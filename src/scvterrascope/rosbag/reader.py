"""Iterate `sensor_msgs/Image[Compressed]` frames from a ROS 2 bag.

Why this wrapper exists:
- `rosbags` is forward-iterator only and reads the message blob on every
  step. The GUI's seek bar needs O(log n) random access. We piggyback
  on the underlying sqlite3 storage to (a) build a timestamp-only index
  per topic at open() (~7 ms for a 5k-frame bag) and (b) translate
  index → timestamp → `rosbags.messages(start=ts)` for O(1) seeks.
- Bag-level duration / start time come from `metadata.yaml` so opening
  is instantaneous regardless of bag length — important for GUI UX.
- We accept many image encodings (bgr8, bgra8, rgb8, rgba8, mono8,
  CompressedImage). The model expects PIL RGB so we normalize here.
- Public API is intentionally small (open/close/topics/seek/iterate)
  so the GUI doesn't depend on rosbags directly.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

LOG = logging.getLogger(__name__)

_IMAGE_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")


@dataclass(frozen=True)
class ImageTopicInfo:
    topic: str
    msgtype: str
    count: int


class BagReader:
    """Open a ROS 2 bag directory and iterate image frames.

    Usage:
        with BagReader(Path('/path/to/bag_dir')) as r:
            topics = r.image_topics()
            for frame in r.iter_topic(topics[0].topic):
                ...

    Or for random access:
        r = BagReader(path).open()
        frame = r.frame_at(topic, index=120)
        r.close()
    """

    def __init__(self, bag_path: str | Path) -> None:
        self.bag_path = Path(bag_path)
        self._reader: Any = None             # rosbags.rosbag2.Reader
        self._typestore: Any = None
        self._connections: list[Any] = []    # rosbags ConnectionExt
        self._all_topics: dict[str, ImageTopicInfo] = {}
        # Bag-level timing read from metadata.yaml (instant; no full scan).
        self._start_ns: int = 0
        self._duration_ns: int = 0
        # Cached forward iterator so step_forward() doesn't re-scan from
        # the start of the bag every call. seek_to() rebuilds it.
        self._active_topic: str | None = None
        self._active_iter: Any = None
        self._active_next_idx: int = 0       # idx of the NEXT frame to yield
        # Per-topic timestamp index built on open() via direct sqlite query
        # (no blob fetch — fast). Used by frame_at() for O(1) random seek.
        self._ts_index: dict[str, list[int]] = {}

    # ----- lifecycle ------------------------------------------------
    def __enter__(self) -> "BagReader":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> "BagReader":
        if self._reader is not None:
            return self
        try:
            from rosbags.rosbag2 import Reader
            from rosbags.typesys import Stores, get_typestore
        except ImportError as exc:
            raise ImportError(
                "BagReader requires `rosbags`. Install via `pip install -e .[dev]` "
                "(rosbags is in pyproject)."
            ) from exc

        if not self.bag_path.is_dir():
            raise FileNotFoundError(
                f"bag path is not a directory: {self.bag_path}. "
                "rosbag2 expects the directory containing metadata.yaml + *.db3."
            )

        self._typestore = get_typestore(Stores.ROS2_HUMBLE)
        # `rosbags`' Reader is itself a context manager but we mirror the
        # idiom so callers can do random-access without `with`.
        self._reader = Reader(self.bag_path)
        self._reader.open()
        self._connections = list(self._reader.connections)

        # Catalogue image topics — instant, no message walk.
        self._all_topics.clear()
        for c in self._connections:
            if c.msgtype in _IMAGE_TYPES:
                self._all_topics[c.topic] = ImageTopicInfo(
                    topic=c.topic, msgtype=c.msgtype, count=int(c.msgcount),
                )

        # Bag-level timing from metadata.yaml (no walk).
        self._start_ns, self._duration_ns = self._read_bag_timing(self.bag_path)

        # Build timestamp-only index per image topic. This is what makes
        # arbitrary-frame seek O(log n) instead of O(n).
        self._ts_index = self._build_timestamp_index(list(self._all_topics))

        LOG.info(
            "opened bag %s — %d image topic(s), %d total image frames",
            self.bag_path.name, len(self._all_topics),
            sum(t.count for t in self._all_topics.values()),
        )
        return self

    @staticmethod
    def _read_bag_timing(bag_path: Path) -> tuple[int, int]:
        """Pull (starting_time_ns, duration_ns) from metadata.yaml without
        walking a single message. Returns (0, 0) if metadata is missing."""
        meta = bag_path / "metadata.yaml"
        if not meta.is_file():
            return 0, 0
        try:
            import yaml

            data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
            info = data.get("rosbag2_bagfile_information", {}) or {}
            start = int(((info.get("starting_time") or {}).get("nanoseconds_since_epoch")) or 0)
            dur = int(((info.get("duration") or {}).get("nanoseconds")) or 0)
            return start, dur
        except Exception as exc:  # noqa: BLE001 — defensive; metadata may be malformed
            LOG.warning("failed to parse %s: %s", meta, exc)
            return 0, 0

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
        self._reader = None
        self._typestore = None
        self._connections = []
        self._active_topic = None
        self._active_iter = None
        self._active_next_idx = 0
        self._ts_index = {}

    # ----- introspection -------------------------------------------
    def image_topics(self) -> list[ImageTopicInfo]:
        """Image-bearing topics in the bag, ordered as discovered."""
        self._require_open()
        return list(self._all_topics.values())

    def frame_count(self, topic: str) -> int:
        self._require_open()
        info = self._all_topics.get(topic)
        return info.count if info else 0

    def duration_seconds(self) -> float:
        return self._duration_ns / 1e9

    @property
    def start_ns(self) -> int:
        return self._start_ns

    # ----- iteration / seek ----------------------------------------
    def iter_topic(
        self, topic: str, *, start_index: int = 0
    ) -> Iterator[Any]:  # yields BagFrame
        """Yield BagFrame objects for `topic` from `start_index` onward.

        Restarts the forward iterator (so this is a fresh walk every
        time it's called). The GUI's hot path is `step_forward` which
        keeps iterator state across calls.
        """
        self._require_open()
        self._restart_iterator(topic, skip=start_index)
        while True:
            frame = self.step_forward(topic)
            if frame is None:
                return
            yield frame

    def step_forward(self, topic: str) -> Any | None:
        """Yield the next BagFrame on `topic`, O(1) amortized.

        Maintains a single live iterator across calls; switching topics
        or seeking backward triggers a rebuild. Returns None at end of
        topic — caller should treat that as "stop" / "loop".
        """
        from scvterrascope.rosbag.types import BagFrame

        self._require_open()
        if self._active_topic != topic or self._active_iter is None:
            self._restart_iterator(topic, skip=0)
        try:
            cn, ts_ns, raw = next(self._active_iter)
        except StopIteration:
            return None
        idx = self._active_next_idx
        self._active_next_idx = idx + 1
        image = self._decode_image(
            self._typestore.deserialize_cdr(raw, cn.msgtype), cn.msgtype
        )
        return BagFrame(
            idx=idx, ros_time_ns=int(ts_ns), image=image,
            topic=cn.topic, encoding=getattr(image, "_bag_encoding", ""),
        )

    def frame_at(self, topic: str, index: int) -> Any:
        """Return a single BagFrame at `index` in `topic`. Random access.

        Two paths:
          1. If the cached forward iterator is already pointing at the
             requested frame, advance it (O(1)). This is the common
             autoplay path.
          2. Otherwise, look up the timestamp via the in-memory index
             and ask `rosbags` to start streaming from that timestamp —
             O(log n) regardless of bag size.
        """
        from scvterrascope.rosbag.types import BagFrame

        self._require_open()
        n = self.frame_count(topic)
        if not 0 <= index < n:
            raise IndexError(f"frame {index} out of range [0, {n}) for topic {topic}")

        # Path 1 — cached iterator already at the target.
        if (
            self._active_topic == topic
            and self._active_iter is not None
            and index == self._active_next_idx
        ):
            frame = self.step_forward(topic)
            if frame is not None:
                return frame

        # Path 2 — timestamp-indexed random access.
        ts_ns_target = self._ts_index[topic][index]
        conn = self._connection_for(topic)
        # `start_ns` is inclusive; ask for the single message at that
        # timestamp, then leave the iterator positioned to yield the
        # NEXT frame on subsequent step_forward calls.
        it = self._reader.messages(connections=[conn], start=ts_ns_target)
        cn, ts_ns, raw = next(it)
        image = self._decode_image(
            self._typestore.deserialize_cdr(raw, cn.msgtype), cn.msgtype
        )
        self._active_topic = topic
        self._active_iter = it
        self._active_next_idx = index + 1
        return BagFrame(
            idx=index, ros_time_ns=int(ts_ns), image=image,
            topic=topic, encoding=getattr(image, "_bag_encoding", ""),
        )

    # ----- internals ------------------------------------------------
    def _require_open(self) -> None:
        if self._reader is None:
            raise RuntimeError("BagReader is closed; call open() first")

    def _connection_for(self, topic: str) -> Any:
        for c in self._connections:
            if c.topic == topic:
                return c
        raise KeyError(f"no connection for topic {topic!r} in {self.bag_path}")

    def _restart_iterator(self, topic: str, *, skip: int) -> None:
        """Build a fresh forward iterator over `topic`, skipping `skip` msgs."""
        conn = self._connection_for(topic)
        it = self._reader.messages(connections=[conn])
        for _ in range(max(0, skip)):
            try:
                next(it)
            except StopIteration:
                break
        self._active_topic = topic
        self._active_iter = it
        self._active_next_idx = max(0, skip)

    def _build_timestamp_index(self, topics: list[str]) -> dict[str, list[int]]:
        """Per-topic ordered list of message timestamps.

        Built once via direct sqlite queries (no blob fetch — ~7 ms per
        5k-frame topic). Storage other than sqlite3 returns an empty
        index, in which case `frame_at` falls back to walk-from-zero.
        """
        import sqlite3
        import yaml

        if not topics:
            return {}
        meta = self.bag_path / "metadata.yaml"
        try:
            data = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
        except OSError as exc:
            LOG.warning("cannot read %s: %s", meta, exc)
            return {}
        info = data.get("rosbag2_bagfile_information", {}) or {}
        if info.get("storage_identifier") != "sqlite3":
            LOG.warning(
                "non-sqlite3 storage %r — skipping timestamp index; seeks will be slow",
                info.get("storage_identifier"),
            )
            return {}
        rel_files = list(info.get("relative_file_paths") or [])
        if not rel_files:
            return {}

        index: dict[str, list[int]] = {t: [] for t in topics}
        for rel in rel_files:
            db_path = self.bag_path / rel
            if not db_path.is_file():
                LOG.warning("missing bag shard %s", db_path)
                continue
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                # Map topic_name → topic_id for THIS shard (ids may differ).
                rows = conn.execute(
                    "SELECT id, name FROM topics WHERE name IN (%s)"
                    % ",".join("?" * len(topics)),
                    topics,
                ).fetchall()
                tid_to_name = {tid: name for tid, name in rows}
                if not tid_to_name:
                    continue
                for tid, name in tid_to_name.items():
                    cur = conn.execute(
                        "SELECT timestamp FROM messages WHERE topic_id = ? "
                        "ORDER BY timestamp",
                        (tid,),
                    )
                    index[name].extend(int(r[0]) for r in cur)
        return index

    def _decode_image(self, msg: Any, msgtype: str) -> Any:
        """`sensor_msgs/Image[Compressed]` → PIL.Image.Image (RGB).

        Encoded variants we care about (others raise so they're loud):
            bgra8, bgr8, rgb8, rgba8, mono8 (Image)
            jpeg, png (CompressedImage)
        """
        import numpy as np
        from PIL import Image

        if msgtype == "sensor_msgs/msg/CompressedImage":
            pil = Image.open(io.BytesIO(bytes(msg.data))).convert("RGB")
            pil._bag_encoding = msg.format  # type: ignore[attr-defined]
            return pil

        # Raw Image: msg.data is uint8 buffer of shape (height, step) bytes.
        encoding = msg.encoding
        h, w = int(msg.height), int(msg.width)
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        if encoding in ("bgra8", "rgba8"):
            arr = buf.reshape(h, w, 4)
            arr = arr[:, :, [2, 1, 0]] if encoding == "bgra8" else arr[:, :, :3]
        elif encoding == "bgr8":
            arr = buf.reshape(h, w, 3)[:, :, ::-1]
        elif encoding == "rgb8":
            arr = buf.reshape(h, w, 3)
        elif encoding == "mono8":
            arr = np.repeat(buf.reshape(h, w, 1), 3, axis=2)
        else:
            raise NotImplementedError(
                f"unsupported sensor_msgs/Image encoding: {encoding!r}. "
                "Add a branch in BagReader._decode_image."
            )
        pil = Image.fromarray(arr.copy(), "RGB")
        pil._bag_encoding = encoding  # type: ignore[attr-defined]
        return pil


