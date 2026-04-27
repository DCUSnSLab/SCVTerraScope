"""BagReader smoke tests against a real ROS 2 bag.

The Phase 2 reader has no synthetic-bag fixtures (rosbags lib doesn't
publish a test-bag generator), so we lean on the SCV bag the user keeps
locally. Tests skip cleanly when the bag isn't present, e.g., on a fresh
clone or CI without the dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCV_BAG = Path("/home/soobin/data/260113_SCV_D2_Detection_01")
EXPECTED_TOPIC = "/zed/zed_node/left/image_rect_color"
EXPECTED_FRAMES = 5222


@pytest.fixture(scope="module")
def scv_bag() -> Path:
    if not (SCV_BAG / "metadata.yaml").is_file():
        pytest.skip(f"SCV bag not present at {SCV_BAG}")
    return SCV_BAG


def test_bag_reader_lists_image_topics(scv_bag: Path) -> None:
    from scvterrascope.rosbag import BagReader

    with BagReader(scv_bag) as r:
        topics = r.image_topics()
        names = [t.topic for t in topics]
        assert EXPECTED_TOPIC in names
        info = next(t for t in topics if t.topic == EXPECTED_TOPIC)
        assert info.msgtype == "sensor_msgs/msg/Image"
        assert info.count == EXPECTED_FRAMES


def test_bag_reader_first_frame_decodes_to_rgb(scv_bag: Path) -> None:
    from scvterrascope.rosbag import BagReader

    with BagReader(scv_bag) as r:
        frame = next(r.iter_topic(EXPECTED_TOPIC))
        assert frame.idx == 0
        assert frame.image.mode == "RGB"
        # ZED left rect color is 1920×1080 with bgra8 encoding.
        assert frame.image.size == (1920, 1080)
        assert frame.encoding == "bgra8"
        assert frame.ros_time_ns > 0


def test_bag_reader_frame_at_random_access(scv_bag: Path) -> None:
    from scvterrascope.rosbag import BagReader

    with BagReader(scv_bag) as r:
        frame_100 = r.frame_at(EXPECTED_TOPIC, 100)
        assert frame_100.idx == 100
        assert frame_100.image.size == (1920, 1080)
        # ROS time of frame 100 must be later than frame 0.
        first = next(r.iter_topic(EXPECTED_TOPIC))
        assert frame_100.ros_time_ns > first.ros_time_ns


def test_bag_reader_duration_and_start(scv_bag: Path) -> None:
    from scvterrascope.rosbag import BagReader

    with BagReader(scv_bag) as r:
        dur = r.duration_seconds()
        # Bag is ~175 seconds long.
        assert 170.0 < dur < 180.0
        assert r.start_ns > 0


def test_bag_reader_rejects_missing_path(tmp_path: Path) -> None:
    from scvterrascope.rosbag import BagReader

    with pytest.raises(FileNotFoundError):
        BagReader(tmp_path / "no_such_bag").open()


def test_bag_reader_iter_skips_to_start_index(scv_bag: Path) -> None:
    from scvterrascope.rosbag import BagReader

    with BagReader(scv_bag) as r:
        frames = []
        for f in r.iter_topic(EXPECTED_TOPIC, start_index=200):
            frames.append(f.idx)
            if len(frames) >= 3:
                break
        assert frames == [200, 201, 202]
