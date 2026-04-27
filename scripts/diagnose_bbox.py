"""Diagnose bbox alignment by visualizing letterbox-frame vs unprojected boxes.

Why this exists: Phase 1-1 GUI worked on CODa 1224×1024 but bbox positions
look wrong on user-collected SCV 1920×1080 frames. Two hypotheses can be
distinguished by *seeing* both intermediate forms side by side:

  - 02 (letterbox + raw boxes): if boxes already miss objects in the
    letterbox frame, the model is OOD on this aspect ratio. Coordinate
    transform code is innocent.
  - 03 (original + unprojected): if 02 is correct but 03 is shifted, the
    unproject formula is the bug.

For each input, writes four artifacts under outputs/diag/<stem>/:
  01_input.png                          — original (untouched)
  02_letterbox_with_raw_boxes.png       — 1024×1024 letterbox + raw xyxy
  03_original_with_unprojected_boxes.png — original + draw_detections
  04_summary.json                       — meta + per-detection coords

Usage:
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/diagnose_bbox.py \
      --checkpoint <path/to/epoch_050.pt> \
      --inputs <image1.jpg> <image2.png> ...
      [--score-threshold 0.3]
      [--top-k 100]
      [--out-dir outputs/diag]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Make the src/ package reachable when running uninstalled.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from scvterrascope.inference import InferenceEngine  # noqa: E402
from scvterrascope.inference.preprocess import (  # noqa: E402
    preprocess_pil,
    unproject_letterbox_xyxy,
)
from scvterrascope.visualization.draw import (  # noqa: E402
    DrawStyle,
    draw_detections,
    palette_for,
)
from scvterrascope.inference.engine import Detection  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--inputs", type=Path, nargs="+", required=True,
        help="One or more image files to diagnose.",
    )
    p.add_argument("--score-threshold", type=float, default=0.30)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--out-dir", type=Path, default=ROOT / "outputs" / "diag",
        help="Diagnostic outputs root (default: outputs/diag/).",
    )
    return p.parse_args(argv)


def _font() -> Any:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=16)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_letterbox_overlay(
    letterbox_pil: Image.Image,
    raw_boxes_xyxy: list[tuple[float, float, float, float]],
    class_names: list[str],
    scores: list[float],
    palette: tuple[tuple[int, int, int], ...],
    class_ids: list[int],
    *,
    score_threshold: float,
    pad_h: int,
    pad_w: int,
) -> Image.Image:
    """Draw raw model boxes (letterbox-frame xyxy) onto the letterbox image.

    Also overlays a translucent rectangle on the padded region so it's
    obvious where the gray padding area is — this makes it easy to see at
    a glance if the model is predicting INTO the padding (a strong OOD signal).
    """
    canvas = letterbox_pil.convert("RGBA").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = _font()
    target = canvas.size[0]

    # Highlight padding area in semi-transparent red (only if non-zero).
    if pad_h > 0:
        draw.rectangle(
            [(0, target - pad_h), (target, target)],
            fill=(255, 0, 0, 60),
        )
    if pad_w > 0:
        draw.rectangle(
            [(target - pad_w, 0), (target, target - pad_h)],
            fill=(255, 0, 0, 60),
        )

    for box, name, score, cid in zip(raw_boxes_xyxy, class_names, scores, class_ids):
        if score < score_threshold:
            continue
        color = palette[(cid - 1) % len(palette)]
        x1, y1, x2, y2 = box
        draw.rectangle([(x1, y1), (x2, y2)], outline=color + (255,), width=3)
        label = f"{name} {score:.2f}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]
        ly1 = max(0, y1 - th - 6)
        draw.rectangle([(x1, ly1), (x1 + tw + 6, ly1 + th + 6)], fill=color + (255,))
        luma = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (0, 0, 0, 255) if luma > 160 else (255, 255, 255, 255)
        draw.text((x1 + 3, ly1 + 3), label, fill=text_color, font=font)

    return canvas.convert("RGB")


def diagnose_one(
    engine: InferenceEngine,
    input_path: Path,
    *,
    out_dir: Path,
    score_threshold: float,
) -> dict[str, Any]:
    import numpy as np
    import torch

    img = Image.open(input_path).convert("RGB")
    pre = preprocess_pil(img, target_size=engine.image_size)

    # Mirror engine.predict()'s forward path so we can capture the
    # letterbox-frame boxes BEFORE they get unprojected and dropped.
    pixel_values = torch.from_numpy(pre.pixel_values).unsqueeze(0).to(engine.device)
    pixel_mask = torch.from_numpy(pre.pixel_mask).unsqueeze(0).long().to(engine.device)
    with torch.inference_mode():
        outputs = engine._model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    target_sizes = torch.tensor(
        [[engine.image_size, engine.image_size]],
        dtype=torch.long,
        device=engine.device,
    )
    results = engine._processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.0, top_k=engine.top_k
    )[0]
    raw_boxes = results["boxes"].cpu().tolist()       # xyxy in letterbox frame
    raw_scores = results["scores"].cpu().tolist()
    raw_labels = results["labels"].cpu().tolist()    # 0-indexed

    # Reconstruct the letterbox PIL image (re-run letterbox on the original
    # so the diagnostic doesn't depend on private fields of pre.pixel_values
    # which is normalized).
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    from scvterrascope.inference.preprocess import letterbox_resize

    letterbox_arr, _scale, (pad_h, pad_w) = letterbox_resize(arr, engine.image_size)
    letterbox_pil = Image.fromarray(letterbox_arr, "RGB")

    # Build Detection list (unprojected) — same as engine would.
    palette = palette_for(max(16, len(engine.taxonomy)))
    detections: list[Detection] = []
    raw_meta: list[dict[str, Any]] = []
    class_names = []
    class_ids = []
    for s, lab, box in zip(raw_scores, raw_labels, raw_boxes):
        try:
            name = engine.taxonomy.name_for(int(lab))
        except IndexError:
            continue
        class_names.append(name)
        class_ids.append(int(lab) + 1)
        ux1, uy1, ux2, uy2 = unproject_letterbox_xyxy(
            tuple(box), scale=pre.scale, original_size=pre.original_size
        )
        detections.append(
            Detection(
                class_id=int(lab) + 1,
                class_name=name,
                score=float(s),
                bbox_xyxy=(ux1, uy1, ux2, uy2),
            )
        )
        raw_meta.append(
            {
                "class_id": int(lab) + 1,
                "class_name": name,
                "score": float(s),
                "letterbox_xyxy": [round(v, 2) for v in box],
                "original_xyxy": [round(v, 2) for v in (ux1, uy1, ux2, uy2)],
            }
        )

    # ---- Render artifacts ----
    stem = input_path.stem
    sub = out_dir / stem
    sub.mkdir(parents=True, exist_ok=True)

    img.save(sub / "01_input.png")

    overlay_lb = draw_letterbox_overlay(
        letterbox_pil,
        raw_boxes,
        class_names,
        raw_scores,
        palette,
        class_ids,
        score_threshold=score_threshold,
        pad_h=pad_h,
        pad_w=pad_w,
    )
    overlay_lb.save(sub / "02_letterbox_with_raw_boxes.png")

    overlay_orig = draw_detections(
        img,
        detections,
        palette=palette,
        score_threshold=score_threshold,
        style=DrawStyle(line_width=3),
    )
    overlay_orig.save(sub / "03_original_with_unprojected_boxes.png")

    summary = {
        "input": str(input_path),
        "image_size_WxH": list(img.size),
        "image_size_HxW": [img.size[1], img.size[0]],
        "letterbox": {
            "target_size": int(engine.image_size),
            "scale": float(pre.scale),
            "new_h": int(engine.image_size - pad_h),
            "new_w": int(engine.image_size - pad_w),
            "pad_h_bottom": int(pad_h),
            "pad_w_right": int(pad_w),
            "pad_area_pct": round(
                (pad_h * engine.image_size + pad_w * (engine.image_size - pad_h))
                / (engine.image_size * engine.image_size) * 100, 2
            ),
        },
        "score_threshold": float(score_threshold),
        "n_above_threshold": sum(1 for s in raw_scores if s >= score_threshold),
        "n_total": len(raw_scores),
        "detections": [d for d in raw_meta if d["score"] >= score_threshold],
    }
    (sub / "04_summary.json").write_text(json.dumps(summary, indent=2))

    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"loading engine from {args.checkpoint}…")
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        device=args.device,
        image_size=args.image_size,
        top_k=args.top_k,
    )
    engine.load()
    print(
        f"loaded — device={engine.device}  "
        f"params={engine.param_count / 1e6:.1f}M  classes={len(engine.taxonomy)}"
    )

    for in_path in args.inputs:
        if not in_path.is_file():
            print(f"WARN: missing {in_path}", file=sys.stderr)
            continue
        print(f"\n=== diagnosing {in_path} ===")
        s = diagnose_one(
            engine, in_path,
            out_dir=args.out_dir,
            score_threshold=args.score_threshold,
        )
        print(
            f"  size={s['image_size_WxH']}  scale={s['letterbox']['scale']:.4f}  "
            f"pad_h={s['letterbox']['pad_h_bottom']}  pad_area={s['letterbox']['pad_area_pct']}%"
        )
        print(f"  detections @ score>={args.score_threshold}: {s['n_above_threshold']}/{s['n_total']}")
        for d in s["detections"][:5]:
            lb = d["letterbox_xyxy"]
            og = d["original_xyxy"]
            print(
                f"    {d['class_name']:14s} score={d['score']:.3f}  "
                f"letterbox=({lb[0]:.0f},{lb[1]:.0f},{lb[2]:.0f},{lb[3]:.0f})  "
                f"original=({og[0]:.0f},{og[1]:.0f},{og[2]:.0f},{og[3]:.0f})"
            )
        print(f"  artifacts → {args.out_dir / in_path.stem}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
