"""
src/evaluation/qualitative.py
------------------------------
Generate qualitative evaluation samples for the production model.

Two split modes are available via ``--split``:

val (default) — Samples 50 test images per damage class from the validation
    set, which has ground-truth labels. Each image is saved with GT bounding
    boxes drawn in green and model predictions drawn in red, allowing direct
    visual comparison between what the model detects and what is annotated.
    Output: ``outputs/qualitative/val/{class}/``

test — Samples 50 images per country from the official test split, which has
    no ground-truth labels. Only model predictions (red) are drawn.
    Useful to visually inspect model behaviour on truly unseen data.
    Output: ``outputs/qualitative/test/{country}/``

both — Runs both modes above.

Why two splits? See DOCUMENTATION/IN DETAIL/evaluation.md.

Usage:
    python -m src.evaluation.qualitative                     # val only
    python -m src.evaluation.qualitative --split test        # test only
    python -m src.evaluation.qualitative --split both        # both
    python -m src.evaluation.qualitative --n-per-class 25   # smaller sample
    python -m src.evaluation.qualitative --conf 0.4
"""

from __future__ import annotations

import argparse
import os
import random
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

from src.db.connection import get_db

load_dotenv()

RANDOM_SEED = 42
CLASS_NAMES = ["D00", "D10", "D20", "D40"]
OUTPUT_DIR = Path("outputs") / "qualitative"
RUNS_DIR = Path("runs") / "train"

_GT_COLOR = (0, 200, 0)    # green — ground truth
_PRED_COLOR = (0, 0, 220)  # red — model predictions
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_THICKNESS = 2


# ---------------------------------------------------------------------------
# Helpers — experiment / checkpoint (same pattern as evaluate.py)
# ---------------------------------------------------------------------------


def _get_experiment(run_id: str | None) -> dict[str, Any]:
    db = get_db()
    if run_id:
        doc = db["experiments"].find_one({"run_id": run_id})
        if doc is None:
            raise RuntimeError(f"No experiment found with run_id='{run_id}'.")
    else:
        doc = db["experiments"].find_one({"is_production": True})
        if doc is None:
            raise RuntimeError("No production model found (is_production=True).")
    return doc


def _resolve_checkpoint(exp: dict[str, Any]) -> Path:
    run_id = exp["run_id"]
    direct = RUNS_DIR / run_id / "weights" / "best.pt"
    if direct.exists():
        return direct
    if RUNS_DIR.exists():
        for candidate in sorted(RUNS_DIR.iterdir()):
            if candidate.name.startswith(run_id):
                ckpt = candidate / "weights" / "best.pt"
                if ckpt.exists():
                    return ckpt
    b2_url = exp.get("checkpoints", {}).get("best_pt")
    if b2_url:
        dl_dir = Path("checkpoints") / run_id
        dl_dir.mkdir(parents=True, exist_ok=True)
        dest = dl_dir / "best.pt"
        if not dest.exists():
            print(f"  Downloading checkpoint from B2: {b2_url}")
            with urllib.request.urlopen(b2_url, timeout=300) as resp:
                with open(dest, "wb") as fh:
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
        return dest
    raise RuntimeError(f"Cannot find best.pt for run_id='{run_id}'.")


# ---------------------------------------------------------------------------
# Helpers — annotation drawing
# ---------------------------------------------------------------------------


def _draw_gt_boxes(img: Any, annotations: list[dict[str, Any]]) -> None:
    for ann in annotations:
        x1, y1 = int(ann.get("xmin", 0)), int(ann.get("ymin", 0))
        x2, y2 = int(ann.get("xmax", 0)), int(ann.get("ymax", 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), _GT_COLOR, _THICKNESS)
        cv2.putText(img, f"GT:{ann.get('label', '')}", (x1, max(y1 - 4, 0)), _FONT, _FONT_SCALE, _GT_COLOR, 1)


def _draw_pred_boxes(img: Any, pred_result: Any, conf_thresh: float) -> None:
    boxes = pred_result.boxes
    if boxes is None:
        return
    for box in boxes:
        c = float(box.conf[0])
        if c < conf_thresh:
            continue
        cls_idx = int(box.cls[0])
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cv2.rectangle(img, (x1, y1), (x2, y2), _PRED_COLOR, _THICKNESS)
        cv2.putText(img, f"{cls_name}:{c:.2f}", (x1, max(y1 - 4, 0)), _FONT, _FONT_SCALE, _PRED_COLOR, 1)


def _save_annotated(img: Any, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# ---------------------------------------------------------------------------
# Val qualitative — 50 per class, GT + predictions
# ---------------------------------------------------------------------------


def _run_val_qualitative(
    model: YOLO,
    rdd_root: str,
    n_per_class: int,
    conf: float,
    device: str,
) -> int:
    """Sample n_per_class images per damage class from the val split.

    Images are selected from MongoDB images_metadata where split=val and
    annotations contain the target class. Both GT (green) and model
    predictions (red) are drawn.

    Returns total number of images saved.
    """
    print("\n--- Val qualitative: GT + predictions ---")
    db = get_db()
    root = Path(rdd_root)
    rng = random.Random(RANDOM_SEED)
    total = 0

    for cls_name in CLASS_NAMES:
        cursor = db["images_metadata"].find(
            {"split": "val", "annotations.label": cls_name},
            {"image_id": 1, "filepath": 1, "annotations": 1, "_id": 0},
        )
        candidates = sorted(list(cursor), key=lambda d: d["image_id"])
        rng_copy = random.Random(RANDOM_SEED)
        rng_copy.shuffle(candidates)
        selected = candidates[:n_per_class]

        out_dir = OUTPUT_DIR / "val" / cls_name
        saved = 0
        for doc in selected:
            img_path = root / doc["filepath"]
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            _draw_gt_boxes(img, doc.get("annotations", []))
            preds = model.predict(source=str(img_path), imgsz=640, conf=conf, device=device, verbose=False, save=False)
            if preds:
                _draw_pred_boxes(img, preds[0], conf)
            _save_annotated(img, out_dir / f"{img_path.stem}_{cls_name}.png")
            saved += 1

        print(f"  [val] {cls_name}: {saved}/{len(selected)} saved to {out_dir}/")
        total += saved

    return total


# ---------------------------------------------------------------------------
# Test qualitative — 50 per country, predictions only (no GT)
# ---------------------------------------------------------------------------


def _run_test_qualitative(
    model: YOLO,
    rdd_root: str,
    n_per_country: int,
    conf: float,
    device: str,
) -> int:
    """Sample n_per_country images per country from the official test split.

    No ground-truth labels are available for the test split, so only model
    predictions (red boxes) are drawn. This gives a visual sense of model
    behaviour on truly unseen, label-free data.

    Returns total number of images saved.
    """
    print("\n--- Test qualitative: predictions only (no GT available) ---")
    db = get_db()
    root = Path(rdd_root)
    rng = random.Random(RANDOM_SEED)

    # Group test images by country.
    cursor = db["images_metadata"].find(
        {"split": "test"},
        {"image_id": 1, "filepath": 1, "country": 1, "_id": 0},
    )
    by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in cursor:
        by_country[doc["country"]].append(doc)

    total = 0
    for country in sorted(by_country.keys()):
        candidates = sorted(by_country[country], key=lambda d: d["image_id"])
        rng_copy = random.Random(RANDOM_SEED)
        rng_copy.shuffle(candidates)
        selected = candidates[:n_per_country]

        out_dir = OUTPUT_DIR / "test" / country
        saved = 0
        for doc in selected:
            img_path = root / doc["filepath"]
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            preds = model.predict(source=str(img_path), imgsz=640, conf=conf, device=device, verbose=False, save=False)
            if preds:
                _draw_pred_boxes(img, preds[0], conf)
            _save_annotated(img, out_dir / f"{img_path.stem}.png")
            saved += 1

        print(f"  [test] {country}: {saved}/{len(selected)} saved to {out_dir}/")
        total += saved

    return total


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------


def generate_qualitative(
    run_id: str | None = None,
    split: str = "val",
    n_per_class: int = 50,
    conf: float = 0.5,
    device: str = "0",
) -> None:
    """Generate qualitative evaluation samples.

    Args:
        run_id: Target experiment run_id, or None to use the production model.
        split: ``"val"`` (GT+predictions), ``"test"`` (predictions only), or
            ``"both"``.
        n_per_class: Images to sample per class (val) or per country (test).
        conf: Confidence threshold for displayed predictions.
        device: CUDA device string.

    Raises:
        EnvironmentError: If ``RDD_DATA_ROOT`` is not set.
        ValueError: If ``split`` is not one of the accepted values.
    """
    if split not in ("val", "test", "both"):
        raise ValueError(f"split must be 'val', 'test', or 'both', got '{split}'.")

    rdd_root = os.getenv("RDD_DATA_ROOT")
    if not rdd_root:
        raise EnvironmentError("RDD_DATA_ROOT is not set in .env.")

    print("=" * 60)
    print("RDDS — Qualitative Evaluation")
    print(f"  split={split}, n_per_class={n_per_class}, conf={conf}")
    print("=" * 60)

    exp = _get_experiment(run_id)
    target_run_id = exp["run_id"]
    print(f"\nModel: {target_run_id}  ({exp.get('model', '?')})")

    checkpoint = _resolve_checkpoint(exp)
    print(f"  Checkpoint: {checkpoint}")
    model = YOLO(str(checkpoint))

    total = 0
    if split in ("val", "both"):
        total += _run_val_qualitative(model, rdd_root, n_per_class, conf, device)
    if split in ("test", "both"):
        total += _run_test_qualitative(model, rdd_root, n_per_class, conf, device)

    print(f"\nDone. {total} annotated images saved under {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate qualitative evaluation samples.\n"
            "  --split val  : 50 images/class from val set, GT + predictions.\n"
            "  --split test : 50 images/country from test set, predictions only.\n"
            "  --split both : runs both."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--split", default="val", choices=["val", "test", "both"])
    parser.add_argument("--n-per-class", type=int, default=50)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_qualitative(
        run_id=args.run_id,
        split=args.split,
        n_per_class=args.n_per_class,
        conf=args.conf,
        device=args.device,
    )
