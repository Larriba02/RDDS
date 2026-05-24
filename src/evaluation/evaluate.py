"""
src/evaluation/evaluate.py
--------------------------
Evaluate a model following the CRDDC2022 protocol.

Two evaluation modes are available via ``--split``:

val (default) — Quantitative evaluation on the fixed validation set.
    The validation set (1 000 images per country, held out during training)
    has ground-truth labels. This is our reported metric set.
    Computes F1, Precision, Recall, mAP@0.5 per class, per country, and
    overall at IoU ≥ 0.5 / conf = 0.5.
    Results are written to MongoDB ``experiments.metrics.evaluation_val``.

both — Runs val evaluation above, then additionally runs inference on the
    official test split (no ground-truth labels available) and writes a
    detection summary to MongoDB ``experiments.metrics.evaluation_test``.
    Individual predictions are saved locally to
    ``outputs/test_predictions.json`` rather than flooding the DB.

Why two splits? See DOCUMENTATION/IN DETAIL/evaluation.md for the full
explanation.

Checkpoint resolution order:
    1. runs/train/<run_id>/weights/best.pt
    2. Any runs/train/*/weights/best.pt whose dir name starts with run_id
    3. Download best.pt from the B2 URL stored in MongoDB

Usage:
    # Quantitative evaluation on the validation set (default):
    python -m src.evaluation.evaluate

    # Val metrics + test inference summary:
    python -m src.evaluation.evaluate --split both

    # Specific run:
    python -m src.evaluation.evaluate --run-id run_20260504_202658_yolo11s

    # Dry run (no MongoDB write):
    python -m src.evaluation.evaluate --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from src.db.connection import get_db

load_dotenv()

DEFAULT_CONF = 0.5
DEFAULT_IOU = 0.5

CLASS_NAMES = ["D00", "D10", "D20", "D40"]
LOG_DIR = Path("logs")
SPLITS_JSON = LOG_DIR / "splits.json"
RUNS_DIR = Path("runs") / "train"
OUTPUT_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Helpers — experiment / checkpoint
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
            raise RuntimeError(
                "No production model found (is_production=True). "
                "Pass --run-id explicitly or train a model first."
            )
    return doc


def _resolve_checkpoint(exp: dict[str, Any]) -> Path:
    run_id = exp["run_id"]

    direct = RUNS_DIR / run_id / "weights" / "best.pt"
    if direct.exists():
        print(f"  Checkpoint: {direct}")
        return direct

    if RUNS_DIR.exists():
        for candidate in sorted(RUNS_DIR.iterdir()):
            if candidate.name.startswith(run_id):
                ckpt = candidate / "weights" / "best.pt"
                if ckpt.exists():
                    print(f"  Checkpoint (fuzzy match): {ckpt}")
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
        else:
            print(f"  Using cached download: {dest}")
        return dest

    raise RuntimeError(
        f"Cannot find best.pt for run_id='{run_id}'. "
        "Check local runs/ directory or MongoDB checkpoints.best_pt."
    )


# ---------------------------------------------------------------------------
# Helpers — image lists
# ---------------------------------------------------------------------------


def _load_splits() -> dict[str, str]:
    if not SPLITS_JSON.exists():
        raise FileNotFoundError(
            f"{SPLITS_JSON} not found. Run 'python -m src.data.split' first."
        )
    with open(SPLITS_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def _load_image_metadata() -> dict[str, dict[str, Any]]:
    db = get_db()
    cursor = db["images_metadata"].find(
        {}, {"image_id": 1, "filepath": 1, "country": 1, "_id": 0}
    )
    result: dict[str, dict[str, Any]] = {}
    for doc in cursor:
        result[doc["image_id"]] = {"filepath": doc["filepath"], "country": doc["country"]}
    if not result:
        raise RuntimeError(
            "images_metadata collection is empty. Run 'python -m src.data.ingest' first."
        )
    return result


def _build_image_lists(rdd_root: str, split: str) -> dict[str, list[Path]]:
    """Build per-country and combined image path lists for the given split.

    Args:
        rdd_root: Absolute path to the RDD2022 dataset root.
        split: ``"val"`` or ``"test"``.

    Returns:
        Dict with country keys + ``"_all"``. Values are lists of absolute Paths.
    """
    splits_map = _load_splits()
    metadata = _load_image_metadata()
    root = Path(rdd_root)

    ids = [img_id for img_id, s in splits_map.items() if s == split]
    if not ids:
        raise RuntimeError(
            f"No images found with split='{split}' in splits.json."
        )

    by_country: dict[str, list[Path]] = defaultdict(list)
    missing = 0
    for img_id in ids:
        meta = metadata.get(img_id)
        if meta is None:
            missing += 1
            continue
        by_country[meta["country"]].append(root / meta["filepath"])

    if missing:
        print(f"  [warn] {missing} image_ids have no MongoDB metadata — skipped.")

    all_paths: list[Path] = [p for paths in by_country.values() for p in paths]
    result: dict[str, list[Path]] = dict(by_country)
    result["_all"] = all_paths

    for key, paths in result.items():
        print(f"  [{split}] {key if key != '_all' else 'ALL'}: {len(paths)} images")

    return result


def _write_image_list(paths: list[Path], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # No explicit encoding: use the system default so Ultralytics can read
    # the file back correctly (Ultralytics also opens txt files without
    # explicit encoding, using the platform default).
    with open(dest, "w") as fh:
        fh.writelines(f"{p}\n" for p in paths)


def _write_val_yaml(image_list_path: Path, yaml_path: Path) -> None:
    data = {
        "train": str(image_list_path),
        "val": str(image_list_path),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Helpers — metric extraction
# ---------------------------------------------------------------------------


def _extract_metrics(val_results: Any) -> dict[str, Any]:
    """Extract aggregate + per-class metrics from a YOLO val() result object."""
    metrics: dict[str, Any] = {}

    rd = getattr(val_results, "results_dict", {})

    def _get(fragment: str) -> float | None:
        for k, v in rd.items():
            if fragment in k.strip():
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None
        return None

    p = _get("precision")
    r = _get("recall")
    map50 = _get("mAP50(B)") or _get("mAP50")

    if p is not None:
        metrics["precision"] = round(p, 6)
    if r is not None:
        metrics["recall"] = round(r, 6)
    if map50 is not None:
        metrics["mAP50"] = round(map50, 6)
    if p is not None and r is not None:
        metrics["F1"] = round(2 * p * r / (p + r), 6) if (p + r) > 0 else 0.0

    box = getattr(val_results, "box", None)
    if box is not None:
        per_p = getattr(box, "p", None)
        per_r = getattr(box, "r", None)
        ap_idx = getattr(box, "ap_class_index", None)
        if per_p is not None and per_r is not None and ap_idx is not None:
            per_class_f1: dict[str, float] = {}
            for i, cls_idx in enumerate(ap_idx):
                if i >= len(per_p) or i >= len(per_r):
                    break
                cls_name = CLASS_NAMES[int(cls_idx)] if int(cls_idx) < len(CLASS_NAMES) else str(cls_idx)
                cp, cr = float(per_p[i]), float(per_r[i])
                per_class_f1[cls_name] = round(2 * cp * cr / (cp + cr), 6) if (cp + cr) > 0 else 0.0
            if per_class_f1:
                metrics["F1_per_class"] = per_class_f1

    return metrics


# ---------------------------------------------------------------------------
# Val evaluation (quantitative — has ground truth)
# ---------------------------------------------------------------------------


def _evaluate_val(
    model: YOLO,
    rdd_root: str,
    conf: float,
    iou: float,
    device: str,
) -> dict[str, Any]:
    """Run val() on the validation split and return the evaluation document."""
    print("\n--- Quantitative evaluation on validation set ---")
    image_lists = _build_image_lists(rdd_root, "val")

    with tempfile.TemporaryDirectory(prefix="rdds_eval_val_") as tmp_dir:
        tmp = Path(tmp_dir)

        # Overall.
        print("\n  >> All countries ...")
        all_list = tmp / "val_all.txt"
        all_yaml = tmp / "data_all.yaml"
        _write_image_list(image_lists["_all"], all_list)
        _write_val_yaml(all_list, all_yaml)
        overall_results = model.val(
            data=str(all_yaml), imgsz=640, conf=conf, iou=iou,
            split="val", device=device, verbose=False, save=False, plots=False,
            workers=0, batch=8,
        )
        overall_metrics = _extract_metrics(overall_results)

        # Per country.
        per_country_metrics: dict[str, dict[str, Any]] = {}
        for country in sorted(k for k in image_lists if k != "_all"):
            print(f"\n  >> {country} ({len(image_lists[country])} images) ...")
            c_list = tmp / f"val_{country}.txt"
            c_yaml = tmp / f"data_{country}.yaml"
            _write_image_list(image_lists[country], c_list)
            _write_val_yaml(c_list, c_yaml)
            c_results = model.val(
                data=str(c_yaml), imgsz=640, conf=conf, iou=iou,
                split="val", device=device, verbose=False, save=False, plots=False,
                workers=0, batch=8,
            )
            per_country_metrics[country] = _extract_metrics(c_results)

    per_country_f1 = {c: m.get("F1", 0.0) for c, m in per_country_metrics.items()}
    per_country_map50 = {c: m.get("mAP50", 0.0) for c, m in per_country_metrics.items()}

    return {
        "split": "val",
        "note": (
            "Fixed held-out validation set (1 000 images/country, never seen during "
            "training). Ground truth available. This is the reported metric."
        ),
        "F1_overall": overall_metrics.get("F1"),
        "precision_overall": overall_metrics.get("precision"),
        "recall_overall": overall_metrics.get("recall"),
        "mAP50_overall": overall_metrics.get("mAP50"),
        "F1_per_class": overall_metrics.get("F1_per_class", {}),
        "F1_per_country": per_country_f1,
        "mAP50_per_country": per_country_map50,
        "n_images": len(image_lists["_all"]),
        "iou_threshold": iou,
        "confidence_threshold": conf,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Test inference (no ground truth — summary only)
# ---------------------------------------------------------------------------


def _evaluate_test_inference(
    model: YOLO,
    rdd_root: str,
    conf: float,
    device: str,
    run_id: str,
) -> dict[str, Any]:
    """Run predict() on the official test split and return a detection summary.

    No F1 or mAP can be computed (no ground truth available).
    Individual predictions are saved to outputs/test_predictions.json.
    """
    print("\n--- Test split inference (no ground truth) ---")
    image_lists = _build_image_lists(rdd_root, "test")

    detections_per_class: dict[str, int] = {c: 0 for c in CLASS_NAMES}
    conf_sums: dict[str, float] = {c: 0.0 for c in CLASS_NAMES}
    images_with_detections = 0
    per_image_records: list[dict[str, Any]] = []

    all_paths = image_lists["_all"]
    print(f"\n  Running inference on {len(all_paths)} test images ...")

    # Build a country lookup from path for per-image records.
    path_to_country: dict[str, str] = {}
    for country, paths in image_lists.items():
        if country == "_all":
            continue
        for p in paths:
            path_to_country[str(p)] = country

    for img_path in all_paths:
        results = model.predict(
            source=str(img_path),
            imgsz=640,
            conf=conf,
            device=device,
            verbose=False,
            save=False,
        )
        if not results:
            continue

        boxes = results[0].boxes
        dets: list[dict[str, Any]] = []
        if boxes is not None and len(boxes) > 0:
            images_with_detections += 1
            for box in boxes:
                cls_idx = int(box.cls[0])
                cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
                c = float(box.conf[0])
                detections_per_class[cls_name] += 1
                conf_sums[cls_name] += c
                dets.append({"label": cls_name, "conf": round(c, 4), "bbox": [round(v) for v in box.xyxy[0].tolist()]})

        per_image_records.append({
            "filepath": str(img_path),
            "country": path_to_country.get(str(img_path), "unknown"),
            "detections": dets,
        })

    # Save per-image predictions locally.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_json = OUTPUT_DIR / f"test_predictions_{run_id}.json"
    with open(pred_json, "w", encoding="utf-8") as fh:
        json.dump(per_image_records, fh, indent=2)
    print(f"  Per-image predictions saved to {pred_json}")

    avg_conf = {
        c: round(conf_sums[c] / detections_per_class[c], 4) if detections_per_class[c] > 0 else 0.0
        for c in CLASS_NAMES
    }

    return {
        "split": "test",
        "note": (
            "Official CRDDC2022 test split. Ground truth not publicly available — "
            "no F1/mAP computed. Detection counts reflect model output only."
        ),
        "n_images": len(all_paths),
        "n_images_with_detections": images_with_detections,
        "detection_rate": round(images_with_detections / len(all_paths), 4) if all_paths else 0.0,
        "detections_per_class": detections_per_class,
        "avg_confidence_per_class": avg_conf,
        "predictions_file": str(pred_json),
        "confidence_threshold": conf,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------


def evaluate(
    run_id: str | None = None,
    split: str = "val",
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    device: str = "0",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate a model and write results to MongoDB.

    Args:
        run_id: Target experiment run_id, or None to use the production model.
        split: ``"val"`` for quantitative evaluation (default), or ``"both"``
            to also run inference on the official test split.
        conf: Confidence threshold.
        iou: IoU threshold for NMS and metric computation.
        device: CUDA device string (``"0"`` or ``"cpu"``).
        dry_run: If True, skip the MongoDB write and return results only.

    Returns:
        Dict with ``"val"`` key (always) and optionally ``"test"`` key.

    Raises:
        EnvironmentError: If ``RDD_DATA_ROOT`` is not set.
        ValueError: If ``split`` is not ``"val"`` or ``"both"``.
    """
    if split not in ("val", "both"):
        raise ValueError(f"split must be 'val' or 'both', got '{split}'.")

    rdd_root = os.getenv("RDD_DATA_ROOT")
    if not rdd_root:
        raise EnvironmentError("RDD_DATA_ROOT is not set in .env.")

    print("=" * 60)
    print("RDDS — Evaluation (CRDDC2022 protocol)")
    print(f"  split={split}, conf={conf}, iou={iou}, device={device}")
    print("=" * 60)

    exp = _get_experiment(run_id)
    target_run_id = exp["run_id"]
    print(f"\nModel: {target_run_id}  ({exp.get('model', '?')})")

    print("\n[1/N] Resolving checkpoint ...")
    checkpoint = _resolve_checkpoint(exp)

    print("\n[2/N] Loading model ...")
    model = YOLO(str(checkpoint))

    output: dict[str, Any] = {}

    # --- Val (always) ---
    print("\n[3/N] Val split evaluation ...")
    val_doc = _evaluate_val(model, rdd_root, conf, iou, device)
    output["val"] = val_doc
    _print_val_summary(val_doc)

    # --- Test (optional) ---
    if split == "both":
        print("\n[4/N] Test split inference ...")
        test_doc = _evaluate_test_inference(model, rdd_root, conf, device, target_run_id)
        output["test"] = test_doc
        _print_test_summary(test_doc)

    # --- MongoDB write ---
    if dry_run:
        print("\n[dry-run] MongoDB write skipped.")
    else:
        print("\nWriting to MongoDB ...")
        db = get_db()
        update: dict[str, Any] = {
            "metrics.evaluation_val": val_doc,
            "metrics.F1": val_doc["F1_overall"],
            "metrics.mAP50": val_doc["mAP50_overall"],
            "metrics.precision": val_doc["precision_overall"],
            "metrics.recall": val_doc["recall_overall"],
        }
        if "test" in output:
            update["metrics.evaluation_test"] = output["test"]
        db["experiments"].update_one({"run_id": target_run_id}, {"$set": update})
        print(f"  MongoDB updated for {target_run_id}.")

    return output


def _print_val_summary(doc: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("VAL SET RESULTS (reported metric)")
    print("=" * 60)
    print(f"  F1 overall:        {doc.get('F1_overall', 0):.4f}")
    print(f"  Precision overall: {doc.get('precision_overall', 0):.4f}")
    print(f"  Recall overall:    {doc.get('recall_overall', 0):.4f}")
    print(f"  mAP@0.5 overall:   {doc.get('mAP50_overall', 0):.4f}")
    print()
    print("  F1 per country:")
    for country, f1 in sorted(doc.get("F1_per_country", {}).items()):
        print(f"    {country:<20} {f1:.4f}")
    if doc.get("F1_per_class"):
        print()
        print("  F1 per class:")
        for cls, f1 in sorted(doc["F1_per_class"].items()):
            print(f"    {cls:<8} {f1:.4f}")
    print("=" * 60)


def _print_test_summary(doc: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("TEST SPLIT INFERENCE SUMMARY (no ground truth)")
    print("=" * 60)
    print(f"  Images processed:        {doc.get('n_images', 0)}")
    print(f"  Images with detections:  {doc.get('n_images_with_detections', 0)}")
    print(f"  Detection rate:          {doc.get('detection_rate', 0):.1%}")
    print()
    print("  Detections per class:")
    for cls, count in sorted(doc.get("detections_per_class", {}).items()):
        avg_c = doc.get("avg_confidence_per_class", {}).get(cls, 0.0)
        print(f"    {cls:<8} {count:>6} detections  (avg conf {avg_c:.3f})")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the production model following CRDDC2022 protocol.\n"
            "  --split val  : F1/mAP on the validation set (has GT labels).\n"
            "  --split both : val metrics + test inference summary (no GT)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--split", default="val", choices=["val", "both"])
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        run_id=args.run_id,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        dry_run=args.dry_run,
    )
