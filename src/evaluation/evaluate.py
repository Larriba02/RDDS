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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
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
            # The B2 bucket is private — anonymous HTTP returns 401. Reuse the
            # authenticated S3 download (boto3 + .env credentials) already used by
            # inference, so cross-machine checkpoints resolve here too.
            from src.inference.predict import _download_from_b2

            _download_from_b2(b2_url, dest)
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


def _per_class_from_box(box: Any) -> dict[str, dict[str, float]]:
    """Per-class precision, recall, AP@0.5, AP@0.5:0.95 and F1 from a YOLO box metric.

    The box arrays (``p``, ``r``, ``ap50``, ``ap``) are indexed positionally and
    mapped back to class names via ``ap_class_index``. Returns ``{}`` when the
    model produced no detections (Ultralytics leaves the arrays empty).
    """
    ap_idx = getattr(box, "ap_class_index", None)
    p = getattr(box, "p", None)
    r = getattr(box, "r", None)
    ap50 = getattr(box, "ap50", None)
    ap = getattr(box, "ap", None)
    if ap_idx is None or p is None or r is None:
        return {}

    out: dict[str, dict[str, float]] = {}
    for i, cls_idx in enumerate(ap_idx):
        if i >= len(p) or i >= len(r):
            break
        name = CLASS_NAMES[int(cls_idx)] if int(cls_idx) < len(CLASS_NAMES) else str(cls_idx)
        cp, cr = float(p[i]), float(r[i])
        rec: dict[str, float] = {
            "precision": round(cp, 6),
            "recall": round(cr, 6),
            "F1": round(2 * cp * cr / (cp + cr), 6) if (cp + cr) > 0 else 0.0,
        }
        if ap50 is not None and i < len(ap50):
            rec["AP50"] = round(float(ap50[i]), 6)
        if ap is not None and i < len(ap):
            rec["AP50_95"] = round(float(ap[i]), 6)
        out[name] = rec
    return out


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
    # "mAP50-95(B)" must be matched before the looser "mAP50" fragment.
    map5095 = _get("mAP50-95(B)") or _get("mAP50-95")

    if p is not None:
        metrics["precision"] = round(p, 6)
    if r is not None:
        metrics["recall"] = round(r, 6)
    if map50 is not None:
        metrics["mAP50"] = round(map50, 6)
    if map5095 is not None:
        metrics["mAP50-95"] = round(map5095, 6)
    if p is not None and r is not None:
        metrics["F1"] = round(2 * p * r / (p + r), 6) if (p + r) > 0 else 0.0

    box = getattr(val_results, "box", None)
    if box is not None:
        per_class = _per_class_from_box(box)
        if per_class:
            metrics["per_class"] = per_class
            # Keep F1_per_class for backward compatibility with older dashboard code.
            metrics["F1_per_class"] = {c: v["F1"] for c, v in per_class.items()}

    return metrics


# ---------------------------------------------------------------------------
# Helpers — detection error analysis (R2): confusion matrix, FP/FN, PR curves
# ---------------------------------------------------------------------------


def _confusion_matrix_doc(val_results: Any) -> dict[str, Any] | None:
    """Serialise the Ultralytics confusion matrix to a MongoDB-friendly dict.

    Ultralytics only *populates* the confusion matrix when ``plots=True`` is
    passed to ``val()`` — call sites must enable it. Orientation is
    ``matrix[predicted, true]`` with the last index = background.
    """
    cm = getattr(val_results, "confusion_matrix", None)
    if cm is None:
        return None
    mat = np.asarray(getattr(cm, "matrix", None))
    if mat is None or mat.size == 0 or mat.ndim != 2:
        return None

    labels = CLASS_NAMES + ["background"]
    if mat.shape[0] != len(labels):
        labels = [str(i) for i in range(mat.shape[0] - 1)] + ["background"]
    return {
        "labels": labels,
        "matrix": mat.astype(int).tolist(),
        "conf": float(getattr(cm, "conf", DEFAULT_CONF)),
        "iou_thres": float(getattr(cm, "iou_thres", 0.45)),
        "note": (
            "rows=predicted, cols=ground truth; last index=background. Matching at "
            "iou_thres (Ultralytics ConfusionMatrix) — the structural FP/FN view, "
            "distinct from the IoU>=0.5 used for the AP/P/R numbers above."
        ),
    }


def _fp_fn_from_matrix(matrix: list[list[int]], labels: list[str]) -> dict[str, dict[str, int]]:
    """Derive per-class TP / FP / FN from a [pred, true] confusion matrix.

    ``FP_background`` (predicted a class where the truth is background) isolates
    hallucinations; ``FN_missed`` (truth is a class, predicted background)
    isolates pure misses — the two error modes the report's analysis needs.
    """
    m = np.asarray(matrix)
    out: dict[str, dict[str, int]] = {}
    n_real = len(labels) - 1  # drop the trailing background row/col
    for c in range(n_real):
        tp = int(m[c, c])
        out[labels[c]] = {
            "TP": tp,
            "FP": int(m[c, :].sum() - tp),
            "FN": int(m[:, c].sum() - tp),
            "FP_background": int(m[c, n_real]),
            "FN_missed": int(m[n_real, c]),
        }
    return out


def _pr_curves_from_box(box: Any, n_points: int = 50) -> dict[str, Any] | None:
    """Per-class precision/recall-vs-confidence curves + the conf=0.5 operating point.

    Built from ``p_curve`` / ``r_curve`` (shape (nc, 1000), populated when
    ``plots=True``) rather than ``prec_values`` (which Ultralytics returns with an
    irregular row count). The confidence axis is downsampled to ``n_points`` so the
    document stays small; the operating point is read at the index nearest conf=0.5.
    """
    ap_idx = getattr(box, "ap_class_index", None)
    px = getattr(box, "px", None)
    p_curve = getattr(box, "p_curve", None)
    r_curve = getattr(box, "r_curve", None)
    if ap_idx is None or px is None or p_curve is None or r_curve is None:
        return None
    px = np.asarray(px)
    p_curve = np.asarray(p_curve)
    r_curve = np.asarray(r_curve)
    if px.size == 0 or p_curve.ndim != 2 or p_curve.shape[0] == 0:
        return None

    idx = np.unique(np.linspace(0, len(px) - 1, min(n_points, len(px))).astype(int))
    op_i = int(np.argmin(np.abs(px - DEFAULT_CONF)))  # confidence nearest 0.5

    out: dict[str, Any] = {
        "confidence": [round(float(px[k]), 4) for k in idx],
        "operating_point_conf": DEFAULT_CONF,
        "per_class": {},
    }
    for row, cls_idx in enumerate(ap_idx):
        if row >= p_curve.shape[0] or row >= r_curve.shape[0]:
            break
        name = CLASS_NAMES[int(cls_idx)] if int(cls_idx) < len(CLASS_NAMES) else str(cls_idx)
        out["per_class"][name] = {
            "precision": [round(float(p_curve[row, k]), 4) for k in idx],
            "recall": [round(float(r_curve[row, k]), 4) for k in idx],
            "precision_at_op": round(float(p_curve[row, op_i]), 6),
            "recall_at_op": round(float(r_curve[row, op_i]), 6),
        }
    return out


# ---------------------------------------------------------------------------
# Helpers — localization error (R2): IoU distribution of matched detections
# ---------------------------------------------------------------------------

POOR_BOX_IOU = 0.1   # below this, a detection is unrelated to the GT, not a "near miss"
TP_IOU = 0.5         # IoU >= this counts as a localized hit (CRDDC2022 protocol)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes. Returns shape (len(a), len(b))."""
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _load_gt_boxes(img_path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Load YOLO-format ground-truth boxes for an image as (classes, xyxy-pixels).

    The label file is the sibling ``labels/<stem>.txt`` of the ``images/`` dir.
    """
    p = Path(img_path)
    label_path = p.parent.parent / "labels" / f"{p.stem}.txt"
    if not label_path.exists():
        return np.zeros((0,), dtype=int), np.zeros((0, 4))
    cls: list[int] = []
    box: list[list[float]] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        box.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        cls.append(c)
    return np.asarray(cls, dtype=int), np.asarray(box, dtype=float).reshape(-1, 4)


def _localization_error(
    model: YOLO,
    image_lists: dict[str, list[Path]],
    conf: float,
    device: str,
    sample_per_country: int,
) -> dict[str, Any]:
    """Distribution of IoU between class-correct predictions and ground-truth boxes.

    For a capped sample per country, every prediction (conf >= 0.5) that has a
    same-class GT box with IoU >= POOR_BOX_IOU is counted as a "class-correct"
    detection and its best same-class IoU is recorded. The share whose best IoU
    is in [POOR_BOX_IOU, 0.5) is the "right class but poor box" rate — detections
    the model got conceptually right but localized too loosely to score as TP at
    IoU >= 0.5. Sampling caps are reported, never silently applied.
    """
    print("\n--- Localization error (IoU of matched detections) ---")
    all_ious: list[float] = []
    per_class: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "poor": 0, "ious": []})
    caps: dict[str, dict[str, int]] = {}
    n_imgs = 0

    for country in sorted(k for k in image_lists if k != "_all"):
        paths = image_lists[country]
        use = paths[:sample_per_country]
        caps[country] = {"available": len(paths), "used": len(use)}
        if len(paths) > len(use):
            print(f"  [{country}] sampling {len(use)}/{len(paths)} images for localization")
        for img_path in use:
            res = model.predict(
                source=str(img_path), imgsz=640, conf=conf, device=device,
                verbose=False, save=False,
            )
            if not res:
                continue
            n_imgs += 1
            r0 = res[0]
            h, w = r0.orig_shape
            gt_cls, gt_box = _load_gt_boxes(img_path, w, h)
            boxes = r0.boxes
            if boxes is None or len(boxes) == 0 or gt_box.shape[0] == 0:
                continue
            pb = boxes.xyxy.cpu().numpy()
            pc = boxes.cls.cpu().numpy().astype(int)
            iou = _iou_xyxy(pb, gt_box)
            for i in range(pb.shape[0]):
                same = np.where(gt_cls == pc[i])[0]
                if same.size == 0:
                    continue  # predicted a class with no GT instance -> class FP, not localization
                best = float(iou[i, same].max())
                if best < POOR_BOX_IOU:
                    continue  # unrelated box -> a hallucination, not a near-miss
                name = CLASS_NAMES[pc[i]] if pc[i] < len(CLASS_NAMES) else str(pc[i])
                per_class[name]["n"] += 1
                per_class[name]["ious"].append(round(best, 4))
                all_ious.append(best)
                if best < TP_IOU:
                    per_class[name]["poor"] += 1

    arr = np.asarray(all_ious)
    bin_edges = np.round(np.arange(POOR_BOX_IOU, 1.0001, 0.1), 2)
    counts, _ = np.histogram(arr, bins=bin_edges) if arr.size else (np.zeros(len(bin_edges) - 1, int), None)

    pc_out: dict[str, dict[str, float]] = {}
    for name, d in per_class.items():
        n = d["n"]
        ious = np.asarray(d["ious"])
        pc_out[name] = {
            "n": n,
            "poor": d["poor"],
            "poor_box_share": round(d["poor"] / n, 4) if n else 0.0,
            "mean_iou": round(float(ious.mean()), 4) if n else 0.0,
        }

    return {
        "sampled_images": n_imgs,
        "sample_per_country": sample_per_country,
        "per_country_cap": caps,
        "poor_box_iou_range": [POOR_BOX_IOU, TP_IOU],
        "n_class_correct": int(arr.size),
        "mean_iou": round(float(arr.mean()), 4) if arr.size else 0.0,
        "median_iou": round(float(np.median(arr)), 4) if arr.size else 0.0,
        "poor_box_share": round(float((arr < TP_IOU).mean()), 4) if arr.size else 0.0,
        "iou_histogram": {
            "bin_edges": [float(x) for x in bin_edges],
            "counts": [int(x) for x in counts],
        },
        "per_class": pc_out,
        "note": (
            "Best same-class IoU per prediction (conf>=0.5). 'poor_box_share' = "
            "fraction with right class but IoU in [0.1, 0.5) — localized too loosely "
            "to count as a hit at IoU>=0.5."
        ),
    }


# ---------------------------------------------------------------------------
# Val evaluation (quantitative — has ground truth)
# ---------------------------------------------------------------------------


def _evaluate_val(
    model: YOLO,
    rdd_root: str,
    conf: float,
    iou: float,
    device: str,
    run_id: str,
    loc_sample: int = 100,
) -> dict[str, Any]:
    """Run val() on the validation split and return the evaluation document.

    ``plots=True`` on every ``val()`` call is required: Ultralytics only
    populates the confusion matrix and the precision/recall curves when plotting
    is enabled. The overall run's plots (confusion_matrix.png, PR_curve.png, ...)
    are kept under ``outputs/eval_<run_id>/`` as report artefacts; the per-country
    plots go to a temp dir and are discarded — only their confusion-matrix counts
    are extracted for the per-country FP/FN breakdown.
    """
    print("\n--- Quantitative evaluation on validation set ---")
    image_lists = _build_image_lists(rdd_root, "val")

    plots_dir = OUTPUT_DIR / f"eval_{run_id}"

    with tempfile.TemporaryDirectory(prefix="rdds_eval_val_") as tmp_dir:
        tmp = Path(tmp_dir)

        # Overall — plots persisted as report artefacts.
        print("\n  >> All countries ...")
        all_list = tmp / "val_all.txt"
        all_yaml = tmp / "data_all.yaml"
        _write_image_list(image_lists["_all"], all_list)
        _write_val_yaml(all_list, all_yaml)
        overall_results = model.val(
            data=str(all_yaml), imgsz=640, conf=conf, iou=iou,
            split="val", device=device, verbose=False, save=False, plots=True,
            workers=0, batch=8, project=str(plots_dir), name="overall", exist_ok=True,
        )
        overall_metrics = _extract_metrics(overall_results)
        confusion = _confusion_matrix_doc(overall_results)
        pr_curves = _pr_curves_from_box(getattr(overall_results, "box", None))
        fp_fn_overall = (
            _fp_fn_from_matrix(confusion["matrix"], confusion["labels"]) if confusion else {}
        )
        plot_paths = _collect_plot_paths(overall_results)

        # Per country — plots into temp (discarded), confusion matrix extracted.
        per_country_metrics: dict[str, dict[str, Any]] = {}
        fp_fn_per_country: dict[str, dict[str, dict[str, int]]] = {}
        for country in sorted(k for k in image_lists if k != "_all"):
            print(f"\n  >> {country} ({len(image_lists[country])} images) ...")
            c_list = tmp / f"val_{country}.txt"
            c_yaml = tmp / f"data_{country}.yaml"
            _write_image_list(image_lists[country], c_list)
            _write_val_yaml(c_list, c_yaml)
            c_results = model.val(
                data=str(c_yaml), imgsz=640, conf=conf, iou=iou,
                split="val", device=device, verbose=False, save=False, plots=True,
                workers=0, batch=8, project=str(tmp), name=f"country_{country}", exist_ok=True,
            )
            per_country_metrics[country] = _extract_metrics(c_results)
            c_conf = _confusion_matrix_doc(c_results)
            if c_conf:
                fp_fn_per_country[country] = _fp_fn_from_matrix(c_conf["matrix"], c_conf["labels"])

    per_country_f1 = {c: m.get("F1", 0.0) for c, m in per_country_metrics.items()}
    per_country_map50 = {c: m.get("mAP50", 0.0) for c, m in per_country_metrics.items()}

    localization = _localization_error(model, image_lists, conf, device, loc_sample)

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
        "mAP50_95_overall": overall_metrics.get("mAP50-95"),
        "F1_per_class": overall_metrics.get("F1_per_class", {}),
        "per_class": overall_metrics.get("per_class", {}),
        "F1_per_country": per_country_f1,
        "mAP50_per_country": per_country_map50,
        "confusion_matrix": confusion,
        "fp_fn": {"overall": fp_fn_overall, "per_country": fp_fn_per_country},
        "pr_curves": pr_curves,
        "localization": localization,
        "plots": plot_paths,
        "n_images": len(image_lists["_all"]),
        "iou_threshold": iou,
        "confidence_threshold": conf,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _collect_plot_paths(val_results: Any) -> dict[str, str]:
    """Map known Ultralytics plot filenames in the run's save_dir to their paths."""
    save_dir = getattr(val_results, "save_dir", None)
    if not save_dir:
        return {}
    d = Path(save_dir)
    wanted = {
        "confusion_matrix_png": "confusion_matrix.png",
        "confusion_matrix_normalized_png": "confusion_matrix_normalized.png",
        "pr_curve_png": "PR_curve.png",
        "f1_curve_png": "F1_curve.png",
    }
    return {key: str(d / fn) for key, fn in wanted.items() if (d / fn).exists()}


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
    loc_sample: int = 100,
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
        loc_sample: Max images per country sampled for the localization-error
            pass (IoU distribution of matched detections).

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
    val_doc = _evaluate_val(model, rdd_root, conf, iou, device, target_run_id, loc_sample)
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
        if val_doc.get("mAP50_95_overall") is not None:
            update["metrics.mAP50-95"] = val_doc["mAP50_95_overall"]
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
    if doc.get("mAP50_95_overall") is not None:
        print(f"  mAP@0.5:0.95:      {doc['mAP50_95_overall']:.4f}")
    print()
    print("  F1 per country:")
    for country, f1 in sorted(doc.get("F1_per_country", {}).items()):
        print(f"    {country:<20} {f1:.4f}")

    per_class = doc.get("per_class") or {}
    if per_class:
        print()
        print("  Per class (P / R / AP50 / AP50-95 / F1):")
        print(f"    {'class':<8}{'P':>8}{'R':>8}{'AP50':>8}{'AP5095':>9}{'F1':>8}")
        for cls in sorted(per_class):
            m = per_class[cls]
            print(
                f"    {cls:<8}{m.get('precision', 0):>8.3f}{m.get('recall', 0):>8.3f}"
                f"{m.get('AP50', 0):>8.3f}{m.get('AP50_95', 0):>9.3f}{m.get('F1', 0):>8.3f}"
            )

    fp_fn = (doc.get("fp_fn") or {}).get("overall") or {}
    if fp_fn:
        print()
        print("  FP/FN breakdown (from confusion matrix @ conf=0.5):")
        print(f"    {'class':<8}{'TP':>7}{'FP':>7}{'FN':>7}{'FP_bg':>8}{'FN_miss':>9}")
        for cls in sorted(fp_fn):
            e = fp_fn[cls]
            print(
                f"    {cls:<8}{e['TP']:>7}{e['FP']:>7}{e['FN']:>7}"
                f"{e['FP_background']:>8}{e['FN_missed']:>9}"
            )

    loc = doc.get("localization") or {}
    if loc.get("n_class_correct"):
        print()
        print(
            f"  Localization ({loc['sampled_images']} imgs sampled, "
            f"{loc['n_class_correct']} class-correct detections):"
        )
        print(f"    mean IoU:        {loc.get('mean_iou', 0):.3f}")
        print(f"    median IoU:      {loc.get('median_iou', 0):.3f}")
        print(
            f"    poor-box share:  {loc.get('poor_box_share', 0):.1%}  "
            "(right class, IoU in [0.1, 0.5))"
        )
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
    parser.add_argument(
        "--loc-sample", type=int, default=100,
        help="Max images per country sampled for the localization-error pass (default: 100).",
    )
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
        loc_sample=args.loc_sample,
    )
