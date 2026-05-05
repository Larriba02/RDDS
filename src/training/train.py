"""
src/training/train.py
---------------------
Phase 0 / Phase 1 training script for RDDS.

Workflow
--------
1. Read ``logs/splits.json`` (written by split.py).
2. Build the training image list:
   - All images where split == "val"  ->fixed val set (never subsampled).
   - All images where split == "train" ->candidate pool.
   - Subsample the candidate pool at ``--sample-ratio`` stratified by
     country using per-country proportional sampling with RANDOM_SEED=42.
3. Write per-run image list files:
   - ``logs/train_images_{run_id}.txt``
   - ``logs/val_images_{run_id}.txt``
4. Generate a temporary ``data.yaml`` pointing to those files.
5. Write the initial MongoDB ``experiments`` document (status="running").
6. Start Ultralytics YOLO11 training.
7. Extract metrics from ``results.csv``.
8. Update MongoDB immediately (status="completed", metrics) — before any
   export/upload so metrics are never lost if later steps crash.
9. Export ``best.pt`` ->``best.onnx`` in a subprocess (crash-safe).
10. Upload ``best.pt``, ``last.pt``, ``best.onnx`` to Backblaze B2.
11. Update MongoDB with checkpoint URLs (if upload succeeded).
12. Log the run to MLflow.
13. Call ``promote.maybe_promote`` to conditionally flip is_production.

Non-negotiable rules (CLAUDE.md §2)
------------------------------------
- RANDOM_SEED = 42 in every training run.
- Val set is ALWAYS 100% of split=="val" images — never subsampled.
- Test split is never touched by this script.
- Every training run writes to MongoDB before, during, and after.
- Checkpoints are uploaded to Backblaze immediately after training.

Usage
-----
python -m src.training.train \\
    --model yolo11s \\
    --sample-ratio 0.10 \\
    --epochs 50 \\
    --batch 8 \\
    --patience 15

python -m src.training.train \\
    --model yolo11m \\
    --sample-ratio 1.0 \\
    --epochs 100 \\
    --batch 32 \\
    --patience 20
"""

from __future__ import annotations

import argparse
import json
import locale
import math
import os
import random
import signal
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from src.db.connection import get_db
from src.training.promote import maybe_promote
from src.training.upload_checkpoint import upload_checkpoints

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
CLASS_NAMES = ["D00", "D10", "D20", "D40"]

LOG_DIR = Path("logs")
SPLITS_JSON = LOG_DIR / "splits.json"
CLASS_DIST_JSON = LOG_DIR / "class_distribution.json"

RUNS_DIR = Path("runs") / "train"


# ---------------------------------------------------------------------------
# Helpers — splits and image lists
# ---------------------------------------------------------------------------


def _load_splits() -> dict[str, str]:
    """Load image_id ->split mapping from logs/splits.json.

    Returns:
        Dict mapping image_id to split string.

    Raises:
        FileNotFoundError: If splits.json does not exist.
    """
    if not SPLITS_JSON.exists():
        raise FileNotFoundError(
            f"{SPLITS_JSON} not found. Run 'python -m src.data.split' first."
        )
    with open(SPLITS_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def _load_image_metadata() -> dict[str, dict[str, Any]]:
    """Pull image filepath + country from MongoDB images_metadata.

    Returns:
        Dict mapping image_id ->{"filepath": str, "country": str}.

    Raises:
        RuntimeError: If the collection is empty (ingest not run yet).
    """
    db = get_db()
    col = db["images_metadata"]
    # Fetch only the fields we need — projection avoids pulling annotations.
    cursor = col.find({}, {"image_id": 1, "filepath": 1, "country": 1, "_id": 0})
    result: dict[str, dict[str, Any]] = {}
    for doc in cursor:
        result[doc["image_id"]] = {
            "filepath": doc["filepath"],
            "country": doc["country"],
        }
    if not result:
        raise RuntimeError(
            "images_metadata collection is empty. "
            "Run 'python -m src.data.ingest' first."
        )
    return result


def _subsample_train_ids(
    train_ids: list[str],
    metadata: dict[str, dict[str, Any]],
    sample_ratio: float,
) -> list[str]:
    """Subsample training image IDs stratified by country.

    Each country contributes ``round(len(country_ids) * sample_ratio)``
    images, sampled without replacement using RANDOM_SEED.

    Args:
        train_ids: Full candidate pool (split == "train").
        metadata: image_id ->{filepath, country} from MongoDB.
        sample_ratio: Fraction of each country's images to include (0, 1].

    Returns:
        Sorted list of sampled image_ids.
    """
    if sample_ratio >= 1.0:
        return sorted(train_ids)

    # Group by country.
    by_country: dict[str, list[str]] = defaultdict(list)
    missing_meta = 0
    for img_id in train_ids:
        meta = metadata.get(img_id)
        if meta is None:
            missing_meta += 1
            continue
        by_country[meta["country"]].append(img_id)

    if missing_meta:
        print(
            f"  [warn] {missing_meta} train image_ids have no MongoDB metadata — skipped."
        )

    rng = random.Random(RANDOM_SEED)
    sampled: list[str] = []
    for country in sorted(by_country.keys()):
        ids = sorted(by_country[country])  # deterministic base order
        rng.shuffle(ids)
        n = max(1, math.ceil(len(ids) * sample_ratio))
        sampled.extend(ids[:n])
        print(f"  Country {country}: {len(ids)} ->{n} images sampled.")

    return sorted(sampled)


def _resolve_image_path(img_id: str, metadata: dict[str, dict[str, Any]]) -> Path | None:
    """Resolve the absolute filesystem path for an image_id.

    The relative filepath stored in MongoDB is joined with RDD_DATA_ROOT.

    Args:
        img_id: image_id to resolve.
        metadata: image_id ->{filepath, country} dict.

    Returns:
        Absolute Path or None if the metadata or env var is missing.
    """
    rdd_root = os.getenv("RDD_DATA_ROOT")
    if not rdd_root:
        return None
    meta = metadata.get(img_id)
    if meta is None:
        return None
    return Path(rdd_root) / meta["filepath"]


def _write_image_list(image_ids: list[str], metadata: dict[str, dict[str, Any]], path: Path) -> int:
    """Write a text file listing one absolute image path per line.

    Lines for images whose paths cannot be resolved are skipped.

    Args:
        image_ids: Ordered list of image_ids to write.
        metadata: image_id ->{filepath, country} dict.
        path: Output .txt file path.

    Returns:
        Number of paths written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for img_id in image_ids:
        abs_path = _resolve_image_path(img_id, metadata)
        if abs_path is not None:
            lines.append(str(abs_path))
    # Use the system preferred encoding so Ultralytics (which calls open() without
    # an explicit encoding) can read the file correctly on both Windows (cp1252)
    # and Linux/macOS (utf-8).
    enc = locale.getpreferredencoding(False)
    path.write_text("\n".join(lines) + "\n", encoding=enc)
    return len(lines)


# ---------------------------------------------------------------------------
# Helpers — class weights
# ---------------------------------------------------------------------------


def _compute_cls_weights() -> list[float] | None:
    """Compute per-class loss weights from logs/class_distribution.json.

    Inverse-frequency weighting: weight_i = max_count / count_i.  Weights
    are normalised so the minimum is 1.0.

    Returns:
        List of 4 floats [w_D00, w_D10, w_D20, w_D40], or None if the file
        is absent (safe fallback — Ultralytics uses uniform weights).
    """
    if not CLASS_DIST_JSON.exists():
        print(
            f"  [warn] {CLASS_DIST_JSON} not found. "
            "Skipping cls_weight calibration — run analyse_distribution.py."
        )
        return None

    with open(CLASS_DIST_JSON, encoding="utf-8") as fh:
        dist = json.load(fh)

    # dist is expected to have a "total" key or per-country keys;
    # we aggregate across all countries.
    totals: dict[str, int] = defaultdict(int)
    for country_data in dist.values():
        if isinstance(country_data, dict):
            for cls_name, count in country_data.items():
                if cls_name in CLASS_NAMES and isinstance(count, int):
                    totals[cls_name] += count

    if not totals:
        print("  [warn] class_distribution.json has unexpected structure -- skipping cls_weight.")
        return None

    counts = [totals.get(c, 1) for c in CLASS_NAMES]
    max_count = max(counts)
    weights = [max_count / max(c, 1) for c in counts]
    # Normalise so min weight is 1.0.
    min_w = min(weights)
    weights = [w / min_w for w in weights]
    print(f"  Class weights (D00, D10, D20, D40): {[round(w, 3) for w in weights]}")
    return weights


# ---------------------------------------------------------------------------
# Helpers — MongoDB
# ---------------------------------------------------------------------------


def _make_run_id(model: str) -> str:
    """Generate a unique run identifier.

    Format: ``run_YYYYMMDD_HHMMSS_<model>``.

    Args:
        model: Model name, e.g. ``yolo11s``.

    Returns:
        Run ID string.
    """
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{now}_{model}"


def _write_initial_mongo_doc(
    run_id: str,
    model: str,
    sample_ratio: float,
    training_image_ids: list[str],
    hyperparams: dict[str, Any],
) -> None:
    """Insert the initial experiments document with status="running".

    Args:
        run_id: Unique run identifier.
        model: Model name string.
        sample_ratio: Fraction of train data used.
        training_image_ids: List of image_ids selected for training.
        hyperparams: Dict of training hyperparameters.
    """
    db = get_db()
    doc: dict[str, Any] = {
        "run_id": run_id,
        "model": model,
        "model_version": None,  # filled in after promotion
        "status": "running",
        "is_production": False,
        "sample_ratio": sample_ratio,
        "training_image_ids": training_image_ids,
        "dataset_countries": sorted(
            set(hyperparams.pop("_countries", []))
        ),
        "hyperparams": hyperparams,
        "metrics": {},
        "checkpoints": {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    db["experiments"].insert_one(doc)
    print(f"  MongoDB: initial doc inserted for {run_id}.")


def _update_mongo_doc(run_id: str, update: dict[str, Any]) -> None:
    """Apply a $set update to the experiments document for this run.

    Args:
        run_id: Unique run identifier.
        update: Dict of fields to set.
    """
    db = get_db()
    db["experiments"].update_one({"run_id": run_id}, {"$set": update})


# ---------------------------------------------------------------------------
# Helpers — data.yaml
# ---------------------------------------------------------------------------


def _write_data_yaml(
    train_list_path: Path,
    val_list_path: Path,
    yaml_path: Path,
) -> None:
    """Write a Ultralytics-compatible data.yaml file.

    Args:
        train_list_path: Absolute path to the train image list .txt file.
        val_list_path: Absolute path to the val image list .txt file.
        yaml_path: Destination .yaml file path.
    """
    data = {
        "train": str(train_list_path.resolve()),
        "val": str(val_list_path.resolve()),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
    print(f"  data.yaml written to {yaml_path}")


# ---------------------------------------------------------------------------
# Helpers — metrics extraction
# ---------------------------------------------------------------------------


def _extract_metrics(run_dir: Path) -> dict[str, float]:
    """Read Ultralytics results.csv and extract final-epoch metrics.

    Columns produced by Ultralytics (>=8.x) include::

        metrics/precision(B), metrics/recall(B), metrics/mAP50(B),
        metrics/mAP50-95(B)

    F1 is computed as the harmonic mean of final precision and recall.

    Args:
        run_dir: Ultralytics output directory (contains results.csv).

    Returns:
        Dict with keys: mAP50, precision, recall, F1.  Empty dict if
        results.csv cannot be parsed.
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"  [warn] results.csv not found in {run_dir}.")
        return {}

    try:
        import csv

        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        if not rows:
            return {}

        # Last row is the final (or best-stopping) epoch.
        last = rows[-1]

        def _get(col_fragment: str) -> float | None:
            for key, val in last.items():
                if col_fragment in key.strip():
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return None
            return None

        precision = _get("precision")
        recall = _get("recall")
        map50 = _get("mAP50(B)") or _get("mAP50")

        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None

        metrics: dict[str, float] = {}
        if map50 is not None:
            metrics["mAP50"] = round(map50, 6)
        if precision is not None:
            metrics["precision"] = round(precision, 6)
        if recall is not None:
            metrics["recall"] = round(recall, 6)
        if f1 is not None:
            metrics["F1"] = round(f1, 6)

        return metrics

    except Exception as exc:
        print(f"  [warn] Could not parse results.csv: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Core train function
# ---------------------------------------------------------------------------


def train(
    model: str = "yolo11s",
    sample_ratio: float = 0.10,
    epochs: int = 50,
    batch: int = 8,
    patience: int = 15,
    imgsz: int = 640,
    amp: bool = True,
    lr0: float = 0.01,
    lrf: float = 0.01,
    cos_lr: bool = False,
    optimizer: str = "auto",
    cache: str | bool = False,
    workers: int = 8,
    skip_upload: bool = False,
    skip_promote: bool = False,
) -> str:
    """Run a full training cycle and return the run_id.

    Args:
        model: Ultralytics model name, e.g. ``yolo11s`` or ``yolo11m``.
        sample_ratio: Fraction of training images to use (0 < ratio <= 1.0).
            Subsampled per country, stratified.
        epochs: Maximum training epochs.
        batch: Batch size.
        patience: Early-stopping patience (epochs without mAP improvement).
        imgsz: Input image size in pixels.
        amp: Enable FP16 mixed-precision training.
        skip_upload: If True, skip B2 upload (useful for smoke tests without
            real B2 credentials).
        skip_promote: If True, skip the promotion step (useful for testing).

    Returns:
        run_id string of the completed experiment.

    Raises:
        FileNotFoundError: If splits.json or the dataset is missing.
        RuntimeError: If training or MongoDB writes fail.
    """
    # ------------------------------------------------------------------
    # 0. Pre-flight checks
    # ------------------------------------------------------------------
    if not os.getenv("RDD_DATA_ROOT"):
        raise EnvironmentError(
            "RDD_DATA_ROOT is not set. Set it in .env or the environment."
        )

    # ------------------------------------------------------------------
    # 1. Build run_id and prepare log directory
    # ------------------------------------------------------------------
    run_id = _make_run_id(model)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"RDDS Training -- run_id: {run_id}")
    print(f"  model={model}  sample_ratio={sample_ratio}  epochs={epochs}")
    print(f"  batch={batch}  imgsz={imgsz}  amp={amp}  patience={patience}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 2. Load splits and metadata
    # ------------------------------------------------------------------
    print("Loading splits ...")
    splits = _load_splits()

    print("Loading image metadata from MongoDB ...")
    metadata = _load_image_metadata()

    # Separate train/val pools.
    all_train_ids = [img_id for img_id, sp in splits.items() if sp == "train"]
    all_val_ids = [img_id for img_id, sp in splits.items() if sp == "val"]

    print(f"  Full train pool: {len(all_train_ids)} images")
    print(f"  Fixed val set:   {len(all_val_ids)} images")

    # ------------------------------------------------------------------
    # 3. Subsample training set at sample_ratio
    # ------------------------------------------------------------------
    print(f"\nSubsampling training set at ratio={sample_ratio} ...")
    selected_train_ids = _subsample_train_ids(all_train_ids, metadata, sample_ratio)
    print(f"  Selected for training: {len(selected_train_ids)} images")

    # Countries present in the training subset.
    countries = sorted(
        {metadata[img_id]["country"] for img_id in selected_train_ids if img_id in metadata}
    )

    # ------------------------------------------------------------------
    # 4. Write per-run image list files and data.yaml
    # ------------------------------------------------------------------
    train_list = LOG_DIR / f"train_images_{run_id}.txt"
    val_list = LOG_DIR / f"val_images_{run_id}.txt"
    data_yaml = LOG_DIR / f"data_{run_id}.yaml"

    n_train_written = _write_image_list(selected_train_ids, metadata, train_list)
    n_val_written = _write_image_list(all_val_ids, metadata, val_list)
    print(f"  Train list: {n_train_written} paths ->{train_list}")
    print(f"  Val list:   {n_val_written} paths ->{val_list}")

    if n_val_written == 0:
        if n_train_written == 0:
            raise RuntimeError(
                "Both train and val image lists are empty. "
                "Check that RDD_DATA_ROOT is correct and ingest.py has been run."
            )
        # Smoke-test fallback: tiny dataset has too few images per country to
        # produce a val set (< 1000 per country threshold in split.py).
        print("  [warn] val set is empty -- using train set as val (smoke-test only).")
        effective_val_list = train_list
    else:
        effective_val_list = val_list
    _write_data_yaml(train_list, effective_val_list, data_yaml)

    # ------------------------------------------------------------------
    # 5. Class weights
    # ------------------------------------------------------------------
    cls_weights = _compute_cls_weights()

    # ------------------------------------------------------------------
    # 6. Write initial MongoDB document (status="running")
    # ------------------------------------------------------------------
    hyperparams: dict[str, Any] = {
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "seed": RANDOM_SEED,
        "patience": patience,
        "amp": amp,
        "lr0": lr0,
        "lrf": lrf,
        "cos_lr": cos_lr,
        "optimizer": optimizer,
        "cache": str(cache),
        "workers": workers,
        "cls_weight": cls_weights,
        "_countries": countries,  # popped inside _write_initial_mongo_doc
    }
    print("\nWriting initial MongoDB document ...")
    _write_initial_mongo_doc(
        run_id=run_id,
        model=model,
        sample_ratio=sample_ratio,
        training_image_ids=selected_train_ids,
        hyperparams=dict(hyperparams),  # copy before _countries is popped
    )

    # ------------------------------------------------------------------
    # 7. Ultralytics training
    # ------------------------------------------------------------------
    project_dir = str(RUNS_DIR)
    yolo = YOLO(f"{model}.pt")  # downloads pretrained weights if absent

    # Note on cls_weights: Ultralytics 8.x ``cls`` hyperparameter is a single
    # float that scales the *global* classification loss gain (default 0.5).
    # It does NOT accept a per-class weight vector via the train() API.
    # Per-class weighting would require a custom loss callback and is
    # deferred to a Phase 1 enhancement.  The computed cls_weights are
    # stored in MongoDB for documentation and future reference only.
    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml.resolve()),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "patience": patience,
        "amp": amp,
        "lr0": lr0,
        "lrf": lrf,
        "cos_lr": cos_lr,
        "optimizer": optimizer,
        "cache": cache,
        "workers": workers,
        "seed": RANDOM_SEED,
        "project": project_dir,
        "name": run_id,
        "exist_ok": False,
        "verbose": True,
        "plots": True,
        "save": True,
    }

    # Ultralytics' MLflow callback reads MLFLOW_TRACKING_URI from the env.
    # If unset it defaults to trainer.save_dir.parents[1]/mlflow, a bare
    # Windows path (scheme "C") that MLflow rejects as unsupported.
    # Use pathlib.as_uri() to produce a proper file:/// URI that works on
    # both Windows and Linux.
    os.environ["MLFLOW_TRACKING_URI"] = Path("mlruns").resolve().as_uri()

    # Catch SIGTERM (e.g. SLURM wall-clock kill) and write interrupted status.
    def _sigterm_handler(signum, frame):  # noqa: ANN001
        _update_mongo_doc(run_id, {"status": "interrupted"})
        raise SystemExit(f"SIGTERM received — run {run_id} marked interrupted.")

    signal.signal(signal.SIGTERM, _sigterm_handler)

    print(f"\nStarting Ultralytics training ...")
    try:
        results = yolo.train(**train_kwargs)
    except Exception as exc:
        _update_mongo_doc(run_id, {"status": "failed", "error": str(exc)})
        raise RuntimeError(f"Training failed for run {run_id}: {exc}") from exc

    run_dir = RUNS_DIR / run_id

    # ------------------------------------------------------------------
    # 8. Extract metrics from results.csv
    # ------------------------------------------------------------------
    print("\nExtracting metrics ...")
    metrics = _extract_metrics(run_dir)
    print(f"  Metrics: {metrics}")

    # ------------------------------------------------------------------
    # 9. Update MongoDB with metrics immediately — before any export/upload
    #    so the document is never left in "running" state if later steps crash.
    # ------------------------------------------------------------------
    print("\nUpdating MongoDB document with final metrics ...")
    _update_mongo_doc(
        run_id,
        {
            "status": "completed",
            "metrics": metrics,
            "checkpoints": {},
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "dataset_countries": countries,
            "sample_ratio": sample_ratio,
            "hyperparams": {
                "epochs": epochs,
                "batch": batch,
                "imgsz": imgsz,
                "seed": RANDOM_SEED,
                "patience": patience,
                "amp": amp,
                "lr0": lr0,
                "lrf": lrf,
                "cos_lr": cos_lr,
                "optimizer": optimizer,
                "cache": str(cache),
                "workers": workers,
                "cls_weight": cls_weights,
            },
        },
    )

    # ------------------------------------------------------------------
    # 10. Export best.pt ->best.onnx
    #     Run in a subprocess so a crash (e.g. onnxslim segfault) cannot
    #     kill the main process and lose the MongoDB update above.
    # ------------------------------------------------------------------
    best_pt = run_dir / "weights" / "best.pt"
    best_onnx = run_dir / "weights" / "best.onnx"
    if best_pt.exists():
        print("\nExporting best.pt ->best.onnx ...")
        export_cmd = (
            f"from ultralytics import YOLO; "
            f"YOLO(r'{best_pt}').export(format='onnx', imgsz={imgsz}, simplify=True)"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", export_cmd],
                stdout=subprocess.DEVNULL,   # discard ANSI-heavy Ultralytics output
                stderr=subprocess.PIPE,       # capture errors in binary to avoid cp1252 issues
                timeout=600,
            )
            if result.returncode != 0:
                err_text = result.stderr.decode("utf-8", errors="replace")[:500]
                print(
                    f"  [warn] ONNX export failed (exit {result.returncode}). "
                    f"Continuing without best.onnx.\n{err_text}"
                )
            else:
                print("  ONNX export complete.")
        except subprocess.TimeoutExpired:
            print("  [warn] ONNX export timed out after 600 s. Continuing.")
        except Exception as exc:
            print(f"  [warn] ONNX export error: {exc}. Continuing.")
    else:
        print(f"  [warn] {best_pt} not found -- skipping export.")

    # ------------------------------------------------------------------
    # 11. Upload checkpoints to Backblaze B2
    # ------------------------------------------------------------------
    checkpoint_urls: dict[str, str] = {}
    if not skip_upload:
        print("\nUploading checkpoints to Backblaze B2 ...")
        try:
            checkpoint_urls = upload_checkpoints(run_id=run_id, run_dir=run_dir)
        except Exception as exc:
            print(f"  [warn] Checkpoint upload failed: {exc}. Continuing.")
    else:
        print("\n  [skip] B2 upload skipped (--skip-upload).")

    # Update checkpoint URLs in MongoDB if any were uploaded.
    if checkpoint_urls:
        _update_mongo_doc(run_id, {"checkpoints": checkpoint_urls})

    # ------------------------------------------------------------------
    # 12. MLflow logging  (best_pt already resolved above)
    # ------------------------------------------------------------------
    print("\nLogging to MLflow ...")
    try:
        mlflow.set_experiment("rdds_training")
        with mlflow.start_run(run_name=run_id):
            mlflow.log_params(
                {
                    "model": model,
                    "sample_ratio": sample_ratio,
                    "epochs": epochs,
                    "batch": batch,
                    "imgsz": imgsz,
                    "seed": RANDOM_SEED,
                    "patience": patience,
                    "amp": amp,
                    "n_train_images": len(selected_train_ids),
                    "n_val_images": len(all_val_ids),
                }
            )
            for metric_name, metric_val in metrics.items():
                mlflow.log_metric(metric_name, metric_val)
            if best_pt.exists():
                mlflow.log_artifact(str(best_pt), artifact_path="checkpoints")
    except Exception as exc:
        print(f"  [warn] MLflow logging failed: {exc}. Continuing.")

    # ------------------------------------------------------------------
    # 13. Promote if F1 available
    #
    # WARNING: the F1 used here is the *training-time* value from
    # results.csv (harmonic mean of final-epoch precision and recall on
    # the fixed val set), NOT the CRDDC2022 protocol F1 produced by
    # evaluate.py.  Using training-time F1 is an acceptable proxy during
    # Phase 0 while evaluate.py is not yet implemented.  Before any Phase 1
    # promotion, run evaluate.py and call promote.py manually with the
    # CRDDC2022 F1 value. (CLAUDE.md §3)
    # ------------------------------------------------------------------
    if not skip_promote:
        f1 = metrics.get("F1")
        if f1 is not None:
            print(f"\nRunning promotion check (training-time F1={f1:.4f}) ...")
            print(
                "  [note] This is training-time F1. Before Phase 1 promotion, "
                "use evaluate.py + promote.py with CRDDC2022 F1."
            )
            outcome = maybe_promote(run_id=run_id, f1_new=f1)
            print(f"  Promotion outcome: {outcome}")
        else:
            print(
                "\n  [warn] No F1 metric available — skipping promotion. "
                "Run src/evaluation/evaluate.py to compute F1, then call promote.py manually."
            )
            _update_mongo_doc(run_id, {"status": "completed"})

    print(f"\nTraining complete. run_id={run_id}")
    return run_id


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO11 on RDD2022 with MongoDB + Backblaze integration."
    )
    parser.add_argument(
        "--model",
        default="yolo11s",
        choices=["yolo11s", "yolo11m", "yolo11l", "yolo11x"],
        help="Ultralytics model variant (default: yolo11s).",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=None,
        help="Fraction of training images to use (default: SAMPLE_RATIO from .env).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stopping patience (default: 15).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size in pixels (default: 640).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable FP16 mixed-precision training.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final learning rate as a fraction of lr0 (default: 0.01).",
    )
    parser.add_argument(
        "--cos-lr",
        action="store_true",
        help="Use cosine learning rate schedule instead of linear.",
    )
    parser.add_argument(
        "--optimizer",
        default="auto",
        choices=["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"],
        help="Optimizer (default: auto — Ultralytics selects SGD for YOLO).",
    )
    parser.add_argument(
        "--cache",
        default="False",
        choices=["False", "ram", "disk"],
        help=(
            "Cache images for faster training. 'ram' keeps them in memory "
            "(needs ~4 GB per 3000 images), 'disk' caches as .npy files. "
            "Default: False (re-read from disk each epoch)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of data-loading worker threads (default: 8).",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip Backblaze B2 upload (for local testing).",
    )
    parser.add_argument(
        "--skip-promote",
        action="store_true",
        help="Skip the promotion step (for local testing).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    sample_ratio = args.sample_ratio
    if sample_ratio is None:
        sample_ratio = float(os.getenv("SAMPLE_RATIO", "1.0"))

    # --cache accepts "False" | "ram" | "disk" strings from argparse
    cache_val: str | bool = args.cache
    if cache_val == "False":
        cache_val = False

    train(
        model=args.model,
        sample_ratio=sample_ratio,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        imgsz=args.imgsz,
        amp=not args.no_amp,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        optimizer=args.optimizer,
        cache=cache_val,
        workers=args.workers,
        skip_upload=args.skip_upload,
        skip_promote=args.skip_promote,
    )
