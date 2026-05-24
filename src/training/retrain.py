"""
src/training/retrain.py
-----------------------
Fine-tune the production checkpoint with new images, then evaluate and promote.

Workflow
--------
1.  Validate inputs.  Raise ``EnvironmentError`` immediately if ``RDD_DATA_ROOT``
    is not set (same pre-flight as ``train.py``).
2.  Ingest new images into MongoDB ``images_metadata`` (idempotent, uses
    ``src.data.ingest.ingest``).
3.  Load the production checkpoint from ``runs/`` (fuzzy-match) or fall back
    to downloading from the B2 URL stored in MongoDB.  ``--model`` overrides
    both lookups.
4.  Build mixed training list: new images + a stratified sample of the original
    training pool (controlled by ``--mix-ratio``, default 0.30).  Mixed training
    is **always on** to prevent catastrophic forgetting — the model keeps seeing
    the original distribution while learning from the new data.  Pass
    ``--mix-ratio 0.0`` to disable and train on new images only.
5.  Compose a data.yaml from the mixed image list so Ultralytics fine-tunes from
    the production weights.  Val split uses the fixed val set from
    ``logs/splits.json`` for meaningful early-stopping signal.
    The run_id follows the same ``run_{YYYYMMDD_HHMMSS}_{model}`` convention as
    ``train.py``.
6.  Write a MongoDB ``experiments`` document with ``status="running"`` before
    training starts.  Install a SIGTERM handler (SLURM wall-clock kill) that
    marks the run ``status="interrupted"``.
7.  Fine-tune via Ultralytics ``YOLO.train()``.  Optional fine-tuning knobs:
    ``--lr0``, ``--lrf``, ``--cos-lr``, ``--optimizer``, ``--freeze``.
8.  Extract metrics from ``results.csv`` (same helper as ``train.py``).
9.  Update MongoDB with final metrics immediately (before any export/upload).
10. Export ``best.pt`` → ``best.onnx`` in a subprocess (crash-safe).
11. Upload checkpoints to Backblaze B2 via ``upload_checkpoint.upload_checkpoints``.
12. Run evaluation on the fixed validation set via ``evaluate.evaluate``
    (uses the retrained run_id, not production).
13. Call ``promote.maybe_promote`` with the CRDDC2022 F1 returned by evaluate.
    Promotion rule: F1_new > F1_current + 0.01.
14. Print a human-readable summary.

Non-negotiable rules (CLAUDE.md §2)
------------------------------------
- RANDOM_SEED = 42 in every training run.
- Every training run writes to MongoDB before, during, and after.
- Checkpoints are uploaded to Backblaze immediately after training.
- is_production is toggled atomically via promote.maybe_promote.

Mixed-training rationale
------------------------
Fine-tuning a model on only new images causes catastrophic forgetting: the
network overwrites the weights it learned from tens of thousands of training
images in exchange for a few hundred new ones.  To prevent this, a random
stratified subsample of the original training pool (``--mix-ratio``) is
included alongside the new images.  The default (0.30) means 30% of the
original train split is replayed each retrain run.  Increase for stability,
decrease when new images are many or compute is scarce.

Fine-tuning knobs
-----------------
``--lr0`` / ``--lrf``  Initial and final LR.  Defaults to Ultralytics' built-in
    fine-tuning schedule (lr0=0.01, lrf=0.01), which is already conservative.
    Lower lr0 (e.g. 0.001) for surgical updates on small new-image batches.
``--cos-lr``           Use cosine LR decay instead of linear.  Useful when the
    new dataset is large enough to warrant a full warmup-decay cycle.
``--optimizer``        SGD (default) | Adam | AdamW | auto.
``--freeze N``         Freeze the first N backbone layers.  Recommended values:
    freeze=10 (freeze backbone, train neck+head) for very small new batches;
    freeze=0 (default, train everything) for larger new datasets.  Higher freeze
    means faster training and lower risk of forgetting, at the cost of less
    adaptation capacity.

Usage
-----
# Standard retrain (mixed training on by default):
python -m src.training.retrain \\
    --new-images path/to/new_images/ \\
    --epochs 20 \\
    --batch 8 \\
    --patience 10

# Conservative fine-tune: freeze backbone, low LR, cosine decay:
python -m src.training.retrain \\
    --new-images path/to/new_images/ \\
    --freeze 10 \\
    --lr0 0.001 \\
    --lrf 0.01 \\
    --cos-lr \\
    --optimizer AdamW

# Disable mixed training (new images only):
python -m src.training.retrain \\
    --new-images path/to/new_images/ \\
    --mix-ratio 0.0

# Local checkpoint override (bypass B2 download):
python -m src.training.retrain \\
    --new-images path/to/new_images/ \\
    --model runs/train/myrun/weights/best.pt

# Smoke test (tiny dataset, 1 epoch):
python -m src.training.retrain \\
    --new-images tests/data/tiny_rdd2022/ \\
    --epochs 1 \\
    --batch 2 \\
    --patience 1
"""

from __future__ import annotations

import argparse
import csv
import json
import locale
import os
import random
import signal
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from src.data.ingest import ingest as _ingest_data
from src.db.connection import get_db
from src.evaluation.evaluate import evaluate as _run_evaluate
from src.training.promote import maybe_promote
from src.training.upload_checkpoint import upload_checkpoints

load_dotenv()

# ---------------------------------------------------------------------------
# Constants (mirrors train.py)
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
CLASS_NAMES = ["D00", "D10", "D20", "D40"]

LOG_DIR = Path("logs")
RUNS_DIR = Path("runs") / "train"
SPLITS_JSON = LOG_DIR / "splits.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Default fraction of the original training pool replayed during mixed training.
DEFAULT_MIX_RATIO = 0.30


# ---------------------------------------------------------------------------
# Helpers — run_id
# ---------------------------------------------------------------------------


def _make_run_id(model: str) -> str:
    """Generate a unique run identifier.

    Format: ``run_YYYYMMDD_HHMMSS_<model>_retrain``.

    Args:
        model: Model name, e.g. ``yolo11s``.

    Returns:
        Run ID string.
    """
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{now}_{model}_retrain"


# ---------------------------------------------------------------------------
# Helpers — checkpoint resolution (mirrors evaluate.py / predict.py)
# ---------------------------------------------------------------------------


def _resolve_production_checkpoint(model_override: str | None) -> tuple[Path, str]:
    """Resolve the checkpoint to fine-tune from.

    Resolution order:
        1. ``model_override`` local path (``--model`` flag).
        2. ``runs/train/<run_id>/weights/best.pt`` (exact match).
        3. Any ``runs/train/*/weights/best.pt`` whose dir starts with run_id.
        4. Download ``best.pt`` from the B2 URL stored in MongoDB.

    Args:
        model_override: Optional local path override from ``--model`` CLI flag.

    Returns:
        Tuple of (Path to best.pt, model_name string from the experiment doc).

    Raises:
        RuntimeError: If no checkpoint can be located.
        FileNotFoundError: If the ``--model`` override path does not exist.
    """
    if model_override is not None:
        ckpt = Path(model_override)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"--model path does not exist: {ckpt}"
            )
        print(f"  Checkpoint (--model override): {ckpt}")
        # Still query MongoDB for the model name so the run doc is consistent.
        try:
            db = get_db()
            prod_doc = db["experiments"].find_one({"is_production": True})
            model_name = prod_doc.get("model", "yolo11s") if prod_doc else "yolo11s"
        except Exception:
            model_name = "yolo11s"
        return ckpt, model_name

    # No override — look up the production experiment.
    db = get_db()
    prod_doc = db["experiments"].find_one({"is_production": True})
    if prod_doc is None:
        raise RuntimeError(
            "No production model found (is_production=True). "
            "Train a baseline first or pass --model with a local .pt path."
        )
    run_id = prod_doc["run_id"]
    model_name = prod_doc.get("model", "yolo11s")
    print(f"  Production experiment: {run_id}  ({model_name})")

    direct = RUNS_DIR / run_id / "weights" / "best.pt"
    if direct.exists():
        print(f"  Checkpoint: {direct}")
        return direct, model_name

    if RUNS_DIR.exists():
        for candidate in sorted(RUNS_DIR.iterdir()):
            if candidate.name.startswith(run_id):
                ckpt = candidate / "weights" / "best.pt"
                if ckpt.exists():
                    print(f"  Checkpoint (fuzzy match): {ckpt}")
                    return ckpt, model_name

    b2_url = prod_doc.get("checkpoints", {}).get("best_pt")
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
        return dest, model_name

    raise RuntimeError(
        f"Cannot find best.pt for production run_id='{run_id}'. "
        "Check local runs/ directory or MongoDB checkpoints.best_pt."
    )


# ---------------------------------------------------------------------------
# Helpers — data.yaml for new images
# ---------------------------------------------------------------------------


def _collect_new_image_paths(new_images_dir: Path) -> list[Path]:
    """Walk new_images_dir and collect all image file paths.

    Handles two layouts:
    - Flat: images directly in new_images_dir (or one level of subfolders).
    - RDD2022-style: ``<country>/<split>/images/*.jpg`` hierarchy.

    Args:
        new_images_dir: Root directory supplied via ``--new-images``.

    Returns:
        Sorted list of absolute image Paths.

    Raises:
        ValueError: If no images are found.
    """
    paths: list[Path] = []
    for p in new_images_dir.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    paths.sort()
    if not paths:
        raise ValueError(
            f"No image files found under {new_images_dir}. "
            f"Supported extensions: {sorted(IMAGE_EXTENSIONS)}"
        )
    return paths



def _load_original_splits(
    data_root: Path,
    mix_ratio: float,
) -> tuple[list[Path], list[Path]]:
    """Load original train and val image paths from MongoDB images_metadata.

    ``logs/splits.json`` maps image_id → split label (format: ``{"<md5>":
    "train"|"val"|"test", ...}``).  Full metadata (filepath, country) is
    fetched from MongoDB ``images_metadata``.

    Samples ``mix_ratio`` fraction of the original train split (stratified by
    country, reproducible with RANDOM_SEED) to include in the mixed training
    list.  The full val split is returned for use in the data.yaml val field
    so Ultralytics early-stopping uses the actual held-out set.

    Args:
        data_root: Absolute path to ``RDD_DATA_ROOT``.
        mix_ratio: Fraction of the original train pool to replay (0.0–1.0).

    Returns:
        Tuple of (train_paths, val_paths).  Both may be empty if
        ``splits.json`` is missing, MongoDB is unreachable, or ``mix_ratio``
        is 0.0.
    """
    if mix_ratio <= 0.0:
        return [], []

    if not SPLITS_JSON.exists():
        print(
            f"  [warn] {SPLITS_JSON} not found — mixed training disabled for this run.\n"
            "  Run src/data/split.py first to generate splits.json."
        )
        return [], []

    # splits.json format: {image_id: "train"|"val"|"test"}
    with open(SPLITS_JSON, encoding="utf-8") as fh:
        split_map: dict[str, str] = json.load(fh)

    train_ids = {k for k, v in split_map.items() if v == "train"}
    val_ids = {k for k, v in split_map.items() if v == "val"}

    if not train_ids:
        print("  [warn] No train records found in splits.json — mixed training disabled.")
        return [], []

    # Fetch filepath + country from MongoDB for the relevant image_ids.
    try:
        db = get_db()
        col = db["images_metadata"]
        train_docs = list(col.find(
            {"image_id": {"$in": list(train_ids)}},
            {"_id": 0, "image_id": 1, "filepath": 1, "country": 1},
        ))
        val_docs = list(col.find(
            {"image_id": {"$in": list(val_ids)}},
            {"_id": 0, "image_id": 1, "filepath": 1, "country": 1},
        ))
    except Exception as exc:
        print(
            f"  [warn] MongoDB query failed ({exc}) — mixed training disabled for this run."
        )
        return [], []

    # Stratified subsample of original train by country.
    by_country: dict[str, list[dict]] = {}
    for doc in train_docs:
        by_country.setdefault(doc.get("country", "unknown"), []).append(doc)

    rng = random.Random(RANDOM_SEED)
    sampled: list[dict] = []
    for country_docs in by_country.values():
        k = max(1, round(len(country_docs) * mix_ratio))
        sampled.extend(rng.sample(country_docs, min(k, len(country_docs))))

    def _to_path(doc: dict) -> Path | None:
        fp = doc.get("filepath", "")
        if not fp:
            return None
        p = data_root / fp
        return p if p.exists() else None

    train_paths = [p for doc in sampled if (p := _to_path(doc)) is not None]
    val_paths = [p for doc in val_docs if (p := _to_path(doc)) is not None]

    return train_paths, val_paths


def _write_retrain_data_yaml(
    new_image_paths: list[Path],
    original_train_paths: list[Path],
    original_val_paths: list[Path],
    run_id: str,
) -> Path:
    """Write image-list files and a data.yaml for mixed fine-tuning.

    Train list = new images + sampled original train images (mixed training).
    Val list   = original fixed val set when available, otherwise new images
                 (fallback for smoke tests where splits.json is absent).

    Files written:
    - ``logs/retrain_images_{run_id}.txt``   — new images only (for reference)
    - ``logs/retrain_mixed_{run_id}.txt``    — full train list (new + original)
    - ``logs/retrain_val_{run_id}.txt``      — val list
    - ``logs/retrain_data_{run_id}.yaml``    — data.yaml consumed by Ultralytics

    Args:
        new_image_paths: New images passed via ``--new-images``.
        original_train_paths: Subsampled original train images for replay.
        original_val_paths: Original val images for early-stopping signal.
        run_id: Unique identifier for this retrain run.

    Returns:
        Path to the generated data.yaml file.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    enc = locale.getpreferredencoding(False)

    # New-images list (informational).
    new_list_path = LOG_DIR / f"retrain_images_{run_id}.txt"
    new_list_path.write_text(
        "\n".join(str(p) for p in new_image_paths) + "\n", encoding=enc
    )

    # Mixed train list.
    mixed_paths = new_image_paths + original_train_paths
    mixed_list_path = LOG_DIR / f"retrain_mixed_{run_id}.txt"
    mixed_list_path.write_text(
        "\n".join(str(p) for p in mixed_paths) + "\n", encoding=enc
    )
    print(
        f"  Train list: {len(new_image_paths)} new + "
        f"{len(original_train_paths)} original = {len(mixed_paths)} total"
        f" -> {mixed_list_path}"
    )

    # Val list: prefer original val set; fall back to new images.
    if original_val_paths:
        val_paths = original_val_paths
        val_list_path = LOG_DIR / f"retrain_val_{run_id}.txt"
        val_list_path.write_text(
            "\n".join(str(p) for p in val_paths) + "\n", encoding=enc
        )
        print(f"  Val list:   {len(val_paths)} original val images -> {val_list_path}")
    else:
        # Fallback for smoke tests / missing splits.json.
        val_list_path = mixed_list_path
        print(
            f"  Val list:   (fallback) using train list ({len(mixed_paths)} images)"
        )

    data_yaml_path = LOG_DIR / f"retrain_data_{run_id}.yaml"
    data = {
        "train": str(mixed_list_path.resolve()),
        "val": str(val_list_path.resolve()),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(data_yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
    print(f"  data.yaml: {data_yaml_path}")
    return data_yaml_path


# ---------------------------------------------------------------------------
# Helpers — MongoDB
# ---------------------------------------------------------------------------


def _write_initial_mongo_doc(
    run_id: str,
    model: str,
    new_images_dir: str,
    n_new_images: int,
    hyperparams: dict[str, Any],
) -> None:
    """Insert the initial experiments document with status="running".

    Args:
        run_id: Unique run identifier.
        model: Model name string.
        new_images_dir: Path string for the new-images source.
        n_new_images: Number of new images ingested / used for fine-tuning.
        hyperparams: Dict of training hyperparameters.
    """
    db = get_db()
    doc: dict[str, Any] = {
        "run_id": run_id,
        "model": model,
        "model_version": None,
        "status": "running",
        "is_production": False,
        "sample_ratio": 1.0,  # retrain always uses 100% of new images
        "training_image_ids": [],  # not tracked at image_id granularity for retrain
        "dataset_countries": [],
        "hyperparams": hyperparams,
        "retrain_source": new_images_dir,
        "retrain_n_images": n_new_images,
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
# Helpers — metrics extraction (mirrors train.py)
# ---------------------------------------------------------------------------


def _extract_metrics(run_dir: Path) -> dict[str, float]:
    """Read Ultralytics results.csv and extract final-epoch metrics.

    Args:
        run_dir: Ultralytics output directory (contains results.csv).

    Returns:
        Dict with keys: mAP50, precision, recall, F1.  Empty if not parseable.
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"  [warn] results.csv not found in {run_dir}.")
        return {}

    try:
        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        if not rows:
            return {}

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

        metrics: dict[str, float] = {}
        if map50 is not None:
            metrics["mAP50"] = round(map50, 6)
        if precision is not None:
            metrics["precision"] = round(precision, 6)
        if recall is not None:
            metrics["recall"] = round(recall, 6)
        if precision is not None and recall is not None and (precision + recall) > 0:
            metrics["F1"] = round(2 * precision * recall / (precision + recall), 6)

        return metrics

    except Exception as exc:
        print(f"  [warn] Could not parse results.csv: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Core retrain function
# ---------------------------------------------------------------------------


def retrain(
    new_images: str | Path,
    epochs: int = 20,
    batch: int = 8,
    patience: int = 10,
    imgsz: int = 640,
    mix_ratio: float = DEFAULT_MIX_RATIO,
    lr0: float | None = None,
    lrf: float | None = None,
    cos_lr: bool = False,
    optimizer: str | None = None,
    freeze: int | None = None,
    model_override: str | None = None,
    skip_upload: bool = False,
    skip_promote: bool = False,
    skip_ingest: bool = False,
) -> str:
    """Fine-tune the production model on new images and promote if better.

    Mixed training is always on by default (``mix_ratio=0.30``) to prevent
    catastrophic forgetting.  The original training pool is sampled from
    ``logs/splits.json``; if that file is absent the run falls back to
    new-images-only training with a warning.

    Args:
        new_images: Directory with new images (and YOLO .txt labels).  Accepts
            flat directories and RDD2022 country/split/images/ hierarchies.
        epochs: Maximum fine-tuning epochs (default 20).
        batch: Batch size (default 8).
        patience: Early-stopping patience on mAP50 (default 10).
        imgsz: Input image size in pixels (default 640).
        mix_ratio: Fraction of the original training pool to replay alongside
            new images (default 0.30).  Set to 0.0 to train on new images only.
        lr0: Initial learning rate.  None → Ultralytics default (0.01).
        lrf: Final LR as a fraction of lr0.  None → Ultralytics default (0.01).
        cos_lr: Use cosine LR schedule instead of linear.
        optimizer: Optimizer name: 'SGD' | 'Adam' | 'AdamW' | 'auto'.
            None → Ultralytics default ('auto').
        freeze: Number of backbone layers to freeze during fine-tuning.
            None / 0 → train all layers.  freeze=10 freezes the backbone and
            trains only the neck and detection head.
        model_override: Optional local path to a .pt checkpoint.  Bypasses
            the MongoDB production-model lookup.
        skip_upload: Skip Backblaze B2 upload (for local testing).
        skip_promote: Skip evaluation and promotion (for local testing).
        skip_ingest: Skip the MongoDB ingest step (when metadata already exists).

    Returns:
        run_id string of the completed retrain experiment.

    Raises:
        EnvironmentError: If ``RDD_DATA_ROOT`` is not set.
        FileNotFoundError: If ``new_images`` does not exist.
        RuntimeError: If fine-tuning or MongoDB writes fail.
    """
    # ------------------------------------------------------------------
    # 0. Pre-flight checks
    # ------------------------------------------------------------------
    data_root_str = os.getenv("RDD_DATA_ROOT")
    if not data_root_str:
        raise EnvironmentError(
            "RDD_DATA_ROOT is not set. Set it in .env or the environment."
        )
    data_root = Path(data_root_str).resolve()

    new_images_dir = Path(new_images).resolve()
    if not new_images_dir.exists():
        raise FileNotFoundError(
            f"--new-images path does not exist: {new_images_dir}"
        )

    if not (0.0 <= mix_ratio <= 1.0):
        raise ValueError(f"--mix-ratio must be between 0.0 and 1.0, got {mix_ratio}")

    # ------------------------------------------------------------------
    # 1. Resolve production checkpoint
    # ------------------------------------------------------------------
    print("\n[1/10] Resolving production checkpoint ...")
    checkpoint, model_name = _resolve_production_checkpoint(model_override)
    run_id = _make_run_id(model_name)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"RDDS Retrain -- run_id: {run_id}")
    print(f"  checkpoint  = {checkpoint}")
    print(f"  new_images  = {new_images_dir}")
    print(f"  mix_ratio   = {mix_ratio}")
    print(f"  epochs={epochs}  batch={batch}  patience={patience}")
    if lr0 is not None:
        print(f"  lr0={lr0}  lrf={lrf}  cos_lr={cos_lr}  optimizer={optimizer}")
    if freeze:
        print(f"  freeze={freeze} backbone layers")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 2. Ingest new images into MongoDB images_metadata (idempotent)
    # ------------------------------------------------------------------
    print("[2/10] Ingesting new images into MongoDB ...")
    if skip_ingest:
        print("  [skip] Ingest skipped (--skip-ingest).")
    else:
        try:
            n_inserted, n_updated = _ingest_data(data_root=new_images_dir)
            print(f"  Ingest result: {n_inserted} new, {n_updated} updated.")
        except Exception as exc:
            print(
                f"  [warn] Ingest raised an exception (continuing): {exc}\n"
                "  Image metadata may already be in MongoDB."
            )

    # ------------------------------------------------------------------
    # 3. Collect new image paths + build mixed training list
    # ------------------------------------------------------------------
    print("\n[3/10] Collecting image paths ...")
    new_image_paths = _collect_new_image_paths(new_images_dir)
    print(f"  New images: {len(new_image_paths)}")

    print(f"  Loading original train/val splits (mix_ratio={mix_ratio}) ...")
    original_train_paths, original_val_paths = _load_original_splits(
        data_root=data_root, mix_ratio=mix_ratio
    )
    if original_train_paths:
        print(f"  Original train sample: {len(original_train_paths)} images")
    if original_val_paths:
        print(f"  Original val set: {len(original_val_paths)} images")

    data_yaml_path = _write_retrain_data_yaml(
        new_image_paths=new_image_paths,
        original_train_paths=original_train_paths,
        original_val_paths=original_val_paths,
        run_id=run_id,
    )
    total_train = len(new_image_paths) + len(original_train_paths)

    # ------------------------------------------------------------------
    # 4. Write initial MongoDB document (status="running")
    # ------------------------------------------------------------------
    hyperparams: dict[str, Any] = {
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "seed": RANDOM_SEED,
        "patience": patience,
        "mix_ratio": mix_ratio,
        "base_checkpoint": str(checkpoint),
    }
    if lr0 is not None:
        hyperparams["lr0"] = lr0
    if lrf is not None:
        hyperparams["lrf"] = lrf
    if cos_lr:
        hyperparams["cos_lr"] = True
    if optimizer is not None:
        hyperparams["optimizer"] = optimizer
    if freeze:
        hyperparams["freeze"] = freeze

    print("\n[4/10] Writing initial MongoDB document ...")
    _write_initial_mongo_doc(
        run_id=run_id,
        model=model_name,
        new_images_dir=str(new_images_dir),
        n_new_images=len(new_image_paths),
        hyperparams=hyperparams,
    )

    # ------------------------------------------------------------------
    # 5. Fine-tune
    # ------------------------------------------------------------------
    # Set MLflow tracking URI (same pattern as train.py).
    os.environ["MLFLOW_TRACKING_URI"] = Path("mlruns").resolve().as_uri()

    # SIGTERM handler — marks run interrupted if SLURM kills the job.
    def _sigterm_handler(signum, frame):  # noqa: ANN001
        _update_mongo_doc(run_id, {"status": "interrupted"})
        raise SystemExit(f"SIGTERM received — retrain run {run_id} marked interrupted.")

    signal.signal(signal.SIGTERM, _sigterm_handler)

    print("\n[5/10] Starting Ultralytics fine-tuning ...")
    yolo = YOLO(str(checkpoint))

    run_dir = RUNS_DIR / run_id
    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml_path.resolve()),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "patience": patience,
        "seed": RANDOM_SEED,
        "project": str(RUNS_DIR),
        "name": run_id,
        "exist_ok": False,
        "verbose": True,
        "plots": True,
        "save": True,
    }
    # Optional fine-tuning knobs — only passed when explicitly set.
    if lr0 is not None:
        train_kwargs["lr0"] = lr0
    if lrf is not None:
        train_kwargs["lrf"] = lrf
    if cos_lr:
        train_kwargs["cos_lr"] = True
    if optimizer is not None:
        train_kwargs["optimizer"] = optimizer
    if freeze:
        train_kwargs["freeze"] = freeze

    try:
        yolo.train(**train_kwargs)
    except Exception as exc:
        _update_mongo_doc(run_id, {"status": "failed", "error": str(exc)})
        raise RuntimeError(f"Fine-tuning failed for run {run_id}: {exc}") from exc

    # ------------------------------------------------------------------
    # 6. Extract training-time metrics from results.csv
    # ------------------------------------------------------------------
    print("\n[6/10] Extracting training-time metrics ...")
    train_metrics = _extract_metrics(run_dir)
    print(f"  Training-time metrics: {train_metrics}")

    # ------------------------------------------------------------------
    # 7. Update MongoDB immediately (status="completed", training metrics)
    #    Do this before any export/upload so MongoDB is never stale.
    # ------------------------------------------------------------------
    print("\n[7/10] Updating MongoDB with training-time metrics ...")
    _update_mongo_doc(
        run_id,
        {
            "status": "completed",
            "metrics": train_metrics,
            "checkpoints": {},
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    # ------------------------------------------------------------------
    # 8. Export best.pt -> best.onnx (subprocess — crash-safe)
    # ------------------------------------------------------------------
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        print("\n[8/10] Exporting best.pt -> best.onnx ...")
        export_cmd = (
            f"from ultralytics import YOLO; "
            f"YOLO(r'{best_pt}').export(format='onnx', imgsz={imgsz}, simplify=True)"
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", export_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
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
        print(f"\n[8/10] [warn] {best_pt} not found -- skipping ONNX export.")

    # ------------------------------------------------------------------
    # 9. Upload checkpoints to Backblaze B2
    # ------------------------------------------------------------------
    checkpoint_urls: dict[str, str] = {}
    if not skip_upload:
        print("\n[9/10] Uploading checkpoints to Backblaze B2 ...")
        try:
            checkpoint_urls = upload_checkpoints(run_id=run_id, run_dir=run_dir)
        except Exception as exc:
            print(f"  [warn] Checkpoint upload failed: {exc}. Continuing.")
    else:
        print("\n[9/10] [skip] B2 upload skipped (--skip-upload).")

    if checkpoint_urls:
        _update_mongo_doc(run_id, {"checkpoints": checkpoint_urls})

    # ------------------------------------------------------------------
    # 10. Evaluate on the fixed validation set (CRDDC2022 protocol F1)
    #     and then promote if the threshold is met.
    # ------------------------------------------------------------------
    f1_before: float | None = None
    f1_after: float | None = None
    promoted = False

    if not skip_promote:
        # Get the current production F1 before overwriting it.
        try:
            db = get_db()
            prod_doc = db["experiments"].find_one({"is_production": True})
            if prod_doc:
                f1_before = prod_doc.get("metrics", {}).get("F1")
        except Exception as exc:
            print(f"  [warn] Could not read current production F1: {exc}")

        print("\n[10/10] Running CRDDC2022 evaluation on the fixed validation set ...")
        try:
            eval_result = _run_evaluate(run_id=run_id, split="val", dry_run=False)
            val_doc = eval_result.get("val", {})
            f1_after = val_doc.get("F1_overall")

            if f1_after is not None:
                # Also store the CRDDC2022 F1 in the top-level metrics field
                # so promote.py can read it via the experiments document.
                _update_mongo_doc(
                    run_id,
                    {
                        "metrics.F1": f1_after,
                        "metrics.evaluation_val": val_doc,
                    },
                )

                print(f"\n  CRDDC2022 F1 (val): {f1_after:.4f}")
                print("  Running promotion check ...")
                outcome = maybe_promote(run_id=run_id, f1_new=f1_after, already_evaluated=True)
                promoted = outcome == "promoted"
                print(f"  Promotion outcome: {outcome}")
            else:
                print(
                    "  [warn] evaluate() returned no F1_overall — "
                    "skipping promotion. Check that the validation set exists."
                )
                _update_mongo_doc(run_id, {"status": "completed"})

        except Exception as exc:
            print(
                f"  [warn] Evaluation / promotion failed: {exc}\n"
                "  Run experiments document is marked 'completed' (not promoted)."
            )
            _update_mongo_doc(run_id, {"status": "completed"})

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("RETRAIN COMPLETE")
    print(f"{'='*60}")
    print(f"  run_id:         {run_id}")
    print(f"  epochs:         {epochs}")
    print(f"  new images:     {len(new_image_paths)}")
    print(f"  original mixed: {len(original_train_paths)} (mix_ratio={mix_ratio})")
    print(f"  total train:    {total_train}")
    if skip_promote:
        print("  F1 before:      (skipped — --skip-promote)")
    elif f1_before is not None:
        print(f"  F1 before:      {f1_before:.4f}")
    else:
        print("  F1 before:      (none — no prior production model)")
    if skip_promote:
        print("  F1 after:       (skipped — --skip-promote)")
    elif f1_after is not None:
        print(f"  F1 after:       {f1_after:.4f}")
    else:
        print("  F1 after:       (evaluation failed — check logs)")
    print(f"  Promoted:       {'YES' if promoted else 'NO'}")
    if checkpoint_urls:
        print(f"  Checkpoints:    {list(checkpoint_urls.keys())}")
    print(f"{'='*60}\n")

    # Log to MLflow (non-fatal on failure).
    try:
        mlflow.set_experiment("rdds_retraining")
        with mlflow.start_run(run_name=run_id):
            mlflow.log_params(
                {
                    "model": model_name,
                    "epochs": epochs,
                    "batch": batch,
                    "imgsz": imgsz,
                    "patience": patience,
                    "seed": RANDOM_SEED,
                    "n_new_images": len(new_image_paths),
                    "base_checkpoint": str(checkpoint),
                }
            )
            if f1_after is not None:
                mlflow.log_metric("F1_crddc2022", f1_after)
            for k, v in train_metrics.items():
                mlflow.log_metric(f"train_{k}", v)
    except Exception as exc:
        print(f"  [warn] MLflow logging failed: {exc}. Continuing.")

    return run_id


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the production YOLO checkpoint on new images.\n\n"
            "Mixed training is on by default (--mix-ratio 0.30): a subsample\n"
            "of the original training pool is replayed alongside the new images\n"
            "to prevent catastrophic forgetting.\n\n"
            "Evaluates on the fixed validation set and promotes if F1 improves\n"
            "by more than the CRDDC2022 threshold (0.01)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Required ---
    parser.add_argument(
        "--new-images",
        required=True,
        help="Directory containing new images (and YOLO .txt labels).",
    )

    # --- Training basics ---
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum fine-tuning epochs (default: 20).",
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
        default=10,
        help="Early-stopping patience on mAP50 (default: 10).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size in pixels (default: 640).",
    )

    # --- Mixed training ---
    parser.add_argument(
        "--mix-ratio",
        type=float,
        default=DEFAULT_MIX_RATIO,
        help=(
            "Fraction of the original training pool to replay alongside new "
            "images (default: 0.30).  Set to 0.0 to disable mixed training."
        ),
    )

    # --- Fine-tuning knobs (all optional) ---
    parser.add_argument(
        "--lr0",
        type=float,
        default=None,
        help=(
            "Initial learning rate (default: Ultralytics default 0.01). "
            "Lower values (e.g. 0.001) for small new-image batches."
        ),
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=None,
        help="Final LR as a fraction of lr0 (default: Ultralytics default 0.01).",
    )
    parser.add_argument(
        "--cos-lr",
        action="store_true",
        help="Use cosine LR decay instead of linear.",
    )
    parser.add_argument(
        "--optimizer",
        default=None,
        choices=["SGD", "Adam", "AdamW", "auto"],
        help="Optimizer (default: Ultralytics 'auto').",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help=(
            "Number of backbone layers to freeze (default: 0, train all). "
            "freeze=10 trains neck+head only — useful for very small batches."
        ),
    )

    # --- Checkpoint ---
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Local path to a .pt checkpoint. "
            "Bypasses the MongoDB production-model lookup."
        ),
    )

    # --- Skip flags (for testing) ---
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip Backblaze B2 upload (for local testing).",
    )
    parser.add_argument(
        "--skip-promote",
        action="store_true",
        help="Skip evaluation and promotion (for local testing).",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip the MongoDB ingest step (when metadata already exists).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    retrain(
        new_images=args.new_images,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        imgsz=args.imgsz,
        mix_ratio=args.mix_ratio,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        optimizer=args.optimizer,
        freeze=args.freeze,
        model_override=args.model,
        skip_upload=args.skip_upload,
        skip_promote=args.skip_promote,
        skip_ingest=args.skip_ingest,
    )
