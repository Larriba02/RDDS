# Retraining Pipeline — Detailed Guide
**RDDS · Group 3 · UFV**  
*Version 1.0 — May 2026*

---

## What this module does

`src/training/retrain.py` fine-tunes the current production model on a new batch
of images and promotes the result to production if it improves the CRDDC2022 F1
by more than the required margin.

This is a **manually triggered** operation, not an autonomous loop. You run it
explicitly when you have new labelled images and want to incorporate them into
the production model.

---

## Usage

```bash
# Fine-tune from production checkpoint with new images
python -m src.training.retrain \
    --new-images path/to/new_images/ \
    --epochs 20 \
    --batch 8 \
    --patience 10

# With local model override (bypass B2 download)
python -m src.training.retrain \
    --new-images path/to/new_images/ \
    --model runs/detect/myrun/weights/best.pt

# Smoke test (tiny dataset, 1 epoch — verifies the pipeline end-to-end)
python -m src.training.retrain \
    --new-images tests/data/tiny_rdd2022/ \
    --epochs 1 \
    --batch 2 \
    --patience 1
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--new-images` | (required) | Directory containing new images and YOLO labels. |
| `--epochs` | 20 | Maximum fine-tuning epochs. |
| `--batch` | 8 | Batch size. |
| `--patience` | 10 | Early-stopping patience on mAP50. |
| `--imgsz` | 640 | Input image size in pixels. |
| `--model` | (MongoDB lookup) | Local `.pt` path override — bypasses production checkpoint lookup. |
| `--skip-upload` | False | Skip Backblaze B2 upload (for local testing). |
| `--skip-promote` | False | Skip evaluation and promotion (for local testing). |
| `--skip-ingest` | False | Skip the MongoDB ingest step (when metadata already exists). |

---

## Step-by-step pipeline

The `retrain()` function executes exactly 10 steps:

### Step 1 — Pre-flight and checkpoint resolution

Raises `EnvironmentError` immediately if `RDD_DATA_ROOT` is not set — same
pre-flight as `train.py`.

Checkpoint resolution order:
1. `--model` local path override.
2. `runs/train/<run_id>/weights/best.pt` (exact match on the production run_id).
3. Any `runs/train/*/weights/best.pt` whose directory name starts with the production run_id (fuzzy match for Ultralytics incremental naming).
4. Download `best.pt` from the B2 URL stored in MongoDB under `experiments.checkpoints.best_pt`.

### Step 2 — Ingest new images

Calls `src.data.ingest.ingest(data_root=new_images_dir)`.

This is **idempotent**: images already in the `images_metadata` collection are
updated (`$set`), not re-inserted. Safe to run multiple times.

If ingest fails (e.g., `splits.json` is missing because the new images are not
part of any split), the error is caught and logged as a warning. Training
continues — the new image metadata simply may not be in MongoDB, but the images
are still on disk and Ultralytics can use them.

### Step 3 — Collect image paths and write data.yaml

All image files under `--new-images` are collected recursively (supports both
flat directories and the RDD2022 country/split/images/ hierarchy).

Two files are written:
- `logs/retrain_images_{run_id}.txt` — one absolute image path per line.
- `logs/retrain_data_{run_id}.yaml` — Ultralytics data descriptor pointing to
  the image list. Both `train` and `val` keys point to the same list: since the
  new-image batch is typically small, using it as both train and val lets
  Ultralytics run validation passes. The authoritative F1 is computed separately
  by `evaluate.py` on the fixed 1,000-per-country val set.

### Step 4 — Write initial MongoDB document

An `experiments` document is inserted before training starts:

```json
{
  "run_id": "run_20260507_184531_yolo11s_retrain",
  "model": "yolo11s",
  "status": "running",
  "is_production": false,
  "hyperparams": { "epochs": 20, "batch": 8, "imgsz": 640, "seed": 42, ... },
  "retrain_source": "/path/to/new_images",
  "retrain_n_images": 14
}
```

A SIGTERM handler is installed at this point. If SLURM's wall-clock limit kills
the job, the handler updates `status` to `"interrupted"` before the process
exits.

### Step 5 — Fine-tune

`YOLO(str(checkpoint)).train(...)` is called with:
- `seed=42` (RANDOM_SEED — non-negotiable).
- `project=runs/train`, `name=run_id`.
- All other Ultralytics defaults (AMP enabled by Ultralytics automatically).

The run directory is `runs/train/{run_id}/`.

### Step 6 — Extract training-time metrics

`results.csv` is read from the run directory. Final-epoch precision, recall,
mAP50, and F1 are extracted. These are training-time metrics (val = same new
images), not the authoritative CRDDC2022 metrics. They are stored in MongoDB
for completeness but are not used for the promotion decision.

### Step 7 — Update MongoDB (status="completed")

MongoDB is updated with training-time metrics **before** ONNX export and B2
upload. This ensures the document is never left in `status="running"` if a
later step crashes.

### Step 8 — Export best.onnx

Runs in a subprocess:
```python
YOLO(best_pt).export(format='onnx', imgsz=640, simplify=True)
```
Crash-safe: if the subprocess fails (e.g. onnxslim segfault), the main process
continues. The warning is printed and the run continues without `best.onnx`.

### Step 9 — Upload to Backblaze B2

`upload_checkpoints(run_id, run_dir)` uploads `best.pt`, `last.pt`, and
`best.onnx` to B2. URLs are written back to `experiments.checkpoints`. If
upload fails, the warning is printed and the run continues (local files are
still on disk).

### Step 10 — Evaluate and promote

`evaluate(run_id=run_id, split="val")` runs CRDDC2022-protocol evaluation on
the fixed 1,000-per-country validation set. The returned `F1_overall` is the
authoritative metric used for the promotion decision.

`maybe_promote(run_id, f1_new)` applies the promotion rule:

| Condition | Outcome |
|-----------|---------|
| `F1_new > F1_current + 0.01` | Promoted — `is_production` flipped atomically via MongoDB transaction. Old production model marked `"superseded"`. |
| `F1_new ∈ [F1_current − 0.005, F1_current + 0.01]` | Noise band — logged as `"completed"`, not promoted. |
| `F1_new < F1_current − 0.005` | Regression — logged as `"completed"`. Investigate before next retrain. |

The transaction guarantees exactly one document has `is_production=True` at any
instant (CLAUDE.md §2 rule 4).

---

## New-images folder layout

Two layouts are supported:

**Flat (any custom batch):**
```
new_images/
├── img001.jpg
├── img001.txt   # YOLO label (or in labels/ sibling)
├── img002.jpg
└── img002.txt
```

**RDD2022-style (e.g. passing the tiny_rdd2022 dataset):**
```
new_images/
├── Japan/
│   └── train/
│       ├── images/
│       │   ├── 00001.jpg
│       │   └── 00002.jpg
│       └── labels/
│           ├── 00001.txt
│           └── 00002.txt
└── Czech/
    └── train/
        ├── images/...
        └── labels/...
```

Ultralytics resolves labels automatically by looking for a `labels/` directory
that is a sibling of the `images/` directory containing the image.

---

## run_id convention

Retrain run IDs follow the same format as `train.py` plus a `_retrain` suffix:

```
run_YYYYMMDD_HHMMSS_{model}_retrain
```

Example: `run_20260507_184531_yolo11s_retrain`

This makes retrain runs distinguishable from fresh training runs in the
dashboard and in MongoDB queries.

---

## What is NOT done by retrain.py

- **No full dataset retraining from scratch.** Only fine-tuning from the
  existing production checkpoint. For full retraining, use `train.py`.
- **No data validation.** The assumption is that new images have already been
  validated and converted to YOLO format before being passed to `--new-images`.
  If new images come from the RDD2022 pipeline, run `validate.py` and
  `convert.py` first.
- **No class-weight recalculation.** Class weights are not recomputed from the
  combined dataset. If the new images significantly shift class balance,
  consider running `analyse_distribution.py` on the full dataset and retraining
  from scratch with updated weights.

---

## Relationship with other modules

| Module | How retrain.py uses it |
|--------|------------------------|
| `src.data.ingest.ingest` | Ingests new image metadata into MongoDB (imported directly). |
| `src.training.upload_checkpoint.upload_checkpoints` | Uploads best.pt, last.pt, best.onnx to B2 (imported directly). |
| `src.evaluation.evaluate.evaluate` | Runs CRDDC2022 evaluation on the fixed val set (imported directly). |
| `src.training.promote.maybe_promote` | Applies promotion rule and atomic MongoDB transaction (imported directly). |
| `src.db.connection.get_db` | MongoDB access for document writes and production experiment lookup. |
