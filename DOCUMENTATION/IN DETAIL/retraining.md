# Retraining Pipeline — Detailed Guide
**RDDS · Group 3 · UFV**  
*Version 1.2 — May 2026*

---

## Scope note (final deliverable)

The retraining workflow is **documented and implemented, but not executed end-to-end
on a real new-data batch** in the final deliverable. `src/training/retrain.py` is
in the repo, exercises every step (ingest → mixed training → evaluate → promote),
and has been **smoke-tested on the synthetic mini-dataset**
(`tests/data/tiny_rdd2022/`, 14 images, 1 epoch) to confirm the pipeline runs
end-to-end without errors.

Two reasons for the execution gap:

1. **Team bandwidth** — the remaining timeline is allocated to finishing the
   YOLO11m main model, evaluation, and the web demo.
2. **Compute cost on personal GPUs** — a meaningful retraining run on top of
   the production model would occupy the same RTX 4050 / RTX 4060 GPUs that
   are needed for the main training and evaluation work.

The manually triggered fine-tuning design remains the recommended long-term
workflow. Everything below describes how to *run* retraining when those
constraints are relaxed.

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
    --model runs/train/myrun/weights/best.pt

# Smoke test (tiny dataset, 1 epoch — verifies the pipeline end-to-end)
python -m src.training.retrain \
    --new-images tests/data/tiny_rdd2022/ \
    --epochs 1 \
    --batch 2 \
    --patience 1 \
    --skip-upload \
    --skip-promote
```

---

## CLI reference

### `python -m src.training.retrain` — Fine-tune production model on new images

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
python -m src.training.retrain --new-images path\to\new_images\ --epochs 20 --batch 8 --patience 10
```

**macOS / Linux**
```bash
source .venv/bin/activate
python -m src.training.retrain --new-images path/to/new_images/ --epochs 20 --batch 8 --patience 10
```

#### Flags

| Flag | Type / Options | Default | Effect | When to use |
|------|---------------|---------|--------|-------------|
| `--new-images` | `path` | (required) | Directory containing new images and YOLO `.txt` labels. Accepts flat or RDD2022-style country/split/images/ hierarchy. | Always required — this is the new data batch to fine-tune on |
| `--epochs` | `int` | `20` | Maximum fine-tuning epochs | Increase for larger new-image batches; keep low (5–10) for tiny patches |
| `--batch` | `int` | `8` | Batch size | Match to GPU VRAM; `8` for 6 GB, `16` for 8 GB |
| `--patience` | `int` | `10` | Early-stopping patience on mAP50 | Lower when fast convergence is expected; raise if training oscillates |
| `--imgsz` | `int` | `640` | Input image size in pixels | Keep at `640` unless images are a different native resolution |
| `--mix-ratio` | `float` (0.0–1.0) | `0.30` | Fraction of the original training pool to replay alongside new images (mixed training). Set `0.0` to train on new images only | Increase toward `1.0` when the new batch is small (< 100 images) to prevent catastrophic forgetting; decrease when compute is scarce or new batch is large |
| `--lr0` | `float` | Ultralytics default (`0.01`) | Initial learning rate | Lower to `0.001` for conservative surgical updates on small new-image batches; leave unset for larger datasets |
| `--lrf` | `float` | Ultralytics default (`0.01`) | Final LR as a fraction of `lr0` | Increase to `0.1` for a more gradual decay; leave unset to mirror `lr0` default |
| `--cos-lr` | flag | off (linear decay) | Use cosine LR decay instead of linear | Enable when the new dataset is large enough to benefit from a full warmup-decay cycle |
| `--optimizer` | `SGD` \| `Adam` \| `AdamW` \| `auto` | Ultralytics default (`auto`) | Optimizer | Use `AdamW` with low `lr0` for small-batch fine-tuning; keep `auto` otherwise |
| `--freeze` | `int` | `None` (train all layers) | Number of backbone layers to freeze during fine-tuning | Use `10` to freeze backbone and train neck+head only — recommended when new batch is < 50 images to avoid overwriting general features |
| `--model` | `path` | MongoDB production lookup | Local `.pt` checkpoint path — bypasses the MongoDB production-model lookup | Use when you want to fine-tune a specific local checkpoint rather than the current production model |
| `--skip-upload` | flag | off | Skip Backblaze B2 checkpoint upload | Use for local testing without real B2 credentials |
| `--skip-promote` | flag | off | Skip evaluation and promotion after fine-tuning | Use for local testing where promotion should not happen |
| `--skip-ingest` | flag | off | Skip the MongoDB ingest step for new images | Use when the new images were already ingested in a previous run |

#### Full example

```powershell
# Windows — conservative fine-tune: freeze backbone, low LR, cosine decay
python -m src.training.retrain `
    --new-images path\to\new_images\ `
    --epochs 20 `
    --batch 8 `
    --patience 10 `
    --imgsz 640 `
    --mix-ratio 0.30 `
    --lr0 0.001 `
    --lrf 0.01 `
    --cos-lr `
    --optimizer AdamW `
    --freeze 10 `
    --model runs\train\run_20260504_202658_yolo11s\weights\best.pt
```
```bash
# macOS / Linux — conservative fine-tune: freeze backbone, low LR, cosine decay
python -m src.training.retrain \
    --new-images path/to/new_images/ \
    --epochs 20 \
    --batch 8 \
    --patience 10 \
    --imgsz 640 \
    --mix-ratio 0.30 \
    --lr0 0.001 \
    --lrf 0.01 \
    --cos-lr \
    --optimizer AdamW \
    --freeze 10 \
    --model runs/train/run_20260504_202658_yolo11s/weights/best.pt
```

---

## Step-by-step pipeline

The `retrain()` function executes the following steps (authoritative numbering in `src/training/retrain.py`):

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

A SIGTERM handler is installed at this point. If the process receives SIGTERM
(e.g. a SLURM wall-clock kill in the original Phase 1 design, or a manual
`kill` on a local run), the handler updates `status` to `"interrupted"` before
the process exits.

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

`upload_checkpoints(run_id, run_dir)` uploads `best.pt`, `last.pt`,
`best.onnx`, and `results.csv` to B2. URLs are written back to
`experiments.checkpoints` (including `results_csv`, which the dashboard uses
as a fallback when the local `results.csv` is missing). If upload fails, the
warning is printed and the run continues (local files are still on disk).

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
| `src.training.upload_checkpoint.upload_checkpoints` | Uploads best.pt, last.pt, best.onnx, and results.csv to B2 (imported directly). |
| `src.evaluation.evaluate.evaluate` | Runs CRDDC2022 evaluation on the fixed val set (imported directly). |
| `src.training.promote.maybe_promote` | Applies promotion rule and atomic MongoDB transaction (imported directly). |
| `src.db.connection.get_db` | MongoDB access for document writes and production experiment lookup. |
