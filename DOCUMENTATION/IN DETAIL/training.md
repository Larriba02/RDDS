# RDDS — Training Module (IN DETAIL)
**Road Damage Detection System · Group 3 · UFV**
*Version 1.2 — May 2026*

This document describes the implementation of the Step 3 training pipeline:
`src/training/train.py`, `src/training/upload_checkpoint.py`, and
`src/training/promote.py`.

For the high-level architecture see `RDDS_Pipeline.md` Stage 3. For
non-negotiable rules see `CLAUDE.md §2`.

---

## 1. Entry point: `src/training/train.py`

### Invocation

```bash
# Phase 0 — laptop, 10% of data
python -m src.training.train \
    --model yolo11s \
    --sample-ratio 0.10 \
    --epochs 50 \
    --batch 8 \
    --patience 15

# Phase 1 — cluster, full dataset
python -m src.training.train \
    --model yolo11m \
    --sample-ratio 1.0 \
    --epochs 100 \
    --batch 32 \
    --patience 20

# Smoke test (skip B2 upload and promotion)
python -m src.training.train \
    --model yolo11s \
    --sample-ratio 1.0 \
    --epochs 1 \
    --batch 2 \
    --skip-upload \
    --skip-promote
```

### Step-by-step workflow

1. **Generate run_id** — format `run_YYYYMMDD_HHMMSS_<model>` (UTC).
2. **Load splits** — read `logs/splits.json` (written by `split.py`). Separate
   image IDs into `split=="train"` (candidate pool) and `split=="val"` (fixed
   val set).
3. **Load image metadata** — pull `image_id`, `filepath`, `country` from
   MongoDB `images_metadata`. Raises `RuntimeError` if collection is empty.
4. **Subsample training set** — at `--sample-ratio` stratified by country:
   - Val set is ALWAYS 100% of `split=="val"` images — never subsampled.
   - For each country in sorted order, `math.ceil(n * ratio)` images are
     selected (minimum 1) using `random.Random(42).shuffle()`.
   - At `ratio=1.0`, the full pool is used without shuffling.
5. **Write image list files**:
   - `logs/train_images_{run_id}.txt` — one absolute path per line.
   - `logs/val_images_{run_id}.txt` — one absolute path per line.
   - Paths are resolved as `RDD_DATA_ROOT / filepath`.
6. **Write data.yaml** — `logs/data_{run_id}.yaml` pointing to the two list
   files. Format required by Ultralytics for text-file-based datasets.
7. **Compute class weights** — inverse-frequency from `logs/class_distribution.json`.
   Stored in MongoDB for documentation. Not currently passed to Ultralytics
   training (Ultralytics 8.x `cls` param is a scalar, not per-class vector).
8. **Write initial MongoDB document** — `experiments` collection,
   `status="running"`, before training starts. This guarantees a record exists
   even if training crashes.
9. **Pre-training setup**:
   - `mlflow.set_tracking_uri("./mlruns")` is called explicitly before
     Ultralytics training to force a relative local path. Without this,
     Ultralytics' built-in MLflow callback can receive a bare Windows absolute
     path (e.g. `C:\…\runs\train`) that MLflow rejects as an invalid URI.
   - A `SIGTERM` signal handler is installed. When SLURM's wall-clock limit
     kills the job, the handler sets `status="interrupted"` in MongoDB before
     the process exits, so the experiment record is never left stuck in
     `"running"` state.
10. **Run Ultralytics training** — `YOLO(model).train(data=..., seed=42, ...)`.
    On failure, sets `status="failed"` in MongoDB and re-raises.
11. **Extract metrics** — reads `runs/train/{run_id}/results.csv`. Computes F1
    from final-row precision and recall.
12. **Update MongoDB immediately** — `status="completed"`, metrics,
    `completed_at` timestamp. This happens *before* ONNX export and B2 upload
    so that metrics are never lost if those steps crash.
13. **Export ONNX** — runs `YOLO(best.pt).export(format="onnx")` in a
    **subprocess** (crash-safe). If the subprocess exits non-zero (e.g. due to
    an onnxslim segfault), a warning is printed and training continues without
    a `.onnx` file. stdout is discarded; stderr is captured in binary to avoid
    Windows cp1252 decode errors on paths with non-ASCII characters.
14. **Upload checkpoints** — calls `upload_checkpoint.upload_checkpoints()`.
    Skipped if `--skip-upload` is set. If B2 upload succeeds, a second
    MongoDB update records the checkpoint URLs.
15. **MLflow logging** — hyperparams + final metrics logged to local
    `./mlruns/` (per-machine, not shared).
16. **Promote** — calls `maybe_promote(run_id, f1)` if F1 is available.
    Skipped if `--skip-promote` is set or if F1 is None (no detections on
    tiny datasets). On successful promotion, `maybe_promote` automatically
    calls `evaluate()` for the newly promoted run (non-fatal: a warning is
    printed if evaluation fails but the promotion itself is not rolled back).
    See `DOCUMENTATION/IN DETAIL/dashboard.md §4`.

### Prerequisites

- `logs/splits.json` — run `python -m src.data.split` first.
- MongoDB `images_metadata` populated — run `python -m src.data.ingest` first.
- `RDD_DATA_ROOT` set in `.env` pointing to the local dataset root.
  `train()` performs a pre-flight check: if `RDD_DATA_ROOT` is absent it raises
  `EnvironmentError` immediately, before any MongoDB writes or file I/O.
- YOLO pretrained weights downloadable or cached (automatic on first run).

---

## 2. `src/training/upload_checkpoint.py`

Uploads `best.pt`, `last.pt`, and `best.onnx` from `runs/train/{run_id}/weights/`
to Backblaze B2 under the prefix `checkpoints/{run_id}/`.

Returns a dict `{"best_pt": url, "last_pt": url, "best_onnx": url}` which is
stored in the `experiments` MongoDB document under `checkpoints`.

Raises `RuntimeError` if any file fails to upload. Missing files (e.g. `best.onnx`
when export failed) are skipped with a warning, not an error.

The boto3 client is configured with `connect_timeout=30 s` and
`read_timeout=300 s` to tolerate slow B2 connections when uploading large
`.pt` files from the cluster. These values are hardcoded in the private
`_b2_client()` helper and do not need to be set in `.env`.

Environment variables required:
- `BACKBLAZE_KEY_ID`
- `BACKBLAZE_APP_KEY`
- `BACKBLAZE_BUCKET`
- `BACKBLAZE_ENDPOINT` (optional, defaults to `https://s3.us-west-004.backblazeb2.com`)

---

## 3. `src/training/promote.py`

Applies the CRDDC2022 promotion rule (CLAUDE.md §3, RDDS_Dev_Steps.md Appendix A):

| Condition | Outcome |
|---|---|
| No production model exists | Promote unconditionally |
| Current production has no F1 | Promote unconditionally |
| `F1_new > F1_current + 0.01` | **Promote** |
| `F1_new >= F1_current - 0.005` | `completed` (noise band, no promotion) |
| `F1_new < F1_current - 0.005` | `completed` (regression, no promotion) |

**Atomicity:** promotion uses a MongoDB transaction (session + `with_transaction`)
to guarantee exactly one document has `is_production=True` at any instant.
The new model gets `is_production=True, status="promoted"`. The old production
model gets `is_production=False, status="superseded"`.

This works on Atlas free tier because Atlas always runs as a replica set.

### Usage (standalone)

```bash
python -m src.training.promote --run-id run_20260310_001_yolo11s --f1 0.74
```

---

## 4. Non-negotiable invariants enforced by these modules

| Rule (CLAUDE.md §2) | Enforcement |
|---|---|
| `RANDOM_SEED=42` | `seed=42` passed to Ultralytics and `random.Random(42)` for sampling |
| Val set always 100% | `_subsample_train_ids` is only called on `split=="train"` pool |
| `is_production` atomic | MongoDB transaction in `promote._apply_promotion` |
| MongoDB writes before/during/after | `status="running"` before; `status="completed"` after |
| Checkpoints uploaded to B2 | `upload_checkpoints()` called immediately after training |
| No Jupyter notebooks | Python scripts only |

---

## 5. Output artefacts per run

| Artefact | Location |
|---|---|
| Training image list | `logs/train_images_{run_id}.txt` |
| Val image list | `logs/val_images_{run_id}.txt` |
| data.yaml | `logs/data_{run_id}.yaml` |
| Ultralytics run dir | `runs/train/{run_id}/` |
| best.pt | `runs/train/{run_id}/weights/best.pt` |
| last.pt | `runs/train/{run_id}/weights/last.pt` |
| best.onnx | `runs/train/{run_id}/weights/best.onnx` |
| MongoDB doc | `rdds.experiments` collection, `run_id` field |
| MLflow run | `./mlruns/` (local only) |
| B2 checkpoints | `checkpoints/{run_id}/best.pt` etc. |

---

## 6. CLI reference

### `python -m src.training.train` — Full training cycle (Phase 0 / Phase 1)

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
python -m src.training.train --model yolo11s --sample-ratio 0.10 --epochs 50 --batch 8 --patience 15
```

**macOS / Linux**
```bash
source .venv/bin/activate
python -m src.training.train --model yolo11s --sample-ratio 0.10 --epochs 50 --batch 8 --patience 15
```

#### Flags

| Flag | Type / Options | Default | Effect | When to use |
|------|---------------|---------|--------|-------------|
| `--model` | `yolo11s` \| `yolo11m` \| `yolo11l` \| `yolo11x` | `yolo11s` | Ultralytics model variant to train | Use `yolo11s` for Phase 0 laptop runs; `yolo11m` for Phase 1 cluster runs |
| `--sample-ratio` | `float` (0.0–1.0) | `SAMPLE_RATIO` from `.env` | Fraction of training images to use, sampled per country | Phase 0 progression: `0.10` → `0.25` → `0.50` → `1.0` |
| `--epochs` | `int` | `50` | Maximum training epochs | Phase 0: `50`; Phase 1 full run: `100` |
| `--batch` | `int` | `8` | Batch size | Match to GPU VRAM: `8` for RTX 4050 6 GB; `32` for A100 40 GB |
| `--patience` | `int` | `15` | Early-stopping patience (epochs without mAP improvement) | Phase 0: `15`; Phase 1: `20` |
| `--imgsz` | `int` | `640` | Input image size in pixels | Keep at `640` (RDD2022 standard); change only with explicit justification |
| `--no-amp` | flag | off (AMP enabled) | Disable FP16 mixed-precision training | Pass if you see AMP-related NaN losses; otherwise leave AMP on |
| `--lr0` | `float` | `0.01` | Initial learning rate | Lower to `0.001` for fine-tuning; keep default for full training from scratch |
| `--lrf` | `float` | `0.01` | Final LR as a fraction of `lr0` (LR decays from `lr0` to `lr0 * lrf`) | Increase to `0.1` for a more gradual decay schedule |
| `--cos-lr` | flag | off (linear decay) | Use cosine learning rate schedule instead of linear | Enable for longer Phase 1 runs where a warmup-then-decay cycle helps |
| `--optimizer` | `auto` \| `SGD` \| `Adam` \| `AdamW` \| `NAdam` \| `RAdam` \| `RMSProp` | `auto` | Optimizer (Ultralytics selects SGD for YOLO when `auto`) | Keep `auto`; switch to `AdamW` only for experimental runs |
| `--cache` | `False` \| `ram` \| `disk` | `False` | Cache images to speed up training | `ram` if you have ≥16 GB RAM and a small dataset fraction; `disk` on the cluster |
| `--workers` | `int` | `8` | Data-loading worker threads | Lower to `4` on Windows if DataLoader errors appear; keep `8` on Linux |
| `--device` | `str` | `"0"` | CUDA device(s): `"0"` for single GPU, `"0,1,2,3"` for multi-GPU DDP | Match to available hardware; Ultralytics handles DDP spawning internally |
| `--skip-upload` | flag | off | Skip Backblaze B2 checkpoint upload | Use for local smoke tests without real B2 credentials |
| `--skip-promote` | flag | off | Skip the promotion check after training | Use for local testing where promotion should not happen |

#### Full example

```powershell
# Windows — Phase 1 cluster-style full run
python -m src.training.train `
    --model yolo11m `
    --sample-ratio 1.0 `
    --epochs 100 `
    --batch 32 `
    --patience 20 `
    --lr0 0.01 `
    --lrf 0.01 `
    --cos-lr `
    --optimizer auto `
    --cache disk `
    --workers 8 `
    --device 0
```
```bash
# macOS / Linux — Phase 1 cluster-style full run
python -m src.training.train \
    --model yolo11m \
    --sample-ratio 1.0 \
    --epochs 100 \
    --batch 32 \
    --patience 20 \
    --lr0 0.01 \
    --lrf 0.01 \
    --cos-lr \
    --optimizer auto \
    --cache disk \
    --workers 8 \
    --device 0
```

---

### `python -m src.training.promote` — Manually promote a run to production

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
python -m src.training.promote --run-id run_20260310_001_yolo11s --f1 0.74
```

**macOS / Linux**
```bash
source .venv/bin/activate
python -m src.training.promote --run-id run_20260310_001_yolo11s --f1 0.74
```

#### Flags

| Flag | Type / Options | Default | Effect | When to use |
|------|---------------|---------|--------|-------------|
| `--run-id` | `str` | (required) | `run_id` of the candidate experiment in MongoDB | The exact `run_id` printed at the start of `train.py` output |
| `--f1` | `float` | (required) | Overall F1 score (IoU ≥ 0.5) to compare against the current production model | Pass the CRDDC2022 F1 from `evaluate.py` output, not the training-time F1 |

#### Full example

```powershell
# Windows
python -m src.training.promote --run-id run_20260504_202658_yolo11s --f1 0.7412
```
```bash
# macOS / Linux
python -m src.training.promote --run-id run_20260504_202658_yolo11s --f1 0.7412
```
