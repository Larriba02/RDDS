# RDDS — Pipeline Document
**Road Damage Detection System · Group 3 · UFV**  
*Version 2.5 — May 2026*

---

## Project status note (final stretch — 2026-05-23)

The pipeline described in this document was designed for a two-phase training
strategy (laptop sandbox + A100 cluster) and a manually triggered retraining
workflow. Two of those components are **out of scope for the final
deliverable**:

- **Phase 1 / A100 cluster training is not executed.** `scripts/train_cluster.sh`
  and `scripts/submit_sweep.sh` remain in the repository as a design
  artifact. No SLURM run is reported.
- **The retraining pipeline (`src/training/retrain.py`) is smoke-tested
  only.** It is not exercised on real new data before submission. The
  design philosophy is unchanged — manually triggered fine-tuning on
  newly ingested imagery is still the intended workflow. It is out of
  scope **for execution**, for three independent reasons: (a) we have no
  fresh batch of road images outside RDD2022 to retrain on, (b) the team
  does not have the remaining bandwidth, and (c) a real retrain on top
  of the production weights is too costly on M's RTX 4050 and J's RTX
  4060 to fit alongside the rest of the work.

The **final reported models** are both trained locally:

- **YOLO11s baseline** — M, RTX 4050 laptop (Phase 0 at SAMPLE_RATIO=1.0).
- **YOLO11m main model** — J, RTX 4060. This replaces what was previously
  planned as the A100 cluster YOLO11m run.

Sections below that describe Phase 1, the A100 hardware profile, and
autonomous retraining are kept for documentation completeness; flags note
where they are documented-but-not-executed.

---

## Pipeline Overview

The system is a modular end-to-end pipeline that takes raw road images as input and produces structured damage detections as output, with the ability to incorporate new data and retrain the model over time.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RDDS PIPELINE                                   │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   RDD2022    │───▶│    DATA      │───▶│   MONGODB    │              │
│  │   Dataset    │    │  INGESTION   │    │   METADATA   │              │
│  └──────────────┘    └──────────────┘    └──────┬───────┘              │
│                                                  │                      │
│                             ┌────────────────────▼──────────────────┐  │
│                             │          TRAINING                      │  │
│                             │                                        │  │
│                             │  YOLO11s baseline: RTX 4050 (M)        │  │
│                             │       +                                │  │
│                             │  YOLO11m main:     RTX 4060 (J)        │  │
│                             │  (Phase 1 / A100 documented but not    │  │
│                             │   executed — see status note above)    │  │
│                             └────────────────────┬──────────────────┘  │
│                                                  │                      │
│                        ┌─────────────────────────▼──────────┐          │
│                        │           EVALUATION                │          │
│                        │   mAP@0.5 · F1 · P · R per class   │          │
│                        └─────────────────────────┬──────────┘          │
│                                                  │                      │
│              ┌───────────────────────────────────▼──────────────────┐  │
│              │                   INFERENCE                           │  │
│              │   image / video frames  ──▶  detections + MongoDB    │  │
│              └───────────────────────────────────┬──────────────────┘  │
│                                                  │                      │
│                         ┌────────────────────────▼───────────────────┐ │
│                         │   RETRAINING PIPELINE (design artifact)     │ │
│                         │   new data ──▶ fine-tune ──▶ evaluate       │ │
│                         │            ──▶ promote if better            │ │
│                         │   Smoke-tested only — not executed on real  │ │
│                         │   new data in the final deliverable.        │ │
│                         └────────────────────────┬───────────────────┘ │
│                                                  │                      │
│                    ┌─────────────────────────────▼──────────────────┐  │
│                    │   WEB DEMO (optional)  src/api/                 │  │
│                    │   GET /  ·  GET /model  ·  POST /predict        │  │
│                    └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

Each stage is independent and clearly separated. Data, training, inference, and retraining can be modified or rerun without touching the others. All persistent state lives in MongoDB or on the filesystem; nothing is hardcoded between stages.

---

## Stage 1 — Data Ingestion and Preprocessing

### What we do

We download RDD2022 — the most current and comprehensive public dataset for road damage detection — and transform it into a format the model can consume. The dataset contains images from 6 countries (Japan, India, Czech Republic, Norway, United States, China) with bounding box annotations in PascalVOC XML format covering four damage categories:

| Code | Damage Type |
|------|-------------|
| D00 | Longitudinal Crack |
| D10 | Transverse Crack |
| D20 | Alligator Crack |
| D40 | Pothole |

The preprocessing script runs the following steps in order:
### Input  
RDD2022 dataset (official splits per country)

### Steps
1. **Download and extract** country ZIPs to local filesystem.
2. **Validate annotations** — parse every PascalVOC XML file and discard bounding boxes with coordinates outside image dimensions, negative values, or degenerate area. Log all discarded samples explicitly.
3. **Convert to YOLO format** — PascalVOC uses absolute pixel coordinates in XML; YOLO requires normalised coordinates (0–1) in `.txt` files, one per image. Ultralytics provides built-in conversion utilities for this. This step runs once and the output is cached.
4. **Analyse class distribution** — count instances per class before splitting. This is not optional: RDD2022 has severe class imbalance (D00 longitudinal cracks are massively overrepresented versus D40 potholes). The distribution analysis informs the class weights used in training.
5. **Respect official splits**:  
	- **Test split: untouched, always 100%**  
	- **Train split:**  
		-  Stratified by *both* damage class and country
		- Reserve fixed validation set (1000 images per country)
6. **Write metadata to MongoDB** — every image gets a document in the `images_metadata` collection (see MongoDB section below).

### Output  
- YOLO-formatted dataset  
- MongoDB `images_metadata`


### Why this matters

The conversion step (PascalVOC → YOLO `.txt`) is where most silent bugs in road damage detection pipelines originate. Some images in RDD2022 contain annotations with bounding boxes that extend outside the image boundary. If these are not caught here, they corrupt the loss computation during training without raising an explicit error. Catching them explicitly in preprocessing is non-negotiable.

The stratified split by country is equally important. A naive random split risks putting all Norway images in train and none in test, which would make the held-out test set useless for measuring generalisation.

### Dataset strategy by phase

RDD2022 ships with official train/test partitions per country. These are respected as-is. The test split is never used during training or hyperparameter tuning — only for final evaluation.

Within the official train split, a `SAMPLE_RATIO` controls how much data is used for training. The sampling is **stratified by country**: every country contributes the same percentage, guaranteeing geographic representation at any scale.

A fixed baseline of **1,000 images per country** is always reserved for validation and cross-run comparison. This subset is consistent across all experiments — it never changes regardless of `SAMPLE_RATIO`. This allows meaningful comparison of hyperparameter runs throught the project.

| Phase                      | SAMPLE_RATIO     | Purpose                                                        |
| -------------------------- | ---------------- | -------------------------------------------------------------- |
| Phase 0 — M (RTX 4050)     | 0.10→0.25→0.50→1.0 | F1-vs-data curve, pipeline validation. **YOLO11s baseline: F1=0.598.** |
| Step 3.5 — J (RTX 4060)    | 0.10→0.25→1.0    | Originally a YOLO11s hyperparameter funnel; pivoted to direct **YOLO11m main model** training. |
| ~~Phase 1 — cluster (A100)~~ | ~~1.00~~       | Out of scope for the final deliverable — see status note.       |

### Hyperparameter funnel (Step 3.5 — superseded)

> **Status:** Documented for reference. The original justification for the
> funnel was *cheap screening before expensive A100 cluster time*. Phase 1 /
> A100 is now out of scope, so J pivoted from YOLO11s hyperparameter
> screening to training the **YOLO11m main model directly on the RTX 4060**.
> Some early Round 1 screening runs were executed before the pivot.

All candidate configurations were to be first screened at `--sample-ratio 0.10`. Only configs that match or beat the baseline F1 at that ratio advance to `0.25`. The single winner at `0.25` runs at `1.0`. This avoids spending 8–12 h on a full run for a config that would have been eliminated in 2 h.

```
Round 1  0.10   all configs    gate: F1 ≥ 0.411 (Phase 0 baseline at 0.10)
Round 2  0.25   top 2 only     gate: F1 ≥ 0.501 (Phase 0 baseline at 0.25)
Round 3  1.00   winner only    gate: F1 > 0.608 to auto-promote (baseline + 0.01)
```

The **test split is always 100%** regardless of `SAMPLE_RATIO`. Partial test evaluation would make results incomparable across runs and against published benchmarks.


### Centralised ingestion

The full download, validation, format conversion, and upload to cloud storage is performed **once by one team member**. All other machines pull the processed dataset from cloud storage. The raw RDD2022 ZIPs are never downloaded more than once.

---

## Stage 2 — MongoDB: What We Store and Why

### Design principle

**Images live on the filesystem. MongoDB stores everything else.**

MongoDB is used as the persistence layer for metadata, annotations, experiment records, inference outputs, and the model registry. It is *not* a blob store. Training reads images directly from local disk, not from the database, which keeps I/O fast and avoids BSON document size limits.

The reason for choosing MongoDB over a relational database is the heterogeneous and evolving structure of the data. Image metadata has different fields from experiment records, which have different fields from inference results. A relational schema would require complex joins and frequent migrations as the project evolves. MongoDB's document model absorbs this variation naturally.

### Collections

**`images_metadata`** — one document per image.

```json
{
  "image_id": "a3f8c2d1...",
  "filepath": "Japan/train/00001.jpg",
  "country": "Japan",
  "source_device": "smartphone",
  "split": "train",
  "width": 600,
  "height": 600,
  "annotations": [
    { "label": "D00", "xmin": 120, "ymin": 80, "xmax": 340, "ymax": 200 }
  ]
}
```

`image_id` is generated as the MD5 hash of the relative filepath (`Japan/train/00001.jpg`), never as a random ObjectId. This guarantees that any team member who downloads RDD2022 independently will generate the same IDs, making cross-machine experiment records consistent.

`filepath` is always relative to `RDD_DATA_ROOT`, a per-machine environment variable defined in `.env` (never committed to Git). The full path is resolved at runtime:

```python
full_path = os.path.join(os.getenv("RDD_DATA_ROOT"), image["filepath"])
```

**`experiments`** — one document per training run.

```json
{
  "run_id": "run_20260310_001",
  "model": "yolo11m",
  "model_version": "v1.0",
  "status": "promoted",
  "is_production": true,
  "training_image_ids": ["id1", "id2", "..."],
  "dataset_countries": ["Japan", "Czech", "India", "US", "China"],
  "hyperparams": { "epochs": 100, "batch": 32, "imgsz": 640, "seed": 42 },
  "metrics": { "mAP50": 0.84, "F1": 0.81, "precision": 0.83, "recall": 0.79 },
  "checkpoints": {
    "best_pt":   "https://f000.backblazeb2.com/rdds/yolo11m_v1.0_run_20260310/best.pt",
    "last_pt":   "https://f000.backblazeb2.com/rdds/yolo11m_v1.0_run_20260310/last.pt",
    "best_onnx": "https://f000.backblazeb2.com/rdds/yolo11m_v1.0_run_20260310/best.onnx"
  },
  "sample_ratio": 0.10,
  "timestamp": "2026-03-10T14:32:00Z"
}
```

**`status`** tracks the full lifecycle of every run. Four possible values:

- `running` — training is currently in progress.
- `completed` — training finished, metrics available, not promoted.
- `promoted` — was the best model at some point and marked as production.
- `superseded` — was production but was later beaten by a newer model.

**`is_production`** is a separate boolean that answers one question only: which model does the inference module load right now? Exactly one document has this set to `true` at any time. It is set and unset atomically during model promotion.

**`training_image_ids`** records exactly which images were used to train this model version. Any run can be recreated from its document alone.

**`checkpoints`** stores the Backblaze B2 URLs for the three weight files produced by every training run. After training, the script automatically uploads weights to Backblaze and stores the URLs in MongoDB. No machine path is stored — the weights are accessible from anywhere.

`best.pt` is the PyTorch checkpoint with the highest validation mAP during training. `last.pt` is the final epoch checkpoint, kept as a fallback. `best.onnx` is the exported model used by the inference module — it has no runtime dependency on Ultralytics.

**`sample_ratio`** records what fraction of the official train split was used for this run (e.g. `0.10` for Phase 0, `1.0` for full training). Stored per experiment for full reproducibility.

**`predictions`** — one document per inference result.

```json
{
  "pred_id": "<uuid4>",
  "image_id": "<md5 of source path string>",
  "model_version": "v1.0",
  "run_id": "run_20260504_202658_yolo11s",
  "timestamp": "2026-03-15T10:00:00Z",
  "source_path": "/absolute/path/to/image.jpg",
  "detections": [
    { "label": "D40", "bbox": [230, 180, 410, 310], "confidence": 0.91 }
  ]
}
```

Write is idempotent: if a document with the same `image_id` + `model_version` already
exists, the DB insert is skipped (annotated image is still saved to disk).

---

## Experiment Dashboard

`src/dashboard.py` is a Streamlit application that provides a live view of all experiments. It reads from three sources:

| Source | What it provides |
|--------|-----------------|
| MongoDB `experiments` | Canonical run results, hyperparameters, production status |
| `runs/train/{run_id}/results.csv` | Per-epoch training curves (loss, mAP, precision, recall) |
| `mlruns/` (MLflow) | Logged parameters and final metrics per run |

**Pages:**

- **Overview** — Production model card with F1/mAP/precision/recall metrics; F1-vs-data-fraction curve showing Phase 0 diminishing-returns progression.
- **Experiments** — Filterable table of all runs ranked by F1; comparison bar chart; mAP@0.5 vs F1 scatter.
- **Run Detail** — Select any run and see per-epoch training curves (metrics, losses, learning rate) pulled from `results.csv`, plus hyperparameters and Backblaze B2 checkpoint URLs.
- **MLflow** — Tabular view of all MLflow-logged runs with params and metrics. Link to native `mlflow ui` for full per-epoch curves logged by Ultralytics' built-in callback.

**Run:**
```bash
streamlit run src/dashboard.py
# Opens http://localhost:8501
```

The dashboard is read-only and does not modify any state. The "Refresh" button in the sidebar clears the 30-second cache and re-queries MongoDB.

---

## Stage 3 — Training

### Architecture: YOLO11

YOLO11 (Ultralytics, 2024) is the primary architecture. The justification is empirical and specific to this dataset: a direct comparison of YOLOv5, YOLOv8, and YOLO11 on RDD2022 showed YOLO11 outperforming YOLOv8 on all metrics while being more parameter-efficient (9.4M vs 11.2M parameters, 21.5B vs 28.6B FLOPs). Since we are training on exactly this dataset, this comparison is directly applicable.

We use two model sizes:

| Model | Role | Why |
|-------|------|-----|
| **YOLO11s** | Baseline (safety net) | Trains fast even on laptop. Always produces a real, presentable result. |
| **YOLO11m** | Main model | Best accuracy/efficiency trade-off on RDD2022. Target for final results. |

Both are initialised from COCO-pretrained weights (transfer learning). Training from scratch would be slower and typically less accurate at this dataset size.

### Why not ensemble or transformer-only approaches?

Ensemble methods combining YOLO + Cascade RCNN + DETR (Swin Transformer) are behind the state-of-the-art F1=0.86 result (ORDDC 2024). They are not used here because they require significantly more compute, more complex pipelines, and substantially more implementation time than is available in two months. YOLO11m fine-tuned on the full RDD2022 dataset has been shown to reach mAP@0.5 in the 0.82–0.88 range, which is a strong and academically credible result within scope.

### Class imbalance handling

RDD2022 has severe class imbalance: D00 (longitudinal cracks) is massively overrepresented relative to D40 (potholes). A model trained without compensation will learn to detect D00 well and miss D40 consistently. This is not acceptable because D40 is the most operationally important damage category.

Mitigation:
- **Per-class loss weights** (`cls_weight` parameter in Ultralytics): calibrated from the distribution analysis in Stage 1.
- **MixUp augmentation** applied specifically to D40 instances during training.

### Training strategy (final deliverable)

The originally planned two-phase strategy (laptop sandbox + A100 cluster) was
collapsed to **two local training tracks**, both on consumer GPUs:

```
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 0 — YOLO11s baseline (M, RTX 4050, 6 GB VRAM)             │
│                                                                  │
│  Dataset:   All 7 country folders, SAMPLE_RATIO sweep            │
│             0.10 → 0.25 → 0.50 → 1.00                            │
│  Model:     YOLO11s (COCO-pretrained, fine-tuned)                │
│  Config:    batch=8, imgsz=640, epochs=50, amp=True (FP16)       │
│             patience=15 (early stopping on metrics/mAP50)         │
│  Goal:      Map the F1-vs-data curve. Validate the full pipeline │
│             end-to-end (MongoDB writes, MLflow logging,           │
│             Backblaze upload, evaluation, promotion).             │
│  Result:    mAP@0.5 = 0.601, F1 = 0.598 (YOLO11s, SAMPLE_RATIO=1.0, │
│             run_20260504_202658) — promoted to is_production ✓    │
└──────────────────────────────────────────────────────────────────┘
                            +
┌──────────────────────────────────────────────────────────────────┐
│  YOLO11m main model (J, RTX 4060, 8 GB VRAM)                     │
│                                                                  │
│  Dataset:   All 7 country folders, SAMPLE_RATIO=1.0              │
│  Model:     YOLO11m (COCO-pretrained, fine-tuned)                │
│  Config:    batch=16, imgsz=640, amp=True (FP16),                │
│             same protocol as Phase 0 (seed=42, fixed val set,     │
│             CRDDC2022 F1 reporting).                              │
│  Goal:      Final reported main model. Replaces the originally    │
│             planned A100 cluster YOLO11m run.                     │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  PHASE 1 — A100 cluster (NOT EXECUTED)                           │
│                                                                  │
│  Documented in `scripts/train_cluster.sh` and                    │
│  `scripts/submit_sweep.sh`. Out of scope for the final           │
│  deliverable — see project status note at the top of this        │
│  document.                                                       │
└──────────────────────────────────────────────────────────────────┘
```

**Why was Phase 1 dropped?** Local compute on M's RTX 4050 and J's RTX 4060
proved sufficient to produce both the YOLO11s baseline and the YOLO11m main
model within the project timeline. The A100 cluster path was kept as a
design artifact but spending the cluster credits was no longer required to
hit the project objectives.

**Why does FP16 (mixed precision) matter on consumer GPUs?** The RTX 4050
laptop has 6 GB of VRAM and the RTX 4060 has 8 GB. At FP32, YOLO11m at 640px
struggles for batch sizes above 4–8 on these cards. FP16 halves memory usage
with negligible accuracy impact. It is enabled by default in Ultralytics
(`amp=True`).

### Early stopping

YOLO11 runs on RDD2022 typically plateau between epoch 40 and 60. Training for fixed epochs wastes compute and increases the risk of overfitting the validation set. Ultralytics supports early stopping natively via the `patience` argument, which halts training if the monitored metric does not improve for N consecutive epochs.

| Track | patience | Monitor | Rationale |
|-------|----------|---------|-----------|
| YOLO11s baseline (M, RTX 4050) | 15 | `metrics/mAP50` | Short runs, noisy val, stop early if stuck. |
| YOLO11m main (J, RTX 4060) | 20 | `metrics/mAP50` | Longer runs, tolerate more plateaus before stopping. |
| ~~Phase 1 (A100)~~ | ~~20~~ | ~~`metrics/mAP50`~~ | ~~Not executed in the final deliverable.~~ |

The monitored metric is Ultralytics' built-in `metrics/mAP50` on the fixed 1,000-per-country validation set. Early stopping does not replace epoch budgets — it bounds them. The `last.pt` from an early-stopped run is still uploaded to Backblaze alongside `best.pt`.

### Experiment tracking

Every training run is logged to **MLflow**: full hyperparameter config, per-epoch metrics, random seed (fixed at 42 for all runs), and checkpoint path. Results are also written to the `experiments` collection in MongoDB. The random seed is non-negotiable: without it, runs are not reproducible and comparisons between models are unreliable.

---

## Stage 4 — Evaluation

### Protocol

We follow the official challenge evaluation protocol used in GRDDC/CRDDC/ORDDC:
- **IoU threshold: 0.5** — a detection is correct if its bounding box overlaps the ground truth by at least 50%.
- **Confidence threshold: 0.5** — predictions below this are discarded.

This protocol ensures our results are directly comparable to published benchmarks.

### Metrics

The **official CRDDC2022 ranking metric is F1-score** at IoU=0.5 (source: https://crddc2022.sekilab.global/overview/). Our primary reported metric therefore follows the same protocol so results are directly comparable with the challenge leaderboards. mAP@0.5 is still reported as the standard detection metric and as a training-time tracker.

| Metric               | What it measures                                     | Why we report it                                                                             |
| -------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **F1 score**         | Harmonic mean of precision and recall                | **Primary metric — official CRDDC2022 ranking criterion.** Reported per class and globally. |
| **mAP@0.5**          | Mean Average Precision across all classes at IoU=0.5 | Standard detection metric. Used during training for early stopping and `best.pt` selection.  |
| **Precision**        | Of all detections made, how many were correct        | High precision = few false alarms.                                                           |
| **Recall**           | Of all real damages, how many were detected          | High recall = few missed damages.                                                            |
| **Confusion matrix** | Misclassifications between damage types              | Reveals systematic errors (e.g., D00 labelled as D10).                                       |

All metrics are reported **per class** and **globally**. Per-class results are essential because the class imbalance means a high global score can hide poor D40 performance. The CRDDC2022 leaderboard also maintains **per-country F1**; we follow the same convention and report F1 per country as well as the country-average overall F1.

### Qualitative evaluation

A sample of **50 images per class** (200 total) is inspected visually, focusing on:
- **False positives**: detections where there is no real damage (noise, shadows, road markings misclassified).
- **False negatives**: real damages the model missed, especially D40 potholes.

Sampling per class rather than globally ensures D40 — the rarest and most important class — gets adequate visual coverage and is not drowned out by the more frequent D00 samples.

### Test set discipline

The **official RDD2022 test split** is the fixed test set — for all six countries, always at 100%. It is touched exactly once: when reporting final results. It is never used for hyperparameter tuning, architecture selection, or any training decision. Violating this rule makes results incomparable with published benchmarks.

---

## Stage 5 — Inference Module

### What it does

The inference module takes an image or a directory of images, runs the production model (the one marked `is_production=True` in MongoDB), and produces:
1. Annotated images with bounding boxes, class labels, and confidence scores drawn.
2. A structured JSON result per image written to the `predictions` collection in MongoDB.

### Video input

The model processes images, not raw video. For video input, a frame extraction script samples **1 frame per second** from the video file (configurable rate). Road damage is static — it does not move — so sampling at 1 fps provides full coverage without processing redundant duplicate frames. The extracted frames are passed to the inference module as a directory.

There is no requirement to run the model on a mobile device. The workflow is:

```
Mobile phone (capture video/photos)
        ↓
Transfer to PC
        ↓
Frame extraction (if video)
        ↓
Inference module on PC
        ↓
Annotated results + MongoDB
```

This approach is not a compromise — it is the correct design choice. It removes the constraint of running on mobile hardware, which would force the use of smaller, less accurate model variants (nano/small). Running on PC allows YOLO11m with no restrictions.

### Model export

After training, the best checkpoint is exported to **ONNX format**. The inference module loads the ONNX model, not the PyTorch checkpoint. This removes the runtime dependency on Ultralytics and makes the inference pipeline self-contained and portable.

### Web demo — `src/api/` (complete)

A minimal FastAPI application (`src/api/main.py`) provides a browser interface
to the inference pipeline. It is a presentation asset — not a core deliverable —
and was built after Steps 5–7 were complete.

**Routes:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the single-page frontend (`src/api/static/index.html`). |
| `GET` | `/model` | Returns current production model metadata (run_id, model, F1, mAP50, sample_ratio, timestamp). |
| `POST` | `/predict` | Accepts an image upload (jpg/png). Returns JSON detections and a base64-encoded annotated image. Writes one document to MongoDB `predictions`. |

The API loads the `is_production=True` model once per process (cached in
`_model_cache`). Model resolution follows the same checkpoint-lookup order as
`src/inference/predict.py`.

**Run:**
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
# Then open http://localhost:8000
```

**Frontend (`src/api/static/index.html`):** drag-and-drop image upload,
annotated result display with per-class colour coding, and a detections table
showing label, bounding box, and confidence score.

---

## Stage 6 — Retraining Pipeline

**Status:** Out of scope **for execution**, not for design.
`src/training/retrain.py` is implemented and smoke-tested on the synthetic
mini-dataset (`tests/data/tiny_rdd2022/`, 14 images, 1 epoch). It is **not
executed on real new data** before submission. The code and the design
described below are preserved as a documented design artifact — the
project's retraining philosophy is unchanged.

Three independent constraints stopped us from running it on real data
before the deadline:

1. **No new data.** We have no fresh batch of road images outside RDD2022.
   A retrain with no genuinely new data would not exercise the workflow
   in any meaningful way — it would just be another fine-tune on the same
   dataset.
2. **Team bandwidth.** With M finishing the baseline + web demo and J
   finishing the YOLO11m main model, no one has the remaining hours to
   drive a real retrain through to a promotion-or-regression decision and
   write it up.
3. **Compute cost on personal hardware.** A real retrain on top of the
   production weights (full mixed training, ~20 epochs, val + B2 upload
   + evaluation afterwards) is expensive on M's RTX 4050 and J's RTX 4060
   and would crowd out the runs that are still required for the final
   report.

The design intent — manually triggered fine-tuning on freshly ingested
imagery, with atomic `is_production` promotion gated by F1 — remains the
recommended long-term workflow for this system.

### Design philosophy

Retraining is implemented as a **manually triggered function**, not an autonomous continuous process. This is a deliberate scope decision: autonomous retraining systems require monitoring, rollback logic, data drift detection, and alerting infrastructure that is out of scope for a two-month project. A manually triggered function fulfils the project requirement, is fully auditable, and is achievable in the available time.

### How it works

```bash
python -m src.training.retrain \
    --new-images path/to/new_images/ \
    --epochs 20 \
    --batch 8 \
    --patience 10
```

The `retrain()` function pipeline (see `src/training/retrain.py` for the full step-by-step breakdown):

1. **Pre-flight** — raises `EnvironmentError` if `RDD_DATA_ROOT` is not set.
2. **Ingest** — writes new image metadata to MongoDB `images_metadata` (idempotent; skips existing `image_id`s). Calls `src.data.ingest.ingest` directly.
3. **Collect images** — walks `--new-images` recursively; supports flat directories and RDD2022-style `country/split/images/` hierarchy.
4. **Build mixed training list** — new images + stratified sample of the original training pool (`--mix-ratio`, default 0.30). Val split uses the fixed val set from `logs/splits.json`.
5. **Write data.yaml** — generates `logs/retrain_images_{run_id}.txt`, `logs/retrain_mixed_{run_id}.txt`, `logs/retrain_val_{run_id}.txt`, and `logs/retrain_data_{run_id}.yaml`.
6. **Insert MongoDB document** — `status="running"` + SIGTERM handler installed.
7. **Fine-tune** — `YOLO(production_checkpoint).train(seed=42, ...)`. run_id format: `run_YYYYMMDD_HHMMSS_{model}_retrain`.
8. **Extract metrics** — reads `results.csv`; training-time metrics only.
9. **Update MongoDB** — `status="completed"` with training-time metrics, before any export/upload.
10. **Export ONNX** — subprocess call to keep crashes isolated from the main process.
11. **Upload B2** — calls `src.training.upload_checkpoint.upload_checkpoints`.
12. **Evaluate + promote** — calls `src.evaluation.evaluate.evaluate(run_id=..., split="val")` on the fixed 1,000-per-country validation set, then `src.training.promote.maybe_promote` with the returned CRDDC2022 F1.

Checkpoint resolution order: `--model` override → `runs/train/<run_id>/weights/best.pt` → fuzzy match → B2 download from MongoDB.

### Promotion rule

| Condition                                          | Outcome                                                                                                 |     |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | --- |
| `F1_new > F1_current + 0.01`                       | Promoted — `is_production` flipped atomically via MongoDB transaction. Old model marked `"superseded"`. |     |
| `F1_new ∈ [F1_current − 0.005, F1_current + 0.01]` | Noise band — logged as `"completed"`, not promoted.                                                     |     |
| `F1_new < F1_current − 0.005`                      | Regression — logged as `"completed"`. Investigate before next retrain.                                  |     |

### Why fine-tune from checkpoint instead of retraining from scratch?

Fine-tuning from an existing checkpoint is faster (converges in fewer epochs), typically more accurate (starts from a strong initialisation), and preserves the knowledge already learned from the original dataset. Retraining from scratch each time would be wasteful and would make retraining on the laptop impractical.

For full documentation of the retraining pipeline see `DOCUMENTATION/IN DETAIL/retraining.md`.

---

## Summary: Key Technical Decisions

| Decision            | Choice                                   | Reason                                                                                                           |
| ------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Architecture        | YOLO11m                                  | Empirically best on RDD2022 specifically. Better than YOLOv8 with fewer parameters.                              |
| Baseline model      | YOLO11s                                  | Fast to train anywhere. Guarantees a presentable result under any circumstance.                                  |
| Dataset             | RDD2022                                  | Most current benchmark dataset. 6 countries, 47k+ images, CC BY-SA 4.0.                                          |
| Test set            | Official RDD2022 test split, always 100% | Comparable with published benchmarks. Never touched during training.                                             |
| Database            | MongoDB                                  | Heterogeneous document structure fits naturally. Images stay on filesystem.                                      |
| YOLO11s baseline    | RTX 4050 laptop (M)                      | Pipeline validation + F1-vs-data sweep + final reported baseline (F1=0.598).                                     |
| YOLO11m main model  | RTX 4060 (J)                             | Final reported main model. Replaces the originally planned A100 cluster run (out of scope, see status note).     |
| ~~Cluster phase~~   | ~~A100, full dataset~~                   | ~~Out of scope for the final deliverable. Scripts preserved as design artifact.~~                                |
| Video handling      | Frame extraction at 1 fps                | Damage is static. No need to process every frame. No mobile deployment required.                                 |
| Retraining          | Manually triggered fine-tuning           | Achievable in scope. Fully auditable. Meets project requirements.                                                |
| Experiment tracking | MLflow + MongoDB                         | MLflow for training curves and configs. MongoDB for production model registry.                                   |
| Class imbalance     | Per-class loss weights + MixUp on D40    | D40 (potholes) is critically underrepresented. Must be compensated explicitly.                                   |
| Random seed         | 42, fixed in all runs                    | Non-negotiable reproducibility requirement.                                                                      |

---

## Project Context and Constraints
*This section is not part of the technical pipeline. It exists to preserve project context across working sessions.*

### Team

- **M** — Project lead. Responsible for pipeline architecture, data pipeline (Step 2), Phase 0 / YOLO11s baseline training, evaluation (Step 5), inference module (Step 6), web demo (Step 8), and overall delivery.
- **L** — Responsible for **MongoDB only**: Atlas cluster, collections, schemas, indexes, connection. L does not train models.
- **J** — Responsible for training the **YOLO11m main model on the RTX 4060** (replaces what was originally planned as the A100 cluster YOLO11m run).

### Hardware
- **RTX 4050 Laptop (M, 6 GB VRAM):** YOLO11s and YOLO11m fit with FP16 (`amp=True`) and batch=8 at imgsz=640. YOLO11l and above do not fit reliably. This is the machine where the YOLO11s baseline (`run_20260504_202658_yolo11s`) was trained.
- **RTX 4060 (J, 8 GB VRAM):** YOLO11m fits at batch=16, imgsz=640, FP16. In the final deliverable this is the machine where the **YOLO11m main model** is trained — it replaces the originally planned A100 cluster run.
- **University A100 (40 GB VRAM) — not used in the final deliverable.** Originally planned for Phase 1 full-performance training. The SLURM scripts (`scripts/train_cluster.sh`, `scripts/submit_sweep.sh`) are preserved in the repository as a design artifact but were not executed. See project status note at the top of this document.

### Cluster Access Details (documented — not executed)

> The information below describes the A100 cluster environment that the
> repository was designed to target. **No SLURM job was actually submitted
> for the final deliverable.** This section is kept for documentation
> completeness and for any future continuation of the project.

- **Job scheduler:** SLURM — jobs submitted with `sbatch`.
- **Internet access:** Yes — MongoDB Atlas connection from training jobs is viable. No need for a separate `sync_to_mongo()` step; the training script can write directly to Atlas during and after training.
- **Time limit:** No hard limit expected given low cluster demand. Do not abuse this.
- **Local storage per node:** 50 GB. RDD2022 processed (~12 GB) fits comfortably. Copy the dataset to local node storage at the start of each job — do not read from network storage during training.
- **Code format:** Submit Python scripts only. Do not use Jupyter notebooks for cluster jobs.

**Cluster job submission (reference only):**
`scripts/train_cluster.sh` accepts the following via `--export`:
`MODEL`, `SAMPLE_RATIO`, `EPOCHS`, `BATCH`, `PATIENCE`, `DEVICE`, `CACHE`,
`LR0`, `LRF`, `COS_LR`, `OPTIMIZER`, `DATA_ROOT`, `SMOKE_TEST`.
For multi-config sweeps use `scripts/submit_sweep.sh`.

```bash
# Example invocation (documented — not executed for the final deliverable)
sbatch --export=MODEL=yolo11m,SAMPLE_RATIO=1.0,EPOCHS=100,BATCH=32,PATIENCE=20,DATA_ROOT=/path/to/rdd2022,DEVICE=0,CACHE=disk scripts/train_cluster.sh
```

### Shared Database Strategy

Since the team members work on different machines (M on the RTX 4050 laptop, J on the RTX 4060), **MongoDB Atlas** (free tier) is used as the single shared database instance.

- One Atlas cluster, one connection URI shared across all machines.
- The URI is stored in a `.env` file that is never committed to Git.
- All experiments, predictions, and metadata written by any machine are immediately visible to the rest of the team.

**Checkpoint storage:** after every training run — on M's laptop or J's RTX 4060 — the script automatically uploads `best.pt`, `last.pt`, and `best.onnx` to **Backblaze B2** (free tier, sufficient for weights alone). MongoDB Atlas stores the resulting public URLs. No machine needs to be kept online as a checkpoint server. Any team member can download any model version at any time from the stored URL.

**Dataset storage:** RDD2022 is processed once (download → validate → convert → split) by one team member and uploaded to cloud storage. All other machines pull from there. The raw Sekilab ZIPs are never downloaded more than once.

### Timeline
~2 months to final results. This constraint is fixed and eliminates: ensemble methods, semi-supervised learning, transformer-only architectures, knowledge distillation, and any two-stage detector approach. All of these are out of scope by time, not by technical merit.


### Qualitative Demo Video

No annotation required. Two options, either is valid:

- **Record own footage:** phone mounted on dashboard or held by a passenger, 10–15 min driving through secondary streets, industrial areas, or any zone likely to have surface damage. One session is enough.
- **Use dashcam footage from YouTube:** download a video with visibly deteriorated road surface (Spanish roads preferred for relevance). Extract frames and run inference.

The video is used exclusively as a demo asset — to show the model detecting damage on real unseen footage. It is not used for quantitative evaluation and requires no annotation.

### imgsz=1280 Experiment

The standard training resolution is imgsz=640, which is consistent with all published RDD2022 benchmarks. imgsz=1280 can improve detection of fine cracks that are missed at lower resolution, but requires more VRAM and longer training time.

This was treated as an **optional post-baseline experiment**:
1. Train and evaluate the full pipeline at imgsz=640 first.
2. If hardware and time allow, launch a second YOLO11m run at imgsz=1280.
3. Compare mAP@0.5. If it improves, report both results. If there is no time, imgsz=640 is fully valid and comparable with the literature.

Given that Phase 1 / A100 was dropped from the final deliverable and the
YOLO11m main model runs on J's RTX 4060 (8 GB VRAM), imgsz=1280 is **not
attempted** — there isn't enough VRAM headroom at the required batch size.
imgsz=640 results stand as the final reported numbers.

### Confirmed Out of Scope (do not reopen)
- FastAPI / REST API service as a core deliverable. It exists only as an optional demo layer built on top of a finished pipeline.
- Mobile deployment or on-device inference.
- Ensemble methods (YOLO + Cascade RCNN + DETR).
- Semi-supervised learning or knowledge distillation.
- Automatic retraining triggers.
- Manual annotation of own video footage.

---

*This document should be updated whenever a significant pipeline decision changes.*
