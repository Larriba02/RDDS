# Evaluation — In Detail

**Road Damage Detection System · Step 5**  
*Version 1.1 — May 2026*

---

## 1. Dataset split structure

The RDD2022 dataset ships with two distinct image sets per country:

```
data/rdd2022/{country}/train/images/   ← with PascalVOC XML annotations
data/rdd2022/{country}/test/images/    ← no annotations
```

Our `split.py` respects that structure and produces three non-overlapping
sets:

| Set | Images | Origin | Labels | Purpose |
|-----|--------|--------|--------|---------|
| **train** | 31,395 | `train/images/` (random sample) | ✅ | Gradient updates (backprop) |
| **val** | 7,000 | `train/images/` (held out before training) | ✅ | Early stopping + reported metrics |
| **test** | 9,039 | `test/images/` (official competition holdout) | ❌ | Qualitative inspection only |

Per-country breakdown:

```
Country                  train      val     test    total
─────────────────────────────────────────────────────────
China_Drone               1401     1000        0     2401
China_MotorBike            977     1000      500     2477
Czech                     1834     1000      711     3545
India                     6706     1000     1959     9665
Japan                     9511     1000     2629    13140
Norway                    7161     1000     2040    10201
United_States             3805     1000     1200     6005
─────────────────────────────────────────────────────────
TOTAL                    31395     7000     9039    47434
```

**Why is the val set exactly 7,000?**  
It is a deliberate design decision: `split.py` takes exactly 1,000 images
per country from the training pool, stratified by dominant damage class
(`StratifiedShuffleSplit`, `RANDOM_SEED=42`). The 7,000 figure is round
because we made it round, not because it matches any official partition.

**Why does the test set have irregular numbers per country?**  
Because they are the images the CRDDC2022 organisers placed in the `test/`
directories — we have no control over those counts.  China_Drone provided
no test images at all.

**Why does the test set have no labels?**  
The `test/images/` files are the official competition holdout.  The
organisers kept the ground-truth annotations on their servers so that
participants could not overfit to the test set.  Scores were obtained by
uploading predictions to the challenge server, which has been closed since
2022.  Locally we cannot compute F1 or mAP on the test set.

---

## 2. How training uses the val set

During each training epoch, Ultralytics follows this sequence:

```
Epoch N:
  1. Forward + backward pass on train set  → weights updated (backprop)
  2. Inference on val set                  → val mAP50 computed
  3. If val mAP50 is best so far           → save best.pt
  4. If no improvement for `patience` epochs → stop training
```

The val set **never enters backpropagation**.  It influences the model in
one way only: it determines which epoch checkpoint is saved as `best.pt`
(step 3) and when to stop (step 4).  Both of those are based on val mAP50
specifically — not on training loss, not on training accuracy.

---

## 3. Reported metrics and the val set

Because the official test set has no labels, **the val set is our reported
metric set**.  The F1, Precision, Recall, and mAP@0.5 numbers in Step 5
and in MongoDB `experiments.metrics` are all computed on the val set.

### Small selection bias

Using the same val set for both early stopping and final reporting
introduces a small **upward selection bias**: `best.pt` was chosen to
maximise val mAP50, so metrics on that same set are slightly more
optimistic than they would be on a completely independent test set.

With 7,000 val images and typical early stopping windows of 10–30 epochs,
this bias is estimated at **< 0.5–1 % in F1** — real but not large enough
to change conclusions about which model is better.

This is the standard accepted practice for competition ML when test labels
are not publicly available.  The CRDDC2022 winners faced the same
situation; they obtained unbiased scores by submitting to the challenge
server (now closed).

All reported metrics should include a note: *"Evaluated on the fixed
held-out validation set (1,000 images/country).  Metrics carry a small
upward selection bias due to early stopping on the same set."*

---

## 4. Val evaluation (quantitative)

Runs `model.val()` on the 7,000-image val set.  Computes:

- F1 @ IoU ≥ 0.5, conf = 0.5 — overall, per country, per class
- Precision and Recall @ IoU ≥ 0.5 — overall
- mAP@0.5 — overall and per country

Results are written to MongoDB `experiments.metrics.evaluation_val` and
also update the top-level `metrics.F1` / `metrics.mAP50` fields.

```bash
python -m src.evaluation.evaluate              # default: val split
python -m src.evaluation.evaluate --dry-run    # without MongoDB write
```

---

## 5. Test inference (no ground truth)

Runs `model.predict()` on the 9,039 official test images.  No F1 or mAP
can be computed.  What is collected:

- Number of images with at least one detection.
- Detection count and average confidence per class.

Per-image predictions are saved to `outputs/test_predictions_{run_id}.json`
for later inspection.  A summary is written to MongoDB
`experiments.metrics.evaluation_test`.

```bash
python -m src.evaluation.evaluate --split both   # val metrics + test inference
```

---

## 6. Qualitative samples

Visual inspection on top of the quantitative numbers.

### Val qualitative (`--split val`, default)

Selects up to 50 images per damage class from the val set (images that
contain at least one annotation of that class, sampled with RANDOM_SEED=42).

Each image shows:
- **Green boxes** — ground-truth annotations.
- **Red boxes** — model predictions above the confidence threshold.

Allows direct visual comparison of false positives (red without green) and
false negatives (green without red).

Output: `outputs/qualitative/val/{D00,D10,D20,D40}/`

### Test qualitative (`--split test`)

Selects up to 50 images per country from the official test split.  No
ground truth available — only model predictions (red boxes) are drawn.
Useful to spot obvious failure modes on truly unseen data.

Output: `outputs/qualitative/test/{country}/`

```bash
python -m src.evaluation.qualitative --split both
```

---

## 7. CLI reference

### `python -m src.evaluation.evaluate` — Quantitative CRDDC2022 evaluation

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
python -m src.evaluation.evaluate
```

**macOS / Linux**
```bash
source .venv/bin/activate
python -m src.evaluation.evaluate
```

#### Flags

| Flag | Type / Options | Default | Effect | When to use |
|------|---------------|---------|--------|-------------|
| `--run-id` | `str` | `None` (production model) | Target experiment `run_id` | Pass a specific `run_id` to evaluate a non-production run |
| `--split` | `val` \| `both` | `val` | `val` evaluates the 7,000-image val set with ground truth; `both` additionally runs inference on the 9,039 test images (no GT, detection summary only) | Use `both` when you want a test-split inference summary alongside the val metrics |
| `--conf` | `float` | `0.5` | Confidence threshold for predictions | Lower to `0.3` to see more detections (higher recall, lower precision); raise to `0.7` for high-precision inspection |
| `--iou` | `float` | `0.5` | IoU threshold for NMS and metric computation | Keep at `0.5` (CRDDC2022 protocol requirement) |
| `--device` | `str` | `"0"` | CUDA device: `"0"` for GPU, `"cpu"` for CPU | Pass `cpu` when no GPU is available |
| `--dry-run` | flag | off | Skip MongoDB write; results are printed but not stored | Use to inspect metrics without modifying the experiments document |

#### Full example

```powershell
# Windows — evaluate a specific run on both splits, dry run
python -m src.evaluation.evaluate `
    --run-id run_20260504_202658_yolo11s `
    --split both `
    --conf 0.5 `
    --iou 0.5 `
    --device 0 `
    --dry-run
```
```bash
# macOS / Linux — evaluate a specific run on both splits, dry run
python -m src.evaluation.evaluate \
    --run-id run_20260504_202658_yolo11s \
    --split both \
    --conf 0.5 \
    --iou 0.5 \
    --device 0 \
    --dry-run
```

---

### `python -m src.evaluation.qualitative` — Generate visual annotation samples

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
python -m src.evaluation.qualitative
```

**macOS / Linux**
```bash
source .venv/bin/activate
python -m src.evaluation.qualitative
```

#### Flags

| Flag | Type / Options | Default | Effect | When to use |
|------|---------------|---------|--------|-------------|
| `--run-id` | `str` | `None` (production model) | Target experiment `run_id` | Evaluate a non-production run's visual output |
| `--split` | `val` \| `test` \| `both` | `val` | `val` saves GT (green) + predictions (red); `test` saves predictions only (no GT available); `both` runs both | Use `both` for a complete visual review before presenting results |
| `--n-per-class` | `int` | `50` | Images to sample per damage class (val) or per country (test) | Lower to `10–25` for a quick sanity check; keep at `50` for a thorough review |
| `--conf` | `float` | `0.5` | Confidence threshold for displayed predictions | Lower to see more predictions on images with few detections |
| `--device` | `str` | `"0"` | CUDA device: `"0"` for GPU, `"cpu"` for CPU | Pass `cpu` when no GPU is available |

#### Full example

```powershell
# Windows — full qualitative review of production model, both splits
python -m src.evaluation.qualitative `
    --run-id run_20260504_202658_yolo11s `
    --split both `
    --n-per-class 50 `
    --conf 0.5 `
    --device 0
```
```bash
# macOS / Linux — full qualitative review of production model, both splits
python -m src.evaluation.qualitative \
    --run-id run_20260504_202658_yolo11s \
    --split both \
    --n-per-class 50 \
    --conf 0.5 \
    --device 0
```

---

## 8. Checkpoint resolution

Both scripts resolve the model checkpoint in the same order:

1. `runs/train/<run_id>/weights/best.pt` — standard Ultralytics output path.
2. Any `runs/train/*/weights/best.pt` whose directory starts with `run_id`
   — handles Ultralytics index suffixes (e.g. `run_id2`).
3. Download `best.pt` from the B2 URL in `experiments.checkpoints.best_pt`
   — used on machines that did not run training locally.
