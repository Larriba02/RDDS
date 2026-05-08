# Data Ingestion — In Detail
**RDDS · Step 2**  
*Version 1.1 — May 2026*

This document explains the data ingestion pipeline in depth: what each module
does, why it is designed that way, and what to watch out for when running it
on real data.

---

## Overview

The ingestion pipeline converts the raw RDD2022 dataset — country ZIPs with
JPEG images and PascalVOC XML annotations — into:

1. YOLO `.txt` label files in a parallel `labels/` directory alongside each `images/` directory (Ultralytics convention, e.g. `Japan/train/labels/Japan_000001.txt` for `Japan/train/images/Japan_000001.jpg`).
2. A split assignment (`train` / `val` / `test`) for every image.
3. MongoDB documents in `images_metadata` for cross-machine access.
4. `logs/class_distribution.json` for training class-weight calibration.
5. Backblaze B2 upload so teammates and the cluster can pull the processed data.

Run order (all idempotent — safe to re-run):

```
download → validate → convert → analyse_distribution → split → ingest → upload_to_cloud
```

---

## Module reference

### `src/data/download.py`

Downloads country ZIPs from the Sekilab S3 bucket and extracts them under
`RDD_DATA_ROOT`.

- ZIPs go into `RDD_DATA_ROOT/_zips/` and are kept after extraction (for
  debugging and re-extraction without re-downloading).
- Skips download if the ZIP file already exists on disk.
- Skips extraction if the country directory already exists.
- The seven countries are: Japan, India, Czech, Norway, United_States,
  China_MotorBike, China_Drone.

Entry point: `python -m src.data.download [--dest DIR] [--countries C1 C2 ...]`

### `src/data/validate.py`

Parses every PascalVOC XML file and discards any bounding box that fails one
of four rules:

| Rule | What it catches |
|------|-----------------|
| Unknown label | Anything not in {D00, D10, D20, D40} — e.g. noise annotations |
| Negative coordinate | xmin < 0 or ymin < 0 |
| Out-of-bounds | xmax > width or ymax > height |
| Degenerate bbox | xmin ≥ xmax or ymin ≥ ymax (zero or negative area) |

All discarded boxes are logged with their reason to `logs/discarded_annotations.txt`.
The returned dict (`{rel_path: [valid_annotations]}`) is the clean annotation
surface used by `convert.py` and `ingest.py`.

Why this matters: RDD2022 contains a non-trivial number of out-of-bounds
annotations, especially in early versions of the dataset. If they reach the
loss function they silently corrupt training without raising an error.

Entry point: `python -m src.data.validate [--data-root DIR]`

### `src/data/convert.py`

Converts clean PascalVOC XML annotations to YOLO `.txt` format.

**Class map (fixed — do not change without updating `train.py` and `split.py`):**

| Code | Class ID | Damage type |
|------|----------|-------------|
| D00  | 0        | Longitudinal crack |
| D10  | 1        | Transverse crack |
| D20  | 2        | Alligator crack |
| D40  | 3        | Pothole |

Each `.txt` file is placed in a parallel `labels/` directory next to the `images/` directory for its split (e.g. `Japan/train/labels/Japan_000001.txt`). This matches the Ultralytics path convention required during training. Files are named with the same stem as the image. The YOLO format is:

```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```

All coordinates are normalised to [0, 1]. Images with no valid annotations
get an empty `.txt` file (YOLO background convention).

The conversion applies the same validation rules as `validate.py` so it is
safe to run independently without calling `validate.py` first.

Entry point: `python -m src.data.convert [--data-root DIR] [--force]`

### `src/data/analyse_distribution.py`

Counts annotation instances per class per country from the YOLO `.txt` files
(run after `convert.py`). Outputs a table to stdout and saves
`logs/class_distribution.json`.

The JSON includes a `cls_weights` key with inverse-frequency weights
(mean-normalised to 1.0). These are loaded by `train.py` to set the
`cls_weight` parameter in Ultralytics, compensating for the severe class
imbalance in RDD2022 (D00 longitudinal cracks are massively overrepresented
versus D40 potholes).

**Do not skip this step.** Without calibrated class weights, the model learns
to detect D00 well and misses D40 (potholes — the operationally most important
class) consistently.

Entry point: `python -m src.data.analyse_distribution [--data-root DIR] [--split train]`

### `src/data/split.py`

Assigns every image to one of three roles: `train`, `val`, or `test`.

Rules (non-negotiable per CLAUDE.md §2):

1. **Official test split → always `test`.** Never sampled. Never used during
   training or hyperparameter tuning.
2. **Fixed validation set**: exactly 1,000 images per country reserved for
   validation, **stratified by (country, dominant damage class)** using
   `sklearn.model_selection.StratifiedShuffleSplit` with `random_state=42`.
   The dominant class per image is the most frequent label; ties broken
   alphabetically.
3. **Training images**: the remainder is sampled at `SAMPLE_RATIO` (default
   1.0). Sampling is per-country and deterministic (seed 42).

Output files:
- `logs/splits.json` — maps `image_id` → split string.
- `logs/split_summary.txt` — human-readable counts per country and split.

Images excluded by `SAMPLE_RATIO < 1.0` are stored as `"excluded"` in
`splits.json` and skipped by `ingest.py`.

**Fallback for tiny datasets**: when a country has fewer images than
`VAL_PER_COUNTRY` (1 000), stratified splitting is skipped and all images
for that country are assigned `"train"`. This affects only the synthetic
mini-dataset used for smoke tests.

**`SAMPLE_RATIO` design note**: `split.py` should always be run with
`SAMPLE_RATIO=1.0` (the default and the recommended value in `.env`).
This ensures the full training pool is committed to `splits.json` and
available to all team members. The actual fraction of training images used
in a given experiment is controlled at run-time by `train.py --sample-ratio`
(Phase 0 iterations use `--sample-ratio 0.10`, `0.25`, `0.50`, `1.0` in
sequence). Do not set `SAMPLE_RATIO < 1.0` in `.env` unless you intend to
permanently exclude images from the pool stored in MongoDB.

The `image_id` is the MD5 hash of the relative filepath (e.g.
`Japan/train/00001.jpg`). This makes IDs machine-independent: any team
member who downloads RDD2022 independently generates the same IDs.

Entry point: `python -m src.data.split [--data-root DIR] [--sample-ratio RATIO]`

### `src/data/ingest.py`

Writes one MongoDB document per image to `images_metadata`. Reads
`logs/splits.json` for split assignments.

- Idempotent: skips any image whose `image_id` already exists.
- Reads dimensions and annotations from the XML file; falls back to Pillow
  image read if the XML is missing (e.g. images without annotations in
  the test split).
- Uses batched `insert_many` (default batch size 500) for throughput.

Entry point: `python -m src.data.ingest [--data-root DIR] [--batch-size 500]`

### `src/data/upload_to_cloud.py`

Uploads the processed data (YOLO labels and logs) to Backblaze B2 using the
S3-compatible API. Pass `--include-images` to also upload the JPEG files
(large — only needed if teammates cannot access RDD2022 independently).

The B2 endpoint is read from `BACKBLAZE_ENDPOINT` in `.env` (default:
`https://s3.us-west-004.backblazeb2.com`). Change this if your bucket is in
a different region.

Entry point: `python -m src.data.upload_to_cloud [--data-root DIR] [--include-images]`

---

## Synthetic mini-dataset (`tests/data/tiny_rdd2022/`)

A self-contained dataset committed to the repository. Contains:
- 2 countries (Japan, Czech)
- 5 training images + 2 test images per country
- All 4 damage classes covered across the training images
- Pre-generated JPEG images, PascalVOC XML annotations, and YOLO `.txt` labels

Purpose:
1. Smoke-test every pipeline stage without downloading the real dataset.
2. Cluster smoke test in Step 4: `sbatch` a 1-epoch run on this dataset
   before queuing the full A100 job.

Regenerate with:
```
python tests/data/generate_tiny_dataset.py
```

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RDD_DATA_ROOT` | Yes | Absolute path to the local dataset root |
| `MONGO_URI` | Yes (ingest only) | Atlas connection string |
| `BACKBLAZE_KEY_ID` | Yes (upload only) | B2 application key ID |
| `BACKBLAZE_APP_KEY` | Yes (upload only) | B2 application key secret |
| `BACKBLAZE_BUCKET` | Yes (upload only) | Bucket name |
| `BACKBLAZE_ENDPOINT` | No | B2 S3 endpoint (default: us-west-004) |
| `SAMPLE_RATIO` | No | Training fraction for `split.py` (default: 1.0; keep at 1.0 — use `train.py --sample-ratio` to control per-run subsampling) |

---

## Common failure modes

**`RDD_DATA_ROOT is not set`** — copy `.env.example` to `.env` and fill in
the path to the extracted dataset.

**`logs/splits.json not found. Run split.py first.`** — `ingest.py` depends
on `split.py` having been run. Run in order.

**Discard count unexpectedly high** — open `logs/discarded_annotations.txt`
and inspect. Out-of-bounds bboxes are normal in RDD2022 (a few percent).
Unknown labels would indicate a dataset version mismatch.

**B2 upload fails with `NoSuchBucket`** — verify `BACKBLAZE_BUCKET` and
`BACKBLAZE_ENDPOINT` in `.env` match your actual bucket region.

---

## CLI reference

All commands are run from the repo root with the venv activated.

---

### `python -m src.data.download` — Download RDD2022 country ZIPs

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
python -m src.data.download
```

**macOS / Linux**
```bash
source .venv/bin/activate
python -m src.data.download
```

#### Flags

| Flag | Type / Options | Default | Effect | When to use |
|------|---------------|---------|--------|-------------|
| `--dest` | `path` | `RDD_DATA_ROOT` from `.env` | Root directory where ZIPs and country folders are saved | Override when downloading to a non-default location |
| `--countries` | one or more strings | all 7 countries | Restrict which countries to download | Re-download one country, or test the pipeline with Japan only |

Available country names: `Japan`, `India`, `Czech`, `Norway`, `United_States`, `China_MotorBike`, `China_Drone`.

#### Full example

```powershell
# Windows
python -m src.data.download --dest D:\datasets\rdd2022 --countries Japan India Czech
```
```bash
# macOS / Linux
python -m src.data.download --dest /data/rdd2022 --countries Japan India Czech
```

---

### `python -m src.data.validate` — Validate PascalVOC XML annotations

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
python -m src.data.validate
```

**macOS / Linux**
```bash
source .venv/bin/activate
python -m src.data.validate
```

#### Flags

| Flag | Type / Options | Default | Effect | When to use |
|------|---------------|---------|--------|-------------|
| `--data-root` | `path` | `RDD_DATA_ROOT` from `.env` | Dataset root directory to validate | Override when the dataset lives outside `RDD_DATA_ROOT` |

#### Full example

```powershell
# Windows
python -m src.data.validate --data-root D:\datasets\rdd2022
```
```bash
# macOS / Linux
python -m src.data.validate --data-root /data/rdd2022
```

---

### `python -m src.data.convert` — Convert XML annotations to YOLO `.txt`

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
python -m src.data.convert
```

**macOS / Linux**
```bash
source .venv/bin/activate
python -m src.data.convert
```

#### Flags

| Flag | Type / Options | Default | Effect | When to use |
|------|---------------|---------|--------|-------------|
| `--data-root` | `path` | `RDD_DATA_ROOT` from `.env` | Dataset root directory | Override when the dataset lives outside `RDD_DATA_ROOT` |
| `--force` | flag | off | Overwrite existing `.txt` label files | Re-run conversion after fixing annotation issues without deleting label files manually |

#### Full example

```powershell
# Windows
python -m src.data.convert --data-root D:\datasets\rdd2022 --force
```
```bash
# macOS / Linux
python -m src.data.convert --data-root /data/rdd2022 --force
```

---

### `python -m src.data.analyse_distribution` — Count class instances per country

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
python -m src.data.analyse_distribution
```

**macOS / Linux**
```bash
source .venv/bin/activate
python -m src.data.analyse_distribution
```

#### Flags

| Flag | Type / Options | Default | Effect | When to use |
|------|---------------|---------|--------|-------------|
| `--data-root` | `path` | `RDD_DATA_ROOT` from `.env` | Dataset root directory | Override when the dataset lives outside `RDD_DATA_ROOT` |
| `--split` | `train` \| `test` | `train` | Which split's label files to analyse | Pass `test` if you want counts for the test split (no labels, so always 0) |

#### Full example

```powershell
# Windows
python -m src.data.analyse_distribution --data-root D:\datasets\rdd2022 --split train
```
```bash
# macOS / Linux
python -m src.data.analyse_distribution --data-root /data/rdd2022 --split train
```

---

### `python -m src.data.split` — Assign train / val / test splits

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
python -m src.data.split
```

**macOS / Linux**
```bash
source .venv/bin/activate
python -m src.data.split
```

#### Flags

| Flag | Type / Options | Default | Effect | When to use |
|------|---------------|---------|--------|-------------|
| `--data-root` | `path` | `RDD_DATA_ROOT` from `.env` | Dataset root directory | Override when the dataset lives outside `RDD_DATA_ROOT` |
| `--sample-ratio` | `float` (0.0–1.0) | `SAMPLE_RATIO` from `.env` (default 1.0) | Fraction of training images to include in `splits.json` | Keep at 1.0 here; use `train.py --sample-ratio` for per-run subsampling instead |

#### Full example

```powershell
# Windows — always run with sample-ratio 1.0 (recommended)
python -m src.data.split --data-root D:\datasets\rdd2022 --sample-ratio 1.0
```
```bash
# macOS / Linux
python -m src.data.split --data-root /data/rdd2022 --sample-ratio 1.0
```

---

### `python -m src.data.ingest` — Write image metadata to MongoDB

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
python -m src.data.ingest
```

**macOS / Linux**
```bash
source .venv/bin/activate
python -m src.data.ingest
```

#### Flags

| Flag | Type / Options | Default | Effect | When to use |
|------|---------------|---------|--------|-------------|
| `--data-root` | `path` | `RDD_DATA_ROOT` from `.env` | Dataset root directory | Override when the dataset lives outside `RDD_DATA_ROOT` |
| `--batch-size` | `int` | `500` | Documents per bulk-write batch | Lower (e.g. `200`) if Atlas free tier throttles writes; raise if ingest is slow on a fast connection |

#### Full example

```powershell
# Windows
python -m src.data.ingest --data-root D:\datasets\rdd2022 --batch-size 500
```
```bash
# macOS / Linux
python -m src.data.ingest --data-root /data/rdd2022 --batch-size 500
```

---

### `python -m src.data.upload_to_cloud` — Upload processed data to Backblaze B2

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
python -m src.data.upload_to_cloud
```

**macOS / Linux**
```bash
source .venv/bin/activate
python -m src.data.upload_to_cloud
```

#### Flags

| Flag | Type / Options | Default | Effect | When to use |
|------|---------------|---------|--------|-------------|
| `--data-root` | `path` | `RDD_DATA_ROOT` from `.env` | Dataset root directory | Override when the dataset lives outside `RDD_DATA_ROOT` |
| `--include-images` | flag | off | Also upload JPEG image files (large) | Only needed if teammates cannot independently access RDD2022 from Sekilab S3 |
| `--workers` | `int` | `8` | Parallel upload threads | Lower (e.g. `4`) on slow connections; raise (e.g. `16`) on fast cluster links |

#### Full example

```powershell
# Windows
python -m src.data.upload_to_cloud --data-root D:\datasets\rdd2022 --include-images --workers 8
```
```bash
# macOS / Linux
python -m src.data.upload_to_cloud --data-root /data/rdd2022 --include-images --workers 8
```
