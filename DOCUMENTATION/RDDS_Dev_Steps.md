# RDDS — Development Steps Reference
**Road Damage Detection System · Group 3 · UFV**  
*Version 1.2 — March 2026*

This document is a step-by-step development guide. It is designed to be pasted into a new conversation as working memory. Each step has a clear goal, the files/code to produce, and a done criterion. Steps must be completed in order — do not start a step until the previous one is done and verified.

---

## Context Summary

- **Project:** Road damage detection system using deep learning on RDD2022 dataset.
- **Team:** M (lead), L (MongoDB), J (TBD).
- **Hardware:** RTX 4050 laptop (6GB, Phase 0), RTX 4060 teammate (8GB), A100 cluster (40GB, Phase 1).
- **Cluster:** SLURM, `sbatch`, Python scripts only, internet access, 50GB local storage per node.
- **Database:** MongoDB Atlas (shared, free tier). URI in `.env`, never in Git.
- **Weights storage:** Backblaze B2. URLs stored in MongoDB after each run.
- **Dataset:** RDD2022 (CC BY-SA 4.0). 6 countries, ~47k images, PascalVOC XML annotations.
- **Architecture:** YOLO11s (baseline) + YOLO11m (main). COCO pretrained weights, fine-tuned.
- **Timeline:** ~2 months to final results.
- **Full pipeline reference:** RDDS_Pipeline.md (v2.0)
- **Claude memory document:** RDDS_Claude_Memory.md (paste at start of any new conversation)

---

## ✅ STEP 0 — Repository and Environment Setup — DONE

**Owner:** M  
**Status:** Complete

### What was done
- [x] Git installed (v2.53.0) and configured with user identity.
- [x] GitHub account set up. Repository `rdds` created at https://github.com/Larriba02/rdds (private).
- [x] Branch strategy configured: `main` (stable) + `dev` (working branch). Feature branches per stage follow the pattern `feature/step-N-name`.
- [x] Repository cloned locally to working machine.
- [x] `.gitignore` configured — excludes `.env`, `data/`, `checkpoints/`, `runs/`, `mlruns/`, `outputs/`, `logs/`, `__pycache__/`, `.vscode/`.
- [x] `.env.example` created with all 7 required environment variables (committed to Git).
- [x] `requirements.txt` created with 9 pinned dependencies.
- [x] `README.md` written with full setup instructions.
- [x] Step 0 files committed and pushed to `dev`.
- [x] VSCode configured with Python, Pylance, GitLens, and GitHub Pull Requests extensions.
- [x] Teammates (L, J) invited as collaborators and sent onboarding instructions.

### Dependencies (pinned)
```
ultralytics==8.3.0
pymongo==4.7.0
python-dotenv==1.0.1
mlflow==2.13.0
opencv-python==4.9.0.80
boto3==1.34.0
Pillow==10.3.0
numpy==1.26.4
scikit-learn==1.4.2
```

### Done criterion — verified ✅
Any team member can clone, run `pip install -r requirements.txt`, configure `.env`, and the environment is ready.

---

## 🔄 STEP 1 — MongoDB Setup

**Owner:** L  
**Status:** In progress

**Goal:** MongoDB Atlas cluster running with correct collections and schemas. All team members can connect.

### Tasks
- [ ] Create MongoDB Atlas free tier cluster.
- [ ] Create database `rdds` with three collections: `images_metadata`, `experiments`, `predictions`.
- [ ] Create indexes:
  - `images_metadata`: index on `country`, `split`, `image_id`
  - `experiments`: index on `is_production`, `status`, `model`
  - `predictions`: index on `image_id`, `model_version`
- [ ] Share connection URI with team. Store in `.env` as `MONGO_URI`.
- [ ] Write `src/db/connection.py` — single function `get_db()` that returns the database handle using `MONGO_URI` from `.env`.
- [ ] Write a smoke test: `python src/db/test_connection.py` — connects, inserts a dummy document, deletes it, prints OK.

### MongoDB Schemas

**images_metadata:**
```json
{
  "image_id": "md5 of relative filepath",
  "filepath": "Japan/train/00001.jpg",
  "country": "Japan",
  "source_device": "smartphone",
  "split": "train",
  "width": 600, "height": 600,
  "annotations": [{ "label": "D00", "xmin": 120, "ymin": 80, "xmax": 340, "ymax": 200 }]
}
```

**experiments:**
```json
{
  "run_id": "run_20260310_001",
  "model": "yolo11m",
  "model_version": "v1.0",
  "status": "promoted",
  "is_production": true,
  "sample_ratio": 1.0,
  "training_image_ids": ["..."],
  "dataset_countries": ["Japan", "Czech", "India", "US", "China", "Norway"],
  "hyperparams": { "epochs": 100, "batch": 32, "imgsz": 640, "seed": 42 },
  "metrics": { "mAP50": 0.84, "F1": 0.81, "precision": 0.83, "recall": 0.79 },
  "checkpoints": {
    "best_pt":   "https://f000.backblazeb2.com/rdds/.../best.pt",
    "last_pt":   "https://f000.backblazeb2.com/rdds/.../last.pt",
    "best_onnx": "https://f000.backblazeb2.com/rdds/.../best.onnx"
  },
  "timestamp": "2026-03-10T14:32:00Z"
}
```

**predictions:**
```json
{
  "pred_id": "...",
  "image_id": "...",
  "model_version": "v1.0",
  "timestamp": "...",
  "detections": [{ "label": "D40", "bbox": [230, 180, 410, 310], "confidence": 0.91 }]
}
```

**status values:** `running` | `completed` | `promoted` | `superseded`

### Done when
`python src/db/test_connection.py` prints OK on all three machines.

---

## ⏳ STEP 2 — Data Ingestion

**Owner:** M  
**Goal:** RDD2022 downloaded, validated, converted to YOLO format, split, uploaded to cloud, and metadata written to MongoDB. Done once by M. All other machines pull from cloud.

### Tasks
- [ ] `src/data/download.py` — download country ZIPs from Sekilab S3 to local disk.
  ```
  https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_{country}.zip
  ```
  Countries: Japan, India, Czech, Norway, United_States, China_MotorBike, China_Drone

- [ ] `src/data/validate.py` — parse every PascalVOC XML, discard bboxes where:
  - xmin < 0 or ymin < 0
  - xmax > image width or ymax > image height
  - area == 0 (degenerate)
  Log all discarded samples to `logs/discarded_annotations.txt`.

- [ ] `src/data/convert.py` — PascalVOC XML → YOLO `.txt` format.
  Class map: D00=0, D10=1, D20=2, D40=3. One `.txt` per image, same name.

- [ ] `src/data/analyse_distribution.py` — count instances per class per country. Print distribution table. Save to `logs/class_distribution.json`. **This output calibrates cls_weight in training — do not skip.**

- [ ] `src/data/split.py` — respect official RDD2022 train/test partitions. Within the official train split:
  - Reserve fixed 1,000 images per country for validation.
  - Remaining train images sampled at `SAMPLE_RATIO` stratified by country.
  - `image_id` = MD5 hash of relative filepath.

- [ ] `src/data/ingest.py` — write one document per image to MongoDB `images_metadata`. Skip if `image_id` already exists (idempotent).

- [ ] `src/data/upload_to_cloud.py` — upload processed dataset to cloud storage.

### Done when
- MongoDB `images_metadata` populated with all countries.
- `logs/class_distribution.json` exists and shows per-class counts.
- Any team member can pull the processed dataset.

---

## ⏳ STEP 3 — Phase 0 Training

**Owner:** M  
**Goal:** Full pipeline runs end-to-end on laptop. Real (weak) mAP number produced. MongoDB writes confirmed. MLflow logging confirmed.

### Configuration
```
Model:        YOLO11s
Dataset:      All 6 countries, SAMPLE_RATIO=0.10
Batch:        8
imgsz:        640
Epochs:       50
amp:          True (FP16)
seed:         42
cls_weight:   from class_distribution.json
```

### Tasks
- [ ] `src/training/train.py` — Ultralytics YOLO11 training script.
- [ ] `src/training/upload_checkpoint.py` — upload `best.pt`, `last.pt`, `best.onnx` to Backblaze B2.
- [ ] `src/training/promote.py` — compare new model mAP vs current `is_production` model.

### Done when
- Training completes without errors.
- MongoDB `experiments` has one document with real metrics.
- MLflow has one logged run.
- Backblaze has `best.pt` and `best.onnx` for this run.
- `is_production=True` on the YOLO11s run.

---

## ⏳ STEP 4 — Phase 1 Training (A100 Cluster)

**Owner:** M  
**Goal:** Full-performance training on complete dataset. Final mAP numbers.

### Configuration
```
Models:       YOLO11s (confirmed baseline) + YOLO11m (main)
Dataset:      All 6 countries, SAMPLE_RATIO=1.0
Batch:        32
imgsz:        640
Epochs:       100
amp:          True (FP16)
seed:         42
```

### Tasks
- [ ] Create `scripts/train_cluster.sh` — sbatch script.
- [ ] Run YOLO11s first (faster, confirms cluster setup works).
- [ ] Run YOLO11m after YOLO11s completes successfully.
- [ ] Promote best model via `promote.py`.

### Done when
- YOLO11m experiment document in MongoDB with `status: promoted`, `is_production: true`.
- mAP@0.5 in expected range 0.82–0.88.

---

## ⏳ STEP 5 — Evaluation

**Owner:** M  
**Goal:** Final quantitative and qualitative evaluation on official test split.

### Tasks
- [ ] `src/evaluation/evaluate.py` — mAP@0.5, F1, Precision, Recall per class and globally. IoU=0.5, confidence=0.5.
- [ ] `src/evaluation/qualitative.py` — 50 images per class (200 total). Save to `outputs/qualitative/`.
- [ ] Save all metrics to MongoDB `experiments` doc of the production model.

### Done when
- Metrics computed on 100% of official test split.
- Per-class breakdown available.
- 200 qualitative samples saved and inspected.

---

## ⏳ STEP 6 — Inference Module

**Owner:** M  
**Goal:** Script that takes an image or folder and produces annotated output + MongoDB write.

### Tasks
- [ ] `src/inference/predict.py` — loads `is_production=True` model, outputs annotated images, writes to `predictions` collection.
- [ ] `src/inference/extract_frames.py` — extract frames from video at 1fps.

### Done when
- Full video demo workflow works: `video → frames → predict → annotated output`.

---

## ⏳ STEP 7 — Retraining Pipeline

**Owner:** M  
**Goal:** `retrain()` function that fine-tunes from a checkpoint, evaluates, and promotes if better.

### Tasks
- [ ] `src/training/retrain.py` — validate, ingest, load checkpoint, fine-tune, evaluate, promote if better.
- [ ] Test with a small synthetic batch (10–20 images).
- [ ] Verify `is_production` flips correctly on promotion.

### Done when
- `retrain()` runs end-to-end without errors.
- Both promoted and non-promoted outcomes correctly logged in MongoDB.

---

## ⏳ STEP 8 — Optional: Web Demo

**Owner:** M (AI-assisted)  
**Condition:** Only if Steps 3–7 are complete and time allows.

### Tasks
- [ ] `src/api/main.py` — FastAPI: `POST /predict`, `GET /model`.
- [ ] Frontend: single page, upload image/video, display annotated result.

---

## Non-Negotiable Rules (apply to every step)

- `RANDOM_SEED=42` in all training runs. No exceptions.
- Test split is always 100%. Never sampled.
- Fixed validation set: always the same 1,000 images per country.
- `is_production` is set atomically. Never leave two documents with `is_production=True`.
- No Jupyter notebooks in cluster jobs. Python scripts only.
- `.env` is never committed to Git.
- Every training run writes to MongoDB before, during, and after training.
- Checkpoints are uploaded to Backblaze immediately after training.

---

## File Structure Reference

```
rdds/
├── src/
│   ├── db/
│   │   ├── connection.py
│   │   └── test_connection.py
│   ├── data/
│   │   ├── download.py
│   │   ├── validate.py
│   │   ├── convert.py
│   │   ├── analyse_distribution.py
│   │   ├── split.py
│   │   ├── ingest.py
│   │   └── upload_to_cloud.py
│   ├── training/
│   │   ├── train.py
│   │   ├── upload_checkpoint.py
│   │   ├── promote.py
│   │   └── retrain.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── qualitative.py
│   ├── inference/
│   │   ├── predict.py
│   │   └── extract_frames.py
│   └── api/               # optional
│       └── main.py
├── scripts/
│   └── train_cluster.sh
├── logs/
├── outputs/
├── .env.example
├── .gitignore
└── requirements.txt
```

---

*Paste this document together with RDDS_Claude_Memory.md at the start of a new conversation to resume development from any step.*
