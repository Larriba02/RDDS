# RDDS — Inference Module
**Road Damage Detection System · Group 3 · UFV**
*Version 1.0 — May 2026*

---

## Overview

The inference module provides two entry points:

| Module | Entry point | Purpose |
|--------|-------------|---------|
| `src/inference/extract_frames.py` | `python -m src.inference.extract_frames` | Extract frames from a video at 1 fps |
| `src/inference/predict.py` | `python -m src.inference.predict` | Run YOLO inference on images, save annotated output, write to MongoDB |

The intended workflow for video input is:

```
video file
    ↓  extract_frames.py (1 fps)
frames directory
    ↓  predict.py
annotated images  +  MongoDB predictions collection
```

---

## extract_frames.py

### What it does

Opens a video file with OpenCV and extracts one frame per second of content.
Frames are saved as `frame_NNNNNN.jpg` (zero-padded, 6 digits) in the output
directory. Nothing is written to MongoDB — frames are inputs to the prediction
pipeline, not predictions.

### Why 1 fps?

Road damage is static — a pothole does not move between video frames. At a
typical dashcam speed of 40–60 km/h, successive 1-second frames are
approximately 11–17 m apart, which is more than enough resolution to detect
all surface damage without processing redundant near-duplicate frames.

### CLI

```bash
python -m src.inference.extract_frames \
    --video  path/to/video.mp4 \
    --output-dir  outputs/frames/   # default
```

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | required | Path to video file (mp4, avi, mov, mkv, etc.) |
| `--output-dir` | `outputs/frames/` | Directory for extracted frames |

### Output

- `outputs/frames/frame_000001.jpg`, `frame_000002.jpg`, ...
- Printed summary: source FPS, duration, frame interval, total extracted, elapsed time.

---

## predict.py

### What it does

1. Resolves the production model checkpoint (see resolution order below).
2. Loads the model with Ultralytics YOLO.
3. Collects all image files from `--source` (single file or directory).
4. For each image:
   - Runs YOLO predict at `conf=0.5`, `iou=0.5`.
   - Draws bounding boxes on the image using OpenCV (`cv2.rectangle` + `cv2.putText`).
   - Saves the annotated image to `--output-dir`.
   - Writes one document to the MongoDB `predictions` collection (idempotent).
5. Prints a summary of total images, total detections, and elapsed time.

### Checkpoint resolution order

1. `--model <path>` flag — local `.pt` file. Bypasses MongoDB lookup entirely.
   Useful before the A100 cluster run when `best.pt` exists under `runs/` locally.
2. `runs/train/<run_id>/weights/best.pt` — direct local match.
3. `runs/train/<prefix>*/weights/best.pt` — fuzzy match (dir name starts with `run_id`).
4. Download `best.pt` from the Backblaze B2 URL stored in `experiments.checkpoints.best_pt`.

### Bounding box colours (BGR)

| Class | Damage type | Colour |
|-------|-------------|--------|
| D00 | Longitudinal crack | Blue `(255, 0, 0)` |
| D10 | Transverse crack | Green `(0, 255, 0)` |
| D20 | Alligator crack | Yellow `(0, 255, 255)` |
| D40 | Pothole | Red `(0, 0, 255)` |

### MongoDB write — predictions collection

One document per image, schema:

```json
{
  "pred_id":      "<uuid4>",
  "image_id":     "<md5 of source path string>",
  "model_version": "v1.0",
  "run_id":       "run_20260504_202658_yolo11s",
  "timestamp":    "2026-05-07T12:00:00+00:00",
  "source_path":  "/absolute/path/to/image.jpg",
  "detections": [
    { "label": "D40", "bbox": [230, 180, 410, 310], "confidence": 0.91 }
  ]
}
```

**Idempotency:** before inserting, the script checks whether a document with the
same `image_id` + `model_version` already exists. If it does, the insert is
skipped. The annotated image is still saved to disk in either case.

`image_id` is the MD5 hex digest of the source path string (encoded UTF-8),
consistent with the convention used in `src/data/split.py` for `images_metadata`.
Note that this is the absolute path as seen on the machine running inference,
so `image_id` in `predictions` is not the same as `image_id` in `images_metadata`
(which uses the relative filepath). The index `(image_id, model_version)` on the
`predictions` collection exists for the idempotency check.

### CLI

```bash
# Production model from MongoDB (B2 download if needed):
python -m src.inference.predict \
    --source path/to/image.jpg

# Directory of images:
python -m src.inference.predict \
    --source outputs/frames/ \
    --output-dir outputs/predictions/

# Local model override:
python -m src.inference.predict \
    --source path/to/image.jpg \
    --model runs/train/run_20260504_202658_yolo11s/weights/best.pt

# Full video workflow:
python -m src.inference.extract_frames \
    --video path/to/video.mp4 \
    --output-dir outputs/frames/
python -m src.inference.predict \
    --source outputs/frames/ \
    --output-dir outputs/predictions/

# Dry run — annotated images saved, no MongoDB write:
python -m src.inference.predict \
    --source path/to/image.jpg \
    --dry-run
```

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | required | Single image path or folder of images |
| `--output-dir` | `outputs/predictions/` | Directory for annotated output images |
| `--model` | `None` | Local `.pt` path; bypasses MongoDB lookup |
| `--conf` | `0.5` | Confidence threshold |
| `--iou` | `0.5` | IoU threshold for NMS |
| `--device` | `"0"` | CUDA device (`"0"`) or `"cpu"` |
| `--dry-run` | `False` | Skip MongoDB write |

---

## Why not pass video directly to predict.py?

Video files are explicitly rejected by `predict.py` with a clear error message
pointing the user to `extract_frames.py`. The two-step approach is deliberate:

- It makes the frame extraction rate visible and adjustable.
- The extracted frames directory can be inspected before prediction (useful for
  debugging poor video quality).
- `predict.py` only needs to handle images, keeping its logic simple and testable
  with synthetic data.

---

## Dependencies

Both modules require:
- `opencv-python` (`cv2`) — already in `requirements.txt`.
- `ultralytics` — already in `requirements.txt`.
- `src.db.connection.get_db()` — MongoDB handle from connection.py.
