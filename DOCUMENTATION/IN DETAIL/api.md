# Web Demo — Detailed Guide
**RDDS · Group 3 · UFV**  
*Version 1.0 — May 2026*

---

## Purpose and scope

`src/api/` is a minimal FastAPI application that puts a browser interface in
front of the inference pipeline. Its purpose is presentation — showing the
production model detecting road damage in a live demo without requiring anyone
in the audience to understand the command line.

It is not a technical deliverable. It was built after Steps 5–7 were complete
and is explicitly listed as optional in the project scope. Nothing in the
core pipeline depends on it. If you need to evaluate the model, use
`src/evaluation/evaluate.py`. If you need batch inference, use
`src/inference/predict.py`. The API is for one-off interactive use only.

---

## Module layout

```
src/api/
├── __init__.py        # empty package marker
├── main.py            # FastAPI application
└── static/
    └── index.html     # single-page frontend
```

---

## Routes

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves `src/api/static/index.html`. This is the entire frontend — one self-contained HTML file. |
| `GET` | `/model` | Returns JSON metadata for the current production model: `run_id`, `model`, `F1`, `mAP50`, `sample_ratio`, `timestamp`. |
| `POST` | `/predict` | Accepts a multipart image upload (jpg or png). Returns JSON detections and a base64-encoded annotated image. Also writes one document to the MongoDB `predictions` collection. |

### GET /model — why it exists

The model-info endpoint exists so the frontend can show which model version is
currently answering requests. This matters in a demo context: an audience
member asking "what model is this?" should get a concrete answer (run ID, F1
score, training date) rather than "the current one". The data is read directly
from MongoDB — the same `is_production=True` document that every other module
uses as its authority.

### POST /predict — what it returns

```json
{
  "detections": [
    { "label": "D40", "bbox": [230, 180, 410, 310], "confidence": 0.91 },
    { "label": "D00", "bbox": [10, 50, 300, 120], "confidence": 0.76 }
  ],
  "image_b64": "<base64-encoded JPEG string>"
}
```

`image_b64` is the annotated image encoded as a base64 JPEG string, ready to
be dropped into an `<img src="data:image/jpeg;base64,...">` tag. The frontend
does this automatically.

`detections` follows the same schema as the `predictions` collection documents
(label, bbox in pixel coordinates, confidence). An empty list means the model
found no damage above the confidence threshold.

---

## Model caching — `_model_cache`

Loading a YOLO model from disk (or downloading it from B2) takes 1–5 seconds.
Doing this on every request would make the API unusable. `main.py` therefore
implements a simple module-level cache:

```python
_model_cache: dict = {}   # keyed by run_id
```

On the first request after startup, the production model is loaded from the
checkpoint resolved by MongoDB, stored in `_model_cache`, and reused for all
subsequent requests. The cache is never invalidated during the process lifetime
— if the production model is promoted to a new version while the server is
running, the server must be restarted to pick up the new model.

This is intentional: the API is a demo tool, not a long-running service. A
restart is acceptable and is much simpler than implementing a live cache
invalidation mechanism.

Model resolution follows the same four-step lookup order used by
`src/inference/predict.py`:

1. `runs/train/<run_id>/weights/best.pt` — direct local match on the
   production run_id.
2. `runs/train/<prefix>*/weights/best.pt` — fuzzy prefix match (handles
   Ultralytics incremental directory naming).
3. Download `best.pt` from the Backblaze B2 URL stored in
   `experiments.checkpoints.best_pt` in MongoDB.

---

## MongoDB write in POST /predict

After running inference, `POST /predict` writes one document to the MongoDB
`predictions` collection — the same collection used by `src/inference/predict.py`.
The schema is identical:

```json
{
  "pred_id":       "<uuid4>",
  "image_id":      "<md5 of a uuid string representing the upload>",
  "model_version": "<model field from experiments doc>",
  "run_id":        "<production run_id>",
  "timestamp":     "<ISO 8601 UTC>",
  "source_path":   "api_upload",
  "detections":    [ ... ]
}
```

The write is **non-fatal**: if MongoDB is unavailable or the insert fails for
any reason, the API logs a warning and still returns the inference result to
the caller. The demo keeps working even if the database is temporarily
unreachable.

This is the correct trade-off for a presentation tool. For production data
pipelines where every prediction must be persisted, use `predict.py` directly
— it is designed for reliability, not interactivity.

---

## Frontend — src/api/static/index.html

The frontend is a single self-contained HTML file with no external JavaScript
dependencies. It provides:

- A drag-and-drop image upload area (also accepts click-to-browse).
- A model-info card showing the run ID, model name, F1 score, mAP50,
  sample ratio, and training timestamp, loaded from `GET /model` on page
  load.
- An annotated result image displayed after a successful `POST /predict`.
- A detections table listing label, bounding box coordinates, and confidence
  score for each detected damage instance.
- Per-class colour coding in the detections table consistent with the colour
  scheme used by `predict.py` (D00 = blue, D10 = green, D20 = yellow,
  D40 = red).

The entire interaction is a single fetch call to `POST /predict`. No page
reload, no state management library, no build step. The frontend is readable
as plain HTML.

---

## How to run

```bash
# Activate the virtual environment first
.venv\Scripts\activate             # Windows
source .venv/bin/activate          # macOS / Linux

# Start the development server (auto-reload on file changes)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
# Then open http://localhost:8000
```

`--reload` is appropriate for local development and demos. Do not use it in
any environment where multiple concurrent requests are expected — it disables
the worker pool.

`--host 0.0.0.0` makes the server reachable from other machines on the same
network (useful for a live demo on a projector laptop). Use `--host 127.0.0.1`
if you only need local access.

### Prerequisites

`requirements.txt` must include:

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
```

`python-multipart` is required for FastAPI's file upload handling. If it is
missing, `POST /predict` will return a 422 error on any upload.

---

## Dependencies

| Package | Why |
|---------|-----|
| `fastapi` | HTTP framework — route definitions, request parsing, response models. |
| `uvicorn[standard]` | ASGI server — runs the FastAPI application. |
| `python-multipart` | Required by FastAPI for parsing `multipart/form-data` file uploads. |
| `httpx` | HTTP client used by the Step 8 smoke test (`tests/smoke_step8.py`) to exercise the API end-to-end without a browser. |
| `ultralytics` | YOLO model loading and inference — already in `requirements.txt`. |
| `opencv-python` | Drawing bounding boxes on the annotated image — already in `requirements.txt`. |
| `src.db.connection.get_db()` | MongoDB handle — shared with all other modules. |

---

## Scope note

The web demo is a **presentation asset**. It exists to answer the question
"can I see it working in a browser?" with a yes. It is not part of the
evaluation pipeline, does not affect any metrics, and is not referenced in the
CRDDC2022 protocol. Steps 1–7 must be complete before this is touched.
