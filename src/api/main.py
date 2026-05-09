"""
src/api/main.py
---------------
FastAPI web demo — POST /predict (image upload) and GET /model.

Run with:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in the browser.
"""

from __future__ import annotations

import base64
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

from src.db.connection import get_db
from src.inference.predict import (
    CLASS_NAMES,
    _draw_detections,
    _get_production_experiment,
    _resolve_checkpoint,
)

load_dotenv()

app = FastAPI(title="RDDS Web Demo", version="1.0")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB hard cap for image uploads

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Model cache — loaded once per process on first request
# ---------------------------------------------------------------------------

_model_cache: dict[str, YOLO] = {}
_exp_cache: dict[str, Any] = {}


def _get_loaded_model() -> tuple[YOLO, dict[str, Any]]:
    exp = _get_production_experiment()
    run_id = exp["run_id"]
    if run_id not in _model_cache:
        checkpoint = _resolve_checkpoint(exp)
        _model_cache[run_id] = YOLO(str(checkpoint))
        _exp_cache[run_id] = exp
    return _model_cache[run_id], _exp_cache[run_id]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=(STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/model")
async def get_model_info() -> dict[str, Any]:
    """Return metadata for the current production model."""
    try:
        exp = _get_production_experiment()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    metrics = exp.get("metrics", {})
    return {
        "run_id": exp.get("run_id"),
        "model": exp.get("model"),
        "model_version": exp.get("model_version"),
        "F1_overall": metrics.get("F1") or metrics.get("evaluation_val", {}).get("F1_overall"),
        "mAP50": metrics.get("mAP50"),
        "sample_ratio": exp.get("sample_ratio"),
        "timestamp": exp.get("timestamp"),
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept an image upload and return detections + annotated image (base64)."""
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 20 MB).")
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image — unsupported format.")

    try:
        model, exp = _get_loaded_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    results = model.predict(
        source=img,
        imgsz=640,
        conf=0.5,
        iou=0.5,
        verbose=False,
        save=False,
    )

    detections: list[dict[str, Any]] = []
    if results:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_idx = int(box.cls[0])
                cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
                detections.append({
                    "label": cls_name,
                    "bbox": [round(v) for v in box.xyxy[0].tolist()],
                    "confidence": round(float(box.conf[0]), 4),
                })

    annotated = _draw_detections(img.copy(), detections)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buf).decode("utf-8")

    # Write to MongoDB predictions (non-fatal if it fails)
    run_id = exp.get("run_id", "unknown")
    model_version = exp.get("model_version", "unknown")
    try:
        db = get_db()
        db["predictions"].insert_one({
            "pred_id": str(uuid.uuid4()),
            "image_id": f"api_{uuid.uuid4().hex[:8]}",
            "model_version": model_version,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_path": f"api_upload:{file.filename}",
            "detections": detections,
        })
    except Exception:
        pass

    return {
        "detections": detections,
        "annotated_image": img_b64,
        "model_version": model_version,
        "run_id": run_id,
    }
