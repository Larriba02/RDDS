"""
src/db/test_connection.py
--------------------------
Smoke test for the MongoDB Atlas connection.

What it does:
  1. Connects to Atlas via MONGO_URI.
  2. Inserts a dummy document into each of the three RDDS collections.
  3. Verifies the insert (reads the document back).
  4. Deletes the dummy documents (leaves no trace).
  5. Prints a per-collection OK / FAIL and an overall result.

Run:
    python src/db/test_connection.py

Expected output (all passing):
    [images_metadata]  OK
    [experiments]      OK
    [predictions]      OK
    ----------------------------------------
    All checks passed — MongoDB connection is healthy.
"""

import sys
import uuid
from datetime import datetime, timezone

from src.db.connection import get_db

# ── Dummy documents (minimal valid shapes) ────────────────────────────────────

_DUMMY_IMAGE = {
    "image_id": f"__smoke__{uuid.uuid4().hex}",
    "filepath": "__smoke__/test/00000.jpg",
    "country": "__smoke__",
    "source_device": "smoke_test",
    "split": "train",
    "width": 1,
    "height": 1,
    "annotations": [],
}

_DUMMY_EXPERIMENT = {
    "run_id": f"__smoke__{uuid.uuid4().hex}",
    "model": "smoke",
    "model_version": "v0.0",
    "status": "running",
    "is_production": False,
    "sample_ratio": 0.0,
    "training_image_ids": [],
    "dataset_countries": [],
    "hyperparams": {"epochs": 0, "batch": 0, "imgsz": 0, "seed": 42},
    "metrics": {},
    "checkpoints": {},
    "timestamp": datetime.now(timezone.utc).isoformat(),
}

_DUMMY_PREDICTION = {
    "pred_id": f"__smoke__{uuid.uuid4().hex}",
    "image_id": "__smoke__",
    "model_version": "v0.0",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "detections": [],
}


# ── Per-collection test ───────────────────────────────────────────────────────

def _test_collection(db, collection_name: str, doc: dict, key_field: str) -> bool:
    """Insert doc, read it back, delete it. Return True on success."""
    col = db[collection_name]
    key_value = doc[key_field]
    try:
        col.insert_one(doc)
        found = col.find_one({key_field: key_value})
        if found is None:
            raise RuntimeError("Document not found after insert.")
        col.delete_one({key_field: key_value})
        print(f"[{collection_name}]".ljust(22) + "OK")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[{collection_name}]".ljust(22) + f"FAIL — {exc}")
        # Best-effort cleanup
        try:
            col.delete_one({key_field: key_value})
        except Exception:
            pass
        return False


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    try:
        db = get_db()
    except EnvironmentError as exc:
        print(f"[connection] FAIL — {exc}")
        sys.exit(1)

    results = [
        _test_collection(db, "images_metadata", _DUMMY_IMAGE,      "image_id"),
        _test_collection(db, "experiments",     _DUMMY_EXPERIMENT, "run_id"),
        _test_collection(db, "predictions",     _DUMMY_PREDICTION, "pred_id"),
    ]

    print("-" * 40)
    if all(results):
        print("All checks passed — MongoDB connection is healthy.")
    else:
        print("One or more checks FAILED. See output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
