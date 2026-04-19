"""
src/db/setup_atlas.py
---------------------
One-time setup script: creates the three RDDS collections and their indexes
in MongoDB Atlas. Safe to re-run — existing indexes are left untouched.

Run once, by any team member with a valid MONGO_URI:
    python -m src.db.setup_atlas
"""

from pymongo import ASCENDING
from src.db.connection import get_db


def setup_collections_and_indexes() -> None:
    db = get_db()

    # ------------------------------------------------------------------
    # images_metadata
    # ------------------------------------------------------------------
    col_images = db["images_metadata"]

    col_images.create_index([("image_id", ASCENDING)], unique=True, name="idx_image_id")
    col_images.create_index([("country", ASCENDING)], name="idx_country")
    col_images.create_index([("split", ASCENDING)], name="idx_split")
    # Compound: common query pattern (filter by split + country)
    col_images.create_index(
        [("split", ASCENDING), ("country", ASCENDING)],
        name="idx_split_country",
    )

    print("[images_metadata] indexes OK")

    # ------------------------------------------------------------------
    # experiments
    # ------------------------------------------------------------------
    col_exp = db["experiments"]

    col_exp.create_index([("run_id", ASCENDING)], unique=True, name="idx_run_id")
    col_exp.create_index([("is_production", ASCENDING)], name="idx_is_production")
    col_exp.create_index([("status", ASCENDING)], name="idx_status")
    col_exp.create_index([("model", ASCENDING)], name="idx_model")

    print("[experiments] indexes OK")

    # ------------------------------------------------------------------
    # predictions
    # ------------------------------------------------------------------
    col_pred = db["predictions"]

    col_pred.create_index([("pred_id", ASCENDING)], unique=True, name="idx_pred_id")
    col_pred.create_index([("image_id", ASCENDING)], name="idx_image_id")
    col_pred.create_index([("model_version", ASCENDING)], name="idx_model_version")
    # Compound: fetch predictions for a given image + model version
    col_pred.create_index(
        [("image_id", ASCENDING), ("model_version", ASCENDING)],
        name="idx_image_model",
    )

    print("[predictions] indexes OK")


if __name__ == "__main__":
    print("Setting up RDDS collections and indexes in MongoDB Atlas...")
    setup_collections_and_indexes()
    print("\nDone. All collections and indexes are in place.")
