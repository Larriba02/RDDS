"""
Step 7 smoke test — retraining pipeline (1 epoch, skip upload + promote).

Prerequisites:
  - RDD_DATA_ROOT set in .env
  - MongoDB accessible (MONGO_URI in .env)
  - is_production=True experiment with a reachable best.pt (local or B2)
  - logs/splits.json present (used for the val split during fine-tuning)

This runs a real 1-epoch fine-tune on tiny_rdd2022. Expect ~1–3 min on GPU.

Run from repo root:
    python tests/smoke_step7.py
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from dotenv import load_dotenv

load_dotenv()

TINY_DATA = Path(__file__).parent / "data" / "tiny_rdd2022"


def _check_prereqs() -> str | None:
    if not os.getenv("RDD_DATA_ROOT"):
        return "RDD_DATA_ROOT not set in .env"

    splits = Path("logs/splits.json")
    if not splits.exists():
        return "logs/splits.json not found — run 'python -m src.data.split' first"

    try:
        from src.db.connection import get_db
        db = get_db()
        doc = db["experiments"].find_one({"is_production": True})
        if doc is None:
            return "no is_production=True experiment in MongoDB — train a model first"
    except Exception as exc:
        return f"MongoDB unreachable: {exc}"

    return None


def main() -> bool:
    print("=== Step 7 smoke test — Retraining Pipeline ===\n")

    skip_reason = _check_prereqs()
    if skip_reason:
        print(f"SKIP: {skip_reason}")
        return True

    print(f"New images: {TINY_DATA}")
    print("Running 1-epoch retrain (--skip-upload --skip-promote)...\n")

    result = subprocess.run(
        [
            sys.executable, "-m", "src.training.retrain",
            "--new-images", str(TINY_DATA),
            "--epochs", "1",
            "--batch", "2",
            "--patience", "1",
            "--skip-upload",
            "--skip-promote",
        ],
    )
    if result.returncode != 0:
        print("\nStep 7 smoke test FAILED")
        return False

    print("\nStep 7 smoke test PASSED")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
