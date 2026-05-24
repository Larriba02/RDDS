"""
Step 5 smoke test — evaluation (--dry-run) on production model.

Prerequisites:
  - MongoDB accessible (MONGO_URI in .env)
  - is_production=True experiment exists in MongoDB
  - logs/splits.json present (run src.data.split first)
  - best.pt locally in runs/train/ OR B2 URL stored in MongoDB

Run from repo root:
    python tests/smoke_step5.py
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


def _check_prereqs() -> str | None:
    """Return a skip reason string, or None if all prereqs are met."""
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
    print("=== Step 5 smoke test — Evaluation ===\n")

    skip_reason = _check_prereqs()
    if skip_reason:
        print(f"SKIP: {skip_reason}")
        return True

    print("Prerequisites OK — running evaluate --dry-run ...\n")
    result = subprocess.run(
        [sys.executable, "-m", "src.evaluation.evaluate", "--dry-run"],
    )
    if result.returncode != 0:
        print("\nStep 5 smoke test FAILED")
        return False

    print("\nStep 5 smoke test PASSED")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
