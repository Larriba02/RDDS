"""
Step 2 smoke test — data pipeline (validate + convert) on tiny_rdd2022.

Prerequisites: none (standalone, no MongoDB, no trained model).

Run from repo root:
    python tests/smoke_step2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.validate import validate_dataset
from src.data.convert import convert_dataset

TINY_DATA = Path(__file__).parent / "data" / "tiny_rdd2022"


def main() -> bool:
    print("=== Step 2 smoke test — Data Pipeline ===")
    print(f"Dataset: {TINY_DATA}\n")

    if not TINY_DATA.exists():
        print("FAIL: tiny_rdd2022 not found at expected path.")
        return False

    print("[1/2] validate_dataset...")
    annotations = validate_dataset(TINY_DATA)
    if not annotations:
        print("FAIL: validate_dataset returned no images.")
        return False
    print(f"  OK — {len(annotations)} images validated\n")

    print("[2/2] convert_dataset...")
    written = convert_dataset(TINY_DATA, force=True)
    print(f"  OK — {written} label files written\n")

    print("Step 2 smoke test PASSED")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
