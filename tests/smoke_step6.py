"""
Step 6 smoke test — inference (--dry-run) on a tiny test image.

Prerequisites:
  - Any best.pt locally in runs/train/*/weights/ (uses --model to bypass MongoDB)

Skips MongoDB write via --dry-run.

Run from repo root:
    python tests/smoke_step6.py
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

TINY_DATA = Path(__file__).parent / "data" / "tiny_rdd2022"
TEST_IMAGE = TINY_DATA / "Japan" / "test" / "images" / "90001.jpg"
RUNS_DIR = Path("runs") / "train"


def _find_local_checkpoint() -> Path | None:
    """Return the newest best.pt found under runs/train/, or None."""
    if not RUNS_DIR.exists():
        return None
    candidates = sorted(RUNS_DIR.glob("*/weights/best.pt"))
    return candidates[-1] if candidates else None


def main() -> bool:
    print("=== Step 6 smoke test — Inference ===\n")

    if not TEST_IMAGE.exists():
        print("FAIL: test image not found — check tiny_rdd2022 dataset.")
        return False

    checkpoint = _find_local_checkpoint()
    if checkpoint is None:
        print("SKIP: no local best.pt found in runs/train/ — run a training first.")
        return True

    print(f"Checkpoint: {checkpoint}")
    print(f"Source:     {TEST_IMAGE}\n")

    result = subprocess.run(
        [
            sys.executable, "-m", "src.inference.predict",
            "--source", str(TEST_IMAGE),
            "--model", str(checkpoint),
            "--dry-run",
            "--device", "cpu",
        ],
    )
    if result.returncode != 0:
        print("\nStep 6 smoke test FAILED")
        return False

    print("\nStep 6 smoke test PASSED")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
