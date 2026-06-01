"""
Full smoke test suite — runs all step smoke tests in sequence.

Step 1 (MongoDB) is covered by the existing /smoke-test slash command
(src/db/setup_atlas.py + src/db/test_connection.py). This runner calls
it here as well so everything is in one place.

Step 3 (training monitor) runs a real 1-epoch CPU train on the tiny dataset
via smoke_step3.py and asserts the per-epoch progress callback (R1) wrote to
MongoDB; it backs up / restores splits.json and cleans up the tiny docs and
the smoke experiment afterwards. Step 4 (cluster) is exercised manually via
'sbatch scripts/train_cluster.sh ... SMOKE_TEST=1'.

Usage:
    python tests/smoke_all.py            # run all steps
    python tests/smoke_all.py --steps 1 2 6   # run specific steps
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parents[1]
TESTS = REPO / "tests"

STEP_SCRIPTS: dict[int, str | list[str]] = {
    1: [
        [sys.executable, "-m", "src.db.setup_atlas"],
        [sys.executable, "-m", "src.db.test_connection"],
    ],
    2: [sys.executable, str(TESTS / "smoke_step2.py")],
    3: [sys.executable, str(TESTS / "smoke_step3.py")],
    5: [sys.executable, str(TESTS / "smoke_step5.py")],
    6: [sys.executable, str(TESTS / "smoke_step6.py")],
    7: [sys.executable, str(TESTS / "smoke_step7.py")],
    8: [sys.executable, str(TESTS / "smoke_step8.py")],
}

STEP_LABELS = {
    1: "MongoDB Setup",
    2: "Data Pipeline",
    3: "Training Monitor (R1)",
    5: "Evaluation",
    6: "Inference",
    7: "Retraining",
    8: "Web Demo",
}


def run_step(step: int) -> bool:
    label = STEP_LABELS[step]
    script = STEP_SCRIPTS[step]
    print(f"\n{'='*60}")
    print(f"  Step {step} — {label}")
    print(f"{'='*60}")

    if isinstance(script[0], list):
        # Multiple commands (Step 1)
        for cmd in script:
            result = subprocess.run(cmd, cwd=str(REPO))
            if result.returncode != 0:
                print(f"\n[Step {step}] FAILED (exit {result.returncode})")
                return False
    else:
        result = subprocess.run(script, cwd=str(REPO))
        if result.returncode != 0:
            print(f"\n[Step {step}] FAILED (exit {result.returncode})")
            return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="RDDS full smoke test suite.")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=sorted(STEP_SCRIPTS.keys()),
        help="Steps to run (default: all). Example: --steps 1 2 6",
    )
    args = parser.parse_args()

    steps = [s for s in args.steps if s in STEP_SCRIPTS]
    unknown = [s for s in args.steps if s not in STEP_SCRIPTS]
    if unknown:
        print(f"Warning: no smoke test for step(s) {unknown} — skipped.")

    results: dict[int, bool] = {}
    for step in steps:
        results[step] = run_step(step)

    print(f"\n{'='*60}")
    print("  SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for step in steps:
        ok = results[step]
        status = "PASS" if ok else "FAIL"
        label = STEP_LABELS[step]
        print(f"  Step {step:2d}  {label:<20} {status}")
        if not ok:
            all_ok = False
    print(f"{'='*60}")
    print(f"  Overall: {'ALL PASSED' if all_ok else 'SOME FAILURES'}")
    print(f"{'='*60}\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
