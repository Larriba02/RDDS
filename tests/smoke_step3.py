"""
Step 3 / R1 smoke test — live per-epoch training monitoring.

Runs a real 1-epoch YOLO11s training on the tiny synthetic dataset (CPU) and
asserts the ``on_fit_epoch_end`` callback added in ``src/training/train.py``
wrote at least one record to ``experiments.progress`` in MongoDB. This is the
R1 done-criterion ("a 1-epoch run on tests/data/tiny_rdd2022 produces >=1
progress record in MongoDB").

ISOLATION / SAFETY
------------------
In normal use ``train.py`` reads ``logs/splits.json`` and the
``images_metadata`` collection, both of which describe the FULL RDD2022
dataset. To avoid clobbering that production state, this script:

  * backs up ``logs/splits.json`` and restores it on exit;
  * regenerates splits + ingests against ``tests/data/tiny_rdd2022``. The
    image_id is the MD5 of the *relative* path (see split.py), so the tiny
    ids (e.g. md5("Czech/train/images/00001.jpg")) can never collide with the
    real ones, and ingest only upserts — it never drops the collection;
  * deletes the tiny ``images_metadata`` docs and the smoke ``experiments``
    doc on exit.

``RDD_DATA_ROOT`` is set for the subprocesses only — ``.env`` is never touched.

Run from the repo root (writes to Atlas intentionally):
    python tests/smoke_step3.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from dotenv import load_dotenv

load_dotenv()

REPO = Path(__file__).parents[1]
TINY_DATA = REPO / "tests" / "data" / "tiny_rdd2022"
SPLITS_JSON = REPO / "logs" / "splits.json"
SPLITS_BAK = REPO / "logs" / "splits.json.smoke-bak"


def _tiny_uses_safe_names() -> bool:
    """The tiny fixture must use bare filenames (e.g. ``00001.jpg``), NOT the real
    ``<Country>_NNNN.jpg`` convention.

    ``image_id`` is the MD5 of the relative filepath (see split.py), so identical
    relative paths would produce identical ids. If the fixture ever adopted the
    real naming, ingest's upsert would overwrite production ``images_metadata``
    docs and the cleanup ``delete_many`` could remove real data. This precondition
    makes that safety explicit instead of relying on a fixture-naming coincidence.
    """
    for img in TINY_DATA.glob("*/*/images/*"):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        country = img.relative_to(TINY_DATA).parts[0]
        if img.stem.startswith(f"{country}_"):
            return False
    return True


def _run(cmd: list[str], env: dict[str, str]) -> int:
    """Run a subprocess from the repo root with the given environment."""
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(REPO), env=env).returncode


def _experiment_run_ids(db) -> set[str]:
    return {d["run_id"] for d in db["experiments"].find({}, {"run_id": 1, "_id": 0})}


def main() -> bool:
    print("=== Step 3 / R1 smoke test — live training monitoring ===\n")

    if not TINY_DATA.exists():
        print(f"SKIP: tiny dataset not found at {TINY_DATA}")
        return True

    if not _tiny_uses_safe_names():
        print(
            "SKIP: tiny fixture uses real-style '<Country>_*' filenames — image_ids "
            "could collide with production images_metadata. Aborting to protect prod data."
        )
        return True

    # --- MongoDB reachability ------------------------------------------------
    try:
        from src.db.connection import get_db
        db = get_db()
        db["experiments"].estimated_document_count()
    except Exception as exc:
        print(f"SKIP: MongoDB unreachable: {exc}")
        return True

    # --- Subprocess environment: point the pipeline at the tiny dataset ------
    env = os.environ.copy()
    env["RDD_DATA_ROOT"] = str(TINY_DATA)
    env["SAMPLE_RATIO"] = "1.0"

    tiny_ids: list[str] = []
    new_run_id: str | None = None
    splits_backed_up = False

    try:
        # --- Back up the real splits.json before split.py overwrites it ------
        if SPLITS_JSON.exists():
            SPLITS_BAK.write_bytes(SPLITS_JSON.read_bytes())
            splits_backed_up = True
            print(f"Backed up splits.json -> {SPLITS_BAK.name}")

        # --- Regenerate splits + ingest for the tiny dataset -----------------
        if _run([sys.executable, "-m", "src.data.split"], env) != 0:
            print("\nFAILED: split.py errored on the tiny dataset")
            return False

        tiny_ids = list(json.loads(SPLITS_JSON.read_text(encoding="utf-8")).keys())
        print(f"  Tiny splits.json: {len(tiny_ids)} image_ids")

        if _run([sys.executable, "-m", "src.data.ingest"], env) != 0:
            print("\nFAILED: ingest.py errored on the tiny dataset")
            return False

        # --- 1-epoch CPU training run ----------------------------------------
        before = _experiment_run_ids(db)
        rc = _run(
            [
                sys.executable, "-m", "src.training.train",
                "--model", "yolo11s",
                "--sample-ratio", "1.0",
                "--epochs", "1",
                "--batch", "2",
                "--patience", "1",
                "--device", "cpu",
                "--skip-upload",
                "--skip-promote",
            ],
            env,
        )
        new_ids = _experiment_run_ids(db) - before
        new_run_id = next(iter(new_ids), None)

        if rc != 0:
            print(f"\nFAILED: train.py exited {rc}")
            return False
        if new_run_id is None:
            print("\nFAILED: no new experiments document was created")
            return False

        # --- Assert the callback wrote per-epoch progress to MongoDB ---------
        doc = db["experiments"].find_one(
            {"run_id": new_run_id},
            {"_id": 0, "progress": 1, "progress_current_epoch": 1,
             "progress_total_epochs": 1, "progress_updated_at": 1},
        )
        progress = (doc or {}).get("progress") or []
        print(f"\nrun_id: {new_run_id}")
        print(f"  progress records:       {len(progress)}")
        print(f"  progress_current_epoch: {doc.get('progress_current_epoch')}")
        print(f"  progress_updated_at:    {doc.get('progress_updated_at')}")

        if not progress:
            print("\nFAILED: experiments.progress is empty — callback did not fire")
            return False

        rec = progress[-1]
        keys = ", ".join(sorted(rec.keys()))
        print(f"  last record keys: {keys}")
        print(f"  last record F1={rec.get('F1')} mAP50={rec.get('metrics/mAP50(B)')}")

        print("\nStep 3 / R1 smoke test PASSED "
              f"({len(progress)} progress record(s) in MongoDB)")
        return True

    finally:
        # --- Restore production state ---------------------------------------
        print("\n--- cleanup ---")
        if tiny_ids:
            res = db["images_metadata"].delete_many({"image_id": {"$in": tiny_ids}})
            print(f"  deleted {res.deleted_count} tiny images_metadata docs")
        if new_run_id:
            db["experiments"].delete_one({"run_id": new_run_id})
            print(f"  deleted smoke experiments doc {new_run_id}")
        if splits_backed_up:
            SPLITS_JSON.write_bytes(SPLITS_BAK.read_bytes())
            SPLITS_BAK.unlink()
            print("  restored real splits.json")


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
