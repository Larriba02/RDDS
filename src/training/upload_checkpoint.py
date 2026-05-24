"""
src/training/upload_checkpoint.py
----------------------------------
Upload training checkpoints (best.pt, last.pt, best.onnx) and the
per-epoch training curves (results.csv) to Backblaze B2.

The files are uploaded under the prefix:
    checkpoints/<run_id>/best.pt
    checkpoints/<run_id>/last.pt
    checkpoints/<run_id>/best.onnx
    checkpoints/<run_id>/results.csv

The returned dict maps filename to the public download URL, ready to be
stored in the ``experiments`` MongoDB document under ``checkpoints``.

Environment variables required (set in .env):
    BACKBLAZE_KEY_ID     — Application key ID
    BACKBLAZE_APP_KEY    — Application key secret
    BACKBLAZE_BUCKET     — Bucket name
    BACKBLAZE_ENDPOINT   — S3-compatible endpoint URL (optional, defaults to
                           us-west-004 region)

Usage (programmatic — called by train.py after training finishes):
    from src.training.upload_checkpoint import upload_checkpoints

    urls = upload_checkpoints(run_id="run_20260310_001", run_dir=Path("runs/train/exp"))

Usage (standalone — for manual re-upload):
    python -m src.training.upload_checkpoint \\
        --run-id run_20260310_001 \\
        --run-dir runs/train/exp
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

# Files to upload from the Ultralytics run directory.
# ONNX is under weights/ alongside the .pt files for Ultralytics >= 8.x.
_CHECKPOINT_FILES = ("best.pt", "last.pt", "best.onnx")


def _b2_client():
    """Create a boto3 S3 client configured for Backblaze B2.

    Returns:
        boto3 S3 client.

    Raises:
        EnvironmentError: If required B2 credentials are missing.
    """
    key_id = os.getenv("BACKBLAZE_KEY_ID")
    app_key = os.getenv("BACKBLAZE_APP_KEY")
    if not key_id or not app_key:
        raise EnvironmentError(
            "BACKBLAZE_KEY_ID and BACKBLAZE_APP_KEY must be set in .env."
        )
    endpoint = os.getenv(
        "BACKBLAZE_ENDPOINT", "https://s3.us-west-004.backblazeb2.com"
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key,
        config=Config(
            signature_version="s3v4",
            connect_timeout=30,
            read_timeout=300,
        ),
    )


def _bucket_name() -> str:
    """Return the B2 bucket name from the environment.

    Raises:
        EnvironmentError: If BACKBLAZE_BUCKET is not set.
    """
    bucket = os.getenv("BACKBLAZE_BUCKET")
    if not bucket:
        raise EnvironmentError("BACKBLAZE_BUCKET is not set in .env.")
    return bucket


def _public_url(endpoint: str, bucket: str, key: str) -> str:
    """Build the public download URL for a B2 S3-compatible object.

    Args:
        endpoint: S3-compatible endpoint URL (e.g. https://s3.us-west-004.backblazeb2.com).
        bucket: Bucket name.
        key: Object key within the bucket.

    Returns:
        Public download URL string.
    """
    # Strip trailing slash from endpoint for clean URL construction.
    return f"{endpoint.rstrip('/')}/{bucket}/{key}"


def upload_checkpoints(
    run_id: str,
    run_dir: Path,
) -> dict[str, str]:
    """Upload best.pt, last.pt, and best.onnx to Backblaze B2.

    Files are looked up in ``run_dir/weights/``.  Missing files are skipped
    with a warning so that a run that never produced best.onnx (e.g. export
    failed) can still upload the .pt files.

    Args:
        run_id: Unique training run identifier (e.g. ``run_20260310_001``).
        run_dir: Path to the Ultralytics output directory for this run
            (the directory that contains ``weights/``).

    Returns:
        Dict mapping shorthand key to public B2 URL:
        ``{"best_pt": "...", "last_pt": "...", "best_onnx": "..."}``.
        A key is omitted if the corresponding file was not found.

    Raises:
        EnvironmentError: If B2 credentials or bucket name are missing.
        RuntimeError: If one or more uploads fail.
    """
    client = _b2_client()
    bucket = _bucket_name()
    endpoint = os.getenv("BACKBLAZE_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")

    weights_dir = run_dir / "weights"
    urls: dict[str, str] = {}
    failures: list[str] = []

    key_map = {
        "best.pt": "best_pt",
        "last.pt": "last_pt",
        "best.onnx": "best_onnx",
    }

    # (local_path, b2_filename, content_type)
    targets: list[tuple[Path, str, str]] = [
        (weights_dir / fn, fn, "application/octet-stream") for fn in _CHECKPOINT_FILES
    ]
    # results.csv lives at the run root, not under weights/.
    targets.append((run_dir / "results.csv", "results.csv", "text/csv"))
    key_map["results.csv"] = "results_csv"

    for local_path, filename, content_type in targets:
        if not local_path.exists():
            print(f"  [skip] {filename} not found at {local_path}")
            continue

        b2_key = f"checkpoints/{run_id}/{filename}"
        print(f"  Uploading {filename} -> {b2_key} ...")
        try:
            client.upload_file(
                str(local_path),
                bucket,
                b2_key,
                ExtraArgs={"ContentType": content_type},
            )
            url = _public_url(endpoint, bucket, b2_key)
            urls[key_map[filename]] = url
            print(f"  OK: {url}")
        except Exception as exc:
            failures.append(filename)
            print(f"  [FAIL] {filename}: {exc}")

    if failures:
        raise RuntimeError(
            f"upload_checkpoints: {len(failures)} file(s) failed to upload: "
            + ", ".join(failures)
        )

    return urls


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload training checkpoints to Backblaze B2."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run identifier, e.g. run_20260310_001.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Ultralytics output directory containing weights/ subfolder.",
    )
    parser.add_argument(
        "--no-mongo",
        action="store_true",
        help="Skip merging uploaded URLs into the experiments document in MongoDB.",
    )
    return parser.parse_args()


def _merge_mongo_checkpoints(run_id: str, urls: dict[str, str]) -> None:
    """Merge uploaded URLs into the experiments document without overwriting
    existing keys for files that were skipped this run."""
    from src.db.connection import get_db

    db = get_db()
    update = {f"checkpoints.{k}": v for k, v in urls.items()}
    result = db["experiments"].update_one({"run_id": run_id}, {"$set": update})
    if result.matched_count == 0:
        print(f"  [warn] No experiments document found for run_id={run_id}.")
    else:
        print(f"  Mongo updated: {list(urls)} on run_id={run_id}.")


if __name__ == "__main__":
    args = _parse_args()
    result = upload_checkpoints(run_id=args.run_id, run_dir=args.run_dir)
    print("\nCheckpoint URLs:")
    for key, url in result.items():
        print(f"  {key}: {url}")
    if result and not args.no_mongo:
        print("\nUpdating MongoDB ...")
        try:
            _merge_mongo_checkpoints(args.run_id, result)
        except Exception as exc:
            print(f"  [warn] Mongo update failed: {exc}")
