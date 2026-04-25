"""
src/data/upload_to_cloud.py
---------------------------
Upload the processed RDD2022 dataset to Backblaze B2 cloud storage.

This script is run **once** by the team member who performed the initial
preprocessing.  All other machines pull from B2 using the public download
URL stored in the bucket.

What is uploaded:
    - YOLO label files (*.txt) alongside images
    - logs/class_distribution.json
    - logs/splits.json
    Optionally the image files themselves (large — only if storage allows).

The bucket path prefix is: ``rdd2022/<country>/<split>/``.

Environment variables required:
    BACKBLAZE_KEY_ID    — Application key ID
    BACKBLAZE_APP_KEY   — Application key secret
    BACKBLAZE_BUCKET    — Bucket name

Usage:
    python -m src.data.upload_to_cloud [--data-root DIR] [--include-images]
"""

from __future__ import annotations

import argparse
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

def _b2_client():
    """Create a boto3 S3 client configured for Backblaze B2.

    The endpoint URL is derived from the ``BACKBLAZE_ENDPOINT`` environment
    variable.  If not set, defaults to the US-West-004 region endpoint.
    Set ``BACKBLAZE_ENDPOINT`` in ``.env`` if your bucket is in a different
    region (e.g. ``https://s3.eu-central-003.backblazeb2.com``).

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
        config=Config(signature_version="s3v4"),
    )


def _bucket_name() -> str:
    bucket = os.getenv("BACKBLAZE_BUCKET")
    if not bucket:
        raise EnvironmentError("BACKBLAZE_BUCKET is not set in .env.")
    return bucket


def _data_root(override: Path | None = None) -> Path:
    if override:
        return override
    root = os.getenv("RDD_DATA_ROOT")
    if not root:
        raise EnvironmentError(
            "RDD_DATA_ROOT is not set. "
            "Set it in .env or pass --data-root on the command line."
        )
    return Path(root)


def _upload_file(client, bucket: str, local_path: Path, b2_key: str) -> None:
    content_type, _ = mimetypes.guess_type(str(local_path))
    extra = {"ContentType": content_type or "application/octet-stream"}
    client.upload_file(str(local_path), bucket, b2_key, ExtraArgs=extra)


def _upload_batch(client, bucket: str, jobs: list[tuple[Path, str]], workers: int) -> int:
    """Upload a list of (local_path, b2_key) pairs in parallel.

    Returns the number of files successfully uploaded.
    """
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_upload_file, client, bucket, local, key): key
            for local, key in jobs
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                future.result()
                done += 1
                print(f"  [{done}/{len(jobs)}] {key}")
            except Exception as exc:
                print(f"  [FAIL] {key} — {exc}")
    return done


def upload_logs(client, bucket: str, workers: int) -> None:
    log_dir = Path("logs")
    jobs = []
    for fname in ("class_distribution.json", "splits.json"):
        local = log_dir / fname
        if local.exists():
            jobs.append((local, f"rdd2022/logs/{fname}"))
        else:
            print(f"  [skip] {local} not found.")
    if jobs:
        _upload_batch(client, bucket, jobs, workers)


def upload_labels(client, bucket: str, data_root: Path, workers: int) -> int:
    jobs = []
    for country_dir in sorted(data_root.iterdir()):
        if not country_dir.is_dir() or country_dir.name.startswith("_"):
            continue
        country = country_dir.name
        for split in ("train", "test"):
            img_dir = country_dir / split / "images"
            if not img_dir.exists():
                continue
            for txt_path in sorted(img_dir.glob("*.txt")):
                b2_key = f"rdd2022/{country}/{split}/labels/{txt_path.name}"
                jobs.append((txt_path, b2_key))
    return _upload_batch(client, bucket, jobs, workers)


def upload_images(client, bucket: str, data_root: Path, workers: int) -> int:
    jobs = []
    for country_dir in sorted(data_root.iterdir()):
        if not country_dir.is_dir() or country_dir.name.startswith("_"):
            continue
        country = country_dir.name
        for split in ("train", "test"):
            img_dir = country_dir / split / "images"
            if not img_dir.exists():
                continue
            for ext in ("*.jpg", "*.png"):
                for img_path in sorted(img_dir.glob(ext)):
                    b2_key = f"rdd2022/{country}/{split}/images/{img_path.name}"
                    jobs.append((img_path, b2_key))
    return _upload_batch(client, bucket, jobs, workers)


def upload_dataset(
    data_root: Path,
    include_images: bool = False,
    workers: int = 8,
) -> None:
    client = _b2_client()
    bucket = _bucket_name()

    print("Uploading log files …")
    upload_logs(client, bucket, workers)

    print("\nUploading YOLO label files …")
    n_labels = upload_labels(client, bucket, data_root, workers)
    print(f"  {n_labels} label files uploaded.")

    if include_images:
        print("\nUploading image files (this may take a long time) …")
        n_images = upload_images(client, bucket, data_root, workers)
        print(f"  {n_images} image files uploaded.")

    print("\nUpload complete.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload processed RDD2022 dataset to Backblaze B2."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root directory (default: RDD_DATA_ROOT from .env).",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help="Also upload image files (large — optional).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel upload threads (default: 8).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    root = _data_root(args.data_root)
    upload_dataset(root, include_images=args.include_images, workers=args.workers)
