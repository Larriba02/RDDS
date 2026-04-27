"""
src/data/test_b2.py
-------------------
Smoke test for Backblaze B2 access.

Verifies that the B2 credentials in .env are valid and the bucket is
reachable by listing a small sample of objects.

Usage:
    python -m src.data.test_b2
"""

import os
import sys

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointResolutionError
from dotenv import load_dotenv

load_dotenv()


def test_b2() -> None:
    key_id = os.getenv("BACKBLAZE_KEY_ID")
    app_key = os.getenv("BACKBLAZE_APP_KEY")
    bucket = os.getenv("BACKBLAZE_BUCKET")
    endpoint = os.getenv("BACKBLAZE_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")

    missing = [k for k, v in {
        "BACKBLAZE_KEY_ID": key_id,
        "BACKBLAZE_APP_KEY": app_key,
        "BACKBLAZE_BUCKET": bucket,
    }.items() if not v]
    if missing:
        print(f"[B2] FAIL — missing env vars: {', '.join(missing)}")
        sys.exit(1)

    try:
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=key_id,
            aws_secret_access_key=app_key,
            config=Config(signature_version="s3v4"),
        )
        # Get total object count
        total = 0
        paginator = client.get_paginator("list_objects_v2")
        first_page = True
        sample = []
        for page in paginator.paginate(Bucket=bucket):
            total += page.get("KeyCount", 0)
            if first_page:
                sample = [obj["Key"] for obj in page.get("Contents", [])[:5]]
                first_page = False

        print(f"[B2] OK — bucket '{bucket}' accessible ({total} objects total)")
        for key in sample:
            print(f"  {key}")
    except ClientError as exc:
        print(f"[B2] FAIL — {exc}")
        sys.exit(1)
    except EndpointResolutionError as exc:
        print(f"[B2] FAIL — endpoint unreachable: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    test_b2()
