"""
src/data/download.py
--------------------
Download and extract RDD2022 country ZIPs from Sekilab S3.

Usage:
    python -m src.data.download [--dest DIR] [--countries C1 C2 ...]

The script is idempotent: if the ZIP file already exists on disk it is
not downloaded again.  If the extracted directory already exists the
extraction step is skipped.

Environment:
    RDD_DATA_ROOT — destination root for the extracted dataset.
                    Overridden by --dest if supplied.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = (
    "https://bigdatacup.s3.ap-northeast-1.amazonaws.com"
    "/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022"
)

COUNTRIES = [
    "Japan",
    "India",
    "Czech",
    "Norway",
    "United_States",
    "China_MotorBike",
    "China_Drone",
]

CHUNK_SIZE = 1 << 20  # 1 MiB


def _dest_root() -> Path:
    """Return the dataset destination directory from the environment."""
    root = os.getenv("RDD_DATA_ROOT")
    if not root:
        raise EnvironmentError(
            "RDD_DATA_ROOT is not set. "
            "Set it in .env or pass --dest on the command line."
        )
    return Path(root)


def download_country(country: str, dest: Path) -> Path:
    """Download one country ZIP to *dest* and return the local ZIP path.

    Args:
        country: Country name matching the Sekilab S3 filename convention.
        dest: Directory where the ZIP will be saved.

    Returns:
        Path to the downloaded ZIP file.

    Raises:
        requests.HTTPError: If the HTTP request fails.
    """
    dest.mkdir(parents=True, exist_ok=True)
    zip_name = f"RDD2022_{country}.zip"
    zip_path = dest / zip_name

    if zip_path.exists():
        print(f"  [skip] {zip_name} already present.")
        return zip_path

    url = f"{BASE_URL}/{zip_name}"
    print(f"  Downloading {url} …")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r    {pct:5.1f}%", end="", flush=True)
    print()
    return zip_path


def extract_country(zip_path: Path, dest: Path) -> Path:
    """Extract a country ZIP into *dest*.

    Args:
        zip_path: Path to the ZIP file.
        dest: Directory where contents will be extracted.

    Returns:
        Path to the extracted country directory.
    """
    # Derive country name from zip filename: RDD2022_Japan.zip → Japan
    country = zip_path.stem.replace("RDD2022_", "")
    country_dir = dest / country

    if country_dir.exists():
        print(f"  [skip] {country} directory already extracted.")
        return country_dir

    print(f"  Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    return country_dir


def download_all(countries: list[str], dest: Path) -> None:
    """Download and extract all requested countries.

    Args:
        countries: List of country names to process.
        dest: Root destination directory.
    """
    zips_dir = dest / "_zips"
    for country in countries:
        print(f"\n[{country}]")
        zip_path = download_country(country, zips_dir)
        extract_country(zip_path, dest)
    print("\nAll downloads complete.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download RDD2022 country ZIPs from Sekilab S3."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination root directory (default: RDD_DATA_ROOT from .env).",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=COUNTRIES,
        metavar="COUNTRY",
        help=f"Countries to download (default: all {len(COUNTRIES)}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dest = args.dest or _dest_root()
    download_all(args.countries, dest)
