"""
src/data/ingest.py
------------------
Write one MongoDB document per image to the ``images_metadata`` collection.

Each document follows the schema defined in RDDS_Dev_Steps.md §STEP 2:
    image_id  : MD5 hash of relative filepath
    filepath  : e.g. "Japan/train/00001.jpg"
    country   : e.g. "Japan"
    split     : "train" | "val" | "test"
    width     : int
    height    : int
    annotations: list of {label, xmin, ymin, xmax, ymax}

The script is idempotent: images whose ``image_id`` already exists in the
collection are skipped (not updated).  This makes it safe to re-run after
partial ingestion.

Prerequisites:
    - convert.py has been run (YOLO .txt files exist).
    - split.py has been run (logs/splits.json exists).
    - MONGO_URI is set in .env.
    - RDD_DATA_ROOT is set in .env.

Usage:
    python -m src.data.ingest [--data-root DIR] [--batch-size 500]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.db.connection import get_db

load_dotenv()

LOG_DIR = Path("logs")
SPLITS_JSON = LOG_DIR / "splits.json"

CLASS_ID_MAP = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
CLASS_NAME_MAP = {v: k for k, v in CLASS_ID_MAP.items()}


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


def _image_id(rel_path: str) -> str:
    """Compute MD5 hash of the relative filepath string."""
    return hashlib.md5(rel_path.encode("utf-8")).hexdigest()


def _load_splits() -> dict[str, str]:
    """Load the splits mapping from logs/splits.json.

    Returns:
        Dict mapping image_id → split string.

    Raises:
        FileNotFoundError: If splits.json does not exist.
    """
    if not SPLITS_JSON.exists():
        raise FileNotFoundError(
            f"{SPLITS_JSON} not found. Run split.py first."
        )
    with open(SPLITS_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def _read_annotations_from_xml(xml_path: Path) -> tuple[int, int, list[dict]]:
    """Parse PascalVOC XML to extract image dimensions and raw annotations.

    Args:
        xml_path: Path to the annotation XML file.

    Returns:
        Tuple of (width, height, annotations) where annotations is a list
        of dicts with keys label, xmin, ymin, xmax, ymax.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    width = int(size.findtext("width", "0")) if size is not None else 0
    height = int(size.findtext("height", "0")) if size is not None else 0

    annotations: list[dict] = []
    for obj in root.findall("object"):
        label = obj.findtext("name", "").strip()
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = int(float(bndbox.findtext("xmin", "0")))
            ymin = int(float(bndbox.findtext("ymin", "0")))
            xmax = int(float(bndbox.findtext("xmax", "0")))
            ymax = int(float(bndbox.findtext("ymax", "0")))
        except ValueError:
            continue
        annotations.append(
            {"label": label, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        )
    return width, height, annotations


def _build_documents(
    data_root: Path,
    splits: dict[str, str],
) -> list[dict[str, Any]]:
    """Build MongoDB documents for all images.

    Args:
        data_root: Root of the RDD2022 dataset.
        splits: image_id → split mapping from splits.json.

    Returns:
        List of document dicts ready for MongoDB insertion.
    """
    docs: list[dict[str, Any]] = []

    for country_dir in sorted(data_root.iterdir()):
        if not country_dir.is_dir() or country_dir.name.startswith("_"):
            continue
        country = country_dir.name
        for official_split in ("train", "test"):
            img_dir = country_dir / official_split / "images"
            ann_dir = country_dir / official_split / "annotations" / "xmls"
            if not img_dir.exists():
                continue

            for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
                rel_path = f"{country}/{official_split}/{img_path.name}"
                img_id = _image_id(rel_path)

                assigned_split = splits.get(img_id)
                if assigned_split is None or assigned_split == "excluded":
                    continue

                xml_path = ann_dir / f"{img_path.stem}.xml" if ann_dir.exists() else None
                if xml_path and xml_path.exists():
                    width, height, annotations = _read_annotations_from_xml(xml_path)
                else:
                    # Fallback: try to read dimensions from image file
                    try:
                        from PIL import Image as _PIL_Image
                        with _PIL_Image.open(img_path) as im:
                            width, height = im.size
                    except Exception:
                        width, height = 0, 0
                    annotations = []

                doc: dict[str, Any] = {
                    "image_id": img_id,
                    "filepath": rel_path,
                    "country": country,
                    "source_device": "smartphone",  # RDD2022 default
                    "split": assigned_split,
                    "width": width,
                    "height": height,
                    "annotations": annotations,
                }
                docs.append(doc)

    return docs


def ingest(data_root: Path, batch_size: int = 500) -> int:
    """Ingest image metadata into MongoDB.

    Skips images whose ``image_id`` already exists (idempotent).

    Args:
        data_root: Root of the RDD2022 dataset.
        batch_size: Number of documents per bulk-write batch.

    Returns:
        Number of new documents inserted.
    """
    db = get_db()
    col = db["images_metadata"]

    splits = _load_splits()
    docs = _build_documents(data_root, splits)

    print(f"Total images to consider: {len(docs)}")

    # Fetch existing image_ids to skip
    existing_ids: set[str] = set(
        d["image_id"] for d in col.find({}, {"image_id": 1, "_id": 0})
    )
    new_docs = [d for d in docs if d["image_id"] not in existing_ids]
    print(f"Already in MongoDB: {len(existing_ids)} | New: {len(new_docs)}")

    if not new_docs:
        print("Nothing to insert. Collection is up to date.")
        return 0

    # Bulk insert in batches
    inserted = 0
    for i in range(0, len(new_docs), batch_size):
        batch = new_docs[i : i + batch_size]
        result = col.insert_many(batch, ordered=False)
        inserted += len(result.inserted_ids)
        print(f"  Inserted batch {i // batch_size + 1}: {len(result.inserted_ids)} docs")

    print(f"\nIngestion complete: {inserted} new documents written to images_metadata.")
    return inserted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest RDD2022 image metadata into MongoDB images_metadata."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root directory (default: RDD_DATA_ROOT from .env).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Documents per bulk-write batch (default: 500).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    root = _data_root(args.data_root)
    ingest(root, args.batch_size)
