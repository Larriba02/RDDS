"""
src/data/validate.py
--------------------
Parse every PascalVOC XML annotation in RDD2022 and flag/discard bounding
boxes that are invalid:

  - xmin < 0 or ymin < 0
  - xmax > image width or ymax > image height
  - xmin >= xmax or ymin >= ymax  (degenerate / zero-area)

Valid annotations are returned as a dict keyed by relative filepath.
Invalid ones are logged to logs/discarded_annotations.txt.

Usage:
    python -m src.data.validate [--data-root DIR]

Returns:
    Dict[relative_path, list[dict]] — clean annotations per image.
    Written to logs/ as a side-effect.
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path("logs")
DISCARD_LOG = LOG_DIR / "discarded_annotations.txt"

VALID_LABELS = {"D00", "D10", "D20", "D40"}


def _data_root(override: Path | None = None) -> Path:
    """Resolve the dataset root directory."""
    if override:
        return override
    root = os.getenv("RDD_DATA_ROOT")
    if not root:
        raise EnvironmentError(
            "RDD_DATA_ROOT is not set. "
            "Set it in .env or pass --data-root on the command line."
        )
    return Path(root)


def _parse_xml(xml_path: Path) -> tuple[int, int, list[dict[str, Any]]]:
    """Parse a single PascalVOC XML file.

    Args:
        xml_path: Path to the annotation XML file.

    Returns:
        Tuple of (width, height, list of raw annotation dicts).
        Each annotation dict has keys: label, xmin, ymin, xmax, ymax.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.findtext("width", default="0"))
    height = int(size.findtext("height", default="0"))

    annotations: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        label = obj.findtext("name", default="").strip()
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


def _is_valid(ann: dict[str, Any], width: int, height: int) -> tuple[bool, str]:
    """Check a single annotation against validation rules.

    Args:
        ann: Annotation dict with label, xmin, ymin, xmax, ymax.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Tuple of (is_valid, reason_if_invalid).
    """
    if ann.get("label") not in VALID_LABELS:
        return False, f"unknown label '{ann.get('label')}'"
    if ann["xmin"] < 0 or ann["ymin"] < 0:
        return False, "negative coordinate"
    if ann["xmax"] > width or ann["ymax"] > height:
        return False, f"out of bounds (img {width}x{height})"
    if ann["xmin"] >= ann["xmax"] or ann["ymin"] >= ann["ymax"]:
        return False, "degenerate bbox (zero or negative area)"
    return True, ""


def validate_dataset(
    data_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Validate all PascalVOC XMLs under *data_root*.

    Walks every country/train and country/test directory, parses XML
    annotation files, applies the three validation rules, discards bad
    bboxes, and logs them.

    Args:
        data_root: Root of the RDD2022 dataset (contains country dirs).

    Returns:
        Dict mapping relative filepath (e.g. ``Japan/train/00001.jpg``)
        to the list of *valid* annotation dicts for that image.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    clean: dict[str, list[dict[str, Any]]] = {}
    discard_lines: list[str] = []
    total_boxes = 0
    discarded_boxes = 0

    for country_dir in sorted(data_root.iterdir()):
        if not country_dir.is_dir() or country_dir.name.startswith("_"):
            continue
        country = country_dir.name
        for split in ("train", "test"):
            ann_dir = country_dir / split / "annotations" / "xmls"
            img_dir = country_dir / split / "images"
            if not ann_dir.exists():
                continue

            for xml_path in sorted(ann_dir.glob("*.xml")):
                img_stem = xml_path.stem
                # Try .jpg first, then .png
                img_path = img_dir / f"{img_stem}.jpg"
                if not img_path.exists():
                    img_path = img_dir / f"{img_stem}.png"

                rel_path = f"{country}/{split}/{img_path.name}"
                width, height, raw_anns = _parse_xml(xml_path)

                valid_anns: list[dict[str, Any]] = []
                for ann in raw_anns:
                    total_boxes += 1
                    ok, reason = _is_valid(ann, width, height)
                    if ok:
                        valid_anns.append(ann)
                    else:
                        discarded_boxes += 1
                        discard_lines.append(
                            f"{rel_path} | {ann} | reason: {reason}"
                        )
                clean[rel_path] = valid_anns

    # Write discard log
    with open(DISCARD_LOG, "w", encoding="utf-8") as fh:
        fh.write(f"# Discarded annotations — {discarded_boxes}/{total_boxes} boxes\n")
        fh.write("\n".join(discard_lines))
        if discard_lines:
            fh.write("\n")

    print(
        f"Validation complete: {len(clean)} images, "
        f"{total_boxes - discarded_boxes} valid boxes, "
        f"{discarded_boxes} discarded."
    )
    print(f"Discard log: {DISCARD_LOG}")
    return clean


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate PascalVOC annotations in RDD2022."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root directory (default: RDD_DATA_ROOT from .env).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    root = _data_root(args.data_root)
    validate_dataset(root)
