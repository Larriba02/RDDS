"""
src/data/convert.py
-------------------
Convert RDD2022 PascalVOC XML annotations to YOLO .txt format.

Class map (fixed — never change without updating train.py and split.py):
    D00 → 0   Longitudinal crack
    D10 → 1   Transverse crack
    D20 → 2   Alligator crack
    D40 → 3   Pothole

For each image a .txt file is created in a parallel ``labels/`` directory
(e.g. ``Japan/train/labels/Japan_000001.txt`` for an image at
``Japan/train/images/Japan_000001.jpg``).  This matches the Ultralytics
path convention used during training.  Each line in the .txt file has the
format:
    <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>

Images without any valid annotation get an empty .txt file (YOLO
background-only convention).

The step is idempotent: if the .txt file already exists it is not rewritten
unless --force is passed.

Usage:
    python -m src.data.convert [--data-root DIR] [--force]
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CLASS_MAP: dict[str, int] = {
    "D00": 0,
    "D10": 1,
    "D20": 2,
    "D40": 3,
}


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


def _xml_to_yolo_lines(
    xml_path: Path,
    width: int,
    height: int,
    discards: list[str] | None = None,
) -> list[str]:
    """Convert one XML file to a list of YOLO annotation lines.

    Skips bounding boxes whose label is not in CLASS_MAP and boxes that
    are invalid (negative coordinates, out-of-bounds, degenerate).

    Args:
        xml_path: Path to the PascalVOC XML annotation file.
        width: Image width in pixels (from XML <size>).
        height: Image height in pixels (from XML <size>).
        discards: Optional list to which discard records are appended.
            Each entry is a tab-separated string: path, label, coords, reason.

    Returns:
        List of YOLO-format strings ready to be joined with newlines.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def _discard(label: str, coords: str, reason: str) -> None:
        if discards is not None:
            discards.append(f"{xml_path}\t{label}\t{coords}\t{reason}")

    lines: list[str] = []
    for obj in root.findall("object"):
        label = obj.findtext("name", default="").strip()
        if label not in CLASS_MAP:
            continue  # unknown class — out of scope, not a bbox error
        bndbox = obj.find("bndbox")
        if bndbox is None:
            _discard(label, "N/A", "missing bndbox element")
            continue
        try:
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))
        except ValueError:
            _discard(label, "unparseable", "non-numeric coordinate")
            continue

        coords_str = f"xmin={xmin},ymin={ymin},xmax={xmax},ymax={ymax}"
        # Validation (mirror validate.py rules)
        if xmin < 0 or ymin < 0:
            _discard(label, coords_str, "negative coordinate")
            continue
        if xmax > width or ymax > height:
            _discard(label, coords_str, f"out-of-bounds (img {width}x{height})")
            continue
        if xmin >= xmax or ymin >= ymax:
            _discard(label, coords_str, "degenerate bbox (zero or negative area)")
            continue

        # YOLO normalised coords
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        w_norm = (xmax - xmin) / width
        h_norm = (ymax - ymin) / height

        cls_id = CLASS_MAP[label]
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    return lines


def _get_image_dims(xml_path: Path) -> tuple[int, int]:
    """Read width and height from a PascalVOC XML <size> element.

    Args:
        xml_path: Path to the XML file.

    Returns:
        Tuple (width, height) in pixels.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    return width, height


def convert_dataset(data_root: Path, force: bool = False) -> int:
    """Convert all XML annotations in *data_root* to YOLO .txt format.

    Args:
        data_root: Root of the RDD2022 dataset.
        force: Overwrite existing .txt files if True.

    Returns:
        Number of .txt files written.
    """
    written = 0
    skipped = 0
    all_discards: list[str] = []

    for country_dir in sorted(data_root.iterdir()):
        if not country_dir.is_dir() or country_dir.name.startswith("_"):
            continue
        country = country_dir.name  # noqa: F841 (used for context, not yet logged)
        for split in ("train", "test"):
            ann_dir = country_dir / split / "annotations" / "xmls"
            img_dir = country_dir / split / "images"
            if not ann_dir.exists():
                continue

            labels_dir = country_dir / split / "labels"
            labels_dir.mkdir(exist_ok=True)

            for xml_path in sorted(ann_dir.glob("*.xml")):
                # Determine image path
                img_stem = xml_path.stem
                img_path = img_dir / f"{img_stem}.jpg"
                if not img_path.exists():
                    img_path = img_dir / f"{img_stem}.png"

                label_path = labels_dir / f"{img_stem}.txt"

                if label_path.exists() and not force:
                    skipped += 1
                    continue

                width, height = _get_image_dims(xml_path)
                if width == 0 or height == 0:
                    # Cannot normalise without dimensions; skip silently
                    continue

                lines = _xml_to_yolo_lines(xml_path, width, height, discards=all_discards)
                label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                written += 1

    # Write discard log so invalid bboxes are traceable.
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    discard_log = log_dir / "discarded_annotations.txt"
    with open(discard_log, "w", encoding="utf-8") as fh:
        fh.write("# xml_path\tlabel\tcoords\treason\n")
        for line in all_discards:
            fh.write(line + "\n")

    print(
        f"Conversion complete: {written} label files written, "
        f"{skipped} already existed (use --force to overwrite). "
        f"{len(all_discards)} bbox(es) discarded — see {discard_log}."
    )
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RDD2022 PascalVOC XMLs to YOLO .txt labels."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root directory (default: RDD_DATA_ROOT from .env).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .txt label files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    root = _data_root(args.data_root)
    convert_dataset(root, force=args.force)
