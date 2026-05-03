"""
src/data/split.py
-----------------
Assign train / val / test splits for RDD2022.

Rules (non-negotiable per CLAUDE.md §2):

1. The official RDD2022 test split is always kept at 100% — never sampled.
2. Within the official train split, 1,000 images per country are reserved
   as the fixed validation set, stratified by (country, dominant damage
   class) using StratifiedShuffleSplit.
   - Dominant class per image = most frequent label in its YOLO .txt.
   - Ties broken alphabetically (D00 before D10, etc.).
3. The remaining training images are sampled at SAMPLE_RATIO stratified by
   country.  SAMPLE_RATIO is read from the environment (default 1.0).

Output:
    logs/splits.json  — maps image_id → split ("train" | "val" | "test")
    logs/split_summary.txt — human-readable counts

The function also returns a dict:
    {image_id: {"filepath": rel_path, "split": ..., "dominant_class": ...}}

image_id is the MD5 hash of the relative filepath (e.g.
``Japan/train/00001.jpg``), matching the MongoDB convention.

Usage:
    python -m src.data.split [--data-root DIR] [--sample-ratio 0.10]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedShuffleSplit

load_dotenv()

LOG_DIR = Path("logs")
SPLITS_JSON = LOG_DIR / "splits.json"
SPLIT_SUMMARY = LOG_DIR / "split_summary.txt"

VAL_PER_COUNTRY = 1000
RANDOM_SEED = 42
CLASS_NAMES = ["D00", "D10", "D20", "D40"]
CLASS_ID_MAP = {i: name for i, name in enumerate(CLASS_NAMES)}


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


def _sample_ratio(override: float | None = None) -> float:
    if override is not None:
        return override
    return float(os.getenv("SAMPLE_RATIO", "1.0"))


def _image_id(rel_path: str) -> str:
    """Compute MD5 hash of the relative filepath string.

    Args:
        rel_path: Relative path string, e.g. ``Japan/train/00001.jpg``.

    Returns:
        32-character hexadecimal MD5 string.
    """
    return hashlib.md5(rel_path.encode("utf-8")).hexdigest()


def _dominant_class(txt_path: Path) -> str:
    """Return the dominant damage class for one image.

    Reads the YOLO .txt label file and returns the most frequent class
    label.  Ties are broken alphabetically.  Images with no annotations
    (background) return ``"none"``.

    Args:
        txt_path: Path to the YOLO label .txt file.

    Returns:
        Class name string (e.g. ``"D00"``) or ``"none"`` for empty labels.
    """
    if not txt_path.exists():
        return "none"
    try:
        lines = txt_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "none"

    counter: Counter = Counter()
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls_id = int(parts[0])
        except ValueError:
            continue
        cls_name = CLASS_ID_MAP.get(cls_id)
        if cls_name:
            counter[cls_name] += 1

    if not counter:
        return "none"
    max_count = max(counter.values())
    candidates = sorted(k for k, v in counter.items() if v == max_count)
    return candidates[0]


def _collect_images(data_root: Path) -> dict[str, dict[str, Any]]:
    """Walk the dataset and collect per-image metadata.

    Args:
        data_root: Root of the RDD2022 dataset.

    Returns:
        Dict mapping image_id → metadata dict with keys:
        filepath, country, official_split, dominant_class.
    """
    images: dict[str, dict[str, Any]] = {}

    for country_dir in sorted(data_root.iterdir()):
        if not country_dir.is_dir() or country_dir.name.startswith("_"):
            continue
        country = country_dir.name
        for official_split in ("train", "test"):
            img_dir = country_dir / official_split / "images"
            if not img_dir.exists():
                continue
            labels_dir = country_dir / official_split / "labels"
            for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
                rel_path = f"{country}/{official_split}/images/{img_path.name}"
                img_id = _image_id(rel_path)
                # Labels live in labels/ (Ultralytics convention). Fall back to
                # images/ for datasets converted before this convention was adopted.
                txt_path = labels_dir / f"{img_path.stem}.txt"
                if not txt_path.exists():
                    txt_path = img_dir / f"{img_path.stem}.txt"
                dom_cls = _dominant_class(txt_path)
                images[img_id] = {
                    "filepath": rel_path,
                    "country": country,
                    "official_split": official_split,
                    "dominant_class": dom_cls,
                }

    return images


def compute_splits(
    data_root: Path,
    sample_ratio: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Compute train/val/test split assignments.

    Args:
        data_root: Root of the RDD2022 dataset.
        sample_ratio: Fraction of remaining train images to include in
            the active training set (0.0–1.0).

    Returns:
        Dict mapping image_id → metadata dict with a ``split`` key
        (``"train"``, ``"val"``, or ``"test"``) added.
    """
    images = _collect_images(data_root)

    # 1. Official test split → always "test"
    for meta in images.values():
        if meta["official_split"] == "test":
            meta["split"] = "test"

    # 2. Per-country: stratified val reservation
    countries = sorted({m["country"] for m in images.values() if m["official_split"] == "train"})

    rng = random.Random(RANDOM_SEED)

    for country in countries:
        country_train_ids = [
            img_id
            for img_id, meta in images.items()
            if meta["official_split"] == "train" and meta["country"] == country
        ]

        if len(country_train_ids) <= VAL_PER_COUNTRY:
            # Tiny dataset (e.g. mini smoke-test): treat everything as train
            for img_id in country_train_ids:
                images[img_id]["split"] = "train"
            continue

        # Build stratification labels (dominant_class)
        strat_labels = np.array(
            [images[img_id]["dominant_class"] for img_id in country_train_ids]
        )

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=VAL_PER_COUNTRY,
            random_state=RANDOM_SEED,
        )
        indices = list(sss.split(country_train_ids, strat_labels))
        train_idx, val_idx = indices[0]

        val_ids = {country_train_ids[i] for i in val_idx}
        remaining_ids = [country_train_ids[i] for i in train_idx]

        for img_id in val_ids:
            images[img_id]["split"] = "val"

        # 3. Sample remaining at sample_ratio stratified by country
        if sample_ratio >= 1.0:
            for img_id in remaining_ids:
                images[img_id]["split"] = "train"
        else:
            n_sample = max(1, int(len(remaining_ids) * sample_ratio))
            sampled = rng.sample(remaining_ids, n_sample)
            sampled_set = set(sampled)
            for img_id in remaining_ids:
                images[img_id]["split"] = "train" if img_id in sampled_set else "excluded"

    return images


def save_splits(images: dict[str, dict[str, Any]]) -> None:
    """Persist split assignments to logs/.

    Args:
        images: Full image metadata dict returned by compute_splits.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Compact form for splits.json: image_id → split
    splits_compact = {img_id: meta["split"] for img_id, meta in images.items()}
    with open(SPLITS_JSON, "w", encoding="utf-8") as fh:
        json.dump(splits_compact, fh, indent=2)

    # Summary
    split_counts: Counter = Counter(meta["split"] for meta in images.values())
    country_counts: dict[str, Counter] = {}
    for meta in images.values():
        country = meta["country"]
        sp = meta["split"]
        if country not in country_counts:
            country_counts[country] = Counter()
        country_counts[country][sp] += 1

    lines: list[str] = ["Split summary\n" + "=" * 40]
    for country in sorted(country_counts):
        lines.append(f"\n{country}:")
        for sp in ("train", "val", "test", "excluded"):
            n = country_counts[country].get(sp, 0)
            if n:
                lines.append(f"  {sp:<10} {n:>6}")
    lines.append("\n" + "-" * 40)
    lines.append(f"{'TOTAL':<10}")
    for sp in ("train", "val", "test", "excluded"):
        n = split_counts.get(sp, 0)
        if n:
            lines.append(f"  {sp:<10} {n:>6}")

    summary_text = "\n".join(lines) + "\n"
    SPLIT_SUMMARY.write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"Splits saved to: {SPLITS_JSON}")
    print(f"Summary saved to: {SPLIT_SUMMARY}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and save train/val/test splits for RDD2022."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root directory (default: RDD_DATA_ROOT from .env).",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=None,
        help="Fraction of training images to include (default: SAMPLE_RATIO from .env).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    root = _data_root(args.data_root)
    ratio = _sample_ratio(args.sample_ratio)
    print(f"Computing splits: data_root={root}, sample_ratio={ratio}")
    result = compute_splits(root, ratio)
    save_splits(result)
