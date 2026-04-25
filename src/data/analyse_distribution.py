"""
src/data/analyse_distribution.py
---------------------------------
Count annotation instances per damage class per country, then print a
distribution table and save the result to logs/class_distribution.json.

This output is used in train.py to calibrate the ``cls_weight`` parameter
so that rare classes (especially D40 potholes) are not under-penalised
during training.  Do not skip this step.

Usage:
    python -m src.data.analyse_distribution [--data-root DIR] [--split train]

The script reads YOLO .txt label files (output of convert.py), not the
raw XML files, so convert.py must run first.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOG_DIR = Path("logs")
OUTPUT_FILE = LOG_DIR / "class_distribution.json"

CLASS_NAMES: dict[int, str] = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
ALL_CLASSES = list(CLASS_NAMES.values())


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


def count_distribution(
    data_root: Path, split: str = "train"
) -> dict[str, dict[str, int]]:
    """Count annotation instances per class per country.

    Reads YOLO .txt label files from ``<country>/<split>/images/*.txt``.

    Args:
        data_root: Root of the RDD2022 dataset.
        split: Which split to analyse (``"train"`` or ``"test"``).

    Returns:
        Nested dict: ``{country: {class_name: count}}``.
        Also includes a synthetic ``"_total"`` key with global counts.
    """
    distribution: dict[str, dict[str, int]] = {}
    total: dict[str, int] = {cls: 0 for cls in ALL_CLASSES}

    for country_dir in sorted(data_root.iterdir()):
        if not country_dir.is_dir() or country_dir.name.startswith("_"):
            continue
        country = country_dir.name
        img_dir = country_dir / split / "images"
        if not img_dir.exists():
            continue

        counts: dict[str, int] = {cls: 0 for cls in ALL_CLASSES}
        for txt_path in img_dir.glob("*.txt"):
            try:
                lines = txt_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls_id = int(parts[0])
                except ValueError:
                    continue
                cls_name = CLASS_NAMES.get(cls_id)
                if cls_name:
                    counts[cls_name] += 1

        distribution[country] = counts
        for cls in ALL_CLASSES:
            total[cls] += counts[cls]

    distribution["_total"] = total
    return distribution


def _compute_cls_weights(total: dict[str, int]) -> dict[str, float]:
    """Compute inverse-frequency weights for each class.

    The weight for class c is proportional to ``max_count / count_c``,
    then normalised so that the mean weight across all classes is 1.0.

    Args:
        total: Dict mapping class name to instance count.

    Returns:
        Dict mapping class name to float weight.
    """
    counts = [total.get(cls, 1) for cls in ALL_CLASSES]
    max_count = max(counts) if counts else 1
    raw = [max_count / max(c, 1) for c in counts]
    mean_raw = sum(raw) / len(raw)
    weights = {cls: round(raw[i] / mean_raw, 4) for i, cls in enumerate(ALL_CLASSES)}
    return weights


def _print_table(distribution: dict[str, dict[str, int]]) -> None:
    """Print a human-readable distribution table to stdout."""
    header = f"{'Country':<22}" + "".join(f"{cls:>8}" for cls in ALL_CLASSES) + f"{'Total':>10}"
    print(header)
    print("-" * len(header))
    for country, counts in distribution.items():
        if country == "_total":
            continue
        row_total = sum(counts.values())
        line = f"{country:<22}" + "".join(f"{counts[cls]:>8}" for cls in ALL_CLASSES) + f"{row_total:>10}"
        print(line)
    print("-" * len(header))
    total = distribution.get("_total", {})
    row_total = sum(total.values())
    print(
        f"{'TOTAL':<22}"
        + "".join(f"{total.get(cls, 0):>8}" for cls in ALL_CLASSES)
        + f"{row_total:>10}"
    )


def analyse(data_root: Path, split: str = "train") -> dict:
    """Run the full distribution analysis and save results.

    Args:
        data_root: Root of the RDD2022 dataset.
        split: Split to analyse.

    Returns:
        Full result dict written to ``logs/class_distribution.json``.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Analysing class distribution in '{split}' split …\n")
    distribution = count_distribution(data_root, split)
    cls_weights = _compute_cls_weights(distribution["_total"])

    result = {
        "split": split,
        "class_names": CLASS_NAMES,
        "distribution": distribution,
        "cls_weights": cls_weights,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    _print_table(distribution)
    print(f"\nClass weights (inverse-frequency, mean-normalised): {cls_weights}")
    print(f"\nSaved to: {OUTPUT_FILE}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse class distribution in RDD2022 YOLO labels."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root directory (default: RDD_DATA_ROOT from .env).",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test"],
        help="Which split to analyse (default: train).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    root = _data_root(args.data_root)
    analyse(root, args.split)
