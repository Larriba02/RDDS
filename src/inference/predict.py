"""
src/inference/predict.py
------------------------
Run YOLO inference on an image, folder of images, or video file and produce
annotated output with bounding boxes drawn.

Checkpoint resolution order (same as evaluate.py):
    1. ``--model`` local path override (useful before A100 cluster runs).
    2. runs/train/<run_id>/weights/best.pt
    3. Any runs/train/*/weights/best.pt whose dirname starts with run_id.
    4. Download best.pt from the Backblaze B2 URL in MongoDB.

MongoDB writes:
    One document per image is written to the ``predictions`` collection.
    Idempotent: if a document with the same ``image_id`` + ``model_version``
    already exists, the DB write is skipped (the annotated image is still saved).

Class colours (BGR convention for OpenCV):
    D00 → blue   (255, 0,   0)
    D10 → green  (  0, 255, 0)
    D20 → yellow (  0, 255, 255)
    D40 → red    (  0,   0, 255)

Usage:
    # Use production model from MongoDB:
    python -m src.inference.predict --source path/to/image.jpg
    python -m src.inference.predict --source path/to/folder/

    # Full video workflow (extract frames first, then predict):
    python -m src.inference.extract_frames --video path/to/video.mp4 --output-dir outputs/frames/
    python -m src.inference.predict --source outputs/frames/

    # Override with a local .pt file:
    python -m src.inference.predict --source path/to/image.jpg --model runs/train/.../best.pt

    # Custom output directory:
    python -m src.inference.predict --source path/to/folder/ --output-dir outputs/predictions/

    # Skip MongoDB write (dry run):
    python -m src.inference.predict --source path/to/image.jpg --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

from src.db.connection import get_db

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONF = 0.5
DEFAULT_IOU = 0.5
DEFAULT_OUTPUT_DIR = Path("outputs") / "predictions"
RUNS_DIR = Path("runs") / "train"

CLASS_NAMES = ["D00", "D10", "D20", "D40"]

# BGR colours per damage class.
CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "D00": (255, 0, 0),    # blue
    "D10": (0, 255, 0),    # green
    "D20": (0, 255, 255),  # yellow
    "D40": (0, 0, 255),    # red
}
DEFAULT_COLOUR = (128, 128, 128)  # grey for unknown classes.

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


# ---------------------------------------------------------------------------
# Helpers — experiment / checkpoint (mirrors evaluate.py pattern)
# ---------------------------------------------------------------------------


def _get_production_experiment() -> dict[str, Any]:
    """Return the current production experiment document from MongoDB.

    Returns:
        MongoDB document with ``is_production=True``.

    Raises:
        RuntimeError: If no production model exists in MongoDB.
    """
    db = get_db()
    doc = db["experiments"].find_one({"is_production": True})
    if doc is None:
        raise RuntimeError(
            "No production model found (is_production=True). "
            "Train a model first or pass --model with a local .pt path."
        )
    return doc


def _resolve_checkpoint(exp: dict[str, Any]) -> Path:
    """Resolve best.pt for a given experiment document.

    Resolution order:
        1. runs/train/<run_id>/weights/best.pt (exact match).
        2. Any runs/train/*/weights/best.pt whose dir starts with run_id.
        3. Download from the Backblaze B2 URL stored in MongoDB.

    Args:
        exp: MongoDB experiment document.

    Returns:
        Path to best.pt.

    Raises:
        RuntimeError: If no checkpoint can be found.
    """
    run_id = exp["run_id"]

    direct = RUNS_DIR / run_id / "weights" / "best.pt"
    if direct.exists():
        print(f"  Checkpoint: {direct}")
        return direct

    if RUNS_DIR.exists():
        for candidate in sorted(RUNS_DIR.iterdir()):
            if candidate.name.startswith(run_id):
                ckpt = candidate / "weights" / "best.pt"
                if ckpt.exists():
                    print(f"  Checkpoint (fuzzy match): {ckpt}")
                    return ckpt

    b2_url = exp.get("checkpoints", {}).get("best_pt")
    if b2_url:
        dl_dir = Path("checkpoints") / run_id
        dl_dir.mkdir(parents=True, exist_ok=True)
        dest = dl_dir / "best.pt"
        if not dest.exists():
            print(f"  Downloading checkpoint from B2: {b2_url}")
            _download_from_b2(b2_url, dest)
        else:
            print(f"  Using cached download: {dest}")
        return dest

    raise RuntimeError(
        f"Cannot find best.pt for run_id='{run_id}'. "
        "Check local runs/ directory or MongoDB checkpoints.best_pt."
    )


def _download_from_b2(url: str, dest: Path) -> None:
    """Download a Backblaze B2 object to ``dest`` using authenticated S3 API.

    The bucket is private, so anonymous HTTP requests return 401. Credentials
    are read from the same .env variables used by upload_checkpoint.py
    (BACKBLAZE_KEY_ID, BACKBLAZE_APP_KEY, optional BACKBLAZE_ENDPOINT).

    Args:
        url: Full S3-style URL stored in ``experiments.checkpoints.best_pt``,
             e.g. ``https://s3.eu-central-003.backblazeb2.com/<bucket>/<key>``.
        dest: Local destination path.

    Raises:
        RuntimeError: If credentials are missing or download fails.
    """
    import boto3
    from botocore.config import Config

    parsed = urllib.parse.urlparse(url)
    endpoint = f"{parsed.scheme}://{parsed.netloc}"
    # Path is "/<bucket>/<key>" — split into bucket and key.
    parts = parsed.path.lstrip("/").split("/", 1)
    if len(parts) != 2:
        raise RuntimeError(f"Cannot parse bucket/key from URL: {url}")
    bucket, key = parts

    key_id = os.getenv("BACKBLAZE_KEY_ID")
    app_key = os.getenv("BACKBLAZE_APP_KEY")
    if not key_id or not app_key:
        raise RuntimeError(
            "Cannot download checkpoint from B2: BACKBLAZE_KEY_ID and "
            "BACKBLAZE_APP_KEY must be set in .env."
        )

    client = boto3.client(
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
    try:
        client.download_file(bucket, key, str(dest))
    except Exception as exc:
        raise RuntimeError(f"Failed to download {url} from B2: {exc}") from exc


# ---------------------------------------------------------------------------
# Helpers — image_id
# ---------------------------------------------------------------------------


def _image_id(source_path: str) -> str:
    """Compute MD5 hash of the source path string.

    Consistent with the convention in ``src/data/split.py``.

    Args:
        source_path: File path string (as provided by the caller).

    Returns:
        32-character hex MD5 digest.
    """
    return hashlib.md5(source_path.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Helpers — collect source paths
# ---------------------------------------------------------------------------


def _collect_image_paths(source: str | Path) -> list[Path]:
    """Resolve all image paths from a source argument.

    Args:
        source: A single image file, a folder of images, or a video file.
            For video inputs, the caller should first run ``extract_frames``
            and then pass the frames folder.

    Returns:
        Sorted list of image Paths.

    Raises:
        FileNotFoundError: If ``source`` does not exist.
        ValueError: If ``source`` is a video file (unsupported directly here).
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if source.is_file():
        suffix = source.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            raise ValueError(
                f"'{source}' is a video file. "
                "Extract frames first with:\n"
                "  python -m src.inference.extract_frames --video <path> --output-dir <frames_dir>\n"
                "then run predict on the frames folder."
            )
        if suffix not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported image formats: {sorted(IMAGE_EXTENSIONS)}"
            )
        return [source]

    if source.is_dir():
        paths = sorted(
            p for p in source.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not paths:
            raise ValueError(f"No image files found in directory: {source}")
        return paths

    raise ValueError(f"Source must be a file or directory: {source}")


# ---------------------------------------------------------------------------
# Helpers — drawing
# ---------------------------------------------------------------------------


def _draw_detections(
    image: Any,  # numpy ndarray
    detections: list[dict[str, Any]],
) -> Any:
    """Draw bounding boxes and labels on an image (in-place).

    Args:
        image: BGR numpy array read by cv2.
        detections: List of detection dicts with keys ``label``, ``bbox``
            (``[xmin, ymin, xmax, ymax]``), and ``confidence``.

    Returns:
        Modified image array.
    """
    for det in detections:
        label = det["label"]
        bbox = det["bbox"]
        confidence = det["confidence"]
        colour = CLASS_COLOURS.get(label, DEFAULT_COLOUR)

        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Draw bounding box.
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colour, thickness=2)

        # Build label text.
        text = f"{label} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Draw filled rectangle behind text for readability.
        label_y = max(ymin, text_h + baseline + 2)
        cv2.rectangle(
            image,
            (xmin, label_y - text_h - baseline - 2),
            (xmin + text_w, label_y),
            colour,
            thickness=cv2.FILLED,
        )

        # Draw text in white for contrast.
        cv2.putText(
            image,
            text,
            (xmin, label_y - baseline),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return image


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------


def predict(
    source: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_path: str | Path | None = None,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    device: str = "0",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run YOLO inference and write results to MongoDB.

    Args:
        source: Path to a single image, a folder of images, or a video file.
            Video files are rejected with an informative error — use
            ``extract_frames`` first.
        output_dir: Directory where annotated images are saved.
        model_path: Optional local path to a ``.pt`` file. When provided,
            bypasses the MongoDB production-model lookup.
        conf: Confidence threshold for inference.
        iou: IoU threshold for NMS.
        device: CUDA device string (e.g. ``"0"`` or ``"cpu"``).
        dry_run: If True, skip MongoDB writes.

    Returns:
        Summary dict with keys ``total_images``, ``total_detections``,
        ``elapsed_seconds``, and ``output_dir``.

    Raises:
        EnvironmentError: If MongoDB is unavailable and ``--model`` is not set.
        FileNotFoundError: If ``source`` does not exist.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RDDS — Inference")
    print(f"  source={source}, conf={conf}, iou={iou}, device={device}")
    print("=" * 60)

    # --- Resolve model ---
    run_id: str | None = None
    model_version: str | None = None

    if model_path is not None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"--model path does not exist: {model_path}")
        checkpoint = model_path
        print(f"\n[1/4] Using local model override: {checkpoint}")
        # Try to get model_version from MongoDB if available, but don't fail.
        try:
            exp = _get_production_experiment()
            run_id = exp.get("run_id")
            model_version = exp.get("model_version")
            print(f"      (MongoDB production run: {run_id}, version: {model_version})")
        except Exception:
            run_id = "local_override"
            model_version = "local"
            print("      (MongoDB unavailable — run_id='local_override')")
    else:
        print("\n[1/4] Fetching production model from MongoDB ...")
        exp = _get_production_experiment()
        run_id = exp["run_id"]
        model_version = exp.get("model_version", "unknown")
        print(f"  Production model: {run_id}  (version: {model_version})")
        checkpoint = _resolve_checkpoint(exp)

    print(f"\n[2/4] Loading model from {checkpoint} ...")
    model = YOLO(str(checkpoint))

    # --- Collect image paths ---
    print(f"\n[3/4] Collecting images from {source} ...")
    image_paths = _collect_image_paths(source)
    print(f"  Found {len(image_paths)} image(s).")

    # --- Run inference ---
    print(f"\n[4/4] Running inference (conf={conf}, iou={iou}) ...")
    start_time = time.monotonic()

    total_detections = 0
    db_writes = 0
    db_skips = 0

    db = get_db() if not dry_run else None
    predictions_col = db["predictions"] if db is not None else None

    for img_path in image_paths:
        img_path_str = str(img_path)
        img_id = _image_id(img_path_str)

        # Run YOLO inference.
        results = model.predict(
            source=img_path_str,
            imgsz=640,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
            save=False,
        )

        # Parse detections.
        detections: list[dict[str, Any]] = []
        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_idx = int(box.cls[0])
                    cls_name = (
                        CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
                    )
                    bbox_coords = [round(v) for v in box.xyxy[0].tolist()]
                    detections.append({
                        "label": cls_name,
                        "bbox": bbox_coords,
                        "confidence": round(float(box.conf[0]), 4),
                    })

        total_detections += len(detections)

        # --- Annotate and save image ---
        img = cv2.imread(img_path_str)
        if img is not None:
            annotated = _draw_detections(img, detections)
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), annotated)
        else:
            print(f"  [warn] Could not read image for annotation: {img_path_str}")

        # --- MongoDB write ---
        if dry_run:
            continue

        # Idempotency check: skip if same image_id + model_version already exists.
        existing = predictions_col.find_one(  # type: ignore[union-attr]
            {"image_id": img_id, "model_version": model_version},
            {"_id": 1},
        )
        if existing is not None:
            db_skips += 1
            continue

        pred_doc: dict[str, Any] = {
            "pred_id": str(uuid.uuid4()),
            "image_id": img_id,
            "model_version": model_version,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_path": img_path_str,
            "detections": detections,
        }
        predictions_col.insert_one(pred_doc)  # type: ignore[union-attr]
        db_writes += 1

    elapsed = time.monotonic() - start_time

    # --- Summary ---
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"  Images processed:   {len(image_paths)}")
    print(f"  Total detections:   {total_detections}")
    print(f"  Elapsed:            {elapsed:.1f}s")
    print(f"  Output directory:   {output_dir.resolve()}")
    if not dry_run:
        print(f"  MongoDB writes:     {db_writes}")
        print(f"  MongoDB skips (dup): {db_skips}")
    else:
        print("  MongoDB writes:     skipped (--dry-run)")
    print("=" * 60)

    return {
        "total_images": len(image_paths),
        "total_detections": total_detections,
        "elapsed_seconds": round(elapsed, 2),
        "output_dir": str(output_dir.resolve()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RDDS inference — run YOLO on images and write annotated output.\n\n"
            "For video input, first extract frames:\n"
            "  python -m src.inference.extract_frames --video path/to/video.mp4\n"
            "then pass the frames folder as --source."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Single image path, folder of images, or (unsupported directly) video file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for annotated output images. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Local path to a .pt checkpoint file. "
            "Bypasses the MongoDB production-model lookup. "
            "Useful when the cluster has not run yet and best.pt is available locally."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"Confidence threshold. Default: {DEFAULT_CONF}",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help=f"IoU threshold for NMS. Default: {DEFAULT_IOU}",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device (e.g. '0') or 'cpu'. Default: '0'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip MongoDB writes. Annotated images are still saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    predict(
        source=args.source,
        output_dir=args.output_dir,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        dry_run=args.dry_run,
    )
