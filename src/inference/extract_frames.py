"""
src/inference/extract_frames.py
--------------------------------
Extract frames from a video file at 1 fps using OpenCV.

Frames are saved as ``frame_NNNNNN.jpg`` (zero-padded to 6 digits) in the
specified output directory. This module does NOT write to MongoDB — frames
are inputs to the prediction pipeline, not predictions.

Usage:
    python -m src.inference.extract_frames --video path/to/video.mp4
    python -m src.inference.extract_frames --video path/to/video.mp4 --output-dir outputs/frames/

Output:
    - outputs/frames/frame_000001.jpg
    - outputs/frames/frame_000002.jpg
    - ...
    - Prints total frames extracted, video duration, and output directory.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


DEFAULT_OUTPUT_DIR = Path("outputs") / "frames"
TARGET_FPS = 1  # Extract one frame per second.


def extract_frames(video_path: str | Path, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> int:
    """Extract frames at 1 fps from a video file and save as JPEG.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where extracted frames will be saved.
            Created if it does not exist.

    Returns:
        Number of frames extracted.

    Raises:
        FileNotFoundError: If ``video_path`` does not exist.
        RuntimeError: If OpenCV cannot open the video file.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        cap.release()
        raise RuntimeError(
            f"Could not read FPS from video (got {video_fps}). "
            "The file may be corrupt or in an unsupported format."
        )

    duration_seconds = total_video_frames / video_fps
    # How many source frames to skip between each extracted frame.
    # For a 30 fps video and TARGET_FPS=1: sample every 30th frame.
    frame_interval = max(1, round(video_fps / TARGET_FPS))

    print(f"Video:     {video_path}")
    print(f"Source FPS: {video_fps:.2f}  |  Duration: {duration_seconds:.1f}s")
    print(f"Frame interval: every {frame_interval} source frames (target {TARGET_FPS} fps)")
    print(f"Output directory: {output_dir}")

    extracted = 0
    source_frame_idx = 0
    start_time = time.monotonic()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if source_frame_idx % frame_interval == 0:
            extracted += 1
            filename = output_dir / f"frame_{extracted:06d}.jpg"
            cv2.imwrite(str(filename), frame)

        source_frame_idx += 1

    cap.release()
    elapsed = time.monotonic() - start_time

    print(f"\nExtracted {extracted} frames in {elapsed:.1f}s")
    print(f"Frames saved to: {output_dir.resolve()}")

    return extracted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames from a video file at 1 fps.\n"
            "Frames are saved as frame_NNNNNN.jpg in the output directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file (mp4, avi, mov, etc.).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save extracted frames. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    extract_frames(video_path=args.video, output_dir=args.output_dir)
