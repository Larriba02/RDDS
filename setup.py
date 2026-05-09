"""
RDDS — Project Setup Script
Run once after cloning the repository (inside an activated Python 3.12 venv).

    py -3.12 -m venv .venv
    .venv\\Scripts\\activate      # Windows
    python setup.py
"""

import subprocess
import sys
from pathlib import Path

REQUIRED_PYTHON = (3, 12)
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu124"


def check_python_version():
    major, minor = sys.version_info[:2]
    if (major, minor) < REQUIRED_PYTHON:
        print(
            f"\nERROR: Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ required, "
            f"but you are running {major}.{minor}.\n"
            f"Create the venv with: py -3.12 -m venv .venv\n"
        )
        sys.exit(1)
    print(f"Python {major}.{minor}  OK")


def _has_nvidia_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_torch():
    print("\n--- Installing PyTorch ---")
    if _has_nvidia_gpu():
        print(f"  NVIDIA GPU detected — installing torch+cu124")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision",
            "--index-url", TORCH_CUDA_INDEX,
        ])
    else:
        print("  No NVIDIA GPU detected — installing torch (CPU)")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision",
        ])
    print("PyTorch installed.")


def install_dependencies():
    print("\n--- Installing dependencies ---")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
    ])
    print("Dependencies installed.")


def configure_env():
    print("\n--- Environment configuration ---")
    print("Enter your credentials. Press Enter to skip optional fields.\n")

    credentials = {
        "MONGO_URI": "MongoDB Atlas connection string (shared by M)",
        "BACKBLAZE_KEY_ID": "Backblaze B2 Key ID (shared by M)",
        "BACKBLAZE_APP_KEY": "Backblaze B2 Application Key (shared by M)",
        "BACKBLAZE_BUCKET": "Backblaze bucket name (shared by M)",
        "BACKBLAZE_ENDPOINT": "Backblaze B2 endpoint URL (shared by M, e.g. https://s3.eu-central-003.backblazeb2.com)",
    }

    values = {}
    for key, description in credentials.items():
        value = input(f"  {key} [{description}]: ").strip()
        values[key] = value if value else f"<{key.lower()}>"

    values["RANDOM_SEED"] = "42"
    values["SAMPLE_RATIO"] = "0.10"
    rdd_data_root = Path(__file__).parent / "data" / "rdd2022"
    rdd_data_root.mkdir(parents=True, exist_ok=True)
    values["RDD_DATA_ROOT"] = str(rdd_data_root)

    env_path = Path(".env")
    if env_path.exists():
        print("\nWARNING: .env already exists. Overwrite? [y/N] ", end="", flush=True)
        if input().strip().lower() != "y":
            print("Aborted. Existing .env kept.")
            return

    with open(env_path, "w", encoding="utf-8") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")

    print("\n.env created.")
    print("  NOTE: Set RDD_DATA_ROOT in .env manually when you download the dataset in Step 2.")


def configure_ultralytics():
    print("\n--- Configuring Ultralytics ---")
    from ultralytics import settings

    project_root = Path(__file__).parent
    settings.update({
        "datasets_dir": str(project_root / "datasets"),
        "weights_dir":  str(project_root / "checkpoints"),
        "runs_dir":     str(project_root / "runs"),
        "sync":        False,
        "clearml":     False,
        "comet":       False,
        "dvc":         False,
        "neptune":     False,
        "raytune":     False,
        "tensorboard": False,
        "wandb":       False,
        "mlflow":      True,
    })
    print("Ultralytics settings configured.")


def verify():
    print("\n--- Verifying installation ---")
    errors = []
    for module in ("torch", "ultralytics", "pymongo", "mlflow"):
        try:
            __import__(module)
            print(f"  {module:<14} OK")
        except ImportError as e:
            print(f"  {module:<14} ERROR: {e}")
            errors.append(module)

    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"  CUDA available  {'YES — ' + torch.version.cuda if cuda_ok else 'NO (CPU only)'}")

    if errors:
        print(f"\nSetup failed: missing modules {errors}")
        sys.exit(1)


def run_smoke_test():
    print("\n--- Optional smoke test ---")
    print("  Runs Step 1 (MongoDB) + Step 2 (data pipeline) to verify the setup.")
    print("  Requires MONGO_URI to be set in .env.\n")
    answer = input("  Run smoke test now? [y/N] ").strip().lower()
    if answer != "y":
        print("  Skipped. Run manually: python tests/smoke_all.py --steps 1 2")
        return
    print()
    result = subprocess.run(
        [sys.executable, "tests/smoke_all.py", "--steps", "1", "2"],
    )
    if result.returncode != 0:
        print("\nSmoke test reported failures — check output above.")
    else:
        print("\nSmoke test passed.")


def main():
    print("=" * 50)
    print("  RDDS — Project Setup")
    print("  Road Damage Detection System · Group 3 · UFV")
    print("=" * 50)

    check_python_version()
    install_torch()
    install_dependencies()
    configure_env()
    configure_ultralytics()
    verify()
    run_smoke_test()

    print("\n" + "=" * 50)
    print("  Setup complete. You are ready to work.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
