"""
RDDS — Project Setup Script
Run once after cloning the repository.
Sets up dependencies, environment variables, and Ultralytics configuration.
"""

import subprocess
import sys
from pathlib import Path


def install_dependencies():
    print("\n--- Installing dependencies ---")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Dependencies installed.")


def configure_env():
    print("\n--- Environment configuration ---")
    print("Enter your credentials. Press Enter to skip optional fields.\n")

    # Only ask for credentials that the user must provide
    credentials = {
        "MONGO_URI": "MongoDB Atlas connection string (shared by M)",
        "BACKBLAZE_KEY_ID": "Backblaze B2 Key ID (shared by M)",
        "BACKBLAZE_APP_KEY": "Backblaze B2 Application Key (shared by M)",
        "BACKBLAZE_BUCKET": "Backblaze bucket name (shared by M)",
    }

    values = {}
    for key, description in credentials.items():
        value = input(f"  {key} [{description}]: ").strip()
        values[key] = value if value else f"<{key.lower()}>"

    # Fixed values — not configurable
    values["RANDOM_SEED"] = "42"
    values["SAMPLE_RATIO"] = "0.10"
    values["RDD_DATA_ROOT"] = "<set this when dataset is downloaded in Step 2>"

    env_path = Path(".env")
    with open(env_path, "w") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")

    print(f"\n.env created.")
    print("  NOTE: Set RDD_DATA_ROOT in .env manually when you download the dataset in Step 2.")


def configure_ultralytics():
    print("\n--- Configuring Ultralytics ---")
    from ultralytics import settings

    project_root = Path(__file__).parent
    settings.update({
        "weights_dir": str(project_root / "checkpoints"),
        "runs_dir":    str(project_root / "runs"),
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
    try:
        import ultralytics
        import pymongo
        import mlflow
        print("  ultralytics  OK")
        print("  pymongo      OK")
        print("  mlflow       OK")
    except ImportError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)


def main():
    print("=" * 50)
    print("  RDDS — Project Setup")
    print("  Road Damage Detection System · Group 3 · UFV")
    print("=" * 50)

    install_dependencies()
    configure_env()
    configure_ultralytics()
    verify()

    print("\n" + "=" * 50)
    print("  Setup complete. You are ready to work.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()