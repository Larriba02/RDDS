"""
Run once after cloning and installing requirements.
Configures Ultralytics settings for the RDDS project.
"""
from pathlib import Path
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

print("Ultralytics settings configured correctly.")
