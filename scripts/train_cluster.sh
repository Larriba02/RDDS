#!/bin/bash
# =============================================================================
# scripts/train_cluster.sh
# SLURM job script for RDDS Phase 1 training on the A100 cluster.
#
# Usage
# -----
#   # 1-epoch smoke test (run this FIRST to validate the cluster node)
#   sbatch --export=MODEL=yolo11s,SAMPLE_RATIO=1.0,EPOCHS=1,BATCH=32,PATIENCE=1,DATA_ROOT=/path/to/rdd2022,SMOKE_TEST=1 scripts/train_cluster.sh
#
#   # YOLO11s full run
#   sbatch --export=MODEL=yolo11s,SAMPLE_RATIO=1.0,EPOCHS=100,BATCH=32,PATIENCE=20,DATA_ROOT=/path/to/rdd2022 scripts/train_cluster.sh
#
#   # YOLO11m full run (after YOLO11s confirms the pipeline)
#   sbatch --export=MODEL=yolo11m,SAMPLE_RATIO=1.0,EPOCHS=100,BATCH=32,PATIENCE=20,DATA_ROOT=/path/to/rdd2022 scripts/train_cluster.sh
#
# Environment variables (all required via --export or already set):
#   MODEL         yolo11s | yolo11m
#   SAMPLE_RATIO  Fraction of train pool to use  (1.0 for Phase 1)
#   EPOCHS        Hard epoch cap  (1 for smoke test, 100 for Phase 1)
#   BATCH         Batch size      (32 for A100)
#   PATIENCE      Early-stopping patience  (1 for smoke test, 20 for Phase 1)
#   DATA_ROOT     Absolute path to the RDD2022 dataset root on the cluster node
#   SMOKE_TEST    Set to 1 to override DATA_ROOT with the tiny synthetic dataset
#
# Cluster-specific lines are marked  <<< EDIT FOR YOUR CLUSTER >>>
# =============================================================================

# --- SLURM directives --------------------------------------------------------
#SBATCH --job-name=rdds_train
#SBATCH --output=logs/slurm_%j.out      # stdout  (logs/ must exist before sbatch)
#SBATCH --error=logs/slurm_%j.err       # stderr
#SBATCH --partition=gpu                  # <<< EDIT FOR YOUR CLUSTER >>>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # <<< EDIT FOR YOUR CLUSTER >>>
#SBATCH --gres=gpu:a100:1               # <<< EDIT FOR YOUR CLUSTER >>>
#SBATCH --mem=40G                        # <<< EDIT FOR YOUR CLUSTER >>>
#SBATCH --time=24:00:00                  # 24 h wall-clock limit

set -euo pipefail

# --- Defaults (override via --export) ----------------------------------------
MODEL="${MODEL:-yolo11s}"
SAMPLE_RATIO="${SAMPLE_RATIO:-1.0}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-32}"
PATIENCE="${PATIENCE:-20}"
SMOKE_TEST="${SMOKE_TEST:-0}"

# --- Resolve data root -------------------------------------------------------
if [[ "${SMOKE_TEST}" == "1" ]]; then
    # Use the synthetic mini-dataset bundled in the repo
    DATA_ROOT="$(pwd)/tests/data/tiny_rdd2022"
    echo "[smoke-test] Using tiny dataset at ${DATA_ROOT}"
else
    if [[ -z "${DATA_ROOT:-}" ]]; then
        echo "ERROR: DATA_ROOT is not set. Pass it via --export DATA_ROOT=..." >&2
        exit 1
    fi
fi

# --- Module loads ------------------------------------------------------------
# Load CUDA and Python. Exact module names depend on the cluster.
# <<< EDIT FOR YOUR CLUSTER >>>
# module purge
# module load cuda/12.4
# module load python/3.12

# --- Navigate to repo root ---------------------------------------------------
# The script is submitted from the repo root, so $(pwd) is already correct.
REPO_ROOT="$(pwd)"
echo "Repo root: ${REPO_ROOT}"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      ${SLURMD_NODENAME:-$(hostname)}"

# --- Activate virtual environment --------------------------------------------
VENV="${REPO_ROOT}/.venv"
if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "ERROR: venv not found at ${VENV}. Run setup.py on this node first." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

python --version
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"

# --- Ensure logs/ directory exists -------------------------------------------
mkdir -p logs

# --- Export environment variables for sub-scripts ----------------------------
export RDD_DATA_ROOT="${DATA_ROOT}"

# Smoke test: always regenerate splits and ingest (tiny dataset, fast).
# Real runs: only regenerate if splits.json is missing.
#   MongoDB images_metadata must be populated before training.  If it is
#   already populated from the laptop run (47k docs), skip ingest to save time.
#   If MongoDB is empty (first time on this cluster), run ingest too.
if [[ "${SMOKE_TEST}" == "1" ]]; then
    echo "--- [smoke-test] Running split.py ---"
    python -m src.data.split
    echo "--- [smoke-test] Running ingest.py ---"
    python -m src.data.ingest
else
    if [[ ! -f logs/splits.json ]]; then
        echo "--- splits.json not found, regenerating ---"
        python -m src.data.split
    else
        echo "--- splits.json found, skipping split.py ---"
    fi

    # Check MongoDB images_metadata count. Use count_documents with a limit
    # to short-circuit quickly (avoids full collection scan on Atlas free tier).
    MONGO_COUNT=$(python -c "
from src.db.connection import get_db
try:
    n = get_db()['images_metadata'].count_documents({}, limit=40001)
    print(n)
except Exception:
    print(0)
" 2>&1 | tail -1 || echo 0)

    echo "MongoDB images_metadata count: ${MONGO_COUNT}"
    if [[ "${MONGO_COUNT}" -lt 40000 ]]; then
        echo "--- MongoDB not populated (expected ~47k), running ingest.py ---"
        python -m src.data.ingest
    else
        echo "--- MongoDB already populated (${MONGO_COUNT} docs), skipping ingest.py ---"
    fi
fi

# --- Verify MongoDB connectivity before starting a long job ------------------
echo "--- Testing MongoDB connection ---"
python -m src.db.test_connection

# --- Main training run -------------------------------------------------------
echo "--- Starting training ---"
echo "  model=${MODEL}  sample_ratio=${SAMPLE_RATIO}  epochs=${EPOCHS}  batch=${BATCH}  patience=${PATIENCE}"

python -m src.training.train \
    --model        "${MODEL}" \
    --sample-ratio "${SAMPLE_RATIO}" \
    --epochs       "${EPOCHS}" \
    --batch        "${BATCH}" \
    --patience     "${PATIENCE}"

echo "--- Training complete ---"
