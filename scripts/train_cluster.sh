#!/bin/bash
# =============================================================================
# scripts/train_cluster.sh
# SLURM job script for RDDS Phase 1 training on the A100 cluster.
#
# Usage
# -----
#   # 1-epoch smoke test (run this FIRST to validate the cluster node)
#   sbatch --export=MODEL=yolo11s,SAMPLE_RATIO=1.0,EPOCHS=1,BATCH=32,PATIENCE=1,\
#     DATA_ROOT=/path/to/rdd2022,DEVICE=0,CACHE=disk,SMOKE_TEST=1 \
#     scripts/train_cluster.sh
#
#   # YOLO11s full run — single GPU
#   sbatch --export=MODEL=yolo11s,SAMPLE_RATIO=1.0,EPOCHS=100,BATCH=32,PATIENCE=20,\
#     DATA_ROOT=/path/to/rdd2022,DEVICE=0,CACHE=disk \
#     scripts/train_cluster.sh
#
#   # YOLO11s full run — 4-GPU DDP (effective batch = BATCH, split across GPUs)
#   sbatch --export=MODEL=yolo11s,SAMPLE_RATIO=1.0,EPOCHS=100,BATCH=128,PATIENCE=20,\
#     DATA_ROOT=/path/to/rdd2022,DEVICE=0,1,2,3,CACHE=disk \
#     scripts/train_cluster.sh
#
#   # YOLO11m full run — 4-GPU DDP
#   sbatch --export=MODEL=yolo11m,SAMPLE_RATIO=1.0,EPOCHS=100,BATCH=128,PATIENCE=20,\
#     DATA_ROOT=/path/to/rdd2022,DEVICE=0,1,2,3,CACHE=disk \
#     scripts/train_cluster.sh
#
# Parallelism strategy with two clusters
# ----------------------------------------
#   Launch both jobs at the same time from different cluster sessions:
#     Cluster A: MODEL=yolo11s, DEVICE=0,1,2,3 (or however many GPUs you get)
#     Cluster B: MODEL=yolo11m, DEVICE=0,1,2,3
#   Both write to the shared MongoDB Atlas — check the dashboard between runs.
#
# Batch size guidance
# --------------------
#   BATCH is the TOTAL batch across all GPUs. Ultralytics divides it evenly.
#   Examples:
#     1 GPU  → BATCH=32   (32 per GPU)
#     2 GPUs → BATCH=64   (32 per GPU)
#     4 GPUs → BATCH=128  (32 per GPU)
#   If GPUs available is unknown at submission time, use BATCH=32 (safe for 1 GPU)
#   and increase once you know the allocation.
#
# Environment variables (all override-able via --export):
#   MODEL         yolo11s | yolo11m
#   SAMPLE_RATIO  Fraction of train pool to use  (1.0 for Phase 1)
#   EPOCHS        Hard epoch cap  (1 for smoke test, 100 for Phase 1)
#   BATCH         Total batch size across all GPUs
#   PATIENCE      Early-stopping patience  (1 for smoke test, 20 for Phase 1)
#   DEVICE        CUDA device(s): "0" for 1 GPU, "0,1,2,3" for 4-GPU DDP
#   CACHE         "disk" | "ram" | "False"  (disk recommended for cluster)
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
#SBATCH --cpus-per-task=8               # <<< EDIT FOR YOUR CLUSTER >>> (8 per GPU is a good baseline)
#SBATCH --gres=gpu:a100:1               # <<< EDIT FOR YOUR CLUSTER >>> (increase to :4 for 4-GPU DDP)
#SBATCH --mem=40G                        # <<< EDIT FOR YOUR CLUSTER >>> (scale with GPU count: 40G×N)
#SBATCH --time=24:00:00                  # 24 h wall-clock limit

set -euo pipefail

# --- Defaults (override via --export) ----------------------------------------
MODEL="${MODEL:-yolo11s}"
SAMPLE_RATIO="${SAMPLE_RATIO:-1.0}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-32}"
PATIENCE="${PATIENCE:-20}"
DEVICE="${DEVICE:-0}"
CACHE="${CACHE:-disk}"
SMOKE_TEST="${SMOKE_TEST:-0}"

# --- Resolve data root -------------------------------------------------------
if [[ "${SMOKE_TEST}" == "1" ]]; then
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
# module load cuda/12.6
# module load python/3.12

# --- Navigate to repo root ---------------------------------------------------
REPO_ROOT="$(pwd)"
echo "Repo root: ${REPO_ROOT}"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      ${SLURMD_NODENAME:-$(hostname)}"
echo "Device(s): ${DEVICE}"
echo "Cache:     ${CACHE}"

# --- Activate virtual environment --------------------------------------------
VENV="${REPO_ROOT}/.venv"
if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "ERROR: venv not found at ${VENV}. Run setup.py on this node first." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

python --version
python -c "
import torch
n = torch.cuda.device_count()
print(f'torch {torch.__version__} | CUDA {torch.cuda.is_available()} | {n} GPU(s) visible')
for i in range(n):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# --- Ensure logs/ directory exists -------------------------------------------
mkdir -p logs

# --- Export environment variables for sub-scripts ----------------------------
export RDD_DATA_ROOT="${DATA_ROOT}"

# Smoke test: always regenerate splits and ingest (tiny dataset, fast).
# Real runs: only regenerate if splits.json is missing.
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
echo "  model=${MODEL}  sample_ratio=${SAMPLE_RATIO}  epochs=${EPOCHS}"
echo "  batch=${BATCH}  patience=${PATIENCE}  device=${DEVICE}  cache=${CACHE}"

python -m src.training.train \
    --model        "${MODEL}" \
    --sample-ratio "${SAMPLE_RATIO}" \
    --epochs       "${EPOCHS}" \
    --batch        "${BATCH}" \
    --patience     "${PATIENCE}" \
    --device       "${DEVICE}" \
    --cache        "${CACHE}"

echo "--- Training complete ---"
