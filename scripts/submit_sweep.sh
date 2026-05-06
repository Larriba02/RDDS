#!/bin/bash
# =============================================================================
# scripts/submit_sweep.sh
# Submit multiple YOLO11m hyperparameter configs as sequential SLURM jobs.
#
# Usage
# -----
#   bash scripts/submit_sweep.sh
#
# Each config runs only after the previous one completes successfully
# (SLURM --dependency=afterok). If a job fails, subsequent jobs are cancelled.
#
# How to fill in configs
# -----------------------
# 1. Wait for J to finish Step 3.5 and report the best hyperparameter configs.
# 2. Edit the CONFIGS array below — one entry per run.
#    Format: "KEY=val KEY=val ..."  (space-separated, all on one line)
#    Available keys: LR0  LRF  COS_LR(0|1)  OPTIMIZER  EPOCHS  PATIENCE  BATCH
# 3. Set DATA_ROOT, DEVICE, and CACHE for your cluster.
# 4. Run:  bash scripts/submit_sweep.sh
#
# Example entry from J's Step 3.5 results:
#   "LR0=0.01 LRF=0.01 COS_LR=1 OPTIMIZER=auto EPOCHS=100 PATIENCE=20"
# =============================================================================

set -euo pipefail

# --- Cluster settings — edit these before submitting ------------------------
DATA_ROOT="/path/to/rdd2022"    # <<< absolute path on the cluster node
DEVICE="0"                       # <<< "0" for 1 GPU, "0,1,2,3" for 4-GPU DDP
CACHE="disk"
BATCH="32"                       # <<< scale with GPU count (32 per GPU)
SAMPLE_RATIO="1.0"

# --- Configs from J's Step 3.5 results — fill in after J reports ------------
# Add or remove entries as needed. Runs in array order, one at a time.
CONFIGS=(
    # Config A — winner of Step 3.5 Round 3
    # "LR0=0.01 LRF=0.01 COS_LR=1 OPTIMIZER=auto EPOCHS=100 PATIENCE=20"

    # Config B — runner-up of Step 3.5
    # "LR0=0.001 LRF=0.1 COS_LR=0 OPTIMIZER=AdamW EPOCHS=100 PATIENCE=20"
)

# --- Validate ---------------------------------------------------------------
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "ERROR: CONFIGS array is empty. Fill in J's hyperparameter results first." >&2
    exit 1
fi

if [[ "${DATA_ROOT}" == "/path/to/rdd2022" ]]; then
    echo "ERROR: DATA_ROOT is still the placeholder. Set it to the real dataset path." >&2
    exit 1
fi

mkdir -p logs

# --- Submit jobs sequentially via SLURM dependency -------------------------
PREV_JOB_ID=""
SUBMITTED=()

echo "Submitting ${#CONFIGS[@]} YOLO11m config(s) sequentially..."
echo ""

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    CONFIG_NUM=$((i + 1))

    # Parse key=val pairs from the config string into --export format
    EXPORT_VARS="MODEL=yolo11m,SAMPLE_RATIO=${SAMPLE_RATIO},BATCH=${BATCH},DEVICE=${DEVICE},CACHE=${CACHE},DATA_ROOT=${DATA_ROOT}"
    for pair in ${CONFIG}; do
        EXPORT_VARS="${EXPORT_VARS},${pair}"
    done

    # Build sbatch command
    SBATCH_CMD="sbatch --job-name=rdds_m_cfg${CONFIG_NUM} --output=logs/slurm_cfg${CONFIG_NUM}_%j.out --error=logs/slurm_cfg${CONFIG_NUM}_%j.err"

    if [[ -n "${PREV_JOB_ID}" ]]; then
        SBATCH_CMD="${SBATCH_CMD} --dependency=afterok:${PREV_JOB_ID}"
    fi

    SBATCH_CMD="${SBATCH_CMD} --export=${EXPORT_VARS} scripts/train_cluster.sh"

    echo "Config ${CONFIG_NUM}: ${CONFIG}"
    JOB_ID=$(eval "${SBATCH_CMD}" | awk '{print $NF}')
    echo "  -> Job ID: ${JOB_ID}"
    SUBMITTED+=("${JOB_ID}")
    PREV_JOB_ID="${JOB_ID}"
    echo ""
done

# --- Summary ----------------------------------------------------------------
echo "================================================"
echo "Sweep submitted: ${#SUBMITTED[@]} job(s)"
echo ""
echo "Job chain:"
for i in "${!SUBMITTED[@]}"; do
    CONFIG_NUM=$((i + 1))
    echo "  Config ${CONFIG_NUM} -> Job ${SUBMITTED[$i]}"
done
echo ""
echo "Monitor all jobs:"
echo "  squeue -u \$USER"
echo ""
echo "Follow logs in real time (replace JOB_ID):"
for i in "${!SUBMITTED[@]}"; do
    CONFIG_NUM=$((i + 1))
    echo "  tail -f logs/slurm_cfg${CONFIG_NUM}_${SUBMITTED[$i]}.out"
done
echo ""
echo "Cancel entire sweep:"
echo "  scancel ${SUBMITTED[*]}"
echo "================================================"
