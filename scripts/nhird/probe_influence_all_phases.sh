#!/bin/bash
#SBATCH --job-name=probe_influence_all_phases
#SBATCH --partition=cx-hyper-p
#SBATCH --output=runs/probe_influence_all_phases_%j.out
#SBATCH --error=runs/probe_influence_all_phases_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=1536G
#SBATCH --time=2-00:00:00

# Launch probe_influence for multiple phases
set -euo pipefail

source .env

# Base directory for checkpoints
BASE_DIR="${LOCAL_ROOT}/out/OLMo-2-0425-1B_stage1/step1907359-vocab_expansion-midtrain"

# Array of checkpoints to process
CHECKPOINTS=(
    "phase3_dry/step3000-unsharded"
    "phase4_dry/step4000-unsharded"
    "phase5_dry/step5000-unsharded"
    "phase6_dry/step6000-unsharded"
)

echo "=================================================="
echo "Running Probe Influence for Multiple Phases"
echo "=================================================="
echo "Base directory: ${BASE_DIR}"
echo "Number of checkpoints: ${#CHECKPOINTS[@]}"
echo "=================================================="

# Process each checkpoint
for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    echo ""
    echo "=================================================="
    echo "Processing checkpoint: ${CHECKPOINT}"
    echo "=================================================="

    export CHECKPOINT_DIR="${BASE_DIR}/${CHECKPOINT}"

    # Check if checkpoint exists
    if [ ! -d "${CHECKPOINT_DIR}" ]; then
        echo "Warning: Checkpoint directory not found: ${CHECKPOINT_DIR}"
        echo "Skipping..."
        continue
    fi

    # Run probe_influence_v2.sh
    echo "Running probe_influence_v2.sh..."
    bash scripts/nhird/probe_influence_v2.sh

    echo ""
    echo "Completed: ${CHECKPOINT}"
    echo "=================================================="
done

echo ""
echo "=================================================="
echo "All phases processed!"
echo "=================================================="
