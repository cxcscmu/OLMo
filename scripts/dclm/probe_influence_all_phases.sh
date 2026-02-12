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

# Base directory for checkpoints (relative path)
BASE_DIR="out/OLMo-300M/dclm_2.3B_organic+recycled_repetition"

# Array of checkpoints to process
# Generate checkpoints for epochs 12-20 (2184 steps per epoch)
CHECKPOINTS=()
for epoch in 11; do
    step=$((2184 * epoch))
    CHECKPOINTS+=("step${step}-unsharded")
done

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

    CHECKPOINT_PATH="${BASE_DIR}/${CHECKPOINT}"

    # Check if checkpoint exists (use LOCAL_ROOT for verification)
    if [ ! -d "${LOCAL_ROOT}/${CHECKPOINT_PATH}" ]; then
        echo "Warning: Checkpoint directory not found: ${LOCAL_ROOT}/${CHECKPOINT_PATH}"
        echo "Skipping..."
        continue
    fi

    # Run probe_influence.sh with relative checkpoint path as argument
    echo "Running probe_influence.sh..."
    bash scripts/dclm/probe_influence.sh "${CHECKPOINT_PATH}"

    echo ""
    echo "Completed: ${CHECKPOINT}"
    echo "=================================================="
done

echo ""
echo "=================================================="
echo "All phases processed!"
echo "=================================================="
