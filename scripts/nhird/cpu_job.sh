#!/bin/bash
#SBATCH --job-name=subsample_nhird
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=runs/subsample_nhird_%j.out
#SBATCH --error=runs/subsample_nhird_%j.err

set -euo pipefail

# Load environment
source .env

echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: 120G"
echo "=================================================="

# Run the cpu script
# python scripts/nhird/prepare_dclm_cuts.py
bash scripts/nhird/select_data.sh

echo "=================================================="
echo "Subsampling complete!"
echo "=================================================="
