#!/bin/bash
# Launch thought-level GRPO training on VERL
#
# Usage:
#   bash training/scripts/launch_training.sh
#
# Override any config value via CLI args, e.g.:
#   bash launch_training.sh trainer.total_epochs=1 actor_rollout_ref.rollout.n=2

set -x
ulimit -n 65535

# Resolve project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Activate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate srpo

# Ensure CWD is project root (relative paths in config resolve against CWD)
cd "$SRPO_DIR"

# Enable vLLM v1
export VLLM_USE_V1=1

# Fix tensordict C extension linking
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Make our module importable
export PYTHONPATH="${SRPO_DIR}:$PYTHONPATH"

# Unique NCCL port per job to avoid conflicts when multiple jobs share a node
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-$$} % 1000))

# Import the module to trigger @register before training starts
python3 -c "import training" || { echo "Failed to import training"; exit 1; }

CONFIG_PATH="${SRPO_DIR}/training/config"

python3 -m training.scripts.main_ppo_wrapper \
    --config-path="$CONFIG_PATH" \
    --config-name='thought_grpo_math500' \
    "$@"

# Save experiment artifacts (config, metrics, checkpoint) to experiments/
python3 "${SRPO_DIR}/training/scripts/save_experiment.py" "$@"
