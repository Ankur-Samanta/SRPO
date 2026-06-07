#!/bin/bash
# Launch SCoRe baseline training on VERL
#
# Reference: Kumar et al. "Training Language Models to Self-Correct via
#   Reinforcement Learning." ICLR 2025. arXiv:2409.12917.
#
# Usage:
#   bash baselines/score/scripts/launch_training.sh
#
# Override any config value via CLI args, e.g.:
#   bash launch_training.sh trainer.total_epochs=1

set -x
ulimit -n 65535

# Resolve project root (three levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Activate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate scpo

# Ensure CWD is project root (relative paths in config resolve against CWD)
cd "$SCPO_DIR"

# Enable vLLM v1
export VLLM_USE_V1=1

# Fix tensordict C extension linking
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Make our module importable
export PYTHONPATH="${SCPO_DIR}:$PYTHONPATH"

# Validate import (separate process — catches import errors early)
python3 -c "import baselines.score" || { echo "Failed to import baselines.score"; exit 1; }

CONFIG_PATH="${SCPO_DIR}/baselines/score/config"

# Use custom entrypoint that imports baselines.score in the DRIVER process.
# This is critical: verl's external_lib only imports in worker processes,
# but compute_advantage (which needs the "score" estimator + monkey-patch)
# runs in the driver process.
python3 -m baselines.score.scripts.main_score \
    --config-path="$CONFIG_PATH" \
    --config-name='score_math500' \
    "$@"
