#!/bin/bash
# Launch SPO-tree baseline training on VERL
#
# Reference: SPO (arXiv:2505.23564), §5 — tree-based credit assignment.
#
# Usage:
#   bash baselines/spo/scripts/launch_tree_training.sh
#
# Override any config value via CLI args, e.g.:
#   bash launch_tree_training.sh trainer.total_epochs=1

set -x
ulimit -n 65535

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate srpo

cd "$SRPO_DIR"

export VLLM_USE_V1=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="${SRPO_DIR}:$PYTHONPATH"

python3 -c "import baselines.spo" || { echo "Failed to import baselines.spo"; exit 1; }

CONFIG_PATH="${SRPO_DIR}/baselines/spo/config"

# main_spo already imports baselines.spo in the driver, which registers
# both the chain and tree estimators/agent loops. Only the config name differs.
python3 -m baselines.spo.scripts.main_spo \
    --config-path="$CONFIG_PATH" \
    --config-name='spo_tree_math500' \
    "$@"
