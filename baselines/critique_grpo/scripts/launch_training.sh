#!/bin/bash
# Launch Critique-GRPO baseline training on VERL
#
# Reference: Zhang et al. "Critique-GRPO: Advancing LLM Reasoning with
#   Natural Language and Numerical Feedback." arXiv:2506.03106.
#
# Usage:
#   bash baselines/critique_grpo/scripts/launch_training.sh
#
# Override any config value via CLI args, e.g.:
#   bash launch_training.sh trainer.total_epochs=1

set -x
ulimit -n 65535

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scpo

cd "$SCPO_DIR"

export VLLM_USE_V1=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="${SCPO_DIR}:$PYTHONPATH"

python3 -c "import baselines.critique_grpo" || { echo "Failed to import baselines.critique_grpo"; exit 1; }

CONFIG_PATH="${SCPO_DIR}/baselines/critique_grpo/config"

python3 -m baselines.critique_grpo.scripts.main_critique_grpo \
    --config-path="$CONFIG_PATH" \
    --config-name='critique_grpo_math500' \
    "$@"
