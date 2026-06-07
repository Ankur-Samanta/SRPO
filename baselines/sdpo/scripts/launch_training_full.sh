#!/bin/bash
# Launch SDPO-FULL (paper-faithful) baseline training on VERL
#
# Reference: Hübotter et al. "Reinforcement Learning via Self-Distillation."
#   arXiv:2601.20802. Code: https://github.com/lasgroup/SDPO
#
# Differences from launch_training.sh (which runs the "sdpo" token-KL variant):
#   - uses sdpo_full_math500.yaml (full-logit + EMA teacher + JSD + top-k=100)
#
# Usage:
#   bash baselines/sdpo/scripts/launch_training_full.sh
#
# Override any config value via CLI args, e.g.:
#   bash launch_training_full.sh algorithm.self_distillation.distillation_topk=50
#   bash launch_training_full.sh algorithm.self_distillation.alpha=0.0

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

python3 -c "import baselines.sdpo" || { echo "Failed to import baselines.sdpo"; exit 1; }

CONFIG_PATH="${SCPO_DIR}/baselines/sdpo/config"

python3 -m baselines.sdpo.scripts.main_sdpo \
    --config-path="$CONFIG_PATH" \
    --config-name='sdpo_full_math500' \
    '+actor_rollout_ref.actor.self_distillation=${algorithm.self_distillation}' \
    "$@"
