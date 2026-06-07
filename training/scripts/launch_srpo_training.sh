#!/bin/bash
# Launch SRPO training on VERL (two-group GRPO: 4 fresh i.i.d. + 4 oracle-gated corrections)
#
# Usage:
#   bash training/scripts/launch_srpo_training.sh
#
# Override any config value via CLI args, e.g.:
#   bash launch_srpo_training.sh trainer.total_epochs=1 actor_rollout_ref.rollout.n=8

set -x
ulimit -n 65535

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate scpo

cd "$SCPO_DIR"

export VLLM_USE_V1=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="${SCPO_DIR}:$PYTHONPATH"
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-$$} % 1000))

# Auto-set branch-dump dir from experiment_name so each run writes ICS data
# (loc prompts, responses, correction outcomes) to its own subdir. Skipped
# if the user already set the var manually.
if [ -z "${SCGRPO_BRANCH_DUMP_DIR:-}" ]; then
    EXP_NAME=$(echo "$@" | grep -oE 'trainer\.experiment_name=[^ ]+' | head -1 | cut -d= -f2)
    if [ -n "$EXP_NAME" ]; then
        export SCGRPO_BRANCH_DUMP_DIR="${SCPO_DIR}/logs/srpo_localizations/${EXP_NAME}"
        echo "[launcher] SCGRPO_BRANCH_DUMP_DIR=$SCGRPO_BRANCH_DUMP_DIR"
    fi
fi

python3 -c "import training" || { echo "Failed to import training"; exit 1; }

CONFIG_PATH="${SCPO_DIR}/training/config"

python3 -m training.scripts.main_ppo_wrapper \
    --config-path="$CONFIG_PATH" \
    --config-name='srpo_math500' \
    "$@"

python3 "${SCPO_DIR}/training/scripts/save_experiment.py" "$@"
