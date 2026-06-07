#!/bin/bash
# Compute actual per-thought ‖∇_θ L_thought‖ over LoRA params.
# Runs scripts/grad_norm_actual.py against the v2 grad-dump output.
#
# Usage:
#   ./batch_scripts/submit_grad_norm_actual.sh             # full run
#   ./batch_scripts/submit_grad_norm_actual.sh --local     # run on this node
#   ./batch_scripts/submit_grad_norm_actual.sh --test      # SLURM, --limit 1

set -e

SRPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_PATH="/home/${USER}/miniconda3"
CONDA_ENV="srpo"
JOB_NAME="grad_norm_actual"

mkdir -p "${SRPO_DIR}/batch_scripts/logs"

MODE="${1:-slurm}"
EXTRA_ARGS=""
if [[ "$MODE" == "--test" ]]; then
    EXTRA_ARGS="--limit 1"
    JOB_NAME="grad_norm_actual_test"
fi

if [[ "$MODE" == "--local" ]]; then
    set -x
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    cd "${SRPO_DIR}"
    export PYTHONPATH="${SRPO_DIR}:${PYTHONPATH}"
    python3 scripts/grad_norm_actual.py ${EXTRA_ARGS}
else
    sbatch --partition="${SLURM_PARTITION:-q1}" \
           --nodes=1 \
           --gpus-per-node=1 \
           --cpus-per-gpu=10 \
           --time=1:30:00 \
           --job-name="${JOB_NAME}" \
           --output="${SRPO_DIR}/batch_scripts/logs/${JOB_NAME}.out" \
           --error="${SRPO_DIR}/batch_scripts/logs/${JOB_NAME}.err" \
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${SRPO_DIR} && export PYTHONPATH=${SRPO_DIR}:\${PYTHONPATH} && python3 scripts/grad_norm_actual.py ${EXTRA_ARGS}'"
    echo "Submitted: ${JOB_NAME}"
    echo "  out: batch_scripts/logs/${JOB_NAME}.out"
    echo "  err: batch_scripts/logs/${JOB_NAME}.err"
fi
