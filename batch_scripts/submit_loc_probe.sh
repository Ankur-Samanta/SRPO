#!/bin/bash
# Submit the localization-prompt probe to SLURM.
# Runs offline vLLM inference over the saved wrong rollouts in
# training/tests/loc_prompt_examples.json.
#
# Usage:
#   bash batch_scripts/submit_loc_probe.sh                         # defaults: OLMo-3-7B-Instruct, all variants
#   bash batch_scripts/submit_loc_probe.sh --local                 # run inline (on a GPU node)
#   MODEL=allenai/OLMo-3-7B-Instruct bash batch_scripts/submit_loc_probe.sh

set -euo pipefail

SRPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_PATH="/home/${USER}/miniconda3"
CONDA_ENV="srpo"

MODEL="${MODEL:-allenai/OLMo-3-7B-Instruct}"
VARIANTS="${VARIANTS:-l2_default l2_terse l2_selfcheck}"
# Default out path / job name tag derive from the model's short name.
MODEL_TAG_DEFAULT="$(basename "${MODEL}" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g')"
TAG="${TAG:-${MODEL_TAG_DEFAULT}}"
OUT="${OUT:-${SRPO_DIR}/training/tests/loc_probe_${TAG}.jsonl}"
JOB_NAME="loc_probe_${TAG}"

CMD="python ${SRPO_DIR}/training/tests/probe_localization_prompts.py \
    --model ${MODEL} \
    --variants ${VARIANTS} \
    --out ${OUT}"

if [[ "${1:-}" == "--local" ]]; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    export TRANSFORMERS_OFFLINE=1
    cd "${SRPO_DIR}"
    exec ${CMD}
fi

mkdir -p "${SRPO_DIR}/batch_scripts/logs"

sbatch --partition="${SLURM_PARTITION:-q1}" \
       --nodes=1 \
       --gpus-per-node=1 \
       --cpus-per-gpu=10 \
       --time=1:00:00 \
       --job-name="${JOB_NAME}" \
       --output="${SRPO_DIR}/batch_scripts/logs/${JOB_NAME}.out" \
       --error="${SRPO_DIR}/batch_scripts/logs/${JOB_NAME}.err" \
       --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && export TRANSFORMERS_OFFLINE=1 && cd ${SRPO_DIR} && ${CMD}'"

echo "Submitted: ${JOB_NAME}"
echo "Output: ${OUT}"
echo "Logs:   ${SRPO_DIR}/batch_scripts/logs/${JOB_NAME}.{out,err}"
