#!/bin/bash
# ICS inference eval — runs the full self-correction pipeline at eval time.
#
# Unlike submit_eval_baselines.sh (which uses thought_agent at temp=0),
# this script runs thought_ics_agent with:
#   - temperature=0.5 for all phases (initial chain, localization, correction)
#   - self-verification gating (model decides when to correct, no oracle)
#   - autonomy_level=2 (no ground truth in localization prompt)
#   - up to 10 correction iterations per trigger
#   - rollout.n=11 (1 fresh + up to 10 corrections per slot)
#
# The oracle answer is still used for REWARD computation (we need it to
# evaluate whether any trajectory succeeded), but NOT for gating ICS.
#
# Usage:
#   ./submit_eval_ics.sh <model> <dataset>     # single job
#   ./submit_eval_ics.sh <model> all           # all datasets for one model
#   ./submit_eval_ics.sh all all               # all jobs
#   ./submit_eval_ics.sh list                  # show available models

set -e

SCPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH="${SCPO_DIR}/training/scripts/launch_training.sh"
CONDA_PATH="/home/${USER}/miniconda3"
CONDA_ENV="scpo"
EVAL_DATA="${HOME}/data/rlhf/eval"

mkdir -p "${SCPO_DIR}/batch_scripts/logs"

OLMO7B="allenai/OLMo-3-7B-Instruct"

# Datasets (csqa and gpqa excluded — same as cross-eval tables)
ALL_DATASETS=(numinamath_olympiads acereason_math sciknoweval_chemistry sciknoweval_physics sciknoweval_biology sciknoweval_materials hmmt_nov_2025 math_level5 strategyqa csqa)

# ─── Shared overrides ─────────────────────────────────────────────────────────

SHARED_OVERRIDES=(
    actor_rollout_ref.model.external_lib=training
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=true
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    trainer.n_gpus_per_node=2
    trainer.val_only=true
    trainer.val_before_train=true
    'trainer.logger=["console"]'
    # ICS agent loop
    actor_rollout_ref.rollout.agent.default_agent_loop=thought_ics_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path=training/config/thought_agent_config_ics_eval.yaml
    # temp=0.7 for all phases; n=10 total rollouts (1 fresh + up to 9 corrections)
    actor_rollout_ref.rollout.temperature=0.7
    actor_rollout_ref.rollout.n=10
)

OLMO_OVERRIDES=(
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
)

# ─── Model registry ───────────────────────────────────────────────────────────

declare -A MODEL_HF
declare -A MODEL_CKPT
declare -A MODEL_IS_OLMO

# Base models
MODEL_HF[base_olmo7b]="$OLMO7B"; MODEL_CKPT[base_olmo7b]=""; MODEL_IS_OLMO[base_olmo7b]=true

# TGRPO (trained on oly)
MODEL_HF[tgrpo_oly_olmo7b]="$OLMO7B"; MODEL_CKPT[tgrpo_oly_olmo7b]="${SCPO_DIR}/checkpoints/thought_grpo/numina_oly_olmo7b_thought/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[tgrpo_oly_olmo7b]=true

# SCGRPO (trained on oly)

# SCGRPO-rand (random localization, trained on oly)

# Seed 0 reruns (with loss/branch dumps + variance/pass@k metrics)
MODEL_HF[tgrpo_oly_olmo7b_s0]="$OLMO7B"; MODEL_CKPT[tgrpo_oly_olmo7b_s0]="${SCPO_DIR}/checkpoints/thought_grpo/numina_oly_olmo7b_thought_s0/global_step_24/actor/lora_adapter"; MODEL_IS_OLMO[tgrpo_oly_olmo7b_s0]=true

ALL_MODELS=(
 base_olmo7b
 tgrpo_oly_olmo7b
 tgrpo_oly_olmo7b_s0
)

ALL_S0_MODELS=(
 tgrpo_oly_olmo7b_s0
)

# ─── Run a single eval ────────────────────────────────────────────────────────

run_eval() {
    local model_key=$1
    local dataset=$2

    local base_model="${MODEL_HF[$model_key]}"
    local ckpt_dir="${MODEL_CKPT[$model_key]}"
    local is_olmo="${MODEL_IS_OLMO[$model_key]}"
    local test_data="${EVAL_DATA}/${dataset}.parquet"

    if [ ! -f "$test_data" ]; then
        echo "ERROR: test data not found: $test_data"
        exit 1
    fi

    local overrides=(
        "${SHARED_OVERRIDES[@]}"
        actor_rollout_ref.model.path="$base_model"
        data.val_files="$test_data"
        data.train_files="$test_data"
        trainer.experiment_name="eval_ics_${model_key}_${dataset}"
        trainer.project_name=eval_ics
        trainer.resume_mode=disable
        "trainer.default_local_dir='checkpoints/eval_ics/eval_ics_${model_key}_${dataset}'"
    )

    if [ -n "$ckpt_dir" ]; then
        overrides+=(actor_rollout_ref.model.lora_adapter_path="${ckpt_dir}")
    fi

    if [ "$is_olmo" = "true" ]; then
        overrides+=("${OLMO_OVERRIDES[@]}")
    fi

    bash "$LAUNCH" "${overrides[@]}"
}

# ─── SLURM submission ─────────────────────────────────────────────────────────

submit() {
    local job_name=$1
    local model_key=$2
    local dataset=$3

    sbatch --partition=q1 \
           --nodes=1 \
           --gpus-per-node=2 \
           --cpus-per-gpu=10 \
           --time=8:00:00 \
           --job-name="${job_name}" \
           --output="batch_scripts/logs/${job_name}.out" \
           --error="batch_scripts/logs/${job_name}.err" \
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${SCPO_DIR} && bash ${SCPO_DIR}/batch_scripts/submit_eval_ics.sh ${model_key} ${dataset} --local'"

    echo "Submitted: ${job_name}"
}

# ─── Entrypoint ───────────────────────────────────────────────────────────────

MODEL_KEY=${1:?Usage: $0 <model> <dataset> [--local]  (run '$0 list' for help)}

if [ "$MODEL_KEY" = "list" ]; then
    echo "Models:"
    for m in "${ALL_MODELS[@]}"; do
        ckpt="${MODEL_CKPT[$m]}"
        echo "  $m  (${MODEL_HF[$m]}${ckpt:+ + adapter})"
    done
    echo ""
    echo "Datasets: ${ALL_DATASETS[*]}"
    exit 0
fi

DATASET=${2:?Usage: $0 <model> <dataset> [--local]}
MODE=${3:-slurm}

expand_models()   { case $1 in all) echo "${ALL_MODELS[@]}" ;; all_s0) echo "${ALL_S0_MODELS[@]}" ;; *) echo "$1" ;; esac; }
expand_datasets() { case $1 in all) echo "${ALL_DATASETS[@]}" ;; *) echo "$1" ;; esac; }

MODELS=$(expand_models "$MODEL_KEY")
DATASETS=$(expand_datasets "$DATASET")

for m in $MODELS; do
    if [ -z "${MODEL_HF[$m]}" ]; then
        echo "Unknown model: $m (run '$0 list')"
        exit 1
    fi
    for d in $DATASETS; do
        job_name="eval_ics_${m}_${d}"
        if [ "$MODE" = "--local" ]; then
            echo "=== Running ICS eval: ${m} on ${d} ==="
            run_eval "$m" "$d"
        else
            submit "$job_name" "$m" "$d"
        fi
    done
done
