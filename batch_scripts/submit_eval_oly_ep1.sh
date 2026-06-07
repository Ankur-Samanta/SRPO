#!/bin/bash
# Submit full-scale eval for oly-trained ep1 (no-KL) checkpoints:
#
# All checkpoints are at global_step_12 (1 epoch on numinamath_olympiads).
#
# Usage:
#   ./submit_eval_oly_ep1.sh <model_key> <dataset>   # single job
#   ./submit_eval_oly_ep1.sh <model_key> all          # all datasets
#   ./submit_eval_oly_ep1.sh all all                   # all jobs
#   ./submit_eval_oly_ep1.sh list                      # show available jobs

set -e

SCPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH="${SCPO_DIR}/training/scripts/launch_training.sh"
CONDA_PATH="/home/${USER}/miniconda3"
CONDA_ENV="scpo"
EVAL_DATA="${HOME}/data/rlhf/eval"
CKPT_ROOT="${SCPO_DIR}/checkpoints"

mkdir -p "${SCPO_DIR}/batch_scripts/logs"

QWEN14B="Qwen/Qwen2.5-14B-Instruct"
OLMO7B="allenai/OLMo-3-7B-Instruct"

ALL_DATASETS=(numinamath_olympiads acereason_math sciknoweval_chemistry sciknoweval_physics sciknoweval_biology sciknoweval_materials hmmt_nov_2025 math_level5 strategyqa csqa)

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
)

OLMO_OVERRIDES=(
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
)

declare -A MODEL_HF
declare -A MODEL_CKPT
declare -A MODEL_IS_OLMO
declare -A MODEL_N_GPUS   # defaults to 2 if unset

# tgrpo oly ep1 (no KL)

MODEL_HF[tgrpo_oly_ep1_olmo7b]="$OLMO7B"
MODEL_CKPT[tgrpo_oly_ep1_olmo7b]="${CKPT_ROOT}/thought_grpo/numina_oly_olmo7b_thought_ep1/global_step_12/actor/lora_adapter"
MODEL_IS_OLMO[tgrpo_oly_ep1_olmo7b]=true

# scgrpo oly ep1 (no KL)


# Seed 420 variants (Qwen7B) — seed-ablation runs, ep2 / step_24



# Seed 420 variants (OLMo7B) — seed-ablation runs, ep2 / step_24
MODEL_HF[tgrpo_oly_s420_olmo7b]="$OLMO7B"
MODEL_CKPT[tgrpo_oly_s420_olmo7b]="${CKPT_ROOT}/thought_grpo/numina_oly_olmo7b_thought_s420/global_step_24/actor/lora_adapter"
MODEL_IS_OLMO[tgrpo_oly_s420_olmo7b]=true



# score oly ep1

MODEL_HF[score_oly_ep1_olmo7b]="$OLMO7B"
MODEL_CKPT[score_oly_ep1_olmo7b]="${CKPT_ROOT}/score_baseline/numina_oly_olmo7b_score_ep1/global_step_12/actor/lora_adapter"
MODEL_IS_OLMO[score_oly_ep1_olmo7b]=true



# crgrpo oly ep1

MODEL_HF[crgrpo_oly_ep1_olmo7b]="$OLMO7B"
MODEL_CKPT[crgrpo_oly_ep1_olmo7b]="${CKPT_ROOT}/critique_grpo_baseline/numina_oly_olmo7b_cgrpo_ep1/global_step_12/actor/lora_adapter"
MODEL_IS_OLMO[crgrpo_oly_ep1_olmo7b]=true

# tgrpo oly ep2 qwen14b (2 epochs, global_step_24)
MODEL_HF[tgrpo_oly_ep2_qwen14b]="$QWEN14B"
MODEL_CKPT[tgrpo_oly_ep2_qwen14b]="${CKPT_ROOT}/thought_grpo/numina_oly_qwen14b_thought/global_step_24/actor/lora_adapter"
MODEL_IS_OLMO[tgrpo_oly_ep2_qwen14b]=false
MODEL_N_GPUS[tgrpo_oly_ep2_qwen14b]=4

# scgrpo oly ep2 qwen14b (2 epochs, global_step_24)


# crgrpo oly ep2 qwen14b (2 epochs, global_step_24)
MODEL_HF[crgrpo_oly_ep2_qwen14b]="$QWEN14B"
MODEL_CKPT[crgrpo_oly_ep2_qwen14b]="${CKPT_ROOT}/critique_grpo_baseline/numina_oly_qwen14b_cgrpo/global_step_24/actor/lora_adapter"
MODEL_IS_OLMO[crgrpo_oly_ep2_qwen14b]=false
MODEL_N_GPUS[crgrpo_oly_ep2_qwen14b]=4

# score oly ep2 qwen14b (2 epochs, global_step_24)
MODEL_HF[score_oly_ep2_qwen14b]="$QWEN14B"
MODEL_CKPT[score_oly_ep2_qwen14b]="${CKPT_ROOT}/score_baseline/numina_oly_qwen14b_score/global_step_24/actor/lora_adapter"
MODEL_IS_OLMO[score_oly_ep2_qwen14b]=false
MODEL_N_GPUS[score_oly_ep2_qwen14b]=4

ALL_MODELS=(
 tgrpo_oly_ep1_olmo7b
 score_oly_ep1_olmo7b
 crgrpo_oly_ep1_olmo7b
)

run_eval() {
    local model_key=$1
    local dataset=$2
    local base_model="${MODEL_HF[$model_key]}"
    local ckpt_dir="${MODEL_CKPT[$model_key]}"
    local is_olmo="${MODEL_IS_OLMO[$model_key]}"
    local n_gpus="${MODEL_N_GPUS[$model_key]:-2}"
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
        trainer.experiment_name="eval_${model_key}_${dataset}"
        trainer.project_name=eval_oly_ep1
        actor_rollout_ref.rollout.agent.default_agent_loop=thought_agent
        trainer.resume_mode=disable
        "trainer.default_local_dir='checkpoints/eval_oly_ep1/eval_${model_key}_${dataset}'"
        actor_rollout_ref.model.lora_adapter_path="${ckpt_dir}"
        trainer.n_gpus_per_node="${n_gpus}"
        actor_rollout_ref.rollout.tensor_model_parallel_size="${n_gpus}"
    )

    if [ "$is_olmo" = "true" ]; then
        overrides+=("${OLMO_OVERRIDES[@]}")
    fi

    # Small datasets (< 32 rows) need a smaller train_batch_size to avoid empty dataloader
    local n_rows
    n_rows=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$test_data')))" 2>/dev/null || echo 999)
    if [ "$n_rows" -lt 32 ] 2>/dev/null; then
        overrides+=(data.train_batch_size=16)
    fi

    bash "$LAUNCH" "${overrides[@]}"
}

submit() {
    local job_name=$1
    local model_key=$2
    local dataset=$3
    local n_gpus="${MODEL_N_GPUS[$model_key]:-2}"

    sbatch --partition=q1 \
           --nodes=1 \
           --gpus-per-node="${n_gpus}" \
           --cpus-per-gpu=10 \
           --time=4:00:00 \
           --job-name="${job_name}" \
           --output="batch_scripts/logs/${job_name}.out" \
           --error="batch_scripts/logs/${job_name}.err" \
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${SCPO_DIR} && bash ${SCPO_DIR}/batch_scripts/submit_eval_oly_ep1.sh ${model_key} ${dataset} --local'"

    echo "Submitted: ${job_name}"
}

MODEL_KEY=${1:?Usage: $0 <model> <dataset> [--local]  (run '$0 list' for help)}

if [ "$MODEL_KEY" = "list" ]; then
    echo "Models:"
    for m in "${ALL_MODELS[@]}"; do echo "  $m"; done
    echo "Datasets:"
    for d in "${ALL_DATASETS[@]}"; do echo "  $d"; done
    exit 0
fi

DATASET=${2:?Usage: $0 <model> <dataset> [--local]}
MODE=${3:-slurm}

MODELS=$([ "$MODEL_KEY" = "all" ] && echo "${ALL_MODELS[*]}" || echo "$MODEL_KEY")
DATASETS=$([ "$DATASET" = "all" ] && echo "${ALL_DATASETS[*]}" || echo "$DATASET")

for m in $MODELS; do
    if [ -z "${MODEL_HF[$m]}" ]; then
        echo "Unknown model: $m (run '$0 list')"
        exit 1
    fi
    for d in $DATASETS; do
        job_name="eval_ep1_${m}_${d}"
        if [ "$MODE" = "--local" ]; then
            echo "=== Running: ${m} on ${d} ==="
            run_eval "$m" "$d"
        else
            submit "$job_name" "$m" "$d"
        fi
    done
done
