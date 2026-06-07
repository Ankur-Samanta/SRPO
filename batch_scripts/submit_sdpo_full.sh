#!/bin/bash
# Submit SDPO-FULL (paper-faithful) baseline training jobs (verl pipeline)
#
# Differences from submit_sdpo.sh:
#   - LAUNCH points at launch_training_full.sh -> --config-name='sdpo_full_math500'
#   - Job names suffixed _sdpo_full (experiment / checkpoint dirs)
# Everything else (dataset paths, epochs, GPU layout, TP, micro_batch, offload,
# seed overrides, SLURM knobs) mirrors submit_sdpo.sh exactly so an oly run with
# sdpo_full gets the same non-training setup as the existing sdpo runs.
#
# Reference: Hübotter et al. "Reinforcement Learning via Self-Distillation."
#   arXiv:2601.20802.
#
# Usage:
#   ./submit_sdpo_full.sh <job>          # submit to SLURM
#   ./submit_sdpo_full.sh <job> --local  # run locally (set CUDA_VISIBLE_DEVICES first)
#
# Jobs (olympiads @ 2 epochs is the focus):
#   olmo7b_numina_oly        OLMo 3 7B on NuminaMath Olympiads    (2 GPUs, TP=2)
#   all                      Submit all jobs (2 epochs)
#   *_ep1                    1-epoch variant (eval every 4 steps)
#   all_ep1                  Submit all 1-epoch jobs
#   <job>_s0 / _s420  Seed variants

set -e

SCPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH="${SCPO_DIR}/baselines/sdpo/scripts/launch_training_full.sh"
CONDA_PATH="/home/${USER}/miniconda3"
CONDA_ENV="scpo"

mkdir -p "${SCPO_DIR}/batch_scripts/logs"

# ─── Shared overrides for 7B models (TP=2, 2 GPUs) ─────────────────────────
SHARED_OVERRIDES=(
    actor_rollout_ref.model.external_lib=baselines.sdpo
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=true
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    trainer.n_gpus_per_node=2
    "trainer.default_local_dir='checkpoints/\${trainer.project_name}/\${trainer.experiment_name}'"
    trainer.total_epochs=2
)

export VERL_LOGGING_LEVEL=INFO

# ─── Generic job builder ───────────────────────────────────────────────────
_run() {
    local model_path=$1; shift
    local experiment_name=$1; shift
    local extra_overrides=("$@")

    local cmd_args=(
        "${SHARED_OVERRIDES[@]}"
        actor_rollout_ref.model.path="$model_path"
        trainer.experiment_name="$experiment_name"
        "${extra_overrides[@]}"
    )

    bash "$LAUNCH" "${cmd_args[@]}"
}

# ─── Model shortcuts ──────────────────────────────────────────────────────
QWEN14B="Qwen/Qwen2.5-14B-Instruct"
OLMO7B="allenai/OLMo-3-7B-Instruct"

# 14B models: TP=4, 4 GPUs, lower vLLM memory + micro batches
# ppo_mini_batch_size=1, distillation_topk=20, alpha=1.0, teacher_update_rate=0.01,
# optim.lr=1e-6) — first attempt OOM'd at step 1 on lm_head logsumexp; same vocab
QWEN14B_OVERRIDES=(
    trainer.n_gpus_per_node=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.calculate_entropy=false
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.ppo_mini_batch_size=1
    algorithm.self_distillation.distillation_topk=20
    algorithm.self_distillation.alpha=1.0
    algorithm.self_distillation.teacher_update_rate=0.01
    actor_rollout_ref.actor.optim.lr=1e-6
)

OLMO_OVERRIDES=(
    # SDPO-full runs student + teacher forward + topk intermediates;
    # micro=2 OOMs at 40GB A100 for 7B. Drop to 1.
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
)

# Small models (≤3B): single GPU, TP=1
SMALL_MODEL_OVERRIDES=(
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    trainer.n_gpus_per_node=1
)

#
# deviations from the primary sdpo.yaml are needed together:
#   teacher_update_rate=0.01  (vs 0.05)   # slower EMA → stronger anchor
#   optim.lr=1e-6            (vs 1e-5)   # smaller steps → less format drift
#   ppo_mini_batch_size=1    (vs 32)     # no gradient accumulation memory pile-up
#   distillation_topk=20     (vs 100)    # smaller topk tensors
#
#   v1 (rate=0.05, lr=1e-5, paper primary) — COLLAPSED epoch 2 (step 17-24):
#     response length 2900→286, num_turns 13→2, score 0.29→0.05, <think> format
#     corrupted. Broken checkpoint at
#   v2 (rate=0.01, lr=1e-5) — tracked v1 identically through step 17; cancelled
#     at step 18. Rate anchoring alone insufficient; per-step lr dominates drift.
#   v3 (rate=0.01, lr=1e-6) — healthy trajectory through step 13, then OOM at
#     step 14 on the teacher forward (24.61 GiB allocation from lm_head with
#     151936 vocab on long reprompts). Never reached collapse window.
#   v4 (v3 + gpu_memory_utilization=0.3 + calculate_entropy=false) — SAME OOM
#     at step 14, same 24.66 GiB allocation. Memory tweaks didn't touch the
#     actual allocation source.
#   v5 (this config) — adds ppo_mini_batch_size=1 + distillation_topk=20 +
#     avoids 8x gradient-accumulation memory; topk=20 shrinks topk tensors 5x.
#
# Olmo-3-7B doesn't need these overrides (format is robust at rate=0.05 lr=1e-5
# and vocab ~100k doesn't pressure lm_head allocations).

# ─── Data overrides ───────────────────────────────────────────────────────
NUMINA_OLYMPIADS_DATA=(
    data.train_files=/home/${USER}/data/rlhf/numinamath_olympiads/train.parquet
    data.val_files=/home/${USER}/data/rlhf/numinamath_olympiads/test.parquet
)

# ─── Job definitions ────────────────────────────────────────────────────────


# --- NuminaMath Olympiads ---

# ─── Seed variants ───────────────────────────────────────────────────────────
SEED_0_OVERRIDES=(actor_rollout_ref.actor.fsdp_config.seed=0   actor_rollout_ref.ref.fsdp_config.seed=0   critic.model.fsdp_config.seed=0)
SEED_420_OVERRIDES=(actor_rollout_ref.actor.fsdp_config.seed=420 actor_rollout_ref.ref.fsdp_config.seed=420 critic.model.fsdp_config.seed=420)
cmd_olmo7b_numina_oly_s0()    { _run "$OLMO7B"  numina_oly_olmo7b_sdpo_full_s0    "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}" ; }
cmd_olmo7b_numina_oly_s420()  { _run "$OLMO7B"  numina_oly_olmo7b_sdpo_full_s420  "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" ; }
cmd_qwen14b_numina_oly()      { _run "$QWEN14B"      numina_oly_qwen14b_sdpo_full      "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" ; }
cmd_qwen14b_numina_oly_s0()   { _run "$QWEN14B"      numina_oly_qwen14b_sdpo_full_s0   "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SEED_0_OVERRIDES[@]}" ; }
cmd_qwen14b_numina_oly_s420() { _run "$QWEN14B"      numina_oly_qwen14b_sdpo_full_s420 "${NUMINA_OLYMPIADS_DATA[@]}" "${QWEN14B_OVERRIDES[@]}" "${SEED_420_OVERRIDES[@]}" ; }
cmd_olmo7b_numina_oly()  { _run "$OLMO7B"  numina_oly_olmo7b_sdpo_full "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" ; }

# ─── 1-epoch variants (eval every 4 steps) ──────────────────────────────────
EP1_OVERRIDES=(
    trainer.total_epochs=1
    trainer.test_freq=4
)


# --- NuminaMath Olympiads ep1 ---
cmd_olmo7b_numina_oly_ep1() { _run "$OLMO7B" numina_oly_olmo7b_sdpo_full_ep1 "${NUMINA_OLYMPIADS_DATA[@]}" "${OLMO_OVERRIDES[@]}" "${EP1_OVERRIDES[@]}" ; }

# ─── SLURM submission helper ────────────────────────────────────────────────

submit() {
    local job_name=$1
    local cmd=$2
    local n_gpus=${3:-2}
    local time_limit=${4:-24:00:00}

    sbatch --partition=q1 \
           --nodes=1 \
           --gpus-per-node=${n_gpus} \
           --cpus-per-gpu=10 \
           --time=${time_limit} \
           --job-name="${job_name}" \
           --output="batch_scripts/logs/${job_name}.out" \
           --error="batch_scripts/logs/${job_name}.err" \
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${SCPO_DIR} && ${cmd}'"

    echo "Submitted: ${job_name}"
}

# ─── Entrypoint ─────────────────────────────────────────────────────────────

# Parse --training_seed from any position in args
TRAINING_SEED=42
POSITIONAL=()
for arg in "$@"; do
    case $arg in
        --training_seed=*) TRAINING_SEED="${arg#*=}" ;;
        --training_seed)   shift_next=true ;;
        *)
            if [ "${shift_next:-}" = true ]; then
                TRAINING_SEED="$arg"
                shift_next=false
            else
                POSITIONAL+=("$arg")
            fi
            ;;
    esac
done

SEED_OVERRIDES=(
    actor_rollout_ref.actor.fsdp_config.seed="${TRAINING_SEED}"
    actor_rollout_ref.ref.fsdp_config.seed="${TRAINING_SEED}"
    critic.model.fsdp_config.seed="${TRAINING_SEED}"
)
SHARED_OVERRIDES+=("${SEED_OVERRIDES[@]}")

JOB=${POSITIONAL[0]:?Usage: $0 <job> [--local] [--training_seed N]  (see header for job list)}
MODE=${POSITIONAL[1]:-slurm}

run_or_submit() {
    local job_name=$1
    local func=$2
    local case_key=${3:-$JOB}
    local n_gpus=${4:-2}
    local time_limit=${5:-24:00:00}

    if [ "$MODE" = "--local" ]; then
        echo "=== Running locally: ${job_name} ==="
        $func
    else
        submit "$job_name" "bash ${SCPO_DIR}/batch_scripts/submit_sdpo_full.sh ${case_key} --local --training_seed=${TRAINING_SEED}" "$n_gpus" "$time_limit"
    fi
}

case $JOB in
    qwen14b_numina_oly)      run_or_submit "sdpo_full_qwen14b_oly"      cmd_qwen14b_numina_oly      qwen14b_numina_oly      4  48:00:00 ;;
    qwen14b_numina_oly_s0)   run_or_submit "sdpo_full_qwen14b_oly_s0"   cmd_qwen14b_numina_oly_s0   qwen14b_numina_oly_s0   4  48:00:00 ;;
    qwen14b_numina_oly_s420) run_or_submit "sdpo_full_qwen14b_oly_s420" cmd_qwen14b_numina_oly_s420 qwen14b_numina_oly_s420 4  48:00:00 ;;
    olmo7b_numina_oly)      run_or_submit "sdpo_full_olmo7b_oly"       cmd_olmo7b_numina_oly ;;

    # ─── Seed variants (NuminaMath Olympiads) ────────────────────────────────
    olmo7b_numina_oly_s0)    run_or_submit "sdpo_full_olmo7b_oly_s0"    cmd_olmo7b_numina_oly_s0 ;;
    olmo7b_numina_oly_s420)  run_or_submit "sdpo_full_olmo7b_oly_s420"  cmd_olmo7b_numina_oly_s420 ;;
    olmo7b_numina_oly_ep1)  run_or_submit "sdpo_full_olmo7b_oly_ep1"  cmd_olmo7b_numina_oly_ep1 ;;
    all)
 for job in olmo7b_numina_oly; do
            run_or_submit "sdpo_full_${job}" "cmd_${job}" "${job}"
        done
        ;;
    all_ep1)
 for job in olmo7b_numina_oly_ep1; do
            run_or_submit "sdpo_full_${job}" "cmd_${job}" "${job}"
        done
        ;;
    *)
        echo "Unknown job: $JOB"
 echo "Available: olmo7b_numina_oly all"
        echo "         + *_ep1 variants, seed variants (_s0/_s420), and all_ep1"
        exit 1
        ;;
esac
