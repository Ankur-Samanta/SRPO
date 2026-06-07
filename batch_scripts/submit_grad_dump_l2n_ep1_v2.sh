#!/bin/bash
# One-shot per-token gradient dump for SRPO-l2n-ep1 *trained policy*.
#
# Approach: original ckpt is LoRA-only (no FSDP shards), so we cannot resume.
# Instead, start a *fresh* training run that loads the trained LoRA adapter as
# init weights, run exactly ONE training step on a small batch, and dump the
# srpo loss tensors per microbatch + branch dumps.
#
# All names/paths are unique — they DO NOT collide with any existing run.
#
# Prerequisite: training/srpo_loss.py patched with the env-gated
# tensor dump (writes torch.save when SRPO_PER_TOKEN_DUMP_DIR is set).
#
# Outputs (all under unique paths):
#   logs/srpo_per_token/grad_dump_l2n_ep1/loss_pid*_call*.pt
#   logs/srpo_localizations/grad_dump_l2n_ep1/branches_pid*.jsonl
#   wandb run under project=srpo_grad_dump
#
# Usage:
#   ./batch_scripts/submit_grad_dump_l2n_ep1_v2.sh             # SLURM
#   ./batch_scripts/submit_grad_dump_l2n_ep1_v2.sh --local     # interactive

set -e

# ─── Paths (all unique to this run; verified no collisions exist) ───────────
SCPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_PATH="/home/${USER}/miniconda3"
CONDA_ENV="scpo"
JOB_NAME="srpo_graddump_l2n_ep1_v2"
EXP_NAME="grad_dump_l2n_ep1_v2"
PROJECT="srpo_grad_dump"

ORIG_LORA="${SCPO_DIR}/checkpoints/srpo/lcb_medium_olmo7b_srpo_l2new_ep1/global_step_11/actor/lora_adapter"
LORA_INIT="${SCPO_DIR}/checkpoints/${PROJECT}/lora_init/lora_adapter"
TRAIN_CKPT_DIR="${SCPO_DIR}/checkpoints/${PROJECT}/${EXP_NAME}"
DUMP_DIR="${SCPO_DIR}/logs/srpo_per_token/${EXP_NAME}"
BRANCH_DIR="${SCPO_DIR}/logs/srpo_localizations/${EXP_NAME}"

mkdir -p "${SCPO_DIR}/batch_scripts/logs"
mkdir -p "${DUMP_DIR}"

# ─── 1. Copy LoRA adapter to an isolated path ───────────────────────────────
# We point lora_adapter_path at this copy, never at the original — so even if
# verl ever wrote to the path during init, the original is physically isolated.
if [[ ! -d "${LORA_INIT}" ]]; then
    if [[ ! -d "${ORIG_LORA}" ]]; then
        echo "ERROR: original LoRA not found at ${ORIG_LORA}" >&2
        exit 1
    fi
    mkdir -p "$(dirname "${LORA_INIT}")"
    cp -r "${ORIG_LORA}" "${LORA_INIT}"
    echo "LoRA init copy: ${LORA_INIT}"
fi

# ─── 2. Hydra overrides ─────────────────────────────────────────────────────
OLMO7B="allenai/OLMo-3-7B-Instruct"

OVERRIDES=(
    # Shared SRPO setup (mirror SHARED_OVERRIDES from submit_srpo.sh)
    actor_rollout_ref.model.external_lib=training
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=true
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    trainer.n_gpus_per_node=2
    actor_rollout_ref.rollout.agent.default_agent_loop=srpo_agent
    actor_rollout_ref.actor.policy_loss.loss_mode=srpo
    # Model + trained LoRA init (NOT a resume — fresh training state)
    actor_rollout_ref.model.path="${OLMO7B}"
    actor_rollout_ref.model.lora_adapter_path="${LORA_INIT}"
    # OLMo micro-batch sizing (mirror OLMO_OVERRIDES)
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
    # LCB-medium data
    "data.train_files=/home/${USER}/data/rlhf/livecodebench_medium/train.parquet"
    "data.val_files=/home/${USER}/data/rlhf/livecodebench_medium/test.parquet"
    # Seed 42 (matches original)
    actor_rollout_ref.actor.fsdp_config.seed=42
    actor_rollout_ref.ref.fsdp_config.seed=42
    critic.model.fsdp_config.seed=42
    # ── Run isolation ─────────────────────────────────────────────────
    "trainer.project_name=${PROJECT}"
    "trainer.experiment_name=${EXP_NAME}"
    "trainer.default_local_dir='checkpoints/\${trainer.project_name}/\${trainer.experiment_name}'"
    trainer.resume_mode=disable           # NOT resuming — starts fresh at step 0
    trainer.total_epochs=999              # let total_training_steps cap us
    trainer.total_training_steps=1        # exactly 1 update step
    trainer.test_freq=999                 # no val
    trainer.save_freq=999                 # no checkpoint writes
    data.train_batch_size=8               # 8 prompts × 8 rollouts = 64 trajectories (≥ ppo_mini_batch_size=8)
)

# ─── 3. Env vars ────────────────────────────────────────────────────────────
export VERL_LOGGING_LEVEL=INFO
export ICS_L2_PROMPT=l2new
export SRPO_PER_TOKEN_DUMP_DIR="${DUMP_DIR}"
export SRPO_PER_TOKEN_DUMP_MAX=200
export SCGRPO_BRANCH_DUMP_DIR="${BRANCH_DIR}"     # explicit (so launcher auto-derive logic is skipped)
export SCGRPO_BRANCH_DUMP_EVERY=1
export SCGRPO_BRANCH_DUMP_MAX=20

# ─── 4. Submit or run locally ───────────────────────────────────────────────
MODE="${1:-slurm}"

if [[ "$MODE" == "--local" ]]; then
    echo "=== Running locally: ${JOB_NAME} ==="
    echo "    LoRA init:   ${LORA_INIT}"
    echo "    train ckpt:  ${TRAIN_CKPT_DIR} (will be created fresh)"
    echo "    loss dump:   ${DUMP_DIR}"
    echo "    branch dump: ${BRANCH_DIR}"

    # Inline launch_srpo_training.sh's env — skip the destructive save_experiment.py post-step.
    set -x
    ulimit -n 65535
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    cd "${SCPO_DIR}"

    export VLLM_USE_V1=1
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
    export PYTHONPATH="${SCPO_DIR}:${PYTHONPATH}"
    export MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-$$} % 1000))

    python3 -c "import training" || { echo "Failed to import training"; exit 1; }

    CONFIG_PATH="${SCPO_DIR}/training/config"
    python3 -m training.scripts.main_ppo_wrapper \
        --config-path="${CONFIG_PATH}" \
        --config-name='srpo_math500' \
        "${OVERRIDES[@]}"
else
    EXPORTS="ALL,VERL_LOGGING_LEVEL=INFO,ICS_L2_PROMPT=l2new"
    EXPORTS="${EXPORTS},SRPO_PER_TOKEN_DUMP_DIR=${DUMP_DIR},SRPO_PER_TOKEN_DUMP_MAX=200"
    EXPORTS="${EXPORTS},SCGRPO_BRANCH_DUMP_DIR=${BRANCH_DIR},SCGRPO_BRANCH_DUMP_EVERY=1,SCGRPO_BRANCH_DUMP_MAX=20"

    sbatch --partition=q1 \
           --nodes=1 \
           --gpus-per-node=2 \
           --cpus-per-gpu=10 \
           --time=2:00:00 \
           --job-name="${JOB_NAME}" \
           --output="${SCPO_DIR}/batch_scripts/logs/${JOB_NAME}.out" \
           --error="${SCPO_DIR}/batch_scripts/logs/${JOB_NAME}.err" \
           --export="${EXPORTS}" \
           --wrap="bash -c 'source ${CONDA_PATH}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${SCPO_DIR} && bash ${SCPO_DIR}/batch_scripts/submit_grad_dump_l2n_ep1_v2.sh --local'"

    echo "Submitted: ${JOB_NAME}"
    echo "  out: batch_scripts/logs/${JOB_NAME}.out"
    echo "  err: batch_scripts/logs/${JOB_NAME}.err"
fi
