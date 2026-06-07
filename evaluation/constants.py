"""
Constants and default values for SCPO training.

All magic numbers and default configurations should be defined here.
"""

# =============================================================================
# MODEL DEFAULTS
# =============================================================================
DEFAULT_MODEL = "allenai/OLMo-3-7B-Instruct"
DEFAULT_TORCH_DTYPE = "bfloat16"

# =============================================================================
# LORA DEFAULTS
# =============================================================================
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.0
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# =============================================================================
# ALGORITHM SELECTION
# =============================================================================
DEFAULT_ALGORITHM = "dpo"  # "dpo", "kto", "sft", or "grpo"
SUPPORTED_ALGORITHMS = ["dpo", "kto", "sft", "grpo"]

# =============================================================================
# DPO DEFAULTS
# =============================================================================
DEFAULT_BETA = 0.1  # KL penalty coefficient
DEFAULT_LOSS_TYPE = "sigmoid"  # sigmoid, hinge, ipo
DEFAULT_LABEL_SMOOTHING = 0.0

# =============================================================================
# KTO DEFAULTS (from TRL v0.9.6)
# =============================================================================
DEFAULT_KTO_DESIRABLE_WEIGHT = 1.0   # Weight for desirable (correct) losses
DEFAULT_KTO_UNDESIRABLE_WEIGHT = 1.0  # Weight for undesirable (incorrect) losses

# =============================================================================
# GRPO DEFAULTS
# =============================================================================
DEFAULT_GRPO_KL_COEFF = 0.0          # KL penalty coefficient (0 = no KL, no reference model)
DEFAULT_GRPO_NORMALIZE_ADVANTAGES = True

# =============================================================================
# TRAINING DEFAULTS
# =============================================================================
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_NUM_TRAIN_STEPS = 1000
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_WARMUP_RATIO = 0.1  # 10% of training steps

# =============================================================================
# GENERATION DEFAULTS
# =============================================================================
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 1.0
DEFAULT_NUM_GENERATIONS_PER_PROMPT = 2

# =============================================================================
# DATA DEFAULTS
# =============================================================================
DEFAULT_DATASET = "math500"
DEFAULT_NUM_PROMPTS = 100
DEFAULT_MAX_PROMPT_LENGTH = 512
DEFAULT_MAX_COMPLETION_LENGTH = 1024

# Train/Val/Test split ratios (must sum to 1.0)
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1

# =============================================================================
# LOGGING DEFAULTS
# =============================================================================
DEFAULT_LOGGING_STEPS = 10
DEFAULT_EVAL_STEPS = 100
DEFAULT_SAVE_STEPS = 500

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
DEFAULT_SEED = 42
DEFAULT_DATA_SEED = 42  # Separate seed for data splitting (independent of training seed)

# =============================================================================
# PATHS
# =============================================================================
DEFAULT_OUTPUT_DIR = "./experiments"
DEFAULT_CACHE_DIR = "./cache"
EXPERIMENT_CONFIG_FILENAME = "config.yaml"
EXPERIMENT_METRICS_FILENAME = "metrics.json"
EXPERIMENT_CHECKPOINT_PREFIX = "checkpoint"

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that solves math and reasoning problems step by step. Show your work clearly and put your final answer in \\boxed{}."""

DEFAULT_PROMPT_TEMPLATE = """{system_prompt}

Problem: {problem}

Solution:"""

# =============================================================================
# SUPPORTED VALUES
# =============================================================================
SUPPORTED_DATASETS = [
    "math500",
    "math_level5",
    "gsm8k",
    "amc23",
    "aime",
    "gpqa",
    "csqa",
    "mathqa",
    "mmlu_pro",
    "ifeval",
    "sciknoweval_l3",
    "sciknoweval_chemistry",
    "sciknoweval_physics",
    "sciknoweval_biology",
    "sciknoweval_materials",
    "livecodebench",
    "livecodebench_medium",
    "livecodebench_hard",
    "theoremqa",
    "strategyqa",
    "agieval",
    "hotpotqa",
    "humaneval_plus",
    "mbpp_plus",
    "putnambench_lean",
]

SUPPORTED_LOSS_TYPES = [
    "sigmoid",
    "hinge",
    "ipo",
]

SUPPORTED_KTO_LOSS_TYPES = [
    "kto",
]

SUPPORTED_TORCH_DTYPES = [
    "float32",
    "float16",
    "bfloat16",
]

# =============================================================================
# KTO GENERATION SETTINGS
# =============================================================================
# KTO benefits from more samples to get both desirable and undesirable examples
DEFAULT_KTO_GENERATION_MULTIPLIER = 2

# =============================================================================
# VALIDATION TOLERANCES
# =============================================================================
# Tolerance for floating point comparisons (e.g., split ratios summing to 1.0)
SPLIT_RATIO_TOLERANCE = 1e-6

# =============================================================================
# THOUGHT-MDP DEFAULTS
# =============================================================================
DEFAULT_THOUGHT_DELIMITER = "</thought>"
DEFAULT_THOUGHT_TRAINING_FORMAT = "full_trajectory"  # or "step_by_step"
DEFAULT_MAX_THOUGHTS_PER_TRAJECTORY = 20
DEFAULT_MAX_TOKENS_PER_THOUGHT = 512
DEFAULT_NUM_TRAJECTORIES_PER_PROMPT = 8
DEFAULT_THOUGHT_TEMPERATURE = 0.7
DEFAULT_USE_THOUGHT_EXAMPLES = True

SUPPORTED_THOUGHT_TRAINING_FORMATS = ["full_trajectory", "step_by_step"]

# =============================================================================
# THOUGHT-LEVEL GRPO DEFAULTS
# =============================================================================
DEFAULT_THOUGHT_LEVEL_GRPO = False

# =============================================================================
# PROCESS REWARD DEFAULTS
# =============================================================================
DEFAULT_USE_PROCESS_REWARD = False
DEFAULT_PROCESS_REWARD_WEIGHT = 0.3
DEFAULT_PROCESS_REWARD_MAX_TOKENS = 512
DEFAULT_PROCESS_REWARD_TEMPERATURE = 0.3
DEFAULT_PROCESS_REWARD_AGGREGATION = "mean"  # "mean", "min", or "last"

# =============================================================================
# VLLM GENERATION DEFAULTS
# =============================================================================
DEFAULT_USE_VLLM_GENERATION = False
DEFAULT_VLLM_GPU_IDS = "1"
DEFAULT_VLLM_TENSOR_PARALLEL_SIZE = 1
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.4
