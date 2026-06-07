"""SCPO trainer — shared evaluation/data utilities.

The original custom online DPO/KTO/SFT/GRPO training infrastructure has been
removed (superseded by the verl-based pipeline in training). What
remains here are the utilities still used by the eval harness and thought_ics:
dataset loading, evaluation, the thought-MDP rollout, and the thought
evaluator.
"""

# Constants
from .constants import (
    # Model defaults
    DEFAULT_MODEL,
    DEFAULT_TORCH_DTYPE,
    # Generation defaults
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_NUM_GENERATIONS_PER_PROMPT,
    # Data defaults
    DEFAULT_DATASET,
    DEFAULT_NUM_PROMPTS,
    DEFAULT_MAX_PROMPT_LENGTH,
    DEFAULT_MAX_COMPLETION_LENGTH,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    DEFAULT_TEST_RATIO,
    # Paths / misc
    DEFAULT_OUTPUT_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_SEED,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_PROMPT_TEMPLATE,
    SUPPORTED_DATASETS,
    SPLIT_RATIO_TOLERANCE,
    # Thought-MDP constants
    DEFAULT_THOUGHT_DELIMITER,
    DEFAULT_THOUGHT_TRAINING_FORMAT,
    DEFAULT_MAX_THOUGHTS_PER_TRAJECTORY,
    DEFAULT_MAX_TOKENS_PER_THOUGHT,
    DEFAULT_NUM_TRAJECTORIES_PER_PROMPT,
    DEFAULT_THOUGHT_TEMPERATURE,
    DEFAULT_USE_THOUGHT_EXAMPLES,
    SUPPORTED_THOUGHT_TRAINING_FORMATS,
)

# Utilities
from .utils import (
    set_seed,
    get_experiment_id,
    setup_experiment_dir,
    save_config,
    load_config,
    get_torch_dtype,
    get_device,
)

# Log probability computation
from .logprobs import (
    selective_log_softmax,
    compute_logprobs,
    compute_logprobs_batched,
)

# Data loading
from .data_loader import (
    DataSplit,
    DatasetSplits,
    load_dataset_splits,
    load_raw_problems,
    split_problems,
    format_prompt,
    get_train_prompts,
)

# Thought-MDP
from .thought_mdp import (
    ThoughtNode,
    ToTState,
    ToTAction,
    ToTAgent,
    ToTEnvironment,
    TreeSearch,
    get_completed_paths,
    extract_boxed_answers,
    prompt_template_with_examples,
    prompt_template_no_examples,
)

# Thought Evaluator (rubric-based coherence scoring)
from .thought_evaluator import (
    ThoughtEvaluationResult,
    ThoughtEvaluator,
    DEFAULT_RUBRIC_DIMENSIONS,
    build_evaluation_prompt,
    parse_evaluation_response,
)


__all__ = [
    # Constants
    "DEFAULT_MODEL",
    "DEFAULT_TORCH_DTYPE",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_NUM_GENERATIONS_PER_PROMPT",
    "DEFAULT_DATASET",
    "DEFAULT_NUM_PROMPTS",
    "DEFAULT_MAX_PROMPT_LENGTH",
    "DEFAULT_MAX_COMPLETION_LENGTH",
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_VAL_RATIO",
    "DEFAULT_TEST_RATIO",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_SEED",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_PROMPT_TEMPLATE",
    "SUPPORTED_DATASETS",
    "SPLIT_RATIO_TOLERANCE",
    "DEFAULT_THOUGHT_DELIMITER",
    "DEFAULT_THOUGHT_TRAINING_FORMAT",
    "DEFAULT_MAX_THOUGHTS_PER_TRAJECTORY",
    "DEFAULT_MAX_TOKENS_PER_THOUGHT",
    "DEFAULT_NUM_TRAJECTORIES_PER_PROMPT",
    "DEFAULT_THOUGHT_TEMPERATURE",
    "DEFAULT_USE_THOUGHT_EXAMPLES",
    "SUPPORTED_THOUGHT_TRAINING_FORMATS",
    # Utilities
    "set_seed",
    "get_experiment_id",
    "setup_experiment_dir",
    "save_config",
    "load_config",
    "get_torch_dtype",
    "get_device",
    # Log probs
    "selective_log_softmax",
    "compute_logprobs",
    "compute_logprobs_batched",
    # Data loading
    "DataSplit",
    "DatasetSplits",
    "load_dataset_splits",
    "load_raw_problems",
    "split_problems",
    "format_prompt",
    "get_train_prompts",
    # Thought-MDP
    "ThoughtNode",
    "ToTState",
    "ToTAction",
    "ToTAgent",
    "ToTEnvironment",
    "TreeSearch",
    "get_completed_paths",
    "extract_boxed_answers",
    "prompt_template_with_examples",
    "prompt_template_no_examples",
    # Thought Evaluator
    "ThoughtEvaluationResult",
    "ThoughtEvaluator",
    "DEFAULT_RUBRIC_DIMENSIONS",
    "build_evaluation_prompt",
    "parse_evaluation_response",
]

__version__ = "0.1.0"
