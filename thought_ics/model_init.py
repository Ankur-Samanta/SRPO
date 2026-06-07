"""
Model initialization utilities for the ICS module.

Ported from TREE/tree_of_thought.py initialize_model() and initialize_model_3p().
SIERA model imports are optional — a clear error is raised if missing.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Optional SIERA model imports
try:
    from SIERA.src.models.base_manager import BaseModelManager
    from SIERA.src.models.config import ModelConfigLoader, InferenceConfig
    SIERA_MODELS_AVAILABLE = True
except ImportError:
    SIERA_MODELS_AVAILABLE = False


def initialize_model(
    gpu_ids: str = "1",
    tensor_parallel_size: int = 1,
    model_name: str = "llama8b",
    model_seed: int = None
):
    """
    Initialize vLLM model with optional multi-GPU support.

    Args:
        gpu_ids: Comma-separated GPU IDs (e.g., "0,1" for 2 GPUs)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        model_name: Model nickname from models.yaml (default: "llama8b")
                   Options: llama8b, llama70b, qwen7b, qwen32b, qwen2b, llama3b, phi4b
        model_seed: Seed for model generation (None=non-deterministic, int=deterministic)

    Returns:
        Initialized model manager
    """
    if not SIERA_MODELS_AVAILABLE:
        raise ImportError(
            "SIERA model infrastructure not available. "
            "Ensure SIERA.src.models is importable (install SIERA or add to PYTHONPATH)."
        )

    # Only set CUDA_VISIBLE_DEVICES if not already set (e.g., for parallel runs)
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

    config_loader = ModelConfigLoader()
    model_config = config_loader.get_model_config(model_name)

    inference_config = InferenceConfig(
        backend="vllm",
        gpu_memory_utilization=0.8,
        max_num_seqs=32,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
        enable_lora=False,
        model_seed=model_seed
    )

    logger.info(f"Loading model '{model_name}' ({model_config.hf_name}) on GPUs {gpu_ids} with tensor_parallel_size={tensor_parallel_size}...")
    manager = BaseModelManager(
        model_config=model_config,
        device=f"cuda:0",
        inference_config=inference_config
    )
    manager.load_base_model()
    logger.info(f"Model '{model_name}' loaded successfully!")

    return manager


def initialize_model_3p(
    api_key: str,
    model: str = "gpt-4o"
):
    """
    Initialize a third-party model manager for API-based inference.

    This function creates a ThirdPartyModelManager that routes all inference
    to the OpenAI API instead of using local vLLM. No GPU resources are used.

    Args:
        api_key: OpenAI API key
        model: Model to use for generation (default: gpt-4o)

    Returns:
        Initialized ThirdPartyModelManager
    """
    # Lazy import — third_party_localization is vendored under vendor/.
    import sys
    from pathlib import Path
    TREE_DIR = Path(__file__).parent.parent / "vendor"
    if str(TREE_DIR) not in sys.path:
        sys.path.insert(0, str(TREE_DIR))

    from third_party_localization import ThirdPartyModelManager

    logger.info(f"Initializing 3P API model: {model} (no local GPU required)")
    manager = ThirdPartyModelManager(api_key=api_key, model=model)
    manager.load_base_model()  # No-op, but called for consistency

    return manager
