"""
Utility functions for SCPO training.

Includes:
- Seeding for reproducibility
- Experiment directory setup
- Config serialization
"""

import hashlib
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all random sources.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.info(f"Set random seed to {seed}")


def get_experiment_id(config_dict: Dict[str, Any]) -> str:
    """
    Generate a unique experiment ID based on config hash and timestamp.

    Format: {short_hash}_{timestamp}

    Args:
        config_dict: Configuration dictionary

    Returns:
        Unique experiment identifier
    """
    # Create hash from config (excluding output paths to avoid circular dependency)
    config_for_hash = {k: v for k, v in config_dict.items()
                       if k not in ["output_dir", "experiment_id", "experiment_dir"]}
    config_str = json.dumps(config_for_hash, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]

    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{config_hash}_{timestamp}"


def setup_experiment_dir(
    base_dir: Path,
    experiment_name: str,
    config_dict: Dict[str, Any],
) -> Path:
    """
    Create experiment directory and save config.

    Structure:
        base_dir/
            experiment_name/
                config.yaml
                checkpoints/
                logs/

    Args:
        base_dir: Base experiments directory
        experiment_name: Name/ID for this experiment
        config_dict: Full configuration to save

    Returns:
        Path to experiment directory
    """
    from .constants import EXPERIMENT_CONFIG_FILENAME

    experiment_dir = Path(base_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)

    # Save config
    config_path = experiment_dir / EXPERIMENT_CONFIG_FILENAME
    save_config(config_dict, config_path)

    logger.info(f"Created experiment directory: {experiment_dir}")

    return experiment_dir


def save_config(config_dict: Dict[str, Any], path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config_dict: Configuration dictionary
        path: Output path for YAML file
    """
    # Convert Path objects to strings for YAML serialization
    serializable = {}
    for k, v in config_dict.items():
        if isinstance(v, Path):
            serializable[k] = str(v)
        elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], Path):
            serializable[k] = [str(p) for p in v]
        else:
            serializable[k] = v

    with open(path, "w") as f:
        yaml.dump(serializable, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {path}")


def load_config(path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    logger.info(f"Loaded config from {path}")
    return config_dict


def get_torch_dtype(dtype_str: str):
    """
    Convert string dtype to torch dtype.

    Args:
        dtype_str: One of "float32", "float16", "bfloat16"

    Returns:
        torch.dtype
    """
    import torch

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")

    return dtype_map[dtype_str]


def get_device() -> "torch.device":
    """
    Get the best available device (CUDA if available, else CPU).

    Returns:
        torch.device
    """
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device
