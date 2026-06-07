"""
Dataset utilities for the ICS module.

Provides normalize_answer and load_dataset_by_name by importing directly
from TREE's dataset_loaders (same source evaluation.data_loader uses).
This avoids triggering evaluation's heavy __init__ imports.
"""

import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Import directly from the vendored dataset_loaders (same source as
# evaluation.data_loader; see vendor/README.md).
TREE_DIR = Path(__file__).parent.parent / "vendor"
if str(TREE_DIR) not in sys.path:
    sys.path.insert(0, str(TREE_DIR))

try:
    from dataset_loaders import normalize_answer, load_dataset_by_name
except ImportError:
    logger.warning("TREE dataset_loaders not available; normalize_answer and load_dataset_by_name will not work")
    normalize_answer = None
    load_dataset_by_name = None

__all__ = ["normalize_answer", "load_dataset_by_name"]
