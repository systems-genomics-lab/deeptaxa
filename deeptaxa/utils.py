"""
Module: utils.py

This module provides utility functions for:
  - Setting random seeds for reproducibility.
  - Logging device information (e.g., GPU details).
  - Printing experiment parameters for record-keeping.
  - Checking the existence of files to prevent runtime errors.
"""

import torch
import random
import numpy as np
import os
from argparse import Namespace
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.
    
    Deep learning experiments often require consistent results across runs,
    which is achieved by setting fixed seeds for all random number generators.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug("Random seed set to %d", seed)

def get_device_info(verbose: bool = False) -> None:
    """
    Log detailed information about available computation devices (CPU/GPU).
    
    Useful for confirming that the model is utilizing the intended hardware.
    
    Args:
        verbose (bool): If True, logs detailed GPU information.
    """
    if verbose:
        logger.info("CUDA available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("GPU count: %d", torch.cuda.device_count())
            logger.info("Current GPU: %d - %s", torch.cuda.current_device(), torch.cuda.get_device_name(0))

def print_parameters(args: Namespace, verbose: bool) -> None:
    """
    Log all command-line parameters to facilitate experiment reproducibility and debugging.
    
    Records all hyperparameters and configuration options.
    
    Args:
        args (Namespace): Parsed command-line arguments.
        verbose (bool): If True, prints the parameters.
    """
    if verbose:
        logger.info("Parameters:")
        for key, value in vars(args).items():
            logger.info("  %s: %s", key, value)

def check_file(file_path: str) -> None:
    """
    Check if the specified file exists.
    
    If the file does not exist, an error is logged and a FileNotFoundError is raised.
    
    Args:
        file_path (str): The path to the file.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(file_path):
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"File not found: {file_path}")

if __name__ == "__main__":
    pass

