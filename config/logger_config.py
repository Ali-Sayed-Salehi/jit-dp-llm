# logger_config.py

import logging
import sys
import os
from datetime import datetime

def setup_logger(name=None, log_dir=None, debug_mode=False):
    """
    Setup and return a logger.

    Args:
        name (str): Logger name (e.g., __name__)
        log_dir (str): Directory to save log file. If None, no file logging.
        debug_mode (bool): Set True for DEBUG level, False for INFO level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional File Handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"run_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
