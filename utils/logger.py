import os
import sys
import logging
from datetime import datetime

import transformers
import datasets

def setup_logger(output_dir: str = "logs", log_level: int = logging.INFO):
    """
    Sets up the global Python logging configuration.
    Logs are seamlessly streamed to both the console and a timestamped file.

    Args:
        output_dir (str): The directory where log files will be saved.
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"experiment_{timestamp}.log")

    log_format = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(log_format)

    logging.basicConfig(
        level=log_level,
        handlers=[console_handler, file_handler],
        force=True
    )
    
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    datasets.utils.logging.set_verbosity_warning()

    logger = logging.getLogger(__name__)
    logger.info(f"Global Logger initialized successfully. Logs are being saved to: {log_file}")
