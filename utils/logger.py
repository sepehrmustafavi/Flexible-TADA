import logging
import sys
import os
from datetime import datetime

def setup_logger(output_dir: str = "logs", log_level: int = logging.INFO):
    """
    Sets up the global Python logging configuration.
    Logs are seamlessly streamed to both the console and a timestamped file.

    Args:
        output_dir (str): The directory where log files will be saved.
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
    """
    # Step 1: Create the logs directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 2: Generate a unique log filename based on the current timestamp
    # This prevents overwriting logs from previous runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"experiment_{timestamp}.log")

    # Step 3: Define a professional log format
    # Format: [YYYY-MM-DD HH:MM:SS] [LEVEL] [module_name] - Message
    log_format = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Step 4: Configure the Console Handler (Prints to terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)

    # Step 5: Configure the File Handler (Saves to text file)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(log_format)

    # Step 6: Apply to Root Logger
    # Using basicConfig with force=True ensures we override any default loggers 
    # that might have been initialized by other libraries prematurely.
    logging.basicConfig(
        level=log_level,
        handlers=[console_handler, file_handler],
        force=True
    )

    # Step 7: Sync with HuggingFace Transformers logging
    # This ensures HF doesn't spam our logs with unnecessary warnings
    import transformers
    import datasets
    
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    datasets.utils.logging.set_verbosity_warning() # Keep datasets library quiet unless there's an error

    # Initial log to confirm setup
    logger = logging.getLogger(__name__)
    logger.info(f"Global Logger initialized successfully. Logs are being saved to: {log_file}")