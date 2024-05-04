# file: stock_ai_bot/utils/logging.py

import logging
import os

def setup_logging(log_file: str = "application.log"):
    """Sets up the logging configuration.

    Args:
        log_file (str): Path to the log file where logs will be saved.
    """
    # Create a logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Define log formatting
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_logger(name: str):
    """Retrieves a logger instance.

    Args:
        name (str): The name of the logger, usually the module's `__name__`.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)
