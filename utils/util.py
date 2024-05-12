import json
import logging
import os

def load_config(path):
    """Load the configuration file from the given path. Provide a default if not present."""
    default_config = {
        'api': {
            'base_url': 'https://paper-api.alpaca.markets',
            'api_key': 'your_api_key',
            'api_secret': 'your_api_secret'
        },
        'model': {
            'path': 'models/default_model.pth',
            'update_interval': 86400
        },
        'logging': {
            'filename': 'trading_log.log',
            'level': 'INFO',
            'format': '%(asctime)s:%(levelname)s:%(message)s'
        }
    }
    
    if not os.path.exists(path):
        with open(path, 'w') as file:
            json.dump(default_config, file)
        return default_config
    else:
        with open(path, 'r') as file:
            return json.load(file)

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