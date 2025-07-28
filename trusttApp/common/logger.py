from functools import wraps
import time
import logging
from logging.handlers import RotatingFileHandler
import os
from dotenv import load_dotenv


load_dotenv()

def setup_logger(name, log_file, level : str|int = logging.INFO):
    """Function to set up loggers for different services, dynamically finding the log directory."""
    LOG_FILE_PATH = os.getenv("LOG_FILE_PATH")

    # Find the root directory of the project based on this file's location
    # root_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # log_directory = os.path.join(root_directory, 'trusttApp', 'common', 'logs')
    log_directory = os.path.join(os.path.dirname(__file__), LOG_FILE_PATH)

    # Create the log directory if it doesn't exist
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Path to the log file
    log_path = os.path.join(log_directory, log_file)

    # Create a handler
    handler = RotatingFileHandler(log_path, maxBytes=100000000, backupCount=10, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Set up logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug("Entering function: %s", func.__name__)
        start_time = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end_time = time.time()
            logger.debug("Exiting function: %s; Duration: %.2f seconds", func.__name__, end_time - start_time)
    return wrapper
