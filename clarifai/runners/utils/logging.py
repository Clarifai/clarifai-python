import os

from clarifai.utils.logging import get_logger

logger_level = os.environ.get("LOG_LEVEL", "INFO")
logger = get_logger(logger_level, __name__)
