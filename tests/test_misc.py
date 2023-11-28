import logging

from rich.logging import RichHandler

from clarifai.utils.logging import _get_library_name, get_logger


def test_get_logger():
  logger = get_logger("DEBUG", "test_logger")
  assert logger.level == logging.DEBUG
  assert logger.name == "test_logger"
  assert isinstance(logger.handlers[0], RichHandler)


def test_get_logger_defaults():
  logger = get_logger()
  assert logger.level == logging.NOTSET
  assert logger.name == _get_library_name()
  assert isinstance(logger.handlers[0], RichHandler)
