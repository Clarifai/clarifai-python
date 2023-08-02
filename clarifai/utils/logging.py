import logging
from typing import Optional

from rich.logging import RichHandler
from rich.table import Table
from rich.traceback import install

install()


def table_from_dict(data, column_names, title="") -> Table:
  """Use this function for printing tables from a list of dicts."""
  table = Table(title=title, show_header=True, header_style="bold blue")
  for column_name in column_names:
    table.add_column(column_name)
  for row in data:
    req_row = [row.get(column_name, "") for column_name in column_names]
    table.add_row(*req_row)
  return table


def _get_library_name() -> str:
  return __name__.split(".")[0]


def _configure_logger(logger_level: str = "ERROR") -> None:
  logging.basicConfig(
      level=logger_level,
      datefmt='%Y-%m-%d %H:%M:%S',
      handlers=[RichHandler(rich_tracebacks=True)])


def get_logger(logger_level: str = "ERROR", name: Optional[str] = None) -> logging.Logger:
  """Return a logger with the specified name."""

  if name is None:
    name = _get_library_name()

  _configure_logger(logger_level)
  return logging.getLogger(name)
