from abc import ABC, abstractmethod
from argparse import _SubParsersAction


class BaseClarifaiCli(ABC):

  @staticmethod
  @abstractmethod
  def register(parser: _SubParsersAction):
    raise NotImplementedError()

  @abstractmethod
  def run(self):
    raise NotImplementedError()
