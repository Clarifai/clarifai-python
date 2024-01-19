from argparse import Namespace, _SubParsersAction

from ._utils import list_model_upload_examples
from .base import BaseClarifaiCli


class ExampleCli(BaseClarifaiCli):

  @staticmethod
  def register(parser: _SubParsersAction):
    creator_parser = parser.add_parser("example", help="Download/List examples of model upload")
    sub_creator_parser = creator_parser.add_subparsers()

    SubListExampleCli.register(sub_creator_parser)

    creator_parser.set_defaults(func=ExampleCli)


class SubListExampleCli(BaseClarifaiCli):

  @staticmethod
  def register(parser: _SubParsersAction):
    _parser = parser.add_parser("list")
    _parser.add_argument("--force-download", action="store_true", help="Force download examples")
    _parser.set_defaults(func=SubListExampleCli)

  def __init__(self, args: Namespace) -> None:
    self.force_download = args.force_download

  def run(self):
    _list = list_model_upload_examples(self.force_download)
    print(f"Found {len(_list)} examples")
    for each in _list:
      print(f" * {each}")
