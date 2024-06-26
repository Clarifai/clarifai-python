import argparse

from clarifai.models.model_serving.constants import CLARIFAI_PAT_PATH
from ..utils import _persist_pat
from .base import BaseClarifaiCli


class LoginCli(BaseClarifaiCli):

  @staticmethod
  def register(parser: argparse._SubParsersAction):
    upload_parser = parser.add_parser("login", help="Login to Clarifai and save PAT locally")
    upload_parser.set_defaults(func=LoginCli)

  def __init__(self, args: argparse.Namespace) -> None:
    pass

  def _parse_config(self):
    # do something with self.config_path
    raise NotImplementedError()

  def run(self):
    msg = "Get your PAT from https://clarifai.com/settings/security and pass it here: "
    _pat = input(msg)
    _persist_pat(_pat)
    print(f"Your PAT is saved at {CLARIFAI_PAT_PATH}")
