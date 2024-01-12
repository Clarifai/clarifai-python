import os
import shutil
from argparse import Namespace, _SubParsersAction
from typing import List

from InquirerPy import prompt

from ..constants import MAX_HW_DIM
from ..model_config import MODEL_TYPES, get_model_config
from ..pb_model_repository import TritonModelRepository
from ._utils import list_model_upload_examples
from .base import BaseClarifaiCli


class CreateCli(BaseClarifaiCli):

  @staticmethod
  def register(parser: _SubParsersAction):
    creator_parser = parser.add_parser("create", help="Create component of Clarifai platform")
    sub_creator_parser = creator_parser.add_subparsers()

    SubCreateModelCli.register(sub_creator_parser)

    creator_parser.set_defaults(func=CreateCli)


class SubCreateModelCli(BaseClarifaiCli):

  @staticmethod
  def register(parser: _SubParsersAction):
    model_parser = parser.add_parser("model")
    model_parser.add_argument(
        "--working-dir",
        type=str,
        required=True,
        help="Path to your working dir. Create new dir if it does not exist")
    model_parser.add_argument(
        "--from-example",
        required=False,
        action="store_true",
        help="Create repository from example")
    model_parser.add_argument(
        "--example-id",
        required=False,
        type=str,
        choices=list_model_upload_examples(),
        help="Example id, run `clarifai example list` to list of examples")

    model_parser.add_argument(
        "--type",
        type=str,
        choices=MODEL_TYPES,
        required=False,
        help="Clarifai supported model types.")
    model_parser.add_argument(
        "--image-shape",
        nargs='+',
        type=int,
        required=False,
        help="H W dims for models with an image input type. H and W each have a max value of 1024",
        default=[-1, -1])
    model_parser.add_argument(
        "--max-bs", type=int, default=1, required=False, help="Max batch size")

    model_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite working-dir if exists")

    model_parser.set_defaults(func=SubCreateModelCli)

  def __init__(self, args: Namespace) -> None:
    self.working_dir: str = args.working_dir
    self.from_example = args.from_example
    self.example_id = args.example_id
    self.overwrite = args.overwrite

    if os.path.exists(self.working_dir):
      #confirm = inquirer.confirm(f"Do you want to overwrite `{self.working_dir}`?", confirm_letter="Y", reject_letter="n").execute()
      #if not confirm:
      #  print("Cancel process.")
      #  exit(1)
      #else:
      #  pass
      if self.overwrite:
        print(f"Overwrite {self.working_dir}")
      else:
        raise FileExistsError(
            f"{self.working_dir} exists. If you want to overwrite it, please set `--overwrite` flag"
        )

    # prevent wrong args when creating from example
    if not self.from_example:
      if len(args.image_shape) != 2:
        raise ValueError(
            f"image_shape takes 2 values, Height and Width. Got {len(args.image_shape)} values instead."
        )
      if args.image_shape[0] > MAX_HW_DIM or args.image_shape[1] > MAX_HW_DIM:
        raise ValueError(
            f"H and W each have a maximum value of 1024. Got H: {args.image_shape[0]}, W: {args.image_shape[1]}"
        )
      self.image_shape: List[int] = args.image_shape

      self.type: str = args.type
      self.max_bs: int = args.max_bs

    else:
      if not self.example_id:
        questions = [
            {
                "type": "list",
                "message": "Select an example:",
                "choices": list_model_upload_examples(),
            },
        ]
        result = prompt(questions)
        self.example_id = result[0]

  def run(self):
    if self.from_example:
      os.makedirs(self.working_dir, exist_ok=True)
      model_repo, readme = list_model_upload_examples()[self.example_id]
      shutil.copytree(model_repo, self.working_dir, dirs_exist_ok=True)
      if readme:
        shutil.copy(readme, os.path.join(self.working_dir, "readme.md"))

    else:
      model_config = get_model_config(self.type).make_triton_model_config(
          model_name="",
          model_version="1",
          image_shape=self.image_shape,
          max_batch_size=self.max_bs,
      )

      triton_repo = TritonModelRepository(model_config)
      triton_repo.build_repository(self.working_dir)
