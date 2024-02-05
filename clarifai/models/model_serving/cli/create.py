import os
import shutil
from argparse import Namespace, _SubParsersAction
from typing import List

from InquirerPy import prompt

from ..model_config import MODEL_TYPES
from ..repo_build import RepositoryBuilder
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
      if self.overwrite:
        print(f"Overwrite {self.working_dir}")
      else:
        raise FileExistsError(
            f"{self.working_dir} exists. If you want to overwrite it, please set `--overwrite` flag"
        )

    # prevent wrong args when creating from example
    if not self.from_example:
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

      else:
        available_examples = list(list_model_upload_examples().keys())
        assert self.example_id in available_examples, f"Available examples are: {available_examples}, got {self.example_id}."

  def run(self):
    if self.from_example:
      os.makedirs(self.working_dir, exist_ok=True)
      model_repo, readme = list_model_upload_examples()[self.example_id]
      shutil.copytree(model_repo, self.working_dir, dirs_exist_ok=True)
      if readme:
        shutil.copy(readme, os.path.join(self.working_dir, "readme.md"))

    else:
      RepositoryBuilder.init_repository(
          self.type,
          self.working_dir,
          backend="triton",
          max_batch_size=self.max_bs,
          image_shape=self.image_shape)

    from itertools import islice
    from pathlib import Path

    def tree(dir_path: Path,
             level: int = -1,
             limit_to_directories: bool = False,
             length_limit: int = 1000):
      # prefix components:
      space = '    '
      branch = '│   '
      # pointers:
      tee = '├── '
      last = '└── '
      """Given a directory Path object print a visual tree structure"""
      dir_path = Path(dir_path)  # accept string coerceable to Path
      files = 0
      directories = 0

      def inner(dir_path: Path, prefix: str = '', level=-1):
        nonlocal files, directories
        if not level:
          return  # 0, stop iterating
        if limit_to_directories:
          contents = [d for d in dir_path.iterdir() if d.is_dir()]
        else:
          contents = list(dir_path.iterdir())
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
          if path.is_dir():
            yield prefix + pointer + path.name
            directories += 1
            extension = branch if pointer == tee else space
            yield from inner(path, prefix=prefix + extension, level=level - 1)
          elif not limit_to_directories:
            yield prefix + pointer + path.name
            files += 1

      print(dir_path.name)
      iterator = inner(dir_path, level=level)
      for line in islice(iterator, length_limit):
        print(line)
      if next(iterator, None):
        print(f'... length_limit, {length_limit}, reached, counted:')
      print(f'\n{directories} directories' + (f', {files} files' if files else ''))

    print("-" * 75)
    print(f"* Created repository at: {self.working_dir}")
    tree(self.working_dir)
    print()
    print("* Please make sure your code is tested using `test.py` before uploading")
    print("-" * 75)
