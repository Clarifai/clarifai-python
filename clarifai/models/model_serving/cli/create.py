from argparse import Namespace, _SubParsersAction
from typing import List

from ..constants import MAX_HW_DIM
from ..model_config import MODEL_TYPES, get_model_config
from ..pb_model_repository import TritonModelRepository
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
        "--max_bs", type=int, default=1, required=False, help="Max batch size")
    model_parser.set_defaults(func=SubCreateModelCli)

  def __init__(self, args: Namespace) -> None:

    if len(args.image_shape) != 2:
      raise ValueError(
          f"image_shape takes 2 values, Height and Width. Got {len(args.image_shape)} values instead."
      )
    if args.image_shape[0] > MAX_HW_DIM or args.image_shape[1] > MAX_HW_DIM:
      raise ValueError(
          f"H and W each have a maximum value of 1024. Got H: {args.image_shape[0]}, W: {args.image_shape[1]}"
      )
    self.image_shape: List[int] = args.image_shape

    self.working_dir: str = args.working_dir
    self.type: str = args.type
    self.max_bs: int = args.max_bs

  def run(self):
    model_config = get_model_config(self.type).make_triton_model_config(
        model_name="",
        model_version="1",
        image_shape=self.image_shape,
        max_batch_size=self.max_bs,
    )

    triton_repo = TritonModelRepository(model_config)
    triton_repo.build_repository(self.working_dir)
