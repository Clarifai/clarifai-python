from argparse import ArgumentParser

from .build import BuildCli
from .create import CreateCli
from .example_cli import ExampleCli
from .login import LoginCli
from .upload import UploadCli


def main():

  parser = ArgumentParser("clarifai")
  cmd_parser = parser.add_subparsers(help="Clarifai cli helpers")

  UploadCli.register(cmd_parser)
  CreateCli.register(cmd_parser)
  LoginCli.register(cmd_parser)
  ExampleCli.register(cmd_parser)
  BuildCli.register(cmd_parser)

  args = parser.parse_args()

  if not hasattr(args, "func"):
    parser.print_help()
    exit(1)

  # Run
  service = args.func(args)
  service.run()


if __name__ == "__main__":
  main()
