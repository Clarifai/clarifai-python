# Copyright 2023 Clarifai, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Commandline interface for model upload utils."""
import argparse
import os
import subprocess

from ..repo_build import RepositoryBuilder
from .base import BaseClarifaiCli


class BuildCli(BaseClarifaiCli):

  @staticmethod
  def register(parser: argparse._SubParsersAction):
    parser = parser.add_parser("build", help="Build clarifai model for uploading")
    sub_parser = parser.add_subparsers()

    BuildModelSubCli.register(sub_parser)

    parser.set_defaults(func=BuildCli)


class BuildModelSubCli(BaseClarifaiCli):

  @staticmethod
  def register(parser: argparse._SubParsersAction):
    sub_parser = parser.add_parser("model", help="Build Clarifai model")
    sub_parser.add_argument(
        "path",
        type=str,
        nargs='?',
        help="Path to working directory, default is current directory",
        default=".")
    sub_parser.add_argument(
        "--out-path", type=str, required=False, help="Output path of built model", default=None)
    sub_parser.add_argument(
        "--name",
        type=str,
        required=False,
        help="Name of built file, default is `clarifai_model_id` in config if set or `model`",
        default=None)
    sub_parser.add_argument(
        "--no-test",
        action="store_true",
        help="Trigger this flag to skip testing before uploading")
    sub_parser.set_defaults(func=BuildModelSubCli)

  def __init__(self, args: argparse.Namespace) -> None:
    self.path = args.path
    self.no_test = args.no_test
    self.test_path = os.path.join(self.path, "test.py")
    self.output_path = args.out_path or self.path
    self.serving_backend = "triton"
    self.name = args.name

  def run(self):

    # Run test before uploading
    if not self.no_test:
      assert os.path.exists(
          self.test_path), FileNotFoundError(f"Could not find `test.py` in {self.path}")
      result = subprocess.run(f"pytest -s --log-level=INFO {self.test_path}", shell=True)
      assert result.returncode == 0, "Test has failed. Please make sure no error exists in your code."

    # build
    print("Start building...")
    RepositoryBuilder.build(
        self.path, backend=self.serving_backend, output_dir=self.output_path, name=self.name)
