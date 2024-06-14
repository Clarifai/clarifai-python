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

from clarifai.models.model_serving.model_config import get_model_config, load_user_config
from clarifai.models.model_serving.model_config.inference_parameter import InferParamManager

from ..constants import BUILT_MODEL_EXT
from ..utils import login
from .base import BaseClarifaiCli


class UploadCli(BaseClarifaiCli):

  @staticmethod
  def register(parser: argparse._SubParsersAction):
    creator_parser = parser.add_parser("upload", help="Upload component to Clarifai platform")
    sub_creator_parser = creator_parser.add_subparsers()

    UploadModelSubCli.register(sub_creator_parser)

    creator_parser.set_defaults(func=UploadCli)


class UploadModelSubCli(BaseClarifaiCli):

  @staticmethod
  def register(parser: argparse._SubParsersAction):
    upload_parser = parser.add_parser("model", help="Upload Clarifai model")
    upload_parser.add_argument(
        "path",
        type=str,
        nargs='?',
        help=
        "Path to working dir to get clarifai_config.yaml or path to yaml. Default is current directory",
        default=".")
    upload_parser.add_argument(
        "--url", type=str, required=False, help="Direct download url of zip file", default=None)
    upload_parser.add_argument(
        "--file", type=str, required=False, help="Local built file", default=None)
    upload_parser.add_argument("--id", type=str, required=False, help="Model ID")
    upload_parser.add_argument(
        "--user-app",
        type=str,
        required=False,
        help="User ID and App ID separated by '/', e.g., <user_id>/<app_id>")
    upload_parser.add_argument(
        "--no-test",
        action="store_true",
        help="Trigger this flag to skip testing before uploading")
    upload_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Trigger this flag to not resume uploading local file")
    upload_parser.add_argument(
        "--update-version",
        action="store_true",
        required=False,
        help="Update exist model with new version")

    upload_parser.set_defaults(func=UploadModelSubCli)

  def __init__(self, args: argparse.Namespace) -> None:
    self.no_test = args.no_test
    self.no_resume = args.no_resume

    working_dir_or_config = args.path
    # if input a config file, then not running test
    if working_dir_or_config.endswith(".yaml"):
      # to folder
      working_dir_or_config = os.path.split(working_dir_or_config)[0]
      config_yaml_path = working_dir_or_config
      self.test_path = None
      self.no_test = True
    # if it is a directory - working dir then it must contain config and test
    else:
      config_yaml_path = os.path.join(working_dir_or_config, "clarifai_config.yaml")
      self.test_path = os.path.join(working_dir_or_config, "test.py")

    assert os.path.exists(config_yaml_path), FileNotFoundError(
        f"`{config_yaml_path}` does not exist")
    self.config = load_user_config(cfg_path=config_yaml_path)

    self.file = args.file
    self.url = args.url
    if self.file:
      assert not self.url, ValueError("Expected either file or url, not both.")
      assert os.path.exists(self.file), FileNotFoundError
    elif self.url:
      if len(self.url.split(":")) == 1:
        # if URL has no scheme, default to https
        self.url = f"https://{self.url}"
      assert self.url.startswith("http") or self.url.startswith("https") or self.url.startswith(
          "s3"
      ), f"Invalid URL scheme, supported schemes are 'http', 'https', or 's3'. Got {self.url}"
      self.file = None
    else:
      for _fname in os.listdir(working_dir_or_config):
        if _fname.endswith(BUILT_MODEL_EXT):
          self.file = os.path.join(working_dir_or_config, _fname)
          break
      assert self.file, ValueError(
          f"Not using url/file but also not found built file with extension {BUILT_MODEL_EXT}")

    self.user_id, self.app_id = "", ""
    user_app = args.user_app
    self.url: str = args.url
    self.update_version = args.update_version

    clarifai_cfg = self.config.clarifai_model
    self.url: str = args.url
    self.id = args.id or clarifai_cfg.clarifai_model_id
    self.type = clarifai_cfg.type
    self.desc = clarifai_cfg.description
    self.infer_param = clarifai_cfg.inference_parameters
    user_app = user_app or clarifai_cfg.clarifai_user_app_id

    if user_app:
      user_app = user_app.split('/')
      assert len(
          user_app
      ) == 2, f"id must be combination of user_id and app_id separated by `/`, e.g. <user_id>/<app_id>. Got {args.id}"
      self.user_id, self.app_id = user_app

    login()

  def run(self):
    from clarifai.client import App, Model

    # Run test before uploading
    if not self.no_test:
      assert os.path.exists(self.test_path), FileNotFoundError(f"Not found {self.test_path}")
      result = subprocess.run(f"pytest -s --log-level=INFO {self.test_path}", shell=True)
      assert result.returncode == 0, "Test has failed. Please make sure no error exists in your code."

    clarifai_key_map = get_model_config(model_type=self.type).clarifai_model.field_maps
    # inference parameters
    inference_parameters = None
    if isinstance(self.infer_param, str) and os.path.isfile(self.infer_param):
      inference_parameters = InferParamManager(json_path=self.infer_param).get_list_params()
    inputs = clarifai_key_map.input_fields_map
    outputs = clarifai_key_map.output_fields_map

    # if updating new version of existing model
    def update_version():
      model = Model(model_id=self.id, app_id=self.app_id)
      if self.url:
        model.create_version_by_url(
            url=self.url,
            input_field_maps=inputs,
            output_field_maps=outputs,
            inference_parameter_configs=inference_parameters,
            description=self.desc)
      elif self.file:
        model.create_version_by_file(
            file_path=self.file,
            input_field_maps=inputs,
            output_field_maps=outputs,
            inference_parameter_configs=inference_parameters,
            no_resume=self.no_resume,
            description=self.desc)
      else:
        raise ValueError

    if self.update_version:
      update_version()
    else:
      # creating new model
      _ = App(app_id=self.app_id).create_model(self.id, model_type_id=self.type)
      update_version()
