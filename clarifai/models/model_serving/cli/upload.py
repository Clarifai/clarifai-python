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
from typing import Union

from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.models.api import Models
from clarifai.models.model_serving.model_config import (MODEL_TYPES, get_model_config,
                                                        load_user_config)
from clarifai.models.model_serving.model_config.inference_parameter import InferParamManager

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
        "--url", type=str, required=True, help="Direct download url of zip file")
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
        "--update-version",
        action="store_true",
        required=False,
        help="Update exist model with new version")

    upload_parser.set_defaults(func=UploadModelSubCli)

  def __init__(self, args: argparse.Namespace) -> None:
    self.no_test = args.no_test

    working_dir_or_config = args.path
    # if input a config file, then not running test
    if working_dir_or_config.endswith(".yaml"):
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

    self.user_id, self.app_id = "", ""
    user_app = args.user_app
    self.url: str = args.url
    self.update_version = args.update_version
    assert self.url.startswith("http") or self.url.startswith(
        "s3"), f"Invalid url supported http or s3 url. Got {self.url}"

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

    if self.user_id:
      os.environ["CLARIFAI_USER_ID"] = self.user_id
    if self.app_id:
      os.environ["CLARIFAI_APP_ID"] = self.app_id

    _user_id = os.environ.get("CLARIFAI_USER_ID", None)
    _app_id = os.environ.get("CLARIFAI_APP_ID", None)
    assert _user_id or _app_id, f"Missing user-id or app-id, got user-id {_user_id} and app-id {_app_id}"
    login()

  def run(self):

    # Run test before uploading
    if not self.no_test:
      assert os.path.exists(self.test_path), FileNotFoundError(f"Not found {self.test_path}")
      result = subprocess.run(f"pytest -s --log-level=INFO {self.test_path}")
      assert result.returncode == 0, "Test has failed. Please make sure no error exists in your code."

    deploy(
        model_url=self.url,
        model_id=self.id,
        desc=self.desc,
        model_type=self.type,
        update_version=self.update_version,
        inference_parameters=self.infer_param)


def deploy(model_url,
           model_id: str = None,
           model_type: str = None,
           desc: str = "",
           update_version: bool = False,
           inference_parameters: Union[dict, str] = None):
  # init Auth from env vars
  auth = ClarifaiAuthHelper.from_env()
  # init api
  model_api = Models(auth)
  # key map
  assert model_type in MODEL_TYPES, f"model_type should be one of {MODEL_TYPES}"
  clarifai_key_map = get_model_config(model_type=model_type).clarifai_model.field_maps
  # inference parameters
  if isinstance(inference_parameters, str) and os.path.isfile(inference_parameters):
    inference_parameters = InferParamManager(json_path=inference_parameters).get_list_params()
  # if updating new version of existing model
  if update_version:
    resp = model_api.post_model_version(
        model_id=model_id,
        model_zip_url=model_url,
        input=clarifai_key_map.input_fields_map,
        outputs=clarifai_key_map.output_fields_map,
        param_specs=inference_parameters)
  # creating new model
  else:
    # post model
    resp = model_api.upload_model(
        model_id=model_id,
        model_zip_url=model_url,
        model_type=model_type,
        input=clarifai_key_map.input_fields_map,
        outputs=clarifai_key_map.output_fields_map,
        description=desc,
        param_specs=inference_parameters)
  # response
  if resp["status"]["code"] != "SUCCESS":
    raise Exception("Post models failed, details: {}, {}".format(resp["status"]["description"],
                                                                 resp["status"]["details"]))
  else:
    print("Success!")
    print(f'Model version: {resp["model"]["model_version"]["id"]}')
