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

from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.models.api import Models
from clarifai.models.model_serving.model_config import MODEL_TYPES, get_model_config
from clarifai.models.model_serving.model_config.inference_parameter import InferParamManager

from .base import BaseClarifaiCli


class UploadCli(BaseClarifaiCli):

  @staticmethod
  def register(parser: argparse._SubParsersAction):
    upload_parser = parser.add_parser("upload", help="Upload Clarifai model")
    upload_parser.add_argument(
        "--url", type=str, required=True, help="Direct download url of zip file")
    upload_parser.add_argument(
        "--config", type=str, required=True, help="Path to Clarifai config.yaml")
    upload_parser.set_defaults(func=UploadCli)

  def __init__(self, args: argparse.Namespace) -> None:
    self.url: str = args.url
    self.config_path: str = args.config

    # TODO: parse config
    #self._parse_config()

  def _parse_config(self):
    # do something with self.config_path
    raise NotImplementedError()

  def run(self):
    deploy(
        model_url=self.url,
        model_id=self.model_id,
        desc=self.desc,
        model_type=self.model_type,
        update_version=self.update_version,
        inference_params_file=self.infer_param)


def deploy(model_url,
           model_id: str = None,
           model_type: str = None,
           desc: str = "",
           update_version: bool = False,
           inference_params_file: str = ""):
  # init Auth from env vars
  auth = ClarifaiAuthHelper.from_env()
  # init api
  model_api = Models(auth)

  # parsing model name/type.
  # if filename having this format: <model_id>_<model-type>
  # e.i yolov5s_coco_visual-dectector
  # else user has to input model_type and model_id
  zip_filename = model_url.split('/')[-1]
  zip_filename = zip_filename.split('.')[0]

  def _parse_name(name):
    *id_, type_ = name.split('_')
    return "_".join(id_), type_

  # parse model_id
  if not model_id and "_" in zip_filename:
    model_id = _parse_name(zip_filename)[0]
  assert model_id, "Can not parse model_id from url, please input it directly"
  # parse model_type
  if not model_type and "_" in zip_filename:
    model_type = _parse_name(zip_filename)[-1]
  assert model_type, "Can not parse model_type from url, please input it directly"
  # key map
  assert model_type in MODEL_TYPES, f"model_type should be one of {MODEL_TYPES}"
  clarifai_key_map = get_model_config(model_type=model_type).field_maps
  # inference parameters
  inference_parameters = InferParamManager(json_path=inference_params_file).get_list_params()

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
