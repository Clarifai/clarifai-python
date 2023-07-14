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

from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.models.api import Models
from clarifai.models.model_serving.constants import MODEL_TYPES
from clarifai.models.model_serving.model_config.deploy import ClarifaiFieldsMap


def deploy(model_url,
           model_id: str = None,
           model_type: str = None,
           desc: str = "",
           update_version: bool = False):
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
  clarifai_key_map = ClarifaiFieldsMap(model_type=model_type)
  # if updating new version of existing model
  if update_version:
    resp = model_api.post_model_version(
        model_id=model_id,
        model_zip_url=model_url,
        input=clarifai_key_map.input_fields_map,
        outputs=clarifai_key_map.output_fields_map,
    )
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
    )
  # response
  if resp["status"]["code"] != "SUCCESS":
    raise Exception("Post models failed, details: {}, {}".format(resp["status"]["description"],
                                                                 resp["status"]["details"]))
  else:
    print("Success!")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  # args
  parser.add_argument("--url", type=str, required=True, help="Direct download url of zip file")
  parser.add_argument("--model_id", type=str, required=False, default="", help="Custom model id.")
  parser.add_argument(
      "--model_type",
      type=str,
      required=False,
      choices=MODEL_TYPES,
      default="",
      help="Clarifai model type")
  parser.add_argument(
      "--desc", type=str, required=False, default="", help="Short desccription of model")
  parser.add_argument(
      "--update_version",
      action="store_true",
      required=False,
      help="Update exist model with new version")
  args = parser.parse_args()
  deploy(
      model_url=args.url,
      model_id=args.model_id,
      desc=args.desc,
      model_type=args.model_type,
      update_version=args.update_version)


if __name__ == "__main__":
  main()
