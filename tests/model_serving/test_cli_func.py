import os
import shutil
import tempfile
from argparse import Namespace

import pytest
import yaml

from clarifai.models.model_serving import MODEL_TYPES, load_user_config
from clarifai.models.model_serving.cli.build import BuildModelSubCli
from clarifai.models.model_serving.cli.create import SubCreateModelCli
from clarifai.models.model_serving.model_config.config import ModelTypes


def prepare_dir(func):

  def wrap(*args, **kwargs):
    try:
      input_dir = kwargs.pop(
          "input_dir", tempfile.NamedTemporaryFile(prefix="clarifai_model_serving_test_").name)
      output_dir = kwargs.pop(
          "output_dir", tempfile.NamedTemporaryFile(prefix="clarifai_model_serving_test_").name)
      output = func(input_dir=input_dir, output_dir=output_dir)
    finally:
      if os.path.exists(input_dir):
        print(f"Removing {input_dir}")
        shutil.rmtree(input_dir)
      if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    return output

  return wrap


@prepare_dir
def test_create_then_build(input_dir, output_dir):
  for model_type in MODEL_TYPES:
    working_dir = os.path.join(input_dir, model_type)
    ## Step 1. Create model
    ### Should work
    args = Namespace(
        type=model_type,
        working_dir=working_dir,
        max_bs=10,
        image_shape=[100, 100],
        from_example=None,
        example_id=None,
        overwrite=False)
    SubCreateModelCli(args).run()
    ### error due to workind dir exists
    with pytest.raises(FileExistsError):
      SubCreateModelCli(args).run()

    ## Step 2. Build

    ### Build without test
    build_args = Namespace(
        path=working_dir,
        no_test=True,
        test_path=None,
        out_path=None,
        name=None,
    )

    # But some required `labels`` models will fail due to missing labels in config
    if model_type in [
        ModelTypes.text_classifier, ModelTypes.visual_classifier, ModelTypes.visual_detector,
        ModelTypes.visual_segmenter
    ]:
      with pytest.raises(Exception):
        BuildModelSubCli(build_args).run()
      # Make labels
      cfg_file = os.path.join(working_dir, "clarifai_config.yaml")
      config = load_user_config(cfg_file)
      config.clarifai_model.labels = ["1", "2"]
      with open(cfg_file, "w") as f:
        f.write(yaml.dump(config.dump_to_user_config()))

      BuildModelSubCli(build_args).run()
    # Other models will pass
    else:
      BuildModelSubCli(build_args).run()

    ### Failed since `predict` is not implemented
    build_args = Namespace(
        path=working_dir,
        no_test=False,
        test_path=None,
        out_path=None,
        name=None,
    )
    with pytest.raises(Exception):
      BuildModelSubCli(build_args).run()
