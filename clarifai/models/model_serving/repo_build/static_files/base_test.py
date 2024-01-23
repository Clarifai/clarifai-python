import os
from typing import Dict, Iterable, List, Union

import numpy as np
import yaml

from ...model_config import InferParam, InferParamManager, load_user_config


class BaseTest:
  init_inference_parameters = {}

  def __init__(self) -> None:
    import inference
    from inference import InferenceModel
    self.model = InferenceModel()
    self._base_dir = os.path.dirname(inference.__file__)
    self.cfg_path = os.path.join(self._base_dir, "clarifai_config.yaml")
    self.user_config = load_user_config(self.cfg_path)

    user_defined_infer_params = [
        InferParam(**each) for each in self.user_config.clarifai_model.inference_parameters
    ]
    # update init vs user_defined params
    total_infer_params = []
    for k, v in self.init_inference_parameters.items():
      _exist = False
      for user_param in user_defined_infer_params:
        if user_param.path == k:
          print(f"Warning: Overwrite parameter `{k}` with default value `{v}`")
          user_param.default_value = v
          _exist = True
          total_infer_params.append(user_param)
          user_defined_infer_params.remove(user_param)
          break
      if not _exist:
        total_infer_params.append(InferParamManager.from_kwargs(**{k: v}).params[0])

    self.infer_param_manager = InferParamManager(
        params=total_infer_params + user_defined_infer_params)
    self.user_config.clarifai_model.inference_parameters = self.infer_param_manager.get_list_params(
    )
    self._overwrite_cfg()

  def _overwrite_cfg(self):
    config = yaml.dump(self.user_config.dump_to_user_config(),)
    with open(self.cfg_path, "w") as f:
      f.write(config)

  def predict(self, input_data: Union[List[np.ndarray], List[str], Dict[str, Union[List[
      np.ndarray], List[str]]]], **inference_paramters) -> Iterable:
    """
    Test Prediction method is exact `InferenceModel.predict` method with
    checking inference paramters.

    Args:
    -----
    - input_data: A list of input data item to predict on. The type depends on model input type:
      * `image`: List[np.ndarray]
      * `text`: List[str]
      * `multimodal`:
        input_data is list of dict where key is input type name e.i. `image`, `text` and value is list.
        {"image": List[np.ndarray], "text": List[str]}

    - inference_paramters: your inference parameterss.

    Returns:
    --------
      List of your inference model output type
    """
    infer_params = self.infer_param_manager.validate(**inference_paramters)
    return self.model.predict(input_data=input_data, inference_paramters=infer_params)
