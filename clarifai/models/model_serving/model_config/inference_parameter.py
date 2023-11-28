import json
from dataclasses import asdict, dataclass, field
from typing import Any, List


@dataclass(frozen=True)
class InferParamType:
  BOOL: int = 1
  STRING: int = 2
  NUMBER: int = 3
  ENCRYPTED_STRING: int = 21


@dataclass
class InferParam:
  path: str
  field_type: InferParamType = field(default_factory=InferParamType)
  default_value: Any = None
  description: str = ""

  def __post_init__(self):
    assert self.path.isidentifier(
    ), f"`path` must be valid for creating python variable, got {self.path}"
    if self.default_value is not None:
      self.validate_type(self.default_value)

  def validate_type(self, value):
    if self.field_type == InferParamType.BOOL:
      assert isinstance(value, bool), f"`field_type` is `BOOL` (bool), however got {type(value)}"
    elif self.field_type == InferParamType.NUMBER:
      assert isinstance(value, float) or isinstance(
          value, int), f"`field_type` is `NUMBER` (float or int), however got {type(value)}"
    else:
      assert isinstance(
          value,
          str), f"`field_type` is `STRING` or `ENCRYPTED_STRING` (str), however got {type(value)}"

  def todict(self):
    return {k: v for k, v in asdict(self).items()}


@dataclass
class InferParamManager:
  json_path: str = ""
  params: List[InferParam] = field(default_factory=list)
  _dict_params: dict = field(init=False)

  @classmethod
  def from_kwargs(cls, **kwargs):
    params = list()
    for k, v in kwargs.items():
      if isinstance(v, str) and k.startswith("_"):
        _type = InferParamType.ENCRYPTED_STRING
      elif isinstance(v, str):
        _type = InferParamType.STRING
      elif isinstance(v, bool):
        _type = InferParamType.BOOL
      elif isinstance(v, float) or isinstance(v, int):
        _type = InferParamType.NUMBER
      else:
        raise TypeError(f"Unsupported type {type(v)} of argument {k}, support {InferParamType}")
      param = InferParam(path=k, field_type=_type, default_value=v, description=k)
      params.append(param)

    return cls(params=params)

  def __post_init__(self):
    #assert self.params == [] or self.json_path, "`json_path` or `params` must be set"
    self._dict_params = dict()
    if self.params == [] and self.json_path:
      with open(self.json_path, "r") as fp:
        objs = json.load(fp)
        objs = objs if isinstance(objs, list) else [objs]
      self.params = [InferParam(**obj) for obj in objs]
    for param in self.params:
      self._dict_params.update({param.path: param})

  def get_list_params(self):
    list_params = []
    for each in self.params:
      list_params.append(each.todict())
    return list_params

  def export(self, path: str):
    list_params = self.get_list_params()
    with open(path, "w") as fp:
      json.dump(list_params, fp, indent=2)

  def validate(self, **kwargs) -> dict:
    output_kwargs = {k: v.default_value for k, v in self._dict_params.items()}
    assert kwargs == {} or self.params != [], "kwargs are rejected since `params` is empty"

    for key, value in kwargs.items():
      assert key in self._dict_params, f"param `{key}` is not in setting: {list(self._dict_params.keys())}"
      if key in self._dict_params:
        self._dict_params[key].validate_type(value)
      output_kwargs.update({key: value})
    return output_kwargs


def is_number(v: str):
  try:
    _ = float(v)
    return True
  except ValueError:
    return False


def str_to_number(v: str):
  try:
    return int(v)
  except ValueError:
    return float(v)


def parse_req_parameters(req_params: str):
  req_params = json.loads(req_params)
  for k, v in req_params.items():
    if isinstance(v, str):
      if is_number(v):
        v = str_to_number(v)
    req_params.update({k: v})

  return req_params
