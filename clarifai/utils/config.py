import os
from collections import OrderedDict
from dataclasses import dataclass, field

import yaml

from clarifai.utils.constants import DEFAULT_CONFIG


class Context(OrderedDict):
  """
    A context which has a name and a set of key-values as a dict under env.

    You can access the keys directly.
    """

  def __init__(self, name, **kwargs):
    self['name'] = name
    # when loading from config we may have the env: section in yaml already so we get it here.
    if 'env' in kwargs:
      self['env'] = kwargs['env']
    else:  # when consructing as Context(name, key=value) we set it here.
      self['env'] = kwargs

  def __getattr__(self, key):
    try:
      if key == 'name':
        return self[key]
      if key == 'env':
        raise AttributeError("Don't access .env directly")

      # Allow accessing CLARIFAI_PAT type env var names from config as .pat
      envvar_name = 'CLARIFAI_' + key.upper()
      env = self['env']
      if envvar_name in env:
        value = env[envvar_name]
        if value == "ENVVAR":
          return os.environ[envvar_name]
      else:
        value = env[key]

      if isinstance(value, dict):
        return Context(value)

      return value
    except KeyError as e:
      raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from e

  def __setattr__(self, key, value):
    if key == "name":
      self['name'] = value
    else:
      self['env'][key] = value

  def __delattr__(self, key):
    try:
      del self['env'][key]
    except KeyError as e:
      raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from e

  def to_serializable_dict(self):
    return dict(self['env'])


@dataclass
class Config():
  current_context: str
  filename: str
  contexts: OrderedDict[str, Context] = field(default_factory=OrderedDict)

  def __post_init__(self):
    for k, v in self.contexts.items():
      if 'name' not in v:
        v['name'] = k
    self.contexts = {k: Context(**v) for k, v in self.contexts.items()}

  @classmethod
  def from_yaml(cls, filename: str = DEFAULT_CONFIG):
    with open(filename, 'r') as f:
      cfg = yaml.safe_load(f)
    return cls(**cfg, filename=filename)

  def to_dict(self):
    return {
        'current_context': self.current_context,
        'contexts': {k: v.to_serializable_dict()
                     for k, v in self.contexts.items()}
    }

  def to_yaml(self, filename: str = None):
    if filename is None:
      filename = self.filename
    dir = os.path.dirname(filename)
    if len(dir):
      os.makedirs(dir, exist_ok=True)
    _dict = self.to_dict()
    for k, v in _dict['contexts'].items():
      v.pop('name', None)
    with open(filename, 'w') as f:
      yaml.safe_dump(_dict, f)

  @property
  def current(self) -> Context:
    """ get the current Context """
    return self.contexts[self.current_context]
