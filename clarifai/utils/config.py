import click
import sys
import os
import typing as t
import yaml

from dataclasses import dataclass, field
from clarifai.utils.constants import DEFAULT_CONFIG


@dataclass
class AccessToken():
  type: str
  value: str

  def __str__(self):
    return f'{self.type}:{self.value}' if self.type == 'env' else '********'

  def to_serializable_dict(self):
    return self.__dict__

  @classmethod
  def from_serializable_dict(cls, _dict):
    return cls(**_dict)

  # Dictionary protocol methods:

  def __getitem__(self, key: str) -> t.Any:
    if key == 'type':
      return self.type
    elif key == 'value':
      return self.value
    else:
      raise KeyError(key)

  def __setitem__(self, key: str, value: t.Any) -> None:
    if key == 'type':
      self.type = value
    elif key == 'value':
      self.value = value
    else:
      raise KeyError(key)

  def __delitem__(self, key: str) -> None:
    raise TypeError("Cannot delete attributes from AccessToken")

  def __iter__(self):
    return iter(['type', 'value'])

  def __len__(self) -> int:
    return 2

  def __contains__(self, key: str) -> bool:
    return key in ['type', 'value']

  def keys(self):
    return ['type', 'value']

  def values(self):
    return [self.type, self.value]

  def items(self):
    return [('type', self.type), ('value', self.value)]

  def get(self, key: str, default: t.Any = None) -> t.Any:
    try:
      return self[key]
    except KeyError:
      return default


@dataclass
class Context():
  name: str
  user_id: str
  base_url: str
  access_token: AccessToken = field(default_factory=lambda: AccessToken('env', 'CLARIFAI_PAT'))
  env: t.Dict[str, str] = field(default_factory=dict)

  pat: str = None

  def _resolve_pat(self) -> str:
    if self.access_token['type'].lower() == 'env':
      return os.getenv(self.access_token['value'], '')
    elif self.access_token['type'].lower() == 'raw':
      return self.access_token['value']
    else:
      raise Exception('Only "env" and "raw" methods are supported')

  def __post_init__(self):
    self.pat = self._resolve_pat()
    self.access_token = AccessToken(**self.access_token)

  def to_serializable_dict(self):
    result = {
        'name': self.name,
        'user_id': self.user_id,
        'base_url': self.base_url,
        'access_token': self.access_token.to_serializable_dict(),
    }
    if self.env:
      result['env'] = self.env
    return result


@dataclass
class Config():
  current_context: str
  filename: str
  contexts: dict[str, Context] = field(default_factory=dict)

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

  def current(self) -> Context:
    return self.contexts[self.current_context]
