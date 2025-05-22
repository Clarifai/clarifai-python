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
                    if envvar_name not in os.environ:
                        raise AttributeError(
                            f"Environment variable '{envvar_name}' not set. Attempting to load it for config '{self['name']}'. Please set it in your terminal."
                        )
                    return os.environ[envvar_name]
            else:
                value = env[key]

            if isinstance(value, dict):
                return Context(value)

            return value
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from e

    def __hasattr__(self, key):
        if key == "name":
            return True
        else:
            envvar_name = 'CLARIFAI_' + key.upper()
            return envvar_name in self['env'] or key in self['env']

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

    def to_column_names(self):
        """used for displaying on terminal."""
        keys = []
        for k in self['env'].keys():
            if k.startswith("CLARIFAI_"):
                keys.append(k.replace("CLARIFAI_", "", 1))
        return keys

    def to_stripped_lowercase(self):
        dict(self['env'])

    def to_serializable_dict(self):
        return dict(self['env'])

    def set_to_env(self):
        """sets the context env vars to the current os.environ

        Example:
          # This is helpful in scripts so you can do

          from clarifai.utils.config import Config

          Config.from_yaml().current.set_to_env()

        """
        for k, v in self['env'].items():
            if isinstance(v, dict):
                continue
            envvar_name = k.upper()
            if not envvar_name.startswith('CLARIFAI_'):
                envvar_name = 'CLARIFAI_' + envvar_name
            os.environ[envvar_name] = str(v)

    def print_env_vars(self):
        """prints the context env vars to the terminal

        Example:
          # This is helpful in scripts so you can do

          from clarifai.utils.config import Config

          Config.from_yaml().current.print_env_vars()

        """
        for k, v in sorted(self['env'].items()):
            if isinstance(v, dict):
                continue
            envvar_name = k.upper()
            if not envvar_name.startswith('CLARIFAI_'):
                envvar_name = 'CLARIFAI_' + envvar_name
            print(f"export {envvar_name}=\"{v}\"")


@dataclass
class Config:
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
            'contexts': {k: v.to_serializable_dict() for k, v in self.contexts.items()},
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
        """get the current Context"""
        return self.contexts[self.current_context]
