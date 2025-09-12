import os
from collections import OrderedDict
from dataclasses import dataclass, field

import yaml

from clarifai.utils.constants import DEFAULT_BASE, DEFAULT_CONFIG, DEFAULT_UI
from clarifai.utils.logging import logger


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
        """Get the key from the config. You can pass a lowercase key like "pat" and it will check if
        the environment variable CLARIFAI_PAT set and use that first. If no env var, then it checks
        if that env var name is in the config and use that. If not then checks if
        "pat" is in the config, if not then it falls back to CLARIFAI_PAT in the environment
        variables, else raises an AttributeError.
        """
        try:
            if key == 'name':
                return self[key]
            if key == 'env':
                raise AttributeError("Don't access .env directly")

            # Allow accessing CLARIFAI_PAT type env var names from config as .pat
            envvar_name = 'CLARIFAI_' + key.upper()
            env = self['env']
            if envvar_name in os.environ:  # environment variable take precedence.
                value = os.environ[envvar_name]
            elif envvar_name in env:
                value = env[envvar_name]
                if value == "ENVVAR":
                    if envvar_name not in os.environ:
                        raise AttributeError(
                            f"Environment variable '{envvar_name}' not set. Attempting to load it for config '{self['name']}'. Please set it in your terminal."
                        )
                    return os.environ[envvar_name]
            elif key in env:  # check if key is in the config
                value = env[key]
            # below are some default fallback values for UI and API base.
            elif envvar_name == 'CLARIFAI_UI':
                value = DEFAULT_UI
            elif envvar_name == 'CLARIFAI_API_BASE':
                value = DEFAULT_BASE
            else:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{key}' or '{envvar_name}' and '{envvar_name}' is also not in os.environ:"
                )

            if isinstance(value, dict):
                return Context(value)

            return value
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from e

    def get(self, key, default=None):
        """Get the key from the config. You can pass a lowercase
        key like "pat" and it will check if the environment variable CLARIFAI_PAT set and use that
        first.
        If no env var, then it checks if that env var name is in the config and use that.
        If not then checks if "pat" is in the config, if not then it falls back to CLARIFAI_PAT in
        the environment variables, else returns the default value.
        """
        try:
            return self.__getattr__(key)
        except AttributeError:
            return default

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
        """Loads the configuration from a YAML file.
        If the file does not exist, it initializes with empty config.
        """
        cfg = {"current_context": "_empty_", "contexts": {"_empty_": {}}}
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                cfg = yaml.safe_load(f)
        else:
            logger.warning(
                f"Config file {filename} not found, using default config. Run 'clarifai config' on the command line to create a config file."
            )
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
        """Get the current Context or an empty one if your config is not setup."""
        if not self.current_context:
            logger.warning(
                "No current context set, returning empty context. Run 'clarifai config' on the command line to create a config file."
            )
            return Context("_empty_")
        return self.contexts[self.current_context]
