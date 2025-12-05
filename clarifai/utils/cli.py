import importlib
import os
import pkgutil
import sys
import typing as t
from collections import defaultdict
from pathlib import Path
from typing import Optional, OrderedDict

import click
import yaml
from google.protobuf.timestamp_pb2 import Timestamp
from tabulate import tabulate

from clarifai.utils.logging import logger


def from_yaml(filename: str):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        click.echo(f"Error reading YAML file: {e}", err=True)
        return {}


def dump_yaml(data, filename: str):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(data, f)
    except Exception as e:
        click.echo(f"Error writing YAML file: {e}", err=True)


# Dynamically find and import all command modules from the cli directory
def load_command_modules():
    package_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cli')

    for _, module_name, _ in pkgutil.iter_modules([package_dir]):
        if module_name not in ['base', '__main__']:  # Skip the base.py and __main__ file itself
            importlib.import_module(f'clarifai.cli.{module_name}')


def display_co_resources(
    response,
    custom_columns={
        'ID': lambda c: c.id,
        'USER_ID': lambda c: c.user_id,
        'DESCRIPTION': lambda c: c.description,
    },
    sort_by_columns=None,
):
    """
    Display compute orchestration resources listing results using rich.

    :param response: Iterable of resource objects to display.
    :param custom_columns: A dictionary mapping column names to extractor functions.
                           Defaults to ID, USER_ID, and DESCRIPTION.
    :param sort_by_columns: Optional list of (column_name, order) tuples specifying sort order.
                            Only column names present in `custom_columns` are considered.
                            Order should be 'asc' or 'desc'.
    :return: None. Prints the formatted table.
    """
    if sort_by_columns:
        for column, order in reversed(sort_by_columns):  # reversed for stable multi-level sort
            if column in custom_columns:
                reverse = order.lower() == 'desc'
                response = sorted(response, key=custom_columns[column], reverse=reverse)

    formatter = TableFormatter(custom_columns)
    print(formatter.format(list(response), fmt="plain"))


class TableFormatter:
    def __init__(self, custom_columns: OrderedDict):
        """
        Initializes the TableFormatter with column headers and custom column mappings.

        :param headers: List of column headers for the table.
        """
        self.custom_columns = custom_columns

    def format(self, objects, fmt='plain'):
        """
        Formats a list of objects into a table with custom columns.

        :param objects: List of objects to format into a table.
        :return: A string representing the table.
        """
        # Prepare the rows by applying the custom column functions to each object
        rows = []
        for obj in objects:
            #   row = [self.custom_columns[header](obj) for header in self.headers]
            row = [f(obj) for f in self.custom_columns.values()]
            rows.append(row)

        # Create the table
        table = tabulate(rows, headers=self.custom_columns.keys(), tablefmt=fmt)
        return table


class AliasedGroup(click.Group):
    def __init__(
        self,
        name: t.Optional[str] = None,
        commands: t.Optional[
            t.Union[t.MutableMapping[str, click.Command], t.Sequence[click.Command]]
        ] = None,
        **attrs: t.Any,
    ) -> None:
        super().__init__(name, commands, **attrs)
        self.alias_map = {}
        self.command_to_aliases = defaultdict(list)

    def add_alias(self, cmd: click.Command, alias: str) -> None:
        self.alias_map[alias] = cmd
        if alias != cmd.name:
            self.command_to_aliases[cmd].append(alias)

    def command(
        self, aliases=None, *args, **kwargs
    ) -> t.Callable[[t.Callable[..., t.Any]], click.Command]:
        cmd_decorator = super().command(*args, **kwargs)
        if aliases is None:
            aliases = []

        def aliased_decorator(f):
            cmd = cmd_decorator(f)
            if cmd.name:
                self.add_alias(cmd, cmd.name)
            for alias in aliases:
                self.add_alias(cmd, alias)
            return cmd

        f = None
        if args and callable(args[0]):
            (f,) = args
        if f is not None:
            return aliased_decorator(f)
        return aliased_decorator

    def group(
        self, aliases=None, *args, **kwargs
    ) -> t.Callable[[t.Callable[..., t.Any]], click.Group]:
        cmd_decorator = super().group(*args, **kwargs)
        if aliases is None:
            aliases = []

        def aliased_decorator(f):
            cmd = cmd_decorator(f)
            if cmd.name:
                self.add_alias(cmd, cmd.name)
            for alias in aliases:
                self.add_alias(cmd, alias)
            return cmd

        f = None
        if args and callable(args[0]):
            (f,) = args
        if f is not None:
            return aliased_decorator(f)
        return aliased_decorator

    def get_command(self, ctx: click.Context, cmd_name: str) -> t.Optional[click.Command]:
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        return self.alias_map.get(cmd_name)

    def format_commands(self, ctx, formatter):
        sub_commands = self.list_commands(ctx)

        rows = []
        for sub_command in sub_commands:
            cmd = self.get_command(ctx, sub_command)
            if cmd is None or getattr(cmd, 'hidden', False):
                continue
            if cmd in self.command_to_aliases:
                aliases = ', '.join(self.command_to_aliases[cmd])
                sub_command = f'{sub_command} ({aliases})'
            cmd_help = cmd.help
            rows.append((sub_command, cmd_help))

        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


def validate_context(ctx):
    from clarifai.utils.logging import logger

    if ctx.obj == {}:
        logger.error("CLI config file missing. Run `clarifai login` to set up the CLI config.")
        sys.exit(1)


def validate_context_auth(pat: str, user_id: str, api_base: str = None):
    """
    Validate a Personal Access Token (PAT) by making a test API call.

    Args:
        pat (str): The Personal Access Token to validate
        user_id (str): The user ID associated with the token
        api_base (str): The API base URL. Defaults to None (uses default).
    """
    try:
        from clarifai_grpc.grpc.api.status import status_code_pb2

        from clarifai.client.user import User

        logger.info("Validating the Context Credentials...")

        # Create user client for validation
        if api_base:
            user_client = User(user_id=user_id, pat=pat, base_url=api_base)
        else:
            user_client = User(user_id=user_id, pat=pat)

        # Try to get user info as a test API call
        response = user_client.get_user_info()

        if response.status.code == status_code_pb2.SUCCESS:
            logger.info("✅ Context is valid")

    except Exception as e:
        # Check for common authentication errors and provide user-friendly messages
        logger.error("❌ Authentication failed. Please check your token and user ID.")
        raise click.Abort()  # Exit without saving the configuration


def customize_ollama_model(
    model_path, user_id, model_name=None, port=None, context_length=None, verbose=False
):
    """Customize the Ollama model name in the cloned template files.
    Args:
     model_path: Path to the cloned model directory
     model_name: The model name to set (e.g., 'llama3.1', 'mistral') - optional
     port: Port for Ollama server - optional
     context_length: Context length for the model - optional
     verbose: Whether to enable verbose logging - optional (defaults to False)

    """
    config_path = os.path.join(model_path, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Update the user_id in the model section
        config['model']['user_id'] = user_id
        if 'toolkit' not in config or config['toolkit'] is None:
            config['toolkit'] = {}
        if model_name is not None:
            config['toolkit']['model'] = model_name
        if port is not None:
            config['toolkit']['port'] = port
        if context_length is not None:
            config['toolkit']['context_length'] = context_length
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    model_py_path = os.path.join(model_path, "1", "model.py")

    if not os.path.exists(model_py_path):
        logger.warning(f"Model file {model_py_path} not found, skipping model name customization")
        return

    try:
        # Read the model.py file
        with open(model_py_path, 'r', encoding='utf-8') as file:
            content = file.read()
        if model_name:
            # Replace the default model name in the load_model method
            content = content.replace(
                'self.model = os.environ.get("OLLAMA_MODEL_NAME", \'llama3.2\')',
                f'self.model = os.environ.get("OLLAMA_MODEL_NAME", \'{model_name}\')',
            )

        if port:
            # Replace the default port variable in the model.py file
            content = content.replace("PORT = '23333'", f"PORT = '{port}'")

        if context_length:
            # Replace the default context length variable in the model.py file
            content = content.replace(
                "context_length = '8192'", f"context_length = '{context_length}'"
            )

        verbose_str = str(verbose)
        if "VERBOSE_OLLAMA = True" in content:
            content = content.replace("VERBOSE_OLLAMA = True", f"VERBOSE_OLLAMA = {verbose_str}")
        elif "VERBOSE_OLLAMA = False" in content:
            content = content.replace("VERBOSE_OLLAMA = False", f"VERBOSE_OLLAMA = {verbose_str}")

        # Write the modified content back to model.py
        with open(model_py_path, 'w', encoding='utf-8') as file:
            file.write(content)

    except Exception as e:
        logger.error(f"Failed to customize Ollama model name in {model_py_path}: {e}")
        raise


def check_ollama_installed():
    """Check if the Ollama CLI is installed."""
    try:
        import subprocess

        result = subprocess.run(
            ['ollama', '--version'], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            return True
        else:
            return False
    except FileNotFoundError:
        return False


def check_lmstudio_installed():
    """Check if the LM Studio CLI is installed."""
    try:
        import subprocess

        result = subprocess.run(['lms', 'version'], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return True
        else:
            return False
    except FileNotFoundError:
        return False


def _is_package_installed(package_name):
    """Helper function to check if a single package in requirements.txt is installed."""
    import importlib.metadata

    try:
        importlib.metadata.distribution(package_name)
        logger.debug(f"✅ {package_name} - installed")
        return True
    except importlib.metadata.PackageNotFoundError:
        logger.debug(f"❌ {package_name} - not installed")
        return False
    except Exception as e:
        logger.warning(f"Error checking {package_name}: {e}")
        return False


def parse_requirements(model_path: str):
    """Parse requirements.txt in the model directory and return a dictionary of dependencies."""
    from packaging.requirements import Requirement

    requirements_path = Path(model_path) / "requirements.txt"

    if not requirements_path.exists():
        logger.warning(f"requirements.txt not found at {requirements_path}")
        return []

    deps = {}
    for line in requirements_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            req = Requirement(line)
            deps[req.name] = str(req.specifier) if req.specifier else None
        except Exception as e:
            logger.warning(f"⚠️ Could not parse line: {line!r} — {e}")
    return deps


def check_requirements_installed(model_path: str = None, dependencies: dict = None):
    """Check if all dependencies in requirements.txt are installed.
    Args:
        model_path: Path to the model directory
        dependencies: Dictionary of dependencies
    Returns:
        True if all dependencies are installed, False otherwise
    """

    if model_path and dependencies:
        logger.warning(
            "model_path and dependencies cannot be provided together, using dependencies instead"
        )
        dependencies = parse_requirements(model_path)

    try:
        if not dependencies:
            dependencies = parse_requirements(model_path)
        missing = [
            full_req
            for package_name, full_req in dependencies.items()
            if not _is_package_installed(package_name)
        ]

        if not missing:
            logger.info(f"✅ All {len(dependencies)} dependencies are installed!")
            return True

        # Report missing packages
        logger.error(
            f"❌ {len(missing)} of {len(dependencies)} required packages are missing in the current environment"
        )
        logger.error("\n".join(f"  - {pkg}" for pkg in missing))
        requirements_path = Path(model_path) / "requirements.txt"
        logger.warning(f"To install: pip install -r {requirements_path}")
        return False

    except Exception as e:
        logger.error(f"Failed to check requirements: {e}")
        return False


def convert_timestamp_to_string(timestamp: Timestamp) -> str:
    """Converts a Timestamp object to a string.

    Args:
        timestamp (Timestamp): The Timestamp object to convert.

    Returns:
        str: The converted string in ISO 8601 format.
    """
    if not timestamp:
        return ""

    datetime_obj = timestamp.ToDatetime()

    return datetime_obj.strftime('%Y-%m-%dT%H:%M:%SZ')


def customize_huggingface_model(model_path, user_id, model_name):
    config_path = os.path.join(model_path, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Update the user_id in the model section
        config['model']['user_id'] = user_id

        if model_name:
            # Update the repo_id in checkpoints section
            if 'checkpoints' not in config:
                config['checkpoints'] = {}
            config['checkpoints']['repo_id'] = model_name

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Updated Hugging Face model repo_id to: {model_name}")
    else:
        logger.warning(f"config.yaml not found at {config_path}, skipping model configuration")


def customize_lmstudio_model(model_path, user_id, model_name=None, port=None, context_length=None):
    """Customize the LM Studio model name in the cloned template files.
    Args:
     model_path: Path to the cloned model directory
     model_name: The model name to set (e.g., 'qwen/qwen3-4b-thinking-2507') - optional
     port: Port for LM Studio server - optional
     context_length: Context length for the model - optional

    """
    config_path = os.path.join(model_path, 'config.yaml')

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # Update the user_id in the model section
        config['model']['user_id'] = user_id
        if 'toolkit' not in config or config['toolkit'] is None:
            config['toolkit'] = {}
        if model_name is not None:
            config['toolkit']['model'] = model_name
        if port is not None:
            config['toolkit']['port'] = port
        if context_length is not None:
            config['toolkit']['context_length'] = context_length
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Updated LM Studio model configuration in: {config_path}")
    else:
        logger.warning(f"config.yaml not found at {config_path}, skipping model configuration")


def prompt_required_field(message: str, default: Optional[str] = None) -> str:
    """Prompt the user for a required field, optionally with a default.
    Used inside the model CLI to prompt the user for required fields if config.yaml is missing.
    Args:
        message (str): The message to display to the user.
        default (Optional[str]): The default value to use if the user does not enter a value.

    Returns:
        str: The value entered by the user.
    """
    while True:
        prompt = f"{message}"
        if default:
            prompt += f" [{default}]"
        prompt += ": "
        value = input(prompt).strip()
        if not value and default:
            return default
        if value:
            return value
        click.echo("❌ This field is required. Please enter a value.")


def prompt_optional_field(message: str, default: Optional[str] = None) -> Optional[str]:
    """Prompt the user for an optional field.
    Used inside the model CLI to prompt the user for optional fields if config.yaml is missing.
    Args:
        message (str): The message to display to the user.
        default (Optional[str]): The default value to use if the user does not enter a value.

    Returns:
        Optional[str]: The value entered by the user.
    """
    prompt = f"{message}"
    if default:
        prompt += f" [{default}]"
    prompt += ": "
    value = input(prompt).strip()
    if not value:
        return default
    return value


def prompt_int_field(message: str, default: Optional[int] = None) -> int:
    """Prompt the user for an integer field (required).
    Used inside the model CLI to prompt the user for integer fields if config.yaml is missing.
    Args:
        message (str): The message to display to the user.
        default (Optional[int]): The default value to use if the user does not enter a value.

    Returns:
        int: The value entered by the user.
    """
    while True:
        prompt = f"{message}"
        if default is not None:
            prompt += f" [{default}]"
        prompt += ": "
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            click.echo("❌ Please enter a valid integer.")


def prompt_yes_no(message: str, default: Optional[bool] = None) -> bool:
    """Prompt the user for a yes/no decision.
    Used inside the model CLI to prompt the user for yes/no fields if config.yaml is missing.
    Args:
        message (str): The message to display to the user.
        default (Optional[bool]): The default value to use if the user does not enter a value.

    Returns:
        bool: The value entered by the user.
    """
    if default is True:
        suffix = " [Y/n]"
    elif default is False:
        suffix = " [y/N]"
    else:
        suffix = " [y/n]"
    prompt = f"{message}{suffix}: "
    while True:
        response = input(prompt).strip().lower()
        if not response and default is not None:
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        click.echo("❌ Please respond with 'y' or 'n'.")


def print_section(title: str, description: str, note: Optional[str] = None) -> None:
    """Print a section with a title, description, and note.
    Used inside the model CLI to print sections if config.yaml is missing.
    Args:
        title (str): The title of the section.
        description (str): The description of the section.
        note (Optional[str]): The note to display below the section.
    """
    click.echo()
    click.echo(click.style(title, fg="bright_cyan", bold=True))
    if description:
        click.echo(description)
    if note:
        click.echo(click.style(note, fg="yellow"))


def print_field_help(name: str, description: str) -> None:
    """Print a field with a name and description.
    Used inside the model CLI to print fields if config.yaml is missing.
    Args:
        name (str): The name of the field.
        description (str): The description of the field.
    """
    click.echo(click.style(f"➤ {name}", fg="bright_green", bold=True))
    if description:
        click.echo(click.style(f"    {description}", fg="green"))
