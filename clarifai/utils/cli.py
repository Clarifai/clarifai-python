import importlib
import os
import pkgutil
import sys
import typing as t
from collections import defaultdict
from typing import OrderedDict

import click
import yaml
from tabulate import tabulate

from clarifai.utils.logging import logger


def from_yaml(filename: str):
    try:
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        click.echo(f"Error reading YAML file: {e}", err=True)
        return {}


def dump_yaml(data, filename: str):
    try:
        with open(filename, 'w') as f:
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
):
    """Display compute orchestration resources listing results using rich."""

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
        error_msg = str(e)

        # Check for common authentication errors and provide user-friendly messages
        if "PERMISSION_DENIED" in error_msg or "Unauthorized" in error_msg:
            logger.error(f"Invalid PAT token or incorrect user ID '{user_id}': {error_msg}")
        elif "UNAUTHENTICATED" in error_msg:
            logger.error(f"Invalid PAT token or user ID: {error_msg}")
        elif "SSL" in error_msg or "certificate" in error_msg:
            logger.error(f"SSL/Certificate error: {error_msg}")
        elif "Connection" in error_msg or "timeout" in error_msg:
            logger.error(f"Network connection error: {error_msg}")
        else:
            logger.error(f"❌ Validation failed: \n{error_msg}")
            logger.error("Please check your credentials and try again.")
        raise click.Abort()  # Exit without saving the configuration


def customize_ollama_model(model_path, model_name, port, context_length):
    """Customize the Ollama model name in the cloned template files.
    Args:
     model_path: Path to the cloned model directory
     model_name: The model name to set (e.g., 'llama3.1', 'mistral')

    """
    model_py_path = os.path.join(model_path, "1", "model.py")

    if not os.path.exists(model_py_path):
        logger.warning(f"Model file {model_py_path} not found, skipping model name customization")
        return

    try:
        # Read the model.py file
        with open(model_py_path, 'r') as file:
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

        # Write the modified content back to model.py
        with open(model_py_path, 'w') as file:
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


def check_requirements_installed(model_path):
    """Check if all dependencies in requirements.txt are installed."""
    import re
    from pathlib import Path

    requirements_path = Path(model_path) / "requirements.txt"

    if not requirements_path.exists():
        logger.warning(f"requirements.txt not found at {requirements_path}")
        return True

    try:
        package_pattern = re.compile(r'^([a-zA-Z0-9_-]+)')

        # Getting package name and version (for logging)
        requirements = [
            (match.group(1), pack)
            for line in requirements_path.read_text().splitlines()
            if (pack := line.strip())
            and not line.startswith('#')
            and (match := package_pattern.match(line))
        ]

        if not requirements:
            logger.info("No dependencies found in requirements.txt")
            return True

        logger.info(f"Checking {len(requirements)} dependencies...")

        missing = [
            full_req
            for package_name, full_req in requirements
            if not _is_package_installed(package_name)
        ]

        if not missing:
            logger.info(f"✅ All {len(requirements)} dependencies are installed!")
            return True

        # Report missing packages
        logger.error(
            f"❌ {len(missing)} of {len(requirements)} required packages are missing in the current environment"
        )
        logger.error("\n".join(f"  - {pkg}" for pkg in missing))
        logger.warning(f"To install: pip install -r {requirements_path}")
        return False

    except Exception as e:
        logger.error(f"Failed to check requirements: {e}")
        return False
