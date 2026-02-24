import json
import os
import re
import shutil
import socket
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Optional

import click
import yaml

from clarifai.cli.base import cli
from clarifai.errors import UserError
from clarifai.utils.cli import (
    AliasedGroup,
    check_lmstudio_installed,
    check_ollama_installed,
    check_requirements_installed,
    customize_huggingface_model,
    customize_lmstudio_model,
    customize_ollama_model,
    parse_requirements,
    print_field_help,
    print_section,
    prompt_int_field,
    prompt_optional_field,
    prompt_required_field,
    prompt_yes_no,
    validate_context,
)
from clarifai.utils.config import Config, Context
from clarifai.utils.constants import (
    CLI_LOGIN_DOC_URL,
    CONFIG_GUIDE_URL,
    DEFAULT_LOCAL_RUNNER_APP_ID,
    DEFAULT_LOCAL_RUNNER_COMPUTE_CLUSTER_CONFIG,
    DEFAULT_LOCAL_RUNNER_COMPUTE_CLUSTER_ID,
    DEFAULT_LOCAL_RUNNER_MODEL_TYPE,
    DEFAULT_LOCAL_RUNNER_NODEPOOL_CONFIG,
    DEFAULT_LOCAL_RUNNER_NODEPOOL_ID,
)
from clarifai.utils.logging import logger


def find_available_port(start_port=8080):
    """Find the first available port starting from start_port."""
    port = start_port
    while port <= 65535:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(('', port))
                return port
            except OSError:
                port += 1
    raise RuntimeError("No available port found")


def _select_context(ctx_config: Config) -> Optional[Context]:
    contexts_map = getattr(ctx_config, "contexts", {}) or {}
    if not contexts_map:
        return None

    context_names = [name for name in contexts_map.keys() if name and name != "_empty_"]
    if not context_names:
        return None

    current_name = getattr(ctx_config, "current_context", None)

    click.echo()
    click.echo(click.style("Available Clarifai CLI contexts:", fg="bright_cyan", bold=True))
    for idx, name in enumerate(context_names, start=1):
        marker = click.style(" (current)", fg="yellow") if name == current_name else ""
        click.echo(f"  [{idx}] {name}{marker}")
    create_idx = len(context_names) + 1
    click.echo(f"  [{create_idx}] Create new context")

    selection = input(
        "Enter the context number or name to use (press Enter to keep current): "
    ).strip()
    if selection == "":
        return contexts_map.get(current_name)
    if selection.isdigit() and int(selection) == create_idx:
        click.echo(click.style("Launching `clarifai login` to create a new context.", fg="yellow"))
        try:
            subprocess.run(["clarifai", "login"], check=True)
        except subprocess.CalledProcessError as exc:
            click.echo(
                click.style(
                    "`clarifai login` exited with an error. Continuing with existing contexts.",
                    fg="red",
                ),
                err=True,
            )
            return contexts_map.get(current_name)

        try:
            refreshed = Config.from_yaml(filename=ctx_config.filename)
            ctx_config.contexts = refreshed.contexts
            ctx_config.current_context = refreshed.current_context
            ctx_config.to_yaml()
        except Exception as exc:
            click.echo(
                click.style(
                    "Failed to reload contexts after login. Continuing with existing contexts.",
                    fg="red",
                ),
                err=True,
            )
            logger.debug(f"Unable to reload contexts after login: {exc}")
            return contexts_map.get(current_name)
        return ctx_config.contexts.get(ctx_config.current_context)

    if not selection:
        return contexts_map.get(current_name)

    chosen_name: Optional[str] = None
    if selection.isdigit():
        idx = int(selection)
        if 1 <= idx <= len(context_names):
            chosen_name = context_names[idx - 1]
    elif selection in context_names:
        chosen_name = selection

    if not chosen_name:
        click.echo(
            click.style(
                "Unrecognized selection. Continuing with the current context.",
                fg="yellow",
            )
        )
        return contexts_map.get(current_name)

    if chosen_name != current_name:
        ctx_config.current_context = chosen_name
        try:
            ctx_config.to_yaml()
        except Exception as exc:
            logger.debug(f"Unable to context switch: {exc}")
        click.echo(click.style(f"Using context '{chosen_name}' for this upload.", fg="green"))

    return ctx_config.contexts[chosen_name]


def ensure_config_exists_for_upload(ctx, model_path: str) -> None:
    """Ensure config.yaml exists before attempting upload; create interactively if missing."""
    config_path = os.path.join(model_path, "config.yaml")
    if os.path.exists(config_path):
        try:
            is_empty = os.path.getsize(config_path) == 0
            if not is_empty:
                with open(config_path, "r", encoding="utf-8") as config_file:
                    data = config_file.read().strip()
                if data:
                    return
        except OSError:
            pass

    click.echo(
        click.style(
            "⚠️  config.yaml is missing or empty in this model directory.",
            fg="yellow",
        )
    )
    # field which asks whether the user wants to create a new config.yaml file by themselves using the guide or not
    create_config = prompt_yes_no(
        "Do you want to create a new config.yaml file by yourself using the guide?",
        default=True,
    )
    if create_config:
        click.echo(
            click.style(
                f"Please refer to the config guide -> {CONFIG_GUIDE_URL} to create a new config.yaml file.",
                fg="yellow",
            )
        )
        raise click.Abort()

    else:
        click.echo(
            click.style(
                f"ℹ️  We'll create one now interactively. For more details on each field, check out the config guide at -> {CONFIG_GUIDE_URL} ",
                fg="yellow",
            )
        )

    ctx_config = getattr(ctx, "obj", None)
    current_context = None
    if ctx_config is not None:
        contexts_map = getattr(ctx_config, "contexts", {}) or {}
        current_name = getattr(ctx_config, "current_context", None)
        current_context = contexts_map.get(current_name)

    if current_context is None:
        click.echo(
            click.style(
                "No Clarifai CLI context detected. Launching `clarifai login` to configure one now.",
                fg="yellow",
            )
        )
        try:
            subprocess.run(["clarifai", "login"], check=True)
        except subprocess.CalledProcessError as exc:
            click.echo(
                click.style(
                    "`clarifai login` exited with an error. Aborting upload.",
                    fg="red",
                ),
                err=True,
            )
            raise click.Abort() from exc

        config_filename = getattr(ctx_config, "filename", None) if ctx_config else None
        try:
            refreshed_config = (
                Config.from_yaml(filename=config_filename)
                if config_filename
                else Config.from_yaml()
            )
        except Exception as exc:
            click.echo(
                click.style(
                    "Unable to reload Clarifai CLI configuration after login.",
                    fg="red",
                ),
                err=True,
            )
            raise click.Abort() from exc

        ctx.obj = refreshed_config
        ctx_config = refreshed_config
        contexts_map = getattr(ctx_config, "contexts", {}) or {}
        current_name = getattr(ctx_config, "current_context", None)
        current_context = contexts_map.get(current_name)

        if current_context is None:
            click.echo(
                click.style(
                    "Login did not create a usable context. Please run `clarifai login` manually and retry.",
                    fg="red",
                ),
                err=True,
            )
            raise click.Abort()

    if ctx_config is not None:
        selected_context = _select_context(ctx_config)
        if selected_context is not None:
            current_context = selected_context
        elif current_context is None:
            contexts_map = getattr(ctx_config, "contexts", {}) or {}
            current_context = contexts_map.get(getattr(ctx_config, "current_context", None))

    if current_context is None:
        click.echo(
            click.style(
                "No Clarifai context available. Please run `clarifai login` and retry.",
                fg="red",
            ),
            err=True,
        )
        raise click.Abort()

    default_user_id = getattr(current_context, "user_id", None)
    default_app_id = getattr(current_context, "app_id", None)
    current_pat = getattr(current_context, "pat", None)

    default_model_id = os.path.basename(os.path.abspath(model_path))
    if not default_model_id or default_model_id == os.path.sep:
        default_model_id = "my-model"

    try:
        print_section(
            "Clarifai Authentication",
            (
                "We can reuse credentials from your active Clarifai CLI context. "
                "If these values are missing or outdated, run `clarifai login` to create or refresh a context."
            ),
            note=f"Reference: {CLI_LOGIN_DOC_URL}",
        )
        if not all([default_user_id, default_app_id, current_pat]):
            click.echo(
                click.style(
                    "⚠️  Some context details are missing. Run `clarifai login` if the prompts below are empty.",
                    fg="yellow",
                )
            )

        print_field_help(
            "Model ID",
            "Unique identifier for this Clarifai model. Use lowercase letters, numbers, or hyphens.",
        )
        model_id = prompt_required_field("Model ID", default_model_id)

        print_field_help(
            "Clarifai user ID",
            "Owner of the model. Defaults to your active CLI context's user ID.",
        )
        user_id = prompt_required_field("Clarifai user ID", default_user_id)

        print_field_help(
            "Clarifai app ID",
            "Application where the model will live. You can find your app ID (name of your project) in the Clarifai dashboard under projects section. Defaults to the current context's app ID.",
        )
        app_id = prompt_required_field("Clarifai app ID", default_app_id)

        print_field_help(
            "Model type ID",
            "Clarifai model type (e.g., any-to-any, text-to-text, visual-classifier).",
        )
        model_type_id = prompt_required_field("Model type ID", "any-to-any")

        print_section(
            "Build Info",
            (
                "Specify the environment used to build the model. This helps ensure dependency "
                "compatibility between development and deployment environments."
            ),
            note="Supported Python versions: 3.11 and 3.12 (default).",
        )
        print_field_help(
            "Python version for build",
            "Python version used in the build container. Defaults to 3.12.",
        )
        python_version = prompt_required_field("Python version for build", "3.12")

        print_section(
            "Inference Compute Resources",
            (
                "Minimum compute requirements for dedicated Clarifai deployments. "
                "For purely local execution, you may leave these defaults as-is."
            ),
            note=(
                "Specify requests and limits using Kubernetes notation. For GPU workloads, "
                "use wildcard accelerator patterns like ['NVIDIA-*'] to stay flexible."
            ),
        )
        print_field_help(
            "CPU limit",
            "Total CPU cores available to the container (e.g., '1', '2', '4.5').",
        )
        cpu_limit = prompt_required_field("CPU limit", "1")

        print_field_help(
            "CPU memory limit",
            "Upper bound of CPU memory usage (e.g., '2Gi', '1500Mi').",
        )
        cpu_memory = prompt_required_field("CPU memory limit", "2Gi")

        print_field_help(
            "CPU requests",
            "Guaranteed CPU allocation. Uses Kubernetes notation (default 500m ≈ 0.5 CPU).",
        )
        cpu_requests = prompt_required_field("CPU requests", "1")

        print_field_help(
            "CPU memory requests",
            "Guaranteed CPU memory allocation (e.g., '1Gi', '750Mi').",
        )
        cpu_memory_requests = prompt_required_field("CPU memory requests", "1Gi")

        print_field_help(
            "Number of accelerators",
            "Number of GPUs/TPUs requested for inference. Use 0 for CPU-only workloads.",
        )
        num_accelerators = prompt_int_field("Number of accelerators", 1)

        accelerator_types = ["NVIDIA-*"]
        accelerator_memory = None
        if num_accelerators > 0:
            print_field_help(
                "Accelerator types",
                "Comma-separated list of accelerator patterns (e.g., 'NVIDIA-*', 'NVIDIA-A10G').",
            )
            accelerator_types_input = prompt_optional_field(
                "Accelerator types (comma separated)", "NVIDIA-*"
            )

            print_field_help(
                "Accelerator memory",
                "Minimum GPU/TPU memory required (e.g., '15Gi').",
            )
            accelerator_memory = prompt_optional_field("Accelerator memory", "15Gi")

            if accelerator_types_input:
                accelerator_types = [
                    item.strip() for item in accelerator_types_input.split(',') if item.strip()
                ]
            if not accelerator_types:
                accelerator_types = ["NVIDIA-*"]

        inference_compute_info: Dict[str, Any] = {
            "cpu_limit": cpu_limit,
            "cpu_memory": cpu_memory,
            "cpu_requests": cpu_requests,
            "cpu_memory_requests": cpu_memory_requests,
            "num_accelerators": num_accelerators,
        }
        if num_accelerators > 0:
            inference_compute_info["accelerator_type"] = accelerator_types
            inference_compute_info["accelerator_memory"] = accelerator_memory

        config_data: Dict[str, Any] = {
            "model": {
                "id": model_id,
                "user_id": user_id,
                "app_id": app_id,
                "model_type_id": model_type_id,
            },
            "build_info": {
                "python_version": python_version,
            },
            "inference_compute_info": inference_compute_info,
        }

        if prompt_yes_no("Do you want to configure a checkpoints download?", default=False):
            print_field_help(
                "Checkpoint loader type",
                "Source used to pull checkpoints (currently 'huggingface' is supported).",
            )
            loader_type = prompt_required_field("Checkpoint loader type", "huggingface")

            print_field_help(
                "Checkpoint repo_id",
                "Repository or model identifier (e.g., 'owner/model-name').",
            )
            repo_id = prompt_required_field("Checkpoint repo_id (e.g. owner/model)", None)

            print_field_help(
                "Download stage",
                "When checkpoints should be downloaded: upload, build, or runtime.",
            )
            when = prompt_required_field(
                "When to download checkpoints (upload/build/runtime)", "runtime"
            )

            print_field_help(
                "Hugging Face token",
                "Optional token for private Hugging Face repos. Leave blank for public models.",
            )
            hf_token = prompt_optional_field("Hugging Face token (optional)") or None

            checkpoints_section: Dict[str, Any] = {
                "type": loader_type,
                "repo_id": repo_id,
                "when": when,
            }
            if hf_token:
                checkpoints_section["hf_token"] = hf_token

            allowed_patterns = prompt_optional_field(
                "Allowed file patterns (comma separated, optional)"
            )
            if allowed_patterns:
                checkpoints_section["allowed_file_patterns"] = [
                    pattern.strip() for pattern in allowed_patterns.split(',') if pattern.strip()
                ]

            ignored_patterns = prompt_optional_field(
                "Ignored file patterns (comma separated, optional)"
            )
            if ignored_patterns:
                checkpoints_section["ignore_file_patterns"] = [
                    pattern.strip() for pattern in ignored_patterns.split(',') if pattern.strip()
                ]

            config_data["checkpoints"] = checkpoints_section

        if prompt_yes_no("Would you like to set a fixed num_threads value?", default=False):
            print_field_help(
                "num_threads",
                "Optional override for threads used by the local runner. Leave unset to use defaults.",
            )
            num_threads = prompt_int_field("num_threads")
            config_data["num_threads"] = num_threads

        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config_data, handle, sort_keys=False)

        click.echo(f"✅ Created config.yaml at {config_path}.")
        click.echo("   Review and adjust any values before continuing.")

    except (KeyboardInterrupt, EOFError) as exc:
        click.echo("\nOperation cancelled. Aborting upload.", err=True)
        raise click.Abort() from exc


@cli.group(
    ['model'],
    cls=AliasedGroup,
    context_settings={'max_content_width': shutil.get_terminal_size().columns - 10},
)
def model():
    """Build, test, and deploy models.

    \b
    Workflow:   init → serve → deploy
    Observe:    logs, list, predict
    """


def _sanitize_model_id(name):
    """Convert a model name to a valid model.id (lowercase, alphanumeric, hyphens only)."""
    name = name.split('/')[-1]  # "meta-llama/Llama-3-8B" -> "Llama-3-8B"
    name = name.lower()
    name = name.replace('_', '-')
    name = re.sub(r'[^a-z0-9-]', '', name)  # strip invalid chars (dots, etc.)
    name = re.sub(r'-+', '-', name).strip('-')  # collapse/trim hyphens
    return name or "my-model"


def _copy_embedded_toolkit(toolkit, model_path):
    """Copy embedded toolkit template files to model_path."""
    toolkit_dir = Path(__file__).parent / "templates" / "toolkits" / toolkit
    if not toolkit_dir.exists():
        raise UserError(f"Toolkit '{toolkit}' template not found at {toolkit_dir}")
    for item in toolkit_dir.iterdir():
        if item.name == '__pycache__':
            continue
        dest = Path(model_path) / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)


def _ensure_config_defaults(model_path, model_type_id='any-to-any'):
    """Ensure config.yaml has required fields that older clarifai versions assert on.

    When running in env/container mode, the subprocess installs clarifai from PyPI
    which may still assert on model_type_id. This patches it into config.yaml if missing.
    """
    config_path = os.path.join(model_path, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    model = config.get('model', {})
    changed = False
    if 'model_type_id' not in model:
        model['model_type_id'] = model_type_id
        config['model'] = model
        changed = True
    if changed:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _patch_config(config_path, model_id, checkpoints_repo_id=None):
    """Update model.id and optionally checkpoints.repo_id in config.yaml."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    config.setdefault('model', {})['id'] = model_id
    if checkpoints_repo_id:
        config.setdefault('checkpoints', {})['repo_id'] = checkpoints_repo_id
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _print_init_success(model_path, toolkit):
    """Print unified success message after init."""
    from clarifai.runners.models import deploy_output as out

    click.echo()
    out.success(f"Model initialized in {model_path}")
    click.echo()
    if toolkit in ('python', 'mcp', 'openai', None):
        click.echo("  1. Edit 1/model.py with your model logic")
        click.echo("  2. Add dependencies to requirements.txt")
        click.echo()
    click.echo("  Test locally:")
    click.echo(f"    clarifai model serve {model_path}")
    click.echo(
        f"    clarifai model serve {model_path} --mode env       # auto-create venv and install deps"
    )
    click.echo(f"    clarifai model serve {model_path} --mode container # run inside Docker")
    click.echo()
    click.echo("  Deploy to Clarifai:")
    click.echo(f"    clarifai model deploy {model_path} --instance g5.xlarge")
    click.echo("    clarifai model deploy --instance-info              # list available instances")
    click.echo()


@model.command()
@click.argument(
    "model_path",
    type=click.Path(),
    required=False,
    default=None,
)
@click.option(
    '--toolkit',
    type=click.Choice(
        ['vllm', 'sglang', 'huggingface', 'ollama', 'mcp', 'python', 'openai', 'lmstudio'],
        case_sensitive=False,
    ),
    required=False,
    help='Inference toolkit to scaffold. Omit for a blank Python model.',
)
@click.option(
    '--model-name',
    required=False,
    help='Model checkpoint (HF repo_id or ollama tag). Auto-creates directory from name.',
)
@click.pass_context
def init(
    ctx,
    model_path,
    toolkit,
    model_name,
):
    """Scaffold a new model project with a specific toolkit like vLLM, SGLang, HuggingFace, Ollama, etc.

    \b
    Creates a ready-to-serve model directory with config.yaml,
    requirements.txt, and 1/model.py. Pick a toolkit for a specific
    inference engine, or omit --toolkit for a blank Python template.

    \b
    MODEL_PATH  Target directory (default: current dir).
                Auto-created from --model-name if omitted
                (e.g., --model-name org/Model → ./Model/).

    \b
    Toolkits (GPU):
      vllm         High-throughput LLM serving with vLLM
      sglang       Fast LLM serving with SGLang
      huggingface  HuggingFace Transformers (direct inference)

    \b
    Toolkits (local):
      ollama       Ollama (local LLM server)
      lmstudio     LM Studio (local LLM server)

    \b
    Toolkits (other):
      python       Blank Python model (default)
      mcp          MCP tool server (FastMCP)
      openai       OpenAI-compatible API wrapper

    \b
    Examples:
      clarifai model init --toolkit vllm --model-name meta-llama/Llama-3-8B
      clarifai model init --toolkit ollama --model-name llama3.1
      clarifai model init --toolkit mcp my-mcp-server
      clarifai model init my-model
    """
    validate_context(ctx)

    # Validate option combinations
    if model_name and not toolkit:
        logger.error("--model-name can only be used with --toolkit")
        raise click.Abort()

    # Resolve model_path: explicit > derived from model-name > current dir
    if model_path is None:
        if model_name:
            # "meta-llama/Llama-3-8B" -> "./Llama-3-8B"
            model_path = model_name.split('/')[-1]
        else:
            model_path = "."
    model_path = os.path.abspath(model_path)
    os.makedirs(model_path, exist_ok=True)

    # Derive model_id: from --model-name if given, else from directory name
    if model_name:
        model_id = _sanitize_model_id(model_name)
    else:
        model_id = _sanitize_model_id(os.path.basename(model_path))

    # Embedded toolkits: copy from clarifai/cli/templates/toolkits/{name}/
    EMBEDDED_TOOLKITS = ('vllm', 'sglang', 'huggingface', 'ollama', 'lmstudio')
    # Template toolkits: generate from string templates
    TEMPLATE_TOOLKITS = ('mcp', 'openai', 'python')

    if toolkit in EMBEDDED_TOOLKITS:
        # Pre-flight checks for local server toolkits
        if toolkit == 'ollama' and not check_ollama_installed():
            logger.error("Ollama is not installed. Please install it from https://ollama.com/")
            raise click.Abort()
        if toolkit == 'lmstudio' and not check_lmstudio_installed():
            logger.error(
                "LM Studio is not installed. Please install it from https://lmstudio.com/"
            )
            raise click.Abort()

        logger.info(f"Initializing model with {toolkit} toolkit...")
        _copy_embedded_toolkit(toolkit, model_path)

        # Toolkit-specific customization (updates toolkit.model, model.py defaults, etc.)
        user_id = ctx.obj.current.user_id
        if toolkit == 'ollama':
            customize_ollama_model(model_path, user_id, model_name)
        elif toolkit == 'lmstudio':
            customize_lmstudio_model(model_path, user_id, model_name)
        elif toolkit in ('huggingface', 'vllm', 'sglang'):
            customize_huggingface_model(model_path, user_id, model_name)

        # Patch config LAST to ensure sanitized model_id and checkpoint override
        config_path = os.path.join(model_path, "config.yaml")
        # Only set checkpoints for HF-based toolkits; ollama/lmstudio use toolkit.model instead
        hf_repo = model_name if toolkit in ('huggingface', 'vllm', 'sglang') else None
        _patch_config(config_path, model_id=model_id, checkpoints_repo_id=hf_repo)

    else:
        # Template-based initialization (mcp, openai, python, or no toolkit)
        model_type_id = toolkit if toolkit in TEMPLATE_TOOLKITS else None

        if model_type_id:
            logger.info(f"Initializing {model_type_id} model from template...")
        else:
            logger.info("Initializing model with default template...")

        from clarifai.cli.templates.model_templates import (
            get_config_template,
            get_model_template,
            get_requirements_template,
        )

        # Create 1/model.py
        model_version_dir = os.path.join(model_path, "1")
        os.makedirs(model_version_dir, exist_ok=True)
        model_py_path = os.path.join(model_version_dir, "model.py")
        if os.path.exists(model_py_path):
            logger.warning(f"File {model_py_path} already exists, skipping...")
        else:
            with open(model_py_path, 'w') as f:
                f.write(get_model_template(model_type_id))
            logger.info(f"Created {model_py_path}")

        # Create requirements.txt
        requirements_path = os.path.join(model_path, "requirements.txt")
        if os.path.exists(requirements_path):
            logger.warning(f"File {requirements_path} already exists, skipping...")
        else:
            with open(requirements_path, 'w') as f:
                f.write(get_requirements_template(model_type_id))
            logger.info(f"Created {requirements_path}")

        # Create config.yaml
        config_path = os.path.join(model_path, "config.yaml")
        if os.path.exists(config_path):
            logger.warning(f"File {config_path} already exists, skipping...")
        else:
            config_model_type_id = model_type_id or DEFAULT_LOCAL_RUNNER_MODEL_TYPE
            with open(config_path, 'w') as f:
                f.write(get_config_template(model_type_id=config_model_type_id, model_id=model_id))
            logger.info(f"Created {config_path}")

    _print_init_success(model_path, toolkit)


def _ensure_hf_token(ctx, model_path):
    """
    Ensure HF_TOKEN is present in CLI context.
    """
    import yaml

    try:
        config_path = os.path.join(model_path, "config.yaml")
        if os.path.isfile(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            config_hf_token = None
            try:
                if "checkpoints" in config:
                    config_hf_token = config.get("checkpoints").get("hf_token")
            except Exception:
                logger.warning("Failed to read HF_TOKEN from config.yaml.")
        else:
            logger.error("`config.yaml` not found in model path.")
            raise click.Abort()

        hf_token = getattr(ctx.obj.current, "CLARIFAI_HF_TOKEN", None)
        if hf_token:
            logger.debug("CLARIFAI_HF_TOKEN already present in context.")
        else:
            hf_token = os.getenv("HF_TOKEN", None)
            if hf_token:
                logger.info("Loaded HF_TOKEN from environment.")
                ctx.obj.current.CLARIFAI_HF_TOKEN = hf_token
                ctx.obj.to_yaml()
            elif config_hf_token:
                logger.info("Extracted HF_TOKEN from config.yaml.")
                ctx.obj.current.CLARIFAI_HF_TOKEN = config_hf_token
                ctx.obj.to_yaml()
                return
            else:
                logger.debug("config.yaml not found; skipping HF_TOKEN extraction.")
        if not config_hf_token:
            if 'checkpoints' not in config:
                config['checkpoints'] = {}
            config["checkpoints"]["hf_token"] = hf_token
    except Exception as e:
        logger.warning(f"Unexpected error ensuring HF_TOKEN: {e}")


@model.command()
@click.argument("model_path", type=click.Path(exists=True), required=False, default=".")
@click.option(
    '--platform',
    required=False,
    help='Docker build platform (e.g., "linux/amd64"). Overrides config.yaml.',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Show detailed build and upload logs.',
)
@click.pass_context
def upload(ctx, model_path, platform, verbose):
    """Upload a model to Clarifai (without deploying).

    \b
    Builds a Docker image and uploads it to the Clarifai registry.
    Use 'clarifai model deploy' to upload and deploy in one step.

    \b
    MODEL_PATH  Model directory containing config.yaml (default: ".").
    """
    from clarifai.runners.models.model_builder import upload_model

    validate_context(ctx)
    model_path = os.path.abspath(model_path)
    ensure_config_exists_for_upload(ctx, model_path)
    _ensure_hf_token(ctx, model_path)
    upload_model(
        model_path,
        platform=platform,
        pat=ctx.obj.current.pat,
        base_url=ctx.obj.current.api_base,
        verbose=verbose,
    )


@model.command()
@click.argument('model_path', type=click.Path(), required=False, default=None)
@click.option(
    '--instance',
    default=None,
    help='Hardware instance type (e.g., g5.xlarge). Use --instance-info to list options.',
)
@click.option(
    '--instance-info',
    is_flag=True,
    help='Show available instance types with GPU, memory, and pricing, then exit.',
)
@click.option(
    '--model-url',
    default=None,
    help='Deploy an already-uploaded model by its Clarifai URL (skips upload).',
)
@click.option(
    '--model-version-id',
    default=None,
    help='Specific model version to deploy (default: latest).',
)
@click.option(
    '--min-replicas',
    default=1,
    type=int,
    show_default=True,
    help='Minimum number of running replicas.',
)
@click.option(
    '--max-replicas',
    default=5,
    type=int,
    show_default=True,
    help='Maximum replicas for autoscaling.',
)
@click.option(
    '--cloud',
    default=None,
    help='Cloud provider (e.g., aws, gcp). Auto-detected from --instance if omitted.',
)
@click.option(
    '--region',
    default=None,
    help='Cloud region (e.g., us-east-1). Auto-detected from --instance if omitted.',
)
@click.option(
    '--compute-cluster-id',
    default=None,
    help='[Advanced] Existing compute cluster ID (skip auto-creation).',
)
@click.option(
    '--nodepool-id',
    default=None,
    help='[Advanced] Existing nodepool ID (skip auto-creation).',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Show detailed build, upload, and deployment logs.',
)
@click.pass_context
def deploy(
    ctx,
    model_path,
    instance,
    instance_info,
    model_url,
    model_version_id,
    min_replicas,
    max_replicas,
    cloud,
    region,
    compute_cluster_id,
    nodepool_id,
    verbose,
):
    """Deploy a model to Clarifai cloud compute.

    \b
    Uploads, builds, and deploys in one step. Compute infrastructure
    (cluster + nodepool) is auto-created when needed.

    \b
    MODEL_PATH  Local model directory to upload and deploy (default: ".").
                Not needed when using --model-url.

    \b
    Examples:
      clarifai model deploy ./my-model --instance g5.xlarge
      clarifai model deploy --model-url https://clarifai.com/user/app/models/id --instance g5.xlarge
      clarifai model deploy --instance-info
      clarifai model deploy --instance-info --cloud gcp
    """
    if instance_info:
        from clarifai.utils.compute_presets import list_gpu_presets

        pat_val = None
        base_url_val = None
        try:
            validate_context(ctx)
            pat_val = ctx.obj.current.pat
            base_url_val = ctx.obj.current.api_base
        except Exception:
            pass
        click.echo(
            list_gpu_presets(
                pat=pat_val, base_url=base_url_val, cloud_provider=cloud, region=region
            )
        )
        return

    validate_context(ctx)
    user_id = ctx.obj.current.user_id
    app_id = getattr(ctx.obj.current, 'app_id', None)

    # Resolve model_path to absolute if provided
    if model_path:
        model_path = os.path.abspath(model_path)
        if not os.path.isdir(model_path):
            raise click.BadParameter(f"Model path '{model_path}' is not a directory.")

    from clarifai.runners.models.model_deploy import ModelDeployer

    deployer = ModelDeployer(
        model_path=model_path,
        model_url=model_url,
        user_id=user_id,
        app_id=app_id,
        model_version_id=model_version_id,
        instance_type=instance,
        cloud_provider=cloud,
        region=region,
        compute_cluster_id=compute_cluster_id,
        nodepool_id=nodepool_id,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        pat=ctx.obj.current.pat,
        base_url=ctx.obj.current.api_base,
        verbose=verbose,
    )

    result = deployer.deploy()
    _print_deploy_result(result)


def _print_deploy_result(result):
    """Print a formatted deployment result."""
    from clarifai.runners.models import deploy_output as out

    model_url = result['model_url']

    out.phase_header("Ready")
    click.echo()
    out.success("Model deployed successfully!")
    click.echo()
    out.link("Model", model_url)
    out.info("Version", result['model_version_id'])
    out.info("Deployment", result['deployment_id'])
    if result.get('instance_type'):
        out.info("Instance", result['instance_type'])
    if result.get('cloud_provider') or result.get('region'):
        cloud = result.get('cloud_provider', '').upper()
        region = result.get('region', '')
        out.info("Cloud", f"{cloud} / {region}" if cloud and region else cloud or region)

    # Show client script (same as upload output)
    client_script = result.get('client_script')
    if client_script:
        click.echo("\n" + "=" * 60)
        click.echo("# Here is a code snippet to use this model:")
        click.echo("=" * 60)
        click.echo(client_script)
        click.echo("=" * 60)

    from clarifai.runners.utils.code_script import generate_predict_hint

    model_ref = f'{result["user_id"]}/{result["app_id"]}/models/{result["model_id"]}'
    predict_cmd = generate_predict_hint(
        result.get('method_signatures') or [],
        model_ref,
        deployment_id=result.get('deployment_id'),
    )

    out.phase_header("Next Steps")
    out.hint("Predict", predict_cmd)
    out.hint("Logs", f'clarifai model logs --model-url "{model_url}"')


@model.command()
@click.option('--model-url', default=None, help='Clarifai model URL.')
@click.option('--model-id', default=None, help='Model ID (alternative to --model-url).')
@click.option('--model-version-id', default=None, help='Specific version (default: latest).')
@click.option(
    '--follow/--no-follow',
    default=True,
    help='Continuously tail new logs. Use --no-follow to print and exit.',
)
@click.option(
    '--duration',
    default=None,
    type=int,
    help='Stop after N seconds (default: unlimited, Ctrl+C to stop).',
)
@click.option('--compute-cluster-id', default=None, help='[Advanced] Filter by compute cluster.')
@click.option('--nodepool-id', default=None, help='[Advanced] Filter by nodepool.')
@click.pass_context
def logs(
    ctx, model_url, model_id, model_version_id, follow, duration, compute_cluster_id, nodepool_id
):
    """Stream logs from a deployed model's runner.

    \b
    Shows stdout/stderr from the runner pod — useful for monitoring
    model loading, inference, and debugging errors.

    \b
    Examples:
      clarifai model logs --model-url https://clarifai.com/user/app/models/id
      clarifai model logs --model-url <url> --no-follow
      clarifai model logs --model-url <url> --duration 60
    """
    validate_context(ctx)

    from clarifai.errors import UserError
    from clarifai.runners.models.model_deploy import stream_model_logs

    user_id = ctx.obj.current.user_id
    app_id = getattr(ctx.obj.current, 'app_id', None)

    try:
        stream_model_logs(
            model_url=model_url,
            model_id=model_id,
            user_id=user_id,
            app_id=app_id,
            model_version_id=model_version_id,
            compute_cluster_id=compute_cluster_id,
            nodepool_id=nodepool_id,
            pat=ctx.obj.current.pat,
            base_url=ctx.obj.current.api_base,
            follow=follow,
            duration=duration,
        )
    except UserError as e:
        click.echo(click.style(f"\nError: {e}", fg="red"), err=True)
        raise SystemExit(1)


def _run_local_grpc(model_path, mode, port, keep_image, verbose):
    """Run a model locally via a standalone gRPC server (no PAT, no API)."""
    import signal

    from clarifai.runners.models import deploy_output as out
    from clarifai.runners.models.model_builder import ModelBuilder
    from clarifai.runners.models.model_deploy import _quiet_sdk_logger
    from clarifai.runners.models.model_run_locally import ModelRunLocally
    from clarifai.runners.server import ModelServer

    model_path = os.path.abspath(model_path)
    suppress = not verbose

    # ── Phase 1: Validate ──────────────────────────────────────────────
    out.phase_header("Validate")

    with _quiet_sdk_logger(suppress):
        builder = ModelBuilder(model_path, download_validation_only=True)
    config = builder.config
    model_config = config.get('model', {})

    model_id = model_config.get('id', os.path.basename(model_path))
    model_type_id = model_config.get('model_type_id', DEFAULT_LOCAL_RUNNER_MODEL_TYPE)

    # Validate requirements for none mode only (env creates its own venv, container builds image)
    if mode not in ("container", "env"):
        dependencies = parse_requirements(model_path)
        if not check_requirements_installed(dependencies=dependencies):
            raise UserError(f"Requirements not installed for model at {model_path}.")

        if "ollama" in dependencies or config.get('toolkit', {}).get('provider') == 'ollama':
            if not check_ollama_installed():
                raise UserError(
                    "Ollama is not installed. Install from https://ollama.com/ to use the Ollama toolkit."
                )
        elif "lmstudio" in dependencies or config.get('toolkit', {}).get('provider') == 'lmstudio':
            if not check_lmstudio_installed():
                raise UserError(
                    "LM Studio is not installed. Install from https://lmstudio.com/ to use the LM Studio toolkit."
                )

    # Get method signatures to generate test snippet
    use_mocking = mode in ("container", "env")
    with _quiet_sdk_logger(suppress):
        method_signatures = builder.get_method_signatures(mocking=use_mocking)

    out.info("Model", model_id)
    out.info("Type", model_type_id)
    out.info("Port", str(port))

    # ── Phase 2: Prepare environment ─────────────────────────────────
    container_name = None
    image_name = None
    manager = None
    cleanup_done = False

    def _do_cleanup():
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        if mode == "container" and manager is not None:
            try:
                if container_name and manager.container_exists(container_name):
                    manager.stop_docker_container(container_name)
                    manager.remove_docker_container(container_name=container_name)
                if not keep_image and image_name:
                    manager.remove_docker_image(image_name=image_name)
            except Exception:
                pass
        out.status("Stopped.")

    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):
        signal.signal(signal.SIGINT, original_sigint)
        _do_cleanup()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        # Ensure config.yaml has fields required by PyPI clarifai in subprocess
        if mode in ("container", "env"):
            _ensure_config_defaults(model_path, model_type_id=model_type_id)

        if mode == "container":
            manager = ModelRunLocally(model_path)
            if not manager.is_docker_installed():
                raise UserError("Docker is not installed.")
            with _quiet_sdk_logger(suppress):
                manager.builder.create_dockerfile(generate_dockerfile=True)
            image_tag = manager._docker_hash()
            container_name = model_id.lower()
            image_name = f"{container_name}:{image_tag}"
            if not manager.docker_image_exists(image_name):
                out.status("Building Docker image... ")
                with _quiet_sdk_logger(suppress):
                    manager.build_docker_image(image_name=image_name)
        elif mode == "env":
            manager = ModelRunLocally(model_path)
            out.status("Creating virtual environment... ")
            manager.create_temp_venv()
            out.status("Installing requirements... ")
            manager.install_requirements()

        # ── Phase 3: Running ────────────────────────────────────────────
        out.phase_header("Running")
        out.success(f"gRPC server running at localhost:{port}")
        click.echo()

        from clarifai.runners.utils.code_script import generate_client_script

        snippet = generate_client_script(
            method_signatures,
            user_id=None,
            app_id=None,
            model_id=model_id,
            local_grpc_port=port,
            colorize=True,
        )
        out.status("Test with Python:")
        click.echo(snippet)

        out.status("Press Ctrl+C to stop.")
        click.echo()

        # ── Serve ───────────────────────────────────────────────────────
        if mode == "container":
            manager.run_docker_container(
                image_name=image_name,
                container_name=container_name,
                port=port,
            )
        elif mode == "env":
            manager.run_model_server(port)
        else:
            # none mode: run in-process
            with _quiet_sdk_logger(suppress):
                server = ModelServer(model_path=model_path)
            server.serve(port=port, grpc=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        click.echo()
        out.warning(f"Model failed: {e}")
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        _do_cleanup()


@model.command(name="serve", aliases=["local-runner"])
@click.argument(
    "model_path",
    type=click.Path(exists=True),
    required=False,
    default=".",
)
@click.option(
    "--mode",
    type=click.Choice(['none', 'env', 'container'], case_sensitive=False),
    default='none',
    show_default=True,
    help='Execution environment. none: use current Python (fastest, deps must be installed). env: auto-create virtualenv and install deps. container: build and run a Docker image.',
)
@click.option(
    '--grpc',
    is_flag=True,
    help='Standalone gRPC server (no login required). Without this flag, the model registers with the Clarifai API for Playground and API access.',
)
@click.option(
    '-p',
    '--port',
    type=int,
    default=8000,
    show_default=True,
    help="Server port (used with --grpc).",
)
@click.option(
    "--concurrency",
    type=int,
    default=32,
    show_default=True,
    help="Maximum number of concurrent requests.",
)
@click.option(
    '--keep-image',
    is_flag=True,
    help='Keep Docker image after exit (only with --mode container).',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Show detailed SDK and server logs.',
)
@click.pass_context
def serve_cmd(ctx, model_path, grpc, mode, port, concurrency, keep_image, verbose):
    """Run a model locally for development and testing.

    \b
    Starts the model and registers it with Clarifai so you can send
    predictions via the API, SDK, or Playground UI. Use --grpc for a
    standalone gRPC server with no API connection. Cleans up on Ctrl+C.

    \b
    MODEL_PATH  Model directory containing config.yaml (default: ".").

    \b
    Modes:
      none        Run in current Python env (fastest, deps pre-installed)
      env         Auto-create a virtualenv, install deps, then run
      container   Build a Docker image with all deps, then run

    \b
    Examples:
      clarifai model serve ./my-model                   # current env, API-connected
      clarifai model serve --mode env                   # auto-install deps in venv
      clarifai model serve --mode container             # run inside Docker
      clarifai model serve --grpc                       # offline gRPC server
      clarifai model serve --grpc --port 9000           # custom port
      clarifai model serve --mode container --keep-image
    """
    if grpc:
        # Standalone gRPC server — no PAT, no API
        _run_local_grpc(model_path, mode, port, keep_image, verbose)
        return

    from clarifai.client.user import User
    from clarifai.runners.models import deploy_output as out
    from clarifai.runners.models.model_builder import ModelBuilder
    from clarifai.runners.models.model_deploy import _quiet_sdk_logger
    from clarifai.runners.models.model_run_locally import ModelRunLocally
    from clarifai.runners.server import ModelServer
    from clarifai.runners.utils import code_script

    validate_context(ctx)
    model_path = os.path.abspath(model_path)
    suppress = not verbose

    # ── Phase 1: Validate ──────────────────────────────────────────────
    out.phase_header("Validate")

    with _quiet_sdk_logger(suppress):
        builder = ModelBuilder(model_path, download_validation_only=True)
    config = builder.config
    model_config = config.get('model', {})

    model_id = model_config.get('id')
    if not model_id:
        raise UserError(
            "model.id is required in config.yaml.\n"
            "  Add to your config.yaml:\n"
            "    model:\n"
            "      id: my-model"
        )

    model_type_id = model_config.get('model_type_id', DEFAULT_LOCAL_RUNNER_MODEL_TYPE)

    # Resolve user_id: config → context → error
    user_id = model_config.get('user_id')
    if not user_id:
        try:
            user_id = ctx.obj.current.user_id
        except AttributeError:
            pass
    if not user_id:
        raise UserError(
            "user_id not found in config.yaml or CLI context.\n"
            "  Run 'clarifai login' to set up credentials."
        )

    # Resolve app_id: config → context → default
    app_id = model_config.get('app_id')
    if not app_id:
        try:
            app_id = ctx.obj.current.app_id
        except AttributeError:
            pass
    if not app_id:
        app_id = DEFAULT_LOCAL_RUNNER_APP_ID

    pat = ctx.obj.current.pat
    if not pat:
        raise UserError(
            "Personal Access Token (PAT) not found.\n  Run 'clarifai login' to set up credentials."
        )
    base_url = ctx.obj.current.api_base

    # Validate requirements before loading method signatures
    # Skip for container (builds image) and env (creates its own venv)
    dependencies = parse_requirements(model_path)
    if mode not in ("container", "env"):
        if not check_requirements_installed(dependencies=dependencies):
            raise UserError(f"Requirements not installed for model at {model_path}.")

    if "ollama" in dependencies or config.get('toolkit', {}).get('provider') == 'ollama':
        if not check_ollama_installed():
            raise UserError(
                "Ollama is not installed. Install from https://ollama.com/ to use the Ollama toolkit."
            )
    elif "lmstudio" in dependencies or config.get('toolkit', {}).get('provider') == 'lmstudio':
        if not check_lmstudio_installed():
            raise UserError(
                "LM Studio is not installed. Install from https://lmstudio.com/ to use the LM Studio toolkit."
            )

    # Method signatures from ModelBuilder (same as upload/deploy).
    # Use mocking=False for "none" mode since requirements are verified installed.
    # mocking=True pollutes sys.modules with MagicMock'd third-party packages inside
    # clarifai modules (e.g. FastMCP in stdio_mcp_class), which breaks ModelServer.__init__
    # when it later tries to load the model for real.
    # For container/env modes, deps may not be in current env, so mocking is needed.
    use_mocking = mode in ("container", "env")
    with _quiet_sdk_logger(suppress):
        method_signatures = builder.get_method_signatures(mocking=use_mocking)

    out.info("Model", f"{user_id}/{app_id}/models/{model_id}")
    out.info("Type", model_type_id)

    _ensure_hf_token(ctx, model_path)

    # ── Phase 2: Setup ─────────────────────────────────────────────────
    out.phase_header("Setup")

    # Track what we create for cleanup
    created = {}  # resource_name → cleanup_info

    with _quiet_sdk_logger(suppress):
        user = User(user_id=user_id, pat=pat, base_url=base_url)

        # 1. Compute cluster (shared, reusable — never cleaned up)
        cc_id = DEFAULT_LOCAL_RUNNER_COMPUTE_CLUSTER_ID
        try:
            user.compute_cluster(cc_id)
            out.status("Compute cluster ready")
        except Exception:
            out.status("Creating compute cluster... ", nl=False)
            user.create_compute_cluster(
                compute_cluster_id=cc_id,
                compute_cluster_config=DEFAULT_LOCAL_RUNNER_COMPUTE_CLUSTER_CONFIG,
            )
            click.echo("done")

        # 2. Nodepool (shared, reusable — never cleaned up)
        np_id = DEFAULT_LOCAL_RUNNER_NODEPOOL_ID
        try:
            nodepool = user.compute_cluster(cc_id).nodepool(np_id)
            out.status("Nodepool ready")
        except Exception:
            out.status("Creating nodepool... ", nl=False)
            nodepool = user.compute_cluster(cc_id).create_nodepool(
                nodepool_config=DEFAULT_LOCAL_RUNNER_NODEPOOL_CONFIG,
                nodepool_id=np_id,
            )
            click.echo("done")

        # 3. App (shared, reusable — never cleaned up)
        try:
            app = user.app(app_id)
            out.status("App ready")
        except Exception:
            out.status("Creating app... ", nl=False)
            app = user.create_app(app_id)
            click.echo("done")

        # 4. Model (ephemeral if we create it)
        model_existed = False
        try:
            model = app.model(model_id)
            model_existed = True
            if model.model_type_id != model_type_id:
                out.warning(
                    f"Model type mismatch (expected '{model_type_id}', "
                    f"found '{model.model_type_id}'). Recreating."
                )
                app.delete_model(model_id)
                model_existed = False
                raise Exception("recreate")
            out.status("Model ready")
        except Exception:
            if not model_existed:
                out.status("Creating model... ", nl=False)
                model = app.create_model(model_id, model_type_id=model_type_id)
                created['model'] = model_id
                click.echo("done")

        # 5. Model version (always created fresh — always cleaned up)
        out.status("Creating model version... ", nl=False)
        version_model = model.create_version(
            pretrained_model_config={"local_dev": True},
            method_signatures=method_signatures,
        )
        version_model.load_info()
        version_id = version_model.model_version.id
        created['model_version'] = version_id
        click.echo(f"done ({version_id[:8]})")

        # 6. Stale deployment cleanup (from previous crash)
        deployment_id = f"local-{model_id}"
        try:
            nodepool.deployment(deployment_id)
            nodepool.delete_deployments([deployment_id])
        except Exception:
            pass

        # 7. Runner (always created fresh — always cleaned up)
        worker = {
            "model": {
                "id": model_id,
                "model_version": {"id": version_id},
                "user_id": user_id,
                "app_id": app_id,
            }
        }
        out.status("Creating runner... ", nl=False)
        runner = nodepool.create_runner(
            runner_config={
                "runner": {
                    "description": f"local runner for {model_id}",
                    "worker": worker,
                    "num_replicas": 1,
                }
            }
        )
        runner_id = runner.id
        created['runner'] = runner_id
        click.echo("done")

        # 8. Deployment (always created fresh — always cleaned up)
        out.status("Creating deployment... ", nl=False)
        nodepool.create_deployment(
            deployment_id=deployment_id,
            deployment_config={
                "deployment": {
                    "scheduling_choice": 3,
                    "worker": worker,
                    "nodepools": [
                        {
                            "id": np_id,
                            "compute_cluster": {
                                "id": cc_id,
                                "user_id": user_id,
                            },
                        }
                    ],
                    "deploy_latest_version": True,
                }
            },
        )
        created['deployment'] = deployment_id
        click.echo("done")

    # Toolkit customization (before serving)
    if config.get('toolkit', {}).get('provider') == 'ollama':
        try:
            customize_ollama_model(model_path=model_path, user_id=user_id, verbose=verbose)
        except Exception as e:
            raise UserError(f"Failed to customize Ollama model: {e}")
    elif config.get('toolkit', {}).get('provider') == 'lmstudio':
        try:
            customize_lmstudio_model(model_path=model_path, user_id=user_id)
        except Exception as e:
            raise UserError(f"Failed to customize LM Studio model: {e}")

    serving_args = {
        "pool_size": concurrency,
        "num_threads": concurrency,
        "user_id": user_id,
        "compute_cluster_id": cc_id,
        "nodepool_id": np_id,
        "runner_id": runner_id,
        "base_url": base_url,
        "pat": pat,
        "health_check_port": 0,  # OS-assigned port; avoids collisions between local runners
    }

    container_name = None
    image_name = None
    manager = None
    cleanup_done = False

    def _cleanup():
        out.phase_header("Stopping")
        with _quiet_sdk_logger(suppress):
            if 'deployment' in created:
                out.status("Deleting deployment... ", nl=False)
                try:
                    nodepool.delete_deployments([created['deployment']])
                    click.echo("done")
                except Exception:
                    click.echo("failed")
            if 'runner' in created:
                out.status("Deleting runner... ", nl=False)
                try:
                    nodepool.delete_runners([created['runner']])
                    click.echo("done")
                except Exception:
                    click.echo("failed")
            if 'model_version' in created:
                out.status("Deleting model version... ", nl=False)
                try:
                    model.delete_version(version_id=created['model_version'])
                    click.echo("done")
                except Exception:
                    click.echo("failed")
            if 'model' in created:
                out.status("Deleting model... ", nl=False)
                try:
                    app.delete_model(created['model'])
                    click.echo("done")
                except Exception:
                    click.echo("failed")
        out.status("Stopped.")

    def _do_cleanup():
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True
        # Container cleanup (Docker)
        if mode == "container" and manager is not None:
            try:
                if container_name and manager.container_exists(container_name):
                    manager.stop_docker_container(container_name)
                    manager.remove_docker_container(container_name=container_name)
                if not keep_image and image_name:
                    manager.remove_docker_image(image_name=image_name)
            except Exception:
                pass
        # API resource cleanup (always)
        _cleanup()

    # Register SIGINT handler so cleanup runs before BaseRunner's os._exit(130).
    # BaseRunner catches KeyboardInterrupt internally and calls os._exit(130),
    # which bypasses try/finally. Our signal handler fires first.
    import signal

    original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):
        signal.signal(signal.SIGINT, original_sigint)  # Restore so second Ctrl+C force-kills
        _do_cleanup()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        # ── Phase 3: Prepare environment ────────────────────────────────
        # Ensure config.yaml has fields required by PyPI clarifai in subprocess
        if mode in ("container", "env"):
            _ensure_config_defaults(model_path, model_type_id=model_type_id)

        if mode == "container":
            manager = ModelRunLocally(model_path)
            if not manager.is_docker_installed():
                raise UserError("Docker is not installed.")
            with _quiet_sdk_logger(suppress):
                manager.builder.create_dockerfile(generate_dockerfile=True)
            image_tag = manager._docker_hash()
            container_name = model_id.lower()
            image_name = f"{container_name}:{image_tag}"
            if not manager.docker_image_exists(image_name):
                out.status("Building Docker image... ")
                manager.build_docker_image(image_name=image_name)
        elif mode == "env":
            manager = ModelRunLocally(model_path)
            out.status("Creating virtual environment... ")
            manager.create_temp_venv()
            out.status("Installing requirements... ")
            manager.install_requirements()

        # ── Phase 4: Running ────────────────────────────────────────────
        out.phase_header("Running")
        out.success("Model is ready for API requests!")
        click.echo()

        # Code snippet
        snippet = code_script.generate_client_script(
            method_signatures,
            user_id=user_id,
            app_id=app_id,
            model_id=model_id,
            deployment_id=deployment_id,
            base_url=base_url,
            colorize=True,
        )
        click.echo(snippet)

        ui_base = getattr(ctx.obj.current, 'ui', None) or "https://clarifai.com"
        playground_url = (
            f"{ui_base}/playground?model={model_id}__{version_id}"
            f"&user_id={user_id}&app_id={app_id}"
        )
        model_url = f"{ui_base}/{user_id}/{app_id}/models/{model_id}"
        model_ref = f"{user_id}/{app_id}/models/{model_id}"
        predict_cmd = code_script.generate_predict_hint(
            method_signatures or [], model_ref, deployment_id=deployment_id
        )

        out.phase_header("Next Steps")
        out.hint("Predict", predict_cmd)
        out.link("Playground", playground_url)
        out.link("Model URL", model_url)
        click.echo()
        out.status("Press Ctrl+C to stop.")
        click.echo()

        # ── Serve ───────────────────────────────────────────────────────
        if mode == "container":
            manager.run_docker_container(
                image_name=image_name,
                container_name=container_name,
                port=8080,
                is_local_runner=True,
                env_vars={"CLARIFAI_PAT": pat},
                **serving_args,
            )
        elif mode == "env":
            # Run via venv subprocess so model code uses venv's packages
            # Filter to args accepted by clarifai.runners.server CLI
            runner_args = {
                k: v
                for k, v in serving_args.items()
                if k
                in (
                    'pool_size',
                    'num_threads',
                    'user_id',
                    'compute_cluster_id',
                    'nodepool_id',
                    'runner_id',
                    'base_url',
                    'pat',
                )
            }
            manager.run_model_server(grpc=False, **runner_args)
        else:
            server = ModelServer(model_path=model_path, model_runner_local=None)
            server.serve(**serving_args)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # Model failed to load or serve — show clean error, then cleanup below
        click.echo()
        out.warning(f"Model failed: {e}")
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        _do_cleanup()


def _parse_json_param(param_value, param_name):
    """Parse JSON parameter with error handling.

    Args:
        param_value: The JSON string to parse
        param_name: Name of the parameter for error messages

    Returns:
        dict: Parsed JSON dictionary

    Raises:
        ValueError: If JSON parsing fails
    """
    if not param_value or param_value == '{}':
        return {}
    try:
        return json.loads(param_value)
    except json.JSONDecodeError as e:
        logger.error(f"ValueError: Invalid JSON in --{param_name} parameter: {e}")
        raise click.Abort()


def _process_multimodal_inputs(inputs_dict):
    """Process inputs to convert URLs and file paths to appropriate data types.

    Args:
        inputs_dict: Dictionary of input parameters

    Returns:
        dict: Processed inputs with Image/Video/Audio objects where appropriate
    """
    from clarifai.runners.utils.data_types import Audio, Image, Video

    for key, value in list(inputs_dict.items()):
        if not isinstance(value, str):
            continue

        if value.startswith(("http://", "https://")):
            # Convert URL strings to appropriate data types
            if "image" in key.lower():
                inputs_dict[key] = Image(url=value)
            elif "video" in key.lower():
                inputs_dict[key] = Video(url=value)
            elif "audio" in key.lower():
                inputs_dict[key] = Audio(url=value)
        elif os.path.isfile(value):
            # Convert file paths to appropriate data types
            try:
                with open(value, "rb") as f:
                    file_bytes = f.read()
                if "image" in key.lower():
                    inputs_dict[key] = Image(bytes=file_bytes)
                elif "video" in key.lower():
                    inputs_dict[key] = Video(bytes=file_bytes)
                elif "audio" in key.lower():
                    inputs_dict[key] = Audio(bytes=file_bytes)
            except IOError as e:
                logger.error(f"ValueError: Failed to read file {value}: {e}")
                raise click.Abort()

    return inputs_dict


def _validate_model_params(model_id, user_id, app_id, model_url):
    """Validate model identification parameters.

    Args:
        model_id: Model ID
        user_id: User ID
        app_id: App ID
        model_url: Model URL

    Raises:
        ValueError: If validation fails
    """
    # Check if we have either (model_id, user_id, app_id) or model_url
    has_triple = all([model_id, user_id, app_id])
    has_url = bool(model_url)

    if not (has_triple or has_url):
        logger.error(
            "ValueError: Either --model_id & --user_id & --app_id or --model_url must be provided."
        )
        raise click.Abort()


def _validate_compute_params(compute_cluster_id, nodepool_id, deployment_id):
    """Validate compute cluster parameters.

    Args:
        compute_cluster_id: Compute cluster ID
        nodepool_id: Nodepool ID
        deployment_id: Deployment ID

    Raises:
        ValueError: If validation fails
    """
    if any([compute_cluster_id, nodepool_id, deployment_id]):
        has_cluster_nodepool = bool(compute_cluster_id) and bool(nodepool_id)
        has_deployment = bool(deployment_id)

        if not (has_cluster_nodepool or has_deployment):
            logger.error(
                "ValueError: Either --compute_cluster_id & --nodepool_id or --deployment_id must be provided."
            )
            raise click.Abort()


def _resolve_model_ref(model_ref, ui_base=None):
    """Resolve a model reference to a full Clarifai URL.

    Accepts:
      - Full URL: https://clarifai.com/user/app/models/model → passthrough
      - Shorthand: user/app/models/model → prepend UI base

    Args:
        model_ref: Model reference string.
        ui_base: UI base URL. Defaults to DEFAULT_UI.

    Returns:
        str: Full Clarifai model URL.

    Raises:
        click.UsageError: If format is invalid.
    """
    from clarifai.utils.constants import DEFAULT_UI

    if not model_ref:
        return None

    if model_ref.startswith(("http://", "https://")):
        return model_ref

    parts = model_ref.split("/")
    if len(parts) == 4 and parts[2] == "models":
        base = ui_base or DEFAULT_UI
        return f"{base.rstrip('/')}/{model_ref}"

    raise click.UsageError(
        f"Invalid model reference: '{model_ref}'. "
        "Use user_id/app_id/models/model_id or a full URL."
    )


def _get_first_str_param(model_client, method_name):
    """Find the first string-typed input parameter name for a method.

    Args:
        model_client: ModelClient instance with fetched signatures.
        method_name: Method name to inspect.

    Returns:
        str or None: The parameter name, or None if no string param found.
    """
    from clarifai_grpc.grpc.api import resources_pb2

    if not model_client._defined:
        model_client.fetch()
    method_sig = model_client._method_signatures.get(method_name)
    if not method_sig:
        return None
    for field in method_sig.input_fields:
        if field.type == resources_pb2.ModelTypeField.DataType.STR:
            return field.name
    return None


def _get_first_media_param(model_client, method_name):
    """Find the first Image/Video/Audio-typed input parameter name and its type.

    Args:
        model_client: ModelClient instance with fetched signatures.
        method_name: Method name to inspect.

    Returns:
        tuple: (param_name, data_type_enum) or (None, None).
    """
    from clarifai_grpc.grpc.api import resources_pb2

    media_types = {
        resources_pb2.ModelTypeField.DataType.IMAGE: 'image',
        resources_pb2.ModelTypeField.DataType.VIDEO: 'video',
        resources_pb2.ModelTypeField.DataType.AUDIO: 'audio',
    }
    if not model_client._defined:
        model_client.fetch()
    method_sig = model_client._method_signatures.get(method_name)
    if not method_sig:
        return None, None
    for field in method_sig.input_fields:
        if field.type in media_types:
            return field.name, media_types[field.type]
    return None, None


def _coerce_input_value(value, model_client, method_name, param_name):
    """Coerce a string value to the correct type based on the method signature.

    Args:
        value: String value to coerce.
        model_client: ModelClient instance.
        method_name: Method name.
        param_name: Parameter name.

    Returns:
        Coerced value.
    """
    from clarifai_grpc.grpc.api import resources_pb2

    if not model_client._defined:
        model_client.fetch()
    method_sig = model_client._method_signatures.get(method_name)
    if not method_sig:
        return value
    type_map = {
        resources_pb2.ModelTypeField.DataType.INT: int,
        resources_pb2.ModelTypeField.DataType.FLOAT: float,
        resources_pb2.ModelTypeField.DataType.BOOL: lambda v: v.lower() in ('true', '1', 'yes'),
    }
    for field in method_sig.input_fields:
        if field.name == param_name and field.type in type_map:
            try:
                return type_map[field.type](value)
            except (ValueError, TypeError):
                return value
    return value


def _parse_kv_inputs(input_params, model_client, method_name):
    """Parse key=value input parameters into a dict with type coercion.

    Args:
        input_params: Tuple of "key=value" strings.
        model_client: ModelClient instance for type coercion.
        method_name: Method name for signature lookup.

    Returns:
        dict: Parsed parameters.
    """
    result = {}
    for kv in input_params:
        if '=' not in kv:
            raise click.UsageError(f"Invalid input format: '{kv}'. Use key=value.")
        key, value = kv.split('=', 1)
        result[key] = _coerce_input_value(value, model_client, method_name, key)
    return result


def _detect_media_type_from_ext(path):
    """Detect media type from file extension.

    Returns:
        str: 'image', 'video', or 'audio'.
    """
    ext = os.path.splitext(path)[1].lower()
    video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv'}
    audio_exts = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    if ext in video_exts:
        return 'video'
    elif ext in audio_exts:
        return 'audio'
    return 'image'


def _is_streaming_method(model_client, method_name):
    """Check if a method returns a streaming (Iterator) response.

    Args:
        model_client: ModelClient instance.
        method_name: Method name.

    Returns:
        bool: True if the method returns an Iterator type.
    """
    sig_str = model_client.method_signature(method_name)
    # Signature looks like: "def name(...) -> Iterator[str]:"
    return_part = sig_str.split('->')[-1].strip() if '->' in sig_str else ''
    return return_part.lower().startswith('iterator')


def _select_method(model_methods, model_client, explicit_method, is_chat, has_text_input):
    """Select the best method to call based on available methods and flags.

    Priority:
      1. --chat → openai_stream_transport
      2. --method explicit → use that
      3. OpenAI auto-detection (has text input + model has openai_stream_transport)
      4. Streaming method (generate, or any Iterator-returning method)
      5. Fallback to predict

    Returns:
        tuple: (method_name, is_openai_chat_path)
    """
    methods = list(model_methods)

    # 1. --chat flag
    if is_chat:
        if 'openai_stream_transport' in methods:
            return 'openai_stream_transport', True
        elif 'openai_transport' in methods:
            return 'openai_transport', True
        else:
            raise click.UsageError(
                "This model does not support OpenAI chat. Available methods: " + ", ".join(methods)
            )

    # 2. Explicit --method
    if explicit_method:
        if explicit_method in methods:
            return explicit_method, False
        raise click.UsageError(
            f"Method '{explicit_method}' not available. Available methods: " + ", ".join(methods)
        )

    # 3. OpenAI auto-detection for text input
    if has_text_input and 'openai_stream_transport' in methods:
        return 'openai_stream_transport', True

    # 4. Prefer streaming method
    for m in methods:
        if m in ('openai_stream_transport', 'openai_transport'):
            continue
        if _is_streaming_method(model_client, m):
            return m, False

    # 5. Fallback to predict or first available
    if 'predict' in methods:
        return 'predict', False
    return methods[0] if methods else 'predict', False


def _build_chat_request(message):
    """Build an OpenAI-compatible chat request JSON string."""
    return json.dumps(
        {
            "messages": [{"role": "user", "content": message}],
            "stream": True,
        }
    )


def _display_openai_stream(stream_response, output_format):
    """Display streaming OpenAI chat response, handling reasoning_content and content.

    Args:
        stream_response: Iterator of streaming chunks.
        output_format: 'text' or 'json'.
    """
    full_reasoning = []
    full_content = []
    in_reasoning = False

    for chunk in stream_response:
        try:
            if isinstance(chunk, str):
                data = json.loads(chunk) if chunk.strip() else {}
            else:
                data = chunk
        except (json.JSONDecodeError, TypeError):
            if output_format == 'text':
                click.echo(chunk, nl=False)
            full_content.append(str(chunk))
            continue

        if not isinstance(data, dict):
            if output_format == 'text':
                click.echo(str(data), nl=False)
            full_content.append(str(data))
            continue

        # Handle OpenAI SSE format: data contains choices[0].delta
        choices = data.get('choices', [])
        if not choices:
            # Might be raw text content
            if output_format == 'text':
                click.echo(str(data), nl=False)
            full_content.append(str(data))
            continue

        delta = choices[0].get('delta', {})
        reasoning = delta.get('reasoning_content', '')
        content = delta.get('content', '')

        if reasoning and output_format == 'text':
            if not in_reasoning:
                click.echo('<think>', nl=True)
                in_reasoning = True
            click.echo(reasoning, nl=False)
            full_reasoning.append(reasoning)

        if content:
            if in_reasoning and output_format == 'text':
                click.echo('\n</think>', nl=True)
                in_reasoning = False
            if output_format == 'text':
                click.echo(content, nl=False)
            full_content.append(content)

    if in_reasoning and output_format == 'text':
        click.echo('\n</think>', nl=True)

    if output_format == 'text':
        click.echo()  # Final newline
    elif output_format == 'json':
        result = {}
        if full_reasoning:
            result['reasoning'] = ''.join(full_reasoning)
        result['result'] = ''.join(full_content)
        click.echo(json.dumps(result))


@model.command()
@click.argument('model_ref', required=False, default=None)
@click.argument('text_input', required=False, default=None)
@click.option(
    '-i',
    '--input',
    'input_params',
    multiple=True,
    help='Named parameter as key=value (repeatable).',
)
@click.option(
    '--inputs',
    required=False,
    help='All parameters as JSON string.',
)
@click.option(
    '--file',
    'input_file',
    type=click.Path(exists=True),
    required=False,
    help='Input file (image, audio, video).',
)
@click.option(
    '--url',
    'input_url',
    required=False,
    help='Input URL (image, audio, video).',
)
@click.option(
    '--chat',
    'chat_message',
    required=False,
    help='OpenAI chat message (auto-uses OpenAI client).',
)
@click.option(
    '--method',
    'explicit_method',
    required=False,
    default=None,
    help='Method to call. Overrides auto-selection.',
)
@click.option(
    '--info',
    is_flag=True,
    default=False,
    help='Show available methods and their signatures, then exit.',
)
@click.option(
    '-o',
    '--output',
    'output_format',
    type=click.Choice(['text', 'json']),
    default='text',
    help='Output format (default: text).',
)
@click.option(
    '--deployment',
    'deployment_id',
    required=False,
    help='Route to a specific deployment.',
)
@click.option(
    '--model-url',
    'model_url_opt',
    required=False,
    help='Full model URL (alternative to positional MODEL).',
)
# Hidden legacy flags — still functional
@click.option('--model_id', required=False, hidden=True, help='Model ID.')
@click.option('--user_id', required=False, hidden=True, help='User ID.')
@click.option('--app_id', required=False, hidden=True, help='App ID.')
@click.option('--model_url', 'model_url_legacy', required=False, hidden=True, help='Model URL.')
@click.option(
    '-cc_id',
    '--compute_cluster_id',
    required=False,
    hidden=True,
    help='Compute Cluster ID.',
)
@click.option(
    '-np_id',
    '--nodepool_id',
    required=False,
    hidden=True,
    help='Nodepool ID.',
)
@click.option(
    '-dpl_id',
    '--deployment_id',
    'deployment_id_legacy',
    required=False,
    hidden=True,
    help='Deployment ID.',
)
@click.option(
    '-dpl_usr_id',
    '--deployment_user_id',
    required=False,
    hidden=True,
    help='Deployment user ID.',
)
@click.pass_context
def predict(
    ctx,
    model_ref,
    text_input,
    input_params,
    inputs,
    input_file,
    input_url,
    chat_message,
    explicit_method,
    info,
    output_format,
    deployment_id,
    model_url_opt,
    model_id,
    user_id,
    app_id,
    model_url_legacy,
    compute_cluster_id,
    nodepool_id,
    deployment_id_legacy,
    deployment_user_id,
):
    """Run a prediction against a Clarifai model.

    \b
    Arguments:
      MODEL   Model as user_id/app_id/models/model_id or full URL.
      INPUT   Text input (for text models). Use --file or --url for media.

    \b
    Examples:
      clarifai model predict openai/chat-completion/models/GPT-4 "Hello world"
      clarifai model predict openai/chat-completion/models/GPT-4 --info
      echo "Hello" | clarifai model predict openai/chat-completion/models/GPT-4
      clarifai model predict my/app/models/detector --file photo.jpg
      clarifai model predict my/app/models/detector --url https://example.com/img.jpg
      clarifai model predict my/app/models/m -i prompt="Hello" -i max_tokens=200
      clarifai model predict openai/chat-completion/models/GPT-4 --chat "What is AI?"
      clarifai model predict openai/chat-completion/models/GPT-4 "Hello" -o json
    """
    import sys

    from clarifai.client.model import Model
    from clarifai.urls.helper import ClarifaiUrlHelper
    from clarifai.utils.cli import validate_context

    validate_context(ctx)

    # --- Merge legacy flags ---
    model_url = model_url_opt or model_url_legacy
    deployment_id = deployment_id or deployment_id_legacy

    # --- Resolve model URL ---
    # Priority: positional model_ref > --model-url/--model_url > --model_id triple > config
    if model_ref:
        model_url = _resolve_model_ref(model_ref, ui_base=ctx.obj.current.ui)
    elif not model_url:
        # Try legacy triple
        if all([model_id, user_id, app_id]):
            model_url = ClarifaiUrlHelper.clarifai_url(
                user_id=user_id, app_id=app_id, resource_type="models", resource_id=model_id
            )
        else:
            raise click.UsageError(
                "No model specified. Use: clarifai model predict <user/app/models/model> ..."
            )

    logger.debug(f"Using model at URL: {model_url}")

    # --- Validate compute params ---
    _validate_compute_params(compute_cluster_id, nodepool_id, deployment_id)

    # --- Create model instance ---
    model = Model(
        url=model_url,
        pat=ctx.obj.current.pat,
        base_url=ctx.obj.current.api_base,
        compute_cluster_id=compute_cluster_id,
        nodepool_id=nodepool_id,
        deployment_id=deployment_id,
        deployment_user_id=deployment_user_id,
    )

    model_methods = list(model.client.available_methods())

    # --- --info: display methods and exit ---
    if info:
        click.echo(f"Model: {model.id} ({model.user_id}/{model.app_id})\n")
        click.echo("Methods:")
        for m in model_methods:
            sig = model.client.method_signature(m)
            click.echo(f"  {sig}")
        return

    # --- Determine if we have text input (for OpenAI auto-detection) ---
    has_text = bool(text_input or chat_message)
    if not has_text and not input_file and not input_url and not inputs and not input_params:
        # Check stdin
        if not sys.stdin.isatty():
            text_input = sys.stdin.read().strip()
            has_text = bool(text_input)

    # --- Select method ---
    method_name, is_openai_path = _select_method(
        model_methods, model.client, explicit_method, bool(chat_message), has_text
    )
    is_stream = _is_streaming_method(model.client, method_name)

    # --- Build inputs ---
    if is_openai_path:
        # OpenAI chat path
        chat_text = chat_message or text_input
        if not chat_text:
            raise click.UsageError("No input provided for chat. Pass a text argument or --chat.")
        request_body = _build_chat_request(chat_text)
        model_prediction = getattr(model, method_name)(msg=request_body)
        _display_openai_stream(model_prediction, output_format)
        return

    # Build inputs dict from various sources
    inputs_dict = {}

    # --inputs JSON
    if inputs:
        inputs_dict = _parse_json_param(inputs, "inputs")

    # -i key=value pairs (override JSON keys)
    if input_params:
        kv_dict = _parse_kv_inputs(input_params, model.client, method_name)
        inputs_dict.update(kv_dict)

    # --file
    if input_file:
        from clarifai.runners.utils.data_types import Audio, Image, Video

        param_name, media_type = _get_first_media_param(model.client, method_name)
        if not param_name:
            # Fallback: detect from extension, use generic name
            media_type = _detect_media_type_from_ext(input_file)
            param_name = media_type

        with open(input_file, 'rb') as f:
            file_bytes = f.read()

        type_cls = {'image': Image, 'video': Video, 'audio': Audio}[media_type]
        inputs_dict[param_name] = type_cls(bytes=file_bytes)

    # --url
    if input_url:
        from clarifai.runners.utils.data_types import Audio, Image, Video

        param_name, media_type = _get_first_media_param(model.client, method_name)
        if not param_name:
            media_type = _detect_media_type_from_ext(input_url)
            param_name = media_type

        type_cls = {'image': Image, 'video': Video, 'audio': Audio}[media_type]
        inputs_dict[param_name] = type_cls(url=input_url)

    # Positional text input or stdin
    if not inputs_dict and text_input:
        param_name = _get_first_str_param(model.client, method_name)
        if param_name:
            inputs_dict[param_name] = text_input
        else:
            # If no str param found, try passing as first positional
            inputs_dict = {'text': text_input}

    if not inputs_dict:
        raise click.UsageError(
            "No input provided. Pass text, --file, --url, --inputs, or -i key=value.\n"
            "Use --info to see available methods and their parameters."
        )

    # Process multimodal inputs (URL/file strings in dict values)
    inputs_dict = _process_multimodal_inputs(inputs_dict)

    # --- Execute prediction ---
    if method_name not in model_methods:
        raise click.UsageError(
            f"Method '{method_name}' not available. Available methods: " + ", ".join(model_methods)
        )

    model_prediction = getattr(model, method_name)(**inputs_dict)

    # --- Display output ---
    if is_stream:
        if output_format == 'json':
            chunks = []
            for chunk in model_prediction:
                if isinstance(chunk, str):
                    chunks.append(chunk)
            click.echo(json.dumps({"result": ''.join(chunks)}))
        else:
            for chunk in model_prediction:
                if isinstance(chunk, str):
                    click.echo(chunk, nl=False)
            click.echo()
    elif output_format == 'json':
        click.echo(json.dumps({"result": str(model_prediction)}))
    else:
        click.echo(model_prediction)


@model.command(name="list")
@click.argument(
    "user_id",
    required=False,
    default=None,
)
@click.option(
    '-a',
    '--app_id',
    type=str,
    default=None,
    help='Filter by app ID.',
)
@click.pass_context
def list_model(ctx, user_id, app_id):
    """List models for a user or across the platform.

    \b
    USER_ID  User ID to list models for (default: current user).
             Use "all" to list public models across Clarifai.
    """
    from clarifai.client import User

    try:
        pat = ctx.obj.current.pat
    except Exception as e:
        pat = None

    User(pat=pat).list_models(
        user_id=user_id, app_id=app_id, show=True, return_clarifai_model=False
    )
