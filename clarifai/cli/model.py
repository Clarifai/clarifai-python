import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional

import click
import yaml

from clarifai.cli.base import cli
from clarifai.errors import UserError
from clarifai.utils.cli import (
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
    DEFAULT_HF_MODEL_REPO_BRANCH,
    DEFAULT_LMSTUDIO_MODEL_REPO_BRANCH,
    DEFAULT_LOCAL_RUNNER_APP_ID,
    DEFAULT_LOCAL_RUNNER_COMPUTE_CLUSTER_CONFIG,
    DEFAULT_LOCAL_RUNNER_COMPUTE_CLUSTER_ID,
    DEFAULT_LOCAL_RUNNER_MODEL_TYPE,
    DEFAULT_LOCAL_RUNNER_NODEPOOL_CONFIG,
    DEFAULT_LOCAL_RUNNER_NODEPOOL_ID,
    DEFAULT_OLLAMA_MODEL_REPO_BRANCH,
    DEFAULT_PYTHON_MODEL_REPO_BRANCH,
    DEFAULT_SGLANG_MODEL_REPO_BRANCH,
    DEFAULT_TOOLKIT_MODEL_REPO,
    DEFAULT_VLLM_MODEL_REPO_BRANCH,
)
from clarifai.utils.logging import logger
from clarifai.utils.misc import (
    clone_github_repo,
    format_github_repo_url,
)


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
    ['model'], context_settings={'max_content_width': shutil.get_terminal_size().columns - 10}
)
def model():
    """Manage Models: init, upload, deploy\n
    Run Locally: local-runner\n
    Observe: logs, list, predict"""


@model.command()
@click.argument(
    "model_path",
    type=click.Path(),
    required=False,
    default=".",
)
@click.option(
    '--toolkit',
    type=click.Choice(
        ['ollama', 'huggingface', 'lmstudio', 'vllm', 'sglang', 'python', 'mcp', 'openai'],
        case_sensitive=False,
    ),
    required=False,
    help='Toolkit/template to use for model initialization.',
)
@click.option(
    '--model-name',
    required=False,
    help='Model name to configure when using --toolkit (e.g., "meta-llama/Llama-3-8B" for vllm/huggingface, "llama3.1" for ollama).',
)
@click.pass_context
def init(
    ctx,
    model_path,
    toolkit,
    model_name,
):
    """Initialize a new model directory structure.

    \b
    Creates the following structure:
      MODEL_PATH/
      ├── 1/model.py
      ├── requirements.txt
      └── config.yaml

    \b
    Examples:
      clarifai model init my-model
      clarifai model init --toolkit vllm --model-name meta-llama/Llama-3-8B my-llama
      clarifai model init --toolkit ollama --model-name llama3.1 my-ollama
      clarifai model init --toolkit mcp my-mcp-server
      clarifai model init --toolkit openai my-openai-model
    """
    validate_context(ctx)
    user_id = ctx.obj.current.user_id
    # Resolve the absolute path
    model_path = os.path.abspath(model_path)

    # Create the model directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)

    # Validate option combinations
    if model_name and not toolkit:
        logger.error("--model-name can only be used with --toolkit")
        raise click.Abort()

    # Template-based toolkits (mcp, openai) use local templates, not GitHub clone
    TEMPLATE_TOOLKITS = ('mcp', 'openai')

    github_url = None
    branch = None

    # --toolkit option (GitHub-cloned toolkits)
    if toolkit == 'ollama':
        if not check_ollama_installed():
            logger.error(
                "Ollama is not installed. Please install it from `https://ollama.com/` to use the Ollama toolkit."
            )
            raise click.Abort()
        github_url = DEFAULT_TOOLKIT_MODEL_REPO
        branch = DEFAULT_OLLAMA_MODEL_REPO_BRANCH
    elif toolkit == 'huggingface':
        github_url = DEFAULT_TOOLKIT_MODEL_REPO
        branch = DEFAULT_HF_MODEL_REPO_BRANCH
    elif toolkit == 'lmstudio':
        if not check_lmstudio_installed():
            logger.error(
                "LM Studio is not installed. Please install it from `https://lmstudio.com/` to use the LM Studio toolkit."
            )
            raise click.Abort()
        github_url = DEFAULT_TOOLKIT_MODEL_REPO
        branch = DEFAULT_LMSTUDIO_MODEL_REPO_BRANCH
    elif toolkit == 'vllm':
        github_url = DEFAULT_TOOLKIT_MODEL_REPO
        branch = DEFAULT_VLLM_MODEL_REPO_BRANCH
    elif toolkit == 'sglang':
        github_url = DEFAULT_TOOLKIT_MODEL_REPO
        branch = DEFAULT_SGLANG_MODEL_REPO_BRANCH
    elif toolkit == 'python':
        github_url = DEFAULT_TOOLKIT_MODEL_REPO
        branch = DEFAULT_PYTHON_MODEL_REPO_BRANCH

    if toolkit and github_url:
        logger.info(f"Initializing model with {toolkit} toolkit...")

        repo_url = format_github_repo_url(github_url)

        try:
            # Create a temporary directory for cloning
            with tempfile.TemporaryDirectory(prefix="clarifai_model_") as clone_dir:
                # Clone the repository with explicit branch parameter
                if not clone_github_repo(repo_url, clone_dir, None, branch):
                    logger.error(f"Failed to clone repository from {repo_url}")
                    github_url = None  # Fall back to template mode

                else:
                    # Copy the entire repository content to target directory (excluding .git)
                    for item in os.listdir(clone_dir):
                        if item == '.git':
                            continue

                        source_path = os.path.join(clone_dir, item)
                        target_path = os.path.join(model_path, item)

                        if os.path.isdir(source_path):
                            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                        else:
                            shutil.copy2(source_path, target_path)

        except Exception as e:
            logger.error(f"Failed to clone GitHub repository: {e}")
            github_url = None

    if toolkit == 'ollama':
        customize_ollama_model(model_path, user_id, model_name)

    if toolkit == 'lmstudio':
        customize_lmstudio_model(model_path, user_id, model_name)

    if toolkit in ('huggingface', 'vllm', 'sglang'):
        customize_huggingface_model(model_path, user_id, model_name)

    if github_url:
        logger.info(f"Model initialization complete in {model_path}")
        logger.info("Next steps:")
        logger.info("1. Review the model configuration in config.yaml")
        logger.info("2. Deploy: clarifai model deploy %s --instance g5.xlarge" % model_path)

    # Fall back to template-based initialization if no GitHub repo or if GitHub repo failed
    # Also handles template toolkits (mcp, openai) which don't clone from GitHub
    if not github_url:
        # Determine model_type_id from toolkit (mcp/openai) or default
        model_type_id = toolkit if toolkit in TEMPLATE_TOOLKITS else None

        if model_type_id:
            logger.info(f"Initializing {model_type_id} model from template...")
        else:
            logger.info("Initializing model with default templates...")

        from clarifai.cli.templates.model_templates import (
            get_config_template,
            get_model_template,
            get_requirements_template,
        )

        # Create the 1/ subdirectory
        model_version_dir = os.path.join(model_path, "1")
        os.makedirs(model_version_dir, exist_ok=True)

        # Create model.py
        model_py_path = os.path.join(model_version_dir, "model.py")
        if os.path.exists(model_py_path):
            logger.warning(f"File {model_py_path} already exists, skipping...")
        else:
            model_template = get_model_template(model_type_id)
            with open(model_py_path, 'w') as f:
                f.write(model_template)
            logger.info(f"Created {model_py_path}")

        # Create requirements.txt
        requirements_path = os.path.join(model_path, "requirements.txt")
        if os.path.exists(requirements_path):
            logger.warning(f"File {requirements_path} already exists, skipping...")
        else:
            requirements_template = get_requirements_template(model_type_id)
            with open(requirements_path, 'w') as f:
                f.write(requirements_template)
            logger.info(f"Created {requirements_path}")

        # Create config.yaml
        config_path = os.path.join(model_path, "config.yaml")
        if os.path.exists(config_path):
            logger.warning(f"File {config_path} already exists, skipping...")
        else:
            dir_name = os.path.basename(os.path.abspath(model_path))
            config_model_type_id = model_type_id or DEFAULT_LOCAL_RUNNER_MODEL_TYPE

            config_template = get_config_template(
                user_id=user_id, model_type_id=config_model_type_id, model_id=dir_name
            )
            with open(config_path, 'w') as f:
                f.write(config_template)
            logger.info(f"Created {config_path}")

        logger.info(f"Model initialization complete in {model_path}")
        logger.info("Next steps:")
        logger.info("1. Implement your model logic in 1/model.py")
        logger.info("2. Add your model dependencies to requirements.txt")
        logger.info("3. Deploy: clarifai model deploy %s --instance g5.xlarge" % model_path)


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


@model.command(help="Upload a trained model.")
@click.argument("model_path", type=click.Path(exists=True), required=False, default=".")
@click.option(
    '--platform',
    required=False,
    help='Target platform(s) for Docker image build (e.g., "linux/amd64" or "linux/amd64,linux/arm64"). This overrides the platform specified in config.yaml.',
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='Show detailed build logs and SDK messages.',
)
@click.pass_context
def upload(ctx, model_path, platform, verbose):
    """Upload a model to Clarifai.

    MODEL_PATH: Path to the model directory. If not specified, the current directory is used by default.
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
    help="Hardware to deploy on. Run '--instance-info' to see available options.",
)
@click.option(
    '--instance-info',
    is_flag=True,
    help='List all available instance types and exit.',
)
@click.option(
    '--model-url',
    default=None,
    help='Deploy an existing model by URL instead of a local directory.',
)
@click.option(
    '--model-version-id',
    default=None,
    help='Version to deploy. Defaults to the latest version.',
)
@click.option(
    '--min-replicas', default=1, type=int, show_default=True, help='Minimum number of replicas.'
)
@click.option(
    '--max-replicas', default=5, type=int, show_default=True, help='Maximum number of replicas.'
)
@click.option(
    '--cloud',
    default=None,
    help="Cloud provider. Auto-detected from --instance if not set.",
)
@click.option(
    '--region',
    default=None,
    help="Cloud region. Auto-detected from --instance if not set.",
)
@click.option(
    '--compute-cluster-id',
    default=None,
    help='[Advanced] Use an existing compute cluster instead of auto-creating one.',
)
@click.option(
    '--nodepool-id',
    default=None,
    help='[Advanced] Use an existing nodepool instead of auto-creating one.',
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='Show detailed build, deploy, and runner logs.',
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
    """Deploy a model to Clarifai compute.

    Uploads, builds, and deploys in one step. Compute infrastructure
    (cluster + nodepool) is auto-created if needed.

    \b
    Examples:
      clarifai model deploy ./my-model --instance g5.xlarge
      clarifai model deploy --model-url https://clarifai.com/user/app/models/id --instance g5.xlarge
      clarifai model deploy --instance-info
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
    out.info("Model", model_url)
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
    else:
        click.echo("\n  Predict:")
        click.echo('    from clarifai.client import Model')
        click.echo(f'    model = Model(url="{model_url}")')
        click.echo('    model.predict(...)  # see model.method_signatures for available methods')

    click.echo("\n  Check the Model Logs:")
    click.echo(f'    clarifai model logs --model-url "{model_url}"')
    click.echo("")


@model.command(help="Stream model runner logs.")
@click.option('--model-url', default=None, help='Clarifai model URL.')
@click.option('--model-id', default=None, help='Model ID.')
@click.option('--model-version-id', default=None, help='Specific version (default: latest).')
@click.option(
    '--compute-cluster-id', default=None, help='[Advanced] Filter by compute cluster ID.'
)
@click.option('--nodepool-id', default=None, help='[Advanced] Filter by nodepool ID.')
@click.option(
    '--follow/--no-follow',
    default=True,
    help='Continuously tail logs (default: --follow). Use --no-follow to print existing logs and exit.',
)
@click.option(
    '--duration',
    default=None,
    type=int,
    help='Max seconds to stream logs (default: unlimited, until Ctrl+C).',
)
@click.pass_context
def logs(
    ctx, model_url, model_id, model_version_id, compute_cluster_id, nodepool_id, follow, duration
):
    """Stream model runner pod logs.

    \b
    Shows stdout/stderr from the model's runner pod, useful for viewing
    model loading progress, inference logs, and debugging.

    \b
    EXAMPLES:
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

    # Validate requirements for none/env modes
    if mode != "container":
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
    use_mocking = mode == "container"
    with _quiet_sdk_logger(suppress):
        method_signatures = builder.get_method_signatures(mocking=use_mocking)

    out.info("Model", model_id)
    out.info("Type", model_type_id)
    out.info("Port", str(port))

    # ── Phase 2: Running ───────────────────────────────────────────────
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

    # ── Phase 3: Serve (with cleanup) ──────────────────────────────────
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
        if mode == "container":
            manager = ModelRunLocally(model_path)
            if not manager.is_docker_installed():
                raise UserError("Docker is not installed.")
            image_tag = manager._docker_hash()
            container_name = model_id.lower()
            image_name = f"{container_name}:{image_tag}"
            if not manager.docker_image_exists(image_name):
                with _quiet_sdk_logger(suppress):
                    manager.build_docker_image(image_name=image_name)
            manager.run_docker_container(
                image_name=image_name,
                container_name=container_name,
                port=port,
            )
        elif mode == "env":
            manager = ModelRunLocally(model_path)
            manager.create_temp_venv()
            manager.install_requirements()
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


@model.command(name="local-runner")
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
    help='Execution environment. "none" uses your current env (dependencies must be pre-installed), "env" creates a virtualenv and installs all dependencies, "container" builds a Docker image with all dependencies.',
)
@click.option(
    "--concurrency",
    type=int,
    default=32,
    show_default=True,
    help="Number of concurrent requests the local runner will handle.",
)
@click.option(
    '--port',
    '-p',
    type=int,
    default=8000,
    show_default=True,
    help="Port for the gRPC server (only used with --grpc).",
)
@click.option(
    '--grpc',
    is_flag=True,
    help='Run a standalone gRPC server instead of connecting to the Clarifai API. No login required.',
)
@click.option(
    '--keep-image',
    is_flag=True,
    help='Do not remove the Docker image on exit (only applies to --mode container).',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Show full SDK debug output instead of clean summaries.',
)
@click.pass_context
def local_runner(ctx, model_path, mode, concurrency, port, grpc, keep_image, verbose):
    """Run a model locally for testing predictions.

    \b
    By default, starts a local runner connected to the Clarifai API so you
    can send predictions via the API or the Playground UI.

    \b
    With --grpc, starts a standalone gRPC server on localhost instead.
    No login required — fully offline.

    \b
    MODEL_PATH  Path to the model directory (must contain config.yaml).
                Defaults to the current directory.

    \b
    Examples:
      clarifai model local-runner ./my-model
      clarifai model local-runner ./my-model --grpc
      clarifai model local-runner ./my-model --grpc --port 9000
      clarifai model local-runner ./my-model --mode container
      clarifai model local-runner ./my-model --concurrency 8 --verbose
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
    dependencies = parse_requirements(model_path)
    if mode != "container":
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
    # Use mocking=False for non-container modes since requirements are verified installed.
    # mocking=True pollutes sys.modules with MagicMock'd third-party packages inside
    # clarifai modules (e.g. FastMCP in stdio_mcp_class), which breaks ModelServer.__init__
    # when it later tries to load the model for real.
    use_mocking = mode == "container"
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

    # ── Phase 3: Running ───────────────────────────────────────────────
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
    out.status("Code snippet:")
    click.echo(snippet)
    click.echo()

    # Playground URL
    ui_base = getattr(ctx.obj.current, 'ui', None) or "https://clarifai.com"
    out.status("Playground:")
    out.status(
        f"  {ui_base}/playground?model={model_id}__{version_id}&user_id={user_id}&app_id={app_id}"
    )
    click.echo()

    # Model URL
    out.status("Model URL:")
    out.status(f"  {ui_base}/{user_id}/{app_id}/models/{model_id}")
    click.echo()

    out.status("Press Ctrl+C to stop.")
    click.echo()

    # ── Phase 4: Serve (with cleanup) ──────────────────────────────────
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
        if mode == "container":
            manager = ModelRunLocally(model_path)
            if not manager.is_docker_installed():
                raise UserError("Docker is not installed.")
            image_tag = manager._docker_hash()
            container_name = model_id.lower()
            image_name = f"{container_name}:{image_tag}"
            if not manager.docker_image_exists(image_name):
                manager.build_docker_image(image_name=image_name)
            manager.run_docker_container(
                image_name=image_name,
                container_name=container_name,
                port=8080,
                is_local_runner=True,
                env_vars={"CLARIFAI_PAT": pat},
                **serving_args,
            )
        else:
            if mode == "env":
                manager = ModelRunLocally(model_path)
                manager.create_temp_venv()
                manager.install_requirements()
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


@model.command(help="Perform a prediction using the model.")
@click.option(
    '--config',
    type=click.Path(exists=True),
    required=False,
    help='Path to the model predict config file.',
)
@click.option('--model_id', required=False, help='Model ID of the model used to predict.')
@click.option('--user_id', required=False, help='User ID of the model used to predict.')
@click.option('--app_id', required=False, help='App ID of the model used to predict.')
@click.option('--model_url', required=False, help='Model URL of the model used to predict.')
@click.option(
    '-cc_id',
    '--compute_cluster_id',
    required=False,
    help='Compute Cluster ID to use for the model',
)
@click.option('-np_id', '--nodepool_id', required=False, help='Nodepool ID to use for the model')
@click.option(
    '-dpl_id', '--deployment_id', required=False, help='Deployment ID to use for the model'
)
@click.option(
    '-dpl_usr_id',
    '--deployment_user_id',
    required=False,
    help='User ID to use for runner selector (organization or user). If not provided, defaults to PAT owner user_id.',
)
@click.option(
    '--inputs',
    required=False,
    help='JSON string of input parameters for pythonic models (e.g., \'{"prompt": "Hello", "max_tokens": 100}\')',
)
@click.option('--method', required=False, default='predict', help='Method to call on the model.')
@click.pass_context
def predict(
    ctx,
    config,
    model_id,
    user_id,
    app_id,
    model_url,
    compute_cluster_id,
    nodepool_id,
    deployment_id,
    deployment_user_id,
    inputs,
    method,
):
    """Predict using a Clarifai model.

    \b
    Model Identification:
        Use either --model_url OR the combination of --model_id, --user_id, and --app_id

    \b
    Input Methods:
        --inputs: JSON string with parameters (e.g., '{"prompt": "Hello", "max_tokens": 100}')
        --method: Method to call on the model (default is 'predict')

    \b
    Compute Options:
        Use either --deployment_id OR both --compute_cluster_id and --nodepool_id

    \b
    Examples:
        Text model:
            clarifai model predict --model_url <url> --inputs '{"prompt": "Hello world"}'

        With compute cluster:
            clarifai model predict --model_id <id> --user_id <uid> --app_id <aid> \\
                                  --compute_cluster_id <cc_id> --nodepool_id <np_id> \\
                                  --inputs '{"prompt": "Hello"}'
    """
    from clarifai.client.model import Model
    from clarifai.urls.helper import ClarifaiUrlHelper
    from clarifai.utils.cli import from_yaml, validate_context

    validate_context(ctx)

    # Load configuration from file if provided
    if config:
        config_data = from_yaml(config)
        # Override None values with config data
        model_id = model_id or config_data.get('model_id')
        user_id = user_id or config_data.get('user_id')
        app_id = app_id or config_data.get('app_id')
        model_url = model_url or config_data.get('model_url')
        compute_cluster_id = compute_cluster_id or config_data.get('compute_cluster_id')
        nodepool_id = nodepool_id or config_data.get('nodepool_id')
        deployment_id = deployment_id or config_data.get('deployment_id')
        deployment_user_id = deployment_user_id or config_data.get('deployment_user_id')
        inputs = inputs or config_data.get('inputs')
        method = method or config_data.get('method', 'predict')

    # Validate parameters
    _validate_model_params(model_id, user_id, app_id, model_url)
    _validate_compute_params(compute_cluster_id, nodepool_id, deployment_id)

    # Generate model URL if not provided
    if not model_url:
        model_url = ClarifaiUrlHelper.clarifai_url(
            user_id=user_id, app_id=app_id, resource_type="models", resource_id=model_id
        )
    logger.debug(f"Using model at URL: {model_url}")

    # Create model instance
    model = Model(
        url=model_url,
        pat=ctx.obj.current.pat,
        base_url=ctx.obj.current.api_base,
        compute_cluster_id=compute_cluster_id,
        nodepool_id=nodepool_id,
        deployment_id=deployment_id,
        deployment_user_id=deployment_user_id,
    )

    model_methods = model.client.available_methods()
    stream_method = (
        model.client.method_signature(method).split()[-1][:-1].lower().startswith('iter')
    )

    # Determine prediction method and execute
    if inputs and (method in model_methods):
        # Pythonic model prediction with JSON inputs
        inputs_dict = _parse_json_param(inputs, "inputs")
        inputs_dict = _process_multimodal_inputs(inputs_dict)
        model_prediction = getattr(model, method)(**inputs_dict)
    else:
        logger.error(
            f"ValueError: The model does not support the '{method}' method. Please check the model's capabilities."
        )
        raise click.Abort()

    if stream_method:
        for chunk in model_prediction:
            click.echo(chunk, nl=False)
        click.echo()  # Ensure a newline after the stream ends
    else:
        click.echo(model_prediction)


@model.command(name="list")
@click.argument(
    "user_id",
    required=False,
    default=None,
)
@click.option(
    '--app_id',
    '-a',
    type=str,
    default=None,
    show_default=True,
    help="Get all models of an app",
)
@click.pass_context
def list_model(ctx, user_id, app_id):
    """List models of user/community.

    USER_ID: User id. If not specified, the current user is used by default. Set "all" to get all public models in Clarifai platform.
    """
    from clarifai.client import User

    try:
        pat = ctx.obj.current.pat
    except Exception as e:
        pat = None

    User(pat=pat).list_models(
        user_id=user_id, app_id=app_id, show=True, return_clarifai_model=False
    )
