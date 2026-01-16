import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional

import click
import yaml

from clarifai.cli.base import cli, pat_display
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
    DEFAULT_LOCAL_RUNNER_DEPLOYMENT_ID,
    DEFAULT_LOCAL_RUNNER_MODEL_ID,
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
    GitHubDownloader,
    clone_github_repo,
    format_github_repo_url,
    get_list_of_files_to_download,
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
    """Manage & Develop Models: init, download-checkpoints, signatures, upload\n
    Run & Test Models Locally: local-runner, local-grpc, local-test\n
    Model Inference: list, predict"""


@model.command()
@click.argument(
    "model_path",
    type=click.Path(),
    required=False,
    default=".",
)
@click.option(
    '--model-type-id',
    type=click.Choice(['mcp', 'openai'], case_sensitive=False),
    required=False,
    help='Model type: "mcp" for MCPModelClass, "openai" for OpenAIModelClass, or leave empty for default ModelClass.',
)
@click.option(
    '--github-pat',
    required=False,
    help='GitHub Personal Access Token for authentication when cloning private repositories.',
)
@click.option(
    '--github-url',
    required=False,
    help='GitHub repository URL or "user/repo" format to clone a repository from. If provided, the entire repository contents will be copied to the target directory instead of using default templates.',
)
@click.option(
    '--toolkit',
    type=click.Choice(
        ['ollama', 'huggingface', 'lmstudio', 'vllm', 'sglang', 'python'], case_sensitive=False
    ),
    required=False,
    help='Toolkit to use for model initialization. Currently supports "ollama", "huggingface", "lmstudio", "vllm", "sglang" and "python".',
)
@click.option(
    '--model-name',
    required=False,
    help='Model name to configure when using --toolkit. For ollama toolkit, this sets the Ollama model to use (e.g., "llama3.1", "mistral", etc.). For vllm, sglang & huggingface toolkit, this sets the Hugging Face model repo_id (e.g., "unsloth/Llama-3.2-1B-Instruct").\n For lmstudio toolkit, this sets the LM Studio model name (e.g., "qwen/qwen3-4b-thinking-2507").\n',
)
@click.option(
    '--port',
    type=str,
    help='Port to run the Ollama server on. Defaults to 23333.',
    required=False,
)
@click.option(
    '--context-length',
    type=str,
    help='Context length for the Ollama model. Defaults to 8192.',
    required=False,
)
@click.pass_context
def init(
    ctx,
    model_path,
    model_type_id,
    github_pat,
    github_url,
    toolkit,
    model_name,
    port,
    context_length,
):
    """Initialize a new model directory structure.

    Creates the following structure in the specified directory:\n
    ├── 1/\n
    │   └── model.py\n
    ├── requirements.txt\n
    └── config.yaml\n

    If --github-repo is provided, the entire repository contents will be copied to the target
    directory instead of using default templates. The --github-pat option can be used for authentication
    when cloning private repositories. The --branch option can be used to specify a specific
    branch to clone from.

    MODEL_PATH: Path where to create the model directory structure. If not specified, the current directory is used by default.\n

    OPTIONS:\n
    MODEL_TYPE_ID: Type of model to create. If not specified, defaults to "text-to-text" for text models.\n
    GITHUB_PAT: GitHub Personal Access Token for authentication when cloning private repositories.\n
    GITHUB_URL: GitHub repository URL or "repo" format to clone a repository from. If provided, the entire repository contents will be copied to the target directory instead of using default templates.\n
    TOOLKIT: Toolkit to use for model initialization. Currently supports "ollama", "huggingface", "lmstudio", "vllm", "sglang" and "python".\n
    MODEL_NAME: Model name to configure when using --toolkit. For ollama toolkit, this sets the Ollama model to use (e.g., "llama3.1", "mistral", etc.). For vllm, sglang & huggingface toolkit, this sets the Hugging Face model repo_id (e.g., "Qwen/Qwen3-4B-Instruct-2507"). For lmstudio toolkit, this sets the LM Studio model name (e.g., "qwen/qwen3-4b-thinking-2507").\n
    PORT: Port to run the (Ollama/lmstudio) server on. Defaults to 23333.\n
    CONTEXT_LENGTH: Context length for the (Ollama/lmstudio) model. Defaults to 8192.\n
    """
    validate_context(ctx)
    user_id = ctx.obj.current.user_id
    # Resolve the absolute path
    model_path = os.path.abspath(model_path)

    # Create the model directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)

    # Validate parameters
    if port and not port.isdigit():
        logger.error("Invalid value: --port must be a number")
        raise click.Abort()

    if context_length and not context_length.isdigit():
        logger.error("Invalid value: --context-length must be a number")
        raise click.Abort()

    # Validate option combinations
    if model_name and not (toolkit):
        logger.error("--model-name can only be used with --toolkit")
        raise click.Abort()

    if toolkit and (github_url):
        logger.error("Cannot specify both --toolkit and --github-repo")
        raise click.Abort()

    # --toolkit option
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

    if github_url:
        downloader = GitHubDownloader(
            max_retries=3,
            github_token=github_pat,
        )
        if toolkit:
            owner, repo, _, folder_path = downloader.parse_github_url(url=github_url)
        else:
            owner, repo, branch, folder_path = downloader.parse_github_url(url=github_url)
        logger.info(
            f"Parsed GitHub repository: owner={owner}, repo={repo}, branch={branch}, folder_path={folder_path}"
        )
        files_to_download = get_list_of_files_to_download(
            downloader, owner, repo, folder_path, branch, []
        )
        for i, file in enumerate(files_to_download):
            files_to_download[i] = f"{i + 1}. {file}"
        files_to_download = '\n'.join(files_to_download)
        logger.info(f"Files to be downloaded are:\n{files_to_download}")
        input("Press Enter to continue...")
        if not toolkit:
            if folder_path != "":
                try:
                    downloader.download_github_folder(
                        url=github_url,
                        output_dir=model_path,
                        github_token=github_pat,
                    )
                    logger.info(f"Successfully downloaded folder contents to {model_path}")
                    logger.info("Model initialization complete with GitHub folder download")
                    logger.info("Next steps:")
                    logger.info("1. Review the model configuration")
                    logger.info("2. Install any required dependencies manually")
                    logger.info("3. Test the model locally using 'clarifai model local-test'")
                    return

                except Exception as e:
                    logger.error(f"Failed to download GitHub folder: {e}")
                    # Continue with the rest of the initialization process
                    github_url = None  # Fall back to template mode

            elif branch and folder_path == "":
                # When we have a branch but no specific folder path
                logger.info(
                    f"Initializing model from GitHub repository: {github_url} (branch: {branch})"
                )

                # Check if it's a local path or normalize the GitHub repo URL
                if os.path.exists(github_url):
                    repo_url = github_url
                else:
                    repo_url = format_github_repo_url(github_url)
                    repo_url = f"https://github.com/{owner}/{repo}"

                try:
                    # Create a temporary directory for cloning
                    with tempfile.TemporaryDirectory(prefix="clarifai_model_") as clone_dir:
                        # Clone the repository with explicit branch parameter
                        if not clone_github_repo(repo_url, clone_dir, github_pat, branch):
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

                            logger.info(f"Successfully cloned repository to {model_path}")
                            logger.info(
                                "Model initialization complete with GitHub repository clone"
                            )
                            logger.info("Next steps:")
                            logger.info("1. Review the model configuration")
                            logger.info("2. Install any required dependencies manually")
                            logger.info(
                                "3. Test the model locally using 'clarifai model local-test'"
                            )
                            return

                except Exception as e:
                    logger.error(f"Failed to clone GitHub repository: {e}")
                    github_url = None  # Fall back to template mode

    if toolkit:
        logger.info(f"Initializing model from GitHub repository: {github_url}")

        # Check if it's a local path or normalize the GitHub repo URL
        if os.path.exists(github_url):
            repo_url = github_url
        else:
            repo_url = format_github_repo_url(github_url)

        try:
            # Create a temporary directory for cloning
            with tempfile.TemporaryDirectory(prefix="clarifai_model_") as clone_dir:
                # Clone the repository with explicit branch parameter
                if not clone_github_repo(repo_url, clone_dir, github_pat, branch):
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

    if (user_id or model_name or port or context_length) and (toolkit == 'ollama'):
        customize_ollama_model(model_path, user_id, model_name, port, context_length)

    if (user_id or model_name or port or context_length) and (toolkit == 'lmstudio'):
        customize_lmstudio_model(model_path, user_id, model_name, port, context_length)

    if (user_id or model_name) and (
        toolkit == 'huggingface' or toolkit == 'vllm' or toolkit == 'sglang'
    ):
        # Update the config.yaml file with the provided model name
        customize_huggingface_model(model_path, user_id, model_name)

    if github_url:
        logger.info("Model initialization complete with GitHub repository")
        logger.info("Next steps:")
        logger.info("1. Review the model configuration")
        logger.info("2. Install any required dependencies manually")
        logger.info("3. Test the model locally using 'clarifai model local-test'")

    # Fall back to template-based initialization if no GitHub repo or if GitHub repo failed
    if not github_url:
        logger.info("Initializing model with default templates...")
        input("Press Enter to continue...")

        from clarifai.cli.base import input_or_default
        from clarifai.cli.templates.model_templates import (
            get_config_template,
            get_model_template,
            get_requirements_template,
        )

        # Collect additional parameters for OpenAI template
        template_kwargs = {}
        if model_type_id == "openai":
            logger.info("Configuring OpenAI local runner...")
            port = input_or_default("Enter port (default: 8000): ", "8000")
            template_kwargs = {"port": port}

        # Create the 1/ subdirectory
        model_version_dir = os.path.join(model_path, "1")
        os.makedirs(model_version_dir, exist_ok=True)

        # Create model.py
        model_py_path = os.path.join(model_version_dir, "model.py")
        if os.path.exists(model_py_path):
            logger.warning(f"File {model_py_path} already exists, skipping...")
        else:
            model_template = get_model_template(model_type_id, **template_kwargs)
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
            config_model_type_id = DEFAULT_LOCAL_RUNNER_MODEL_TYPE  # default

            config_template = get_config_template(
                user_id=user_id, model_type_id=config_model_type_id
            )
            with open(config_path, 'w') as f:
                f.write(config_template)
            logger.info(f"Created {config_path}")

        logger.info(f"Model initialization complete in {model_path}")
        logger.info("Next steps:")
        logger.info("1. Search for '# TODO: please fill in' comments in the generated files")
        logger.info("2. Update the model configuration in config.yaml")
        logger.info("3. Add your model dependencies to requirements.txt")
        logger.info("4. Implement your model logic in 1/model.py")


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
    '--stage',
    required=False,
    type=click.Choice(['runtime', 'build', 'upload'], case_sensitive=True),
    default="upload",
    show_default=True,
    help='The stage we are calling download checkpoints from. Typically this would "upload" and will download checkpoints if config.yaml checkpoints section has when set to "upload". Other options include "runtime" to be used in load_model or "upload" to be used during model upload. Set this stage to whatever you have in config.yaml to force downloading now.',
)
@click.option(
    '--skip_dockerfile',
    is_flag=True,
    help='Flag to skip generating a dockerfile so that you can manually edit an already created dockerfile. If not provided, intelligently handle existing Dockerfiles with user confirmation.',
)
@click.option(
    '--platform',
    required=False,
    help='Target platform(s) for Docker image build (e.g., "linux/amd64" or "linux/amd64,linux/arm64"). This overrides the platform specified in config.yaml.',
)
@click.pass_context
def upload(ctx, model_path, stage, skip_dockerfile, platform):
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
        stage,
        skip_dockerfile,
        platform=platform,
        pat=ctx.obj.current.pat,
        base_url=ctx.obj.current.api_base,
    )


@model.command(help="Download model checkpoint files.")
@click.argument(
    "model_path",
    type=click.Path(exists=True),
    required=False,
    default=".",
)
@click.option(
    '--out_path',
    type=click.Path(exists=False),
    required=False,
    default=None,
    help='Option path to write the checkpoints to. This will place them in {out_path}/1/checkpoints If not provided it will default to {model_path}/1/checkpoints where the config.yaml is read.',
)
@click.option(
    '--stage',
    required=False,
    type=click.Choice(['runtime', 'build', 'upload'], case_sensitive=True),
    default="build",
    show_default=True,
    help='The stage we are calling download checkpoints from. Typically this would be in the build stage which is the default. Other options include "runtime" to be used in load_model or "upload" to be used during model upload. Set this stage to whatever you have in config.yaml to force downloading now.',
)
@click.pass_context
def download_checkpoints(ctx, model_path, out_path, stage):
    """Download checkpoints from external source to local model_path

    MODEL_PATH: Path to the model directory. If not specified, the current directory is used by default.
    """

    from clarifai.runners.models.model_builder import ModelBuilder

    validate_context(ctx)
    _ensure_hf_token(ctx, model_path)
    builder = ModelBuilder(model_path, download_validation_only=True)
    builder.download_checkpoints(stage=stage, checkpoint_path_override=out_path)


@model.command(help="Generate model method signatures.")
@click.argument(
    "model_path",
    type=click.Path(exists=True),
    required=False,
    default=".",
)
@click.option(
    '--out_path',
    type=click.Path(exists=False),
    required=False,
    default=None,
    help='Path to write the method signature defitions to. If not provided, use stdout.',
)
def signatures(model_path, out_path):
    """Generate method signatures for the model.

    MODEL_PATH: Path to the model directory. If not specified, the current directory is used by default.
    """

    from clarifai.runners.models.model_builder import ModelBuilder

    builder = ModelBuilder(model_path, download_validation_only=True)
    signatures = builder.method_signatures_yaml()
    if out_path:
        with open(out_path, 'w') as f:
            f.write(signatures)
    else:
        click.echo(signatures)


@model.command(name="local-test", help="Execute all model unit tests locally.")
@click.argument(
    "model_path",
    type=click.Path(exists=True),
    required=False,
    default=".",
)
@click.option(
    '--mode',
    type=click.Choice(['env', 'container'], case_sensitive=False),
    default='env',
    show_default=True,
    help='Specify how to test the model locally: "env" for virtual environment or "container" for Docker container. Defaults to "env".',
)
@click.option(
    '--keep_env',
    is_flag=True,
    help='Keep the virtual environment after testing the model locally (applicable for virtualenv mode). Defaults to False.',
)
@click.option(
    '--keep_image',
    is_flag=True,
    help='Keep the Docker image after testing the model locally (applicable for container mode). Defaults to False.',
)
@click.option(
    '--skip_dockerfile',
    is_flag=True,
    help='Flag to skip generating a dockerfile so that you can manually edit an already created dockerfile. If not provided, intelligently handle existing Dockerfiles with user confirmation.',
)
@click.pass_context
def test_locally(
    ctx, model_path, keep_env=False, keep_image=False, mode='env', skip_dockerfile=False
):
    """Test model locally.

    MODEL_PATH: Path to the model directory. If not specified, the current directory is used by default.
    """
    try:
        from clarifai.runners.models import model_run_locally

        validate_context(ctx)
        _ensure_hf_token(ctx, model_path)
        if mode == 'env' and keep_image:
            raise ValueError("'keep_image' is applicable only for 'container' mode")
        if mode == 'container' and keep_env:
            raise ValueError("'keep_env' is applicable only for 'env' mode")

        if mode == "env":
            click.echo("Testing model locally in a virtual environment...")
            model_run_locally.main(model_path, run_model_server=False, keep_env=keep_env)
        elif mode == "container":
            click.echo("Testing model locally inside a container...")
            model_run_locally.main(
                model_path,
                inside_container=True,
                run_model_server=False,
                keep_image=keep_image,
                skip_dockerfile=skip_dockerfile,
            )
        click.echo("Model tested successfully.")
    except Exception as e:
        click.echo(f"Failed to test model locally: {e}", err=True)


@model.command(name="local-grpc", help="Run the model locally via a gRPC server.")
@click.argument(
    "model_path",
    type=click.Path(exists=True),
    required=False,
    default=".",
)
@click.option(
    '--port',
    '-p',
    type=int,
    default=8000,
    show_default=True,
    help="The port to host the gRPC server for running the model locally. Defaults to 8000.",
)
@click.option(
    '--mode',
    type=click.Choice(['env', 'container'], case_sensitive=False),
    default='env',
    show_default=True,
    help='Specifies how to run the model: "env" for virtual environment or "container" for Docker container. Defaults to "env".',
)
@click.option(
    '--keep_env',
    is_flag=True,
    help='Keep the virtual environment after testing the model locally (applicable for virtualenv mode). Defaults to False.',
)
@click.option(
    '--keep_image',
    is_flag=True,
    help='Keep the Docker image after testing the model locally (applicable for container mode). Defaults to False.',
)
@click.option(
    '--skip_dockerfile',
    is_flag=True,
    help='Flag to skip generating a dockerfile so that you can manually edit an already created dockerfile. If not provided, intelligently handle existing Dockerfiles with user confirmation.',
)
@click.pass_context
def run_locally(ctx, model_path, port, mode, keep_env, keep_image, skip_dockerfile=False):
    """Run the model locally and start a gRPC server to serve the model.

    MODEL_PATH: Path to the model directory. If not specified, the current directory is used by default.
    """
    model_path = os.path.abspath(model_path)
    try:
        from clarifai.runners.models import model_run_locally

        validate_context(ctx)
        _ensure_hf_token(ctx, model_path)
        if mode == 'env' and keep_image:
            raise ValueError("'keep_image' is applicable only for 'container' mode")
        if mode == 'container' and keep_env:
            raise ValueError("'keep_env' is applicable only for 'env' mode")

        if mode == "env":
            click.echo("Running model locally in a virtual environment...")
            model_run_locally.main(model_path, run_model_server=True, keep_env=keep_env, port=port)
        elif mode == "container":
            click.echo("Running model locally inside a container...")
            model_run_locally.main(
                model_path,
                inside_container=True,
                run_model_server=True,
                port=port,
                keep_image=keep_image,
                skip_dockerfile=skip_dockerfile,
            )
        click.echo(f"Model server started locally from {model_path} in {mode} mode.")
    except Exception as e:
        click.echo(f"Failed to starts model server locally: {e}", err=True)


@model.command(name="local-runner", help="Run the model locally for dev, debug, or local compute.")
@click.argument(
    "model_path",
    type=click.Path(exists=True),
    required=False,
    default=".",
)
@click.option(
    "--pool_size",
    type=int,
    default=32,
    show_default=True,
    help="The number of threads to use. On community plan, the compute time allocation is drained at a rate proportional to the number of threads.",
)  # pylint: disable=range-builtin-not-iterating
@click.option(
    '--suppress-toolkit-logs',
    is_flag=True,
    help='Show detailed logs including Ollama server output. By default, Ollama logs are suppressed.',
)
@click.option(
    "--mode",
    type=click.Choice(['env', 'container', 'none'], case_sensitive=False),
    default='none',
    show_default=True,
    help='Specifies how to run the model: "env" for virtual environment, "container" for Docker container, or "none" to skip creating environment and directly run the model. Defaults to "none".',
)
@click.option(
    '--keep_image',
    is_flag=True,
    help='Keep the Docker image after testing the model locally (applicable for container mode). Defaults to False.',
)
@click.pass_context
def local_runner(ctx, model_path, pool_size, suppress_toolkit_logs, mode, keep_image):
    """Run the model as a local runner to help debug your model connected to the API or to
      leverage local compute resources manually. This relies on many variables being present in the env
      of the currently selected context. If they are not present then default values will be used to
      ease the setup of a local runner and your context yaml will be updated in place. The required
      env vars are:

    \b
      CLARIFAI_PAT:

    \b
      # for where the model that represents the local runner should be:
    \b
      CLARIFAI_USER_ID:
      CLARIFAI_APP_ID:
      CLARIFAI_MODEL_ID:
    \b
      # for where the local runner should be in a compute cluster
      # note the user_id of the compute cluster is the same as the user_id of the model.
    \b
      CLARIFAI_COMPUTE_CLUSTER_ID:
      CLARIFAI_NODEPOOL_ID:

      # The following will be created in your context since it's generated by the API

      CLARIFAI_RUNNER_ID:


    Additionally using the provided model path, if the config.yaml file does not contain the model
    information that matches the above CLARIFAI_USER_ID, CLARIFAI_APP_ID, CLARIFAI_MODEL_ID then the
    config.yaml will be updated to include the model information. This is to ensure that the model
    that starts up in the local runner is the same as the one you intend to call in the API.

    MODEL_PATH: Path to the model directory. If not specified, the current directory is used by default.
    MODE: Specifies how to run the model: "env" for virtual environment or "container" for Docker container. Defaults to "env".
    KEEP_IMAGE: Keep the Docker image after testing the model locally (applicable for container mode). Defaults to False.
    """
    from clarifai.client.user import User
    from clarifai.runners.models.model_builder import ModelBuilder
    from clarifai.runners.models.model_run_locally import ModelRunLocally
    from clarifai.runners.server import ModelServer

    validate_context(ctx)
    model_path = os.path.abspath(model_path)
    _ensure_hf_token(ctx, model_path)
    builder = ModelBuilder(model_path, download_validation_only=True)
    manager = ModelRunLocally(model_path)

    port = 8080
    if mode == "env":
        manager.create_temp_venv()
        manager.install_requirements()

    dependencies = parse_requirements(model_path)
    if mode != "container":
        logger.info("> Checking local runner requirements...")
        # Post check while running `clarifai model local-runner` we check if the toolkit is ollama
        if not check_requirements_installed(dependencies=dependencies):
            logger.error(f"Requirements not installed for model at {model_path}.")
            raise click.Abort()

    if "ollama" in dependencies or builder.config.get('toolkit', {}).get('provider') == 'ollama':
        logger.info("Verifying Ollama installation...")
        if not check_ollama_installed():
            logger.error(
                "Ollama application is not installed. Please install it from `https://ollama.com/` to use the Ollama toolkit."
            )
            raise click.Abort()
    elif (
        "lmstudio" in dependencies
        or builder.config.get('toolkit', {}).get('provider') == 'lmstudio'
    ):
        logger.info("Verifying LM Studio installation...")
        if not check_lmstudio_installed():
            logger.error(
                "LM Studio application is not installed. Please install it from `https://lmstudio.com/` to use the LM Studio toolkit."
            )
            raise click.Abort()

    # Load model config
    config_file = os.path.join(model_path, 'config.yaml')
    if not os.path.exists(config_file):
        logger.error(
            f"config.yaml not found in {model_path}. Please ensure you are passing the correct directory."
        )
        raise click.Abort()
    config = ModelBuilder._load_config(config_file)

    uploaded_model_type_id = config.get('model', {}).get(
        'model_type_id', DEFAULT_LOCAL_RUNNER_MODEL_TYPE
    )

    logger.info("> Verifying local runner setup...")
    logger.info(f"Current context: {ctx.obj.current.name}")
    user_id = ctx.obj.current.user_id
    logger.info(f"Current user_id: {user_id}")
    if not user_id:
        logger.error(f"User with ID '{user_id}' not found. Use 'clarifai login' to setup context.")
        raise click.Abort()
    pat = ctx.obj.current.pat
    display_pat = pat_display(pat) if pat else ""
    logger.info(f"Current PAT: {display_pat}")
    if not pat:
        logger.error(
            "Personal Access Token (PAT) not found. Use 'clarifai login' to setup context."
        )
        raise click.Abort()
    user = User(user_id=user_id, pat=ctx.obj.current.pat, base_url=ctx.obj.current.api_base)
    logger.debug("Checking if a local runner compute cluster exists...")

    # see if ctx has CLARIFAI_COMPUTE_CLUSTER_ID, if not use default
    try:
        compute_cluster_id = ctx.obj.current.compute_cluster_id
    except AttributeError:
        compute_cluster_id = DEFAULT_LOCAL_RUNNER_COMPUTE_CLUSTER_ID
    logger.info(f"Current compute_cluster_id: {compute_cluster_id}")

    try:
        compute_cluster = user.compute_cluster(compute_cluster_id)
        if compute_cluster.cluster_type != 'local-dev':
            raise ValueError(
                f"Compute cluster {user_id}/{compute_cluster_id} is not a compute cluster of type 'local-dev'. Please use a compute cluster of type 'local-dev'."
            )
        try:
            compute_cluster_id = ctx.obj.current.compute_cluster_id
        except AttributeError:  # doesn't exist in context but does in API then update the context.
            ctx.obj.current.CLARIFAI_COMPUTE_CLUSTER_ID = compute_cluster.id
            ctx.obj.to_yaml()  # save to yaml file.
    except ValueError:
        raise
    except Exception as e:
        logger.warning(f"Failed to get compute cluster with ID '{compute_cluster_id}':\n{e}")
        y = input(
            f"Compute cluster not found. Do you want to create a new compute cluster {user_id}/{compute_cluster_id}? (y/n): "
        )
        if y.lower() != 'y':
            raise click.Abort()
        # Create a compute cluster with default configuration for local runner.
        compute_cluster = user.create_compute_cluster(
            compute_cluster_id=compute_cluster_id,
            compute_cluster_config=DEFAULT_LOCAL_RUNNER_COMPUTE_CLUSTER_CONFIG,
        )
        ctx.obj.current.CLARIFAI_COMPUTE_CLUSTER_ID = compute_cluster_id
        ctx.obj.to_yaml()  # save to yaml file.

    # Now check if there is a nodepool created in this compute cluser
    try:
        nodepool_id = ctx.obj.current.nodepool_id
    except AttributeError:
        nodepool_id = DEFAULT_LOCAL_RUNNER_NODEPOOL_ID
    logger.info(f"Current nodepool_id: {nodepool_id}")

    try:
        nodepool = compute_cluster.nodepool(nodepool_id)
        try:
            nodepool_id = ctx.obj.current.nodepool_id
        except AttributeError:  # doesn't exist in context but does in API then update the context.
            ctx.obj.current.CLARIFAI_NODEPOOL_ID = nodepool.id
            ctx.obj.to_yaml()  # save to yaml file.
    except Exception as e:
        logger.warning(f"Failed to get nodepool with ID '{nodepool_id}':\n{e}")
        y = input(
            f"Nodepool not found. Do you want to create a new nodepool {user_id}/{compute_cluster_id}/{nodepool_id}? (y/n): "
        )
        if y.lower() != 'y':
            raise click.Abort()
        nodepool = compute_cluster.create_nodepool(
            nodepool_config=DEFAULT_LOCAL_RUNNER_NODEPOOL_CONFIG, nodepool_id=nodepool_id
        )
        ctx.obj.current.CLARIFAI_NODEPOOL_ID = nodepool_id
        ctx.obj.to_yaml()  # save to yaml file.

    logger.debug("Checking if model is created to call for local development...")
    # see if ctx has CLARIFAI_APP_ID, if not use default
    try:
        app_id = ctx.obj.current.app_id
    except AttributeError:
        app_id = DEFAULT_LOCAL_RUNNER_APP_ID
    logger.info(f"Current app_id: {app_id}")

    try:
        app = user.app(app_id)
        try:
            app_id = ctx.obj.current.app_id
        except AttributeError:  # doesn't exist in context but does in API then update the context.
            ctx.obj.current.CLARIFAI_APP_ID = app.id
            ctx.obj.to_yaml()  # save to yaml file.
    except Exception as e:
        logger.warning(f"Failed to get app with ID '{app_id}':\n{e}")
        y = input(f"App not found. Do you want to create a new app {user_id}/{app_id}? (y/n): ")
        if y.lower() != 'y':
            raise click.Abort()
        app = user.create_app(app_id)
        ctx.obj.current.CLARIFAI_APP_ID = app_id
        ctx.obj.to_yaml()  # save to yaml file.

    # Within this app we now need a model to call as the local runner.
    try:
        model_id = ctx.obj.current.model_id
    except AttributeError:
        model_id = DEFAULT_LOCAL_RUNNER_MODEL_ID
    logger.info(f"Current model_id: {model_id}")

    try:
        model = app.model(model_id)
        current_model_type_id = model.model_type_id
        try:
            model_id = ctx.obj.current.model_id
        except AttributeError:  # doesn't exist in context but does in API then update the context.
            ctx.obj.current.CLARIFAI_MODEL_ID = model.id
            ctx.obj.to_yaml()  # save to yaml file.
        if current_model_type_id != uploaded_model_type_id:
            logger.warning(
                f"Model type ID mismatch: expected '{uploaded_model_type_id}', found '{current_model_type_id}'. Deleting the model."
            )
            app.delete_model(model_id)
            raise Exception
    except Exception as e:
        logger.warning(f"Failed to get model with ID '{model_id}':\n{e}")
        y = input(
            f"Model not found. Do you want to create a new model {user_id}/{app_id}/models/{model_id}? (y/n): "
        )
        if y.lower() != 'y':
            raise click.Abort()

        model = app.create_model(model_id, model_type_id=uploaded_model_type_id)
        ctx.obj.current.CLARIFAI_MODEL_TYPE_ID = uploaded_model_type_id
        ctx.obj.current.CLARIFAI_MODEL_ID = model_id
        ctx.obj.to_yaml()  # save to yaml file.

    # Now we need to create a version for the model if no version exists. Only need one version that
    # mentions it's a local runner.
    model_versions = list(model.list_versions())
    method_signatures = manager._get_method_signatures()

    create_new_version = False
    if len(model_versions) == 0:
        logger.warning("No model versions found. Creating a new version for local runner.")
        create_new_version = True
    else:
        # Try to patch the latest version, and fallback to creating a new one if that fails.
        latest_version = model_versions[0]
        logger.warning(f"Attempting to patch latest version: {latest_version.model_version.id}")

        try:
            patched_model = model.patch_version(
                version_id=latest_version.model_version.id,
                pretrained_model_config={"local_dev": True},
                method_signatures=method_signatures,
            )
            patched_model.load_info()
            version = patched_model.model_version
            logger.info(f"Successfully patched version {version.id}")
            ctx.obj.current.CLARIFAI_MODEL_VERSION_ID = version.id
            ctx.obj.to_yaml()  # save to yaml file.
        except Exception as e:
            logger.warning(f"Failed to patch model version: {e}. Creating a new version instead.")
            create_new_version = True

    if create_new_version:
        version = model.create_version(
            pretrained_model_config={"local_dev": True}, method_signatures=method_signatures
        ).model_version
        ctx.obj.current.CLARIFAI_MODEL_VERSION_ID = version.id
        ctx.obj.to_yaml()

    logger.info(f"Current model version {version.id}")

    worker = {
        "model": {
            "id": f"{model.id}",
            "model_version": {
                "id": f"{version.id}",
            },
            "user_id": f"{user_id}",
            "app_id": f"{app_id}",
        },
    }

    try:
        # if it's already in our context then we'll re-use the same one.
        # note these are UUIDs, we cannot provide a runner ID.
        runner_id = ctx.obj.current.runner_id

        try:
            runner = nodepool.runner(runner_id)
            # ensure the deployment is using the latest version.
            if runner.worker.model.model_version.id != version.id:
                nodepool.delete_runners([runner_id])
                logger.warning("Deleted runner that was for an old model version ID.")
                raise AttributeError(
                    "Runner deleted because it was associated with an outdated model version."
                )
        except Exception as e:
            logger.warning(f"Failed to get runner with ID '{runner_id}':\n{e}")
            raise AttributeError("Runner not found in nodepool.")
    except AttributeError:
        logger.info(
            f"Creating the local runner tying this '{user_id}/{app_id}/models/{model.id}' model (version: {version.id}) to the '{user_id}/{compute_cluster_id}/{nodepool_id}' nodepool."
        )
        try:
            logger.info("Checking for existing runners in the nodepool...")
            runners = nodepool.list_runners(
                model_version_ids=[version.id],
            )
            runner_id = None
            for runner in runners:
                logger.info(
                    f"Found existing runner {runner.id} for model version {version.id}. Reusing it."
                )
                runner_id = runner.id
                break  # use the first one we find.
            if runner_id is None:
                logger.warning("No existing runners found in nodepool. Creating a new one.\n")
                runner = nodepool.create_runner(
                    runner_config={
                        "runner": {
                            "description": "local runner for model testing",
                            "worker": worker,
                            "num_replicas": 1,
                        }
                    }
                )
                runner_id = runner.id
        except Exception as e:
            logger.warning(
                f"Failed to list existing runners in nodepool {e}...Creating a new one.\n"
            )
            runner = nodepool.create_runner(
                runner_config={
                    "runner": {
                        "description": "local runner for model testing",
                        "worker": worker,
                        "num_replicas": 1,
                    }
                }
            )
            runner_id = runner.id
        ctx.obj.current.CLARIFAI_RUNNER_ID = runner.id
        ctx.obj.to_yaml()

    logger.info(f"Current runner_id: {runner_id}")

    # To make it easier to call the model without specifying a runner selector
    # we will also create a deployment tying the model to the nodepool.
    try:
        deployment_id = ctx.obj.current.deployment_id
    except AttributeError:
        deployment_id = DEFAULT_LOCAL_RUNNER_DEPLOYMENT_ID
    try:
        deployment = nodepool.deployment(deployment_id)
        # ensure the deployment is using the latest version.
        if deployment.worker.model.model_version.id != version.id:
            nodepool.delete_deployments([deployment_id])
            logger.warning("Deleted deployment that was for an old model version ID.")
            raise Exception(
                "Deployment deleted because it was associated with an outdated model version."
            )
        try:
            deployment_id = ctx.obj.current.deployment_id
        except AttributeError:  # doesn't exist in context but does in API then update the context.
            ctx.obj.current.CLARIFAI_DEPLOYMENT_ID = deployment.id
            ctx.obj.to_yaml()  # save to yaml file.
    except Exception as e:
        logger.warning(f"Failed to get deployment with ID {deployment_id}:\n{e}")
        y = input(
            f"Deployment not found. Do you want to create a new deployment {user_id}/{compute_cluster_id}/{nodepool_id}/{deployment_id}? (y/n): "
        )
        if y.lower() != 'y':
            raise click.Abort()
        nodepool.create_deployment(
            deployment_id=deployment_id,
            deployment_config={
                "deployment": {
                    "scheduling_choice": 3,  # 3 means by price
                    "worker": worker,
                    "nodepools": [
                        {
                            "id": f"{nodepool_id}",
                            "compute_cluster": {
                                "id": f"{compute_cluster_id}",
                                "user_id": f"{user_id}",
                            },
                        }
                    ],
                    "deploy_latest_version": True,
                }
            },
        )
        ctx.obj.current.CLARIFAI_DEPLOYMENT_ID = deployment_id
        ctx.obj.to_yaml()  # save to yaml file.

    logger.info(f"Current deployment_id: {deployment_id}")

    # Now that we have all the context in ctx.obj, we need to update the config.yaml in
    # the model_path directory with the model object containing user_id, app_id, model_id, version_id
    # The config.yaml doens't match what we created above.
    if 'model' in config and model_id != config['model'].get('id'):
        logger.info(f"Current model section of config.yaml: {config.get('model', {})}")
        y = input(
            "Do you want to backup config.yaml to config.yaml.bk then update the config.yaml with the new model information? (y/n): "
        )
        if y.lower() != 'y':
            raise click.Abort()
        config = ModelBuilder._set_local_runner_model(
            config, user_id, app_id, model_id, uploaded_model_type_id
        )
        ModelBuilder._backup_config(config_file)
        ModelBuilder._save_config(config_file, config)

    # Post check while running `clarifai model local-runner` we check if the toolkit is ollama
    if builder.config.get('toolkit', {}).get('provider') == 'ollama':
        try:
            logger.info("Customizing Ollama model with provided parameters...")
            customize_ollama_model(
                model_path=model_path,
                user_id=user_id,
                verbose=False if suppress_toolkit_logs else True,
            )
        except Exception as e:
            logger.error(f"Failed to customize Ollama model: {e}")
            raise click.Abort()
    elif builder.config.get('toolkit', {}).get('provider') == 'lmstudio':
        try:
            logger.info("Customizing LM Studio model with provided parameters...")
            customize_lmstudio_model(
                model_path=model_path,
                user_id=user_id,
            )
        except Exception as e:
            logger.error(f"Failed to customize LM Studio model: {e}")
            raise click.Abort()

    logger.info("✅ Starting local runner...")

    def print_code_snippet():
        if ctx.obj.current is None:
            logger.debug("Context is None. Skipping code snippet generation.")
        else:
            from clarifai.runners.utils import code_script

            snippet = code_script.generate_client_script(
                method_signatures,
                user_id=ctx.obj.current.user_id,
                app_id=ctx.obj.current.app_id,
                model_id=ctx.obj.current.model_id,
                deployment_id=ctx.obj.current.deployment_id,
                base_url=ctx.obj.current.api_base,
                colorize=True,
            )
            logger.info(
                "✅ Your model is running locally and is ready for requests from the API...\n"
            )
            logger.info(
                f"> Code Snippet: To call your model via the API, use this code snippet:\n{snippet}"
            )
            logger.info(
                f"> Playground:   To chat with your model, visit: {ctx.obj.current.ui}/playground?model={ctx.obj.current.model_id}__{ctx.obj.current.model_version_id}&user_id={ctx.obj.current.user_id}&app_id={ctx.obj.current.app_id}\n"
            )
            logger.info(
                f"> API URL:      To call your model via the API, use this model URL: {ctx.obj.current.ui}/{ctx.obj.current.user_id}/{ctx.obj.current.app_id}/models/{ctx.obj.current.model_id}\n"
            )
            logger.info("Press CTRL+C to stop the runner.\n")

    serving_args = {
        "pool_size": pool_size,
        "num_threads": pool_size,
        "user_id": user_id,
        "compute_cluster_id": compute_cluster_id,
        "nodepool_id": nodepool_id,
        "runner_id": runner_id,
        "base_url": ctx.obj.current.api_base,
        "pat": ctx.obj.current.pat,
        "context": ctx.obj.current,
    }

    if mode == "container":
        try:
            if not manager.is_docker_installed():
                raise click.abort()

            image_tag = manager._docker_hash()
            model_id = manager.config['model']['id'].lower()
            # must be in lowercase
            image_name = f"{model_id}:{image_tag}"
            container_name = model_id
            if not manager.docker_image_exists(image_name):
                manager.build_docker_image(image_name=image_name)

            manager.build_docker_image(image_name=image_name)
            print_code_snippet()
            manager.run_docker_container(
                image_name=image_name,
                container_name=container_name,
                port=port,
                is_local_runner=True,
                env_vars={"CLARIFAI_PAT": ctx.obj.current.pat},
                **serving_args,
            )

        finally:
            if manager.container_exists(container_name):
                manager.stop_docker_container(container_name)
                manager.remove_docker_container(container_name=container_name)
            if not keep_image:
                manager.remove_docker_image(image_name=image_name)
    else:
        print_code_snippet()
        # This reads the config.yaml from the model_path so we alter it above first.
        server = ModelServer(model_path=model_path, model_runner_local=None)
        server.serve(**serving_args)


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
