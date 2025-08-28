import os
import shutil
import tempfile

import click

from clarifai.cli.base import cli, pat_display
from clarifai.utils.cli import (
    check_ollama_installed,
    check_requirements_installed,
    customize_ollama_model,
    parse_requirements,
    validate_context,
)
from clarifai.utils.constants import (
    DEFAULT_LOCAL_RUNNER_APP_ID,
    DEFAULT_LOCAL_RUNNER_COMPUTE_CLUSTER_CONFIG,
    DEFAULT_LOCAL_RUNNER_COMPUTE_CLUSTER_ID,
    DEFAULT_LOCAL_RUNNER_DEPLOYMENT_ID,
    DEFAULT_LOCAL_RUNNER_MODEL_ID,
    DEFAULT_LOCAL_RUNNER_MODEL_TYPE,
    DEFAULT_LOCAL_RUNNER_NODEPOOL_CONFIG,
    DEFAULT_LOCAL_RUNNER_NODEPOOL_ID,
    DEFAULT_OLLAMA_MODEL_REPO,
    DEFAULT_OLLAMA_MODEL_REPO_BRANCH,
)
from clarifai.utils.logging import logger
from clarifai.utils.misc import GitHubDownloader, clone_github_repo, format_github_repo_url


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
    type=click.Choice(['ollama'], case_sensitive=False),
    required=False,
    help='Toolkit to use for model initialization. Currently supports "ollama".',
)
@click.option(
    '--model-name',
    required=False,
    help='Model name to configure when using --toolkit. For ollama toolkit, this sets the Ollama model to use (e.g., "llama3.1", "mistral", etc.).',
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
def init(
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

    MODEL_PATH: Path where to create the model directory structure. If not specified, the current directory is used by default.
    MODEL_TYPE_ID: Type of model to create. If not specified, defaults to "text-to-text" for text models.
    GITHUB_PAT: GitHub Personal Access Token for authentication when cloning private repositories.
    GITHUB_URL: GitHub repository URL or "repo" format to clone a repository from. If provided, the entire repository contents will be copied to the target directory instead of using default templates.
    TOOLKIT: Toolkit to use for model initialization. Currently supports "ollama".
    MODEL_NAME: Model name to configure when using --toolkit. For ollama toolkit, this sets the Ollama model to use (e.g., "llama3.1", "mistral", etc.).
    PORT: Port to run the Ollama server on. Defaults to 23333.
    CONTEXT_LENGTH: Context length for the Ollama model. Defaults to 8192.
    """
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
        github_url = DEFAULT_OLLAMA_MODEL_REPO
        branch = DEFAULT_OLLAMA_MODEL_REPO_BRANCH

    if github_url:
        if not toolkit:
            owner, repo, branch, folder_path = GitHubDownloader().parse_github_url(url=github_url)
            logger.info(
                f"Parsed GitHub repository: owner={owner}, repo={repo}, branch={branch}, folder_path={folder_path}"
            )
            if folder_path != "":
                downloader = GitHubDownloader(
                    max_retries=3,
                    github_token=github_pat,
                )
                try:
                    downloader.download_github_folder(
                        url=github_url,
                        output_dir=model_path,
                        github_token=github_pat,
                    )
                    logger.info(f"Successfully downloaded folder contents to {model_path}")
                    logger.info("Model initialization complete with GitHub folder download")
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

    if (model_name or port or context_length) and (toolkit == 'ollama'):
        customize_ollama_model(model_path, model_name, port, context_length)

    if github_url:
        logger.info("Model initialization complete with GitHub repository")
        logger.info("Next steps:")
        logger.info("1. Review the model configuration")
        logger.info("2. Install any required dependencies manually")
        logger.info("3. Test the model locally using 'clarifai model local-test'")

    # Fall back to template-based initialization if no GitHub repo or if GitHub repo failed
    if not github_url:
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
            config_model_type_id = "text-to-text"  # default

            config_template = get_config_template(config_model_type_id)
            with open(config_path, 'w') as f:
                f.write(config_template)
            logger.info(f"Created {config_path}")

        logger.info(f"Model initialization complete in {model_path}")
        logger.info("Next steps:")
        logger.info("1. Search for '# TODO: please fill in' comments in the generated files")
        logger.info("2. Update the model configuration in config.yaml")
        logger.info("3. Add your model dependencies to requirements.txt")
        logger.info("4. Implement your model logic in 1/model.py")


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
@click.pass_context
def upload(ctx, model_path, stage, skip_dockerfile):
    """Upload a model to Clarifai.

    MODEL_PATH: Path to the model directory. If not specified, the current directory is used by default.
    """
    from clarifai.runners.models.model_builder import upload_model

    validate_context(ctx)
    upload_model(
        model_path,
        stage,
        skip_dockerfile,
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
def download_checkpoints(model_path, out_path, stage):
    """Download checkpoints from external source to local model_path

    MODEL_PATH: Path to the model directory. If not specified, the current directory is used by default.
    """

    from clarifai.runners.models.model_builder import ModelBuilder

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
def test_locally(model_path, keep_env=False, keep_image=False, mode='env', skip_dockerfile=False):
    """Test model locally.

    MODEL_PATH: Path to the model directory. If not specified, the current directory is used by default.
    """
    try:
        from clarifai.runners.models import model_run_locally

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
def run_locally(model_path, port, mode, keep_env, keep_image, skip_dockerfile=False):
    """Run the model locally and start a gRPC server to serve the model.

    MODEL_PATH: Path to the model directory. If not specified, the current directory is used by default.
    """
    try:
        from clarifai.runners.models import model_run_locally

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
    '--verbose',
    is_flag=True,
    help='Show detailed logs including Ollama server output. By default, Ollama logs are suppressed.',
)
@click.pass_context
def local_runner(ctx, model_path, pool_size, verbose):
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
    """
    from clarifai.client.user import User
    from clarifai.runners.models.model_builder import ModelBuilder
    from clarifai.runners.server import ModelServer

    validate_context(ctx)
    builder = ModelBuilder(model_path, download_validation_only=True)
    logger.info("> Checking local runner requirements...")
    if not check_requirements_installed(model_path):
        logger.error(f"Requirements not installed for model at {model_path}.")
        raise click.Abort()

    # Post check while running `clarifai model local-runner` we check if the toolkit is ollama
    dependencies = parse_requirements(model_path)
    if "ollama" in dependencies or builder.config.get('toolkit', {}).get('provider') == 'ollama':
        logger.info("Verifying Ollama installation...")
        if not check_ollama_installed():
            logger.error(
                "Ollama application is not installed. Please install it from `https://ollama.com/` to use the Ollama toolkit."
            )
            raise click.Abort()

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
        try:
            model_id = ctx.obj.current.model_id
        except AttributeError:  # doesn't exist in context but does in API then update the context.
            ctx.obj.current.CLARIFAI_MODEL_ID = model.id
            ctx.obj.to_yaml()  # save to yaml file.
    except Exception as e:
        logger.warning(f"Failed to get model with ID '{model_id}':\n{e}")
        y = input(
            f"Model not found. Do you want to create a new model {user_id}/{app_id}/models/{model_id}? (y/n): "
        )
        if y.lower() != 'y':
            raise click.Abort()
        try:
            model_type_id = ctx.obj.current.model_type_id
        except AttributeError:
            model_type_id = DEFAULT_LOCAL_RUNNER_MODEL_TYPE

        model = app.create_model(model_id, model_type_id=model_type_id)
        ctx.obj.current.CLARIFAI_MODEL_TYPE_ID = model_type_id
        ctx.obj.current.CLARIFAI_MODEL_ID = model_id
        ctx.obj.to_yaml()  # save to yaml file.

    # Now we need to create a version for the model if no version exists. Only need one version that
    # mentions it's a local runner.
    model_versions = [v for v in model.list_versions()]
    method_signatures = builder.get_method_signatures(mocking=False)
    if len(model_versions) == 0:
        logger.warning("No model versions found. Creating a new version for local runner.")
        version = model.create_version(
            pretrained_model_config={"local_dev": True}, method_signatures=method_signatures
        ).model_version
        ctx.obj.current.CLARIFAI_MODEL_VERSION_ID = version.id
        ctx.obj.to_yaml()
    else:
        model.patch_version(
            version_id=model_versions[0].model_version.id,
            pretrained_model_config={"local_dev": True},
            method_signatures=method_signatures,
        )
        version = model_versions[0].model_version
        ctx.obj.current.CLARIFAI_MODEL_VERSION_ID = version.id
        ctx.obj.to_yaml()  # save to yaml file.

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
    config_file = os.path.join(model_path, 'config.yaml')
    if not os.path.exists(config_file):
        logger.error(
            f"config.yaml not found in {model_path}. Please ensure you are passing the correct directory."
        )
        raise click.Abort()
    config = ModelBuilder._load_config(config_file)
    model_type_id = config.get('model', {}).get('model_type_id', DEFAULT_LOCAL_RUNNER_MODEL_TYPE)
    # The config.yaml doens't match what we created above.
    if 'model' in config and model_id != config['model'].get('id'):
        logger.info(f"Current model section of config.yaml: {config.get('model', {})}")
        y = input(
            "Do you want to backup config.yaml to config.yaml.bk then update the config.yaml with the new model information? (y/n): "
        )
        if y.lower() != 'y':
            raise click.Abort()
        config = ModelBuilder._set_local_runner_model(
            config, user_id, app_id, model_id, model_type_id
        )
        ModelBuilder._backup_config(config_file)
        ModelBuilder._save_config(config_file, config)

    if not check_requirements_installed(model_path):
        logger.error(f"Requirements not installed for model at {model_path}.")
        raise click.Abort()

    # Post check while running `clarifai model local-runner` we check if the toolkit is ollama
    if builder.config.get('toolkit', {}).get('provider') == 'ollama':
        if not check_ollama_installed():
            logger.error(
                "Ollama is not installed. Please install it from `https://ollama.com/` to use the Ollama toolkit."
            )
            raise click.Abort()

        try:
            logger.info("Customizing Ollama model with provided parameters...")
            customize_ollama_model(
                model_path=model_path,
                verbose=True if verbose else False,
            )
        except Exception as e:
            logger.error(f"Failed to customize Ollama model: {e}")
            raise click.Abort()

    logger.info("✅ Starting local runner...")

    # This reads the config.yaml from the model_path so we alter it above first.
    server = ModelServer(model_path)
    server.serve(
        pool_size=pool_size,
        num_threads=pool_size,
        user_id=user_id,
        compute_cluster_id=compute_cluster_id,
        nodepool_id=nodepool_id,
        runner_id=runner_id,
        base_url=ctx.obj.current.api_base,
        pat=ctx.obj.current.pat,
        context=ctx.obj.current,
    )


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
@click.option('--file_path', required=False, help='File path of file for the model to predict')
@click.option('--url', required=False, help='URL to the file for the model to predict')
@click.option('--bytes', required=False, help='Bytes to the file for the model to predict')
@click.option('--input_type', required=False, help='Type of input')
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
    '--inference_params', required=False, default='{}', help='Inference parameters to override'
)
@click.option('--output_config', required=False, default='{}', help='Output config to override')
@click.pass_context
def predict(
    ctx,
    config,
    model_id,
    user_id,
    app_id,
    model_url,
    file_path,
    url,
    bytes,
    input_type,
    compute_cluster_id,
    nodepool_id,
    deployment_id,
    inference_params,
    output_config,
):
    """Predict using the given model"""
    import json

    from clarifai.client.model import Model
    from clarifai.utils.cli import from_yaml, validate_context

    validate_context(ctx)
    if config:
        config = from_yaml(config)
        (
            model_id,
            user_id,
            app_id,
            model_url,
            file_path,
            url,
            bytes,
            input_type,
            compute_cluster_id,
            nodepool_id,
            deployment_id,
            inference_params,
            output_config,
        ) = (
            config.get(k, v)
            for k, v in [
                ('model_id', model_id),
                ('user_id', user_id),
                ('app_id', app_id),
                ('model_url', model_url),
                ('file_path', file_path),
                ('url', url),
                ('bytes', bytes),
                ('input_type', input_type),
                ('compute_cluster_id', compute_cluster_id),
                ('nodepool_id', nodepool_id),
                ('deployment_id', deployment_id),
                ('inference_params', inference_params),
                ('output_config', output_config),
            ]
        )
    if (
        sum(
            [
                opt[1]
                for opt in [(model_id, 1), (user_id, 1), (app_id, 1), (model_url, 3)]
                if opt[0]
            ]
        )
        != 3
    ):
        raise ValueError(
            "Either --model_id & --user_id & --app_id or --model_url must be provided."
        )
    if compute_cluster_id or nodepool_id or deployment_id:
        if (
            sum(
                [
                    opt[1]
                    for opt in [(compute_cluster_id, 0.5), (nodepool_id, 0.5), (deployment_id, 1)]
                    if opt[0]
                ]
            )
            != 1
        ):
            raise ValueError(
                "Either --compute_cluster_id & --nodepool_id or --deployment_id must be provided."
            )
    if model_url:
        model = Model(
            url=model_url,
            pat=ctx.obj['pat'],
            base_url=ctx.obj['base_url'],
            compute_cluster_id=compute_cluster_id,
            nodepool_id=nodepool_id,
            deployment_id=deployment_id,
        )
    else:
        model = Model(
            model_id=model_id,
            user_id=user_id,
            app_id=app_id,
            pat=ctx.obj['pat'],
            base_url=ctx.obj['base_url'],
            compute_cluster_id=compute_cluster_id,
            nodepool_id=nodepool_id,
            deployment_id=deployment_id,
        )

    if inference_params:
        inference_params = json.loads(inference_params)
    if output_config:
        output_config = json.loads(output_config)

    if file_path:
        model_prediction = model.predict_by_filepath(
            filepath=file_path,
            input_type=input_type,
            inference_params=inference_params,
            output_config=output_config,
        )
    elif url:
        model_prediction = model.predict_by_url(
            url=url,
            input_type=input_type,
            inference_params=inference_params,
            output_config=output_config,
        )
    elif bytes:
        bytes = str.encode(bytes)
        model_prediction = model.predict_by_bytes(
            input_bytes=bytes,
            input_type=input_type,
            inference_params=inference_params,
            output_config=output_config,
        )  ## TO DO: Add support for input_id
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
        pat = ctx.obj.contexts["default"]["env"]["CLARIFAI_PAT"]
    except Exception as e:
        pat = None

    User(pat=pat).list_models(
        user_id=user_id, app_id=app_id, show=True, return_clarifai_model=False
    )
