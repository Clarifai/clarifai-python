import os
import shutil

import click

from clarifai.cli.base import cli
from clarifai.utils.cli import validate_context
from clarifai.utils.constants import (
    DEFAULT_LOCAL_DEV_APP_ID,
    DEFAULT_LOCAL_DEV_COMPUTE_CLUSTER_CONFIG,
    DEFAULT_LOCAL_DEV_COMPUTE_CLUSTER_ID,
    DEFAULT_LOCAL_DEV_DEPLOYMENT_ID,
    DEFAULT_LOCAL_DEV_MODEL_ID,
    DEFAULT_LOCAL_DEV_MODEL_TYPE,
    DEFAULT_LOCAL_DEV_NODEPOOL_CONFIG,
    DEFAULT_LOCAL_DEV_NODEPOOL_ID,
)
from clarifai.utils.logging import logger


@cli.group(
    ['model'], context_settings={'max_content_width': shutil.get_terminal_size().columns - 10}
)
def model():
    """Manage models: upload, test, local dev, predict, etc"""


@model.command()
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
    help='Flag to skip generating a dockerfile so that you can manually edit an already created dockerfile.',
)
def upload(model_path, stage, skip_dockerfile):
    """Upload a model to Clarifai.

    MODEL_PATH: Path to the model directory. If not specified, the current directory is used by default.
    """
    from clarifai.runners.models.model_builder import upload_model

    upload_model(model_path, stage, skip_dockerfile)


@model.command()
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


@model.command()
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


@model.command()
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
    help='Flag to skip generating a dockerfile so that you can manually edit an already created dockerfile. Apply for `--mode conatainer`.',
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


@model.command()
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
    help='Flag to skip generating a dockerfile so that you can manually edit an already created dockerfile. Apply for `--mode conatainer`.',
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


@model.command()
@click.argument(
    "model_path",
    type=click.Path(exists=True),
    required=False,
    default=".",
)
@click.pass_context
def local_dev(ctx, model_path):
    """Run the model as a local dev runner to help debug your model connected to the API or to
    leverage local compute resources manually. This relies on many variables being present in the env
    of the currently selected context. If they are not present then default values will be used to
    ease the setup of a local dev runner and your context yaml will be updated in place. The required
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
      # for where the local dev runner should be in a compute cluser
      # note the user_id of the compute cluster is the same as the user_id of the model.

    \b
      CLARIFAI_COMPUTE_CLUSTER_ID:
      CLARIFAI_NODEPOOL_ID:

      # The following will be created in your context since it's generated by the API

      CLARIFAI_RUNNER_ID:


    Additionally using the provided model path, if the config.yaml file does not contain the model
    information that matches the above CLARIFAI_USER_ID, CLARIFAI_APP_ID, CLARIFAI_MODEL_ID then the
    config.yaml will be updated to include the model information. This is to ensure that the model
    that starts up in the local dev runner is the same as the one you intend to call in the API.

    MODEL_PATH: Path to the model directory. If not specified, the current directory is used by default.
    """
    from clarifai.client.user import User
    from clarifai.runners.models.model_builder import ModelBuilder
    from clarifai.runners.server import serve

    validate_context(ctx)
    logger.info("Checking setup for local development runner...")
    logger.info(f"Current context: {ctx.obj.current.name}")
    user_id = ctx.obj.current.user_id
    user = User(user_id=user_id, pat=ctx.obj.current.pat, base_url=ctx.obj.current.api_base)
    logger.info(f"Current user_id: {user_id}")
    logger.debug("Checking if a local dev compute cluster exists...")

    # see if ctx has CLARIFAI_COMPUTE_CLUSTER_ID, if not use default
    try:
        compute_cluster_id = ctx.obj.current.compute_cluster_id
    except AttributeError:
        compute_cluster_id = DEFAULT_LOCAL_DEV_COMPUTE_CLUSTER_ID
    logger.info(f"Current compute_cluster_id: {compute_cluster_id}")

    try:
        compute_cluster = user.compute_cluster(compute_cluster_id)
        if compute_cluster.cluster_type != 'local-dev':
            raise ValueError(
                f"Compute cluster {user_id}/{compute_cluster_id} is not a local-dev compute cluster. Please create a local-dev compute cluster."
            )
        try:
            compute_cluster_id = ctx.obj.current.compute_cluster_id
        except AttributeError:  # doesn't exist in context but does in API then update the context.
            ctx.obj.current.CLARIFAI_COMPUTE_CLUSTER_ID = compute_cluster.id
            ctx.obj.to_yaml()  # save to yaml file.
    except ValueError:
        raise
    except Exception as e:
        logger.info(f"Failed to get compute cluster with ID {compute_cluster_id}: {e}")
        y = input(
            f"Compute cluster not found. Do you want to create a new compute cluster {user_id}/{compute_cluster_id}? (y/n): "
        )
        if y.lower() != 'y':
            raise click.Abort()
        # Create a compute cluster with default configuration for local dev.
        compute_cluster = user.create_compute_cluster(
            compute_cluster_id=compute_cluster_id,
            compute_cluster_config=DEFAULT_LOCAL_DEV_COMPUTE_CLUSTER_CONFIG,
        )
        ctx.obj.current.CLARIFAI_COMPUTE_CLUSTER_ID = compute_cluster_id
        ctx.obj.to_yaml()  # save to yaml file.

    # Now check if there is a nodepool created in this compute cluser
    try:
        nodepool_id = ctx.obj.current.nodepool_id
    except AttributeError:
        nodepool_id = DEFAULT_LOCAL_DEV_NODEPOOL_ID
    logger.info(f"Current nodepool_id: {nodepool_id}")

    try:
        nodepool = compute_cluster.nodepool(nodepool_id)
        try:
            nodepool_id = ctx.obj.current.nodepool_id
        except AttributeError:  # doesn't exist in context but does in API then update the context.
            ctx.obj.current.CLARIFAI_NODEPOOL_ID = nodepool.id
            ctx.obj.to_yaml()  # save to yaml file.
    except Exception as e:
        logger.info(f"Failed to get nodepool with ID {nodepool_id}: {e}")
        y = input(
            f"Nodepool not found. Do you want to create a new nodepool {user_id}/{compute_cluster_id}/{nodepool_id}? (y/n): "
        )
        if y.lower() != 'y':
            raise click.Abort()
        nodepool = compute_cluster.create_nodepool(
            nodepool_config=DEFAULT_LOCAL_DEV_NODEPOOL_CONFIG, nodepool_id=nodepool_id
        )
        ctx.obj.current.CLARIFAI_NODEPOOL_ID = nodepool_id
        ctx.obj.to_yaml()  # save to yaml file.

    logger.debug("Checking if model is created to call for local development...")
    # see if ctx has CLARIFAI_APP_ID, if not use default
    try:
        app_id = ctx.obj.current.app_id
    except AttributeError:
        app_id = DEFAULT_LOCAL_DEV_APP_ID
    logger.info(f"Current app_id: {app_id}")

    try:
        app = user.app(app_id)
        try:
            app_id = ctx.obj.current.app_id
        except AttributeError:  # doesn't exist in context but does in API then update the context.
            ctx.obj.current.CLARIFAI_APP_ID = app.id
            ctx.obj.to_yaml()  # save to yaml file.
    except Exception as e:
        logger.info(f"Failed to get app with ID {app_id}: {e}")
        y = input(f"App not found. Do you want to create a new app {user_id}/{app_id}? (y/n): ")
        if y.lower() != 'y':
            raise click.Abort()
        app = user.create_app(app_id)
        ctx.obj.current.CLARIFAI_APP_ID = app_id
        ctx.obj.to_yaml()  # save to yaml file.

    # Within this app we now need a model to call as the local dev runner.
    try:
        model_id = ctx.obj.current.model_id
    except AttributeError:
        model_id = DEFAULT_LOCAL_DEV_MODEL_ID
    logger.info(f"Current model_id: {model_id}")

    try:
        model = app.model(model_id)
        try:
            model_id = ctx.obj.current.model_id
        except AttributeError:  # doesn't exist in context but does in API then update the context.
            ctx.obj.current.CLARIFAI_MODEL_ID = model.id
            ctx.obj.to_yaml()  # save to yaml file.
    except Exception as e:
        logger.info(f"Failed to get model with ID {model_id}: {e}")
        y = input(
            f"Model not found. Do you want to create a new model {user_id}/{app_id}/models/{model_id}? (y/n): "
        )
        if y.lower() != 'y':
            raise click.Abort()
        model = app.create_model(model_id, model_type_id=DEFAULT_LOCAL_DEV_MODEL_TYPE)
        ctx.obj.current.CLARIFAI_MODEL_ID = model_id
        ctx.obj.to_yaml()  # save to yaml file.

    # Now we need to create a version for the model if no version exists. Only need one version that
    # mentions it's a local dev runner.
    model_versions = [v for v in model.list_versions()]
    if len(model_versions) == 0:
        logger.info("No model versions found. Creating a new version for local dev runner.")
        version = model.create_version(pretrained_model_config={"local_dev": True}).model_version
        logger.info(f"Created model version {version.id}")
    else:
        version = model_versions[0].model_version

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
        except Exception as e:
            raise AttributeError("Runner not found in nodepool.") from e
    except AttributeError:
        logger.info(
            f"Create the local dev runner tying this\n  {user_id}/{app_id}/models/{model.id} model (version: {version.id}) to the\n  {user_id}/{compute_cluster_id}/{nodepool_id} nodepool."
        )
        runner = nodepool.create_runner(
            runner_config={
                "runner": {
                    "description": "Local dev runner for model testing",
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
        deployment_id = DEFAULT_LOCAL_DEV_DEPLOYMENT_ID
    try:
        deployment = nodepool.deployment(deployment_id)
        try:
            deployment_id = ctx.obj.current.deployment_id
        except AttributeError:  # doesn't exist in context but does in API then update the context.
            ctx.obj.current.CLARIFAI_DEPLOYMENT_ID = deployment.id
            ctx.obj.to_yaml()  # save to yaml file.
    except Exception as e:
        logger.info(f"Failed to get deployment with ID {deployment_id}: {e}")
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
        raise ValueError(
            f"config.yaml not found in {model_path}. Please ensure you are passing the correct directory."
        )
    config = ModelBuilder._load_config(config_file)
    # The config.yaml doens't match what we created above.
    if 'model' in config and model_id != config['model'].get('id'):
        logger.info(f"Current model section of config.yaml: {config.get('model', {})}")
        y = input(
            "Do you want to backup config.yaml to config.yaml.bk then update the config.yaml with the new model information? (y/n): "
        )
        if y.lower() != 'y':
            raise click.Abort()
        config = ModelBuilder._set_local_dev_model(
            config, user_id, app_id, model_id, DEFAULT_LOCAL_DEV_MODEL_TYPE
        )
        ModelBuilder._backup_config(config_file)
        ModelBuilder._save_config(config_file, config)

    # client_model = Model(
    # TODO: once we can generate_client_script from ModelBuilder or similar
    # we should be able to put the exact function call in place.
    # model_script = model.generate_client_script()

    builder = ModelBuilder(model_path, download_validation_only=True)
    method_signatures = builder.get_method_signatures()

    from clarifai.runners.utils import code_script

    snippet = code_script.generate_client_script(
        method_signatures,
        user_id=user_id,
        app_id=app_id,
        model_id=model_id,
        deployment_id=deployment_id,
        use_ctx=True,
        base_url=ctx.obj.current.api_base,
    )

    # TODO: put in the ClarifaiUrlHelper to create the model url.

    logger.info("""\n
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# About to start up the local dev runner in this terminal...
# Here is a code snippet to call this model once it start from another terminal:
""")
    logger.info(snippet)

    logger.info("Now starting the local dev runner...")

    # This reads the config.yaml from the model_path so we alter it above first.
    serve(
        model_path,
        user_id=user_id,
        compute_cluster_id=compute_cluster_id,
        nodepool_id=nodepool_id,
        runner_id=runner_id,
        base_url=ctx.obj.current.api_base,
        pat=ctx.obj.current.pat,
    )


@model.command()
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
        model = Model(url=model_url, pat=ctx.obj['pat'], base_url=ctx.obj['base_url'])
    else:
        model = Model(
            model_id=model_id,
            user_id=user_id,
            app_id=app_id,
            pat=ctx.obj['pat'],
            base_url=ctx.obj['base_url'],
        )

    if inference_params:
        inference_params = json.loads(inference_params)
    if output_config:
        output_config = json.loads(output_config)

    if file_path:
        model_prediction = model.predict_by_filepath(
            filepath=file_path,
            input_type=input_type,
            compute_cluster_id=compute_cluster_id,
            nodepool_id=nodepool_id,
            deployment_id=deployment_id,
            inference_params=inference_params,
            output_config=output_config,
        )
    elif url:
        model_prediction = model.predict_by_url(
            url=url,
            input_type=input_type,
            compute_cluster_id=compute_cluster_id,
            nodepool_id=nodepool_id,
            deployment_id=deployment_id,
            inference_params=inference_params,
            output_config=output_config,
        )
    elif bytes:
        bytes = str.encode(bytes)
        model_prediction = model.predict_by_bytes(
            input_bytes=bytes,
            input_type=input_type,
            compute_cluster_id=compute_cluster_id,
            nodepool_id=nodepool_id,
            deployment_id=deployment_id,
            inference_params=inference_params,
            output_config=output_config,
        )  ## TO DO: Add support for input_id
    click.echo(model_prediction)
