import os
import shutil

import click

from clarifai.cli.base import cli
from clarifai.utils.cli import (
    AliasedGroup,
    convert_timestamp_to_string,
    display_co_resources,
    resolve_id,
    validate_context,
)
from clarifai.utils.logging import logger

_DEFAULT_PIPELINE_ID = "hello-world-pipeline"


def _ensure_pipeline_compute(
    ctx, user_id, instance, cloud, region, compute_cluster_id, nodepool_id
):
    """Resolve instance type and auto-create compute cluster/nodepool if needed.

    Uses the same deterministic ID and get-or-create pattern as model deploy.

    Returns:
        tuple: (compute_cluster_id, nodepool_id)
    """
    from clarifai.utils.compute_presets import (
        get_compute_cluster_config,
        get_deploy_compute_cluster_id,
        get_deploy_nodepool_id,
        get_nodepool_config,
        resolve_gpu,
    )

    gpu_preset = resolve_gpu(instance, pat=ctx.obj.current.pat, base_url=ctx.obj.current.api_base)
    if not gpu_preset:
        raise ValueError(
            f"Unknown instance type '{instance}'. Use 'clarifai list-instances' to see available options."
        )

    cloud = cloud or gpu_preset.get('cloud_provider', 'aws')
    region = region or gpu_preset.get('region', 'us-east-1')
    instance_type_id = gpu_preset['instance_type_id']

    cc_id = compute_cluster_id or get_deploy_compute_cluster_id(cloud, region)
    np_id = nodepool_id or get_deploy_nodepool_id(instance_type_id)

    from clarifai.client.user import User

    user = User(user_id=user_id, pat=ctx.obj.current.pat, base_url=ctx.obj.current.api_base)

    # Get-or-create compute cluster
    try:
        user.compute_cluster(cc_id)
    except Exception:
        logger.info(f"Creating compute cluster '{cc_id}'...")
        cc_config = get_compute_cluster_config(user_id, cloud, region)
        user.create_compute_cluster(compute_cluster_config=cc_config)

    # Get-or-create nodepool
    from clarifai.client.compute_cluster import ComputeCluster

    cc = ComputeCluster(
        compute_cluster_id=cc_id,
        user_id=user_id,
        pat=ctx.obj.current.pat,
        base_url=ctx.obj.current.api_base,
    )
    try:
        cc.nodepool(np_id)
    except Exception:
        logger.info(f"Creating nodepool '{np_id}'...")
        np_config = get_nodepool_config(
            instance_type_id=instance_type_id,
            compute_cluster_id=cc_id,
            user_id=user_id,
            compute_info=gpu_preset.get("inference_compute_info"),
        )
        cc.create_nodepool(nodepool_config=np_config)

    return cc_id, np_id


@cli.group(
    ['pipeline', 'pl'],
    cls=AliasedGroup,
    context_settings={'max_content_width': shutil.get_terminal_size().columns - 10},
)
def pipeline():
    """Create and manage pipelines."""


@pipeline.command()
@click.argument("path", type=click.Path(exists=True), required=False, default=".")
@click.option(
    '--no-lockfile',
    is_flag=True,
    help='Skip creating config-lock.yaml file.',
)
def upload(path, no_lockfile):
    """Upload a pipeline with associated pipeline steps to Clarifai.

    PATH: Path to the pipeline configuration file or directory containing config.yaml. If not specified, the current directory is used by default.
    """
    from clarifai.runners.pipelines.pipeline_builder import upload_pipeline

    upload_pipeline(path, no_lockfile=no_lockfile)


@pipeline.command()
@click.option(
    '--config',
    type=click.Path(exists=True),
    required=False,
    help='Path to the pipeline run config file.',
)
@click.option('--pipeline_id', required=False, help='Pipeline ID to run.')
@click.option('--pipeline_version_id', required=False, help='Pipeline Version ID to run.')
@click.option(
    '--pipeline_version_run_id',
    required=False,
    help='Pipeline Version Run ID. If not provided, a UUID will be generated.',
)
@click.option('--user_id', required=False, help='User ID of the pipeline.')
@click.option('--app_id', required=False, help='App ID that contains the pipeline.')
@click.option(
    '--nodepool_id',
    required=False,
    help='[Advanced] Existing nodepool ID (skip auto-creation).',
)
@click.option(
    '--compute_cluster_id',
    required=False,
    help='[Advanced] Existing compute cluster ID (skip auto-creation).',
)
@click.option(
    '--instance',
    default=None,
    help='Hardware instance type (e.g., g5.xlarge, A10G). Auto-creates compute cluster and nodepool.',
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
@click.option('--pipeline_url', required=False, help='Pipeline URL to run.')
@click.option(
    '--timeout',
    type=int,
    default=3600,
    help='Maximum time to wait for completion in seconds. Default 3600 (1 hour).',
)
@click.option(
    '--monitor_interval',
    type=int,
    default=10,
    help='Interval between status checks in seconds. Default 10.',
)
@click.option(
    '--log_file',
    type=click.Path(),
    required=False,
    help='Path to file where logs should be written. If not provided, logs are displayed on console.',
)
@click.option(
    '--monitor',
    is_flag=True,
    default=False,
    help='Monitor an existing pipeline run instead of starting a new one. Requires pipeline_version_run_id.',
)
@click.option(
    '--set',
    'override_params',
    multiple=True,
    help='Override parameter values inline. Format: --set key=value. Can be used multiple times.',
)
@click.option(
    '--overrides-file',
    type=click.Path(exists=True),
    help='Path to JSON/YAML file containing parameter overrides.',
)
@click.pass_context
def run(
    ctx,
    config,
    pipeline_id,
    pipeline_version_id,
    pipeline_version_run_id,
    user_id,
    app_id,
    nodepool_id,
    compute_cluster_id,
    instance,
    cloud,
    region,
    pipeline_url,
    timeout,
    monitor_interval,
    log_file,
    monitor,
    override_params,
    overrides_file,
):
    """Run a pipeline and monitor its progress."""
    import json

    from clarifai.client.pipeline import Pipeline
    from clarifai.utils.cli import from_yaml, validate_context

    validate_context(ctx)

    # Try to load from config-lock.yaml first if no config is specified
    lockfile_path = os.path.join(os.getcwd(), "config-lock.yaml")
    if not config and os.path.exists(lockfile_path):
        logger.info("Found config-lock.yaml, using it as default config source")
        config = lockfile_path

    if config:
        config_data = from_yaml(config)

        # Handle both regular config format and lockfile format
        if 'pipeline' in config_data and isinstance(config_data['pipeline'], dict):
            pipeline_config = config_data['pipeline']
            pipeline_id = pipeline_config.get('id', pipeline_id)
            pipeline_version_id = pipeline_config.get('version_id', pipeline_version_id)
            user_id = pipeline_config.get('user_id', user_id)
            app_id = pipeline_config.get('app_id', app_id)
        else:
            # Fallback to flat config structure
            pipeline_id = config_data.get('pipeline_id', pipeline_id)
            pipeline_version_id = config_data.get('pipeline_version_id', pipeline_version_id)
            user_id = config_data.get('user_id', user_id)
            app_id = config_data.get('app_id', app_id)

        pipeline_version_run_id = config_data.get(
            'pipeline_version_run_id', pipeline_version_run_id
        )
        nodepool_id = config_data.get('nodepool_id', nodepool_id)
        compute_cluster_id = config_data.get('compute_cluster_id', compute_cluster_id)
        # Read compute section from inside pipeline config for auto-creation support
        pipeline_sect = (
            config_data.get('pipeline', {})
            if isinstance(config_data.get('pipeline'), dict)
            else {}
        )
        compute_section = pipeline_sect.get('compute', {})
        instance = instance or compute_section.get('instance')
        cloud = cloud or compute_section.get('cloud')
        region = region or compute_section.get('region')
        pipeline_url = config_data.get('pipeline_url', pipeline_url)
        timeout = config_data.get('timeout', timeout)
        monitor_interval = config_data.get('monitor_interval', monitor_interval)
        log_file = config_data.get('log_file', log_file)
        monitor = config_data.get('monitor', monitor)
    elif ctx.obj.current:
        if not user_id:
            user_id = ctx.obj.current.get('user_id', '')
        if not app_id:
            app_id = ctx.obj.current.get('app_id', '')
        if not pipeline_id:
            pipeline_id = ctx.obj.current.get('pipeline_id', '')
        if not pipeline_version_id:
            pipeline_version_id = ctx.obj.current.get('pipeline_version_id', '')
        if not nodepool_id:
            nodepool_id = ctx.obj.current.get('nodepool_id', '')
        if not compute_cluster_id:
            compute_cluster_id = ctx.obj.current.get('compute_cluster_id', '')

    # Auto-resolve compute cluster and nodepool from --instance if not explicitly provided
    if not compute_cluster_id or not nodepool_id:
        if instance:
            compute_cluster_id, nodepool_id = _ensure_pipeline_compute(
                ctx, user_id, instance, cloud, region, compute_cluster_id, nodepool_id
            )
        else:
            raise ValueError(
                "--instance is required when --compute_cluster_id and --nodepool_id are not both provided.\n"
                "  Example: clarifai pipeline run --instance g5.xlarge\n"
                "  Or provide both: --compute_cluster_id <id> --nodepool_id <id>\n"
                "  List available instances: clarifai list-instances"
            )

    # When monitor flag is used, pipeline_version_run_id is mandatory
    if monitor and not pipeline_version_run_id:
        raise ValueError("--pipeline_version_run_id is required when using --monitor flag.")

    if pipeline_url:
        # When using pipeline_url, other parameters are optional (will be parsed from URL)
        required_params_provided = True
    else:
        # When not using pipeline_url, all individual parameters are required
        required_params_provided = all([pipeline_id, user_id, app_id, pipeline_version_id])

    if not required_params_provided:
        raise ValueError(
            "Either --user_id & --app_id & --pipeline_id & --pipeline_version_id or --pipeline_url must be provided."
        )

    if pipeline_url:
        pipeline = Pipeline(
            url=pipeline_url,
            pat=ctx.obj.current.pat,
            base_url=ctx.obj.current.api_base,
            pipeline_version_run_id=pipeline_version_run_id,
            nodepool_id=nodepool_id,
            compute_cluster_id=compute_cluster_id,
            log_file=log_file,
        )
    else:
        pipeline = Pipeline(
            pipeline_id=pipeline_id,
            pipeline_version_id=pipeline_version_id,
            pipeline_version_run_id=pipeline_version_run_id,
            user_id=user_id,
            app_id=app_id,
            nodepool_id=nodepool_id,
            compute_cluster_id=compute_cluster_id,
            pat=ctx.obj.current.pat,
            base_url=ctx.obj.current.api_base,
            log_file=log_file,
        )

    # Process input argument overrides
    input_args_override = None
    if override_params or overrides_file:
        from clarifai_grpc.grpc.api import resources_pb2

        # Start with an empty dict for all overrides
        all_overrides = {}

        # Load overrides from file if provided
        if overrides_file:
            from clarifai.utils.cli import from_yaml

            try:
                if overrides_file.endswith(('.yaml', '.yml')):
                    file_overrides = from_yaml(overrides_file)
                else:  # assume JSON
                    import json

                    with open(overrides_file, 'r') as f:
                        file_overrides = json.load(f)

                all_overrides.update(file_overrides)
            except Exception as e:
                raise ValueError(f"Failed to load overrides file {overrides_file}: {e}")

        # Process inline --set parameters (these take precedence over file)
        for param in override_params:
            if '=' not in param:
                raise ValueError(f"Invalid --set format: {param}. Expected format: key=value")
            key, value = param.split('=', 1)
            all_overrides[key] = value

        # Build the OrchestrationArgsOverride proto if we have any overrides
        if all_overrides:
            parameters = []
            for key, value in all_overrides.items():
                parameters.append(
                    resources_pb2.ArgoParameterOverride(
                        name=key,
                        value=str(value),  # Argo parameters are always strings
                    )
                )

            input_args_override = resources_pb2.OrchestrationArgsOverride(
                argo_args_override=resources_pb2.ArgoArgsOverride(parameters=parameters)
            )

    if monitor:
        # Monitor existing pipeline run instead of starting new one
        result = pipeline.monitor_only(timeout=timeout, monitor_interval=monitor_interval)
    else:
        # Start new pipeline run and monitor it
        result = pipeline.run(
            timeout=timeout,
            monitor_interval=monitor_interval,
            input_args_override=input_args_override,
        )
    click.echo(json.dumps(result, indent=2, default=str))


@pipeline.command()
@click.argument(
    "pipeline_path",
    type=click.Path(),
    required=False,
    default=".",
)
@click.option(
    '--template',
    required=False,
    help='Initialize from a template (e.g., image-classification, text-prep)',
)
@click.option('--user_id', required=False, help='User ID for the pipeline.')
@click.option('--app_id', required=False, help='App ID for the pipeline.')
@click.option(
    '--pipeline_id',
    required=False,
    default=_DEFAULT_PIPELINE_ID,
    show_default=True,
    help='Pipeline ID.',
)
@click.option(
    '--steps',
    required=False,
    multiple=True,
    help='Pipeline step names. Can be specified multiple times (e.g., --steps stepA --steps stepB). Ignored when --template is used.',
)
@click.option(
    '--num_steps',
    required=False,
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help='Number of pipeline steps to create when --steps is not specified. Ignored when --template or --steps is used.',
)
@click.option(
    '--set',
    'override_params',
    multiple=True,
    help='Template parameter overrides. Format: --set key=value. Can be used multiple times. Only used with --template.',
)
def init(pipeline_path, template, user_id, app_id, pipeline_id, steps, num_steps, override_params):
    """Initialize a new pipeline project structure.

    Creates a pipeline project structure either from a template or using flag-based inputs.

    When using --template, initializes from a predefined template with specific
    parameters and structure. Without --template, creates a custom pipeline structure
    using the provided flags.

    Creates the following structure in the specified directory:
    ├── config.yaml          # Pipeline configuration
    ├── stepA/               # First pipeline step
    │   ├── config.yaml     # Step A configuration
    │   ├── requirements.txt # Step A dependencies
    │   └── 1/
    │       └── pipeline_step.py  # Step A implementation
    ├── stepB/               # Second pipeline step
    │   ├── config.yaml     # Step B configuration
    │   ├── requirements.txt # Step B dependencies
    │   └── 1/
    │       └── pipeline_step.py  # Step B implementation
    └── README.md           # Documentation

    PIPELINE_PATH: Path where to create the pipeline project structure. If not specified, the current directory is used by default.

    Examples:

        # user_id/app_id auto-detected from global config (~/.clarifai/config.yaml)
        clarifai pipeline init

        # Initialize with explicit IDs and steps
        clarifai pipeline init --user_id=my_user --app_id=my_app --pipeline_id=my-pipeline --steps stepA --steps stepB

        # Initialize with a specific number of steps
        clarifai pipeline init --user_id=my_user --app_id=my_app --pipeline_id=my-pipeline --num_steps=3

        # Initialize from a template
        clarifai pipeline init --template=image-classification --user_id=my_user --app_id=my_app

        # Initialize from a template with custom parameters
        clarifai pipeline init --template=image-classification --user_id=my_user --app_id=my_app --set model_name=resnet50
    """
    # Resolve user_id and app_id from flag → global config → prompt
    user_id = resolve_id(user_id, 'user_id', 'User ID')
    app_id = resolve_id(app_id, 'app_id', 'App ID')

    # Common setup logic
    pipeline_path = _prepare_pipeline_path(pipeline_path, template)
    if not pipeline_path:
        return  # Error already shown in _prepare_pipeline_path

    # Resolve step names: explicit --steps take precedence, then generate from --num_steps
    if steps:
        resolved_steps = [*steps]
    else:
        default_names = ["stepA", "stepB", "stepC", "stepD", "stepE", "stepF"]
        resolved_steps = [
            default_names[i] if i < len(default_names) else f"step{i + 1}"
            for i in range(num_steps)
        ]

    # Branch to specific initialization method
    if template:
        success = _init_from_template(
            pipeline_path,
            template,
            user_id=user_id,
            app_id=app_id,
            pipeline_id=pipeline_id,
            override_params=override_params,
        )
    else:
        success = _init_flag_based(
            pipeline_path,
            user_id=user_id,
            app_id=app_id,
            pipeline_id=pipeline_id,
            step_names=resolved_steps,
        )

    # Common completion logic
    if success:
        _show_completion_message(pipeline_path)


def _prepare_pipeline_path(pipeline_path, template_name):
    """Prepare and validate the pipeline path.

    Args:
        pipeline_path: Original path argument
        template_name: Template name (if using template initialization)

    Returns:
        Absolute path to use, or None if there's an error
    """
    # If pipeline_path is current directory and using template, create new directory with template name
    if pipeline_path == "." and template_name:
        pipeline_path = template_name

    # Resolve the absolute path
    pipeline_path = os.path.abspath(pipeline_path)

    # For template initialization, check if directory exists and is not empty
    # For interactive initialization, allow existing directories (files will be skipped individually)
    if template_name and os.path.exists(pipeline_path) and os.listdir(pipeline_path):
        click.echo(
            f"Error: Directory '{pipeline_path}' already exists and is not empty.", err=True
        )
        click.echo("Please choose a different directory or remove the existing one.", err=True)
        return None

    # Create the pipeline directory
    os.makedirs(pipeline_path, exist_ok=True)
    return pipeline_path


def _show_completion_message(pipeline_path):
    """Show common completion message.

    Args:
        pipeline_path: Path where pipeline was created
    """
    logger.info(f"Pipeline initialization complete in {pipeline_path}")
    logger.info("Next steps:")
    logger.info("1. Review and customize the generated pipeline steps")
    logger.info("2. Add any additional dependencies to requirements.txt files")
    logger.info("3. Run 'clarifai pipeline upload config.yaml' to upload your pipeline")


def _init_from_template(
    pipeline_path, template_name, user_id=None, app_id=None, pipeline_id=None, override_params=None
):
    """Initialize pipeline from a template.

    Args:
        pipeline_path: Destination path for the pipeline (already prepared)
        template_name: Name of the template to use
        user_id: User ID for the pipeline (optional, uses placeholder if not provided)
        app_id: App ID for the pipeline (optional, uses placeholder if not provided)
        pipeline_id: Pipeline ID (optional, defaults to template_name)
        override_params: Iterable of "key=value" strings for template parameter overrides

    Returns:
        bool: True if successful, False otherwise
    """
    from clarifai.utils.template_manager import TemplateManager

    click.echo(f"Initializing pipeline from template: {template_name}")
    click.echo()

    try:
        # Initialize template manager and get template info
        template_manager = TemplateManager()
        template_info = template_manager.get_template_info(template_name)

        if not template_info:
            click.echo(f"Error: Template '{template_name}' not found", err=True)
            click.echo("Use 'clarifai pipelinetemplate ls' to see available templates", err=True)
            return False

        # Show template information
        click.echo(f"Template Type: {template_info['type']}")
        click.echo(f"Steps: {', '.join(template_info['step_directories'])}")

        parameters = template_info['parameters']
        if parameters:
            click.echo(f"Parameters: {len(parameters)} available")
        click.echo()

        # user_id and app_id already resolved by the init command caller
        effective_user_id = user_id or "your_user_id"
        effective_app_id = app_id or "your_app_id"
        effective_pipeline_id = (
            pipeline_id if pipeline_id and pipeline_id != _DEFAULT_PIPELINE_ID else template_name
        )

        # Build parameter substitutions from flags
        parameter_substitutions = {}

        # Parse --set overrides
        if override_params:
            for param in override_params:
                if '=' not in param:
                    raise ValueError(f"Invalid --set format: '{param}'. Expected key=value.")
                key, value = param.split('=', 1)
                parameter_substitutions[key] = value

        # Warn about template parameters that were not overridden
        if parameters:
            overridden_keys = set(parameter_substitutions.keys())
            for param in parameters:
                param_name = param['name']
                if param_name not in overridden_keys:
                    default_value = param['default_value']
                    logger.info(
                        f"Using default value for template parameter '{param_name}': {default_value}"
                    )

        # Add basic info to parameter substitutions
        parameter_substitutions['user_id'] = effective_user_id
        parameter_substitutions['app_id'] = effective_app_id
        parameter_substitutions['id'] = effective_pipeline_id

        click.echo(
            f"Creating pipeline '{effective_pipeline_id}' from template '{template_name}'..."
        )

        # Copy template with substitutions
        success = template_manager.copy_template(
            template_name, pipeline_path, parameter_substitutions
        )

        if not success:
            click.echo("Error: Failed to create pipeline from template", err=True)

        return success

    except Exception as e:
        logger.error(f"Template initialization error: {e}")
        click.echo(f"Error: {e}", err=True)
        return False


def _init_flag_based(
    pipeline_path, user_id=None, app_id=None, pipeline_id=_DEFAULT_PIPELINE_ID, step_names=None
):
    """Flag-based pipeline initialization.

    Args:
        pipeline_path: Destination path for the pipeline (already prepared)
        user_id: User ID for the pipeline (optional, uses placeholder if not provided)
        app_id: App ID for the pipeline (optional, uses placeholder if not provided)
        pipeline_id: Pipeline ID (default: 'hello-world-pipeline')
        step_names: List of pipeline step names (default: ['stepA', 'stepB'])

    Returns:
        bool: True if successful, False otherwise
    """
    from clarifai.cli.templates.pipeline_templates import (
        get_pipeline_config_template,
        get_pipeline_step_config_template,
        get_pipeline_step_requirements_template,
        get_pipeline_step_template,
        get_readme_template,
    )

    if step_names is None:
        step_names = ["stepA", "stepB"]

    # user_id and app_id already resolved by the init command caller
    effective_user_id = user_id or "your_user_id"
    effective_app_id = app_id or "your_app_id"

    try:
        click.echo(f"Creating pipeline '{pipeline_id}' with steps: {', '.join(step_names)}")

        # Create pipeline config.yaml
        config_path = os.path.join(pipeline_path, "config.yaml")
        if os.path.exists(config_path):
            logger.warning(f"File {config_path} already exists, skipping...")
        else:
            config_template = get_pipeline_config_template(
                pipeline_id=pipeline_id,
                user_id=effective_user_id,
                app_id=effective_app_id,
                step_names=step_names,
            )
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_template)
            logger.info(f"Created {config_path}")

        # Create README.md
        readme_path = os.path.join(pipeline_path, "README.md")
        if os.path.exists(readme_path):
            logger.warning(f"File {readme_path} already exists, skipping...")
        else:
            readme_template = get_readme_template()
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_template)
            logger.info(f"Created {readme_path}")

        # Create pipeline steps
        for step_id in step_names:
            step_dir = os.path.join(pipeline_path, step_id)
            os.makedirs(step_dir, exist_ok=True)

            # Create the 1/ subdirectory for the step version
            step_version_dir = os.path.join(step_dir, "1")
            os.makedirs(step_version_dir, exist_ok=True)

            # Create step config.yaml
            step_config_path = os.path.join(step_dir, "config.yaml")
            if os.path.exists(step_config_path):
                logger.warning(f"File {step_config_path} already exists, skipping...")
            else:
                step_config_template = get_pipeline_step_config_template(
                    step_id=step_id, user_id=effective_user_id, app_id=effective_app_id
                )
                with open(step_config_path, 'w', encoding='utf-8') as f:
                    f.write(step_config_template)
                logger.info(f"Created {step_config_path}")

            # Create step requirements.txt
            step_requirements_path = os.path.join(step_dir, "requirements.txt")
            if os.path.exists(step_requirements_path):
                logger.warning(f"File {step_requirements_path} already exists, skipping...")
            else:
                step_requirements_template = get_pipeline_step_requirements_template()
                with open(step_requirements_path, 'w', encoding='utf-8') as f:
                    f.write(step_requirements_template)
                logger.info(f"Created {step_requirements_path}")

            # Create step pipeline_step.py
            step_py_path = os.path.join(step_version_dir, "pipeline_step.py")
            if os.path.exists(step_py_path):
                logger.warning(f"File {step_py_path} already exists, skipping...")
            else:
                step_py_template = get_pipeline_step_template(step_id)
                with open(step_py_path, 'w', encoding='utf-8') as f:
                    f.write(step_py_template)
                logger.info(f"Created {step_py_path}")

        return True

    except Exception as e:
        logger.error(f"Pipeline initialization error: {e}")
        click.echo(f"Error: {e}", err=True)
        return False


@pipeline.command()
@click.argument(
    "lockfile_path", type=click.Path(exists=True), required=False, default="config-lock.yaml"
)
def validate_lock(lockfile_path):
    """Validate a config-lock.yaml file for schema and reference consistency.

    LOCKFILE_PATH: Path to the config-lock.yaml file. If not specified, looks for config-lock.yaml in current directory.
    """
    from clarifai.runners.utils.pipeline_validation import PipelineConfigValidator
    from clarifai.utils.cli import from_yaml

    try:
        # Load the lockfile
        lockfile_data = from_yaml(lockfile_path)

        # Validate required fields
        if "pipeline" not in lockfile_data:
            raise ValueError("'pipeline' section not found in lockfile")

        pipeline = lockfile_data["pipeline"]
        required_fields = ["id", "user_id", "app_id", "version_id"]

        for field in required_fields:
            if field not in pipeline:
                raise ValueError(f"Required field '{field}' not found in pipeline section")
            if not pipeline[field]:
                raise ValueError(f"Required field '{field}' cannot be empty")

        # Validate orchestration spec if present
        if "orchestration_spec" in pipeline:
            # Create a temporary config structure for validation
            temp_config = {
                "pipeline": {
                    "id": pipeline["id"],
                    "user_id": pipeline["user_id"],
                    "app_id": pipeline["app_id"],
                    "orchestration_spec": pipeline["orchestration_spec"],
                }
            }

            # Use existing validator to check orchestration spec
            validator = PipelineConfigValidator()
            validator._validate_orchestration_spec(temp_config)

        logger.info(f"✅ Lockfile {lockfile_path} is valid")
        logger.info(f"Pipeline: {pipeline['id']}")
        logger.info(f"User: {pipeline['user_id']}")
        logger.info(f"App: {pipeline['app_id']}")
        logger.info(f"Version: {pipeline['version_id']}")

    except Exception as e:
        logger.error(f"❌ Lockfile validation failed: {e}")
        raise click.Abort()


@pipeline.command(['ls'])
@click.option('--page_no', required=False, help='Page number to list.', default=1)
@click.option('--per_page', required=False, help='Number of items per page.', default=16)
@click.option(
    '--app_id',
    required=True,
    help='App ID to list pipelines from.',
)
@click.option(
    '--user_id',
    required=False,
    help='User ID to list pipelines from. If not provided, uses current user.',
)
@click.pass_context
def list(ctx, page_no, per_page, app_id, user_id):
    """List all pipelines for the user."""
    validate_context(ctx)

    target_user_id = user_id or ctx.obj.current.user_id

    from clarifai.client.app import App

    app = App(
        app_id=app_id,
        user_id=target_user_id,
        pat=ctx.obj.current.pat,
        base_url=ctx.obj.current.api_base,
    )
    response = app.list_pipelines(page_no=page_no, per_page=per_page)

    display_co_resources(
        response,
        custom_columns={
            'ID': lambda p: getattr(p, 'pipeline_id', ''),
            'USER_ID': lambda p: getattr(p, 'user_id', ''),
            'APP_ID': lambda p: getattr(p, 'app_id', ''),
            'VERSION_ID': lambda p: getattr(p, 'pipeline_version_id', ''),
            'VISIBILITY': lambda p: getattr(p, 'visibility', ''),
            'DESCRIPTION': lambda p: getattr(p, 'description', ''),
            'CREATED_AT': lambda ps: convert_timestamp_to_string(getattr(ps, 'created_at', '')),
            'MODIFIED_AT': lambda ps: convert_timestamp_to_string(getattr(ps, 'modified_at', '')),
        },
        sort_by_columns=[
            ('CREATED_AT', 'desc'),
            ('ID', 'asc'),
        ],
    )
