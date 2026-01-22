import os
import shutil

import click

from clarifai.cli.base import cli
from clarifai.utils.cli import (
    AliasedGroup,
    convert_timestamp_to_string,
    display_co_resources,
    validate_context,
)
from clarifai.utils.logging import logger


@cli.group(
    ['pipeline', 'pl'],
    cls=AliasedGroup,
    context_settings={'max_content_width': shutil.get_terminal_size().columns - 10},
)
def pipeline():
    """Manage pipelines: upload, init, list, etc"""


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
@click.option('--nodepool_id', required=False, help='Nodepool ID to run the pipeline on.')
@click.option(
    '--compute_cluster_id', required=False, help='Compute Cluster ID to run the pipeline on.'
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

    # compute_cluster_id and nodepool_id are mandatory regardless of whether pipeline_url is provided
    if not compute_cluster_id or not nodepool_id:
        raise ValueError("--compute_cluster_id and --nodepool_id are mandatory parameters.")

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
def init(pipeline_path, template):
    """Initialize a new pipeline project structure.

    Creates a pipeline project structure either from a template or interactively.

    When using --template, initializes from a predefined template with specific
    parameters and structure. Without --template, uses the interactive flow
    to create a custom pipeline structure.

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
    """
    # Common setup logic
    pipeline_path = _prepare_pipeline_path(pipeline_path, template)
    if not pipeline_path:
        return  # Error already shown in _prepare_pipeline_path

    # Branch to specific initialization method
    if template:
        success = _init_from_template(pipeline_path, template)
    else:
        success = _init_interactive(pipeline_path)

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


def _init_from_template(pipeline_path, template_name):
    """Initialize pipeline from a template.

    Args:
        pipeline_path: Destination path for the pipeline (already prepared)
        template_name: Name of the template to use

    Returns:
        bool: True if successful, False otherwise
    """
    from clarifai.utils.template_manager import TemplateManager

    click.echo("Welcome to Clarifai Pipeline Template Initialization!")
    click.echo(f"Using template: {template_name}")
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
            click.echo(f"Parameters: {len(parameters)} required")
        click.echo()

        # Collect basic pipeline information
        click.echo("Please provide the following information:")
        user_id = click.prompt("User ID", type=str)
        app_id = click.prompt("App ID", type=str)

        # Use template name as default pipeline ID
        default_pipeline_id = template_name
        pipeline_id = click.prompt("Pipeline ID", default=default_pipeline_id, type=str)

        # Collect template-specific parameters
        parameter_substitutions = {}
        if parameters:
            click.echo("\nTemplate Parameters:")
            for param in parameters:
                param_name = param['name']
                default_value = param['default_value']

                # Format prompt as "param_name (default: value)"
                prompt_text = f"{param_name} (default: {default_value})"
                value = click.prompt(prompt_text, default=default_value)

                # Map parameter name to user's new value for substitution
                # Only add to substitutions if the value actually changed
                if value != default_value:
                    parameter_substitutions[param_name] = value

        # Add basic info to parameter substitutions
        parameter_substitutions['user_id'] = user_id
        parameter_substitutions['app_id'] = app_id
        parameter_substitutions['id'] = pipeline_id

        click.echo(f"\nCreating pipeline '{pipeline_id}' from template '{template_name}'...")

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


def _init_interactive(pipeline_path):
    """Interactive pipeline initialization (original behavior).

    Args:
        pipeline_path: Destination path for the pipeline (already prepared)

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

    try:
        # Prompt for user inputs
        click.echo("Welcome to Clarifai Pipeline Initialization!")
        click.echo("Please provide the following information:")

        user_id = click.prompt("User ID", type=str)
        app_id = click.prompt("App ID", type=str)
        pipeline_id = click.prompt("Pipeline ID", default="hello-world-pipeline", type=str)
        num_steps = click.prompt("Number of pipeline steps", default=2, type=int)

        # Get step names
        step_names = []
        default_names = ["stepA", "stepB", "stepC", "stepD", "stepE", "stepF"]

        for i in range(num_steps):
            default_name = default_names[i] if i < len(default_names) else f"step{i + 1}"
            step_name = click.prompt(f"Name for step {i + 1}", default=default_name, type=str)
            step_names.append(step_name)

        click.echo(f"\nCreating pipeline '{pipeline_id}' with steps: {', '.join(step_names)}")

        # Create pipeline config.yaml
        config_path = os.path.join(pipeline_path, "config.yaml")
        if os.path.exists(config_path):
            logger.warning(f"File {config_path} already exists, skipping...")
        else:
            config_template = get_pipeline_config_template(
                pipeline_id=pipeline_id, user_id=user_id, app_id=app_id, step_names=step_names
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
                    step_id=step_id, user_id=user_id, app_id=app_id
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
        logger.error(f"Interactive initialization error: {e}")
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
    required=False,
    help='App ID to list pipelines from. If not provided, lists across all apps.',
)
@click.pass_context
def list(ctx, page_no, per_page, app_id):
    """List all pipelines for the user."""
    validate_context(ctx)

    from clarifai.client.app import App
    from clarifai.client.user import User

    if app_id:
        app = App(
            app_id=app_id,
            user_id=ctx.obj.current.user_id,
            pat=ctx.obj.current.pat,
            base_url=ctx.obj.current.api_base,
        )
        response = app.list_pipelines(page_no=page_no, per_page=per_page)
    else:
        user = User(
            user_id=ctx.obj.current.user_id,
            pat=ctx.obj.current.pat,
            base_url=ctx.obj.current.api_base,
        )
        response = user.list_pipelines(page_no=page_no, per_page=per_page)

    display_co_resources(
        response,
        custom_columns={
            'ID': lambda p: getattr(p, 'pipeline_id', ''),
            'USER_ID': lambda p: getattr(p, 'user_id', ''),
            'APP_ID': lambda p: getattr(p, 'app_id', ''),
            'VERSION_ID': lambda p: getattr(p, 'pipeline_version_id', ''),
            'DESCRIPTION': lambda p: getattr(p, 'description', ''),
            'CREATED_AT': lambda ps: convert_timestamp_to_string(getattr(ps, 'created_at', '')),
            'MODIFIED_AT': lambda ps: convert_timestamp_to_string(getattr(ps, 'modified_at', '')),
        },
        sort_by_columns=[
            ('CREATED_AT', 'desc'),
            ('ID', 'asc'),
        ],
    )
