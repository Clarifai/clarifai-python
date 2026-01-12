"""CLI commands for managing pipeline templates."""

import shutil

import click

from clarifai.cli.base import cli
from clarifai.utils.cli import AliasedGroup, display_co_resources
from clarifai.utils.logging import logger
from clarifai.utils.template_manager import TemplateManager


@cli.group(
    ['pipelinetemplate', 'pt'],
    cls=AliasedGroup,
    context_settings={'max_content_width': shutil.get_terminal_size().columns - 10},
)
def pipelinetemplate():
    """Manage pipeline templates: list, discover, etc"""


@pipelinetemplate.command(name='list', aliases=['ls'])
@click.option(
    '--type',
    'template_type',
    required=False,
    help='Filter templates by type (train, data, agent, etc.)',
)
def list_templates(template_type):
    """List available pipeline templates.

    Lists all available pipeline templates from the template repository.
    Templates are organized by type and can be filtered using the --type option.

    Examples:
        clarifai pipelinetemplate ls                    # List all templates
        clarifai pipelinetemplate ls --type=train       # List only training templates
        clarifai pipelinetemplate ls --type=data        # List only data processing templates
    """
    try:
        template_manager = TemplateManager()
        templates = template_manager.list_templates(template_type)

        if not templates:
            if template_type:
                click.echo(f"No templates found for type '{template_type}'")
            else:
                click.echo("No templates found")
            return

        # Display templates in a table format
        def format_template_data(template_list):
            """Format template data for display."""
            for template in template_list:
                description = template['description']
                # Ellipsize description to 80 characters
                if len(description) > 80:
                    description = description[:77] + "..."

                yield {
                    'NAME': template['name'],
                    'TYPE': template['type'],
                    'DESCRIPTION': description,
                }

        # Custom columns for template display
        custom_columns = {
            'NAME': lambda t: t['NAME'],
            'TYPE': lambda t: t['TYPE'],
            'DESCRIPTION': lambda t: t['DESCRIPTION'],
        }

        # Display the templates
        display_co_resources(
            format_template_data(templates),
            custom_columns=custom_columns,
            sort_by_columns=[('TYPE', 'asc'), ('NAME', 'asc')],
        )

        # Show summary
        total_count = len(templates)
        if template_type:
            click.echo(f"\nFound {total_count} template(s) of type '{template_type}'")
        else:
            click.echo(f"\nFound {total_count} template(s) total")
            # Show available types only when not filtering by type
            types = sorted(set(t['type'] for t in templates))
            click.echo(f"Available types: {', '.join(types)}")

    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        click.echo(f"Error: Could not list templates - {e}", err=True)


@pipelinetemplate.command()
@click.argument('template_name', required=True)
def info(template_name):
    """Show detailed information about a specific template.

    Displays detailed information about a template including its parameters,
    step structure, and description.

    TEMPLATE_NAME: Name of the template to show information for

    Examples:
        clarifai pipelinetemplate info image-classification
        clarifai pipelinetemplate info text-prep
    """
    try:
        template_manager = TemplateManager()
        template_info = template_manager.get_template_info(template_name)

        if not template_info:
            click.echo(f"Template '{template_name}' not found", err=True)
            return

        click.echo(f"Template: {template_info['name']}")
        click.echo(f"Type: {template_info['type']}")

        # Show template description
        use_case = template_info.get('use_case', '')
        if use_case and use_case != "Pipeline template":
            click.echo(f"Description: {use_case}")

        click.echo()

        # Show pipeline steps with descriptions
        steps = template_info.get('step_directories', [])
        step_descriptions = template_info.get('step_descriptions', {})
        if steps:
            click.echo("Pipeline Steps:")
            for i, step in enumerate(steps, 1):
                if step in step_descriptions:
                    # Show step name and description on the same line
                    description = step_descriptions[step]
                    click.echo(f"  {i}. {step} ({description})")
                else:
                    click.echo(f"  {i}. {step}")
            click.echo()

        # Show parameters
        parameters = template_info.get('parameters', [])
        if parameters:
            click.echo("Template Parameters:")
            for param in parameters:
                click.echo(f"  {param['name']} ({param['description']})")
            click.echo()
        else:
            click.echo("No template parameters found")

        # Show pipeline config info
        config = template_info.get('config', {})
        pipeline_config = config.get('pipeline', {})
        if 'id' in pipeline_config:
            click.echo(f"Default Pipeline ID: {pipeline_config['id']}")

    except Exception as e:
        logger.error(f"Error getting template info: {e}")
        click.echo(f"Error: Could not get template information - {e}", err=True)
