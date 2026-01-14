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
            message = (
                f"No templates found for type '{template_type}'"
                if template_type
                else "No templates found"
            )
            click.echo(message)
            return

        # Display templates in a table format
        def format_template_data(template_list):
            """Format template data for display."""
            for template in template_list:
                yield {
                    'NAME': template['name'],
                    'TYPE': template['type'],
                }

        # Display the templates
        display_co_resources(
            format_template_data(templates),
            custom_columns={
                'NAME': lambda t: t['NAME'],
                'TYPE': lambda t: t['TYPE'],
            },
            sort_by_columns=[('TYPE', 'asc'), ('NAME', 'asc')],
        )

        # Show summary
        total_count = len(templates)
        summary_message = f"\nFound {total_count} template(s)"

        if template_type:
            summary_message += f" of type '{template_type}'"
        else:
            summary_message += " total"
            # Show available types only when not filtering by type
            types = sorted(set(t['type'] for t in templates))
            summary_message += f"\nAvailable types: {', '.join(types)}"

        click.echo(summary_message)

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
        click.echo()

        # Show pipeline steps (no descriptions)
        steps = template_info.get('step_directories', [])
        if steps:
            click.echo("Pipeline Steps:")
            for i, step in enumerate(steps, 1):
                click.echo(f"  {i}. {step}")
            click.echo()

        # Show parameters with defaults
        parameters = template_info.get('parameters', [])
        if parameters:
            click.echo("Template Parameters:")
            for param in parameters:
                default_value = param['default_value']
                click.echo(f"  {param['name']} (default: {default_value})")
            click.echo()
        else:
            click.echo("No template parameters found")

    except Exception as e:
        logger.error(f"Error getting template info: {e}")
        click.echo(f"Error: Could not get template information - {e}", err=True)
