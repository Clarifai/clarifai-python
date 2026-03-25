"""CLI commands for managing Clarifai agent skills."""

import click

from clarifai.cli.base import cli
from clarifai.utils.cli import AliasedGroup


@cli.group(['skills', 'sk'], cls=AliasedGroup, short_help='Manage Clarifai agent skills.')
def skills():
    """Install and manage Clarifai skills for AI coding assistants.

    \b
    Quick start:
      clarifai skills install              # install all skills (auto-detects agents)
      clarifai skills list --remote        # browse available skills
      clarifai skills list --installed     # see what's installed
      clarifai skills update               # update to latest
    """


@skills.command()
@click.argument('skill_names', nargs=-1)
@click.option('--claude', is_flag=True, help='Install for Claude Code.')
@click.option('--codex', is_flag=True, help='Install for OpenAI Codex.')
@click.option('--cursor', is_flag=True, help='Install for Cursor.')
@click.option('--all-agents', is_flag=True, help='Install for all supported agents.')
@click.option(
    '--global/--local',
    'global_',
    default=True,
    show_default=True,
    help='Install globally (~/), or project-level (./).',
)
@click.option('--force', is_flag=True, help='Overwrite existing skills.')
@click.option(
    '--source',
    type=click.Path(exists=True),
    help='Install from local skills repo clone instead of GitHub.',
)
def install(skill_names, claude, codex, cursor, all_agents, global_, force, source):
    """Install Clarifai skills for AI assistants.

    \b
    If no agent flag is specified, auto-detects installed agents.
    If no SKILL_NAMES given, installs all available skills.

    \b
    Examples:
      clarifai skills install                              # all skills, auto-detect agents
      clarifai skills install --claude                     # all skills for Claude
      clarifai skills install clarifai-cli --claude        # one skill for Claude
      clarifai skills install --claude --cursor            # all skills, multi-agent
      clarifai skills install --local                      # project-level install
    """
    from clarifai.utils.skills import (
        install_skills,
        list_remote_skills,
        resolve_agents,
    )

    agents = resolve_agents(claude, codex, cursor, all_agents)
    skill_ids = list(skill_names) if skill_names else None

    # Try to validate against registry, but proceed even if registry is unavailable
    try:
        click.echo("Fetching Clarifai skills registry...")
        remote = list_remote_skills()
        available_ids = [s["id"] for s in remote]
        if skill_ids:
            for sid in skill_ids:
                if sid not in available_ids:
                    raise click.ClickException(
                        f"Unknown skill '{sid}'. Available: {', '.join(available_ids)}"
                    )
        count = len(skill_ids) if skill_ids else len(available_ids)
        click.echo(f"Found {len(available_ids)} available skills. Installing {count}...")
    except click.ClickException:
        raise
    except Exception:
        click.echo("Registry unavailable, downloading all skills from GitHub...")
        skill_ids = skill_ids or None

    try:
        downloaded, linked_agents = install_skills(
            skill_ids=skill_ids, agents=agents, global_=global_, force=force, source=source
        )
    except Exception as e:
        raise click.ClickException(f"Installation failed: {e}")

    scope = "global" if global_ else "project"
    click.echo(f"\nInstalled {len(downloaded)} skills ({scope}).")
    if linked_agents:
        click.echo(f"Linked to: {', '.join(a.title() for a in linked_agents)}")
    click.echo("\nDone! Skills are now available.")


@skills.command(['ls'], name='list')
@click.option('--remote', 'show_remote', is_flag=True, help='Show available skills from registry.')
@click.option('--installed', 'show_installed', is_flag=True, help='Show installed skills.')
@click.option('--claude', is_flag=True, help='Filter by Claude.')
@click.option('--codex', is_flag=True, help='Filter by Codex.')
@click.option('--cursor', is_flag=True, help='Filter by Cursor.')
@click.option(
    '--global/--local',
    'global_',
    default=True,
)
def list_skills(show_remote, show_installed, claude, codex, cursor, global_):
    """List available or installed Clarifai skills.

    \b
    Examples:
      clarifai skills list --remote        # available skills
      clarifai skills list --installed     # installed skills
    """
    from clarifai.utils.skills import (
        list_installed_skills,
        list_remote_skills,
        resolve_agents,
    )

    if not show_remote and not show_installed:
        show_remote = True  # default to showing remote

    if show_remote:
        click.echo("Fetching Clarifai skills registry...\n")
        try:
            remote = list_remote_skills()
        except Exception as e:
            raise click.ClickException(f"Failed to fetch registry: {e}")

        click.echo(f"Available Clarifai Skills ({len(remote)}):\n")
        if not remote:
            click.echo("  (no skills found in registry)")
        else:
            max_name = max(len(s["id"]) for s in remote)
            for s in remote:
                click.echo(f"  {s['id']:<{max_name}}  {s['description'][:80]}")
            click.echo("\nInstall: clarifai skills install")

    if show_installed:
        agents = resolve_agents(claude, codex, cursor, False)
        installed = list_installed_skills(agents, global_)
        scope = "global" if global_ else "project"
        click.echo(f"\nInstalled Clarifai Skills ({scope}):\n")
        for agent, skill_list in installed.items():
            click.echo(f"  {agent.title()}:")
            if skill_list:
                for sid in skill_list:
                    click.echo(f"    {sid}")
            else:
                click.echo("    (none)")


@skills.command()
@click.option('--claude', is_flag=True, help='Update for Claude.')
@click.option('--codex', is_flag=True, help='Update for Codex.')
@click.option('--cursor', is_flag=True, help='Update for Cursor.')
@click.option('--all-agents', is_flag=True, help='Update for all agents.')
@click.option(
    '--global/--local',
    'global_',
    default=True,
)
@click.option(
    '--source', type=click.Path(exists=True), help='Update from local skills repo clone.'
)
def update(claude, codex, cursor, all_agents, global_, source):
    """Update installed skills to the latest version.

    \b
    Examples:
      clarifai skills update               # update all (auto-detect agents)
      clarifai skills update --claude      # update for Claude only
    """
    from clarifai.utils.skills import install_skills, resolve_agents

    agents = resolve_agents(claude, codex, cursor, all_agents)

    click.echo("Updating skills...")
    try:
        downloaded, linked_agents = install_skills(
            skill_ids=None, agents=agents, global_=global_, force=True, source=source
        )
    except Exception as e:
        raise click.ClickException(f"Update failed: {e}")

    click.echo(
        f"Updated {len(downloaded)} skills for {', '.join(a.title() for a in linked_agents)}."
    )


@skills.command()
@click.argument('skill_names', nargs=-1)
@click.option('--claude', is_flag=True, help='Remove from Claude.')
@click.option('--codex', is_flag=True, help='Remove from Codex.')
@click.option('--cursor', is_flag=True, help='Remove from Cursor.')
@click.option('--all-agents', is_flag=True, help='Remove from all agents.')
@click.option('--all', 'remove_all', is_flag=True, help='Remove all Clarifai skills.')
@click.option(
    '--global/--local',
    'global_',
    default=True,
)
def remove(skill_names, claude, codex, cursor, all_agents, remove_all, global_):
    """Remove installed Clarifai skills.

    \b
    Examples:
      clarifai skills remove --all                       # remove everything
      clarifai skills remove clarifai-cli --claude       # remove one skill
    """
    from clarifai.utils.skills import remove_skills, resolve_agents

    agents = resolve_agents(claude, codex, cursor, all_agents)
    skill_ids = list(skill_names) if skill_names else None

    if not skill_ids and not remove_all:
        raise click.ClickException("Specify skill names or use --all to remove all skills.")

    removed = remove_skills(
        skill_ids=skill_ids, agents=agents, global_=global_, remove_all=remove_all
    )

    if removed:
        click.echo(f"Removed {len(removed)} skills: {', '.join(removed)}")
    else:
        click.echo("No skills found to remove.")
