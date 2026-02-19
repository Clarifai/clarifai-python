"""Structured output helpers for `clarifai model deploy`.

Provides phase headers, status lines, and progress indicators using click.echo
so that deploy output is visually organized into clear phases (Validate, Upload,
Build, Deploy, Monitor, Ready) rather than a wall of undifferentiated log lines.
"""

import click

HEADER_WIDTH = 58


def phase_header(title):
    """Print a phase separator: ── Title ────────────────────"""
    bar = "\u2500" * max(1, HEADER_WIDTH - len(title) - 4)
    click.echo(click.style(f"\n\u2500\u2500 {title} {bar}", fg="cyan", bold=True))


def info(label, value):
    """Print a labeled info line: '  Label:  value'"""
    styled_label = click.style(f"{label}:", fg="white", bold=True)
    click.echo(f"  {styled_label:30s} {value}")


def status(message, nl=True):
    """Print a status message."""
    click.echo(f"  {message}", nl=nl)


def inline_progress(message):
    """Print inline progress (overwrites current line)."""
    click.echo(f"\r  {message:<70}", nl=False)


def clear_inline():
    """Clear inline progress line."""
    click.echo(f"\r{' ':74}\r", nl=False)


def success(message):
    """Print a success message."""
    click.echo(click.style(f"  {message}", fg="green"))


def warning(message):
    """Print a warning."""
    click.echo(click.style(f"  [warning] {message}", fg="yellow"))


def event(message):
    """Print a deployment event (dimmed)."""
    click.echo(click.style(f"  {message}", fg="bright_black"))
