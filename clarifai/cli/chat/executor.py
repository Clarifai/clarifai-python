"""Command executor for Clarifai CLI agent.

This module provides the ability to execute shell commands (specifically clarifai CLI commands)
and Python SDK code, returning the results. Designed to be used by the chat interface to execute
commands on behalf of the user.
"""

import io
import os
import re
import subprocess
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import List, Optional, Tuple

from clarifai.utils.logging import logger

# Commands that are safe to execute without confirmation
SAFE_COMMANDS = {
    'clarifai model list',
    'clarifai pipeline list',
    'clarifai pipelinetemplate list',
    'clarifai deployment list',
    'clarifai computecluster list',
    'clarifai nodepool list',
    'clarifai artifact list',
    'clarifai config list',
    'clarifai config show',
    'clarifai --help',
    'clarifai model --help',
    'clarifai pipeline --help',
    'clarifai --version',
}

# Command prefixes that are safe (read-only operations)
SAFE_PREFIXES = [
    'clarifai model list',
    'clarifai pipeline list',
    'clarifai pipelinetemplate list',
    'clarifai pipelinetemplate discover',
    'clarifai deployment list',
    'clarifai computecluster list',
    'clarifai nodepool list',
    'clarifai artifact list',
    'clarifai artifact get',
    'clarifai config list',
    'clarifai config show',
    'clarifai model signatures',
]

# Commands that should NEVER be auto-executed (require user interaction or are destructive)
BLOCKED_COMMANDS = [
    'clarifai login',  # Interactive
    'clarifai logout',  # Destructive
]

# Dangerous command patterns (checked at start of command)
BLOCKED_PREFIXES = [
    'rm ',
    'rm.exe',
    'del ',
    'del.exe',
    'rmdir',
    'format ',
    'sudo ',
]

# Placeholder patterns that indicate the command needs user input
# These are common patterns used by LLMs when providing example commands
PLACEHOLDER_PATTERNS = [
    r'\bYOUR_\w+\b',  # YOUR_APP_ID, YOUR_USER_ID, YOUR_PAT, etc.
    r'\bMY_\w+\b',  # MY_APP_ID, MY_USER_ID, etc.
    r'<[a-z_]+>',  # <app_id>, <user_id>, etc.
    r'\[YOUR[^\]]*\]',  # [YOUR APP ID], [YOUR_APP_ID]
    r'\{[a-z_]+\}',  # {app_id}, {user_id}
    r'REPLACE_\w+',  # REPLACE_THIS, REPLACE_WITH_YOUR_ID
    r'xxx+',  # xxx, xxxx (placeholder markers)
    r'\.\.\.+',  # ... (ellipsis as placeholder)
    r'<insert[^>]*>',  # <insert your value here>
    r'EXAMPLE_\w+',  # EXAMPLE_APP_ID
    r'\bFOO\b|\bBAR\b|\bBAZ\b',  # Common placeholder names
]

# Patterns that can be auto-substituted with environment variable values
# Maps regex pattern -> (env_var_name, description)
# The user_id and PAT are available since they're used to authenticate with the agent
SUBSTITUTION_PATTERNS = [
    # User ID substitutions
    (r'\bYOUR_USER_ID\b', 'CLARIFAI_USER_ID'),
    (r'\bMY_USER_ID\b', 'CLARIFAI_USER_ID'),
    (r'<user_id>', 'CLARIFAI_USER_ID'),
    (r'\{user_id\}', 'CLARIFAI_USER_ID'),
    (r'--user_id\s+YOUR_USER_ID', '--user_id {CLARIFAI_USER_ID}'),
    (r'--user_id\s+<user_id>', '--user_id {CLARIFAI_USER_ID}'),
    # Note: PAT should NOT be substituted into commands for security
    # Commands should use the environment variable directly
]

# Safe Python SDK patterns (read-only operations)
# These patterns are checked to auto-execute Python code without confirmation
SAFE_PYTHON_PATTERNS = [
    r'\.list_apps\s*\(',
    r'\.list_models\s*\(',
    r'\.list_datasets\s*\(',
    r'\.list_workflows\s*\(',
    r'\.list_concepts\s*\(',
    r'\.list_pipelines\s*\(',
    r'\.list_versions\s*\(',
    r'\.list_compute_clusters\s*\(',
    r'\.list_nodepools\s*\(',
    r'\.list_deployments\s*\(',
    r'\.list_runners\s*\(',
    r'\.get_input_count\s*\(',
]

# Dangerous Python patterns that should NEVER be auto-executed
BLOCKED_PYTHON_PATTERNS = [
    r'\.delete\s*\(',
    r'\.create_app\s*\(',
    r'\.create_dataset\s*\(',
    r'\.create_model\s*\(',
    r'\.create_workflow\s*\(',
    r'\.upload\s*\(',
    r'\.train\s*\(',
    r'\.evaluate\s*\(',
    r'\.export\s*\(',
    r'\bos\.system\s*\(',
    r'\bsubprocess\.',
    r'\bexec\s*\(',
    r'\beval\s*\(',
    r'\bopen\s*\([^)]*["\']w',  # open with write mode
    r'\brmtree\s*\(',
    r'\bremove\s*\(',
    r'\bunlink\s*\(',
]


def get_user_id_from_env() -> Optional[str]:
    """Get the current user ID from environment variables.

    Returns:
        The user ID if available, None otherwise
    """
    return os.environ.get('CLARIFAI_USER_ID')


# Module-level variable to store the current user_id (set by chat command)
_current_user_id: Optional[str] = None


def set_current_user_id(user_id: str) -> None:
    """Set the current user ID for command substitution.

    This should be called by the chat command with the user_id from the config context.
    """
    global _current_user_id
    _current_user_id = user_id


def get_current_user_id() -> Optional[str]:
    """Get the current user ID from module variable or environment.

    Returns:
        The user ID if available, None otherwise
    """
    return _current_user_id or os.environ.get('CLARIFAI_USER_ID')


def substitute_known_values(command: str) -> Tuple[str, List[str]]:
    """Substitute placeholder values with known environment variable values.

    Only substitutes safe values (user_id). PAT is never substituted for security.

    Args:
        command: The command string with potential placeholders

    Returns:
        Tuple of (substituted_command, list of substitutions made)
    """
    substitutions = []
    result = command

    user_id = get_current_user_id()

    if user_id:
        # Substitute user_id placeholders
        user_id_patterns = [
            (r'\bYOUR_USER_ID\b', user_id),
            (r'\bMY_USER_ID\b', user_id),
            (r'<user_id>', user_id),
            (r'\{user_id\}', user_id),
        ]

        for pattern, replacement in user_id_patterns:
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                substitutions.append(f"user_id -> {user_id}")
                break  # Only report once

    return result, substitutions


def is_safe_python_code(code: str) -> bool:
    """Check if Python code is safe to execute without confirmation.

    Args:
        code: The Python code string to check

    Returns:
        True if the code only contains read-only SDK operations, False otherwise
    """
    # Check for blocked patterns first
    for pattern in BLOCKED_PYTHON_PATTERNS:
        if re.search(pattern, code):
            return False

    # Check if it contains at least one safe pattern (SDK list operation)
    has_safe_pattern = any(re.search(pattern, code) for pattern in SAFE_PYTHON_PATTERNS)

    # Must have a safe pattern and use clarifai imports
    if has_safe_pattern and ('clarifai' in code or 'User(' in code or 'App(' in code):
        return True

    return False


def is_clarifai_python_code(code: str) -> bool:
    """Check if Python code uses the Clarifai SDK.

    Args:
        code: The Python code string to check

    Returns:
        True if the code imports or uses clarifai SDK
    """
    clarifai_indicators = [
        'from clarifai',
        'import clarifai',
        'clarifai.client',
        'User(',
        'App(',
        'Model(',
        'Workflow(',
        'Dataset(',
        'Inputs(',
        'Search(',
    ]
    return any(indicator in code for indicator in clarifai_indicators)


def execute_python(code: str, timeout: int = 60) -> 'CommandResult':
    """Execute Python code and return the result.

    Captures stdout/stderr and returns the output.
    Sets up CLARIFAI_USER_ID environment variable if available from config.

    Args:
        code: The Python code to execute
        timeout: Maximum time to wait for execution (seconds) - not fully implemented

    Returns:
        CommandResult with the execution details
    """
    logger.debug(f"Executing Python code:\n{code}")

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Ensure CLARIFAI_USER_ID is set for SDK calls
    user_id = get_current_user_id()
    old_user_id = os.environ.get('CLARIFAI_USER_ID')
    if user_id and not old_user_id:
        os.environ['CLARIFAI_USER_ID'] = user_id

    # Create a namespace with common imports pre-loaded
    namespace = {
        '__name__': '__main__',
        '__builtins__': __builtins__,
    }

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, namespace)

        stdout = stdout_capture.getvalue().strip()
        stderr = stderr_capture.getvalue().strip()

        return CommandResult(
            command=f"python:\n{code[:100]}..." if len(code) > 100 else f"python:\n{code}",
            returncode=0,
            stdout=stdout,
            stderr=stderr,
            success=True,
        )

    except Exception as e:
        error_msg = traceback.format_exc()
        return CommandResult(
            command=f"python:\n{code[:100]}..." if len(code) > 100 else f"python:\n{code}",
            returncode=1,
            stdout=stdout_capture.getvalue().strip(),
            stderr=error_msg,
            success=False,
        )
    finally:
        # Restore original CLARIFAI_USER_ID if we modified it
        if user_id and not old_user_id:
            del os.environ['CLARIFAI_USER_ID']


@dataclass
class CommandResult:
    """Result of a command execution."""

    command: str
    returncode: int
    stdout: str
    stderr: str
    success: bool

    @property
    def output(self) -> str:
        """Get combined output, preferring stdout."""
        if self.stdout:
            return self.stdout
        return self.stderr


def is_safe_command(command: str) -> bool:
    """Check if a command is safe to execute without confirmation.

    Args:
        command: The command string to check

    Returns:
        True if the command is safe (read-only), False otherwise
    """
    cmd_lower = command.lower().strip()

    # Check blocked clarifai commands (exact match at start)
    for blocked in BLOCKED_COMMANDS:
        if cmd_lower.startswith(blocked.lower()):
            return False

    # Check dangerous command prefixes
    for prefix in BLOCKED_PREFIXES:
        if cmd_lower.startswith(prefix.lower()):
            return False

    # Check exact matches in safe commands
    if cmd_lower in SAFE_COMMANDS:
        return True

    # Check safe prefixes
    for prefix in SAFE_PREFIXES:
        if cmd_lower.startswith(prefix.lower()):
            return True

    # All other commands require confirmation
    return False


def is_clarifai_command(command: str) -> bool:
    """Check if a command is a clarifai CLI command.

    Args:
        command: The command string to check

    Returns:
        True if the command starts with 'clarifai'
    """
    return command.strip().lower().startswith('clarifai')


def has_placeholder_values(command: str) -> Tuple[bool, Optional[str]]:
    """Check if a command contains placeholder values that need user input.

    Args:
        command: The command string to check

    Returns:
        Tuple of (has_placeholder, matched_placeholder)
    """
    for pattern in PLACEHOLDER_PATTERNS:
        match = re.search(pattern, command, re.IGNORECASE)
        if match:
            return True, match.group(0)
    return False, None


def execute_command(command: str, timeout: int = 60) -> CommandResult:
    """Execute a shell command and return the result.

    Args:
        command: The command to execute
        timeout: Maximum time to wait for command completion (seconds)

    Returns:
        CommandResult with the execution details
    """
    logger.debug(f"Executing command: {command}")

    try:
        # Use shell=True on Windows for proper command handling
        # Use subprocess.DEVNULL for stdin to prevent hanging on input prompts
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace',
            stdin=subprocess.DEVNULL,
            check=False,  # Prevent hanging on input prompts
        )

        return CommandResult(
            command=command,
            returncode=result.returncode,
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip(),
            success=result.returncode == 0,
        )

    except subprocess.TimeoutExpired:
        return CommandResult(
            command=command,
            returncode=-1,
            stdout='',
            stderr=f'Command timed out after {timeout} seconds',
            success=False,
        )
    except Exception as e:
        return CommandResult(
            command=command,
            returncode=-1,
            stdout='',
            stderr=str(e),
            success=False,
        )


def parse_commands_from_response(
    response_text: str,
) -> Tuple[List[Tuple[str, List[str], str]], List[Tuple[str, str]]]:
    """Parse executable commands and Python code from LLM response text.

    Looks for commands in:
    1. <execute>command</execute> tags (CLI commands)
    2. ```bash or ```shell code blocks with clarifai commands
    3. ```python code blocks with clarifai SDK code

    Substitutes known values (like user_id from environment) and filters out
    commands that still contain placeholder values (e.g., YOUR_APP_ID).

    Args:
        response_text: The LLM response text

    Returns:
        Tuple of (executable_commands, skipped_commands)
        where executable_commands is a list of (command, substitutions_made, type) tuples
        type is either 'cli' or 'python'
        and skipped_commands is a list of (command, reason) tuples
    """
    commands = []
    skipped = []

    def process_cli_command(cmd: str):
        """Process a single CLI command - substitute values and check for placeholders."""
        # First, substitute known values from environment
        substituted_cmd, substitutions = substitute_known_values(cmd)

        # Then check if there are still placeholders remaining
        has_placeholder, placeholder = has_placeholder_values(substituted_cmd)
        if has_placeholder:
            skipped.append((cmd, f"contains placeholder '{placeholder}'"))
        else:
            commands.append((substituted_cmd, substitutions, 'cli'))

    def process_python_code(code: str):
        """Process Python code - check for placeholders and clarifai usage."""
        # Check for placeholders
        has_placeholder, placeholder = has_placeholder_values(code)
        if has_placeholder:
            skipped.append(
                (
                    code[:80] + "..." if len(code) > 80 else code,
                    f"contains placeholder '{placeholder}'",
                )
            )
        elif is_clarifai_python_code(code):
            commands.append((code, [], 'python'))
        # If not clarifai code, just ignore it (don't add to skipped)

    # Pattern 1: <execute>command</execute> tags
    execute_pattern = r'<execute>(.*?)</execute>'
    execute_matches = re.findall(execute_pattern, response_text, re.DOTALL)
    for match in execute_matches:
        cmd = match.strip()
        if cmd and is_clarifai_command(cmd):
            process_cli_command(cmd)

    # Pattern 2: ```bash or ```shell code blocks
    bash_block_pattern = r'```(?:bash|shell|sh|powershell|ps1)\s*\n(.*?)```'
    bash_matches = re.findall(bash_block_pattern, response_text, re.DOTALL | re.IGNORECASE)
    for match in bash_matches:
        # Extract individual commands from the code block
        for line in match.strip().split('\n'):
            cmd = line.strip()
            # Skip comments and empty lines
            if cmd and not cmd.startswith('#') and is_clarifai_command(cmd):
                process_cli_command(cmd)

    # Pattern 3: ```python code blocks
    python_block_pattern = r'```python\s*\n(.*?)```'
    python_matches = re.findall(python_block_pattern, response_text, re.DOTALL | re.IGNORECASE)
    for match in python_matches:
        code = match.strip()
        if code:
            process_python_code(code)

    return commands, skipped


def format_command_output(result: CommandResult, max_lines: int = 50) -> str:
    """Format command output for display.

    Args:
        result: The CommandResult to format
        max_lines: Maximum number of lines to show

    Returns:
        Formatted output string
    """
    output_lines = result.output.split('\n')

    if len(output_lines) > max_lines:
        truncated = output_lines[:max_lines]
        truncated.append(f'... ({len(output_lines) - max_lines} more lines)')
        output = '\n'.join(truncated)
    else:
        output = result.output

    status = "[OK]" if result.success else "[FAIL]"
    return f"{status} `{result.command}`\n\n```\n{output}\n```"
