"""Chat module for Clarifai CLI.

This module provides the interactive chat interface, agent system,
command execution, skills management, and RAG capabilities.
"""

from clarifai.cli.chat.agent import (
    ClarifaiAgent,
    build_agent_system_prompt,
    parse_skill_calls_from_response,
    parse_tool_calls_from_response,
)
from clarifai.cli.chat.command import DEFAULT_CHAT_MODEL_URL, chat, sanitize_sensitive_data
from clarifai.cli.chat.executor import (
    CommandResult,
    execute_command,
    execute_python,
    format_command_output,
    get_current_user_id,
    get_user_id_from_env,
    has_placeholder_values,
    is_clarifai_command,
    is_clarifai_python_code,
    is_safe_command,
    is_safe_python_code,
    parse_commands_from_response,
    set_current_user_id,
    substitute_known_values,
)
from clarifai.cli.chat.rag import ClarifaiCodeRAG, build_system_prompt_with_rag
from clarifai.cli.chat.skills import (
    Skill,
    SkillLoader,
    SkillRegistry,
    build_skills_system_prompt,
    parse_skill_markdown,
)

__all__ = [
    # Command
    'chat',
    'DEFAULT_CHAT_MODEL_URL',
    'sanitize_sensitive_data',
    # Agent
    'ClarifaiAgent',
    'build_agent_system_prompt',
    'parse_skill_calls_from_response',
    'parse_tool_calls_from_response',
    # Executor
    'execute_command',
    'execute_python',
    'parse_commands_from_response',
    'is_safe_command',
    'is_safe_python_code',
    'is_clarifai_command',
    'is_clarifai_python_code',
    'has_placeholder_values',
    'substitute_known_values',
    'get_user_id_from_env',
    'get_current_user_id',
    'set_current_user_id',
    'format_command_output',
    'CommandResult',
    # Skills
    'Skill',
    'SkillLoader',
    'SkillRegistry',
    'parse_skill_markdown',
    'build_skills_system_prompt',
    # RAG
    'ClarifaiCodeRAG',
    'build_system_prompt_with_rag',
]
