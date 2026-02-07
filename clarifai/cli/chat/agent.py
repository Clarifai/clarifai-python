"""Agent system for Clarifai CLI - provides skill-based guidance.

This module provides a skills-based approach where markdown documentation files
guide the LLM on how to handle different types of requests. Skills are loaded
from markdown files with YAML frontmatter.
"""

import json
import re
from typing import Any, Dict, List

from clarifai.cli.chat.skills import SkillRegistry, build_skills_system_prompt
from clarifai.utils.logging import logger


class ClarifaiAgent:
    """Agent for Clarifai CLI that uses markdown-based skills for guidance.

    Skills are documentation files that tell the LLM how to route requests
    and provide accurate guidance. They are NOT executable code.
    """

    def __init__(self, pat: str = None, user_id: str = None, skills_dir: str = None):
        """Initialize the agent.

        Args:
            pat: Personal Access Token (kept for context, passed to skills)
            user_id: Clarifai user ID (kept for context)
            skills_dir: Optional custom skills directory path
        """
        self.pat = pat
        self.user_id = user_id
        self.skills = SkillRegistry(pat=pat, user_id=user_id, skills_dir=skills_dir)
        logger.debug(f"Initialized agent with {len(self.skills.skills)} skills")

    def get_skills_for_llm(self) -> List[Dict[str, Any]]:
        """Get skill definitions formatted for LLM."""
        return self.skills.get_skills_for_llm()

    def get_skill_names(self) -> List[str]:
        """Get list of loaded skill names."""
        return list(self.skills.skills.keys())

    def reload_skills(self):
        """Reload skills from disk."""
        self.skills.reload()

    # Backward compatibility - these no longer execute anything
    # but are kept to avoid breaking existing code
    @property
    def tools(self) -> Dict:
        """Backward compatibility property - returns skills."""
        return self.skills.skills

    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Alias for get_skills_for_llm for backward compatibility."""
        return self.get_skills_for_llm()


def build_agent_system_prompt(agent: ClarifaiAgent) -> str:
    """Build a system prompt that includes all loaded skill documentation.

    Args:
        agent: ClarifaiAgent instance with loaded skills

    Returns:
        System prompt string with skill routing and documentation
    """
    return build_skills_system_prompt(agent.skills)


# Legacy parsing functions - kept for potential future use or backward compat
def parse_skill_calls_from_response(response_text: str) -> List[Dict[str, Any]]:
    """Parse skill references from LLM response text.

    Note: With markdown-based skills, the LLM provides guidance rather than
    requesting skill execution. This function is kept for potential future
    use where skills might have executable components.

    Args:
        response_text: Raw LLM response text

    Returns:
        List of skill reference dicts or empty list if none found
    """
    skill_calls = []

    # Try skill_call format
    skill_pattern = r"<skill_call>(.*?)</skill_call>"
    skill_matches = re.findall(skill_pattern, response_text, re.DOTALL)

    for match in skill_matches:
        try:
            skill_call = json.loads(match)
            if "skill" in skill_call:
                skill_calls.append(skill_call)
        except json.JSONDecodeError:
            logger.debug(f"Could not parse skill call: {match}")

    # Also check for legacy tool_call format
    tool_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_matches = re.findall(tool_pattern, response_text, re.DOTALL)

    for match in tool_matches:
        try:
            tool_call = json.loads(match)
            if "tool" in tool_call:
                skill_calls.append(
                    {"skill": tool_call.get("tool"), "params": tool_call.get("params", {})}
                )
        except json.JSONDecodeError:
            logger.debug(f"Could not parse tool call: {match}")

    return skill_calls


# Backward compatibility alias
def parse_tool_calls_from_response(response_text: str) -> List[Dict[str, Any]]:
    """Backward compatibility alias for parse_skill_calls_from_response."""
    return parse_skill_calls_from_response(response_text)
