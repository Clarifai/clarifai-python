"""Skills-based approach for Clarifai CLI agent.

Skills are loaded from markdown files with YAML frontmatter. Each skill provides
guidance to the LLM on how to handle specific types of requests.

Skill Markdown Format:
```markdown
---
name: skill-name
description: When to use this skill and what it handles
---

# Skill Title

Detailed documentation, examples, code snippets, etc.
```
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from clarifai.utils.logging import logger


@dataclass
class Skill:
    """A skill loaded from a markdown file.

    Skills are documentation/guidance for the LLM, not executable code.
    They tell the LLM how to route requests and what advice to give.
    """

    name: str
    description: str
    content: str  # Full markdown content after frontmatter
    source_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert skill to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "source_file": self.source_file,
        }


def parse_skill_markdown(content: str, source_file: str = None) -> Optional[Skill]:
    """Parse a skill markdown file with YAML frontmatter.

    Args:
        content: Raw markdown content
        source_file: Optional source file path for debugging

    Returns:
        Skill object or None if parsing fails
    """
    # Match YAML frontmatter between --- delimiters
    frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
    match = re.match(frontmatter_pattern, content, re.DOTALL)

    if not match:
        logger.warning(f"No frontmatter found in skill file: {source_file}")
        return None

    frontmatter_text = match.group(1)
    markdown_body = match.group(2).strip()

    # Parse YAML frontmatter manually (avoid yaml dependency for simple case)
    frontmatter = {}
    current_key = None
    current_value = []

    for line in frontmatter_text.split('\n'):
        # Check if this is a new key
        if ':' in line and not line.startswith(' ') and not line.startswith('\t'):
            # Save previous key-value if exists
            if current_key:
                frontmatter[current_key] = ' '.join(current_value).strip()

            key, value = line.split(':', 1)
            current_key = key.strip()
            current_value = [value.strip()]
        elif current_key:
            # Continuation of previous value
            current_value.append(line.strip())

    # Save last key-value
    if current_key:
        frontmatter[current_key] = ' '.join(current_value).strip()

    name = frontmatter.get('name')
    description = frontmatter.get('description', '')

    if not name:
        logger.warning(f"Skill missing 'name' in frontmatter: {source_file}")
        return None

    return Skill(
        name=name,
        description=description,
        content=markdown_body,
        source_file=source_file,
    )


class SkillLoader:
    """Loads skills from markdown files in a directory."""

    def __init__(self, skills_dir: str = None):
        """Initialize the skill loader.

        Args:
            skills_dir: Directory containing skill markdown files.
                       If None, uses default skills directory.
        """
        if skills_dir is None:
            # Default to skills directory relative to this file (chat/skills/)
            skills_dir = os.path.join(os.path.dirname(__file__), 'skills')

        self.skills_dir = skills_dir
        self.skills: Dict[str, Skill] = {}
        self._load_skills()

    def _load_skills(self):
        """Load all skill markdown files from the skills directory recursively."""
        if not os.path.exists(self.skills_dir):
            logger.debug(f"Skills directory does not exist: {self.skills_dir}")
            return

        # Recursively find all SKILL.md files in subdirectories
        skills_path = Path(self.skills_dir)
        for skill_file in skills_path.rglob('SKILL.md'):
            self._load_skill_file(str(skill_file))

        logger.debug(f"Loaded {len(self.skills)} skills from {self.skills_dir}")

    def _load_skill_file(self, filepath: str):
        """Load a single skill file along with its references and examples."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            skill = parse_skill_markdown(content, filepath)
            if skill:
                # Load additional content from references/ and examples/ subdirectories
                skill_dir = Path(filepath).parent
                additional_content = []

                for subdir_name in ['references', 'examples']:
                    subdir = skill_dir / subdir_name
                    if subdir.exists() and subdir.is_dir():
                        for md_file in sorted(subdir.glob('*.md')):
                            try:
                                with open(md_file, 'r', encoding='utf-8') as f:
                                    file_content = f.read()
                                # Add section header based on filename
                                section_name = (
                                    md_file.stem.replace('-', ' ').replace('_', ' ').title()
                                )
                                additional_content.append(
                                    f"\n\n## {subdir_name.title()}: {section_name}\n\n{file_content}"
                                )
                                logger.debug(
                                    f"  Added {subdir_name}/{md_file.name} to skill {skill.name}"
                                )
                            except Exception as e:
                                logger.warning(f"Failed to load {md_file}: {e}")

                # Append additional content to skill
                if additional_content:
                    skill.content += ''.join(additional_content)

                self.skills[skill.name] = skill
                logger.debug(f"Loaded skill: {skill.name}")
        except Exception as e:
            logger.warning(f"Failed to load skill from {filepath}: {e}")

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self.skills.get(name)

    def get_all_skills(self) -> List[Skill]:
        """Get all loaded skills."""
        return list(self.skills.values())

    def reload(self):
        """Reload skills from disk."""
        self.skills.clear()
        self._load_skills()


class SkillRegistry:
    """Registry of available skills for the agent.

    This is now a thin wrapper around SkillLoader that maintains
    backward compatibility with the previous implementation.
    """

    def __init__(self, pat: str = None, user_id: str = None, skills_dir: str = None):
        """Initialize the skill registry.

        Args:
            pat: Personal Access Token (kept for backward compat, not used)
            user_id: Clarifai user ID (kept for backward compat, not used)
            skills_dir: Directory containing skill markdown files
        """
        self.pat = pat
        self.user_id = user_id
        self.loader = SkillLoader(skills_dir)

    @property
    def skills(self) -> Dict[str, Skill]:
        """Get all skills as a dict."""
        return self.loader.skills

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self.loader.get_skill(name)

    def get_all_skills(self) -> List[Skill]:
        """Get all registered skills."""
        return self.loader.get_all_skills()

    def get_skills_for_llm(self) -> List[Dict[str, Any]]:
        """Get skill definitions formatted for LLM."""
        return [skill.to_dict() for skill in self.get_all_skills()]

    def reload(self):
        """Reload skills from disk."""
        self.loader.reload()


def build_skills_system_prompt(registry: SkillRegistry) -> str:
    """Build a system prompt that includes all loaded skills.

    Args:
        registry: SkillRegistry instance with loaded skills

    Returns:
        System prompt string with skill routing information
    """
    skills = registry.get_all_skills()

    if not skills:
        return _get_fallback_system_prompt()

    # Build skill index for routing
    skill_index = "\n".join([f"- **{skill.name}**: {skill.description}" for skill in skills])

    # Include full skill content
    skill_docs = "\n\n---\n\n".join(
        [f"# SKILL: {skill.name}\n\n{skill.content}" for skill in skills]
    )

    return f"""You are a helpful Clarifai CLI assistant with access to specialized skills.

## CREDENTIALS

The user is already authenticated. The following credentials are available from environment variables:
- **user_id**: Available as CLARIFAI_USER_ID - use directly in commands without placeholders
- **PAT**: Available as CLARIFAI_PAT - automatically used by CLI, never include in commands

When writing commands, use the actual user_id if you know it (from previous commands or user input).
If you must use a placeholder, use `YOUR_USER_ID` and it will be auto-substituted with the actual value.
Never use placeholders like `YOUR_PAT` - the PAT is handled automatically by the CLI.

## SDK ACTIONS (Preferred for SDK Operations)

For SDK operations (list apps, delete apps, create datasets, etc.), use structured JSON actions in a ```json code block.
See the "Clarifai Python SDK" skill documentation below for the full list of available actions and examples.

## CLI COMMANDS

For CLI-specific operations, use ```bash code blocks:
```bash
clarifai model list YOUR_USER_ID
```

**Safe CLI commands** (auto-executed):
- `clarifai model list YOUR_USER_ID` - List user's models
- `clarifai pipeline list` - List pipelines
- `clarifai pipelinetemplate list` - List templates
- `clarifai deployment list` - List deployments
- `clarifai config show` - Show config
- Any `--help` commands

**Interactive commands** (user must run in terminal):
- `clarifai login` - Requires interactive input
- `clarifai model init` - Initialize a model project
- `clarifai model upload` - Upload a model

## SKILL ROUTING

When a user makes a request, determine which skill applies based on their descriptions:

{skill_index}

## DECISION LOGIC

1. **Match request to skill(s)**: Read the skill descriptions to find the best match
2. **Use skill guidance**: Follow the patterns, examples, and recommendations in the matched skill
3. **For questions**: Answer directly using skill knowledge without executing commands
4. **For SDK operations**: Use a JSON action in a ```json code block (see SDK skill for examples)
5. **For CLI commands**: Include the command in a ```bash code block
6. **For multi-step tasks**: Execute one step at a time, wait for results

## RESPONSE FORMAT

- Keep responses concise and actionable
- For SDK operations, use a ```json action block
- For CLI operations, use a ```bash code block
- For questions, answer directly from skill knowledge

---

## SKILL DOCUMENTATION

{skill_docs}

---

Remember: Use the skill documentation above to provide accurate, detailed guidance. 
If no skill matches, answer based on general Clarifai knowledge or ask for clarification."""


def _get_fallback_system_prompt() -> str:
    """Get fallback system prompt when no skills are loaded."""
    return """You are a helpful Clarifai CLI assistant.

## CREDENTIALS

The user is already authenticated. The following credentials are available from environment variables:
- **user_id**: Available as CLARIFAI_USER_ID - use directly in commands without placeholders
- **PAT**: Available as CLARIFAI_PAT - automatically used by CLI, never include in commands

## SDK ACTIONS

For SDK operations (list apps, delete apps, etc.), use JSON actions in a ```json code block:

```json
{"action": "list_apps"}
```

```json
{"action": "delete_app", "app_id": "my-app"}
```

Available actions: list_apps, list_models, list_datasets, list_workflows, list_concepts, list_pipelines, list_compute_clusters, list_runners, get_user_info, create_app, delete_app, create_dataset, delete_model, delete_dataset, delete_workflow

## CLI COMMANDS

Use ```bash blocks for CLI commands:
```bash
clarifai model list YOUR_USER_ID
```

**Safe CLI commands** (auto-executed):
- `clarifai model list YOUR_USER_ID` - List user's models
- `clarifai pipeline list` - List pipelines  
- `clarifai deployment list` - List deployments
- `clarifai config show` - Show config

## CAPABILITIES

I can help you with:
- SDK operations via JSON actions (list apps, delete apps, create datasets, etc.)
- CLI commands (model, pipeline, deployment, artifact, etc.)
- Understanding Clarifai concepts and workflows
- Troubleshooting issues

## RESPONSE FORMAT

- For SDK operations, use a ```json action block
- For CLI operations, use a ```bash code block
- For questions, answer directly"""
