"""Chat command for Clarifai CLI."""

import os
import sys

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme

import clarifai
from clarifai.cli.agent import (
    ClarifaiAgent,
    build_agent_system_prompt,
    parse_tool_calls_from_response,
)
from clarifai.cli.base import cli
from clarifai.cli.rag import ClarifaiCodeRAG, build_system_prompt_with_rag
from clarifai.client.model import Model
from clarifai.utils.cli import validate_context
from clarifai.utils.logging import logger

# Default model URL for GPT-OSS-120B chat completion
DEFAULT_CHAT_MODEL_URL = "https://clarifai.com/openai/chat-completion/models/gpt-oss-120b"

# Color constants (Clarifai brand colors from clarifai.com)
BRAND = '#04AFFF'      # Primary Clarifai cyan
SUCCESS = 'green'      # Success states
ERROR = 'red'          # Errors
WARNING = 'yellow'     # Warnings
MUTED = 'dim'          # Muted/subtle text

# Custom theme for Rich console - uses brand colors for code/links
custom_theme = Theme({
    "markdown.code": BRAND,              # Inline code: `code`
    "markdown.code_block": BRAND,        # Code blocks: ```code```
    "markdown.link": BRAND,              # Links: [text](url)
    "markdown.link_url": f"dim {BRAND}", # URL part of links (subdued)
    "markdown.item.bullet": BRAND,       # Bullet list markers (-, *)
    "markdown.item.number": BRAND,       # Numbered list markers (1., 2., etc.)
})

# Rich console for formatted output with custom theme
console = Console(theme=custom_theme)


def get_cli_command_for_tool(tool_name: str, params: dict) -> str:
    """Convert an agent tool call to the equivalent CLI command.

    Args:
        tool_name: Name of the agent tool (e.g., 'user_delete_app')
        params: Parameters passed to the tool

    Returns:
        Equivalent CLI command string, or None if no mapping exists
    """
    # Map tool names to CLI commands
    tool_to_cli = {
        # User operations
        'user_list_apps': 'clarifai app ls',
        'user_create_app': 'clarifai app create {app_id}',
        'user_delete_app': 'clarifai app delete {app_id}',
        # App operations
        'app_list_models': 'clarifai model ls --app_id {app_id}',
        'app_list_datasets': 'clarifai dataset ls --app_id {app_id}',
        'app_list_workflows': 'clarifai workflow ls --app_id {app_id}',
        'app_create_model': 'clarifai model create {model_id} --app_id {app_id}',
        'app_create_dataset': 'clarifai dataset create {dataset_id} --app_id {app_id}',
        'app_delete_model': 'clarifai model delete {model_id} --app_id {app_id}',
        'app_delete_dataset': 'clarifai dataset delete {dataset_id} --app_id {app_id}',
        # Model operations
        'model_predict': 'clarifai model predict --model_id {model_id} --app_id {app_id}',
        'model_list_versions': 'clarifai model version ls --model_id {model_id} --app_id {app_id}',
        # Dataset operations
        'dataset_list_inputs': 'clarifai dataset inputs ls --dataset_id {dataset_id} --app_id {app_id}',
        'dataset_upload_from_url': 'clarifai dataset inputs upload --dataset_id {dataset_id} --app_id {app_id} --url {url}',
        # Inputs operations
        'inputs_upload_from_url': 'clarifai inputs upload --app_id {app_id} --url {url}',
        'inputs_upload_from_file': 'clarifai inputs upload --app_id {app_id} --file {file_path}',
        'inputs_list': 'clarifai inputs ls --app_id {app_id}',
        # Pipeline operations
        'user_list_pipelines': 'clarifai pipeline ls --app_id {app_id}',
        'pipeline_list_versions': 'clarifai pipeline version ls --pipeline_id {pipeline_id} --app_id {app_id}',
    }

    template = tool_to_cli.get(tool_name)
    if not template:
        return None

    # Substitute params into template
    try:
        cmd = template.format(**params)
        return cmd
    except KeyError:
        # If some params are missing, return template with available params
        for key, value in params.items():
            template = template.replace('{' + key + '}', str(value))
        return template


def normalize_response_with_tools(response_text: str, agent: ClarifaiAgent = None) -> str:
    """Normalize model response to prioritize tool calls and strip excessive explanation.

    If the response contains tool calls, move them to the front and remove redundant explanation.
    Ensures tool calls are formatted correctly.

    Args:
        response_text: Raw model response
        agent: ClarifaiAgent instance to check for valid tool names

    Returns:
        Normalized response with tool calls prioritized
    """
    import re

    # Extract all tool calls from the response
    tool_call_pattern = r'<tool_call>\s*(\{[^}]*"tool"[^}]*\})\s*</tool_call>'
    tool_calls = re.findall(tool_call_pattern, response_text)

    if not tool_calls:
        return response_text

    # Remove tool calls from the original response
    response_without_tools = re.sub(tool_call_pattern, '', response_text)
    response_without_tools = response_without_tools.strip()

    # Remove excessive explanation before tool calls
    # If the explanation is more than 2 lines before tools, truncate it
    explanation_lines = response_without_tools.split('\n')
    if len(explanation_lines) > 2:
        # Keep only brief explanation (first 1-2 lines)
        brief_explanation = '\n'.join(explanation_lines[:1])
    else:
        brief_explanation = response_without_tools

    # Reconstruct: tool calls first, then brief explanation
    normalized = ""
    for tool_call in tool_calls:
        normalized += f"<tool_call>{tool_call}</tool_call>\n"

    if brief_explanation and brief_explanation.strip():
        normalized += brief_explanation.strip()

    return normalized.strip()


def sanitize_sensitive_data(text: str) -> str:
    """Sanitize sensitive credentials from response text.

    Masks PAT (Personal Access Token), auth tokens, and other sensitive data
    by replacing them with asterisks.

    Args:
        text: Text to sanitize

    Returns:
        Text with sensitive data masked
    """
    import re

    # 1. Sanitize any 32-character hex string that looks like a PAT/token
    #    This is the most aggressive pattern - catch bare hex values
    text = re.sub(r'\b[a-f0-9]{32}\b', lambda m: '*' * 32, text, flags=re.IGNORECASE)

    # 2. Sanitize PAT values in key:value format (various quote types)
    # Handle both ASCII and Unicode quotes
    text = re.sub(
        r"(['\"`" "'']pat['\"`" "'']\\s*:\\s*['\"`" "''])[a-f0-9*]{32}(['\"`" "''])",
        r"\1" + "*" * 32 + r"\2",
        text,
        flags=re.IGNORECASE,
    )

    # 3. Sanitize token values
    text = re.sub(
        r"(['\"`" "'']token['\"`" "'']\\s*:\\s*['\"`" "''])[a-f0-9*]{32}(['\"`" "''])",
        r"\1" + "*" * 32 + r"\2",
        text,
        flags=re.IGNORECASE,
    )

    # 4. Sanitize after keyword patterns
    # e.g., "PAT: 6b09c6dc2a694266921a3f62d25c9197" or "Personal Access Token: 6b09c6dc..."
    text = re.sub(
        r'((?:PAT|Token|API[_-]?Key|Secret|Authorization|Auth|Credential|password)\s*[:=]\s*)[a-f0-9*]{32}',
        r"\1" + "*" * 32,
        text,
        flags=re.IGNORECASE,
    )

    # 5. Sanitize environment variable format
    text = re.sub(
        r'(CLARIFAI_PAT\s*=\s*)[a-f0-9*]{32}', r"\1" + "*" * 32, text, flags=re.IGNORECASE
    )

    # 6. Sanitize in parentheses or brackets
    text = re.sub(
        r'([\(\[\{]\s*)[a-f0-9]{32}([\)\]\}])',
        lambda m: m.group(1) + "*" * 32 + m.group(2),
        text,
        flags=re.IGNORECASE,
    )

    # 7. Ensure already-partially masked values don't create issues
    # Convert patterns like "6b09c6dc*" back to full mask
    text = re.sub(r'[a-f0-9]{16,}\*+', "*" * 32, text, flags=re.IGNORECASE)

    return text


@cli.command()
@click.pass_context
def chat(ctx):
    """Start an interactive session using the Clarifai CLI assistant.

    You must be logged in first. Uses the current context's credentials.

    Example:
        $ clarifai login
        $ clarifai chat
    """
    # Validate that user is authenticated
    validate_context(ctx)

    # Get the current context
    current_context = ctx.obj.current

    if current_context.name == '_empty_' or not current_context.pat:
        click.secho("Error: No valid authentication found. Please login first.", fg=ERROR)
        click.echo("Run: clarifai login")
        sys.exit(1)

    # Get chat model URL from config or use default
    chat_model_url = current_context.get('chat_model_url', DEFAULT_CHAT_MODEL_URL)

    # Try to initialize the model
    model_available = True
    try:
        console.print(f"[bold {BRAND}]Clarifai CLI Assistant[/bold {BRAND}]")
        console.print(f"[{MUTED}]Your AI-powered guide to Clarifai[/{MUTED}]\n")

        # Initialize model with the current context's PAT
        with console.status(f"[{BRAND}]Connecting to AI model...[/{BRAND}]", spinner="dots"):
            model = Model(
                url=chat_model_url,
                pat=current_context.pat,
            )

        # Initialize the agent for tool calling
        try:
            agent = ClarifaiAgent(
                pat=current_context.pat,
                user_id=current_context.user_id,
            )
            console.print(f"[{SUCCESS}]✓[/{SUCCESS}] Agent ready for command execution")
        except Exception as e:
            click.secho(f"(Warning: Could not initialize agent: {e})", fg=WARNING)
            agent = None

        # Initialize RAG system for CLI documentation
        try:
            # Find the clarifai-python root
            clarifai_root = os.path.dirname(os.path.dirname(clarifai.__file__))
            rag = ClarifaiCodeRAG(clarifai_root)
            console.print(f"[{SUCCESS}]✓[/{SUCCESS}] Knowledge base loaded")
        except Exception as e:
            click.secho(f"(Warning: Could not load knowledge base: {e})", fg=WARNING)
            rag = None

        # Platform-specific EOF shortcut
        eof_shortcut = "Ctrl+Z" if sys.platform == "win32" else "Ctrl+D"

        console.print()
        console.print(f"[{MUTED}]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/{MUTED}]")
        console.print(
            f"[bold {BRAND}]Quick Commands:[/bold {BRAND}] [{BRAND}]help[/{BRAND}] | [{BRAND}]history[/{BRAND}] | [{BRAND}]clear[/{BRAND}] | [{BRAND}]exit[/{BRAND}] | [{MUTED}]{eof_shortcut} to quit[/{MUTED}]"
        )
        console.print(f"[{MUTED}]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/{MUTED}]\n")

        # Interactive chat loop
        conversation_history = []

        while True:
            try:
                # Get user input with brand color (cursor appears after prompt)
                console.print(f"[bold {BRAND}]>[/bold {BRAND}] ", end="")
                try:
                    user_input = input().strip()
                    # Add blank line after user input for better readability
                    if user_input:
                        print()
                except EOFError:
                    click.echo()
                    click.secho("Goodbye!", fg=WARNING)
                    break

                if not user_input:
                    continue

                # Check for special commands
                if user_input.lower() in ('exit', 'quit', 'bye'):
                    click.secho("Goodbye!", fg=WARNING)
                    break

                if user_input.lower() in ('help', '?'):
                    click.echo(
                        "\nChat Commands:\n"
                        "  exit, quit, bye  - Exit the chat\n"
                        "  history, hist    - Show conversation history\n"
                        "  clear            - Clear conversation history\n"
                        "  help, ?          - Show this help message\n"
                    )
                    continue

                if user_input.lower() in ('history', 'hist'):
                    if not conversation_history:
                        click.echo("No conversation history yet.\n")
                    else:
                        click.echo("\nConversation History:")
                        for i, msg in enumerate(conversation_history, 1):
                            role = "You" if msg['role'] == 'user' else "Assistant"
                            click.echo(f"\n{i}. {role}:")
                            click.echo(
                                f"   {msg['message'][:300]}..."
                                if len(msg['message']) > 300
                                else f"   {msg['message']}"
                            )
                        click.echo()
                    continue

                if user_input.lower() == 'clear':
                    conversation_history = []
                    # Clear the terminal screen
                    click.clear()
                    # Reprint a minimal header
                    console.print(f"[bold {BRAND}]Clarifai CLI Assistant[/bold {BRAND}]")
                    console.print(f"[{MUTED}]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/{MUTED}]")
                    console.print(f"[{SUCCESS}]✓[/{SUCCESS}] Screen and conversation history cleared.\n")
                    continue

                # Add to conversation history
                conversation_history.append({'role': 'user', 'message': user_input})

                # click.secho("Thinking...", fg=WARNING)

                # Build conversation context for follow-up questions
                conversation_context = ""
                if len(conversation_history) > 1:
                    conversation_context = "\n## Conversation History:\n"
                    # Include all previous messages (not just last 4) for full context
                    for msg in conversation_history[:-1]:  # Exclude the current user message
                        role = "User" if msg['role'] == 'user' else "Assistant"
                        # Keep full messages, only truncate very long ones at 500 chars
                        msg_text = msg['message']
                        if len(msg_text) > 500:
                            msg_text = msg_text[:500] + "..."
                        conversation_context += f"{role}: {msg_text}\n\n"

                # Build enhanced prompt with RAG context and agent tools
                if agent:
                    # Use agent system prompt as the primary prompt (includes tool definitions)
                    system_prompt = build_agent_system_prompt(agent)
                # Use RAG or fallback system prompt
                elif rag:
                    system_prompt = build_system_prompt_with_rag(rag, user_input)
                else:
                    system_prompt = """You are an expert Clarifai CLI assistant. Your role is to help users with:
1. Using the Clarifai CLI commands (login, chat, config, deployment, pipeline, model, artifact, etc.)
2. Understanding CLI options, parameters, and flags
3. Troubleshooting CLI issues and errors
4. Writing CLI scripts and automation
5. Understanding CLI integration with Clarifai resources
6. Answering meta-questions about our conversation

RESPONSE RULES:
- Keep responses CONCISE and FOCUSED (max 300 words)
- Use bullet points, tables, or code examples when helpful
- Answer ALL CLI-related questions directly
- For meta-questions about our conversation, reference the conversation history
- For general Clarifai API questions NOT related to CLI, refer to: https://docs.clarifai.com"""

                # Create enhanced input with system prompt, history, and question
                enhanced_input = f"{system_prompt}{conversation_context}\n\nCurrent User Question: {user_input}\n\nRespond concisely (max 300 words)."

                # Send to model with inference parameters to limit response
                try:
                    response = model.predict_by_bytes(
                        input_bytes=enhanced_input.encode('utf-8'),
                        input_type='text',
                        inference_params={
                            'max_tokens': '500'  # Limit output tokens for concise responses
                        },
                    )

                    # Extract the text response
                    if response and hasattr(response, 'outputs') and len(response.outputs) > 0:
                        output = response.outputs[0]
                        if hasattr(output, 'data') and hasattr(output.data, 'text'):
                            assistant_message = output.data.text.raw

                            # Normalize response to prioritize tool calls and reduce explanation
                            if agent:
                                assistant_message = normalize_response_with_tools(
                                    assistant_message, agent
                                )

                            # Remove special model control tokens that shouldn't be displayed
                            import re

                            assistant_message = re.sub(r'<\|[a-z_]+\|>', '', assistant_message)
                            assistant_message = re.sub(r'</?[a-z_]+>', '', assistant_message)

                            # Sanitize sensitive data (PAT, tokens, etc.)
                            assistant_message = sanitize_sensitive_data(assistant_message)

                            # Remove bare JSON tool calls from the displayed message (they'll be parsed separately)
                            assistant_message = re.sub(
                                r'\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"params"\s*:\s*\{[^}]*\}\s*\}',
                                '',
                                assistant_message,
                            )
                            # Clean up any extra whitespace left behind
                            assistant_message = re.sub(r'\n\s*\n', '\n', assistant_message).strip()

                            # Check for tool calls in the response
                            tool_calls = []
                            if agent:
                                tool_calls = parse_tool_calls_from_response(
                                    output.data.text.raw
                                )  # Parse from original before cleaning

                            # Execute any tool calls
                            tool_results = {}
                            if tool_calls:
                                console.print(f"[{BRAND}]Executing actions...[/{BRAND}]\n")
                                for tool_call in tool_calls:
                                    tool_name = tool_call.get('tool')
                                    params = tool_call.get('params', {})

                                    # Show the equivalent CLI command if available
                                    cli_cmd = get_cli_command_for_tool(tool_name, params)
                                    if cli_cmd:
                                        console.print(
                                            f"[{MUTED}]CLI equivalent:[/{MUTED}] [{BRAND}]{cli_cmd}[/{BRAND}]"
                                        )
                                    else:
                                        # Fallback: show the API operation
                                        console.print(
                                            f"[{MUTED}]API operation:[/{MUTED}] [{BRAND}]{tool_name}({', '.join(f'{k}={v}' for k, v in params.items())})[/{BRAND}]"
                                        )

                                    # Execute the tool
                                    result = agent.execute_tool(tool_name, params)
                                    tool_results[tool_name] = result

                                    if result.get('success'):
                                        console.print(f"[{SUCCESS}]  ✓ Done[/{SUCCESS}]")
                                    else:
                                        console.print(
                                            f"[{ERROR}]  ✗ Error: {result.get('error')}[/{ERROR}]"
                                        )
                                click.echo()  # Spacing

                                # If tools were executed, get a follow-up response from the model
                                if tool_results:
                                    tool_summary = "\n".join(
                                        [
                                            f"- {name}: {'Success' if r.get('success') else 'Error'} - {r.get('result') or r.get('error')}"
                                            for name, r in tool_results.items()
                                        ]
                                    )
                                    follow_up_input = f"Tool execution results:\n{tool_summary}\n\nPlease provide a summary of what was accomplished."

                                    try:
                                        follow_up_response = model.predict_by_bytes(
                                            input_bytes=follow_up_input.encode('utf-8'),
                                            input_type='text',
                                            inference_params={'max_tokens': '300'},
                                        )
                                        if (
                                            follow_up_response
                                            and hasattr(follow_up_response, 'outputs')
                                            and len(follow_up_response.outputs) > 0
                                        ):
                                            follow_up_output = follow_up_response.outputs[0]
                                            if hasattr(follow_up_output, 'data') and hasattr(
                                                follow_up_output.data, 'text'
                                            ):
                                                # Use the follow-up response as the final assistant message
                                                assistant_message = follow_up_output.data.text.raw
                                                # Remove special model control tokens
                                                import re

                                                assistant_message = re.sub(
                                                    r'<\|[a-z_]+\|>', '', assistant_message
                                                )
                                                assistant_message = re.sub(
                                                    r'</?[a-z_]+>', '', assistant_message
                                                )
                                                # Sanitize sensitive data
                                                assistant_message = sanitize_sensitive_data(
                                                    assistant_message
                                                )
                                    except Exception as e:
                                        logger.warning(f"Failed to get follow-up response: {e}")

                            # Add to conversation history
                            conversation_history.append(
                                {'role': 'assistant', 'message': assistant_message}
                            )

                            # Display response with rich markdown rendering
                            console.print(Markdown(assistant_message))
                        else:
                            click.secho("No text response received", fg=ERROR)
                    else:
                        click.secho("Invalid response format", fg=ERROR)

                except Exception as e:
                    error_msg = str(e).lower()

                    # Check for common reasons assistant might be unavailable
                    if 'not found' in error_msg or 'invalid' in error_msg:
                        click.secho("Assistant is currently unavailable.", fg=WARNING)
                        click.echo("  Please check your credentials or account status.")
                    elif 'permission' in error_msg or 'unauthorized' in error_msg:
                        click.secho("Assistant is currently unavailable.", fg=WARNING)
                        click.echo("  Access denied. Please verify your account status.")
                    else:
                        click.secho("Assistant is temporarily unavailable.", fg=WARNING)
                        click.echo("  Please try again in a moment.")

                    logger.exception("Chat model prediction error")

                click.echo()  # Add spacing between exchanges

            except KeyboardInterrupt:
                click.echo()
                click.secho("Chat interrupted. Goodbye!", fg=WARNING)
                break
            except (EOFError, click.exceptions.Abort):
                click.echo()
                click.secho("Goodbye!", fg=WARNING)
                break
            except Exception as e:
                click.secho(f"Error: {str(e)}", fg=ERROR)
                logger.exception("Chat error")
    except ImportError as e:
        click.secho(f"Error: Failed to import Clarifai SDK. {str(e)}", fg=ERROR)
        sys.exit(1)
    except Exception as e:
        click.secho("Assistant is currently unavailable.\n", fg=WARNING)
        click.secho("This could be due to:", fg=WARNING)
        click.echo("  • Network connectivity issues")
        click.echo("  • API authentication problems\n")
        logger.exception("Chat initialization error")
        sys.exit(1)
