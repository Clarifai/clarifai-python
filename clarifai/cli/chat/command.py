"""Chat command for Clarifai CLI."""

import os
import re
import sys

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme

import clarifai
from clarifai.cli.chat.actions import (
    parse_action_from_response,
    execute_action,
    get_action,
)
from clarifai.cli.chat.agent import (
    ClarifaiAgent,
    build_agent_system_prompt,
)
from clarifai.cli.base import cli
from clarifai.cli.chat.executor import (
    execute_command,
    execute_python,
    parse_commands_from_response,
    is_safe_command,
    is_safe_python_code,
    format_command_output,
    set_current_user_id,
)
from clarifai.cli.chat.rag import ClarifaiCodeRAG
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
        r"(['\"`\u201c\u201d\u2018\u2019]pat['\"`\u201c\u201d\u2018\u2019]\s*:\s*['\"`\u201c\u201d\u2018\u2019])[a-f0-9*]{32}(['\"`\u201c\u201d\u2018\u2019])",
        r"\1" + "*" * 32 + r"\2",
        text,
        flags=re.IGNORECASE,
    )

    # 3. Sanitize token values
    text = re.sub(
        r"(['\"`\u201c\u201d\u2018\u2019]token['\"`\u201c\u201d\u2018\u2019]\s*:\s*['\"`\u201c\u201d\u2018\u2019])[a-f0-9*]{32}(['\"`\u201c\u201d\u2018\u2019])",
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

    # Set the user_id for command substitution (so placeholders like YOUR_USER_ID get replaced)
    if current_context.user_id:
        set_current_user_id(current_context.user_id)

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
            console.print(f"[{SUCCESS}][OK][/{SUCCESS}] Agent ready for command execution")
        except Exception as e:
            click.secho(f"(Warning: Could not initialize agent: {e})", fg=WARNING)
            agent = None

        # Initialize RAG system for CLI documentation
        try:
            # Find the clarifai-python root
            clarifai_root = os.path.dirname(os.path.dirname(clarifai.__file__))
            rag = ClarifaiCodeRAG(clarifai_root)
            console.print(f"[{SUCCESS}][OK][/{SUCCESS}] Knowledge base loaded")
        except Exception as e:
            click.secho(f"(Warning: Could not load knowledge base: {e})", fg=WARNING)
            rag = None

        # Platform-specific EOF shortcut
        eof_shortcut = "Ctrl+Z" if sys.platform == "win32" else "Ctrl+D"

        console.print()
        console.print(f"[{MUTED}]-----------------------------------------------[/{MUTED}]")
        console.print(
            f"[bold {BRAND}]Quick Commands:[/bold {BRAND}] [{BRAND}]help[/{BRAND}] | [{BRAND}]history[/{BRAND}] | [{BRAND}]clear[/{BRAND}] | [{BRAND}]exit[/{BRAND}] | [{MUTED}]{eof_shortcut} to quit[/{MUTED}]"
        )
        console.print(f"[{MUTED}]-----------------------------------------------[/{MUTED}]\n")

        # Interactive chat loop
        conversation_history = []

        while True:
            try:
                # Get user input
                user_input = click.prompt(
                    click.style("You", fg='cyan'), show_default=False, default=''
                )

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
                    console.print(f"[{MUTED}]-----------------------------------------------[/{MUTED}]")
                    console.print(f"[{SUCCESS}][OK][/{SUCCESS}] Screen and conversation history cleared.\n")
                    continue

                # Add to conversation history
                conversation_history.append({'role': 'user', 'message': user_input})

                # click.secho("Thinking...", fg='yellow')

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
                # Combine skills (for action guidance) AND RAG (for codebase context)
                system_prompt_parts = []
                
                # Add skills guidance if agent is available
                if agent:
                    system_prompt_parts.append(build_agent_system_prompt(agent))
                
                # Add RAG context for codebase questions
                if rag:
                    search_results = rag.search(user_input, top_k=3)
                    if search_results:
                        rag_context = "\n## Relevant Code References:\n"
                        for file_path, snippet in search_results:
                            rag_context += f"\n### From {file_path}:\n```\n{snippet}\n```"
                        system_prompt_parts.append(rag_context)
                
                # Combine or use fallback
                if system_prompt_parts:
                    system_prompt = "\n\n".join(system_prompt_parts)
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

                            # Remove special model control tokens that shouldn't be displayed
                            assistant_message = re.sub(r'<\|[a-z_]+\|>', '', assistant_message)
                            assistant_message = re.sub(r'</?[a-z_]+>', '', assistant_message)
                            
                            # Remove any skill/tool call tags (skills are now guidance, not executable)
                            assistant_message = re.sub(r'<skill_call>.*?</skill_call>', '', assistant_message, flags=re.DOTALL)
                            assistant_message = re.sub(r'<tool_call>.*?</tool_call>', '', assistant_message, flags=re.DOTALL)

                            # Sanitize sensitive data (PAT, tokens, etc.)
                            assistant_message = sanitize_sensitive_data(assistant_message)

                            # Clean up any extra whitespace left behind
                            assistant_message = re.sub(r'\n\s*\n', '\n', assistant_message).strip()

                            # Add to conversation history
                            conversation_history.append(
                                {'role': 'assistant', 'message': assistant_message}
                            )

                            # Display response with rich markdown rendering
                            console.print(
                                Markdown(f"**Assistant:**\n\n{assistant_message}"),
                                style='green',
                            )
                            
                            # Parse and execute commands from the response
                            # First try to parse JSON actions (preferred method)
                            action_data = parse_action_from_response(assistant_message)
                            
                            if action_data:
                                action_name = action_data.get('action')
                                action_params = {k: v for k, v in action_data.items() if k != 'action'}
                                action_def = get_action(action_name)
                                
                                if action_def:
                                    console.print(f"\n[{MUTED}]--- SDK Action ---[/{MUTED}]")
                                    console.print(f"[{BRAND}]Action:[/{BRAND}] {action_name}")
                                    if action_params:
                                        console.print(f"[{MUTED}]Params: {action_params}[/{MUTED}]")
                                    
                                    if action_def.needs_confirmation:
                                        if click.confirm(f"Execute {action_name}?", default=False):
                                            result = execute_action(action_name, action_params)
                                            if result.success:
                                                console.print(f"[{SUCCESS}][OK][/{SUCCESS}] {result.message}")
                                            else:
                                                console.print(f"[{ERROR}][FAIL][/{ERROR}] {result.error}")
                                            conversation_history.append({
                                                'role': 'system',
                                                'message': f"Action {action_name} result: {result.message or result.error}"
                                            })
                                        else:
                                            console.print(f"[{WARNING}]Skipped[/{WARNING}]")
                                    else:
                                        result = execute_action(action_name, action_params)
                                        if result.success:
                                            console.print(f"[{SUCCESS}][OK][/{SUCCESS}] {result.message}")
                                        else:
                                            console.print(f"[{ERROR}][FAIL][/{ERROR}] {result.error}")
                                        conversation_history.append({
                                            'role': 'system',
                                            'message': f"Action {action_name} result: {result.message or result.error}"
                                        })
                                    console.print(f"[{MUTED}]------------------[/{MUTED}]")
                                else:
                                    console.print(f"[{WARNING}]Unknown action: {action_name}[/{WARNING}]")
                            
                            # Also check for CLI commands (bash blocks)
                            commands, skipped = parse_commands_from_response(assistant_message)
                            
                            if skipped:
                                console.print(f"\n[{MUTED}]--- Skipped Commands (need your input) ---[/{MUTED}]")
                                for cmd, reason in skipped:
                                    console.print(f"[{WARNING}]! Skipped:[/{WARNING}] `{cmd[:60]}...`" if len(cmd) > 60 else f"[{WARNING}]! Skipped:[/{WARNING}] `{cmd}`")
                                    console.print(f"  [{MUTED}]Reason: {reason} - please replace with actual values[/{MUTED}]")
                                console.print(f"[{MUTED}]-----------------------------------------[/{MUTED}]")
                            
                            if commands:
                                console.print(f"\n[{MUTED}]--- Command Execution ---[/{MUTED}]")
                                for cmd, substitutions, cmd_type in commands:
                                    if substitutions:
                                        console.print(f"[{MUTED}]Auto-substituted: {', '.join(substitutions)}[/{MUTED}]")
                                    
                                    if cmd_type == 'python':
                                        # Python SDK code
                                        if is_safe_python_code(cmd):
                                            console.print(f"[{BRAND}]Running Python (SDK):[/{BRAND}]")
                                            console.print(f"[{MUTED}]{cmd[:100]}...[/{MUTED}]" if len(cmd) > 100 else f"[{MUTED}]{cmd}[/{MUTED}]")
                                            result = execute_python(cmd)
                                            console.print(Markdown(format_command_output(result)))
                                            conversation_history.append({
                                                'role': 'system',
                                                'message': f"Python code returned:\n{result.output[:500]}"
                                            })
                                        else:
                                            # Ask for confirmation for non-safe Python code
                                            console.print(f"[{WARNING}]Python code to execute:[/{WARNING}]")
                                            console.print(f"[{MUTED}]{cmd[:200]}...[/{MUTED}]" if len(cmd) > 200 else f"[{MUTED}]{cmd}[/{MUTED}]")
                                            if click.confirm("Execute this Python code?", default=False):
                                                result = execute_python(cmd)
                                                console.print(Markdown(format_command_output(result)))
                                                conversation_history.append({
                                                    'role': 'system', 
                                                    'message': f"Python code returned:\n{result.output[:500]}"
                                                })
                                            else:
                                                console.print(f"[{WARNING}]Skipped[/{WARNING}]")
                                    else:
                                        # CLI command
                                        if is_safe_command(cmd):
                                            # Execute safe commands automatically
                                            console.print(f"[{BRAND}]Running:[/{BRAND}] {cmd}")
                                            result = execute_command(cmd)
                                            console.print(Markdown(format_command_output(result)))
                                            # Add result to conversation history for context
                                            conversation_history.append({
                                                'role': 'system',
                                                'message': f"Command `{cmd}` returned:\n{result.output[:500]}"
                                            })
                                        else:
                                            # Ask for confirmation for non-safe commands
                                            if click.confirm(f"Execute: {cmd}?", default=False):
                                                console.print(f"[{BRAND}]Running:[/{BRAND}] {cmd}")
                                                result = execute_command(cmd)
                                                console.print(Markdown(format_command_output(result)))
                                                conversation_history.append({
                                                    'role': 'system', 
                                                    'message': f"Command `{cmd}` returned:\n{result.output[:500]}"
                                                })
                                            else:
                                                console.print(f"[{WARNING}]Skipped:[/{WARNING}] {cmd}")
                                console.print(f"[{MUTED}]--------------------------[/{MUTED}]")
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
