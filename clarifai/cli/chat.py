"""Chat command for Clarifai CLI."""

import os
import sys

import click
from rich.console import Console
from rich.markdown import Markdown

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

# Rich console for formatted output
console = Console()


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
    
    # Sanitize PAT values (32-character hex strings commonly used as PAT)
    # Pattern: 'pat': 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
    text = re.sub(
        r"('pat'\s*:\s*')[a-f0-9]{32}(')",
        r"\1" + "*" * 32 + r"\2",
        text,
        flags=re.IGNORECASE
    )
    
    # Also handle "pat=XXXXXXXX..." format
    text = re.sub(
        r"(pat\s*=\s*)[a-f0-9]{32}",
        r"\1" + "*" * 32,
        text,
        flags=re.IGNORECASE
    )
    
    # Sanitize token values (similar pattern)
    text = re.sub(
        r"('token'\s*:\s*')[a-f0-9]{32}(')",
        r"\1" + "*" * 32 + r"\2",
        text,
        flags=re.IGNORECASE
    )
    
    # Sanitize CLARIFAI_PAT environment variable format
    text = re.sub(
        r"(CLARIFAI_PAT\s*=\s*)[a-f0-9]{32}",
        r"\1" + "*" * 32,
        text,
        flags=re.IGNORECASE
    )
    
    # Sanitize any hex token-like values that appear in quotes (general case)
    # Match patterns like: "token": "abc123..." or 'key': 'abc123...'
    text = re.sub(
        r"(['\"](?:token|pat|password|secret|authorization|api[_-]key)['\"]?\s*:?\s*['\"])[a-f0-9A-F]{16,}(['\"])",
        lambda m: m.group(1) + "*" * 32 + m.group(2),
        text,
        flags=re.IGNORECASE
    )
    
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
        click.secho("Error: No valid authentication found. Please login first.", fg='red')
        click.echo("Run: clarifai login")
        sys.exit(1)

    # Get chat model URL from config or use default
    chat_model_url = current_context.get('chat_model_url', DEFAULT_CHAT_MODEL_URL)

    # Try to initialize the model
    model_available = True
    try:
        click.secho("Initializing assistant...", fg='cyan')

        # Initialize model with the current context's PAT
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
            click.secho("Agent ready for command execution.", fg='blue')
        except Exception as e:
            click.secho(f"(Warning: Could not initialize agent: {e})", fg='yellow')
            agent = None

        click.secho("Connected! Type 'exit' or 'quit' to leave.\n", fg='green')

        # Initialize RAG system for CLI documentation
        try:
            # Find the clarifai-python root
            clarifai_root = os.path.dirname(os.path.dirname(clarifai.__file__))
            rag = ClarifaiCodeRAG(clarifai_root)
            click.secho("(Knowledge base ready.)\n", fg='blue')
        except Exception as e:
            click.secho(f"(Warning: Could not load knowledge base: {e})\n", fg='yellow')
            rag = None

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
                    click.secho("Goodbye!", fg='yellow')
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
                            role = "You" if msg['role'] == 'user' else "🤖 Assistant"
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
                    click.secho("Conversation history cleared.\n", fg='green')
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
                if agent:
                    # Use agent system prompt as the primary prompt (includes tool definitions)
                    system_prompt = build_agent_system_prompt(agent)
                else:
                    # Use RAG or fallback system prompt
                    if rag:
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
                                assistant_message = normalize_response_with_tools(assistant_message, agent)
                            
                            # Remove special model control tokens that shouldn't be displayed
                            import re
                            assistant_message = re.sub(r'<\|[a-z_]+\|>', '', assistant_message)
                            assistant_message = re.sub(r'</?[a-z_]+>', '', assistant_message)
                            
                            # Sanitize sensitive data (PAT, tokens, etc.)
                            assistant_message = sanitize_sensitive_data(assistant_message)
                            
                            # Remove bare JSON tool calls from the displayed message (they'll be parsed separately)
                            assistant_message = re.sub(r'\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"params"\s*:\s*\{[^}]*\}\s*\}', '', assistant_message)
                            # Clean up any extra whitespace left behind
                            assistant_message = re.sub(r'\n\s*\n', '\n', assistant_message).strip()

                            # Check for tool calls in the response
                            tool_calls = []
                            if agent:
                                tool_calls = parse_tool_calls_from_response(output.data.text.raw)  # Parse from original before cleaning

                            # Execute any tool calls
                            tool_results = {}
                            if tool_calls:
                                click.secho("Executing tools...\n", fg='blue')
                                for tool_call in tool_calls:
                                    tool_name = tool_call.get('tool')
                                    params = tool_call.get('params', {})
                                    click.echo(
                                        f"→ Executing: {tool_name}({', '.join(f'{k}={v}' for k, v in params.items())})"
                                    )

                                    # Execute the tool
                                    result = agent.execute_tool(tool_name, params)
                                    tool_results[tool_name] = result

                                    if result.get('success'):
                                        click.secho(
                                            f"  ✓ {tool_name}: {result.get('result')}",
                                            fg='green',
                                        )
                                    else:
                                        click.secho(
                                            f"  ✗ {tool_name}: {result.get('error')}",
                                            fg='red',
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
                                                assistant_message = re.sub(r'<\|[a-z_]+\|>', '', assistant_message)
                                                assistant_message = re.sub(r'</?[a-z_]+>', '', assistant_message)
                                                # Sanitize sensitive data
                                                assistant_message = sanitize_sensitive_data(assistant_message)
                                    except Exception as e:
                                        logger.warning(f"Failed to get follow-up response: {e}")

                            # Add to conversation history
                            conversation_history.append(
                                {'role': 'assistant', 'message': assistant_message}
                            )

                            # Display response with rich markdown rendering
                            console.print(
                                Markdown(f"**Assistant:**\n\n{assistant_message}"),
                                style='green',
                            )
                        else:
                            click.secho("No text response received", fg='red')
                    else:
                        click.secho("Invalid response format", fg='red')

                except Exception as e:
                    error_msg = str(e).lower()

                    # Check for common reasons assistant might be unavailable
                    if 'not found' in error_msg or 'invalid' in error_msg:
                        click.secho("Assistant is currently unavailable.", fg='yellow')
                        click.echo("  Please check your credentials or account status.")
                    elif 'permission' in error_msg or 'unauthorized' in error_msg:
                        click.secho("Assistant is currently unavailable.", fg='yellow')
                        click.echo("  Access denied. Please verify your account status.")
                    else:
                        click.secho("Assistant is temporarily unavailable.", fg='yellow')
                        click.echo("  Please try again in a moment.")

                    logger.exception("Chat model prediction error")

                click.echo()  # Add spacing between exchanges

            except KeyboardInterrupt:
                click.echo()
                click.secho("Chat interrupted. Goodbye!", fg='yellow')
                break
            except Exception as e:
                click.secho(f"Error: {str(e)}", fg='red')
                logger.exception("Chat error")
    except ImportError as e:
        click.secho(f"Error: Failed to import Clarifai SDK. {str(e)}", fg='red')
        sys.exit(1)
    except Exception as e:
        click.secho("Assistant is currently unavailable.\n", fg='yellow')
        click.secho("This could be due to:", fg='yellow')
        click.echo("  • Network connectivity issues")
        click.echo("  • API authentication problems\n")
        logger.exception("Chat initialization error")
        sys.exit(1)
