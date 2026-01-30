"""Chat command for Clarifai CLI."""

import os
import sys

import click

import clarifai
from clarifai.cli.base import cli
from clarifai.cli.rag import ClarifaiCodeRAG, build_system_prompt_with_rag
from clarifai.client.model import Model
from clarifai.utils.cli import validate_context
from clarifai.utils.logging import logger

# Default model URL for GPT-OSS-120B chat completion
DEFAULT_CHAT_MODEL_URL = "https://clarifai.com/openai/chat-completion/models/gpt-oss-120b"


@cli.command()
@click.pass_context
def chat(ctx):
    """Start an interactive chat session using the Clarifai chat model.

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
    try:
        click.secho("Initializing chat model...", fg='cyan')

        # Initialize model with the current context's PAT
        model = Model(
            url=chat_model_url,
            pat=current_context.pat,
        )

        click.secho("Connected! Type 'exit' or 'quit' to leave.\n", fg='green')

        # Initialize RAG system for CLI documentation
        try:
            # Find the clarifai-python root
            clarifai_root = os.path.dirname(os.path.dirname(clarifai.__file__))
            rag = ClarifaiCodeRAG(clarifai_root)
            click.secho("(CLI reference system ready)\n", fg='blue')
        except Exception as e:
            click.secho(f"(Warning: Could not load CLI reference system: {e})\n", fg='yellow')
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

                # Check for exit commands
                if user_input.lower() in ('exit', 'quit', 'bye'):
                    click.secho("Goodbye!", fg='yellow')
                    break

                # Add to conversation history
                conversation_history.append({'role': 'user', 'message': user_input})

                click.secho("Thinking...", fg='yellow')

                # Build conversation context for follow-up questions
                conversation_context = ""
                if len(conversation_history) > 2:  # Include last exchange if available
                    conversation_context = "\n## Previous Context:\n"
                    # Include last 2 exchanges (4 messages max) for context
                    for msg in conversation_history[-4:-1]:
                        role = "User" if msg['role'] == 'user' else "Assistant"
                        conversation_context += (
                            f"{role}: {msg['message'][:200]}\n"  # Limit to 200 chars per msg
                        )

                # Build enhanced prompt with RAG context
                system_prompt = None
                if rag:
                    system_prompt = build_system_prompt_with_rag(rag, user_input)
                else:
                    system_prompt = """You are a Clarifai CLI expert. Help users with CLI commands, options, and troubleshooting.
For non-CLI questions, direct users to: https://docs.clarifai.com

IMPORTANT: Keep responses CONCISE and FOCUSED. Use bullet points when appropriate. Maximum 300 words."""

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

                            # Add to conversation history
                            conversation_history.append(
                                {'role': 'assistant', 'message': assistant_message}
                            )

                            # Display response
                            click.secho(f"Assistant: {assistant_message}", fg='green')
                        else:
                            click.secho("No text response received", fg='red')
                    else:
                        click.secho("Invalid response format", fg='red')

                except Exception as e:
                    click.secho(f"Error calling model: {str(e)}", fg='red')
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
        click.secho(f"Error initializing chat model: {str(e)}", fg='red')
        logger.exception("Chat initialization error")
        sys.exit(1)
