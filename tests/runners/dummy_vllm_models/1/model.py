import os
import sys

from openai import OpenAI

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.utils.logging import logger

PYTHON_EXEC = sys.executable


def vllm_openai_server(checkpoints, **kwargs):
    """Start vLLM OpenAI compatible server."""

    from clarifai.runners.utils.model_utils import (
        execute_shell_command,
        terminate_process,
        wait_for_server,
    )

    # Start building the command
    cmds = [
        PYTHON_EXEC,
        '-m',
        'vllm.entrypoints.openai.api_server',
        '--model',
        checkpoints,
        '--enforce-eager',
    ]
    # Add all parameters from kwargs to the command
    for key, value in kwargs.items():
        if value is None:  # Skip None values
            continue
        param_name = key.replace('_', '-')
        if isinstance(value, bool):
            if value:  # Only add the flag if True
                cmds.append(f'--{param_name}')
        else:
            cmds.extend([f'--{param_name}', str(value)])
    # Create server instance
    server = type(
        'Server',
        (),
        {
            'host': kwargs.get('host', '0.0.0.0'),
            'port': kwargs.get('port', 23333),
            'backend': "vllm",
            'process': None,
        },
    )()

    try:
        server.process = execute_shell_command(" ".join(cmds))
        wait_for_server(f"http://{server.host}:{server.port}")
        logger.info("Server started successfully at " + f"http://{server.host}:{server.port}")
    except Exception as e:
        logger.error(f"Failed to start vllm server: {str(e)}")
        if server.process:
            terminate_process(server.process)
        raise RuntimeError(f"Failed to start vllm server: {str(e)}")

    return server


class VllmFacebookOpt125M(OpenAIModelClass):
    """
    A Model that integrates with the Clarifai platform and uses the vLLM OpenAI compatible server for inference.
    """

    client = True  # This will be set in load_model method
    model = True  # This will be set in load_model method

    def load_model(self):
        """Load the model here."""
        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        checkpoints = builder.config["checkpoints"]["repo_id"]
        self.server = vllm_openai_server(checkpoints)

        self.client = OpenAI(
            api_key="notset",
            base_url=f'http://{self.server.host}:{self.server.port}/v1',
        )
        self.model = self.client.models.list().data[0].id
        logger.info("vLLM OpenAI server loaded successfully.")

    @OpenAIModelClass.method
    def predict(self, prompt: str, temperature: float = 0.7, max_tokens: int = 25) -> str:
        """
        Predict method that uses the vLLM OpenAI compatible server to generate text based on the provided prompt.
        """
        return "Test response from vLLM OpenAI server."
