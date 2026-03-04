import os
import sys
from typing import Iterator, List

from openai import OpenAI

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.runners.utils.data_utils import Param
from clarifai.runners.utils.openai_convertor import build_openai_messages
from clarifai.utils.logging import logger

PYTHON_EXEC = sys.executable


def sglang_openai_server(checkpoints, **kwargs):
    """Start SGLang OpenAI-compatible server."""
    from clarifai.runners.utils.model_utils import (
        execute_shell_command,
        terminate_process,
        wait_for_server,
    )

    cmds = [
        PYTHON_EXEC,
        '-m',
        'sglang.launch_server',
        '--model-path',
        checkpoints,
    ]
    for key, value in kwargs.items():
        if value is None:
            continue
        param_name = key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                cmds.append(f'--{param_name}')
        else:
            cmds.extend([f'--{param_name}', str(value)])

    server = type(
        'Server',
        (),
        {
            'host': kwargs.get('host', '0.0.0.0'),
            'port': kwargs.get('port', 23333),
            'process': None,
        },
    )()

    try:
        server.process = execute_shell_command(" ".join(cmds))
        url = f"http://{server.host}:{server.port}"
        logger.info(f"Waiting for SGLang server at {url}")
        wait_for_server(url)
        logger.info(f"SGLang server started at {url}")
    except Exception as e:
        logger.error(f"Failed to start SGLang server: {e}")
        if server.process:
            terminate_process(server.process)
        raise RuntimeError(f"Failed to start SGLang server: {e}")
    return server


class SGLangModel(OpenAIModelClass):
    client = True
    model = True

    def load_model(self):
        server_args = {
            'dtype': 'auto',
            'kv_cache_dtype': 'auto',
            'tp_size': 1,
            'context_length': None,
            'device': 'cuda',
            'port': 23333,
            'host': '0.0.0.0',
            'mem_fraction_static': 0.9,
        }

        model_path = os.path.dirname(os.path.dirname(__file__))
        builder = ModelBuilder(model_path, download_validation_only=True)
        config = builder.config
        stage = config["checkpoints"]["when"]
        checkpoints = config["checkpoints"]["repo_id"]
        if stage in ["build", "runtime"]:
            checkpoints = builder.download_checkpoints(stage=stage)

        self.server = sglang_openai_server(checkpoints, **server_args)
        self.client = OpenAI(
            api_key="notset",
            base_url=f"http://{self.server.host}:{self.server.port}/v1",
        )
        self.model = self.client.models.list().data[0].id

    @OpenAIModelClass.method
    def predict(
        self,
        prompt: str = "",
        chat_history: List[dict] = None,
        max_tokens: int = Param(
            default=512,
            description="The maximum number of tokens to generate.",
        ),
        temperature: float = Param(
            default=0.7,
            description="Sampling temperature (higher = more random).",
        ),
        top_p: float = Param(
            default=0.8,
            description="Nucleus sampling threshold.",
        ),
    ) -> str:
        """Return a single completion."""
        messages = build_openai_messages(prompt=prompt, messages=chat_history)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content

    @OpenAIModelClass.method
    def generate(
        self,
        prompt: str = "",
        chat_history: List[dict] = None,
        max_tokens: int = Param(
            default=512,
            description="The maximum number of tokens to generate.",
        ),
        temperature: float = Param(
            default=0.7,
            description="Sampling temperature (higher = more random).",
        ),
        top_p: float = Param(
            default=0.8,
            description="Nucleus sampling threshold.",
        ),
    ) -> Iterator[str]:
        """Stream a completion response."""
        messages = build_openai_messages(prompt=prompt, messages=chat_history)
        for chunk in self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        ):
            if chunk.choices:
                text = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ''
                yield text
