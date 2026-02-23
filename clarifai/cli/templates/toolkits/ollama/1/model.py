import json
import os
import subprocess
from typing import Iterator, List

from openai import OpenAI

from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.runners.utils.data_types import Image
from clarifai.runners.utils.data_utils import Param
from clarifai.runners.utils.openai_convertor import build_openai_messages

if not os.environ.get('OLLAMA_HOST'):
    PORT = '23333'
    os.environ["OLLAMA_HOST"] = f'127.0.0.1:{PORT}'
OLLAMA_HOST = os.environ.get('OLLAMA_HOST')

if not os.environ.get('OLLAMA_CONTEXT_LENGTH'):
    os.environ["OLLAMA_CONTEXT_LENGTH"] = '8192'

VERBOSE_OLLAMA = False


def run_ollama_server(model_name: str = 'llama3.2'):
    """Start Ollama server and pull the model."""
    from clarifai.runners.utils.model_utils import execute_shell_command

    try:
        execute_shell_command(
            "ollama serve",
            stdout=None if VERBOSE_OLLAMA else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if VERBOSE_OLLAMA else subprocess.DEVNULL,
        )
        execute_shell_command(
            f"ollama pull {model_name}",
            stdout=None if VERBOSE_OLLAMA else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if VERBOSE_OLLAMA else subprocess.DEVNULL,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start Ollama server: {e}")


def has_image_content(image: Image) -> bool:
    return bool(getattr(image, 'url', None) or getattr(image, 'bytes', None))


class OllamaModel(OpenAIModelClass):
    client = True
    model = True

    def load_model(self):
        self.model = os.environ.get("OLLAMA_MODEL_NAME", 'llama3.2')
        run_ollama_server(model_name=self.model)
        self.client = OpenAI(api_key="notset", base_url=f"http://{OLLAMA_HOST}/v1")

    @OpenAIModelClass.method
    def predict(
        self,
        prompt: str = "",
        image: Image = None,
        images: List[Image] = None,
        chat_history: List[dict] = None,
        tools: List[dict] = None,
        tool_choice: str = None,
        max_tokens: int = Param(
            default=2048,
            description="The maximum number of tokens to generate.",
        ),
        temperature: float = Param(
            default=0.7,
            description="Sampling temperature (higher = more random).",
        ),
        top_p: float = Param(
            default=0.95,
            description="Nucleus sampling threshold.",
        ),
    ) -> str:
        """Return a single completion."""
        if tools is not None and tool_choice is None:
            tool_choice = "auto"

        img_content = image if has_image_content(image) else None
        messages = build_openai_messages(
            prompt=prompt, image=img_content, images=images, messages=chat_history
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        if response.usage is not None:
            self.set_output_context(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        if response.choices[0] and response.choices[0].message.tool_calls:
            tool_calls = response.choices[0].message.tool_calls
            return json.dumps([tc.to_dict() for tc in tool_calls], indent=2)
        return response.choices[0].message.content

    @OpenAIModelClass.method
    def generate(
        self,
        prompt: str = "",
        image: Image = None,
        images: List[Image] = None,
        chat_history: List[dict] = None,
        tools: List[dict] = None,
        tool_choice: str = None,
        max_tokens: int = Param(
            default=2048,
            description="The maximum number of tokens to generate.",
        ),
        temperature: float = Param(
            default=0.7,
            description="Sampling temperature (higher = more random).",
        ),
        top_p: float = Param(
            default=0.95,
            description="Nucleus sampling threshold.",
        ),
    ) -> Iterator[str]:
        """Stream a completion response."""
        if tools is not None and tool_choice is None:
            tool_choice = "auto"

        img_content = image if has_image_content(image) else None
        messages = build_openai_messages(
            prompt=prompt, image=img_content, images=images, messages=chat_history
        )
        for chunk in self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            stream_options={"include_usage": True},
        ):
            if chunk.usage is not None:
                if chunk.usage.prompt_tokens or chunk.usage.completion_tokens:
                    self.set_output_context(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                    )
            if chunk.choices:
                if chunk.choices[0].delta.tool_calls:
                    tool_calls_json = [tc.to_dict() for tc in chunk.choices[0].delta.tool_calls]
                    yield json.dumps(tool_calls_json, indent=2)
                else:
                    text = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ''
                    yield text
