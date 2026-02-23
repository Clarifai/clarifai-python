import json
import os
import socket
import subprocess
import sys
import time
from typing import Iterator, List

from openai import OpenAI

from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.runners.utils.data_types import Image
from clarifai.runners.utils.data_utils import Param
from clarifai.runners.utils.openai_convertor import build_openai_messages
from clarifai.utils.logging import logger

VERBOSE_LMSTUDIO = True
LMS_MODEL_NAME = "LiquidAI/LFM2-1.2B"
LMS_PORT = 11434
LMS_CONTEXT_LENGTH = 4096


def _stream_command(cmd, verbose=True):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    if verbose and process.stdout:
        for line in iter(process.stdout.readline, ""):
            if line:
                logger.info(f"[lms] {line.rstrip()}")
    ret = process.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed ({ret}): {cmd}")
    return True


def _wait_for_port(port, timeout=30.0):
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            try:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    return True
            except Exception:
                pass
        time.sleep(0.5)
    raise RuntimeError(f"LM Studio server did not start on port {port} within {timeout}s")


def run_lms_server(model_name='LiquidAI/LFM2-1.2B', port=11434, context_length=4096):
    """Download model, load it, and start the LM Studio server."""
    try:
        _stream_command(
            f"lms get https://huggingface.co/{model_name} --verbose",
            verbose=VERBOSE_LMSTUDIO,
        )
        _stream_command("lms unload --all", verbose=VERBOSE_LMSTUDIO)
        _stream_command(
            f"lms load {model_name} --verbose --context-length {context_length}",
            verbose=VERBOSE_LMSTUDIO,
        )
        subprocess.Popen(
            f"lms server start --port {port}",
            shell=True,
            stdout=None if not VERBOSE_LMSTUDIO else sys.stdout,
            stderr=None if not VERBOSE_LMSTUDIO else sys.stderr,
        )
        _wait_for_port(port)
        logger.info(f"LM Studio server started on port {port}")
    except Exception as e:
        raise RuntimeError(f"Failed to start LM Studio server: {e}")


def has_image_content(image: Image) -> bool:
    return bool(getattr(image, 'url', None) or getattr(image, 'bytes', None))


class LMStudioModel(OpenAIModelClass):
    client = True
    model = True

    def load_model(self):
        self.model = LMS_MODEL_NAME
        self.port = LMS_PORT
        run_lms_server(
            model_name=self.model,
            port=self.port,
            context_length=LMS_CONTEXT_LENGTH,
        )
        self.client = OpenAI(api_key="notset", base_url=f"http://localhost:{self.port}/v1")

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
