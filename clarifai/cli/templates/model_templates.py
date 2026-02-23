"""Templates for model initialization."""

from clarifai import __version__


def get_model_class_template() -> str:
    """Return the template for a basic ModelClass-based model."""
    return '''from typing import Iterator, List
from clarifai.runners.models.model_class import ModelClass
from clarifai.runners.utils.data_utils import Param

class MyModel(ModelClass):
    """A custom model."""

    def load_model(self):
        """Initialize your model here. Called once when the model starts."""
        pass

    @ModelClass.method
    def predict(
        self,
        prompt: str = "",
        chat_history: List[dict] = None,
        max_tokens: int = Param(default=256, description="The maximum number of tokens to generate."),
        temperature: float = Param(default=1.0, description="Sampling temperature (higher = more random)."),
        top_p: float = Param(default=1.0, description="Nucleus sampling threshold."),
    ) -> str:
        """Return a single response."""
        return f"Echo: {prompt}"

    @ModelClass.method
    def generate(
        self,
        prompt: str = "",
        chat_history: List[dict] = None,
        max_tokens: int = Param(default=256, description="The maximum number of tokens to generate."),
        temperature: float = Param(default=1.0, description="Sampling temperature (higher = more random)."),
        top_p: float = Param(default=1.0, description="Nucleus sampling threshold."),
    ) -> Iterator[str]:
        """Stream a response."""
        for word in f"Echo: {prompt}".split():
            yield word + " "
'''


def get_mcp_model_class_template() -> str:
    """Return the template for an MCPModelClass-based model."""
    return '''from typing import Any

from fastmcp import FastMCP
from pydantic import Field

from clarifai.runners.models.mcp_class import MCPModelClass

server = FastMCP("my-mcp-server", instructions="A sample MCP server.", stateless_http=True)


@server.tool("hello", description="Say hello to someone")
def hello(name: str = Field(description="Name to greet")) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"


@server.resource("config://version")
def get_version():
    """Return the server version."""
    return "1.0.0"


class MyModel(MCPModelClass):
    """MCP model that exposes tools, resources, and prompts."""

    def get_server(self) -> FastMCP:
        """Return the FastMCP server instance."""
        return server
'''


def get_openai_model_class_template(port: str = "8000") -> str:
    """Return the template for an OpenAIModelClass-based model."""
    return f'''from typing import List, Iterator
from openai import OpenAI
from clarifai.runners.models.openai_class import OpenAIModelClass
from clarifai.runners.utils.data_utils import Param
from clarifai.runners.utils.openai_convertor import build_openai_messages

class MyModel(OpenAIModelClass):
    """Wraps an OpenAI-compatible API endpoint."""

    client = OpenAI(
        api_key="local-key",
        base_url="http://localhost:{port}/v1",
    )

    model = client.models.list().data[0].id

    def load_model(self):
        """Optional initialization logic."""
        pass

    @OpenAIModelClass.method
    def predict(
        self,
        prompt: str = "",
        chat_history: List[dict] = None,
        max_tokens: int = Param(default=256, description="The maximum number of tokens to generate."),
        temperature: float = Param(default=1.0, description="Sampling temperature (higher = more random)."),
        top_p: float = Param(default=1.0, description="Nucleus sampling threshold."),
    ) -> str:
        """Run a single prompt completion."""
        messages = build_openai_messages(prompt, chat_history)
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
        max_tokens: int = Param(default=256, description="The maximum number of tokens to generate."),
        temperature: float = Param(default=1.0, description="Sampling temperature (higher = more random)."),
        top_p: float = Param(default=1.0, description="Nucleus sampling threshold."),
    ) -> Iterator[str]:
        """Stream a completion response."""
        messages = build_openai_messages(prompt, chat_history)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices:
                text = (chunk.choices[0].delta.content
                        if (chunk and chunk.choices[0].delta.content) is not None else '')
                yield text
'''


def get_config_template(
    user_id: str = None,
    model_type_id: str = "any-to-any",
    model_id: str = "my-model",
    simplified: bool = True,
) -> str:
    """Return the template for config.yaml.

    Args:
        user_id: User ID to include in the config. In simplified mode, this is omitted
                 (resolved from CLI context at deploy time).
        model_type_id: Model type ID.
        model_id: Model ID.
        simplified: If True, generate simplified config (no TODOs, compute.instance shorthand).
                    If False, generate verbose config with all fields.
    """
    if simplified:
        return f'''model:
  id: "{model_id}"
  model_type_id: "{model_type_id}"

compute:
  instance: g5.xlarge  # Run 'clarifai model deploy --instance-info' to see all options.
  # cloud: aws          # Cloud provider (aws, gcp, vultr). Auto-detected from instance.
  # region: us-east-1   # Cloud region. Auto-detected from instance.

# Uncomment to auto-download model checkpoints:
# checkpoints:
#   repo_id: owner/model-name
'''
    else:
        return _get_verbose_config_template(user_id, model_type_id, model_id)


def _get_verbose_config_template(
    user_id: str = None, model_type_id: str = "any-to-any", model_id: str = "my-model"
) -> str:
    """Return the verbose template for config.yaml (original format)."""
    return f'''model:
  id: "{model_id}"
  user_id: "{user_id}"
  app_id: "app_id"
  model_type_id: "{model_type_id}"

build_info:
  python_version: "3.12"

inference_compute_info:
  cpu_limit: "1"
  cpu_memory: "1Gi"
  cpu_requests: "0.5"
  cpu_memory_requests: "512Mi"
  num_accelerators: 1
  accelerator_type: ["NVIDIA-*"]
  accelerator_memory: "1Gi"

# checkpoints:
#   type: "huggingface"
#   repo_id: "your-model-repo"
#   when: "runtime"
'''


def get_requirements_template(model_type_id: str = None) -> str:
    """Return the template for requirements.txt."""
    req = f'clarifai>={__version__}\n'
    if model_type_id == "mcp":
        req += "fastmcp\n"
    elif model_type_id == "openai":
        req += "openai\n"
    return req


# Mapping of model type IDs to their corresponding templates
MODEL_TYPE_TEMPLATES = {
    "mcp": get_mcp_model_class_template,
    "openai": get_openai_model_class_template,
}


def get_model_template(model_type_id: str = None, **kwargs) -> str:
    """Get the appropriate model template based on model_type_id."""
    if model_type_id in MODEL_TYPE_TEMPLATES:
        template_func = MODEL_TYPE_TEMPLATES[model_type_id]
        import inspect

        sig = inspect.signature(template_func)
        if len(sig.parameters) > 0:
            return template_func(**kwargs)
        else:
            return template_func()
    return get_model_class_template()
