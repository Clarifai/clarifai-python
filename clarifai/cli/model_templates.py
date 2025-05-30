"""Templates for model initialization."""

from clarifai import __version__


def get_model_class_template() -> str:
    """Return the template for a basic ModelClass-based model."""
    return '''from typing import Iterator

from clarifai.runners.models.model_class import ModelClass


class MyModel(ModelClass):
    """A custom model implementation using ModelClass."""

    def load_model(self):
        """Load the model here.
        
        # TODO: please fill in
        # Add your model loading logic here
        """
        pass

    @ModelClass.method
    def predict(self, prompt: str = "") -> str:
        """This is the method that will be called when the runner is run. It takes in an input and
        returns an output.
        
        # TODO: please fill in
        # Implement your prediction logic here
        """
        # Example implementation:
        # output_text = prompt + " processed"
        # return output_text
        
        return "Hello World"

    # Optional: Add more methods as needed
    # @ModelClass.method
    # def generate(self, prompt: str = "") -> Iterator[str]:
    #     """Example yielding a streamed response."""
    #     # TODO: please fill in
    #     # Implement your generation logic here
    #     for i in range(5):
    #         output_text = prompt + f" generated {i}"
    #         yield output_text
'''


def get_mcp_model_class_template() -> str:
    """Return the template for an MCPModelClass-based model."""
    return '''from typing import Any

from fastmcp import FastMCP  # use fastmcp v2 not the built in mcp
from pydantic import Field

from clarifai.runners.models.mcp_class import MCPModelClass

# TODO: please fill in
# Configure your FastMCP server
server = FastMCP("my-mcp-server", instructions="", stateless_http=True)


# TODO: please fill in
# Add your tools, resources, and prompts here
@server.tool("example_tool", description="An example tool")
def example_tool(input_param: Any = Field(description="Example input parameter")):
    """Example tool implementation."""
    # TODO: please fill in
    # Implement your tool logic here
    return f"Processed: {input_param}"


# Static resource example
@server.resource("config://version")
def get_version():
    """Example static resource."""
    # TODO: please fill in
    # Return your resource data
    return "1.0.0"


@server.prompt()
def example_prompt(text: str) -> str:
    """Example prompt template."""
    # TODO: please fill in
    # Define your prompt template
    return f"Process this text: {text}"


class MyModel(MCPModelClass):
    """A custom model implementation using MCPModelClass."""

    def get_server(self) -> FastMCP:
        """Return the FastMCP server instance."""
        return server
'''


def get_openai_model_class_template() -> str:
    """Return the template for an OpenAIModelClass-based model."""
    return '''from openai import OpenAI

from clarifai.runners.models.openai_class import OpenAIModelClass


class MyModel(OpenAIModelClass):
    """A custom model implementation using OpenAIModelClass."""
    
    # TODO: please fill in
    # Configure your OpenAI-compatible client for local model
    client = OpenAI(
        api_key="local-key",  # TODO: please fill in - use your local API key
        base_url="http://localhost:8000/v1",  # TODO: please fill in - your local model server endpoint
    )
    
    # TODO: please fill in
    # Specify the model name to use
    model = "my-local-model"  # TODO: please fill in - replace with your local model name
    
    def load_model(self):
        """Optional: Add any additional model loading logic here."""
        # TODO: please fill in (optional)
        # Add any initialization logic if needed
        pass
'''


def get_config_template(model_type_id: str = "text-to-text") -> str:
    """Return the template for config.yaml."""
    return f'''# Configuration file for your Clarifai model

model:
  id: "my-model"  # TODO: please fill in - replace with your model ID
  user_id: "user_id"  # TODO: please fill in - replace with your user ID
  app_id: "app_id"  # TODO: please fill in - replace with your app ID
  model_type_id: "{model_type_id}"

build_info:
  python_version: "3.12"

# TODO: please fill in - adjust compute requirements for your model
inference_compute_info:
  cpu_limit: "1"
  cpu_memory: "1Gi"
  cpu_request: "0.5"  # TODO: please fill in - CPU request amount
  memory_request: "512Mi"  # TODO: please fill in - memory request amount
  num_accelerators: 0
  accelerator_request: 0  # TODO: please fill in - number of accelerators requested

# TODO: please fill in (optional) - add checkpoints section if needed
# checkpoints:
#   type: "huggingface"  # supported type
#   repo_id: "your-model-repo"  # for huggingface
#   when: "build"  # or "runtime", "upload"
'''


def get_requirements_template() -> str:
    """Return the template for requirements.txt."""
    return f'''# Clarifai SDK - required
clarifai>={__version__}

# TODO: please fill in - add your model's dependencies here
# Examples:
# torch>=2.0.0
# transformers>=4.30.0
# numpy>=1.21.0
# pillow>=9.0.0
'''


# Mapping of model type IDs to their corresponding templates
MODEL_TYPE_TEMPLATES = {
    "mcp": get_mcp_model_class_template,
    "openai": get_openai_model_class_template,
}


def get_model_template(model_type_id: str = None) -> str:
    """Get the appropriate model template based on model_type_id."""
    if model_type_id in MODEL_TYPE_TEMPLATES:
        return MODEL_TYPE_TEMPLATES[model_type_id]()
    return get_model_class_template()