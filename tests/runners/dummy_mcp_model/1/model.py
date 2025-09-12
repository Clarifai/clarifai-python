from typing import Any

from fastmcp import FastMCP  # use fastmcp v2 not the built in mcp
from pydantic import Field

from clarifai.runners.models.mcp_class import MCPModelClass

server = FastMCP("my-first-mcp-server", instructions="", stateless_http=True)


@server.tool("calculate_sum", description="Add two numbers together")
def sum(a: Any = Field(description="first number"), b: Any = Field(description="second number")):
    return float(a) + float(b)


# Static resource
@server.resource("config://version")
def get_version():
    return "2.0.1"


@server.prompt()
def summarize_request(text: str) -> str:
    """Generate a prompt asking for a summary."""
    return f"Please summarize the following text:\n\n{text}"


class MyModelClass(MCPModelClass):
    def get_server(self) -> FastMCP:
        return server
