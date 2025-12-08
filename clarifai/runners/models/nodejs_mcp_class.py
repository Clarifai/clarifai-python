"""Base class for bridging Node.js MCP servers with Python FastMCP servers."""

import os
import traceback
from contextlib import asynccontextmanager
from typing import Any, Optional

import yaml

from clarifai.runners.models.mcp_class import MCPModelClass
from clarifai.utils.logging import logger

try:
    from fastmcp import FastMCP
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import TextContent, Tool
except ImportError:
    FastMCP = None
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    Tool = None
    TextContent = None


class NodeMCPClient:
    """Client for communicating with a Node.js MCP server."""

    def __init__(
        self,
        command: str = "npx",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        self.command = command
        self.args = args or []
        if not self.args:
            raise ValueError("args must be provided")
        self.env = env or {}

    @asynccontextmanager
    async def _get_session(self):
        """Get a fresh session for a single operation."""
        server_params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        stdio_ctx = stdio_client(server_params)
        stdio_transport = await stdio_ctx.__aenter__()
        read_stream, write_stream = stdio_transport

        session_ctx = ClientSession(read_stream, write_stream)
        session = await session_ctx.__aenter__()
        await session.initialize()

        try:
            yield session
        finally:
            try:
                await session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                await stdio_ctx.__aexit__(None, None, None)
            except Exception:
                pass

    async def list_tools(self) -> list[Tool]:
        """List all tools from the Node.js MCP server."""
        async with self._get_session() as session:
            result = await session.list_tools()
            return result.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the Node.js MCP server."""
        async with self._get_session() as session:
            result = await session.call_tool(name, arguments)

            output_parts = []
            for content in result.content:
                if isinstance(content, TextContent):
                    output_parts.append(content.text)
                elif hasattr(content, 'text'):
                    output_parts.append(content.text)
                else:
                    output_parts.append(str(content))

            output = "\n".join(output_parts)

            if result.isError:
                raise RuntimeError(f"Tool error: {output}")

            return output


class NodeJSMCPModelClass(MCPModelClass):
    """Base class for bridging Node.js MCP servers with Python FastMCP servers.

    This class automatically discovers and registers all tools from a Node.js MCP server
    into a Python FastMCP server, making them available to MCP clients.

    Configuration is read from config.yaml in the 'mcp' section:

    Example config.yaml:
        mcp:
          command: "npx"
          args: ["-y", "@modelcontextprotocol/server-github"]
          env:
            GITHUB_PERSONAL_ACCESS_TOKEN: "your-token-here"

    Subclasses should simply inherit from this class:

        class MCPModel(NodeJSMCPModelClass):
            pass
    """

    def __init__(self):
        super().__init__()
        self._node_client: Optional[NodeMCPClient] = None
        self._server: Optional[FastMCP] = None
        self._tools_registered = False

    def _json_type_to_python(self, json_type: str) -> type:
        """Convert JSON schema type to Python type."""
        return {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }.get(json_type, str)

    def _create_tool_function(
        self, tool_name: str, properties: dict, required: list, node_client: NodeMCPClient
    ) -> callable:
        """
        Create a tool function with explicit parameters (no **kwargs).

        FastMCP requires functions with explicit signatures, so we dynamically
        generate the function code and execute it.
        """
        func_name = tool_name.replace("-", "_").replace(".", "_")

        # Build parameter list (required first, then optional with defaults)
        required_params = [p for p in properties.keys() if p in required]
        optional_params = [f"{p}=None" for p in properties.keys() if p not in required]
        params_str = ", ".join(required_params + optional_params)

        # Build function body
        body_lines = ["    try:"]
        body_lines.append("        args = {}")
        for param in properties.keys():
            body_lines.append(f"        if {param} is not None:")
            body_lines.append(f"            args['{param}'] = {param}")
        body_lines.append(f"        return await node_client.call_tool('{tool_name}', args)")
        body_lines.append("    except Exception as e:")
        body_lines.append("        import traceback")
        body_lines.append("        error_type = type(e).__name__")
        body_lines.append("        error_msg = str(e) if str(e) else repr(e)")
        body_lines.append(
            "        tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))"
        )
        body_lines.append(
            f"        return f'Error executing {tool_name} ({{error_type}}): {{error_msg}}\\n\\nTraceback:\\n{{tb}}'"
        )

        code = f"async def {func_name}({params_str}) -> str:\n" + "\n".join(body_lines)

        # Execute to create function
        namespace = {"node_client": node_client}
        exec(code, namespace)
        func = namespace[func_name]

        # Set annotations for FastMCP
        annotations = {"return": str}
        for param, schema in properties.items():
            ptype = self._json_type_to_python(schema.get("type", "string"))
            annotations[param] = ptype if param in required else Optional[ptype]
        func.__annotations__ = annotations

        return func

    def _find_config_file(self) -> Optional[str]:
        """Find config.yaml file in common locations."""

        # parent directory (for models in 1/ subdirectory)
        if os.path.exists("../config.yaml"):
            print("config.yaml at ../config.yaml")
            return "../config.yaml"

        # Try absolute path from common model structure
        current_dir = os.getcwd()
        if os.path.exists(os.path.join(current_dir, "config.yaml")):
            return os.path.join(current_dir, "config.yaml")
        return None

    def _load_secrets(self) -> dict[str, Any]:
        """Load secrets from .secrets.yaml."""
        config_path = self._find_config_file()
        if not config_path:
            raise FileNotFoundError(
                "config.yaml not found. Please ensure config.yaml exists in the model directory "
                "with an 'mcp_server' section containing 'command', 'args', and optionally 'env'."
            )
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError("config.yaml is empty")

        secrets_config = config.get("secrets", None)
        if not secrets_config:
            return {}

        secrets = []
        for secret in secrets_config:
            secrets.append(
                {
                    "id": secret.get("id"),
                    "value": secret.get("value", None),
                    "env_var": secret.get("env_var"),
                }
            )
        return secrets

    def _load_mcp_config(self) -> dict[str, Any]:
        """Load MCP configuration from config.yaml."""
        config_path = self._find_config_file()
        if not config_path:
            raise FileNotFoundError(
                "config.yaml not found. Please ensure config.yaml exists in the model directory "
                "with an 'mcp_server' section containing 'command', 'args', and optionally 'env'."
            )

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError("config.yaml is empty")

        mcp_config = config.get("mcp_server")
        if not mcp_config:
            raise ValueError(
                "No 'mcp_server' section found in config.yaml. Please add an 'mcp_server' section with "
                "'command', 'args', and optionally 'env' fields."
            )

        command = mcp_config.get("command")
        args = mcp_config.get("args")
        env = mcp_config.get("env", {})

        if not command:
            raise ValueError("'command' is required in the 'mcp_server' section of config.yaml")
        if not args:
            raise ValueError("'args' is required in the 'mcp_server' section of config.yaml")

        return {
            "command": command,
            "args": args if isinstance(args, list) else [args],
            "env": env if isinstance(env, dict) else {},
        }

    def _get_node_client(self) -> NodeMCPClient:
        """Get or create the Node.js MCP client."""
        if self._node_client is None:
            mcp_config = self._load_mcp_config()
            env = mcp_config["env"]
            secrets = self._load_secrets()
            for secret in secrets:
                if secret["value"] is not None:
                    env[secret["env_var"]] = secret["value"]
                elif os.environ.get(secret["env_var"], None) is not None:
                    env[secret["env_var"]] = os.environ.get(secret["env_var"])
            self._node_client = NodeMCPClient(
                command=mcp_config["command"], args=mcp_config["args"], env=env
            )
        return self._node_client

    def _get_server(self) -> FastMCP:
        """Get or create the FastMCP server."""
        if self._server is None:
            if FastMCP is None:
                raise ImportError(
                    "fastmcp package is required to use MCP functionality. "
                    "Install it with: pip install fastmcp"
                )

            # Create lifespan function that has access to self
            @asynccontextmanager
            async def lifespan(server: FastMCP):
                """Lifespan manager: discovers and registers tools from Node.js server on startup."""
                if self._tools_registered:
                    yield
                    return

                logger.info(
                    "Starting Node.js MCP bridge server...",
                )

                try:
                    # Discover tools from Node.js server
                    node_client = self._get_node_client()
                    tools = await node_client.list_tools()

                    logger.info(f"✅ Discovered {len(tools)} tools from Node.js MCP server...")

                    # Register each tool as a FastMCP tool
                    for tool in tools:
                        tool_name = tool.name
                        tool_desc = tool.description or f"Tool: {tool_name}"
                        schema = tool.inputSchema or {}
                        properties = schema.get("properties", {})
                        required = schema.get("required", [])

                        # Create function with explicit parameters
                        func = self._create_tool_function(
                            tool_name, properties, required, node_client
                        )
                        func.__doc__ = tool_desc

                        # Register with FastMCP
                        server.add_tool(func, name=tool_name, description=tool_desc)

                        # Update the tool's parameters schema with the actual schema from Node.js MCP server
                        if hasattr(server, '_tool_manager') and hasattr(
                            server._tool_manager, '_tools'
                        ):
                            registered_tool = server._tool_manager._tools.get(tool_name)
                            if registered_tool and schema:
                                registered_tool.parameters = schema

                        logger.info(f"   ✅ {tool_name}")

                    logger.info(f"✅ Bridge server ready with {len(tools)} tools")
                    self._tools_registered = True

                except Exception as e:
                    logger.info(f"⚠️  Error in lifespan: {e}")
                    traceback.print_exc()
                    raise

                try:
                    yield  # Server runs here
                finally:
                    logger.info("🔧 Bridge server shutting down...")

            self._server = FastMCP(
                "nodejs-mcp-bridge",
                instructions="""
Bridge to Node.js MCP server.
All tools from the Node.js MCP server are automatically available.
""".strip(),
                lifespan=lifespan,
            )
        return self._server

    def get_server(self) -> FastMCP:
        """Return the FastMCP server instance."""
        return self._get_server()
