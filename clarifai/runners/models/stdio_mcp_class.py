"""Bridge a stdio MCP server to a Python FastMCP server.

The implementation keeps a **single long‑lived session** for the whole
FastMCP server lifetime:

*   The stdio process is started once (the first time it is needed).
*   The same `ClientSession` object is reused for every subsequent call.
*   The stdio process is shut down cleanly when the FastMCP server's lifespan
    context exits.
"""

import asyncio
import inspect
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


class StdioMCPClient:
    """A thin wrapper around a stdio MCP server that re‑uses a single session."""

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

        self._stdio_ctx = None
        self._session_ctx = None
        self._session: Optional[ClientSession] = None
        self._started = False
        self._lock = asyncio.Lock()

    async def _ensure_started(self) -> None:
        """Start the stdio process and MCP session if not already running."""
        if self._started:
            return

        async with self._lock:
            # Double-check after acquiring lock
            if self._started:
                return

            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env,
            )
            self._stdio_ctx = stdio_client(server_params)
            stdio_transport = await self._stdio_ctx.__aenter__()
            read_stream, write_stream = stdio_transport

            self._session_ctx = ClientSession(read_stream, write_stream)
            self._session = await self._session_ctx.__aenter__()
            await self._session.initialize()

            self._started = True
            logger.debug("StdioMCPClient: stdio process + MCP session started")

    async def close(self) -> None:
        """Gracefully shut down the stdio process and MCP session."""
        async with self._lock:
            if not self._started:
                return

            if self._session_ctx is not None:
                try:
                    await self._session_ctx.__aexit__(None, None, None)
                except Exception:
                    logger.exception("Error while closing MCP session")

            if self._stdio_ctx is not None:
                try:
                    await self._stdio_ctx.__aexit__(None, None, None)
                except Exception:
                    logger.exception("Error while closing stdio transport")

            self._started = False
            self._session = None
            self._session_ctx = None
            self._stdio_ctx = None
            logger.debug("StdioMCPClient: stdio process + MCP session stopped")

    async def list_tools(self) -> list[Tool]:
        """List all tools from the stdio MCP server."""
        await self._ensure_started()
        result = await self._session.list_tools()
        return result.tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the stdio MCP server."""
        await self._ensure_started()
        result = await self._session.call_tool(name, arguments)

        output_parts = []
        for content in result.content:
            if isinstance(content, TextContent):
                output_parts.append(content.text)
            elif hasattr(content, "text"):
                output_parts.append(content.text)
            else:
                output_parts.append(str(content))

        output = "\n".join(output_parts)

        if result.isError:
            raise RuntimeError(f"Tool error: {output}")

        return output


class StdioMCPModelClass(MCPModelClass):
    """Base class for bridging stdio MCP servers with Python FastMCP servers.

    This class automatically discovers and registers all tools from a stdio MCP server
    into a Python FastMCP server, making them available to MCP clients.

    Configuration is read from config.yaml in the 'mcp' section:

    Example config.yaml:
        mcp:
          command: "npx"
          args: ["-y", "@modelcontextprotocol/server-github"]
          env:
            GITHUB_PERSONAL_ACCESS_TOKEN: "your-token-here"

    Subclasses should simply inherit from this class:

        class MCPModel(StdioMCPModelClass):
            pass
    """

    def __init__(self):
        super().__init__()
        self._stdio_client: Optional[StdioMCPClient] = None
        self._server: Optional[FastMCP] = None
        # Flag to indicate whether tools have been registered with the FastMCP server.
        # Prevents duplicate registration. Reset to False on shutdown to allow re-registration if restarted.
        self._tools_registered = False

    def _json_type_to_python(self, json_type: str) -> type:
        return {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }.get(json_type, str)

    def _create_tool_function(
        self,
        tool_name: str,
        properties: dict,
        required: list,
        stdio_client: StdioMCPClient,
    ) -> callable:
        """Create a FastMCP‑compatible coroutine function dynamically."""
        func_name = tool_name.replace("-", "_").replace(".", "_")

        required_params = [p for p in properties if p in required]
        optional_params = [f"{p}=None" for p in properties if p not in required]
        params_str = ", ".join(required_params + optional_params)

        body = [
            "    try:",
            "        args = {}",
        ]
        for param in properties:
            body.append(f"        if {param} is not None:")
            body.append(f"            args['{param}'] = {param}")
        body.append(f"        return await stdio_client.call_tool('{tool_name}', args)")
        body.extend(
            [
                "    except Exception as e:",
                "        import traceback",
                "        error_type = type(e).__name__",
                "        error_msg = str(e) if str(e) else repr(e)",
                "        tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))",
                f"        return f'Error executing {tool_name} ({{error_type}}): {{error_msg}}\\n\\nTraceback:\\n{{tb}}'",
            ]
        )

        code = f"async def {func_name}({params_str}) -> str:\n" + "\n".join(body)

        namespace = {"stdio_client": stdio_client}
        exec(code, namespace)
        func = namespace[func_name]

        annotations = {"return": str}
        for param, schema in properties.items():
            py_type = self._json_type_to_python(schema.get("type", "string"))
            annotations[param] = py_type if param in required else Optional[py_type]
        func.__annotations__ = annotations

        return func

    def _find_config_file(self) -> Optional[str]:
        """Find config.yaml file in the same directory as the class file."""
        # Get the file path of the actual class (subclass) being instantiated
        try:
            class_file = inspect.getfile(self.__class__)
            class_dir = os.path.dirname(os.path.abspath(class_file))

            config_path = os.path.join(os.path.dirname(class_dir), "config.yaml")
            if os.path.exists(config_path):
                return config_path

        except (OSError, TypeError) as e:
            logger.warning(f"Could not determine class file location: {e}")

        return None

    def _load_secrets(self) -> list[dict[str, Any]]:
        config_path = self._find_config_file()
        if not config_path:
            raise FileNotFoundError("config.yaml not found")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return [
            {
                "id": s.get("id"),
                "value": s.get("value"),
                "env_var": s.get("env_var"),
            }
            for s in config.get("secrets", [])
        ]

    def _load_mcp_config(self) -> dict[str, Any]:
        config_path = self._find_config_file()
        if not config_path:
            raise FileNotFoundError("config.yaml not found")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        mcp_cfg = config.get("mcp_server")
        if not mcp_cfg:
            raise ValueError("Missing 'mcp_server' section in config.yaml")

        command = mcp_cfg.get("command")
        args = mcp_cfg.get("args")
        env = mcp_cfg.get("env", {})

        if not command:
            raise ValueError("'command' missing in mcp_server")
        if not args:
            raise ValueError("'args' missing in mcp_server")

        return {
            "command": command,
            "args": args if isinstance(args, list) else [args],
            "env": env if isinstance(env, dict) else {},
        }

    def _get_stdio_client(self) -> StdioMCPClient:
        """Get or create the stdio MCP client."""
        if self._stdio_client is None:
            cfg = self._load_mcp_config()
            env = dict(cfg["env"])

            for secret in self._load_secrets():
                env_var = secret.get("env_var")
                if not env_var:
                    continue
                if secret.get("value") is not None:
                    env[env_var] = secret["value"]
                elif os.getenv(env_var) is not None:
                    env[env_var] = os.getenv(env_var)

            self._stdio_client = StdioMCPClient(
                command=cfg["command"],
                args=cfg["args"],
                env=env,
            )
        return self._stdio_client

    def get_server(self) -> FastMCP:
        """Return the FastMCP server instance."""
        if self._server is not None:
            return self._server

        if FastMCP is None:
            raise ImportError("fastmcp package is required – install with `pip install fastmcp`")

        @asynccontextmanager
        async def lifespan(server: FastMCP):
            """Discover stdio tools and keep the session alive."""
            if self._tools_registered:
                yield
                return

            logger.info("🚀 Starting stdio MCP bridge...")
            stdio_client = self._get_stdio_client()

            try:
                tools = await stdio_client.list_tools()
                logger.info(f"✅ Discovered {len(tools)} tools from stdio MCP")

                for tool in tools:
                    name = tool.name
                    desc = tool.description or f"Tool: {name}"
                    schema = tool.inputSchema or {}
                    props = schema.get("properties", {})
                    required = schema.get("required", [])

                    func = self._create_tool_function(name, props, required, stdio_client)
                    func.__doc__ = desc
                    server.add_tool(func, name=name, description=desc)

                    # Preserve original JSON‑schema
                    if hasattr(server, "_tool_manager") and hasattr(
                        server._tool_manager, "_tools"
                    ):
                        reg = server._tool_manager._tools.get(name)
                        if reg and schema:
                            reg.parameters = schema

                    logger.debug(f"   ✅ Registered {name}")

                self._tools_registered = True
                logger.info("✅ Bridge server ready")

            except Exception as exc:
                logger.error(f"❌ Error during bridge startup: {exc}")
                traceback.print_exc()
                raise

            try:
                yield
            finally:
                logger.info("🛑 Shutting down stdio MCP bridge...")
                try:
                    await stdio_client.close()
                except Exception:
                    logger.exception("Error while closing StdioMCPClient")
                self._stdio_client = None
                logger.info("🛑 Bridge shutdown complete")

        self._server = FastMCP(
            "stdio-mcp-bridge",
            instructions="Bridge to a stdio MCP server. All tools are automatically available.",
            lifespan=lifespan,
        )
        return self._server

    async def _background_shutdown(self) -> None:
        """Override to also close the stdio client."""
        # Close stdio client first
        if self._stdio_client is not None:
            try:
                await self._stdio_client.close()
            except Exception:
                logger.exception("Error while closing StdioMCPClient")
            self._stdio_client = None

        # Then call parent shutdown
        await super()._background_shutdown()

    def shutdown(self):
        """
        Cleanly shut down the server and reset the tools registration flag.
        Call this when the FastMCP server is shutting down.
        """
        self._tools_registered = False
        super().shutdown()
