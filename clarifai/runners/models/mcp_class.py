"""Base class for creating Model Context Protocol (MCP) servers."""

import asyncio
import json
from typing import Any

from fastmcp import Client, FastMCP  # use fastmcp v2 not the built in mcp
from mcp import types
from mcp.shared.exceptions import McpError

from clarifai.runners.models.model_class import ModelClass

# class MCPServerProvider(abc.ABC):
#     """
#     Base class for creating Model Context Protocol (MCP) servers.

#     This class provides a base implementation of the MCP server, including
#     methods for handling requests and responses, as well as error handling and
#     logging.

#     Attributes:
#     _server: The FastMCP server instance.
#     _tools: List of tools available in the server.
#     _resources: List of resources available in the server.
#     _prompts: List of prompts available in the server.

#     Methods:
#     get_server(): Returns the FastMCP server instance.
#     mcp_transport(msg): Handles incoming messages and sends them to the FastMCP server.
#     """

#     @abc.abstractmethod
#     def get_server(self) -> FastMCP:
#         """Required method for each subclass to implement to return the FastMCP server to use."""
#         if self._server is None:
#             raise ValueError("Server not initialized")
#         return self._server


class MCPModelClass(ModelClass):
    """Base class for wrapping FastMCP servers as a model running in Clarfai. This handles
    all the transport between the API and the MCP server here. Simply subclass this and implement
    the get_server() method to return the FastMCP server instance. The server is then used to
    handle all the requests and responses.
    """

    def load_model(self):
        # in memory transport provided in fastmcp v2 so we can easily use the client functions.
        self.client = Client(self.get_server())

    def get_server(self) -> FastMCP:
        """Required method for each subclass to implement to return the FastMCP server to use."""
        raise NotImplementedError("Subclasses must implement get_server() method")

    @ModelClass.method
    def mcp_transport(self, msg: str) -> str:
        """The single model method to get the jsonrpc message and send it to the FastMCP server then
        return it's response.

        """

        async def send_notification(client_message: types.ClientNotification) -> None:
            async with self.client:
                # Strip the jsonrpc field since send_notification will also pass it in for some reason.
                client_message = types.ClientNotification.model_validate(
                    client_message.model_dump(
                        by_alias=True, mode="json", exclude_none=True, exclude={"jsonrpc"}
                    )
                )
                try:
                    return await self.client.session.send_notification(client_message)
                except McpError as e:
                    return types.JSONRPCError(jsonrpc="2.0", error=e.error)

        async def send_request(client_message: types.ClientRequest, id: str) -> Any:
            async with self.client:
                # Strip the jsonrpc and id fields as send_request sets them again too.
                client_message = types.ClientRequest.model_validate(
                    client_message.model_dump(
                        by_alias=True, mode="json", exclude_none=True, exclude={"jsonrpc", "id"}
                    )
                )

                result_type = None
                if isinstance(client_message.root, types.PingRequest):
                    result_type = types.EmptyResult
                elif isinstance(client_message.root, types.InitializeRequest):
                    return await self.client.session.initialize()
                elif isinstance(client_message.root, types.SetLevelRequest):
                    result_type = types.EmptyResult
                elif isinstance(client_message.root, types.ListResourcesRequest):
                    result_type = types.ListResourcesResult
                elif isinstance(client_message.root, types.ListResourceTemplatesRequest):
                    result_type = types.ListResourceTemplatesResult
                elif isinstance(client_message.root, types.ReadResourceRequest):
                    result_type = types.ReadResourceResult
                elif isinstance(client_message.root, types.SubscribeRequest):
                    result_type = types.EmptyResult
                elif isinstance(client_message.root, types.UnsubscribeRequest):
                    result_type = types.EmptyResult
                elif isinstance(client_message.root, types.ListPromptsRequest):
                    result_type = types.ListPromptsResult
                elif isinstance(client_message.root, types.GetPromptRequest):
                    result_type = types.GetPromptResult
                elif isinstance(client_message.root, types.CompleteRequest):
                    result_type = types.CompleteResult
                elif isinstance(client_message.root, types.ListToolsRequest):
                    result_type = types.ListToolsResult
                elif isinstance(client_message.root, types.CallToolRequest):
                    result_type = types.CallToolResult
                else:
                    # this is a special case where we need to return the list of tools.
                    raise NotImplementedError(f"Method {client_message.method} not implemented")
                # Call the mcp server using send_request() or send_notification() depending on the method.
                try:
                    return await self.client.session.send_request(client_message, result_type)
                except McpError as e:
                    return types.JSONRPCError(jsonrpc="2.0", id=id, error=e.error)

        # The message coming here is the generic request. We look at it's .method
        # to determine which client function to call and to further subparse the params.
        # Note(zeiler): unfortunately the pydantic types in mcp/types.py are not consistent.
        # The JSONRPCRequest are supposed to have an id but the InitializeRequest
        # does not have it.
        d = json.loads(msg)

        # If we have an id it's a JSONRPCRequest
        if not d.get('method', None).startswith("notifications/"):
            # rpc_message2 = types.JSONRPCRequest.model_validate(rpc_message)
            # underlying: types.JSONRPCRequest = rpc_message.root
            client_message = types.ClientRequest.model_validate(d)
            response = asyncio.run(
                send_request(client_message, id=d.get('id', ""))
            )  # underlying.id))
        else:  # JSONRPCRequest
            client_message = types.ClientNotification.model_validate(d)
            response = asyncio.run(send_notification(client_message))
        if response is None:
            return ""
        # return as a serialized json string
        return response.model_dump_json(by_alias=True, exclude_none=True)
