"""Agent system for Clarifai CLI - enables tool calling and command execution."""

import json
from typing import Any, Callable, Dict, List

from clarifai.client.app import App
from clarifai.utils.logging import logger


class Tool:
    """Represents an agent tool that can be called."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        required_params: List[str],
        optional_params: List[str] = None,
    ):
        """Initialize a Tool.

        Args:
            name: Tool name (snake_case)
            description: Human-readable description of what the tool does
            func: Callable that executes the tool
            required_params: List of required parameter names
            optional_params: List of optional parameter names
        """
        self.name = name
        self.description = description
        self.func = func
        self.required_params = required_params
        self.optional_params = optional_params or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to JSON schema format for LLM."""
        properties = {}

        # Add required parameters
        for param in self.required_params:
            properties[param] = {
                "type": "string",
                "description": f"Required parameter: {param}",
            }

        # Add optional parameters
        for param in self.optional_params:
            properties[param] = {
                "type": "string",
                "description": f"Optional parameter: {param}",
            }

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": self.required_params,
            },
        }

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters.

        Args:
            params: Dictionary of parameters

        Returns:
            Dict with 'success' bool and 'result' or 'error'
        """
        try:
            # Validate required parameters
            missing = [p for p in self.required_params if p not in params]
            if missing:
                return {
                    "success": False,
                    "error": f"Missing required parameters: {', '.join(missing)}",
                }

            result = self.func(**params)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ClarifaiAgent:
    """Agent for executing Clarifai CLI commands."""

    def __init__(self, pat: str, user_id: str):
        """Initialize the agent.

        Args:
            pat: Personal Access Token for authentication
            user_id: Clarifai user ID
        """
        self.pat = pat
        self.user_id = user_id
        self.tools: Dict[str, Tool] = {}
        self._register_tools()

    def _register_tools(self):
        """Register all available tools."""
        # App operations
        self.register_tool(
            Tool(
                name="create_app",
                description="Create a new Clarifai app with the specified ID",
                func=self._create_app,
                required_params=["app_id"],
                optional_params=["name", "description"],
            )
        )

        self.register_tool(
            Tool(
                name="list_apps",
                description="List all apps for the current user",
                func=self._list_apps,
                required_params=[],
                optional_params=["page_no", "per_page"],
            )
        )

        # Model operations
        self.register_tool(
            Tool(
                name="create_model",
                description="Create a new model in an app",
                func=self._create_model,
                required_params=["app_id", "model_id"],
                optional_params=["description", "model_type"],
            )
        )

        self.register_tool(
            Tool(
                name="list_models",
                description="List all models in an app",
                func=self._list_models,
                required_params=["app_id"],
                optional_params=["page_no", "per_page"],
            )
        )

        # Dataset operations
        self.register_tool(
            Tool(
                name="create_dataset",
                description="Create a new dataset in an app",
                func=self._create_dataset,
                required_params=["app_id", "dataset_id"],
                optional_params=["description"],
            )
        )

        self.register_tool(
            Tool(
                name="list_datasets",
                description="List all datasets in an app",
                func=self._list_datasets,
                required_params=["app_id"],
                optional_params=["page_no", "per_page"],
            )
        )

        # Workflow operations
        self.register_tool(
            Tool(
                name="list_workflows",
                description="List all workflows in an app",
                func=self._list_workflows,
                required_params=["app_id"],
                optional_params=["page_no", "per_page"],
            )
        )

    def register_tool(self, tool: Tool):
        """Register a new tool."""
        self.tools[tool.name] = tool

    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get tool definitions formatted for LLM function calling."""
        return [tool.to_dict() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            params: Parameters to pass to the tool

        Returns:
            Dict with execution result
        """
        if tool_name not in self.tools:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        tool = self.tools[tool_name]
        return tool.execute(params)

    # ===== App Operations =====

    def _create_app(self, app_id: str, name: str = None, description: str = None) -> str:
        """Create a new app."""
        try:
            from clarifai.client.user import User

            user = User(user_id=self.user_id, pat=self.pat)
            kwargs = {}
            if name:
                kwargs['name'] = name
            if description:
                kwargs['description'] = description
            app = user.create_app(app_id=app_id, **kwargs)
            return f"App '{app_id}' created successfully"
        except Exception as e:
            raise Exception(f"Failed to create app: {str(e)}")

    def _list_apps(self, page_no: int = None, per_page: int = None) -> List[str]:
        """List all apps for the user."""
        try:
            from clarifai.client.user import User

            user = User(user_id=self.user_id, pat=self.pat)
            apps = []
            for app in user.list_apps(page_no=page_no, per_page=per_page):
                apps.append({"id": app.id, "name": getattr(app, "name", "N/A")})
            return apps
        except Exception as e:
            raise Exception(f"Failed to list apps: {str(e)}")

    # ===== Model Operations =====

    def _create_model(
        self, app_id: str, model_id: str, description: str = None, model_type: str = None
    ) -> str:
        """Create a new model in an app."""
        try:
            app = App(app_id=app_id, user_id=self.user_id, pat=self.pat)
            model = app.create_model(model_id=model_id, description=description)
            return f"Model '{model_id}' created successfully in app '{app_id}'"
        except Exception as e:
            raise Exception(f"Failed to create model: {str(e)}")

    def _list_models(self, app_id: str, page_no: int = None, per_page: int = None) -> List[str]:
        """List all models in an app."""
        try:
            app = App(app_id=app_id, user_id=self.user_id, pat=self.pat)
            models = []
            for model in app.list_models(page_no=page_no, per_page=per_page):
                models.append(
                    {
                        "id": model.id,
                        "model_type": getattr(model, "model_type", "N/A"),
                    }
                )
            return models
        except Exception as e:
            raise Exception(f"Failed to list models: {str(e)}")

    # ===== Dataset Operations =====

    def _create_dataset(self, app_id: str, dataset_id: str, description: str = None) -> str:
        """Create a new dataset in an app."""
        try:
            app = App(app_id=app_id, user_id=self.user_id, pat=self.pat)
            dataset = app.create_dataset(dataset_id=dataset_id, description=description)
            return f"Dataset '{dataset_id}' created successfully in app '{app_id}'"
        except Exception as e:
            raise Exception(f"Failed to create dataset: {str(e)}")

    def _list_datasets(self, app_id: str, page_no: int = None, per_page: int = None) -> List[str]:
        """List all datasets in an app."""
        try:
            app = App(app_id=app_id, user_id=self.user_id, pat=self.pat)
            datasets = []
            for dataset in app.list_datasets(page_no=page_no, per_page=per_page):
                datasets.append({"id": dataset.id})
            return datasets
        except Exception as e:
            raise Exception(f"Failed to list datasets: {str(e)}")

    # ===== Workflow Operations =====

    def _list_workflows(self, app_id: str, page_no: int = None, per_page: int = None) -> List[str]:
        """List all workflows in an app."""
        try:
            app = App(app_id=app_id, user_id=self.user_id, pat=self.pat)
            workflows = []
            for workflow in app.list_workflows(page_no=page_no, per_page=per_page):
                workflows.append({"id": workflow.id})
            return workflows
        except Exception as e:
            raise Exception(f"Failed to list workflows: {str(e)}")


def parse_tool_calls_from_response(response_text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from LLM response text.

    Looks for JSON-formatted tool calls in the response.
    Format: <tool_call>{"tool": "tool_name", "params": {"key": "value"}}</tool_call>

    Args:
        response_text: Raw LLM response text

    Returns:
        List of tool call dicts or empty list if none found
    """
    tool_calls = []
    import re

    # Look for <tool_call>...</tool_call> blocks
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, response_text, re.DOTALL)

    for match in matches:
        try:
            tool_call = json.loads(match)
            if "tool" in tool_call and "params" in tool_call:
                tool_calls.append(tool_call)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool call: {match}")

    return tool_calls


def format_tool_call_in_response(tool_name: str, params: Dict[str, Any]) -> str:
    """Format a tool call for embedding in LLM response.

    Args:
        tool_name: Name of the tool
        params: Parameters for the tool

    Returns:
        Formatted tool call string
    """
    tool_call = {"tool": tool_name, "params": params}
    return f"<tool_call>{json.dumps(tool_call)}</tool_call>"


def build_agent_system_prompt(agent: ClarifaiAgent) -> str:
    """Build a system prompt that instructs the LLM to use agent tools.

    Args:
        agent: ClarifaiAgent instance with registered tools

    Returns:
        System prompt string
    """
    tools_list = ""
    for tool in agent.tools.values():
        params_str = ", ".join(
            [f"{p} (required)" for p in tool.required_params]
            + [f"{p} (optional)" for p in tool.optional_params]
        )
        tools_list += f"- {tool.name}: {tool.description}\n  Parameters: {params_str}\n"

    return f"""You are an expert Clarifai CLI assistant with the ability to execute commands.

AVAILABLE TOOLS:
{tools_list}

TOOL USAGE RULES:
1. When user asks for an action (create, list, delete), use the appropriate tool
2. Format tool calls as: <tool_call>{{"tool": "tool_name", "params": {{"param1": "value1"}}}}</tool_call>
3. Always include tool calls in your response when relevant
4. After executing a tool, summarize the result for the user
5. Ask for clarification if required parameters are missing

RESPONSE FORMAT:
- Start with your analysis/explanation
- Include tool calls where appropriate (using the <tool_call> format)
- End with a summary of what was accomplished or next steps

EXAMPLE:
User: "Create a new app called my_vision_app"
Your response: "I'll create a new app with ID 'my_vision_app'.
<tool_call>{{"tool": "create_app", "params": {{"app_id": "my_vision_app", "name": "my_vision_app"}}}}</tool_call>
The app has been created successfully!"

Remember:
- Keep responses concise and actionable
- For general questions about Clarifai, provide helpful explanations
- For command-related questions, use tools to demonstrate functionality"""
