"""Agent system for Clarifai CLI - enables tool calling and command execution."""

import inspect
import json
from typing import Any, Callable, Dict, List

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
        self._auto_discover_tools()

    def _auto_discover_tools(self):
        """Auto-discover and register tools from clarifai.client module."""
        from clarifai import client as client_module

        # Classes to introspect for tools
        target_classes = [
            'User',
            'App',
            'Model',
            'Dataset',
            'Inputs',
            'Workflow',
            'Pipeline',
            'Search',
        ]

        for class_name in target_classes:
            if not hasattr(client_module, class_name):
                continue

            cls = getattr(client_module, class_name)
            self._register_class_methods(cls, class_name)

        logger.debug(f"Auto-discovered {len(self.tools)} tools from clarifai.client")

    def _register_class_methods(self, cls: type, class_name: str):
        """Register public methods from a class as tools.

        Args:
            cls: The class to introspect
            class_name: Name of the class (for namespacing)
        """
        # Skip private and special methods
        excluded_patterns = {'__', '_', 'from_', 'load_info', '__getattr__', '__str__'}

        for name in dir(cls):
            # Skip private/special methods
            if any(name.startswith(p) for p in excluded_patterns):
                continue

            # Skip property-like methods that shouldn't be tools
            if name in {'id', 'app_id', 'user_id', 'model_id', 'dataset_id'}:
                continue

            try:
                attr = getattr(cls, name)

                # Only process callable methods
                if not callable(attr):
                    continue

                # Skip class methods and static methods
                if isinstance(attr, (classmethod, staticmethod)):
                    continue

                # Create a wrapper that binds the method to the agent's context
                tool_wrapper = self._create_method_wrapper(cls, name, class_name)
                if tool_wrapper is None:
                    continue

                # Extract signature info
                sig = inspect.signature(attr)
                description = self._extract_description(attr)

                # Determine required vs optional params (skip 'self')
                required_params = []
                optional_params = []

                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue

                    if param.default == inspect.Parameter.empty:
                        required_params.append(param_name)
                    else:
                        optional_params.append(param_name)

                # Create tool name: class_method
                tool_name = f"{class_name.lower()}_{name}"

                tool = Tool(
                    name=tool_name,
                    description=description,
                    func=tool_wrapper,
                    required_params=required_params,
                    optional_params=optional_params,
                )

                self.register_tool(tool)

            except Exception as e:
                logger.debug(f"Could not register {class_name}.{name}: {str(e)}")

    def _create_method_wrapper(self, cls: type, method_name: str, class_name: str) -> Callable | None:
        """Create a wrapper function that calls a class method with proper initialization.

        Args:
            cls: The class containing the method
            method_name: Name of the method
            class_name: Name of the class

        Returns:
            Wrapper function or None if creation fails
        """

        def wrapper(**kwargs) -> Any:
            try:
                # Instantiate the class with agent's credentials
                if class_name == 'User':
                    instance = cls(user_id=self.user_id, pat=self.pat)
                else:
                    # For App, Model, Dataset, etc. - require app_id
                    app_id = kwargs.pop('app_id', None)
                    if not app_id and class_name in ['App', 'Model', 'Dataset', 'Inputs', 'Workflow']:
                        raise ValueError(f"{class_name} requires 'app_id' parameter")

                    if class_name in ['Model']:
                        model_id = kwargs.pop('model_id', None)
                        if not model_id:
                            raise ValueError("Model requires 'model_id' parameter")
                        instance = cls(
                            app_id=app_id, model_id=model_id, user_id=self.user_id, pat=self.pat
                        )
                    elif class_name in ['Dataset']:
                        dataset_id = kwargs.pop('dataset_id', None)
                        if not dataset_id:
                            raise ValueError("Dataset requires 'dataset_id' parameter")
                        instance = cls(
                            app_id=app_id, dataset_id=dataset_id, user_id=self.user_id, pat=self.pat
                        )
                    elif class_name in ['Workflow']:
                        workflow_id = kwargs.pop('workflow_id', None)
                        if not workflow_id:
                            raise ValueError("Workflow requires 'workflow_id' parameter")
                        instance = cls(
                            app_id=app_id, workflow_id=workflow_id, user_id=self.user_id, pat=self.pat
                        )
                    elif class_name in ['Pipeline']:
                        pipeline_id = kwargs.pop('pipeline_id', None)
                        if not pipeline_id:
                            raise ValueError("Pipeline requires 'pipeline_id' parameter")
                        instance = cls(
                            app_id=app_id, pipeline_id=pipeline_id, user_id=self.user_id, pat=self.pat
                        )
                    elif class_name in ['Search']:
                        instance = cls(app_id=app_id, user_id=self.user_id, pat=self.pat)
                    else:
                        instance = cls(app_id=app_id, user_id=self.user_id, pat=self.pat)

                # Call the method
                method = getattr(instance, method_name)
                result = method(**kwargs)

                # Handle generators - convert to list for JSON serialization
                if inspect.isgenerator(result):
                    result = list(result)

                # Convert objects to dicts for JSON serialization
                result = self._convert_to_serializable(result)

                return result

            except Exception as e:
                raise Exception(f"Error calling {class_name}.{method_name}: {str(e)}")

        return wrapper

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Recursively convert Clarifai objects to serializable dicts.

        Args:
            obj: Object to convert

        Returns:
            Serializable object (dict, list, or primitive)
        """
        return self._convert_to_serializable_impl(obj, set(), 0)

    def _convert_to_serializable_impl(self, obj: Any, seen: set, depth: int) -> Any:
        """Implementation of recursive conversion with circular reference detection.

        Args:
            obj: Object to convert
            seen: Set of object ids we've already processed
            depth: Current recursion depth

        Returns:
            Serializable object
        """
        # Prevent infinite recursion
        if depth > 10:
            return "[Max depth exceeded]"

        # Handle primitives
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        # Handle lists
        if isinstance(obj, list):
            return [self._convert_to_serializable_impl(item, seen, depth + 1) for item in obj]

        # Handle dicts
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                result[k] = self._convert_to_serializable_impl(v, seen, depth + 1)
            return result

        # Check for circular references
        obj_id = id(obj)
        if obj_id in seen:
            return f"[Circular reference to {type(obj).__name__}]"

        # Handle objects with __dict__
        if hasattr(obj, '__dict__'):
            seen.add(obj_id)
            try:
                obj_dict = vars(obj).copy()
                result = {}

                # Skip certain problematic attributes
                skip_attrs = {'stub', '_stub', 'grpc_channel', '_grpc_channel', 'credentials', '_credentials'}

                for k, v in obj_dict.items():
                    if k in skip_attrs or k.startswith('_'):
                        continue
                    result[k] = self._convert_to_serializable_impl(v, seen, depth + 1)

                return result
            finally:
                seen.discard(obj_id)

        # Fallback to string representation
        return str(obj)

        return wrapper

    def _extract_description(self, method: Callable) -> str:
        """Extract description from method docstring.

        Args:
            method: The method to extract description from

        Returns:
            First line of docstring or generic description
        """
        if method.__doc__:
            first_line = method.__doc__.strip().split('\n')[0]
            return first_line
        return f"Call {method.__name__}"

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
