"""Action registry for chat agent SDK calls.

Maps action names to direct SDK method calls, avoiding code generation.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from clarifai.client import App, User
from clarifai.cli.chat.executor import get_current_user_id

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    data: Any
    message: str
    error: Optional[str] = None


@dataclass  
class ActionDef:
    """Definition of an available action."""
    name: str
    description: str
    params: Dict[str, str]  # param_name -> description
    required_params: List[str]
    needs_confirmation: bool
    handler: Callable


def _get_user() -> User:
    """Get authenticated User instance with user_id from config."""
    user_id = get_current_user_id()
    return User(user_id=user_id)


def _get_app(app_id: str) -> App:
    """Get App instance for the given app_id."""
    user_id = get_current_user_id()
    return App(user_id=user_id, app_id=app_id)


# Action handlers
def _list_apps(params: Dict) -> ActionResult:
    """List all apps for the user."""
    user = _get_user()
    apps = list(user.list_apps())
    
    if not apps:
        return ActionResult(
            success=True,
            data=[],
            message="No apps found."
        )
    
    output = []
    for app in apps:
        output.append(f"- {app.id}: {getattr(app, 'description', '') or '(no description)'}")
    
    return ActionResult(
        success=True,
        data=[{"id": app.id, "description": getattr(app, 'description', '')} for app in apps],
        message=f"Found {len(apps)} app(s):\n" + "\n".join(output)
    )


def _delete_app(params: Dict) -> ActionResult:
    """Delete an app by ID."""
    app_id = params.get("app_id")
    if not app_id:
        return ActionResult(
            success=False,
            data=None,
            message="",
            error="app_id is required"
        )
    
    user = _get_user()
    user.delete_app(app_id)
    
    return ActionResult(
        success=True,
        data={"deleted": app_id},
        message=f"App '{app_id}' has been deleted."
    )


def _create_app(params: Dict) -> ActionResult:
    """Create a new app."""
    app_id = params.get("app_id")
    if not app_id:
        return ActionResult(
            success=False,
            data=None,
            message="",
            error="app_id is required"
        )
    
    base_workflow = params.get("base_workflow", "Empty")
    description = params.get("description", "")
    
    user = _get_user()
    app = user.create_app(app_id=app_id, base_workflow=base_workflow, description=description)
    
    return ActionResult(
        success=True,
        data={"id": app.id},
        message=f"App '{app_id}' has been created."
    )


def _list_models(params: Dict) -> ActionResult:
    """List models. If app_id provided, list models in that app. Otherwise list user's models."""
    app_id = params.get("app_id")
    
    if app_id:
        app = _get_app(app_id)
        models = list(app.list_models())
        scope = f"app '{app_id}'"
    else:
        user = _get_user()
        models = list(user.list_models())
        scope = "your account"
    
    if not models:
        return ActionResult(
            success=True,
            data=[],
            message=f"No models found in {scope}."
        )
    
    output = []
    for model in models:
        model_id = getattr(model, 'id', str(model))
        output.append(f"- {model_id}")
    
    return ActionResult(
        success=True,
        data=[{"id": getattr(m, 'id', str(m))} for m in models],
        message=f"Found {len(models)} model(s) in {scope}:\n" + "\n".join(output[:20])
    )


def _list_datasets(params: Dict) -> ActionResult:
    """List datasets in an app."""
    app_id = params.get("app_id")
    if not app_id:
        return ActionResult(
            success=False,
            data=None,
            message="",
            error="app_id is required to list datasets"
        )
    
    app = _get_app(app_id)
    datasets = list(app.list_datasets())
    
    if not datasets:
        return ActionResult(
            success=True,
            data=[],
            message=f"No datasets found in app '{app_id}'."
        )
    
    output = []
    for ds in datasets:
        ds_id = getattr(ds, 'id', str(ds))
        output.append(f"- {ds_id}")
    
    return ActionResult(
        success=True,
        data=[{"id": getattr(ds, 'id', str(ds))} for ds in datasets],
        message=f"Found {len(datasets)} dataset(s) in app '{app_id}':\n" + "\n".join(output)
    )


def _list_workflows(params: Dict) -> ActionResult:
    """List workflows in an app."""
    app_id = params.get("app_id")
    if not app_id:
        return ActionResult(
            success=False,
            data=None,
            message="",
            error="app_id is required to list workflows"
        )
    
    app = _get_app(app_id)
    workflows = list(app.list_workflows())
    
    if not workflows:
        return ActionResult(
            success=True,
            data=[],
            message=f"No workflows found in app '{app_id}'."
        )
    
    output = []
    for wf in workflows:
        wf_id = getattr(wf, 'id', str(wf))
        output.append(f"- {wf_id}")
    
    return ActionResult(
        success=True,
        data=[{"id": getattr(wf, 'id', str(wf))} for wf in workflows],
        message=f"Found {len(workflows)} workflow(s) in app '{app_id}':\n" + "\n".join(output)
    )


def _list_concepts(params: Dict) -> ActionResult:
    """List concepts in an app."""
    app_id = params.get("app_id")
    if not app_id:
        return ActionResult(
            success=False,
            data=None,
            message="",
            error="app_id is required to list concepts"
        )
    
    app = _get_app(app_id)
    concepts = list(app.list_concepts())
    
    if not concepts:
        return ActionResult(
            success=True,
            data=[],
            message=f"No concepts found in app '{app_id}'."
        )
    
    output = []
    for c in concepts:
        c_id = getattr(c, 'id', str(c))
        output.append(f"- {c_id}")
    
    return ActionResult(
        success=True,
        data=[{"id": getattr(c, 'id', str(c))} for c in concepts],
        message=f"Found {len(concepts)} concept(s) in app '{app_id}':\n" + "\n".join(output)
    )


def _delete_model(params: Dict) -> ActionResult:
    """Delete a model from an app."""
    app_id = params.get("app_id")
    model_id = params.get("model_id")
    
    if not app_id:
        return ActionResult(success=False, data=None, message="", error="app_id is required")
    if not model_id:
        return ActionResult(success=False, data=None, message="", error="model_id is required")
    
    app = _get_app(app_id)
    app.delete_model(model_id)
    
    return ActionResult(
        success=True,
        data={"deleted": model_id},
        message=f"Model '{model_id}' has been deleted from app '{app_id}'."
    )


def _delete_dataset(params: Dict) -> ActionResult:
    """Delete a dataset from an app."""
    app_id = params.get("app_id")
    dataset_id = params.get("dataset_id")
    
    if not app_id:
        return ActionResult(success=False, data=None, message="", error="app_id is required")
    if not dataset_id:
        return ActionResult(success=False, data=None, message="", error="dataset_id is required")
    
    app = _get_app(app_id)
    app.delete_dataset(dataset_id)
    
    return ActionResult(
        success=True,
        data={"deleted": dataset_id},
        message=f"Dataset '{dataset_id}' has been deleted from app '{app_id}'."
    )


def _delete_workflow(params: Dict) -> ActionResult:
    """Delete a workflow from an app."""
    app_id = params.get("app_id")
    workflow_id = params.get("workflow_id")
    
    if not app_id:
        return ActionResult(success=False, data=None, message="", error="app_id is required")
    if not workflow_id:
        return ActionResult(success=False, data=None, message="", error="workflow_id is required")
    
    app = _get_app(app_id)
    app.delete_workflow(workflow_id)
    
    return ActionResult(
        success=True,
        data={"deleted": workflow_id},
        message=f"Workflow '{workflow_id}' has been deleted from app '{app_id}'."
    )


def _create_dataset(params: Dict) -> ActionResult:
    """Create a dataset in an app."""
    app_id = params.get("app_id")
    dataset_id = params.get("dataset_id")
    
    if not app_id:
        return ActionResult(success=False, data=None, message="", error="app_id is required")
    if not dataset_id:
        return ActionResult(success=False, data=None, message="", error="dataset_id is required")
    
    app = _get_app(app_id)
    dataset = app.create_dataset(dataset_id=dataset_id)
    
    return ActionResult(
        success=True,
        data={"id": dataset.id},
        message=f"Dataset '{dataset_id}' has been created in app '{app_id}'."
    )


def _get_user_info(params: Dict) -> ActionResult:
    """Get current user information."""
    user = _get_user()
    info = user.get_user_info()
    
    return ActionResult(
        success=True,
        data={"id": info.id, "email": getattr(info, 'email', '')},
        message=f"User ID: {info.id}"
    )


def _list_pipelines(params: Dict) -> ActionResult:
    """List pipelines."""
    app_id = params.get("app_id")
    
    if app_id:
        app = _get_app(app_id)
        pipelines = list(app.list_pipelines())
        scope = f"app '{app_id}'"
    else:
        user = _get_user()
        pipelines = list(user.list_pipelines())
        scope = "your account"
    
    if not pipelines:
        return ActionResult(
            success=True,
            data=[],
            message=f"No pipelines found in {scope}."
        )
    
    output = []
    for p in pipelines:
        p_id = getattr(p, 'id', str(p))
        output.append(f"- {p_id}")
    
    return ActionResult(
        success=True,
        data=[{"id": getattr(p, 'id', str(p))} for p in pipelines],
        message=f"Found {len(pipelines)} pipeline(s) in {scope}:\n" + "\n".join(output[:20])
    )


def _list_compute_clusters(params: Dict) -> ActionResult:
    """List compute clusters."""
    user = _get_user()
    clusters = list(user.list_compute_clusters())
    
    if not clusters:
        return ActionResult(
            success=True,
            data=[],
            message="No compute clusters found."
        )
    
    output = []
    for c in clusters:
        c_id = getattr(c, 'id', str(c))
        output.append(f"- {c_id}")
    
    return ActionResult(
        success=True,
        data=[{"id": getattr(c, 'id', str(c))} for c in clusters],
        message=f"Found {len(clusters)} compute cluster(s):\n" + "\n".join(output)
    )


def _list_runners(params: Dict) -> ActionResult:
    """List runners."""
    user = _get_user()
    runners = list(user.list_runners())
    
    if not runners:
        return ActionResult(
            success=True,
            data=[],
            message="No runners found."
        )
    
    output = []
    for r in runners:
        r_id = getattr(r, 'id', str(r))
        output.append(f"- {r_id}")
    
    return ActionResult(
        success=True,
        data=[{"id": getattr(r, 'id', str(r))} for r in runners],
        message=f"Found {len(runners)} runner(s):\n" + "\n".join(output)
    )


# Action registry
ACTIONS: Dict[str, ActionDef] = {
    "list_apps": ActionDef(
        name="list_apps",
        description="List all apps for the current user",
        params={},
        required_params=[],
        needs_confirmation=False,
        handler=_list_apps,
    ),
    "delete_app": ActionDef(
        name="delete_app",
        description="Delete an app by ID",
        params={"app_id": "The ID of the app to delete"},
        required_params=["app_id"],
        needs_confirmation=True,
        handler=_delete_app,
    ),
    "create_app": ActionDef(
        name="create_app",
        description="Create a new app",
        params={
            "app_id": "The ID for the new app",
            "base_workflow": "Base workflow (default: Empty)",
            "description": "App description",
        },
        required_params=["app_id"],
        needs_confirmation=True,
        handler=_create_app,
    ),
    "list_models": ActionDef(
        name="list_models",
        description="List models (optionally in a specific app)",
        params={"app_id": "Optional app ID to list models from"},
        required_params=[],
        needs_confirmation=False,
        handler=_list_models,
    ),
    "list_datasets": ActionDef(
        name="list_datasets",
        description="List datasets in an app",
        params={"app_id": "The app ID"},
        required_params=["app_id"],
        needs_confirmation=False,
        handler=_list_datasets,
    ),
    "list_workflows": ActionDef(
        name="list_workflows",
        description="List workflows in an app",
        params={"app_id": "The app ID"},
        required_params=["app_id"],
        needs_confirmation=False,
        handler=_list_workflows,
    ),
    "list_concepts": ActionDef(
        name="list_concepts",
        description="List concepts in an app",
        params={"app_id": "The app ID"},
        required_params=["app_id"],
        needs_confirmation=False,
        handler=_list_concepts,
    ),
    "list_pipelines": ActionDef(
        name="list_pipelines",
        description="List pipelines (optionally in a specific app)",
        params={"app_id": "Optional app ID"},
        required_params=[],
        needs_confirmation=False,
        handler=_list_pipelines,
    ),
    "list_compute_clusters": ActionDef(
        name="list_compute_clusters",
        description="List compute clusters",
        params={},
        required_params=[],
        needs_confirmation=False,
        handler=_list_compute_clusters,
    ),
    "list_runners": ActionDef(
        name="list_runners",
        description="List runners",
        params={},
        required_params=[],
        needs_confirmation=False,
        handler=_list_runners,
    ),
    "delete_model": ActionDef(
        name="delete_model",
        description="Delete a model from an app",
        params={"app_id": "The app ID", "model_id": "The model ID to delete"},
        required_params=["app_id", "model_id"],
        needs_confirmation=True,
        handler=_delete_model,
    ),
    "delete_dataset": ActionDef(
        name="delete_dataset",
        description="Delete a dataset from an app",
        params={"app_id": "The app ID", "dataset_id": "The dataset ID to delete"},
        required_params=["app_id", "dataset_id"],
        needs_confirmation=True,
        handler=_delete_dataset,
    ),
    "delete_workflow": ActionDef(
        name="delete_workflow",
        description="Delete a workflow from an app",
        params={"app_id": "The app ID", "workflow_id": "The workflow ID to delete"},
        required_params=["app_id", "workflow_id"],
        needs_confirmation=True,
        handler=_delete_workflow,
    ),
    "create_dataset": ActionDef(
        name="create_dataset",
        description="Create a dataset in an app",
        params={"app_id": "The app ID", "dataset_id": "The dataset ID to create"},
        required_params=["app_id", "dataset_id"],
        needs_confirmation=True,
        handler=_create_dataset,
    ),
    "get_user_info": ActionDef(
        name="get_user_info",
        description="Get current user information",
        params={},
        required_params=[],
        needs_confirmation=False,
        handler=_get_user_info,
    ),
}


def get_action(name: str) -> Optional[ActionDef]:
    """Get action definition by name."""
    return ACTIONS.get(name)


def list_actions() -> List[ActionDef]:
    """List all available actions."""
    return list(ACTIONS.values())


def execute_action(name: str, params: Dict = None) -> ActionResult:
    """Execute an action by name with given parameters.
    
    Args:
        name: The action name
        params: Parameters for the action
        
    Returns:
        ActionResult with the execution outcome
    """
    params = params or {}
    
    action = get_action(name)
    if not action:
        return ActionResult(
            success=False,
            data=None,
            message="",
            error=f"Unknown action: {name}"
        )
    
    # Check required params
    missing = [p for p in action.required_params if p not in params]
    if missing:
        return ActionResult(
            success=False,
            data=None,
            message="",
            error=f"Missing required parameters: {', '.join(missing)}"
        )
    
    try:
        return action.handler(params)
    except Exception as e:
        logger.exception(f"Error executing action {name}")
        return ActionResult(
            success=False,
            data=None,
            message="",
            error=str(e)
        )


def parse_action_from_response(response: str) -> Optional[Dict]:
    """Parse an action JSON from an LLM response.
    
    Looks for JSON in ```json or ```action code blocks, or inline JSON.
    
    Args:
        response: The LLM response text
        
    Returns:
        Parsed action dict with 'action' and optional params, or None
    """
    import re
    
    # Only parse actions from explicit ```json or ```action code blocks
    # Do NOT parse inline JSON to avoid picking up examples from explanatory text
    patterns = [
        r'```(?:json|action)\s*\n(.*?)\n```',
        r'```\s*\n(\{.*?"action".*?\})\n```',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                if isinstance(data, dict) and 'action' in data:
                    return data
            except json.JSONDecodeError:
                continue
    
    return None


def get_actions_prompt() -> str:
    """Generate a prompt describing available actions for the LLM."""
    lines = ["## Available Actions", ""]
    lines.append("Return actions as JSON in a ```json code block:")
    lines.append("")
    lines.append("```json")
    lines.append('{"action": "action_name", "param1": "value1"}')
    lines.append("```")
    lines.append("")
    lines.append("### Read-only actions (auto-executed):")
    
    for action in ACTIONS.values():
        if not action.needs_confirmation:
            params_str = ", ".join([f'"{k}": "{v}"' for k, v in action.params.items()])
            if params_str:
                example = f'{{"action": "{action.name}", {params_str}}}'
            else:
                example = f'{{"action": "{action.name}"}}'
            lines.append(f"- **{action.name}**: {action.description}")
            lines.append(f"  Example: `{example}`")
    
    lines.append("")
    lines.append("### Actions requiring confirmation:")
    
    for action in ACTIONS.values():
        if action.needs_confirmation:
            params_str = ", ".join([f'"{k}": "..."' for k in action.required_params])
            example = f'{{"action": "{action.name}", {params_str}}}'
            lines.append(f"- **{action.name}**: {action.description}")
            lines.append(f"  Required: {', '.join(action.required_params) or 'none'}")
            lines.append(f"  Example: `{example}`")
    
    return "\n".join(lines)
