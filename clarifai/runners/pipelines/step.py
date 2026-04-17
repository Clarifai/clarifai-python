import inspect
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

from clarifai.runners.pipelines.compute import ComputeConfig

_ACTIVE_PIPELINE = threading.local()
_STEP_REF_URL_PATTERN = re.compile(
    r'^users/(?P<user_id>[^/]+)/apps/(?P<app_id>[^/]+)/pipeline_steps/(?P<step_id>[^/]+)/versions/(?P<version_id>[^/]+)$'
)


def _parse_step_ref_url(step_url: str) -> Dict[str, str]:
    parsed_url = urlparse(step_url)
    resource_path = parsed_url.path if (parsed_url.scheme or parsed_url.netloc) else step_url
    normalized_path = resource_path.strip('/')
    if normalized_path.startswith('v2/'):
        normalized_path = normalized_path[3:]

    match = _STEP_REF_URL_PATTERN.match(normalized_path)
    if match is None:
        raise ValueError(
            'step_ref.from_url() expects a versioned pipeline step URL or resource path of the form '
            "'users/{user_id}/apps/{app_id}/pipeline_steps/{step_id}/versions/{version_id}'"
        )

    return match.groupdict()


def _set_active_pipeline(pipeline):
    _ACTIVE_PIPELINE.current = pipeline


def _clear_active_pipeline():
    _ACTIVE_PIPELINE.current = None


def get_active_pipeline():
    return getattr(_ACTIVE_PIPELINE, 'current', None)


@dataclass(frozen=True)
class WorkflowInputRef:
    name: str

    def as_argo_value(self) -> str:
        return f"{{{{workflow.parameters.{self.name}}}}}"


@dataclass(frozen=True)
class OutputRef:
    node: 'StepNode'
    output_name: str = 'result'

    def as_argo_value(self) -> str:
        return f"{{{{steps.{self.node.name}.outputs.parameters.{self.output_name}}}}}"


class StepNode:
    """A concrete task instance in a pipeline DAG."""

    def __init__(
        self, pipeline, step_definition: 'StepDefinition', name: str, arguments: Dict[str, Any]
    ):
        self.pipeline = pipeline
        self.step_definition = step_definition
        self.name = name
        self.arguments = arguments
        self.dependencies = set()

    def output(self, param_name: str = 'result') -> OutputRef:
        return OutputRef(node=self, output_name=param_name)

    def __rshift__(self, other):
        self.pipeline.add_dependency(self, other)
        return other

    def __rrshift__(self, other):
        if isinstance(other, list):
            for item in other:
                self.pipeline.add_dependency(item, self)
            return self
        raise TypeError(f"Unsupported dependency source: {type(other)!r}")


class StepDefinition:
    is_managed = True

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        id: Optional[str] = None,
        requirements=None,
        compute: Optional[ComputeConfig] = None,
        python_version: str = '3.12',
        secrets: Optional[Dict[str, str]] = None,
    ):
        self.func = func
        self.id = id or func.__name__.replace('_', '-')
        self.requirements = requirements or []
        self.compute = compute or ComputeConfig()
        self.python_version = python_version
        self.secrets = secrets or {}
        self.signature = inspect.signature(func)

    @property
    def __name__(self):
        return self.func.__name__

    def template_ref(self, default_user_id: str, default_app_id: str) -> str:
        return f'users/{default_user_id}/apps/{default_app_id}/pipeline_steps/{self.id}'

    def __call__(self, *args, **kwargs):
        pipeline = get_active_pipeline()
        if pipeline is None:
            return self.func(*args, **kwargs)

        task_name = kwargs.pop('name', None)
        bound = self.signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return pipeline.add_node(self, arguments=dict(bound.arguments), name=task_name)

    def test(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def get_input_params(self):
        params = []
        for param in self.signature.parameters.values():
            if param.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            param_config = {'name': param.name}
            if param.default is not inspect._empty:
                param_config['default'] = str(param.default)
            annotation = param.annotation
            if annotation is not inspect._empty:
                annotation_name = getattr(annotation, '__name__', str(annotation))
                param_config['description'] = f'Auto-generated from annotation {annotation_name}'
            params.append(param_config)
        return params


class ExistingStepDefinition:
    is_managed = False

    @classmethod
    def from_url(
        cls, step_url: str, *, secrets: Optional[Dict[str, str]] = None
    ) -> 'ExistingStepDefinition':
        parsed_step = _parse_step_ref_url(step_url)
        return cls(
            id=parsed_step['step_id'],
            version_id=parsed_step['version_id'],
            user_id=parsed_step['user_id'],
            app_id=parsed_step['app_id'],
            secrets=secrets,
        )

    def __init__(
        self,
        *,
        id: str,
        version_id: str,
        user_id: Optional[str] = None,
        app_id: Optional[str] = None,
        secrets: Optional[Dict[str, str]] = None,
    ):
        if not version_id:
            raise ValueError('version_id is required for step_ref()')
        self.id = id
        self.version_id = version_id
        self.user_id = user_id
        self.app_id = app_id
        self.secrets = secrets or {}

    @property
    def __name__(self):
        return self.id.replace('-', '_')

    def template_ref(self, default_user_id: str, default_app_id: str) -> str:
        user_id = self.user_id or default_user_id
        app_id = self.app_id or default_app_id
        return f'users/{user_id}/apps/{app_id}/pipeline_steps/{self.id}/versions/{self.version_id}'

    def __call__(self, *args, **kwargs):
        pipeline = get_active_pipeline()
        if pipeline is None:
            raise RuntimeError('step_ref() instances can only be used inside an active Pipeline')

        if args:
            raise TypeError('step_ref() calls only support keyword arguments')

        task_name = kwargs.pop('name', None)
        return pipeline.add_node(self, arguments=dict(kwargs), name=task_name)

    def test(self, *args, **kwargs):
        raise RuntimeError('step_ref() does not support local execution')

    def get_input_params(self):
        return []


def step(
    *,
    id: Optional[str] = None,
    requirements=None,
    compute: Optional[ComputeConfig] = None,
    python_version: str = '3.12',
    secrets: Optional[Dict[str, str]] = None,
):
    def decorator(func: Callable[..., Any]) -> StepDefinition:
        return StepDefinition(
            func,
            id=id,
            requirements=requirements,
            compute=compute,
            python_version=python_version,
            secrets=secrets,
        )

    return decorator


def step_ref(
    *,
    id: str,
    version_id: str,
    user_id: Optional[str] = None,
    app_id: Optional[str] = None,
    secrets: Optional[Dict[str, str]] = None,
) -> ExistingStepDefinition:
    return ExistingStepDefinition(
        id=id,
        version_id=version_id,
        user_id=user_id,
        app_id=app_id,
        secrets=secrets,
    )


def _step_ref_from_url(
    step_url: str, *, secrets: Optional[Dict[str, str]] = None
) -> ExistingStepDefinition:
    return ExistingStepDefinition.from_url(step_url, secrets=secrets)


step_ref.from_url = _step_ref_from_url
