import inspect
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from clarifai.runners.pipelines.compute import ComputeConfig

_ACTIVE_PIPELINE = threading.local()


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
        return f"{{{{tasks.{self.node.name}.outputs.parameters.{self.output_name}}}}}"


class StepNode:
    """A concrete task instance in a pipeline DAG."""

    def __init__(self, pipeline, step_definition: 'StepDefinition', name: str, arguments: Dict[str, Any]):
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
            if param.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
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