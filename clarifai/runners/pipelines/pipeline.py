import importlib.util
import os
import tempfile
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional

import yaml

from clarifai.runners.pipelines.codegen import generate_step_directory
from clarifai.runners.pipelines.step import (
    OutputRef,
    StepNode,
    WorkflowInputRef,
    _clear_active_pipeline,
    _set_active_pipeline,
)


class Pipeline:
    """Code-first pipeline definition that compiles to the existing config format."""

    def __init__(self, id: str, user_id: str, app_id: str, visibility: str = 'PRIVATE'):
        self.id = id
        self.user_id = user_id
        self.app_id = app_id
        self.visibility = visibility
        self.nodes: List[StepNode] = []
        self._node_names = set()
        self._workflow_params = OrderedDict()
        self._active = False
        self._uploaded_pipeline_version_id: Optional[str] = None

    def __enter__(self):
        self._active = True
        _set_active_pipeline(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        _clear_active_pipeline()
        self._active = False
        if exc_type is None:
            self.validate()

    def input(self, param_name: str, default: Any = None) -> WorkflowInputRef:
        self._workflow_params[param_name] = default
        return WorkflowInputRef(name=param_name)

    def add_node(
        self, step_definition, arguments: Dict[str, Any], name: Optional[str] = None
    ) -> StepNode:
        if not self._active:
            raise RuntimeError(
                'Pipeline nodes can only be created inside an active Pipeline context'
            )

        node_name = name or self._generate_task_name(step_definition.id)
        node = StepNode(self, step_definition, node_name, arguments)
        self.nodes.append(node)
        self._node_names.add(node_name)

        for value in arguments.values():
            if isinstance(value, OutputRef):
                self.add_dependency(value.node, node)
            elif isinstance(value, WorkflowInputRef):
                self._workflow_params.setdefault(value.name, None)

        return node

    def add_dependency(self, upstream, downstream):
        for source in self._normalize_dependency_operand(upstream):
            for target in self._normalize_dependency_operand(downstream):
                if not isinstance(source, StepNode) or not isinstance(target, StepNode):
                    raise TypeError('Dependencies must be StepNode instances')
                target.dependencies.add(source.name)

    def _normalize_dependency_operand(self, value) -> Iterable[StepNode]:
        if isinstance(value, list):
            return value
        return [value]

    def _generate_task_name(self, step_id: str) -> str:
        candidate = step_id
        suffix = 1
        while candidate in self._node_names:
            candidate = f'{step_id}-{suffix}'
            suffix += 1
        return candidate

    def validate(self):
        nodes_by_name = {node.name: node for node in self.nodes}
        for node in self.nodes:
            missing = [
                dependency for dependency in node.dependencies if dependency not in nodes_by_name
            ]
            if missing:
                raise ValueError(f'Unknown dependencies for {node.name}: {missing}')

        visited = set()
        visiting = set()

        def dfs(node: StepNode):
            if node.name in visited:
                return
            if node.name in visiting:
                raise ValueError(f'Cycle detected in pipeline graph at {node.name}')
            visiting.add(node.name)
            for dependency_name in node.dependencies:
                dfs(nodes_by_name[dependency_name])
            visiting.remove(node.name)
            visited.add(node.name)

        for node in self.nodes:
            dfs(node)

    def _serialize_argument_value(self, value: Any):
        if isinstance(value, WorkflowInputRef):
            return value.as_argo_value()
        if isinstance(value, OutputRef):
            return value.as_argo_value()
        if isinstance(value, bool):
            return 'true' if value else 'false'
        return str(value)

    def _topological_layers(self) -> List[List[StepNode]]:
        """Group nodes into layers where each layer's dependencies are satisfied by earlier layers."""
        nodes_by_name = {node.name: node for node in self.nodes}
        remaining = {node.name for node in self.nodes}
        satisfied: set = set()
        layers: List[List[StepNode]] = []

        while remaining:
            layer = [
                nodes_by_name[name]
                for name in sorted(remaining)
                if nodes_by_name[name].dependencies <= satisfied
            ]
            if not layer:
                raise ValueError('Cycle detected in pipeline graph')
            for node in layer:
                remaining.discard(node.name)
                satisfied.add(node.name)
            layers.append(layer)

        return layers

    def to_argo_spec(self) -> Dict[str, Any]:
        self.validate()
        parameters = []
        for name, default in self._workflow_params.items():
            item = {'name': name}
            if default is not None:
                item['value'] = str(default)
            parameters.append(item)

        step_groups = []
        for layer in self._topological_layers():
            group = []
            for node in layer:
                template_ref_name = node.step_definition.template_ref(self.user_id, self.app_id)
                step_entry = {
                    'name': node.name,
                    'templateRef': {
                        'name': template_ref_name,
                        'template': template_ref_name,
                    },
                }
                if node.arguments:
                    step_entry['arguments'] = {
                        'parameters': [
                            {'name': key, 'value': self._serialize_argument_value(value)}
                            for key, value in node.arguments.items()
                        ]
                    }
                group.append(step_entry)
            step_groups.append(group)

        return {
            'apiVersion': 'argoproj.io/v1alpha1',
            'kind': 'Workflow',
            'spec': {
                'entrypoint': self.id,
                'arguments': {'parameters': parameters},
                'templates': [
                    {
                        'name': self.id,
                        'steps': step_groups,
                    }
                ],
            },
        }

    def to_config(self) -> Dict[str, Any]:
        used_step_ids = []
        seen = set()
        step_version_secrets = {}
        for node in self.nodes:
            if node.step_definition.is_managed and node.step_definition.id not in seen:
                used_step_ids.append(node.step_definition.id)
                seen.add(node.step_definition.id)
            if node.step_definition.secrets:
                step_version_secrets[node.name] = node.step_definition.secrets

        config = {
            'pipeline': {
                'id': self.id,
                'user_id': self.user_id,
                'app_id': self.app_id,
                'visibility': {'gettable': self.visibility},
                'step_directories': used_step_ids,
                'orchestration_spec': {
                    'argo_orchestration_spec': yaml.safe_dump(
                        self.to_argo_spec(), default_flow_style=False, sort_keys=False
                    )
                },
            }
        }

        if step_version_secrets:
            config['pipeline']['config'] = {'step_version_secrets': step_version_secrets}

        return config

    def generate(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        step_definitions = OrderedDict()
        for node in self.nodes:
            if node.step_definition.is_managed:
                step_definitions.setdefault(node.step_definition.id, node.step_definition)

        for step_definition in step_definitions.values():
            generate_step_directory(step_definition, output_dir, self.user_id, self.app_id)

        config_path = os.path.join(output_dir, 'config.yaml')
        with open(config_path, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(self.to_config(), handle, default_flow_style=False, sort_keys=False)
        return config_path

    def upload(self, no_lockfile: bool = False) -> Optional[str]:
        from clarifai.runners.pipelines.pipeline_builder import PipelineBuilder

        temp_dir = tempfile.mkdtemp(prefix='clarifai-pipeline-')
        config_path = self.generate(temp_dir)
        builder = PipelineBuilder(config_path)
        if not builder.ensure_app_exists():
            raise RuntimeError('Failed to verify or create app for pipeline upload')
        if not builder.upload_pipeline_steps():
            raise RuntimeError('Failed to upload pipeline steps')
        lockfile_data = None
        if not no_lockfile:
            lockfile_data = builder.prepare_lockfile_with_step_versions()
        success, pipeline_version_id = builder.create_pipeline()
        if not success:
            raise RuntimeError('Failed to create pipeline')
        if not no_lockfile and lockfile_data:
            builder.save_lockfile(
                builder.update_lockfile_with_pipeline_info(lockfile_data, pipeline_version_id)
            )
        self._uploaded_pipeline_version_id = pipeline_version_id
        return pipeline_version_id

    def run(
        self,
        inputs=None,
        timeout: int = 3600,
        monitor_interval: int = 10,
        input_args_override=None,
        nodepool_id: Optional[str] = None,
        compute_cluster_id: Optional[str] = None,
        log_file: Optional[str] = None,
        base_url: Optional[str] = None,
        pat: Optional[str] = None,
        no_lockfile: bool = False,
    ):
        from clarifai.client.pipeline import Pipeline as PipelineClient

        pipeline_version_id = self._uploaded_pipeline_version_id or self.upload(
            no_lockfile=no_lockfile
        )
        client_kwargs = {
            'pipeline_id': self.id,
            'pipeline_version_id': pipeline_version_id,
            'user_id': self.user_id,
            'app_id': self.app_id,
            'nodepool_id': nodepool_id,
            'compute_cluster_id': compute_cluster_id,
            'log_file': log_file,
        }
        if base_url is not None:
            client_kwargs['base_url'] = base_url
        if pat is not None:
            client_kwargs['pat'] = pat

        client = PipelineClient(**client_kwargs)
        return client.run(
            inputs=inputs,
            timeout=timeout,
            monitor_interval=monitor_interval,
            input_args_override=input_args_override,
        )


def load_pipeline_from_file(file_path: str) -> Pipeline:
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Could not load pipeline module from {file_path}')

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    pipelines = [value for value in module.__dict__.values() if isinstance(value, Pipeline)]
    if not pipelines:
        raise ValueError(f'No Pipeline instance found in {file_path}')
    if len(pipelines) > 1:
        raise ValueError(f'Multiple Pipeline instances found in {file_path}; expose only one')
    return pipelines[0]
