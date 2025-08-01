import os
import uuid
from typing import Any, Dict, Generator, List

import yaml
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.resources_pb2 import Concept, ConceptRelation
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.json_format import MessageToDict

from clarifai.client.base import BaseClient
from clarifai.client.dataset import Dataset
from clarifai.client.input import Inputs
from clarifai.client.lister import Lister
from clarifai.client.model import Model
from clarifai.client.module import Module
from clarifai.client.pipeline import Pipeline
from clarifai.client.pipeline_step import PipelineStep
from clarifai.client.search import Search
from clarifai.client.workflow import Workflow
from clarifai.constants.model import TRAINABLE_MODEL_TYPES
from clarifai.errors import UserError
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.utils.constants import DEFAULT_BASE
from clarifai.utils.logging import display_concept_relations_tree, display_workflow_tree, logger
from clarifai.utils.misc import concept_relations_accumulation
from clarifai.workflows.utils import get_yaml_output_info_proto, is_same_yaml_model
from clarifai.workflows.validate import validate


class App(Lister, BaseClient):
    """App is a class that provides access to Clarifai API endpoints related to App information."""

    def __init__(
        self,
        url: str = None,
        app_id: str = None,
        user_id: str = None,
        base_url: str = DEFAULT_BASE,
        pat: str = None,
        token: str = None,
        root_certificates_path: str = None,
        **kwargs,
    ):
        """Initializes an App object.

        Args:
            url (str): The URL to initialize the app object.
            app_id (str): The App ID for the App to interact with.
            base_url (str): Base API url. Default "https://api.clarifai.com"
            pat (str): A personal access token for authentication. Can be set as env var CLARIFAI_PAT
            token (str): A session token for authentication. Accepts either a session token or a pat. Can be set as env var CLARIFAI_SESSION_TOKEN
            root_certificates_path (str): Path to the SSL root certificates file, used to establish secure gRPC connections.
            **kwargs: Additional keyword arguments to be passed to the App.
                - name (str): The name of the app.
                - description (str): The description of the app.
        """
        if url and app_id:
            raise UserError("You can only specify one of url or app_id.")
        if url:
            user_id, app_id = ClarifaiUrlHelper.split_clarifai_app_url(url)
            kwargs = {'user_id': user_id}
        if user_id:
            kwargs = {'user_id': user_id}

        self.kwargs = {**kwargs, 'id': app_id}
        self.app_info = resources_pb2.App(**self.kwargs)
        self.logger = logger
        BaseClient.__init__(
            self,
            user_id=self.user_id,
            app_id=self.id,
            base=base_url,
            pat=pat,
            token=token,
            root_certificates_path=root_certificates_path,
        )
        Lister.__init__(self)

    def list_datasets(
        self, page_no: int = None, per_page: int = None
    ) -> Generator[Dataset, None, None]:
        """Lists all the datasets for the app.

        Args:
            page_no (int): The page number to list.
            per_page (int): The number of items per page.

        Yields:
            Dataset: Dataset objects for the datasets in the app.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> all_datasets = list(app.list_datasets())

        Note:
            Defaults to 16 per page if page_no is specified and per_page is not specified.
            If both page_no and per_page are None, then lists all the resources.
        """
        request_data = dict(
            user_app_id=self.user_app_id,
        )
        all_datasets_info = self.list_pages_generator(
            self.STUB.ListDatasets,
            service_pb2.ListDatasetsRequest,
            request_data,
            per_page=per_page,
            page_no=page_no,
        )
        for dataset_info in all_datasets_info:
            if 'version' in dataset_info:
                dataset_info['version'].pop('metrics', None)
                dataset_info['version'].pop('export_info', None)
            yield Dataset.from_auth_helper(auth=self.auth_helper, **dataset_info)

    def list_models(
        self,
        filter_by: Dict[str, Any] = {},
        only_in_app: bool = True,
        page_no: int = None,
        per_page: int = None,
    ) -> Generator[Model, None, None]:
        """Lists all the available models for the user.

        Args:
            filter_by (dict): A dictionary of filters to apply to the list of models.
            only_in_app (bool): If True, only return models that are in the app.
            page_no (int): The page number to list.
            per_page (int): The number of items per page.

        Yields:
            Model: Model objects for the models in the app.

        Example:
            >>> from clarifai.client.user import User
            >>> app = User(user_id="user_id").app(app_id="app_id")
            >>> all_models = list(app.list_models())

        Note:
            Defaults to 16 per page if page_no is specified and per_page is not specified.
            If both page_no and per_page are None, then lists all the resources.
        """
        request_data = dict(user_app_id=self.user_app_id, **filter_by)
        all_models_info = self.list_pages_generator(
            self.STUB.ListModels,
            service_pb2.ListModelsRequest,
            request_data,
            per_page=per_page,
            page_no=page_no,
        )

        for model_info in all_models_info:
            if 'model_version' not in list(model_info.keys()):
                continue
            if only_in_app:
                if model_info['app_id'] != self.id:
                    continue
            yield Model.from_auth_helper(auth=self.auth_helper, **model_info)

    def list_workflows(
        self,
        filter_by: Dict[str, Any] = {},
        only_in_app: bool = True,
        page_no: int = None,
        per_page: int = None,
    ) -> Generator[Workflow, None, None]:
        """Lists all the available workflows for the user.

        Args:
            filter_by (dict): A dictionary of filters to apply to the list of workflows.
            only_in_app (bool): If True, only return workflows that are in the app.
            page_no (int): The page number to list.
            per_page (int): The number of items per page.

        Yields:
            Workflow: Workflow objects for the workflows in the app.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> all_workflows = list(app.list_workflows())

        Note:
            Defaults to 16 per page if page_no is specified and per_page is not specified.
            If both page_no and per_page are None, then lists all the resources.
        """
        request_data = dict(user_app_id=self.user_app_id, **filter_by)
        all_workflows_info = self.list_pages_generator(
            self.STUB.ListWorkflows,
            service_pb2.ListWorkflowsRequest,
            request_data,
            per_page=per_page,
            page_no=page_no,
        )

        for workflow_info in all_workflows_info:
            if only_in_app:
                if workflow_info['app_id'] != self.id:
                    continue
            yield Workflow.from_auth_helper(auth=self.auth_helper, **workflow_info)

    def list_pipelines(
        self,
        filter_by: Dict[str, Any] = {},
        only_in_app: bool = True,
        page_no: int = None,
        per_page: int = None,
    ) -> Generator[dict, None, None]:
        """Lists all the pipelines for the user.

        Args:
            filter_by (dict): A dictionary of filters to apply to the list of pipelines.
            only_in_app (bool): If True, only return pipelines that are in the app.
            page_no (int): The page number to list.
            per_page (int): The number of items per page.

        Yields:
            Dict: Dictionaries containing information about the pipelines.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> all_pipelines = list(app.list_pipelines())

        Note:
            Defaults to 16 per page if page_no is specified and per_page is not specified.
            If both page_no and per_page are None, then lists all the resources.
        """
        request_data = dict(user_app_id=self.user_app_id, **filter_by)
        all_pipelines_info = self.list_pages_generator(
            self.STUB.ListPipelines,
            service_pb2.ListPipelinesRequest,
            request_data,
            per_page=per_page,
            page_no=page_no,
        )

        for pipeline_info in all_pipelines_info:
            pipeline = self._process_pipeline_info(
                pipeline_info, self.auth_helper, self.id, only_in_app
            )
            if pipeline is not None:
                yield pipeline

    @staticmethod
    def _process_pipeline_info(pipeline_info, auth_helper, app_id=None, only_in_app=False):
        """Helper method to process pipeline info and create Pipeline objects.

        Args:
            pipeline_info: Raw pipeline info from API
            auth_helper: Auth helper instance
            app_id: App ID to filter by (if only_in_app is True)
            only_in_app: Whether to filter by app_id

        Returns:
            Pipeline object or None if filtered out
        """
        if only_in_app and app_id:
            if pipeline_info.get('app_id') != app_id:
                return None

        # Map API field names to constructor parameter names
        pipeline_kwargs = pipeline_info.copy()
        if 'id' in pipeline_kwargs:
            pipeline_kwargs['pipeline_id'] = pipeline_kwargs.pop('id')
        if 'pipeline_version' in pipeline_kwargs:
            pipeline_version = pipeline_kwargs.pop('pipeline_version')
            pipeline_kwargs['pipeline_version_id'] = pipeline_version.get('id', '')

        return Pipeline.from_auth_helper(auth=auth_helper, **pipeline_kwargs)

    @staticmethod
    def _process_pipeline_step_info(
        pipeline_step_info, auth_helper, app_id=None, only_in_app=False
    ):
        """Helper method to process pipeline step info and create PipelineStep objects.

        Args:
            pipeline_step_info: Raw pipeline step info from API
            auth_helper: Auth helper instance
            app_id: App ID to filter by (if only_in_app is True)
            only_in_app: Whether to filter by app_id

        Returns:
            PipelineStep object or None if filtered out
        """
        if only_in_app and app_id:
            if pipeline_step_info.get('app_id') != app_id:
                return None

        # Map API field names to constructor parameter names
        step_kwargs = pipeline_step_info.copy()
        if 'pipeline_step' in step_kwargs:
            pipeline_step = step_kwargs.pop('pipeline_step')
            step_kwargs['pipeline_step_id'] = pipeline_step.get('id', '')

        return PipelineStep.from_auth_helper(auth=auth_helper, **step_kwargs)

    def list_pipeline_steps(
        self,
        pipeline_id: str = None,
        filter_by: Dict[str, Any] = {},
        only_in_app: bool = True,
        page_no: int = None,
        per_page: int = None,
    ) -> Generator[dict, None, None]:
        """Lists all the pipeline steps for the user.

        Args:
            pipeline_id (str): If provided, only list pipeline steps from this pipeline.
            filter_by (dict): A dictionary of filters to apply to the list of pipeline steps.
            only_in_app (bool): If True, only return pipeline steps that are in the app.
            page_no (int): The page number to list.
            per_page (int): The number of items per page.

        Yields:
            Dict: Dictionaries containing information about the pipeline steps.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> all_pipeline_steps = list(app.list_pipeline_steps())

        Note:
            Defaults to 16 per page if page_no is specified and per_page is not specified.
            If both page_no and per_page are None, then lists all the resources.
        """
        request_data = dict(user_app_id=self.user_app_id, **filter_by)
        if pipeline_id:
            request_data['pipeline_id'] = pipeline_id

        all_pipeline_steps_info = self.list_pages_generator(
            self.STUB.ListPipelineStepVersions,
            service_pb2.ListPipelineStepVersionsRequest,
            request_data,
            per_page=per_page,
            page_no=page_no,
        )

        for pipeline_step_info in all_pipeline_steps_info:
            pipeline_step = self._process_pipeline_step_info(
                pipeline_step_info, self.auth_helper, self.id, only_in_app
            )
            if pipeline_step is not None:
                yield pipeline_step

    def list_modules(
        self,
        filter_by: Dict[str, Any] = {},
        only_in_app: bool = True,
        page_no: int = None,
        per_page: int = None,
    ) -> Generator[Module, None, None]:
        """Lists all the available modules for the user.

        Args:
            filter_by (dict): A dictionary of filters to apply to the list of modules.
            only_in_app (bool): If True, only return modules that are in the app.
            page_no (int): The page number to list.
            per_page (int): The number of items per page.

        Yields:
            Module: Module objects for the modules in the app.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> all_modules = list(app.list_modules())

        Note:
            Defaults to 16 per page if page_no is specified and per_page is not specified.
            If both page_no and per_page are None, then lists all the resources.
        """
        request_data = dict(user_app_id=self.user_app_id, **filter_by)
        all_modules_info = self.list_pages_generator(
            self.STUB.ListModules,
            service_pb2.ListModulesRequest,
            request_data,
            per_page=per_page,
            page_no=page_no,
        )

        for module_info in all_modules_info:
            if only_in_app:
                if module_info['app_id'] != self.id:
                    continue
            yield Module.from_auth_helper(auth=self.auth_helper, **module_info)

    def list_installed_module_versions(
        self, filter_by: Dict[str, Any] = {}, page_no: int = None, per_page: int = None
    ) -> Generator[Module, None, None]:
        """Lists all installed module versions in the app.

        Args:
            filter_by (dict): A dictionary of filters to apply to the list of installed module versions.
            page_no (int): The page number to list.
            per_page (int): The number of items per page.

        Yields:
            Module: Module objects for the installed module versions in the app.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> all_installed_module_versions = list(app.list_installed_module_versions())

        Note:
            Defaults to 16 per page if page_no is specified and per_page is not specified.
            If both page_no and per_page are None, then lists all the resources.
        """
        request_data = dict(user_app_id=self.user_app_id, **filter_by)
        all_imv_infos = self.list_pages_generator(
            self.STUB.ListInstalledModuleVersions,
            service_pb2.ListInstalledModuleVersionsRequest,
            request_data,
            per_page=per_page,
            page_no=page_no,
        )
        for imv_info in all_imv_infos:
            del imv_info['deploy_url']
            del imv_info['installed_module_version_id']  # TODO: remove this after the backend fix
            yield Module.from_auth_helper(
                auth=self.auth_helper,
                module_id=imv_info['module_version']['module_id'],
                **imv_info,
            )

    def get_input_count(self) -> int:
        """Get count of all the inputs in the app.

        Returns:
            input_count: count of inputs in the app.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> input_count = app.get_input_count()
        """
        request = service_pb2.GetInputCountRequest(user_app_id=self.user_app_id)
        response = self._grpc_request(self.STUB.GetInputCount, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info("\nGetting App input Counts\n%s", response.status)

        return response.counts.processed

    def list_concepts(
        self, page_no: int = None, per_page: int = None
    ) -> Generator[Concept, None, None]:
        """Lists all the concepts for the app.
        Args:
            page_no (int): The page number to list.
            per_page (int): The number of items per page.

        Yields:
            Concept: Concepts in the app.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> all_concepts = list(app.list_concepts())

        Note:
            Defaults to 16 per page if page_no is specified and per_page is not specified.
            If both page_no and per_page are None, then lists all the resources.
        """
        request_data = dict(user_app_id=self.user_app_id)
        all_concepts_infos = self.list_pages_generator(
            self.STUB.ListConcepts,
            service_pb2.ListConceptsRequest,
            request_data,
            per_page=per_page,
            page_no=page_no,
        )
        for concept_info in all_concepts_infos:
            concept_info['id'] = concept_info.pop('concept_id')
            yield Concept(**concept_info)

    def search_concept_relations(
        self,
        concept_id: str = None,
        predicate: str = None,
        page_no: int = None,
        per_page: int = None,
        show_tree: bool = False,
    ) -> Generator[ConceptRelation, None, None]:
        """List all the concept relations of the app.

        Args:
            concept_id (str, optional): The concept ID to filter the concept relations.
            predicate (str, optional): Type of relation to filter the concept relations. For ex : 'hypernym', 'hyponym' or 'synonym'
            page_no (int, optional): The page number to list.
            per_page (int, optional): The number of items per page.
            show_tree (bool, optional): If True, prints out rich tree representation of concept relations.

        Yields:
            ConceptRelation: ConceptRelations in the app.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> all_concept_relations = list(app.search_concept_relations())

        Note:
            Defaults to 16 per page if page_no is specified and per_page is not specified.
            If both page_no and per_page are None, then lists all the resources.
        """
        request_data = dict(
            user_app_id=self.user_app_id, concept_id=concept_id, predicate=predicate
        )
        all_concept_relations_infos = self.list_pages_generator(
            self.STUB.ListConceptRelations,
            service_pb2.ListConceptRelationsRequest,
            request_data,
            per_page=per_page,
            page_no=page_no,
        )
        relations_dict = {}
        for concept_relation_info in all_concept_relations_infos:
            if show_tree:
                current_subject_concept = concept_relation_info['subject_concept']['id']
                current_object_concept = concept_relation_info['object_concept']['id']
                current_predicate = concept_relation_info['predicate']
                relations_dict = concept_relations_accumulation(
                    relations_dict,
                    current_subject_concept,
                    current_object_concept,
                    current_predicate,
                )
            concept_relation_info['id'] = concept_relation_info.pop('concept_relation_id')
            yield ConceptRelation(**concept_relation_info)
        if show_tree:
            display_concept_relations_tree(relations_dict)

    def list_trainable_model_types(self) -> List[str]:
        """Lists all the trainable model types.

        Returns:
            templates (List): List all the trainable model types.

        Example:
            >>> from clarifai.client.app import App
            >>> print(app.list_trainable_model_types())
        """
        return TRAINABLE_MODEL_TYPES

    def create_dataset(self, dataset_id: str, **kwargs) -> Dataset:
        """Creates a dataset for the app.

        Args:
            dataset_id (str): The dataset ID for the dataset to create.
            **kwargs: Additional keyword arguments to be passed to the Dataset.

        Returns:
            Dataset: A Dataset object for the specified dataset ID.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> dataset = app.create_dataset(dataset_id="dataset_id")
        """
        request = service_pb2.PostDatasetsRequest(
            user_app_id=self.user_app_id, datasets=[resources_pb2.Dataset(id=dataset_id, **kwargs)]
        )
        response = self._grpc_request(self.STUB.PostDatasets, request)
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info("\nDataset created\n%s", response.status)

        return Dataset.from_auth_helper(self.auth_helper, dataset_id=dataset_id, **kwargs)

    def create_model(self, model_id: str, **kwargs) -> Model:
        """Creates a model for the app.

        Args:
            model_id (str): The model ID for the model to create.
            **kwargs: Additional keyword arguments to be passed to the Model.

        Returns:
            Model: A Model object for the specified model ID.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> model = app.create_model(model_id="model_id")
        """
        request = service_pb2.PostModelsRequest(
            user_app_id=self.user_app_id, models=[resources_pb2.Model(id=model_id, **kwargs)]
        )
        response = self._grpc_request(self.STUB.PostModels, request)
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info(f"Model with ID '{model_id}' is created:\n{response.status}")
        kwargs.update(
            {
                'model_id': model_id,
                'model_type_id': response.model.model_type_id,
            }
        )

        return Model.from_auth_helper(auth=self.auth_helper, **kwargs)

    def _process_workflow_config(self, config_filepath: str):
        with open(config_filepath, 'r') as file:
            workflow_config = yaml.safe_load(file)

        workflow_config = validate(workflow_config)
        workflow = workflow_config['workflow']

        # Get all model objects from the workflow nodes.
        all_models = []
        for node in workflow['nodes']:
            output_info = get_yaml_output_info_proto(node['model'].get('output_info', None))
            try:
                model = self.model(
                    model_id=node['model']['model_id'],
                    model_version={"id": node['model'].get('model_version_id', "")},
                    user_id=node['model'].get('user_id', self.user_app_id.user_id),
                    app_id=node['model'].get('app_id', self.user_app_id.app_id),
                )
            except Exception as e:
                if "Model does not exist" in str(e):
                    model = self.create_model(
                        **{k: v for k, v in node['model'].items() if k != 'output_info'}
                    )
                    model_version = model.create_version(output_info=output_info)
                    all_models.append(model_version.model_info)
                    continue

            # If the model version ID is specified, or if the yaml model is the same as the one in the api
            if node["model"].get("model_version_id", "") or is_same_yaml_model(
                model.model_info, node["model"]
            ):
                all_models.append(model.model_info)
            else:  # Create a new model version
                model = model.create_version(output_info=output_info)
                all_models.append(model.model_info)

        # Convert nodes to resources_pb2.WorkflowNodes.
        nodes = []
        for i, yml_node in enumerate(workflow['nodes']):
            node = resources_pb2.WorkflowNode(
                id=yml_node['id'],
                model=all_models[i],
            )
            # Add node inputs if they exist, i.e. if these nodes do not connect directly to the input.
            if yml_node.get("node_inputs"):
                for ni in yml_node.get("node_inputs"):
                    node.node_inputs.append(resources_pb2.NodeInput(node_id=ni['node_id']))
            nodes.append(node)

        return workflow, nodes

    def create_workflow(
        self, config_filepath: str, generate_new_id: bool = False, display: bool = True
    ) -> Workflow:
        """Creates a workflow for the app.

        Args:
            config_filepath (str): The path to the yaml workflow config file.
            generate_new_id (bool): If True, generate a new workflow ID.
            display (bool): If True, display the workflow nodes tree.

        Returns:
            Workflow: A Workflow object for the specified workflow config.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> workflow = app.create_workflow(config_filepath="config.yml")
        """
        if not os.path.exists(config_filepath):
            raise UserError(f"Workflow config file not found at {config_filepath}")

        workflow, nodes = self._process_workflow_config(config_filepath)

        workflow_id = workflow['id']
        if generate_new_id:
            workflow_id = str(uuid.uuid4())

        # Create the workflow.
        request = service_pb2.PostWorkflowsRequest(
            user_app_id=self.user_app_id,
            workflows=[resources_pb2.Workflow(id=workflow_id, nodes=nodes)],
        )

        response = self._grpc_request(self.STUB.PostWorkflows, request)
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info("\nWorkflow created\n%s", response.status)

        dict_response = MessageToDict(response, preserving_proto_field_name=True)
        # Display the workflow nodes tree.
        wf = dict_response["workflows"][0]
        if display:
            display_workflow_tree(wf["nodes"])
        kwargs = self.process_response_keys(wf, "workflow")

        return Workflow.from_auth_helper(auth=self.auth_helper, **kwargs)

    def create_module(self, module_id: str, description: str, **kwargs) -> Module:
        """Creates a module for the app.

        Args:
            module_id (str): The module ID for the module to create.
            description (str): The description of the module to create.
            **kwargs: Additional keyword arguments to be passed to the module.

        Returns:
            Module: A Module object for the specified module ID.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> module = app.create_module(module_id="module_id")
        """
        request = service_pb2.PostModulesRequest(
            user_app_id=self.user_app_id,
            modules=[resources_pb2.Module(id=module_id, description=description, **kwargs)],
        )
        response = self._grpc_request(self.STUB.PostModules, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info("\nModule created\n%s", response.status)

        return Module.from_auth_helper(auth=self.auth_helper, module_id=module_id, **kwargs)

    def create_concepts(self, concept_ids: List[str], concepts: List[str] = []) -> None:
        """Add concepts to the app.

        Args:
            concept_ids (List[str]): List of concept IDs of concepts to add to the app.
            concepts (List[str], optional): List of concepts names of concepts to add to the app.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> app.add_concepts(concept_ids=["concept_id_1", "concept_id_2", ..])
        """
        if not concepts:
            concepts = list(concept_ids)
        concepts_to_add = [
            resources_pb2.Concept(id=concept_id, name=concept)
            for concept_id, concept in zip(concept_ids, concepts)
        ]
        request = service_pb2.PostConceptsRequest(
            user_app_id=self.user_app_id, concepts=concepts_to_add
        )
        response = self._grpc_request(self.STUB.PostConcepts, request)
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info("\nConcepts added\n%s", response.status)

    def create_concept_relations(
        self, subject_concept_id: str, object_concept_ids: List[str], predicates: List[str]
    ) -> None:
        """Creates concept relations between concepts of the app.

        Args:
            subject_concept_id (str): The concept ID of the subject concept.
            object_concept_ids (List[str]): List of concepts IDs of object concepts.
            predicates (List[str]): List of predicates corresponding to each relation between subject and objects.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> app.create_concept_relation(subject_concept_id="subject_concept_id", object_concept_ids=["object_concept_id_1", "object_concept_id_2", ..], predicates=["predicate_1", "predicate_2", ..])
        """
        assert len(object_concept_ids) == len(predicates)
        subject_relations = [
            resources_pb2.ConceptRelation(
                object_concept=resources_pb2.Concept(id=object_concept_id), predicate=predicate
            )
            for object_concept_id, predicate in zip(object_concept_ids, predicates)
        ]
        request = service_pb2.PostConceptRelationsRequest(
            user_app_id=self.user_app_id,
            concept_id=subject_concept_id,
            concept_relations=subject_relations,
        )
        response = self._grpc_request(self.STUB.PostConceptRelations, request)
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info("\nConcept Relations created\n%s", response.status)

    def dataset(self, dataset_id: str, dataset_version_id: str = None, **kwargs) -> Dataset:
        """Returns a Dataset object for the existing dataset ID.

        Args:
            dataset_id (str): The dataset ID for the dataset to interact with.
            dataset_version_id (str): The version ID for the dataset version to interact with.

        Returns:
            Dataset: A Dataset object for the existing dataset ID.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> dataset = app.dataset(dataset_id="dataset_id")
        """
        request = service_pb2.GetDatasetRequest(
            user_app_id=self.user_app_id, dataset_id=dataset_id
        )
        response = self._grpc_request(self.STUB.GetDataset, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        dict_response = MessageToDict(response, preserving_proto_field_name=True)
        kwargs = self.process_response_keys(dict_response['dataset'], 'dataset')
        kwargs['version'] = response.dataset.version if response.dataset.version else None
        kwargs['dataset_version_id'] = dataset_version_id
        return Dataset.from_auth_helper(auth=self.auth_helper, **kwargs)

    def model(self, model_id: str, model_version: Dict = {'id': ""}, **kwargs) -> Model:
        """Returns a Model object for the existing model ID.

        Args:
            model_id (str): The model ID for the model to interact with.
            model_version (Dict): The model version ID for the model version to interact with.

        Returns:
            Model: A Model object for the existing model ID.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> model_v1 = app.model(model_id="model_id", model_version={"id":"model_version_id")
        """
        # Change user_app_id based on whether user_id or app_id is specified.
        if kwargs.get("user_id") or kwargs.get("app_id"):
            request = service_pb2.GetModelRequest(
                user_app_id=self.auth_helper.get_user_app_id_proto(
                    kwargs.get("user_id"), kwargs.get("app_id")
                ),
                model_id=model_id,
                version_id=model_version["id"],
            )
        else:
            request = service_pb2.GetModelRequest(
                user_app_id=self.user_app_id, model_id=model_id, version_id=model_version["id"]
            )
        response = self._grpc_request(self.STUB.GetModel, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        dict_response = MessageToDict(response, preserving_proto_field_name=True)

        kwargs = self.process_response_keys(dict_response['model'], 'model')
        kwargs['model_version'] = (
            response.model.model_version if response.model.model_version else None
        )

        m = Model.from_auth_helper(self.auth_helper, **kwargs)
        return m

    def workflow(self, workflow_id: str, **kwargs) -> Workflow:
        """Returns a workflow object for the existing workflow ID.

        Args:
            workflow_id (str): The workflow ID for the workflow to interact with.

        Returns:
            Workflow: A Workflow object for the existing workflow ID.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> workflow = app.workflow(workflow_id="workflow_id")
        """
        request = service_pb2.GetWorkflowRequest(
            user_app_id=self.user_app_id, workflow_id=workflow_id
        )
        response = self._grpc_request(self.STUB.GetWorkflow, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        dict_response = MessageToDict(response, preserving_proto_field_name=True)
        kwargs = self.process_response_keys(dict_response['workflow'], "workflow")

        return Workflow.from_auth_helper(auth=self.auth_helper, **kwargs)

    def module(self, module_id: str, **kwargs) -> Module:
        """Returns a Module object for the existing module ID.

        Args:
            module_id (str): The module ID for the module to interact with.
            module_version_id (str): The module version ID for the module version to interact with.

        Returns:
            Module: A Module object for the existing module ID.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> module = app.module(module_id="module_id")
        """
        request = service_pb2.GetModuleRequest(user_app_id=self.user_app_id, module_id=module_id)
        response = self._grpc_request(self.STUB.GetModule, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        dict_response = MessageToDict(response, preserving_proto_field_name=True)
        kwargs = self.process_response_keys(dict_response['module'], 'module')

        return Module.from_auth_helper(auth=self.auth_helper, **kwargs)

    def inputs(
        self,
    ):
        """Returns an Input object.

        Returns:
            Inputs: An input object.
        """
        return Inputs.from_auth_helper(self.auth_helper)

    def patch_dataset(self, dataset_id: str, action: str = 'merge', **kwargs) -> Dataset:
        """Patches a dataset for the app.

        Args:
            dataset_id (str): The dataset ID for the dataset to create.
            action (str): The action to perform on the dataset (merge/overwrite/remove).
            **kwargs: Additional keyword arguments to be passed to patch the Dataset.

        Returns:
            Dataset: A Dataset object for the specified dataset ID.
        """
        if "visibility" in kwargs:
            kwargs["visibility"] = resources_pb2.Visibility(gettable=kwargs["visibility"])
        if "image_url" in kwargs:
            kwargs["image"] = resources_pb2.Image(url=kwargs.pop("image_url"))
        request = service_pb2.PatchDatasetsRequest(
            user_app_id=self.user_app_id,
            datasets=[resources_pb2.Dataset(id=dataset_id, **kwargs)],
            action=action,
        )
        response = self._grpc_request(self.STUB.PatchDatasets, request)
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info("\nDataset patched\n%s", response.status)

        return Dataset.from_auth_helper(self.auth_helper, dataset_id=dataset_id, **kwargs)

    def patch_model(self, model_id: str, action: str = 'merge', **kwargs) -> Model:
        """Patches a model of the app.

        Args:
            model_id (str): The model ID of the model to patch.
            action (str): The action to perform on the model (merge/overwrite/remove).
            **kwargs: Additional keyword arguments to be passed to patch the Model.

        Returns:
            Model: A Model object of the specified model ID.
        """
        if "visibility" in kwargs:
            kwargs["visibility"] = resources_pb2.Visibility(gettable=kwargs["visibility"])
        if "image_url" in kwargs:
            kwargs["image"] = resources_pb2.Image(url=kwargs.pop("image_url"))
        request = service_pb2.PatchModelsRequest(
            user_app_id=self.user_app_id,
            models=[resources_pb2.Model(id=model_id, **kwargs)],
            action=action,
        )
        response = self._grpc_request(self.STUB.PatchModels, request)
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info("\nModel %s patched successfully\n%s", model_id, response.status)
        kwargs.update(
            {
                'model_id': model_id,
            }
        )

        return Model.from_auth_helper(self.auth_helper, **kwargs)

    def patch_workflow(
        self, workflow_id: str, action: str = 'merge', config_filepath: str = None, **kwargs
    ) -> Workflow:
        """Patches a workflow of the app.

        Args:
            workflow_id (str): The Workflow ID of the workflow to patch.
            action (str): The action to perform on the workflow (merge/overwrite/remove).
            config_filepath (str): The path to the yaml workflow config file.
            **kwargs: Additional keyword arguments to be passed to patch the Workflow.

        Returns:
            Workflow: A Workflow object of the specified workflow ID.
        """
        if config_filepath:
            if not os.path.exists(config_filepath):
                raise UserError(f"Workflow config file not found at {config_filepath}")
            _, kwargs['nodes'] = self._process_workflow_config(config_filepath)
        if "visibility" in kwargs:
            kwargs["visibility"] = resources_pb2.Visibility(gettable=kwargs["visibility"])
        if "image_url" in kwargs:
            kwargs["image"] = resources_pb2.Image(url=kwargs.pop("image_url"))

        request = service_pb2.PatchWorkflowsRequest(
            user_app_id=self.user_app_id,
            workflows=[resources_pb2.Workflow(id=workflow_id, **kwargs)],
            action=action,
        )
        response = self._grpc_request(self.STUB.PatchWorkflows, request)
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info("\nWorkflow patched\n%s", response.status)
        kwargs.update(
            {
                'workflow_id': workflow_id,
            }
        )

        return Workflow.from_auth_helper(self.auth_helper, **kwargs)

    def delete_dataset(self, dataset_id: str) -> None:
        """Deletes an dataset for the user.

        Args:
            dataset_id (str): The dataset ID for the app to delete.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> app.delete_dataset(dataset_id="dataset_id")
        """
        request = service_pb2.DeleteDatasetsRequest(
            user_app_id=self.user_app_id, dataset_ids=[dataset_id]
        )
        response = self._grpc_request(self.STUB.DeleteDatasets, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info("\nDataset Deleted\n%s", response.status)

    def delete_model(self, model_id: str) -> None:
        """Deletes an model for the user.

        Args:
            model_id (str): The model ID for the app to delete.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> app.delete_model(model_id="model_id")
        """
        request = service_pb2.DeleteModelsRequest(user_app_id=self.user_app_id, ids=[model_id])
        response = self._grpc_request(self.STUB.DeleteModels, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info("\nModel Deleted\n%s", response.status)

    def delete_workflow(self, workflow_id: str) -> None:
        """Deletes an workflow for the user.

        Args:
            workflow_id (str): The workflow ID for the app to delete.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> app.delete_workflow(workflow_id="workflow_id")
        """
        request = service_pb2.DeleteWorkflowsRequest(
            user_app_id=self.user_app_id, ids=[workflow_id]
        )
        response = self._grpc_request(self.STUB.DeleteWorkflows, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info("\nWorkflow Deleted\n%s", response.status)

    def delete_module(self, module_id: str) -> None:
        """Deletes an module for the user.

        Args:
            module_id (str): The module ID for the app to delete.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> app.delete_module(module_id="module_id")
        """
        request = service_pb2.DeleteModulesRequest(user_app_id=self.user_app_id, ids=[module_id])
        response = self._grpc_request(self.STUB.DeleteModules, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info("\nModule Deleted\n%s", response.status)

    def delete_concept_relations(
        self, concept_id: str, concept_relation_ids: List[str] = []
    ) -> None:
        """Delete concept relations of a concept of the app.

        Args:
            concept_id (str): The concept ID of the concept to delete relations for.
            concept_relation_ids (List[str], optional): List of concept relation IDs of concept relations.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> app.delete_concept_relations(concept_id="concept_id", concept_relation_ids=["concept_relation_id_1", "concept_relation_id_2", ..])
        """
        if not concept_relation_ids:
            concept_relation_ids = [
                concept_relation.id
                for concept_relation in list(self.search_concept_relations(concept_id=concept_id))
            ]
        request = service_pb2.DeleteConceptRelationsRequest(
            user_app_id=self.user_app_id, concept_id=concept_id, ids=concept_relation_ids
        )
        response = self._grpc_request(self.STUB.DeleteConceptRelations, request)
        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(response.status)
        self.logger.info("\nConcept Relations Deleted\n%s", response.status)

    def search(self, **kwargs) -> Model:
        """Returns a Search object for the user and app ID.

        Args:
            see the Search class in clarifai.client.search for kwargs.

        Returns:
            Search: A Search object for the user and app ID.

        Example:
            >>> from clarifai.client.app import App
            >>> app = App(app_id="app_id", user_id="user_id")
            >>> search_client = app.search(top_k=12, metric="euclidean")
        """
        kwargs.get("user_id", self.user_app_id.user_id)
        kwargs.get("app_id", self.user_app_id.app_id)
        return Search.from_auth_helper(auth=self.auth_helper, **kwargs)

    def __getattr__(self, name):
        return getattr(self.app_info, name)

    def __str__(self):
        init_params = [param for param in self.kwargs.keys()]
        attribute_strings = [
            f"{param}={getattr(self.app_info, param)}"
            for param in init_params
            if hasattr(self.app_info, param)
        ]
        return f"Clarifai App Details: \n{', '.join(attribute_strings)}\n"
