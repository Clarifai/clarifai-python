from abc import ABC, abstractmethod
from enum import Enum

from clarifai.actions.configure.configure import Configure
from clarifai.actions.datasets.datasets import Datasets
from clarifai.actions.models.models import Models
from clarifai.actions.workflows.workflows import Workflows


class Actions(Enum):
    CONFIGURE = 'configure'
    DATASETS = 'datasets'
    MODELS = 'models'
    WORKFLOWS = 'workflows'


class ActionFactory(object):
    def execute(self, args):
        action = args.action
        action_path = self._get_action_path(action)
        action_path.execute(args)

    @staticmethod
    def _get_action_path(action):
        if action == Actions.CONFIGURE.value:
            return Configure()
        elif action == Actions.DATASETS.value:
            return Datasets()
        elif action == Actions.MODELS.value:
            return Models()
        elif action == Actions.WORKFLOWS.value:
            return Workflows()

