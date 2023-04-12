from clarifai_grpc.grpc.api import resources_pb2

from clarifai.client import V2Stub
from clarifai.listing.concepts import concepts_generator
from clarifai.listing.datasets import datasets_generator
from clarifai.listing.inputs import dataset_inputs_generator, inputs_generator
from clarifai.listing.installed_module_versions import installed_module_versions_generator
from clarifai.listing.models import models_generator
from clarifai.listing.module_versions import module_versions_generator
from clarifai.listing.modules import modules_generator


class ClarifaiResourceLister(object):

  def __init__(self, stub: V2Stub, user_id: str, app_id: str, page_size: int = 16):
    """
        Helper class for common listing of resources in an Clarifai App.

        Args:
          stub: grpc client V2Strub for our API.
          user_id: the user to list from.
          app_id: the app in the user_id account to list from.
          page_size: the pagination size to use while iterating.
        """
    self.stub = stub
    self.user_id = user_id
    self.app_id = app_id
    self.user_app_id_proto = resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)
    self.default_page_size = page_size

  def list_all_models(self, page_size: int = None, only_in_app: bool = False):
    """
        This lists all the Models accessible in app. Not recommended for large apps.

        Params:
          page_size: how many elements per page to fetch
          only_in_app: if only models created in the app should be returned

        Returns:
          inputs: a list of Model protos for all the inputs in the app.
        """
    return [item for item in self.models_generator(page_size, only_in_app)]

  def models_generator(self, page_size: int = None, only_in_app: bool = False):
    """
        This lists all the models in an app. Not recommended for large apps.

        Params:
          page_size: how many elements per page to fetch
          only_in_app: if only models created in the app should be returned

        Returns:
          gen: a generator that yields a single Model proto at a time.
        """
    page_size = self.default_page_size if page_size is None else page_size
    return models_generator(self.stub, self.user_id, self.app_id, page_size, only_in_app)

  def list_all_datasets(self, page_size: int = None):
    """
        This lists all Datasets in app. Not recommended for large apps.

        Params:
          page_size: how many elements per page to fetch

        Returns:
          inputs: a list of Dataset protos for all the inputs in the app.
        """
    return [item for item in self.datasets_generator(page_size)]

  def datasets_generator(self, page_size: int = None):
    """
        This lists all the Datasets in an app. Not recommended for large apps.

        Params:
          page_size: how many elements per page to fetch

        Returns:
          gen: a generator that yields a single Dataset proto at a time.
        """
    page_size = self.default_page_size if page_size is None else page_size
    return datasets_generator(self.stub, self.user_id, self.app_id, page_size)

  def list_all_concepts(self, page_size: int = None):
    """
        This lists all the concepts in an app. Not recommended for large apps.

        Returns:
          concepts: a list of Concept protos for all the concepts in the app.
        """
    return [item for item in self.concepts_generator(page_size)]

  def concepts_generator(self, page_size: int = None):
    """
        This lists all the concepts in an app. Not recommended for large apps.

        Returns:
          gen: a generator that yields a single Concept proto at a time.
        """
    page_size = self.default_page_size if page_size is None else page_size
    return concepts_generator(self.stub, self.user_id, self.app_id, page_size)

  def list_all_inputs(self, page_size: int = None):
    """
        This lists all the inputs in an app. Not recommended for large apps.

        Returns:
          inputs: a list of Input protos for all the inputs in the app.
        """
    return [item for item in self.inputs_generator(page_size)]

  def inputs_generator(self, page_size: int = None):
    """
        This lists all the inputs in an app. Not recommended for large apps.

        Returns:
          gen: a generator that yields a single Input proto at a time.
        """
    page_size = self.default_page_size if page_size is None else page_size
    return inputs_generator(self.stub, self.user_id, self.app_id, page_size)

  def list_all_dataset_inputs(self, page_size: int = None, dataset_id: str = None):
    """
        This lists all the inputs in a dataset. Not recommended for large datasets.

        Returns:
          inputs: a list of Input protos for all the inputs in the app.
        """
    return [item for item in self.dataset_inputs_generator(page_size, dataset_id)]

  def dataset_inputs_generator(self, page_size: int = None, dataset_id: str = None):
    """
        This lists all the inputs in a dataset. Not recommended for large datasets.

        Returns:
          gen: a generator that yields a single Input proto at a time.
        """
    page_size = self.default_page_size if page_size is None else page_size
    return dataset_inputs_generator(
        stub=self.stub,
        user_id=self.user_id,
        app_id=self.app_id,
        page_size=page_size,
        dataset_id=dataset_id)

  def list_all_installed_module_versions(self, page_size: int = None):
    """
    This lists all the installed_module_versions in an app. Not recommended for large apps.

    Returns:
      installed_module_versions: a list of InstalledModuleVersion protos for all the installed_module_versions in the app.
    """
    return [item for item in self.installed_module_versions_generator(page_size)]

  def installed_module_versions_generator(self, page_size: int = None):
    """
    This lists all the installed_module_versions in an app. Not recommended for large apps.

    Returns:
      gen: a generator that yields a single InstalledModuleVersion proto at a time.
    """
    page_size = self.default_page_size if page_size is None else page_size
    return installed_module_versions_generator(self.stub, self.user_id, self.app_id, page_size)

  def list_all_modules(self, page_size: int = None):
    """
    This lists all the modules in an app. Not recommended for large apps.

    Returns:
      modules: a list of Module protos for all the modules in the app.
    """
    return [item for item in self.module_generator(page_size)]

  def module_generator(self, page_size: int = None):
    """
    This lists all the module in an app. Not recommended for large apps.

    Returns:
      gen: a generator that yields a single Module proto at a time.
    """
    page_size = self.default_page_size if page_size is None else page_size
    return modules_generator(self.stub, self.user_id, self.app_id, page_size)

  def list_all_module_versions(self, module_id: str, page_size: int = None):
    """
    This lists all the module_versions in an app. Not recommended for large apps.

    Returns:
      module_versions: a list of ModuleVersion protos for all the module_versions in the app.
    """
    return [item for item in self.module_versions_generator(module_id, page_size)]

  def module_versions_generator(self, module_id: str, page_size: int = None):
    """
    This lists all the module_versions in an app. Not recommended for large apps.

    Returns:
      gen: a generator that yields a single ModuleVersion proto at a time.
    """
    page_size = self.default_page_size if page_size is None else page_size
    return module_versions_generator(self.stub, self.user_id, self.app_id, module_id, page_size)
