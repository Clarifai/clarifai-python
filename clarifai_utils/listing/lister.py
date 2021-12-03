from clarifai_grpc.grpc.api import resources_pb2
from clarifai_grpc.grpc.api.service_pb2_grpc import V2Stub

from clarifai_utils.listing.concepts import concepts_generator
from clarifai_utils.listing.inputs import inputs_generator


class ClarifaiResourceLister(object):

  def __init__(self, stub: V2Stub, metadata: tuple, user_id: str, app_id: str,
               page_size: int = 16):
    """
    Helper class for common listing of resources in an Clarifai App.

    Args:
      stub: grpc client V2Strub for our API.
      metadata: the auth metadata for the grpc stub.
      user_id: the user to list from.
      app_id: the app in the user_id account to list from.
      page_size: the pagination size to use while iterating.
    """
    self.stub = stub
    self.metadata = metadata
    self.user_id = user_id
    self.app_id = app_id
    self.user_app_id_proto = resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)
    self.default_page_size = page_size

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
    return concepts_generator(self.stub, self.metadata, self.user_id, self.app_id, page_size)

  def list_all_inputs(self, page_size: int = None):
    """
    This lists all the inputs in an app. Not recommended for large apps.

    Returns:
      inputs: a list of Input protos for all the inputs in the app.
    """
    return [item for item in self.inputs_generator(page_size)]

  def inputs_generator(self, page_size: int = None):
    """
    This lists all the concepts in an app. Not recommended for large apps.

    Returns:
      gen: a generator that yields a single Input proto at a time.
    """
    page_size = self.default_page_size if page_size is None else page_size
    return inputs_generator(self.stub, self.metadata, self.user_id, self.app_id, page_size)
