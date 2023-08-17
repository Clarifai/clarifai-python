from typing import Union

from clarifai_grpc.grpc.api import resources_pb2, service_pb2  # noqa: F401
from clarifai_grpc.grpc.api.resources_pb2 import Text

from clarifai.client.input import Inputs


class Text(Inputs):
  """
    Text is a class that provides access to protos related to Text.
    Inherits from Inputs for input methods.
    """

  def __init__(self, user_id: str = '', app_id: str = '', logger_level: str = "INFO"):
    """Initializes an Input object.
        Args:
            user_id (str): A user ID for authentication.
            app_id (str): An app ID for the application to interact with.
        """
    self.user_id = user_id
    self.app_id = app_id
    super().__init__(user_id=self.user_id, app_id=self.app_id, logger_level=logger_level)

  def _get_proto(self, input_id: str, dataset_id: Union[str, None], textpb: resources_pb2.Text,
                 **kwargs) -> Text:
    """Create input proto for text data type.
        Args:
            input_id (str): The input ID for the input to create.
            dataset_id (str): The dataset ID for the dataset to add the input to.
            textpb (resources_pb2.Text): The text proto to be used for the input.
            **kwargs: Additional keyword arguments to be passed to the Input
                - geo_info (list): A list of longitude and latitude for the geo point.
                - labels (list): A list of labels for the input.
                - metadata (Struct): A Struct of metadata for the input.
        Returns:
            Text: An Text object for the specified input ID."""
    geo_pb = resources_pb2.Geo(geo_point=resources_pb2.GeoPoint(
        longitude=kwargs['geo_info'][0], latitude=kwargs['geo_info'][
            1])) if 'geo_info' in kwargs else None
    concepts=[
            resources_pb2.Concept(
              id=f"id-{''.join(_label.split(' '))}", name=_label, value=1.)\
            for _label in kwargs['labels']
        ]if 'labels' in kwargs else None
    metadata = kwargs['metadata'] if 'metadata' in kwargs else None

    if dataset_id:
      return resources_pb2.Input(
          id=input_id,
          dataset_ids=[dataset_id],
          data=resources_pb2.Data(text=textpb, geo=geo_pb, concepts=concepts, metadata=metadata))

    return resources_pb2.Input(
        id=input_id,
        data=resources_pb2.Data(text=textpb, geo=geo_pb, concepts=concepts, metadata=metadata))

  def get_input_from_url(self, input_id: str, url: str, dataset_id: str = None, **kwargs) -> Text:
    """Create input proto for text data type from url.
        Args:
            input_id (str): The input ID for the input to create.
            url (str): The url to be used for the input.
            dataset_id (str): The dataset ID for the dataset to add the input to.
            **kwargs: Additional keyword arguments to be passed to the Input
        Returns:
            Text: An Input object for the specified input ID.
        Example:
            >>> from clarifai.client.input.text import Text
            >>> text_obj = Text()
            >>> input_protos = text_obj.get_input_from_url(input_id = 'demo', url='https://samples.clarifai.com/featured-models/xgen-code.txt)
        """
    text_pb = resources_pb2.Text(url=url)
    return self._get_proto(input_id, dataset_id, text_pb, **kwargs)

  def get_input(self, input_id: str, raw_text: str, dataset_id: str = None, **kwargs) -> Text:
    """Create input proto for text data type from filename.
        Args:
            input_id (str): The input ID for the input to create.
            raw_text (str): The raw text input.
            dataset_id (str): The dataset ID for the dataset to add the input to.
            **kwargs: Additional keyword arguments to be passed to the Input
        Returns:
            Text: An Input object for the specified input ID.
        Example:
            >>> from clarifai.client.input.text import Text
            >>> text_obj = Text()
            >>> input_protos = text_obj.get_input(input_id = 'demo', raw_text = 'This is a test')
        """
    text_pb = resources_pb2.Text(raw=raw_text)
    return self._get_proto(input_id, dataset_id, text_pb, **kwargs)

  def upload_from_url(self, input_id: str, url: str, dataset_id: str = None, **kwargs) -> str:
    """upload text from url.
        Args:
            input_id (str): The input ID for the input to create.
            url (str): The url for the text.
            dataset_id (str): The dataset ID for the dataset to add the input to.
        Returns:
            input_job_id (str): job id for the upload request.

        input_pb = self._get_proto(input_id, dataset_id, resources_pb2.Text(url=url), **kwargs)
        return self.upload_inputs([input_pb])
        Example:
            >>> from clarifai.client.input.text import Text
            >>> text_obj = Text()
            >>> text_obj.upload_from_url(input_id = 'demo', url='https://samples.clarifai.com/featured-models/xgen-code.txt)
        """

  def upload(self, input_id: str, raw_text: str, dataset_id: str = None, **kwargs) -> str:
    """upload text from raw text.
        Args:
            input_id (str): The input ID for the input to create.
            raw_text (str): The raw text.
            dataset_id (str): The dataset ID for the dataset to add the input to.
        Returns:
            input_job_id (str): job id for the upload request.
        Example:
            >>> from clarifai.client.input.text import Text
            >>> text_obj = Text()
            >>> text_obj.upload(input_id = 'demo', raw_text = 'This is a test')
        """
    input_pb = self._get_proto(input_id, dataset_id, resources_pb2.Text(raw=raw_text), **kwargs)
    return self.upload_inputs([input_pb])
