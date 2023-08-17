import os
from typing import Union

from clarifai_grpc.grpc.api import resources_pb2, service_pb2  # noqa: F401
from clarifai_grpc.grpc.api.resources_pb2 import Audio

from clarifai.client.input import Inputs


class Audio(Inputs):
  """
    Audio is a class that provides access to protos related to Audio.
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

  def _get_proto(self, input_id: str, dataset_id: Union[str, None], audiopb: resources_pb2.Audio,
                 **kwargs) -> Audio:
    """Create input proto for audio data type.
        Args:
            input_id (str): The input ID for the input to create.
            dataset_id (str): The dataset ID for the dataset to add the input to.
            audiopb (resources_pb2.Audio): The audio proto to be used for the input.
            **kwargs: Additional keyword arguments to be passed to the Input
                - geo_info (list): A list of longitude and latitude for the geo point.
                - labels (list): A list of labels for the input.
                - metadata (Struct): A Struct of metadata for the input.
        Returns:
            Audio: An Audio object for the specified input ID."""
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
          data=resources_pb2.Data(audio=audiopb, geo=geo_pb, concepts=concepts, metadata=metadata))

    return resources_pb2.Input(
        id=input_id,
        data=resources_pb2.Data(audio=audiopb, geo=geo_pb, concepts=concepts, metadata=metadata))

  def get_input_from_url(self, input_id: str, url: str, dataset_id: str = None, **kwargs) -> Audio:
    """Create input proto for audio data type from url.
        Args:
            input_id (str): The input ID for the input to create.
            url (str): The url for the audio.
            dataset_id (str): The dataset ID for the dataset to add the input to.
            **kwargs: Additional keyword arguments to be passed to the Input
        Returns:
            Audio: An Input object for the specified input ID.
        Example:
            >>> from clarifai.client.input.audio import Audio
            >>> audio_obj = Audio()
            >>> input_protos = audio_obj.get_inputs_from_url(input_id = 'demo', url='https://samples.clarifai.com/english_audio_sample.mp3')
        """
    audio_pb = resources_pb2.Audio(url=url)
    return self._get_proto(input_id=input_id, dataset_id=dataset_id, audiopb=audio_pb, **kwargs)

  def get_input_from_filename(self, input_id: str, filename: str, dataset_id: str = None,
                              **kwargs) -> Audio:
    """Create input proto for audio data type from filename.
        Args:
            input_id (str): The input ID for the input to create.
            filename (str): The filename for the audio.
            dataset_id (str): The dataset ID for the dataset to add the input to.
            **kwargs: Additional keyword arguments to be passed to the Input
        Returns:
            Audio: An Input object for the specified input ID.
        Example:
            >>> from clarifai.client.input.audio import Audio
            >>> audio_obj = Audio()
            >>> input_protos = audio_obj.get_inputs_from_filename(input_id = 'demo', filename='demo.mp3')
        """
    audio_pb = resources_pb2.Audio(base64=os.path.basename(filename))
    return self._get_proto(input_id=input_id, dataset_id=dataset_id, audiopb=audio_pb, **kwargs)

  def get_input_from_bytes(self, input_id: str, bytes: bytes, dataset_id: str = None,
                           **kwargs) -> Audio:
    """Create input proto for audio data type from bytes.
        Args:
            input_id (str): The input ID for the input to create.
            bytes (bytes): The bytes for the audio.
            dataset_id (str): The dataset ID for the dataset to add the input to.
            **kwargs: Additional keyword arguments to be passed to the Input
        Returns:
            Audio: An Input object for the specified input ID.
        Example:
            >>> from clarifai.client.input.audio import Audio
            >>> audio_obj = Audio()
            >>> bytes = open('demo.mp3', 'rb').read()
            >>> input_protos = audio_obj.get_inputs_from_bytes(input_id = 'demo', bytes='bytes')
        """
    audio_pb = resources_pb2.Audio(base64=bytes)
    return self._get_proto(input_id=input_id, dataset_id=dataset_id, audiopb=audio_pb, **kwargs)

  def upload_from_url(self, input_id: str, url: str, dataset_id: str = None, **kwargs) -> str:
    """upload audio from url.
        Args:
            input_id (str): The input ID for the input to create.
            url (str): The url for the audio.
            dataset_id (str): The dataset ID for the dataset to add the input to.
        Returns:
            input_job_id (str): job id for the upload request.
        Example:
            >>> from clarifai.client.input.audio import Audio
            >>> audio_obj = Audio(user_id = 'user_id', app_id = 'demo_app')
            >>> audio_obj.upload_from_url(input_id = 'demo', url='https://samples.clarifai.com/english_audio_sample.mp3')
        """
    input_pb = self._get_proto(input_id, dataset_id, resources_pb2.Audio(url=url), **kwargs)
    return self.upload_inputs([input_pb])

  def upload_from_filename(self, input_id: str, filename: str, dataset_id: str = None,
                           **kwargs) -> str:
    """upload audio from filename.
        Args:
            input_id (str): The input ID for the input to create.
            filename (str): The filename for the audio.
            dataset_id (str): The dataset ID for the dataset to add the input to.
        Returns:
            input_job_id (str): job id for the upload request.
        Example:
            >>> from clarifai.client.input.audio import Audio
            >>> audio_obj = Audio(user_id = 'user_id', app_id = 'demo_app')
            >>> audio_obj.upload_from_filename(input_id = 'demo', filename='demo.mp3')
        """
    input_pb = self._get_proto(
        input_id, dataset_id, resources_pb2.Audio(base64=os.path.basename(filename)), **kwargs)
    return self.upload_inputs([input_pb])

  def upload_from_bytes(self, input_id: str, bytes: bytes, dataset_id: str = None,
                        **kwargs) -> str:
    """upload audio from bytes.
        Args:
            input_id (str): The input ID for the input to create.
            bytes (bytes): The bytes for the audio.
            dataset_id (str): The dataset ID for the dataset to add the input to.
        Returns:
            input_job_id (str): job id for the upload request.
        Example:
            >>> from clarifai.client.input.audio import Audio
            >>> audio_obj = Audio(user_id = 'user_id', app_id = 'demo_app')
            >>> bytes = open('demo.mp3', 'rb').read()
            >>> audio_obj.upload_from_bytes(input_id = 'demo', bytes = bytes)
        """
    input_pb = self._get_proto(input_id, dataset_id, resources_pb2.Audio(base64=bytes), **kwargs)
    return self.upload_inputs([input_pb])
