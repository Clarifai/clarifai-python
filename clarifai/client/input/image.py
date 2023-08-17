import os
from typing import List, Union

from clarifai_grpc.grpc.api import resources_pb2, service_pb2  # noqa: F401
from clarifai_grpc.grpc.api.resources_pb2 import Annotation, Input

from clarifai.client.input import Inputs


class Image(Inputs):
  """
    Image is a class that provides access to protos related to Image.
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

  def _get_proto(self, input_id: str, dataset_id: Union[str, None], imagepb: resources_pb2.Image,
                 **kwargs) -> Input:
    """Create input proto for image data type.
        Args:
            input_id (str): The input ID for the input to create.
            dataset_id (str): The dataset ID for the dataset to add the input to.
            imagepb (resources_pb2.Image): The image proto to be used for the input.
            **kwargs: Additional keyword arguments to be passed to the Input
                - geo_info (list): A list of longitude and latitude for the geo point.
                - labels (list): A list of labels for the input.
                - metadata (Struct): A Struct of metadata for the input.
        Returns:
            Input: An Input object for the specified input ID."""
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
          data=resources_pb2.Data(image=imagepb, geo=geo_pb, concepts=concepts, metadata=metadata))

    return resources_pb2.Input(
        id=input_id,
        data=resources_pb2.Data(image=imagepb, geo=geo_pb, concepts=concepts, metadata=metadata))

  def get_input_from_url(self, input_id: str, url: str, dataset_id: str = None, **kwargs) -> Input:
    """Create input proto for image data type from url.
        Args:
            input_id (str): The input ID for the input to create.
            url (str): The url for the image.
            dataset_id (str): The dataset ID for the dataset to add the input to.
        Returns:
            Input: An Input object for the specified input ID.
        Example:
            >>> from clarifai.client.input.image import Image
            >>> image_obj = Image()
            >>> input_proto = img_obj.get_input_from_url(input_id = 'demo', url='https://samples.clarifai.com/metro-north.jpg')
        """
    image_pb = resources_pb2.Image(url=url)
    return self._get_proto(input_id, dataset_id, image_pb, **kwargs)

  def get_input_from_filename(self, input_id: str, filename: str, dataset_id: str = None,
                              **kwargs) -> Input:
    """Create input proto for image data type from filename.
        Args:
            input_id (str): The input ID for the input to create.
            filename (str): The filename for the image.
            dataset_id (str): The dataset ID for the dataset to add the input to.
        Returns:
            Input: An Input object for the specified input ID.
        Example:
            >>> from clarifai.client.input.image import Image
            >>> image_obj = Image()
            >>> input_proto = img_obj.get_input_from_filename(input_id = 'demo', filename='demo.jpg')
        """
    image_pb = resources_pb2.Image(base64=open(filename, 'rb').read())
    return self._get_proto(input_id, dataset_id, image_pb, **kwargs)

  def get_input_from_bytes(self, input_id: str, bytes: bytes, dataset_id: str = None,
                           **kwargs) -> Input:
    """Create input proto for image data type from bytes.
        Args:
            input_id (str): The input ID for the input to create.
            bytes (bytes): The bytes for the image.
            dataset_id (str): The dataset ID for the dataset to add the input to.
        Returns:
            Input: An Input object for the specified input ID.
        Example:
            >>> from clarifai.client.input.image import Image
            >>> image_obj = Image()
            >>> bytes = open('demo.jpg', 'rb').read()
            >>> input_proto = img_obj.get_input_from_bytes(input_id = 'demo', bytes=bytes)
        """
    image_pb = resources_pb2.Image(base64=bytes)
    return self._get_proto(input_id, dataset_id, image_pb, **kwargs)

  def get_inputs_from_folder(self, folder_path: str, dataset_id: str = None) -> List[Input]:
    """Create input protos for image data type from folder.
        The folder should only contain images. The filename of the image is used as the input_id.
        Args:
            folder_path (str): Path to the folder containing images.
        Returns:
            list of Input: A list of Input objects for the specified folder.
        Example:
            >>> from clarifai.client.input.image import Image
            >>> image_obj = Image()
            >>> input_protos = img_obj.get_inputs_from_folder(folder_path='demo_folder')
        """
    inputs = []
    for filename in os.listdir(folder_path):
      if filename.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'tiff', 'webp']:
        continue
      input_id = filename.split('.')[0]
      image_pb = resources_pb2.Image(base64=open(os.path.join(folder_path, filename), 'rb').read())
      inputs.append(self._get_proto(input_id, dataset_id, image_pb))
    return inputs

  def get_annotation_proto(self, input_id: str, label: str, annotations: List) -> Annotation:
    """Create an annotation proto for each bounding box, label input pair.
        Args:
            input_id (str): The input ID for the annotation to create.
            label (str): annotation label
            annotations (List): a list of a single bbox's coordinates. # Annotations ordering: [xmin, ymin, xmax, ymax]
        Returns:
            An annotation object for the specified input ID.
        Example:
            >>> from clarifai.client.input.image import Image
            >>> image_obj = Image()
            >>> img_obj.get_annotation_proto(input_id='demo', label='demo', annotations=[x_min, y_min, x_max, y_max])
        """
    input_annot_proto = resources_pb2.Annotation(
        input_id=input_id,
        data=resources_pb2.Data(regions=[
            resources_pb2.Region(
                region_info=resources_pb2.RegionInfo(bounding_box=resources_pb2.BoundingBox(
                    # Annotations ordering: [xmin, ymin, xmax, ymax]
                    # top_row must be less than bottom row
                    # left_col must be less than right col
                    top_row=annotations[1],  #y_min
                    left_col=annotations[0],  #x_min
                    bottom_row=annotations[3],  #y_max
                    right_col=annotations[2]  #x_max
                )),
                data=resources_pb2.Data(concepts=[
                    resources_pb2.Concept(
                        id=f"id-{''.join(label.split(' '))}", name=label, value=1.)
                ]))
        ]))

    return input_annot_proto

  def get_mask_proto(self, input_id: str, label: str, polygons: List[List[float]]) -> Annotation:
    """Create an annotation proto for each polygon box, label input pair.
        Args:
            input_id (str): The input ID for the annotation to create.
            label (str): annotation label
            polygons (List): Polygon x,y points iterable
        Returns:
            An annotation object for the specified input ID.
        Example:
            >>> from clarifai.client.input.image import Image
            >>> image_obj = Image()
            >>> img_obj.get_mask_proto(input_id='demo', label='demo', polygons=[[[x,y],...,[x,y]],...])
        """
    input_mask_proto = resources_pb2.Annotation(
        input_id=input_id,
        data=resources_pb2.Data(regions=[
            resources_pb2.Region(
                region_info=resources_pb2.RegionInfo(polygon=resources_pb2.Polygon(
                    points=[
                        resources_pb2.Point(
                            row=_point[1],  # row is y point
                            col=_point[0],  # col is x point
                            visibility="VISIBLE") for _point in polygons
                    ])),
                data=resources_pb2.Data(concepts=[
                    resources_pb2.Concept(
                        id=f"id-{''.join(label.split(' '))}", name=label, value=1.)
                ]))
        ]))

    return input_mask_proto

  def upload_from_url(self, input_id: str, url: str, dataset_id: str = None, **kwargs) -> str:
    """upload image from url.
        Args:
            input_id (str): The input ID for the input to create.
            url (str): The url for the image.
            dataset_id (str): The dataset ID for the dataset to add the input to.
        Returns:
            input_job_id (str): job id for the upload request.
        Example:
            >>> from clarifai.client.input.image import Image
            >>> image_obj = Image(user_id = 'user_id', app_id = 'demo_app')
            >>> img_obj.upload_from_url(input_id='demo', url='https://samples.clarifai.com/metro-north.jpg')
        """
    input_pb = self._get_proto(input_id, dataset_id, resources_pb2.Image(url=url), **kwargs)
    return self.upload_inputs([input_pb])

  def upload_from_filename(self, input_id: str, filename: str, dataset_id: str = None,
                           **kwargs) -> str:
    """upload image from filename.
        Args:
            input_id (str): The input ID for the input to create.
            filename (str): The filename for the image.
            dataset_id (str): The dataset ID for the dataset to add the input to.
        Returns:
            input_job_id (str): job id for the upload request.
        Example:
            >>> from clarifai.client.input.image import Image
            >>> image_obj = Image(user_id = 'user_id', app_id = 'demo_app')
            >>> img_obj.upload_from_filename(input_id='demo', filename='demo.jpg')
        """
    input_pb = self._get_proto(
        input_id, dataset_id, resources_pb2.Image(base64=open(filename, 'rb').read()), **kwargs)
    return self.upload_inputs([input_pb])

  def upload_from_bytes(self, input_id: str, bytes: bytes, dataset_id: str = None,
                        **kwargs) -> str:
    """upload image from bytes.
        Args:
            input_id (str): The input ID for the input to create.
            bytes (bytes): The bytes for the image.
            dataset_id (str): The dataset ID for the dataset to add the input to.
        Returns:
            input_job_id (str): job id for the upload request.
        Example:
            >>> from clarifai.client.input.image import Image
            >>> image_obj = Image(user_id = 'user_id', app_id = 'demo_app')
            >>> bytes = open('demo.jpg', 'rb').read()
            >>> img_obj.upload_from_bytes(input_id='demo', bytes=bytes)
        """
    input_pb = self._get_proto(input_id, dataset_id, resources_pb2.Image(base64=bytes), **kwargs)
    return self.upload_inputs([input_pb])

  def upload_from_folder(self, folder_path: str, dataset_id: str = None) -> str:
    """Upload images from folder.
        The folder should only contain images. The filename of the image is used as the input_id.
        Args:
            folder_path (str): Path to the folder containing images.
        Returns:
            input_job_id (str): job id for the upload request.
        Example:
            >>> from clarifai.client.input.image import Image
            >>> image_obj = Image(user_id = 'user_id', app_id = 'demo_app')
            >>> img_obj.upload_from_folder(folder_path='demo_folder')
        """
    inputs = self.get_inputs_from_folder(folder_path, dataset_id)
    return self._bulk_upload(inputs)
