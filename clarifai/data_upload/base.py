from typing import Dict, List


class ClarifaiDataset:
  """
  Dataset base class
  """

  def __init__(self, csv_file_path: str, dataset_id: str, split: str) -> None:
    self.csv_file_path = csv_file_path
    self.dataset_id = dataset_id
    self.split = split
    self._all_input_protos = []

  def __len__(self):
    """
    Get size of all input protos
    """
    return len(self._all_input_protos)

  def to_list(self, input_protos: List):
    """
    Return a list of input protos.
    """
    return list(input_protos)

  def create_input_protos(self,
                          image_path: str,
                          label: str,
                          input_id: str,
                          dataset_id: str,
                          metadata: Dict,
                          use_urls=False,
                          allow_dups=False):
    """
    Create input protos for each image, label input pair.
    Args:
    	`image_path`: full image path or valid url with
    		ending with an image extension.
    	`label`: image label
    	`input_id: unique input id
    	`dataset_id`: Clarifai dataset id
    	`metadata`: image metadata
    	`use_urls`: If set to True it means all image_paths are provided as urls and
    		hence uploading will attempt to read images from urls.
    		The default behavior is reading images from local storage.
    	`allow_dups`: Boolean indicating whether to allow duplicate url inputs
    Returns:
    	An input proto representing a single row input
    """
    raise NotImplementedError()

  def _get_input_protos(self) -> List:
    """
    Create input protos for each row of the dataframe.
    Returns:
    	A list of input protos
    """
    raise NotImplementedError()


class Chunker:
  """
  Split an input sequence into small chunks.
  """

  def __init__(self, seq: List, size: int) -> None:
    self.seq = seq
    self.size = size

  def chunk(self):
    """
    Chunk input sequence.
    """
    return [self.seq[pos:pos + self.size] for pos in range(0, len(self.seq), self.size)]
