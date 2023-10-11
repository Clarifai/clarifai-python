import json
import os
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, Iterator, List, Optional

import requests
from clarifai_grpc.grpc.api import resources_pb2
from google.protobuf.json_format import MessageToDict
from PIL import ImageFile
from tqdm import tqdm

from clarifai.errors import UserError
from clarifai.utils.logging import get_logger

logger = get_logger("INFO", __name__)


class DatasetExportReader:

  def __init__(self,
               session: requests.Session,
               archive_url: Optional[str] = None,
               local_archive_path: Optional[str] = None):
    """Download/Reads the zipfile archive and yields every api.Input object.

    Args:
        session: requests.Session object
        archive_url: URL of the DatasetVersionExport archive
        local_archive_path: Path to the DatasetVersionExport archive
    """
    self.input_count = 0
    self.temp_file = None
    self.session = session

    assert archive_url or local_archive_path, UserError(
        "Either archive_url or local_archive_path must be provided.")
    assert not (archive_url and local_archive_path), UserError(
        "Only one of archive_url or local_archive_path must be provided.")

    if archive_url:
      logger.info('url: %s' % archive_url)
      self.temp_file = self._download_temp_archive(archive_url)
      self.archive = zipfile.ZipFile(self.temp_file)
    else:
      logger.info("path: %s" % local_archive_path)
      self.archive = zipfile.ZipFile(local_archive_path)

    self.file_name_list = self.archive.namelist()
    assert "mimetype" in self.file_name_list, "Missing mimetype file in the dataset export archive."
    assert self.archive.read("mimetype") == b"application/x.clarifai-data+protobuf"
    self.file_name_list.remove("mimetype")

    logger.info("Obtained file name list. %d entries." % len(self.file_name_list))
    self.split_dir = os.path.dirname(self.file_name_list[0]) if len(self.file_name_list) else ""

  def _download_temp_archive(self, archive_url: str,
                             chunk_size: int = 128) -> tempfile.TemporaryFile:
    """Downloads the temp archive of InputBatches."""
    r = self.session.get(archive_url, stream=True)
    temp_file = tempfile.TemporaryFile()
    for chunk in r.iter_content(chunk_size=chunk_size):
      temp_file.write(chunk)

    return temp_file

  def __len__(self) -> int:
    if not self.input_count:
      if self.file_name_list is not None:
        for filename in self.file_name_list:
          self.input_count += int(filename.split('_n')[-1])

    return self.input_count

  def __iter__(self) -> Iterator[resources_pb2.Input]:
    """Loops through all InputBatches in the DatasetVersionExport and yields every api.Input object"""
    if self.file_name_list is not None:
      for filename in self.file_name_list:
        db = resources_pb2.InputBatch().FromString(self.archive.read(filename))
        for db_input in db.inputs:
          yield db_input

  def __enter__(self) -> 'DatasetExportReader':
    return self

  def __exit__(self, *args: Any) -> None:
    self.close()

  def close(self) -> None:
    logger.info("closing file objects.")
    self.archive.close()
    if self.temp_file:
      self.temp_file.close()


class InputAnnotationDownloader:

  def __init__(self,
               session: requests.Session,
               input_iterator: DatasetExportReader,
               num_workers: int = 4):
    """Downloads the archive from the URL into an archive of inputs, annotations in the directory format
    {split}/inputs and {split}/annotations.

    Args:
        session: requests.Session object
        input_iterator: Iterable of DatasetExportReader object
        num_workers: Number of threads to use for downloading
    """
    self.input_iterator = input_iterator
    self.num_workers = min(num_workers, 10)  # Max 10 threads
    self.num_inputs_annotations = 0
    self.split_prefix = None
    self.session = session
    self.input_ext = dict(image=".png", text=".txt", audio=".mp3", video=".mp4")
    if isinstance(self.input_iterator, DatasetExportReader):
      self.split_prefix = self.input_iterator.split_dir

  def _save_image_to_archive(self, new_archive: zipfile.ZipFile, hosted_url: str,
                             file_name: str) -> None:
    """Use PIL ImageFile to return image parsed from the response bytestring (from requests) and append to zip file."""
    p = ImageFile.Parser()
    p.feed(self.session.get(hosted_url).content)
    image = p.close()
    image_file = BytesIO()
    image.save(image_file, 'PNG')
    new_archive.writestr(file_name, image_file.getvalue())

  def _save_text_to_archive(self, new_archive: zipfile.ZipFile, hosted_url: str,
                            file_name: str) -> None:
    """Gets the text response bytestring (from requests) and append to zip file."""
    text_content = self.session.get(hosted_url).content
    new_archive.writestr(file_name, text_content)

  def _save_audio_to_archive(self, new_archive: zipfile.ZipFile, hosted_url: str,
                             file_name: str) -> None:
    """Gets the audio response bytestring (from requests) as chunks and append to zip file."""
    audio_response = self.session.get(hosted_url, stream=True)
    audio_stream = BytesIO()
    # Retrieve the audio content in chunks and write to the BytesIO object
    for chunk in audio_response.iter_content(chunk_size=128):
      audio_stream.write(chunk)
    new_archive.writestr(file_name, audio_stream.getvalue())

  def _save_video_to_archive(self, new_archive: zipfile.ZipFile, hosted_url: str,
                             file_name: str) -> None:
    """Gets the video response bytestring (from requests) as chunks and append to zip file."""
    video_response = self.session.get(hosted_url)
    video_stream = BytesIO()
    # Retrieve the video content in chunks and write to the BytesIO object
    for chunk in video_response.iter_content(chunk_size=128):
      video_stream.write(chunk)
    new_archive.writestr(file_name, video_stream.getvalue())

  def _save_annotation_to_archive(self, new_archive: zipfile.ZipFile, annot_data: List[Dict],
                                  file_name: str) -> None:
    """Gets the annotation response bytestring (from requests) and append to zip file."""
    # Serialize the dictionary to a JSON string
    json_str = json.dumps(annot_data)
    # Convert the JSON string to bytes
    bytes_object = json_str.encode()

    new_archive.writestr(file_name, bytes_object)

  def _write_archive(self, input_, new_archive, split: Optional[str] = None) -> None:
    """Writes the input, annotation archive into prefix dir."""
    data_dict = MessageToDict(input_.data)
    input_type = list(filter(lambda x: x in list(data_dict.keys()),
                             list(self.input_ext.keys())))[0]
    hosted = getattr(input_.data, input_type).hosted
    if hosted.prefix:
      assert 'orig' in hosted.sizes
      hosted_url = f"{hosted.prefix}/orig/{hosted.suffix}"
      file_name = os.path.join(split, "inputs", input_.id + self.input_ext[input_type])
      if input_type == "image":
        self._save_image_to_archive(new_archive, hosted_url, file_name)
      elif input_type == "text":
        self._save_text_to_archive(new_archive, hosted_url, file_name)
      elif input_type == "audio":
        self._save_audio_to_archive(new_archive, hosted_url, file_name)
      elif input_type == "video":
        self._save_video_to_archive(new_archive, hosted_url, file_name)
      self.num_inputs_annotations += 1

    if data_dict.get("concepts") or data_dict.get("regions"):
      file_name = os.path.join(split, "annotations", input_.id + ".json")
      annot_data = data_dict.get("concepts") or data_dict.get("regions")

      self._save_annotation_to_archive(new_archive, annot_data, file_name)
      self.num_inputs_annotations += 1

  def _check_output_archive(self, save_path: str) -> None:
    try:
      archive = zipfile.ZipFile(save_path, 'r')
    except zipfile.BadZipFile as e:
      raise e
    assert len(
        archive.namelist()
    ) == self.num_inputs_annotations, "Archive has %d inputs+annotations | expecting %d inputs+annotations" % (
        len(archive.namelist()), self.num_inputs_annotations)

  def download_archive(self, save_path: str, split: Optional[str] = None) -> None:
    """Downloads the archive from the URL into an archive of inputs, annotations in the directory format
    {split}/inputs and {split}/annotations.
    """
    with zipfile.ZipFile(save_path, "a") as new_archive:
      with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
        with tqdm(total=len(self.input_iterator), desc='Downloading Dataset') as progress:
          # Submit all jobs to the executor and store the returned futures
          futures = [
              executor.submit(self._write_archive, input_, new_archive, split)
              for input_ in self.input_iterator
          ]

          for _ in as_completed(futures):
            progress.update()

    self._check_output_archive(save_path)
    logger.info("Downloaded %d inputs+annotations to %s" % (self.num_inputs_annotations,
                                                            save_path))
