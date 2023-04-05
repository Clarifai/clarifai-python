import os
import tempfile
import zipfile
from io import BytesIO

import requests
from clarifai_grpc.grpc.api import resources_pb2
from PIL import ImageFile
from tqdm import tqdm


class DatasetExportReader:
  """
  Unpacks the zipfile from DatasetVersionExport
  - Downloads the temp archive onto disk
  - Reads DatasetVersionExports archive in memory without extracting all
  - Yield each api.Input object.
  """

  def __init__(self, archive_url=None, local_archive_path=None):

    self.input_count = 0
    self.temp_file = None

    assert archive_url or local_archive_path, "Must use one input."

    if archive_url:
      print('url: %s' % archive_url)
      self.temp_file = self._download_temp_archive(archive_url)
      self.archive = zipfile.ZipFile(self.temp_file)
    else:
      print("path: %s" % local_archive_path)
      self.archive = zipfile.ZipFile(local_archive_path)

    self.file_name_list = self.archive.namelist()
    assert "mimetype" in self.file_name_list, "Missing mimetype file in the dataset export archive."
    assert self.archive.read("mimetype") == b"application/x.clarifai-data+protobuf"
    self.file_name_list.remove("mimetype")

    print("Obtained file name list. %d entries." % len(self.file_name_list))
    self.split_dir = os.path.dirname(self.file_name_list[0]) if len(self.file_name_list) else ""

  def _download_temp_archive(self, archive_url, chunk_size=128):
    """
    Downloads the temp archive of InputBatches.
    """
    r = requests.get(archive_url, stream=True)
    temp_file = tempfile.TemporaryFile()
    for chunk in r.iter_content(chunk_size=chunk_size):
      temp_file.write(chunk)

    return temp_file

  def __len__(self):
    if not self.input_count:
      if self.file_name_list is not None:
        for filename in self.file_name_list:
          self.input_count += int(filename.split('_n')[-1])

    return self.input_count

  def __iter__(self):
    """
    Loops through all InputBatches in the DatasetVersionExport and yields every api.Input object
    """
    if self.file_name_list is not None:
      for filename in self.file_name_list:
        db = resources_pb2.InputBatch().FromString(self.archive.read(filename))
        for db_input in db.inputs:
          yield db_input
      print("DONE")

  def __enter__(self):
    return self

  def __exit__(self, *args):
    self.close()

  def close(self):
    print("closing file objects.")
    self.archive.close()
    if self.temp_file:
      self.temp_file.close()


class InputDownloader:
  """
  Takes an iterator or a list of api.Input instances as input,
  and has a method for downloading all inputs (currently only images) of that data.
  Has the ability of either writing to a new ZIP archive OR a filesystem directory.
  """

  def __init__(self, input_iterator):
    self.input_iterator = input_iterator
    self.num_inputs = 0
    self.split_prefix = None
    if isinstance(self.input_iterator, DatasetExportReader):
      self.split_prefix = self.input_iterator.split_dir

  def _save_image_to_archive(self, new_archive, hosted_url, name):
    """
    Use PIL ImageFile to return image parsed from the response bytestring (from requests) and append to zip file.
    """
    p = ImageFile.Parser()
    p.feed(requests.get(hosted_url).content)
    image = p.close()
    image_file = BytesIO()
    image.save(image_file, 'JPEG')
    new_archive.writestr(name, image_file.getvalue())

  def _write_image_archive(self, save_path, split):
    """
    Writes the image archive into prefix dir.
    """
    try:
      total = len(self.input_iterator)
    except TypeError:
      total = None
    with zipfile.ZipFile(save_path, "a") as new_archive:
      for input_ in tqdm(self.input_iterator, desc="Writing image archive", total=total):
        # checks for image
        hosted = input_.data.image.hosted
        if hosted.prefix:
          assert 'orig' in hosted.sizes
          hosted_url = f"{hosted.prefix}/orig/{hosted.suffix}"
          full_name = os.path.join(split, input_.id + ".jpg")

          self._save_image_to_archive(new_archive, hosted_url, full_name)
          self.num_inputs += 1

  def _check_output_archive(self, save_path):
    try:
      archive = zipfile.ZipFile(save_path, 'r')
    except zipfile.BadZipFile as e:
      raise e
    assert len(
        archive.namelist()) == self.num_inputs, "Archive has %d inputs | expecting %d inputs" % (
            len(archive.namelist()), self.num_inputs)

  def download_image_archive(self, save_path, split=None):
    """
    Downloads the archive from the URL into an archive of images in the directory format {split}/{image}.
    """
    self._write_image_archive(save_path, split=split or self.split_prefix)
    self._check_output_archive(save_path)


if __name__ == "__main__":
  import sys
  if len(sys.argv) < 2:
    print(f"usage: {sys.argv[0]} <archive-url> [<save-path>]")
    sys.exit(2)
  archive_url = sys.argv[1]
  save_path = sys.argv[2] if len(sys.argv) > 2 else "output.zip"

  with DatasetExportReader(archive_url=archive_url) as reader:
    InputDownloader(reader).download_image_archive(save_path=save_path)
