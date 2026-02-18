import json
import os
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Any, Dict, Iterator, List, Optional

import requests
from clarifai_grpc.grpc.api import resources_pb2
from google.protobuf.json_format import MessageToDict
from PIL import ImageFile
from tqdm import tqdm

from clarifai.constants.dataset import CONTENT_TYPE
from clarifai.errors import UserError
from clarifai.utils.logging import logger


class DatasetExportReader:
    def __init__(
        self,
        session: requests.Session = None,
        archive_url: Optional[str] = None,
        local_archive_path: Optional[str] = None,
    ):
        """Download/Reads the zipfile archive and yields every api.Input object.

        Args:
            session: requests.Session object
            archive_url: URL of the DatasetVersionExport archive
            local_archive_path: Path to the DatasetVersionExport archive
        """
        self.input_count = None
        self.temp_file = None
        self.session = session
        if not self.session:
            self.session = requests.Session()

        assert archive_url or local_archive_path, UserError(
            "Either archive_url or local_archive_path must be provided."
        )
        assert not (archive_url and local_archive_path), UserError(
            "Only one of archive_url or local_archive_path must be provided."
        )

        if archive_url:
            logger.info('url: %s' % archive_url)
            self.temp_file = self._download_temp_archive(archive_url)
            self.archive = zipfile.ZipFile(self.temp_file)
        else:
            logger.info("path: %s" % local_archive_path)
            self.archive = zipfile.ZipFile(local_archive_path)

        self.file_name_list = self.archive.namelist()
        assert "mimetype" in self.file_name_list, (
            "Missing mimetype file in the dataset export archive."
        )
        assert self.archive.read("mimetype") == b"application/x.clarifai-data+protobuf"
        self.file_name_list.remove("mimetype")

        logger.info("Obtained file name list. %d entries." % len(self.file_name_list))
        self.split_dir = (
            os.path.dirname(self.file_name_list[0]) if len(self.file_name_list) else ""
        )

    def _download_temp_archive(
        self, archive_url: str, chunk_size: int = 128
    ) -> tempfile.TemporaryFile:
        """Downloads the temp archive of InputBatches."""
        r = self.session.get(archive_url, stream=True)
        if r.headers['content-type'] == CONTENT_TYPE['json']:
            raise Exception("File is a json file :\n {}".format(r.json()))
        elif r.headers['content-type'] != CONTENT_TYPE['zip']:
            raise Exception('File is not a zip file')
        temp_file = tempfile.TemporaryFile()
        for chunk in r.iter_content(chunk_size=chunk_size):
            temp_file.write(chunk)

        return temp_file

    def __len__(self) -> int:
        if self.input_count is None:
            input_count = 0
            if self.file_name_list is not None:
                for filename in self.file_name_list:
                    input_count += int(filename.split('_n')[-1])
            self.input_count = input_count

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
    def __init__(
        self, session: requests.Session, input_iterator: DatasetExportReader, num_workers: int = 4
    ):
        """Downloads the archive from the URL into an archive of inputs, annotations in the directory format
        {split}/inputs and {split}/annotations.

        Args:
            session: requests.Session object
            input_iterator: Iterable of DatasetExportReader object
            num_workers: Number of threads to use for downloading
        """
        self.input_iterator = input_iterator
        self.num_workers = min(num_workers, 10)  # Max 10 threads
        self.num_inputs = 0
        self.num_annotations = 0
        self.split_prefix = None
        self.session = session
        self.input_ext = dict(image=".png", text=".txt", audio=".mp3", video=".mp4")
        self.zip_lock = threading.Lock()  # Lock to protect concurrent ZIP file writes
        if isinstance(self.input_iterator, DatasetExportReader):
            self.split_prefix = self.input_iterator.split_dir

    def _save_image_to_archive(
        self, new_archive: zipfile.ZipFile, hosted_url: str, file_name: str
    ) -> None:
        """Use PIL ImageFile to return image parsed from the response bytestring (from requests) and append to zip file."""
        try:
            response = self.session.get(hosted_url, timeout=60)
            response.raise_for_status()
            p = ImageFile.Parser()
            p.feed(response.content)
            image = p.close()
            image_file = BytesIO()
            image.save(image_file, 'PNG')
            # Thread-safe write to ZIP archive
            with self.zip_lock:
                new_archive.writestr(file_name, image_file.getvalue())
        except requests.exceptions.Timeout:
            raise Exception(f"Timeout downloading image from {hosted_url}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error downloading image from {hosted_url}: {e}")
        except Exception as e:
            raise Exception(f"Error processing image from {hosted_url}: {e}")

    def _save_text_to_archive(
        self, new_archive: zipfile.ZipFile, hosted_url: str, file_name: str
    ) -> None:
        """Gets the text response bytestring (from requests) and append to zip file."""
        try:
            response = self.session.get(hosted_url, timeout=60)
            response.raise_for_status()
            # Thread-safe write to ZIP archive
            with self.zip_lock:
                new_archive.writestr(file_name, response.content)
        except requests.exceptions.Timeout:
            raise Exception(f"Timeout downloading text from {hosted_url}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error downloading text from {hosted_url}: {e}")

    def _save_audio_to_archive(
        self, new_archive: zipfile.ZipFile, hosted_url: str, file_name: str
    ) -> None:
        """Gets the audio response bytestring (from requests) as chunks and append to zip file."""
        try:
            audio_response = self.session.get(hosted_url, stream=True, timeout=60)
            audio_response.raise_for_status()
            audio_stream = BytesIO()
            # Retrieve the audio content in chunks and write to the BytesIO object
            for chunk in audio_response.iter_content(chunk_size=128):
                audio_stream.write(chunk)
            # Thread-safe write to ZIP archive
            with self.zip_lock:
                new_archive.writestr(file_name, audio_stream.getvalue())
        except requests.exceptions.Timeout:
            raise Exception(f"Timeout downloading audio from {hosted_url}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error downloading audio from {hosted_url}: {e}")

    def _save_video_to_archive(
        self, new_archive: zipfile.ZipFile, hosted_url: str, file_name: str
    ) -> None:
        """Gets the video response bytestring (from requests) as chunks and append to zip file."""
        try:
            video_response = self.session.get(hosted_url, stream=True, timeout=120)
            video_response.raise_for_status()
            video_stream = BytesIO()
            # Retrieve the video content in chunks and write to the BytesIO object
            for chunk in video_response.iter_content(chunk_size=128):
                video_stream.write(chunk)
            # Thread-safe write to ZIP archive
            with self.zip_lock:
                new_archive.writestr(file_name, video_stream.getvalue())
        except requests.exceptions.Timeout:
            raise Exception(f"Timeout downloading video from {hosted_url}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error downloading video from {hosted_url}: {e}")

    def _save_annotation_to_archive(
        self, new_archive: zipfile.ZipFile, annot_data: List[Dict], file_name: str
    ) -> None:
        """Gets the annotation response bytestring (from requests) and append to zip file."""
        # Fill zero values for missing bounding box keys
        for annot in annot_data:
            if annot.get('regionInfo') and annot['regionInfo'].get('boundingBox'):
                bbox = annot['regionInfo']['boundingBox']
                bbox.setdefault('topRow', 0)
                bbox.setdefault('leftCol', 0)
                bbox.setdefault('bottomRow', 0)
                bbox.setdefault('rightCol', 0)
        # Serialize the dictionary to a JSON string
        json_str = json.dumps(annot_data)
        # Convert the JSON string to bytes
        bytes_object = json_str.encode()

        # Thread-safe write to ZIP archive
        with self.zip_lock:
            new_archive.writestr(file_name, bytes_object)

    def _write_archive(self, input_, new_archive, split: Optional[str] = None) -> Optional[str]:
        """Writes the input, annotation archive into prefix dir.

        Returns:
            "skipped" if input was skipped due to missing hosted URL, None otherwise
        """
        try:
            data_dict = MessageToDict(input_.data)
            input_type = list(
                filter(lambda x: x in list(data_dict.keys()), list(self.input_ext.keys()))
            )[0]
            hosted = getattr(input_.data, input_type).hosted

            if not hosted.prefix:
                logger.debug(f"Skipping input {input_.id} - missing hosted URL")
                return "skipped"

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

            self.num_inputs += 1

            if data_dict.get("metadata") or data_dict.get("concepts") or data_dict.get("regions"):
                file_name = os.path.join(split, "annotations", input_.id + ".json")
                annot_data = (
                    [{"metadata": data_dict.get("metadata", {})}]
                    + data_dict.get("regions", [])
                    + data_dict.get("concepts", [])
                )

                self._save_annotation_to_archive(new_archive, annot_data, file_name)
                self.num_annotations += 1

            return None

        except Exception as e:
            logger.error(f"Error processing input {input_.id}: {e}")
            raise

    def _check_output_archive(self, save_path: str) -> None:
        try:
            archive = zipfile.ZipFile(save_path, 'r')
        except zipfile.BadZipFile as e:
            raise e
        assert len(archive.namelist()) == self.num_inputs + self.num_annotations, (
            "Archive has %d inputs+annotations | expecting %d inputs+annotations"
            % (len(archive.namelist()), self.num_inputs + self.num_annotations)
        )

    def download_archive(self, save_path: str, split: Optional[str] = None) -> None:
        """Downloads the archive from the URL into an archive of inputs, annotations in the directory format
        {split}/inputs and {split}/annotations.
        """
        failed_inputs = []
        skipped_inputs = []

        with zipfile.ZipFile(save_path, "a") as new_archive:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                with tqdm(total=len(self.input_iterator), desc='Downloading Dataset') as progress:
                    # Submit all jobs to the executor and store the returned futures with input IDs
                    future_to_input = {}
                    for input_ in self.input_iterator:
                        future = executor.submit(self._write_archive, input_, new_archive, split)
                        future_to_input[future] = input_.id

                    # Check each completed future for exceptions
                    for future in as_completed(future_to_input):
                        input_id = future_to_input[future]
                        try:
                            result = future.result()
                            if result == "skipped":
                                skipped_inputs.append(input_id)
                        except Exception as e:
                            failed_inputs.append((input_id, str(e)))
                            logger.error(f"Failed to download input {input_id}: {e}")
                        progress.update()

        # Log summary of failures and skips
        if failed_inputs:
            logger.warning(
                f"Failed to download {len(failed_inputs)} inputs: "
                f"{[input_id for input_id, _ in failed_inputs[:10]]}"
                + (" ..." if len(failed_inputs) > 10 else "")
            )
        if skipped_inputs:
            logger.warning(
                f"Skipped {len(skipped_inputs)} inputs (missing hosted URL): "
                f"{skipped_inputs[:10]}" + (" ..." if len(skipped_inputs) > 10 else "")
            )

        self._check_output_archive(save_path)
        logger.info(
            "Downloaded %d inputs and %d annotations to %s"
            % (self.num_inputs, self.num_annotations, save_path)
        )

        if failed_inputs or skipped_inputs:
            logger.warning(
                f"Download completed with issues: {len(failed_inputs)} failed, "
                f"{len(skipped_inputs)} skipped out of {len(self.input_iterator)} total inputs"
            )
