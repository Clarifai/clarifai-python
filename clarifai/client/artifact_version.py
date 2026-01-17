import os
from typing import Generator, Optional

import requests
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf.timestamp_pb2 import Timestamp
from tqdm import tqdm

from clarifai.client.base import BaseClient
from clarifai.constants.artifact import (
    ARTIFACT_VISIBILITY_ORG,
    ARTIFACT_VISIBILITY_PRIVATE,
    ARTIFACT_VISIBILITY_PUBLIC,
    DEFAULT_ARTIFACT_VISIBILITY,
    DEFAULT_ARTIFACTS_PAGE_SIZE,
    DEFAULT_DOWNLOAD_MAX_RETRIES,
    DEFAULT_UPLOAD_MAX_RETRIES,
    PROGRESS_BAR_DESCRIPTION_DOWNLOAD,
    PROGRESS_BAR_DESCRIPTION_UPLOAD,
    PROGRESS_BAR_UNIT,
    UPLOAD_CHUNK_SIZE,
)
from clarifai.errors import UserError
from clarifai.utils.logging import logger
from clarifai.utils.misc import format_bytes


class ArtifactVersion(BaseClient):
    """ArtifactVersion client for managing artifact versions in Clarifai."""

    def __init__(
        self,
        artifact_id: str = "",
        version_id: str = "",
        user_id: str = "",
        app_id: str = "",
        **kwargs,
    ):
        """Initialize the ArtifactVersion client.

        Args:
            artifact_id: The artifact ID.
            version_id: The artifact version ID.
            user_id: The user ID.
            app_id: The app ID.
            **kwargs: Additional keyword arguments to be passed to the BaseClient.
        """
        super().__init__(**kwargs)
        # Derive default user_id from auth_helper if it exists and is set
        if hasattr(self, "auth_helper") and self.auth_helper:
            default_user_id = getattr(self.auth_helper, "user_id", "")
        else:
            default_user_id = ""
        self.artifact_id = artifact_id
        self.version_id = version_id
        self.user_id = user_id or default_user_id
        self.app_id = app_id

    @property
    def id(self) -> str:
        """Get the artifact version ID."""
        return self.version_id

    def __repr__(self) -> str:
        init_params = [f"artifact_id='{self.artifact_id}'", f"version_id='{self.version_id}'"]
        if self.user_id:
            init_params.append(f"user_id='{self.user_id}'")
        if self.app_id:
            init_params.append(f"app_id='{self.app_id}'")
        return f"ArtifactVersion({', '.join(init_params)})"

    def upload(
        self,
        file_path: str,
        artifact_id: str = "",
        description: str = "",
        visibility: Optional[str] = None,
        expires_at: Optional[Timestamp] = None,
        version_id: str = "",
        user_id: str = "",
        app_id: str = "",
    ) -> "ArtifactVersion":
        """Upload a file as a new artifact version with streaming support.

        Args:
            file_path: Path to the file to upload.
            artifact_id: The artifact ID. Defaults to the artifact ID from initialization.
            description: Description for the artifact version.
            visibility: Visibility setting ("private", "public", or "org"). Defaults to "private" if not provided.
            expires_at: Optional expiration timestamp.
            version_id: Optional version ID to assign.
            user_id: The user ID. Defaults to the user ID from initialization.
            app_id: The app ID. Defaults to the app ID from initialization.

        Returns:
            An ArtifactVersion object for the uploaded version.

        Raises:
            Exception: If the upload fails.
        """
        artifact_id = artifact_id or self.artifact_id
        user_id = user_id or self.user_id
        app_id = app_id or self.app_id

        # Set default visibility if not provided
        if visibility is None:
            visibility = DEFAULT_ARTIFACT_VISIBILITY

        if not artifact_id:
            raise UserError("artifact_id is required")
        if not user_id:
            raise UserError("user_id is required")
        if not app_id:
            raise UserError("app_id is required")
        if not file_path:
            raise UserError("file_path is required")
        if not os.path.exists(file_path):
            raise UserError(f"File does not exist: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise UserError(f"File is empty: {file_path}")

        logger.info(f"Uploading file: {file_path} ({format_bytes(file_size)})")

        # Handle artifact creation/validation
        from clarifai.client.artifact import Artifact

        # Create artifact client
        artifact = Artifact(
            artifact_id=artifact_id,
            user_id=user_id,
            app_id=app_id,
            pat=getattr(self, 'pat', None),
            base=getattr(self, 'base', None),
        )

        # Ensure artifact exists (create if it doesn't)
        try:
            artifact.get()
            logger.info(f"Artifact {artifact_id} exists")
        except Exception:
            logger.info(f"Creating artifact {artifact_id}")
            artifact.create(artifact_id=artifact_id)

        try:
            # Perform streaming upload with retry logic
            return self._streaming_upload_with_retry(
                file_path=file_path,
                artifact_id=artifact_id,
                description=description,
                visibility=visibility,
                expires_at=expires_at,
                version_id=version_id,
                user_id=user_id,
                app_id=app_id,
            )
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise

    def _streaming_upload_with_retry(
        self,
        file_path: str,
        artifact_id: str,
        description: str,
        visibility: str,
        expires_at: Optional[Timestamp],
        version_id: str,
        user_id: str,
        app_id: str,
        max_retries: int = DEFAULT_UPLOAD_MAX_RETRIES,
    ) -> "ArtifactVersion":
        """Perform streaming upload with retry logic and automatic progress tracking."""
        file_size = os.path.getsize(file_path)

        for attempt in range(max_retries):
            logger.debug(f"Upload attempt {attempt + 1}/{max_retries}")

            # Progress bar setup - always show for better UX
            progress_bar = tqdm(
                total=file_size,
                unit=PROGRESS_BAR_UNIT,
                unit_scale=True,
                desc=PROGRESS_BAR_DESCRIPTION_UPLOAD,
            )

            try:
                uploaded_bytes = 0
                chunk_count = 0
                final_version_id = version_id

                # Create iterator and track chunk sizes for accurate progress
                upload_iterator = self._artifact_version_upload_iterator(
                    file_path,
                    artifact_id,
                    description,
                    visibility,
                    expires_at,
                    version_id,
                    user_id,
                    app_id,
                )

                # Pre-calculate actual chunk sizes for accurate progress tracking
                chunk_sizes = []
                iterator_requests = []

                for request in upload_iterator:
                    iterator_requests.append(request)
                    if hasattr(request, 'content_part') and request.content_part.data:
                        chunk_sizes.append(len(request.content_part.data))

                # Perform streaming upload following pipeline_step pattern (use STUB with automatic metadata)
                try:
                    data_chunk_index = 0  # Track data chunks separately from total responses
                    for response in self.STUB.PostArtifactVersionsUpload(iter(iterator_requests)):
                        if chunk_count == 0:
                            # First response is config upload - extract version ID
                            if hasattr(response, 'artifact_version_id'):
                                final_version_id = response.artifact_version_id
                                logger.info(f"Created artifact version: {final_version_id}")
                        # Update progress with actual chunk size from pre-calculated sizes
                        elif data_chunk_index < len(chunk_sizes):
                            actual_chunk_size = chunk_sizes[data_chunk_index]
                            uploaded_bytes += actual_chunk_size
                            progress_bar.update(actual_chunk_size)
                            data_chunk_index += 1
                        else:
                            logger.warning(
                                f"Unexpected data chunk response at index {data_chunk_index}"
                            )

                        chunk_count += 1

                        # Check for errors (following model upload pattern)
                        finished_status = [
                            status_code_pb2.SUCCESS,
                            status_code_pb2.UPLOAD_DONE,
                        ]
                        uploading_in_progress_status = [status_code_pb2.UPLOAD_IN_PROGRESS]

                        if (
                            hasattr(response, 'status')
                            and response.status.code
                            not in finished_status + uploading_in_progress_status
                        ):
                            error_details = {
                                'code': response.status.code,
                                'description': response.status.description,
                                'details': response.status.details
                                if hasattr(response.status, 'details')
                                else 'No details',
                            }
                            logger.error(f"Server error: {error_details}")
                            raise Exception(f"Upload failed: {response.status.description}")

                except Exception as streaming_error:
                    progress_bar.close()
                    logger.error(f"Error processing streaming response: {streaming_error}")
                    raise Exception(f"Upload failed: {streaming_error}")

                progress_bar.close()
                logger.info(f"Upload completed successfully: {final_version_id}")

                return ArtifactVersion(
                    artifact_id=artifact_id,
                    version_id=final_version_id,
                    user_id=user_id,
                    app_id=app_id,
                )

            except Exception as e:
                progress_bar.close()
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        f"Upload attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                    )
                    import time

                    time.sleep(wait_time)
                    # Reset progress tracking state before retrying
                    uploaded_bytes = 0
                    chunk_count = 0
                    continue
                else:
                    # Last attempt failed, raise with context
                    raise UserError(f"Upload failed after {max_retries} attempts: {e}")

    def _artifact_version_upload_iterator(
        self,
        file_path: str,
        artifact_id: str,
        description: str,
        visibility: str,
        expires_at: Optional[Timestamp],
        version_id: str,
        user_id: str,
        app_id: str,
    ):
        """Iterator for uploading artifact version in chunks (following pipeline_step pattern)."""
        file_size = os.path.getsize(file_path)

        # First yield the upload config
        yield self._create_upload_config(
            artifact_id=artifact_id,
            description=description,
            visibility=visibility,
            expires_at=expires_at,
            version_id=version_id,
            user_id=user_id,
            app_id=app_id,
            file_size=file_size,
        )

        # Then yield file content in chunks (same as pipeline_step)
        with open(file_path, "rb") as f:
            logger.info("Uploading artifact content...")
            logger.debug(f"File size: {file_size}")
            logger.debug(f"Chunk size: {UPLOAD_CHUNK_SIZE}")

            offset = 0
            part_id = 1
            while offset < file_size:
                try:
                    current_chunk_size = min(UPLOAD_CHUNK_SIZE, file_size - offset)
                    chunk = f.read(current_chunk_size)
                    if not chunk:
                        break

                    yield service_pb2.PostArtifactVersionsUploadRequest(
                        content_part=resources_pb2.UploadContentPart(
                            data=chunk,
                            part_number=part_id,
                            range_start=offset,
                        )
                    )
                    offset += len(chunk)
                    part_id += 1
                except Exception as e:
                    logger.exception(f"\nError uploading file: {e}")
                    break

        if offset == file_size:
            logger.info("Upload complete!")

    def _create_upload_config(
        self,
        artifact_id: str,
        description: str,
        visibility: str,
        expires_at: Optional[Timestamp],
        version_id: str,
        user_id: str,
        app_id: str,
        file_size: int,
    ):
        """Create upload config message."""
        # Convert visibility string to enum with validation
        # Note: upload() method sets visibility to DEFAULT_ARTIFACT_VISIBILITY if None was passed
        valid_visibility_values = [
            ARTIFACT_VISIBILITY_PRIVATE,
            ARTIFACT_VISIBILITY_PUBLIC,
            ARTIFACT_VISIBILITY_ORG,
        ]

        if visibility == ARTIFACT_VISIBILITY_PRIVATE:
            visibility_enum = resources_pb2.Visibility.Gettable.PRIVATE
        elif visibility == ARTIFACT_VISIBILITY_PUBLIC:
            visibility_enum = resources_pb2.Visibility.Gettable.PUBLIC
        elif visibility == ARTIFACT_VISIBILITY_ORG:
            visibility_enum = resources_pb2.Visibility.Gettable.ORG
        else:
            raise UserError(
                f"Invalid visibility value: '{visibility}'. "
                f"Valid values are: {', '.join(valid_visibility_values)}"
            )

        artifact_version = resources_pb2.ArtifactVersion(
            description=description,
            visibility=resources_pb2.Visibility(gettable=visibility_enum),
            expires_at=expires_at,
        )

        if version_id:
            artifact_version.id = version_id

        return service_pb2.PostArtifactVersionsUploadRequest(
            upload_config=service_pb2.PostArtifactVersionsUploadConfig(
                user_app_id=self.auth_helper.get_user_app_id_proto(user_id=user_id, app_id=app_id),
                artifact_id=artifact_id,
                artifact_version=artifact_version,
                total_size=file_size,
                storage_request_size=file_size,  # For now, same as total size
            )
        )

    def download(
        self,
        output_path: str = "",
        artifact_id: str = "",
        version_id: str = "",
        user_id: str = "",
        app_id: str = "",
        force: bool = False,
    ) -> str:
        """Download an artifact version with automatic progress tracking and retry logic.

        Args:
            output_path: The local path to save the file. If not provided, uses the original filename.
            artifact_id: The artifact ID. Defaults to the artifact ID from initialization.
            version_id: The artifact version ID. Defaults to the version ID from initialization.
            user_id: The user ID. Defaults to the user ID from initialization.
            app_id: The app ID. Defaults to the app ID from initialization.
            force: Whether to overwrite existing files without prompting.

        Returns:
            The path where the file was saved.

        Raises:
            Exception: If the download fails.
        """
        artifact_id = artifact_id or self.artifact_id
        version_id = version_id or self.version_id
        user_id = user_id or self.user_id
        app_id = app_id or self.app_id

        if not artifact_id:
            raise UserError("artifact_id is required")
        if not user_id:
            raise UserError("user_id is required")
        if not app_id:
            raise UserError("app_id is required")
        if not version_id:
            raise UserError("version_id is required")

        # Get artifact version info
        version_pb = self.get(
            artifact_id=artifact_id, version_id=version_id, user_id=user_id, app_id=app_id
        )

        # Extract download info from protobuf
        if not hasattr(version_pb, 'upload') or not version_pb.upload.content_url:
            raise UserError("No download URL available for this artifact version")

        content_url = version_pb.upload.content_url
        content_name = version_pb.upload.content_name or "downloaded_file"
        total_size = version_pb.upload.content_length or 0

        # Determine output path
        if not output_path:
            output_path = content_name
        elif os.path.isdir(output_path):
            output_path = os.path.join(output_path, content_name)

        return self._download_with_retry(content_url, output_path, total_size, force)

    def _download_with_retry(
        self,
        content_url: str,
        output_path: str,
        total_size: int,
        force: bool,
        max_retries: int = DEFAULT_DOWNLOAD_MAX_RETRIES,
    ) -> str:
        """Download file with automatic retry logic and resume support."""
        for attempt in range(max_retries):
            logger.debug(f"Download attempt {attempt + 1}/{max_retries}")

            # Handle existing file and resume
            resume_byte_pos = 0
            if os.path.exists(output_path):
                if total_size > 0:
                    existing_size = os.path.getsize(output_path)
                    if existing_size < total_size:
                        resume_byte_pos = existing_size
                        logger.info(f"Resuming download from byte {resume_byte_pos}")
                    elif existing_size == total_size:
                        logger.info("File already downloaded completely")
                        return output_path

                if resume_byte_pos == 0:
                    if not force:
                        import click

                        if not click.confirm(f"File '{output_path}' already exists. Overwrite?"):
                            raise UserError("Download cancelled by user")

            # Prepare download headers for resume and authentication
            headers = {}
            if resume_byte_pos > 0:
                headers['Range'] = f'bytes={resume_byte_pos}-'

            # Add authentication headers
            if hasattr(self, 'pat') and self.pat:
                headers['Authorization'] = f'Key {self.pat}'
            elif hasattr(self, 'token') and self.token:
                headers['x-clarifai-session-token'] = self.token

            # Download the file with progress tracking
            logger.info(f"Downloading to {output_path}")

            try:
                response = requests.get(content_url, stream=True, headers=headers)
                response.raise_for_status()

                # Check if server supports range requests
                if resume_byte_pos > 0 and response.status_code != 206:
                    logger.warning("Server does not support resume, starting fresh download")
                    resume_byte_pos = 0
                    # Re-add authentication headers for the retry request
                    auth_headers = {}
                    if hasattr(self, 'pat') and self.pat:
                        auth_headers['Authorization'] = f'Key {self.pat}'
                    elif hasattr(self, 'token') and self.token:
                        auth_headers['x-clarifai-session-token'] = self.token
                    response = requests.get(content_url, stream=True, headers=auth_headers)
                    response.raise_for_status()

                # Get content length
                content_length = total_size
                if 'content-length' in response.headers:
                    content_length = int(response.headers['content-length'])
                    if resume_byte_pos > 0:
                        content_length += resume_byte_pos

                # Progress bar setup - always show for better UX
                progress_bar = tqdm(
                    total=content_length,
                    initial=resume_byte_pos,
                    unit=PROGRESS_BAR_UNIT,
                    unit_scale=True,
                    desc=PROGRESS_BAR_DESCRIPTION_DOWNLOAD,
                )

                try:
                    # Write file
                    mode = 'ab' if resume_byte_pos > 0 else 'wb'
                    with open(output_path, mode) as f:
                        for chunk in response.iter_content(
                            chunk_size=8192
                        ):  # 8KB chunks for downloads
                            if chunk:  # Filter out keep-alive chunks
                                f.write(chunk)
                                progress_bar.update(len(chunk))

                    progress_bar.close()

                    # Verify file size if known
                    if total_size > 0:
                        actual_size = os.path.getsize(output_path)
                        if actual_size != total_size:
                            logger.warning(
                                f"Downloaded file size ({actual_size}) does not match expected size ({total_size})"
                            )

                    logger.info(f"Download completed: {output_path}")
                    return output_path

                except Exception as e:
                    progress_bar.close()
                    raise

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.warning(
                        f"Download attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                    )
                    import time

                    time.sleep(wait_time)
                    continue
                # Last attempt failed, raise with context
                elif isinstance(e, requests.RequestException):
                    raise UserError(f"Download failed due to network error: {e}")
                else:
                    raise UserError(f"Download failed after {max_retries} attempts: {e}")

    def delete(
        self,
        artifact_id: str = "",
        version_id: str = "",
        user_id: str = "",
        app_id: str = "",
        **kwargs,
    ) -> bool:
        """Delete this artifact version.

        Args:
            artifact_id: The artifact ID. Defaults to the artifact ID from initialization.
            version_id: The artifact version ID. Defaults to the version ID from initialization.
            user_id: The user ID. Defaults to the user ID from initialization.
            app_id: The app ID. Defaults to the app ID from initialization.
            **kwargs: Additional keyword arguments to be passed to the BaseClient.

        Returns:
            True if deletion was successful.

        Raises:
            Exception: If the artifact version deletion fails.
        """
        artifact_id = artifact_id or self.artifact_id
        version_id = version_id or self.version_id
        user_id = user_id or self.user_id
        app_id = app_id or self.app_id

        if not artifact_id:
            raise UserError("artifact_id is required")
        if not version_id:
            raise UserError("version_id is required")
        if not user_id:
            raise UserError("user_id is required")
        if not app_id:
            raise UserError("app_id is required")

        request = service_pb2.DeleteArtifactVersionRequest(
            user_app_id=self.auth_helper.get_user_app_id_proto(user_id=user_id, app_id=app_id),
            artifact_id=artifact_id,
            artifact_version_id=version_id,
        )

        response = self._grpc_request(self.STUB.DeleteArtifactVersion, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(f"Failed to delete artifact version: {response.status.description}")

        return True

    def get(
        self,
        artifact_id: str = "",
        version_id: str = "",
        user_id: str = "",
        app_id: str = "",
        **kwargs,
    ) -> resources_pb2.ArtifactVersion:
        """Get information about this artifact version.

        Args:
            artifact_id: The artifact ID. Defaults to the artifact ID from initialization.
            version_id: The artifact version ID. Defaults to the version ID from initialization.
            user_id: The user ID. Defaults to the user ID from initialization.
            app_id: The app ID. Defaults to the app ID from initialization.
            **kwargs: Additional keyword arguments to be passed to the BaseClient.

        Returns:
            The artifact version protobuf object.

        Raises:
            Exception: If the artifact version retrieval fails.
        """
        artifact_id = artifact_id or self.artifact_id
        version_id = version_id or self.version_id
        user_id = user_id or self.user_id
        app_id = app_id or self.app_id

        if not artifact_id:
            raise UserError("artifact_id is required")
        if not version_id:
            raise UserError("version_id is required")
        if not user_id:
            raise UserError("user_id is required")
        if not app_id:
            raise UserError("app_id is required")

        request = service_pb2.GetArtifactVersionRequest(
            user_app_id=self.auth_helper.get_user_app_id_proto(user_id=user_id, app_id=app_id),
            artifact_id=artifact_id,
            artifact_version_id=version_id,
        )

        response = self._grpc_request(self.STUB.GetArtifactVersion, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(f"Failed to get artifact version: {response.status.description}")

        return response.artifact_version

    def list(
        self,
        artifact_id: str = "",
        user_id: str = "",
        app_id: str = "",
        page: int = 1,
        per_page: int = DEFAULT_ARTIFACTS_PAGE_SIZE,
        **kwargs,
    ) -> Generator[resources_pb2.ArtifactVersion, None, None]:
        """List artifact versions for this artifact.

        Args:
            artifact_id: The artifact ID. Defaults to the artifact ID from initialization.
            user_id: The user ID. Defaults to the user ID from initialization.
            app_id: The app ID. Defaults to the app ID from initialization.
            page: The page number for pagination. Defaults to 1.
            per_page: The number of results per page. Defaults to 20.
            **kwargs: Additional keyword arguments to be passed to the BaseClient.

        Yields:
            ArtifactVersion protobuf objects.

        Raises:
            UserError: If required parameters are missing.
            Exception: If the artifact version listing fails.
        """
        artifact_id = artifact_id or self.artifact_id
        user_id = user_id or self.user_id
        app_id = app_id or self.app_id

        if not artifact_id:
            raise UserError("artifact_id is required")
        if not user_id:
            raise UserError("user_id is required")
        if not app_id:
            raise UserError("app_id is required")

        request = service_pb2.ListArtifactVersionsRequest(
            user_app_id=self.auth_helper.get_user_app_id_proto(user_id=user_id, app_id=app_id),
            artifact_id=artifact_id,
            page=page,
            per_page=per_page,
        )

        response = self._grpc_request(self.STUB.ListArtifactVersions, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(f"Failed to list artifact versions: {response.status.description}")

        for version_pb in response.artifact_versions:
            yield version_pb
