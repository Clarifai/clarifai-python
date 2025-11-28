from typing import Dict, Generator

from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.base import BaseClient
from clarifai.constants.artifact import (
    DEFAULT_ARTIFACTS_PAGE_SIZE,
)
from clarifai.errors import UserError


def handle_grpc_error(func, *args, **kwargs):
    """Handle gRPC errors with graceful fallbacks for missing APIs.

    Args:
        func: The gRPC function to call
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function call

    Raises:
        UserError: If the API is not available or other errors occur
    """
    try:
        return func(*args, **kwargs)
    except AttributeError as e:
        if "has no attribute" in str(e) and any(
            name in str(e) for name in ["Artifact", "PostArtifact", "ListArtifact"]
        ):
            raise UserError(
                "Artifact API is not yet available in the current gRPC client version. Please update your clarifai-grpc package."
            )
        raise UserError(f"gRPC method not found: {e}")
    except Exception as e:
        # Handle other gRPC errors
        if hasattr(e, 'code') and hasattr(e, 'details'):
            raise UserError(f"API Error: {e.details()}")
        raise UserError(f"Request failed: {e}")


class Artifact(BaseClient):
    """Artifact client for managing artifacts in Clarifai."""

    def __init__(self, artifact_id: str = "", user_id: str = "", app_id: str = "", **kwargs):
        """Initialize the Artifact client.

        Args:
            artifact_id: The artifact ID.
            user_id: The user ID.
            app_id: The app ID.
            **kwargs: Additional keyword arguments to be passed to the BaseClient.
        """
        super().__init__(**kwargs)
        self.artifact_id = artifact_id
        self.user_id = user_id or self.auth_helper.user_id
        self.app_id = app_id

    @property
    def id(self) -> str:
        """Get the artifact ID."""
        return self.artifact_id

    def __repr__(self) -> str:
        init_params = [f"artifact_id='{self.artifact_id}'"]
        if self.user_id:
            init_params.append(f"user_id='{self.user_id}'")
        if self.app_id:
            init_params.append(f"app_id='{self.app_id}'")
        return f"Artifact({', '.join(init_params)})"

    def create(
        self,
        artifact_id: str,
        user_id: str = "",
        app_id: str = "",
    ) -> "Artifact":
        """Create a new artifact.

        Args:
            artifact_id: The artifact ID to create.
            user_id: The user ID. Defaults to the user ID from initialization.
            app_id: The app ID. Defaults to the app ID from initialization.

        Returns:
            An Artifact object for the created artifact.

        Raises:
            Exception: If the artifact creation fails.
        """
        user_id = user_id or self.user_id
        app_id = app_id or self.app_id

        if not user_id:
            raise UserError("user_id is required")
        if not app_id:
            raise UserError("app_id is required")

        request = service_pb2.PostArtifactsRequest(
            user_app_id=self.auth_helper.get_user_app_id_proto(user_id=user_id, app_id=app_id),
            artifacts=[
                resources_pb2.Artifact(
                    id=artifact_id,
                    user_id=user_id,
                    app_id=app_id,
                )
            ],
        )

        response = handle_grpc_error(self._grpc_request, self.STUB.PostArtifacts, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(f"Failed to create artifact: {response.status.description}")

        return Artifact(
            artifact_id=artifact_id, user_id=user_id, app_id=app_id, **self._get_client_params()
        )

    def delete(
        self,
        artifact_id: str = "",
        user_id: str = "",
        app_id: str = "",
    ) -> bool:
        """Delete an artifact.

        Args:
            artifact_id: The artifact ID. Defaults to the artifact ID from initialization.
            user_id: The user ID. Defaults to the user ID from initialization.
            app_id: The app ID. Defaults to the app ID from initialization.

        Returns:
            True if deletion was successful.

        Raises:
            Exception: If the artifact deletion fails.
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

        request = service_pb2.DeleteArtifactRequest(
            user_app_id=self.auth_helper.get_user_app_id_proto(user_id=user_id, app_id=app_id),
            id=artifact_id,
        )

        response = handle_grpc_error(self._grpc_request, self.STUB.DeleteArtifact, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(f"Failed to delete artifact: {response.status.description}")

        return True

    def info(
        self,
        artifact_id: str = "",
        user_id: str = "",
        app_id: str = "",
    ) -> Dict:
        """Get information about an artifact.

        Args:
            artifact_id: The artifact ID. Defaults to the artifact ID from initialization.
            user_id: The user ID. Defaults to the user ID from initialization.
            app_id: The app ID. Defaults to the app ID from initialization.

        Returns:
            A dictionary containing the artifact information.

        Raises:
            Exception: If the artifact retrieval fails.
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

        request = service_pb2.GetArtifactRequest(
            user_app_id=self.auth_helper.get_user_app_id_proto(user_id=user_id, app_id=app_id),
            id=artifact_id,
        )

        response = handle_grpc_error(self._grpc_request, self.STUB.GetArtifact, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(f"Failed to get artifact: {response.status.description}")

        return {
            "id": response.artifact.id,
            "user_id": response.artifact.user_id,
            "app_id": response.artifact.app_id,
            "created_at": response.artifact.created_at.ToDatetime()
            if response.artifact.created_at
            else None,
            "modified_at": response.artifact.modified_at.ToDatetime()
            if response.artifact.modified_at
            else None,
            "deleted_at": response.artifact.deleted_at.ToDatetime()
            if response.artifact.deleted_at
            else None,
            "artifact_version": {
                "id": response.artifact.artifact_version.id,
                "description": response.artifact.artifact_version.description,
            }
            if response.artifact.artifact_version
            else None,
        }

    @staticmethod
    def list(
        user_id: str,
        app_id: str,
        page: int = 1,
        per_page: int = DEFAULT_ARTIFACTS_PAGE_SIZE,
        **kwargs,
    ) -> Generator["Artifact", None, None]:
        """List artifacts in an app.

        Args:
            user_id: The user ID.
            app_id: The app ID.
            page: The page number for pagination. Defaults to 1.
            per_page: The number of results per page. Defaults to 20.
            **kwargs: Additional keyword arguments to be passed to the BaseClient.

        Yields:
            Artifact objects.

        Raises:
            Exception: If the artifact listing fails.
        """
        client = BaseClient(**kwargs)

        request = service_pb2.ListArtifactsRequest(
            user_app_id=client.auth_helper.get_user_app_id_proto(user_id=user_id, app_id=app_id),
            page=page,
            per_page=per_page,
        )

        response = handle_grpc_error(client._grpc_request, client.STUB.ListArtifacts, request)

        if response.status.code != status_code_pb2.SUCCESS:
            raise Exception(f"Failed to list artifacts: {response.status.description}")

        for artifact_pb in response.artifacts:
            yield Artifact(artifact_id=artifact_pb.id, user_id=user_id, app_id=app_id, **kwargs)

    def _get_client_params(self) -> Dict:
        """Get the client parameters for creating new instances."""
        return {
            "user_id": self.user_id,
            "app_id": self.app_id,
        }

    def exists(
        self,
        artifact_id: str = "",
        user_id: str = "",
        app_id: str = "",
    ) -> bool:
        """Check if an artifact exists.

        Args:
            artifact_id: The artifact ID. Defaults to the artifact ID from initialization.
            user_id: The user ID. Defaults to the user ID from initialization.
            app_id: The app ID. Defaults to the app ID from initialization.

        Returns:
            True if the artifact exists, False otherwise.
        """
        try:
            self.info(artifact_id=artifact_id, user_id=user_id, app_id=app_id)
            return True
        except Exception:
            return False

    def _get_client_params(self) -> Dict:
        """Get the client parameters for creating new instances."""
        return {
            "user_id": self.user_id,
            "app_id": self.app_id,
        }
