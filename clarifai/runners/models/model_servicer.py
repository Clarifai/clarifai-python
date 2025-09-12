import os
from itertools import tee
from typing import Iterator

from clarifai_grpc.grpc.api import service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2

from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.utils.secrets import inject_secrets

from ..utils.url_fetcher import ensure_urls_downloaded

_RAISE_EXCEPTIONS = os.getenv("RAISE_EXCEPTIONS", "false").lower() in ("true", "1")


class ModelServicer(service_pb2_grpc.V2Servicer):
    """
    This is the servicer that will handle the gRPC requests from either the dev server or runner loop.
    """

    def __init__(self, model):
        """
        Args:
            model: The class that will handle the model logic. Must implement predict(),
        generate(), stream().
        """
        self.model = model

        # Try to create auth helper from environment variables if available
        self._auth_helper = None
        try:
            user_id = os.environ.get("CLARIFAI_USER_ID")
            pat = os.environ.get("CLARIFAI_PAT")
            token = os.environ.get("CLARIFAI_SESSION_TOKEN")
            base_url = os.environ.get("CLARIFAI_API_BASE", "https://api.clarifai.com")

            if user_id and (pat or token):
                self._auth_helper = ClarifaiAuthHelper(
                    user_id=user_id,
                    app_id="",  # app_id not needed for URL fetching
                    pat=pat or "",
                    token=token or "",
                    base=base_url,
                    validate=False,  # Don't validate since app_id is empty
                )
        except Exception:
            # If auth helper creation fails, proceed without authentication
            self._auth_helper = None

    def PostModelOutputs(
        self, request: service_pb2.PostModelOutputsRequest, context=None
    ) -> service_pb2.MultiOutputResponse:
        """
        This is the method that will be called when the servicer is run. It takes in an input and
        returns an output.
        """

        # Download any urls that are not already bytes.
        ensure_urls_downloaded(request, auth_helper=self._auth_helper)
        inject_secrets(request)

        try:
            return self.model.predict_wrapper(request)
        except Exception as e:
            if _RAISE_EXCEPTIONS:
                raise
            return service_pb2.MultiOutputResponse(
                status=status_pb2.Status(
                    code=status_code_pb2.MODEL_PREDICTION_FAILED,
                    description="Failed",
                    details="",
                    internal_details=str(e),
                )
            )

    def GenerateModelOutputs(
        self, request: service_pb2.PostModelOutputsRequest, context=None
    ) -> Iterator[service_pb2.MultiOutputResponse]:
        """
        This is the method that will be called when the servicer is run. It takes in an input and
        returns an output.
        """
        # Download any urls that are not already bytes.
        ensure_urls_downloaded(request, auth_helper=self._auth_helper)
        inject_secrets(request)

        try:
            yield from self.model.generate_wrapper(request)
        except Exception as e:
            if _RAISE_EXCEPTIONS:
                raise
            yield service_pb2.MultiOutputResponse(
                status=status_pb2.Status(
                    code=status_code_pb2.MODEL_PREDICTION_FAILED,
                    description="Failed",
                    details="",
                    internal_details=str(e),
                )
            )

    def StreamModelOutputs(
        self, request: Iterator[service_pb2.PostModelOutputsRequest], context=None
    ) -> Iterator[service_pb2.MultiOutputResponse]:
        """
        This is the method that will be called when the servicer is run. It takes in an input and
        returns an output.
        """
        # Duplicate the iterator
        request, request_copy = tee(request)

        # Download any urls that are not already bytes.
        for req in request:
            ensure_urls_downloaded(req, auth_helper=self._auth_helper)
            inject_secrets(req)

        try:
            yield from self.model.stream_wrapper(request_copy)
        except Exception as e:
            if _RAISE_EXCEPTIONS:
                raise
            yield service_pb2.MultiOutputResponse(
                status=status_pb2.Status(
                    code=status_code_pb2.MODEL_PREDICTION_FAILED,
                    description="Failed",
                    details="",
                    internal_details=str(e),
                )
            )

    def set_model(self, model):
        self.model = model
