import time
from typing import Iterator, Optional, Union

from clarifai_grpc.grpc.api import service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from clarifai_protocol import BaseRunner
from clarifai_protocol.utils.health import HealthProbeRequestHandler, start_health_server_thread

from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.utils.constants import STATUS_FAIL, STATUS_MIXED, STATUS_OK, STATUS_UNKNOWN
from clarifai.utils.logging import get_req_id_from_context, logger
from clarifai.utils.secrets import inject_secrets, req_secrets_context

from ..utils.url_fetcher import ensure_urls_downloaded
from .model_class import ModelClass


class ModelRunner(BaseRunner):
    """
    This is a subclass of the runner class which will handle only the work items relevant to models.
    """

    def __init__(
        self,
        model: ModelClass,
        runner_id: str,
        nodepool_id: str,
        compute_cluster_id: str,
        user_id: Optional[str] = None,
        check_runner_exists: bool = True,
        base_url: str = "https://api.clarifai.com",
        pat: Optional[str] = None,
        token: Optional[str] = None,
        num_parallel_polls: int = 4,
        health_check_port: Union[int, None] = 8080,
        **kwargs,
    ) -> None:
        super().__init__(
            runner_id,
            nodepool_id,
            compute_cluster_id,
            user_id,
            check_runner_exists,
            base_url,
            pat,
            token,
            num_parallel_polls,
            **kwargs,
        )
        self.model = model

        # Store authentication parameters for URL fetching
        self._user_id = user_id
        self._pat = pat
        self._token = token
        self._base_url = base_url

        # Create auth helper if we have sufficient authentication information
        self._auth_helper: Optional[ClarifaiAuthHelper] = None
        if self._user_id and (self._pat or self._token):
            try:
                self._auth_helper = ClarifaiAuthHelper(
                    user_id=self._user_id,
                    app_id="",  # app_id not needed for URL fetching
                    pat=self._pat or "",
                    token=self._token or "",
                    base=self._base_url,
                    validate=False,  # Don't validate since app_id is empty
                )
            except Exception:
                # If auth helper creation fails, proceed without authentication
                self._auth_helper = None

        # if the model has a handle_liveness_probe method, call it to determine liveness
        # otherwise rely on self.is_alive from the protocol
        if hasattr(self.model, 'handle_liveness_probe'):
            HealthProbeRequestHandler.handle_liveness_probe = self.model.handle_liveness_probe

        # if the model has a handle_readiness_probe method, call it to determine readiness
        # otherwise rely on self.is_ready from the protocol
        if hasattr(self.model, 'handle_readiness_probe'):
            HealthProbeRequestHandler.handle_readiness_probe = self.model.handle_readiness_probe

        # if the model has a handle_startup_probe method, call it to determine startup
        # otherwise rely on self.is_startup from the protocol
        if hasattr(self.model, 'handle_startup_probe'):
            HealthProbeRequestHandler.handle_startup_probe = self.model.handle_startup_probe

        # After model load successfully set the health probe to ready and startup
        HealthProbeRequestHandler.is_ready = True
        HealthProbeRequestHandler.is_startup = True

        start_health_server_thread(port=health_check_port, address='')

    def get_runner_item_output_for_status(
        self, status: status_pb2.Status
    ) -> service_pb2.RunnerItemOutput:
        """
        Set the error message in the RunnerItemOutput message subfield, used during exception handling
        where we may only have a status to return.

        Args:
          status: status_pb2.Status - the status to return

        Returns:
          service_pb2.RunnerItemOutput - the RunnerItemOutput message with the status set
        """
        rio = service_pb2.RunnerItemOutput(
            multi_output_response=service_pb2.MultiOutputResponse(status=status)
        )
        return rio

    def runner_item_predict(
        self, runner_item: service_pb2.RunnerItem
    ) -> service_pb2.RunnerItemOutput:
        """
        Run the model on the given request. You shouldn't need to override this method, see run_input
        for the implementation to process each input in the request.

        Args:
          request: service_pb2.PostModelOutputsRequest - the request to run the model on

        Returns:
          service_pb2.MultiOutputResponse - the response from the model's run_input implementation.
        """

        if not runner_item.HasField('post_model_outputs_request'):
            raise Exception("Unexpected work item type: {}".format(runner_item))
        request = runner_item.post_model_outputs_request
        ensure_urls_downloaded(request, auth_helper=self._auth_helper)
        inject_secrets(request)
        start_time = time.time()
        req_id = get_req_id_from_context()
        status_str = STATUS_UNKNOWN
        # Operation name for logging purposes
        endpoint = "model_predict"

        # if method_name == '_GET_SIGNATURES' then the request is for getting signatures and we don't want to log it.
        # This is a workaround to avoid logging the _GET_SIGNATURES method call.
        method_name = None
        logging = True
        if len(request.inputs) > 0 and '_method_name' in request.inputs[0].data.metadata:
            method_name = request.inputs[0].data.metadata['_method_name']
        if method_name == '_GET_SIGNATURES':
            logging = False

        # Use req_secrets_context to temporarily set request-type secrets as environment variables
        with req_secrets_context(request):
            resp = self.model.predict_wrapper(request)

        # if we have any non-successful code already it's an error we can return.
        if (
            resp.status.code != status_code_pb2.SUCCESS
            and resp.status.code != status_code_pb2.ZERO
        ):
            status_str = f"{resp.status.code} ERROR"
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"{endpoint} | {status_str} | {duration_ms:.2f}ms | req_id={req_id}")
            return service_pb2.RunnerItemOutput(multi_output_response=resp)
        successes = []
        for output in resp.outputs:
            if not output.HasField('status') or not output.status.code:
                raise Exception(
                    "Output must have a status code, please check the model implementation."
                )
            successes.append(output.status.code == status_code_pb2.SUCCESS)
        if all(successes):
            status = status_pb2.Status(
                code=status_code_pb2.SUCCESS,
                description="Success",
            )
            status_str = STATUS_OK
        elif any(successes):
            status = status_pb2.Status(
                code=status_code_pb2.MIXED_STATUS,
                description="Mixed Status",
            )
            status_str = STATUS_MIXED
        else:
            status = status_pb2.Status(
                code=status_code_pb2.FAILURE,
                description="Failed",
            )
            status_str = STATUS_FAIL

        resp.status.CopyFrom(status)
        if logging:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"{endpoint} | {status_str} | {duration_ms:.2f}ms | req_id={req_id}")
        return service_pb2.RunnerItemOutput(multi_output_response=resp)

    def runner_item_generate(
        self, runner_item: service_pb2.RunnerItem
    ) -> Iterator[service_pb2.RunnerItemOutput]:
        # Call the generate() method the underlying model implements.

        if not runner_item.HasField('post_model_outputs_request'):
            raise Exception("Unexpected work item type: {}".format(runner_item))
        request = runner_item.post_model_outputs_request
        ensure_urls_downloaded(request, auth_helper=self._auth_helper)
        inject_secrets(request)

        # --- Live logging additions ---
        start_time = time.time()
        req_id = get_req_id_from_context()
        status_str = STATUS_UNKNOWN
        endpoint = "model_generate"

        # Use req_secrets_context to temporarily set request-type secrets as environment variables
        with req_secrets_context(request):
            for resp in self.model.generate_wrapper(request):
                # if we have any non-successful code already it's an error we can return.
                if (
                    resp.status.code != status_code_pb2.SUCCESS
                    and resp.status.code != status_code_pb2.ZERO
                ):
                    status_str = f"{resp.status.code} ERROR"
                    duration_ms = (time.time() - start_time) * 1000
                    logger.info(
                        f"{endpoint} | {status_str} | {duration_ms:.2f}ms | req_id={req_id}"
                    )
                    yield service_pb2.RunnerItemOutput(multi_output_response=resp)
                    continue
                successes = []
                for output in resp.outputs:
                    if not output.HasField('status') or not output.status.code:
                        raise Exception(
                            "Output must have a status code, please check the model implementation."
                        )
                    successes.append(output.status.code == status_code_pb2.SUCCESS)
                if all(successes):
                    status = status_pb2.Status(
                        code=status_code_pb2.SUCCESS,
                        description="Success",
                    )
                    status_str = STATUS_OK
                elif any(successes):
                    status = status_pb2.Status(
                        code=status_code_pb2.MIXED_STATUS,
                        description="Mixed Status",
                    )
                    status_str = STATUS_MIXED
                else:
                    status = status_pb2.Status(
                        code=status_code_pb2.FAILURE,
                        description="Failed",
                    )
                    status_str = STATUS_FAIL
                resp.status.CopyFrom(status)

                yield service_pb2.RunnerItemOutput(multi_output_response=resp)

        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"{endpoint} | {status_str} | {duration_ms:.2f}ms | req_id={req_id}")

    def runner_item_stream(
        self, runner_item_iterator: Iterator[service_pb2.RunnerItem]
    ) -> Iterator[service_pb2.RunnerItemOutput]:
        # Call the generate() method the underlying model implements.
        start_time = time.time()
        req_id = get_req_id_from_context()
        status_str = STATUS_UNKNOWN
        endpoint = "model_stream"

        # Get the first request to establish secrets context
        first_request = None
        runner_items = list(runner_item_iterator)  # Convert to list to avoid consuming iterator
        if runner_items:
            first_request = runner_items[0].post_model_outputs_request

        # Use req_secrets_context based on the first request (secrets should be consistent across stream)
        with req_secrets_context(first_request):
            for resp in self.model.stream_wrapper(
                pmo_iterator(iter(runner_items), auth_helper=self._auth_helper)
            ):
                # if we have any non-successful code already it's an error we can return.
                if (
                    resp.status.code != status_code_pb2.SUCCESS
                    and resp.status.code != status_code_pb2.ZERO
                ):
                    status_str = f"{resp.status.code} ERROR"
                    duration_ms = (time.time() - start_time) * 1000
                    logger.info(
                        f"{endpoint} | {status_str} | {duration_ms:.2f}ms | req_id={req_id}"
                    )
                    yield service_pb2.RunnerItemOutput(multi_output_response=resp)
                    continue
                successes = []
                for output in resp.outputs:
                    if not output.HasField('status') or not output.status.code:
                        raise Exception(
                            "Output must have a status code, please check the model implementation."
                        )
                    successes.append(output.status.code == status_code_pb2.SUCCESS)
                if all(successes):
                    status = status_pb2.Status(
                        code=status_code_pb2.SUCCESS,
                        description="Success",
                    )
                    status_str = STATUS_OK
                elif any(successes):
                    status = status_pb2.Status(
                        code=status_code_pb2.MIXED_STATUS,
                        description="Mixed Status",
                    )
                    status_str = STATUS_MIXED
                else:
                    status = status_pb2.Status(
                        code=status_code_pb2.RUNNER_PROCESSING_FAILED,
                        description="Runner Processing Failed",
                    )
                    status_str = STATUS_FAIL
                resp.status.CopyFrom(status)

                yield service_pb2.RunnerItemOutput(multi_output_response=resp)

        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"{endpoint} | {status_str} | {duration_ms:.2f}ms | req_id={req_id}")

    def set_model(self, model: ModelClass):
        """Set the model for this runner."""
        self.model = model


def pmo_iterator(runner_item_iterator, auth_helper=None):
    for runner_item in runner_item_iterator:
        if not runner_item.HasField('post_model_outputs_request'):
            raise Exception("Unexpected work item type: {}".format(runner_item))
        ensure_urls_downloaded(runner_item.post_model_outputs_request, auth_helper=auth_helper)
        inject_secrets(runner_item.post_model_outputs_request)
        yield runner_item.post_model_outputs_request
