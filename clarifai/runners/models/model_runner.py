from typing import Iterator

from clarifai_grpc.grpc.api import service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from clarifai_protocol import BaseRunner
from clarifai_protocol.utils.health import HealthProbeRequestHandler

from clarifai.client.auth.helper import ClarifaiAuthHelper

from ..utils.url_fetcher import ensure_urls_downloaded
from .model_class import ModelClass


class ModelRunner(BaseRunner, HealthProbeRequestHandler):
    """
    This is a subclass of the runner class which will handle only the work items relevant to models.
    """

    def __init__(
        self,
        model: ModelClass,
        runner_id: str,
        nodepool_id: str,
        compute_cluster_id: str,
        user_id: str = None,
        check_runner_exists: bool = True,
        base_url: str = "https://api.clarifai.com",
        pat: str = None,
        token: str = None,
        num_parallel_polls: int = 4,
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
        self._auth_helper = None
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

        # After model load successfully set the health probe to ready and startup
        HealthProbeRequestHandler.is_ready = True
        HealthProbeRequestHandler.is_startup = True

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

        resp = self.model.predict_wrapper(request)
        # if we have any non-successful code already it's an error we can return.
        if (
            resp.status.code != status_code_pb2.SUCCESS
            and resp.status.code != status_code_pb2.ZERO
        ):
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
        elif any(successes):
            status = status_pb2.Status(
                code=status_code_pb2.MIXED_STATUS,
                description="Mixed Status",
            )
        else:
            status = status_pb2.Status(
                code=status_code_pb2.FAILURE,
                description="Failed",
            )

        resp.status.CopyFrom(status)
        return service_pb2.RunnerItemOutput(multi_output_response=resp)

    def runner_item_generate(
        self, runner_item: service_pb2.RunnerItem
    ) -> Iterator[service_pb2.RunnerItemOutput]:
        # Call the generate() method the underlying model implements.

        if not runner_item.HasField('post_model_outputs_request'):
            raise Exception("Unexpected work item type: {}".format(runner_item))
        request = runner_item.post_model_outputs_request
        ensure_urls_downloaded(request, auth_helper=self._auth_helper)

        for resp in self.model.generate_wrapper(request):
            # if we have any non-successful code already it's an error we can return.
            if (
                resp.status.code != status_code_pb2.SUCCESS
                and resp.status.code != status_code_pb2.ZERO
            ):
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
            elif any(successes):
                status = status_pb2.Status(
                    code=status_code_pb2.MIXED_STATUS,
                    description="Mixed Status",
                )
            else:
                status = status_pb2.Status(
                    code=status_code_pb2.FAILURE,
                    description="Failed",
                )
            resp.status.CopyFrom(status)

            yield service_pb2.RunnerItemOutput(multi_output_response=resp)

    def runner_item_stream(
        self, runner_item_iterator: Iterator[service_pb2.RunnerItem]
    ) -> Iterator[service_pb2.RunnerItemOutput]:
        # Call the generate() method the underlying model implements.
        for resp in self.model.stream_wrapper(pmo_iterator(runner_item_iterator)):
            # if we have any non-successful code already it's an error we can return.
            if (
                resp.status.code != status_code_pb2.SUCCESS
                and resp.status.code != status_code_pb2.ZERO
            ):
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
            elif any(successes):
                status = status_pb2.Status(
                    code=status_code_pb2.MIXED_STATUS,
                    description="Mixed Status",
                )
            else:
                status = status_pb2.Status(
                    code=status_code_pb2.FAILURE,
                    description="Failed",
                )
            resp.status.CopyFrom(status)

            yield service_pb2.RunnerItemOutput(multi_output_response=resp)


def pmo_iterator(runner_item_iterator, auth_helper=None):
    for runner_item in runner_item_iterator:
        if not runner_item.HasField('post_model_outputs_request'):
            raise Exception("Unexpected work item type: {}".format(runner_item))
        ensure_urls_downloaded(runner_item.post_model_outputs_request, auth_helper=auth_helper)
        yield runner_item.post_model_outputs_request
