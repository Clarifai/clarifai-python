"""
This is simply the main file for the server that imports ModelRunner implementation
and starts the server.
"""

import argparse
import os
from concurrent import futures

from clarifai_grpc.grpc.api import service_pb2_grpc
from clarifai_protocol.utils.grpc_server import GRPCServer

from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.runners.models.model_runner import ModelRunner
from clarifai.runners.models.model_servicer import ModelServicer
from clarifai.utils.logging import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help="The port to host the gRPC server at.",
        choices=range(1024, 65535),
    )
    parser.add_argument(
        '--pool_size',
        type=int,
        default=os.environ.get('CLARIFAI_NUM_THREADS', 32),
        help="The number of threads to use for the gRPC server. Runner (ie. grpc=False) threads are read from the config file and ModelBuilder defaults.",
        choices=range(1, 129),
    )  # pylint: disable=range-builtin-not-iterating
    parser.add_argument(
        '--max_queue_size',
        type=int,
        default=10,
        help='Max queue size of requests before we begin to reject requests (default: 10).',
        choices=range(1, 21),
    )  # pylint: disable=range-builtin-not-iterating
    parser.add_argument(
        '--max_msg_length',
        type=int,
        default=1024 * 1024 * 1024,
        help='Max message length of grpc requests (default: 1 GB).',
    )
    parser.add_argument(
        '--enable_tls',
        action='store_true',
        default=False,
        help='Set to true to enable TLS (default: False) since this server is meant for local development only.',
    )
    parser.add_argument(
        '--grpc',
        action='store_true',
        default=False,
        help='Set to true to start the gRPC server (default: False). If set to false, the server will not start and only the runner loop will start to fetch work from the API.',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='The path to the model directory that contains implemention of the model.',
    )

    parsed_args = parser.parse_args()

    serve(
        model_path=parsed_args.model_path,
        port=parsed_args.port,
        pool_size=parsed_args.pool_size,
        max_queue_size=parsed_args.max_queue_size,
        max_msg_length=parsed_args.max_msg_length,
        enable_tls=parsed_args.enable_tls,
        grpc=parsed_args.grpc,
    )


def serve(
    model_path,
    port=8000,
    pool_size=32,
    num_threads=0,
    max_queue_size=10,
    max_msg_length=1024 * 1024 * 1024,
    enable_tls=False,
    grpc=False,
    user_id: str = os.environ.get("CLARIFAI_USER_ID", None),
    compute_cluster_id: str = os.environ.get("CLARIFAI_COMPUTE_CLUSTER_ID", None),
    nodepool_id: str = os.environ.get("CLARIFAI_NODEPOOL_ID", None),
    runner_id: str = os.environ.get("CLARIFAI_RUNNER_ID", None),
    base_url: str = os.environ.get("CLARIFAI_API_BASE", "https://api.clarifai.com"),
    pat: str = os.environ.get("CLARIFAI_PAT", None),
    context=None,  # This is the current context object that contains user_id, app_id, model_id, etc.
):
    builder = ModelBuilder(model_path, download_validation_only=True)

    model = builder.create_model_instance()

    # `num_threads` can be set in config.yaml or via the environment variable CLARIFAI_NUM_THREADS="<integer>".
    # Note: The value in config.yaml takes precedence over the environment variable.
    if num_threads == 0:
        num_threads = builder.config.get("num_threads")
    # Setup the grpc server for local development.
    if grpc:
        # initialize the servicer with the runner so that it gets the predict(), generate(), stream() classes.
        servicer = ModelServicer(model)

        server = GRPCServer(
            futures.ThreadPoolExecutor(
                max_workers=pool_size,
                thread_name_prefix="ServeCalls",
            ),
            max_msg_length,
            max_queue_size,
        )
        server.add_port_to_server('[::]:%s' % port, enable_tls)

        service_pb2_grpc.add_V2Servicer_to_server(servicer, server)
        server.start()
        logger.info("Started server on port %s", port)
        logger.info(f"Access the model at http://localhost:{port}")
        server.wait_for_termination()
    else:  # start the runner with the proper env variables and as a runner protocol.
        # initialize the Runner class. This is what the user implements.
        runner = ModelRunner(
            model=model,
            user_id=user_id,
            compute_cluster_id=compute_cluster_id,
            nodepool_id=nodepool_id,
            runner_id=runner_id,
            base_url=base_url,
            pat=pat,
            num_parallel_polls=num_threads,
        )

        if context is None:
            logger.debug("Context is None. Skipping code snippet generation.")
        else:
            method_signatures = builder.get_method_signatures(mocking=False)
            from clarifai.runners.utils import code_script

            snippet = code_script.generate_client_script(
                method_signatures,
                user_id=context.user_id,
                app_id=context.app_id,
                model_id=context.model_id,
                deployment_id=context.deployment_id,
                base_url=context.api_base,
            )
            logger.info(
                "✅ Your model is running locally and is ready for requests from the API...\n"
            )
            logger.info(
                f"> Code Snippet: To call your model via the API, use this code snippet:\n{snippet}"
            )
            logger.info(
                f"> Playground:   To chat with your model, visit: {context.ui}/playground?model={context.model_id}__{context.model_version_id}&user_id={context.user_id}&app_id={context.app_id}\n"
            )
            logger.info(
                f"> API URL:      To call your model via the API, use this model URL: {context.ui}/users/{context.user_id}/apps/{context.app_id}/models/{context.model_id}\n"
            )
            logger.info("Press CTRL+C to stop the runner.\n")
        runner.start()  # start the runner to fetch work from the API.


if __name__ == '__main__':
    main()
