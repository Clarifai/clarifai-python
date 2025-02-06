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
      default=32,
      help="The number of threads to use for the gRPC server.",
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
      help=
      'Set to true to enable TLS (default: False) since this server is meant for local development only.',
  )
  parser.add_argument(
      '--grpc',
      action='store_true',
      default=False,
      help=
      'Set to true to start the gRPC server (default: False). If set to false, the server will not start and only the runner loop will start to fetch work from the API.',
  )
  parser.add_argument(
      '--model_path',
      type=str,
      required=True,
      help='The path to the model directory that contains implemention of the model.',
  )

  parsed_args = parser.parse_args()

  builder = ModelBuilder(parsed_args.model_path, download_validation_only=True)

  model = builder.create_model_instance()

  # Setup the grpc server for local development.
  if parsed_args.grpc:

    # initialize the servicer with the runner so that it gets the predict(), generate(), stream() classes.
    servicer = ModelServicer(model)

    server = GRPCServer(
        futures.ThreadPoolExecutor(
            max_workers=parsed_args.pool_size,
            thread_name_prefix="ServeCalls",
        ),
        parsed_args.max_msg_length,
        parsed_args.max_queue_size,
    )
    server.add_port_to_server('[::]:%s' % parsed_args.port, parsed_args.enable_tls)

    service_pb2_grpc.add_V2Servicer_to_server(servicer, server)
    server.start()
    logger.info("Started server on port %s", parsed_args.port)
    logger.info(f"Access the model at http://localhost:{parsed_args.port}")
    server.wait_for_termination()
  else:  # start the runner with the proper env variables and as a runner protocol.

    # initialize the Runner class. This is what the user implements.
    runner = ModelRunner(
        model=model,
        runner_id=os.environ["CLARIFAI_RUNNER_ID"],
        nodepool_id=os.environ["CLARIFAI_NODEPOOL_ID"],
        compute_cluster_id=os.environ["CLARIFAI_COMPUTE_CLUSTER_ID"],
        base_url=os.environ.get("CLARIFAI_API_BASE", "https://api.clarifai.com"),
        num_parallel_polls=int(os.environ.get("CLARIFAI_NUM_THREADS", 1)),
    )
    runner.start()  # start the runner to fetch work from the API.


if __name__ == '__main__':
  main()
