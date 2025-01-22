"""
This is simply the main file for the server that imports ModelRunner implementation
and starts the server.
"""

import argparse
import importlib.util
import inspect
import os
import sys
from concurrent import futures

from clarifai_grpc.grpc.api import service_pb2_grpc
from clarifai_protocol import BaseRunner
from clarifai_protocol.utils.grpc_server import GRPCServer

from clarifai.runners.models.model_servicer import ModelServicer
from clarifai.runners.models.model_upload import ModelUploader
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
      '--start_dev_server',
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

  uploader = ModelUploader(parsed_args.model_path)

  # server config with init args etc.
  # server_info contains: model_file, model_class, model_args
  server_config = uploader.config["server_info"]

  model_file = server_config.get("model_file")
  if model_file:
    model_file = os.path.join(parsed_args.model_path, model_file)
    if not os.path.exists(model_file):
      raise Exception(f"Model file {model_file} does not exist.")
  else:
    # look for default model.py file location
    for loc in ["model.py", "1/model.py"]:
      model_file = os.path.join(parsed_args.model_path, loc)
      if os.path.exists(model_file):
        break
    if not os.path.exists(model_file):
      raise Exception("Model file not found.")

  module_name = os.path.basename(model_file).replace(".py", "")

  spec = importlib.util.spec_from_file_location(module_name, model_file)
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)

  if server_config.get("model_class"):
    model_class = getattr(module, server_config["model_class"])
  else:
    # Find all classes in the model.py file that are subclasses of ModelClass
    classes = [
        cls for _, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, ModelClass) and cls.__module__ == runner_module.__name__
    ]
    #  Ensure there is exactly one subclass of BaseRunner in the model.py file
    if len(classes) != 1:
      raise Exception("Could not determine model class. Please specify it in the config with server_info.model_class.")
    model_class = classes[0]

  model_args = server_config.get("model_args", {})

  # initialize the model class with the args.
  model = model_class(**model_args)

  # Setup the grpc server for local development.
  if parsed_args.start_dev_server:

    # We validate that we have checkpoints downloaded before constructing MyRunner which
    # will call load_model()
    uploader.download_checkpoints()

    # initialize the Runner class. This is what the user implements.
    # we aren't going to call runner.start() to engage with the API so IDs are not necessary.
    runner = ModelRunner(
        runner_id="n/a",
        nodepool_id="n/a",
        compute_cluster_id="n/a",
        user_id="n/a",
        health_check_port=None,  # not needed when running local server
        model=model,
    )

    # initialize the servicer with the runner so that it gets the predict(), generate(), stream() classes.
    servicer = ModelServicer(runner)

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
        runner_id=os.environ["CLARIFAI_RUNNER_ID"],
        nodepool_id=os.environ["CLARIFAI_NODEPOOL_ID"],
        compute_cluster_id=os.environ["CLARIFAI_COMPUTE_CLUSTER_ID"],
        base_url=os.environ["CLARIFAI_API_BASE"],
        num_parallel_polls=int(os.environ.get("CLARIFAI_NUM_THREADS", 1)),
        model=model,
    )
    runner.start()  # start the runner to fetch work from the API.


if __name__ == '__main__':
  main()
