import abc
import grpc

from clarifai_grpc.grpc.api import service_pb2_grpc


class V2Stub(abc.ABC):
  """Abstract base class of clarifai api rpc client stubs"""


class RpcCallable(abc.ABC):
  """Abstract base class of clarifai api rpc callables"""


# add grpc classes as subclasses of the abcs, so they also succeed in isinstance calls
def _register_classes():
  V2Stub.register(service_pb2_grpc.V2Stub)
  for name in dir(grpc):
    if name.endswith('Callable'):
      RpcCallable.register(getattr(grpc, name))


_register_classes()
