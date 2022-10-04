from google.protobuf.json_format import _Parser
from google.protobuf.message import Message  # noqa

from clarifai.rest.grpc.proto.clarifai.api.utils import extensions_pb2

# Python 3 deprecates getargspec and introduces getfullargspec, which Python 2 doesn't have.
try:
  from inspect import getfullargspec as get_args
except ImportError:
  from inspect import getargspec as get_args

DEFAULT_MAX_RECURSION_DEPTH = 64


def dict_to_protobuf(protobuf_class, js_dict, ignore_unknown_fields=False):
  # type: (type(Message), dict, bool) -> Message
  message = protobuf_class()

  parser = _CustomParser(ignore_unknown_fields, None, DEFAULT_MAX_RECURSION_DEPTH)

  parser.ConvertMessage(js_dict, message, None)
  return message


class _CustomParser(_Parser):

  def _ConvertFieldValuePair(self, js, message, path=None):
    """
    Because of fields with custom extensions such as cl_default_float, we need
    to adjust the original's method's JSON object parameter by setting them explicitly to the
    default value.
    """

    message_descriptor = message.DESCRIPTOR
    for f in message_descriptor.fields:
      default_float = f.GetOptions().Extensions[extensions_pb2.cl_default_float]
      if default_float:
        if f.name not in js:
          js[f.name] = default_float

    super(_CustomParser, self)._ConvertFieldValuePair(js, message, path)
