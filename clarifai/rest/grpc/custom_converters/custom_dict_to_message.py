from google.protobuf.json_format import _Parser

from clarifai.rest.grpc.proto.clarifai.api.utils import extensions_pb2


def dict_to_protobuf(protobuf_class, js_dict):
  message = protobuf_class()
  parser = _CustomParser(ignore_unknown_fields=False)
  parser.ConvertMessage(js_dict, message)
  return message


class _CustomParser(_Parser):

  def _ConvertFieldValuePair(self, js, message):
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

    super(_CustomParser, self)._ConvertFieldValuePair(js, message)
