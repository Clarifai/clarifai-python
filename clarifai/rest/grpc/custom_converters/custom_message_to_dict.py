from google.protobuf import descriptor
from google.protobuf.json_format import _IsMapEntry, _Printer

from clarifai.rest.grpc.proto.clarifai.api.utils import extensions_pb2


def protobuf_to_dict(object_protobuf):
  printer = _CustomPrinter(
      including_default_value_fields=False,
      preserving_proto_field_name=True,
      use_integers_for_enums=True)
  # pylint: disable=protected-access
  return printer._MessageToJsonObject(object_protobuf)


class _CustomPrinter(_Printer):

  def _RegularMessageToJsonObject(self, message, js):
    """
    Because of the fields with the custom extension `cl_show_if_empty`, we need to adjust the
    original's method's return JSON object and keep these fields.
    """

    js = super(_CustomPrinter, self)._RegularMessageToJsonObject(message, js)

    message_descriptor = message.DESCRIPTOR
    for field in message_descriptor.fields:

      if not field.GetOptions().Extensions[extensions_pb2.cl_show_if_empty]:
        continue

      # Singular message fields and oneof fields will not be affected.
      if ((field.label != descriptor.FieldDescriptor.LABEL_REPEATED and
           field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE) or
          field.containing_oneof):
        continue
      if self.preserving_proto_field_name:
        name = field.name
      else:
        name = field.json_name
      if name in js:
        # Skip the field which has been serailized already.
        continue
      if _IsMapEntry(field):
        js[name] = {}
      elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
        js[name] = []
      else:
        js[name] = self._FieldToJsonObject(field, field.default_value)

    return js
