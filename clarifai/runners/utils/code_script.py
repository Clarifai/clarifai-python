import json
from typing import List

from clarifai_grpc.grpc.api import resources_pb2

from clarifai.runners.utils import data_utils
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.utils.constants import MCP_TRANSPORT_NAME, OPENAI_TRANSPORT_NAME


def has_signature_method(
    name: str, method_signatures: List[resources_pb2.MethodSignature]
) -> bool:
    """
    Check if a method signature with the given name exists in the list of method signatures.

    :param name: The name of the method to check.
    :param method_signatures: List of MethodSignature objects to search in.
    :return: True if a method with the given name exists, False otherwise.
    """
    return any(
        method_signature.name == name for method_signature in method_signatures if method_signature
    )


def generate_client_script(
    method_signatures: List[resources_pb2.MethodSignature],
    user_id,
    app_id,
    model_id,
    base_url: str = None,
    deployment_id: str = None,
    compute_cluster_id: str = None,
    nodepool_id: str = None,
    use_ctx: bool = False,
) -> str:
    url_helper = ClarifaiUrlHelper()

    # Provide an mcp client config if there is a method named "mcp_transport"
    if has_signature_method(MCP_TRANSPORT_NAME, method_signatures):
        mcp_url = url_helper.mcp_api_url(
            user_id,
            app_id,
            model_id,
        )

        _CLIENT_TEMPLATE = f"""
import asyncio
import os

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

transport = StreamableHttpTransport(
    url="{mcp_url}",
    headers={{"Authorization": "Bearer " + os.environ["CLARIFAI_PAT"]}},
)

async def main():
    async with Client(transport) as client:
        tools = await client.list_tools()
        print(f"Available tools: {{tools}}")
        # TODO: update the dictionary of arguments passed to call_tool to make sense for your MCP.
        result = await client.call_tool(tools[0].name, {{"a": 5, "b": 3}})
        print(f"Result: {{result[0].text}}")

if __name__ == "__main__":
    asyncio.run(main())
"""
        return _CLIENT_TEMPLATE

    if has_signature_method(OPENAI_TRANSPORT_NAME, method_signatures):
        openai_api_base = url_helper.openai_api_url()
        model_ui_url = url_helper.clarifai_url(user_id, app_id, "models", model_id)
        _CLIENT_TEMPLATE = f"""
import os

from openai import OpenAI

client = OpenAI(
    base_url="{openai_api_base}",
    api_key=os.environ['CLARIFAI_PAT'],
)

response = client.chat.completions.create(
    model="{model_ui_url}",
    messages=[
        {{"role": "system", "content": "Talk like a pirate."}},
        {{
            "role": "user",
            "content": "How do I check if a Python object is an instance of a class?",
        }},
    ],
    temperature=0.7,
    stream=False,  # stream=True also works, just iterator over the response
)
print(response)
"""
        return _CLIENT_TEMPLATE
    # Generate client template
    _CLIENT_TEMPLATE = (
        "import os\n\n"
        "from clarifai.client import Model\n"
        "from clarifai.runners.utils import data_types\n\n"
        "{model_section}\n"
    )
    if deployment_id and (compute_cluster_id or nodepool_id):
        raise ValueError(
            "You can only specify one of deployment_id or compute_cluster_id and nodepool_id."
        )
    if compute_cluster_id and nodepool_id:
        deployment_id = None
    else:
        deployment_id = (
            'os.environ.get("CLARIFAI_DEPLOYMENT_ID", None)'
            if not deployment_id
            else repr(deployment_id)
        )

    deployment_line = (
        f'deployment_id={deployment_id},  # Only needed for dedicated deployed models'
        if deployment_id
        else ""
    )
    compute_cluster_line = (
        f'compute_cluster_id="{compute_cluster_id}",' if compute_cluster_id else ""
    )
    nodepool_line = (
        f'nodepool_id="{nodepool_id}",  # Only needed for dedicated nodepool'
        if nodepool_id
        else ""
    )

    base_url_str = ""
    if base_url is not None:
        base_url_str = f'base_url="{base_url}",'

    # Join all non-empty lines
    optional_lines = "\n    ".join(
        line
        for line in [deployment_line, compute_cluster_line, nodepool_line, base_url_str]
        if line
    )

    if use_ctx:
        model_section = """
model = Model.from_current_context()
"""
    else:
        model_ui_url = url_helper.clarifai_url(user_id, app_id, "models", model_id)
        if optional_lines:
            model_args = f'"{model_ui_url}",\n    {optional_lines}'
        else:
            model_args = f'"{model_ui_url}"'
        model_section = f"model = Model(\n    {model_args}\n)"

    client_template = _CLIENT_TEMPLATE.format(model_section=model_section.strip("\n"))

    # Generate method signatures
    method_signatures_str = []
    for method_signature in method_signatures:
        if method_signature is None:
            continue
        method_name = method_signature.name
        client_script_str = f"response = model.{method_name}("
        annotations = _get_annotations_source(method_signature)
        param_lines = []
        for idx, (param_name, (param_type, default_value, required)) in enumerate(
            annotations.items()
        ):
            if param_name == "return":
                continue
            if default_value is None and required:
                default_value = _set_default_value(param_type)
            if not default_value and idx == 0:
                default_value = _set_default_value(param_type)
            if param_type == "str" and default_value is not None:
                default_value = json.dumps(default_value)
            if default_value is not None:
                param_lines.append(f"    {param_name}={default_value},")
        if param_lines:
            client_script_str += "\n" + "\n".join(param_lines) + "\n)"
        else:
            client_script_str += ")"
        if method_signature.method_type == resources_pb2.RunnerMethodType.UNARY_UNARY:
            client_script_str += "\nprint(response)"
        elif method_signature.method_type == resources_pb2.RunnerMethodType.UNARY_STREAMING:
            client_script_str += "\nfor res in response:\n    print(res)"
        client_script_str += "\n"
        method_signatures_str.append(client_script_str)

    method_signatures_str = "\n".join(method_signatures_str)
    # Combine all parts
    script_lines = []
    script_lines.append("\n# Clarifai Model Client Script")
    if not use_ctx:
        script_lines.append(
            "# Set the environment variables `CLARIFAI_DEPLOYMENT_ID` and `CLARIFAI_PAT` to run this script."
        )
    script_lines.append("# Example usage:")
    script_lines.append(client_template)
    script_lines.append("# Example model prediction from different model methods:\n")
    script_lines.append(method_signatures_str)
    script_lines.append("")
    script = "\n".join(script_lines)
    return script


# get annotations source with default values
def _get_annotations_source(method_signature: resources_pb2.MethodSignature) -> dict:
    annotations = {}
    for input_field in method_signature.input_fields:
        param_name = input_field.name
        param_type = _get_base_type(input_field)
        if input_field.iterator:
            param_type = f"Iterator[{param_type}]"
        default_value = None
        if data_utils.Param.get_default(input_field):
            default_value = _parse_default_value(input_field)
        annotations[param_name] = (param_type, default_value, input_field.required)
    if not method_signature.output_fields:
        raise ValueError("MethodSignature must have at least one output field")
    for output_field in method_signature.output_fields:
        param_name = output_field.name
        param_type = _get_base_type(output_field)
        if output_field.iterator:
            param_type = f"Iterator[{param_type}]"
        annotations[param_name] = (param_type, None, output_field.required)
    return annotations


def _get_base_type(field: resources_pb2.ModelTypeField) -> str:
    data_type = field.type
    if data_type == resources_pb2.ModelTypeField.DataType.NAMED_FIELDS:
        annotations = {}
        for type_arg in field.type_args:
            field_name = type_arg.name
            field_type = _get_base_type(type_arg)
            annotations[field_name] = field_type
        class_name = f"NamedFields[{', '.join(f'{k}: {v}' for k, v in annotations.items())}]"
        return class_name
    elif data_type == resources_pb2.ModelTypeField.DataType.TUPLE:
        type_args_str = [_get_base_type(ta) for ta in field.type_args]
        return f"Tuple[{', '.join(type_args_str)}]"
    elif data_type == resources_pb2.ModelTypeField.DataType.LIST:
        if len(field.type_args) != 1:
            raise ValueError("List type must have exactly one type argument")
        element_type = _get_base_type(field.type_args[0])
        return f"List[{element_type}]"
    else:
        type_map = {
            resources_pb2.ModelTypeField.DataType.STR: "str",
            resources_pb2.ModelTypeField.DataType.BYTES: "bytes",
            resources_pb2.ModelTypeField.DataType.INT: "int",
            resources_pb2.ModelTypeField.DataType.FLOAT: "float",
            resources_pb2.ModelTypeField.DataType.BOOL: "bool",
            resources_pb2.ModelTypeField.DataType.NDARRAY: "np.ndarray",
            resources_pb2.ModelTypeField.DataType.JSON_DATA: "data_types.JSON",
            resources_pb2.ModelTypeField.DataType.TEXT: "data_types.Text",
            resources_pb2.ModelTypeField.DataType.IMAGE: "data_types.Image",
            resources_pb2.ModelTypeField.DataType.CONCEPT: "data_types.Concept",
            resources_pb2.ModelTypeField.DataType.REGION: "data_types.Region",
            resources_pb2.ModelTypeField.DataType.FRAME: "data_types.Frame",
            resources_pb2.ModelTypeField.DataType.AUDIO: "data_types.Audio",
            resources_pb2.ModelTypeField.DataType.VIDEO: "data_types.Video",
        }
        return type_map.get(data_type, "Any")


def _map_default_value(field_type):
    """
    Map the default value of a field type to a string representation.
    """
    default_value = None

    if field_type == "str":
        default_value = 'What is the future of AI?'
    elif field_type == "bytes":
        default_value = b""
    elif field_type == "int":
        default_value = 0
    elif field_type == "float":
        default_value = 0.0
    elif field_type == "bool":
        default_value = False
    elif field_type == "data_types.Image":
        default_value = 'data_types.Image.from_url("https://samples.clarifai.com/metro-north.jpg")'
    elif field_type == "data_types.Text":
        default_value = 'data_types.Text("What is the future of AI?")'
    elif field_type == "data_types.Audio":
        default_value = 'data_types.Audio.from_url("https://samples.clarifai.com/audio.mp3")'
    elif field_type == "data_types.Video":
        default_value = 'data_types.Video.from_url("https://samples.clarifai.com/video.mp4")'
    elif field_type == "data_types.Concept":
        default_value = 'data_types.Concept(id="concept_id", name="dog", value=0.95)'
    elif field_type == "data_types.Region":
        default_value = 'data_types.Region(box=[0.1, 0.1, 0.5, 0.5],)'
    elif field_type == "data_types.Frame":
        default_value = 'data_types.Frame.from_url("https://samples.clarifai.com/video.mp4", 0)'
    elif field_type == "data_types.NDArray":
        default_value = 'data_types.NDArray([1, 2, 3])'
    else:
        default_value = None
    return default_value


def _set_default_value(field_type):
    """
    Set the default value of a field if it is not set.
    """
    is_iterator = False
    if field_type.startswith("Iterator["):
        is_iterator = True
        field_type = field_type[9:-1]
    default_value = None
    default_value = _map_default_value(field_type)
    if field_type.startswith("List["):
        element_type = field_type[5:-1]
        element_type_default_value = _map_default_value(element_type)
        if element_type_default_value is not None:
            default_value = f"[{element_type_default_value}]"
    elif field_type.startswith("Tuple["):
        element_types = field_type[6:-1].split(", ")
        element_type_defaults = [_map_default_value(et) for et in element_types]
        default_value = f"({', '.join([str(et) for et in element_type_defaults])})"
    elif field_type.startswith("NamedFields["):
        element_types = field_type[12:-1].split(", ")
        element_type_defaults = [_map_default_value(et) for et in element_types]
        default_value = f"{{{', '.join([str(et) for et in element_type_defaults])}}}"

    if is_iterator:
        if field_type == "str":
            default_value = f"iter(['{default_value}'])"
        else:
            default_value = f"iter([{default_value}])"
    return default_value


def _parse_default_value(field: resources_pb2.ModelTypeField):
    if not field.default:
        return None
    default_str = field.default
    data_type = field.type

    try:
        if data_type == resources_pb2.ModelTypeField.DataType.INT:
            return str(int(default_str))
        elif data_type == resources_pb2.ModelTypeField.DataType.FLOAT:
            return str(float(default_str))
        elif data_type == resources_pb2.ModelTypeField.DataType.BOOL:
            return 'True' if default_str.lower() == 'true' else 'False'
        elif data_type == resources_pb2.ModelTypeField.DataType.STR:
            return default_str
        elif data_type == resources_pb2.ModelTypeField.DataType.BYTES:
            return f"b{repr(default_str.encode('utf-8'))}"
        elif data_type == resources_pb2.ModelTypeField.DataType.JSON_DATA:
            parsed = json.loads(default_str)
            return repr(parsed)
        elif data_type in (
            resources_pb2.ModelTypeField.DataType.LIST,
            resources_pb2.ModelTypeField.DataType.TUPLE,
            resources_pb2.ModelTypeField.DataType.NAMED_FIELDS,
        ):
            parsed = json.loads(default_str)
            return repr(parsed)
        else:
            return repr(default_str)
    except (ValueError, json.JSONDecodeError):
        return repr(default_str)
