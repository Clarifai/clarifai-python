from datetime import datetime
from typing import Any, Callable

from clarifai_grpc.grpc.api import resources_pb2
from google.protobuf import struct_pb2
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.wrappers_pb2 import BoolValue

from clarifai.client.auth import create_stub
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.constants.base import COMPUTE_ORCHESTRATION_RESOURCES
from clarifai.errors import ApiError, UserError
from clarifai.utils.constants import CLARIFAI_PAT_ENV_VAR, CLARIFAI_SESSION_TOKEN_ENV_VAR
from clarifai.utils.misc import get_from_dict_env_or_config


class BaseClient:
    """BaseClient is the base class for all the classes interacting with Clarifai endpoints.

    Args:
        **kwargs: Additional keyword arguments to be passed to the ClarifaiAuthHelper.
            - user_id (str): A user ID for authentication.
            - app_id (str): An app ID for the application to interact with.
            - pat (str): A personal access token for authentication.
            - token (str): A session token for authentication. Accepts either a session token or a pat.
            - base (str): The base URL for the API endpoint. Defaults to 'https://api.clarifai.com'.
            - ui (str): The URL for the UI. Defaults to 'https://clarifai.com'.
            - root_certificates_path (str): Path to the SSL root certificates file, used to establish secure gRPC connections.


    Attributes:
        auth_helper (ClarifaiAuthHelper): An instance of ClarifaiAuthHelper for authentication.
        STUB (Stub): The gRPC Stub object for API interaction.
        metadata (tuple): The gRPC metadata containing the personal access token.
        userDataObject (UserAppIDSet): The protobuf object representing user and app IDs.
        base (str): The base URL for the API endpoint.
    """

    def __init__(self, **kwargs):
        token, pat = "", ""
        try:
            pat = get_from_dict_env_or_config(key="pat", env_key=CLARIFAI_PAT_ENV_VAR, **kwargs)
        except UserError:
            try:
                token = get_from_dict_env_or_config(
                    key="token", env_key=CLARIFAI_SESSION_TOKEN_ENV_VAR, **kwargs
                )
            except UserError:
                pass
        finally:
            if not (token or pat):
                raise UserError(
                    "Authentication Required. Please authenticate in one of the following ways:\n\n"
                    "- Pass your Personal Access Token ('pat') or session token ('token') as arguments to your function.\n"
                    "- Set the CLARIFAI_PAT or CLARIFAI_SESSION_TOKEN environment variables in your environment.\n"
                    "- Run `clarifai login` in your terminal to configure CLI authentication."
                )

        # Also try to get user_id and base from CLI config if not provided
        if not kwargs.get('user_id'):
            try:
                user_id = get_from_dict_env_or_config(
                    key="user_id", env_key="CLARIFAI_USER_ID", **kwargs
                )
                kwargs['user_id'] = user_id
            except UserError:
                pass  # user_id is optional for some use cases

        if not kwargs.get('base'):
            try:
                base = get_from_dict_env_or_config(
                    key="base", env_key="CLARIFAI_API_BASE", **kwargs
                )
                kwargs['base'] = base
            except UserError:
                pass  # base has a default value

        kwargs.update({'token': token, 'pat': pat})

        self.auth_helper = ClarifaiAuthHelper(**kwargs, validate=False)
        self.STUB = create_stub(self.auth_helper)
        self._async_stub = None
        self.metadata = self.auth_helper.metadata
        self.pat = self.auth_helper.pat
        self.token = self.auth_helper._token
        self.user_app_id = self.auth_helper.get_user_app_id_proto()
        self.base = self.auth_helper.base
        self.root_certificates_path = self.auth_helper._root_certificates_path

    @property
    def async_stub(self):
        """Returns the asynchronous gRPC stub for the API interaction.
        Lazy initialization of async stub"""
        if self._async_stub is None:
            self._async_stub = create_stub(self.auth_helper, is_async=True)
        return self._async_stub

    @classmethod
    def from_env(cls, validate: bool = False):
        auth = ClarifaiAuthHelper.from_env(validate=validate)
        return cls.from_auth_helper(auth)

    @classmethod
    def from_auth_helper(cls, auth: ClarifaiAuthHelper, **kwargs):
        default_kwargs = {
            "user_id": kwargs.get("user_id", None) or auth.user_id,
            "app_id": kwargs.get("app_id", None) or auth.app_id,
            "pat": kwargs.get("pat", None) or auth.pat,
            "token": kwargs.get("token", None) or auth._token,
            "root_certificates_path": kwargs.get("root_certificates_path", None)
            or auth._root_certificates_path,
        }
        _base = kwargs.get("base", None) or auth.base
        _clss = cls.__mro__[0]
        if _clss == BaseClient:
            kwargs = {
                **default_kwargs,
                "base": _base,  # Baseclient uses `base`
                "ui": kwargs.get("ui", None) or auth.ui,
            }
        else:
            # Remove user_id and app_id if a custom URL is provided
            if kwargs.get("url"):
                default_kwargs.pop("user_id", "")
                default_kwargs.pop("app_id", "")
            # Remove app_id if the class name is a compute orchestration resource
            if any(
                co_resource in _clss.__name__ for co_resource in COMPUTE_ORCHESTRATION_RESOURCES
            ):
                default_kwargs.pop("app_id", "")
            kwargs.update({**default_kwargs, "base_url": _base})

        return cls(**kwargs)

    def _grpc_request(self, method: Callable, argument: Any):
        """Makes a gRPC request to the API.

        Args:
            method (Callable): The gRPC method to call.
            argument (Any): The argument to pass to the gRPC method.

        Returns:
            res (Any): The result of the gRPC method call.
        """

        try:
            res = method(argument, metadata=self.auth_helper.metadata)
            # MessageToDict(res) TODO global debug logger
            return res
        except ApiError:
            raise Exception("ApiError")

    def convert_string_to_timestamp(self, date_str) -> Timestamp:
        """Converts a string to a Timestamp object.

        Args:
            date_str (str): The string to convert.

        Returns:
            Timestamp: The converted Timestamp object.
        """
        # Parse the string into a Python datetime object
        try:
            datetime_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError:
            try:
                datetime_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
            except ValueError:
                return Timestamp()

        # Convert the datetime object to a Timestamp object
        timestamp_obj = Timestamp()
        timestamp_obj.FromDatetime(datetime_obj)

        return timestamp_obj

    def process_response_keys(self, old_dict, listing_resource=None):
        """Converts keys in a response dictionary to resource proto format.

        Args:
            old_dict (dict): The dictionary to convert.
            listing_resource (str, optional): The resource type for which the keys are being processed.

        Returns:
            new_dict (dict): The dictionary with processed keys.
        """
        if listing_resource:
            old_dict[f'{listing_resource}_id'] = old_dict['id']
            old_dict.pop('id')

        def convert_recursive(item):
            if isinstance(item, dict):
                new_item = {}
                for key, value in item.items():
                    if key == 'default_value':
                        # Map infer param value to proto value
                        value_map = dict(number_value=None, string_value=None, bool_value=None)

                        def map_fn(v):
                            return (
                                'number_value'
                                if isinstance(v, float) or isinstance(v, int)
                                else 'string_value'
                                if isinstance(v, str)
                                else 'bool_value'
                                if isinstance(v, bool)
                                else None
                            )

                        value_map[map_fn(value)] = value
                        value = struct_pb2.Value(**value_map)
                    if key in ['created_at', 'modified_at', 'completed_at']:
                        value = self.convert_string_to_timestamp(value)
                    elif key in ['workflow_recommended', 'is_template']:
                        value = BoolValue(value=True)
                    elif key in ['fields_map', 'params']:
                        value_s = struct_pb2.Struct()
                        value_s.update(value)
                        value = value_s
                    elif key == 'metrics':
                        continue
                    elif key == 'size':
                        value = int(value)
                    elif key == 'image_info':
                        value = resources_pb2.ImageInfo(**value)
                    elif key == 'hosted_image_info':
                        continue
                    elif key in ['metadata', 'presets']:
                        if isinstance(value, dict) and value != {}:
                            value_s = struct_pb2.Struct()
                            value_s.update(value)
                            value = value_s
                        else:
                            continue
                    new_item[key] = convert_recursive(value)
                return new_item
            elif isinstance(item, list):
                return [convert_recursive(element) for element in item]
            else:
                return item

        new_dict = convert_recursive(old_dict)
        return new_dict
