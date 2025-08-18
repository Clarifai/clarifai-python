import os
import urllib.request
from typing import Any, Dict, Tuple

import grpc
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2_grpc

from clarifai import __version__
from clarifai.utils.constants import (
    CLARIFAI_PAT_ENV_VAR,
    CLARIFAI_SESSION_TOKEN_ENV_VAR,
    DEFAULT_BASE,
    DEFAULT_UI,
)

REQUEST_ID_PREFIX_HEADER = "x-clarifai-request-id-prefix"
REQUEST_ID_PREFIX = f"sdk-python-{__version__}"

# Map from base domain to True / False for whether the base has https or http.
# This is filled in get_stub() if it's not in there already.
base_https_cache = {}
ui_https_cache = {}


def clear_cache() -> None:
    """Clears the cache."""
    base_https_cache.clear()
    ui_https_cache.clear()


def https_cache(cache: dict, url: str) -> str:
    """This is a helper function to cache whether a url is http or https."""
    HTTPS = True
    HTTP = False

    # If http or https is provided, we trust that it is correct.
    # Note: this always stores the url without http:// or https://
    if url.startswith("https://"):
        url = url.replace("https://", "")
        cache[url] = HTTPS
    elif url.startswith("http://"):
        url = url.replace("http://", "")
        cache[url] = HTTP
    elif url.endswith(":443"):
        # If it ends with :443 then we know it's https.
        # trim it off the right
        url = url[:-4]
        cache[url] = HTTPS
    elif url.find('.clarifai.com') >= 0:
        # We know our endpoints are https.
        cache[url] = HTTPS
    elif url not in cache:
        # need to test ones that we don't have in the cache yet.
        try:  # make request to https endpoint.
            urllib.request.urlopen("https://%s/v2/auth/methods" % url, timeout=5)
            cache[url] = HTTPS  # cache it.
        except Exception as e:
            if "SSL" in str(e):  # if ssl error then we know it's http.
                cache[url] = HTTP
                # For http urls we need host:port format.
                if ":" not in url:
                    raise Exception(
                        "When providing an insecure url it must have both host:port format"
                    )
            else:
                raise Exception(
                    "Could not get a valid response from url: %s, is the API running there?" % url
                ) from e
    return url


class ClarifaiAuthHelper:
    def __init__(
        self,
        user_id: str = "",
        app_id: str = "",
        pat: str = "",
        token: str = "",
        base: str = DEFAULT_BASE,
        ui: str = DEFAULT_UI,
        root_certificates_path: str = None,
        validate: bool = True,
    ):
        """
            A helper to get the authorization information needed to make API calls with the grpc
            client to a specific app using a personal access token.

            There are classmethods to make this object easily from either query_params provided by streamlit or from env vars.

            Note: only one of personal access token (pat) or a session token (token) can be provided.
            Always use PATs in your code and never session tokens, those are only provided internal UI code.

        Args:
          user_id: a user id who owns the resource you want to make calls to.
          app_id: an app id for the application that owns the resource you want to interact with
          pat: a personal access token.
          token: a session token (internal use only, always use a PAT).
          base: a url to the API endpoint to hit. Examples include api.clarifai.com,
        https://api.clarifai.com (default), https://host:port, http://host:port, host:port (will be treated as http, not https). It's highly recommended to include the http:// or https:// otherwise we need to check the endpoint to determine if it has SSL during this __init__
          ui: a url to the UI. Examples include clarifai.com,
        https://clarifai.com (default), https://host:port, http://host:port, host:port (will be treated as http, not https). It's highly recommended to include the http:// or https:// otherwise we need to check the endpoint to determine if it has SSL during this __init__
          root_certificates_path: path to the root certificates file. This is only used for grpc secure channels.
          validate: whether to validate the inputs. This is useful for overriding vars then validating
        """

        self.user_id = user_id
        self.app_id = app_id
        self._pat = pat
        self._token = token
        self._root_certificates_path = root_certificates_path

        self.set_base(base)
        self.set_ui(ui)
        if validate:
            self.validate()

    def validate(self):
        if self.user_id == "":
            raise Exception(
                "Need 'user_id' to not be empty in the query params or user CLARIFAI_USER_ID env var"
            )
        if self._pat != "" and self._token != "":
            raise Exception(
                "A personal access token OR a session token need to be provided, but you cannot provide both."
            )
        elif self._pat == "" and self._token == "":
            raise Exception(
                "Need 'pat' or 'token' in the query params or use one of the CLARIFAI_PAT or CLARIFAI_SESSION_TOKEN env vars"
            )
        if (self._root_certificates_path) and (not os.path.exists(self._root_certificates_path)):
            raise Exception(
                "Root certificates path %s does not exist" % self._root_certificates_path
            )

    @classmethod
    def from_streamlit(cls, st: Any) -> "ClarifaiAuthHelper":
        """This is a convenient method to check the environment variables first to see if there are
        required variables for auth, then override them with any additional query parameters that may
        have been passed in.

        Note: if a .streamlit/secrets.toml is present then st.secrets will auto populate the
        corresponding environment variables and we will pick them up from there, OVERWRITING whatever
        matching env var that may already be present.

            Args:
              st: the streamlit package typically as: 'import streamlit as st'
            Returns:
              auth: this class instantiated
        """
        # start with the env vars (potentially loaded from secrets.toml)
        # Don't validate yet as we'll layer on the query params next.
        auth = ClarifaiAuthHelper.from_env(validate=False)

        # Then add in the query params.
        try:
            if st.query_params:
                auth.add_streamlit_query_params(st.query_params)
            else:
                auth.add_streamlit_query_params(st.session_state)
        except Exception as e:
            st.error(e)
            st.stop()
            raise e

        # Then validate.
        try:
            auth.validate()
        except Exception as e:
            st.error(e)
            st.stop()
            raise e

        return auth

    @classmethod
    def from_streamlit_query_params(cls, query_params: Any = "") -> "ClarifaiAuthHelper":
        """Initialize from streamlit queryparams. The following things will be looked for:
        user_id: as 'user_id' in query_params
        app_id: as 'app_id' in query_params
        one of:
          token: as 'token' in query_params
          pat: as 'pat' in query_params
        optionally:
          base: as 'base' in query_params.
          ui: as 'ui' in query_params.

        """

        # Setup an empty one (not from env).
        auth = ClarifaiAuthHelper("", "", "", "", validate=False)

        # Then add in the query params.
        auth.add_streamlit_query_params(query_params)

        # Then validate.
        auth.validate()

        return auth

    def add_streamlit_query_params(self, query_params: Any = "") -> "ClarifaiAuthHelper":
        """Initialize from streamlit queryparams. The following things will be looked for:
              user_id: as 'user_id' in query_params
              app_id: as 'app_id' in query_params
              one of:
                token: as 'token' in query_params
                pat: as 'pat' in query_params
              optionally:
                base: as 'base' in query_params.

        Args:
          query_params: the streamlit.query_params response or streamlit.session_state.
        """

        if query_params == "":  # empty response from streamlit
            query_params = {}
        if "user_id" in query_params:
            self.user_id = query_params["user_id"]
        if "app_id" in query_params:
            self.app_id = query_params["app_id"]
        if "token" in query_params:
            self._token = query_params["token"]
        if "pat" in query_params:
            self._pat = query_params["pat"]
        if "base" in query_params:
            self.set_base(query_params["base"])
        if "ui" in query_params:
            self.set_ui(query_params["ui"])
        if "root_certificates_path" in query_params:
            self._root_certificates_path = query_params["root_certificates_path"]

    @classmethod
    def from_env(cls, validate: bool = True) -> "ClarifaiAuthHelper":
        """Will look for the following env vars:
        user_id: CLARIFAI_USER_ID env var.
        app_id: CLARIFAI_APP_ID env var.
        one of:
          token: CLARIFAI_SESSION_TOKEN env var.
          pat: CLARIFAI_PAT env var.
        base: CLARIFAI_API_BASE env var.
        root_certificates_path: CLARIFAI_ROOT_CERTIFICATES_PATH env var.
        """
        user_id = os.environ.get("CLARIFAI_USER_ID", "")
        app_id = os.environ.get("CLARIFAI_APP_ID", "")
        token = os.environ.get(CLARIFAI_SESSION_TOKEN_ENV_VAR, "")
        pat = os.environ.get(CLARIFAI_PAT_ENV_VAR, "")
        base = os.environ.get("CLARIFAI_API_BASE", DEFAULT_BASE)
        ui = os.environ.get("CLARIFAI_UI", DEFAULT_UI)
        root_certificates_path = os.environ.get("CLARIFAI_ROOT_CERTIFICATES_PATH", None)
        return cls(user_id, app_id, pat, token, base, ui, root_certificates_path, validate)

    def get_user_app_id_proto(
        self,
        user_id: str = None,
        app_id: str = None,
    ) -> resources_pb2.UserAppIDSet:
        """Get the gRPC metadata that contains either the session token or the PAT to use.

        Args:
          user_id: optional user_id to override the default
          app_id: optional app_id to override the default.

        Returns:
          metadata: the metadata need to send with all grpc API calls in the API client.
        """
        user_id = self.user_id if user_id is None else user_id
        app_id = self.app_id if app_id is None else app_id
        return resources_pb2.UserAppIDSet(user_id=user_id, app_id=app_id)

    @property
    def metadata(self):
        """Get the gRPC metadata that contains either the session token or the PAT to use.

        Returns:
          metadata: the metadata need to send with all grpc API calls in the API client.
        """
        if self._pat != "":
            return (
                ("authorization", "Key %s" % self._pat),
                (REQUEST_ID_PREFIX_HEADER, REQUEST_ID_PREFIX),
            )
        elif self._token != "":
            return (
                ("x-clarifai-session-token", self._token),
                (REQUEST_ID_PREFIX_HEADER, REQUEST_ID_PREFIX),
            )
        else:
            raise Exception(
                "'token' or 'pat' needed to be provided in the query params or env vars."
            )

    def get_stub(self) -> service_pb2_grpc.V2Stub:
        stub, channel = self.get_stub_and_channel()
        return stub

    def get_stub_and_channel(self) -> Tuple[service_pb2_grpc.V2Stub, grpc.Channel]:
        """Get the API gRPC stub and channel based on the API endpoint base.

        Returns:
          stub: The service_pb2_grpc.V2Stub stub for the API.
        """
        if self._base not in base_https_cache:
            raise Exception("Cannot determine if base %s is https" % self._base)

        https = base_https_cache[self._base]
        if https:
            channel = ClarifaiChannel.get_grpc_channel(
                base=self._base, root_certificates_path=self._root_certificates_path
            )
        else:
            if self._base.find(":") >= 0:
                host, port = self._base.split(":")
            else:
                host = self._base
                port = 80
            channel = ClarifaiChannel.get_insecure_grpc_channel(base=host, port=port)
        stub = service_pb2_grpc.V2Stub(channel)
        return stub, channel

    def get_async_stub(self) -> service_pb2_grpc.V2Stub:
        """Get the API gRPC async stub using the right channel based on the API endpoint base.
        Returns:
        stub: The service_pb2_grpc.V2Stub stub for the API.
        """
        if self._base not in base_https_cache:
            raise Exception("Cannot determine if base %s is https" % self._base)

        https = base_https_cache[self._base]
        if https:
            channel = ClarifaiChannel.get_aio_grpc_channel(
                base=self._base, root_certificates_path=self._root_certificates_path
            )
        else:
            if self._base.find(":") >= 0:
                host, port = self._base.split(":")
            else:
                host = self._base
                port = 80
            channel = ClarifaiChannel.get_aio_insecure_grpc_channel(base=host, port=port)
        stub = service_pb2_grpc.V2Stub(channel)
        return stub

    @property
    def ui(self) -> str:
        """Return the domain for the UI."""
        if self._ui not in ui_https_cache:
            raise Exception("Cannot determine if ui %s is https" % self._ui)
        https = ui_https_cache[self._ui]
        if https:
            if not self._ui.startswith("https://"):
                return "https://" + self._ui
            return self._ui
        if not self._ui.startswith("http://"):
            return "http://" + self._ui
        return self._ui

    def set_base(self, base: str):
        """Set the base domain for the API."""
        base = DEFAULT_BASE if base is None else base
        self._base = https_cache(base_https_cache, base)

    def set_ui(self, ui: str):
        """Set the domain for the UI."""
        self._ui = https_cache(ui_https_cache, ui)

    @property
    def base(self) -> str:
        """Return the base domain for the API."""
        if self._base not in base_https_cache:
            raise Exception("Cannot determine if base %s is https" % self._base)
        https = base_https_cache[self._base]
        if https:
            if not self._base.startswith("https://"):
                return "https://" + self._base
            return self._base
        if not self._base.startswith("http://"):
            return "http://" + self._base
        return self._base

    @property
    def pat(self) -> str:
        """Return the personal access token."""
        return self._pat

    def __str__(self):
        return "ClarifaiAuthHelper:\n- base: %s\n- user_id: %s\n- app_id: %s\n" % (
            self._base,
            self.user_id,
            self.app_id,
        )

    @classmethod
    def required_env_vars(cls):
        """Return the list of the required environment variables."""
        return ["CLARIFAI_USER_ID", "CLARIFAI_APP_ID", "CLARIFAI_PAT"]

    @classmethod
    def validate_secrets_dict(cls, toml_dict: Dict[str, Any]):
        """Validate the secrets.toml file has been filled with non-empty values for all the auth
        parameters that are present.

        We don't load the file here so that we don't need the tomli package dependency. You can simply
        do:
        import tomli

        d = tomli.load(open("secrets.toml"))
        ClarifaiAuthHelper.validate_secrets_dict(d)

        """
        # We don't validate the bases because they have sensible defaults
        auth_keys = cls.required_env_vars()

        for k, v in toml_dict.items():
            if k in auth_keys:
                if v == "":
                    raise Exception("'%s' in secrets.toml cannot be empty" % k)
        # for all the keys that are not present, they have a non empty value.
        return True
