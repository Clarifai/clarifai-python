from clarifai.client.base import BaseClient
from clarifai.client.lister import Lister
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai.utils.constants import DEFAULT_BASE


class PipelineStep(Lister, BaseClient):
    """PipelineStep is a class that provides access to Clarifai API endpoints related to PipelineStep information."""

    def __init__(
        self,
        url: str = None,
        pipeline_step_id: str = None,
        pipeline_step_version_id: str = None,
        user_id: str = None,
        app_id: str = None,
        pipeline_id: str = None,
        base_url: str = DEFAULT_BASE,
        pat: str = None,
        token: str = None,
        root_certificates_path: str = None,
        **kwargs,
    ):
        """Initializes a PipelineStep object.

        Args:
            url (str): The URL to initialize the pipeline step object.
            pipeline_step_id (str): The PipelineStep ID for the PipelineStep to interact with.
            pipeline_step_version_id (str): The PipelineStep version ID for the PipelineStep to interact with.
            user_id (str): The User ID for the PipelineStep to interact with.
            app_id (str): The App ID for the PipelineStep to interact with.
            pipeline_id (str): The Pipeline ID for the PipelineStep to interact with.
            base_url (str): Base API url. Default "https://api.clarifai.com"
            pat (str): A personal access token for authentication.
            token (str): A session token for authentication.
            root_certificates_path (str): Path to the SSL root certificates file.
            **kwargs: Additional keyword arguments to be passed to the BaseClient.
        """
        if url:
            user_id, app_id, _, pipeline_step_id, pipeline_step_version_id = (
                ClarifaiUrlHelper.split_clarifai_url(url)
            )

        # Store all kwargs as attributes for API data
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.kwargs = {
            "pipeline_step_id": pipeline_step_id,
            "pipeline_step_version_id": pipeline_step_version_id,
            "user_id": user_id,
            "app_id": app_id,
            "pipeline_id": pipeline_id,
            **kwargs,
        }

        BaseClient.__init__(
            self,
            user_id=user_id,
            app_id=app_id,
            base=base_url,
            pat=pat,
            token=token,
            root_certificates_path=root_certificates_path,
        )
        Lister.__init__(self)

        self.pipeline_step_id = pipeline_step_id
        self.pipeline_step_version_id = pipeline_step_version_id
        self.pipeline_id = pipeline_id
        self.user_id = user_id
        self.app_id = app_id
