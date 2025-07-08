import os
from pathlib import Path

DEFAULT_UI = os.environ.get("CLARIFAI_UI", "https://clarifai.com")
DEFAULT_BASE = os.environ.get("CLARIFAI_API_BASE", "https://api.clarifai.com")

MCP_TRANSPORT_NAME = "mcp_transport"
OPENAI_TRANSPORT_NAME = "openai_transport"

CLARIFAI_PAT_ENV_VAR = "CLARIFAI_PAT"
CLARIFAI_SESSION_TOKEN_ENV_VAR = "CLARIFAI_SESSION_TOKEN"
CLARIFAI_USER_ID_ENV_VAR = "CLARIFAI_USER_ID"

HOME_PATH = Path.home()
DEFAULT_CONFIG = HOME_PATH / '.config/clarifai/config'

# Default clusters, etc. for local dev runner easy setup
DEFAULT_LOCAL_DEV_COMPUTE_CLUSTER_ID = "local-dev-compute-cluster"
DEFAULT_LOCAL_DEV_NODEPOOL_ID = "local-dev-nodepool"
DEFAULT_LOCAL_DEV_DEPLOYMENT_ID = "local-dev-deployment"
DEFAULT_LOCAL_DEV_MODEL_ID = "local-dev-model"
DEFAULT_LOCAL_DEV_APP_ID = "local-dev-runner-app"

# FIXME: should have any-to-any for these cases.
DEFAULT_LOCAL_DEV_MODEL_TYPE = "text-to-text"

DEFAULT_LOCAL_DEV_COMPUTE_CLUSTER_CONFIG = {
    "compute_cluster": {
        "id": DEFAULT_LOCAL_DEV_COMPUTE_CLUSTER_ID,
        "description": "Default Local Dev Compute Cluster",
        "cloud_provider": {
            "id": "local",
        },
        "region": "na",
        "managed_by": "user",
        "cluster_type": "local-dev",
    }
}

DEFAULT_LOCAL_DEV_NODEPOOL_CONFIG = {
    "nodepool": {
        "id": DEFAULT_LOCAL_DEV_NODEPOOL_ID,
        "description": "Default Local Dev Nodepool",
        "compute_cluster": {
            "id": DEFAULT_LOCAL_DEV_COMPUTE_CLUSTER_ID,
            "user_id": None,  # This will be set when creating the compute cluster
        },
        "instance_types": [
            {
                "id": "local",
                "compute_info": {
                    "cpu_limit": str(os.cpu_count()),
                    "cpu_memory": "16Gi",  # made up as we don't schedule based on this for local dev.
                    "num_accelerators": 0,  # TODO if we need accelerator detection for local dev.
                },
            }
        ],
        "node_capacity_type": {
            "capacity_types": [1],
        },
        "min_instances": 1,
        "max_instances": 1,
    }
}
