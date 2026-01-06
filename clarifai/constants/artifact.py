"""Constants for artifact management."""

# Default artifact visibility
DEFAULT_ARTIFACT_VISIBILITY = "private"

# Artifact visibility options
ARTIFACT_VISIBILITY_PRIVATE = "private"
ARTIFACT_VISIBILITY_PUBLIC = "public"
ARTIFACT_VISIBILITY_ORG = "org"

# Upload retry settings
DEFAULT_UPLOAD_MAX_RETRIES = 3
DEFAULT_DOWNLOAD_MAX_RETRIES = 3

# Default pagination settings for artifacts
DEFAULT_ARTIFACTS_PAGE_SIZE = 20

# File upload settings
UPLOAD_CHUNK_SIZE = 14 * 1024 * 1024  # 14MB chunks

# Progress bar settings
PROGRESS_BAR_UNIT = 'B'
PROGRESS_BAR_DESCRIPTION_UPLOAD = "Uploading"
PROGRESS_BAR_DESCRIPTION_DOWNLOAD = "Downloading"

# RFC3339 timestamp formats
RFC3339_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
RFC3339_FORMAT_NO_MICROSECONDS = "%Y-%m-%dT%H:%M:%SZ"
