"""Constants for artifact management."""

# Default artifact visibility
DEFAULT_ARTIFACT_VISIBILITY = "private"

# Artifact visibility options
ARTIFACT_VISIBILITY_PRIVATE = "private"
ARTIFACT_VISIBILITY_PUBLIC = "public"

# Upload status values
UPLOAD_STATUS_SUCCESS = "SUCCESS"
UPLOAD_STATUS_PENDING = "PENDING"
UPLOAD_STATUS_FAILED = "FAILED"

# Default pagination settings for artifacts
DEFAULT_ARTIFACTS_PAGE_SIZE = 20
MAX_ARTIFACTS_PAGE_SIZE = 128

# File upload settings
MAX_UPLOAD_SIZE = 10 * 1024 * 1024 * 1024  # 10GB
UPLOAD_CHUNK_SIZE = 14 * 1024 * 1024  # 14MB chunks

# RFC3339 timestamp formats
RFC3339_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
RFC3339_FORMAT_NO_MICROSECONDS = "%Y-%m-%dT%H:%M:%SZ"
