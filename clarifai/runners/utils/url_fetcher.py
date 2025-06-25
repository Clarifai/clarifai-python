import concurrent.futures

import fsspec

from clarifai.utils.logging import logger


def download_input(input, auth_helper=None):
    _download_input_data(input.data, auth_helper=auth_helper)
    if input.data.parts:
        for i in range(len(input.data.parts)):
            _download_input_data(input.data.parts[i].data, auth_helper=auth_helper)


def _download_input_data(input_data, auth_helper=None):
    """
    This function will download any urls that are not already bytes.

    Args:
        input_data: The input data containing URLs to download
        auth_helper: Optional ClarifaiAuthHelper instance for authentication
    """
    # Get auth headers if auth_helper is provided
    auth_kwargs = {}
    if auth_helper is not None:
        auth_kwargs = _get_auth_kwargs(auth_helper)

    if input_data.image.url and not input_data.image.base64:
        # Download the image
        with fsspec.open(input_data.image.url, 'rb', **auth_kwargs) as f:
            input_data.image.base64 = f.read()
    if input_data.video.url and not input_data.video.base64:
        # Download the video
        with fsspec.open(input_data.video.url, 'rb', **auth_kwargs) as f:
            input_data.video.base64 = f.read()
    if input_data.audio.url and not input_data.audio.base64:
        # Download the audio
        with fsspec.open(input_data.audio.url, 'rb', **auth_kwargs) as f:
            input_data.audio.base64 = f.read()
    if input_data.text.url and not input_data.text.raw:
        # Download the text
        with fsspec.open(input_data.text.url, 'r', **auth_kwargs) as f:
            input_data.text.raw = f.read()


def _get_auth_kwargs(auth_helper):
    """
    Convert ClarifaiAuthHelper metadata to fsspec-compatible kwargs.

    Args:
        auth_helper: ClarifaiAuthHelper instance

    Returns:
        dict: kwargs to pass to fsspec.open() for authentication
    """
    if auth_helper is None:
        return {}

    try:
        # Get authentication metadata from the helper
        metadata = auth_helper.metadata

        # Convert gRPC metadata tuples to HTTP headers dict
        headers = {}
        for key, value in metadata:
            # Skip non-auth headers
            if key in ('authorization', 'x-clarifai-session-token'):
                headers[key] = value

        # Return fsspec-compatible kwargs
        return {'client_kwargs': {'headers': headers}}
    except Exception as e:
        logger.warning(f"Failed to get authentication headers: {e}")
        return {}


def ensure_urls_downloaded(request, max_threads=128, auth_helper=None):
    """
    This function will download any urls that are not already bytes and parallelize with a thread pool.

    Args:
        request: The request containing inputs with URLs to download
        max_threads: Maximum number of threads to use for parallel downloads
        auth_helper: Optional ClarifaiAuthHelper instance for authentication
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for input in request.inputs:
            futures.append(executor.submit(download_input, input, auth_helper))
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.exception(f"Error downloading input: {e}")
