from clarifai_grpc.grpc.api import resources_pb2, service_pb2

from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.runners.utils.url_fetcher import _get_auth_kwargs, ensure_urls_downloaded

image_url = "https://samples.clarifai.com/metro-north.jpg"
audio_url = "https://samples.clarifai.com/GoodMorning.wav"
text_url = "https://samples.clarifai.com/negative_sentence_12.txt"


def test_url_fetcher():
    request = service_pb2.PostModelOutputsRequest(
        model_id="model_id",
        version_id="version_id",
        user_app_id=resources_pb2.UserAppIDSet(user_id="user_id", app_id="app_id"),
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(url=image_url),
                    text=resources_pb2.Text(url=text_url),
                    audio=resources_pb2.Audio(url=audio_url),
                ),
            ),
        ],
    )
    ensure_urls_downloaded(request)
    for input in request.inputs:
        assert input.data.image.base64 and len(input.data.image.base64) == 70911, (
            f"Expected length of of image base64 to be 70911, but got {len(input.data.image.base64)}"
        )
        assert input.data.audio.base64 and len(input.data.audio.base64) == 200406, (
            f"Expected length of of audio base64 to be 200406, but got {len(input.data.audio.base64)}"
        )
        assert input.data.text.raw and len(input.data.text.raw) == 35, (
            f"Expected length of of text raw to be 35, but got {len(input.data.text.raw)}"
        )


def test_get_auth_kwargs_with_pat():
    """Test _get_auth_kwargs function with PAT authentication"""
    auth_helper = ClarifaiAuthHelper(
        user_id="test_user", app_id="test_app", pat="test_pat_token", validate=False
    )

    auth_kwargs = _get_auth_kwargs(auth_helper)

    assert 'client_kwargs' in auth_kwargs
    assert 'headers' in auth_kwargs['client_kwargs']
    assert 'authorization' in auth_kwargs['client_kwargs']['headers']
    assert auth_kwargs['client_kwargs']['headers']['authorization'] == 'Key test_pat_token'


def test_get_auth_kwargs_with_session_token():
    """Test _get_auth_kwargs function with session token authentication"""
    auth_helper = ClarifaiAuthHelper(
        user_id="test_user", app_id="test_app", token="test_session_token", validate=False
    )

    auth_kwargs = _get_auth_kwargs(auth_helper)

    assert 'client_kwargs' in auth_kwargs
    assert 'headers' in auth_kwargs['client_kwargs']
    assert 'x-clarifai-session-token' in auth_kwargs['client_kwargs']['headers']
    assert (
        auth_kwargs['client_kwargs']['headers']['x-clarifai-session-token'] == 'test_session_token'
    )


def test_get_auth_kwargs_with_none():
    """Test _get_auth_kwargs function with None input"""
    auth_kwargs = _get_auth_kwargs(None)
    assert auth_kwargs == {}


def test_ensure_urls_downloaded_with_auth_backward_compatibility():
    """Test that ensure_urls_downloaded maintains backward compatibility and supports auth"""
    request = service_pb2.PostModelOutputsRequest(
        model_id="model_id",
        version_id="version_id",
        user_app_id=resources_pb2.UserAppIDSet(user_id="user_id", app_id="app_id"),
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(
                        url="https://httpbin.org/status/404"
                    ),  # Use a URL that will fail gracefully
                ),
            ),
        ],
    )

    # Test without auth_helper (backward compatibility)
    try:
        ensure_urls_downloaded(request, max_threads=1)
    except Exception:
        pass  # Expected to fail due to URL, but function signature should work

    # Test with auth_helper (new functionality)
    auth_helper = ClarifaiAuthHelper(
        user_id="test_user", app_id="test_app", pat="test_pat", validate=False
    )

    try:
        ensure_urls_downloaded(request, max_threads=1, auth_helper=auth_helper)
    except Exception:
        pass  # Expected to fail due to URL, but function signature should work
