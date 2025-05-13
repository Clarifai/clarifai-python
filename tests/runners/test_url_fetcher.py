from clarifai_grpc.grpc.api import resources_pb2, service_pb2

from clarifai.runners.utils.url_fetcher import ensure_urls_downloaded

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
