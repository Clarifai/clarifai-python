import time
import uuid
from typing import Dict, List

from clarifai.runners.utils.data_types import Audio, Image, Video
from clarifai.runners.utils.data_utils import process_audio, process_image, process_video


def generate_id():
    return f"chatcmpl-{uuid.uuid4().hex}"


def _format_non_streaming_response(
    generated_text,
    model="custom-model",
    id=None,
    created=None,
    prompt_tokens=None,
    completion_tokens=None,
    finish_reason="stop",
):
    if id is None:
        id = generate_id()
    if created is None:
        created = int(time.time())

    response = {
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text,
                },
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }

    if prompt_tokens is not None and completion_tokens is not None:
        response["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    return response


def _format_streaming_response(
    generated_chunks,
    model="custom-model",
    id=None,
    created=None,
    finish_reason="stop",
):
    if id is None:
        id = generate_id()
    if created is None:
        created = int(time.time())

    for chunk in generated_chunks:
        yield {
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": chunk,
                    },
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
        }

    # Final chunk indicating completion
    yield {
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }


def openai_response(
    generated_text,
    model="custom-model",
    id=None,
    created=None,
    prompt_tokens=None,
    completion_tokens=None,
    finish_reason="stop",
    stream=True,
):
    if stream:
        return _format_streaming_response(generated_text, model, id, created, finish_reason)
    else:
        return _format_non_streaming_response(
            generated_text, model, id, created, prompt_tokens, completion_tokens, finish_reason
        )


def openai_to_hf_messages(openai_messages):
    """
    Converts OpenAI-style chat messages into a format compatible with Hugging Face's
    `tokenizer.apply_chat_template()` function, supporting all modalities (text, images, etc.).

    Args:
        openai_messages (list): List of OpenAI-style messages, where each message is a dict with
                                'role' (str) and 'content' (str or list of parts).

    Returns:
        list: Hugging Face-compatible messages. Each message is a dict with 'role' and 'content'.
              Content is a string (text-only) or a list of parts (multimodal).
    """
    hf_messages = []
    for msg in openai_messages:
        role = msg['role']
        content = msg['content']

        if isinstance(content, list):
            # Handle multimodal content (e.g., text + images)
            converted_content = []
            for part in content:
                if part['type'] == 'text':
                    converted_content.append({'type': 'text', 'text': part['text']})
                elif part['type'] == 'image_url':
                    # Handle image (extract base64 or URL)
                    image_url = part["image_url"]["url"]
                    if image_url.startswith("data:image"):
                        # Base64-encoded image
                        b64_img = image_url.split(",")[1]
                        converted_content.append({'type': 'image', 'base64': b64_img})
                    else:
                        # URL (model must handle downloads)
                        converted_content.append({'type': 'image', 'url': image_url})
                elif part['type'] == 'video_url':
                    video_url = part["video_url"]["url"]
                    if video_url.startswith("data:video"):
                        raise ValueError("Base64 video data is not supported in HF format.")
                    else:
                        # URL (model must handle downloads)
                        converted_content.append({'type': 'video', 'url': video_url})
                else:
                    raise ValueError(f"Unsupported content type: {part['type']} for conversion.")
            hf_content = converted_content
        else:
            # Text-only content (string)
            hf_content = content

        hf_messages.append({'role': role, 'content': hf_content})

    return hf_messages


def build_openai_messages(
    prompt: str = None,
    image: Image = None,
    images: List[Image] = None,
    audio: Audio = None,
    audios: List[Audio] = None,
    video: Video = None,
    videos: List[Video] = None,
    messages: List[Dict] = None,
) -> List[Dict]:
    """
    Construct OpenAI-compatible messages from input components.
      Args:
          prompt (str): The prompt text.
          image (Image): Clarifai Image object.
          images (List[Image]): List of Clarifai Image objects.
          audio (Audio): Clarifai Audio object.
          audios (List[Audio]): List of Clarifai Audio objects.
          video (Video): Clarifai Video object.
          videos (List[Video]): List of Clarifai Video objects.
          messages (List[Dict]): List of chat messages.
      Returns:
          List[Dict]: Formatted chat messages.
    """

    openai_messages = []
    # Add previous conversation history
    if messages and is_openai_chat_format(messages):
        openai_messages.extend(messages)

    content = []
    if prompt.strip():
        # Build content array for current message
        content.append({'type': 'text', 'text': prompt})
    # Add single image if present
    if image:
        content.append(process_image(image))
    # Add multiple images if present
    if images:
        for img in images:
            content.append(process_image(img))
    # Add single audio if present
    if audio:
        content.append(process_audio(audio))
    # Add multiple audios if present
    if audios:
        for audio in audios:
            content.append(process_audio(audio))
    # Add single video if present
    if video:
        content.append(process_video(video))
    # Add multiple videos if present
    if videos:
        for video in videos:
            content.append(process_video(video))

    if content:
        # Append complete user message
        openai_messages.append({'role': 'user', 'content': content})

    return openai_messages


def is_openai_chat_format(messages):
    """
    Verify if the given argument follows the OpenAI chat messages format.

    Args:
        messages (list): A list of dictionaries representing chat messages.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(messages, list):
        return False

    valid_roles = {"system", "user", "assistant", "function"}

    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
        if msg["role"] not in valid_roles:
            return False

        content = msg["content"]

        # Content should be either a string (text message) or a multimodal list
        if isinstance(content, str):
            continue  # Valid text message

        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    return False
    return True
