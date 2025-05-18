import time
import uuid


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
