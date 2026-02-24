"""Tests for the improved `clarifai model predict` CLI command."""

import json
from unittest.mock import MagicMock

import click
import pytest

from clarifai.cli.model import (
    _build_chat_request,
    _coerce_input_value,
    _detect_media_type_from_ext,
    _get_first_media_param,
    _get_first_str_param,
    _is_streaming_method,
    _parse_kv_inputs,
    _resolve_model_ref,
    _select_method,
)


# ---------------------------------------------------------------------------
# _resolve_model_ref
# ---------------------------------------------------------------------------
class TestResolveModelRef:
    def test_full_url_passthrough(self):
        url = "https://clarifai.com/openai/chat-completion/models/GPT-4"
        assert _resolve_model_ref(url) == url

    def test_http_url_passthrough(self):
        url = "http://clarifai.com/openai/chat-completion/models/GPT-4"
        assert _resolve_model_ref(url) == url

    def test_shorthand_default_base(self):
        result = _resolve_model_ref("openai/chat-completion/models/GPT-4")
        assert result == "https://clarifai.com/openai/chat-completion/models/GPT-4"

    def test_shorthand_custom_base(self):
        result = _resolve_model_ref(
            "openai/chat-completion/models/GPT-4",
            ui_base="https://web-dev.clarifai.com",
        )
        assert result == "https://web-dev.clarifai.com/openai/chat-completion/models/GPT-4"

    def test_shorthand_trailing_slash_base(self):
        result = _resolve_model_ref(
            "openai/chat-completion/models/GPT-4",
            ui_base="https://clarifai.com/",
        )
        assert result == "https://clarifai.com/openai/chat-completion/models/GPT-4"

    def test_invalid_no_models_keyword(self):
        with pytest.raises(click.UsageError, match="Invalid model reference"):
            _resolve_model_ref("openai/chat-completion/GPT-4")

    def test_invalid_too_few_parts(self):
        with pytest.raises(click.UsageError, match="Invalid model reference"):
            _resolve_model_ref("openai/models/GPT-4")

    def test_invalid_too_many_parts(self):
        with pytest.raises(click.UsageError, match="Invalid model reference"):
            _resolve_model_ref("openai/app/models/GPT-4/extra")

    def test_none_returns_none(self):
        assert _resolve_model_ref(None) is None

    def test_empty_returns_none(self):
        assert _resolve_model_ref("") is None


# ---------------------------------------------------------------------------
# Mock helpers for method signature-based tests
# ---------------------------------------------------------------------------
def _make_field(name, data_type, iterator=False):
    """Create a mock ModelTypeField."""
    from clarifai_grpc.grpc.api import resources_pb2

    field = resources_pb2.ModelTypeField()
    field.name = name
    field.type = data_type
    field.iterator = iterator
    return field


def _make_method_sig(name, input_fields, output_fields):
    """Create a mock MethodSignature."""
    from clarifai_grpc.grpc.api import resources_pb2

    sig = resources_pb2.MethodSignature()
    sig.name = name
    sig.input_fields.extend(input_fields)
    sig.output_fields.extend(output_fields)
    return sig


def _make_model_client(method_sigs_dict, sig_strings=None):
    """Create a mock model client with pre-set method signatures.

    Args:
        method_sigs_dict: Dict of method_name -> MethodSignature proto.
        sig_strings: Dict of method_name -> signature display string.
    """
    client = MagicMock()
    client._defined = True
    client._method_signatures = method_sigs_dict
    client.available_methods.return_value = list(method_sigs_dict.keys())
    if sig_strings:
        client.method_signature.side_effect = lambda m: sig_strings[m]
    return client


# ---------------------------------------------------------------------------
# _get_first_str_param
# ---------------------------------------------------------------------------
class TestGetFirstStrParam:
    def test_finds_str_param(self):
        from clarifai_grpc.grpc.api import resources_pb2

        sig = _make_method_sig(
            "predict",
            [_make_field("prompt", resources_pb2.ModelTypeField.DataType.STR)],
            [_make_field("return", resources_pb2.ModelTypeField.DataType.STR)],
        )
        client = _make_model_client({"predict": sig})
        assert _get_first_str_param(client, "predict") == "prompt"

    def test_skips_non_str(self):
        from clarifai_grpc.grpc.api import resources_pb2

        sig = _make_method_sig(
            "predict",
            [
                _make_field("count", resources_pb2.ModelTypeField.DataType.INT),
                _make_field("text", resources_pb2.ModelTypeField.DataType.STR),
            ],
            [_make_field("return", resources_pb2.ModelTypeField.DataType.STR)],
        )
        client = _make_model_client({"predict": sig})
        assert _get_first_str_param(client, "predict") == "text"

    def test_no_str_param(self):
        from clarifai_grpc.grpc.api import resources_pb2

        sig = _make_method_sig(
            "predict",
            [_make_field("image", resources_pb2.ModelTypeField.DataType.IMAGE)],
            [_make_field("return", resources_pb2.ModelTypeField.DataType.STR)],
        )
        client = _make_model_client({"predict": sig})
        assert _get_first_str_param(client, "predict") is None

    def test_missing_method(self):
        client = _make_model_client({})
        assert _get_first_str_param(client, "nonexistent") is None


# ---------------------------------------------------------------------------
# _get_first_media_param
# ---------------------------------------------------------------------------
class TestGetFirstMediaParam:
    def test_finds_image_param(self):
        from clarifai_grpc.grpc.api import resources_pb2

        sig = _make_method_sig(
            "predict",
            [_make_field("image", resources_pb2.ModelTypeField.DataType.IMAGE)],
            [_make_field("return", resources_pb2.ModelTypeField.DataType.STR)],
        )
        client = _make_model_client({"predict": sig})
        name, media_type = _get_first_media_param(client, "predict")
        assert name == "image"
        assert media_type == "image"

    def test_finds_video_param(self):
        from clarifai_grpc.grpc.api import resources_pb2

        sig = _make_method_sig(
            "predict",
            [_make_field("clip", resources_pb2.ModelTypeField.DataType.VIDEO)],
            [_make_field("return", resources_pb2.ModelTypeField.DataType.STR)],
        )
        client = _make_model_client({"predict": sig})
        name, media_type = _get_first_media_param(client, "predict")
        assert name == "clip"
        assert media_type == "video"

    def test_finds_audio_param(self):
        from clarifai_grpc.grpc.api import resources_pb2

        sig = _make_method_sig(
            "predict",
            [_make_field("audio", resources_pb2.ModelTypeField.DataType.AUDIO)],
            [_make_field("return", resources_pb2.ModelTypeField.DataType.STR)],
        )
        client = _make_model_client({"predict": sig})
        name, media_type = _get_first_media_param(client, "predict")
        assert name == "audio"
        assert media_type == "audio"

    def test_no_media_param(self):
        from clarifai_grpc.grpc.api import resources_pb2

        sig = _make_method_sig(
            "predict",
            [_make_field("prompt", resources_pb2.ModelTypeField.DataType.STR)],
            [_make_field("return", resources_pb2.ModelTypeField.DataType.STR)],
        )
        client = _make_model_client({"predict": sig})
        name, media_type = _get_first_media_param(client, "predict")
        assert name is None
        assert media_type is None


# ---------------------------------------------------------------------------
# _coerce_input_value
# ---------------------------------------------------------------------------
class TestCoerceInputValue:
    def test_coerce_int(self):
        from clarifai_grpc.grpc.api import resources_pb2

        sig = _make_method_sig(
            "predict",
            [_make_field("max_tokens", resources_pb2.ModelTypeField.DataType.INT)],
            [_make_field("return", resources_pb2.ModelTypeField.DataType.STR)],
        )
        client = _make_model_client({"predict": sig})
        assert _coerce_input_value("200", client, "predict", "max_tokens") == 200

    def test_coerce_float(self):
        from clarifai_grpc.grpc.api import resources_pb2

        sig = _make_method_sig(
            "predict",
            [_make_field("temperature", resources_pb2.ModelTypeField.DataType.FLOAT)],
            [_make_field("return", resources_pb2.ModelTypeField.DataType.STR)],
        )
        client = _make_model_client({"predict": sig})
        assert _coerce_input_value("0.7", client, "predict", "temperature") == 0.7

    def test_coerce_bool_true(self):
        from clarifai_grpc.grpc.api import resources_pb2

        sig = _make_method_sig(
            "predict",
            [_make_field("stream", resources_pb2.ModelTypeField.DataType.BOOL)],
            [_make_field("return", resources_pb2.ModelTypeField.DataType.STR)],
        )
        client = _make_model_client({"predict": sig})
        assert _coerce_input_value("true", client, "predict", "stream") is True
        assert _coerce_input_value("yes", client, "predict", "stream") is True
        assert _coerce_input_value("1", client, "predict", "stream") is True

    def test_coerce_bool_false(self):
        from clarifai_grpc.grpc.api import resources_pb2

        sig = _make_method_sig(
            "predict",
            [_make_field("stream", resources_pb2.ModelTypeField.DataType.BOOL)],
            [_make_field("return", resources_pb2.ModelTypeField.DataType.STR)],
        )
        client = _make_model_client({"predict": sig})
        assert _coerce_input_value("false", client, "predict", "stream") is False

    def test_str_passthrough(self):
        from clarifai_grpc.grpc.api import resources_pb2

        sig = _make_method_sig(
            "predict",
            [_make_field("prompt", resources_pb2.ModelTypeField.DataType.STR)],
            [_make_field("return", resources_pb2.ModelTypeField.DataType.STR)],
        )
        client = _make_model_client({"predict": sig})
        assert _coerce_input_value("hello", client, "predict", "prompt") == "hello"

    def test_unknown_param_passthrough(self):
        client = _make_model_client({})
        assert _coerce_input_value("hello", client, "predict", "unknown") == "hello"


# ---------------------------------------------------------------------------
# _parse_kv_inputs
# ---------------------------------------------------------------------------
class TestParseKvInputs:
    def test_simple_kv(self):
        from clarifai_grpc.grpc.api import resources_pb2

        sig = _make_method_sig(
            "predict",
            [
                _make_field("prompt", resources_pb2.ModelTypeField.DataType.STR),
                _make_field("max_tokens", resources_pb2.ModelTypeField.DataType.INT),
            ],
            [_make_field("return", resources_pb2.ModelTypeField.DataType.STR)],
        )
        client = _make_model_client({"predict": sig})
        result = _parse_kv_inputs(("prompt=Hello", "max_tokens=200"), client, "predict")
        assert result == {"prompt": "Hello", "max_tokens": 200}

    def test_value_with_equals(self):
        """Values can contain = signs."""
        client = _make_model_client({})
        result = _parse_kv_inputs(("prompt=a=b=c",), client, "predict")
        assert result == {"prompt": "a=b=c"}

    def test_invalid_no_equals(self):
        client = _make_model_client({})
        with pytest.raises(click.UsageError, match="Invalid input format"):
            _parse_kv_inputs(("no-equals-here",), client, "predict")


# ---------------------------------------------------------------------------
# _detect_media_type_from_ext
# ---------------------------------------------------------------------------
class TestDetectMediaTypeFromExt:
    def test_image_extensions(self):
        for ext in [".jpg", ".png", ".gif", ".bmp", ".tiff"]:
            assert _detect_media_type_from_ext(f"photo{ext}") == "image"

    def test_video_extensions(self):
        for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            assert _detect_media_type_from_ext(f"clip{ext}") == "video"

    def test_audio_extensions(self):
        for ext in [".wav", ".mp3", ".flac", ".aac", ".ogg"]:
            assert _detect_media_type_from_ext(f"sound{ext}") == "audio"

    def test_unknown_defaults_to_image(self):
        assert _detect_media_type_from_ext("file.xyz") == "image"

    def test_url_path(self):
        assert _detect_media_type_from_ext("https://example.com/photo.jpg") == "image"
        assert _detect_media_type_from_ext("https://example.com/clip.mp4") == "video"


# ---------------------------------------------------------------------------
# _is_streaming_method
# ---------------------------------------------------------------------------
class TestIsStreamingMethod:
    def test_streaming_signature(self):
        client = MagicMock()
        client.method_signature.return_value = "def generate(prompt: str) -> Iterator[str]:"
        assert _is_streaming_method(client, "generate") is True

    def test_unary_signature(self):
        client = MagicMock()
        client.method_signature.return_value = "def predict(prompt: str) -> str:"
        assert _is_streaming_method(client, "predict") is False

    def test_no_return_type(self):
        client = MagicMock()
        client.method_signature.return_value = "def predict(prompt: str):"
        assert _is_streaming_method(client, "predict") is False


# ---------------------------------------------------------------------------
# _select_method
# ---------------------------------------------------------------------------
class TestSelectMethod:
    def _client_with_sigs(self, sig_strings):
        client = MagicMock()
        client.method_signature.side_effect = lambda m: sig_strings[m]
        return client

    def test_chat_selects_openai_stream(self):
        methods = ['predict', 'openai_stream_transport', 'openai_transport']
        client = self._client_with_sigs(
            {
                'predict': 'def predict(prompt: str) -> str:',
                'openai_stream_transport': 'def openai_stream_transport(msg: str) -> Iterator[str]:',
                'openai_transport': 'def openai_transport(msg: str) -> str:',
            }
        )
        method, is_openai = _select_method(
            methods, client, None, is_chat=True, has_text_input=True
        )
        assert method == 'openai_stream_transport'
        assert is_openai is True

    def test_chat_fallback_to_non_stream(self):
        methods = ['predict', 'openai_transport']
        client = self._client_with_sigs(
            {
                'predict': 'def predict(prompt: str) -> str:',
                'openai_transport': 'def openai_transport(msg: str) -> str:',
            }
        )
        method, is_openai = _select_method(
            methods, client, None, is_chat=True, has_text_input=True
        )
        assert method == 'openai_transport'
        assert is_openai is True

    def test_chat_no_openai_methods_errors(self):
        methods = ['predict']
        client = self._client_with_sigs(
            {
                'predict': 'def predict(prompt: str) -> str:',
            }
        )
        with pytest.raises(click.UsageError, match="does not support OpenAI chat"):
            _select_method(methods, client, None, is_chat=True, has_text_input=True)

    def test_explicit_method(self):
        methods = ['predict', 'generate']
        client = self._client_with_sigs(
            {
                'predict': 'def predict(prompt: str) -> str:',
                'generate': 'def generate(prompt: str) -> Iterator[str]:',
            }
        )
        method, is_openai = _select_method(
            methods, client, 'predict', is_chat=False, has_text_input=True
        )
        assert method == 'predict'
        assert is_openai is False

    def test_explicit_method_not_available(self):
        methods = ['predict']
        client = self._client_with_sigs(
            {
                'predict': 'def predict(prompt: str) -> str:',
            }
        )
        with pytest.raises(click.UsageError, match="not available"):
            _select_method(methods, client, 'nonexistent', is_chat=False, has_text_input=True)

    def test_auto_openai_for_text(self):
        methods = ['predict', 'openai_stream_transport']
        client = self._client_with_sigs(
            {
                'predict': 'def predict(prompt: str) -> str:',
                'openai_stream_transport': 'def openai_stream_transport(msg: str) -> Iterator[str]:',
            }
        )
        method, is_openai = _select_method(
            methods, client, None, is_chat=False, has_text_input=True
        )
        assert method == 'openai_stream_transport'
        assert is_openai is True

    def test_prefer_streaming_no_openai(self):
        methods = ['predict', 'generate']
        client = self._client_with_sigs(
            {
                'predict': 'def predict(prompt: str) -> str:',
                'generate': 'def generate(prompt: str) -> Iterator[str]:',
            }
        )
        method, is_openai = _select_method(
            methods, client, None, is_chat=False, has_text_input=True
        )
        assert method == 'generate'
        assert is_openai is False

    def test_fallback_to_predict(self):
        methods = ['predict']
        client = self._client_with_sigs(
            {
                'predict': 'def predict(prompt: str) -> str:',
            }
        )
        method, is_openai = _select_method(
            methods, client, None, is_chat=False, has_text_input=True
        )
        assert method == 'predict'
        assert is_openai is False

    def test_no_text_skips_openai_auto(self):
        """Without text input, don't auto-select OpenAI path."""
        methods = ['predict', 'openai_stream_transport']
        client = self._client_with_sigs(
            {
                'predict': 'def predict(prompt: str) -> str:',
                'openai_stream_transport': 'def openai_stream_transport(msg: str) -> Iterator[str]:',
            }
        )
        method, is_openai = _select_method(
            methods, client, None, is_chat=False, has_text_input=False
        )
        assert method == 'predict'
        assert is_openai is False


# ---------------------------------------------------------------------------
# _build_chat_request
# ---------------------------------------------------------------------------
class TestBuildChatRequest:
    def test_builds_valid_json(self):
        result = _build_chat_request("What is AI?")
        data = json.loads(result)
        assert data["messages"] == [{"role": "user", "content": "What is AI?"}]
        assert data["stream"] is True

    def test_preserves_special_chars(self):
        result = _build_chat_request('Say "hello" & <goodbye>')
        data = json.loads(result)
        assert data["messages"][0]["content"] == 'Say "hello" & <goodbye>'
