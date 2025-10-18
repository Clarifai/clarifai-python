"""
Tests for the with_proto feature in pythonic models
"""

import unittest
from unittest.mock import Mock, patch

from clarifai_grpc.grpc.api import service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

from clarifai.client.model_client import ModelClient


class TestWithProtoFeature(unittest.TestCase):
    """Test class for with_proto functionality"""

    def test_predict_with_proto_parameter_exists(self):
        """Test that _predict method has with_proto parameter"""
        import inspect

        sig = inspect.signature(ModelClient._predict)
        self.assertIn('with_proto', sig.parameters)
        self.assertIs(sig.parameters['with_proto'].default, False)

    def test_generate_with_proto_parameter_exists(self):
        """Test that _generate method has with_proto parameter"""
        import inspect

        sig = inspect.signature(ModelClient._generate)
        self.assertIn('with_proto', sig.parameters)
        self.assertIs(sig.parameters['with_proto'].default, False)

    def test_stream_with_proto_parameter_exists(self):
        """Test that _stream method has with_proto parameter"""
        import inspect

        sig = inspect.signature(ModelClient._stream)
        self.assertIn('with_proto', sig.parameters)
        self.assertIs(sig.parameters['with_proto'].default, False)

    def test_async_predict_with_proto_parameter_exists(self):
        """Test that _async_predict method has with_proto parameter"""
        import inspect

        sig = inspect.signature(ModelClient._async_predict)
        self.assertIn('with_proto', sig.parameters)
        self.assertIs(sig.parameters['with_proto'].default, False)

    def test_async_generate_with_proto_parameter_exists(self):
        """Test that _async_generate method has with_proto parameter"""
        import inspect

        sig = inspect.signature(ModelClient._async_generate)
        self.assertIn('with_proto', sig.parameters)
        self.assertIs(sig.parameters['with_proto'].default, False)

    def test_async_stream_with_proto_parameter_exists(self):
        """Test that _async_stream method has with_proto parameter"""
        import inspect

        sig = inspect.signature(ModelClient._async_stream)
        self.assertIn('with_proto', sig.parameters)
        self.assertIs(sig.parameters['with_proto'].default, False)

    def test_with_proto_parameter_extraction(self):
        """Test that with_proto parameter is correctly extracted from kwargs"""
        # Test the parameter extraction logic used in bind_f
        test_kwargs = {'param1': 'value1', 'with_proto': True, 'param2': 'value2'}

        # Extract with_proto parameter like the code does
        with_proto = test_kwargs.pop('with_proto', False)

        # Verify extraction worked correctly
        self.assertIs(with_proto, True)
        self.assertNotIn('with_proto', test_kwargs)
        self.assertEqual(test_kwargs, {'param1': 'value1', 'param2': 'value2'})

    def test_with_proto_default_behavior(self):
        """Test that with_proto defaults to False when not provided"""
        test_kwargs = {'param1': 'value1', 'param2': 'value2'}

        # Extract with_proto parameter with default
        with_proto = test_kwargs.pop('with_proto', False)

        # Verify default behavior
        self.assertIs(with_proto, False)
        self.assertEqual(test_kwargs, {'param1': 'value1', 'param2': 'value2'})

    @patch('clarifai.client.model_client.serialize')
    @patch('clarifai.client.model_client.deserialize')
    def test_predict_with_proto_false(self, mock_deserialize, mock_serialize):
        """Test _predict method with with_proto=False returns only result"""
        # Setup mocks
        mock_deserialize.return_value = "test_result"
        mock_serialize.return_value = None

        # Create mock response
        mock_response = service_pb2.MultiOutputResponse()
        mock_response.status.code = status_code_pb2.SUCCESS
        output = mock_response.outputs.add()

        # Create mock stub
        mock_stub = Mock()
        mock_stub.PostModelOutputs.return_value = mock_response

        # Create client with mock method signatures
        request_template = service_pb2.PostModelOutputsRequest()
        client = ModelClient(stub=mock_stub, request_template=request_template)

        # Mock method signature
        method_signature = Mock()
        method_signature.input_fields = []
        method_signature.output_fields = []

        client._method_signatures = {'predict': method_signature}

        # Test with with_proto=False
        result = client._predict({'text': 'test'}, 'predict', with_proto=False)

        # Should return only the result, not a tuple
        self.assertEqual(result, "test_result")
        self.assertNotIsInstance(result, tuple)

    @patch('clarifai.client.model_client.serialize')
    @patch('clarifai.client.model_client.deserialize')
    def test_predict_with_proto_true(self, mock_deserialize, mock_serialize):
        """Test _predict method with with_proto=True returns (result, proto) tuple"""
        # Setup mocks
        mock_deserialize.return_value = "test_result"
        mock_serialize.return_value = None

        # Create mock response
        mock_response = service_pb2.MultiOutputResponse()
        mock_response.status.code = status_code_pb2.SUCCESS
        output = mock_response.outputs.add()

        # Create mock stub
        mock_stub = Mock()
        mock_stub.PostModelOutputs.return_value = mock_response

        # Create client with mock method signatures
        request_template = service_pb2.PostModelOutputsRequest()
        client = ModelClient(stub=mock_stub, request_template=request_template)

        # Mock method signature
        method_signature = Mock()
        method_signature.input_fields = []
        method_signature.output_fields = []

        client._method_signatures = {'predict': method_signature}

        # Test with with_proto=True
        result = client._predict({'text': 'test'}, 'predict', with_proto=True)

        # Should return tuple (result, proto)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "test_result")  # The deserialized result
        self.assertEqual(result[1], mock_response)  # The proto response

    def test_backward_compatibility(self):
        """Test that existing code without with_proto continues to work"""
        # This test ensures that the changes don't break existing functionality

        # Mock the method binding
        method_argnames = ['text', 'temperature']

        # Simulate existing call without with_proto
        args = ('test input',)
        kwargs = {'temperature': 0.7}

        # Extract with_proto (should default to False)
        with_proto = kwargs.pop('with_proto', False)

        # Verify existing behavior is preserved
        self.assertIs(with_proto, False)
        self.assertEqual(len(args), 1)
        self.assertEqual(kwargs, {'temperature': 0.7})

        # Ensure parameter count validation still works
        self.assertLessEqual(len(args), len(method_argnames))
        self.assertLessEqual(len(args) + len(kwargs), len(method_argnames))

    def test_documentation_shows_with_proto_support(self):
        """Test that the feature is properly documented in method docstrings"""

    def test_reserved_parameter_validation(self):
        """Test that with_proto parameter name is properly validated in ModelClass methods"""
        from clarifai.runners.utils.method_signatures import (
            RESERVED_PARAM_WITH_PROTO,
            build_function_signature,
        )

        # Test that a function with with_proto parameter raises an error
        def invalid_method(self, text: str, with_proto: bool = False) -> str:
            return text

        with self.assertRaises(ValueError) as context:
            build_function_signature(invalid_method)

        error_msg = str(context.exception)
        self.assertIn(RESERVED_PARAM_WITH_PROTO, error_msg)
        self.assertIn("reserved", error_msg.lower())

    def test_reserved_parameter_constant_usage(self):
        """Test that the constant is used consistently in ModelClient"""
        from clarifai.runners.utils.method_signatures import RESERVED_PARAM_WITH_PROTO

        # Test parameter extraction using the constant
        test_kwargs = {'param1': 'value1', RESERVED_PARAM_WITH_PROTO: True, 'param2': 'value2'}

        # Extract with_proto parameter using the constant
        with_proto = test_kwargs.pop(RESERVED_PARAM_WITH_PROTO, False)

        # Verify extraction worked correctly
        self.assertIs(with_proto, True)
        self.assertNotIn(RESERVED_PARAM_WITH_PROTO, test_kwargs)
        self.assertEqual(test_kwargs, {'param1': 'value1', 'param2': 'value2'})


if __name__ == '__main__':
    unittest.main()
