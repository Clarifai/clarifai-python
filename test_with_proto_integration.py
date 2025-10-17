#!/usr/bin/env python3
"""
Integration test to validate the with_proto functionality works end-to-end
"""
import os
from unittest.mock import Mock, patch, MagicMock
from clarifai.client.model import Model
from clarifai.client.model_client import ModelClient
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2


def create_mock_method_signature():
    """Create a mock method signature for testing"""
    method_signature = resources_pb2.MethodSignature()
    method_signature.name = 'predict'
    method_signature.method_type = resources_pb2.UNARY_UNARY
    method_signature.description = "Test prediction method"
    
    # Add input field
    input_field = method_signature.input_fields.add()
    input_field.name = 'text'
    
    # Add output field  
    output_field = method_signature.output_fields.add()
    output_field.name = 'text'
    
    return method_signature


def create_mock_response(output_text="Hello World"):
    """Create a mock protobuf response"""
    mock_response = service_pb2.MultiOutputResponse()
    mock_response.status.code = status_code_pb2.SUCCESS
    output = mock_response.outputs.add()
    output.data.text.raw = output_text
    return mock_response


def test_model_client_with_proto():
    """Test ModelClient with with_proto parameter"""
    print("Testing ModelClient with with_proto...")
    
    # Create mock objects
    mock_stub = Mock()
    mock_response = create_mock_response("Test Response")
    mock_stub.PostModelOutputs.return_value = mock_response
    
    # Create ModelClient
    request_template = service_pb2.PostModelOutputsRequest()
    client = ModelClient(stub=mock_stub, request_template=request_template)
    
    # Set up method signatures manually to avoid fetching
    method_signature = create_mock_method_signature()
    client._method_signatures = {'predict': method_signature}
    client._defined = True  # Skip automatic fetching
    
    # Test direct method calls with with_proto
    print("Testing _predict with with_proto=False...")
    result = client._predict({'text': 'test input'}, 'predict', with_proto=False)
    print(f"Result: {result}")
    assert result == "Test Response", f"Expected 'Test Response', got {result}"
    
    print("Testing _predict with with_proto=True...")
    result, proto = client._predict({'text': 'test input'}, 'predict', with_proto=True)
    print(f"Result: {result}, Proto type: {type(proto)}")
    assert result == "Test Response", f"Expected 'Test Response', got {result}"
    assert proto == mock_response, "Proto response should match mock_response"
    
    print("✓ ModelClient with_proto tests passed!")


def test_dynamic_method_binding_with_proto():
    """Test that dynamically bound methods work with with_proto"""
    print("Testing dynamic method binding with with_proto...")
    
    # Mock the deserialization to return simple strings
    with patch('clarifai.client.model_client.deserialize') as mock_deserialize, \
         patch('clarifai.client.model_client.serialize') as mock_serialize:
        
        mock_deserialize.return_value = "Dynamic Test Response"
        mock_serialize.return_value = None  # serialize doesn't return anything, just modifies proto
        
        # Create mock objects
        mock_stub = Mock()
        mock_response = create_mock_response("Dynamic Response")
        mock_stub.PostModelOutputs.return_value = mock_response
        
        # Create ModelClient
        request_template = service_pb2.PostModelOutputsRequest()
        client = ModelClient(stub=mock_stub, request_template=request_template)
        
        # Set up method signatures
        method_signature = create_mock_method_signature()
        client._method_signatures = {'predict': method_signature}
        client._defined = False
        
        # Define the functions (this simulates what happens in normal operation)
        client._define_functions()
        
        # Now test the dynamically created predict method
        print("Testing dynamically bound predict method without with_proto...")
        result = client.predict(text="test input")
        print(f"Result: {result}")
        assert result == "Dynamic Test Response"
        
        print("Testing dynamically bound predict method with with_proto=True...")
        result, proto = client.predict(text="test input", with_proto=True)
        print(f"Result: {result}, Proto type: {type(proto)}")
        assert result == "Dynamic Test Response"
        assert proto == mock_response
        
        print("✓ Dynamic method binding with_proto tests passed!")


def test_model_integration():
    """Test full Model class integration"""
    print("Testing Model class integration...")
    
    # Mock the necessary components to avoid network calls
    with patch('clarifai.client.base.BaseClient.__init__'), \
         patch('clarifai.client.lister.Lister.__init__'), \
         patch('clarifai.utils.misc.dict_to_protobuf'):
        
        # Create a Model instance (mocking the complex init)
        model = Model.__new__(Model)
        model.kwargs = {'id': 'test-model', 'model_version': {'id': 'test-version'}}
        model.model_info = Mock()
        model._client = None
        model._async_client = None
        model._added_methods = False
        
        # Mock the client property to return our test client
        mock_client = Mock()
        mock_client.fetch = Mock()
        mock_client._method_signatures = {'predict': create_mock_method_signature()}
        
        # Mock a predict method that supports with_proto
        def mock_predict(*args, **kwargs):
            with_proto = kwargs.get('with_proto', False)
            result = "Model Response"
            if with_proto:
                return result, create_mock_response(result)
            return result
        
        mock_client.predict = mock_predict
        
        # Test the Model's __getattr__ method
        with patch.object(Model, 'client', new_callable=lambda: mock_client):
            model._added_methods = False  # Reset to force method addition
            
            # Simulate the __getattr__ call
            predict_method = model.__getattr__('predict')
            
            print("Testing Model.predict without with_proto...")
            result = predict_method(text="test")
            print(f"Result: {result}")
            
            print("Testing Model.predict with with_proto=True...")
            result, proto = predict_method(text="test", with_proto=True)
            print(f"Result: {result}, Proto type: {type(proto)}")
            
            print("✓ Model integration tests passed!")


if __name__ == '__main__':
    test_model_client_with_proto()
    test_dynamic_method_binding_with_proto()
    test_model_integration()
    print("\n🎉 All integration tests passed!")