#!/usr/bin/env python3
"""
Example demonstrating the with_proto functionality in Clarifai Python SDK

This feature allows pythonic models to return both the processed result 
and the raw protobuf response from the API.
"""
import os

from clarifai.client.model import Model


def demo_with_proto_functionality():
    """
    Demonstrate how to use with_proto parameter with pythonic models
    """
    print("=== Clarifai with_proto Feature Demo ===\n")

    # Example model URL and configuration
    # Note: Replace with actual model URL and credentials for real usage
    model_url = "https://clarifai.com/user/app/models/my-pythonic-model"
    deployment_id = "my-deployment-id"
    pat = os.getenv('CLARIFAI_PAT')
    if not pat:
        pat = 'your-pat-token'  # Placeholder for demo purposes

    try:
        # Initialize the model client
        model = Model(
            url=model_url,
            pat=pat,
            deployment_id=deployment_id,
        )

        print("Model initialized successfully!\n")

        # Example 1: Basic predict without with_proto (existing behavior)
        print("1. Standard predict call (existing behavior):")
        print("   response = model.predict(prompt='What is AI?')")
        print("   # Returns only the processed result\n")

        # Example 2: Predict with with_proto=True (NEW feature)  
        print("2. Predict with protobuf response (NEW feature):")
        print("   response, proto = model.predict(")
        print("       prompt='What is AI?',")
        print("       with_proto=True  # <- This is the new parameter")
        print("   )")
        print("   # Returns tuple: (processed_result, raw_protobuf_response)\n")

        # Example 3: Generate streaming with with_proto
        print("3. Streaming generate with protobuf:")
        print("   for response, proto in model.generate(")
        print("       prompt='Explain quantum computing',")
        print("       with_proto=True")
        print("   ):")
        print("       print(f'Generated: {response}')")
        print("       print(f'Status: {proto.status.code}')")
        print("       # Access raw metadata like timestamps, request IDs, etc.\n")

        # Example 4: Custom model method with with_proto
        print("4. Custom model methods support with_proto:")
        print("   # Any pythonic model method automatically supports with_proto")
        print("   result, proto = model.my_custom_method(")
        print("       input_data='some data',")
        print("       temperature=0.7,")
        print("       with_proto=True")
        print("   )")
        print("   # Works with any method defined in the ModelClass\n")

        # Benefits section
        print("Benefits of with_proto=True:")
        print("- Access to complete API response metadata")
        print("- Debugging capabilities (status codes, request IDs)")
        print("- Performance metrics (latency, processing info)")
        print("- Raw data for advanced use cases")
        print("- Backward compatible (existing code unchanged)")

    except Exception as e:
        print(f"Demo setup note: {e}")
        print("This is a demonstration script showing the API usage.")
        print("For actual usage, ensure you have valid model URLs and credentials.\n")

        # Show the implementation regardless
        show_implementation_details()

def show_implementation_details():
    """Show technical implementation details"""
    print("\n=== Implementation Details ===\n")

    print("The with_proto feature is implemented by modifying the ModelClient class:")
    print("1. All method binding now extracts 'with_proto' parameter")
    print("2. Core methods (_predict, _generate, _stream) accept with_proto")
    print("3. When with_proto=True, methods return (result, proto_response)")
    print("4. When with_proto=False (default), methods return just result")
    print("5. Fully backward compatible - no existing code needs changes\n")

    print("Supported methods:")
    print("- Synchronous: predict(), generate(), stream(), custom_method()")
    print("- Asynchronous: async_predict(), async_generate(), async_stream()")
    print("- All methods support with_proto parameter consistently\n")

def technical_example():
    """Show a technical example of what the protobuf response contains"""
    print("=== What's in the Protobuf Response? ===\n")

    print("The protobuf response contains rich metadata:")
    print("""
    proto.status.code          # Success/error status
    proto.status.description   # Human readable status
    proto.status.details      # Additional error details
    proto.status.req_id       # Request ID for debugging
    proto.status.percent_completed  # Progress for long operations

    proto.outputs[0].data     # Raw output data
    proto.outputs[0].status   # Per-output status
    proto.model.id           # Model information
    proto.model.model_version.id  # Model version used

    # And much more depending on the model and operation
    """)

    print("Example usage for debugging:")
    print("""
    try:
        result, proto = model.predict(text="Hello", with_proto=True)
        print(f"Success! Request ID: {proto.status.req_id}")
    except Exception as e:
        print(f"Error: {proto.status.description}")
        print(f"Debug with Request ID: {proto.status.req_id}")
    """)

if __name__ == '__main__':
    demo_with_proto_functionality()
    technical_example()
    print("\n✅ Demo complete! The with_proto feature is ready to use.")
