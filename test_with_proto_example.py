#!/usr/bin/env python3
"""
Example showing how the with_proto feature works
"""

def demonstrate_with_proto_functionality():
    """
    This demonstrates how the with_proto functionality would work in practice.
    This is a conceptual example since we can't easily test against a real model.
    """
    print("=== Clarifai with_proto Feature Demonstration ===\n")
    
    print("The with_proto feature allows any pythonic model method to return")
    print("both the processed result AND the raw protobuf response from the API.\n")
    
    print("Example usage:")
    print("```python")
    print("from clarifai.client import Model")
    print("")
    print("model = Model(")
    print('    url="https://clarifai.com/user/app/models/my-model",')
    print('    pat="*****",')
    print('    deployment_id="my-deploy",')
    print(")")
    print("")
    print("# Normal usage (existing behavior)")
    print('response = model.predict(')
    print('    prompt="What is the future of AI?",')
    print('    reasoning_effort="medium"')
    print(")")
    print("print(response)  # Just the processed result")
    print("")
    print("# With proto information (NEW feature)")
    print('response, proto = model.predict(')
    print('    prompt="What is the future of AI?",')
    print('    reasoning_effort="medium",')
    print('    with_proto=True  # NEW parameter')
    print(")")
    print("print(response)  # The processed result (same as before)")
    print("print(proto)     # The raw protobuf response with additional metadata")
    print("```\n")
    
    print("The feature works with ANY pythonic model method:")
    print("- model.predict(..., with_proto=True)")
    print("- model.generate(..., with_proto=True)")  
    print("- model.stream(..., with_proto=True)")
    print("- model.my_custom_method(..., with_proto=True)")
    print("")
    
    print("Benefits:")
    print("- Access to raw API response metadata")
    print("- Debugging and inspection capabilities")
    print("- Backward compatible - existing code unchanged")
    print("- Works with all model types and custom methods")


def show_implementation_details():
    """Show what was implemented"""
    print("\n=== Implementation Details ===\n")
    
    print("Changes made to clarifai/client/model_client.py:")
    print("")
    print("1. Modified method binding to accept 'with_proto' parameter:")
    print("   - Extracts with_proto from kwargs before validation")
    print("   - Passes with_proto to underlying _predict/_generate/_stream methods")
    print("")
    print("2. Updated core methods to support with_proto:")
    print("   - _predict() now returns (result, proto) when with_proto=True")
    print("   - _async_predict() has same functionality for async calls")  
    print("   - _generate() returns (result, proto) for each yielded item")
    print("   - _async_generate() has same functionality for async streaming")
    print("   - _stream() returns (result, proto) for each streamed item")
    print("   - _async_stream() has same functionality for async streaming")
    print("")
    print("3. Maintains backward compatibility:")
    print("   - with_proto defaults to False")
    print("   - Existing behavior unchanged when with_proto=False")
    print("   - No changes to method signatures or existing APIs")


if __name__ == '__main__':
    demonstrate_with_proto_functionality()
    show_implementation_details()
    print("\n✅ Feature implementation complete!")