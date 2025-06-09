"""Templates for initializing pipeline step projects."""

from clarifai.versions import CLIENT_VERSION


def get_config_template():
    """Get the config.yaml template for pipeline steps."""
    return """pipeline_step:
  id: "text-classifier-train-upload-step"  # TODO: please fill in
  user_id: "your_user_id"  # TODO: please fill in
  app_id: "your_app_id"  # TODO: please fill in

pipeline_step_input_params:
  - name: param_a
  - name: param_b
    default: "default-param-b-value"
    description: "param_b is the second parameter of the pipeline step"
    accepted_values:  # list of accepted values for param_b
      - "param_b_allowed_value1"
      - "param_b_allowed_value2"
      - "param_b_allowed_value3"

build_info:
  python_version: "3.12"

pipeline_step_compute_info:
  cpu_limit: "500m"
  cpu_memory: "500Mi"
  num_accelerators: 0
"""


def get_pipeline_step_template():
    """Get the pipeline_step.py template."""
    return '''"""
This is a template for pipeline step implementation.
"""

class PipelineStepProcessor:
    """Template pipeline step processor class."""

    def __init__(self):
        """Initialize the pipeline step processor."""
        pass

    def load_pipeline_step(self):
        """Load any necessary resources for the pipeline step."""
        # TODO: Implement loading logic here
        pass

    def process(self, param_a, param_b="default-param-b-value"):
        """
        Process the pipeline step with given parameters.

        Args:
            param_a: First parameter
            param_b: Second parameter with default value

        Returns:
            Processing result
        """
        # TODO: Implement your pipeline step logic here
        print(f"Processing with param_a: {param_a}, param_b: {param_b}")

        # Example processing logic
        result = {
            "status": "success",
            "param_a": param_a,
            "param_b": param_b,
            "processed": True
        }

        return result
'''


def get_requirements_template():
    """Get the requirements.txt template."""
    return f'''clarifai=={CLIENT_VERSION}
# Add your pipeline step dependencies here
# Example:
# torch>=1.9.0
# transformers>=4.20.0
'''
