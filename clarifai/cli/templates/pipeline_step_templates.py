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
    default: "param_b_allowed_value1"
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
    return '''import argparse

import clarifai
from clarifai.utils.logging import logger

def main():
    parser = argparse.ArgumentParser(description='Concatenate two strings.')
    parser.add_argument('--param_a', type=str, required=True, help='First string to concatenate')
    parser.add_argument('--param_b', type=str, required=True, help='Second string to concatenate')

    args = parser.parse_args()

    logger.info(clarifai.__version__)

    logger.info(f"Concatenation Output: {args.param_a + args.param_b}")


if __name__ == "__main__":
    main()
'''


def get_requirements_template():
    """Get the requirements.txt template."""
    return f'''clarifai=={CLIENT_VERSION}
# Add your pipeline step dependencies here
# Example:
# torch>=1.9.0
# transformers>=4.20.0
'''
