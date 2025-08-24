"""Templates for initializing pipeline projects."""

from clarifai.versions import CLIENT_VERSION


def get_pipeline_config_template(pipeline_name=None, user_id=None, app_id=None):
    """Get the config.yaml template for pipelines."""
    # Use provided values or defaults
    pipeline_id = pipeline_name if pipeline_name else "hello-world-pipeline"
    user_id_value = user_id if user_id else "your_user_id"
    app_id_value = app_id if app_id else "your_app_id"

    # Generate TODO comments only if using default values
    pipeline_id_comment = "" if pipeline_name else "  # TODO: please fill in"
    user_id_comment = "" if user_id else "  # TODO: please fill in" 
    app_id_comment = "" if app_id else "  # TODO: please fill in"
    step_a_comment = "" if user_id and app_id else "  # TODO: please fill in"
    step_b_comment = "" if user_id and app_id else "  # TODO: please fill in"

    return f"""pipeline:
  id: "{pipeline_id}"{pipeline_id_comment}
  user_id: "{user_id_value}"{user_id_comment}
  app_id: "{app_id_value}"{app_id_comment}
  step_directories:
    - stepA
    - stepB
  orchestration_spec:
    argo_orchestration_spec: |
      apiVersion: argoproj.io/v1alpha1
      kind: Workflow
      spec:
        entrypoint: sequence
        templates:
        - name: sequence
          steps:
          - - name: step-a
              templateRef:
                name: users/{user_id_value}/apps/{app_id_value}/pipeline_steps/stepA{step_a_comment}
                template: users/{user_id_value}/apps/{app_id_value}/pipeline_steps/stepA{step_a_comment}
              arguments:
                parameters:
                  - name: input_text
                    value: "Input Text to Step A"
          - - name: step-b
              templateRef:
                name: users/{user_id_value}/apps/{app_id_value}/pipeline_steps/stepB{step_b_comment}
                template: users/{user_id_value}/apps/{app_id_value}/pipeline_steps/stepB{step_b_comment}
              arguments:
                parameters:
                  - name: input_text
                    value: "Input Text to Step B"
"""


def get_pipeline_step_config_template(step_id: str, user_id=None, app_id=None):
    """Get the config.yaml template for a pipeline step."""
    # Use provided values or defaults
    user_id_value = user_id if user_id else "your_user_id"
    app_id_value = app_id if app_id else "your_app_id"

    # Generate TODO comments only if using default values
    user_id_comment = "" if user_id else "  # TODO: please fill in"
    app_id_comment = "" if app_id else "  # TODO: please fill in"

    return f"""pipeline_step:
  id: "{step_id}"
  user_id: "{user_id_value}"{user_id_comment}
  app_id: "{app_id_value}"{app_id_comment}

pipeline_step_input_params:
  - name: input_text
    description: "Text input for processing"

build_info:
  python_version: "3.12"

pipeline_step_compute_info:
  cpu_limit: "500m"
  cpu_memory: "500Mi"
  num_accelerators: 0
"""


def get_pipeline_step_template(step_id: str):
    """Get the pipeline_step.py template for a pipeline step."""
    return f'''import argparse

import clarifai
from clarifai.utils.logging import logger

def main():
    parser = argparse.ArgumentParser(description='{step_id} processing step.')
    parser.add_argument('--input_text', type=str, required=True, help='Text input for processing')

    args = parser.parse_args()

    logger.info(clarifai.__version__)

    # TODO: Implement your pipeline step logic here
    logger.info(f"{step_id} processed: {{args.input_text}}")


if __name__ == "__main__":
    main()
'''


def get_pipeline_step_requirements_template():
    """Get the requirements.txt template for pipeline steps."""
    return f'''clarifai=={CLIENT_VERSION}
# Add your pipeline step dependencies here
# Example:
# torch>=1.9.0
# transformers>=4.20.0
'''


def get_readme_template():
    """Get the README.md template for the pipeline project."""
    return """# Pipeline Project

This project contains a Clarifai pipeline with associated pipeline steps.

## Structure

```
├── config.yaml          # Pipeline configuration
├── stepA/               # First pipeline step
│   ├── config.yaml     # Step A configuration
│   ├── requirements.txt # Step A dependencies
│   └── 1/
│       └── pipeline_step.py  # Step A implementation
├── stepB/               # Second pipeline step
│   ├── config.yaml     # Step B configuration
│   ├── requirements.txt # Step B dependencies
│   └── 1/
│       └── pipeline_step.py  # Step B implementation
└── README.md           # This file
```

## Getting Started

1. **Configure the pipeline**: Edit `config.yaml` and update the TODO fields:
   - Set your `user_id` and `app_id`
   - Update the pipeline `id`
   - Modify the Argo orchestration spec as needed

2. **Configure pipeline steps**: For each step directory (stepA, stepB):
   - Edit `config.yaml` and fill in the TODO fields
   - Update `requirements.txt` with your dependencies
   - Implement your logic in `1/pipeline_step.py`

3. **Upload the pipeline**:
   ```bash
   clarifai pipeline upload config.yaml
   ```

This will:
- Upload the pipeline steps from the `step_directories`
- Create the pipeline with proper orchestration
- Link all components together

## Pipeline Steps

### stepA
TODO: Describe what stepA does

### stepB
TODO: Describe what stepB does

## Customization

- Add more pipeline steps by creating new directories and adding them to `step_directories` in `config.yaml`
- Modify the Argo orchestration spec to change the execution flow
- Update compute resources in each step's `config.yaml` as needed
"""
