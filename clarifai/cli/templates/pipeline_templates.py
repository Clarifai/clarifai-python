"""Templates for initializing pipeline projects."""

from clarifai.versions import CLIENT_VERSION


def get_pipeline_config_template(
    pipeline_id="hello-world-pipeline",
    user_id="your_user_id",
    app_id="your_app_id",
    step_names=None,
):
    """Get the config.yaml template for pipelines."""
    if step_names is None:
        step_names = ["stepA", "stepB"]

    step_directories = "\n".join(f"    - {step}" for step in step_names)

    # Generate step templates for orchestration
    step_templates = []
    for i, step_name in enumerate(step_names):
        step_templates.append(f"""          - - name: step-{i}
              templateRef:
                name: users/{user_id}/apps/{app_id}/pipeline_steps/{step_name}
                template: users/{user_id}/apps/{app_id}/pipeline_steps/{step_name}
              arguments:
                parameters:
                  - name: input_text
                    value: "{{{{workflow.parameters.input_text}}}}\"""")

    steps_yaml = "\n".join(step_templates)

    return f"""pipeline:
  id: "{pipeline_id}"
  user_id: "{user_id}"
  app_id: "{app_id}"
  step_directories:
{step_directories}
  orchestration_spec:
    argo_orchestration_spec: |
      apiVersion: argoproj.io/v1alpha1
      kind: Workflow
      spec:
        entrypoint: sequence
        arguments:
          parameters:
            - name: input_text
              value: "Input Text Here"
        templates:
        - name: sequence
          steps:
{steps_yaml}
  # Optional: Define secrets for pipeline steps
  # config:
  #   step_version_secrets:
  #     step-0:
  #       API_KEY: users/{user_id}/apps//secrets/my-api-key
  #       DB_PASSWORD: users/{user_id}/apps/secrets/db-secret
  #     step-1:
  #       EMAIL_TOKEN: users/{user_id}/apps/secrets/email-token
"""


def get_pipeline_step_config_template(step_id: str, user_id="your_user_id", app_id="your_app_id"):
    """Get the config.yaml template for a pipeline step."""
    return f"""pipeline_step:
  id: "{step_id}"
  user_id: "{user_id}"
  app_id: "{app_id}"

pipeline_step_input_params:
  - name: input_text
    description: "Text input for processing"

build_info:
  python_version: "3.12"
  # platform: "linux/amd64,linux/arm64"  # Optional: Specify target platform(s) for Docker image build

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

1. **Implement pipeline steps**: For each step directory:
   - Update `requirements.txt` with your dependencies
   - Implement your logic in `1/pipeline_step.py`

2. **Upload the pipeline**:
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
