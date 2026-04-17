#!/usr/bin/env python3
"""Example pipeline defined with the Clarifai pipeline DSL.

This example demonstrates:
- code-first step definitions with `@step`
- mixed local and pre-existing remote steps
- helper-function extraction within a single step
- diamond DAG composition using `>>`
- step-level secret configuration
- generation of the YAML/config-based pipeline bundle

Usage:

  python examples/pipeline_dsl_text_pipeline.py --generate ./generated-pipeline

After generation, the resulting directory can be uploaded with either:

  clarifai pipeline upload ./generated-pipeline

or directly from this file:

  clarifai pipeline upload examples/pipeline_dsl_text_pipeline.py

Only locally managed steps are generated into the bundle. Pre-existing steps
declared with `step_ref(...)` are emitted as versioned `templateRef`s and are
not included in `step_directories`.
"""

import argparse

from clarifai.runners.pipelines import ComputeConfig, Pipeline, step, step_ref


def normalize_text(value: str) -> str:
    """Small helper intentionally kept outside the step for codegen extraction."""
    return " ".join(value.strip().split())


@step(
    id="prepare-text",
    requirements=["transformers>=4.0"],
    compute=ComputeConfig(cpu_limit="500m", cpu_memory="500Mi"),
)
def prepare_text(input_text: str) -> str:
    """Normalize text before downstream processing."""
    cleaned = normalize_text(input_text)
    return cleaned.lower()


summarize = step_ref.from_url(
    "https://api.clarifai.com/v2/users/demo-user/apps/shared-app/pipeline_steps/summarize/versions/summary-v1",
    secrets={"OPENAI_API_KEY": "users/demo-user/secrets/openai-key"},
)


classify_sentiment = step_ref(
    id="classify-sentiment",
    user_id="demo-user",
    app_id="shared-app",
    version_id="sentiment-v3",
)


@step(id="assemble-report")
def assemble_report(summary: str, sentiment: str) -> str:
    """Join branch outputs into a single final payload."""
    return f"Summary: {summary}\nSentiment: {sentiment}"


with Pipeline(id="text-analysis-pipeline", user_id="demo-user", app_id="demo-app") as pipeline:
    raw_text = pipeline.input("input_text")

    prepared = prepare_text(input_text=raw_text)
    summary = summarize(input_text=prepared.output())
    sentiment = classify_sentiment(input_text=prepared.output())
    report = assemble_report(
        summary=summary.output(),
        sentiment=sentiment.output(),
    )

    prepared >> [summary, sentiment] >> report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a code-first Clarifai pipeline bundle.")
    parser.add_argument(
        "--generate",
        default="./generated-pipeline",
        help="Directory where the generated pipeline bundle should be written.",
    )
    args = parser.parse_args()

    config_path = pipeline.generate(args.generate)
    print(f"Generated pipeline bundle at: {config_path}")


if __name__ == "__main__":
    main()