#!/usr/bin/env python3
"""Example pipeline defined with the Clarifai pipeline DSL.

This example demonstrates:
- code-first step definitions with `@step`
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
"""

import argparse

from clarifai.runners.pipelines import ComputeConfig, Pipeline, step


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


@step(
    id="summarize",
    requirements=["openai>=1.0"],
    compute=ComputeConfig(cpu_limit="1000m", cpu_memory="1Gi", num_accelerators=1),
    secrets={"OPENAI_API_KEY": "users/demo-user/secrets/openai-key"},
)
def summarize(input_text: str) -> str:
    """Placeholder summary logic for the example pipeline."""
    return f"Summary of: {input_text[:80]}"


@step(id="classify-sentiment", requirements=["textblob>=0.18.0"])
def classify_sentiment(input_text: str) -> str:
    """A lightweight branch that simulates sentiment classification."""
    positive_words = {"great", "good", "love", "excellent", "happy"}
    tokens = set(input_text.split())
    return "positive" if tokens & positive_words else "neutral"


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