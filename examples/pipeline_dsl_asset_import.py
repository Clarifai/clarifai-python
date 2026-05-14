from clarifai.runners.pipelines import ComputeInfo, Pipeline, step


def normalize_text(value: str, suffix: str) -> str:
    """Small helper intentionally kept outside the step for codegen extraction."""
    #    import importlib.util
    #    from pathlib import Path
    #
    #    module_path = Path(__file__).with_name("text_utils.py")
    #    spec = importlib.util.spec_from_file_location("text_utils", module_path)
    #    if spec is None or spec.loader is None:
    #        raise RuntimeError(f"Could not load helper module from {module_path}")
    #
    #    module = importlib.util.module_from_spec(spec)
    #    spec.loader.exec_module(module)
    import text_utils as module

    return module.concat(module.clean_text(value), suffix)


@step(
    id="prepare-text",
    requirements=["transformers>=4.0"],
    assets=["./text_utils.py"],
    compute=ComputeInfo(cpu_limit="500m", cpu_memory="500Mi"),
)
def prepare_text(input_text: str, suffix: str) -> str:
    """Normalize text before downstream processing."""
    cleaned = normalize_text(input_text, suffix)
    return cleaned


@step(id="print-text", compute=ComputeInfo(cpu_limit="500m", cpu_memory="500Mi"))
def print_text(text: str):
    print(text)


with Pipeline(id="dsl-demo-pipeline", user_id="peter-rizzi", app_id="pipeline-app") as pipeline:
    raw_text = pipeline.input("input_text")

    prepared = prepare_text(input_text=raw_text, suffix='-2')
    prepared
