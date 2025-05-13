import shutil
from pathlib import Path

import pytest
import yaml

from clarifai.runners.models.model_builder import ModelBuilder


@pytest.fixture
def my_tmp_path(tmp_path):
    return tmp_path


@pytest.mark.parametrize("num_threads", [-1, 0, 3, 1.5, "a", None])
def test_num_threads(my_tmp_path, num_threads, monkeypatch):
    """
    Clone dummy_runner_models with different num_threads settings for testing
    """
    tests_dir = Path(__file__).parent.resolve()
    original_dummy_path = tests_dir / "dummy_runner_models"
    if not original_dummy_path.exists():
        # Adjust or raise an error if you cannot locate the dummy_runner_models folder
        raise FileNotFoundError(
            f"Could not find dummy_runner_models at {original_dummy_path}. "
            "Adjust path or ensure it exists."
        )

    # Copy the entire folder to tmp_path
    target_folder = my_tmp_path / "dummy_runner_models"
    shutil.copytree(original_dummy_path, target_folder)

    # Update the config.yaml to override the app_id with the ephemeral one
    config_yaml_path = target_folder / "config.yaml"
    with config_yaml_path.open("r") as f:
        config = yaml.safe_load(f)

    monkeypatch.delenv("CLARIFAI_NUM_THREADS", raising=False)
    if num_threads is not None:
        config["num_threads"] = num_threads

    # Rewrite config.yaml
    with config_yaml_path.open("w") as f:
        yaml.dump(config, f, sort_keys=False)

    # no num_threads
    if num_threads is None:
        # default is 16
        builder = ModelBuilder(target_folder, validate_api_ids=False)
        assert builder.config.get("num_threads") == 16
        # set by env var if unset in config.yaml
        monkeypatch.setenv("CLARIFAI_NUM_THREADS", "4")
        builder = ModelBuilder(target_folder, validate_api_ids=False)
        assert builder.config.get("num_threads") == 4

    elif num_threads == 3:
        builder = ModelBuilder(target_folder, validate_api_ids=False)
        assert builder.config.get("num_threads") == num_threads

        # set by env var if unset in config.yaml
        monkeypatch.setenv("CLARIFAI_NUM_THREADS", "14")
        builder = ModelBuilder(target_folder, validate_api_ids=False)
        assert builder.config.get("num_threads") == num_threads

    elif num_threads in [-1, 0, "a", 1.5]:
        with pytest.raises(AssertionError):
            builder = ModelBuilder(target_folder, validate_api_ids=False)
