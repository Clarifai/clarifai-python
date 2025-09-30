"""Tests for requirements.txt validation."""

from pathlib import Path


def test_requirements_loosened_constraints():
    """Test that requirements.txt has loosened constraints for specified packages.

    This test ensures that PR #787 changes are applied correctly,
    loosening version constraints from == to >= for specific packages.
    """
    # Get the requirements.txt path
    repo_root = Path(__file__).parent.parent
    requirements_path = repo_root / "requirements.txt"

    assert requirements_path.exists(), "requirements.txt should exist"

    # Read requirements
    with open(requirements_path, 'r') as f:
        content = f.read()

    # Check that specific packages use >= instead of ==
    loosened_packages = {
        'uv': '0.7.12',
        'ruff': '0.11.4',
        'psutil': '7.0.0',
        'pydantic_core': '2.33.2',
        'packaging': '25.0',
    }

    for package, version in loosened_packages.items():
        # Should have >= constraint
        expected_constraint = f"{package}>={version}"
        assert expected_constraint in content, (
            f"Expected {expected_constraint} in requirements.txt"
        )

        # Should NOT have == constraint
        forbidden_constraint = f"{package}=={version}"
        assert forbidden_constraint not in content, (
            f"Should not have exact constraint {forbidden_constraint}"
        )

    # Check that schema is still pinned (as mentioned in PR description)
    assert "schema==0.7.5" in content, "schema should remain pinned to ==0.7.5"


def test_requirements_file_format():
    """Test that requirements.txt is properly formatted."""
    repo_root = Path(__file__).parent.parent
    requirements_path = repo_root / "requirements.txt"

    with open(requirements_path, 'r') as f:
        lines = f.readlines()

    # Check that there are no duplicate blank lines at the end
    non_empty_lines = [line for line in lines if line.strip()]
    assert len(lines) - len(non_empty_lines) <= 1, "Should have at most one trailing newline"

    # Check that all lines are properly formatted (no leading/trailing whitespace issues)
    for i, line in enumerate(lines):
        if line.strip():  # Skip empty lines
            assert not line.startswith(' '), f"Line {i + 1} should not start with space: {line!r}"
            assert not line.endswith(' \n'), (
                f"Line {i + 1} should not have trailing space: {line!r}"
            )
