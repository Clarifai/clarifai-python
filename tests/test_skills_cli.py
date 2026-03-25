"""Tests for clarifai skills CLI and utilities."""

import shutil
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from clarifai.cli.base import cli
from clarifai.utils.skills import (
    _copy_skills_from_local,
    _is_safe_tar_member,
    _link_or_copy,
    detect_agents,
    install_skills,
    remove_skills,
    resolve_agents,
)


@pytest.fixture
def tmp_dir():
    """Create a temp directory for tests."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def mock_skills_repo(tmp_dir):
    """Create a mock skills repo structure."""
    repo = tmp_dir / "mock-skills"
    skills_dir = repo / ".github" / "skills"

    for skill in ["clarifai-cli", "clarifai-inference"]:
        skill_dir = skills_dir / skill
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill}\ndescription: Test skill\n---\n# {skill}\n"
        )
        refs = skill_dir / "references"
        refs.mkdir()
        (refs / "ref.md").write_text("# Reference\n")

    (repo / "AGENTS.md").write_text("# Agents\n")
    return repo


# --- _is_safe_tar_member tests ---


class TestSafeTarMember:
    def test_rejects_absolute_path(self):
        member = tarfile.TarInfo(name="/etc/passwd")
        member.type = tarfile.REGTYPE
        assert not _is_safe_tar_member(member, "/tmp/dest")

    def test_rejects_parent_traversal(self):
        member = tarfile.TarInfo(name="foo/../../etc/passwd")
        member.type = tarfile.REGTYPE
        assert not _is_safe_tar_member(member, "/tmp/dest")

    def test_rejects_symlink(self):
        member = tarfile.TarInfo(name="foo/link")
        member.type = tarfile.SYMTYPE
        assert not _is_safe_tar_member(member, "/tmp/dest")

    def test_rejects_hardlink(self):
        member = tarfile.TarInfo(name="foo/link")
        member.type = tarfile.LNKTYPE
        assert not _is_safe_tar_member(member, "/tmp/dest")

    def test_accepts_safe_path(self):
        member = tarfile.TarInfo(name="clarifai-cli/SKILL.md")
        member.type = tarfile.REGTYPE
        assert _is_safe_tar_member(member, "/tmp/dest")

    def test_accepts_directory(self):
        member = tarfile.TarInfo(name="clarifai-cli/references/")
        member.type = tarfile.DIRTYPE
        assert _is_safe_tar_member(member, "/tmp/dest")

    def test_rejects_fifo(self):
        member = tarfile.TarInfo(name="foo/fifo")
        member.type = tarfile.FIFOTYPE
        assert not _is_safe_tar_member(member, "/tmp/dest")

    def test_rejects_device(self):
        member = tarfile.TarInfo(name="foo/dev")
        member.type = tarfile.CHRTYPE
        assert not _is_safe_tar_member(member, "/tmp/dest")


# --- _link_or_copy tests ---


class TestLinkOrCopy:
    def test_creates_symlink(self, tmp_dir):
        source = tmp_dir / "source"
        source.mkdir()
        (source / "file.txt").write_text("hello")
        target = tmp_dir / "target"

        _link_or_copy(source, target)

        assert target.is_symlink()
        assert (target / "file.txt").read_text() == "hello"

    def test_replaces_existing_symlink(self, tmp_dir):
        source1 = tmp_dir / "source1"
        source1.mkdir()
        source2 = tmp_dir / "source2"
        source2.mkdir()
        (source2 / "new.txt").write_text("new")
        target = tmp_dir / "target"
        target.symlink_to(source1)

        _link_or_copy(source2, target)

        assert target.is_symlink()
        assert (target / "new.txt").read_text() == "new"

    def test_replaces_existing_dir(self, tmp_dir):
        source = tmp_dir / "source"
        source.mkdir()
        (source / "file.txt").write_text("hello")
        target = tmp_dir / "target"
        target.mkdir()
        (target / "old.txt").write_text("old")

        _link_or_copy(source, target)

        assert target.is_symlink()
        assert not (target / "old.txt").exists()

    def test_replaces_existing_file(self, tmp_dir):
        source = tmp_dir / "source"
        source.mkdir()
        target = tmp_dir / "target"
        target.write_text("i am a file")

        _link_or_copy(source, target)

        assert target.is_symlink()

    def test_falls_back_to_copy(self, tmp_dir):
        source = tmp_dir / "source"
        source.mkdir()
        (source / "file.txt").write_text("hello")
        target = tmp_dir / "target"

        with patch("clarifai.utils.skills.Path.symlink_to", side_effect=OSError("no symlinks")):
            _link_or_copy(source, target)

        assert target.is_dir()
        assert not target.is_symlink()
        assert (target / "file.txt").read_text() == "hello"


# --- _copy_skills_from_local tests ---


class TestCopySkillsFromLocal:
    def test_copies_all_skills(self, mock_skills_repo, tmp_dir):
        dest = tmp_dir / "dest"
        dest.mkdir()

        result = _copy_skills_from_local(mock_skills_repo, dest, None)

        assert sorted(result) == ["clarifai-cli", "clarifai-inference"]
        assert (dest / "clarifai-cli" / "SKILL.md").exists()
        assert (dest / "clarifai-inference" / "SKILL.md").exists()
        assert (dest / "AGENTS.md").exists()

    def test_copies_specific_skill(self, mock_skills_repo, tmp_dir):
        dest = tmp_dir / "dest"
        dest.mkdir()

        result = _copy_skills_from_local(mock_skills_repo, dest, ["clarifai-cli"])

        assert result == ["clarifai-cli"]
        assert (dest / "clarifai-cli" / "SKILL.md").exists()
        assert not (dest / "clarifai-inference").exists()

    def test_raises_on_missing_skills_dir(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            _copy_skills_from_local(tmp_dir / "nonexistent", tmp_dir, None)


# --- resolve_agents tests ---


class TestResolveAgents:
    def test_explicit_claude(self):
        assert resolve_agents(claude=True, codex=False, cursor=False, all_agents=False) == [
            "claude"
        ]

    def test_all_agents(self):
        result = resolve_agents(claude=False, codex=False, cursor=False, all_agents=True)
        assert set(result) == {"claude", "codex", "cursor"}

    def test_multiple(self):
        result = resolve_agents(claude=True, codex=True, cursor=False, all_agents=False)
        assert result == ["claude", "codex"]


# --- install_skills + remove_skills integration ---


class TestInstallRemove:
    def test_install_and_remove(self, mock_skills_repo, tmp_dir):
        central = tmp_dir / "central"
        agent_dir = tmp_dir / "agent"

        with (
            patch("clarifai.utils.skills.CENTRAL_DIR", {"global": central}),
            patch(
                "clarifai.utils.skills.AGENT_DIRS",
                {"claude": {"global": agent_dir}},
            ),
        ):
            downloaded, linked = install_skills(
                skill_ids=None,
                agents=["claude"],
                global_=True,
                force=True,
                source=str(mock_skills_repo),
            )

            assert sorted(downloaded) == ["clarifai-cli", "clarifai-inference"]
            assert linked == ["claude"]
            assert (agent_dir / "clarifai-cli").exists()

            removed = remove_skills(skill_ids=["clarifai-cli"], agents=["claude"], global_=True)

            assert removed == ["clarifai-cli"]
            assert not (agent_dir / "clarifai-cli").exists()
            assert (agent_dir / "clarifai-inference").exists()

    def test_install_skips_when_not_forced(self, mock_skills_repo, tmp_dir):
        central = tmp_dir / "central"
        agent_dir = tmp_dir / "agent"

        with (
            patch("clarifai.utils.skills.CENTRAL_DIR", {"global": central}),
            patch(
                "clarifai.utils.skills.AGENT_DIRS",
                {"claude": {"global": agent_dir}},
            ),
        ):
            # First install
            install_skills(
                skill_ids=None,
                agents=["claude"],
                global_=True,
                force=True,
                source=str(mock_skills_repo),
            )

            # Modify a file
            (central / "clarifai-cli" / "marker.txt").write_text("modified")

            # Second install without force — should skip download, just re-link
            install_skills(
                skill_ids=None,
                agents=["claude"],
                global_=True,
                force=False,
                source=str(mock_skills_repo),
            )

            # marker.txt should still exist (wasn't re-downloaded)
            assert (central / "clarifai-cli" / "marker.txt").exists()


# --- CLI tests ---


class TestCLI:
    def test_skills_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["skills", "--help"])
        assert result.exit_code == 0
        assert "install" in result.output
        assert "list" in result.output
        assert "update" in result.output
        assert "remove" in result.output

    def test_install_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["skills", "install", "--help"])
        assert result.exit_code == 0
        assert "--claude" in result.output
        assert "--global" in result.output
        assert "--source" in result.output

    def test_remove_requires_args(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["skills", "remove"])
        assert result.exit_code != 0
        assert "Specify skill names or use --all" in result.output

    @patch("clarifai.utils.skills.list_remote_skills")
    def test_list_remote_empty(self, mock_remote):
        mock_remote.return_value = []
        runner = CliRunner()
        result = runner.invoke(cli, ["skills", "list", "--remote"])
        assert result.exit_code == 0
        assert "no skills found" in result.output

    @patch("clarifai.utils.skills.list_remote_skills")
    def test_list_remote(self, mock_remote):
        mock_remote.return_value = [
            {"id": "clarifai-cli", "description": "CLI operations"},
        ]
        runner = CliRunner()
        result = runner.invoke(cli, ["skills", "list", "--remote"])
        assert result.exit_code == 0
        assert "clarifai-cli" in result.output
        assert "CLI operations" in result.output

    def test_sk_alias(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["sk", "--help"])
        assert result.exit_code == 0
        assert "install" in result.output

    @patch("clarifai.utils.skills.install_skills")
    @patch("clarifai.utils.skills.list_remote_skills")
    def test_install_happy_path(self, mock_remote, mock_install):
        mock_remote.return_value = [
            {"id": "clarifai-cli", "description": "CLI"},
        ]
        mock_install.return_value = (["clarifai-cli"], ["claude"])
        runner = CliRunner()
        result = runner.invoke(cli, ["skills", "install", "--claude", "clarifai-cli"])
        assert result.exit_code == 0
        assert "Installed 1 skills" in result.output

    @patch("clarifai.utils.skills.remove_skills")
    def test_remove_happy_path(self, mock_remove):
        mock_remove.return_value = ["clarifai-cli"]
        runner = CliRunner()
        result = runner.invoke(cli, ["skills", "remove", "clarifai-cli", "--claude"])
        assert result.exit_code == 0
        assert "Removed 1 skills" in result.output


# --- detect_agents tests ---


class TestDetectAgents:
    def test_detects_from_global_dirs(self, tmp_dir):
        with patch(
            "clarifai.utils.skills.AGENT_DIRS",
            {
                "claude": {
                    "global": tmp_dir / ".claude" / "skills",
                    "local": tmp_dir / "local" / ".claude" / "skills",
                },
                "codex": {
                    "global": tmp_dir / ".codex" / "skills",
                    "local": tmp_dir / "local" / ".codex" / "skills",
                },
            },
        ):
            (tmp_dir / ".claude").mkdir()
            # .codex parent doesn't exist, so only claude should be detected
            result = detect_agents()
            assert result == ["claude"]

    def test_detects_from_local_dirs(self, tmp_dir):
        local_claude = tmp_dir / ".claude"
        local_claude.mkdir()
        with patch(
            "clarifai.utils.skills.AGENT_DIRS",
            {
                "claude": {
                    "global": Path("/nonexistent/.claude/skills"),
                    "local": local_claude / "skills",
                },
            },
        ):
            result = detect_agents()
            assert result == ["claude"]

    def test_fallback_to_claude(self):
        with patch(
            "clarifai.utils.skills.AGENT_DIRS",
            {
                "claude": {
                    "global": Path("/nonexistent/.claude/skills"),
                    "local": Path("/nonexistent/.claude/skills"),
                },
            },
        ):
            result = detect_agents()
            assert result == ["claude"]
