"""Core logic for installing and managing Clarifai agent skills."""

import io
import json
import os
import shutil
import tarfile
from pathlib import Path

import requests

SKILLS_REPO = "Clarifai/skills"
SKILLS_BRANCH = "main"
MARKETPLACE_URL = (
    f"https://raw.githubusercontent.com/{SKILLS_REPO}/{SKILLS_BRANCH}/marketplace.json"
)
TARBALL_URL = f"https://api.github.com/repos/{SKILLS_REPO}/tarball/{SKILLS_BRANCH}"
COMMIT_API_URL = f"https://api.github.com/repos/{SKILLS_REPO}/commits/{SKILLS_BRANCH}"


def _gh_headers() -> dict:
    """Get GitHub auth headers if token available."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        try:
            import subprocess

            result = subprocess.run(
                ["gh", "auth", "token"], capture_output=True, text=True, timeout=5, check=False
            )
            if result.returncode == 0:
                token = result.stdout.strip()
        except Exception:
            pass
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


VERSION_FILE = ".clarifai-skills-version"

AGENT_DIRS = {
    "claude": {"global": Path.home() / ".claude" / "skills", "local": Path(".claude") / "skills"},
    "codex": {"global": Path.home() / ".codex" / "skills", "local": Path(".codex") / "skills"},
    "cursor": {"global": Path.home() / ".cursor" / "skills", "local": Path(".cursor") / "skills"},
    "copilot": {
        "global": Path.home() / ".github-copilot" / "skills",
        "local": Path(".github-copilot") / "skills",
    },
    "gemini": {"global": Path.home() / ".gemini" / "skills", "local": Path(".gemini") / "skills"},
}

CENTRAL_DIR = {
    "global": Path.home() / ".agents" / "skills",
    "local": Path(".agents") / "skills",
}


def fetch_marketplace() -> dict:
    """Fetch the skills marketplace index from GitHub."""
    headers = _gh_headers()
    resp = requests.get(MARKETPLACE_URL, timeout=15, headers=headers)
    if resp.status_code == 404:
        api_url = f"https://api.github.com/repos/{SKILLS_REPO}/contents/marketplace.json?ref={SKILLS_BRANCH}"
        resp = requests.get(api_url, timeout=15, headers=headers)
        resp.raise_for_status()
        import base64

        content = base64.b64decode(resp.json()["content"])
        return json.loads(content)
    resp.raise_for_status()
    return resp.json()


def list_remote_skills() -> list[dict]:
    """List all available skills from the remote registry."""
    marketplace = fetch_marketplace()
    return marketplace.get("skills", [])


def detect_agents() -> list[str]:
    """Auto-detect which agent directories exist on the system."""
    detected = []
    for agent, dirs in AGENT_DIRS.items():
        global_parent = dirs["global"].parent
        local_parent = dirs["local"].parent
        if global_parent.exists() or local_parent.exists():
            detected.append(agent)
    return detected or ["claude"]


def resolve_agents(
    claude: bool = False,
    codex: bool = False,
    cursor: bool = False,
    copilot: bool = False,
    gemini: bool = False,
    all_agents: bool = False,
) -> list[str]:
    """Resolve which agents to target based on CLI flags."""
    if all_agents:
        return list(AGENT_DIRS.keys())
    flag_map = {
        "claude": claude,
        "codex": codex,
        "cursor": cursor,
        "copilot": copilot,
        "gemini": gemini,
    }
    agents = [name for name, enabled in flag_map.items() if enabled]
    if not agents:
        agents = detect_agents()
    return agents


def _find_local_skills_repo() -> Path | None:
    """Try to find a local clone of the skills repo."""
    candidates = [
        Path.home() / "work" / "skills",
        Path.home() / "skills",
        Path.home() / "clarifai" / "skills",
        Path.cwd() / "skills",
        Path.cwd().parent / "skills",
    ]
    for p in candidates:
        if (p / ".github" / "skills").is_dir():
            return p
    return None


def _is_safe_tar_member(member: tarfile.TarInfo, dest: str) -> bool:
    """Reject tar members with path traversal, absolute paths, links, or special files."""
    # Only allow regular files and directories
    if not (member.isfile() or member.isdir()):
        return False
    # Reject absolute paths
    if member.name.startswith("/") or member.name.startswith("\\"):
        return False
    # Reject parent traversal
    if ".." in member.name.split("/"):
        return False
    # Verify resolved path stays within dest
    resolved = os.path.realpath(os.path.join(dest, member.name))
    if not resolved.startswith(os.path.realpath(dest) + os.sep) and resolved != os.path.realpath(
        dest
    ):
        return False
    return True


def download_skills(
    dest: Path, skill_ids: list[str] | None = None, source: str | None = None
) -> list[str]:
    """Download skills to dest directory.

    Priority: explicit --source > GitHub download > auto-detected local clone.
    Returns list of skill IDs that were downloaded.
    """
    if source and Path(source).exists():
        return _copy_skills_from_local(Path(source), dest, skill_ids)

    try:
        return _download_skills_from_github(dest, skill_ids)
    except Exception as gh_err:
        local = _find_local_skills_repo()
        if local:
            return _copy_skills_from_local(local, dest, skill_ids)
        raise RuntimeError(
            f"GitHub download failed ({gh_err}) and no local skills repo found.\n"
            f"Options:\n"
            f"  1. Set GITHUB_TOKEN or run 'gh auth login' for private repo access\n"
            f"  2. Use --source /path/to/skills (local clone)\n"
            f"  3. Wait for the repo to be made public"
        ) from gh_err


def _copy_skills_from_local(source: Path, dest: Path, skill_ids: list[str] | None) -> list[str]:
    """Copy skills from a local skills repo clone."""
    skills_dir = source / ".github" / "skills"
    if not skills_dir.exists():
        raise FileNotFoundError(f"No .github/skills/ found in {source}")

    downloaded = []
    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        if skill_ids and skill_dir.name not in skill_ids:
            continue
        target = dest / skill_dir.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(skill_dir, target)
        downloaded.append(skill_dir.name)

    agents_md = source / "AGENTS.md"
    if agents_md.exists():
        shutil.copy2(agents_md, dest / "AGENTS.md")

    return downloaded


def _download_skills_from_github(dest: Path, skill_ids: list[str] | None) -> list[str]:
    """Download skills from GitHub tarball with safe extraction."""
    headers = _gh_headers()
    # Handle redirects manually to preserve Authorization header across hosts
    # (GitHub redirects api.github.com → codeload.github.com, and requests drops auth)
    resp = requests.get(TARBALL_URL, timeout=60, headers=headers, allow_redirects=False)
    if resp.status_code in (301, 302, 307, 308):
        redirect_url = resp.headers["Location"]
        resp = requests.get(redirect_url, timeout=60, headers=headers)
    resp.raise_for_status()

    downloaded = []
    dest_str = str(dest)
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        prefix = None
        for member in tar.getmembers():
            if prefix is None:
                prefix = member.name.split("/")[0]

            skills_prefix = f"{prefix}/.github/skills/"
            if not member.name.startswith(skills_prefix):
                if member.name == f"{prefix}/AGENTS.md" and member.isfile():
                    member.name = "AGENTS.md"
                    if _is_safe_tar_member(member, dest_str):
                        tar.extract(member, dest)
                continue

            relative = member.name[len(skills_prefix) :]
            if not relative:
                continue

            skill_id = relative.split("/")[0]
            if skill_ids and skill_id not in skill_ids:
                continue

            member.name = relative
            if not _is_safe_tar_member(member, dest_str):
                continue
            tar.extract(member, dest)

            if skill_id not in downloaded:
                downloaded.append(skill_id)

    return downloaded


def _write_version(dest: Path):
    """Write current commit SHA to version file for update tracking."""
    try:
        headers = _gh_headers()
        resp = requests.get(COMMIT_API_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        sha = resp.json()["sha"]
    except Exception:
        sha = "unknown"
    (dest / VERSION_FILE).write_text(sha)


def _read_version(dest: Path) -> str:
    """Read the locally stored version."""
    version_file = dest / VERSION_FILE
    if version_file.exists():
        return version_file.read_text().strip()
    return ""


def _link_or_copy(source: Path, target: Path):
    """Create a symlink from target -> source, falling back to copy on failure (e.g. Windows)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    # Remove any existing target (file, symlink, or directory)
    if target.is_symlink() or target.exists():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    try:
        target.symlink_to(source)
    except (OSError, NotImplementedError):
        # Fallback: copy instead of symlink (Windows without Developer Mode, etc.)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def install_skills(
    skill_ids: list[str] | None,
    agents: list[str],
    global_: bool = True,
    force: bool = False,
    source: str | None = None,
) -> tuple[list[str], list[str]]:
    """Install skills: download to central dir, symlink to agent dirs.

    Returns (downloaded_skills, linked_agents).
    """
    scope = "global" if global_ else "local"
    central = CENTRAL_DIR[scope]

    # If not forced, check if already installed and skip download
    if not force and central.exists() and skill_ids is None:
        existing = [
            d.name for d in central.iterdir() if d.is_dir() and d.name.startswith("clarifai-")
        ]
        if existing:
            # Already installed — just re-link to requested agents without touching version metadata
            downloaded = existing
            linked_agents = _link_skills_to_agents(central, downloaded, agents, scope)
            return downloaded, linked_agents

    central.mkdir(parents=True, exist_ok=True)

    # When force-updating all skills, clean existing clarifai-* dirs to remove stale files
    if force and skill_ids is None and central.exists():
        for d in central.iterdir():
            if d.is_dir() and d.name.startswith("clarifai-"):
                shutil.rmtree(d)
        for f in [central / "AGENTS.md", central / VERSION_FILE]:
            if f.exists():
                f.unlink()

    downloaded = download_skills(central, skill_ids, source=source)
    _write_version(central)
    linked_agents = _link_skills_to_agents(central, downloaded, agents, scope)
    return downloaded, linked_agents


def _link_skills_to_agents(
    central: Path, skill_ids: list[str], agents: list[str], scope: str
) -> list[str]:
    """Create symlinks (or copies) from agent dirs to central skills."""
    linked_agents = []
    for agent in agents:
        agent_dir = AGENT_DIRS[agent][scope]
        agent_dir.mkdir(parents=True, exist_ok=True)

        for skill_id in skill_ids:
            source_dir = central / skill_id
            target = agent_dir / skill_id
            _link_or_copy(source_dir.resolve(), target)

        agents_md = central / "AGENTS.md"
        if agents_md.exists():
            _link_or_copy(agents_md.resolve(), agent_dir / "AGENTS.md")

        linked_agents.append(agent)
    return linked_agents


def list_installed_skills(agents: list[str], global_: bool = True) -> dict[str, list[str]]:
    """List installed Clarifai skills per agent.

    Returns {agent_name: [skill_ids]}.
    """
    scope = "global" if global_ else "local"
    result = {}
    for agent in agents:
        agent_dir = AGENT_DIRS[agent][scope]
        if not agent_dir.exists():
            result[agent] = []
            continue
        skills = sorted(
            d.name
            for d in agent_dir.iterdir()
            if (d.is_dir() or d.is_symlink()) and d.name.startswith("clarifai-")
        )
        result[agent] = skills
    return result


def check_for_updates(global_: bool = True) -> bool:
    """Check if remote version differs from local."""
    scope = "global" if global_ else "local"
    central = CENTRAL_DIR[scope]
    local_version = _read_version(central)
    if not local_version or local_version == "unknown":
        return True
    try:
        headers = _gh_headers()
        resp = requests.get(COMMIT_API_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        remote_sha = resp.json()["sha"]
        return remote_sha != local_version
    except Exception:
        return True


def remove_skills(
    skill_ids: list[str] | None,
    agents: list[str],
    global_: bool = True,
    remove_all: bool = False,
) -> list[str]:
    """Remove skills from agent dirs and central dir.

    Returns list of removed skill IDs.
    """
    scope = "global" if global_ else "local"
    central = CENTRAL_DIR[scope]

    if remove_all:
        # Collect skill IDs from central dir and all targeted agent dirs
        # so --all works even if central is missing/out-of-sync
        collected: set[str] = set()
        if central.exists():
            collected.update(
                d.name for d in central.iterdir() if d.is_dir() and d.name.startswith("clarifai-")
            )
        for agent in agents:
            agent_dir = AGENT_DIRS[agent][scope]
            if agent_dir.exists():
                collected.update(
                    d.name
                    for d in agent_dir.iterdir()
                    if (d.is_dir() or d.is_symlink()) and d.name.startswith("clarifai-")
                )
        skill_ids = sorted(collected)

    if not skill_ids:
        return []

    removed = []
    for skill_id in skill_ids:
        for agent in agents:
            agent_dir = AGENT_DIRS[agent][scope]
            target = agent_dir / skill_id
            if target.is_symlink() or target.exists():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()

        source = central / skill_id
        if source.exists():
            shutil.rmtree(source)

        removed.append(skill_id)

    # If all skills removed, clean up AGENTS.md and version file
    remaining = (
        [d for d in central.iterdir() if d.is_dir() and d.name.startswith("clarifai-")]
        if central.exists()
        else []
    )
    if not remaining:
        for f in [central / "AGENTS.md", central / VERSION_FILE]:
            if f.exists():
                f.unlink()
        for agent in agents:
            agent_dir = AGENT_DIRS[agent][scope]
            target = agent_dir / "AGENTS.md"
            if target.is_symlink() or target.exists():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()

    return removed
