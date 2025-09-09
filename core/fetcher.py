from __future__ import annotations
import os, shutil, time
from dataclasses import dataclass
from typing import List
from git import Repo, GitCommandError


@dataclass
class CloneResult:
    repo_dir: str
    commit: str
    branch: str
    top_level: List[str]
    took_sec: float
    reused_existing: bool


def _ensure_workspace() -> str:
    """
    Ensure a stable local workspace directory exists (./workspace) and return its absolute path.
    Keeping clones here lets us reuse and fast-update repositories between runs.
    """
    ws = os.path.abspath("./workspace")
    os.makedirs(ws, exist_ok=True)
    return ws


def _list_top_level(repo_dir: str, max_items: int = 50) -> List[str]:
    """
    Return a compact listing of top-level entries in `repo_dir`, up to `max_items`.
    - Appends '/' to directory names.
    - Skips the '.git' directory.
    - Sorts directories first, then files, alphabetically (case-insensitive).
    - Returns [] if the folder is missing or unreadable.
    """
    try:
        names = [n for n in os.listdir(repo_dir) if not n.startswith(".git")]
    except (FileNotFoundError, PermissionError):
        return []

    names.sort(key=lambda n: (not os.path.isdir(os.path.join(repo_dir, n)), n.lower()))

    entries: List[str] = []
    for name in names:
        path = os.path.join(repo_dir, name)
        entries.append(name + "/" if os.path.isdir(path) else name)
        if len(entries) >= max_items:
            break

    return entries


def _repo_cache_path_by_branch(url: str, branch: str) -> str:
    """
    Return a deterministic, filesystem-safe path under the workspace for this URL+branch.
    """
    base = _ensure_workspace()
    safe_url = (
        url.replace("://", "_")
           .replace("/", "_")
           .replace("@", "_")
           .replace(":", "_")
           .replace("\\", "_")
           .strip()
    )
    safe_branch = (branch or "unknown").replace("/", "_").replace(":", "_").replace("\\", "_").strip()
    return os.path.join(base, f"{safe_url}__{safe_branch}")

def _remote_heads(repo: "Repo") -> set[str]:
    """
    Return the set of remote branch names (e.g., {'main', 'master', 'develop'}).
    """
    try:
        return {r.remote_head for r in repo.remotes.origin.refs if getattr(r, "remote_head", None) and r.remote_head != "HEAD"}
    except Exception:
        return set()

def _remote_heads_by_url(url: str) -> set[str]:
    """
    Return the set of remote branch names (e.g., {'main', 'master', 'develop'})
    without cloning the repo (uses `git ls-remote --heads`).
    """
    from git import Git
    heads = set()
    try:
        out = Git().ls_remote("--heads", url)  # lines like: "<sha>\trefs/heads/<name>"
        for line in (out or "").splitlines():
            try:
                _sha, ref = line.strip().split("\t", 1)
                if ref.startswith("refs/heads/"):
                    heads.add(ref.split("/", 2)[-1])
            except ValueError:
                continue
    except Exception:
        pass
    return heads

def _detect_default_branch_by_url(url: str) -> str:
    """
    Detect the remote's default branch by querying origin/HEAD symbolically,
    without cloning (uses `git ls-remote --symref <url> HEAD`).
    Falls back to common names, then any head.
    """
    from git import Git
    try:
        out = Git().ls_remote("--symref", url, "HEAD")
        # Example first line: "ref: refs/heads/develop HEAD"
        for line in (out or "").splitlines():
            line = line.strip()
            if line.startswith("ref:"):
                parts = line.split()
                # parts[1] should be "refs/heads/<branch>"
                ref = parts[1] if len(parts) > 1 else ""
                if ref.startswith("refs/heads/"):
                    return ref.split("/")[-1]
    except Exception:
        pass

    # Fallbacks: prefer common names, then any head we can find
    heads = _remote_heads_by_url(url)
    for cand in ("main", "master"):
        if cand in heads:
            return cand
    return sorted(heads)[0] if heads else "main"


def _detect_default_branch_by_url(url: str) -> str:
    """
    Detect the remote's default branch by querying origin/HEAD symbolically,
    without cloning (uses `git ls-remote --symref <url> HEAD`).
    Falls back to common names, then any head.
    """
    from git import Git
    try:
        out = Git().ls_remote("--symref", url, "HEAD")
        # Example first line: "ref: refs/heads/develop HEAD"
        for line in (out or "").splitlines():
            line = line.strip()
            if line.startswith("ref:"):
                parts = line.split()
                # parts[1] should be "refs/heads/<branch>"
                ref = parts[1] if len(parts) > 1 else ""
                if ref.startswith("refs/heads/"):
                    return ref.split("/")[-1]
    except Exception:
        pass

    # Fallbacks: prefer common names, then any head we can find
    heads = _remote_heads_by_url(url)
    for cand in ("main", "master"):
        if cand in heads:
            return cand
    return sorted(heads)[0] if heads else "main"


def clone_or_update_repo(url: str, branch: str | None = None) -> CloneResult:
    """
    Clone/update shallowly (depth=1) into a cache path that reflects the *actual* branch.
    Behavior:
      - If user requested a branch and it exists remotely → use it.
      - Else → use the remote's default branch (origin/HEAD).
      - Clone/update at ./workspace/<safe_url>__<actual_branch>.
    """
    depth = 1  # shallow, latest only
    t0 = time.time()

    # Decide the branch BEFORE cloning so the path matches the actual branch used.
    remote_heads = _remote_heads_by_url(url)
    if branch and branch in remote_heads:
        use_branch = branch
    else:
        use_branch = _detect_default_branch_by_url(url)

    target = _repo_cache_path_by_branch(url, use_branch)
    reused_existing = False

    try:
        if os.path.isdir(os.path.join(target, ".git")):
            # Fast update existing per-branch clone
            repo = Repo(target)
            repo.git.fetch("--depth", str(depth), "origin", use_branch)
            repo.git.checkout(use_branch)
            repo.git.reset("--hard", f"origin/{use_branch}")
            reused_existing = True
        else:
            # Fresh per-branch clone
            if os.path.isdir(target):
                shutil.rmtree(target, ignore_errors=True)
            repo = Repo.clone_from(url, target, branch=use_branch, depth=depth)

        commit = repo.head.commit.hexsha
        took_sec = round(time.time() - t0, 2)
        return CloneResult(
            repo_dir=target,
            commit=commit,
            branch=use_branch,
            top_level=_list_top_level(target),
            took_sec=took_sec,
            reused_existing=reused_existing,
        )
    except GitCommandError as e:
        raise RuntimeError(f"Git error: {e}") from e
