from __future__ import annotations
import os
import pathlib
from typing import List, Dict


def list_text_files(root: str, max_files: int = 5000) -> List[str]:
    """
    Return a list of relative paths for likely-text files under `root`.
    Keep this minimal: just an extension allowlist and a few ignores.
    """
    allow_ext = {
        ".py", ".yaml", ".yml", ".json", ".toml", ".xml",
        ".md", ".js", ".ts", ".ini", ".cfg", ".txt",
    }
    ignore_dirs = {".git", "__pycache__", ".venv", "env", "node_modules", "dist", "build"}

    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune ignore dirs in-place for speed
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]

        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            ext = pathlib.Path(name).suffix.lower()
            if ext in allow_ext:
                out.append(rel)
                if len(out) >= max_files:
                    return out
    return out


def read_for_preview(
    root: str,
    rel_path: str,
    show_full: bool,
    soft_cap_chars: int = 50_000,  # cap when "Show full file" is on
    head_chars: int = 2_000,       # cap for head preview
) -> tuple[str, str]:
    """
    Read file text for preview (characters-based).
    - If show_full=False: return first `head_chars` characters.
      If truncated by the head limit, append a visible marker: "[...]".
    - If show_full=True: return up to `soft_cap_chars` characters (no marker).
    Returns (text, notice).
    """
    import os
    abspath = os.path.join(root, rel_path)

    try:
        size = os.path.getsize(abspath)
    except Exception:
        size = None

    limit = soft_cap_chars if show_full else head_chars

    try:
        with open(abspath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(limit + 1)  # +1 to detect truncation
    except Exception as e:
        return f"(error reading {rel_path}: {e})", ""

    truncated = len(text) > limit
    if truncated:
        text = text[:limit]
        if not show_full:
            # Add a clear visual indicator that there is more content
            if not text.endswith("\n"):
                text += "\n"
            text += "[...]"

    # Build a concise notice
    if truncated:
        notice = (
            f"Full view truncated to {limit:,} characters to keep the UI responsive."
            if show_full else
            f"Head preview: showing first {head_chars:,} characters."
        )
    else:
        if size is not None:
            shown = min(size, limit)
            notice = (
                f"Read {shown:,} of {size:,} bytes."
                if show_full else
                f"Read first {min(size, head_chars):,} bytes."
            )
        else:
            notice = ""

    return text, notice
