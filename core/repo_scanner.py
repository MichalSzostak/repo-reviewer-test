from __future__ import annotations
import os
import pathlib
from typing import List, Dict


def list_text_files(root: str, max_files: int = 5000) -> List[str]:
    allow_ext = {
        ".py", ".yaml", ".yml",
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

def read_text_file(path: str, cap_bytes: int = 1_000_000) -> str:
    """
    Read a text file safely with a soft byte cap (to avoid huge prompts).
    Returns the first ~cap_bytes worth of text; decoding errors are ignored.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(cap_bytes)
    except Exception as e:
        base = os.path.basename(path)
        return f"(error reading {base}: {e})"


def lang_from_path(path: str) -> str:
    """
    Infer a broad 'language' from the filename extension.
    Used only for light classification; not for syntax highlighting.
    """
    p = pathlib.Path((path or "").lower())
    # Handle common multi-suffix names gracefully (e.g., .template.yaml)
    suffixes = "".join(p.suffixes[-2:])  # e.g., ".template.yaml" -> ".template.yaml"
    last = p.suffix.lower() if p.suffix else ""

    mapping = {
        ".py": "python",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".js": "javascript",
        ".ts": "typescript",
        ".md": "markdown",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
        ".txt": "text",
        ".sh": "shell",
        ".ps1": "shell",
    }

    # Prefer the last suffix; fall back to multi-suffix check
    if last in mapping:
        return mapping[last]
    if suffixes.endswith(".template.yaml") or suffixes.endswith(".template.json"):
        # still treated as yaml/json at this stage; CFN detection is in choose_bucket()
        return "yaml" if suffixes.endswith(".yaml") else "json"

    return "text"


def looks_like_cfn_text(text: str) -> bool:
    """
    Lightweight heuristic to detect CloudFormation templates (YAML/JSON).
    Checks only the head of the file for common CFN markers.
    """
    if not text:
        return False

    head = text[:20000] if isinstance(text, str) else str(text)[:20000]

    markers = (
        "AWSTemplateFormatVersion",
        "Resources:",
        '"Resources"',
        "Parameters:",
        '"Parameters"',
        "Mappings:",
        '"Mappings"',
        "Transform: AWS::Serverless-2016-10-31",
        "Transform: 'AWS::Serverless-2016-10-31'",
        "Transform: AWS::LanguageExtensions",
        "Type: AWS::",        # YAML resource type line
        '"Type": "AWS::',     # JSON resource type line
    )
    return any(m in head for m in markers)


def choose_bucket(path: str, text: str) -> str:
    """
    Route a file into one of {'python','cfn','text'} based on path + content.
    - 'python' for .py
    - 'cfn' for YAML/JSON that *looks* like CloudFormation
    - 'text' for everything else (README, Markdown, configs, etc.)
    """
    language = lang_from_path(path)
    if language == "python":
        return "python"
    if language in {"yaml", "json"} and looks_like_cfn_text(text):
        return "cfn"
    return "text"