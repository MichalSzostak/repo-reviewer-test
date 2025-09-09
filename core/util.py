import os
from typing import List, Dict

from rich.console import Console

from rich.tree import Tree

from core.model_manager import LoadedModel, generate_text

def summaries_to_markdown(summaries: List[Dict]) -> str:
    """
    Render per-file summaries into a single Markdown doc grouped by directory.

    Expected item shape (bucket/model optional):
      {
        "path": "core/utils.py",
        "summary": "- bullet\n- bullet",
        "bucket": "python",
        "model": "Qwen/Qwen2.5-3B-Instruct"
      }

    Returns a markdown string safe to pass to gr.Markdown(value=...).
    """
    if not summaries:
        return "# Repo summaries\n\n(none)\n"

    items = sorted(summaries, key=lambda s: s.get("path", ""))

    lines = ["# Repo summaries", ""]
    current_dir = None

    for item in items:
        path = item.get("path", "")
        directory = os.path.dirname(path) or "."
        name = os.path.basename(path) or path
        summary = (item.get("summary") or "").strip()
        bucket = item.get("bucket")
        model = item.get("model")

        # Start a new directory section if needed
        if directory != current_dir:
            if current_dir is not None:
                lines.append("")  # spacing between sections
            lines.append(f"## {directory}")
            lines.append("")
            current_dir = directory

        # File header with optional meta
        meta_bits = []
        if bucket:
            meta_bits.append(f"`{bucket}`")
        if model:
            meta_bits.append(f"`{model}`")
        meta = (" — " + " · ".join(meta_bits)) if meta_bits else ""
        lines.append(f"### {name}{meta}")

        # Body: include summary exactly as produced (no post-processing here)
        if summary:
            for ln in summary.splitlines():
                lines.append(ln.rstrip())
        else:
            lines.append("_(no summary)_")

        lines.append("")  # blank line after each file

    return "\n".join(lines)


def render_repo_tree(repo_dir: str, *, max_depth: int = 4, per_dir_limit: int = 200) -> str:
    """
    Render a recursive directory tree (dirs first, then files), skipping .git.
    Limits depth and entries per directory to keep output readable.
    """
    def add_dir(node: Tree, path: str, depth: int):
        if depth > max_depth:
            return
        try:
            names = [n for n in os.listdir(path) if n != ".git"]
        except Exception:
            return
        # dirs first, then files; case-insensitive
        names.sort(key=lambda n: (not os.path.isdir(os.path.join(path, n)), n.lower()))

        count = 0
        for name in names:
            full = os.path.join(path, name)
            if os.path.isdir(full):
                child = node.add(f"[yellow]{name}/[/]")
                add_dir(child, full, depth + 1)
            else:
                node.add(name)
            count += 1
            if count >= per_dir_limit:
                node.add("[dim]… (more files truncated)[/]")
                break

    tree = Tree(f"[bold cyan]{os.path.basename(repo_dir)}[/]  (root)")
    add_dir(tree, repo_dir, depth=1)

    # Use a fresh console to avoid duplicated output from previous prints
    local_console = Console(record=True, width=120)
    local_console.print(tree)
    ansi = local_console.export_text(clear=True)
    return f"```\n{ansi}\n```"

def _summary_messages(file_path: str, file_text: str):
    system = (
        "You are a senior engineer. Return 3-5 concise bullets (each starting with '- '). "
        "Focus on: purpose, key entry points/APIs, libraries explicitly imported in THIS file, and cross-file ties ONLY if visible. "
        "Do NOT restate the request. Do NOT copy code or identifiers. Do NOT include CLI or install steps. "
        "If something is not evident from THIS file, write 'not evident'."
    )
    user = (
        f"File path: {file_path}\n\n"
        f"<file>\n{file_text}\n</file>\n\n"
        "- "
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _reasoning_messages(ticket_text: str, summaries_block: str):
    system = (
        "Produce exactly three Markdown sections and nothing else:\n"
        "# What to read first (top files)\n"
        "List 3-5 file paths. For each: a one-sentence rationale (why it matters for the ticket) "
        "followed by 1-3 short bullets adapted from its summary (no code/CLI).\n"
        "# Prerequisites\n"
        "3-6 bullets of topics/libraries/services a contributor should review.\n"
        "# Work plan\n"
        "3-7 concrete steps; call out risks where relevant.\n"
        "Rules: Do NOT repeat or quote the ticket. Do NOT ask questions. Be concise."
    )
    user = f"Ticket:\n{ticket_text}\n\nFile summaries:\n{summaries_block}\n"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _summarise_one_file(file_path: str, text: str, model: LoadedModel, *, max_in: int, max_out: int) -> str:
    # Single call through the unified generator
    return generate_text(
        model,
        messages=_summary_messages(file_path, text),
        max_input_tokens=int(max_in),
        max_new_tokens=int(max_out),
    )


def _make_summaries_block(summaries: list[dict]) -> str:
    """
    Compact, plain block for the reasoner: 'File: path\n<bullets>\n'
    """
    parts = []
    for item in summaries:
        path = item.get("path", "???")
        summ = (item.get("summary") or "").strip()
        if summ:
            parts.append(f"File: {path}\n{summ}\n")
    return "\n".join(parts).strip()
