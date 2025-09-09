from __future__ import annotations


import json
import os
import time
from typing import List, Dict, Tuple

from core.simple_summary import read_text_file, summarise_file_text, lang_from_path, choose_bucket


def summarise_files_batch(
    repo_root: str,
    rel_paths: List[str],
    models_state: Dict[str, tuple],
    max_files: int,
    max_input_tokens: int,
    max_new_tokens: int,
    overlap: int,
) -> Tuple[List[Dict], float]:
    """
    Summarise up to `max_files` files using the language-appropriate model in models_state.
    models_state: {"python": (tok, model, name), "yaml": (tok, model, name)}
    Returns (summaries, took_sec) where summaries = [{"path","lang","summary"}...]
    """
    t0 = time.time()
    out: List[Dict] = []
    count = 0

    for rel in rel_paths: 
        print(f'summarise_file_batch: starting pass {count + 1}...')
        if count >= max_files:
            break


        base = os.path.basename(rel).lower()
        if (
            base.startswith(("license", "copying", "notice"))
            or base in {"package-lock.json", "poetry.lock", "yarn.lock", "pnpm-lock.yaml", "go.sum"}
            or base.endswith((".min.js", ".min.css"))
        ):
            continue
        lang = lang_from_path(rel)
        
        abspath = os.path.join(repo_root, rel)
        text = read_text_file(abspath)
        bucket = choose_bucket(rel, text)
        tok, model, _name = (
            models_state.get(bucket)
            or models_state.get("text")
            or models_state.get("python")
        )
        try:
            s = summarise_file_text(
                file_text=text,
                file_path=rel,
                tok=tok,
                model=model,
                max_input_tokens=int(max_input_tokens),
                max_new_tokens=int(max_new_tokens),
                overlap=int(overlap),
            )
            print(json.dumps({"path": rel, "lang": lang, "summary": s}, indent=4))
        except Exception as e:
            s = f"(summary error: {e})"

        out.append({"path": rel, "lang": lang, "summary": s})
        count += 1

    return out, round(time.time() - t0, 2)


def summaries_to_markdown(summaries: List[Dict]) -> str:
    """
    Render summaries grouped by directory. Bullets per file, no input echoed.
    """
    if not summaries:
        return "# File summaries\n\n(none)\n"

    summaries = sorted(summaries, key=lambda s: s["path"])
    lines = ["# Repo summaries", ""]
    current_dir = None

    for item in summaries:
        path = item["path"]
        d = os.path.dirname(path) or "."
        name = os.path.basename(path)

        if d != current_dir:
            lines.append(f"## {d}")
            lines.append("")
            current_dir = d

        lines.append(f"### {name}")
        for ln in (item["summary"] or "").splitlines():
            if ln.strip():
                lines.append(ln)
        lines.append("")

    return "\n".join(lines)
