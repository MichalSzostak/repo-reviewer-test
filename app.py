from __future__ import annotations

import os
import time

import gradio as gr

from rich.console import Console

import core.util as util
from core.model_manager import get_or_load_model, generate_text, LoadedModel
from core.repo_fetcher import clone_or_update_repo, list_remote_branches, detect_default_branch
from core.repo_scanner import list_text_files, read_for_preview, lang_from_path, read_text_file, choose_bucket


console = Console(record=True)

# -----------------------------
# Gradio callbacks
# -----------------------------

def on_clone_click(url: str, selected_branch: str | None, force_fresh: bool):
    if not url or not url.strip():
        return (
            gr.update(value=""),
            gr.update(value="Please provide a repository URL."),
            None,
            gr.update(choices=[], value=None, interactive=False),
        )
    try:
        repo_url = url.strip()
        branch_to_use = (selected_branch or "").strip() or detect_default_branch(repo_url)

        res = clone_or_update_repo(
            repo_url,
            branch=branch_to_use,
            force_fresh=bool(force_fresh),
        )

        branches = list_remote_branches(repo_url)
        branch_update = gr.update(choices=branches, value=res.branch, interactive=True)

        note = " (fresh clone)" if force_fresh else (" (reused cache)" if res.reused_existing else "")
        msg = (
            f"**Cloned/Updated OK**{note}  \n"
            f"- Path: `{res.repo_dir}`  \n"
            f"- Branch: `{res.branch}`  \n"
            f"- Commit: `{res.commit}`  \n"
            f"- Reused existing: `{res.reused_existing}`  \n"
            f"- Took: `{res.took_sec}s`"
        )
        tree_md = util.render_repo_tree(res.repo_dir, max_depth=4, per_dir_limit=200)
        return gr.update(value=msg), gr.update(value=tree_md), res.repo_dir, branch_update

    except Exception as e:
        return (
            gr.update(value=""),
            gr.update(value=f"**Error:** {e}"),
            None,
            gr.update(choices=[], value=None, interactive=False),
        )



def on_list_files_click(repo_dir: str | None):
    if not repo_dir:
        return gr.update(choices=[], value=None), gr.update(value="(clone first)")
    files = list_text_files(repo_dir)
    preview = "(no text files found)" if not files else f"{len(files)} files that can be analyzed"
    default_val = files[0] if files else None
    return gr.update(choices=files, value=default_val), gr.update(value=preview)


def on_preview_click(repo_dir: str | None, rel_path: str | None, show_full: bool):
    if not (repo_dir and rel_path):
        return gr.update(value="(select a file)"), gr.update(value="")
    text, notice = read_for_preview(
        repo_dir, rel_path,
        show_full=bool(show_full),
        soft_cap_chars=50_000,
        head_chars=2_000,
    )
    lang = lang_from_path(rel_path)
    supported = {"python","yaml","json","javascript","typescript","markdown","toml","ini","xml","html","css","shell"}
    if lang in supported:
        return gr.update(value=text, language=lang), gr.update(value=notice)
    else:
        return gr.update(value=text), gr.update(value=notice)


def on_load_models_click(py_model_name: str, txt_model_name: str, cfn_model_name: str):
    """
    Load three generators: python, general text, cloudformation.
    Uses get_or_load_model (cached).
    """
    try:
        py = get_or_load_model(py_model_name,  min_context_tokens=2048, prefer_fp16=True)
        tx = get_or_load_model(txt_model_name, min_context_tokens=2048, prefer_fp16=True)
        cf = get_or_load_model(cfn_model_name, min_context_tokens=2048, prefer_fp16=True)

        status = (
            "**Models ready**  \n"
            f"- Python: `{py.model_id}` (context≈{py.context_tokens})  \n"
            f"- Text: `{tx.model_id}` (context≈{tx.context_tokens})  \n"
            f"- CloudFormation: `{cf.model_id}` (context≈{cf.context_tokens})"
        )
        return {
            "python": py,
            "text":   tx,
            "cfn":    cf,
        }, gr.update(value=status)

    except Exception as e:
        return None, gr.update(value=f"**Model load error:** {e}")


def on_summarise_repo_click(
    repo_dir: str | None,
    max_files: int,
    max_in: int, max_out: int, overlap_unused: int,  # overlap kept for UI parity
    models_state: dict | None,
):
    if not repo_dir:
        return None, gr.update(value="(clone first)"), gr.update(value="")
    if not models_state:
        return None, gr.update(value="(load models first)"), gr.update(value="")

    files = list_text_files(repo_dir)
    if not files:
        return None, gr.update(value="(no text-like files found)"), gr.update(value="")
    files = files[: int(max_files)]
    total = len(files)
    summaries = []
    t0 = time.time()
    taken = 0
    for idx, rel in enumerate(files, start=1):
        yield summaries, gr.update(value=f"Summarizing **{rel}** ({idx} of {total})…"), gr.update(value=util.summaries_to_markdown(summaries))
        if taken >= int(max_files):
            break

        # Simple skip list (keep it minimal)
        base = os.path.basename(rel).lower()
        if (
            base.startswith(("license", "copying", "notice"))
            or base in {"package-lock.json","poetry.lock","yarn.lock","pnpm-lock.yaml","go.sum"}
            or base.endswith((".min.js",".min.css"))
        ):
            continue

        abspath = os.path.join(repo_dir, rel)
        text = read_text_file(abspath)
        bucket = choose_bucket(rel, text)  # 'python' | 'cfn' | 'text'
        handle: LoadedModel = (models_state.get(bucket) or models_state.get("text") or models_state.get("python"))

        try:
            s = util._summarise_one_file(rel, text, handle, max_in=max_in, max_out=max_out)
        except Exception as e:
            s = f"(summary error: {e})"

        summaries.append({
            "path": rel,
            "lang": lang_from_path(rel),
            "bucket": bucket,
            "model": handle.model_id,
            "summary": s,
        })
        taken += 1

    report = [f"Summarised {len(summaries)} file(s)."]
    report += [f"- {s['path']}" for s in summaries[:10]]
    if len(summaries) > 10:
        report.append("...")
    took = round(time.time() - t0, 2)
    yield summaries, gr.update(value=f"Done. Summarised {len(summaries)} file(s) in {took}s."), gr.update(value=util.summaries_to_markdown(summaries))

def on_save_summaries_click(summaries: list | None):
    if not summaries:
        return gr.update(value="(nothing to save)")
    os.makedirs("./out", exist_ok=True)
    out_path = os.path.abspath("./out/repo_summaries.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(util.summaries_to_markdown(summaries))
    return gr.update(value=f"Saved: `{out_path}`")


def on_ticket_reason_click(
    ticket_text: str | None,
    summaries: list | None,
    models_state: dict | None,
    max_in: int, max_out: int,
):
    if not (ticket_text and ticket_text.strip()):
        return gr.update(value="(paste a ticket/bug text)")
    if not summaries:
        return gr.update(value="(summarise the repo first)")
    if not models_state:
        return gr.update(value="(load models first)")

    # Use the general text model by default for reasoning
    handle: LoadedModel = (models_state.get("text") or models_state.get("python"))

    summaries_block = util._make_summaries_block(summaries)
    try:
        answer = generate_text(
            handle,
            messages=util._reasoning_messages(ticket_text.strip(), summaries_block),
            max_input_tokens=int(max_in),
            max_new_tokens=int(max_out),
        )
        header = f"**Reasoner model:** `{handle.model_id}`"
        return gr.update(value=f"{header}\n\n{answer}")
    except Exception as e:
        return gr.update(value=f"(reasoning error: {e})")


def on_summarize_click(
    repo_dir: str | None,
    rel_path: str | None,
    max_in: int, max_out: int, overlap_unused: int,
    models_state: dict | None
):
    if not (repo_dir and rel_path):
        return gr.update(value="(clone and select a file)")
    if not models_state:
        return gr.update(value="(load models first)")

    abspath = os.path.join(repo_dir, rel_path)
    text = read_text_file(abspath)
    bucket = choose_bucket(rel_path, text)
    handle: LoadedModel = (models_state.get(bucket) or models_state.get("text") or models_state.get("python"))

    try:
        summary = util._summarise_one_file(rel_path, text, handle, max_in=max_in, max_out=max_out)
        header = f"**Model:** `{handle.model_id}` | **Bucket:** `{bucket}` | **File:** `{rel_path}`"
        return gr.update(value=f"{header}\n\n{summary}")
    except Exception as e:
        return gr.update(value=f"**Summarisation error:** {e}")


def on_save_summary_click(repo_dir: str | None, rel_path: str | None, summary_md: str | None):
    if not summary_md:
        return gr.update(value="(nothing to save)")
    try:
        os.makedirs("./out", exist_ok=True)
        safe_name = rel_path.replace("/", "_").replace("\\", "_") if rel_path else "summary"
        out_path = os.path.abspath(os.path.join("./out", f"summary_{safe_name}.md"))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(summary_md)
        return gr.update(value=f"Saved: `{out_path}`")
    except Exception as e:
        return gr.update(value=f"**Save error:** {e}")


# -----------------------------
# UI
# -----------------------------

with gr.Blocks(title="Repo Explainer") as demo:
    
    # --- Clone / Update ---
    with gr.Row():
        url = gr.Textbox(
            label="Repository URL (HTTPS or SSH)",
            placeholder="https://github.com/..."
        )
        branch_dd = gr.Dropdown(
            label="Branch",
            choices=[],
            value=None,
            interactive=False,
        )
    with gr.Row():
        clone_btn = gr.Button("Clone repo / Update branch", variant="primary")
        force_fresh_cb = gr.Checkbox(
            label="Fresh clone (delete local cache for selected branch)",
            value=False,
        )
    clone_status = gr.Markdown()
    tree_md = gr.Markdown()

    # States
    repo_dir_state = gr.State()
    models_state = gr.State()
    summaries_state = gr.State()  # list of {"path","lang","bucket","model","summary"}

    # Wire clone
    clone_event = clone_btn.click(
        fn=on_clone_click,
        inputs=[url, branch_dd, force_fresh_cb],
        outputs=[clone_status, tree_md, repo_dir_state, branch_dd],
    )

    gr.Markdown("---")

    # --- Files + preview ---
    gr.Markdown("## Files")
    files_info = gr.Markdown()
    files_dropdown = gr.Dropdown(choices=[], label="Select a file", interactive=True)
    preview_btn = gr.Button("Preview file")
    show_full_cb = gr.Checkbox(label="Show full file (may be slow)", value=False)
    preview_code = gr.Code(label="Preview")
    preview_notice = gr.Markdown()

    clone_event.then(
        fn=on_list_files_click,
        inputs=[repo_dir_state],
        outputs=[files_dropdown, files_info],
    )

    preview_btn.click(
        fn=on_preview_click,
        inputs=[repo_dir_state, files_dropdown, show_full_cb],
        outputs=[preview_code, preview_notice],
    )

    gr.Markdown("---")

    # --- Models & settings ---
    gr.Markdown("## Models & Settings")
    with gr.Row():
        py_model_name  = gr.Textbox(label="Python model",          value="Qwen/Qwen2.5-3B-Instruct")
        txt_model_name = gr.Textbox(label="General text model",    value="Qwen/Qwen2.5-3B-Instruct")
        cfn_model_name = gr.Textbox(label="CloudFormation model",  value="Qwen/Qwen2.5-3B-Instruct")
    load_models_btn = gr.Button("Load models")
    model_status = gr.Markdown()

    with gr.Row():
        max_input_tokens = gr.Slider(128, 4096, value=2048, step=64, label="Max input tokens (per window)")
        overlap_tokens = gr.Slider(0, 512, value=64, step=16, label="Token overlap")  # not used by generator; kept for diagnostics
        max_new_tokens = gr.Slider(64, 1024, value=256, step=32, label="Max new tokens (output)")

    load_models_btn.click(
        fn=on_load_models_click,
        inputs=[py_model_name, txt_model_name, cfn_model_name],
        outputs=[models_state, model_status],
    )

    gr.Markdown("---")

    # --- Repo summaries (loop) ---
    gr.Markdown("## Repo summaries")
    with gr.Row():
        max_files_to_summarise = gr.Slider(1, 500, value=50, step=1, label="Max files to summarise")
        summarise_repo_btn = gr.Button("Summarise repo")
    summarise_report_md = gr.Markdown()
    summaries_md = gr.Markdown()

    summarise_repo_btn.click(
        fn=on_summarise_repo_click,
        inputs=[repo_dir_state, max_files_to_summarise, max_input_tokens, max_new_tokens, overlap_tokens, models_state],
        outputs=[summaries_state, summarise_report_md, summaries_md],
    )

    gr.Markdown("---")

    # --- Ticket reasoning ---
    gr.Markdown("## Ticket reasoning")
    ticket_text = gr.Textbox(label="Ticket / bug report / feature request", lines=6, placeholder="Paste the ticket text here")
    reason_btn = gr.Button("Suggest plan and reading")
    reason_md = gr.Markdown()

    reason_btn.click(
        fn=on_ticket_reason_click,
        inputs=[ticket_text, summaries_state, models_state, max_input_tokens, max_new_tokens],
        outputs=[reason_md],
    )

if __name__ == "__main__":
    demo.launch()
