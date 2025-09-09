from __future__ import annotations
import os
import gradio as gr
from rich.tree import Tree
from rich.console import Console

from core.batch_summary import summarise_files_batch, summaries_to_markdown
from core.diagnostics import measure_text_tokens, chunk_plan, gpu_memory_info
from core.fetcher import clone_or_update_repo
from core.scanner import list_text_files, read_for_preview
from core.simple_summary import (
    load_generator, read_text_file, summarise_file_text, lang_from_path, choose_bucket
)
from core.reasoner import reason_over_summaries

console = Console(record=True)


def render_top_level(repo_dir: str, entries: list[str]) -> str:
    """
    Pretty-print a simple top-level tree using rich, return as Markdown code block.
    """
    tree = Tree(f"[bold cyan]{os.path.basename(repo_dir)}[/]  (top-level)")
    for e in entries:
        if e.endswith("/"):
            tree.add(f"[yellow]{e}[/]")
        else:
            tree.add(e)
    console.clear()
    console.print(tree)
    ansi = console.export_text(clear=False)
    return f"```\n{ansi}\n```"


def on_clone_click(url: str, branch: str):
    if not url.strip():
        return gr.update(value=""), gr.update(value="Please provide a repository URL."), None
    try:
        requested = (branch or "").strip() or None
        res = clone_or_update_repo(url.strip(), branch=requested)

        note = ""
        if requested and res.branch != requested:
            note = f"\n- Note: requested branch `{requested}` not found; using default `{res.branch}`."

        msg = (
            f"**Cloned OK**  \n"
            f"- Path: `{res.repo_dir}`  \n"
            f"- Branch: `{res.branch}`  \n"
            f"- Commit: `{res.commit}`  \n"
            f"- Reused existing: `{res.reused_existing}`  \n"
            f"- Took: `{res.took_sec}s`"
            f"{note}"
        )
        tree_md = render_top_level(res.repo_dir, res.top_level)
        return gr.update(value=msg), gr.update(value=tree_md), res.repo_dir
    except Exception as e:
        return gr.update(value=""), gr.update(value=f"**Error:** {e}"), None


def on_list_files_click(repo_dir: str | None):
    if not repo_dir:
        return gr.update(choices=[], value=None), gr.update(value="(clone first)")
    files = list_text_files(repo_dir)
    preview = "(no text files found)" if not files else f"{len(files)} text-like files"
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

    supported = {
        "python", "yaml", "json", "javascript", "typescript",
        "markdown", "toml", "ini", "xml", "html", "css", "shell",
    }
    if lang in supported:
        return gr.update(value=text, language=lang), gr.update(value=notice)
    else:
        return gr.update(value=text), gr.update(value=notice)


def on_load_models_click(py_model_name: str, txt_model_name: str, cfn_model_name: str):
    """
    Load three generators: python, general text, cloudformation.
    """
    try:
        py_tok,  py_model,  py_ctx,  _ = load_generator(py_model_name,  fp16=True)
        txt_tok, txt_model, txt_ctx, _ = load_generator(txt_model_name, fp16=True)
        cfn_tok, cfn_model, cfn_ctx, _ = load_generator(cfn_model_name, fp16=True)

        status = (
            f"**Models loaded**  \n"
            f"- Python: `{py_model_name}` (ctx≈{py_ctx})  \n"
            f"- Text: `{txt_model_name}` (ctx≈{txt_ctx})  \n"
            f"- CloudFormation: `{cfn_model_name}` (ctx≈{cfn_ctx})"
        )
        return {
            "python": (py_tok,  py_model,  py_model_name),
            "text":   (txt_tok, txt_model, txt_model_name),
            "cfn":    (cfn_tok, cfn_model, cfn_model_name),
        }, gr.update(value=status)

    except Exception as e:
        return None, gr.update(value=f"**Model load error:** {e}")


def on_summarise_repo_click(
    repo_dir: str | None,
    max_files: int,
    max_in: int, max_out: int, overlap: int,
    models_state: dict | None,
):
    try:
        if not repo_dir:
            return None, gr.update(value="(clone first)")
        if not models_state:
            return None, gr.update(value="(load models first)")

        files = list_text_files(repo_dir)
        if not files:
            return None, gr.update(value="(no text-like files found)")

        summaries, took = summarise_files_batch(
            repo_root=repo_dir,
            rel_paths=files,
            models_state=models_state,
            max_files=int(max_files),
            max_input_tokens=int(max_in),
            max_new_tokens=int(max_out),
            overlap=int(overlap),
        )

        report = [f"Summarised {len(summaries)} file(s) in {took}s."]
        report += [f"- {s['path']}" for s in summaries[:10]]
        if len(summaries) > 10:
            report.append("...")
        return summaries, gr.update(value="\n".join(report)), gr.update(value=summaries_to_markdown(summaries))
    except Exception as e:
        return None, gr.update(value=f'**Batch error:** {e}'), gr.update(value="")

def on_save_summaries_click(summaries: list | None):
    if not summaries:
        return gr.update(value="(nothing to save)")
    os.makedirs("./out", exist_ok=True)
    out_path = os.path.abspath("./out/repo_summaries.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(summaries_to_markdown(summaries))
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

    tok, model, model_name = models_state.get("text", models_state.get("python"))
    try:
        answer = reason_over_summaries(
            ticket_text=ticket_text.strip(),
            summaries=summaries,
            tok=tok,
            model=model,
            max_input_tokens=int(max_in),
            max_new_tokens=int(max_out),
        )
        header = f"**Reasoner model:** `{model_name}`"
        return gr.update(value=f"{header}\n\n{answer}")
    except Exception as e:
        return gr.update(value=f"(reasoning error: {e})")


def on_summarize_click(repo_dir: str | None, rel_path: str | None, max_in: int, max_out: int, overlap: int, models_state: dict | None):
    if not (repo_dir and rel_path):
        return gr.update(value="(clone and select a file)")
    if not models_state:
        return gr.update(value="(load models first)")

    abspath = os.path.join(repo_dir, rel_path)
    text = read_text_file(abspath)

    bucket = choose_bucket(rel_path, text)  # 'python' | 'cfn' | 'text'
    tok, model, model_name = (
        models_state.get(bucket)
        or models_state.get("text")
        or models_state.get("python")
    )

    try:
        summary = summarise_file_text(
            file_text=text,
            file_path=rel_path,
            tok=tok,
            model=model,
            max_input_tokens=int(max_in),
            max_new_tokens=int(max_out),
            overlap=int(overlap),
        )
        header = f"**Model:** `{model_name}` | **Bucket:** `{bucket}` | **File:** `{rel_path}`"
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


def on_diagnostics_click(repo_dir: str | None, rel_path: str | None, max_in: int, overlap: int, models_state: dict | None):
    if not (repo_dir and rel_path):
        return gr.update(value="(clone and select a file)")
    if not models_state:
        return gr.update(value="(load models first)")

    lang = lang_from_path(rel_path)
    tok, _model, model_name = models_state.get(lang, models_state.get("python"))

    abspath = os.path.join(repo_dir, rel_path)
    try:
        with open(abspath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            bucket = choose_bucket(rel_path, text)
            tok, _model, model_name = (
                models_state.get(bucket)
                or models_state.get("text")
                or models_state.get("python")
            )
    except Exception as e:
        return gr.update(value=f"Error reading file: {e}")

    plan = chunk_plan(text, tok, max_tokens=int(max_in), overlap=int(overlap))
    gpu = gpu_memory_info()

    md = []
    md.append(f"**File:** `{rel_path}`")
    md.append(f"**Language:** `{lang}`")
    md.append(f"**Tokenizer (from model):** `{model_name}`")
    md.append("")
    md.append("### Tokenization")
    md.append(f"- Total tokens in file: **{plan['total_tokens']}**")
    md.append(f"- Chunk window: **{int(max_in)}** tokens, overlap: **{int(overlap)}**")
    md.append(f"- Number of chunks planned: **{plan['num_chunks']}**")
    if plan["num_chunks"] > 0:
        sizes = [c["size"] for c in plan["chunks"]]
        md.append(f"- Chunk sizes (first 10): {sizes[:10]}")
    md.append("")
    md.append("### GPU memory (best-effort)")
    if gpu["available"]:
        md.append(f"- Allocated: **{gpu['current_mb']} MB**")
        md.append(f"- Reserved: **{gpu['reserved_mb']} MB**")
        md.append(f"- Total: **{gpu['total_mb']} MB**")
    else:
        md.append("- CUDA not available (running on CPU or no GPU detected)")
    return gr.update(value="\n".join(md))


with gr.Blocks(title="Repo Explainer – Minimal Summariser") as demo:
    gr.Markdown("# Repo Explainer (Minimal)\nClone → list files → load model(s) → summarise selected file.\n\n> Uses token-aware map-reduce if the file is too large for one pass.")

    # --- Clone / Update ---
    with gr.Row():
        url = gr.Textbox(label="Repository URL (HTTPS or SSH)", placeholder="https://github.com/base2Services/cloudformation-custom-resources-python")
        branch = gr.Textbox(label="Branch", value="main")

    clone_btn = gr.Button("Clone / Update", variant="primary")
    clone_status = gr.Markdown()
    tree_md = gr.Markdown()

    # States
    repo_dir_state = gr.State()
    models_state = gr.State()
    summaries_state = gr.State()  # list of {"path","lang","summary"}

    # Wire clone (no .then yet; components referenced later)
    clone_event = clone_btn.click(
        fn=on_clone_click,
        inputs=[url, branch],
        outputs=[clone_status, tree_md, repo_dir_state],
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

    # Now that components exist, chain list refresh after clone
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
        py_model_name  = gr.Textbox(label="Python model",          value="Qwen/Qwen2.5-Coder-3B-Instruct")
        txt_model_name = gr.Textbox(label="General text model",    value="microsoft/Phi-3.1-mini-4k-instruct")
        cfn_model_name = gr.Textbox(label="CloudFormation model",  value="Qwen/Qwen2.5-3B-Instruct")
    load_models_btn = gr.Button("Load models")
    model_status = gr.Markdown()

    # Sliders
    with gr.Row():
        max_input_tokens = gr.Slider(128, 4096, value=2048, step=64, label="Max input tokens (per window)")
        overlap_tokens = gr.Slider(0, 512, value=64, step=16, label="Token overlap")
        max_new_tokens = gr.Slider(64, 1024, value=256, step=32, label="Max new tokens (output)")

    load_models_btn.click(
        fn=on_load_models_click,
        inputs=[py_model_name, yml_model_name],
        outputs=[models_state, model_status],
    )

    gr.Markdown("---")

    # --- Repo summaries (batch) ---
    gr.Markdown("## Repo summaries")
    with gr.Row():
        max_files_to_summarise = gr.Slider(1, 500, value=50, step=1, label="Max files to summarise")
        summarise_repo_btn = gr.Button("Summarise repo")
    summarise_report_md = gr.Markdown()
    summaries_md = gr.Markdown() 
    save_summaries_btn = gr.Button("Save summaries")
    save_summaries_status = gr.Markdown()

    summarise_repo_btn.click(
        fn=on_summarise_repo_click,
        inputs=[repo_dir_state, max_files_to_summarise, max_input_tokens, max_new_tokens, overlap_tokens, models_state],
        outputs=[summaries_state, summarise_report_md, summaries_md],  # <— add summaries_md
    )
    save_summaries_btn.click(
        fn=on_save_summaries_click,
        inputs=[summaries_state],
        outputs=[save_summaries_status],
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

    gr.Markdown("---")

    # --- Diagnostics ---
    gr.Markdown("## Diagnostics")
    diag_btn = gr.Button("Analyze selected file")
    diag_md = gr.Markdown()

    diag_btn.click(
        fn=on_diagnostics_click,
        inputs=[repo_dir_state, files_dropdown, max_input_tokens, overlap_tokens, models_state],
        outputs=[diag_md],
    )

    gr.Markdown("---")

    # --- Single-file summary + save ---
    summarise_btn = gr.Button("Summarise selected file", variant="primary")
    summary_md = gr.Markdown()
    save_btn = gr.Button("Save summary")
    save_status = gr.Markdown()

    summarise_btn.click(
        fn=on_summarize_click,
        inputs=[repo_dir_state, files_dropdown, max_input_tokens, max_new_tokens, overlap_tokens, models_state],
        outputs=[summary_md],
    )
    save_btn.click(
        fn=on_save_summary_click,
        inputs=[repo_dir_state, files_dropdown, summary_md],
        outputs=[save_status],
    )

if __name__ == "__main__":
    demo.launch()
