from __future__ import annotations

import os
from typing import List, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
)

# -----------------------------
# Cache location
# -----------------------------

def _project_cache_dir() -> str:
    root = os.path.abspath("./.hf_cache")
    os.makedirs(root, exist_ok=True)
    return root

# -----------------------------
# Model loading (simple)
# -----------------------------

def model_ctx_window(model) -> int:
    cfg = getattr(model, "config", None)
    for attr in (
        "n_positions",
        "max_position_embeddings",
        "max_seq_len",
        "max_seq_length",
        "seq_length",
        "rope_scaling_max_position_embeddings",
    ):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return 512

def load_generator(model_name: str, fp16: bool = True):
    """
    Minimal loader:
    - Uses device_map='auto' so we land on GPU if available.
    - No quantization paths. Keep it simple & reliable on Windows.
    Returns: (tokenizer, model, ctx_tokens, used_quant=False)
    """
    cache_dir = _project_cache_dir()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    is_s2s = bool(getattr(config, "is_encoder_decoder", False))

    torch_dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else None
    cls = AutoModelForSeq2SeqLM if is_s2s else AutoModelForCausalLM
    model = cls.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=cache_dir,
    )

    # Ensure a pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

    ctx = model_ctx_window(model)

    # Optional: compact device map summary so you know where it landed
    try:
        dm = getattr(model, "hf_device_map", None)
        if dm:
            from collections import Counter
            c = Counter(dm.values())
            print(f"[load] {model_name} ctx≈{ctx}; device_map={dict(c)}")
        else:
            print(f"[load] {model_name} ctx≈{ctx}; (no device_map reported)")
    except Exception:
        pass

    return tokenizer, model, ctx, False  # used_quant=False (we don't quantize here)

# -----------------------------
# Token helpers
# -----------------------------

def count_tokens(text: str, tok) -> int:
    return len(tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0])

def _truncate_to_tokens(text: str, tok, max_tokens: int) -> Tuple[str, bool, int]:
    """
    Return (possibly truncated_text, was_truncated, original_tokens)
    """
    enc = tok(text, return_tensors="pt", add_special_tokens=False, truncation=False)
    ids = enc["input_ids"][0]
    orig = ids.shape[-1]
    if orig <= max_tokens:
        return text, False, orig
    trimmed = ids[:max_tokens]
    out = tok.decode(trimmed, skip_special_tokens=False)
    return out, True, orig

# -----------------------------
# Generation
# -----------------------------

def _encode_with_clamp(text: str, tok, model, reserve: int = 64):
    ctx = model_ctx_window(model)
    used_max_len = max(8, ctx - int(reserve))
    enc = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=used_max_len,
        padding=False,
    )
    if torch.cuda.is_available():
        enc = {k: v.cuda() for k, v in enc.items()}
    return enc["input_ids"], enc.get("attention_mask", None), used_max_len

def generate_text(prompt: str, tok, model, max_new_tokens: int = 256) -> str:
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=model_ctx_window(model) - 64, padding=False)
    input_len = enc["input_ids"].shape[1]
    if torch.cuda.is_available():
        enc = {k: v.cuda() for k, v in enc.items()}

    gen = model.generate(
        **enc,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        num_beams=1,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    new_ids = gen[0, input_len:]
    return tok.decode(new_ids, skip_special_tokens=True).strip()

# -----------------------------
# File helpers
# -----------------------------

def read_text_file(path: str, cap_bytes: int = 1_000_000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(cap_bytes)
    except Exception as e:
        return f"(error reading {os.path.basename(path)}: {e})"

def lang_from_path(path: str) -> str:
    p = (path or "").lower()
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
    for ext, lang in mapping.items():
        if p.endswith(ext):
            return lang
    return "text"

def looks_like_cfn_text(text: str) -> bool:
    # very lightweight heuristics that work well in practice
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
    )
    t = text if isinstance(text, str) else str(text)
    # quick check to avoid scanning huge files pointlessly
    head = t[:20000]
    return any(m in head for m in markers)

def choose_bucket(path: str, text: str) -> str:
    """Return one of {'python','cfn','text'} based on path+content."""
    lang = lang_from_path(path)
    if lang == "python":
        return "python"
    if lang in {"yaml", "json"} and looks_like_cfn_text(text):
        return "cfn"
    return "text"

# -----------------------------
# Single-pass summariser (simple)
# -----------------------------
def _apply_chat_template(tok, messages: list[dict], max_input_tokens: int) -> dict:
    """
    Build a chat-formatted prompt if the tokenizer supports it; otherwise fall back.
    Returns tokenized inputs (ready for generate) truncated to max_input_tokens.
    """
    try:
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # adds the assistant header token(s)
        )
    except Exception:
        # Fallback for tokenizers without a template
        system = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
        user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        prompt = f"<<SYSTEM>>\n{system}\n\n<<USER>>\n{user}\n\n<<ASSISTANT>>\n"

    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=int(max_input_tokens), padding=False)
    return enc

def _only_bullets(text: str, max_lines: int = 5) -> str:
    """
    Keep only bullet lines; drop everything else (no prompts, no questions).
    """
    lines = [ln.strip() for ln in text.splitlines()]
    bullets = [ln for ln in lines if ln.startswith(("- ", "* ", "• "))]
    if not bullets:
        # as a last resort, take the first non-empty lines and prefix with '- '
        bullets = [f"- {ln}" for ln in lines if ln][:max_lines]
    return "\n".join(bullets[:max_lines])


def summarise_file_text(
    file_text: str,
    file_path: str,
    tok,
    model,
    max_input_tokens: int = 1024,
    max_new_tokens: int = 160,
    overlap: int = 0,
) -> str:
    """
    Compact, bullet-only summary using chat template when available.
    """
    ctx = model_ctx_window(model)
    budget = max(128, min(int(max_input_tokens), ctx - 64))

    # Rules as a single sentence to reduce echoing
    system_msg = (
        "You are a senior engineer. Return 3-5 concise bullets (start with '- '). "
        "Focus on purpose, key entry points/APIs, important libraries explicitly present in this file, and cross-file ties. "
        "Do not restate the request, ask questions, quote code, or include setup/CLI commands. "
        "If something is not evident from this file, say 'not evident' rather than guessing."
    )
    # Bullet anchor at the end:
    user_msg = (
        f"File: {file_path}\n\n"
        f"<file>\n{file_text}\n</file>\n\n"
        "- "
    )

    # Use chat template if available; otherwise fall back to plain prompt
    try:
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": system_msg},
             {"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = f"<<SYSTEM>>\n{system_msg}\n\n<<USER>>\n{user_msg}\n\n<<ASSISTANT>>\n"

    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=budget, padding=False)
    input_len = enc["input_ids"].shape[1]  # <-- length of the prompt
    if model.device.type == "cuda":
        enc = {k: v.to(model.device) for k, v in enc.items()}

    gen = model.generate(
        **enc,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        num_beams=1,
        no_repeat_ngram_size=3,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    new_ids = gen[0, input_len:]
    raw = tok.decode(new_ids, skip_special_tokens=True).strip()

    # --- Inline cleanup (no helpers) ---
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    # drop instruction echoes
    bad_fragments = ("respond with", "do not", "don't", "rules:", "focus on")
    lines = [ln for ln in lines if not any(b in ln.lower() for b in bad_fragments)]
    # keep only bullet-ish lines
    lines = [ln for ln in lines if ln.lstrip().startswith(("- ", "* ", "• "))]
    if not lines:
        # last resort: force the first few lines into bullets
        lines = [f"- {ln}" for ln in (raw.splitlines()[:5] if raw else []) if ln.strip()]
    out = "\n".join(lines[:5])

    # Light truncation note (cheap heuristic)
    try:
        orig_len = len(tok(file_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0])
        if orig_len > (budget // 2):
            out += "\n\n> Note: input truncated to fit the model window."
    except Exception:
        pass

    return out
