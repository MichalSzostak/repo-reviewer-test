# core/model_manager.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)


# -----------------------------
# Types
# -----------------------------

@dataclass
class LoadedModel:
    model_id: str
    tokenizer: Any
    model: Any
    context_tokens: int
    supports_chat_template: bool
    device_map_summary: Optional[Dict[str, int]]  # e.g., {0: 28, "cpu": 12}


# -----------------------------
# Cache & config
# -----------------------------

_MODEL_CACHE: Dict[str, LoadedModel] = {}


def _project_cache_dir() -> str:
    root = os.path.abspath("./.hf_cache")
    os.makedirs(root, exist_ok=True)
    return root


# -----------------------------
# Capability checks
# -----------------------------

def _has_chat_template(tokenizer) -> bool:
    """
    True iff the tokenizer exposes a non-empty chat template.
    """
    try:
        return hasattr(tokenizer, "apply_chat_template") and bool(getattr(tokenizer, "chat_template", None))
    except Exception:
        return False


def _detect_context_window_from_config(config) -> Optional[int]:
    """
    Try several common config attributes for context window.
    """
    candidate_fields = [
        "max_position_embeddings",
        "n_positions",
        "max_seq_len",
        "max_seq_length",
        "seq_length",
        "rope_scaling_max_position_embeddings",
    ]
    for name in candidate_fields:
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _detect_context_window(tokenizer, config) -> int:
    """
    Conservative context window detection. Prefer model config; fall back to tokenizer.
    Reject absurd sentinel values from tokenizer (e.g., huge ints).
    """
    cfg_val = _detect_context_window_from_config(config)
    if isinstance(cfg_val, int) and cfg_val > 0:
        return cfg_val

    tok_val = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_val, int) and 64 <= tok_val < 1_000_000:
        return tok_val

    return 2048


def _summarize_device_map(model) -> Optional[Dict[str, int]]:
    """
    Compact summary of the HF device map, if present.
    """
    try:
        device_map = getattr(model, "hf_device_map", None)
        if not device_map:
            return None
        from collections import Counter
        return dict(Counter(device_map.values()))
    except Exception:
        return None


# -----------------------------
# Public API (load/cache)
# -----------------------------

def get_or_load_model(
    model_id: str,
    *,
    min_context_tokens: int = 2048,
    prefer_fp16: bool = True,
) -> LoadedModel:
    """
    Load (or return cached) tokenizer + model with capability checks.
    """
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    cache_dir = _project_cache_dir()

    # 1) tokenizer + config
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, cache_dir=cache_dir)
        config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer/config for '{model_id}'. "
            f"Ensure the model exists and you have access (HF token if private). Underlying error: {e}"
        )

    supports_chat_template = _has_chat_template(tokenizer)
    context_tokens = _detect_context_window(tokenizer, config)
    if context_tokens < int(min_context_tokens):
        raise RuntimeError(
            f"Model '{model_id}' reports a context window of ~{context_tokens} tokens, "
            f"below the required minimum ({min_context_tokens}). Choose a larger-context model."
        )

    # 2) model class
    is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
    model_cls = AutoModelForSeq2SeqLM if is_encoder_decoder else AutoModelForCausalLM

    # 3) load weights
    dtype = torch.float16 if (prefer_fp16 and torch.cuda.is_available()) else None
    try:
        model = model_cls.from_pretrained(
            model_id,
            device_map="auto",
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            dtype=dtype,  # <- use only the modern 'dtype' parameter
        )
    except TypeError:
        model = model_cls.from_pretrained(
            model_id,
            device_map="auto",
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            torch_dtype=dtype,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights for '{model_id}': {e}")

    device_map_summary = _summarize_device_map(model)
    if device_map_summary:
        print(f"[load] {model_id} context≈{context_tokens}; device_map={device_map_summary}")
    else:
        print(f"[load] {model_id} context≈{context_tokens}; (no device_map reported)")

    # pad token guard (prevents warnings on generate)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or getattr(tokenizer, "unk_token", None)

    handle = LoadedModel(
        model_id=model_id,
        tokenizer=tokenizer,
        model=model,
        context_tokens=context_tokens,
        supports_chat_template=supports_chat_template,
        device_map_summary=device_map_summary,
    )
    _MODEL_CACHE[model_id] = handle
    return handle


def is_loaded(model_id: str) -> bool:
    return model_id in _MODEL_CACHE


def get_loaded(model_id: str) -> Optional[LoadedModel]:
    return _MODEL_CACHE.get(model_id)


def unload(model_id: str) -> bool:
    return _MODEL_CACHE.pop(model_id, None) is not None


def list_loaded() -> Dict[str, Tuple[int, bool]]:
    return {mid: (h.context_tokens, h.supports_chat_template) for mid, h in _MODEL_CACHE.items()}


# -----------------------------
# Inference utilities
# -----------------------------

def build_inputs(
    handle: LoadedModel,
    *,
    messages: Optional[List[Dict[str, str]]] = None,
    prompt: Optional[str] = None,
    max_input_tokens: int = 2048,
    reserve_tokens: int = 64,
) -> Tuple[Dict[str, torch.Tensor], int, bool]:
    """
    Tokenize inputs for generation, respecting a token budget.
    - If `messages` provided and chat template supported → use chat template.
    - Else, use plain prompt (either `prompt` arg or a simple system/user fallback).
    Returns: (tokenized_inputs, used_max_len, used_chat_template)
    """
    tokenizer = handle.tokenizer
    model = handle.model

    # Effective cap: don't exceed the model's context window
    effective_cap = max(64, min(int(max_input_tokens), int(handle.context_tokens) - int(reserve_tokens)))

    if messages and handle.supports_chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # plain fallback
        if messages and not prompt:
            # Build a minimal plain-text fallback from the last system+user turns
            system_parts = [m["content"] for m in messages if m.get("role") == "system"]
            user_parts = [m["content"] for m in messages if m.get("role") == "user"]
            system_text = system_parts[-1] if system_parts else ""
            user_text = user_parts[-1] if user_parts else ""
            text = f"<<SYSTEM>>\n{system_text}\n\n<<USER>>\n{user_text}\n\n<<ASSISTANT>>\n"
        else:
            text = prompt or ""

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=effective_cap,
        padding=False,
    )

    # Move to model's primary device when available
    target_device = getattr(model, "device", None)
    if target_device is not None:
        enc = {k: v.to(target_device) for k, v in enc.items()}

    used_chat = bool(messages and handle.supports_chat_template)
    return enc, effective_cap, used_chat


def generate_text(
    handle: LoadedModel,
    *,
    messages: Optional[List[Dict[str, str]]] = None,
    prompt: Optional[str] = None,
    max_input_tokens: int = 2048,
    max_new_tokens: int = 256,
    reserve_tokens: int = 64,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_beams: int = 1,
    return_meta: bool = False,
) -> str | Tuple[str, Dict[str, Any]]:
    """
    Single entry-point for inference. Provide either `messages` (preferred) or `prompt`.
    - Uses chat template when available; else safe plain fallback.
    - Avoids passing sampling params when not sampling (prevents warnings).
    """
    tokenizer = handle.tokenizer
    model = handle.model

    enc, used_cap, used_chat = build_inputs(
        handle,
        messages=messages,
        prompt=prompt,
        max_input_tokens=max_input_tokens,
        reserve_tokens=reserve_tokens,
    )

    input_len = enc["input_ids"].shape[1] if "input_ids" in enc else 0

    cap_limit = min(int(max_input_tokens), int(handle.context_tokens) - int(reserve_tokens))
    cap_hit = input_len >= max(8, cap_limit - 4)  # small margin to avoid off-by-one
    if cap_hit:
        msg = (
            f"(file too large for a single-pass summary — "
            f"encoded ~{input_len} tokens vs limit ~{cap_limit})."
        )
        if return_meta:
            meta = {
                "model_id": handle.model_id,
                "used_chat_template": used_chat,
                "input_cap_tokens": used_cap,
                "context_tokens": handle.context_tokens,
                "sampling": {"do_sample": do_sample, "temperature": temperature, "top_p": top_p, "num_beams": num_beams},
                "truncated": True,
            }
            return msg, meta
        return msg

    # Required ids
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        num_beams=int(num_beams),
        do_sample=bool(do_sample),
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )

    # Only include sampling controls when sampling is enabled; avoids “ignored flags” warnings
    if do_sample:
        gen_kwargs.update(temperature=float(temperature), top_p=float(top_p))

    output_ids = model.generate(**enc, **gen_kwargs)

    new_ids = output_ids[0, input_len:] if input_len > 0 else output_ids[0]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    if return_meta:
        meta = {
            "model_id": handle.model_id,
            "used_chat_template": used_chat,
            "input_cap_tokens": used_cap,
            "context_tokens": handle.context_tokens,
            "sampling": {"do_sample": do_sample, "temperature": temperature, "top_p": top_p, "num_beams": num_beams},
        }
        return text, meta
    return text
