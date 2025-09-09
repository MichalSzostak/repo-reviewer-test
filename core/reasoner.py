from __future__ import annotations
from typing import List, Dict

def _cap_list(items: List[str], limit: int) -> List[str]:
    return items[:limit] if len(items) > limit else items

def reason_over_summaries(
    ticket_text: str,
    summaries: List[Dict],
    tok,
    model,
    max_input_tokens: int = 2048,
    max_new_tokens: int = 384,
) -> str:
    """
    Minimal reasoner: feed the ticket + all file summaries with a strict format.
    No parsing or ranking on our side; the model picks relevant files itself.
    """
    # Build a compact context of path + summary (already bullets).
    # Keep it simple and let the model choose what's relevant.
    ctx_lines = []
    for item in summaries:
        path = item.get("path", "???")
        summ = (item.get("summary") or "").strip()
        if not summ:
            continue
        ctx_lines.append(f"File: {path}\n{summ}\n")
    ctx_block = "\n".join(ctx_lines).strip()

    system_msg = (
        "You are a senior engineer. Based on the ticket and the provided file summaries, "
        "produce EXACTLY the following Markdown sections and nothing else:\n\n"
        "# What to read first (top files)\n"
        "List 3-5 file paths. For each:\n"
        "• one-sentence rationale (why this file matters for the ticket)\n"
        "• then 1-3 short bullets lifted/adapted from its summary (no code/CLI)\n\n"
        "# Prerequisites\n"
        "3-6 bullets of topics/libraries/services a contributor should review\n\n"
        "# Work plan\n"
        "3-7 concrete steps, call out risks where relevant\n\n"
        "Rules: Do NOT repeat or quote the ticket or the summaries. Do NOT ask questions. "
        "Do NOT include code blocks or shell commands. Be concise."
    )

    user_msg = (
        f"Ticket:\n{ticket_text}\n\n"
        "File summaries:\n"
        f"{ctx_block}\n"
    )

    try:
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": system_msg},
             {"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = f"<<SYSTEM>>\n{system_msg}\n\n<<USER>>\n{user_msg}\n\n<<ASSISTANT>>\n"

    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=int(max_input_tokens), padding=False)
    input_len = enc["input_ids"].shape[1]  # <-- length of the prompt
    if hasattr(model, "device") and getattr(model.device, "type", None) == "cuda":
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
    new_ids = gen[0, input_len:]  # <-- keep only the completion
    return tok.decode(new_ids, skip_special_tokens=True).strip()
