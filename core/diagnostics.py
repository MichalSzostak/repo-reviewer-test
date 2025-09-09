from __future__ import annotations

def measure_text_tokens(text: str, tokenizer) -> int:
    """
    Count tokens for `text` using the provided tokenizer.
    """
    return len(tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0])

def chunk_plan(text: str, tokenizer, max_tokens: int, overlap: int) -> dict:
    """
    Return a simple plan for how the text would be chunked at (max_tokens, overlap).
    Does not allocate model memory; only tokenizes once to count.
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=False)
    ids = enc["input_ids"][0].tolist()
    total = len(ids)

    if total == 0:
        return {"total_tokens": 0, "num_chunks": 0, "chunks": []}

    chunks = []
    start = 0
    while start < total:
        end = min(start + max_tokens, total)
        size = end - start
        chunks.append({"start_token": start, "end_token": end, "size": size})
        if end == total:
            break
        start = max(0, end - overlap)

    return {
        "total_tokens": total,
        "num_chunks": len(chunks),
        "chunks": chunks,
    }

def gpu_memory_info() -> dict:
    """
    Best-effort GPU VRAM stats (CUDA only). Safe on CPU-only machines.
    Returns {"available": bool, "current_mb": int, "reserved_mb": int, "total_mb": int} or zeros.
    """
    info = {"available": False, "current_mb": 0, "reserved_mb": 0, "total_mb": 0}
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            info["available"] = True
            info["current_mb"] = int(torch.cuda.memory_allocated(dev) / (1024 * 1024))
            info["reserved_mb"] = int(torch.cuda.memory_reserved(dev) / (1024 * 1024))
            info["total_mb"] = int(torch.cuda.get_device_properties(dev).total_memory / (1024 * 1024))
    except Exception:
        pass
    return info
