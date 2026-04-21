import ollama

SYSTEM_PROMPT = (
    "You are a precise analyst. Given a transcript, produce Markdown with these sections: "
    "**Summary** (3-5 sentences), **Key Points** (bulleted), **Topics** (comma-separated tags), "
    "**Notable Quotes** (2-3 direct quotes if present, otherwise omit). "
    "Do not invent facts. If the transcript is partial, say so."
)

CHUNK_SYSTEM_PROMPT = (
    "You are summarizing one chunk of a longer transcript. "
    "Produce a concise bulleted summary of the facts, claims, and notable quotes in this chunk only. "
    "Do not add introductions or conclusions."
)

REDUCE_SYSTEM_PROMPT = (
    "You are given bulleted summaries of consecutive chunks of a single transcript. "
    "Merge them into a final analysis. Output Markdown with sections: "
    "**Summary** (3-5 sentences), **Key Points** (bulleted), **Topics** (comma-separated tags), "
    "**Notable Quotes** (2-3 if present)."
)

CHUNK_CHAR_LIMIT = 24_000


def analyze(transcript: str, model: str = "qwen3.5:latest") -> str:
    if len(transcript) <= CHUNK_CHAR_LIMIT:
        return _chat(SYSTEM_PROMPT, transcript, model)

    chunks = _chunk(transcript, CHUNK_CHAR_LIMIT)
    partials = [_chat(CHUNK_SYSTEM_PROMPT, c, model) for c in chunks]
    joined = "\n\n---\n\n".join(
        f"## Chunk {i + 1}\n{p}" for i, p in enumerate(partials)
    )
    return _chat(REDUCE_SYSTEM_PROMPT, joined, model)


def _chat(system: str, user: str, model: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response["message"]["content"].strip()


def _chunk(text: str, limit: int) -> list[str]:
    # Split on paragraph boundaries where possible, fall back to hard slices.
    paragraphs = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para) + 1
        if current_len + para_len > limit and current:
            chunks.append("\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len
    if current:
        chunks.append("\n".join(current))

    # Any chunk still over the limit (a single massive paragraph) gets hard-sliced.
    final: list[str] = []
    for c in chunks:
        if len(c) <= limit:
            final.append(c)
        else:
            for i in range(0, len(c), limit):
                final.append(c[i : i + limit])
    return final
