"""Result guardrails: min-score filter, dedup, context-window cap, truncation, sentinels."""

from __future__ import annotations

from ovgenai_retrieval.hits import Hit

DEFAULT_TRUNCATION_HINT = (
    "[Results truncated to fit the context window. "
    "Refine the query or narrow the --subfolder scope to see more specific matches.]"
)

EMPTY_RESULT_SENTINEL = "No relevant documentation found for this query."


def filter_min_score(hits: list[Hit], threshold: float) -> list[Hit]:
    """Drop hits whose ``score`` is below ``threshold``."""
    return [h for h in hits if h.score >= threshold]


def dedup_by_doc_id(hits: list[Hit]) -> list[Hit]:
    """Keep first occurrence per ``doc_id``."""
    seen: set[str] = set()
    out: list[Hit] = []
    for h in hits:
        if h.doc_id in seen:
            continue
        seen.add(h.doc_id)
        out.append(h)
    return out


def cap_context_tokens(hits: list[Hit], max_tokens: int, tokens_per_char: float = 0.25) -> list[Hit]:
    """Greedy-truncate the tail of ``hits`` to fit within ``max_tokens``.

    We approximate tokens with ``ceil(len(content) * tokens_per_char)``; the
    default 0.25 corresponds to ~4 chars per token, the conventional rough cut.
    """
    out: list[Hit] = []
    budget = max_tokens
    for h in hits:
        est = max(1, int(len(h.content) * tokens_per_char + 0.5))
        if est > budget:
            break
        budget -= est
        out.append(h)
    return out


def truncate_with_guidance(
    hits: list[Hit],
    max_chars: int,
    hint_text: str = DEFAULT_TRUNCATION_HINT,
) -> tuple[list[Hit], str | None]:
    """Cap the aggregate ``content`` character count across ``hits``.

    Returns the surviving hits plus a hint string if anything was dropped, or
    ``None`` when the full list fit within ``max_chars``. Callers are expected
    to render the hint alongside the result list (REQ-RG-1, REQ-ST-8).
    """
    if max_chars <= 0:
        return [], hint_text
    out: list[Hit] = []
    used = 0
    truncated = False
    for h in hits:
        needed = len(h.content)
        if used + needed > max_chars:
            truncated = True
            break
        used += needed
        out.append(h)
    return out, (hint_text if truncated else None)


def empty_result_sentinel() -> str:
    """Return the canonical "no hits" message (REQ-RG-2)."""
    return EMPTY_RESULT_SENTINEL


def truncate_text_with_guidance(
    text: str,
    max_chars: int,
    hint: str = DEFAULT_TRUNCATION_HINT,
) -> tuple[str, str | None]:
    """Cap a pre-formatted result string at ``max_chars`` (REQ-RG-1).

    Sibling to :func:`truncate_with_guidance`, which operates on a list of
    :class:`Hit`. Use this variant when the MCP wrapper has already
    rendered hits into a single string (the common shape for settings
    results). Returns ``(text, None)`` when the input fit, or
    ``(truncated_text, hint)`` otherwise; callers decide how to render
    the hint (prepend vs. append).
    """
    if max_chars <= 0:
        return "", hint
    if len(text) <= max_chars:
        return text, None
    # Cut on a line boundary if one exists near the cap so we don't
    # slice mid-token.
    head = text[:max_chars]
    nl = head.rfind("\n")
    if nl >= int(max_chars * 0.8):
        head = head[:nl]
    return head, hint


def filter_by_subfolder(hits: list[Hit], prefix: str) -> list[Hit]:
    """Keep only hits whose ``file_path`` starts with ``prefix`` (REQ-ST-7)."""
    if not prefix:
        return hits
    p = prefix.rstrip("/") + "/"
    return [h for h in hits if h.file_path and (h.file_path.startswith(p) or h.file_path == prefix.rstrip("/"))]
