from ovgenai_retrieval.guardrails import (
    DEFAULT_TRUNCATION_HINT,
    cap_context_tokens,
    dedup_by_doc_id,
    empty_result_sentinel,
    filter_by_subfolder,
    filter_min_score,
    truncate_text_with_guidance,
    truncate_with_guidance,
)
from ovgenai_retrieval.hits import Hit


def _h(doc_id: str, content: str, score: float = 1.0, file_path: str | None = None) -> Hit:
    return Hit(doc_id=doc_id, content=content, score=score, file_path=file_path)


def test_filter_min_score():
    hits = [_h("a", "x", 0.9), _h("b", "y", 0.3), _h("c", "z", 0.6)]
    assert [h.doc_id for h in filter_min_score(hits, 0.5)] == ["a", "c"]


def test_dedup():
    hits = [_h("a", "x"), _h("b", "y"), _h("a", "x2")]
    out = dedup_by_doc_id(hits)
    assert [h.doc_id for h in out] == ["a", "b"]


def test_cap_context_tokens_truncates_tail():
    hits = [_h("a", "x" * 1000), _h("b", "x" * 1000), _h("c", "x" * 1000)]
    out = cap_context_tokens(hits, max_tokens=500)
    # 1000 chars * 0.25 = 250 tokens/hit; budget 500 fits 2
    assert len(out) == 2
    assert [h.doc_id for h in out] == ["a", "b"]


def test_truncate_with_guidance_fits():
    hits = [_h("a", "x" * 100), _h("b", "x" * 100)]
    out, hint = truncate_with_guidance(hits, max_chars=1000)
    assert len(out) == 2
    assert hint is None


def test_truncate_with_guidance_drops_tail():
    hits = [_h("a", "x" * 400), _h("b", "x" * 400), _h("c", "x" * 400)]
    out, hint = truncate_with_guidance(hits, max_chars=900)
    assert [h.doc_id for h in out] == ["a", "b"]
    assert hint == DEFAULT_TRUNCATION_HINT


def test_truncate_with_guidance_custom_hint():
    hits = [_h("a", "x" * 100), _h("b", "x" * 100)]
    out, hint = truncate_with_guidance(hits, max_chars=50, hint_text="Too much!")
    assert out == []
    assert hint == "Too much!"


def test_truncate_with_guidance_zero_budget():
    hits = [_h("a", "x")]
    out, hint = truncate_with_guidance(hits, max_chars=0)
    assert out == []
    assert hint == DEFAULT_TRUNCATION_HINT


def test_empty_result_sentinel_is_stable():
    s = empty_result_sentinel()
    assert isinstance(s, str) and s
    assert "no relevant" in s.lower()


def test_filter_by_subfolder_prefix_match():
    hits = [
        _h("a", "x", file_path="kit/extensions/omni.ui/README.md"),
        _h("b", "x", file_path="kit/settings/renderer.md"),
        _h("c", "x", file_path="kit/extensions/carb/index.md"),
    ]
    out = filter_by_subfolder(hits, "kit/extensions")
    assert [h.doc_id for h in out] == ["a", "c"]


def test_filter_by_subfolder_handles_trailing_slash():
    hits = [_h("a", "x", file_path="a/b/c.md")]
    assert filter_by_subfolder(hits, "a/b/") == hits
    assert filter_by_subfolder(hits, "a/b") == hits


def test_filter_by_subfolder_skips_missing_path():
    hits = [_h("a", "x", file_path=None), _h("b", "x", file_path="kit/x.md")]
    out = filter_by_subfolder(hits, "kit")
    assert [h.doc_id for h in out] == ["b"]


def test_filter_by_subfolder_empty_prefix_passthrough():
    hits = [_h("a", "x", file_path="kit/x.md")]
    assert filter_by_subfolder(hits, "") == hits


# ---- truncate_text_with_guidance (REQ-RG-1 text variant) -------------------


def test_truncate_text_fits():
    text = "short"
    out, hint = truncate_text_with_guidance(text, 100)
    assert out == text
    assert hint is None


def test_truncate_text_drops_tail():
    text = "a" * 1500
    out, hint = truncate_text_with_guidance(text, 500)
    assert len(out) <= 500
    assert hint == DEFAULT_TRUNCATION_HINT


def test_truncate_text_zero_budget():
    out, hint = truncate_text_with_guidance("anything", 0)
    assert out == ""
    assert hint == DEFAULT_TRUNCATION_HINT


def test_truncate_text_custom_hint():
    text = "x" * 200
    out, hint = truncate_text_with_guidance(text, 50, hint="use a prefix_filter!")
    assert len(out) <= 50
    assert hint == "use a prefix_filter!"


def test_truncate_text_prefers_line_boundary():
    # 100 lines of varying length; cap well under total. Should cut on a newline.
    lines = [f"line {i:03d}" for i in range(100)]
    text = "\n".join(lines)
    out, hint = truncate_text_with_guidance(text, 400)
    assert hint is not None
    # Truncation should land on a newline boundary when one is near the cap.
    assert not out.endswith(" line")  # no mid-line cut
    assert out.count("\n") >= 1
