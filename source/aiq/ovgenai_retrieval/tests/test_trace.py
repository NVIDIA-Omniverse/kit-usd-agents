import json
from pathlib import Path

from ovgenai_retrieval.hits import Hit
from ovgenai_retrieval.trace import TraceEmitter


def test_trace_emits_ndjson(tmp_path: Path):
    out = tmp_path / "trace.ndjson"
    em = TraceEmitter(out, condition="FS+skills")
    with em.time_tool("search_kit_knowledge", "q1", bundle_id="bid") as slot:
        slot["hits"] = [Hit(doc_id="a", content="c", score=0.9, file_path="p.md")]
    with em.time_tool("search_kit_knowledge", "q2", bundle_id="bid") as slot:
        slot["hits"] = []

    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    assert rec1["tool"] == "search_kit_knowledge"
    assert rec1["query"] == "q1"
    assert rec1["condition"] == "FS+skills"
    assert rec1["bundle_id"] == "bid"
    assert rec1["hits"][0]["doc_id"] == "a"
    assert "latency_ms" in rec1


def test_trace_noop_when_path_none():
    em = TraceEmitter(None)
    with em.time_tool("tool", "q") as slot:
        slot["hits"] = []
    # No assertion needed — just confirming it doesn't crash
