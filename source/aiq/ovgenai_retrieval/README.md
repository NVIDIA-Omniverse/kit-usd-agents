# ovgenai-retrieval

Shared hybrid-retrieval library used by the Omniverse GenAI MCP servers and the [`ovgenai-agent-search`](https://github.com/NVIDIA-dev/ovgenai-agent-search) CLI. One retrieval core, two delivery surfaces — guarantees apples-to-apples comparison under the Kit Agent Evaluation SRD.

## Features

- **Bundle loader** — auto-detects legacy pickle / legacy JSON / new-manifest bundles; backward-compatible.
- **Hybrid retrieval** — FAISS semantic + BM25 lexical combined via Reciprocal Rank Fusion (RRF) by default; optional cross-encoder reranker.
- **Pluggable BM25 backends** — `rank_bm25` (default, pure Python), `tantivy` (opt-in, fast), `whoosh` (opt-in, pure Python).
- **Catalog index** — by-name / prefix / filter lookup for structured JSON bundles (extensions_database, usd_atlas, etc.).
- **Query expansion** — glossary-based, ported from the Kit MCPs improvement work.
- **Guardrails** — min-score threshold, dedup, context-window cap.
- **Trace emitter** — records trajectories compatible with Andrey Voroshilov's MCP Eval Tracker schema.

## Installation

```bash
pip install ovgenai-retrieval          # core (faiss-cpu, rank_bm25, numpy)
pip install ovgenai-retrieval[rerank]  # + cross-encoder reranker
pip install ovgenai-retrieval[all]     # everything
```

## Quick start

```python
from ovgenai_retrieval import HybridRetriever, CatalogIndex, load_bundle

bundle = load_bundle("/path/to/kit_v110_1_bundle")

if bundle.type in ("knowledge", "code", "settings", "extensions_semantic"):
    r = HybridRetriever(bundle, fusion="rrf", top_k=10)
    hits = r.retrieve("how do I duplicate a USD prim?", top_k=5)
    for h in hits:
        print(f"{h.score:.3f} {h.file_path}:{h.line_start}-{h.line_end}")
        print(h.content[:200])
        print()
elif bundle.type == "catalog":
    c = CatalogIndex(bundle)
    ext = c.get_by_name("omni.ui")
```

## Bundle formats supported

| Generation | Files present | Backward compat |
| :---- | :---- | :---- |
| Legacy pickle | `index.faiss`, `index.pkl` | ✅ BM25 built in-process; `bundle_type` inferred from dir name |
| Legacy JSON | `index.faiss`, `index.json` | ✅ same |
| New manifest | `manifest.json`, `index.faiss`, `index.json`, `bm25.<backend>`, `files/` | ✅ Full-featured path |

See `docs/manifest-schema.md` for the v1 manifest schema.

## License

Apache-2.0.
