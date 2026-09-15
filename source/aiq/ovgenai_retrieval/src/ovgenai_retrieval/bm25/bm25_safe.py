"""Safe BM25 (de)serialization — JSON instead of pickle.

Mirrors the ``faiss_safe`` pattern from kit-usd-agents
(``source/aiq/*/src/*/utils/faiss_safe.py``): the .pkl path is banned for
the same pickle-loading-flagged reason. This module stores
the ``rank_bm25.BM25Okapi`` state as a plain JSON document containing only
strings, numbers, and mappings — safe to load anywhere without pickle.

Format version 1::

    {
      "format_version": 1,
      "backend": "rank_bm25",
      "k1": 1.2,
      "b": 0.75,
      "epsilon": 0.25,
      "corpus_size": N,
      "avgdl": float,
      "average_idf": float,
      "doc_freqs": [{"<token>": <count>, ...}, ...],  # per-doc term counts
      "idf":       {"<token>": <weight>, ...},
      "doc_len":   [...]
    }

Callers should use :func:`save_bm25_json` and :func:`load_bm25_json`
exclusively. :func:`convert_pkl_to_json` migrates legacy sidecars.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1


def _bm25_to_safe_dict(bm25: Any) -> dict[str, Any]:
    """Extract serializable state from a ``rank_bm25.BM25Okapi``."""
    return {
        "format_version": FORMAT_VERSION,
        "backend": "rank_bm25",
        "k1": float(getattr(bm25, "k1", 1.2)),
        "b": float(getattr(bm25, "b", 0.75)),
        "epsilon": float(getattr(bm25, "epsilon", 0.25)),
        "corpus_size": int(getattr(bm25, "corpus_size", 0)),
        "avgdl": float(getattr(bm25, "avgdl", 0.0)),
        "average_idf": float(getattr(bm25, "average_idf", 0.0)),
        "doc_freqs": [dict(d) for d in getattr(bm25, "doc_freqs", [])],
        "idf": dict(getattr(bm25, "idf", {})),
        "doc_len": list(getattr(bm25, "doc_len", [])),
    }


def save_bm25_json(bm25: Any, path: str | Path) -> None:
    """Write the BM25Okapi state as a safe JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_bm25_to_safe_dict(bm25), f, ensure_ascii=False)


def load_bm25_json(path: str | Path) -> Any:
    """Reconstruct a ``BM25Okapi`` from its JSON sidecar.

    Skips the constructor's ``_initialize`` pass (which expects a corpus)
    by allocating a bare instance and populating the persisted fields
    directly. The returned object answers ``get_scores(query_tokens)`` the
    same way a freshly-built instance would.
    """
    from rank_bm25 import BM25Okapi

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fmt = data.get("format_version")
    if fmt != FORMAT_VERSION:
        raise ValueError(f"Unsupported bm25_safe format_version={fmt!r}; expected {FORMAT_VERSION}")
    if data.get("backend") not in (None, "rank_bm25"):
        raise ValueError(
            f"bm25_safe backend mismatch: got {data.get('backend')!r}, "
            f"only 'rank_bm25' is supported by load_bm25_json"
        )

    bm25 = BM25Okapi.__new__(BM25Okapi)
    bm25.k1 = float(data.get("k1", 1.2))
    bm25.b = float(data.get("b", 0.75))
    bm25.epsilon = float(data.get("epsilon", 0.25))
    bm25.corpus_size = int(data.get("corpus_size", 0))
    bm25.avgdl = float(data.get("avgdl", 0.0))
    bm25.average_idf = float(data.get("average_idf", 0.0))
    bm25.doc_freqs = [dict(d) for d in data.get("doc_freqs", [])]
    bm25.idf = dict(data.get("idf", {}))
    bm25.doc_len = list(data.get("doc_len", []))
    bm25.tokenizer = None
    return bm25


def convert_pkl_to_json(pkl_path: str | Path, json_path: str | Path | None = None, *, delete_pkl: bool = False) -> bool:
    """Migrate a legacy ``bm25.pkl`` to ``bm25.json`` format.

    Mirrors :func:`ovgenai_retrieval._compat.load_pickle_docstore`'s gated
    pickle handling: only runs if the caller has already decided the source
    is trusted. Returns ``True`` if a conversion happened, ``False`` if the
    JSON already existed (idempotent).
    """
    import pickle  # local import — kept out of the safe path

    pkl_path = Path(pkl_path)
    json_path = Path(json_path) if json_path else pkl_path.with_suffix(".json")
    if json_path.exists():
        logger.info("bm25_safe: %s already exists — skipping", json_path)
        return False
    if not pkl_path.exists():
        raise FileNotFoundError(f"No legacy {pkl_path} to convert")

    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    # Payload shape produced by build_bm25_sidecar_from_bundle: {"bm25": BM25Okapi, "corpus_len": int}
    if isinstance(payload, dict) and "bm25" in payload:
        bm25 = payload["bm25"]
    else:
        bm25 = payload  # older variant where bm25 was pickled directly

    save_bm25_json(bm25, json_path)
    if delete_pkl:
        pkl_path.unlink()
        logger.info("bm25_safe: removed %s", pkl_path)
    return True
