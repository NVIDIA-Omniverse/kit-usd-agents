"""HybridRetriever — FAISS semantic + BM25 lexical, fused via RRF, optionally reranked.

The apples-to-apples guarantee for the MCP-vs-agent-search eval lives here: both
sides instantiate the same ``HybridRetriever`` on the same bundle.
"""

from __future__ import annotations

from typing import Optional

from ovgenai_retrieval.bm25 import build_backend
from ovgenai_retrieval.bm25.base import Bm25Index, default_tokenize
from ovgenai_retrieval.bundle import Bundle
from ovgenai_retrieval.embedder import Embedder, EmbedderFactory
from ovgenai_retrieval.fusion import rrf_fuse
from ovgenai_retrieval.guardrails import cap_context_tokens, dedup_by_doc_id, filter_min_score
from ovgenai_retrieval.hits import Hit
from ovgenai_retrieval.models import DEFAULT_EMBEDDING_MODEL
from ovgenai_retrieval.query_expansion import expand as expand_query
from ovgenai_retrieval.rerank import CrossEncoderReranker, NvidiaReranker
from ovgenai_retrieval.scoring import bm25_confidence, combined_confidence, semantic_confidence
from ovgenai_retrieval.semantic import SemanticSearcher


class HybridRetriever:
    """Composes semantic + lexical search + RRF + optional reranker."""

    def __init__(
        self,
        bundle: Bundle,
        *,
        top_k: int = 10,
        fusion: str = "rrf",  # "rrf" | "semantic_only" | "lexical_only"
        rerank: bool = False,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        rerank_backend: str = "cross_encoder",  # "cross_encoder" | "nvidia"
        rrf_k: int = 60,
        bm25_backend: str | None = None,
        embedder: Embedder | None = None,
        query_expand: bool = True,
        qe_domains: set[str] | list[str] | None = None,
        min_score: float | None = None,
        max_context_tokens: int | None = None,
        semantic_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        if bundle.type == "catalog":
            raise TypeError("HybridRetriever does not serve catalog bundles; use CatalogIndex.")
        if bundle.docstore is None:
            raise ValueError("Bundle has no docstore; cannot retrieve.")

        self._bundle = bundle
        self._top_k = top_k
        self._fusion = fusion
        self._rrf_k = rrf_k
        self._do_rerank = rerank
        self._query_expand = query_expand
        # Glossary categories (e.g. {"physics", "sensors"} for an Isaac MCP)
        # that scope which abbreviations are eligible for expansion. None
        # means "every category" — the same legacy-merge behaviour. Cast
        # to set so callers can pass a list/tuple too.
        self._qe_domains = set(qe_domains) if qe_domains else None
        self._min_score = min_score
        self._max_context_tokens = max_context_tokens
        self._semantic_weight = semantic_weight
        self._bm25_weight = bm25_weight

        # Validate and cache the row-ordered records before constructing either
        # search backend. A corrupt mapping must never be allowed to associate a
        # valid FAISS row with the wrong document.
        self._records = bundle.iter_records()

        # ---- Semantic side ------------------------------------------------
        self._semantic: Optional[SemanticSearcher] = None
        if bundle.has_faiss and fusion != "lexical_only":
            model = bundle.manifest.embeddings.model if bundle.manifest.embeddings else DEFAULT_EMBEDDING_MODEL
            if not embedder and model in {"", "unknown"}:
                raise ValueError(
                    "The legacy FAISS bundle does not record its embedding model. "
                    "Pass a matching embedder explicitly or reindex the bundle."
                )
            self._embedder = embedder or EmbedderFactory.create(model=model)
            expected_dimension = bundle.manifest.embeddings.dim if bundle.manifest.embeddings else None
            self._semantic = SemanticSearcher(
                bundle.faiss_index_path,
                self._embedder,
                expected_dimension=expected_dimension,
                expected_count=len(self._records),
            )

        # ---- Lexical side -------------------------------------------------
        self._lexical: Optional[Bm25Index] = None
        if fusion != "semantic_only":
            self._lexical = self._load_or_build_bm25(bm25_backend)

        # ---- Reranker -----------------------------------------------------
        self._reranker: Optional[object] = None
        if rerank:
            if rerank_backend == "nvidia":
                # ``rerank_model`` defaults to a sentence-transformers cross-
                # encoder ID that isn't valid for the NVIDIA endpoint, so let
                # NvidiaReranker pick its own default unless the caller
                # explicitly overrode it with an NVIDIA-style model string.
                if rerank_model.startswith("nvidia/"):
                    self._reranker = NvidiaReranker(model=rerank_model)
                else:
                    self._reranker = NvidiaReranker()
            elif rerank_backend == "cross_encoder":
                self._reranker = CrossEncoderReranker(rerank_model)
            else:
                raise ValueError(
                    f"Unknown rerank_backend: {rerank_backend!r}. " "Expected 'cross_encoder' or 'nvidia'."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[Hit]:
        k = top_k if top_k is not None else self._top_k

        query_text = expand_query(query, domains=self._qe_domains) if self._query_expand else query

        # Oversearch each side a bit so fusion has room to differentiate
        oversearch = max(k * 3, 20)

        semantic_ranking: list[tuple[int, float]] = []
        if self._semantic is not None and self._fusion != "lexical_only":
            semantic_ranking = self._semantic.top_k(query, k=oversearch)

        lexical_ranking: list[tuple[int, float]] = []
        if self._lexical is not None and self._fusion != "semantic_only":
            lexical_ranking = self._lexical.top_k(default_tokenize(query_text), k=oversearch)

        # ---- Fuse -------------------------------------------------------
        if self._fusion == "rrf":
            fused = rrf_fuse(
                [
                    [i for i, _ in semantic_ranking],
                    [i for i, _ in lexical_ranking],
                ],
                k=self._rrf_k,
                weights=[self._semantic_weight, self._bm25_weight],
            )
        elif self._fusion == "semantic_only":
            fused = [(i, s) for i, s in semantic_ranking]
        elif self._fusion == "lexical_only":
            fused = [(i, s) for i, s in lexical_ranking]
        else:
            raise ValueError(f"Unknown fusion mode: {self._fusion!r}")

        # ---- Build Hits (pre-rerank) -----------------------------------
        sem_scores = dict(semantic_ranking)
        lex_scores = dict(lexical_ranking)
        sem_ranks = {i: r + 1 for r, (i, _) in enumerate(semantic_ranking)}
        lex_ranks = {i: r + 1 for r, (i, _) in enumerate(lexical_ranking)}

        hits: list[Hit] = []
        for row_idx, fused_score in fused[:oversearch]:
            if row_idx < 0 or row_idx >= len(self._records):
                continue
            rec = self._records[row_idx]
            sem_conf = semantic_confidence(sem_scores[row_idx]) if row_idx in sem_scores else None
            lex_conf = bm25_confidence(lex_scores[row_idx]) if row_idx in lex_scores else None
            confidence = combined_confidence(sem_conf, lex_conf)

            meta = rec.metadata
            in_sem = row_idx in sem_scores
            in_lex = row_idx in lex_scores
            if in_sem and in_lex:
                match_mode = "both"
            elif in_lex:
                match_mode = "keyword"
            else:
                match_mode = "semantic"
            hits.append(
                Hit(
                    doc_id=rec.doc_id,
                    content=rec.content,
                    score=confidence if self._fusion == "rrf" else float(fused_score),
                    index_text=rec.index_text,
                    file_path=meta.get("file_path"),
                    line_start=meta.get("line_start"),
                    line_end=meta.get("line_end"),
                    section_hierarchy=list(meta.get("section_hierarchy") or []),
                    url=meta.get("url"),
                    match_mode=match_mode,
                    metadata={
                        k: v
                        for k, v in meta.items()
                        if k not in {"file_path", "line_start", "line_end", "section_hierarchy", "url"}
                    },
                    provenance={
                        "semantic_rank": sem_ranks.get(row_idx),
                        "lexical_rank": lex_ranks.get(row_idx),
                        "fused_score": float(fused_score),
                        "bundle_id": self._bundle.manifest.bundle_id,
                    },
                )
            )

        # ---- Optional reranker -----------------------------------------
        if self._reranker is not None and hits:
            # Some legacy bundles keep the searchable passage only in
            # ``index_text``. Sending an empty ``content`` string makes the
            # NVIDIA rerank API reject the entire request.
            texts = [h.content or h.index_text for h in hits]
            scores = self._reranker.score(query, texts)
            # Stable: assign new scores, re-sort
            for h, s in zip(hits, scores):
                h.provenance["rerank_score"] = float(s)
                h.score = float(s)
            hits.sort(key=lambda h: h.score, reverse=True)

        # ---- Guardrails + final top-k ----------------------------------
        hits = dedup_by_doc_id(hits)
        if self._min_score is not None:
            hits = filter_min_score(hits, self._min_score)
        hits = hits[:k]
        if self._max_context_tokens is not None:
            hits = cap_context_tokens(hits, self._max_context_tokens)
        return hits

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_or_build_bm25(self, backend_override: str | None) -> Bm25Index:
        m = self._bundle.manifest
        chosen = backend_override or (m.lexical.backend if m.lexical else "rank_bm25")
        # If a sidecar is present, load it
        if self._bundle.has_lexical_sidecar:
            sidecar = self._bundle.path / m.lexical.file  # type: ignore[union-attr]
            from ovgenai_retrieval.bm25 import build_backend as _bb  # noqa: F401

            if chosen == "rank_bm25":
                from ovgenai_retrieval.bm25.rank_bm25_backend import RankBm25Index

                try:
                    return RankBm25Index.load(sidecar)
                except FileNotFoundError:
                    # Legacy ``bm25.pkl`` or missing file — try the safe
                    # sibling (``bm25.json``) if it exists, else fall through
                    # and rebuild from the docstore.
                    json_sibling = sidecar.with_suffix(".json")
                    if json_sibling.exists():
                        return RankBm25Index.load(json_sibling)
            # Other backends: fall through to build from docstore
        # Build in-process from the docstore ``index_text`` column
        corpus = [default_tokenize(rec.index_text or rec.content) for rec in self._bundle.iter_records()]
        return build_backend(chosen, corpus)
