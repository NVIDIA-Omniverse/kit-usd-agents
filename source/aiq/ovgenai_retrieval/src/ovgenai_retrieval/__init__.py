"""ovgenai-retrieval — shared hybrid-retrieval library for Omniverse agent tools."""

from ovgenai_retrieval.bundle import Bundle, load_bundle
from ovgenai_retrieval.catalog import CatalogIndex
from ovgenai_retrieval.hits import Hit
from ovgenai_retrieval.hybrid import HybridRetriever
from ovgenai_retrieval.manifest import Manifest
from ovgenai_retrieval.models import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANK_ENDPOINT,
    DEFAULT_RERANK_MODEL,
)
from ovgenai_retrieval.sanitize import sanitize_identifier, sanitize_query
from ovgenai_retrieval.shim import hits_to_documents, hits_to_documents_with_scores, maybe_load_hybrid
from ovgenai_retrieval.upgrade import (
    build_bm25_sidecar_from_bundle,
    discover_bundles,
    upgrade_bundle,
    write_or_update_manifest,
)

__all__ = [
    "Bundle",
    "CatalogIndex",
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_RERANK_ENDPOINT",
    "DEFAULT_RERANK_MODEL",
    "Hit",
    "HybridRetriever",
    "Manifest",
    "build_bm25_sidecar_from_bundle",
    "discover_bundles",
    "hits_to_documents",
    "hits_to_documents_with_scores",
    "load_bundle",
    "maybe_load_hybrid",
    "sanitize_identifier",
    "sanitize_query",
    "upgrade_bundle",
    "write_or_update_manifest",
]

__version__ = "0.2.0"
