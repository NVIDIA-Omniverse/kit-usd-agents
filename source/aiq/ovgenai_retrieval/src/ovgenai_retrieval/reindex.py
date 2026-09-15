"""Rebuild safe JSON-backed FAISS bundles with a new embedding model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

import faiss
import numpy as np
from ovgenai_retrieval._compat import DocStore, load_json_docstore
from ovgenai_retrieval.manifest import Manifest
from ovgenai_retrieval.models import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_EMBEDDING_MODEL
from ovgenai_retrieval.upgrade import build_bm25_sidecar_from_bundle


class DocumentEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingCache:
    """SQLite-backed vector cache shared across bundles and interrupted runs."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL
            )
            """
        )
        self._db.commit()

    @staticmethod
    def key(model: str, text: str) -> str:
        payload = f"{model}\0{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, model: str, text: str, dimension: int) -> np.ndarray | None:
        row = self._db.execute(
            "SELECT dimension, vector FROM embeddings WHERE cache_key = ?",
            (self.key(model, text),),
        ).fetchone()
        if row is None or int(row[0]) != dimension:
            return None
        vector = np.frombuffer(row[1], dtype=np.float32).copy()
        return vector if vector.size == dimension else None

    def put(self, model: str, text: str, vector: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(vector, dtype=np.float32)
        self._db.execute(
            """
            INSERT OR REPLACE INTO embeddings(cache_key, model, dimension, vector)
            VALUES (?, ?, ?, ?)
            """,
            (self.key(model, text), model, int(contiguous.size), contiguous.tobytes()),
        )

    def commit(self) -> None:
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def discover_bundles(roots: Iterable[str | os.PathLike]) -> list[Path]:
    """Return every JSON-backed FAISS bundle found below ``roots``."""
    found: set[Path] = set()
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        candidates = [root_path] if root_path.is_dir() else []
        candidates.extend(path.parent for path in root_path.rglob("index.faiss"))
        for candidate in candidates:
            if (candidate / "index.faiss").is_file() and (candidate / "index.json").is_file():
                found.add(candidate.resolve())
    return sorted(found)


def _ordered_records(docstore: DocStore) -> list[Any]:
    if len(docstore.records) != len(docstore.id_to_faiss_index):
        raise ValueError(
            "docstore and index_to_docstore_id counts differ: "
            f"{len(docstore.records)} != {len(docstore.id_to_faiss_index)}"
        )
    by_index: dict[int, Any] = {}
    for record in docstore.records:
        index = docstore.id_to_faiss_index.get(record.doc_id)
        if index is not None:
            by_index[index] = record
    expected = list(range(len(by_index)))
    if len(by_index) != len(docstore.records) or sorted(by_index) != expected:
        raise ValueError("index_to_docstore_id must contain contiguous rows starting at zero")
    return [by_index[index] for index in expected]


def _embed_batch(
    embedder: DocumentEmbedder,
    texts: list[str],
    retries: int,
) -> list[list[float]]:
    for attempt in range(retries + 1):
        try:
            vectors = embedder.embed_documents(texts)
            if len(vectors) != len(texts):
                raise ValueError(f"embedding service returned {len(vectors)} vectors for {len(texts)} texts")
            return vectors
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(30.0, 2.0**attempt))
    raise AssertionError("unreachable")


def _validate_vector(vector: list[float] | np.ndarray, dimension: int) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or array.size != dimension:
        raise ValueError(f"embedding dimension mismatch: got shape {array.shape}, expected ({dimension},)")
    if not np.isfinite(array).all():
        raise ValueError("embedding service returned a non-finite vector")
    if float(np.linalg.norm(array)) == 0.0:
        raise ValueError("embedding service returned a zero vector")
    return array


def _manifest_data(bundle_path: Path) -> dict[str, Any]:
    manifest_path = bundle_path / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return Manifest.synthesize_from_legacy(bundle_path).to_dict()


def _write_metadata(
    bundle_path: Path,
    manifest: dict[str, Any],
    records: list[Any],
    model: str,
    dimension: int,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    manifest["schema_version"] = "1"
    manifest.setdefault("bundle_id", bundle_path.name)
    manifest["created_at"] = timestamp
    manifest["created_by"] = "ovgenai-retrieval.reindex"
    manifest["embeddings"] = {
        "backend": "faiss",
        "file": "index.faiss",
        "model": model,
        "dim": dimension,
        "metric": "cosine",
    }
    manifest["docstore"] = {"format": "json", "file": "index.json"}
    if (bundle_path / "bm25.json").exists():
        manifest["lexical"] = {
            "backend": "rank_bm25",
            "file": "bm25.json",
            "tokenizer": "whitespace+lowercase",
            "k1": 1.2,
            "b": 0.75,
        }
    all_have_index_text = bool(records) and all(record.index_text for record in records)
    any_have_index_text = any(record.index_text for record in records)
    if all_have_index_text:
        index_key = "index_text"
    elif any_have_index_text:
        index_key = "index_text_or_page_content"
    else:
        index_key = "page_content"
    metadata_fields = sorted({key for record in records for key in record.metadata})
    manifest["chunks"] = {
        "count": len(records),
        "index_key": index_key,
        "metadata_fields": metadata_fields,
    }
    (bundle_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metadata_path = bundle_path / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(metadata, dict):
            metadata["embedding_model"] = model
            metadata["embedding_dimension"] = dimension
            metadata["reindexed_at"] = timestamp
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )


def reindex_bundle(
    bundle_path: str | os.PathLike,
    embedder: DocumentEmbedder,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    batch_size: int = 64,
    workers: int = 4,
    retries: int = 5,
    cache: EmbeddingCache | None = None,
) -> dict[str, Any]:
    """Re-embed one bundle while preserving its document-to-row mapping."""
    path = Path(bundle_path).resolve()
    if batch_size < 1 or workers < 1 or retries < 0:
        raise ValueError("batch_size/workers must be positive and retries cannot be negative")
    if not (path / "index.json").is_file() or not (path / "index.faiss").is_file():
        raise FileNotFoundError(f"Expected index.faiss and index.json in {path}")

    manifest = _manifest_data(path)
    docstore = load_json_docstore(path / "index.json")
    records = _ordered_records(docstore)
    if not records:
        raise ValueError(f"Cannot reindex an empty bundle: {path}")
    texts = [record.index_text or record.content for record in records]
    if any(not text for text in texts):
        raise ValueError(f"Bundle contains empty index text: {path}")

    owned_cache = cache is None
    active_cache = cache or EmbeddingCache(path / ".reindex-cache.sqlite3")
    index = faiss.IndexFlatL2(dimension)
    window_size = batch_size * workers
    try:
        for start in range(0, len(texts), window_size):
            window = texts[start : start + window_size]
            missing: dict[str, str] = {}
            for text in window:
                if active_cache.get(model, text, dimension) is None:
                    missing.setdefault(active_cache.key(model, text), text)

            batches = [
                list(missing.values())[offset : offset + batch_size] for offset in range(0, len(missing), batch_size)
            ]
            if batches:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(_embed_batch, embedder, batch, retries): batch for batch in batches}
                    for future in as_completed(futures):
                        batch = futures[future]
                        vectors = future.result()
                        for text, vector in zip(batch, vectors):
                            active_cache.put(model, text, _validate_vector(vector, dimension))
                active_cache.commit()

            matrix = np.vstack([active_cache.get(model, text, dimension) for text in window]).astype(
                np.float32, copy=False
            )
            index.add(matrix)
            print(f"[{path}] embedded {min(start + len(window), len(texts))}/{len(texts)}")

        if int(index.ntotal) != len(records):
            raise ValueError(f"rebuilt index has {index.ntotal} rows for {len(records)} records")
        temp_path = path / f".index.faiss.{os.getpid()}.tmp"
        try:
            faiss.write_index(index, str(temp_path))
            os.replace(temp_path, path / "index.faiss")
        finally:
            temp_path.unlink(missing_ok=True)

        # A source refresh can change the docstore row count. Install the
        # matching manifest before the BM25 builder reloads and validates the
        # bundle, then write it again after BM25 exists to record the sidecar.
        _write_metadata(path, manifest, records, model, dimension)
        bm25_result = build_bm25_sidecar_from_bundle(
            path,
            force=True,
            allow_pickle=False,
        )
        if bm25_result["status"] == "skipped":
            raise RuntimeError(f"BM25 rebuild failed for {path}: {bm25_result.get('reason')}")
        _write_metadata(path, manifest, records, model, dimension)
    finally:
        if owned_cache:
            active_cache.close()
            (path / ".reindex-cache.sqlite3").unlink(missing_ok=True)

    return {"path": str(path), "rows": len(records), "dimension": dimension, "model": model}


def _already_current(path: Path, model: str, dimension: int) -> bool:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        info = manifest["embeddings"]
        index = faiss.read_index(str(path / "index.faiss"))
        mapping = load_json_docstore(path / "index.json").id_to_faiss_index
        return (
            info.get("model") == model
            and int(info.get("dim", -1)) == dimension
            and int(index.d) == dimension
            and int(index.ntotal) == len(mapping)
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", default=[], help="Bundle directory (repeatable).")
    parser.add_argument("--root", action="append", default=[], help="Discovery root (repeatable).")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--dimension", type=int, default=DEFAULT_EMBEDDING_DIMENSION)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--cache", type=Path, default=Path(".ovgenai-reindex-cache.sqlite3"))
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    roots = [*args.bundle, *args.root]
    if not roots:
        parser.error("provide at least one --bundle or --root")
    bundles = discover_bundles(roots)
    if not bundles:
        print("No JSON-backed FAISS bundles found.", file=sys.stderr)
        return 2
    for path in bundles:
        state = "current" if _already_current(path, args.model, args.dimension) else "reindex"
        print(f"[{state}] {path}")
    if args.dry_run:
        return 0

    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.base_url:
        parser.error(f"{args.api_key_env} is not set and --base-url was not provided")
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    embedder_kwargs: dict[str, Any] = {
        "model": args.model,
        "truncate": "END",
        "max_batch_size": args.batch_size,
    }
    if api_key:
        embedder_kwargs["api_key"] = api_key
    if args.base_url:
        embedder_kwargs["base_url"] = args.base_url
    embedder = NVIDIAEmbeddings(**embedder_kwargs)

    completed = 0
    with EmbeddingCache(args.cache) as cache:
        for path in bundles:
            if not args.force and _already_current(path, args.model, args.dimension):
                continue
            reindex_bundle(
                path,
                embedder,
                model=args.model,
                dimension=args.dimension,
                batch_size=args.batch_size,
                workers=args.workers,
                retries=args.retries,
                cache=cache,
            )
            completed += 1
    print(f"Reindexed {completed} bundle(s); {len(bundles) - completed} already current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
