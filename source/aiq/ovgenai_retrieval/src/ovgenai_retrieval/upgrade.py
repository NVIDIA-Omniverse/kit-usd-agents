"""In-place upgrade of legacy bundles.

Legacy bundles (produced by the pre-manifest-v1 rag-prep pipeline, or by
vendored kit-usd-agents data before the Phase-4 builder upgrades) contain only
``index.faiss`` + ``index.pkl`` (or ``index.json``). They work, but:

- No BM25 sidecar, so ``HybridRetriever`` has to rebuild BM25 in-process on
  every load. That's slow on large corpora (tens of seconds for a 60k-chunk
  bundle).
- No ``manifest.json``, so the loader has to heuristic-guess bundle_type,
  embedder model, and chunk metadata_fields.

This module converts such bundles in place with zero re-embedding and zero
LLM calls. Only chunk text (already stored in the docstore) is needed.

BM25 sidecar format is ``bm25.json`` — **not pickle**. Pickle is banned
across this stack (security scanners flag ``*.pkl`` files); this
mirrors the faiss_safe migration in kit-usd-agents. Legacy ``bm25.pkl``
bundles are migrated automatically on upgrade when encountered.

CLI:

    ovgenai-upgrade-bundle --bundle /path/to/legacy_bundle
    ovgenai-upgrade-bundle --root /path/to/many/bundles   # walk + upgrade each
    ovgenai-upgrade-bundle --bundle /path/... --force    # overwrite existing sidecar

API:

    from ovgenai_retrieval.upgrade import upgrade_bundle
    result = upgrade_bundle("/path/to/bundle", allow_pickle=True)
    # → {"bm25": {"status": "created"|"migrated"|"exists", "path": ...}, ...}
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from ovgenai_retrieval.bm25.base import default_tokenize
from ovgenai_retrieval.bm25.bm25_safe import convert_pkl_to_json, save_bm25_json
from ovgenai_retrieval.bundle import load_bundle
from ovgenai_retrieval.manifest import ChunksInfo, LexicalInfo, Manifest

SAFE_BM25_FILENAME = "bm25.json"
LEGACY_BM25_FILENAME = "bm25.pkl"


def build_bm25_sidecar_from_bundle(
    bundle_path: str | os.PathLike,
    *,
    filename: str = SAFE_BM25_FILENAME,
    k1: float = 1.2,
    b: float = 0.75,
    force: bool = False,
    allow_pickle: bool = True,
    migrate_legacy_pkl: bool = True,
) -> dict[str, Any]:
    """Build a rank_bm25 sidecar from an existing bundle's docstore.

    Writes ``bm25.json`` by default (pickle output is refused for the same
    security-scanner rationale as kit-usd-agents's faiss_safe migration refuses ``*.pkl``). If a legacy
    ``bm25.pkl`` already exists, it is migrated to ``bm25.json`` in place
    and the pickle is removed — returns status ``"migrated"``.

    Returns a dict with ``status`` ("created" | "migrated" | "exists" |
    "skipped") plus paths.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return {"status": "skipped", "reason": "rank_bm25 not installed"}

    if filename.endswith(".pkl"):
        return {
            "status": "skipped",
            "reason": "pickle BM25 sidecars are banned for security; pass filename=bm25.json",
        }

    bdir = Path(bundle_path)
    sidecar = bdir / filename
    legacy = bdir / LEGACY_BM25_FILENAME

    # Case 1: the JSON sidecar already exists and we're not forcing.
    if sidecar.exists() and not force:
        # If a legacy .pkl happens to sit next to it, remove the pkl silently
        # so the bundle ends up single-sidecar JSON-only. This keeps the security-scan posture
        # scans of vendored bundles clean after a one-shot upgrade.
        if legacy.exists():
            try:
                legacy.unlink()
            except OSError:
                pass
        return {"status": "exists", "path": str(sidecar)}

    # Case 2: only the legacy .pkl exists — migrate it in place.
    if not sidecar.exists() and legacy.exists() and migrate_legacy_pkl:
        try:
            convert_pkl_to_json(legacy, sidecar, delete_pkl=True)
            return {"status": "migrated", "path": str(sidecar), "legacy": str(legacy)}
        except Exception:
            # Fall through to rebuilding from the trusted docstore.
            pass

    bundle = load_bundle(bdir, allow_pickle=allow_pickle)
    if bundle.type == "catalog":
        return {
            "status": "skipped",
            "reason": "catalog bundles don't use BM25",
            "path": str(bdir),
        }
    if bundle.docstore is None:
        return {"status": "skipped", "reason": "bundle has no docstore", "path": str(bdir)}

    records = bundle.iter_records()
    corpus = [default_tokenize(r.index_text or r.content) for r in records]
    try:
        bm25 = BM25Okapi(corpus, k1=k1, b=b)
    except TypeError:
        bm25 = BM25Okapi(corpus)
        bm25.k1 = k1
        bm25.b = b

    save_bm25_json(bm25, sidecar)
    # If a legacy .pkl was left behind, clean it up now that JSON is present.
    if legacy.exists():
        try:
            legacy.unlink()
        except OSError:
            pass
    return {"status": "created", "path": str(sidecar), "corpus_len": len(corpus)}


def write_or_update_manifest(
    bundle_path: str | os.PathLike,
    *,
    embedder_name: str = "unknown",
    force: bool = False,
) -> dict[str, Any]:
    """Write a manifest.json (v1) for a bundle that lacks one.

    Inspects the bundle directory to detect docstore format and BM25 sidecar
    presence, and synthesizes an accurate manifest. If manifest.json already
    exists and ``force=False``, no-op.
    """
    bdir = Path(bundle_path)
    manifest_path = bdir / "manifest.json"
    if manifest_path.exists() and not force:
        return {"status": "exists", "path": str(manifest_path)}

    # Use the library's legacy synthesizer as the starting point.
    m = Manifest.synthesize_from_legacy(bdir)

    # Enrich with what the caller explicitly knows. A legacy FAISS binary does
    # not identify the model that produced it, so the safe default is unknown.
    if m.embeddings is not None:
        m.embeddings.model = embedder_name

    # Lexical sidecar — stamp if present on disk. Prefer JSON over the
    # legacy pickle, even if both happen to coexist during a migration.
    bm25 = None
    for candidate in (bdir / SAFE_BM25_FILENAME, bdir / "bm25.npz", bdir / LEGACY_BM25_FILENAME):
        if candidate.exists():
            bm25 = candidate
            break
    if bm25 is not None:
        m.lexical = LexicalInfo(backend="rank_bm25", file=bm25.name)

    # Count chunks from the docstore.
    try:
        bundle = load_bundle(bdir, allow_pickle=True)
        if bundle.docstore is not None:
            m.chunks = ChunksInfo(
                count=len(bundle.docstore.records),
                index_key="index_text",
                metadata_fields=["file_path", "line_start", "line_end", "section_hierarchy", "url"],
            )
    except Exception:
        pass

    m.created_by = m.created_by or "ovgenai-retrieval.upgrade"
    m.write(manifest_path)
    return {"status": "created", "path": str(manifest_path)}


def upgrade_bundle(
    bundle_path: str | os.PathLike,
    *,
    allow_pickle: bool = True,
    force: bool = False,
    embedder_name: str = "unknown",
) -> dict[str, Any]:
    """One-shot upgrade: BM25 sidecar (JSON) + manifest.json.

    Migrates any legacy ``bm25.pkl`` to ``bm25.json`` in place (REQ compatible
    with the faiss_safe migration pattern in kit-usd-agents-master).
    """
    bm25_result = build_bm25_sidecar_from_bundle(bundle_path, allow_pickle=allow_pickle, force=force)
    mf_result = write_or_update_manifest(bundle_path, embedder_name=embedder_name, force=force)
    return {
        "bundle_path": str(bundle_path),
        "bm25": bm25_result,
        "manifest": mf_result,
    }


def discover_bundles(root: str | os.PathLike) -> list[Path]:
    """Find every directory under ``root`` that looks like a bundle."""
    root_p = Path(root)
    if not root_p.is_dir():
        return []
    out: list[Path] = []
    for d in sorted(root_p.rglob("*")):
        if not d.is_dir():
            continue
        if (d / "index.faiss").exists() or (d / "manifest.json").exists():
            out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="In-place upgrade of legacy bundles — emit bm25.json + manifest.json.",
    )
    ap.add_argument("--bundle", type=Path, help="Upgrade a single bundle directory.")
    ap.add_argument("--root", type=Path, help="Walk this directory and upgrade every bundle found.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing bm25.json / manifest.json.")
    ap.add_argument(
        "--no-pickle",
        action="store_true",
        help="Refuse to load bundles that only have index.pkl (legacy LangChain pickle).",
    )
    ap.add_argument(
        "--embedder",
        default="unknown",
        help="Known embedder model name to stamp into the synthesized manifest.",
    )
    args = ap.parse_args()

    if bool(args.bundle) == bool(args.root):
        ap.error("provide exactly one of --bundle or --root")

    paths = [args.bundle] if args.bundle else discover_bundles(args.root)
    if not paths:
        print(f"No bundles found under {args.root}", file=sys.stderr)
        return 2

    allow_pickle = not args.no_pickle
    total_ok = 0
    total_skipped = 0
    for p in paths:
        try:
            result = upgrade_bundle(
                p,
                allow_pickle=allow_pickle,
                force=args.force,
                embedder_name=args.embedder,
            )
            bm = result["bm25"]["status"]
            mf = result["manifest"]["status"]
            print(f"  [{p}]  bm25={bm}  manifest={mf}")
            total_ok += 1
        except Exception as e:
            print(f"  [{p}]  SKIP ({e!r})", file=sys.stderr)
            total_skipped += 1

    print(f"\nUpgraded: {total_ok}  Skipped: {total_skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
