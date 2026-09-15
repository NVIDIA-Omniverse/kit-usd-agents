## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import json
import os
from pathlib import Path

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from lc_agent import get_retriever_registry

DEFAULT_EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"
DEFAULT_EMBEDDING_DIMENSION = 2048


def _get_nvidia_embedder(api_key: str = None, func_id: str = None, model: str = None):
    embedding = NVIDIAEmbeddings(
        model=model or DEFAULT_EMBEDDING_MODEL,
        truncate="END",
        nvidia_api_key=api_key,
    )

    if not func_id:
        func_id = os.environ.get("NVIDIA_EMBEDDING_FUNC_ID", None)

    if func_id:
        base_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/{func_id}"
        base_url = base_url.replace("{func_id}", func_id)
        embedding._client.infer_path = base_url

    return embedding


def _load_faiss_index(folder: str, embedder, expected_model: str) -> FAISS:
    # Pickle-free loader: reads the native FAISS binary (index.faiss) for the
    # vector matrix, and a JSON docstore (index.json) for Document content
    # and the vector-row -> doc-id mapping. Replaces the old
    # FAISS.load_local(..., allow_dangerous_deserialization=True) call that
    # pickled an InMemoryDocstore -- pickle files were being flagged as
    # legacy pickle metadata by container scanners.
    folder = Path(folder)
    faiss_path = folder / "index.faiss"
    json_path = folder / "index.json"
    manifest_path = folder / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing embedding provenance at {manifest_path}; rebuild this index before loading it"
        )

    raw_index = faiss.read_index(str(faiss_path))
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    docstore = InMemoryDocstore(
        {
            doc_id: Document(page_content=d["page_content"], metadata=d["metadata"])
            for doc_id, d in payload["docstore"].items()
        }
    )
    index_to_docstore_id = {int(k): v for k, v in payload["index_to_docstore_id"].items()}

    rows = sorted(index_to_docstore_id)
    if rows != list(range(int(raw_index.ntotal))):
        raise ValueError(f"FAISS row mapping in {json_path} does not match {raw_index.ntotal} vectors")
    missing_ids = set(index_to_docstore_id.values()) - set(docstore._dict)
    if missing_ids or len(docstore._dict) != int(raw_index.ntotal):
        raise ValueError(
            f"FAISS docstore in {json_path} does not match the index: "
            f"vectors={raw_index.ntotal}, documents={len(docstore._dict)}, missing_ids={len(missing_ids)}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedding_info = manifest.get("embeddings") or {}
    manifest_model = embedding_info.get("model")
    manifest_dimension = embedding_info.get("dim")
    if manifest_model != expected_model:
        raise ValueError(
            f"Embedding model mismatch in {folder}: index={manifest_model!r}, "
            f"configured={expected_model!r}. Reindex this bundle."
        )
    if manifest_dimension != int(raw_index.d):
        raise ValueError(
            f"Embedding dimension mismatch in {folder}: index={raw_index.d}, "
            f"manifest={manifest_dimension}. Reindex this bundle."
        )
    if expected_model == DEFAULT_EMBEDDING_MODEL and raw_index.d != DEFAULT_EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding dimension mismatch in {folder}: index={raw_index.d}, "
            f"{DEFAULT_EMBEDDING_MODEL} requires {DEFAULT_EMBEDDING_DIMENSION}. Reindex this bundle."
        )

    return FAISS(
        embedding_function=embedder,
        index=raw_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )


def register_faiss_retriever(
    name, vectordb_index_name: str, top_k: int = 3, api_key: str = None, func_id: str = None, model: str = None
):
    effective_model = model or DEFAULT_EMBEDDING_MODEL
    embedder = _get_nvidia_embedder(api_key=api_key, func_id=func_id, model=effective_model)
    if not embedder:
        return

    vectordb = _load_faiss_index(vectordb_index_name, embedder, effective_model)

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    get_retriever_registry().register(name, retriever)


def register_all(top_k: int = 3, api_key: str = None, func_id: str = None, model: str = None):

    # Code retriever
    faiss_index_code_embedqa = "../data/faiss_usd_code_3346"
    faiss_index_code_embedqa = os.path.abspath(f"{__file__}/{faiss_index_code_embedqa}")
    register_faiss_retriever("embedqa", faiss_index_code_embedqa, top_k, api_key, func_id, model)

    # Metafunction retriever
    faiss_usd_metafunctions = "../data/faiss_usd_metafunctions_01"
    faiss_usd_metafunctions = os.path.abspath(f"{__file__}/{faiss_usd_metafunctions}")
    register_faiss_retriever("usd_metafunctions", faiss_usd_metafunctions, top_k, api_key, func_id, model)

    # Knowledge retriever
    faiss_index_usd_knowledge_qa = "../data/faiss_usd_knowledge_sdgqa"
    faiss_index_usd_knowledge_qa = os.path.abspath(f"{__file__}/{faiss_index_usd_knowledge_qa}")
    register_faiss_retriever("usd_knowledge_qa", faiss_index_usd_knowledge_qa, top_k, api_key, func_id, model)

    # Code 06262024 retriever
    faiss_index_usd_code06262024 = "../data/faiss_usd_code_06262024"
    faiss_index_usd_code06262024 = os.path.abspath(f"{__file__}/{faiss_index_usd_code06262024}")
    register_faiss_retriever("usd_code06262024", faiss_index_usd_code06262024, top_k, api_key, func_id, model)


def unregister_all():
    get_retriever_registry().unregister("embedqa")
    get_retriever_registry().unregister("usd_metafunctions")
    get_retriever_registry().unregister("usd_knowledge_qa")
    get_retriever_registry().unregister("usd_code06262024")
