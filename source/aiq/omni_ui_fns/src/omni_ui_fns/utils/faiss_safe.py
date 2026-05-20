# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Safe FAISS (de)serialization using JSON instead of pickle.

LangChain's ``FAISS.save_local`` / ``FAISS.load_local`` pair the binary
``<name>.faiss`` index with a ``<name>.pkl`` that pickles the docstore and
index-to-id map. Loading that pickle requires
``allow_dangerous_deserialization=True``, which can execute arbitrary code if
the file is tampered with.

``save_faiss_safe`` / ``load_faiss_safe`` write and read a ``<name>.json``
alongside ``<name>.faiss`` instead. The JSON format only contains strings,
numbers, and plain mappings — safe to load anywhere.

Format version 1::

    {
      "format_version": 1,
      "docstore": {"<id>": {"page_content": "...", "metadata": {...}}, ...},
      "index_to_docstore_id": {"<int_index>": "<id>", ...}
    }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1


def save_faiss_safe(vectorstore: Any, folder_path: str, index_name: str = "index") -> None:
    """Save FAISS to ``<folder>/<index_name>.faiss`` + ``<folder>/<index_name>.json``."""
    import faiss as faiss_lib

    path = Path(folder_path)
    path.mkdir(parents=True, exist_ok=True)

    faiss_lib.write_index(vectorstore.index, str(path / f"{index_name}.faiss"))

    docstore_dict = _extract_docstore_dict(vectorstore.docstore)
    safe = {
        "format_version": FORMAT_VERSION,
        "docstore": {
            k: {"page_content": v.page_content, "metadata": dict(v.metadata)} for k, v in docstore_dict.items()
        },
        "index_to_docstore_id": {str(k): v for k, v in vectorstore.index_to_docstore_id.items()},
    }
    with open(path / f"{index_name}.json", "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False)


def load_faiss_safe(
    folder_path: str,
    embeddings: Any,
    index_name: str = "index",
    **kwargs: Any,
) -> Any:
    """Load FAISS from ``<folder>/<index_name>.faiss`` + ``<folder>/<index_name>.json``.

    Raises ``FileNotFoundError`` with migration guidance if only the legacy
    ``.pkl`` is present. **Never** falls back to pickle loading.
    """
    import faiss as faiss_lib
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    path = Path(folder_path)
    json_path = path / f"{index_name}.json"
    faiss_path = path / f"{index_name}.faiss"
    pkl_path = path / f"{index_name}.pkl"

    if not json_path.exists():
        if pkl_path.exists():
            raise FileNotFoundError(
                f"Legacy pickle metadata at {pkl_path} — run the faiss_safe converter "
                f"to produce {json_path}. Pickle loading is disabled for security "
                f"because legacy pickle metadata is not safe to load."
            )
        raise FileNotFoundError(f"Missing FAISS metadata JSON at {json_path}")
    if not faiss_path.exists():
        raise FileNotFoundError(f"Missing FAISS index at {faiss_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        safe = json.load(f)
    if safe.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported faiss_safe format_version={safe.get('format_version')!r}; " f"expected {FORMAT_VERSION}"
        )

    index = faiss_lib.read_index(str(faiss_path))
    documents = {
        k: Document(page_content=v["page_content"], metadata=v.get("metadata", {})) for k, v in safe["docstore"].items()
    }
    docstore = InMemoryDocstore(documents)
    index_to_id = {int(k): v for k, v in safe["index_to_docstore_id"].items()}

    return FAISS(embeddings, index, docstore, index_to_id, **kwargs)


def _extract_docstore_dict(docstore: Any) -> dict:
    """Pull the ``{id: Document}`` mapping out of a docstore, across LangChain versions."""
    for attr in ("_dict", "dict"):
        d = getattr(docstore, attr, None)
        if isinstance(d, dict):
            return d
    raise TypeError(
        f"Unsupported docstore type {type(docstore).__name__}: "
        f"expected InMemoryDocstore with a `_dict` or `dict` attribute"
    )


def convert_pkl_to_json(folder_path: str, index_name: str = "index", delete_pkl: bool = False) -> bool:
    """One-shot: read legacy ``<index_name>.pkl``, write ``<index_name>.json``.

    Requires langchain runtime (to unpickle Document / InMemoryDocstore). Set
    ``delete_pkl=True`` to remove the pickle after successful conversion.

    Returns ``True`` if a conversion happened, ``False`` if ``.json`` already
    existed (idempotent).
    """
    import pickle

    path = Path(folder_path)
    pkl_path = path / f"{index_name}.pkl"
    json_path = path / f"{index_name}.json"

    if json_path.exists():
        logger.info("faiss_safe: %s already exists — skipping", json_path)
        return False
    if not pkl_path.exists():
        raise FileNotFoundError(f"No legacy {pkl_path} to convert")

    with open(pkl_path, "rb") as f:
        docstore, index_to_id = pickle.load(f)

    docstore_dict = _extract_docstore_dict(docstore)
    safe = {
        "format_version": FORMAT_VERSION,
        "docstore": {
            k: {"page_content": v.page_content, "metadata": dict(v.metadata)} for k, v in docstore_dict.items()
        },
        "index_to_docstore_id": {str(k): v for k, v in index_to_id.items()},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False)

    if delete_pkl:
        pkl_path.unlink()
        logger.info("faiss_safe: removed %s", pkl_path)

    return True
