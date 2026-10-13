# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Hermetic fakes for exercising search services against a failing embedding backend.

The helpers build a tiny FAISS bundle on disk so the real service constructors
load an index, and provide embedders that fail the way the real backends do:
``LocalEmbedder`` raises ``requests.HTTPError`` from ``raise_for_status()``,
while ``langchain_nvidia_ai_endpoints.NVIDIAEmbeddings`` raises a bare
``Exception`` whose message starts with ``[<status>] <reason>``.
"""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import requests

RETIRED_MODEL = "nvidia/nv-embedqa-e5-v5"
_DIM = 4


class FailingEmbedder:
    """Embedder whose every request fails with an HTTP error from the backend."""

    def __init__(self, status_code: int = 410, reason: str = "Gone", style: str = "http_error"):
        self.model = RETIRED_MODEL
        self.status_code = status_code
        self.reason = reason
        self.style = style
        self.calls = 0

    def _fail(self) -> None:
        self.calls += 1
        url = "https://integrate.api.nvidia.com/v1/embeddings"
        if self.style == "nvidia_client":
            # Shape of NVIDIAEmbeddings' error: a bare Exception carrying the status.
            raise Exception(f"[{self.status_code}] {self.reason}\nThe model '{self.model}' is no longer available.")
        response = requests.Response()
        response.status_code = self.status_code
        response.reason = self.reason
        response.url = url
        raise requests.HTTPError(f"{self.status_code} Client Error: {self.reason} for url: {url}", response=response)

    def embed_query(self, text: str) -> List[float]:
        self._fail()
        return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._fail()
        return []


def stub_hybrid_shim(monkeypatch) -> None:
    """Replace ``isaacsim_fns.utils.hybrid_shim`` with a no-hybrid stub.

    The real shim imports ``ovgenai_retrieval``, which is not a dependency of the
    poetry test environment; the settings and extension services load the shim
    inside their FAISS init ``try`` block, so an import failure would discard
    the fixture index. The stub keeps those services on the dense FAISS path.
    """
    import sys
    import types

    stub = types.ModuleType("isaacsim_fns.utils.hybrid_shim")
    stub.maybe_load_hybrid = lambda *args, **kwargs: None
    stub.hits_to_documents_with_scores = lambda hits: []
    monkeypatch.setitem(sys.modules, "isaacsim_fns.utils.hybrid_shim", stub)
    import isaacsim_fns.utils as utils_pkg

    monkeypatch.setattr(utils_pkg, "hybrid_shim", stub, raising=False)


def write_faiss_bundle(path: Path, docs: Sequence[Tuple[str, Dict]]) -> Path:
    """Write a minimal faiss_safe bundle containing ``docs`` (page_content, metadata)."""
    import faiss
    import numpy as np
    from isaacsim_fns.utils.faiss_safe import save_faiss_safe
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    assert 0 < len(docs) <= _DIM
    index = faiss.IndexFlatL2(_DIM)
    index.add(np.eye(len(docs), _DIM, dtype=np.float32))
    ids = [f"doc{i}" for i in range(len(docs))]
    docstore = InMemoryDocstore(
        {doc_id: Document(page_content=content, metadata=dict(meta)) for doc_id, (content, meta) in zip(ids, docs)}
    )
    store = FAISS(lambda text: [0.0] * _DIM, index, docstore, dict(enumerate(ids)))
    save_faiss_safe(store, str(path))
    return path
