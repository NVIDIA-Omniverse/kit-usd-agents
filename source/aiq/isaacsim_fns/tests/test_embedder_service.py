# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from isaacsim_fns.config import DEFAULT_EMBEDDING_ENDPOINT
from isaacsim_fns.services.embedder_service import EmbeddingBackendError, LocalEmbedder, guard_backend_errors


class _Response:
    def __init__(self, embeddings):
        self._embeddings = embeddings

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"embedding": embedding} for embedding in self._embeddings]}


class _Requests:
    def __init__(self):
        self.calls = []

    def post(self, url, *, json, headers, timeout):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return _Response([[float(i)] for i, _ in enumerate(json["input"])])


def test_hosted_embedding_endpoint_uses_integrate_api():
    assert DEFAULT_EMBEDDING_ENDPOINT == "https://integrate.api.nvidia.com/v1"


def test_local_embedder_uses_supported_input_types():
    embedder = LocalEmbedder("https://embedder", model="nvidia/nemotron-3-embed-1b")
    requests = _Requests()
    embedder._requests = requests

    assert embedder.embed_documents(["one", "two"]) == [[0.0], [1.0]]
    assert embedder.embed_query("three") == [0.0]

    document_call, query_call = requests.calls
    assert document_call["url"] == "https://embedder/v1/embeddings"
    assert document_call["json"]["input_type"] == "passage"
    assert query_call["json"]["input_type"] == "query"


class _RaisingEmbedder:
    model = "nvidia/retired-model"

    def embed_query(self, text):
        raise Exception("[410] Gone")

    def embed_documents(self, texts):
        raise Exception("[410] Gone")


def test_guarded_embedder_exposes_model_and_names_it_in_errors():
    wrapped = guard_backend_errors(_RaisingEmbedder())
    # load_faiss_safe reads ``embeddings.model`` for its provenance check.
    assert wrapped.model == "nvidia/retired-model"

    with pytest.raises(EmbeddingBackendError, match=r"model: nvidia/retired-model.*\[410\] Gone"):
        wrapped.embed_query("q")
