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

"""Embedding services for the USD RAG MCP server."""

from typing import Any, Dict, List, Optional

import requests
from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from ..config import DEFAULT_EMBEDDING_ENDPOINT, DEFAULT_EMBEDDING_MODEL, get_effective_api_key
from .embedder_service import EmbedderFactory


class CustomEndpointEmbeddings(Embeddings):
    """Custom embeddings implementation for specific endpoints."""

    def __init__(
        self,
        endpoint_url: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
        api_key: str = None,
    ):
        """Initialize the CustomEndpointEmbeddings.

        Args:
            endpoint_url: The endpoint URL (e.g. https://integrate.api.nvidia.com/v1)
            model: The model to use
            api_key: The API key to use
        """
        self.endpoint_url = endpoint_url
        self.model = model
        self.api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents.

        Args:
            texts: The list of texts to embed

        Returns:
            List of embeddings
        """
        embedding_payload = {
            "input": texts,
            "input_type": "query",
            "encoding_format": "float",
            "truncate": "END",
        }

        if self.model is not None:
            embedding_payload["model"] = self.model

        embedding_headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }

        if self.api_key is not None:
            embedding_headers["Authorization"] = f"Bearer {self.api_key}"

        # Construct the embeddings URL
        # For local services, endpoint might be like http://localhost:8004
        # For NVIDIA API, it might already include the full path
        if "/embeddings" in self.endpoint_url:
            # Already has embeddings in the path
            url = self.endpoint_url
        elif self.endpoint_url.endswith("/v1"):
            # Has /v1 already, append /embeddings
            url = f"{self.endpoint_url}/embeddings"
        else:
            # No /v1, add full path
            url = f"{self.endpoint_url}/v1/embeddings"

        embedding_response = requests.post(
            url,
            json=embedding_payload,
            headers=embedding_headers,
            timeout=180,  # 180 second timeout to prevent indefinite hangs
        )
        embedding_response.raise_for_status()

        return [emb["embedding"] for emb in embedding_response.json()["data"]]

    def embed_query(self, text: str) -> List[float]:
        """Embed a query.

        Args:
            text: The text to embed

        Returns:
            The embedding
        """
        return self.embed_documents([text])[0]


def create_embeddings_with_config(config: Optional[Dict[str, Any]] = None) -> Embeddings:
    """Create an embeddings instance using provided configuration and factory pattern.

    Args:
        config: Configuration dict with 'model', 'endpoint', and 'api_key'

    Returns:
        Embeddings instance
    """
    if not config:
        return create_embeddings()

    endpoint_url = config.get("endpoint")
    api_key = config.get("api_key", "")
    model = config.get("model", DEFAULT_EMBEDDING_MODEL)

    # If endpoint is explicitly None or empty, don't set it
    if endpoint_url is None or endpoint_url == "":
        endpoint_url = None

    if not api_key:
        api_key = get_effective_api_key("embeddings")

    if endpoint_url is not None:
        # Use factory with local backend
        return EmbedderFactory.create(backend="local", local_url=endpoint_url, model=model)
    else:
        # Use factory with default backend from environment
        return EmbedderFactory.create(api_key=api_key, model=model)


def create_embeddings(endpoint_url: str = None, api_key: str = "") -> Embeddings:
    """Create an embeddings instance using factory pattern.

    Args:
        endpoint_url: Optional custom endpoint URL for local embedder
        api_key: API key for authentication

    Returns:
        Embeddings instance
    """
    if not api_key:
        api_key = get_effective_api_key("embeddings")

    model = DEFAULT_EMBEDDING_MODEL

    if endpoint_url is not None:
        # Use factory with local backend
        return EmbedderFactory.create(backend="local", local_url=endpoint_url, model=model)
    else:
        # Use factory with default backend from environment
        return EmbedderFactory.create(api_key=api_key, model=model)
