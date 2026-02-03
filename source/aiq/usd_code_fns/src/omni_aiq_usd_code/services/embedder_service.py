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

"""Embedder service wrapper for switching between NVIDIA API and local deployment."""

import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Try to import NVIDIA embeddings (optional dependency)
try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings  # type: ignore

    NVIDIA_EMBEDDINGS_AVAILABLE = True
except ImportError:
    logger.warning("NVIDIA embeddings not available.")
    NVIDIA_EMBEDDINGS_AVAILABLE = False

# Environment variable to control embedder backend
ENV_EMBEDDER_BACKEND = "KIT_EMBEDDER_BACKEND"  # "nvidia_api" or "local"
ENV_LOCAL_EMBEDDER_URL = "KIT_LOCAL_EMBEDDER_URL"  # URL for local embedder


class LocalEmbedder:
    """Wrapper for local embedder API that mimics LangChain embeddings interface."""

    def __init__(self, base_url: str, model: str = "nvidia/nv-embedqa-e5-v5"):
        """Initialize local embedder.

        Args:
            base_url: Base URL for local embedder (e.g., "http://10.34.1.127:8001")
            model: Model name to use
        """
        self.base_url = base_url
        self.model = model

        try:
            import requests

            self._requests = requests
        except ImportError:
            logger.error("requests library not available for local embedder")
            self._requests = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not self._requests:
            raise RuntimeError("requests library not available")

        if not texts:
            return []

        try:
            url = f"{self.base_url}/v1/embeddings"
            headers = {"accept": "application/json", "Content-Type": "application/json"}

            payload = {
                "input": texts,
                "model": self.model,
                "input_type": "search_document",
            }

            response = self._requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()
            embeddings = [item["embedding"] for item in data.get("data", [])]

            logger.debug(f"Successfully embedded {len(texts)} documents via local API")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to embed documents via local API: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        if not self._requests:
            raise RuntimeError("requests library not available")

        try:
            url = f"{self.base_url}/v1/embeddings"
            headers = {"accept": "application/json", "Content-Type": "application/json"}

            payload = {
                "input": [text],
                "model": self.model,
                "input_type": "query",
            }

            response = self._requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()
            embedding = data["data"][0]["embedding"]

            logger.debug("Successfully embedded query via local API")
            return embedding

        except Exception as e:
            logger.error(f"Failed to embed query via local API: {e}")
            raise

    def __call__(self, text: str) -> List[float]:
        """Make embedder callable for FAISS compatibility.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.embed_query(text)


class EmbedderFactory:
    """Factory for creating embedder instances based on configuration."""

    _instance: Optional[Any] = None

    @staticmethod
    def create(
        api_key: Optional[str] = None,
        model: str = "nvidia/nv-embedqa-e5-v5",
        backend: Optional[str] = None,
        local_url: Optional[str] = None,
    ) -> Any:
        """Create an embedder instance based on configuration.

        Args:
            api_key: NVIDIA API key (used for nvidia_api backend)
            model: Model name to use
            backend: Embedder backend ("nvidia_api" or "local"), defaults to environment variable
            local_url: Local embedder URL (used for local backend), defaults to environment variable

        Returns:
            Embedder instance (either NVIDIAEmbeddings or LocalEmbedder)

        Raises:
            RuntimeError: If required backend is not available
        """
        # Determine backend to use
        if backend is None:
            backend = os.getenv(ENV_EMBEDDER_BACKEND, "nvidia_api")

        logger.info(f"Creating embedder with backend: {backend}")

        if backend == "local":
            # Use local embedder
            if local_url is None:
                local_url = os.getenv(ENV_LOCAL_EMBEDDER_URL)

            if not local_url:
                raise ValueError(
                    f"Local embedder URL must be provided via {ENV_LOCAL_EMBEDDER_URL} "
                    "environment variable or local_url parameter"
                )

            logger.info(f"Using local embedder at {local_url}")
            return LocalEmbedder(base_url=local_url, model=model)

        elif backend == "nvidia_api":
            # Use NVIDIA API embedder
            if not NVIDIA_EMBEDDINGS_AVAILABLE:
                raise RuntimeError(
                    "NVIDIA embeddings not available. Please install " "langchain-nvidia-ai-endpoints package"
                )

            if api_key is None:
                api_key = os.getenv("NVIDIA_API_KEY", "")

            logger.info("Using NVIDIA API embedder")
            return NVIDIAEmbeddings(model=model, nvidia_api_key=api_key, truncate="END")

        else:
            raise ValueError(f"Unknown embedder backend: {backend}. " f"Must be 'nvidia_api' or 'local'")

    @staticmethod
    def get_instance(
        api_key: Optional[str] = None,
        model: str = "nvidia/nv-embedqa-e5-v5",
        backend: Optional[str] = None,
        local_url: Optional[str] = None,
    ) -> Any:
        """Get or create a singleton embedder instance.

        Args:
            api_key: NVIDIA API key
            model: Model name
            backend: Embedder backend
            local_url: Local embedder URL

        Returns:
            Embedder instance
        """
        if EmbedderFactory._instance is None:
            EmbedderFactory._instance = EmbedderFactory.create(
                api_key=api_key, model=model, backend=backend, local_url=local_url
            )
        return EmbedderFactory._instance

    @staticmethod
    def reset() -> None:
        """Reset the singleton instance."""
        EmbedderFactory._instance = None
