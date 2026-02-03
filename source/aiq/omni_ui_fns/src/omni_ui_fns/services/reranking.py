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

"""Reranking service for improving search relevance with support for NVIDIA API and local deployment."""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from ..config import (
    DEFAULT_RERANK_ENDPOINT,
    DEFAULT_RERANK_MODEL,
    ENV_LOCAL_RERANKER_URL,
    ENV_RERANKER_BACKEND,
    get_effective_api_key,
)

logger = logging.getLogger(__name__)


class Reranker:
    """Service for reranking search results using NVIDIA reranking models (NVIDIA API)."""

    def __init__(
        self, endpoint_url: str = DEFAULT_RERANK_ENDPOINT, api_key: str = "", model: str = DEFAULT_RERANK_MODEL
    ):
        """Initialize the Reranker.

        Args:
            endpoint_url: The reranking service endpoint URL
            api_key: API key for authentication
            model: The model to use for reranking
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model = model

    def rerank(self, query: str, passages: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """Rerank passages based on relevance to the query.

        Args:
            query: The search query
            passages: List of passages to rerank
            top_k: Number of top results to return

        Returns:
            List of reranked results with scores
        """
        if not passages:
            return []

        # Prepare the request payload
        payload = {
            "model": self.model,
            "query": {"text": query},
            "passages": [{"text": passage} for passage in passages],
            "truncate": "END",
        }

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.post(self.endpoint_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            result = response.json()
            rankings = result.get("rankings", [])

            # Sort by score and limit to top_k
            rankings.sort(key=lambda x: x.get("logit", 0), reverse=True)
            return rankings[:top_k]

        except requests.exceptions.RequestException as e:
            logger.error(f"Reranking request failed: {e}")
            # Return original order as fallback
            return [{"index": i, "logit": 0} for i in range(min(top_k, len(passages)))]
        except Exception as e:
            logger.error(f"Unexpected error during reranking: {e}")
            return [{"index": i, "logit": 0} for i in range(min(top_k, len(passages)))]


class LocalReranker:
    """Wrapper for local reranker API that mimics the Reranker interface."""

    def __init__(self, base_url: str, model: str = DEFAULT_RERANK_MODEL):
        """Initialize local reranker.

        Args:
            base_url: Base URL for local reranker (e.g., "http://10.34.1.127:8002")
            model: Model name to use
        """
        self.base_url = base_url.rstrip("/")
        self.model = model

    def rerank(self, query: str, passages: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """Rerank passages based on relevance to the query using local API.

        Args:
            query: The search query
            passages: List of passages to rerank
            top_k: Number of top results to return

        Returns:
            List of reranked results with scores
        """
        if not passages:
            return []

        try:
            url = f"{self.base_url}/v1/ranking"
            headers = {"accept": "application/json", "Content-Type": "application/json"}

            payload = {
                "model": self.model,
                "query": {"text": query},
                "passages": [{"text": passage} for passage in passages],
                "truncate": "END",
            }

            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            result = response.json()
            rankings = result.get("rankings", [])

            # Sort by score and limit to top_k
            rankings.sort(key=lambda x: x.get("logit", 0), reverse=True)

            logger.debug(f"Successfully reranked {len(passages)} passages via local API")
            return rankings[:top_k]

        except requests.exceptions.RequestException as e:
            logger.error(f"Local reranking request failed: {e}")
            # Return original order as fallback
            return [{"index": i, "logit": 0} for i in range(min(top_k, len(passages)))]
        except Exception as e:
            logger.error(f"Unexpected error during local reranking: {e}")
            return [{"index": i, "logit": 0} for i in range(min(top_k, len(passages)))]


class RerankerFactory:
    """Factory for creating reranker instances based on configuration."""

    _instance: Optional[Any] = None

    @staticmethod
    def create(
        api_key: Optional[str] = None,
        model: str = DEFAULT_RERANK_MODEL,
        backend: Optional[str] = None,
        local_url: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> Optional[Any]:
        """Create a reranker instance based on configuration.

        Args:
            api_key: NVIDIA API key (used for nvidia_api backend)
            model: Model name to use
            backend: Reranker backend ("nvidia_api" or "local"), defaults to environment variable
            local_url: Local reranker URL (used for local backend), defaults to environment variable
            endpoint_url: Custom endpoint URL for NVIDIA API backend

        Returns:
            Reranker instance (either Reranker or LocalReranker) or None if creation fails
        """
        # Determine backend to use
        if backend is None:
            backend = os.getenv(ENV_RERANKER_BACKEND, "nvidia_api")

        logger.info(f"Creating reranker with backend: {backend}")

        if backend == "local":
            # Use local reranker
            if local_url is None:
                local_url = os.getenv(ENV_LOCAL_RERANKER_URL)

            if not local_url:
                raise ValueError(
                    f"Local reranker URL must be provided via {ENV_LOCAL_RERANKER_URL} "
                    "environment variable or local_url parameter"
                )

            logger.info(f"Using local reranker at {local_url}")
            return LocalReranker(base_url=local_url, model=model)

        elif backend == "nvidia_api":
            # Use NVIDIA API reranker
            if api_key is None:
                api_key = get_effective_api_key("reranking")

            if not api_key:
                logger.warning("No API key available for reranker")
                return None

            endpoint = endpoint_url or DEFAULT_RERANK_ENDPOINT

            logger.info("Using NVIDIA API reranker")
            return Reranker(endpoint_url=endpoint, api_key=api_key, model=model)

        else:
            raise ValueError(f"Unknown reranker backend: {backend}. Must be 'nvidia_api' or 'local'")

    @staticmethod
    def get_instance(
        api_key: Optional[str] = None,
        model: str = DEFAULT_RERANK_MODEL,
        backend: Optional[str] = None,
        local_url: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> Optional[Any]:
        """Get or create a singleton reranker instance."""
        if RerankerFactory._instance is None:
            RerankerFactory._instance = RerankerFactory.create(
                api_key=api_key, model=model, backend=backend, local_url=local_url, endpoint_url=endpoint_url
            )
        return RerankerFactory._instance

    @staticmethod
    def reset() -> None:
        """Reset the singleton instance."""
        RerankerFactory._instance = None


# Legacy functions for backward compatibility
def create_reranker(endpoint_url: str = None, api_key: str = None, model: str = None) -> Optional[Reranker]:
    """Create a reranker instance (legacy function).

    Args:
        endpoint_url: Optional custom endpoint URL
        api_key: Optional API key (uses environment if not provided)
        model: Optional model name

    Returns:
        Reranker instance or None if creation fails
    """
    return RerankerFactory.create(
        api_key=api_key,
        model=model or DEFAULT_RERANK_MODEL,
        backend="nvidia_api",
        endpoint_url=endpoint_url,
    )


def create_reranker_with_config(config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """Create a reranker instance using provided configuration.

    Args:
        config: Configuration dict with 'model', 'endpoint', 'api_key', 'backend', 'local_url'

    Returns:
        Reranker instance or None if creation fails
    """
    if not config:
        return RerankerFactory.create()

    return RerankerFactory.create(
        api_key=config.get("api_key"),
        model=config.get("model", DEFAULT_RERANK_MODEL),
        backend=config.get("backend"),
        local_url=config.get("local_url"),
        endpoint_url=config.get("endpoint"),
    )
