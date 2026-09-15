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

"""Configuration module for Isaac Sim MCP tools."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

# Get the package directory
PACKAGE_DIR = Path(__file__).parent

# Data paths - relative to package directory
DATA_DIR = PACKAGE_DIR / "data"
INSTRUCTIONS_DIR = DATA_DIR / "instructions"
EXTENSIONS_INDEX_PATH = DATA_DIR / "extensions_index"
CODE_EXAMPLES_INDEX_PATH = DATA_DIR / "code_examples_index"
TEST_EXAMPLES_INDEX_PATH = DATA_DIR / "test_examples_index"

# API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Default configuration values
DEFAULT_MCP_PORT = 9902
DEFAULT_TIMEOUT = 30.0

# Usage logging configuration
USAGE_LOGGING_ENABLED_BY_DEFAULT = True
USAGE_LOGGING_TIMEOUT = 30.0

# OpenSearch configuration for usage analytics
OPEN_SEARCH_URL = "https://search-omnigenai-usage-e6htsydkjhq7tktdqbflrqg3aa.us-west-2.es.amazonaws.com"

# RAG Configuration for Isaac Sim Code Examples
DEFAULT_RAG_LENGTH_CODE = 30000
DEFAULT_RAG_TOP_K_CODE = 90
DEFAULT_RERANK_CODE = 10

# Reranking Configuration
DEFAULT_RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"
DEFAULT_RERANK_ENDPOINT = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking"

# Embedding Configuration
DEFAULT_EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"
DEFAULT_EMBEDDING_ENDPOINT = "https://integrate.api.nvidia.com/v1"

# Environment variable names
ENV_DISABLE_LOGGING = "KIT_MCP_DISABLE_USAGE_LOGGING"
ENV_MCP_PORT = "MCP_PORT"
ENV_ISAACSIM_VERSION = "MCP_ISAACSIM_VERSION"
ENV_EMBEDDER_BACKEND = "KIT_EMBEDDER_BACKEND"  # "nvidia_api" or "local"
ENV_LOCAL_EMBEDDER_URL = "KIT_LOCAL_EMBEDDER_URL"  # URL for local embedder (e.g., "http://10.34.1.127:8001")
ENV_EMBEDDING_MODEL = "KIT_EMBEDDING_MODEL"  # override the embedding model without a rebuild
ENV_RERANK_MODEL = "KIT_RERANK_MODEL"  # override the rerank model without a rebuild
ENV_RERANK_ENDPOINT = "KIT_RERANK_ENDPOINT"  # override the rerank URL outright


def get_env_bool(env_var: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    value = os.environ.get(env_var, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    elif value in ("false", "0", "no", "off"):
        return False
    return default


def get_env_int(env_var: str, default: int) -> int:
    """Get integer value from environment variable."""
    try:
        return int(os.environ.get(env_var, str(default)))
    except ValueError:
        return default


def get_env_float(env_var: str, default: float) -> float:
    """Get float value from environment variable."""
    try:
        return float(os.environ.get(env_var, str(default)))
    except ValueError:
        return default


# Runtime configuration
MCP_PORT = get_env_int(ENV_MCP_PORT, DEFAULT_MCP_PORT)
USAGE_LOGGING_ENABLED = not get_env_bool(ENV_DISABLE_LOGGING, False)
ISAACSIM_VERSION = os.environ.get(ENV_ISAACSIM_VERSION, "6.1")

# Model selection. These are the values the services actually use; the DEFAULT_*
# constants above are only the fallback. Hosted models get retired periodically
# (nv-embedqa-e5-v5 and llama-nemotron-rerank-1b-v2 both reached EOL on
# 2026-08-25), so these are env-overridable to allow repointing a running
# deployment without a code change and rebuild.
#
# NOTE: changing the embedding model requires a matching index. The bundled
# FAISS data is built with EMBEDDING_MODEL; pointing at a model with a
# different vector width will fail at query time, not at startup.
EMBEDDING_MODEL = os.environ.get(ENV_EMBEDDING_MODEL, DEFAULT_EMBEDDING_MODEL)
RERANK_MODEL = os.environ.get(ENV_RERANK_MODEL, DEFAULT_RERANK_MODEL)
RERANK_ENDPOINT = os.environ.get(ENV_RERANK_ENDPOINT) or (
    DEFAULT_RERANK_ENDPOINT
    if RERANK_MODEL == DEFAULT_RERANK_MODEL
    else f"https://ai.api.nvidia.com/v1/retrieval/{RERANK_MODEL}/reranking"
)


def get_effective_api_key(service: Optional[str] = None) -> Optional[str]:
    """Get the effective API key for a service.

    Args:
        service: The service name ('embeddings' or 'reranking')

    Returns:
        The API key from environment variable
    """
    return NVIDIA_API_KEY
