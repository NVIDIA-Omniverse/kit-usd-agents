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

"""Configuration module for Kit MCP tools."""

import os
from pathlib import Path
from typing import Optional

# Get the package directory
PACKAGE_DIR = Path(__file__).parent

# Kit version (defined early since paths depend on it)
ENV_KIT_VERSION = "MCP_KIT_VERSION"
KIT_VERSION = os.environ.get(ENV_KIT_VERSION, "109.0")

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

# Fuzzy matching configuration
# Used for "did you mean?" suggestions across extension and API lookups
DEFAULT_FUZZY_MATCH_THRESHOLD = 0.5

# Usage logging configuration
USAGE_LOGGING_ENABLED_BY_DEFAULT = True
USAGE_LOGGING_TIMEOUT = 30.0

# OpenSearch configuration for usage analytics
OPEN_SEARCH_URL = "https://search-omnigenai-usage-e6htsydkjhq7tktdqbflrqg3aa.us-west-2.es.amazonaws.com"

# RAG Configuration for Kit Code Examples
DEFAULT_RAG_LENGTH_CODE = 30000
DEFAULT_RAG_TOP_K_CODE = 90
DEFAULT_RERANK_CODE = 10

# RAG Configuration for Kit Knowledge
DEFAULT_RAG_LENGTH_KNOWLEDGE = 30000
DEFAULT_RAG_TOP_K_KNOWLEDGE = 45
DEFAULT_RERANK_KNOWLEDGE = 10

# Knowledge FAISS index path
KNOWLEDGE_INDEX_PATH = DATA_DIR / KIT_VERSION / "knowledge"

# Reranking Configuration
DEFAULT_RERANK_MODEL = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
DEFAULT_RERANK_ENDPOINT = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"

# Embedding Configuration
DEFAULT_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
DEFAULT_EMBEDDING_ENDPOINT = "https://ai.api.nvidia.com/v1"

# Environment variable names
ENV_DISABLE_LOGGING = "KIT_MCP_DISABLE_USAGE_LOGGING"
ENV_MCP_PORT = "MCP_PORT"
ENV_EMBEDDER_BACKEND = "KIT_EMBEDDER_BACKEND"  # "nvidia_api" or "local"
ENV_LOCAL_EMBEDDER_URL = "KIT_LOCAL_EMBEDDER_URL"  # URL for local embedder (e.g., "http://10.34.1.127:8001")
ENV_RERANKER_BACKEND = "KIT_RERANKER_BACKEND"  # "nvidia_api" or "local"
ENV_LOCAL_RERANKER_URL = "KIT_LOCAL_RERANKER_URL"  # URL for local reranker (e.g., "http://10.34.1.127:8002")


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


def get_effective_api_key(service: Optional[str] = None) -> Optional[str]:
    """Get the effective API key for a service.

    Args:
        service: The service name ('embeddings' or 'reranking')

    Returns:
        The API key from environment variable
    """
    return NVIDIA_API_KEY
