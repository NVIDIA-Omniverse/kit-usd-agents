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

"""Configuration module for USD Code tools."""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

# Get the package directory
PACKAGE_DIR = Path(__file__).parent

# Data paths - USD 25.02
# Note: Version selection may be added in the future
DATA_DIR = PACKAGE_DIR / "data" / "v25.02"
FAISS_CODE_INDEX_PATH = DATA_DIR / "code_rag"
FAISS_KNOWLEDGE_INDEX_PATH = DATA_DIR / "knowledge_rag"
USD_ATLAS_FILE_PATH = DATA_DIR / "usd_atlas_v25_02.json"

logging.info(f"Using USD 25.02 code RAG from: {FAISS_CODE_INDEX_PATH}")
logging.info(f"Using USD 25.02 atlas from: {USD_ATLAS_FILE_PATH}")

# API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# RAG Configuration
DEFAULT_RAG_LENGTH_CODE = 30000
DEFAULT_RAG_TOP_K_CODE = 90
DEFAULT_RERANK_CODE = 10

DEFAULT_RAG_LENGTH_KNOWLEDGE = 30000
DEFAULT_RAG_TOP_K_KNOWLEDGE = 45
DEFAULT_RERANK_KNOWLEDGE = 10

# Reranking Configuration
DEFAULT_RERANK_MODEL = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
DEFAULT_RERANK_ENDPOINT = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"

# Embedding Configuration
DEFAULT_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
DEFAULT_EMBEDDING_ENDPOINT = "https://ai.api.nvidia.com/v1"

# Environment variable names for embedder/reranker backend configuration
ENV_EMBEDDER_BACKEND = "KIT_EMBEDDER_BACKEND"  # "nvidia_api" or "local"
ENV_LOCAL_EMBEDDER_URL = "KIT_LOCAL_EMBEDDER_URL"  # URL for local embedder (e.g., "http://10.34.1.127:8001")
ENV_RERANKER_BACKEND = "KIT_RERANKER_BACKEND"  # "nvidia_api" or "local"
ENV_LOCAL_RERANKER_URL = "KIT_LOCAL_RERANKER_URL"  # URL for local reranker (e.g., "http://10.34.1.127:8002")


def get_effective_api_key(service: Optional[str] = None) -> Optional[str]:
    """Get the effective API key for a service.

    Args:
        service: The service name ('embeddings' or 'reranking')

    Returns:
        The API key from environment variable
    """
    return NVIDIA_API_KEY
