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

"""Function to get UI window examples using FAISS vector search."""

import logging
import os
import time
from typing import Any, Dict, Optional

from ..services.telemetry import ensure_telemetry_initialized, telemetry
from ..services.ui_window_examples_retrieval import create_ui_window_examples_retriever, get_ui_window_examples

logger = logging.getLogger(__name__)


async def get_window_examples(
    request: str,
    top_k: int = 5,
    format_type: str = "formatted",
    embedding_config: Optional[Dict[str, Any]] = None,
    faiss_index_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Get UI window examples using semantic search.

    Args:
        request: Query describing the desired UI window example
        top_k: Number of examples to return
        format_type: Format for results ("structured", "formatted", "raw")
        embedding_config: Configuration for embeddings
        faiss_index_path: Path to FAISS index (uses default if None)

    Returns:
        Dictionary with success status and result/error
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()

    # Record start time for telemetry
    start_time = time.perf_counter()

    # Prepare telemetry data
    telemetry_data = {
        "request": request,
        "top_k": top_k,
        "format_type": format_type,
        "has_embedding_config": embedding_config is not None,
        "has_faiss_index_path": faiss_index_path is not None,
    }

    success = True
    error_msg = None

    logger.info(f"[DEBUG] get_window_examples called with request: {request}")

    try:
        # Set default FAISS index path if not provided
        if faiss_index_path is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(current_dir, "data")
            faiss_index_path = os.path.join(data_dir, "ui_window_examples_faiss")

        # Extract API key from embedding config
        api_key = ""
        if embedding_config and "api_key" in embedding_config:
            api_key = embedding_config["api_key"] or ""

        # Create retriever
        retriever = create_ui_window_examples_retriever(faiss_index_path=faiss_index_path, api_key=api_key, top_k=top_k)

        # Get results
        results = get_ui_window_examples(user_query=request, retriever=retriever, top_k=top_k, format_type=format_type)

        if not results:
            return {"success": False, "error": "No UI window examples found for the given query", "result": ""}

        # Format result based on type
        if format_type == "structured":
            result_text = f"Found {len(results)} UI window examples:\n\n"
            for i, result in enumerate(results, 1):
                result_text += f"### Example {i}\n"
                result_text += f"**File:** `{result['file_path']}`\n"
                result_text += f"**Function:** `{result['class_name']}.{result['function_name']}()` (Line {result['line_number']})\n\n"
                result_text += f"**Description:**\n{result['description']}\n\n"
                result_text += f"**Code:**\n```python\n{result['code']}\n```\n\n"
                result_text += "---\n\n"
        elif format_type == "formatted":
            result_text = results  # Already formatted as string
        else:  # raw
            result_text = str(results)

        logger.info(
            f"Successfully retrieved {len(results) if isinstance(results, list) else 'formatted'} UI window examples"
        )

        return {"success": True, "result": result_text, "count": len(results) if isinstance(results, list) else None}

    except Exception as e:
        error_msg = f"Failed to retrieve UI window examples: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}

    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_window_examples",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg,
        )
