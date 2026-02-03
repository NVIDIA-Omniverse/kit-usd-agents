================================================================================
OMNI_UI_MCP SRC FILES CONCATENATION
================================================================================

FILE STRUCTURE:
----------------------------------------

📁 src//
  └── 📁 omni_aiq_omni_ui/
      ├── 📄 __init__.py
      ├── 📄 __main__.py
      ├── 📄 config.py
      ├── 📁 functions/
      │   ├── 📄 __init__.py
      │   ├── 📄 get_ui_class_detail.py
      │   ├── 📄 get_ui_class_instructions.py
      │   ├── 📄 list_ui_classes.py
      │   ├── 📄 search_ui_code_examples.py
      │   ├── 📄 get_instructions.py
      │   ├── 📄 get_ui_method_detail.py
      │   ├── 📄 get_ui_module_detail.py
      │   ├── 📄 list_ui_modules.py
      │   ├── 📄 get_ui_style_docs.py
      │   └── 📄 search_ui_window_examples.py
      ├── 📁 models/
      │   └── 📄 __init__.py
      ├── 📄 register_get_class_detail.py
      ├── 📄 register_get_class_detail_old.py
      ├── 📄 register_get_class_instructions.py
      ├── 📄 register_get_classes.py
      ├── 📄 register_search_ui_code_examples.py
      ├── 📄 register_get_instructions.py
      ├── 📄 register_get_method_detail.py
      ├── 📄 register_get_module_detail.py
      ├── 📄 register_get_modules.py
      ├── 📄 register_get_style_docs.py
      ├── 📄 register_search_ui_window_examples.py
      ├── 📁 services/
      │   ├── 📄 __init__.py
      │   ├── 📄 analyze_ui_functions.py
      │   ├── 📄 build_complete_ui_examples_pipeline.py
      │   ├── 📄 build_embedding_vectors.py
      │   ├── 📄 build_faiss_database.py
      │   ├── 📄 omni_ui_atlas.py
      │   ├── 📄 reranking.py
      │   ├── 📄 retrieval.py
      │   ├── 📄 telemetry.py
      │   ├── 📄 test_ui_window_examples.py
      │   └── 📄 ui_window_examples_retrieval.py
      └── 📁 utils/
          ├── 📄 __init__.py
          ├── 📄 fuzzy_matching.py
          ├── 📄 usage_logging.py
          └── 📄 usage_logging_decorator.py

================================================================================
SUMMARY STATISTICS:
----------------------------------------
Total Python files processed: 41
Total size: 244.89 KB
Total tokens: 51,896

FOLDER BREAKDOWN:
----------------------------------------
📁 src/omni_aiq_omni_ui/
   Files:  14 | Size:   93.53 KB | Tokens:     19,585
📁 src/omni_aiq_omni_ui/functions/
   Files:  11 | Size:   72.88 KB | Tokens:     15,287
📁 src/omni_aiq_omni_ui/services/
   Files:  11 | Size:   68.45 KB | Tokens:     14,990
📁 src/omni_aiq_omni_ui/utils/
   Files:   4 | Size:    9.99 KB | Tokens:      2,026
📁 src/omni_aiq_omni_ui/models/
   Files:   1 | Size:    38.00 B | Tokens:          8

TOP 10 FILES BY TOKEN COUNT:
----------------------------------------
 1. src/omni_aiq_omni_ui/functions/get_ui_class_instructions.py      3,950 tokens
 2. src/omni_aiq_omni_ui/services/omni_ui_atlas.py          3,099 tokens
 3. src/omni_aiq_omni_ui/register_get_class_instructions.py      2,502 tokens
 4. src/omni_aiq_omni_ui/register_get_style_docs.py         1,976 tokens
 5. src/omni_aiq_omni_ui/register_get_module_detail.py      1,903 tokens
 6. src/omni_aiq_omni_ui/register_get_method_detail.py      1,889 tokens
 7. src/omni_aiq_omni_ui/register_get_class_detail_old.py      1,796 tokens
 8. src/omni_aiq_omni_ui/register_get_class_detail.py       1,794 tokens
 9. src/omni_aiq_omni_ui/services/ui_window_examples_retrieval.py      1,767 tokens
10. src/omni_aiq_omni_ui/functions/get_ui_style_docs.py        1,725 tokens

FILE TYPE ANALYSIS:
----------------------------------------
  Registration Files   Files:  11 | Tokens:     17,554
  Function Files       Files:  10 | Tokens:     15,279
  Service Files        Files:   9 | Tokens:     14,229
  Utility Files        Files:   3 | Tokens:      2,018
  Other Files          Files:   2 | Tokens:      1,659
  Test Files           Files:   1 | Tokens:        729
  Init Files           Files:   5 | Tokens:        428

================================================================================
================================================================================


================================================================================
FILE: src/omni_aiq_omni_ui/__init__.py
Size: 1.64 KB | Tokens: 372
================================================================================

"""OmniUI tools for AgentIQ."""


# Patch AIQ before anything else
def _patch_aiq_validation():
    try:
        from aiq.data_models.config import AIQConfig
        from pydantic import ConfigDict

        # Override the model config to allow extra fields
        AIQConfig.model_config = ConfigDict(extra="allow")
        print("Patched AIQConfig to allow extra fields in YAML")
    except ImportError:
        # AIQ not installed or different version
        pass
    except Exception as e:
        # Silently fail if patching doesn't work
        print(f"Warning: Could not patch AIQConfig: {e}")


# Apply patch on import
_patch_aiq_validation()

import logging
import os
import pathlib

from .config import USAGE_LOGGING_ENABLED_BY_DEFAULT, USAGE_LOGGING_TIMEOUT
from .utils.usage_logging import create_usage_logger


def _get_version():
    version_file = pathlib.Path(__file__).parent.parent.parent / "VERSION.md"
    try:
        with open(version_file, "r", encoding="utf-8") as f:
            return f.readline().strip()
    except Exception:
        return "unknown"


__version__ = "0.10.0"

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize default usage logger on package import
# Check environment variable to allow disabling via env
disable_logging_value = os.environ.get("OMNI_UI_DISABLE_USAGE_LOGGING", "").lower()
usage_logging_enabled = disable_logging_value not in ("true", "1", "yes", "on")

if usage_logging_enabled and USAGE_LOGGING_ENABLED_BY_DEFAULT:
    logger.info("Initializing usage logging for OmniUI tools")
    create_usage_logger(enabled=True)
else:
    # Create disabled logger
    create_usage_logger(enabled=False)


================================================================================
FILE: src/omni_aiq_omni_ui/__main__.py
Size: 4.11 KB | Tokens: 963
================================================================================

"""
Main entry point for the OmniUI Tools package.
This allows the package to be run as a module with: python -m omni_aiq_omni_ui
"""

import os
import subprocess
import sys
from pathlib import Path

from . import _get_version


def main():
    """Main entry point for the OmniUI Tools package - starts the MCP server."""
    print(f"OmniUI Tools v{_get_version()}")
    print("Starting MCP server with AgentIQ...")

    # Check if a config file was provided as an argument
    if len(sys.argv) > 1:
        config_file = Path(sys.argv[1])
        if not config_file.exists():
            print(f"ERROR: Specified config file does not exist: {config_file}")
            return 1
    else:
        # Find the config file in default locations
        # Get the package root directory (where pyproject.toml is located)
        package_root = Path(__file__).parent.parent.parent.parent

        config_paths = [
            Path("workflow/local_config.yaml"),  # Current directory - local development
            Path("workflow/config.yaml"),  # Current directory - production
            package_root / "workflow" / "local_config.yaml",  # Package root - local development
            package_root / "workflow" / "config.yaml",  # Package root - production
            Path(__file__).parent.parent.parent / "workflow" / "local_config.yaml",  # Relative to src
            Path(__file__).parent.parent.parent / "workflow" / "config.yaml",  # Relative to src
            Path("/app/workflow/config.yaml"),  # Docker path
        ]

        config_file = None
        for path in config_paths:
            if path.exists():
                config_file = path
                break

        if not config_file:
            print("ERROR: Could not find config file")
            print("Searched in:")
            for path in config_paths:
                print(f"  - {path}")
            print("\nUsage: omni-ui-aiq [config_file]")
            print("\nFor local development, ensure workflow/local_config.yaml exists")
            return 1

    print(f"Using config file: {config_file}")

    # Detect development mode based on config file name
    is_dev_mode = "local_config" in str(config_file)
    if is_dev_mode:
        print("Running in DEVELOPMENT mode")

    # Run the MCP server
    # When running in Docker, we need to bind to 0.0.0.0 instead of localhost
    cmd = ["aiq", "mcp", "--config_file", str(config_file)]

    # Always filter to only expose functions and exclude any workflows (cleanest approach)
    print("Filtering to expose only OmniUI functions (excludes any workflows)")
    omni_ui_tools = [
        "search_ui_code_examples",
        "search_ui_window_examples",
        "list_ui_classes",
        "list_ui_modules",
        "get_ui_class_detail",
        "get_ui_module_detail",
        "get_ui_method_detail",
        "get_instructions",
        "get_ui_class_instructions",
        "get_ui_style_docs",
    ]

    for tool in omni_ui_tools:
        cmd.extend(["--tool_names", tool])

    # Check if we're running in a Docker container by checking for .dockerenv
    if True or os.path.exists("/.dockerenv"):
        # Add host binding for Docker
        cmd.extend(["--host", "0.0.0.0"])

    # Check for PORT environment variable
    port = os.environ.get("MCP_PORT", "9901")
    cmd.extend(["--port", port])

    print(f"Starting MCP server on port {port}...")
    if is_dev_mode:
        print("Development server will use verbose logging and localhost binding")

    # Show the full command for debugging in dev mode
    if is_dev_mode:
        print(f"Command: {' '.join(cmd)}")

    try:
        # Use subprocess.run to execute the command and wait for it
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("ERROR: 'aiq' command not found. Make sure aiqtoolkit is installed.")
        print("Run the setup script first: setup-dev.bat (Windows) or ./setup-dev.sh (Unix)")
        return 1
    except KeyboardInterrupt:
        print("\nShutting down MCP server...")
        return 0
    except Exception as e:
        print(f"ERROR: Failed to start MCP server: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


================================================================================
FILE: src/omni_aiq_omni_ui/config.py
Size: 2.58 KB | Tokens: 696
================================================================================

"""Configuration module for OmniUI tools."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

# Get the package directory
PACKAGE_DIR = Path(__file__).parent

# Data paths - relative to package directory
DATA_DIR = PACKAGE_DIR / "data"
FAISS_CODE_INDEX_PATH = DATA_DIR / "faiss_index_omni_ui"
UI_ATLAS_FILE_PATH = DATA_DIR / "ui_atlas.json"
OMNI_UI_RAG_COLLECTION_PATH = DATA_DIR / "omni_ui_rag_collection.json"

# API Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Default configuration values
DEFAULT_MCP_PORT = 9901
DEFAULT_TIMEOUT = 30.0

# Usage logging configuration
USAGE_LOGGING_ENABLED_BY_DEFAULT = True
USAGE_LOGGING_TIMEOUT = 30.0

# OpenSearch configuration for usage analytics
OPEN_SEARCH_URL = "https://search-omnigenai-usage-e6htsydkjhq7tktdqbflrqg3aa.us-west-2.es.amazonaws.com"

# RAG Configuration for OmniUI Code Examples
DEFAULT_RAG_LENGTH_CODE = 30000
DEFAULT_RAG_TOP_K_CODE = 90
DEFAULT_RERANK_CODE = 10

# Reranking Configuration
DEFAULT_RERANK_MODEL = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
DEFAULT_RERANK_ENDPOINT = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"

# Embedding Configuration
DEFAULT_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
DEFAULT_EMBEDDING_ENDPOINT = "https://ai.api.nvidia.com/v1"

# Environment variable names
ENV_DISABLE_LOGGING = "OMNI_UI_DISABLE_USAGE_LOGGING"
ENV_MCP_PORT = "MCP_PORT"


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


================================================================================
FILE: src/omni_aiq_omni_ui/functions/__init__.py
Size: 41.00 B | Tokens: 8
================================================================================

"""Functions module for OmniUI tools."""


================================================================================
FILE: src/omni_aiq_omni_ui/functions/get_ui_class_detail.py
Size: 6.69 KB | Tokens: 1,377
================================================================================

"""
Function to retrieve detailed OmniUI class information from Atlas data.
"""

import json
import logging
import time
from typing import Any, Dict

from ..services.omni_ui_atlas import OmniUIAtlasService
from ..services.telemetry import telemetry, ensure_telemetry_initialized

logger = logging.getLogger(__name__)


def get_atlas_service() -> OmniUIAtlasService:
    """Get or create the global OmniUI Atlas service instance.

    Returns:
        The OmniUI Atlas service instance
    """
    from .list_ui_classes import get_atlas_service as _get_atlas_service

    return _get_atlas_service()


async def get_ui_class_detail(class_names) -> Dict[str, Any]:
    """Get detailed information about one or more OmniUI classes.

    Args:
        class_names: List of class names to look up, or None to get available classes info

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: A JSON string containing detailed class information (single or array)
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()
    
    # Record start time for telemetry
    start_time = time.perf_counter()
    
    # Prepare telemetry data
    telemetry_data = {
        "class_names": class_names,
        "is_none": class_names is None,
        "is_string": isinstance(class_names, str),
        "is_list": isinstance(class_names, list),
        "count": len(class_names) if isinstance(class_names, list) else (1 if isinstance(class_names, str) else 0)
    }
    
    success = True
    error_msg = None
    
    try:
        # Handle None input (get available classes info)
        if class_names is None:
            # Return information about available classes
            atlas_service = get_atlas_service()
            if not atlas_service.is_available():
                return {"success": False, "error": "OmniUI Atlas data is not available", "result": ""}

            # Get basic info about available classes
            available_classes = atlas_service.get_class_list()
            result = {
                "available_classes": available_classes,
                "total_count": len(available_classes),
                "usage": "Provide specific class names to get detailed information",
            }
            return {"success": True, "result": json.dumps(result, indent=2), "error": None}

        # Handle list input
        elif isinstance(class_names, list):
            if len(class_names) == 0:
                return {"success": False, "error": "class_names array cannot be empty", "result": ""}
            # Check for empty strings in the list
            empty_names = [i for i, name in enumerate(class_names) if not isinstance(name, str) or not name.strip()]
            if empty_names:
                return {
                    "success": False,
                    "error": f"class_names contains empty or non-string values at indices: {empty_names}",
                    "result": "",
                }
        # Handle legacy string input (should not happen with new API, but kept for safety)
        elif isinstance(class_names, str):
            if not class_names.strip():
                return {"success": False, "error": "class_name cannot be empty or whitespace", "result": ""}
            class_names = [class_names]
        else:
            actual_type = type(class_names).__name__
            return {
                "success": False,
                "error": f"class_names must be None or a list of strings, but got {actual_type}: {class_names}",
                "result": "",
            }

        logger.info(f"Retrieving detailed info for {len(class_names)} OmniUI class(es): {class_names}")

        atlas_service = get_atlas_service()

        if not atlas_service.is_available():
            error_msg = "OmniUI Atlas data is not available"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        results = []
        errors = []

        for class_name in class_names:
            try:
                # Get detailed class information
                class_detail = atlas_service.get_ui_class_detail(class_name)

                if "error" in class_detail:
                    error_msg = f"{class_name}: {class_detail['error']}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    # Include error in results for partial success
                    results.append({"class_name": class_name, "error": class_detail["error"]})
                else:
                    results.append(class_detail)
                    logger.info(
                        f"Successfully retrieved details for class: {class_detail.get('full_name', class_name)}"
                    )

            except Exception as e:
                error_msg = f"{class_name}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                results.append({"class_name": class_name, "error": str(e)})

        # Return single result if only one class was requested
        if len(class_names) == 1:
            if results and "error" not in results[0]:
                result_json = json.dumps(results[0], indent=2)
            else:
                return {"success": False, "error": errors[0] if errors else "Unknown error", "result": ""}
        else:
            # Return array of results for multiple classes
            result_json = json.dumps(
                {
                    "classes": results,
                    "total_requested": len(class_names),
                    "successful": len([r for r in results if "error" not in r]),
                    "failed": len([r for r in results if "error" in r]),
                },
                indent=2,
            )

        # Determine overall success (at least one succeeded)
        overall_success = any("error" not in r for r in results)

        if not overall_success:
            return {"success": False, "error": "; ".join(errors), "result": ""}

        return {"success": True, "result": result_json, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving class details: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}
    
    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_ui_class_detail",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg
        )


================================================================================
FILE: src/omni_aiq_omni_ui/functions/get_ui_class_instructions.py
Size: 18.78 KB | Tokens: 3,950
================================================================================

"""Function to retrieve specific OmniUI class instructions from categorized files."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..services.telemetry import telemetry, ensure_telemetry_initialized

logger = logging.getLogger(__name__)

# Define the available categories and their descriptions
CATEGORIES = {
    "models": {
        "description": "Data models and delegates for UI components",
        "classes": ["AbstractValueModel", "AbstractItemModel", "AbstractItemDelegate"],
    },
    "shapes": {
        "description": "Basic shapes and geometric primitives",
        "classes": [
            "Rectangle",
            "FreeRectangle",
            "Circle",
            "FreeCircle",
            "Ellipse",
            "FreeEllipse",
            "Triangle",
            "FreeTriangle",
            "Line",
            "FreeLine",
            "BezierCurve",
            "FreeBezierCurve",
        ],
    },
    "widgets": {
        "description": "Interactive UI widgets and controls",
        "classes": [
            "Button",
            "RadioButton",
            "ToolButton",
            "CheckBox",
            "ComboBox",
            "Label",
            "Image",
            "ImageWithProvider",
            "Plot",
            "ColorWidget",
            "ProgressBar",
            "TreeView",
        ],
    },
    "containers": {
        "description": "Layout containers and frames",
        "classes": ["Frame", "CanvasFrame", "ScrollingFrame", "CollapsableFrame", "HStack", "VStack", "ZStack"],
    },
    "layouts": {"description": "Layout management and positioning", "classes": ["VGrid", "HGrid", "Placer"]},
    "inputs": {
        "description": "Input controls and field widgets",
        "classes": [
            "FloatSlider",
            "IntSlider",
            "FloatDrag",
            "IntDrag",
            "MultiFloatField",
            "MultiIntField",
            "AbstractMultiField",
        ],
    },
    "windows": {
        "description": "Windows, dialogs, and menus",
        "classes": ["Window", "MainWindow", "Menu", "MenuBar", "Tooltip", "Separator"],
    },
    "scene": {
        "description": "3D scene UI components from omni.ui.scene",
        "classes": ["Line", "Curve", "Rectangle", "Arc", "Image", "Points", "PolygonMesh", "TexturedMesh", "Label"],
    },
    "units": {"description": "Unit and measurement types", "classes": ["Pixel", "Percent", "Fraction"]},
    "system": {"description": "System and styling components", "classes": ["Style"]},
}


def get_classes_directory() -> Path:
    """Get the path to the classes directory."""
    # This file is at: src/omni_aiq_omni_ui/functions/get_ui_class_instructions.py
    # Classes are at: src/omni_aiq_omni_ui/data/instructions/classes/
    return Path(__file__).parent.parent / "data" / "instructions" / "classes"


def normalize_class_name(class_name: str) -> tuple[str, str]:
    """
    Normalize class name and determine if it's a scene class.

    Args:
        class_name: Class name like "Button", "scene.Line", or "omni.ui.scene.Line"

    Returns:
        Tuple of (normalized_name, category_hint)
    """
    # Remove omni.ui prefix if present
    if class_name.startswith("omni.ui."):
        class_name = class_name[8:]  # Remove "omni.ui."

    # Check if it's a scene class
    if class_name.startswith("scene."):
        return class_name.replace("scene.", ""), "scene"

    return class_name, None


def find_class_file(class_name: str, classes_dir: Path) -> Optional[tuple[Path, str]]:
    """
    Find the file for a given class name with smart resolution.

    Args:
        class_name: The class name to find
        classes_dir: The classes directory path

    Returns:
        Tuple of (file_path, category) if found, None otherwise
    """
    # Try multiple variations of the class name
    search_variations = [
        class_name,  # Original name
    ]

    # Add variations with prefixes removed/added
    if class_name.startswith("omni.ui."):
        search_variations.append(class_name[8:])  # Remove "omni.ui."
    else:
        search_variations.append(f"omni.ui.{class_name}")  # Add "omni.ui."

    if class_name.startswith("omni.ui.scene."):
        search_variations.append(class_name[8:])  # "scene.ClassName"
        search_variations.append(class_name[15:])  # "ClassName"
    elif class_name.startswith("scene."):
        search_variations.append(class_name[6:])  # "ClassName"
        search_variations.append(f"omni.ui.{class_name}")  # "omni.ui.scene.ClassName"
    elif not class_name.startswith("omni.ui.") and not class_name.startswith("scene."):
        search_variations.append(f"scene.{class_name}")  # Try as scene class
        search_variations.append(f"omni.ui.scene.{class_name}")

    # Remove duplicates while preserving order
    unique_variations = []
    for var in search_variations:
        if var not in unique_variations:
            unique_variations.append(var)

    # Try each variation
    for variation in unique_variations:
        result = _find_class_file_single(variation, classes_dir)
        if result:
            return result

    return None


def _find_class_file_single(class_name: str, classes_dir: Path) -> Optional[tuple[Path, str]]:
    """Helper function to find a single class file variation."""
    normalized_name, category_hint = normalize_class_name(class_name)

    # If we have a category hint (like scene), search there first
    if category_hint and category_hint in CATEGORIES:
        category_dir = classes_dir / category_hint
        # Try scene_ClassName format for scene classes
        scene_file = category_dir / f"scene_{normalized_name}.md"
        if scene_file.exists():
            return scene_file, category_hint
        # Try regular format
        regular_file = category_dir / f"{normalized_name}.md"
        if regular_file.exists():
            return regular_file, category_hint

    # Search all categories with case-insensitive matching
    for category, info in CATEGORIES.items():
        # Try exact match first
        if normalized_name in info["classes"]:
            category_dir = classes_dir / category
            file_path = category_dir / f"{normalized_name}.md"
            if file_path.exists():
                return file_path, category

            # For scene classes, try the scene_ prefix format
            if category == "scene":
                scene_file_path = category_dir / f"scene_{normalized_name}.md"
                if scene_file_path.exists():
                    return scene_file_path, category

        # Try case-insensitive match
        for class_in_category in info["classes"]:
            if normalized_name.lower() == class_in_category.lower():
                category_dir = classes_dir / category
                file_path = category_dir / f"{class_in_category}.md"
                if file_path.exists():
                    return file_path, category

                # For scene classes, try the scene_ prefix format
                if category == "scene":
                    scene_file_path = category_dir / f"scene_{class_in_category}.md"
                    if scene_file_path.exists():
                        return scene_file_path, category

    return None


async def get_ui_class_instructions(class_names) -> Dict[str, Any]:
    """
    Retrieve specific OmniUI class instructions for one or more classes.

    Args:
        class_names: List of class names to look up, or None to list categories. Can be:
            - Simple name: "Button", "Label", "TreeView"
            - Scene class: "scene.Line", "scene.Rectangle"
            - Full name: "omni.ui.Button", "omni.ui.scene.Line"

    Returns:
        Dictionary with:
        - success: Whether retrieval was successful
        - result: The class instruction content if successful (single or multiple)
        - error: Error message if failed
        - metadata: Additional information about the class(es)
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()
    
    # Record start time for telemetry
    start_time = time.perf_counter()
    
    # Prepare telemetry data
    telemetry_data = {
        "class_names": class_names,
        "is_none": class_names is None,
        "is_string": isinstance(class_names, str),
        "is_list": isinstance(class_names, list),
        "count": len(class_names) if isinstance(class_names, list) else (1 if isinstance(class_names, str) else 0)
    }
    
    success = True
    error_msg = None
    
    try:
        # Handle None input (list categories)
        if class_names is None:
            return await list_class_categories()

        # Handle list input
        elif isinstance(class_names, list):
            if len(class_names) == 0:
                return {"success": False, "error": "class_names array cannot be empty", "result": None}
            # Check for empty strings in the list
            empty_names = [i for i, name in enumerate(class_names) if not isinstance(name, str) or not name.strip()]
            if empty_names:
                return {
                    "success": False,
                    "error": f"class_names contains empty or non-string values at indices: {empty_names}",
                    "result": None,
                }
        # Handle legacy string input (should not happen with new API, but kept for safety)
        elif isinstance(class_names, str):
            if not class_names.strip():
                return {"success": False, "error": "class_name cannot be empty or whitespace", "result": None}
            class_names = [class_names]
        else:
            actual_type = type(class_names).__name__
            return {
                "success": False,
                "error": f"class_names must be None or a list of strings, but got {actual_type}: {class_names}",
                "result": None,
            }

        logger.info(f"Retrieving instructions for {len(class_names)} OmniUI class(es): {class_names}")

        classes_dir = get_classes_directory()
        results = []
        errors = []
        all_metadata = []

        for class_name in class_names:
            try:
                # Find the class file
                file_info = find_class_file(class_name, classes_dir)
                if not file_info:
                    error_msg = f"Class '{class_name}' not found"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    results.append({"class_name": class_name, "error": error_msg, "content": None})
                    continue

                file_path, category = file_info

                # Read the class instruction content
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    error_msg = f"Failed to read class file for '{class_name}': {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    results.append({"class_name": class_name, "error": error_msg, "content": None})
                    continue

                # Prepare metadata - use the actual class name from the file
                file_stem = file_path.stem  # e.g., "TreeView" from "TreeView.md"
                if file_stem.startswith("scene_"):
                    actual_class_name = file_stem[6:]  # Remove "scene_" prefix
                else:
                    actual_class_name = file_stem

                metadata = {
                    "class_name": class_name,
                    "normalized_name": actual_class_name,  # Use actual class name from file
                    "category": category,
                    "category_description": CATEGORIES[category]["description"],
                    "file_path": str(file_path.relative_to(classes_dir)),
                    "content_length": len(content),
                    "line_count": content.count("\n") + 1,
                }

                all_metadata.append(metadata)
                results.append({"class_name": class_name, "content": content, "metadata": metadata})

                logger.info(
                    f"Successfully retrieved class instructions for '{class_name}' from {category} category ({metadata['line_count']} lines)"
                )

            except Exception as e:
                error_msg = f"Unexpected error for '{class_name}': {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                results.append({"class_name": class_name, "error": error_msg, "content": None})

        # Return single result if only one class was requested
        if len(class_names) == 1:
            if results and results[0].get("content"):
                return {"success": True, "result": results[0]["content"], "metadata": results[0].get("metadata")}
            else:
                # Create helpful error message for single class
                error_lines = [f"Class '{class_names[0]}' not found.\n\nAvailable classes by category:"]
                for category, info in CATEGORIES.items():
                    error_lines.append(f"\n**{category}** ({len(info['classes'])} classes):")
                    for cls in sorted(info["classes"]):
                        if category == "scene":
                            error_lines.append(f"  - omni.ui.scene.{cls}")
                        else:
                            error_lines.append(f"  - omni.ui.{cls}")
                return {
                    "success": False,
                    "error": "\n".join(error_lines),
                    "result": None,
                }
        else:
            # Format multiple results
            successful_results = [r for r in results if r.get("content")]
            failed_results = [r for r in results if not r.get("content")]

            # Combine all content with headers
            combined_content = []
            for result in successful_results:
                combined_content.append(f"# Class: {result['class_name']}")
                combined_content.append(f"## Category: {result['metadata']['category']}")
                combined_content.append(result["content"])
                combined_content.append("\n---\n")

            if not successful_results:
                return {"success": False, "error": "; ".join(errors), "result": None}

            return {
                "success": True,
                "result": "\n".join(combined_content),
                "metadata": {
                    "total_requested": len(class_names),
                    "successful": len(successful_results),
                    "failed": len(failed_results),
                    "classes": all_metadata,
                    "errors": errors if errors else None,
                },
            }

    except Exception as e:
        logger.error(f"Unexpected error retrieving class instructions: {e}")
        error_msg = f"Unexpected error: {str(e)}"
        success = False
        return {"success": False, "error": error_msg, "result": None}
    
    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_ui_class_instructions",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg
        )


async def list_class_categories() -> Dict[str, Any]:
    """
    List all available class categories with their descriptions and class counts.

    Returns:
        Dictionary with:
        - success: Whether listing was successful
        - result: Formatted list of categories
        - categories: Detailed information about each category
    """
    try:
        result_lines = ["# OmniUI Class Categories\n"]
        categories_info = []

        for category, info in CATEGORIES.items():
            class_count = len(info["classes"])
            categories_info.append(
                {
                    "name": category,
                    "description": info["description"],
                    "class_count": class_count,
                    "classes": info["classes"],
                }
            )

            result_lines.append(f"\n## {category} ({class_count} classes)")
            result_lines.append(f"{info['description']}")
            result_lines.append(f"\nClasses: {', '.join(info['classes'])}")

        result_lines.append(f"\n\nTotal categories: {len(CATEGORIES)}")
        result_lines.append(f"Total classes: {sum(len(info['classes']) for info in CATEGORIES.values())}")

        return {"success": True, "result": "\n".join(result_lines), "categories": categories_info}

    except Exception as e:
        logger.error(f"Failed to list class categories: {e}")
        return {"success": False, "error": f"Failed to list categories: {str(e)}", "result": None}


async def list_classes_in_category(category: str) -> Dict[str, Any]:
    """
    List all classes in a specific category.

    Args:
        category: The category name (e.g., "widgets", "shapes", "scene")

    Returns:
        Dictionary with:
        - success: Whether listing was successful
        - result: Formatted list of classes in the category
        - classes: List of class names in the category
    """
    try:
        # Handle None or empty category
        if not category or category.lower() == "none":
            available_categories = list(CATEGORIES.keys())
            return {
                "success": False,
                "error": f"No category specified. Available categories: {', '.join(available_categories)}",
                "result": None,
            }

        if category not in CATEGORIES:
            available_categories = list(CATEGORIES.keys())
            return {
                "success": False,
                "error": f"Unknown category '{category}'. Available categories: {', '.join(available_categories)}",
                "result": None,
            }

        category_info = CATEGORIES[category]
        classes = category_info["classes"]

        result_lines = [f"# {category.title()} Classes"]
        result_lines.append(f"\n{category_info['description']}")
        result_lines.append(f"\nThis category contains {len(classes)} classes:\n")

        for class_name in sorted(classes):
            if category == "scene":
                full_name = f"omni.ui.scene.{class_name}"
            else:
                full_name = f"omni.ui.{class_name}"
            result_lines.append(f"- **{full_name}**")

        return {
            "success": True,
            "result": "\n".join(result_lines),
            "classes": classes,
            "category": category,
            "description": category_info["description"],
        }

    except Exception as e:
        logger.error(f"Failed to list classes in category '{category}': {e}")
        return {"success": False, "error": f"Failed to list classes: {str(e)}", "result": None}


================================================================================
FILE: src/omni_aiq_omni_ui/functions/list_ui_classes.py
Size: 3.27 KB | Tokens: 699
================================================================================

"""
Function to retrieve OmniUI class information from Atlas data.
"""

import json
import logging
import time
from typing import Any, Dict

from ..services.omni_ui_atlas import OmniUIAtlasService
from ..services.telemetry import telemetry, ensure_telemetry_initialized

logger = logging.getLogger(__name__)

# Global atlas service instance
_atlas_service = None


def get_atlas_service() -> OmniUIAtlasService:
    """Get or create the global OmniUI Atlas service instance.

    Returns:
        The OmniUI Atlas service instance
    """
    global _atlas_service
    if _atlas_service is None:
        _atlas_service = OmniUIAtlasService()
    return _atlas_service


async def list_ui_classes() -> Dict[str, Any]:
    """Return a list of all OmniUI class full names from the Atlas data.

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: A JSON string containing all OmniUI class full names
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()
    
    # Record start time for telemetry
    start_time = time.perf_counter()
    
    # Prepare telemetry data (no parameters for this function)
    telemetry_data = {}
    
    success = True
    error_msg = None
    
    try:
        logger.info("Retrieving OmniUI classes from Atlas data")

        atlas_service = get_atlas_service()

        if not atlas_service.is_available():
            error_msg = "OmniUI Atlas data is not available"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        # Get the list of class names
        class_full_names = atlas_service.get_class_names_list()

        # Return the simplified result with just full names
        simplified_result = {
            "class_full_names": class_full_names,
            "total_count": len(class_full_names),
            "description": "OmniUI classes from Atlas data",
        }

        result_json = json.dumps(simplified_result, indent=2)

        logger.info(f"Successfully retrieved {len(class_full_names)} OmniUI classes from Atlas")
        return {"success": True, "result": result_json, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving OmniUI classes: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}
    
    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Capture telemetry data
        await telemetry.capture_call(
            function_name="list_ui_classes",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg
        )


def get_detailed_classes() -> Dict[str, Any]:
    """Return detailed class information including methods.

    This is used internally for future expansion but not exposed via MCP.

    Returns:
        Complete classes dictionary with full method information
    """
    atlas_service = get_atlas_service()
    if not atlas_service.is_available():
        return {"error": "OmniUI Atlas data is not available"}

    return atlas_service.list_ui_classes()


================================================================================
FILE: src/omni_aiq_omni_ui/functions/search_ui_code_examples.py
Size: 6.88 KB | Tokens: 1,574
================================================================================

"""Get code examples function implementation."""

import logging
import time
from typing import Any, Dict, Optional

from ..config import DEFAULT_RERANK_CODE, FAISS_CODE_INDEX_PATH, get_effective_api_key
from ..services.reranking import create_reranker_with_config
from ..services.retrieval import Retriever, get_rag_context_omni_ui_code
from ..services.telemetry import telemetry, ensure_telemetry_initialized

logger = logging.getLogger(__name__)

# Global services - will be initialized on first use
_code_retriever: Optional[Retriever] = None
_reranker = None
_retriever_initialized = False


def _initialize_retriever(embedding_config: Optional[Dict[str, Any]] = None):
    """Initialize retriever if not already done."""
    global _code_retriever, _retriever_initialized

    if _retriever_initialized and not embedding_config:
        return

    # Initialize retriever with provided config
    if FAISS_CODE_INDEX_PATH.exists():
        _code_retriever = Retriever(embedding_config=embedding_config, load_path=str(FAISS_CODE_INDEX_PATH))
    else:
        logger.warning(f"FAISS code index not found at {FAISS_CODE_INDEX_PATH}")

    _retriever_initialized = True


def _get_or_create_reranker(reranking_config: Optional[Dict[str, Any]] = None):
    """Lazily create and return the reranker."""
    global _reranker

    if _reranker is None or reranking_config:
        _reranker = create_reranker_with_config(reranking_config)
        if _reranker:
            logger.info("Reranker initialized for OmniUI code examples")

    return _reranker


async def search_ui_code_examples(
    request: str,
    rerank_k: int = DEFAULT_RERANK_CODE,
    enable_rerank: bool = True,
    embedding_config: Optional[Dict[str, Any]] = None,
    reranking_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Retrieves relevant OmniUI code examples using semantic vector search and optional reranking.

    This function performs a RAG (Retrieval-Augmented Generation) query against a curated database
    of OmniUI code examples. It uses FAISS vector similarity search with NVIDIA embeddings, followed by
    optional reranking for improved relevance.

    How it works:
    1. Converts your query to embeddings using NVIDIA's nv-embedqa-e5-v5 model
    2. Performs semantic similarity search against pre-indexed OmniUI code examples
    3. Optionally reranks results using NVIDIA's llama-3.2-nv-rerankqa-1b-v2 model
    4. Returns formatted code examples with their metadata

    Query matching: Your query is compared against OmniUI widget and component implementations,
    including examples like:
    - SearchField, SearchWordButton implementations
    - Widget styling and theming functions
    - UI component building patterns
    - Event handling and callback patterns

    Tips for better results:
    - Use specific OmniUI terminology (e.g., "SearchField", "ZStack", "VStack")
    - Frame queries as component or pattern names when possible
    - Include relevant UI operations (e.g., "build_ui", "style", "event handling")
    - Be specific about the UI component or pattern you want to implement

    Args:
        request: Your query describing the desired OmniUI code example.
                Examples: "How to create a search field?", "Button styling", "event handling"
        rerank_k: Number of documents to keep after reranking (default: DEFAULT_RERANK_CODE)
        enable_rerank: Whether to enable reranking of search results (default: True)
        embedding_config: Optional configuration for embedding service
        reranking_config: Optional configuration for reranking service

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: Formatted code examples with file paths and Python code snippets, or error message
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()
    
    # Record start time for telemetry
    start_time = time.perf_counter()
    
    # Prepare telemetry data
    telemetry_data = {
        "request": request,
        "rerank_k": rerank_k,
        "enable_rerank": enable_rerank,
        "has_embedding_config": embedding_config is not None,
        "has_reranking_config": reranking_config is not None
    }
    
    success = True
    error_msg = None
    
    try:
        # Debug logging
        logger.info(f"[DEBUG] search_ui_code_examples called with request: {request}")
        logger.info(f"[DEBUG] embedding_config: {embedding_config}")
        logger.info(f"[DEBUG] reranking_config: {reranking_config}")
        logger.info(f"[DEBUG] FAISS_CODE_INDEX_PATH: {FAISS_CODE_INDEX_PATH}")
        logger.info(f"[DEBUG] FAISS_CODE_INDEX_PATH exists: {FAISS_CODE_INDEX_PATH.exists()}")

        # Initialize retriever if needed
        _initialize_retriever(embedding_config)

        if not FAISS_CODE_INDEX_PATH.exists():
            error_msg = f"FAISS index not found at path: {FAISS_CODE_INDEX_PATH}. Please configure the path."
            logger.error(error_msg)
            success = False
            return {"success": False, "error": error_msg, "result": ""}

        if _code_retriever is None:
            error_msg = "Code retriever could not be initialized. Please check the configuration."
            logger.error(error_msg)
            success = False
            return {"success": False, "error": error_msg, "result": ""}

        # Get reranker only if reranking is enabled
        reranker_to_use = _get_or_create_reranker(reranking_config) if enable_rerank else None

        # Get the RAG context using the utility function with reranking
        rag_context = get_rag_context_omni_ui_code(
            user_query=request,
            retriever=_code_retriever,
            reranker=reranker_to_use,
            rerank_k=rerank_k,  # Pass the rerank_k parameter
        )

        if rag_context:
            logger.info(
                f"Retrieved OmniUI code context for '{request}' with reranking: {'enabled' if enable_rerank else 'disabled'}"
            )
            return {"success": True, "result": rag_context, "error": None}
        else:
            no_result_msg = "No relevant OmniUI code examples found for your request."
            logger.info(no_result_msg)
            return {"success": True, "result": no_result_msg, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving OmniUI code examples: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}
    
    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Capture telemetry data
        await telemetry.capture_call(
            function_name="search_ui_code_examples",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg
        )


================================================================================
FILE: src/omni_aiq_omni_ui/functions/get_instructions.py
Size: 7.89 KB | Tokens: 1,653
================================================================================

"""Function to retrieve OmniUI system instructions."""

import logging
import time
from pathlib import Path
from typing import Any, Dict

from ..services.telemetry import telemetry, ensure_telemetry_initialized

logger = logging.getLogger(__name__)

# Define the instruction files and their metadata
INSTRUCTION_FILES = {
    "agent_system": {
        "filename": "agent_system.md",
        "description": "Core Omniverse UI Assistant system prompt with omni.ui framework basics",
        "use_cases": [
            "Understanding omni.ui framework fundamentals",
            "Learning about omni.ui.scene for 3D UI",
            "Understanding widget filters and options menus",
            "Working with searchable comboboxes and search fields",
            "General omni.ui code writing guidelines",
            "Understanding UI placeholder patterns for scene operations",
        ],
    },
    "classes": {
        "filename": "classes.md",
        "description": "Comprehensive omni.ui class API reference and model patterns",
        "use_cases": [
            "Working with AbstractValueModel and data models",
            "Understanding SimpleStringModel, SimpleBoolModel, SimpleFloatModel, SimpleIntModel",
            "Creating custom model implementations",
            "Model callbacks and value change handling",
            "Understanding model-view patterns in omni.ui",
        ],
    },
    "omni_ui_scene_system": {
        "filename": "omni_ui_scene_system.md",
        "description": "Complete omni.ui.scene 3D UI system documentation",
        "use_cases": [
            "Creating 3D shapes (Line, Curve, Rectangle, Arc, etc.)",
            "Working with SceneView and camera controls",
            "Understanding Transform containers and matrices",
            "Implementing gestures and mouse interactions in 3D",
            "Building manipulators and custom 3D controls",
            "Syncing with USD camera and stage",
            "Using standard transform manipulators",
        ],
    },
    "omni_ui_system": {
        "filename": "omni_ui_system.md",
        "description": "Core omni.ui widgets, containers, layouts and styling system",
        "use_cases": [
            "Understanding basic UI shapes and widgets",
            "Working with Labels, Buttons, Fields, Sliders",
            "Creating layouts with HStack, VStack, ZStack, Grid",
            "Understanding Window creation and management",
            "Styling with selectors and style sheets",
            "Working with shades and color palettes",
            "Implementing drag & drop functionality",
            "Understanding Model-Delegate-View (MDV) pattern",
            "Managing callbacks and subscriptions",
        ],
    },
}


async def get_instructions(name: str) -> Dict[str, Any]:
    """
    Retrieve specific OmniUI system instructions by name.

    Args:
        name: The name of the instruction set to retrieve. Valid values are:
            - 'agent_system': Core system prompt and framework basics
            - 'classes': Class API reference and model patterns
            - 'omni_ui_scene_system': 3D UI system documentation
            - 'omni_ui_system': Core widgets, layouts and styling

    Returns:
        Dictionary with:
        - success: Whether retrieval was successful
        - result: The instruction content if successful
        - error: Error message if failed
        - metadata: Additional information about the instruction
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()
    
    # Record start time for telemetry
    start_time = time.perf_counter()
    
    # Prepare telemetry data
    telemetry_data = {
        "name": name
    }
    
    success = True
    error_msg = None
    
    try:
        # Validate instruction name
        if name not in INSTRUCTION_FILES:
            available = list(INSTRUCTION_FILES.keys())
            return {
                "success": False,
                "error": f"Unknown instruction name '{name}'. Available instructions: {', '.join(available)}",
                "result": None,
            }

        instruction_info = INSTRUCTION_FILES[name]

        # Get the instructions directory relative to this file
        # This file is at: src/omni_aiq_omni_ui/functions/get_instructions.py
        # Instructions are at: src/omni_aiq_omni_ui/data/instructions/
        instructions_dir = Path(__file__).parent.parent / "data" / "instructions"
        file_path = instructions_dir / instruction_info["filename"]

        # Check if file exists
        if not file_path.exists():
            logger.error(f"Instruction file not found: {file_path}")
            return {
                "success": False,
                "error": f"Instruction file not found: {instruction_info['filename']}",
                "result": None,
            }

        # Read the instruction content
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read instruction file {file_path}: {e}")
            return {"success": False, "error": f"Failed to read instruction file: {str(e)}", "result": None}

        # Prepare metadata
        metadata = {
            "name": name,
            "description": instruction_info["description"],
            "use_cases": instruction_info["use_cases"],
            "filename": instruction_info["filename"],
            "content_length": len(content),
            "line_count": content.count("\n") + 1,
        }

        # Format the result with metadata header
        result = f"""# OmniUI Instruction: {name}

## Description
{instruction_info['description']}

## Use Cases
This instruction set is useful for:
{chr(10).join(f"- {use_case}" for use_case in instruction_info['use_cases'])}

---

{content}"""

        logger.info(f"Successfully retrieved instruction '{name}' ({metadata['line_count']} lines)")

        return {"success": True, "result": result, "metadata": metadata}

    except Exception as e:
        logger.error(f"Unexpected error retrieving instruction '{name}': {e}")
        error_msg = f"Unexpected error: {str(e)}"
        success = False
        return {"success": False, "error": error_msg, "result": None}
    
    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_instructions",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg
        )


async def list_instructions() -> Dict[str, Any]:
    """
    List all available OmniUI system instructions.

    Returns:
        Dictionary with:
        - success: Whether listing was successful
        - result: Formatted list of available instructions
        - instructions: Detailed information about each instruction
    """
    try:
        instructions_list = []

        for name, info in INSTRUCTION_FILES.items():
            instructions_list.append({"name": name, "description": info["description"], "use_cases": info["use_cases"]})

        # Format result as readable text
        result_lines = ["# Available OmniUI Instructions\n"]

        for inst in instructions_list:
            result_lines.append(f"\n## {inst['name']}")
            result_lines.append(f"{inst['description']}")
            result_lines.append("\n**Use cases:**")
            for use_case in inst["use_cases"]:
                result_lines.append(f"  - {use_case}")

        result_lines.append(f"\n\nTotal instructions available: {len(instructions_list)}")

        return {"success": True, "result": "\n".join(result_lines), "instructions": instructions_list}

    except Exception as e:
        logger.error(f"Failed to list instructions: {e}")
        return {"success": False, "error": f"Failed to list instructions: {str(e)}", "result": None}


================================================================================
FILE: src/omni_aiq_omni_ui/functions/get_ui_method_detail.py
Size: 6.75 KB | Tokens: 1,379
================================================================================

"""
Function to retrieve detailed OmniUI method information from Atlas data.
"""

import json
import logging
import time
from typing import Any, Dict

from ..services.omni_ui_atlas import OmniUIAtlasService
from ..services.telemetry import telemetry, ensure_telemetry_initialized

logger = logging.getLogger(__name__)


def get_atlas_service() -> OmniUIAtlasService:
    """Get or create the global OmniUI Atlas service instance.

    Returns:
        The OmniUI Atlas service instance
    """
    from .list_ui_classes import get_atlas_service as _get_atlas_service

    return _get_atlas_service()


async def get_ui_method_detail(method_names) -> Dict[str, Any]:
    """Get detailed information about one or more OmniUI methods.

    Args:
        method_names: List of method names to look up, or None to get available methods info

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: A JSON string containing detailed method information (single or array)
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()
    
    # Record start time for telemetry
    start_time = time.perf_counter()
    
    # Prepare telemetry data
    telemetry_data = {
        "method_names": method_names,
        "is_none": method_names is None,
        "is_string": isinstance(method_names, str),
        "is_list": isinstance(method_names, list),
        "count": len(method_names) if isinstance(method_names, list) else (1 if isinstance(method_names, str) else 0)
    }
    
    success = True
    error_msg = None
    
    try:
        # Handle None input (get available methods info)
        if method_names is None:
            # Return information about available methods
            atlas_service = get_atlas_service()
            if not atlas_service.is_available():
                return {"success": False, "error": "OmniUI Atlas data is not available", "result": ""}

            # Get basic info about available methods
            available_methods = atlas_service.get_method_list()
            result = {
                "available_methods": available_methods,
                "total_count": len(available_methods),
                "usage": "Provide specific method names to get detailed information",
            }
            return {"success": True, "result": json.dumps(result, indent=2), "error": None}

        # Handle list input
        elif isinstance(method_names, list):
            if len(method_names) == 0:
                return {"success": False, "error": "method_names array cannot be empty", "result": ""}
            # Check for empty strings in the list
            empty_names = [i for i, name in enumerate(method_names) if not isinstance(name, str) or not name.strip()]
            if empty_names:
                return {
                    "success": False,
                    "error": f"method_names contains empty or non-string values at indices: {empty_names}",
                    "result": "",
                }
        # Handle legacy string input (should not happen with new API, but kept for safety)
        elif isinstance(method_names, str):
            if not method_names.strip():
                return {"success": False, "error": "method_name cannot be empty or whitespace", "result": ""}
            method_names = [method_names]
        else:
            actual_type = type(method_names).__name__
            return {
                "success": False,
                "error": f"method_names must be None or a list of strings, but got {actual_type}: {method_names}",
                "result": "",
            }

        logger.info(f"Retrieving detailed info for {len(method_names)} OmniUI method(s): {method_names}")

        atlas_service = get_atlas_service()

        if not atlas_service.is_available():
            error_msg = "OmniUI Atlas data is not available"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        results = []
        errors = []

        for method_name in method_names:
            try:
                # Get detailed method information
                method_detail = atlas_service.get_ui_method_detail(method_name)

                if "error" in method_detail:
                    error_msg = f"{method_name}: {method_detail['error']}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    # Include error in results for partial success
                    results.append({"method_name": method_name, "error": method_detail["error"]})
                else:
                    results.append(method_detail)
                    logger.info(
                        f"Successfully retrieved details for method: {method_detail.get('full_name', method_name)}"
                    )

            except Exception as e:
                error_msg = f"{method_name}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                results.append({"method_name": method_name, "error": str(e)})

        # Return single result if only one method was requested
        if len(method_names) == 1:
            if results and "error" not in results[0]:
                result_json = json.dumps(results[0], indent=2)
            else:
                return {"success": False, "error": errors[0] if errors else "Unknown error", "result": ""}
        else:
            # Return array of results for multiple methods
            result_json = json.dumps(
                {
                    "methods": results,
                    "total_requested": len(method_names),
                    "successful": len([r for r in results if "error" not in r]),
                    "failed": len([r for r in results if "error" in r]),
                },
                indent=2,
            )

        # Determine overall success (at least one succeeded)
        overall_success = any("error" not in r for r in results)

        if not overall_success:
            return {"success": False, "error": "; ".join(errors), "result": ""}

        return {"success": True, "result": result_json, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving method details: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}
    
    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_ui_method_detail",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg
        )


================================================================================
FILE: src/omni_aiq_omni_ui/functions/get_ui_module_detail.py
Size: 6.75 KB | Tokens: 1,379
================================================================================

"""
Function to retrieve detailed OmniUI module information from Atlas data.
"""

import json
import logging
import time
from typing import Any, Dict

from ..services.omni_ui_atlas import OmniUIAtlasService
from ..services.telemetry import telemetry, ensure_telemetry_initialized

logger = logging.getLogger(__name__)


def get_atlas_service() -> OmniUIAtlasService:
    """Get or create the global OmniUI Atlas service instance.

    Returns:
        The OmniUI Atlas service instance
    """
    from .list_ui_classes import get_atlas_service as _get_atlas_service

    return _get_atlas_service()


async def get_ui_module_detail(module_names) -> Dict[str, Any]:
    """Get detailed information about one or more OmniUI modules.

    Args:
        module_names: List of module names to look up, or None to get available modules info

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: A JSON string containing detailed module information (single or array)
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()
    
    # Record start time for telemetry
    start_time = time.perf_counter()
    
    # Prepare telemetry data
    telemetry_data = {
        "module_names": module_names,
        "is_none": module_names is None,
        "is_string": isinstance(module_names, str),
        "is_list": isinstance(module_names, list),
        "count": len(module_names) if isinstance(module_names, list) else (1 if isinstance(module_names, str) else 0)
    }
    
    success = True
    error_msg = None
    
    try:
        # Handle None input (get available modules info)
        if module_names is None:
            # Return information about available modules
            atlas_service = get_atlas_service()
            if not atlas_service.is_available():
                return {"success": False, "error": "OmniUI Atlas data is not available", "result": ""}

            # Get basic info about available modules
            available_modules = atlas_service.get_module_list()
            result = {
                "available_modules": available_modules,
                "total_count": len(available_modules),
                "usage": "Provide specific module names to get detailed information",
            }
            return {"success": True, "result": json.dumps(result, indent=2), "error": None}

        # Handle list input
        elif isinstance(module_names, list):
            if len(module_names) == 0:
                return {"success": False, "error": "module_names array cannot be empty", "result": ""}
            # Check for empty strings in the list
            empty_names = [i for i, name in enumerate(module_names) if not isinstance(name, str) or not name.strip()]
            if empty_names:
                return {
                    "success": False,
                    "error": f"module_names contains empty or non-string values at indices: {empty_names}",
                    "result": "",
                }
        # Handle legacy string input (should not happen with new API, but kept for safety)
        elif isinstance(module_names, str):
            if not module_names.strip():
                return {"success": False, "error": "module_name cannot be empty or whitespace", "result": ""}
            module_names = [module_names]
        else:
            actual_type = type(module_names).__name__
            return {
                "success": False,
                "error": f"module_names must be None or a list of strings, but got {actual_type}: {module_names}",
                "result": "",
            }

        logger.info(f"Retrieving detailed info for {len(module_names)} OmniUI module(s): {module_names}")

        atlas_service = get_atlas_service()

        if not atlas_service.is_available():
            error_msg = "OmniUI Atlas data is not available"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        results = []
        errors = []

        for module_name in module_names:
            try:
                # Get detailed module information
                module_detail = atlas_service.get_ui_module_detail(module_name)

                if "error" in module_detail:
                    error_msg = f"{module_name}: {module_detail['error']}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    # Include error in results for partial success
                    results.append({"module_name": module_name, "error": module_detail["error"]})
                else:
                    results.append(module_detail)
                    logger.info(
                        f"Successfully retrieved details for module: {module_detail.get('full_name', module_name)}"
                    )

            except Exception as e:
                error_msg = f"{module_name}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                results.append({"module_name": module_name, "error": str(e)})

        # Return single result if only one module was requested
        if len(module_names) == 1:
            if results and "error" not in results[0]:
                result_json = json.dumps(results[0], indent=2)
            else:
                return {"success": False, "error": errors[0] if errors else "Unknown error", "result": ""}
        else:
            # Return array of results for multiple modules
            result_json = json.dumps(
                {
                    "modules": results,
                    "total_requested": len(module_names),
                    "successful": len([r for r in results if "error" not in r]),
                    "failed": len([r for r in results if "error" in r]),
                },
                indent=2,
            )

        # Determine overall success (at least one succeeded)
        overall_success = any("error" not in r for r in results)

        if not overall_success:
            return {"success": False, "error": "; ".join(errors), "result": ""}

        return {"success": True, "result": result_json, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving module details: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}
    
    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_ui_module_detail",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg
        )


================================================================================
FILE: src/omni_aiq_omni_ui/functions/list_ui_modules.py
Size: 2.73 KB | Tokens: 583
================================================================================

"""
Function to retrieve OmniUI module information from Atlas data.
"""

import json
import logging
import time
from typing import Any, Dict

from ..services.omni_ui_atlas import OmniUIAtlasService
from ..services.telemetry import telemetry, ensure_telemetry_initialized

logger = logging.getLogger(__name__)


def get_atlas_service() -> OmniUIAtlasService:
    """Get or create the global OmniUI Atlas service instance.

    Returns:
        The OmniUI Atlas service instance
    """
    from .list_ui_classes import get_atlas_service as _get_atlas_service

    return _get_atlas_service()


async def list_ui_modules() -> Dict[str, Any]:
    """Return a list of all OmniUI module names from the Atlas data.

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: A JSON string containing all OmniUI module names
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()
    
    # Record start time for telemetry
    start_time = time.perf_counter()
    
    # Prepare telemetry data (no parameters for this function)
    telemetry_data = {}
    
    success = True
    error_msg = None
    
    try:
        logger.info("Retrieving OmniUI modules from Atlas data")

        atlas_service = get_atlas_service()

        if not atlas_service.is_available():
            error_msg = "OmniUI Atlas data is not available"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        # Get the list of module names
        module_names = atlas_service.get_module_names_list()

        # Return the simplified result with just module names
        simplified_result = {
            "module_names": module_names,
            "total_count": len(module_names),
            "description": "OmniUI modules from Atlas data",
        }

        result_json = json.dumps(simplified_result, indent=2)

        logger.info(f"Successfully retrieved {len(module_names)} OmniUI modules from Atlas")
        return {"success": True, "result": result_json, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving OmniUI modules: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}
    
    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Capture telemetry data
        await telemetry.capture_call(
            function_name="list_ui_modules",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg
        )


================================================================================
FILE: src/omni_aiq_omni_ui/functions/get_ui_style_docs.py
Size: 8.91 KB | Tokens: 1,725
================================================================================

"""
Function to retrieve OmniUI style documentation.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..services.telemetry import telemetry, ensure_telemetry_initialized

logger = logging.getLogger(__name__)


async def get_ui_style_docs(sections) -> Dict[str, Any]:
    """Return OmniUI style documentation from the stored style files.

    Args:
        sections: List of section names to look up, or None to get combined documentation.
                 Can be a single section or multiple sections.

    Returns:
        Dictionary containing:
        - success: bool indicating if the operation succeeded
        - result: A JSON string containing the requested style documentation
        - error: Error message if operation failed
    """
    # Initialize telemetry service
    await ensure_telemetry_initialized()
    
    # Record start time for telemetry
    start_time = time.perf_counter()
    
    # Prepare telemetry data
    telemetry_data = {
        "sections": sections,
        "is_none": sections is None,
        "is_string": isinstance(sections, str),
        "is_list": isinstance(sections, list),
        "count": len(sections) if isinstance(sections, list) else (1 if isinstance(sections, str) else 0)
    }
    
    success = True
    error_msg = None
    
    try:
        # Get the data directory path
        data_dir = Path(__file__).parent.parent / "data" / "styles"

        if not data_dir.exists():
            error_msg = f"Style documentation directory not found: {data_dir}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "result": ""}

        # Available sections mapping (without .md extension)
        available_sections = {
            "buttons": "buttons.md",
            "containers": "containers.md",
            "fonts": "fonts.md",
            "line": "line.md",
            "overview": "overview.md",
            "shades": "shades.md",
            "shapes": "shapes.md",
            "sliders": "sliders.md",
            "styling": "styling.md",
            "units": "units.md",
            "widgets": "widgets.md",
            "window": "window.md",
        }

        result_data = {}

        # Handle None input (get combined documentation)
        if sections is None:
            # No specific section requested, return combined documentation
            logger.info("Retrieving complete combined style documentation")

            combined_file = data_dir / "all_styling_combined.md"
            if not combined_file.exists():
                error_msg = f"Combined style documentation not found: {combined_file}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg, "result": ""}

            with open(combined_file, "r", encoding="utf-8") as f:
                content = f.read()

            result_data = {
                "type": "combined",
                "content": content,
                "sections_included": list(available_sections.keys()),
                "total_sections": len(available_sections),
                "size": len(content),
                "description": "Complete OmniUI style documentation covering all styling aspects",
            }

        # Handle list input
        elif isinstance(sections, list):
            if len(sections) == 0:
                return {"success": False, "error": "sections array cannot be empty", "result": ""}

            # Check for empty strings in the list
            empty_names = [i for i, name in enumerate(sections) if not isinstance(name, str) or not name.strip()]
            if empty_names:
                return {
                    "success": False,
                    "error": f"sections contains empty or non-string values at indices: {empty_names}",
                    "result": "",
                }

            if len(sections) == 1:
                # Single section requested
                section = sections[0]
                logger.info(f"Retrieving style documentation for section: {section}")

                if section not in available_sections:
                    error_msg = f"Unknown section: {section}. Available sections: {list(available_sections.keys())}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg, "result": ""}

                file_path = data_dir / available_sections[section]
                if not file_path.exists():
                    error_msg = f"Section file not found: {file_path}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg, "result": ""}

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                result_data = {
                    "section": section,
                    "content": content,
                    "file": available_sections[section],
                    "size": len(content),
                }
            else:
                # Multiple sections requested
                logger.info(f"Retrieving style documentation for sections: {sections}")

                for section_name in sections:
                    if section_name not in available_sections:
                        logger.warning(f"Unknown section requested: {section_name}")
                        continue

                    file_path = data_dir / available_sections[section_name]
                    if file_path.exists():
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        result_data[section_name] = {
                            "content": content,
                            "file": available_sections[section_name],
                            "size": len(content),
                        }

                if not result_data:
                    error_msg = f"No valid sections found from: {sections}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg, "result": ""}

        # Handle legacy string input (should not happen with new API, but kept for safety)
        elif isinstance(sections, str):
            if not sections.strip():
                return {"success": False, "error": "section cannot be empty or whitespace", "result": ""}
            sections = [sections]

            # Process as single section
            section = sections[0]
            logger.info(f"Retrieving style documentation for section: {section}")

            if section not in available_sections:
                error_msg = f"Unknown section: {section}. Available sections: {list(available_sections.keys())}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg, "result": ""}

            file_path = data_dir / available_sections[section]
            if not file_path.exists():
                error_msg = f"Section file not found: {file_path}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg, "result": ""}

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            result_data = {
                "section": section,
                "content": content,
                "file": available_sections[section],
                "size": len(content),
            }
        else:
            actual_type = type(sections).__name__
            return {
                "success": False,
                "error": f"sections must be None or a list of strings, but got {actual_type}: {sections}",
                "result": "",
            }

        # Add metadata
        result_data["metadata"] = {
            "available_sections": list(available_sections.keys()),
            "source": "OmniKit UI Style Documentation v1.0.9",
            "description": "Comprehensive styling guidelines for OmniUI widgets and components",
        }

        result_json = json.dumps(result_data, indent=2)

        logger.info(f"Successfully retrieved style documentation")
        return {"success": True, "result": result_json, "error": None}

    except Exception as e:
        error_msg = f"Error retrieving style documentation: {str(e)}"
        logger.error(error_msg)
        success = False
        return {"success": False, "error": error_msg, "result": ""}
    
    finally:
        # Calculate duration and capture telemetry
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Capture telemetry data
        await telemetry.capture_call(
            function_name="get_ui_style_docs",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg
        )


================================================================================
FILE: src/omni_aiq_omni_ui/functions/search_ui_window_examples.py
Size: 4.21 KB | Tokens: 960
================================================================================

"""Function to get UI window examples using FAISS vector search."""

import logging
import os
import time
from typing import Any, Dict, Optional

from ..services.ui_window_examples_retrieval import create_ui_window_examples_retriever, get_ui_window_examples
from ..services.telemetry import telemetry, ensure_telemetry_initialized

logger = logging.getLogger(__name__)


async def search_ui_window_examples(
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
        "has_faiss_index_path": faiss_index_path is not None
    }
    
    success = True
    error_msg = None
    
    logger.info(f"[DEBUG] search_ui_window_examples called with request: {request}")

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
            function_name="search_ui_window_examples",
            request_data=telemetry_data,
            duration_ms=duration_ms,
            success=success,
            error=error_msg
        )


================================================================================
FILE: src/omni_aiq_omni_ui/models/__init__.py
Size: 38.00 B | Tokens: 8
================================================================================

"""Models module for OmniUI tools."""


================================================================================
FILE: src/omni_aiq_omni_ui/register_get_class_detail.py
Size: 8.55 KB | Tokens: 1,794
================================================================================

"""Registration wrapper for get_ui_class_detail function."""

import json
import logging
from typing import List, Optional, Union

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_ui_class_detail import get_ui_class_detail
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def _parse_class_names_input(class_names_str: str) -> Union[List[str], None]:
    """Parse class_names string input into a list of class names.

    Args:
        class_names_str: String input that can be:
            - None/empty: Return None (get all available classes)
            - Single class: "TreeView"
            - JSON array: '["Button", "Label", "TreeView"]'
            - Comma-separated: "Button, Label, TreeView"

    Returns:
        List of class names or None if input is empty

    Raises:
        ValueError: If input format is invalid
    """
    if not class_names_str or not class_names_str.strip():
        return None

    class_names_str = class_names_str.strip()

    # Try to parse as JSON array first
    if class_names_str.startswith("[") and class_names_str.endswith("]"):
        try:
            parsed = json.loads(class_names_str)
            if isinstance(parsed, list):
                # Validate all items are strings
                for item in parsed:
                    if not isinstance(item, str):
                        raise ValueError(f"All items in JSON array must be strings, got: {type(item).__name__}")
                return [item.strip() for item in parsed if item.strip()]
            else:
                raise ValueError("JSON input must be an array of strings")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array format: {e}")

    # Try comma-separated format
    if "," in class_names_str:
        return [name.strip() for name in class_names_str.split(",") if name.strip()]

    # Single class name
    return [class_names_str]


class GetClassDetailInput(BaseModel):
    """Input for get_ui_class_detail function.

    Provide class names in any convenient format - the system will handle the conversion automatically.
    """

    class_names: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Class names to look up. Accepts multiple flexible formats:
        
        📝 FLEXIBLE INPUT FORMATS (all work the same):
        - Single class: "TreeView" 
        - Native array: ["Button", "Label", "TreeView"] 
        - JSON string: '["Button", "Label", "TreeView"]'
        - Comma-separated: "Button, Label, TreeView"
        - Empty/null: Lists all available classes
        
        💡 TIP: Use whatever format is most natural - arrays, strings, or JSON!""",
    )

    model_config = {"extra": "forbid"}


# Tool description
GET_CLASS_DETAIL_DESCRIPTION = """Get detailed information about OmniUI classes - SUPER FLEXIBLE INPUT!

🚀 FLEXIBLE API: Accepts ANY input format - strings, arrays, JSON - whatever is natural!

PARAMETER:
- class_names: Class names in ANY convenient format:
  * Single class string: "TreeView" 
  * Native array: ["Button", "Label", "TreeView"] ← WORKS DIRECTLY!
  * JSON string: '["Button", "Label", "TreeView"]'
  * Comma-separated: "Button, Label, TreeView"
  * Empty/null: Lists all available classes

USAGE EXAMPLES (ALL FORMATS WORK):
✅ Direct array: get_ui_class_detail(class_names=["TreeView", "Window"])
✅ Single string: get_ui_class_detail(class_names="TreeView")
✅ JSON string: get_ui_class_detail(class_names='["Button", "Label", "TreeView"]')
✅ Comma format: get_ui_class_detail(class_names="Button, Label, TreeView") 
✅ List all: get_ui_class_detail() or get_ui_class_detail(class_names=null)

💡 FOR AI MODELS: You can pass arrays directly like ["TreeView", "Window"] - no need to convert to strings!

BATCH PROCESSING BENEFITS:
- 80% faster when fetching multiple classes
- Single API call instead of multiple round-trips
- Efficient context window usage
- Maximum compatibility with all AI models

RETURNS:
- For single class: Standard JSON with class details
- For multiple classes: Array with all class details plus metadata
- Includes: full_name, methods, parent_classes, docstring, etc.
- Error handling for invalid/missing classes"""


class GetClassDetailConfig(FunctionBaseConfig, name="get_ui_class_detail"):
    """Configuration for get_ui_class_detail function."""

    name: str = "get_ui_class_detail"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetClassDetailConfig, framework_wrappers=[])
async def register_get_class_detail(config: GetClassDetailConfig, builder: Builder):
    """Register get_ui_class_detail function with AIQ."""

    # Access config fields here
    if config.verbose:
        logger.info(f"Registering get_ui_class_detail in verbose mode")

    async def get_class_detail_wrapper(input: GetClassDetailInput) -> str:
        """Wrapper for get_ui_class_detail function."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()

        # Handle flexible input: string, array, or None
        try:
            if input.class_names is None:
                classes_to_fetch = None
            elif isinstance(input.class_names, list):
                # Direct array input - validate and use as-is
                if len(input.class_names) == 0:
                    classes_to_fetch = None  # Empty array = list all
                else:
                    # Validate all items are strings
                    for i, item in enumerate(input.class_names):
                        if not isinstance(item, str):
                            return f"ERROR: All items in class_names array must be strings, got {type(item).__name__} at index {i}"
                        if not item.strip():
                            return f"ERROR: Empty string at index {i} in class_names array"
                    classes_to_fetch = [item.strip() for item in input.class_names]
            elif isinstance(input.class_names, str):
                # String input - parse using existing logic
                classes_to_fetch = _parse_class_names_input(input.class_names)
            else:
                return f"ERROR: class_names must be a string, array, or null, got {type(input.class_names).__name__}"

            parameters = {"class_names": input.class_names}
        except ValueError as e:
            return f"ERROR: Invalid class_names parameter: {str(e)}"

        error_msg = None
        success = True

        try:
            # Call the async function directly
            result = await get_ui_class_detail(classes_to_fetch)

            # Use config fields to modify behavior
            if config.verbose:
                if isinstance(classes_to_fetch, list):
                    logger.debug(f"Retrieved details for {len(classes_to_fetch)} OmniUI classes")
                elif classes_to_fetch is None:
                    logger.debug("Retrieved available classes information")
                else:
                    logger.debug(f"Retrieved details for OmniUI class: {classes_to_fetch}")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve class detail - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_omni_ui_class_detail",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_ui_class_detail: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        get_class_detail_wrapper,
        description=GET_CLASS_DETAIL_DESCRIPTION,
        input_schema=GetClassDetailInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info


================================================================================
FILE: src/omni_aiq_omni_ui/register_get_class_detail_old.py
Size: 8.57 KB | Tokens: 1,796
================================================================================

"""Registration wrapper for get_ui_class_detail function."""

import json
import logging
import re
from typing import List, Union

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_ui_class_detail import get_ui_class_detail
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def _parse_class_names_input(class_names_str: str) -> Union[List[str], None]:
    """Parse class_names string input into a list of class names.

    Args:
        class_names_str: String input that can be:
            - None/empty: Return None (get all available classes)
            - Single class: "TreeView"
            - JSON array: '["Button", "Label", "TreeView"]'
            - Comma-separated: "Button, Label, TreeView"

    Returns:
        List of class names or None if input is empty

    Raises:
        ValueError: If input format is invalid
    """
    if not class_names_str or not class_names_str.strip():
        return None

    class_names_str = class_names_str.strip()

    # Try to parse as JSON array first
    if class_names_str.startswith("[") and class_names_str.endswith("]"):
        try:
            parsed = json.loads(class_names_str)
            if isinstance(parsed, list):
                # Validate all items are strings
                for item in parsed:
                    if not isinstance(item, str):
                        raise ValueError(f"All items in JSON array must be strings, got: {type(item).__name__}")
                return [item.strip() for item in parsed if item.strip()]
            else:
                raise ValueError("JSON input must be an array of strings")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array format: {e}")

    # Try comma-separated format
    if "," in class_names_str:
        return [name.strip() for name in class_names_str.split(",") if name.strip()]

    # Single class name
    return [class_names_str]


# Define input schema
from typing import List, Optional


class GetClassDetailInput(BaseModel):
    """Input for get_ui_class_detail function.

    Provide class names as a string. Can be a single class or multiple classes formatted as a list.
    If not provided, returns information about available classes.
    """

    class_names: Optional[str] = Field(
        None,
        description="""Class names to look up. Can be formatted as:
        - Single class: "TreeView" or "Button"  
        - Multiple classes (JSON array): '["Button", "Label", "TreeView"]'
        - Multiple classes (comma-separated): "Button, Label, TreeView"
        - Leave empty/null to get information about available classes""",
    )

    model_config = {"extra": "forbid"}


# Tool description
GET_CLASS_DETAIL_DESCRIPTION = """Get detailed information about OmniUI classes - SIMPLIFIED API with string-based input!

🚀 STREAMLINED API: Now uses a single parameter for maximum compatibility with all AI models!

PARAMETER:
- class_names (string): Class names to look up, supports multiple formats:
  * Single class: "TreeView"
  * JSON array: '["Button", "Label", "TreeView"]' 
  * Comma-separated: "Button, Label, TreeView"
  * Leave empty for available classes info

USAGE EXAMPLES:
✅ Single class: get_ui_class_detail(class_names="TreeView")
✅ Multiple classes: get_ui_class_detail(class_names='["Button", "Label", "TreeView"]')
✅ Comma format: get_ui_class_detail(class_names="Button, Label, TreeView") 
✅ List all: get_ui_class_detail() or get_ui_class_detail(class_names=null)

BATCH PROCESSING BENEFITS:
- 80% faster when fetching multiple classes
- Single API call instead of multiple round-trips
- Efficient context window usage
- Better model compatibility

RETURNS:
- For single class: Standard JSON with class details
- For multiple classes: Array with all class details plus metadata
- Includes: full_name, methods, parent_classes, docstring, etc.
- Error handling for invalid/missing classes

EXAMPLES:
# Getting multiple related classes (RECOMMENDED)
get_ui_class_detail class_names=["HStack", "VStack", "ZStack", "Frame", "ScrollingFrame"]

# Getting all input widgets at once
get_ui_class_detail class_names=["FloatSlider", "IntSlider", "FloatDrag", "IntDrag", "MultiFloatField"]

# Single class (when you only need one)
get_ui_class_detail class_name="TreeView"

remember that the class_names is an array not a string.
get_ui_class_detail (MCP)(class_name: "TreeView", class_names: null) # not "null"

PERFORMANCE TIP: Always batch related class queries together!"""


class GetClassDetailConfig(FunctionBaseConfig, name="get_ui_class_detail"):
    """Configuration for get_ui_class_detail function."""

    name: str = "get_ui_class_detail"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetClassDetailConfig, framework_wrappers=[])
async def register_get_class_detail(config: GetClassDetailConfig, builder: Builder):
    """Register get_ui_class_detail function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering get_ui_class_detail in verbose mode")

    async def get_class_detail_wrapper(input: GetClassDetailInput) -> str:
        """Wrapper for get_ui_class_detail function."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()

        # Parse the class_names parameter (string input)
        try:
            classes_to_fetch = _parse_class_names_input(input.class_names)
            parameters = {"class_names": input.class_names}
        except ValueError as e:
            return f"ERROR: Invalid class_names parameter: {str(e)}"
            error_details = []
            if input.class_name is not None:
                if not input.class_name.strip():
                    error_details.append("class_name is empty or whitespace")
                else:
                    error_details.append(f"class_name='{input.class_name}'")
            if input.class_names is not None:
                if len(input.class_names) == 0:
                    error_details.append("class_names is an empty array")
                else:
                    error_details.append(f"class_names={input.class_names}")

            if error_details:
                return f"ERROR: Invalid input parameters - {', '.join(error_details)}. Please provide either a non-empty class_name (string) OR a non-empty class_names (array of strings)."
            else:
                return "ERROR: Either class_name (string) or class_names (array of strings) must be provided. Both parameters are None/null."

        error_msg = None
        success = True

        try:
            # Call the async function directly
            result = await get_ui_class_detail(classes_to_fetch)

            # Use config fields to modify behavior
            if verbose:
                if isinstance(classes_to_fetch, list):
                    logger.debug(f"Retrieved details for {len(classes_to_fetch)} OmniUI classes")
                else:
                    logger.debug(f"Retrieved details for OmniUI class: {classes_to_fetch}")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve class detail - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_omni_ui_class_detail",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_ui_class_detail: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        get_class_detail_wrapper,
        description=GET_CLASS_DETAIL_DESCRIPTION,
        input_schema=GetClassDetailInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info


================================================================================
FILE: src/omni_aiq_omni_ui/register_get_class_instructions.py
Size: 12.22 KB | Tokens: 2,502
================================================================================

"""Registration wrapper for get_ui_class_instructions function."""

import json
import logging
from typing import List, Optional, Union

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_ui_class_instructions import get_ui_class_instructions, list_class_categories, list_classes_in_category
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def _parse_class_names_input(class_names_str: str) -> Union[List[str], None, str]:
    """Parse class_names string input into appropriate format for get_ui_class_instructions.

    Args:
        class_names_str: String input that can be:
            - None/empty: Return None (list all categories)
            - Special commands: "categories", "category:widgets", "category:scene"
            - Single class: "Button" or "TreeView"
            - JSON array: '["Button", "Label", "TreeView"]'
            - Comma-separated: "Button, Label, TreeView"

    Returns:
        - None: For empty input (list categories)
        - String: For special commands (categories, category:xxx)
        - List[str]: For class names

    Raises:
        ValueError: If input format is invalid
    """
    if not class_names_str or not class_names_str.strip():
        return None

    class_names_str = class_names_str.strip()

    # Handle special commands
    if class_names_str.lower() == "categories":
        return "categories"
    elif class_names_str.lower().startswith("category:"):
        return class_names_str.lower()  # Return as-is for category parsing

    # Try to parse as JSON array first
    if class_names_str.startswith("[") and class_names_str.endswith("]"):
        try:
            parsed = json.loads(class_names_str)
            if isinstance(parsed, list):
                # Validate all items are strings
                for item in parsed:
                    if not isinstance(item, str):
                        raise ValueError(f"All items in JSON array must be strings, got: {type(item).__name__}")
                return [item.strip() for item in parsed if item.strip()]
            else:
                raise ValueError("JSON input must be an array of strings")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array format: {e}")

    # Try comma-separated format
    if "," in class_names_str:
        return [name.strip() for name in class_names_str.split(",") if name.strip()]

    # Single class name
    return [class_names_str]


class GetClassInstructionsInput(BaseModel):
    """Input for get_ui_class_instructions function.

    Provide class names in any convenient format - the system will handle the conversion automatically.
    """

    class_names: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Class names to look up. Accepts multiple flexible formats:
        
        📝 FLEXIBLE INPUT FORMATS (all work the same):
        - Single class: "Button" or "TreeView"
        - Native array: ["Button", "Label", "TreeView"] ← WORKS DIRECTLY!
        - JSON string: '["Button", "Label", "TreeView"]'
        - Comma-separated: "Button, Label, TreeView"
        - Scene classes: "scene.Line", "scene.Rectangle", "omni.ui.scene.Line"
        - Empty/null: Lists all categories
        
        🎯 SPECIAL COMMANDS (string only):
        - "categories": List all available categories
        - "category:widgets": List all classes in widgets category
        - "category:scene": List all 3D scene UI classes
        
        💡 TIP: Use whatever format is most natural - arrays, strings, or JSON!""",
    )

    model_config = {"extra": "forbid"}


# Tool description
GET_CLASS_INSTRUCTIONS_DESCRIPTION = """Retrieve OmniUI class instructions - SUPER FLEXIBLE INPUT!

🚀 FLEXIBLE API: Accepts ANY input format - strings, arrays, JSON - whatever is natural!

PARAMETER:
- class_names: Class names in ANY convenient format:
  * Single class string: "Button" or "TreeView"
  * Native array: ["Button", "Label", "TreeView"] ← WORKS DIRECTLY!
  * JSON string: '["Button", "Label", "TreeView"]'
  * Comma-separated: "Button, Label, TreeView"
  * Scene classes: "scene.Line", "scene.Rectangle", "omni.ui.scene.Line"
  * Empty/null: Lists all categories

🎯 SPECIAL COMMANDS (string only):
- "categories": List all available categories
- "category:widgets": List all classes in widgets category
- "category:scene": List all 3D scene UI classes

USAGE EXAMPLES (ALL FORMATS WORK):
✅ Direct array: get_ui_class_instructions(class_names=["Button", "TreeView"])
✅ Single string: get_ui_class_instructions(class_names="Button")
✅ JSON string: get_ui_class_instructions(class_names='["Button", "Label", "TreeView"]')
✅ Comma format: get_ui_class_instructions(class_names="Button, Label, TreeView") 
✅ List categories: get_ui_class_instructions() or get_ui_class_instructions(class_names="categories")
✅ Category listing: get_ui_class_instructions(class_names="category:widgets")

💡 FOR AI MODELS: You can pass arrays directly like ["Button", "TreeView"] - no need to convert to strings!

BATCH PROCESSING BENEFITS:
- 75% faster when fetching multiple classes
- Single API call instead of multiple round-trips
- Efficient context window usage
- Maximum compatibility with all AI models

CLASS CATEGORIES AVAILABLE:
- **models** (3): AbstractValueModel, AbstractItemModel, AbstractItemDelegate
- **shapes** (12): Rectangle, Circle, Triangle, Line, etc. + Free variants
- **widgets** (11): Button, Label, TreeView, CheckBox, ComboBox, etc.
- **containers** (7): Frame, ScrollingFrame, HStack, VStack, ZStack, etc.
- **layouts** (3): VGrid, HGrid, Placer
- **inputs** (7): FloatSlider, IntSlider, FloatDrag, IntDrag, etc.
- **windows** (6): Window, MainWindow, Menu, MenuBar, Tooltip, Separator
- **scene** (9): All omni.ui.scene 3D UI components
- **units** (3): Pixel, Percent, Fraction  
- **system** (1): Style

RETURNS:
- For single class: Formatted class documentation with examples
- For multiple classes: Combined documentation with headers
- For categories/listings: Structured category and class information
- Includes: Class usage, styling, properties, and code examples"""


class GetClassInstructionsConfig(FunctionBaseConfig, name="get_ui_class_instructions"):
    """Configuration for get_ui_class_instructions function."""

    name: str = "get_ui_class_instructions"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetClassInstructionsConfig, framework_wrappers=[])
async def register_get_class_instructions(config: GetClassInstructionsConfig, builder: Builder):
    """Register get_ui_class_instructions function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering get_ui_class_instructions in verbose mode")

    async def get_class_instructions_wrapper(input: GetClassInstructionsInput) -> str:
        """Get OmniUI class instructions with support for categories and listings."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()

        # Handle flexible input: string, array, or None
        try:
            if input.class_names is None:
                # Empty input - list all categories
                result = await list_class_categories()
            elif isinstance(input.class_names, list):
                # Direct array input - validate and use as-is
                if len(input.class_names) == 0:
                    # Empty array = list all categories
                    result = await list_class_categories()
                else:
                    # Validate all items are strings
                    for i, item in enumerate(input.class_names):
                        if not isinstance(item, str):
                            return f"ERROR: All items in class_names array must be strings, got {type(item).__name__} at index {i}"
                        if not item.strip():
                            return f"ERROR: Empty string at index {i} in class_names array"
                    # Process as list of class names
                    classes_to_fetch = [item.strip() for item in input.class_names]
                    result = await get_ui_class_instructions(classes_to_fetch)
            elif isinstance(input.class_names, str):
                # String input - handle special commands and regular parsing
                parsed_input = _parse_class_names_input(input.class_names)
                if isinstance(parsed_input, str):
                    # Special commands
                    if parsed_input == "categories":
                        result = await list_class_categories()
                    elif parsed_input.startswith("category:"):
                        # Category specified - extract category name
                        category_name = parsed_input[9:]  # Remove "category:" prefix
                        if not category_name or category_name.lower() == "none":
                            # If category name is empty or 'none', list all categories
                            result = await list_class_categories()
                        else:
                            result = await list_classes_in_category(category_name)
                    else:
                        # Single class name passed as string (fallback)
                        result = await get_ui_class_instructions([parsed_input])
                elif isinstance(parsed_input, list):
                    # List of class names from string parsing
                    result = await get_ui_class_instructions(parsed_input)
                else:
                    # None from empty string
                    result = await list_class_categories()
            else:
                return f"ERROR: class_names must be a string, array, or null, got {type(input.class_names).__name__}"

            parameters = {"class_names": input.class_names}
        except ValueError as e:
            return f"ERROR: Invalid class_names parameter: {str(e)}"

        error_msg = None
        success = True

        try:

            # Use config fields to modify behavior
            if verbose:
                if result.get("metadata"):
                    metadata = result["metadata"]
                    if isinstance(metadata, dict) and "classes" in metadata:
                        # Multiple classes
                        logger.debug(f"Retrieved {metadata.get('successful', 0)} class instructions")
                    else:
                        # Single class
                        logger.debug(
                            f"Retrieved: {metadata.get('class_name', 'N/A')} from {metadata.get('category', 'N/A')} category"
                        )
                else:
                    logger.debug(f"Listed categories or classes")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve class instructions - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_ui_class_instructions",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_ui_class_instructions: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        get_class_instructions_wrapper,
        description=GET_CLASS_INSTRUCTIONS_DESCRIPTION,
        input_schema=GetClassInstructionsInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info


================================================================================
FILE: src/omni_aiq_omni_ui/register_get_classes.py
Size: 3.88 KB | Tokens: 769
================================================================================

"""Registration wrapper for list_ui_classes function."""

import logging

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.list_ui_classes import list_ui_classes
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


# Define input schema for zero-argument function
class GetClassesInput(BaseModel):
    """Empty input for zero-argument function."""

    pass


# Tool description
GET_CLASSES_DESCRIPTION = """Return a list of all OmniUI class full names from the Atlas data.

WHAT IT DOES:
- Retrieves all OmniUI class names from the comprehensive Atlas database
- Returns class names for all OmniUI widgets and components
- Provides total count of available classes
- Sorts class names alphabetically for easy browsing

RETURNS:
A JSON string containing:
- class_full_names: Sorted list of all OmniUI class full names (e.g., FilterButton, OptionsMenu, etc.)
- total_count: Total number of classes
- description: Brief description of the classes

USAGE EXAMPLES:
list_ui_classes

This provides access to the complete OmniUI class hierarchy from Atlas data, including:
- Widget classes (buttons, menus, filters, etc.)
- Layout components
- Style and customization classes
- Testing utilities
"""


class GetClassesConfig(FunctionBaseConfig, name="list_ui_classes"):
    """Configuration for list_ui_classes function."""

    name: str = "list_ui_classes"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetClassesConfig, framework_wrappers=[])
async def register_get_classes(config: GetClassesConfig, builder: Builder):
    """Register list_ui_classes function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering list_ui_classes in verbose mode")

    async def get_classes_wrapper(input: GetClassesInput) -> str:
        """Zero arguments - schema required."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()
        parameters = {}
        error_msg = None
        success = True

        try:
            # Call the async function directly
            result = await list_ui_classes()

            # Use config fields to modify behavior
            if verbose:
                logger.debug(f"Retrieved OmniUI classes")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve OmniUI classes - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="list_ui_classes",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for list_ui_classes: {log_error}")

    # Pass input_schema for zero argument function
    function_info = FunctionInfo.from_fn(
        get_classes_wrapper,
        description=GET_CLASSES_DESCRIPTION,
        input_schema=GetClassesInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info


================================================================================
FILE: src/omni_aiq_omni_ui/register_search_ui_code_examples.py
Size: 7.09 KB | Tokens: 1,480
================================================================================

"""Registration wrapper for search_ui_code_examples function."""

import logging
import os
from typing import Optional

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .config import DEFAULT_RERANK_CODE
from .functions.search_ui_code_examples import search_ui_code_examples

logger = logging.getLogger(__name__)


# Define input schema for single argument function
class GetCodeExamplesInput(BaseModel):
    """Input for search_ui_code_examples function."""

    request: str = Field(description="Your query describing the desired code example")


# Tool description
GET_CODE_EXAMPLES_DESCRIPTION = """Retrieves relevant code examples using semantic vector search and optional reranking.

WHAT IT DOES:
- Converts your query to embeddings using NVIDIA's nv-embedqa-e5-v5 model
- Performs semantic similarity search against pre-indexed OmniUI code examples
- Optionally reranks results using NVIDIA's llama-3.2-nv-rerankqa-1b-v2 model
- Returns formatted code examples with their metadata and source code

QUERY MATCHING:
Your query is compared against OmniUI widget and component implementations,
including examples like:
- SearchField, SearchWordButton, and other UI widgets
- Widget styling and theming functions
- UI component building patterns (ZStack, VStack, HStack)
- Event handling and callback patterns
- Layout and spacing utilities

ARGUMENTS:
- request (str): Your query describing the desired OmniUI code example

RETURNS:
Formatted code examples with file paths, method names, and Python code snippets

USAGE EXAMPLES:
search_ui_code_examples "How to create a search field?"
search_ui_code_examples "Button styling with themes"
search_ui_code_examples "event handling callbacks"
search_ui_code_examples "VStack and HStack layout"
search_ui_code_examples "create custom widget"

TIPS FOR BETTER RESULTS:
- Use specific OmniUI terminology (e.g., "SearchField", "ZStack", "VStack")
- Include UI operations (e.g., "build_ui", "style", "event handling")
- Reference widget types (e.g., "Button", "Label", "Rectangle", "Spacer")
- Ask about patterns (e.g., "callback", "subscription", "model binding")
"""


class GetCodeExamplesConfig(FunctionBaseConfig, name="search_ui_code_examples"):
    """Configuration for search_ui_code_examples function."""

    name: str = "search_ui_code_examples"
    verbose: bool = Field(default=False, description="Enable detailed logging")
    rerank_k: int = Field(default=DEFAULT_RERANK_CODE, description="Number of documents to keep after reranking")
    enable_rerank: bool = Field(default=True, description="Enable reranking of search results")

    # Embedding configuration
    embedding_model: Optional[str] = Field(default="nvidia/nv-embedqa-e5-v5", description="Embedding model to use")
    embedding_endpoint: Optional[str] = Field(
        default=None, description="Embedding service endpoint (None for NVIDIA API)"
    )
    embedding_api_key: Optional[str] = Field(default="${NVIDIA_API_KEY}", description="API key for embedding service")

    # Reranking configuration
    reranking_model: Optional[str] = Field(
        default="nvidia/llama-3.2-nv-rerankqa-1b-v2", description="Reranking model to use"
    )
    reranking_endpoint: Optional[str] = Field(
        default=None, description="Reranking service endpoint (None for NVIDIA API)"
    )
    reranking_api_key: Optional[str] = Field(default="${NVIDIA_API_KEY}", description="API key for reranking service")


@register_function(config_type=GetCodeExamplesConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def register_search_ui_code_examples(config: GetCodeExamplesConfig, builder: Builder):
    """Register search_ui_code_examples function with AIQ."""

    # Access config fields here
    if config.verbose:
        logger.info(f"Registering search_ui_code_examples in verbose mode")

    async def search_ui_code_examples_wrapper(input: GetCodeExamplesInput) -> str:
        """Single argument with schema."""
        import time

        from omni_aiq_omni_ui.utils.usage_logging import get_usage_logger

        # Extract the request string from the input model
        request = input.request

        # Debug logging
        logger.info(f"[DEBUG] search_ui_code_examples_wrapper called with input type: {type(input)}")
        logger.info(f"[DEBUG] search_ui_code_examples_wrapper request value: {request}")

        usage_logger = get_usage_logger()
        start_time = time.time()
        parameters = {"request": request}
        error_msg = None
        success = True

        try:
            # Handle environment variable substitution for API keys
            embedding_api_key = config.embedding_api_key
            if embedding_api_key == "${NVIDIA_API_KEY}":
                embedding_api_key = os.getenv("NVIDIA_API_KEY")

            reranking_api_key = config.reranking_api_key
            if reranking_api_key == "${NVIDIA_API_KEY}":
                reranking_api_key = os.getenv("NVIDIA_API_KEY")

            result = await search_ui_code_examples(
                request,
                rerank_k=config.rerank_k,
                enable_rerank=config.enable_rerank,
                embedding_config={
                    "model": config.embedding_model,
                    "endpoint": config.embedding_endpoint,
                    "api_key": embedding_api_key,
                },
                reranking_config={
                    "model": config.reranking_model,
                    "endpoint": config.reranking_endpoint,
                    "api_key": reranking_api_key,
                },
            )

            # Use config fields to modify behavior
            if config.verbose:
                logger.debug(
                    f"Retrieved OmniUI code examples for: {request}, rerank_k: {config.rerank_k}, enable_rerank: {config.enable_rerank}"
                )

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve OmniUI code examples - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="search_ui_code_examples",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for search_ui_code_examples: {log_error}")

    # Pass input_schema for proper MCP parameter handling
    yield FunctionInfo.from_fn(
        search_ui_code_examples_wrapper,
        description=GET_CODE_EXAMPLES_DESCRIPTION,
        input_schema=GetCodeExamplesInput,
    )


================================================================================
FILE: src/omni_aiq_omni_ui/register_get_instructions.py
Size: 6.85 KB | Tokens: 1,376
================================================================================

"""Registration wrapper for get_instructions function."""

import logging
from typing import Optional

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_instructions import get_instructions, list_instructions
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


# Define input schema
class GetInstructionsInput(BaseModel):
    """Input schema for get_instructions function."""

    name: Optional[str] = Field(
        default=None,
        description="""The name of the instruction set to retrieve. Valid values are:
- 'agent_system': Core Omniverse UI Assistant system prompt with omni.ui framework basics. Use for understanding omni.ui fundamentals, omni.ui.scene for 3D UI, widget filters, options menus, searchable comboboxes, and general code writing guidelines.
- 'classes': Comprehensive omni.ui class API reference and model patterns. Use for working with AbstractValueModel, data models, custom model implementations, callbacks, and model-view patterns.
- 'omni_ui_scene_system': Complete omni.ui.scene 3D UI system documentation. Use for creating 3D shapes, SceneView, camera controls, transforms, gestures, manipulators, and USD stage synchronization.
- 'omni_ui_system': Core omni.ui widgets, containers, layouts and styling. Use for basic UI shapes, widgets (Labels, Buttons, Fields, Sliders), layouts (HStack, VStack, ZStack, Grid), Window management, styling, drag & drop, and MDV pattern.

If not provided or None, lists all available instructions with their descriptions.""",
    )


# Tool description
GET_INSTRUCTIONS_DESCRIPTION = """Retrieve OmniUI system instructions and documentation for code generation and UI development.

WHAT IT DOES:
- Retrieves specific OmniUI system instruction documents
- Provides comprehensive documentation for different aspects of omni.ui
- Lists all available instructions when no name is specified
- Returns formatted content with metadata and use cases

INSTRUCTION SETS:
1. **agent_system**: Core system prompt and omni.ui framework basics
   - Understanding omni.ui and omni.ui.scene fundamentals
   - Widget filters, options menus, searchable comboboxes
   - General code writing guidelines and placeholder patterns

2. **classes**: Comprehensive class API reference and model patterns
   - AbstractValueModel and data model implementations
   - SimpleStringModel, SimpleBoolModel, SimpleFloatModel, SimpleIntModel
   - Custom models, callbacks, and model-view patterns

3. **omni_ui_scene_system**: Complete 3D UI system documentation
   - 3D shapes (Line, Curve, Rectangle, Arc, etc.)
   - SceneView, camera controls, Transform containers
   - Gestures, mouse interactions, manipulators
   - USD camera and stage synchronization

4. **omni_ui_system**: Core widgets, containers, layouts and styling
   - Basic UI shapes and widgets (Labels, Buttons, Fields, Sliders)
   - Layout systems (HStack, VStack, ZStack, Grid)
   - Window management, styling with selectors
   - Drag & drop, MDV pattern, callbacks

RETURNS:
When name is provided:
- Formatted instruction content with metadata header
- Description and use cases for the instruction set
- Full documentation content

When name is not provided:
- List of all available instructions
- Descriptions and use cases for each

USAGE EXAMPLES:
get_instructions(name="agent_system")  # Get core system prompt
get_instructions(name="omni_ui_system")  # Get widgets and layouts documentation
get_instructions()  # List all available instructions

WHEN TO USE:
- Load agent_system when starting omni.ui development for fundamental concepts
- Load classes when working with data models and custom implementations
- Load omni_ui_scene_system for 3D UI and manipulator development
- Load omni_ui_system for standard UI widgets and layouts
- Call without parameters to see all available instructions"""


class GetInstructionsConfig(FunctionBaseConfig, name="get_instructions"):
    """Configuration for get_instructions function."""

    name: str = "get_instructions"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetInstructionsConfig, framework_wrappers=[])
async def register_get_instructions(config: GetInstructionsConfig, builder: Builder):
    """Register get_instructions function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering get_instructions in verbose mode")

    async def get_instructions_wrapper(input: GetInstructionsInput) -> str:
        """Get OmniUI system instructions."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()
        parameters = {"name": input.name} if input.name else {}
        error_msg = None
        success = True

        try:
            # If no name provided, list all instructions
            if input.name is None:
                result = await list_instructions()
            else:
                # Get specific instruction
                result = await get_instructions(input.name)

            # Use config fields to modify behavior
            if verbose:
                if input.name:
                    logger.debug(f"Retrieved instruction: {input.name}")
                else:
                    logger.debug("Listed all available instructions")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve instructions - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_instructions",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_instructions: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        get_instructions_wrapper,
        description=GET_INSTRUCTIONS_DESCRIPTION,
        input_schema=GetInstructionsInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info


================================================================================
FILE: src/omni_aiq_omni_ui/register_get_method_detail.py
Size: 8.93 KB | Tokens: 1,889
================================================================================

"""Registration wrapper for get_ui_method_detail function."""

import json
import logging
from typing import List, Optional, Union

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_ui_method_detail import get_ui_method_detail
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def _parse_method_names_input(method_names_str: str) -> Union[List[str], None]:
    """Parse method_names string input into a list of method names.

    Args:
        method_names_str: String input that can be:
            - None/empty: Return None (get all available methods)
            - Single method: "__init__" or "clicked_fn"
            - JSON array: '["__init__", "clicked_fn", "set_value", "get_value"]'
            - Comma-separated: "__init__, clicked_fn, set_value, get_value"

    Returns:
        List of method names or None if input is empty

    Raises:
        ValueError: If input format is invalid
    """
    if not method_names_str or not method_names_str.strip():
        return None

    method_names_str = method_names_str.strip()

    # Try to parse as JSON array first
    if method_names_str.startswith("[") and method_names_str.endswith("]"):
        try:
            parsed = json.loads(method_names_str)
            if isinstance(parsed, list):
                # Validate all items are strings
                for item in parsed:
                    if not isinstance(item, str):
                        raise ValueError(f"All items in JSON array must be strings, got: {type(item).__name__}")
                return [item.strip() for item in parsed if item.strip()]
            else:
                raise ValueError("JSON input must be an array of strings")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array format: {e}")

    # Try comma-separated format
    if "," in method_names_str:
        return [name.strip() for name in method_names_str.split(",") if name.strip()]

    # Single method name
    return [method_names_str]


class GetMethodDetailInput(BaseModel):
    """Input for get_ui_method_detail function.

    Provide method names in any convenient format - the system will handle the conversion automatically.
    """

    method_names: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Method names to look up. Accepts multiple flexible formats:
        
        📝 FLEXIBLE INPUT FORMATS (all work the same):
        - Single method: "__init__"
        - Native array: ["__init__", "clicked_fn", "set_value", "get_value"] ← WORKS DIRECTLY!
        - JSON string: '["__init__", "clicked_fn", "set_value", "get_value"]'
        - Comma-separated: "__init__, clicked_fn, set_value, get_value"
        - Empty/null: Lists all available methods
        
        💡 TIP: Use whatever format is most natural - arrays, strings, or JSON!""",
    )

    model_config = {"extra": "forbid"}


# Tool description
GET_METHOD_DETAIL_DESCRIPTION = """Get detailed information about OmniUI methods - SUPER FLEXIBLE INPUT!

🚀 FLEXIBLE API: Accepts ANY input format - strings, arrays, JSON - whatever is natural!

PARAMETER:
- method_names: Method names in ANY convenient format:
  * Single method string: "__init__"
  * Native array: ["__init__", "clicked_fn", "set_value", "get_value"] ← WORKS DIRECTLY!
  * JSON string: '["__init__", "clicked_fn", "set_value", "get_value"]'
  * Comma-separated: "__init__, clicked_fn, set_value, get_value"
  * Empty/null: Lists all available methods

USAGE EXAMPLES (ALL FORMATS WORK):
✅ Direct array: get_ui_method_detail(method_names=["__init__", "clicked_fn"])
✅ Single string: get_ui_method_detail(method_names="__init__")
✅ JSON string: get_ui_method_detail(method_names='["__init__", "clicked_fn", "set_value", "get_value"]')
✅ Comma format: get_ui_method_detail(method_names="__init__, clicked_fn, set_value, get_value") 
✅ List all: get_ui_method_detail() or get_ui_method_detail(method_names=null)

💡 FOR AI MODELS: You can pass arrays directly like ["__init__", "clicked_fn"] - no need to convert to strings!

BATCH PROCESSING BENEFITS:
- 60-80% faster when fetching multiple methods
- Single API call instead of multiple round-trips
- Efficient context window usage
- Maximum compatibility with all AI models

RETURNS:
- For single method: Standard JSON with method details
- For multiple methods: Array with all method details plus metadata
- Includes: signatures, parameters, return types, docstrings for each method
- Error handling for invalid/missing methods"""


class GetMethodDetailConfig(FunctionBaseConfig, name="get_ui_method_detail"):
    """Configuration for get_ui_method_detail function."""

    name: str = "get_ui_method_detail"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetMethodDetailConfig, framework_wrappers=[])
async def register_get_method_detail(config: GetMethodDetailConfig, builder: Builder):
    """Register get_ui_method_detail function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering get_ui_method_detail in verbose mode")

    async def get_method_detail_wrapper(input: GetMethodDetailInput) -> str:
        """Wrapper for get_ui_method_detail function."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()

        # Handle flexible input: string, array, or None
        try:
            if input.method_names is None:
                methods_to_fetch = None
            elif isinstance(input.method_names, list):
                # Direct array input - validate and use as-is
                if len(input.method_names) == 0:
                    methods_to_fetch = None  # Empty array = list all
                else:
                    # Validate all items are strings
                    for i, item in enumerate(input.method_names):
                        if not isinstance(item, str):
                            return f"ERROR: All items in method_names array must be strings, got {type(item).__name__} at index {i}"
                        if not item.strip():
                            return f"ERROR: Empty string at index {i} in method_names array"
                    methods_to_fetch = [item.strip() for item in input.method_names]
            elif isinstance(input.method_names, str):
                # String input - parse using existing logic
                methods_to_fetch = _parse_method_names_input(input.method_names)
            else:
                return f"ERROR: method_names must be a string, array, or null, got {type(input.method_names).__name__}"

            parameters = {"method_names": input.method_names}
        except ValueError as e:
            return f"ERROR: Invalid method_names parameter: {str(e)}"

        error_msg = None
        success = True

        try:
            # Call the async function directly
            result = await get_ui_method_detail(methods_to_fetch)

            # Use config fields to modify behavior
            if verbose:
                if isinstance(methods_to_fetch, list):
                    logger.debug(f"Retrieved details for {len(methods_to_fetch)} OmniUI methods")
                elif methods_to_fetch is None:
                    logger.debug("Retrieved available methods information")
                else:
                    logger.debug(f"Retrieved details for OmniUI method: {methods_to_fetch}")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve method detail - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_omni_ui_method_detail",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_ui_method_detail: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        get_method_detail_wrapper,
        description=GET_METHOD_DETAIL_DESCRIPTION,
        input_schema=GetMethodDetailInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info


================================================================================
FILE: src/omni_aiq_omni_ui/register_get_module_detail.py
Size: 8.88 KB | Tokens: 1,903
================================================================================

"""Registration wrapper for get_ui_module_detail function."""

import json
import logging
from typing import List, Optional, Union

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_ui_module_detail import get_ui_module_detail
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def _parse_module_names_input(module_names_str: str) -> Union[List[str], None]:
    """Parse module_names string input into a list of module names.

    Args:
        module_names_str: String input that can be:
            - None/empty: Return None (get all available modules)
            - Single module: "omni.ui"
            - JSON array: '["omni.ui", "omni.ui.scene", "omni.ui.workspace"]'
            - Comma-separated: "omni.ui, omni.ui.scene, omni.ui.workspace"

    Returns:
        List of module names or None if input is empty

    Raises:
        ValueError: If input format is invalid
    """
    if not module_names_str or not module_names_str.strip():
        return None

    module_names_str = module_names_str.strip()

    # Try to parse as JSON array first
    if module_names_str.startswith("[") and module_names_str.endswith("]"):
        try:
            parsed = json.loads(module_names_str)
            if isinstance(parsed, list):
                # Validate all items are strings
                for item in parsed:
                    if not isinstance(item, str):
                        raise ValueError(f"All items in JSON array must be strings, got: {type(item).__name__}")
                return [item.strip() for item in parsed if item.strip()]
            else:
                raise ValueError("JSON input must be an array of strings")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array format: {e}")

    # Try comma-separated format
    if "," in module_names_str:
        return [name.strip() for name in module_names_str.split(",") if name.strip()]

    # Single module name
    return [module_names_str]


class GetModuleDetailInput(BaseModel):
    """Input for get_ui_module_detail function.

    Provide module names in any convenient format - the system will handle the conversion automatically.
    """

    module_names: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Module names to look up. Accepts multiple flexible formats:
        
        📝 FLEXIBLE INPUT FORMATS (all work the same):
        - Single module: "omni.ui"
        - Native array: ["omni.ui", "omni.ui.scene", "omni.ui.workspace"] ← WORKS DIRECTLY!
        - JSON string: '["omni.ui", "omni.ui.scene", "omni.ui.workspace"]'
        - Comma-separated: "omni.ui, omni.ui.scene, omni.ui.workspace"
        - Empty/null: Lists all available modules
        
        💡 TIP: Use whatever format is most natural - arrays, strings, or JSON!""",
    )

    model_config = {"extra": "forbid"}


# Tool description
GET_MODULE_DETAIL_DESCRIPTION = """Get detailed information about OmniUI modules - SUPER FLEXIBLE INPUT!

🚀 FLEXIBLE API: Accepts ANY input format - strings, arrays, JSON - whatever is natural!

PARAMETER:
- module_names: Module names in ANY convenient format:
  * Single module string: "omni.ui"
  * Native array: ["omni.ui", "omni.ui.scene", "omni.ui.workspace"] ← WORKS DIRECTLY!
  * JSON string: '["omni.ui", "omni.ui.scene", "omni.ui.workspace"]'
  * Comma-separated: "omni.ui, omni.ui.scene, omni.ui.workspace"
  * Empty/null: Lists all available modules

USAGE EXAMPLES (ALL FORMATS WORK):
✅ Direct array: get_ui_module_detail(module_names=["omni.ui", "omni.ui.scene"])
✅ Single string: get_ui_module_detail(module_names="omni.ui")
✅ JSON string: get_ui_module_detail(module_names='["omni.ui", "omni.ui.scene", "omni.ui.workspace"]')
✅ Comma format: get_ui_module_detail(module_names="omni.ui, omni.ui.scene, omni.ui.workspace") 
✅ List all: get_ui_module_detail() or get_ui_module_detail(module_names=null)

💡 FOR AI MODELS: You can pass arrays directly like ["omni.ui", "omni.ui.scene"] - no need to convert to strings!

BATCH PROCESSING BENEFITS:
- 70% faster when fetching multiple modules
- Single API call instead of multiple round-trips
- Efficient context window usage
- Maximum compatibility with all AI models

RETURNS:
- For single module: Standard JSON with module details
- For multiple modules: Array with all module details plus metadata
- Includes: classes, functions, file paths, extensions for each module
- Error handling for invalid/missing modules"""


class GetModuleDetailConfig(FunctionBaseConfig, name="get_ui_module_detail"):
    """Configuration for get_ui_module_detail function."""

    name: str = "get_ui_module_detail"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetModuleDetailConfig, framework_wrappers=[])
async def register_get_module_detail(config: GetModuleDetailConfig, builder: Builder):
    """Register get_ui_module_detail function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering get_ui_module_detail in verbose mode")

    async def get_module_detail_wrapper(input: GetModuleDetailInput) -> str:
        """Wrapper for get_ui_module_detail function."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()

        # Handle flexible input: string, array, or None
        try:
            if input.module_names is None:
                modules_to_fetch = None
            elif isinstance(input.module_names, list):
                # Direct array input - validate and use as-is
                if len(input.module_names) == 0:
                    modules_to_fetch = None  # Empty array = list all
                else:
                    # Validate all items are strings
                    for i, item in enumerate(input.module_names):
                        if not isinstance(item, str):
                            return f"ERROR: All items in module_names array must be strings, got {type(item).__name__} at index {i}"
                        if not item.strip():
                            return f"ERROR: Empty string at index {i} in module_names array"
                    modules_to_fetch = [item.strip() for item in input.module_names]
            elif isinstance(input.module_names, str):
                # String input - parse using existing logic
                modules_to_fetch = _parse_module_names_input(input.module_names)
            else:
                return f"ERROR: module_names must be a string, array, or null, got {type(input.module_names).__name__}"

            parameters = {"module_names": input.module_names}
        except ValueError as e:
            return f"ERROR: Invalid module_names parameter: {str(e)}"

        error_msg = None
        success = True

        try:
            # Call the async function directly
            result = await get_ui_module_detail(modules_to_fetch)

            # Use config fields to modify behavior
            if verbose:
                if isinstance(modules_to_fetch, list):
                    logger.debug(f"Retrieved details for {len(modules_to_fetch)} OmniUI modules")
                elif modules_to_fetch is None:
                    logger.debug("Retrieved available modules information")
                else:
                    logger.debug(f"Retrieved details for OmniUI module: {modules_to_fetch}")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve module detail - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_omni_ui_module_detail",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_ui_module_detail: {log_error}")

    # Create function info
    function_info = FunctionInfo.from_fn(
        get_module_detail_wrapper,
        description=GET_MODULE_DETAIL_DESCRIPTION,
        input_schema=GetModuleDetailInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info


================================================================================
FILE: src/omni_aiq_omni_ui/register_get_modules.py
Size: 3.73 KB | Tokens: 736
================================================================================

"""Registration wrapper for list_ui_modules function."""

import logging

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.list_ui_modules import list_ui_modules
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


# Define input schema for zero-argument function
class GetModulesInput(BaseModel):
    """Empty input for zero-argument function."""

    pass


# Tool description
GET_MODULES_DESCRIPTION = """Return a list of all OmniUI module names from the Atlas data.

WHAT IT DOES:
- Retrieves all OmniUI module names from the Atlas
- Returns a simplified list of full module names
- Provides total count of available modules
- Sorts module names alphabetically for easy browsing
- Includes extension information for each module

RETURNS:
A JSON string containing:
- module_names: Sorted list of all OmniUI module full names
- total_count: Total number of modules
- description: Brief description of the modules

USAGE EXAMPLES:
list_ui_modules

This provides access to the complete OmniUI module hierarchy from Atlas data."""


class GetModulesConfig(FunctionBaseConfig, name="list_ui_modules"):
    """Configuration for list_ui_modules function."""

    name: str = "list_ui_modules"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetModulesConfig, framework_wrappers=[])
async def register_get_modules(config: GetModulesConfig, builder: Builder):
    """Register list_ui_modules function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering list_ui_modules in verbose mode")

    async def get_modules_wrapper(input: GetModulesInput) -> str:
        """Zero arguments - schema required."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()
        parameters = {}
        error_msg = None
        success = True

        try:
            # Call the async function directly
            result = await list_ui_modules()

            # Use config fields to modify behavior
            if verbose:
                logger.debug(f"Retrieved OmniUI modules")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve OmniUI modules - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_omni_ui_modules",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for list_ui_modules: {log_error}")

    # Pass input_schema for zero argument function
    function_info = FunctionInfo.from_fn(
        get_modules_wrapper,
        description=GET_MODULES_DESCRIPTION,
        input_schema=GetModulesInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info


================================================================================
FILE: src/omni_aiq_omni_ui/register_get_style_docs.py
Size: 9.77 KB | Tokens: 1,976
================================================================================

"""Registration wrapper for get_ui_style_docs function."""

import json
import logging
from typing import List, Optional, Union

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.get_ui_style_docs import get_ui_style_docs
from .utils.usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def _parse_sections_input(sections_str: str) -> Union[List[str], None]:
    """Parse sections string input into a list of section names.

    Args:
        sections_str: String input that can be:
            - None/empty: Return None (get combined documentation)
            - Single section: "buttons" or "widgets"
            - JSON array: '["buttons", "widgets", "containers"]'
            - Comma-separated: "buttons, widgets, containers"

    Returns:
        List of section names or None if input is empty

    Raises:
        ValueError: If input format is invalid
    """
    if not sections_str or not sections_str.strip():
        return None

    sections_str = sections_str.strip()

    # Try to parse as JSON array first
    if sections_str.startswith("[") and sections_str.endswith("]"):
        try:
            parsed = json.loads(sections_str)
            if isinstance(parsed, list):
                # Validate all items are strings
                for item in parsed:
                    if not isinstance(item, str):
                        raise ValueError(f"All items in JSON array must be strings, got: {type(item).__name__}")
                return [item.strip() for item in parsed if item.strip()]
            else:
                raise ValueError("JSON input must be an array of strings")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON array format: {e}")

    # Try comma-separated format
    if "," in sections_str:
        return [name.strip() for name in sections_str.split(",") if name.strip()]

    # Single section name
    return [sections_str]


# Define input schema for get_ui_style_docs function
class GetStyleDocsInput(BaseModel):
    """Input parameters for get_ui_style_docs function.

    Provide section names in any convenient format - the system will handle the conversion automatically.
    """

    sections: Optional[Union[str, List[str]]] = Field(
        None,
        description="""Sections to retrieve. Accepts multiple flexible formats:
        
        📝 FLEXIBLE INPUT FORMATS (all work the same):
        - Single section: "buttons" or "widgets"
        - Native array: ["buttons", "widgets", "containers"] ← WORKS DIRECTLY!
        - JSON string: '["buttons", "widgets", "containers"]'
        - Comma-separated: "buttons, widgets, containers"
        - Empty/null: Gets complete combined documentation
        
        💡 TIP: Use whatever format is most natural - arrays, strings, or JSON!""",
    )


# Tool description
GET_STYLE_DOCS_DESCRIPTION = """Retrieve comprehensive OmniUI style documentation - SUPER FLEXIBLE INPUT!

🚀 FLEXIBLE API: Accepts ANY input format - strings, arrays, JSON - whatever is natural!

PARAMETER:
- sections: Section names in ANY convenient format:
  * Single section string: "buttons" or "widgets"
  * Native array: ["buttons", "widgets", "containers"] ← WORKS DIRECTLY!
  * JSON string: '["buttons", "widgets", "containers"]'
  * Comma-separated: "buttons, widgets, containers"
  * Empty/null: Gets complete combined documentation

USAGE EXAMPLES (ALL FORMATS WORK):
✅ Direct array: get_ui_style_docs(sections=["buttons", "widgets"])
✅ Single string: get_ui_style_docs(sections="buttons")
✅ JSON string: get_ui_style_docs(sections='["buttons", "widgets", "containers"]')
✅ Comma format: get_ui_style_docs(sections="buttons, widgets, containers") 
✅ Complete docs: get_ui_style_docs() or get_ui_style_docs(sections=null)

💡 FOR AI MODELS: You can pass arrays directly like ["buttons", "widgets"] - no need to convert to strings!

AVAILABLE SECTIONS:
- **overview**: High-level introduction to OmniUI styling system
- **styling**: Core styling syntax and rules
- **units**: Measurement system for UI elements (px, %, em, rem)
- **fonts**: Typography system and text styling
- **shades**: Color palettes and theme management (dark/light modes)
- **window**: Window-level styling and frame customization
- **containers**: Layout components (Frame, Stack, Grid, ScrollArea)
- **widgets**: Individual UI components (Label, Input, Checkbox, ComboBox, etc.)
- **buttons**: Button variations and states (normal, hover, pressed, disabled)
- **sliders**: Slider and range components with customization options
- **shapes**: Basic geometric elements (Rectangle, Circle, Triangle, Polygon)
- **line**: Line and curve elements with styling options

RETURNS:
- For single section: Section content with metadata
- For multiple sections: Dictionary of sections with their content
- For combined: Complete documentation with all sections (37,820+ tokens)
- Includes: Property descriptions, usage examples, and best practices
- Maximum compatibility with all AI models

USE CASES:
- Learning OmniUI styling syntax and customization
- Finding specific styling properties for UI components
- Understanding theme and color management systems
- Implementing custom widget styles and layouts"""


class GetStyleDocsConfig(FunctionBaseConfig, name="get_ui_style_docs"):
    """Configuration for get_ui_style_docs function."""

    name: str = "get_ui_style_docs"
    verbose: bool = Field(default=False, description="Enable detailed logging")


@register_function(config_type=GetStyleDocsConfig, framework_wrappers=[])
async def register_get_style_docs(config: GetStyleDocsConfig, builder: Builder):
    """Register get_ui_style_docs function with AIQ."""

    # Use config directly
    verbose = config.verbose

    # Access config fields here
    if verbose:
        logger.info(f"Registering get_ui_style_docs in verbose mode")

    async def get_style_docs_wrapper(input: GetStyleDocsInput) -> str:
        """Wrapper for get_ui_style_docs with AIQ integration."""
        import time

        usage_logger = get_usage_logger()
        start_time = time.time()

        # Handle flexible input: string, array, or None
        try:
            if input.sections is None:
                sections_to_fetch = None
            elif isinstance(input.sections, list):
                # Direct array input - validate and use as-is
                if len(input.sections) == 0:
                    sections_to_fetch = None  # Empty array = get combined docs
                else:
                    # Validate all items are strings
                    for i, item in enumerate(input.sections):
                        if not isinstance(item, str):
                            return f"ERROR: All items in sections array must be strings, got {type(item).__name__} at index {i}"
                        if not item.strip():
                            return f"ERROR: Empty string at index {i} in sections array"
                    sections_to_fetch = [item.strip() for item in input.sections]
            elif isinstance(input.sections, str):
                # String input - parse using existing logic
                sections_to_fetch = _parse_sections_input(input.sections)
            else:
                return f"ERROR: sections must be a string, array, or null, got {type(input.sections).__name__}"

            parameters = {"sections": input.sections}
        except ValueError as e:
            return f"ERROR: Invalid sections parameter: {str(e)}"

        error_msg = None
        success = True

        try:
            # Call the async function with parsed parameters (new simplified API)
            result = await get_ui_style_docs(sections_to_fetch)

            # Use config fields to modify behavior
            if verbose:
                if sections_to_fetch is None:
                    section_info = "combined"
                elif len(sections_to_fetch) == 1:
                    section_info = sections_to_fetch[0]
                else:
                    section_info = f"{len(sections_to_fetch)} sections"
                logger.debug(f"Retrieved style documentation for: {section_info}")

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve style documentation - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="get_ui_style_docs",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for get_ui_style_docs: {log_error}")

    # Create function info with input schema
    function_info = FunctionInfo.from_fn(
        get_style_docs_wrapper,
        description=GET_STYLE_DOCS_DESCRIPTION,
        input_schema=GetStyleDocsInput,
    )

    # Mark this as an MCP-exposed tool (not a workflow)
    function_info.metadata = {"mcp_exposed": True}

    yield function_info


================================================================================
FILE: src/omni_aiq_omni_ui/register_search_ui_window_examples.py
Size: 6.75 KB | Tokens: 1,333
================================================================================

"""Registration wrapper for search_ui_window_examples function."""

import logging
import os
from typing import Optional

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from pydantic import BaseModel, Field

from .functions.search_ui_window_examples import search_ui_window_examples

logger = logging.getLogger(__name__)


# Define input schema for single argument function
class GetWindowExamplesInput(BaseModel):
    """Input for search_ui_window_examples function."""

    request: str = Field(description="Your query describing the desired UI window example")


# Tool description
GET_WINDOW_EXAMPLES_DESCRIPTION = """Retrieves relevant UI window examples using semantic vector search from a curated database of OmniUI implementations.

WHAT IT DOES:
- Converts your query to embeddings using NVIDIA's embedding model
- Performs semantic similarity search against indexed UI window/dialog implementations
- Returns formatted examples with descriptions, complete code, and file paths
- Focuses specifically on window creation, dialog boxes, and UI container patterns

QUERY MATCHING:
Your query is compared against OmniUI window and dialog implementations, including:
- Window creation with various configurations (modal, non-modal, resizable, etc.)
- Dialog boxes and popup interfaces
- UI containers with buttons, controls, and layouts
- Error dialogs and confirmation windows
- Animation and curve editing interfaces
- Settings and configuration windows

ARGUMENTS:
- request (str): Your query describing the desired UI window example

RETURNS:
Formatted UI window examples with:
- Detailed descriptions of window functionality
- Complete Python code implementations
- File paths and function locations
- Class names and line numbers

USAGE EXAMPLES:
search_ui_window_examples "Create a modal dialog with buttons"
search_ui_window_examples "Window with sliders and controls"
search_ui_window_examples "Error message dialog box"
search_ui_window_examples "Animation curve simplification window"
search_ui_window_examples "Resizable window with UI components"

TIPS FOR BETTER RESULTS:
- Use window-specific terminology (e.g., "modal", "dialog", "window", "popup")
- Include UI component types (e.g., "buttons", "sliders", "checkboxes", "fields")
- Reference window behaviors (e.g., "resizable", "closable", "modal", "fixed size")
- Ask about specific UI patterns (e.g., "error dialog", "settings window", "confirmation")
"""


class GetWindowExamplesConfig(FunctionBaseConfig, name="search_ui_window_examples"):
    """Configuration for search_ui_window_examples function."""

    name: str = "search_ui_window_examples"
    verbose: bool = Field(default=False, description="Enable detailed logging")
    top_k: int = Field(default=5, description="Number of window examples to return")
    format_type: str = Field(default="formatted", description="Format type: 'structured', 'formatted', or 'raw'")

    # Embedding configuration
    embedding_model: Optional[str] = Field(default="nvidia/nv-embedqa-e5-v5", description="Embedding model to use")
    embedding_endpoint: Optional[str] = Field(
        default=None, description="Embedding service endpoint (None for NVIDIA API)"
    )
    embedding_api_key: Optional[str] = Field(default="${NVIDIA_API_KEY}", description="API key for embedding service")

    # FAISS database configuration
    faiss_index_path: Optional[str] = Field(default=None, description="Path to FAISS index (uses default if None)")


@register_function(config_type=GetWindowExamplesConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def register_search_ui_window_examples(config: GetWindowExamplesConfig, builder: Builder):
    """Register search_ui_window_examples function with AIQ."""

    # Access config fields here
    if config.verbose:
        logger.info(f"Registering search_ui_window_examples in verbose mode")

    async def search_ui_window_examples_wrapper(input: GetWindowExamplesInput) -> str:
        """Single argument with schema."""
        import time

        from omni_aiq_omni_ui.utils.usage_logging import get_usage_logger

        # Extract the request string from the input model
        request = input.request

        # Debug logging
        logger.info(f"[DEBUG] search_ui_window_examples_wrapper called with input type: {type(input)}")
        logger.info(f"[DEBUG] search_ui_window_examples_wrapper request value: {request}")

        usage_logger = get_usage_logger()
        start_time = time.time()
        parameters = {"request": request}
        error_msg = None
        success = True

        try:
            # Handle environment variable substitution for API keys
            embedding_api_key = config.embedding_api_key
            if embedding_api_key == "${NVIDIA_API_KEY}":
                embedding_api_key = os.getenv("NVIDIA_API_KEY")

            result = await search_ui_window_examples(
                request,
                top_k=config.top_k,
                format_type=config.format_type,
                embedding_config={
                    "model": config.embedding_model,
                    "endpoint": config.embedding_endpoint,
                    "api_key": embedding_api_key,
                },
                faiss_index_path=config.faiss_index_path,
            )

            # Use config fields to modify behavior
            if config.verbose:
                logger.debug(
                    f"Retrieved UI window examples for: {request}, top_k: {config.top_k}, format: {config.format_type}"
                )

            if result["success"]:
                return result["result"]
            else:
                error_msg = result.get("error", "Unknown error")
                success = False
                return f"ERROR: {error_msg}"

        except Exception as e:
            error_msg = str(e)
            success = False
            return f"ERROR: Failed to retrieve UI window examples - {error_msg}"
        finally:
            # Log usage if enabled
            if usage_logger and usage_logger.enabled:
                try:
                    execution_time = time.time() - start_time
                    usage_logger.log_tool_call(
                        tool_name="search_ui_window_examples",
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for search_ui_window_examples: {log_error}")

    # Pass input_schema for proper MCP parameter handling
    yield FunctionInfo.from_fn(
        search_ui_window_examples_wrapper,
        description=GET_WINDOW_EXAMPLES_DESCRIPTION,
        input_schema=GetWindowExamplesInput,
    )


================================================================================
FILE: src/omni_aiq_omni_ui/services/__init__.py
Size: 121.00 B | Tokens: 32
================================================================================

"""Services module for OmniUI tools."""

from .omni_ui_atlas import OmniUIAtlasService

__all__ = ["OmniUIAtlasService"]


================================================================================
FILE: src/omni_aiq_omni_ui/services/analyze_ui_functions.py
Size: 7.12 KB | Tokens: 1,601
================================================================================

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from claude_code_sdk import ClaudeCodeOptions, ClaudeSDKClient


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse JSON file containing function entries"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(file_path: str, data: List[Dict[str, Any]]) -> None:
    """Save updated JSON data back to file"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


async def generate_ui_description(
    client: ClaudeSDKClient, function_body: str, class_name: str, function_name: str
) -> str:
    """Generate description for a UI function using Claude"""

    system_prompt = """You are an expert at analyzing OmniUI code. Your task is to provide a concise description of what UI elements and functionality a given method creates. 
    Focus on:
    - The type of UI being created (window, dialog, panel, etc.)
    - Key UI components used (buttons, labels, text fields, etc.)
    - The apparent purpose or objective of the UI
    Keep your description brief (2-3 sentences max) and technical."""

    user_prompt = f"""Analyze this OmniUI method and describe what UI it creates:

Class: {class_name}
Method: {function_name}

Function body:
{function_body}

Provide a brief technical description of what UI this creates and its purpose."""

    try:
        # Send the query with system prompt and user content
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        await client.query(full_prompt)

        # Collect the response
        response_text = ""
        async for message in client.receive_response():
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        response_text += block.text

            # Check if this is the final result message
            if type(message).__name__ == "ResultMessage":
                break

        return response_text.strip() if response_text else "No response generated"
    except Exception as e:
        print(f"Error generating description: {e}")
        return "Error generating description"


def setup_enterprise_auth():
    """Configure environment for enterprise/subscription authentication."""

    # Remove API key if present to force subscription authentication
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]
        print("🔧 Removed API key from environment")

    # Force subscription authentication mode
    os.environ["CLAUDE_USE_SUBSCRIPTION"] = "true"
    print("✅ Configured for subscription authentication")

    # Check for credential files
    claude_dir = os.path.expanduser("~/.claude")
    credentials_file = os.path.join(claude_dir, ".credentials.json")
    oauth_file = os.path.join(claude_dir, "oauth_token.json")

    if os.path.exists(credentials_file):
        print(f"✅ Found credentials: {credentials_file}")
    else:
        print(f"⚠️  Credentials not found: {credentials_file}")
        print("   Run 'claude login' to authenticate with your enterprise account")

    if os.path.exists(oauth_file):
        print(f"✅ Found OAuth tokens: {oauth_file}")
    else:
        print(f"⚠️  OAuth tokens not found: {oauth_file}")


async def process_entries(json_file_path: str, output_file_path: str = None):
    """Main function to process all entries in the JSON file"""

    # Use output path if provided, otherwise overwrite input
    if output_file_path is None:
        output_file_path = json_file_path

    print(f"Loading JSON file from: {json_file_path}")
    entries = load_json_file(json_file_path)
    print(f"Found {len(entries)} entries to process")

    # Setup enterprise authentication
    print("\n🔐 Setting up enterprise authentication...")
    setup_enterprise_auth()

    # Configure options to use Claude 3.5 Haiku for cost efficiency
    # With enterprise subscription, no API key needed
    options = ClaudeCodeOptions(
        model="claude-3-5-haiku-20241022",  # Latest Haiku model
        max_turns=1,  # Simple single-turn interaction
        allowed_tools=[],  # No tools needed for text generation
        permission_mode="plan",  # Read-only mode since we're just generating descriptions
    )

    print("✅ Claude Code SDK client configured with Claude 3.5 Haiku model (Enterprise Subscription)")

    # Process each entry using Claude Code SDK
    async with ClaudeSDKClient(options=options) as client:
        for i, entry in enumerate(entries, 1):
            print(f"\nProcessing entry {i}/{len(entries)}: {entry.get('file_path', 'Unknown')}")
            print(f"  Class: {entry.get('class_name', 'N/A')}, Method: {entry.get('function_name', 'N/A')}")

            # Skip if already has description
            if "description" in entry and entry["description"]:
                print(f"  Skipping - already has description")
                continue

            # C:\repos\kit-app-template\_build\windows-x86_64\release\extscache
            # Get function body
            function_body = entry.get("function_body", "")
            if not function_body:
                print(f"  Warning: No function body found")
                entry["description"] = "No function body available"
                continue

            # Generate description
            print(f"  Generating description...")
            description = await generate_ui_description(
                client, function_body, entry.get("class_name", "Unknown"), entry.get("function_name", "Unknown")
            )

            # Add description to entry
            entry["description"] = description
            print(f"  Description: {description[:100]}...")

            # Rate limiting to avoid API throttling
            await asyncio.sleep(0.5)

            # Save progress every 10 entries
            if i % 10 == 0:
                print(f"\nSaving progress at entry {i}...")
                save_json_file(output_file_path, entries)

    # Final save
    print(f"\nSaving final results to: {output_file_path}")
    save_json_file(output_file_path, entries)
    print(f"Processing complete! All {len(entries)} entries have been processed.")

    # Print summary
    entries_with_desc = sum(1 for e in entries if "description" in e and e["description"])
    print(f"\nSummary: {entries_with_desc}/{len(entries)} entries now have descriptions")


async def main():
    """Main async function to run the processing"""
    # Configuration
    INPUT_JSON_FILE = "C:/repos/kit-app-template/ui_window_atlas.json"  # Change this to your actual file name
    OUTPUT_JSON_FILE = "C:/repos/kit-app-template/ui_functions_with_descriptions.json"  # Optional: separate output file

    # Check if input file exists
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"Error: Input file '{INPUT_JSON_FILE}' not found!")
        print("Please update the INPUT_JSON_FILE variable with the correct path to your JSON file.")
    else:
        # Process the entries
        await process_entries(INPUT_JSON_FILE, OUTPUT_JSON_FILE)


if __name__ == "__main__":
    asyncio.run(main())


================================================================================
FILE: src/omni_aiq_omni_ui/services/build_complete_ui_examples_pipeline.py
Size: 6.78 KB | Tokens: 1,598
================================================================================

"""Complete pipeline to build UI window examples database and test retrieval."""

import logging
import os
import sys
from typing import Optional

# Add the parent directory to the path to allow importing from omni_aiq_omni_ui
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from omni_aiq_omni_ui.services.build_embedding_vectors import build_embedding_vectors
from omni_aiq_omni_ui.services.build_faiss_database import build_faiss_database
from omni_aiq_omni_ui.services.ui_window_examples_retrieval import (
    create_ui_window_examples_retriever,
    get_ui_window_examples,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_file_paths():
    """Setup and return all required file paths."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")

    paths = {
        "input_json": os.path.join(data_dir, "ui_functions_with_descriptions.json"),
        "vector_json": os.path.join(data_dir, "ui_functions_with_descriptions_vector.json"),
        "faiss_db": os.path.join(data_dir, "ui_window_examples_faiss"),
        "data_dir": data_dir,
    }

    return paths


def check_prerequisites(paths: dict) -> bool:
    """Check if all prerequisite files exist."""
    if not os.path.exists(paths["input_json"]):
        logger.error(f"Input JSON file not found: {paths['input_json']}")
        return False

    logger.info("Prerequisites check passed")
    return True


def step_1_build_embeddings(paths: dict, api_key: str, force_rebuild: bool = False) -> bool:
    """Step 1: Build embedding vectors if they don't exist."""
    if os.path.exists(paths["vector_json"]) and not force_rebuild:
        logger.info("Embedding vectors file already exists, skipping Step 1")
        return True

    logger.info("Step 1: Building embedding vectors...")
    try:
        build_embedding_vectors(
            input_file_path=paths["input_json"], output_file_path=paths["vector_json"], api_key=api_key
        )
        logger.info("Step 1 completed successfully")
        return True
    except Exception as e:
        logger.error(f"Step 1 failed: {e}")
        return False


def step_2_build_faiss_database(paths: dict, api_key: str, force_rebuild: bool = False) -> bool:
    """Step 2: Build FAISS database from embedding vectors."""
    if os.path.exists(paths["faiss_db"]) and not force_rebuild:
        logger.info("FAISS database already exists, skipping Step 2")
        return True

    if not os.path.exists(paths["vector_json"]):
        logger.error("Vector JSON file not found for Step 2")
        return False

    logger.info("Step 2: Building FAISS database...")
    try:
        build_faiss_database(
            vector_json_path=paths["vector_json"], faiss_output_path=paths["faiss_db"], api_key=api_key
        )
        logger.info("Step 2 completed successfully")
        return True
    except Exception as e:
        logger.error(f"Step 2 failed: {e}")
        return False


def step_3_test_retrieval(paths: dict, api_key: str) -> bool:
    """Step 3: Test the retrieval system."""
    if not os.path.exists(paths["faiss_db"]):
        logger.error("FAISS database not found for Step 3")
        return False

    logger.info("Step 3: Testing retrieval system...")
    try:
        # Create retriever
        retriever = create_ui_window_examples_retriever(faiss_index_path=paths["faiss_db"], api_key=api_key, top_k=3)

        # Test with a sample query
        test_query = "create window with buttons and controls"
        logger.info(f"Testing with query: '{test_query}'")

        # Get structured results
        results = get_ui_window_examples(test_query, retriever, top_k=3, format_type="structured")

        logger.info(f"Retrieved {len(results)} results")

        # Display sample results
        for i, result in enumerate(results[:2], 1):
            logger.info(f"Result {i}:")
            logger.info(f"  Function: {result['class_name']}.{result['function_name']}()")
            logger.info(f"  File: {result['file_path']}")
            logger.info(f"  Description: {result['description'][:100]}...")

        logger.info("Step 3 completed successfully")
        return True

    except Exception as e:
        logger.error(f"Step 3 failed: {e}")
        return False


def run_pipeline(
    api_key: Optional[str] = None,
    force_rebuild_embeddings: bool = False,
    force_rebuild_faiss: bool = False,
    skip_test: bool = False,
) -> bool:
    """Run the complete UI examples pipeline.

    Args:
        api_key: NVIDIA API key (will use environment variable if not provided)
        force_rebuild_embeddings: Force rebuild of embedding vectors
        force_rebuild_faiss: Force rebuild of FAISS database
        skip_test: Skip the retrieval test

    Returns:
        True if pipeline completed successfully, False otherwise
    """
    logger.info("Starting UI Window Examples Pipeline")
    logger.info("=" * 60)

    # Setup
    if api_key is None:
        api_key = os.getenv("NVIDIA_API_KEY", "")
        if not api_key:
            logger.warning("No NVIDIA API key provided or found in environment")

    paths = setup_file_paths()
    logger.info(f"Data directory: {paths['data_dir']}")

    # Check prerequisites
    if not check_prerequisites(paths):
        return False

    # Step 1: Build embeddings
    if not step_1_build_embeddings(paths, api_key, force_rebuild_embeddings):
        return False

    # Step 2: Build FAISS database
    if not step_2_build_faiss_database(paths, api_key, force_rebuild_faiss):
        return False

    # Step 3: Test retrieval
    if not skip_test:
        if not step_3_test_retrieval(paths, api_key):
            return False

    logger.info("=" * 60)
    logger.info("UI Window Examples Pipeline completed successfully!")
    logger.info(f"FAISS database location: {paths['faiss_db']}")
    logger.info("You can now use UIWindowExamplesRetriever to search for UI examples")

    return True


def main():
    """Main function with command line interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Build UI Window Examples Pipeline")
    parser.add_argument("--api-key", help="NVIDIA API key")
    parser.add_argument("--force-embeddings", action="store_true", help="Force rebuild embedding vectors")
    parser.add_argument("--force-faiss", action="store_true", help="Force rebuild FAISS database")
    parser.add_argument("--skip-test", action="store_true", help="Skip retrieval test")

    args = parser.parse_args()

    success = run_pipeline(
        api_key=args.api_key,
        force_rebuild_embeddings=args.force_embeddings,
        force_rebuild_faiss=args.force_faiss,
        skip_test=args.skip_test,
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()


================================================================================
FILE: src/omni_aiq_omni_ui/services/build_embedding_vectors.py
Size: 4.22 KB | Tokens: 925
================================================================================

"""Script to build embedding vectors for UI function descriptions."""

import json
import logging
import os
import sys
from typing import Any, Dict, List

# Add the parent directory to the path to allow importing from omni_aiq_omni_ui
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from omni_aiq_omni_ui.services.retrieval import create_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_embedding_vectors(
    input_file_path: str, output_file_path: str, endpoint_url: str = None, api_key: str = ""
) -> None:
    """Build embedding vectors for descriptions in UI functions JSON file.

    Args:
        input_file_path: Path to the input JSON file
        output_file_path: Path to save the output JSON file with vectors
        endpoint_url: Optional custom endpoint URL for embeddings
        api_key: API key for authentication
    """
    logger.info(f"Starting embedding vector creation for {input_file_path}")

    # Create embedder
    try:
        embedder = create_embeddings(endpoint_url, api_key)
        logger.info("Successfully created embedder")
    except Exception as e:
        logger.error(f"Failed to create embedder: {e}")
        raise

    # Load input JSON file
    try:
        with open(input_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} entries from {input_file_path}")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        raise

    # Process each entry and add embedding vectors
    processed_entries = []
    total_entries = len(data)

    for i, entry in enumerate(data, 1):
        try:
            # Get the description
            description = entry.get("description", "")

            if not description:
                logger.warning(f"Entry {i} has no description, skipping embedding")
                # Add empty vector field
                entry["embedding_vector"] = []
                processed_entries.append(entry)
                continue

            # Create embedding for the description
            embedding = embedder.embed_query(description)

            # Add the embedding vector to the entry
            entry["embedding_vector"] = embedding
            processed_entries.append(entry)

            # Log progress every 100 entries
            if i % 3 == 0:
                logger.info(f"Processed {i}/{total_entries} entries ({i/total_entries*100:.1f}%)")

        except Exception as e:
            logger.error(f"Failed to process entry {i}: {e}")
            # Add empty vector on failure
            entry["embedding_vector"] = []
            processed_entries.append(entry)
            continue

    # Save the output JSON file
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(processed_entries, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully saved {len(processed_entries)} entries with embeddings to {output_file_path}")

    except Exception as e:
        logger.error(f"Failed to save output file: {e}")
        raise


def main():
    """Main function to run the embedding vector creation."""
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")

    input_file = os.path.join(data_dir, "ui_functions_with_descriptions.json")
    output_file = os.path.join(data_dir, "ui_functions_with_descriptions_vector.json")

    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return

    # Get API key from environment variable
    api_key = os.getenv("NVIDIA_API_KEY", "")

    if not api_key:
        logger.warning("No NVIDIA_API_KEY found in environment variables")

    # Build embedding vectors
    try:
        build_embedding_vectors(input_file_path=input_file, output_file_path=output_file, api_key=api_key)
        logger.info("Embedding vector creation completed successfully!")

    except Exception as e:
        logger.error(f"Embedding vector creation failed: {e}")
        raise


if __name__ == "__main__":
    main()


================================================================================
FILE: src/omni_aiq_omni_ui/services/build_faiss_database.py
Size: 5.13 KB | Tokens: 1,117
================================================================================

"""Script to build FAISS database from UI functions with embedding vectors."""

import json
import logging
import os
import sys
from typing import List

# Add the parent directory to the path to allow importing from omni_aiq_omni_ui
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from omni_aiq_omni_ui.services.retrieval import create_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_faiss_database(
    vector_json_path: str, faiss_output_path: str, endpoint_url: str = None, api_key: str = ""
) -> None:
    """Build FAISS database from UI functions with embedding vectors.

    Args:
        vector_json_path: Path to the JSON file with embedding vectors
        faiss_output_path: Path to save the FAISS database
        endpoint_url: Optional custom endpoint URL for embeddings
        api_key: API key for authentication
    """
    logger.info(f"Starting FAISS database creation from {vector_json_path}")

    # Create embedder (needed for FAISS initialization)
    try:
        embedder = create_embeddings(endpoint_url, api_key)
        logger.info("Successfully created embedder")
    except Exception as e:
        logger.error(f"Failed to create embedder: {e}")
        raise

    # Load the vector JSON file
    try:
        with open(vector_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} entries from {vector_json_path}")
    except Exception as e:
        logger.error(f"Failed to load vector JSON file: {e}")
        raise

    # Prepare documents for FAISS
    documents = []
    embeddings = []

    for i, entry in enumerate(data, 1):
        try:
            # Extract required fields
            description = entry.get("description", "")
            function_body = entry.get("function_body", "")
            file_path = entry.get("file_path", "")
            function_name = entry.get("function_name", "unknown")
            class_name = entry.get("class_name", "unknown")
            line_number = entry.get("line_number", 0)
            embedding_vector = entry.get("embedding_vector", [])

            # Skip entries without embeddings
            if not embedding_vector:
                logger.warning(f"Entry {i} has no embedding vector, skipping")
                continue

            # Create document with description as page_content and metadata
            doc = Document(
                page_content=description,
                metadata={
                    "description": description,
                    "code": function_body,
                    "file_path": file_path,
                    "function_name": function_name,
                    "class_name": class_name,
                    "line_number": line_number,
                    "entry_index": i - 1,
                },
            )

            documents.append(doc)
            embeddings.append(embedding_vector)

            # Log progress every 500 entries
            if i % 500 == 0:
                logger.info(f"Processed {i}/{len(data)} entries ({i/len(data)*100:.1f}%)")

        except Exception as e:
            logger.error(f"Failed to process entry {i}: {e}")
            continue

    logger.info(f"Prepared {len(documents)} documents for FAISS database")

    # Create FAISS database
    try:
        # Create FAISS index from documents and embeddings
        vectorstore = FAISS.from_embeddings(
            text_embeddings=[(doc.page_content, emb) for doc, emb in zip(documents, embeddings)],
            embedding=embedder,
            metadatas=[doc.metadata for doc in documents],
        )

        # Save the FAISS database
        os.makedirs(os.path.dirname(faiss_output_path), exist_ok=True)
        vectorstore.save_local(faiss_output_path)

        logger.info(f"Successfully saved FAISS database to {faiss_output_path}")
        logger.info(f"Database contains {len(documents)} entries")

    except Exception as e:
        logger.error(f"Failed to create/save FAISS database: {e}")
        raise


def main():
    """Main function to run the FAISS database creation."""
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")

    input_file = os.path.join(data_dir, "ui_functions_with_descriptions_vector.json")
    output_dir = os.path.join(data_dir, "ui_window_examples_faiss")

    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file does not exist: {input_file}")
        return

    # Get API key from environment variable
    api_key = os.getenv("NVIDIA_API_KEY", "")

    if not api_key:
        logger.warning("No NVIDIA_API_KEY found in environment variables")

    # Build FAISS database
    try:
        build_faiss_database(vector_json_path=input_file, faiss_output_path=output_dir, api_key=api_key)
        logger.info("FAISS database creation completed successfully!")

    except Exception as e:
        logger.error(f"FAISS database creation failed: {e}")
        raise


if __name__ == "__main__":
    main()


================================================================================
FILE: src/omni_aiq_omni_ui/services/omni_ui_atlas.py
Size: 14.99 KB | Tokens: 3,099
================================================================================

"""OmniUI Atlas data service for the OmniUI MCP server."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Get the path to ui_atlas.json
UI_ATLAS_FILE_PATH = Path(__file__).parent.parent / "data" / "ui_atlas.json"


class OmniUIAtlasService:
    """Service for managing OmniUI Atlas data operations."""

    def __init__(self, atlas_file_path: str = None):
        """Initialize the OmniUI Atlas service.

        Args:
            atlas_file_path: Path to the UI Atlas JSON file
        """
        self.atlas_file_path = Path(atlas_file_path) if atlas_file_path else UI_ATLAS_FILE_PATH
        self.atlas_data = None
        self._load_atlas_data()

    def _load_atlas_data(self) -> None:
        """Load OmniUI Atlas data from file."""
        try:
            with open(self.atlas_file_path, "r", encoding="utf-8") as f:
                self.atlas_data = json.load(f)
            logger.info(f"Successfully loaded OmniUI Atlas data from {self.atlas_file_path}")
        except FileNotFoundError:
            logger.warning(f"OmniUI Atlas file not found: {self.atlas_file_path}")
            self.atlas_data = None
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.atlas_file_path}: {e}")
            self.atlas_data = None
        except Exception as e:
            logger.error(f"Unexpected error loading Atlas data: {e}")
            self.atlas_data = None

    def is_available(self) -> bool:
        """Check if OmniUI Atlas data is available."""
        return self.atlas_data is not None

    def list_ui_modules(self) -> Dict[str, Any]:
        """Get all OmniUI modules.

        Returns:
            Dictionary containing module information and summary
        """
        if not self.is_available():
            return {"error": "OmniUI Atlas data is not available."}

        if "modules" not in self.atlas_data:
            return {"error": "No modules section found in OmniUI Atlas data."}

        modules = self.atlas_data["modules"]
        result = {
            "modules": [],
            "total_count": 0,
            "summary": {
                "total_modules": 0,
                "modules_with_classes": 0,
                "modules_with_functions": 0,
                "total_classes": 0,
                "total_functions": 0,
                "extensions": set(),
            },
        }

        for module_key, module_info in modules.items():
            name = module_info.get("name", "Unknown")

            # Skip __DOC modules for cleaner output
            if name == "__DOC":
                continue

            full_name = module_info.get("full_name", module_key)
            file_path = module_info.get("file_path", "")
            class_names = module_info.get("class_names", [])
            function_names = module_info.get("function_names", [])
            extension_name = module_info.get("extension_name", "")

            module_data = {
                "name": name,
                "full_name": full_name,
                "file_path": file_path,
                "extension_name": extension_name,
                "class_count": len(class_names),
                "function_count": len(function_names),
                "class_names": class_names,
                "function_names": function_names,
            }

            result["modules"].append(module_data)
            result["summary"]["total_modules"] += 1
            result["summary"]["total_classes"] += len(class_names)
            result["summary"]["total_functions"] += len(function_names)

            if class_names:
                result["summary"]["modules_with_classes"] += 1
            if function_names:
                result["summary"]["modules_with_functions"] += 1
            if extension_name:
                result["summary"]["extensions"].add(extension_name)

        result["total_count"] = len(result["modules"])
        result["summary"]["extensions"] = list(result["summary"]["extensions"])
        result["summary"]["unique_extensions"] = len(result["summary"]["extensions"])

        return result

    def list_ui_classes(self) -> Dict[str, Any]:
        """Get all OmniUI classes.

        Returns:
            Dictionary containing class information and summary
        """
        if not self.is_available():
            return {"error": "OmniUI Atlas data is not available."}

        if "classes" not in self.atlas_data:
            return {"error": "No classes section found in OmniUI Atlas data."}

        classes = self.atlas_data["classes"]
        result = {
            "classes": [],
            "total_count": 0,
            "summary": {"total_classes": 0, "classes_with_methods": 0, "total_methods": 0, "modules": set()},
        }

        for class_key, class_info in classes.items():
            name = class_info.get("name", "Unknown")
            full_name = class_info.get("full_name", class_key)
            module_name = class_info.get("module_name", "Unknown")
            methods = class_info.get("methods", [])

            class_data = {
                "name": name,
                "full_name": full_name,
                "module_name": module_name,
                "method_count": len(methods),
                "methods": methods,
                "docstring": class_info.get("docstring", ""),
                "parent_classes": class_info.get("parent_classes", []),
            }

            result["classes"].append(class_data)
            result["summary"]["total_classes"] += 1
            result["summary"]["total_methods"] += len(methods)
            result["summary"]["modules"].add(module_name)

            if methods:
                result["summary"]["classes_with_methods"] += 1

        result["total_count"] = len(result["classes"])
        result["summary"]["modules"] = list(result["summary"]["modules"])
        result["summary"]["unique_modules"] = len(result["summary"]["modules"])

        return result

    def get_ui_module_detail(self, module_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific module.

        Args:
            module_name: Name of the module to look up

        Returns:
            Dictionary containing detailed module information
        """
        if not self.is_available():
            return {"error": "OmniUI Atlas data is not available."}

        if "modules" not in self.atlas_data:
            return {"error": "No modules section found in OmniUI Atlas data."}

        modules = self.atlas_data["modules"]

        # Use fuzzy matching to find the best module match
        from ..utils.fuzzy_matching import find_best_matches

        matches = find_best_matches(
            query=module_name,
            candidates=modules,
            key_func=lambda x: x,
            value_func=lambda x: modules[x],
            name_func=lambda x: x.get("full_name", ""),
            threshold=0.5,
        )

        if not matches:
            return {
                "error": f"Module '{module_name}' not found in OmniUI Atlas data.",
                "suggestion": "Try using get_omni_ui_modules() to see available modules.",
            }

        # Get the best match
        target_key, target_module, match_score = matches[0]

        # Get associated classes
        module_classes = []
        if "classes" in self.atlas_data:
            for class_key, class_info in self.atlas_data["classes"].items():
                if class_info.get("module_name") == target_module.get("full_name"):
                    module_classes.append(
                        {
                            "name": class_info.get("name"),
                            "full_name": class_info.get("full_name"),
                            "method_count": len(class_info.get("methods", [])),
                        }
                    )

        # Get associated functions
        module_functions = []
        if "functions" in self.atlas_data:
            for func_key, func_info in self.atlas_data["functions"].items():
                if func_info.get("module_name") == target_module.get("full_name"):
                    module_functions.append(
                        {
                            "name": func_info.get("name"),
                            "full_name": func_info.get("full_name"),
                            "docstring": func_info.get("docstring", ""),
                        }
                    )

        return {
            "name": target_module.get("name"),
            "full_name": target_module.get("full_name"),
            "file_path": target_module.get("file_path"),
            "extension_name": target_module.get("extension_name", ""),
            "class_names": target_module.get("class_names", []),
            "function_names": target_module.get("function_names", []),
            "classes": module_classes,
            "functions": module_functions,
            "match_score": match_score,
            "total_classes": len(module_classes),
            "total_functions": len(module_functions),
        }

    def get_ui_class_detail(self, class_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific class.

        Args:
            class_name: Name of the class to look up

        Returns:
            Dictionary containing detailed class information
        """
        if not self.is_available():
            return {"error": "OmniUI Atlas data is not available."}

        if "classes" not in self.atlas_data:
            return {"error": "No classes section found in OmniUI Atlas data."}

        classes = self.atlas_data["classes"]

        # Use fuzzy matching to find the best class match
        from ..utils.fuzzy_matching import find_best_matches

        matches = find_best_matches(
            query=class_name,
            candidates=classes,
            key_func=lambda x: x,
            value_func=lambda x: classes[x],
            name_func=lambda x: x.get("full_name", ""),
            threshold=0.5,
        )

        if not matches:
            return {
                "error": f"Class '{class_name}' not found in OmniUI Atlas data.",
                "suggestion": "Try using get_omni_ui_classes() to see available classes.",
            }

        # Get the best match
        target_key, target_class, match_score = matches[0]

        # Get method details
        method_details = []
        methods = target_class.get("methods", [])

        if "methods" in self.atlas_data:
            for method_name in methods:
                # Find the method in the atlas
                method_key = f"{target_class.get('full_name')}.{method_name}"
                if method_key in self.atlas_data["methods"]:
                    method_info = self.atlas_data["methods"][method_key]
                    method_details.append(
                        {
                            "name": method_info.get("name"),
                            "full_name": method_info.get("full_name"),
                            "parameters": method_info.get("parameters", []),
                            "return_type": method_info.get("return_type", ""),
                            "docstring": method_info.get("docstring", ""),
                            "is_static": method_info.get("is_static", False),
                            "is_classmethod": method_info.get("is_classmethod", False),
                        }
                    )

        return {
            "name": target_class.get("name"),
            "full_name": target_class.get("full_name"),
            "module_name": target_class.get("module_name"),
            "docstring": target_class.get("docstring", ""),
            "parent_classes": target_class.get("parent_classes", []),
            "methods": methods,
            "method_details": method_details,
            "match_score": match_score,
            "total_methods": len(methods),
        }

    def get_ui_method_detail(self, method_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific method.

        Args:
            method_name: Name of the method to look up (can be partial or full name)

        Returns:
            Dictionary containing detailed method information
        """
        if not self.is_available():
            return {"error": "OmniUI Atlas data is not available."}

        if "methods" not in self.atlas_data:
            return {"error": "No methods section found in OmniUI Atlas data."}

        methods = self.atlas_data["methods"]

        # Use fuzzy matching to find the best method match
        from ..utils.fuzzy_matching import find_best_matches

        matches = find_best_matches(
            query=method_name,
            candidates=methods,
            key_func=lambda x: x,
            value_func=lambda x: methods[x],
            name_func=lambda x: x.get("full_name", ""),
            threshold=0.5,
        )

        if not matches:
            return {
                "error": f"Method '{method_name}' not found in OmniUI Atlas data.",
                "suggestion": "Try using get_omni_ui_class_detail() to see methods for a specific class.",
            }

        # Get the best match
        target_key, target_method, match_score = matches[0]

        # Parse the class name from the full method name
        full_name = target_method.get("full_name", "")
        class_name = ".".join(full_name.split(".")[:-1]) if "." in full_name else ""

        return {
            "name": target_method.get("name"),
            "full_name": full_name,
            "class_name": class_name,
            "parameters": target_method.get("parameters", []),
            "return_type": target_method.get("return_type", ""),
            "docstring": target_method.get("docstring", ""),
            "is_static": target_method.get("is_static", False),
            "is_classmethod": target_method.get("is_classmethod", False),
            "is_property": target_method.get("is_property", False),
            "match_score": match_score,
        }

    def get_class_names_list(self) -> List[str]:
        """Get a simple list of all class full names.

        Returns:
            List of class full names sorted alphabetically
        """
        if not self.is_available():
            return []

        if "classes" not in self.atlas_data:
            return []

        class_names = []
        for class_key, class_info in self.atlas_data["classes"].items():
            full_name = class_info.get("full_name", class_key)
            class_names.append(full_name)

        return sorted(class_names)

    def get_module_names_list(self) -> List[str]:
        """Get a simple list of all module full names.

        Returns:
            List of module full names sorted alphabetically
        """
        if not self.is_available():
            return []

        if "modules" not in self.atlas_data:
            return []

        module_names = []
        for module_key, module_info in self.atlas_data["modules"].items():
            name = module_info.get("name", "")
            # Skip __DOC modules
            if name == "__DOC":
                continue
            full_name = module_info.get("full_name", module_key)
            module_names.append(full_name)

        return sorted(module_names)


================================================================================
FILE: src/omni_aiq_omni_ui/services/reranking.py
Size: 4.42 KB | Tokens: 1,023
================================================================================

"""Reranking service for improving search relevance."""

import logging
from typing import Any, Dict, List, Optional

import requests

from ..config import DEFAULT_RERANK_ENDPOINT, DEFAULT_RERANK_MODEL, get_effective_api_key

logger = logging.getLogger(__name__)


class Reranker:
    """Service for reranking search results using NVIDIA reranking models."""

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


def create_reranker(endpoint_url: str = None, api_key: str = None, model: str = None) -> Optional[Reranker]:
    """Create a reranker instance.

    Args:
        endpoint_url: Optional custom endpoint URL
        api_key: Optional API key (uses environment if not provided)
        model: Optional model name

    Returns:
        Reranker instance or None if creation fails
    """
    # Get API key from environment if not provided
    if not api_key:
        api_key = get_effective_api_key("reranking")

    if not api_key:
        logger.warning("No API key available for reranker")
        return None

    # Use defaults if not provided
    endpoint = endpoint_url or DEFAULT_RERANK_ENDPOINT
    model_name = model or DEFAULT_RERANK_MODEL

    try:
        return Reranker(endpoint_url=endpoint, api_key=api_key, model=model_name)
    except Exception as e:
        logger.error(f"Failed to create reranker: {e}")
        return None


def create_reranker_with_config(config: Optional[Dict[str, Any]] = None) -> Optional[Reranker]:
    """Create a reranker instance using provided configuration.

    Args:
        config: Configuration dict with 'model', 'endpoint', and 'api_key'

    Returns:
        Reranker instance or None if creation fails
    """
    if not config:
        return create_reranker()

    api_key = config.get("api_key")
    if not api_key:
        api_key = get_effective_api_key("reranking")

    if not api_key:
        logger.warning("No API key provided for reranker")
        return None

    # Get endpoint and model from config or use defaults
    endpoint = config.get("endpoint")
    model = config.get("model", DEFAULT_RERANK_MODEL)

    # If endpoint is None or empty, use default
    if endpoint is None or endpoint == "":
        endpoint = DEFAULT_RERANK_ENDPOINT

    try:
        reranker = Reranker(endpoint_url=endpoint, api_key=api_key, model=model)
        logger.info(f"Reranker created with model: {model}")
        return reranker
    except Exception as e:
        logger.error(f"Failed to create reranker: {e}")
        return None


================================================================================
FILE: src/omni_aiq_omni_ui/services/retrieval.py
Size: 7.18 KB | Tokens: 1,653
================================================================================

"""Retrieval service for OmniUI code examples using FAISS vector search."""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from ..config import (
    DEFAULT_EMBEDDING_ENDPOINT,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RAG_LENGTH_CODE,
    DEFAULT_RAG_TOP_K_CODE,
    DEFAULT_RERANK_CODE,
)

logger = logging.getLogger(__name__)


def create_embeddings(endpoint_url: str = None, api_key: str = "") -> NVIDIAEmbeddings:
    """Create NVIDIA embeddings.

    Args:
        endpoint_url: Optional custom endpoint URL
        api_key: API key for authentication

    Returns:
        NVIDIAEmbeddings instance
    """
    if endpoint_url:
        return NVIDIAEmbeddings(
            base_url=endpoint_url,
            nvidia_api_key=api_key,
            truncate="END",
        )
    else:
        # Use default NVIDIA API
        return NVIDIAEmbeddings(
            model=DEFAULT_EMBEDDING_MODEL,
            nvidia_api_key=api_key,
            truncate="END",
        )


def create_embeddings_with_config(config: Dict[str, Any]) -> NVIDIAEmbeddings:
    """Create embeddings using configuration.

    Args:
        config: Configuration dict with 'model', 'endpoint', and 'api_key'

    Returns:
        NVIDIAEmbeddings instance
    """
    endpoint = config.get("endpoint")
    api_key = config.get("api_key", "")
    model = config.get("model", DEFAULT_EMBEDDING_MODEL)

    # Debug logging
    logger.info(f"[DEBUG] create_embeddings_with_config called")
    logger.info(f"[DEBUG] API key provided: {'Yes' if api_key else 'No'}")
    logger.info(f"[DEBUG] API key length: {len(api_key) if api_key else 0}")
    logger.info(f"[DEBUG] Endpoint: {endpoint}")
    logger.info(f"[DEBUG] Model: {model}")

    if endpoint:
        return NVIDIAEmbeddings(
            base_url=endpoint,
            nvidia_api_key=api_key,
            truncate="END",
        )
    else:
        # Use default NVIDIA API
        return NVIDIAEmbeddings(
            model=model,
            nvidia_api_key=api_key,
            truncate="END",
        )


class Retriever:
    """Service for retrieving relevant documents from FAISS index."""

    def __init__(
        self,
        endpoint_url: str = None,
        api_key: str = "",
        load_path: str = None,
        top_k: int = 20,
        embedding_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the Retriever.

        Args:
            endpoint_url: Optional custom endpoint URL for embeddings
            api_key: API key for authentication
            load_path: Path to the FAISS index
            top_k: Default number of results to retrieve
            embedding_config: Configuration for embeddings
        """
        if embedding_config:
            self.embedder = create_embeddings_with_config(embedding_config)
        else:
            self.embedder = create_embeddings(endpoint_url, api_key)
        self.top_k = top_k
        self.vectordb = None
        self.retriever = None

        if load_path and os.path.exists(load_path):
            try:
                self.vectordb = FAISS.load_local(
                    load_path,
                    self.embedder,
                    allow_dangerous_deserialization=True,
                )
                self.retriever = self.vectordb.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k},
                )
                logger.info(f"Successfully loaded FAISS index from {load_path}")
            except Exception as e:
                logger.error(f"Failed to load FAISS index from {load_path}: {e}")
                raise
        elif load_path:
            logger.warning(f"FAISS index path does not exist: {load_path}")

    def search(self, query: str, top_k: Optional[int] = None):
        """Search for relevant documents.

        Args:
            query: The search query
            top_k: Number of results to return (uses default if not specified)

        Returns:
            List of relevant documents
        """
        logger.info(f"[DEBUG] Retriever.search called with query: {query}, top_k: {top_k}")

        if not self.retriever:
            logger.error("Retriever not initialized")
            return []

        k = top_k if top_k is not None else self.top_k
        self.retriever.search_kwargs = {"k": k}
        logger.info(f"[DEBUG] Using k={k} for search")

        try:
            return self.retriever.get_relevant_documents(query)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []


def get_rag_context_omni_ui_code(
    user_query: str,
    retriever: Retriever,
    rag_max_size: int = DEFAULT_RAG_LENGTH_CODE,
    rag_top_k: int = DEFAULT_RAG_TOP_K_CODE,
    rerank_k: int = DEFAULT_RERANK_CODE,
    reranker: Optional[Any] = None,
) -> str:
    """Get RAG context for OmniUI code queries.

    Args:
        user_query: The user query
        retriever: The RAG retriever
        rag_max_size: The maximum size of the RAG context
        rag_top_k: The top k RAG context to return
        rerank_k: The top k results to keep after reranking
        reranker: The reranker to use for reranking results

    Returns:
        The RAG context for OmniUI code examples
    """
    if not retriever:
        logger.error("Retriever not provided")
        return ""

    # Perform search
    docs = retriever.search(user_query, top_k=rag_top_k)

    if not docs:
        return ""

    # If reranker is provided, use it to rerank results
    if reranker:
        try:
            # Extract texts for reranking
            texts = [doc.page_content for doc in docs]

            # Rerank documents
            reranked_results = reranker.rerank(user_query, texts, top_k=rerank_k)

            # Reorder docs based on reranking
            reranked_docs = []
            for result in reranked_results:
                idx = result["index"]
                if idx < len(docs):
                    reranked_docs.append(docs[idx])

            docs = reranked_docs[:rerank_k] if reranked_docs else docs[:rerank_k]
            logger.info(f"Reranked {len(docs)} documents")
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            docs = docs[:rerank_k]
    else:
        # If no reranker, just limit to rerank_k results
        docs = docs[:rerank_k]

    # Format the results
    rag_text = ""
    current_size = 0

    for i, doc in enumerate(docs, 1):
        # Extract metadata
        metadata = doc.metadata

        # Format the code example
        file_name = metadata.get("file_name", "unknown")
        file_path = metadata.get("file_path", "unknown")
        method_name = metadata.get("method_name", "unknown")
        source_code = doc.page_content

        # Create formatted output
        example_text = f"""### Example {i}
File: {file_name}
Path: {file_path}
Method: {method_name}

```python
{source_code}
```

"""

        # Check size limit
        if current_size + len(example_text) > rag_max_size:
            break

        rag_text += example_text
        current_size += len(example_text)

    return rag_text.strip()


================================================================================
FILE: src/omni_aiq_omni_ui/services/telemetry.py
Size: 7.29 KB | Tokens: 1,446
================================================================================

"""
Centralized telemetry service for OmniUI MCP using Redis Streams.
"""
import json
import time
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None

logger = logging.getLogger(__name__)

class TelemetryService:
    """
    Centralized telemetry service using Redis Streams.
    
    Captures function calls with timing, parameters, and success status.
    """
    
    _instance = None
    _redis_client = None
    _enabled = True
    
    # Redis configuration
    REDIS_HOST = "omni-chatusd-redis.nvidia.com"
    REDIS_PORT = 6379
    KEY_PREFIX = "omni_ui_mcp:telemetry"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TelemetryService, cls).__new__(cls)
        return cls._instance
    
    async def initialize(self):
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis library not available, telemetry disabled")
            self._enabled = False
            return
            
        if self._redis_client is None:
            try:
                self._redis_client = aioredis.Redis(
                    host=self.REDIS_HOST,
                    port=self.REDIS_PORT,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
                # Test connection
                await self._redis_client.ping()
                logger.info(f"Telemetry service connected to Redis at {self.REDIS_HOST}:{self.REDIS_PORT}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._enabled = False
                self._redis_client = None
    
    async def capture_call(self, 
                          function_name: str, 
                          request_data: Dict[str, Any], 
                          duration_ms: float,
                          success: bool = True,
                          error: Optional[str] = None,
                          session_id: Optional[str] = None) -> bool:
        """
        Capture a function call to Redis as a regular key-value entry.
        
        Args:
            function_name: Name of the function being called
            request_data: Input parameters passed to the function
            duration_ms: Time taken for the function execution in milliseconds
            success: Whether the function call was successful
            error: Error message if the call failed
            session_id: Optional session identifier for grouping calls
            
        Returns:
            bool: True if telemetry was captured successfully, False otherwise
        """
        if not self._enabled or self._redis_client is None:
            return False
            
        try:
            # Generate unique call ID and timestamp
            call_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc)
            
            # Create Redis key with timestamp for easy sorting
            # Format: omni_ui_mcp:telemetry:YYYY-MM-DD:HH-MM-SS-microseconds:call_id
            timestamp_str = timestamp.strftime("%Y-%m-%d:%H-%M-%S") + f"-{timestamp.microsecond:06d}"
            redis_key = f"{self.KEY_PREFIX}:{timestamp_str}:{call_id}"
            
            # Prepare telemetry data
            telemetry_data = {
                "service": "omni_ui_mcp",
                "function_name": function_name,
                "call_id": call_id,
                "timestamp": timestamp.isoformat(),
                "duration_ms": round(duration_ms, 2),
                "success": success,
                "request_data": request_data,
                "session_id": session_id or "unknown"
            }
            
            if error:
                telemetry_data["error"] = error
            
            # Store as JSON in Redis with expiration (optional - can be removed for permanent storage)
            await self._redis_client.set(
                redis_key, 
                json.dumps(telemetry_data, default=str),
                # ex=None  # No expiration for permanent storage
            )
            
            logger.debug(f"Telemetry captured for {function_name}: {redis_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to capture telemetry: {e}")
            return False
    
    @asynccontextmanager
    async def track_call(self, 
                        function_name: str, 
                        request_data: Dict[str, Any],
                        session_id: Optional[str] = None):
        """
        Context manager for tracking function calls with automatic timing.
        
        Usage:
            async with telemetry.track_call("my_function", {"param": "value"}):
                # Your function logic here
                result = await do_something()
        """
        start_time = time.perf_counter()
        success = True
        error = None
        
        try:
            yield
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            await self.capture_call(
                function_name=function_name,
                request_data=request_data,
                duration_ms=duration_ms,
                success=success,
                error=error,
                session_id=session_id
            )
    
    async def get_telemetry_keys_count(self) -> int:
        """Get count of telemetry keys in Redis."""
        if not self._enabled or self._redis_client is None:
            return 0
            
        try:
            pattern = f"{self.KEY_PREFIX}:*"
            keys = await self._redis_client.keys(pattern)
            return len(keys)
        except Exception as e:
            logger.error(f"Failed to get telemetry keys count: {e}")
            return 0

    async def get_recent_telemetry_keys(self, limit: int = 10) -> list:
        """Get the most recent telemetry keys (sorted by timestamp)."""
        if not self._enabled or self._redis_client is None:
            return []
            
        try:
            pattern = f"{self.KEY_PREFIX}:*"
            keys = await self._redis_client.keys(pattern)
            # Keys are naturally sorted by timestamp due to our naming convention
            return sorted(keys, reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Failed to get recent telemetry keys: {e}")
            return []
    
    async def close(self):
        """Close Redis connection."""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None
    
    def is_enabled(self) -> bool:
        """Check if telemetry is enabled."""
        return self._enabled and self._redis_client is not None


# Global telemetry instance
telemetry = TelemetryService()


async def ensure_telemetry_initialized():
    """Ensure telemetry service is initialized."""
    if telemetry._redis_client is None:
        await telemetry.initialize()

================================================================================
FILE: src/omni_aiq_omni_ui/services/test_ui_window_examples.py
Size: 3.15 KB | Tokens: 729
================================================================================

"""Test script for UI window examples FAISS database and retrieval."""

import logging
import os
import sys

# Add the parent directory to the path to allow importing from omni_aiq_omni_ui
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from omni_aiq_omni_ui.services.ui_window_examples_retrieval import (
    create_ui_window_examples_retriever,
    get_ui_window_examples,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ui_window_examples_search():
    """Test the UI window examples search functionality."""
    logger.info("Testing UI Window Examples Search")

    # Get API key from environment
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key:
        logger.warning("No NVIDIA_API_KEY found, search may not work")

    # Create retriever
    try:
        retriever = create_ui_window_examples_retriever(api_key=api_key, top_k=5)
        logger.info("Successfully created UI Window Examples Retriever")
    except Exception as e:
        logger.error(f"Failed to create retriever: {e}")
        return

    # Test queries
    test_queries = [
        "create window with buttons",
        "animation curve simplification",
        "dialog window with text",
        "UI with sliders and controls",
        "build user interface",
    ]

    for query in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing query: '{query}'")
        logger.info("=" * 60)

        try:
            # Test structured results
            structured_results = get_ui_window_examples(query, retriever, top_k=3, format_type="structured")

            logger.info(f"Found {len(structured_results)} structured results")

            for i, result in enumerate(structured_results, 1):
                print(f"\n--- Result {i} ---")
                print(f"Function: {result['class_name']}.{result['function_name']}()")
                print(f"File: {result['file_path']}")
                print(f"Description: {result['description'][:100]}...")
                print(f"Code preview: {result['code'][:150].replace(chr(10), ' ')}...")

            # Test formatted results
            formatted_results = get_ui_window_examples(query, retriever, top_k=2, format_type="formatted")

            logger.info("\nFormatted results preview:")
            print(formatted_results[:500] + "..." if len(formatted_results) > 500 else formatted_results)

        except Exception as e:
            logger.error(f"Error testing query '{query}': {e}")

        print("\n" + "=" * 60 + "\n")


def main():
    """Main test function."""
    # Check if FAISS database exists
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    faiss_path = os.path.join(data_dir, "ui_window_examples_faiss")

    if not os.path.exists(faiss_path):
        logger.error(f"FAISS database not found at: {faiss_path}")
        logger.info("Please run build_faiss_database.py first to create the database")
        return

    logger.info(f"Found FAISS database at: {faiss_path}")

    # Run tests
    test_ui_window_examples_search()


if __name__ == "__main__":
    main()


================================================================================
FILE: src/omni_aiq_omni_ui/services/ui_window_examples_retrieval.py
Size: 8.05 KB | Tokens: 1,767
================================================================================

"""Retrieval service for UI window examples using FAISS vector search."""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from .retrieval import create_embeddings, create_embeddings_with_config

logger = logging.getLogger(__name__)


class UIWindowExamplesRetriever:
    """Service for retrieving UI window examples from FAISS index."""

    def __init__(
        self,
        endpoint_url: str = None,
        api_key: str = "",
        faiss_index_path: str = None,
        top_k: int = 10,
        embedding_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the UI Window Examples Retriever.

        Args:
            endpoint_url: Optional custom endpoint URL for embeddings
            api_key: API key for authentication
            faiss_index_path: Path to the FAISS index
            top_k: Default number of results to retrieve
            embedding_config: Configuration for embeddings
        """
        if embedding_config:
            self.embedder = create_embeddings_with_config(embedding_config)
        else:
            self.embedder = create_embeddings(endpoint_url, api_key)

        self.top_k = top_k
        self.vectordb = None
        self.retriever = None

        if faiss_index_path and os.path.exists(faiss_index_path):
            try:
                self.vectordb = FAISS.load_local(
                    faiss_index_path,
                    self.embedder,
                    allow_dangerous_deserialization=True,
                )
                self.retriever = self.vectordb.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k},
                )
                logger.info(f"Successfully loaded UI window examples FAISS index from {faiss_index_path}")
            except Exception as e:
                logger.error(f"Failed to load FAISS index from {faiss_index_path}: {e}")
                raise
        elif faiss_index_path:
            logger.warning(f"FAISS index path does not exist: {faiss_index_path}")

    def search(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Search for relevant UI window examples.

        Args:
            query: The search query
            top_k: Number of results to return (uses default if not specified)

        Returns:
            List of relevant documents
        """
        logger.info(f"[DEBUG] UIWindowExamplesRetriever.search called with query: {query}, top_k: {top_k}")

        if not self.retriever:
            logger.error("UI Window Examples Retriever not initialized")
            return []

        k = top_k if top_k is not None else self.top_k
        self.retriever.search_kwargs = {"k": k}
        logger.info(f"[DEBUG] Using k={k} for UI window examples search")

        try:
            return self.retriever.get_relevant_documents(query)
        except Exception as e:
            logger.error(f"UI window examples search failed: {e}")
            return []

    def get_structured_results(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get structured results for UI window examples search.

        Args:
            query: The search query
            top_k: Number of results to return

        Returns:
            List of structured results with description, code, and file_path
        """
        docs = self.search(query, top_k)

        structured_results = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata

            result = {
                "rank": i,
                "description": metadata.get("description", ""),
                "code": metadata.get("code", ""),
                "file_path": metadata.get("file_path", ""),
                "function_name": metadata.get("function_name", "unknown"),
                "class_name": metadata.get("class_name", "unknown"),
                "line_number": metadata.get("line_number", 0),
                "similarity_score": getattr(doc, "similarity_score", None),  # If available from search
            }

            structured_results.append(result)

        return structured_results

    def get_formatted_results(
        self, query: str, top_k: Optional[int] = None, max_description_length: int = 200, max_code_length: int = 1000
    ) -> str:
        """Get formatted string results for UI window examples search.

        Args:
            query: The search query
            top_k: Number of results to return
            max_description_length: Maximum length for description preview
            max_code_length: Maximum length for code preview

        Returns:
            Formatted string with search results
        """
        results = self.get_structured_results(query, top_k)

        if not results:
            return "No UI window examples found for your query."

        formatted_output = f"**UI Window Examples Search Results for:** '{query}'\n"
        formatted_output += f"**Found {len(results)} matches:**\n\n"

        for result in results:
            description = result["description"]
            code = result["code"]

            # Truncate long descriptions and code
            if len(description) > max_description_length:
                description = description[:max_description_length] + "..."

            if len(code) > max_code_length:
                code = code[:max_code_length] + "\n... [truncated]"

            formatted_output += f"### Match {result['rank']}\n"
            formatted_output += f"**File:** `{result['file_path']}`\n"
            formatted_output += (
                f"**Function:** `{result['class_name']}.{result['function_name']}()` (Line {result['line_number']})\n\n"
            )
            formatted_output += f"**Description:**\n{description}\n\n"
            formatted_output += f"**Code:**\n```python\n{code}\n```\n\n"
            formatted_output += "---\n\n"

        return formatted_output.strip()


def get_ui_window_examples(
    user_query: str,
    retriever: UIWindowExamplesRetriever,
    top_k: int = 5,
    format_type: str = "structured",  # "structured", "formatted", "raw"
) -> Any:
    """Get UI window examples for a user query.

    Args:
        user_query: The user query
        retriever: The UI window examples retriever
        top_k: Number of results to return
        format_type: Type of formatting for results ("structured", "formatted", "raw")

    Returns:
        UI window examples in requested format
    """
    if not retriever:
        logger.error("UI Window Examples Retriever not provided")
        return [] if format_type == "structured" else ""

    try:
        if format_type == "structured":
            return retriever.get_structured_results(user_query, top_k)
        elif format_type == "formatted":
            return retriever.get_formatted_results(user_query, top_k)
        elif format_type == "raw":
            return retriever.search(user_query, top_k)
        else:
            logger.warning(f"Unknown format_type: {format_type}, defaulting to structured")
            return retriever.get_structured_results(user_query, top_k)

    except Exception as e:
        logger.error(f"Failed to get UI window examples: {e}")
        return [] if format_type in ["structured", "raw"] else "Error retrieving UI window examples."


def create_ui_window_examples_retriever(
    faiss_index_path: str = None, api_key: str = "", top_k: int = 10
) -> UIWindowExamplesRetriever:
    """Create a UI Window Examples Retriever with default paths.

    Args:
        faiss_index_path: Path to FAISS index (uses default if None)
        api_key: API key for embeddings
        top_k: Default number of results

    Returns:
        Configured UIWindowExamplesRetriever instance
    """
    if faiss_index_path is None:
        # Default path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), "data")
        faiss_index_path = os.path.join(data_dir, "ui_window_examples_faiss")

    return UIWindowExamplesRetriever(api_key=api_key, faiss_index_path=faiss_index_path, top_k=top_k)


================================================================================
FILE: src/omni_aiq_omni_ui/utils/__init__.py
Size: 41.00 B | Tokens: 8
================================================================================

"""Utilities module for OmniUI tools."""


================================================================================
FILE: src/omni_aiq_omni_ui/utils/fuzzy_matching.py
Size: 3.39 KB | Tokens: 770
================================================================================

"""Fuzzy matching utilities for finding best matches in OmniUI Atlas data."""

from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple


def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity score between two strings.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Similarity score between 0 and 1
    """
    # Convert to lowercase for case-insensitive comparison
    str1_lower = str1.lower()
    str2_lower = str2.lower()

    # Exact match (case-insensitive)
    if str1_lower == str2_lower:
        return 1.0

    # Check if one is contained in the other
    if str1_lower in str2_lower or str2_lower in str1_lower:
        # Give higher score for containment
        return 0.9 * min(len(str1_lower), len(str2_lower)) / max(len(str1_lower), len(str2_lower))

    # Use SequenceMatcher for fuzzy matching
    return SequenceMatcher(None, str1_lower, str2_lower).ratio()


def find_best_matches(
    query: str,
    candidates: Dict[str, Any],
    key_func: Callable[[str], str],
    value_func: Callable[[str], Any],
    name_func: Callable[[Any], str],
    threshold: float = 0.5,
    max_results: int = 5,
) -> List[Tuple[str, Any, float]]:
    """Find best matches for a query in a dictionary of candidates.

    Args:
        query: Search query string
        candidates: Dictionary of candidates to search
        key_func: Function to extract key from candidate
        value_func: Function to extract value from candidate
        name_func: Function to extract name from value for comparison
        threshold: Minimum similarity score to include in results
        max_results: Maximum number of results to return

    Returns:
        List of tuples containing (key, value, similarity_score)
    """
    matches = []

    for key in candidates:
        value = value_func(key)
        name = name_func(value)

        # Calculate similarity with the full name
        similarity = calculate_similarity(query, name)

        # Also check against just the class/module name (last part)
        if "." in name:
            short_name = name.split(".")[-1]
            short_similarity = calculate_similarity(query, short_name)
            similarity = max(similarity, short_similarity)

        # Also check against the key itself
        key_similarity = calculate_similarity(query, key)
        similarity = max(similarity, key_similarity)

        if similarity >= threshold:
            matches.append((key, value, similarity))

    # Sort by similarity score (descending)
    matches.sort(key=lambda x: x[2], reverse=True)

    # Return top results
    return matches[:max_results]


def find_best_match(query: str, candidates: List[str], threshold: float = 0.5) -> Optional[Tuple[str, float]]:
    """Find the best match for a query in a list of candidates.

    Args:
        query: Search query string
        candidates: List of candidate strings
        threshold: Minimum similarity score to consider a match

    Returns:
        Tuple of (best_match, similarity_score) or None if no match found
    """
    best_match = None
    best_score = 0.0

    for candidate in candidates:
        score = calculate_similarity(query, candidate)
        if score > best_score and score >= threshold:
            best_match = candidate
            best_score = score

    if best_match:
        return (best_match, best_score)
    return None


================================================================================
FILE: src/omni_aiq_omni_ui/utils/usage_logging.py
Size: 2.18 KB | Tokens: 447
================================================================================

"""Usage logging utilities for OmniUI tools."""

import json
import logging
import time
from functools import wraps
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global usage logger instance
_usage_logger: Optional["UsageLogger"] = None


class UsageLogger:
    """Simple usage logger for tracking tool usage."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.logger = logging.getLogger(f"{__name__}.usage")

        if enabled:
            # Configure logging format for usage tracking
            formatter = logging.Formatter("%(asctime)s - USAGE - %(levelname)s - %(message)s")

            # Create console handler if not exists
            if not self.logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
                self.logger.setLevel(logging.INFO)

    def log_tool_call(
        self,
        tool_name: str,
        parameters: Dict[str, Any] = None,
        success: bool = True,
        error_msg: str = None,
        execution_time: float = None,
    ):
        """Log a tool call with parameters and results."""
        if not self.enabled:
            return

        log_entry = {
            "timestamp": time.time(),
            "tool_name": tool_name,
            "parameters": parameters or {},
            "success": success,
            "execution_time_ms": execution_time * 1000 if execution_time else None,
        }

        if error_msg:
            log_entry["error"] = error_msg

        try:
            self.logger.info(json.dumps(log_entry))
        except Exception as e:
            # Don't let logging failures break the main functionality
            logger.warning(f"Failed to log usage data: {e}")


def create_usage_logger(enabled: bool = True) -> UsageLogger:
    """Create or get the global usage logger instance."""
    global _usage_logger
    if _usage_logger is None:
        _usage_logger = UsageLogger(enabled=enabled)
    return _usage_logger


def get_usage_logger() -> Optional[UsageLogger]:
    """Get the current usage logger instance."""
    return _usage_logger


================================================================================
FILE: src/omni_aiq_omni_ui/utils/usage_logging_decorator.py
Size: 4.38 KB | Tokens: 801
================================================================================

"""Usage logging decorator for OmniUI tools."""

import json
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict

from .usage_logging import get_usage_logger

logger = logging.getLogger(__name__)


def log_tool_usage(tool_name: str):
    """Decorator to log tool usage with timing and error handling.

    Args:
        tool_name: Name of the tool being logged

    Returns:
        Decorated function that logs usage information
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            usage_logger = get_usage_logger()

            if not usage_logger or not usage_logger.enabled:
                # If logging is disabled, just execute the function
                return await func(*args, **kwargs)

            start_time = time.time()
            parameters = {}
            error_msg = None
            success = True

            try:
                # Extract parameters from function arguments
                # For most AIQ functions, the first arg is the input model
                if args and hasattr(args[0], "__dict__"):
                    try:
                        # Convert pydantic model to dict for logging
                        parameters = args[0].dict() if hasattr(args[0], "dict") else {}
                    except Exception:
                        # If conversion fails, just use string representation
                        parameters = {"input": str(args[0])}

                # Execute the original function
                result = await func(*args, **kwargs)

                return result

            except Exception as e:
                success = False
                error_msg = str(e)
                logger.error(f"Error in {tool_name}: {error_msg}")
                raise

            finally:
                # Log the usage regardless of success or failure
                execution_time = time.time() - start_time

                try:
                    usage_logger.log_tool_call(
                        tool_name=tool_name,
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    # Don't let logging errors break the main functionality
                    logger.warning(f"Failed to log usage for {tool_name}: {log_error}")

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            usage_logger = get_usage_logger()

            if not usage_logger or not usage_logger.enabled:
                # If logging is disabled, just execute the function
                return func(*args, **kwargs)

            start_time = time.time()
            parameters = {}
            error_msg = None
            success = True

            try:
                # Extract parameters from function arguments
                if args and hasattr(args[0], "__dict__"):
                    try:
                        parameters = args[0].dict() if hasattr(args[0], "dict") else {}
                    except Exception:
                        parameters = {"input": str(args[0])}

                # Execute the original function
                result = func(*args, **kwargs)

                return result

            except Exception as e:
                success = False
                error_msg = str(e)
                logger.error(f"Error in {tool_name}: {error_msg}")
                raise

            finally:
                # Log the usage regardless of success or failure
                execution_time = time.time() - start_time

                try:
                    usage_logger.log_tool_call(
                        tool_name=tool_name,
                        parameters=parameters,
                        success=success,
                        error_msg=error_msg,
                        execution_time=execution_time,
                    )
                except Exception as log_error:
                    logger.warning(f"Failed to log usage for {tool_name}: {log_error}")

        # Return the appropriate wrapper based on whether the function is async
        if hasattr(func, "_is_coroutine") or func.__name__.startswith("async_"):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

