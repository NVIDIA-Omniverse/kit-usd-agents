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


def _ignore_verified_embedding_model_warning():
    """Silence the endpoint client's stale model-type metadata warning."""
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=(
            r"Found nvidia/nemotron-3-embed-1b in available_models, but type is " r"unknown and inference may fail\."
        ),
        category=UserWarning,
        module=r"langchain_nvidia_ai_endpoints\._common",
    )


# omni.kit.pip_archive loads before this extension and registers stale copies
# of some packages (typing_extensions 4.12.2, websockets 12.0) that shadow our
# pip_core_prebundle. The langchain stack needs newer ones: langchain-core>=1.3.2
# imports langchain_protocol, whose PEP 728 TypedDict(extra_items=...) requires
# typing_extensions>=4.13, and langgraph imports websockets.asyncio, which only
# exists in websockets>=13. Kit's fast importer resolves every submodule from
# its own flat index (ignoring the parent package's __path__), so neither
# sys.path order nor a sys.modules swap is enough — imports can even end up
# mixing files from both locations. Install a meta-path finder ahead of it that
# serves these modules and all their submodules from pip_core_prebundle, and
# purge stale copies that were already imported.
def _prefer_prebundled_modules():
    import importlib.machinery
    import os
    import sys

    try:
        ext_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        prebundle = os.path.join(ext_root, "pip_core_prebundle")
        if not os.path.isdir(prebundle):
            return

        # typing_extensions: PEP 728 TypedDict(extra_items=...) needs >=4.13
        # websockets: websockets.asyncio needs >=13
        # pydantic stack: 2.11.9 fails on langchain_core ToolCall (internal issue 6585942).
        names = (
            "typing_extensions",
            "websockets",
            "pydantic",
            "pydantic_core",
            "typing_inspection",
            "annotated_types",
        )

        class _PrebundlePreferredFinder:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] not in names:
                    return None
                parts = fullname.split(".")
                location = os.path.join(prebundle, *parts[:-1])
                return importlib.machinery.PathFinder.find_spec(fullname, [location])

        sys.meta_path.insert(0, _PrebundlePreferredFinder())

        # Drop copies (and their submodules) that were already imported from
        # elsewhere so re-imports resolve through the finder above.
        for name in names:
            module = sys.modules.get(name)
            if module is None:
                continue
            origin = getattr(module, "__file__", None)
            if origin is not None and os.path.abspath(origin).startswith(prebundle + os.sep):
                continue
            for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
                del sys.modules[key]
    except Exception as exc:  # never block extension load on the shim
        import carb

        carb.log_warn(f"pip_core_prebundle compatibility shim failed: {exc}")


_ignore_verified_embedding_model_warning()
del _ignore_verified_embedding_model_warning
_prefer_prebundled_modules()
del _prefer_prebundled_modules
