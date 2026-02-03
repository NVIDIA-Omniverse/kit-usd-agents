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

"""
Centralized OmniUI Atlas service singleton.

This module provides a single global instance of the OmniUI Atlas service
to avoid duplicate instances across different function modules.
"""

from ..services.omni_ui_atlas import OmniUIAtlasService

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
