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

import omni.usd
from pxr import Sdf

from .double_run_utils import merge_layer_content


class EditTargetManager:
    """
    Manages the edit target of a USD stage, including setting up and restoring
    the edit target, and merging layers into the root layer.
    """

    def __init__(self):
        self._stage = omni.usd.get_context().get_stage()
        self._original_edit_target = self._stage.GetEditTarget()
        self._new_layer = None

    def set_edit_target(self):
        """
        Sets up a new edit target layer and sets it as the current edit target.
        """
        self._new_layer = Sdf.Layer.CreateAnonymous()
        session_layer = self._stage.GetSessionLayer()
        session_layer.subLayerPaths.insert(0, self._new_layer.identifier)

        edit_target = self._stage.GetEditTargetForLocalLayer(self._new_layer)
        self._stage.SetEditTarget(edit_target)

    def restore_edit_target(self):
        """
        Restores the original edit target.
        """
        self._stage.SetEditTarget(self._original_edit_target)

    def merge_to_root_layer(self):
        """
        Merges the new layer into the root layer and removes it from the session layer.
        """
        if self._new_layer and not self._new_layer.empty:
            root_layer = self._stage.GetRootLayer()
            merge_layer_content(self._new_layer, root_layer)

            # Remove the new layer from the session layer
            session_layer = self._stage.GetSessionLayer()
            if self._new_layer.identifier in session_layer.subLayerPaths:
                session_layer.subLayerPaths.remove(self._new_layer.identifier)
