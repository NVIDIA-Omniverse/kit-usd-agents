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

import omni.kit.commands
import omni.usd
from pxr import Sdf

from .double_run_utils import merge_layer_content


class DoubleRunUSDCodeGenCommand(omni.kit.commands.Command):
    """
    Command to merge layers into the root layer upon saving the stage.
    """

    def __init__(self, layer):
        self._layer = layer
        self._done = False

        # Listen to USD stage activity
        stage_event_stream = omni.usd.get_context().get_stage_event_stream()
        self._stage_event_sub = stage_event_stream.create_subscription_to_pop(
            self._on_stage_event, name="DoubleRunUSDCodeGenCommand"
        )

    def _on_stage_event(self, event):
        if self._done:
            return

        SAVING = int(omni.usd.StageEventType.SAVING)
        if event.type == SAVING:
            self._done = True

            print("Saving stage. Merging DoubleRunUSDCodeGenCommand layers.")
            stage = omni.usd.get_context().get_stage()
            session_layer = stage.GetSessionLayer()
            root_layer = stage.GetRootLayer()
            layers_to_merge = []

            for layer_identifier in reversed(session_layer.subLayerPaths):
                current_layer = Sdf.Layer.Find(layer_identifier)
                if "DoubleRunUSDCodeGenCommand" in current_layer.customLayerData:
                    layers_to_merge.append((layer_identifier, current_layer))

            for layer_identifier, current_layer in layers_to_merge:
                print("Merging", layer_identifier)
                merge_layer_content(current_layer, root_layer)
                session_layer.subLayerPaths.remove(layer_identifier)

    def do(self):
        stage = omni.usd.get_context().get_stage()
        session_layer = stage.GetSessionLayer()
        layer_identifier = self._layer.identifier

        # Mark it as a layer that needs to be saved
        custom_layer_data = self._layer.customLayerData
        custom_layer_data["DoubleRunUSDCodeGenCommand"] = "yes"
        self._layer.customLayerData = custom_layer_data
        print("Marked layer for merging:", layer_identifier)

        if layer_identifier not in session_layer.subLayerPaths:
            session_layer.subLayerPaths.insert(0, layer_identifier)

    def undo(self):
        stage = omni.usd.get_context().get_stage()
        session_layer = stage.GetSessionLayer()
        layer_identifier = self._layer.identifier
        if layer_identifier in session_layer.subLayerPaths:
            session_layer.subLayerPaths.remove(layer_identifier)
