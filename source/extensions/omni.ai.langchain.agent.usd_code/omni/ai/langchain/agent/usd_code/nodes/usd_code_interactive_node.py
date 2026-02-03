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
from lc_agent_usd import USDCodeInteractiveNode as USDCodeInteractiveNodeBase
from pxr import UsdGeom

from ..utils.chat_model_utils import sanitize_messages_with_expert_type


class USDCodeInteractiveNode(USDCodeInteractiveNodeBase):
    def __init__(self, **kwargs):
        # We need to dynamically replace all the "{default_prim}" with the real default prim path
        if "system_message" not in kwargs:
            usd_context = omni.usd.get_context()
            selection = usd_context.get_selection().get_selected_prim_paths()
            selection = f"{selection}" if selection else "no"
            stage = usd_context.get_stage()
            default_prim_path = "/World"
            up_axis = "Y"
            if stage:
                default_prim = stage.GetDefaultPrim()
                if default_prim:
                    default_prim_path = default_prim.GetPath().pathString
                up_axis = UsdGeom.GetStageUpAxis(stage)

            super().__init__(default_prim_path=default_prim_path, up_axis=up_axis, selection=selection, **kwargs)
        else:
            super().__init__(**kwargs)

    def _sanitize_messages_for_chat_model(self, messages, chat_model_name, chat_model):
        """Sanitizes messages and adds metafunction expert type for USD operations."""
        messages = super()._sanitize_messages_for_chat_model(messages, chat_model_name, chat_model)
        return sanitize_messages_with_expert_type(messages, "metafunction")
