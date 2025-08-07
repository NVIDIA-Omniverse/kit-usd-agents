## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

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
