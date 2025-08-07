## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

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
