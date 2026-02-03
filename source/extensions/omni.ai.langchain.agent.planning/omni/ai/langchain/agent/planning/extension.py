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

import omni.ext
from lc_agent import get_node_factory

from .nodes import ChatUSDPlanningNetworkNode, ChatUSDPlanningSupervisorNode, PlanningGenNode, PlanningNetworkNode


class PlanningExtension(omni.ext.IExt):
    """
    Extension for Scene Planning Agent.

    This extension provides detailed planning capabilities for USD scenes,
    allowing users to:
    - Create comprehensive plans for scene creation and modification
    - Specify all objects, properties, and steps in detail
    - Validate execution against the original plan

    The extension integrates with Chat USD to provide a comprehensive
    USD development experience with planning as the central component.
    """

    def on_startup(self, ext_id):
        """
        Called when the extension is started.

        Args:
            ext_id: The extension id
        """
        self._ext_id = ext_id
        self._register_planning_agent()

    def on_shutdown(self):
        """Called when the extension is shut down."""
        self._unregister_planning_agent()

    def _register_planning_agent(self):
        """Register the planning agent components."""
        get_node_factory().register(PlanningGenNode, name="PlanningGenNode", hidden=True)
        get_node_factory().register(PlanningNetworkNode, name="ChatUSD_Planning", hidden=True)
        get_node_factory().register(ChatUSDPlanningSupervisorNode, name="ChatUSDPlanningSupervisorNode", hidden=True)
        get_node_factory().register(ChatUSDPlanningNetworkNode, name="ChatUSD with planning")

    def _unregister_planning_agent(self):
        """Unregister the planning agent components."""
        get_node_factory().unregister(PlanningGenNode)
        get_node_factory().unregister(PlanningNetworkNode)
        get_node_factory().unregister(ChatUSDPlanningSupervisorNode)
        get_node_factory().unregister(ChatUSDPlanningNetworkNode)
