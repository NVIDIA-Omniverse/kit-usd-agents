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

import re
from typing import Any, Dict

from langchain_core.messages import AIMessage
from lc_agent import NetworkModifier


class PlanningModifier(NetworkModifier):
    """
    Modifier that processes and enhances planning responses.

    This modifier:
    1. Captures generated plans
    2. Validates plan completeness
    3. Reformats plans for better readability
    4. Tracks plan execution status
    """

    def __init__(self):
        super().__init__()
        self.current_plan = None
        self.plan_status = {}

    async def on_post_invoke_async(self, network, node):
        """
        Post-invoke hook that processes plans from the PlanningGenNode.

        Args:
            network: The current network being executed
            node: The node that was just invoked
        """
        if (
            node.invoked
            and isinstance(node.outputs, AIMessage)
            and node.outputs.content
            and not network.get_children(node)
        ):
            # Extract plan from the node output
            plan_content = node.outputs.content.strip()
            if self._is_valid_plan(plan_content):
                # Store the plan in node metadata for later reference
                node.metadata["plan"] = self._extract_plan(plan_content)
                node.metadata["plan_complete"] = True

                # Also store it in the network metadata so other agents can access it
                network.metadata["current_plan"] = node.metadata["plan"]

                # Set this as the current plan
                self.current_plan = node.metadata["plan"]
                self.plan_status = {step: "pending" for step in node.metadata["plan"]["steps"].keys()}

                # No need to create a new node, let the plan be returned directly

    def _is_valid_plan(self, content: str) -> bool:
        """
        Check if the content contains a valid plan.

        Args:
            content: The content to check

        Returns:
            bool: True if content contains a valid plan
        """
        # Check if content contains a plan header and at least one step
        plan_header = re.search(r"PLAN:\s*(.+?)(?:\n|$)", content)
        steps = re.findall(r"Step \d+:", content)
        return plan_header is not None and len(steps) > 0

    def _extract_plan(self, content: str) -> Dict[str, Any]:
        """
        Extract and structure the plan from the content.

        Args:
            content: The content containing the plan

        Returns:
            dict: Structured plan with title and steps
        """
        # Extract plan title
        plan_title_match = re.search(r"PLAN:\s*(.+?)(?:\n|$)", content)
        plan_title = plan_title_match.group(1).strip() if plan_title_match else "Untitled Plan"

        # Extract steps
        steps = {}
        step_matches = re.finditer(r"Step (\d+):\s*(.+?)(?=\nStep \d+:|$)", content, re.DOTALL)

        for match in step_matches:
            step_number = match.group(1)
            step_content = match.group(2).strip()

            # Extract details from bullet points
            details = []
            for line in step_content.split("\n"):
                if line.strip().startswith("-"):
                    details.append(line.strip()[1:].strip())

            steps[f"step_{step_number}"] = {"title": step_content.split("\n")[0].strip(), "details": details}

        return {"title": plan_title, "steps": steps}
