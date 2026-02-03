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

import omni.ui as ui
from langchain_core.messages import AIMessage
from omni.ai.langchain.widget.core.agent_delegate import DefaultDelegate
from omni.ui import color as cl

STYLE = {
    "Label.Tool": {
        "color": cl.input_border_color,
        "font_size": 14,
        "margin": 5,
    }
}


class SupervisorNodeDelegate(DefaultDelegate):
    """Minimal delegate for the Supervisor node."""

    def build_agent_widget(self, network, node):
        if isinstance(node.outputs, AIMessage):
            if node.outputs.content:
                # There is a text response, so use the default delegate
                return super().build_agent_widget(network, node)

            # It's a tool call
            with ui.ZStack(style=STYLE):
                # Background
                ui.Rectangle(style_type_name_override="Rectangle.Bot.ChatGPT")

                tools = [tool_call["name"] for tool_call in node.outputs.tool_calls]
                if tools:
                    text = "Tool: " + ", ".join(tools)
                else:
                    text = "Choosing a Tool..."

                with ui.HStack():
                    ui.Spacer(width=75)
                    ui.Label(text, style_type_name_override="Label.Tool")


class ToolNodeDelegate(DefaultDelegate):
    """Minimal delegate for the Tool node."""

    def build_agent_widget(self, network, node):
        ui.Line(height=2, tooltip="Tool Invoked")
