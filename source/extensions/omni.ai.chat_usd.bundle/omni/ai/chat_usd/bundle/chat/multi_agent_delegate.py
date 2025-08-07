## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

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
