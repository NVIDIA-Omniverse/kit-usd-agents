## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import asyncio
import copy
import json
from typing import Any, List, Tuple

import omni.kit.app
import omni.ui as ui
from langchain_core.messages import BaseMessage
from lc_agent import (
    FromRunnableNode,
    NetworkNode,
    RunnableAINode,
    RunnableHumanNode,
    RunnableNetwork,
    RunnableNode,
    USDAssistantNode,
)
from omni.kit.widget.graph.graph_model import GraphModel
from omni.kit.widget.graph.graph_node_delegate import GraphNodeDelegate
from omni.kit.widget.graph.graph_view import GraphConnectionDescription, GraphView

# The main colors
LABEL_COLOR = 0xFFB4B4B4
BACKGROUND_COLOR = 0xFF34302A
BORDER_DEFAULT = 0xFFDBA656
BORDER_SELECTED = 0xFFFFFFFF
CONNECTION = 0xFF80C280
NODE_BACKGROUND = 0xFF675853
NODE_BACKGROUND_SELECTED = 0xFF7F6C66
ICON_PATH = "C:/dev/kit-ai-agent/_build/windows-x86_64/release/kit/exts/omni.kit.widget.graph/icons"

BORDER_WIDTH = 16
CONNECTION_CURVE = 60


def color_to_hex(color: tuple) -> int:
    """Convert float rgb to int"""

    def to_int(f: float) -> int:
        return int(255 * max(0.0, min(1.0, f)))

    red = to_int(color[0])
    green = to_int(color[1])
    blue = to_int(color[2])
    alpha = to_int(color[3]) if len(color) > 3 else 255
    return (alpha << 8 * 3) + (blue << 8 * 2) + (green << 8 * 1) + red


def split_string_to_max_length(text, max_len):
    wrapped_lines = []
    lines = text.split("\n")

    for line in lines:
        if len(line) <= max_len:
            wrapped_lines.append(line)
            continue

        # Split existing line into words to wrap
        words = line.split()
        current_line = ""

        for word in words:
            # Check if adding the new word exceeds the max length
            if len(current_line) + len(word) + 1 > max_len:
                # Append the current line and start a new one
                wrapped_lines.append(current_line)
                current_line = word
            else:
                # Add the word to the current line
                current_line += " " + word if current_line else word

        # Don't forget to add the last line if not empty
        if current_line:
            wrapped_lines.append(current_line)

    # Join the wrapped lines with newlines
    return "\n".join(wrapped_lines)


def trim_long_strings(data, max_length=120):
    """Recursively trim long strings in a dictionary to a specified maximum length."""
    if isinstance(data, dict):
        return {key: trim_long_strings(value, max_length) for key, value in data.items()}
    elif isinstance(data, list):
        return [trim_long_strings(item, max_length) for item in data]
    elif isinstance(data, str) and len(data) > max_length:
        return data[:max_length] + "..."
    else:
        return data


class RunnableNetworkDelegateClassic(GraphNodeDelegate):
    @staticmethod
    def get_style():
        """Return style that can be used with this delegate"""
        style = GraphNodeDelegate.get_style().copy()

        additional = {"Tooltip": {"font_size": 12.0}}

        style.update(additional)

        return style

    def get_output_tooltip(self, node: RunnableNode):
        """Generate tooltip from agent outputs"""
        if hasattr(node, "outputs"):
            if isinstance(node.outputs, BaseMessage) and node.outputs.content:
                tooltip = f"{node.outputs.content}"
            elif isinstance(node.outputs, BaseMessage):
                tooltip = json.dumps(json.loads(node.outputs.json()), indent=2)
            else:
                tooltip = f"{node.outputs}"
            # Split each line to maximum 130 characters
            tooltip = split_string_to_max_length(tooltip, 130)
            return tooltip
        return ""

    def node_header(self, model, node_desc):
        """Called to create widgets of the node background"""
        agent = node_desc.node

        # Create a deep copy of the metadata to avoid modifying the original
        metadata = copy.deepcopy(agent.metadata)

        # Body tooltip is metadata
        exclude = {"description", "examples", "messages", "icon_color", "icon_url"}
        filtered_metadata = {key: metadata[key] for key in sorted(metadata.keys()) if key not in exclude}

        # Trim long strings in the copied metadata
        trimmed_metadata = trim_long_strings(filtered_metadata, max_length=120)

        # Create the tooltip with the trimmed metadata
        tooltip = json.dumps(trimmed_metadata, indent=2)

        with ui.Frame(tooltip=tooltip):
            super().node_header(model, node_desc)

    def node_header_output(self, model, node_desc):
        """Called to create widgets of the port output"""
        # Get the agent and generate output tooltip
        runnable_node = node_desc.node
        tooltip = self.get_output_tooltip(runnable_node)

        with ui.Frame(width=0, tooltip=tooltip):
            result = super().node_header_output(model, node_desc)

        return result

    def connection(self, model, source: GraphConnectionDescription, target: GraphConnectionDescription):
        """Called to create the connection between ports"""
        port_type = str(model[source.port].type)

        # Connection tooltip is output
        runnable_node = source.node
        tooltip = self.get_output_tooltip(runnable_node)

        if target.is_tangent_reversed != source.is_tangent_reversed:
            # It's the same node connection. Set tangent in pixels.
            start_tangent_width = ui.Pixel(20)
            end_tangent_width = ui.Pixel(20)
        else:
            # If the connection is reversed, we need to mirror tangents
            source_reverced_tangent = -1.0 if target.is_tangent_reversed else 1.0
            target_reverced_gangent = -1.0 if source.is_tangent_reversed else 1.0
            start_tangent_width = ui.Percent(-CONNECTION_CURVE * source_reverced_tangent)
            end_tangent_width = ui.Percent(CONNECTION_CURVE * target_reverced_gangent)

        ui.FreeBezierCurve(
            target.widget,
            source.widget,
            tooltip=tooltip,
            start_tangent_width=start_tangent_width,
            end_tangent_width=end_tangent_width,
            name=port_type,
            style_type_name_override="Graph.Connection",
        )


class RunnableNetworkGraphModel(GraphModel):
    """Graph model for visualizing an agent network."""

    def __init__(self, agent_network: "RunnableNetwork"):
        super().__init__()
        self.agent_network = agent_network
        self.build_graph_model()
        self.positions = {}
        self._selected_nodes = []

    def build_graph_model(self):
        """Build the internal graph model using the agent network."""

        # when we have too many nodes under one node, we experience performance issues when operating on the node.
        # so we only pass the current node as the node in the model.
        def create_slimmed_agent(agent):
            as_dict = agent.model_dump()
            slimmed_agent = RunnableNode.model_validate(as_dict)
            # slimmed_agent = agent.copy(deep=True)
            if hasattr(agent, "nodes"):
                slimmed_agent.nodes = []
            return slimmed_agent

        self._ids = []
        self._nodes = []
        self._full_nodes = []
        for agent in self.agent_network.nodes:
            self._ids.append(self.agent_network._get_node_id(agent))
            self._nodes.append(create_slimmed_agent(agent))
            self._full_nodes.append(agent)

    def _get_agent_from_item(self, item) -> Any:
        """Retrieve the agent based on the item, which could be a port or node ID."""
        if isinstance(item, str) and ("/input" in item or "/output" in item):
            # If item is a port representation, extract the agent ID and return the agent
            agent_id = int(item.split("/")[0])
            return self._full_nodes[agent_id]
        return None

    @property
    def nodes(self, item=None) -> List[Any]:
        """Return all nodes (agents) in the network."""
        return self._nodes

    @property
    def name(self, item) -> str:
        """Return the agent's ID as name."""
        if isinstance(item, RunnableNode):
            return item.name or str(type(item).__name__)
        port_type = item.split("/")[-1]
        if port_type == "input":
            return "parents"
        if port_type == "output":
            return ""

    @property
    def position(self, item=None) -> Tuple[float, float]:
        """Position agents onto the graph. Implement actual position logic as needed."""
        return (0.0, 0.0)  # Replace with actual layout logic

    @property
    def ports(self, item=None) -> List[str]:
        """Return ports for an agent, always 'input' and 'output'."""
        if isinstance(item, str):
            if item.endswith("/input") or item.endswith("/output"):
                return None
        id = self._ids[self.nodes.index(item)]
        return [f"{id}/input", f"{id}/output"]

    @property
    def inputs(self, item) -> List[Any]:
        """Return a list of agent IDs that are connected to this port as inputs."""
        if isinstance(item, str) and "/input" in item:
            agent = self._get_agent_from_item(item)
            if agent:
                return [
                    f"{self.agent_network._get_node_id(p)}/output"
                    for p in self.agent_network.get_parents(agent)
                    if p in self.agent_network
                ]
        return []

    @property
    def outputs(self, item) -> List[Any]:
        """Return an empty list as output connections are not maintained in this model."""
        return None

    def can_connect(self, source, target) -> bool:
        """Determines if a connection can be made between source and target. Always false for read-only."""
        return False

    @property
    def position(self, item=None):
        """Returns the position of the node"""
        return self.positions.get(item)

    @position.setter
    def position(self, value, item=None):
        """The node position setter"""
        self.positions[item] = value

    @property
    def expansion_state(self, item=None):
        return self.ExpansionState.CLOSED

    @expansion_state.setter
    def expansion_state(self, value, item=None):
        pass

    @property
    def display_color(self, item):
        """The node color."""
        if isinstance(item, RunnableNode):
            if isinstance(item, FromRunnableNode):
                return (0.5058823529411764, 0.5607843137254902, 0.7058823529411765)
            if isinstance(item, RunnableAINode):
                return (0.09803921568627451, 0.12549019607843137, 0.20392156862745098)
            if isinstance(item, RunnableHumanNode):
                return (0.2627450980392157, 0.3333333333333333, 0.5215686274509804)
            if isinstance(item, USDAssistantNode):
                return (0.7058823529411765, 0.5058823529411764, 0.5607843137254902)
            if isinstance(item, NetworkNode):
                return (0.12549019607843137, 0.2627450980392157, 0.20392156862745098)
            if hasattr(item, "subnetwork"):
                return (0.12549019607843137, 0.12627450980392157, 0.20392156862745098)

    @property
    def selection(self):
        """return the selected node"""
        return self._selected_nodes

    @selection.setter
    def selection(self, value):
        """set the selection"""
        if value and value != self._selected_nodes:
            self._selected_nodes = value
        elif len(value) == 0:
            self._selected_nodes = []

        self._selection_changed()


class NetworkGraphWindow(ui.Window):
    def __init__(self, agent_network: "RunnableNetwork"):
        self.agent_network = agent_network
        super().__init__(
            agent_network.metadata.get("name", None) or type(agent_network).__name__, width=1200, height=600
        )
        self.delegate = RunnableNetworkDelegateClassic()
        self.model = RunnableNetworkGraphModel(agent_network)
        self.view = None
        self.frame.set_build_fn(self.build)
        self.subwindow = None

    def destroy(self):
        self.view = None

        if self.view:
            self.view.destroy()
            self.view = None

        if self.model:
            self.model.destroy()
            self.model = None

        if self.delegate:
            self.delegate.destroy()
            self.delegate = None

        if self.subwindow:
            self.subwindow.destroy()
            self.subwindow = None

        super().destroy()

    def build(self):
        self.view = GraphView(
            delegate=self.delegate, model=self.model, style=self.delegate.get_style(), horizontal=1, raster_nodes=1
        )

        self.view.set_mouse_double_clicked_fn(lambda x, y, b, m: self.__on_mouse_mouse_double_clicked(b))

        async def focus(view):
            for _ in range(2):
                await omni.kit.app.get_app().next_update_async()
            view.focus_on_nodes()

        asyncio.ensure_future(focus(self.view))

    def __on_mouse_mouse_double_clicked(self, button):
        if self.view:
            selection = self.view.selection
        else:
            selection = []

        self.on_left_mouse_button_double_clicked(selection)

    def on_left_mouse_button_double_clicked(self, items):
        """this is an override for the function in base Class while left mouse button is double clicked
        we enter into a subgraph when we double click a CompoundNode
        """
        # double click on a compound node will open the subgraph
        if not items or len(items) != 1:
            return

        # Open compound
        node = items[0]
        if isinstance(node, NetworkNode):
            node_full = self.model._full_nodes[self.model.nodes.index(node)]
            self.subwindow = NetworkGraphWindow(node_full)
        elif hasattr(node, "subnetwork"):
            node_full = node.subnetwork
            self.subwindow = NetworkGraphWindow(node_full)
