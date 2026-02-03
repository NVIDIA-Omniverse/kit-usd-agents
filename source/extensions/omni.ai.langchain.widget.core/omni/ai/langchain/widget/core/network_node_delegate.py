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

import weakref

import omni.ui as ui

from .agent_delegate import DefaultDelegate
from .chat_view import ChatView


def _get_subnodes_count(node: "NetworkNode | RunnableNATNode") -> int:
    if hasattr(node, "subnetwork"):
        node = node.subnetwork
        if node is None:
            return 0

    if not hasattr(node, "nodes"):
        return 0

    return len([n for n in node.nodes if n.metadata.get("contribute_to_ui", True)])


class NetworkNodeChatView(ChatView):
    """Customized ChatView with no prompt field and no scroll bar"""

    def __init__(self, network=None, **kwargs):
        super().__init__(network=network, **kwargs)

        self._visible = True
        self._tree_view = None

    def build_view_header(self):
        pass

    def build_view_footer(self):
        pass

    def build_view_body(self):
        if self._tree_model:
            self._tree_model.destroy()

        self._tree_model = self._create_tree_model()
        self._tree_view = ui.TreeView(
            self._tree_model, delegate=self._delegate, root_visible=False, header_visible=False, visible=self._visible
        )

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        self._visible = value
        if self._tree_view:
            self._tree_view.visible = value


class NetworkNodeDelegate(DefaultDelegate):
    """Delegate for subnetworks"""

    def build_agent_widget(self, network, node):
        def hide_chat_view(chat_view, body, button, node):
            visible = chat_view.visible

            chat_view.visible = not visible
            body.visible = visible

            # Appears when clicking the button
            subnodes_count = _get_subnodes_count(node)
            button_text = f"Expand ({subnodes_count})" if visible else f"Collapse ({subnodes_count})"
            button.text = button_text

        # Creat ChatView on the top of the area
        with ui.VStack(height=0):
            with ui.HStack(height=0):
                # Just a small indent
                ui.Spacer(width=50)

                # Auto-expand if this node is the only node at its level in the parent network
                auto_expand = (
                    hasattr(network, "nodes")
                    and len([n for n in network.nodes if n.metadata.get("contribute_to_ui", True)]) == 1
                )

                chat_view = NetworkNodeChatView()
                chat_view.visible = auto_expand
                if hasattr(node, "subnetwork"):
                    chat_view.network = node.subnetwork
                else:
                    chat_view.network = node

            with ui.ZStack():
                with ui.HStack(content_clipping=1, height=0):
                    ui.Spacer()
                    ui.Line(width=20, alignment=ui.Alignment.V_CENTER)

                    # Appears on create
                    subnodes_count = _get_subnodes_count(node)
                    button_text = f"Collapse ({subnodes_count})" if auto_expand else f"Expand ({subnodes_count})"
                    button = ui.Button(button_text, height=20, width=0, name="expand-collapse")
                    ui.Line(width=20, alignment=ui.Alignment.V_CENTER)
                    ui.Spacer()

                with ui.Frame(height=0, visible=not auto_expand) as frame:
                    with ui.VStack():
                        ui.Spacer(height=30)
                        data = DefaultDelegate.build_agent_widget(self, network, node)

        button.set_clicked_fn(
            lambda v=weakref.proxy(chat_view), f=weakref.proxy(frame), b=weakref.proxy(button): hide_chat_view(
                v, f, b, node
            )
        )

        # Store the button for future reference
        data.chat_view = chat_view
        data.body_frame = frame
        data.expand_button = button

        return data

    def _build_agent_header(self, network, agent):
        # Space for button
        ui.Spacer(height=20)

    def need_rebuild_agent_widget(self, network, node, data) -> bool:
        if hasattr(data, "chat_view") and hasattr(data, "expand_button"):
            visible = data.chat_view.visible

            # Check if the subnetwork has been populated and update the chat_view network
            # This is necessary for nodes like RunnableNATNode where the subnetwork
            # is created during invocation, not during widget creation
            if hasattr(node, "subnetwork"):
                current_subnetwork = node.subnetwork
                if current_subnetwork is not None and data.chat_view.network != current_subnetwork:
                    # Subnetwork is now available, assign it to the chat view
                    data.chat_view.network = current_subnetwork

            # Auto-expand if this node is the only one at its level in the parent network
            should_auto_expand = (
                hasattr(network, "nodes")
                and len([n for n in network.nodes if n.metadata.get("contribute_to_ui", True)]) == 1
            )

            # Apply auto-expand logic: show chat_view and hide body_frame when there's only 1 node at this level
            if should_auto_expand and not visible:
                data.chat_view.visible = True
                data.body_frame.visible = False
                visible = True

            subnodes_count = _get_subnodes_count(node)
            button_text = f"Collapse ({subnodes_count})" if visible else f"Expand ({subnodes_count})"
            data.expand_button.text = button_text

        return DefaultDelegate.need_rebuild_agent_widget(self, network, node, data)
