## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import weakref

import omni.ui as ui

from .agent_delegate import DefaultDelegate
from .chat_view import ChatView


def _get_subnodes_count(node: "NetworkNode | RunnableAIQNode") -> int:
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

                chat_view = NetworkNodeChatView()
                chat_view.visible = False
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
                    button = ui.Button(f"Expand ({subnodes_count})", height=20, width=0, name="expand-collapse")
                    ui.Line(width=20, alignment=ui.Alignment.V_CENTER)
                    ui.Spacer()

                with ui.Frame(height=0) as frame:
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

            # Appears when new node
            subnodes_count = _get_subnodes_count(node)
            button_text = f"Collapse ({subnodes_count})" if visible else f"Expand ({subnodes_count})"
            data.expand_button.text = button_text

        return DefaultDelegate.need_rebuild_agent_widget(self, network, node, data)
