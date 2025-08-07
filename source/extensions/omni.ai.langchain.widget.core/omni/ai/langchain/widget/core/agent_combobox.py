## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from functools import partial
from typing import Callable, List

import carb.tokens
import omni.ui as ui
from lc_agent import get_node_factory

from .chat_view import get_node_names, get_persistent_node_name
from .style import agent_combo_menu_style

ICON_PATH = carb.tokens.get_tokens_interface().resolve("${omni.ai.langchain.widget.core}/icons")


class _AgentItem(ui.AbstractItem):
    def __init__(self, name, description, selected):
        super().__init__()
        self.name_model = ui.SimpleStringModel(name)
        self.description = description
        self.selected = selected


class AgentComboModel(ui.AbstractItemModel):
    def __init__(self, menu_options: List[str], selection: str):
        super().__init__()
        self._items = []

        if selection not in menu_options:
            selection = menu_options[0]
        for option in menu_options:
            node = get_node_factory().create_node(option)
            metadata = getattr(node, "metadata", {})
            description = metadata.get("description", "")
            if description:
                description = description.split("\n", 1)[0]
            self._items.append(_AgentItem(option, description, option == selection))

    def get_item_children(self, item):
        if item is not None:
            return []
        return self._items

    def get_item_value_model_count(self, item):
        return 1

    def get_item_value_model(self, item, column_id):
        if item:
            return item.name_model

    def on_selection_changed(self, selection: _AgentItem):
        for item in self._items:
            selected = selection == item
            if item.selected != selected:
                item.selected = selected
                self._item_changed(item)


class AgentComboDelegate(ui.AbstractItemDelegate):
    def __init__(self):
        super().__init__()

    def build_header(self, column_id=0):
        ui.Label("Model", name="model", height=30)

    def build_widget(self, model, item, column_id, level, expanded):
        stack = ui.HStack(height=50)
        with stack:
            with ui.VStack(spacing=2):
                ui.Spacer()
                value_model = model.get_item_value_model(item, column_id)
                ui.Label(value_model.as_string, name="agent-name", height=0)
                if desp := item.description:
                    ui.Label(desp, name="agent-description", width=225, height=0, word_wrap=True)
                ui.Spacer()
            icon_stack = ui.HStack(visible=item.selected)
            with icon_stack:
                ui.Spacer()
                with ui.VStack(width=0):
                    ui.Spacer()
                    ui.Image(width=16, height=16, name="agent-active")
                    ui.Spacer()
                # leave space for scrollbar if the drop-down menu is scrollable
                ui.Spacer(width=20)

        def on_hovered(hovered):
            icon_stack.visible = hovered
            model.on_selection_changed(item)

        stack.set_mouse_hovered_fn(lambda hovered: on_hovered(hovered))


class _AgentComboBoxMenu:
    def __init__(self, selection: int, **kwargs):
        """
        Initialize the menu.
        """
        self._window: ui.Window = None
        self._parent: ui.Widget = kwargs.get("parent", None)
        self._width: int = kwargs.get("width", 210)
        self._height: int = kwargs.get("height", 300)
        self._selection_changed_fn: Callable = kwargs.get("selection_changed_fn", None)
        self._visibility_changed_fn: Callable = kwargs.get("visibility_changed_fn", None)
        self._build_ui(selection)

    def _build_ui(self, selection: int):
        window_flags = (
            ui.WINDOW_FLAGS_NO_RESIZE
            | ui.WINDOW_FLAGS_POPUP
            | ui.WINDOW_FLAGS_NO_TITLE_BAR
            | ui.WINDOW_FLAGS_NO_BACKGROUND
            | ui.WINDOW_FLAGS_NO_MOVE
            | ui.WINDOW_FLAGS_NO_SCROLLBAR
        )
        self._window = ui.Window("ComboBoxMenu", flags=window_flags, auto_resize=True)
        if self._visibility_changed_fn:
            self._window.set_visibility_changed_fn(self._visibility_changed_fn)
        self._window.frame.style = agent_combo_menu_style
        with self._window.frame:
            with ui.ZStack(width=self._width, height=0):
                ui.Rectangle()
                with ui.VStack():
                    ui.Spacer(height=4)
                    with ui.HStack():
                        ui.Spacer(width=4)
                        bounded_frame = ui.Frame(
                            width=self._width,
                            build_fn=partial(self._frame_build_fn, selection),
                            name="menu",
                        )
                        collapsed_frame = ui.ScrollingFrame(
                            width=self._width,
                            height=self._height,
                            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                            build_fn=partial(self._frame_build_fn, selection),
                            visible=False,
                        )

                        def frame_size_changed():
                            if bounded_frame.computed_height > self._height:
                                bounded_frame.visible = False
                                collapsed_frame.visible = True
                            else:
                                bounded_frame.visible = True
                                collapsed_frame.visible = False

                        bounded_frame.set_computed_content_size_changed_fn(frame_size_changed)
                    ui.Spacer(height=4)

    def _on_selection_changed(self, selection: List[ui.AbstractItem]):
        if selection:
            self.hide()
            if self._selection_changed_fn:
                self._selection_changed_fn(selection[0].name_model.as_string)

    def _frame_build_fn(self, selection: int):
        with ui.VStack():
            self._model = AgentComboModel(get_node_names(), selection)
            self._delegate = AgentComboDelegate()
            with ui.HStack():
                ui.Spacer(width=8)
                self._treeview = ui.TreeView(
                    self._model, width=self._width, delegate=self._delegate, root_visible=False, header_visible=True
                )
            self._treeview.set_selection_changed_fn(self._on_selection_changed)
            ui.Spacer(height=8)

    def show(self, offset_x: int = 0, offset_y: int = 0):
        """
        Show the window. It will be positioned relative to the parent window.

        Keyword Args:
            offset_x(int): Offset in x direction. Default is 0.
            offset_y(int): Offset in y direction. Default is 0
        """
        if self._parent:
            self._window.position_x = self._parent.screen_position_x + offset_x
            self._window.position_y = self._parent.screen_position_y + offset_y
        elif offset_x != 0 or offset_y != 0:
            self._window.position_x = offset_x
            self._window.position_y = offset_y
        self._window.visible = True
        if self._visibility_changed_fn:
            self._visibility_changed_fn(True)

    def hide(self):
        """
        Hides the window.
        """
        self._window.visible = False
        if self._visibility_changed_fn:
            self._visibility_changed_fn(False)


class AgentComboBox:
    def __init__(self, **kwargs):
        """
        Initialize the menu.

        Args:
            options(List): List of options to show in combo box
        """
        self._width = kwargs.get("width", 200)
        self._height = kwargs.get("height", 32)
        self._selection_changed_fn: Callable = kwargs.get("selection_changed_fn", None)
        self._button = None
        self._label = None
        self._build_ui()

    def _build_ui(self):
        agent_names = get_node_names()
        self.selection = get_persistent_node_name()
        if agent_names and self.selection not in agent_names:
            self.selection = agent_names[0]

        with ui.ZStack(width=self._width, height=self._height):
            self._button = ui.Button(" ", clicked_fn=self.show_menu, name="agent-combo-button")
            with ui.HStack():
                ui.Spacer(width=16)
                self._label = ui.Label(self.selection, name="agent-combo-label")
                ui.Spacer()
                self._arrow_button = ui.Button(width=32, name="agent-combo-arrow", checked=False)
                ui.Spacer(width=10)

    def show_menu(self):
        """Show the menu for the combobox."""
        menu = _AgentComboBoxMenu(
            self.selection,
            parent=self._button,
            width=270,
            selection_changed_fn=self._on_selection_changed,
            visibility_changed_fn=self._on_menu_visibility_changed,
        )
        menu.show(offset_x=-1, offset_y=self._height - 4)

    def _on_selection_changed(self, option: str):
        self.selection = option
        self._label.text = option
        if self._selection_changed_fn:
            self._selection_changed_fn(option)

    def _on_menu_visibility_changed(self, visible):
        self._arrow_button.checked = visible

    def set_selection(self, option: str):
        if option in get_node_names():
            self._on_selection_changed(option)
