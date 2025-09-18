## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import abc
import asyncio
import os
from enum import Enum, auto
from functools import partial
from typing import Any, Dict, NamedTuple, Tuple, Union

import carb.input
import omni.appwindow
import omni.kit.app
import omni.kit.clipboard as clipboard
import omni.ui as ui
from lc_agent import RunnableHumanImageNode, RunnableHumanNode, RunnableNetwork, RunnableNode, get_node_factory

from .style import chat_window_style
from .utils.formated_text import FormatedText
from .utils.indeterminate_progress_indicator import IndeterminateProgressIndicator

try:
    from omni.ai_agent.telemetry import response_feedback_data_send_event

    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False

from omni.kit.window.filepicker import FilePickerDialog


def _process_prompt(model):
    """Process prompt and extract image paths if present."""
    prompt = model.as_string

    # Extract any @image() references from the prompt
    import re

    image_pattern = r"@image\(([^)]+)\)"
    embedded_images = re.findall(image_pattern, prompt)

    return prompt, embedded_images


class RoleItem(ui.AbstractItem):
    def __init__(self, role_dict):
        super().__init__()
        self._role_dict = role_dict
        self._role_model = ui.SimpleStringModel(self._role_dict["role"])

    @property
    def role_model(self):
        return self._role_model

    @property
    def content(self):
        return self._role_dict["content"]


class RoleModel(ui.AbstractItemModel):
    def __init__(self, data_list):
        super().__init__()
        self.__items = [RoleItem(data) for data in data_list]

    def get_item_children(self, item):
        if item is not None:
            return []
        return self.__items

    def get_item_value_model_count(self, item):
        return 1

    def get_item_value_model(self, item, column_id):
        if item:
            return item.role_model


class RoleDelegate(ui.AbstractItemDelegate):
    def build_branch(self, model, item, column_id, level, expanded):
        pass

    def build_widget(self, model: RoleModel, item, column_id, level, expanded):
        """Create a widget per column per item"""
        if item is None:
            return

        item_model: ui.SimpleStringModel = model.get_item_value_model(item, column_id)

        with ui.HStack(height=40):
            ui.Spacer(width=10)

            with ui.VStack(width=20):
                ui.Spacer()
                ui.Image(name="chat-message-line", width=20, height=20)
                ui.Spacer()

            ui.Spacer(width=5)

            with ui.VStack():
                ui.Spacer()
                ui.StringField(
                    item_model,
                    enabled=False,
                    style_type_name_override="Label",
                    height=0,
                    tooltip=item_model.as_string,
                    name="history_title",
                )
                ui.Spacer()

            ui.Spacer(width=10)


class MessagesWidget:
    def __init__(self, data_list):
        self.tree_view_model = RoleModel(data_list)
        self.tree_view_delegate = RoleDelegate()

        with ui.HStack():
            with ui.ScrollingFrame(
                name="history", horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF, width=150
            ):
                self.tree_view = ui.TreeView(self.tree_view_model, delegate=self.tree_view_delegate)
                self.tree_view.set_selection_changed_fn(self.on_tree_view_selection_changed)
            with ui.ScrollingFrame():
                with ui.HStack():
                    ui.Spacer(width=20)
                    self.content = FormatedText("")
                    ui.Spacer(width=20)

    def on_tree_view_selection_changed(self, selection):
        if selection:
            self.content.update(selection[0].content)
        else:
            self.content.update("")


class AgentDelegate(abc.ABC):
    """
    AgentDelegate is an abstract base class for building UI for agents and
    network.
    """

    @abc.abstractmethod
    def build_agent_widget(self, network, agent) -> Any:
        """
        Builds the UI for the given agent within the network.
        """
        pass

    @abc.abstractmethod
    def build_request_widget(self, network):
        """
        Builds the UI for a request within the network.
        """
        pass

    @abc.abstractmethod
    def on_core_changed(self, core_name: str, network: RunnableNetwork):
        """
        Builds the UI for a request within the network.
        """
        pass

    def need_rebuild_agent_widget(self, network, agent, data) -> bool:
        """
        Returns True if the widget should be rebuilt.
        """
        return True


class LayoutDelegate(AgentDelegate):
    """
    LayoutDelegate provides base implementation for build_agent_widget and
    build_request_widget. It also provides hooks for building different parts of
    the agent UI.
    """

    _stop_astream_event = asyncio.Event()

    def __init__(self):
        super().__init__()
        self._data = None
        self._field_state = None
        self._stop_stack = None
        self._send_stack = None

    def build_agent_widget(self, network, agent) -> Any:
        """
        Builds the UI for the given agent within the network using
        the layout components.
        """

        def _collapse(expand_body_frame, collapsed_body_frame, collapse_btn):
            if collapse_btn.checked:
                # need to rebuild the collapsed frame since the agent output has been outdated.
                collapsed_body_frame.rebuild()
                expand_body_frame.visible = False
                collapsed_body_frame.visible = True
            else:
                expand_body_frame.visible = True
                collapsed_body_frame.visible = False

        def _body_content():
            style_name = "Label.User" if agent.name == "UserAgent" else "Label.Agent"
            with ui.VStack(height=0):
                FormatedText(self._get_agent_output(agent), style_name=style_name, scroll_to_bottom=False)

        def _build_collapse_button():
            with ui.HStack(width=0):
                ui.Spacer(width=10)
                with ui.ZStack(width=28):
                    btn = ui.ToolButton(name="collapse", width=28, height=28, visible=False)
                    ui.Spacer()
                ui.Spacer(width=10)
            return btn

        def _build_delete_and_copy_buttons():
            btn_stack = ui.VStack(width=0, height=30, visible=False)
            with btn_stack:
                ui.Spacer(height=4)
                with ui.ZStack():
                    ui.Rectangle(style_type_name_override=f"Rectangle.User.Buttons")
                    with ui.HStack():
                        with ui.VStack():
                            ui.Spacer()
                            ui.Button(
                                name="Copy-text",
                                image_width=25,
                                image_height=25,
                                height=0,
                                clicked_fn=lambda a=agent: self.copy_text(a.outputs.content),
                            )
                            ui.Spacer()
                        with ui.VStack():
                            ui.Spacer()
                            ui.Button(
                                name="delete",
                                image_width=20,
                                image_height=20,
                                height=0,
                                clicked_fn=lambda n=network, a=agent: self.delete_agent(n, a),
                            )
                            ui.Spacer()
            return btn_stack

        with ui.VStack():
            with ui.HStack():
                if agent.name == "UserAgent":
                    ui.Spacer(width=ui.Fraction(2))
                    stack = ui.ZStack(width=ui.Fraction(8))
                    with stack:
                        ui.Rectangle(style_type_name_override=f"Rectangle.User")
                        ui.Spacer(height=45)  # set a minimum height for the user message
                        with ui.VStack():
                            ui.Spacer(height=10)
                            with ui.HStack(height=0):
                                ui.Spacer(width=15)
                                data = self._build_agent_body(network, agent)
                                ui.Spacer(width=2)
                            ui.Spacer(height=10)
                        # copy and delete buttons when hovered
                        with ui.HStack():
                            ui.Spacer()
                            btn_stack = _build_delete_and_copy_buttons()
                            ui.Spacer(width=4)

                    ui.Spacer(width=2)

                    def on_hovered(hovered):
                        btn_stack.visible = hovered

                    stack.set_mouse_hovered_fn(lambda hovered: on_hovered(hovered))
                else:
                    collapse_height = 100

                    btn = _build_collapse_button()
                    self._build_agent_icon(network, agent)
                    ui.Spacer(width=20)
                    expand_body_frame = ui.Frame()
                    with expand_body_frame:
                        with ui.VStack(height=0):
                            data = self._build_agent_body(network, agent)

                    collapsed_body_frame = ui.ScrollingFrame(
                        height=collapse_height,
                        visible=False,
                        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                    )
                    collapsed_body_frame.set_build_fn(_body_content)

                    def frame_size_changed():
                        if expand_body_frame.computed_height > collapse_height:
                            btn.visible = True

                    expand_body_frame.set_computed_content_size_changed_fn(frame_size_changed)
                    btn.set_clicked_fn(lambda b=btn: _collapse(expand_body_frame, collapsed_body_frame, btn))

            self._build_agent_footer(network, agent)

        return data

    def build_request_widget(self, network):
        """
        Builds the UI for a message request within the network.

        Args:
            network (RunnableNetwork): The agent network that the widget is a part of.
        """

        class FieldState:
            """
            Manages the state of a field (input box) in the UI.
            """

            def __init__(self, network, send_message_func):
                """
                Initializes a new instance of FieldState.

                Args:
                    network (RunnableNetwork): The agent network that the field is a part of.
                    send_message_func (function): A callback function to send messages when prompted.
                """
                self._network = network
                self._send_message_func = send_message_func

                app_window = omni.appwindow.get_default_app_window()
                self._key_input = carb.input.acquire_input_interface()
                self._keyboard = app_window.get_keyboard()

                self._loop_task = None
                self._loop_event = None
                self.model = None
                self.edit = False

                self._image_path = None  # Legacy field for compatibility
                self._enter_was_pressed = False  # Track if Enter was pressed in the previous frame

            def __del__(self):
                """
                Deletes the instance of FieldState & cancels any running tasks, if any.
                """
                self.destroy()

            def destroy(self):
                if self._loop_event is not None:
                    self._loop_event.set()

                if self._loop_task is not None:
                    self._loop_task.cancel()

                self._network = None
                self._send_message_func = None

            @property
            def edit(self):
                """
                Returns the editing state of the field.

                Returns:
                    bool: True if the field is being edited, False otherwise.
                """
                return self._edit

            @edit.setter
            def edit(self, value):
                """
                Sets the editing state of the field & starts/stops the loop task accordingly.

                Args:
                    value (bool): The new editing state of the field.
                """
                self._edit = value
                if value and self._loop_task is None:
                    if self._loop_event is not None:
                        self._loop_event.set()

                    self._loop_event = asyncio.Event()

                    self._loop_task = asyncio.ensure_future(self._loop(self._loop_event))
                elif not value and self._loop_task is not None:
                    self._loop_event.set()
                    self._loop_event = None
                    self._loop_task.cancel()
                    self._loop_task = None

            def send_message_from_field(self):
                """
                Sends a message from the input field to the agent network if any text was entered.
                Clears the field after sending the message.
                """
                if not self.model:
                    return

                prompt, images = _process_prompt(self.model)
                if not prompt and not images:
                    return

                self._send_message_func(self._network, prompt or "Please analyze the attached image(s)", images)
                self.model.as_string = ""

            async def _loop(self, loop_event):
                """
                Updates the state of the input field as long as the field is being edited.
                Sends a message from the field once Enter is pressed.
                """
                while True:
                    await omni.kit.app.get_app().next_update_async()

                    KeyboardInput = carb.input.KeyboardInput
                    key_input = self._key_input

                    def is_key_down(key):
                        return key_input.get_keyboard_button_flags(self._keyboard, key) & carb.input.BUTTON_FLAG_DOWN

                    enter_pressed = is_key_down(KeyboardInput.ENTER) or is_key_down(KeyboardInput.NUMPAD_ENTER)
                    shift_down = is_key_down(KeyboardInput.LEFT_SHIFT) or is_key_down(KeyboardInput.RIGHT_SHIFT)
                    alt_down = is_key_down(KeyboardInput.LEFT_ALT) or is_key_down(KeyboardInput.RIGHT_ALT)
                    ctrl_down = is_key_down(KeyboardInput.LEFT_CONTROL) or is_key_down(KeyboardInput.RIGHT_CONTROL)

                    if enter_pressed and not (shift_down or alt_down or ctrl_down):
                        # Only send message if Enter wasn't pressed in the previous frame
                        if not self._enter_was_pressed:
                            self.send_message_from_field()
                        self._enter_was_pressed = True
                    else:
                        self._enter_was_pressed = False

                    if loop_event.is_set():
                        break

        if self._field_state is not None:
            self._field_state.destroy()

        self._field_state = FieldState(network, self._submit_prompt)

        def input_field_changed(model, input_stack):
            content = model.as_string
            numbers_of_lines = content.count("\n") + 1
            input_stack.height = ui.Pixel(32 + 18 * numbers_of_lines)

        def on_begin_edit(field_state, model):
            field_state.edit = True
            field_state.model = model

        def on_end_edit(field_state, model):
            field_state.edit = False
            field_state.model = None

        with ui.VStack():
            input_stack = ui.HStack(height=50)
            with input_stack:
                ui.Spacer(width=20)

                with ui.ZStack():
                    input_field = ui.StringField(style_type_name_override="InputField", multiline=True).model
                    input_field.add_begin_edit_fn(partial(on_begin_edit, self._field_state))
                    input_field.add_end_edit_fn(partial(on_end_edit, self._field_state))
                    input_field.add_value_changed_fn(lambda m, s=input_stack: input_field_changed(m, s))

                    def _on_voice_input_button_clicked():
                        print("Voice input button clicked")

                    with ui.HStack():
                        ui.Spacer()
                        with ui.VStack(width=0, content_clipping=True):
                            ui.Spacer()
                            self._voice_btn = ui.Button(
                                name="voice",
                                tooltip="audio input",
                                height=32,
                                width=32,
                                clicked_fn=_on_voice_input_button_clicked,
                                visible=0,
                            )
                            ui.Spacer()
                        ui.Spacer(width=2)

                # Here we are calling the new function that builds buttons
                with ui.Frame(width=0):
                    self.build_request_buttons(network, input_field)

                ui.Spacer(width=15)
            ui.Spacer(height=18)

    def build_request_buttons(self, network, input_model):
        """Builds the buttons for the request widget"""

        def send_clicked_fn(network, model):
            prompt, images = _process_prompt(model)
            if prompt or images:
                self._submit_prompt(network, prompt, images)
            model.as_string = ""

        self._input_filepicker = None

        def _on_input_image_button_clicked():
            def _on_apply_input_image(filename, dirname):
                # Handle both (file, directory) and (filename, dirname) parameter names
                if dirname and filename:
                    image_path = os.path.join(dirname, filename) if dirname != "." else filename
                    # Append @image(path) to the text field
                    current_text = input_model.as_string
                    if current_text and not current_text.endswith(" "):
                        current_text += " "
                    input_model.as_string = f"{current_text}@image({image_path})"
                    print(f"[Image Support] Added image reference: @image({image_path})")
                if self._input_filepicker:
                    self._input_filepicker.hide()

            if not self._input_filepicker:
                self._input_filepicker = FilePickerDialog(
                    "Select Image",
                    click_apply_handler=_on_apply_input_image,
                    file_extension_options=[(".png", "PNG Image"), (".jpg", "JPEG Image"), (".jpeg", "JPEG Image")],
                )
            self._input_filepicker.show()

        with ui.HStack():
            ui.Spacer(width=10)
            with ui.VStack(width=0, content_clipping=True):
                ui.Spacer()
                self._upload_btn = ui.Button(
                    name="upload",
                    tooltip="Attach images (click to select)",
                    height=32,
                    width=32,
                    clicked_fn=_on_input_image_button_clicked,
                    visible=True,  # Make visible by default
                )
                ui.Spacer()
            ui.Spacer(width=2)
            with ui.ZStack():
                self._stop_stack = ui.VStack(width=40, content_clipping=True, visible=False)
                with self._stop_stack:
                    ui.Spacer()
                    ui.Button(name="stop", height=32, width=32, clicked_fn=self.stop_astream)
                    ui.Spacer()
                self._send_stack = ui.VStack(width=40, content_clipping=True, visible=True)
                with self._send_stack:
                    ui.Spacer()
                    ui.Button(
                        name="send",
                        height=32,
                        width=32,
                        clicked_fn=lambda n=network, m=input_model: send_clicked_fn(n, m),
                    )
                    ui.Spacer()
        # Fire up logic that checks whether the 'Send Image' button should be hidden or not.
        self.on_core_changed(network.chat_model_name, network)

    def stop_astream(self):
        """
        Stops the node astream process.
        """
        # Set the stop event to signal all running processes to terminate
        self._stop_astream_event.set()
        self._stop_stack.visible = False
        self._send_stack.visible = True

    def on_core_changed(self, core_name: str, network: RunnableNetwork):
        # LLM Cores don't have metadata which we can query on the modalities support.
        # So, filter by the name for now.
        # In the future, hopefully we can add metadata methods to LLM Cores which will simplify this kind of logic.
        if hasattr(self, "_send_image_btn"):
            if core_name == "NVCF NeVA 22B" or core_name == "NVCF Fuyu 8B" or core_name == "Gemini Pro Vision":
                self._send_image_btn.visible = True
            else:
                self._send_image_btn.visible = False

    def _submit_prompt(self, network, prompt, images=None):
        """
        Submits a user prompt to the network, creating a new agent with the prompt.
        If images are provided, uses RunnableHumanImageNode instead.

        Args:
            network (RunnableNetwork): The agent network to add the agent to.
            prompt (str): The user's message prompt.
            images (list): Optional list of image paths or URLs.
        """
        if images and len(images) > 0:
            # Use RunnableHumanImageNode when images are present
            network.add_node(get_node_factory().create_node("UserAgentImage", prompt, images))
        else:
            # Regular text-only message
            network.add_node(get_node_factory().create_node("UserAgent", prompt))
        self._process_network(network)

    def _process_network(self, network):
        async def _process(stop_astream_event):
            async_generator = network.astream()

            await omni.kit.app.get_app().next_update_async()
            self._stop_stack.visible = True
            self._send_stack.visible = False

            try:
                async for _ in async_generator:
                    if stop_astream_event.is_set():
                        await async_generator.aclose()
            except asyncio.CancelledError:
                # We always cancel the task. It's not a problem.
                pass
            except BaseException:
                raise
            finally:
                for node in network.get_leaf_nodes():
                    network._event_callback(RunnableNetwork.Event.NODE_INVOKED, {"node": node, "network": network})

                self._stop_stack.visible = False
                self._send_stack.visible = True

        self._stop_astream_event.clear()
        asyncio.ensure_future(_process(self._stop_astream_event))

    def _build_agent_icon(self, network, agent: RunnableNode):
        """
        Builds the icon of the agent's UI.
        """
        icon_size = 26
        with ui.ZStack(width=icon_size, height=icon_size):
            ui.Circle(style_type_name_override=f"Circle.Agent")
            ui.Label("N", alignment=ui.Alignment.CENTER, name="N")

    def _build_agent_header(self, network, agent):
        """
        Builds the header part of the agent's UI.
        """
        pass

    def _build_agent_body_left(self, network, agent):
        """
        Builds the left part of the agent's UI body.
        """
        pass

    def _build_agent_body(self, network, agent):
        """
        Builds the main part of the agent's UI body.
        """
        style_name = "Label.User" if agent.name == "UserAgent" else "Label.Agent"
        ui.Label(
            agent.outputs.content or "",
            style_type_name_override=style_name,
            word_wrap=True,
            alignment=ui.Alignment.LEFT_TOP,
        )

    def _build_agent_body_right(self, network, agent):
        """
        Builds the right part of the agent's UI body.
        """
        pass

    def copy_text(self, text: str):
        """Copy item text to clipboard"""
        print(f"copying text to clipboard: {text}")
        clipboard.copy(text)

    def delete_agent(self, network: RunnableNetwork, node: RunnableNode):
        """Delete the agent from the network"""
        network.remove_node(node)

    def _build_agent_footer(self, network, agent):
        """
        Builds the footer part of the agent's UI.
        """

        def send_rated_event(network: RunnableNetwork, agent: RunnableNode, rating: int):
            if not TELEMETRY_AVAILABLE:
                return

            """Send telemetry data"""
            messages = agent.metadata.get("messages", [])
            user_prompt = ""
            for message in messages:
                role = message["role"]
                if role == "user":
                    user_prompt = message["content"]
                    break
            response = agent.output
            context = agent.metadata.get("context", None)
            if not context:
                if hasattr(agent, "_system"):
                    context = agent._system
            if not context:
                context = ""
            agent_role = ""  # TODO: NOTE: This is a stub
            agent_parents = ""
            if hasattr(agent, "parents"):
                for parent in agent.parents():
                    parent_name = parent.name
                    if not parent_name:
                        parent_name = "None"
                    if len(agent_parents) > 0:
                        agent_parents += ", "
                    agent_parents += parent_name
            agent_type = agent.name
            if not agent_type:
                agent_type = "None"
            network_name = network.metadata.get("name", None)
            if not network_name:
                network_name = "None"
            model_name = agent.metadata.get("core", None)
            if not model_name:
                model_name = "None"
            # response_feedback_data_send_event(
            #     user_prompt,
            #     response,
            #     context,
            #     agent_role,
            #     agent_parents,
            #     agent_type,
            #     network_name,
            #     model_name,
            #     "like",
            #     rating,
            # )

        def refresh_agent(network: RunnableNetwork, node: RunnableNode):
            """Refresh the agent in the network"""
            children = network.get_all_children(node)
            for child in children:
                network.remove_node(child)

            parent = network.get_parents(node)[0]
            parent.invoked = False
            network.remove_node(node)
            self._process_network(network)

        with ui.VStack():
            ui.Spacer(height=10)
            if agent.name != "UserAgent":
                with ui.HStack(height=40):
                    ui.Spacer(width=95)
                    ui.Label(agent.name or type(agent).__name__, name="bot_name", width=0)
                    ui.Spacer(width=20)
                    ui.Button(
                        name="Thumb-up",
                        width=30,
                        enabled=0,  # , clicked_fn=lambda n=network, a=agent: send_rated_event(n, a, 1)
                    )
                    ui.Button(
                        name="Thumb-down",
                        width=30,
                        enabled=0,  # , clicked_fn=lambda n=network, a=agent: send_rated_event(n, a, 0)
                    )
                    # ui.Button(
                    #     name="show-systems",
                    #     width=30,
                    #     clicked_fn=lambda agent=agent: self._show_context(agent),
                    # )
                    ui.Button(
                        name="Copy-text",
                        width=40,
                        clicked_fn=lambda a=agent: self.copy_text(a.outputs.content),
                    )
                    ui.Spacer()
                    ui.Button(name="delete", width=30, clicked_fn=lambda n=network, a=agent: self.delete_agent(n, a))
                    ui.Button(
                        name="refresh",
                        width=30,
                        clicked_fn=lambda n=network, a=agent: refresh_agent(n, a),
                        visible=network.name == "Main",
                    )
                ui.Spacer(height=10)

                agent_error = agent.metadata.get("error", None)
                if agent_error:

                    def split_string(string, limit):
                        words = string.split()
                        lines = []
                        temp_line = ""

                        for word in words:
                            if len(temp_line) + len(word) + 1 > limit:
                                lines.append(temp_line.strip())
                                temp_line = ""
                            temp_line += word + " "
                        lines.append(temp_line.strip())

                        return "\n".join(lines)

                    ui.Spacer(width=15)
                    if agent_error.startswith("Network length"):
                        error_text = "Network length"
                    if (
                        agent_error.startswith("1 validation error for ChatOpenAI")
                        and "`OPENAI_API_KEY`" in agent_error
                        and (network.chat_model_name is None or network.chat_model_name.startswith("nvidia"))
                    ):
                        error_text = "Missing API Key"
                        agent_error = "Please set the API Key to use USD NIMs"
                    elif "unauthorized" in agent_error.lower():
                        error_text = "API Key Unauthorized. Please check your API Key and restart Kit."
                    else:
                        error_text = "Error"
                    with ui.HStack(height=40):
                        ui.Spacer(width=95)
                        ui.Label(error_text, name="error", width=0, tooltip=split_string(str(agent_error), 100))

                num_tokens = agent.metadata.get("num_tokens", 0)
                num_input_tokens = agent.metadata.get("num_input_tokens", 0)

                num_tokens_text = ""
                if num_input_tokens:
                    num_tokens_text = f"{num_input_tokens} / {num_tokens or 0}"
                elif num_tokens:
                    num_tokens_text = f"{num_tokens}"

                tokens_label = ui.Label(num_tokens_text, width=0)

                if self._data:
                    self._data.tokens = tokens_label
                    tooltip = self._get_tokens_tooltip(agent)
                    if tooltip:
                        self._data.tokens.tooltip = tooltip
                    self._data = None

    def _get_style(self):
        """
        Returns omni.ui style dict that will be applied to the root frame.
        """
        pass

    def _show_context(self, agent):
        def check_format(variable):
            if type(variable) is list:
                for item in variable:
                    if type(item) is dict:
                        if "role" in item.keys() and "content" in item.keys():
                            continue
                        else:
                            return False
                    else:
                        return False
                return True
            else:
                return False

        messages = agent.metadata.get("messages", None)
        if messages and check_format(messages):
            messages = messages[:]
            messages.append({"role": "output", "content": agent.outputs.content})
        else:
            messages = None

        context = agent.metadata.get("context", None)

        if not context:
            if hasattr(agent, "_system"):
                context = agent._system

        if not context:
            context = agent.outputs.content

        if not context:
            return

        self.__window = ui.Window(
            "Context", width=500, height=500, flags=ui.WINDOW_FLAGS_NO_SCROLL_WITH_MOUSE | ui.WINDOW_FLAGS_NO_SCROLLBAR
        )
        self.__window.frame.style = chat_window_style
        with self.__window.frame:
            if messages:
                self.__messages_widget = MessagesWidget(messages)
            else:
                with ui.ScrollingFrame():
                    ui.Label(context, word_wrap=True, alignment=ui.Alignment.LEFT_TOP)

    def _get_tokens_tooltip(self, agent):
        # Create a tooltip with detailed data
        time_to_first_token = agent.metadata.get("time_to_first_token", None)
        average_tokens_per_sec = agent.metadata.get("average_tokens_per_sec", None)
        delivered_tokens_per_sec = agent.metadata.get("delivered_tokens_per_sec", None)
        evaluation_time = agent.metadata.get("evaluation_time", None)

        tooltip_parts = []
        if time_to_first_token is not None:
            tooltip_parts.append(f"First token: {time_to_first_token:.1f}s")
        if average_tokens_per_sec is not None:
            tooltip_parts.append(f"Avg tokens/s: {average_tokens_per_sec:.1f}")
        if delivered_tokens_per_sec is not None:
            tooltip_parts.append(f"Delivered tokens/s: {delivered_tokens_per_sec:.1f}")
        if evaluation_time is not None:
            tooltip_parts.append(f"Eval time: {evaluation_time:.1f}s")

        return "\n".join(tooltip_parts)


class DefaultDelegate(LayoutDelegate):
    """
    DefaultDelegate provides a way to process Markdown output and generate UI
    blocks.

    The implementation is itegrated with the new Markdown Widget we need to
    implement.
    """

    class _BlockType(Enum):
        """The part of MarkdownWidget"""

        H1 = auto()
        H2 = auto()
        H3 = auto()
        PARAGRAPH = auto()
        LIST = auto()
        CODE = auto()
        IMAGE = auto()
        LINK = auto()
        TABLE = auto()

    class _Block(NamedTuple):
        """The part of MarkdownWidget"""

        block_type: "_BlockType"
        text: Union[str, Tuple[str]]
        source: str
        metadata: Dict[str, Any]

    def _get_agent_output(self, agent):
        """
        The ability to filter the agent output
        """
        output = agent.outputs
        if output:
            if isinstance(output.content, list):
                # Safely extract text from multi-modal content
                text_parts = []
                entry = output.content[0]
                if isinstance(entry, dict) and entry.get("type") == "text":
                    text_parts.append(entry.get("text", ""))
                elif isinstance(entry, str):
                    text_parts.append(entry)
                return "".join(text_parts)
            else:
                result = output.content
                if isinstance(result, str) and result.startswith("FINAL "):
                    result = result[6:]
                return result
        else:
            return None

    def need_rebuild_agent_widget(self, network, agent, data) -> bool:
        """
        Returns True if the widget should be rebuilt.
        """
        if not data or not data.formatted_text:
            return True

        if agent.metadata.get("error", None):
            return True

        agent_output = self._get_agent_output(agent)
        if agent_output:
            if data.progress:
                data.progress.visible = False
                data.progress = None
            formated_text = data.formatted_text
            formated_text.visible = True
            formated_text.update(agent_output)

        # Updata tokens
        if data.tokens:
            num_tokens = agent.metadata.get("num_tokens", None)
            num_input_tokens = agent.metadata.get("num_input_tokens", None)

            num_tokens_text = ""
            if num_input_tokens:
                num_tokens_text = f"{num_input_tokens} / {num_tokens or 0}"
            elif num_tokens:
                num_tokens_text = f"{num_tokens}"

            data.tokens.text = num_tokens_text
            tooltip = self._get_tokens_tooltip(agent)
            if tooltip:
                data.tokens.tooltip = tooltip

        # We just updated it. No need to rebuild.

        return False

    def _build_agent_body(self, network, agent):
        """
        Builds the main part of the agent's UI body.
        """

        class WidgetData:
            def __init__(self):
                self.formatted_text: Any = None
                self.progress: Any = None
                self.tokens: Any = None
                self.images: list = []

        self._data = WidgetData()

        agent_output = self._get_agent_output(agent)
        text_visible = bool(agent_output)
        progress_visible = not text_visible and "error" not in agent.metadata
        style_name = "Label.User" if agent.name == "UserAgent" else "Label.Agent"
        self._data.formatted_text = FormatedText(agent_output, visible=text_visible, style_name=style_name)
        if progress_visible:
            self._data.progress = IndeterminateProgressIndicator(visible=progress_visible)

        return self._data

    def _build_md_block_component(self, network, agent, block: _Block):
        """
        Builds the UI component for the given markdown block.
        """
        pass

    def _build_md_block_h1(self, network, agent, block: _Block):
        """
        Builds the UI component for an H1 markdown block.
        """
        pass

    def _build_md_block_h2(self, network, agent, block: _Block):
        """
        Builds the UI component for an H2 markdown block.
        """
        pass

    def _build_md_block_h3(self, network, agent, block: _Block):
        """
        Builds the UI component for an H3 markdown block.
        """
        pass

    def _build_md_block_text(self, network, agent, block: _Block):
        """
        Builds the UI component for a text markdown block.
        """
        pass

    def _build_md_block_picture(self, network, agent, block: _Block):
        """
        Builds the UI component for a picture markdown block.
        """
        pass

    def _build_md_block_table(self, network, agent, block: _Block):
        """
        Builds the UI component for a table markdown block.
        """
        pass
