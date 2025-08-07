## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import asyncio
import json
from typing import Dict, List

import omni.ui as ui
from lc_agent import RunnableNetwork, RunnableNode
from omni.ai.langchain.widget.core import DefaultDelegate
from omni.ai.langchain.widget.core.utils.indeterminate_progress_indicator import IndeterminateProgressIndicator
from omni.ui import color as cl

# Define colors
cl.background_color = cl.shade(cl("#00000000"))
cl.selected_color = cl.shade(cl("#4a90e2"))


class USDSearchImageWidget:
    def __init__(
        self, query: str, images: List[str], usd_paths: List[str], bounding_boxs: List[list] = [], *args, **kwargs
    ):
        self._frame = ui.Frame(*args, **kwargs)
        self._query = query
        self._image_preview = None
        self._images = images
        self._usd_paths = usd_paths
        self._bounding_boxs = bounding_boxs
        self._selected_items = set()
        self._image_frames = {}
        self._build_ui()

    def _build_ui(self):
        with self._frame:
            with ui.VStack():
                ui.Label(f"USD Search Found {len(self._images)} Assets for the query: {self._query}")
                ui.Spacer(height=5)
                with ui.VGrid(height=0, column_width=206, row_height=236, spacing=10, padding=0) as self._grid:
                    for i, image in enumerate(self._images):
                        self._build_image_item(i, image)

                # this should "deselect all" but is not working
                self._grid.set_mouse_released_fn(self._on_background_click)

    def _build_image_item(self, index: int, image: str):
        with ui.ZStack(content_clipping=True, selected=False) as frame:
            self._image_frames[index] = frame
            ui.Rectangle(
                style={
                    "background_color": cl.background_color,
                    "border_width": 3,
                    "border_radius": 4,
                    ":checked": {"border_color": cl.selected_color},
                }
            )
            with ui.VStack():
                ui.Spacer(height=10)
                with ui.HStack():
                    ui.Spacer(width=10)
                    with ui.VStack():
                        file_url = self._usd_paths[index]
                        short_url = file_url.split("/")[-1]
                        if len(short_url) > 30:
                            short_url = short_url[:27] + "..."
                        img = ui.Image(image, width=186, height=186)
                        img.set_mouse_released_fn(lambda x, y, b, m, idx=index: self._on_image_click(x, y, idx, b, m))

                        self._set_drag_fn(img, index)
                        ui.Label(
                            f"{short_url}", style={"font_size": 14}, tooltip=file_url, alignment=ui.Alignment.CENTER
                        )

                    ui.Spacer(width=10)
                ui.Spacer(height=20)

    def _on_image_click(self, x, y, index: int, button: int, modifier):
        # print(f"Image click: {x}, {y}, {index}, {button}, {modifier}")
        if button == 0:  # Left click
            frame = self._image_frames[index]
            if (
                frame.screen_position_x < x < frame.screen_position_x + frame.computed_content_width
                and frame.screen_position_y < y < frame.screen_position_y + frame.computed_content_height
            ):
                self._image_frames[index].checked = not self._image_frames[index].checked
                if self._image_frames[index].checked:
                    self._selected_items.add(index)
                else:
                    if index in self._selected_items:
                        self._selected_items.remove(index)

        elif button == 1:  # Right click
            self._show_context_menu(index)

    def _on_background_click(self, x, y, button, modifier):
        # print(f"Background click: {x}, {y}, {button}, {modifier}")
        if (
            self._grid.screen_position_x < x < self._grid.screen_position_x + self._grid.computed_content_width
            and self._grid.screen_position_y < y < self._grid.screen_position_y + self._grid.computed_content_height
        ):

            if button == 0:  # Left click
                import copy

                _selections = copy.copy(self._selected_items)

                async def __delay_unselect():
                    import omni.kit.app

                    await omni.kit.app.get_app().next_update_async()

                    # Only deselect all if no selection changed
                    if self._selected_items == _selections:
                        for index in range(len(self._image_frames)):
                            self._image_frames[index].checked = False

                        self._selected_items.clear()

                asyncio.ensure_future(__delay_unselect())

    def _set_drag_fn(self, image_widget: ui.Image, index: int):
        def _get_drag_data(index):
            thumbnail = self._images[index]
            icon_size = 64
            ui.ImageWithProvider(thumbnail, width=icon_size, height=icon_size)

            selected_urls = (
                [self._usd_paths[i] for i in self._selected_items] if self._selected_items else [self._usd_paths[index]]
            )
            return "\n".join(selected_urls)

        image_widget.set_drag_fn(lambda: _get_drag_data(index))

    def _show_context_menu(self, index: int):
        self._menu = ui.Menu()
        with self._menu:
            ui.MenuItem("Copy URL", triggered_fn=lambda: self._copy_url(index))
        self._menu.show()

    def _copy_url(self, index: int):
        import omni.kit.clipboard as clipboard

        urls = [self._usd_paths[i] for i in self._selected_items] if self._selected_items else [self._usd_paths[index]]
        # print(f"Copying URLs: {urls}")
        # build a url string with the selected urls separated by new lines
        url = "\n".join(urls)
        clipboard.copy(url)


class USDSearchImageDelegate(DefaultDelegate):
    def _build_agent_body(self, network: "RunnableNetwork", agent: RunnableNode):
        # print("USDSearchImageDelegate: _build_agent_body")
        if agent.outputs:
            try:
                output_data: Dict[str, list] = json.loads(agent.outputs.content)
                # output_data is a dictionary with query and results
                with ui.VStack():
                    for query, data in output_data.items():
                        images = [item["image"] for item in data]
                        usd_paths = [item["url"] for item in data]
                        # bounding_boxs = [item.get("bbox_dimension", None) for item in data]
                        # dont want to get boudning box for now
                        USDSearchImageWidget(query, images, usd_paths)
            except Exception as e:
                return super()._build_agent_body(network, agent)

    def need_rebuild_agent_widget(self, network, agent, data) -> bool:
        return True
