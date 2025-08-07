## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import carb.tokens
import omni.ui as ui
import omni.ui.color_utils as cl
import omni.ui.constant_utils as ct

icon_path = carb.tokens.get_tokens_interface().resolve("${omni.ai.langchain.widget.core}/icons")
font_path = carb.tokens.get_tokens_interface().resolve("${omni.ai.langchain.widget.core}/data/fonts")

cl.base_background_color = cl.color("292929")

cl.combo_background_color = cl.color("343434")
cl.combo_text_color = cl.color("B6B6B6")

cl.bot_background_color = cl.color("262626")

cl.agent_icon_background_color = cl.color("76b900")

cl.history_highlight_color = cl.color("#2A63BA")

cl.chat_button_color = cl.color("5E5E5E")
cl.chat_button_color_hovered = cl.color("9E9EDE")
cl.chat_button_color_pressed = cl.color("FEFEFE")
cl.chat_button_color_disabled = cl.color("404040")

cl.code_background_color = cl.color("101010")
cl.code_header_background_color = cl.color("3E3E3E")
cl.code_border_color = cl.color("3E3E3E")

cl.line_color = cl.color("3B3B3B")

cl.send_button_color_hovered = cl.color(0.5, 0.5, 1, 1.0)
cl.send_button_color_pressed = cl.color(0.8, 0.8, 1, 1.0)

cl.stop_button_color_hovered = cl.color(1.0, 0.2, 0.2, 1.0)
cl.stop_button_color_pressed = cl.color(1.0, 0.5, 0.5, 1.0)

cl.error_color = cl.color(1.0, 0.2, 0.2, 1.0)

cl.conversation_history_background_color = cl.color("202020")
ct.font_size = 20

cl.button_image = cl.color("E2E2E2")
cl.input_field_bg_color = cl.color("121212")
cl.chat_text_color = cl.color("D8D8D8")

cl.button_bg_color = cl.color("181818")
cl.button_bg_border_color = cl.color("696969")


chat_window_style = {
    "Button": {
        "background_color": 0x0,
    },
    "Button:checked": {
        "background_color": 0x0,
    },
    "Button:hovered": {
        "background_color": 0x0,
    },
    "Button:pressed": {
        "background_color": 0x0,
    },
    "Button.Image:hovered": {"color": cl.chat_button_color_hovered},
    "Button.Image:pressed": {"color": cl.chat_button_color_pressed},
    "Button.Image:disabled": {"color": cl.chat_button_color_disabled},
    "Button::agent-combo-button": {
        "background_color": cl.combo_background_color,
        "border_radius": 7,
    },
    "Button::agent-combo-button:hovered": {
        "background_color": cl.combo_background_color,
    },
    "Button::agent-combo-button:pressed": {
        "background_color": cl.combo_background_color,
    },
    "Button::example_conversation": {
        "border_color": cl.color(0.7),
        "border_width": 1,
        "border_radius": 0,
        "padding": 9,
    },
    "Button::example_conversation:hovered": {
        "border_color": cl.color.white,
    },
    "Button.Image::agent-combo-arrow": {
        "image_url": f"{icon_path}/openIconContainer.svg",
    },
    "Button.Image::agent-combo-arrow:checked": {
        "image_url": f"{icon_path}/closedIconContainer.svg",
    },
    "Button.Image::collapse": {
        "color": cl.button_image,
        "image_url": f"{icon_path}/openIconContainer.svg",
    },
    "Button.Image::collapse:checked": {
        "color": cl.button_image,
        "image_url": f"{icon_path}/closedIcon.svg",
    },
    "Button.Image::send": {
        "color": cl.button_image,
        "image_url": f"{icon_path}/submitButton.svg",
    },
    "Button.Image::send:hovered": {
        "color": cl.send_button_color_hovered,
    },
    "Button.Image::send:pressed": {
        "color": cl.send_button_color_pressed,
    },
    "Button::send:pressed": {
        "color": cl.send_button_color_pressed,
    },
    "Button.Image::voice": {
        "color": cl.button_image,
        "image_url": f"{icon_path}/voice.svg",
    },
    "Button.Image::upload": {
        "color": cl.button_image,
        "image_url": f"{icon_path}/uploadButton.svg",
    },
    "Button.Image::stop": {
        "color": cl.button_image,
        "image_url": f"{icon_path}/stop-button.svg",
    },
    "Button.Image::stop:hovered": {
        "color": cl.stop_button_color_hovered,
    },
    "Button.Image::stop:pressed": {
        "color": cl.stop_button_color_pressed,
    },
    "Button.Image::Thumb-up": {"image_url": f"{icon_path}/IcoThumbUpLine.svg"},
    "Button.Image::Thumb-down": {"image_url": f"{icon_path}/IcoThumbDownLine.svg"},
    "Button.Image::Copy-text": {"image_url": f"{icon_path}/copy-text.svg"},
    "Button.Image::show-systems": {"image_url": f"{icon_path}/show-systems.svg", "color": cl.chat_button_color},
    "Button.Image::chat-history-delete": {
        "background_color": 0x0,
        "image_url": f"{icon_path}/chat-history-delete.svg",
    },
    "Button.Image::delete": {
        "image_url": f"{icon_path}/chat-history-delete.svg",
    },
    "Button.Image::refresh": {
        "background_color": 0x0,
        "image_url": f"{icon_path}/IcoRefreshLine.svg",
    },
    "Button.Image::refresh:hovered": {
        "background_color": 0x0,
        "image_url": f"{icon_path}/IcoRefreshLine.svg",
    },
    "Button.Label::expand-collapse": {
        "color": cl.chat_button_color,
    },
    "Button.Label::example_conversation": {
        "color": cl.color.white,
        "font_size": 22,
        "font": f"{font_path}/NVIDIASans_Rg.ttf",
    },
    "Button.Label::example_conversation:hovered": {
        "color": cl.color.white,
    },
    "Button.Image::chat-history-save": {
        "color": cl.chat_button_color,
        "image_url": f"{icon_path}/chat-history-save.svg",
    },
    "Button.Image::chat-history-save:hovered": {
        "image_url": f"{icon_path}/chat-history-save.svg",
    },
    "Button.Image::chat-history-load": {
        "color": cl.chat_button_color,
        "image_url": f"{icon_path}/chat-history-load.svg",
    },
    "Button.Image::chat-history-load:hovered": {
        "image_url": f"{icon_path}/chat-history-load.svg",
    },
    "Button.Image::execute": {
        "image_url": f"{icon_path}/play-circle-fill.svg",
    },
    "Button.Image::copy-code": {
        "image_url": f"{icon_path}/copy-text.svg",
    },
    "Button.Image::list_view_toggle": {
        "image_url": f"{icon_path}/list-view-toggle.svg",
    },
    "Button.Image::new_chat": {"image_url": f"{icon_path}/new-chat-button.svg"},
    "CollapsableFrame::tool": {
        "background_color": cl.color.transparent,
        "secondary_color": cl.color.transparent,
        "border_color": cl.code_border_color,
        "border_width": 0.5,
        "border_radius": 4,
    },
    "CollapsableFrame::tool:hovered": {
        "background_color": cl.color.transparent,
        "secondary_color": cl.color.transparent,
    },
    "ComboBox::model-combo": {
        "color": cl.color.white,
        "background_color": cl.base_background_color,
        "font_size": 18,
        "font": f"{font_path}/NVIDIASans_Rg.ttf",
    },
    "Circle.Agent": {
        "background_color": cl.agent_icon_background_color,
    },
    "Field": {
        "background_color": 0x0,
        "color": cl.color.white,
        "font_size": ct.font_size,
        "font": f"{font_path}/NVIDIASans_Rg.ttf",
    },
    "Field::agent": {
        "background_color": cl.conversation_history_background_color,
        "margin": 1,
    },
    "Field:pressed": {
        "background_color": 0x0,
    },
    "Image.User": {"border_radius": 20, "image_url": f"{icon_path}/me.png"},
    "Image.Bot.ChatGPT": {"image_url": f"{icon_path}/openai-logo.svg"},
    "InputField": {
        "background_color": cl.input_field_bg_color,
        "border_color": cl.button_image,
        "border_width": 1,
        "padding": 4,
        "border_radius": 7,
        "font_size": 16,
        "font": f"{font_path}/NVIDIASans_Rg.ttf",
    },
    "Image::combo_button": {
        "image_url": f"{icon_path}/openIconContainer.svg",
    },
    "Label": {
        "font": f"{font_path}/NVIDIASans_Rg.ttf",
        "font_size": 18,
    },
    "Label::code_language": {
        "font": f"{font_path}/NVIDIASans_Lt.ttf",
        "font_size": ct.font_size,
    },
    "Label::error": {
        "color": cl.error_color,
    },
    "Label.Agent": {
        "color": cl.chat_text_color,
        "font": f"{font_path}/NVIDIASans_It.ttf",
        "font_size": 18,
    },
    "Label.User": {
        "color": cl.chat_text_color,
        "font": f"{font_path}/NVIDIASans_Rg.ttf",
        "font_size": 18,
    },
    "Label::N": {
        "color": cl.conversation_history_background_color,
        "font_size": 16,
        "font": f"{font_path}/NVIDIASans_Lt.ttf",
    },
    "Label.Code": {
        "color": cl.color.white,
        "font": f"{font_path}/NVIDIASans_Rg.ttf",
        "font_size": 18,
    },
    "Label::bot_name": {
        "color": cl.color.white,
        "font_size": 13,
    },
    "Label::history_title": {
        "color": cl.color.white,
        "background_color": cl.color.transparent,
        "font_size": 17,
        "font": f"{font_path}/NVIDIASans_Md.ttf",
    },
    "Label::new_conversation": {
        "color": cl.color.white,
        "alignment": ui.Alignment.CENTER,
        "font_size": 30,
    },
    "Label::model_name": {
        "color": cl.color.white,
        "font_size": 20,
    },
    "Label::agent_desp": {
        "color": cl.color.white,
        "alignment": ui.Alignment.CENTER,
    },
    "Label::agent-combo-label": {"color": cl.combo_text_color},
    "Line": {
        "color": cl.line_color,
        "border_width": 2,
    },
    "Progress": {
        "background_color": cl.conversation_history_background_color,
        "border_radius": 1,
        "border_width": 1,
        "border_color": cl.color.white,
    },
    "Progress::highlight": {
        "background_color": cl.color.white,
    },
    "Rectangle.User": {
        "background_color": cl.conversation_history_background_color,
        "border_radius": 10,
    },
    "Rectangle.User.Buttons": {
        "background_color": cl.button_bg_color,
        "border_radius": 7,
        "border_width": 1,
        "border_color": cl.button_bg_border_color,
    },
    "Rectangle.Bot.ChatGPT": {
        "background_color": cl.bot_background_color,
    },
    "Rectangle::chat-history-highlight": {
        "background_color": cl.color.transparent,
    },
    "Rectangle::chat-history-highlight:checked": {
        "background_color": cl.base_background_color,
    },
    "Rectangle::chat-history-indicator": {
        "background_color": cl.history_highlight_color,
    },
    "Rectangle::code-background": {
        "background_color": cl.code_background_color,
        "border_radius": 4,
        "corner_flag": ui.CornerFlag.BOTTOM,
        "border_width": 1,
        "border_color": cl.code_border_color,
    },
    "Rectangle::code-header-background": {
        "background_color": cl.code_header_background_color,
        "border_radius": 4,
        "corner_flag": ui.CornerFlag.TOP,
    },
    "Rectangle::header_view_background": {"background_color": cl.base_background_color},
    "Rectangle::history": {
        "background_color": cl.conversation_history_background_color,
    },
    "ScrollingFrame::history": {
        "background_color": cl.conversation_history_background_color,
        "secondary_color": cl.color.gray,
        "scrollbar_size": 4,
    },
    "TreeView::chat": {
        "background_selected_color": ui.color.transparent
    },  # treeview item background color when the item is hovered
    "TreeView::chat:selected": {
        "background_color": ui.color.transparent
    },  # treeview item background color when the item is selected
    "TreeView::history": {
        "background_color": cl.color.transparent,
    },
    "TreeView::history:selected": {"background_color": cl.base_background_color},
    "Window": {"background_color": cl.base_background_color},
}

agent_combo_menu_style = {
    "background_color": cl.combo_background_color,
    "border_radius": 7,
    "shadow_color": cl.color.transparent,
    "Label::model": {
        "color": cl.combo_text_color,
        "font_size": 12,
        "font": f"{font_path}/NVIDIASans_Rg.ttf",
    },
    "Label::agent-name": {
        "color": cl.combo_text_color,
        "font_size": 16,
        "font": f"{font_path}/NVIDIASans_Rg.ttf",
    },
    "Label::agent-description": {
        "color": cl.combo_text_color,
        "font_size": 14,
        "font": f"{font_path}/NVIDIASans_Rg.ttf",
    },
    "Image::agent-active": {
        "image_url": f"{icon_path}/activeIconContainer.svg",
        "color": 0xFF00B976,
    },
    "TreeView": {
        "background_selected_color": ui.color.transparent
    },  # treeview item background color when the item is hovered
    "TreeView:selected": {
        "background_color": ui.color.transparent
    },  # treeview item background color when the item is selected
}
