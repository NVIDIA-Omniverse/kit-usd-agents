## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from pathlib import Path
from typing import Optional

import lc_agent_usd
import usdcode
from lc_agent_usd import USDCodeGenNode
from lc_agent_usd.nodes.usd_meta_functions_parser import extract_function_signatures

from ..utils.chat_model_utils import sanitize_messages_with_expert_type


def read_md_file(file_path: str):
    with open(file_path, "r") as file:
        return file.read()


# Get the paths
SYSTEM_PATH = Path(__file__).parent.joinpath("systems")
LC_AGENT_USD_SYSTEM_PATH = Path(lc_agent_usd.__file__).parent.joinpath("nodes/systems")
METAFUNCTION_GET_PATH = Path(usdcode.__file__).parent.joinpath("usd_meta_functions_get.py")

identity = read_md_file(f"{SYSTEM_PATH}/scene_info_identity.md")
task = read_md_file(f"{SYSTEM_PATH}/scene_info_task.md")
selection = read_md_file(f"{SYSTEM_PATH}/scene_info_selection.md")
examples = read_md_file(f"{SYSTEM_PATH}/scene_info_examples.md")
instructions = read_md_file(f"{SYSTEM_PATH}/scene_info_instructions.md")

metafunctions = read_md_file(f"{LC_AGENT_USD_SYSTEM_PATH}/usd_code_interactive_metafunctions.md")
metafunction_get = extract_function_signatures(f"{METAFUNCTION_GET_PATH}")

system_message2 = f"""
{identity}
{task}
{selection}
{metafunctions}
{metafunction_get}
{examples}
{instructions}
"""


class SceneInfoGenNode(USDCodeGenNode):
    system_message: Optional[str] = system_message2

    def _sanitize_messages_for_chat_model(self, messages, chat_model_name, chat_model):
        """Sanitizes messages and adds metafunction expert type for scene information."""
        messages = super()._sanitize_messages_for_chat_model(messages, chat_model_name, chat_model)
        return sanitize_messages_with_expert_type(messages, "metafunction")
