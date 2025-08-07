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
from importlib.util import find_spec
import usdcode
import functools

# Replace the import and path definitions
# Remove: import usdcode as mf
mf_spec = find_spec("usdcode")
mf_package_path = Path(mf_spec.origin).parent

# Update the path definitions
SYSTEM_PATH = Path(__file__).parent.joinpath("systems")
METAFUNCTION_SET_PATH = mf_package_path.joinpath("usd_meta_functions_set.py")
METAFUNCTION_GET_PATH = mf_package_path.joinpath("usd_meta_functions_get.py")

from .usd_code_gen_node import USDCodeGenNode
from .usd_meta_functions_parser import extract_module_functions


def read_md_file(file_path: str):
    with open(file_path, "r") as file:
        return file.read()


@functools.lru_cache(maxsize=1)
def get_usd_code_interactive_system_message():
    """
    Generates and caches the system message for USD code interactive node.
    This function is only called once per session due to lru_cache.
    """
    identity = read_md_file(f"{SYSTEM_PATH}/usd_code_interactive_identity.md")
    code_structure = read_md_file(f"{SYSTEM_PATH}/usd_code_interactive_code_structure.md")
    selection = read_md_file(f"{SYSTEM_PATH}/usd_code_interactive_selection.md")
    examples = read_md_file(f"{SYSTEM_PATH}/usd_code_interactive_examples.md")
    final_instructions = read_md_file(f"{SYSTEM_PATH}/usd_code_interactive_final_instructions.md")
    lc_agent_usd_final_instructions = read_md_file(f"{SYSTEM_PATH}/final_instructions.md")
    guardrails = read_md_file(f"{SYSTEM_PATH}/usd_code_interactive_guardrails.md")

    metafunctions = read_md_file(f"{SYSTEM_PATH}/usd_code_interactive_metafunctions.md")
    metafunction_set = extract_module_functions(usdcode)

    return f"""
{identity}

{code_structure}

{selection}

{metafunctions}

# The following functions are available for you to use in the module usdcode:

{metafunction_set}

# Here are some examples of code snippets:

{examples}

# Final Instructions:
{final_instructions}
{lc_agent_usd_final_instructions}

# Guardrails:
{guardrails}
"""


class USDCodeInteractiveNode(USDCodeGenNode):
    system_message: Optional[str] = None

    def __init__(self, default_prim_path="/World", up_axis="Y", selection="no", **kwargs):
        # We need to dynamically replace all the "{default_prim}" with the real default prim path
        if "system_message" not in kwargs:
            # Get the cached system message template
            system_message = get_usd_code_interactive_system_message()

            # Apply custom replacements
            system_message = (
                system_message.replace("{default_prim}", default_prim_path)
                .replace("{up_axis}", up_axis.lower())
                .replace("{selection}", selection)
            )

            kwargs["system_message"] = system_message

        super().__init__(**kwargs)
