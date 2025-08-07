## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import re
from langchain_core.messages import SystemMessage
from lc_agent import RunnableNode
from pathlib import Path
from typing import Optional

SYSTEM_PATH = Path(__file__).parent.joinpath("systems")


def read_md_file(file_path: str):
    with open(file_path, "r") as file:
        return file.read()


identity = read_md_file(f"{SYSTEM_PATH}/identity.md")
code_structure = read_md_file(f"{SYSTEM_PATH}/code_structure.md")
snippet_structure = read_md_file(f"{SYSTEM_PATH}/snippet_structure.md")
code_verification = read_md_file(f"{SYSTEM_PATH}/code_verification.md")
final_instructions = read_md_file(f"{SYSTEM_PATH}/final_instructions.md")
guardrails = read_md_file(f"{SYSTEM_PATH}/guardrails.md")
common_mistakes = read_md_file(f"{SYSTEM_PATH}/common_mistakes.md")
code_fixer_tool = read_md_file(f"{SYSTEM_PATH}/code_fixer.md")


class USDCodeGenNode(RunnableNode):
    system_message: Optional[str] = None

    def __init__(self, enable_code_fixer=False, use_code_verification=False, **kwargs):
        super().__init__(**kwargs)

        tools = ""
        if enable_code_fixer:
            tools = f"You have access to few Tools:\n{code_fixer_tool}"

        # Check if code verification is enabled
        if use_code_verification:
            code_verification_prompt = code_verification
        else:
            code_verification_prompt = ""

        usd_code_system = f"""
        {identity}

        In general answer to the point and be concise, with just few comments like that
        {code_structure}

        If you are ask to write a function or snippet use the following template:
        {snippet_structure}

        {code_verification_prompt}

        {tools}

        Here is a list of the common pitfall to avoid:
        {common_mistakes}

        Final Instructions:
        {final_instructions}

        Guardrails:
        {guardrails}
        """

        # Check if a system message is provided
        if self.system_message:
            pass
        else:
            self.system_message = usd_code_system

    async def _acombine_inputs(self, input, config, parents_result, **kwargs):
        # Get the combined inputs from parent class
        result = await super()._acombine_inputs(input, config, parents_result, **kwargs)

        if not self.system_message:
            return result

        iterated = set()
        for p in self._iterate_chain(iterated):
            if p is self:
                continue

            parent_outputs = p.outputs
            if isinstance(parent_outputs, SystemMessage):
                return result
            elif isinstance(parent_outputs, list):
                for o in parent_outputs:
                    if isinstance(o, SystemMessage):
                        return result

        if not result:
            return [SystemMessage(content=self.system_message)]
        elif isinstance(result[0], SystemMessage):
            system_copy = result[0].copy()
            result[0] = system_copy
            system_copy.content = f"{self.system_message}\n\n{system_copy.content}"
            return result
        # else:

        # Add the system message at the start of the sequence
        return [SystemMessage(content=self.system_message), *result]
