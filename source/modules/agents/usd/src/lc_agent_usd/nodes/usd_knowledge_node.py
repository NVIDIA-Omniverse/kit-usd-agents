## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from lc_agent import RunnableNode
from lc_agent import RunnableSystemAppend
from typing import Optional

from pathlib import Path
SYSTEM_PATH = Path(__file__).parent.joinpath("systems")


def read_md_file(file_path: str):
    with open(file_path, "r") as file:
        return file.read()


identity = read_md_file(f"{SYSTEM_PATH}/identity.md")
guardrails = read_md_file(f"{SYSTEM_PATH}/guardrails.md")

KNOWLEDGE_SYSTEM = f"""
{identity}

in general answer to the point and be concise, answer in few lines and ask if the user needs more information

if the user ask for details and follow up,
structure you answer with sections so they can easily ask for detail on a specific section one

Guardrails:
{guardrails}
"""


class USDKnowledgeNode(RunnableNode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.inputs.append(RunnableSystemAppend(system_message=KNOWLEDGE_SYSTEM))
