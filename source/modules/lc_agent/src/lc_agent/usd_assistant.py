## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from .runnable_node import RunnableNode
from .runnable_utils import RunnableSystemAppend


_SYSTEM = """
You are the Omniverse USD Assistant tool for Pixar USD.

You have to assist and guide the writing and debugging of Pixar USD code on Python.
"""


class USDAssistantNode(RunnableNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.inputs.append(RunnableSystemAppend(system_message=_SYSTEM))
