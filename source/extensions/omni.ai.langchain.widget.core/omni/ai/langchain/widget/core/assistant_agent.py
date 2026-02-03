# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from lc_agent import RunnableNode, RunnableSystemAppend

_SYSTEM = """
You are the Omniverse Assistant tool.

You have to assist and guide the writing and debugging code in Python.
"""


class AssistantAgent(RunnableNode):
    """This is an example default agent for the AI widget. It is extends from  RunnableNode."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs.append(RunnableSystemAppend(system_message=_SYSTEM))
        self.metadata["description"] = "Generic AI Assistant"
        self.metadata["examples"] = ["Who are you in two words?"]
