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

from typing import Optional

import carb.settings
from lc_agent import NetworkNode
from lc_agent_usd import CodeExtractorModifier, NetworkLenghtModifier, USDCodeGenPatcherModifier
from lc_agent_usd.modifiers.code_interpreter_modifier import CodeInterpreterModifier

MAX_RETRIES_SETTINGS = "/exts/omni.ai.langchain.agent.usd_code/max_retries"
MAX_RETRIES_DEFAULT = 3


class USDCodeNetworkNode(NetworkNode):
    default_node: str = "USDCodeNode"

    def __init__(self, snippet_verification=False, max_retries: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)

        if max_retries is None:
            max_retries = carb.settings.get_settings().get(MAX_RETRIES_SETTINGS) or MAX_RETRIES_DEFAULT

        self.add_modifier(NetworkLenghtModifier(max_length=max_retries))
        self.add_modifier(CodeExtractorModifier(snippet_verification=snippet_verification))
        # self.add_modifier(USDCodeGenPatcherModifier())
        self.add_modifier(CodeInterpreterModifier())

        self.metadata["description"] = "Agent to answer USD knowledge and generate Python USD code."
        self.metadata["examples"] = [
            "How to create a new layer?",
            "How do I read attributes from a prim?",
            "Show me how to traverse a stage?",
        ]
