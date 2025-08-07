## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import re

from lc_agent import NetworkModifier
from lc_agent import RunnableAINode
from lc_agent import RunnableHumanNode
from lc_agent import NetworkNode

from .code_interpreter_modifier import CodeInterpreterModifier


class CodeFixerModifier(NetworkModifier):
    def on_post_invoke(self, network: "RunnableNetwork", node: "RunnableNode"):
        # check if the node has the FixCode instruction
        if "@FixCode" not in node.outputs.content:
            return

        output = node.outputs.content
        # check if we need to fix the code
        matches = re.findall(r"@FixCode\((.*?)<\*\*>(.*?)\)@EndFixCode", output, re.DOTALL)

        if not matches:
            # no matches found
            # we still need to add a node to follow up
            node >> RunnableHumanNode(
                human_message="the 'FixCode' instruction you provided is not correct, I didn't find any matches."
            )
            return

        # now we need to find the code in parent nodes
        code = None
        current_node = node
        parent = None
        original_code_node = None
        while True:
            parent = network.get_parents(current_node)[0]
            if not parent or parent not in network.nodes:  # NetworkNode are connected to their parent
                # we are at the root node of the network
                break

            if "interpreter_code" in parent.metadata:
                original_code_node = parent
                code = parent.metadata["interpreter_code"]
                break
            current_node = parent

        if not code:
            # we can't do anything without the code but it is unexpected
            node >> RunnableHumanNode(human_message="CodeFixer could not find code in parent nodes")
            return

        # iterate over the matches and replace the code
        for source_text, new_text in matches:
            code = code.replace(source_text, new_text)

        new_message = f"""I have updated the code hope, let's see if it works now.\n```python\n{code}\n```"""

        new_node = RunnableHumanNode(
            human_message=new_message, name=str(type(self).__name__), metadata={}, invoked=True
        )

        # we need to remove the code from the broken node from the history
        original_code_node.metadata["contribute_to_history"] = False

        if isinstance(network, NetworkNode):
            network.outputs.content = new_message

        metadata = {"contribute_to_history": False, "contribute_to_ui": False}

        extracted_node = RunnableAINode(
            ai_message=code,
            name=str(type(self).__name__),
            metadata=metadata,
        )
        # Now we insert the new Nodes in the network
        node >> new_node >> extracted_node

        # Now we run the code interpreter
        CodeInterpreterModifier().on_post_invoke(network, extracted_node)
