## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from ..nodes.usd_code_gen_node import USDCodeGenNode
from ..nodes.judge_node import JUDGE_FIX_PROMPT
from ..nodes.judge_node import JUDGE_PROMPT
from ..nodes.judge_node import JudgeNode
from lc_agent import NetworkModifier
from lc_agent import RunnableAINode
from lc_agent import RunnableHumanNode
from lc_agent import get_node_factory
from typing import Optional

# pyright: reportUnusedExpression=false


def extract_score(feedback: str):
    words = feedback.replace("**", "").split()

    for word in words:
        # Try to find a number at the beginning of the word
        for i, char in enumerate(word):
            if not char.isdigit() and char != ".":
                break
        possible_score = word[:i] if not char.isdigit() else word[: i + 1]

        # Check if the extracted part is a valid number
        if possible_score.replace(".", "", 1).isdigit():
            # Keep only the integer part if it's a decimal
            score = int(float(possible_score))
            return score

    # Return an error if no score found
    return None


class JudgeModifier(NetworkModifier):
    def __init__(self, judge_model_name: Optional[str]):
        self._judge_model_name = judge_model_name

    def on_post_invoke(self, network: "RunnableNetwork", node: "RunnableNode"):
        if (
            # If we need a judge
            node.invoked
            and type(node) is RunnableAINode
            and not network.get_children(node)
            and "interpreter_ok" in node.metadata
        ):
            # Get the question, code, and result
            parent = node
            while parent:
                if (
                    type(parent) is RunnableHumanNode
                    and "interpreter_error" not in parent.metadata
                ):
                    break
                parent = parent.parents[0] if parent.parents else None
            question = parent.outputs.content
            code = node.metadata["interpreter_code"]
            result = node.metadata["interpreter_result"]

            prompt = (
                JUDGE_PROMPT.replace("{question}", question)
                .replace("{code}", code)
                .replace("{execution_result}", result)
            )
            # Create a parallel branch to judge the code and the question
            #
            #  RunnableHumanNode
            #         |
            #  USDCodeGenNode[node]   RunnableHumanNode
            #                              |
            #                           JudgeNode
            #
            (
                None
                >> RunnableHumanNode(human_message=prompt)
                >> JudgeNode(chat_model_name=self._judge_model_name)
            )
        elif (
            # If it's called after judge
            node.invoked
            and type(node) is JudgeNode
            and not network.get_children(node)
        ):
            result: str = node.outputs.content
            score = extract_score(result)

            new_parents = network.get_leaf_nodes()

            # Mute judge and JUDGE_PROMPT
            for n in new_parents:
                n.metadata["contribute_to_history"] = False
            for p in node.parents:
                p.metadata["contribute_to_history"] = False

            if score and score >= 7:
                # Good. Finish the conversation.
                #
                #  RunnableHumanNode
                #         |
                #  USDCodeGenNode[M]
                #         |
                #          \            RunnableHumanNode[M]
                #           \               /
                #            JudgeNode [M][node]
                new_parents = [n for n in new_parents if n != node]
                node._add_parent(new_parents[-1])
                return
            else:
                #  USDCodeGenNode
                #         |
                #  USDCodeGenNode[M] RunnableHumanNode[M]
                #         |                  |
                #          \            JudgeNode[M][node]
                #           \               /
                #           RunnableHumanNode
                new_parents >> RunnableHumanNode(
                    human_message=JUDGE_FIX_PROMPT.replace("{judge}", result)
                )
