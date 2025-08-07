## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from lc_agent import NetworkModifier
from lc_agent import RunnableAINode
from lc_agent import RunnableNetwork
from lc_agent import RunnableNode
from .code_interpreter_modifier import remove_brackets

# pyright: reportUnusedExpression=false


def remove_main_block(code):
    """Removes the main block from the given code snippet."""
    # We need to remove this block because otherwise the code will not be
    # executed
    lines = code.split("\n")
    main_block_found = False
    main_block_indentation = None
    result = []

    for line in lines:
        if line.strip() in ['if __name__ == "__main__":', "if __name__ == '__main__':"]:
            main_block_found = True
        elif main_block_found:
            if main_block_indentation is None:
                main_block_indentation = len(line) - len(line.lstrip())
            if line.strip() and len(line) - len(line.lstrip()) >= main_block_indentation:
                result.append(line[main_block_indentation:])
        else:
            result.append(line)

    return "\n".join(result)


class CodePatcherModifier(NetworkModifier):
    def _patch_code(self, code):
        """Patches the given code snippet."""
        return remove_main_block(code)

    def on_post_invoke(self, network: RunnableNetwork, node: RunnableNode):
        if (
            # If it's a final result
            node.invoked
            and type(node) is RunnableAINode
            and not network.get_children(node)
            and "interpreter_code" not in node.metadata
            and "code_patcher" not in node.metadata
        ):
            code_snippet = remove_brackets(node.outputs.content)
            if not code_snippet:
                return

            patched_code = self._patch_code(code_snippet)

            if patched_code != code_snippet:
                # Mute this node, add a new one with output
                node.metadata["contribute_to_history"] = False
                node >> RunnableAINode(
                    ai_message=f"```python\n{patched_code}\n```",
                    name=str(type(self).__name__),
                    metadata={"code_patcher": True},
                )
