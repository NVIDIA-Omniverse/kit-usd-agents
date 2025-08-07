## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from collections import defaultdict
from lc_agent import RunnableNetwork, RunnableNode
from lc_agent_rag_modifiers import RetrieverMessage, SystemRagModifier
from typing import Literal

# RAG content explaining the purpose of the usdcode module
RAG_CONTENT = """
The usdcode module contains metafunctions that can be used to perform various operations on the USD stage.

Always use them to save time and resources.
"""

# Usage instructions for metafunctions
METAFUNCTION_USAGE = """
You can call these metafunctions directly from the `usdcode` module that is pre-imported.
Don't call it like this: `{group}.{method_name}` as it's not a part of USD.
Call it like this: `usdcode.MF{group}.{method_name}`.
"""


class MFRetrieverMessage(RetrieverMessage):
    """
    Custom RetrieverMessage for metafunctions, handling metadata and formatting.
    """

    name: Literal["mf_retriever_message"] = "mf_retriever_message"

    def __init__(self, metadata, **kwargs):
        super().__init__(**kwargs)
        # Store metadata as a private attribute
        object.__setattr__(self, "_metadata", metadata)

    def _format_message(self, retriever_results):
        """
        Format the retriever results into a structured message.

        Args:
            retriever_results: Results from the retriever.

        Returns:
            str: Formatted message containing metafunction information.
        """
        groups = {}
        already_have = defaultdict(set)

        if "retreived_metafunctions" not in self._metadata:
            self._metadata["retreived_metafunctions"] = []
        retreived_metafunctions = self._metadata["retreived_metafunctions"]

        for rag_result in retriever_results or []:
            retreived_metafunctions.append(rag_result.metadata)

        combined_metafunctions = self._combine_metafunctions(retreived_metafunctions)
        retreived_metafunctions[:] = combined_metafunctions

        examples = []
        methods = []

        for idx, rag_result in enumerate(retreived_metafunctions):
            group, method_name = self._process_rag_result(rag_result, already_have, groups)
            methods.append((group, method_name))
            example = self._format_example(rag_result, idx, group, method_name)
            examples.append(example)

        rag_content = self._generate_rag_content(groups)
        examples_text = "There are examples of how to use metafunctions:\n\n" + "\n".join(examples)
        result = (
            RAG_CONTENT + rag_content + examples_text + METAFUNCTION_USAGE.format(group=group, method_name=method_name)
        )

        self._debug_print_methods(result, methods)

        return result

    def _process_rag_result(self, rag_result, already_have, groups):
        group = rag_result.get("class_name")
        if not group:
            return None, None

        method_name = rag_result.get("method_name")
        if method_name in already_have[group]:
            return None, None

        already_have[group].add(method_name)
        group = group.split(".")[1]

        if group not in groups:
            groups[group] = f"class MF{group}:\n"

        metafunction_signature = f"  @staticmethod\n  {rag_result.get('signature')}\n"
        groups[group] += metafunction_signature

        return group, method_name

    def _format_example(self, rag_result, idx, group, method_name):
        example = rag_result.get("test")
        example_lines = [
            e
            for e in example.splitlines()
            if "stage =" not in e
            and "Usd.Stage.CreateInMemory()" not in e
            and not (e.strip().startswith("#") and "test" in e.lower())
        ]
        example = "\n".join(example_lines)
        example = example.replace(f" {method_name}(", f" usdcode.MF{group}.{method_name}(")
        example = example.replace(f"\n{method_name}(", f"\nusdcode.MF{group}.{method_name}(")
        example = example.replace(f"({method_name}(", f"(usdcode.MF{group}.{method_name}(")
        example = example.replace(f"[{method_name}(", f"[usdcode.MF{group}.{method_name}(")
        example = example.replace(f"{{{method_name}(", f"{{usdcode.MF{group}.{method_name}(")

        return f"Example {idx + 1}:\n\n{example}\n"

    def _generate_rag_content(self, groups):
        return "".join(f"{group_signature}\n" for group_signature in groups.values())

    def _debug_print_methods(self, result, methods):
        if False:  # Set to True for debugging
            print("RAG {")
            for group, method_name in methods:
                print(f"  {group}.{method_name}")
            if False:
                for line in result.splitlines():
                    print(f"  {line}")
            print("}")

    def _combine_metafunctions(self, metafunctions):
        combined = set()
        result = []
        for mf in metafunctions:
            key = (mf["class_name"], mf["method_name"])
            if key not in combined:
                combined.add(key)
                result.append(mf)
        return result


class MFRagModifier(SystemRagModifier):
    """
    RAG modifier for metafunctions, injecting relevant information into the network.
    """

    def __init__(self, retriever_name="usd_metafunctions", **kwargs):
        super().__init__(retriever_name=retriever_name, **kwargs)

    def _inject_rag(self, network: RunnableNetwork, node: RunnableNode, question: str):
        # Collect retreived_metafunctions from parents and their ancestors
        parent_metafunctions = self._collect_parent_metafunctions(network, node)

        # Combine parent metafunctions (removing duplicates)
        if "retreived_metafunctions" not in node.metadata:
            node.metadata["retreived_metafunctions"] = []
        combined_metafunctions = self._combine_metafunctions(
            node.metadata["retreived_metafunctions"] + parent_metafunctions
        )

        # Update node's metadata with combined metafunctions
        node.metadata["retreived_metafunctions"][:] = combined_metafunctions

        # Create MFRetrieverMessage with reference to node's metadata
        retriever_message = MFRetrieverMessage(
            metadata=node.metadata,
            question=question,
            retriever_name=self._retriever_name,
            type="system",
            top_k=self._top_k,
            max_tokens=self._max_tokens,
        )

        # Insert RAG content into the system prompt
        node.inputs.insert(1, retriever_message)

    def _collect_parent_metafunctions(self, network: RunnableNetwork, node: RunnableNode):
        all_metafunctions = []
        visited = set()

        def collect_from_ancestors(current_node):
            if current_node in visited:
                return
            visited.add(current_node)

            if hasattr(current_node, "metadata") and "retreived_metafunctions" in current_node.metadata:
                all_metafunctions.extend(current_node.metadata["retreived_metafunctions"])

            parents = network.get_parents(current_node)
            if parents:
                for parent in parents:
                    collect_from_ancestors(parent)

        collect_from_ancestors(node)
        return all_metafunctions

    def _combine_metafunctions(self, metafunctions):
        combined = set()
        result = []
        for mf in metafunctions:
            key = (mf["class_name"], mf["method_name"])
            if key not in combined:
                combined.add(key)
                result.append(mf)
        return result
