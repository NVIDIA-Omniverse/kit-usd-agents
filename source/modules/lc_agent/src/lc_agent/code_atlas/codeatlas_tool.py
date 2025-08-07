# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from .codeatlas_cache import CodeAtlasCache
from .codeatlas_topic import CodeAtlasTopics
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from pydantic import Field
from typing import Optional, Type


class CodeAtlasInput(BaseModel):
    """
    Specification for the input data required by the Code Atlas.
    """

    lookup_type: str = Field(
        "MODULE",
        description="The type of entity to look up. "
        "This determines what to search for. Can be 'MODULE', 'CLASS', or 'USED_WITH'. "
        "MODULE: Look up a module by name. If the name is not specified, all modules are listed. "
        "CLASS: Look up a class by name. "
        "USED_WITH: Look up source code that is using the given class. ",
    )
    lookup_name: str = Field(
        "",
        description="The name of the module, class or class the methods are using based on lookup_type.",
    )
    classes: bool = Field(
        False,
        description="Flag to indicate if classes should be included. Without this flag on the classes will not be included to the modules.",
    )
    methods: bool = Field(
        False,
        description="Flag to indicate if methods should be included. Without this flag on the methods will not be included to the classes.",
    )
    full_source: bool = Field(
        False,
        description="Flag to indicate if method bodies with the source code should be included.",
    )
    docs: bool = Field(
        False, description="Flag to indicate if docstrings should be included."
    )


class CodeAtlasTool(BaseTool):
    """
    CodeAtlasTool is responsible for executing code snippets and providing information about the API of USD libraries.
    """

    name: str = "CodeAtlas"
    description: str = (
        "This tool is used to look up classes, methods, and modules in the {library_name} library. "
        "It can also be used to find source code that uses a specific class in the {library_name} library. "
        "The tool can be used to look up a module by name. "
        "The tool can be used to list all modules. "
        "It can also be used to look up a class by name. "
        "The tool can also be used to look up source code that is using the given class. "
        "Always check how to use the class in the real source code with USED_WITH before writing any code. "
        "Always do batch look up in Action Input with a list of dicts if it's possible. Don't ignore batch look up, it saves time."
    )
    args_schema: Type[CodeAtlasInput] = CodeAtlasInput  # Pydantic model for argument validation
    ask_human_input: bool = False

    cache: Optional[CodeAtlasCache] = None
    topic: Optional[CodeAtlasTopics] = None

    def _run(
        self,
        lookup_type: str = "MODULE",
        lookup_name: str = "",
        classes: bool = False,
        methods: bool = False,
        full_source: bool = False,
        docs: bool = False,
    ) -> str:
        if self.cache is None:
            return ""

        cache = self.cache

        if lookup_type == "MODULE" and lookup_name:
            return cache.lookup_module(
                lookup_name,
                classes=classes,
                methods=methods,
                method_bodies=full_source,
                docs=docs,
            )
        elif lookup_type == "CLASS" and lookup_name:
            fail_message = (
                f"Can't find the class `{lookup_name}`. "
                "Please try a different name or don't specify namespace."
            )
            return (
                cache.lookup_class(
                    lookup_name,
                    methods=methods,
                    method_bodies=full_source,
                    docs=docs,
                )
                or fail_message
            )
        elif lookup_type == "USED_WITH" and lookup_name:
            return cache.lookup_used_with(
                lookup_name,
                method_bodies=full_source,
                docs=docs,
            )
        elif lookup_type == "TOPIC" and lookup_name:
            if not self.topic or not self.topic.topics:
                return "Can't find any topic."

            topic_name = lookup_name.split(":")[0]
            topic = next((t for t in self.topic.topics if t.topic == topic_name), None)
            if not topic:
                return f"Topic {topic_name} is not available."

            return topic.content

        elif lookup_type == "MODULE" and not lookup_name:
            return "\n".join(
                [
                    cache.lookup_module(
                        module.full_name or module.name,
                        classes=classes,
                        methods=methods,
                        method_bodies=full_source,
                        docs=docs,
                    )
                    or ""
                    for _, module in cache._modules.items()
                ]
            )
