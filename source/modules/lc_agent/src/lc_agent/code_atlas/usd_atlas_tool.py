## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from .codeatlas_tool import CodeAtlasTool
from .codeatlas_cache import CodeAtlasCache
from .codeatlas_topic import CodeAtlasTopics
import os


class USDAtlasTool(CodeAtlasTool):
    name: str = "USDAtlasTool"
    description: str = (
        "This toolkit serves as a comprehensive resource for exploring classes, methods, and modules within the Pixar USD API. "
        "It facilitates the discovery of code snippets that utilize specific USD classes. "
        "Users may query the toolkit to retrieve information on a particular USD module or enumerate all available modules. "
        "Additionally, it provides the means to search for a class by name. "
        "Further capabilities include pinpointing code instances where the queried class is employed. "
        "Prior to commencing any USD scripting, it is recommended to examine usage patterns of classes with the USED_WITH function. "
        "For efficiency, batch queries should be conducted in the Action Input using a sequence of dictionary objects when feasible. "
        "Do not overlook the advantages of batch querying, as it is a time-efficient practice."
    )

    def __init__(self):
        super().__init__()

        current_dir = os.path.dirname(os.path.realpath(__file__))

        # Full source code of the USD library
        self.cache = CodeAtlasCache()
        self.cache.load(f"{current_dir}/../data/usd_atlas.json")

        # Topics are like a mini wiki for the tool
        self.topic = CodeAtlasTopics()
        self.topic.load(f"{current_dir}/../data/usd_atlas.md")
