## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from .modifiers.usd_knowledge_rag_modifier import USDKnowledgeRagModifier
from lc_agent import NetworkNode
from lc_agent import get_node_factory

from .nodes.usd_knowledge_node import USDKnowledgeNode
get_node_factory().register(USDKnowledgeNode)


class USDKnowledgeNetworkNode(NetworkNode):
    """
    A specialized knowledge tool focused exclusively on answering questions about OpenUSD (Universal Scene Description) 
    concepts, terminology, and functionality. This tool provides explanations and information about USD features 
    and concepts.

    IMPORTANT: This tool is strictly limited to USD knowledge questions and cannot:
    - Generate or explain code (use the code generation tool instead)
    - Answer questions about Python programming
    - Provide information about Omniverse
    - Answer questions about other 3D file formats or graphics systems

    Example appropriate questions:
    - "What is the difference between a USD Layer and a Stage?"
    - "Can you explain what USD composition arcs are?"
    - "What are the different types of USD variants?"
    - "How does USD handle references vs payloads?"
    - "What is the purpose of USD schemas?"
    """
    
    default_node: str = "USDKnowledgeNode"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_modifier(USDKnowledgeRagModifier())
