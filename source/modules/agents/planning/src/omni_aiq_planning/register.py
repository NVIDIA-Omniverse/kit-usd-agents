## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from lc_agent import RunnableSystemAppend
from lc_agent_aiq import LCAgentFunction
from lc_agent_aiq import replace_md_file_references
from aiq.data_models.component_ref import LLMRef
from pydantic import Field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PlanningConfig(FunctionBaseConfig, name="planning"):
    """Configuration for the planning agent."""

    name: str = "planning"
    system_message: Optional[str] = Field(
        default=None,
        description="Additional system message to append to the planning agent. The base system message is already included.",
    )
    llm_name: Optional[LLMRef] = Field(default=None, description="The LLM model to use with the multi-agent router")
    short_plan: bool = Field(
        default=False,
        description="Whether to generate short plans without details (only step titles). Default is False for detailed plans.",
    )
    add_details: bool = Field(
        default=False,
        description="Whether to add details to the plan. "
        "If True, the planning agent will be asked to provide details for each step dynamically. "
        "Default is False.",
    )


@register_function(config_type=PlanningConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def planning_function(config: PlanningConfig, builder: Builder):
    """Register the planning function with AIQ.

    This function creates detailed plans for scene creation and modification tasks.
    It acts as the project architect, detailing all objects, properties, and steps
    needed for scene modifications.
    """
    from omni_aiq_planning.nodes.planning_node import PlanningGenNode, PlanningNetworkNode

    system_message = replace_md_file_references(config.system_message)

    class ExtraPlanningGenNode(PlanningGenNode):
        """Extended planning generation node with optional additional system message."""

        def __init__(self, **kwargs):
            # Pass short_plan to parent class
            kwargs["short_plan"] = config.short_plan
            kwargs["add_details"] = config.add_details
            super().__init__(**kwargs)

            # Append additional system message if provided
            if system_message:
                self.inputs.append(RunnableSystemAppend(system_message=system_message))

    # Register the planning function
    yield LCAgentFunction(
        config=config,
        builder=builder,
        lc_agent_node_type=PlanningNetworkNode,
        lc_agent_node_gen_type=ExtraPlanningGenNode,
        hidden=True,
    )
