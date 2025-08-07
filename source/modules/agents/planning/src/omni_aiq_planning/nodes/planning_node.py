## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from pathlib import Path
from typing import List
from lc_agent import NetworkNode, RunnableNode, RunnableSystemAppend, RunnableNetwork
from omni_aiq_planning.modifiers.planning_modifier import PlanningModifier

SYSTEM_PATH = Path(__file__).parent.joinpath("systems")


def read_md_file(file_path: str):
    with open(file_path, "r") as file:
        return file.read()


PLANNING_GEN_SYSTEM_PATH = SYSTEM_PATH.joinpath("planning_gen_system.md")
PLANNING_GEN_SYSTEM = read_md_file(str(PLANNING_GEN_SYSTEM_PATH))


class PlanningGenNode(RunnableNode):
    """
    Node responsible for creating detailed plans for complex task execution.

    This node creates comprehensive, step-by-step plans that include:
    - All steps and actions required to complete the task
    - Specific requirements and specifications for each step
    - Dependencies and relationships between steps
    - Optimal execution sequence and timing
    """

    def __init__(self, short_plan: bool = False, add_details: bool = False, **kwargs):
        super().__init__(**kwargs)

        # Process system message based on short_plan option
        system_message = PLANNING_GEN_SYSTEM
        if short_plan:
            # For short plans: keep content within <short> tags, remove everything else including <long> content
            import re

            # Remove content within <long> tags
            system_message = re.sub(r"<long>.*?</long>", "", system_message, flags=re.DOTALL)
            # Remove <short> tags but keep their content
            system_message = system_message.replace("<short>", "").replace("</short>", "")
            # Add instruction to produce short plans
            system_message = (
                "**IMPORTANT: Generate SHORT plans only. Include ONLY step titles without any details, bullet points, or success criteria.**\n\n"
                + system_message
            )
        else:
            # For detailed plans: remove content within <short> tags, keep <long> content
            import re

            # Remove content within <short> tags
            system_message = re.sub(r"<short>.*?</short>", "", system_message, flags=re.DOTALL)
            # Remove <long> tags but keep their content
            system_message = system_message.replace("<long>", "").replace("</long>", "")

        # Add system message from external file
        self.inputs.append(RunnableSystemAppend(system_message=system_message))

        if add_details:
            self.metadata["add_details"] = True


class PlanningNetworkNode(NetworkNode):
    """
    Tool for creating detailed plans for complex tasks.

    ## Use Case: Strategic Task Planning

    This tool creates comprehensive, detailed plans for complex multi-step tasks.
    It acts as a strategic planner, breaking down complex goals into manageable,
    actionable steps with clear dependencies and execution sequences.

    The planning system is designed to:
    - Break down user requests into specific, actionable steps
    - Identify all components, resources, or entities involved
    - Define exact requirements and specifications for each step
    - Establish dependencies and relationships between steps
    - Create an optimal execution sequence
    - Provide success criteria and validation points

    ## When to Use This Tool:

    Use this tool when the user request involves:
    1. Complex multi-step processes requiring coordination
    2. Tasks with multiple components or dependencies
    3. Projects requiring detailed specifications and sequencing
    4. Any task requiring detailed planning before execution
    5. Workflows where order of operations is critical
    6. Tasks that need clear success criteria and validation
    7. ANY complex task - this should be the first step for structured execution

    ## Capabilities:

    This tool creates detailed plans that include:
    - List of all steps, actions, and components involved
    - Specific requirements and specifications for each step
    - Clear dependencies and relationships between steps
    - Optimal execution sequence and timing
    - Success criteria and validation points
    - Resource requirements and constraints
    - Risk mitigation strategies when applicable

    ## Examples:

    - "Deploy a multi-service application with database setup and API configuration"
    - "Create a comprehensive testing strategy for a software release"
    - "Plan a data migration from legacy system to new platform"
    - "Design a workflow for processing customer orders"
    - "Build a content creation pipeline with review stages"
    - "Organize a multi-phase project with team coordination"
    """

    default_node: str = "PlanningGenNode"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Add the planning modifier
        self.add_modifier(PlanningModifier())
