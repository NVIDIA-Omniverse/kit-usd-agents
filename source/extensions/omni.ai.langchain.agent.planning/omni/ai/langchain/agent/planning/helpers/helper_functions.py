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

"""
Helper functions for the planning agent.

This module provides utility functions for plan management and tracking.
"""


def get_current_plan(stage_metadata):
    """
    Get the current plan from stage metadata if available.

    Args:
        stage_metadata: The stage metadata to check

    Returns:
        dict: The current plan or None if not found
    """
    return stage_metadata.get("current_plan")


def check_plan_step_status(stage_metadata, step_id):
    """
    Check the status of a specific plan step.

    Args:
        stage_metadata: The stage metadata to check
        step_id: The step ID to check

    Returns:
        str: The step status or "unknown" if not found
    """
    plan = get_current_plan(stage_metadata)
    if plan and "status" in plan and step_id in plan["status"]:
        return plan["status"][step_id]
    return "unknown"


def update_plan_step_status(stage_metadata, step_id, status):
    """
    Update the status of a plan step.

    Args:
        stage_metadata: The stage metadata to update
        step_id: The step ID to update
        status: The new status (pending, in_progress, completed, failed)

    Returns:
        bool: True if successfully updated, False otherwise
    """
    plan = get_current_plan(stage_metadata)
    if not plan:
        return False

    if "status" not in plan:
        plan["status"] = {}

    plan["status"][step_id] = status
    return True
