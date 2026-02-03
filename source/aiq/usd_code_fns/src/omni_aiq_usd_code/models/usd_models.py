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

"""Data models for USD-related structures."""

import json
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class USDCodeFeedback:
    """Represents feedback for USD code improvement."""

    title: str
    type: str  # "code" or "knowledge"
    finally_good_bad: str  # "good" or "bad"
    details: str
    query: str = ""
    timestamp: Optional[str] = None

    def __post_init__(self):
        """Validate and normalize feedback type and finally_good_bad."""
        # Normalize type to lowercase for case-insensitive validation
        if self.type:
            self.type = self.type.lower().strip()

        if self.type not in ["code", "knowledge"]:
            raise ValueError("Feedback type must be either 'code' or 'knowledge'")

        # Normalize finally_good_bad to lowercase for case-insensitive validation
        if self.finally_good_bad:
            self.finally_good_bad = self.finally_good_bad.lower().strip()

        if self.finally_good_bad not in ["good", "bad"]:
            raise ValueError("finally_good_bad must be either 'good' or 'bad'")

    @property
    def is_code_feedback(self) -> bool:
        """Check if this is code feedback."""
        return self.type == "code"

    @property
    def is_knowledge_feedback(self) -> bool:
        """Check if this is knowledge feedback."""
        return self.type == "knowledge"

    @property
    def is_good_feedback(self) -> bool:
        """Check if this is positive feedback."""
        return self.finally_good_bad == "good"

    @property
    def is_bad_feedback(self) -> bool:
        """Check if this is negative feedback."""
        return self.finally_good_bad == "bad"

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(
            {
                "title": self.title,
                "type": self.type,
                "finally_good_bad": self.finally_good_bad,
                "details": self.details,
                "query": self.query,
                "timestamp": self.timestamp,
            },
            indent=2,
        )
