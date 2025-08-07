# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from pydantic import BaseModel
from typing import List
import re
# import carb.tokens


class TopicSection(BaseModel):
    topic: str
    description: str
    content: str


class CodeAtlasTopics:
    def __init__(self):
        self._topics: List[TopicSection] = []

    @property
    def topics(self):
        return self._topics

    def load(self, file_path):
        # token = carb.tokens.get_tokens_interface()
        # file_path = token.resolve(file_path)

        with open(file_path, 'r') as md_file:
            md_content = md_file.read()

        sections = re.split(r"\n(?=# )", md_content)

        for section in sections:
            if not section.strip():
                continue  # Skip empty sections

            heading_and_rest = section.split("\n", 1)
            topic = heading_and_rest[0].lstrip("# ").strip()
            rest = heading_and_rest[1] if len(heading_and_rest) > 1 else ""

            # Find the end of the description (before an empty line or another heading)
            description_end_match = re.search(r"\n(\s*\n|# )", rest)

            # Extract description and remaining content
            if description_end_match:
                description = rest[: description_end_match.start()].strip()
                content = rest[description_end_match.end() :].strip()
            else:
                description = rest.strip()
                content = ""

            self._topics.append(TopicSection(topic=topic, description=description, content=content))
