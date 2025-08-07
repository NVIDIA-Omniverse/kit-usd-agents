## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import uuid


def get_id_string() -> str:
    """
    Generate a UUID string
    """
    return str(uuid.uuid4())


class UUIDMixin:
    def uuid(self) -> str:
        """return the unique identifier for the mode otheriwe create one"""
        uuid = self.metadata.get("uuid", None)
        if not uuid:
            # we create one
            uuid = get_id_string()
            self.metadata["uuid"] = uuid

        return uuid
