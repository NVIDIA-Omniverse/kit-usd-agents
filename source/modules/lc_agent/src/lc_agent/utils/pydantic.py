## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""
Pydantic compatibility layer for LC Agent.

This module provides a compatibility layer for Pydantic imports, allowing the code to work
with both old and new versions of LangChain. The old version of LangChain uses
`langchain_core.pydantic_v1` while the new version requires direct imports from `pydantic`.

Usage:
    from lc_agent.utils.pydantic import BaseModel, Field, PrivateAttr, validator
"""

import importlib.util
import sys
from typing import Any, Dict, List, Optional, Type, Union, Callable

USE_PYDANTIC_V1 = False

# Import from the appropriate source
if USE_PYDANTIC_V1:
    from langchain_core.pydantic_v1 import (
        BaseModel,
        Field,
        PrivateAttr,
        validator,
        root_validator,
        create_model,
        ValidationError,
        Extra,
    )
else:
    from pydantic import (
        BaseModel,
        Field,
        PrivateAttr,
        validator,
        root_validator,
        create_model,
        ValidationError,
        Extra,
    )

# Export the Config class for compatibility
if USE_PYDANTIC_V1:
    from langchain_core.pydantic_v1 import BaseConfig as Config
else:
    from pydantic import ConfigDict

    # Create a compatibility wrapper for Config
    class Config:
        """Compatibility wrapper for Pydantic Config"""
        @classmethod
        def with_options(cls, **kwargs):
            return ConfigDict(**kwargs)

# Add any additional compatibility functions or classes here
def is_using_pydantic_v1() -> bool:
    """
    Returns True if using langchain_core.pydantic_v1, False if using pydantic directly.
    
    This can be useful for conditional logic in your code if needed.
    """
    return USE_PYDANTIC_V1 
