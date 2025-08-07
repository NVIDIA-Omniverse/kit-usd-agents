# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from pydantic import BaseModel, Field
from typing import List
from typing import Optional


class CodeAtlasObjectInfo(BaseModel):
    name: str = Field(..., description="Name of the Python object")
    full_name: Optional[str] = Field(None, description="Fully qualified object name")
    docstring: Optional[str] = Field(None, description="Docstring of the module")
    line_number: Optional[int] = Field(None, description="Line number where the object is defined in the source code")


# Model for storing information about a module
class CodeAtlasModuleInfo(CodeAtlasObjectInfo):
    """Data model for storing Python module information."""

    file_path: str = Field(..., description="File system path to Python module file")
    class_names: List[str] = Field([], description="List of class names in the module")
    function_names: List[str] = Field([], description="List of public function names included in the module")
    equivelant_modules: List[str] = Field([], description="List of equivelant modules") # e.g. omni.ui.scene == omni.ui_scene
    extension_name: Optional[str] = Field(None, description="Kit extension name of the module, only set on the root module")


# TODO: split it
class CodeAtlasArgumentInfo(CodeAtlasObjectInfo):
    """Data model for storing Python argument information of a method or function."""

    type_annotation: Optional[str] = Field(None, description="Type annotation of the argument")
    default_value: Optional[str] = Field(None, description="Default value of the argument")
    is_variadic: bool = Field(False, description="Indicator if argument is variadic (*args or **kwargs)")
    parent_method: Optional[str] = Field(None, description="Name of the method or function this argument belongs to")


class CodeAtlasMethodInfo(CodeAtlasObjectInfo):
    """Data model for storing Python method or function information."""

    module_name: str = Field(..., description="Name of the module containing the class")
    parent_class: Optional[str] = Field(None, description="Containing class if method is part of a class")
    return_type: Optional[str] = Field(None, description="Return type of the method or function")
    arguments: List[CodeAtlasArgumentInfo] = Field([], description="List of arguments for the method or function")
    is_class_method: bool = Field(False, description="Indicator if method is a class method")
    is_static_method: bool = Field(False, description="Indicator if method is a static method")
    is_async_method: bool = Field(False, description="Indicator if method is an async method")
    decorators: List[str] = Field([], description="List of decorators applied to the method")
    source_code: Optional[str] = Field(None, description="The full source code of the method")
    class_usages: List[str] = Field([], description="List of fully qualified class names used in the method")


class CodeAtlasClassInfo(CodeAtlasObjectInfo):
    """Data model for storing Python class information."""

    module_name: str = Field(..., description="Name of the module containing the class")
    methods: List[str] = Field([], description="List of methods in the class")
    class_variables: List[str] = Field([], description="List of class variables")
    parent_classes: List[str] = Field([], description="List of parent classes this class inherits from")
    decorators: List[str] = Field([], description="List of decorators applied to the class")
