## Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""Tests for USD tool register-layer input schemas."""

import sys
from pathlib import Path

import pytest

# Add the src directory to the path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from omni_aiq_usd_code.register_get_usd_class_detail import (  # noqa: E402
    GetUSDClassDetailInput,
    _parse_class_names_input,
)
from omni_aiq_usd_code.register_get_usd_method_detail import (  # noqa: E402
    GetUSDMethodDetailInput,
    _parse_method_names_input,
)
from omni_aiq_usd_code.register_get_usd_module_detail import (  # noqa: E402
    GetUSDModuleDetailInput,
    _parse_module_names_input,
)


def _field_schema(model_cls, field_name):
    schema = model_cls.model_json_schema()
    return schema["properties"][field_name]


@pytest.mark.parametrize(
    ("model_cls", "field_name"),
    [
        (GetUSDModuleDetailInput, "module_names"),
        (GetUSDClassDetailInput, "class_names"),
        (GetUSDMethodDetailInput, "method_names"),
    ],
)
def test_detail_register_schemas_accept_native_arrays(model_cls, field_name):
    field_schema = _field_schema(model_cls, field_name)

    assert any(branch.get("type") == "array" for branch in field_schema["anyOf"])
    assert any(branch.get("type") == "string" for branch in field_schema["anyOf"])


def test_module_names_accept_native_list_and_json_array_string():
    assert (
        _parse_module_names_input(GetUSDModuleDetailInput(module_names=["Usd", "UsdGeom"]).module_names)
        == "Usd,UsdGeom"
    )
    assert (
        _parse_module_names_input(GetUSDModuleDetailInput(module_names='["Usd", "UsdGeom"]').module_names)
        == "Usd,UsdGeom"
    )


def test_class_names_accept_native_list_and_json_array_string():
    assert (
        _parse_class_names_input(GetUSDClassDetailInput(class_names=["UsdStage", "UsdPrim"]).class_names)
        == "UsdStage,UsdPrim"
    )
    assert (
        _parse_class_names_input(GetUSDClassDetailInput(class_names='["UsdStage", "UsdPrim"]').class_names)
        == "UsdStage,UsdPrim"
    )


def test_method_names_accept_native_list_and_json_array_string():
    assert (
        _parse_method_names_input(GetUSDMethodDetailInput(method_names=["GetPrim", "CreatePrim"]).method_names)
        == "GetPrim,CreatePrim"
    )
    assert (
        _parse_method_names_input(
            GetUSDMethodDetailInput(method_names='["GetPrim", "CreatePrim"]', class_name="UsdStage").method_names
        )
        == "GetPrim,CreatePrim"
    )
