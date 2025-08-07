## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from typing import List, Optional

from pxr import Kind, Sdf, Usd, UsdGeom


def get_prim_kinds(kind_name: str) -> list[str]:
    """
    Get all prim kinds that derive from the given kind name.

    Args:
        kind_name (str): The name of the kind to find derived kinds for.

    Returns:
        list[str]: A list of all prim kinds that derive from the given kind.
    """
    if not Kind.Registry.HasKind(kind_name):
        raise ValueError(f"Kind '{kind_name}' does not exist in the registry.")
    all_kinds = Kind.Registry.GetAllKinds()
    derived_kinds = [kind for kind in all_kinds if Kind.Registry.IsA(kind, kind_name)]
    return derived_kinds


def filter_prims_by_kind(prims: List[Usd.Prim], kind: str) -> List[Usd.Prim]:
    """Filter a list of prims by kind.

    Args:
        prims (List[Usd.Prim]): A list of prims to filter.
        kind (str): The kind to filter by.

    Returns:
        List[Usd.Prim]: A list of prims that match the specified kind.
    """
    if not Kind.Registry.HasKind(kind):
        raise ValueError(f"Unknown kind: {kind}")
    filtered_prims = [prim for prim in prims if Usd.ModelAPI(prim).GetKind() == kind]
    return filtered_prims


def assign_kind_to_prims(stage: Usd.Stage, prim_paths: List[str], kind: str) -> None:
    """Assign a kind to a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths.
        kind (str): The kind to assign.

    Raises:
        ValueError: If the kind is not known to the registry.
    """
    if not Kind.Registry.HasKind(kind):
        raise ValueError(f"Kind '{kind}' is not known to the registry.")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path '{prim_path}' does not exist. Skipping.")
            continue
        prim.SetMetadata("kind", kind)


def validate_stage_kinds(stage: Usd.Stage) -> bool:
    """Validate that all prims on the stage have known kinds."""
    for prim in stage.TraverseAll():
        kind = Usd.ModelAPI(prim).GetKind()
        if kind and (not Kind.Registry.HasKind(kind)):
            print(f"Error: Prim {prim.GetPath()} has unknown kind '{kind}'")
            return False
    return True


def convert_to_component(kind: Kind.Tokens) -> Optional[Kind.Tokens]:
    """Convert a Kind.Tokens value to its component counterpart if applicable.

    Args:
        kind (Kind.Tokens): The Kind.Tokens value to convert.

    Returns:
        Optional[Kind.Tokens]: The corresponding component Kind.Tokens value if applicable, otherwise None.
    """
    if kind == Kind.Tokens.model:
        return Kind.Tokens.component
    elif kind == Kind.Tokens.group:
        return Kind.Tokens.assembly
    else:
        return None


def validate_assembly_structure(stage: Usd.Stage) -> List[Usd.Prim]:
    """Validate the assembly structure of a USD stage.

    This function checks if the stage has a valid assembly structure, which means:
    1. The stage must have a default prim.
    2. The default prim must be a valid assembly (has kind 'assembly').
    3. All descendants of the default prim must also be valid assemblies.

    Args:
        stage (Usd.Stage): The USD stage to validate.

    Returns:
        List[Usd.Prim]: A list of invalid prims that do not meet the assembly structure requirements.
    """
    invalid_prims: List[Usd.Prim] = []
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise ValueError("Stage does not have a default prim.")
    if default_prim.GetMetadata("kind") != Kind.Tokens.assembly:
        invalid_prims.append(default_prim)
    for descendant in Usd.PrimRange(default_prim):
        if descendant.GetMetadata("kind") != Kind.Tokens.assembly:
            invalid_prims.append(descendant)
    return invalid_prims


def set_kind(prim: Usd.Prim, kind_name: Kind.Tokens) -> None:
    """Set the kind for a given prim."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    kind_attr = prim.GetAttribute("kind")
    if not kind_attr.IsValid():
        kind_attr = prim.CreateAttribute("kind", Sdf.ValueTypeNames.Token)
    kind_attr.Set(kind_name)


def get_kind(prim: Usd.Prim) -> str:
    """Get the kind for a given prim."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    kind_attr = prim.GetAttribute("kind")
    if not kind_attr.IsValid():
        return ""
    kind_value = kind_attr.Get()
    if kind_value is None:
        return ""
    return kind_value
