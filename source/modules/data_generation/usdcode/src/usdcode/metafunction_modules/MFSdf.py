## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdSkel, Vt

from .add_op import *


def convert_angular_unit(value: float, source_unit: str, target_unit: str) -> float:
    """Convert an angular value from one unit to another.

    Args:
        value (float): The angular value to convert.
        source_unit (str): The source angular unit ('degrees' or 'radians').
        target_unit (str): The target angular unit ('degrees' or 'radians').

    Returns:
        float: The converted angular value.
    """
    if source_unit == "degrees":
        degrees = value
    elif source_unit == "radians":
        degrees = value * 180.0 / 3.14159265359
    else:
        raise ValueError(f"Unsupported source angular unit: {source_unit}")
    if target_unit == "degrees":
        return degrees
    elif target_unit == "radians":
        return degrees * 3.14159265359 / 180.0
    else:
        raise ValueError(f"Unsupported target angular unit: {target_unit}")


def resolve_asset_paths(asset_paths: List[Sdf.AssetPath]) -> List[Tuple[str, str]]:
    """
    Resolve a list of asset paths and return a list of tuples containing the original path and resolved path.

    Args:
        asset_paths (List[Sdf.AssetPath]): A list of asset paths to resolve.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the original path and resolved path.
    """
    resolved_paths = []
    for asset_path in asset_paths:
        original_path = asset_path.path
        resolved_path = asset_path.resolvedPath
        if not resolved_path:
            resolved_path = original_path
        resolved_paths.append((original_path, resolved_path))
    return resolved_paths


def update_asset_references(layer: Sdf.Layer, old_asset_path: str, new_asset_path: str) -> int:
    """
    Update all references to the given old asset path with the new asset path in the given layer.

    Args:
        layer (Sdf.Layer): The layer in which to update asset references.
        old_asset_path (str): The old asset path to replace.
        new_asset_path (str): The new asset path to use as a replacement.

    Returns:
        int: The number of asset references that were updated.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    old_path = Sdf.AssetPath(old_asset_path)
    new_path = Sdf.AssetPath(new_asset_path)
    num_updated = 0
    for prim in layer.rootPrims:
        stack = [prim]
        while stack:
            current_prim = stack.pop()
            for prop in current_prim.properties:
                if isinstance(prop, Sdf.AttributeSpec):
                    attr = prop
                    if attr.typeName == Sdf.ValueTypeNames.Asset:
                        current_path = attr.default
                        if current_path == old_path:
                            attr.default = new_path
                            num_updated += 1
            stack.extend(current_prim.nameChildren)
    return num_updated


def replace_asset_paths(asset_path_array: Sdf.AssetPathArray, old_path: str, new_path: str) -> None:
    """
    Replace all occurrences of the given old_path with the new_path in the AssetPathArray.

    Args:
        asset_path_array (Sdf.AssetPathArray): The AssetPathArray to modify.
        old_path (str): The old asset path to replace.
        new_path (str): The new asset path to replace with.

    Raises:
        ValueError: If the old_path is empty or not found in the AssetPathArray.
    """
    if not old_path:
        raise ValueError("The old_path cannot be empty.")
    found = False
    for i in range(len(asset_path_array)):
        asset_path = asset_path_array[i]
        if old_path in asset_path.path:
            new_asset_path = Sdf.AssetPath(asset_path.path.replace(old_path, new_path))
            asset_path_array[i] = new_asset_path
            found = True
    if not found:
        raise ValueError(f"The old_path '{old_path}' was not found in the AssetPathArray.")


def find_prims_with_asset_path(stage: Usd.Stage, asset_path: Sdf.AssetPath) -> List[Usd.Prim]:
    """Find all prims on the stage that have the specified asset path.

    Args:
        stage (Usd.Stage): The USD stage to search.
        asset_path (Sdf.AssetPath): The asset path to search for.

    Returns:
        List[Usd.Prim]: A list of prims that have the specified asset path.
    """
    prims_with_asset_path = []
    for prim in stage.TraverseAll():
        attributes = prim.GetAttributes()
        for attr in attributes:
            if attr.GetTypeName() == "asset[]":
                asset_path_array = attr.Get()
                if asset_path in asset_path_array:
                    prims_with_asset_path.append(prim)
                    break
    return prims_with_asset_path


def collect_asset_paths(prim: Usd.Prim) -> Sdf.AssetPathArray:
    """Recursively collect all asset paths under the given prim.

    Args:
        prim (Usd.Prim): The prim to start the search from.

    Returns:
        Sdf.AssetPathArray: An array of all unique asset paths found.
    """
    asset_paths = set()
    for descendant in Usd.PrimRange(prim):
        for attr in descendant.GetAttributes():
            if attr.GetTypeName() == "asset":
                asset_path = attr.Get()
                if asset_path:
                    asset_paths.add(asset_path)
    return Sdf.AssetPathArray(list(asset_paths))


def validate_asset_paths(asset_paths: Sdf.AssetPathArray) -> bool:
    """Validate an array of asset paths.

    Args:
        asset_paths (Sdf.AssetPathArray): Array of asset paths to validate.

    Returns:
        bool: True if all asset paths are valid, False otherwise.
    """
    for asset_path in asset_paths:
        if asset_path.path == "":
            print(f"Asset path is empty.")
            return False
        if asset_path.resolvedPath == "":
            print(f"Asset path '{asset_path.path}' has an unsupported URI scheme.")
            return False
        if not Sdf.Layer.FindOrOpen(asset_path.resolvedPath):
            print(f"Asset path '{asset_path.path}' does not resolve to a valid file.")
            return False
    return True


def copy_attribute(source_prim: Usd.Prim, source_attr_name: str, dest_prim: Usd.Prim, dest_attr_name: str):
    """Copy an attribute from one prim to another.

    Args:
        source_prim (Usd.Prim): The prim to copy the attribute from.
        source_attr_name (str): The name of the attribute to copy.
        dest_prim (Usd.Prim): The prim to copy the attribute to.
        dest_attr_name (str): The name to give the copied attribute.

    Raises:
        ValueError: If the source attribute does not exist or has an invalid value.
    """
    if not source_prim.HasAttribute(source_attr_name):
        raise ValueError(f"Attribute {source_attr_name} does not exist on prim {source_prim.GetPath()}")
    source_attr = source_prim.GetAttribute(source_attr_name)
    attr_value = source_attr.Get()
    if attr_value is None:
        raise ValueError(f"Attribute {source_attr_name} on prim {source_prim.GetPath()} has no value")
    dest_attr = dest_prim.CreateAttribute(dest_attr_name, source_attr.GetTypeName())
    dest_attr.Set(attr_value)
    for key in source_attr.GetAllMetadata():
        dest_attr.SetMetadata(key, source_attr.GetMetadata(key))


def rename_attribute(attribute: Usd.Attribute, new_name: str) -> bool:
    """Rename an attribute on a prim.

    Args:
        attribute (Usd.Attribute): The attribute to rename.
        new_name (str): The new name for the attribute.

    Returns:
        bool: True if the attribute was successfully renamed, False otherwise.
    """
    if not attribute:
        return False
    prim = attribute.GetPrim()
    if not prim:
        return False
    if prim.HasAttribute(new_name):
        return False
    old_name = attribute.GetName()
    new_attribute = prim.CreateAttribute(new_name, attribute.GetTypeName())
    if attribute.HasValue():
        new_attribute.Set(attribute.Get())
    prim.RemoveProperty(old_name)
    return True


def get_attribute_connections(attribute: Sdf.AttributeSpec) -> List[Sdf.Path]:
    """
    Get the list of connection paths for the given attribute.

    Args:
        attribute (Sdf.AttributeSpec): The attribute to get connections for.

    Returns:
        List[Sdf.Path]: The list of connection paths.
    """
    if not attribute:
        raise ValueError("Invalid attribute specified.")
    connection_path_list = attribute.connectionPathList
    if not connection_path_list.isExplicit:
        return []
    connection_paths = connection_path_list.explicitItems
    return connection_paths


def validate_stage_authoring(stage: Usd.Stage) -> bool:
    """Validate the authoring of a USD stage.

    This function checks if the stage has any authoring errors by attempting to
    export it to a string. If an AuthoringError is raised during the export, it
    indicates that there are authoring errors in the stage.

    Args:
        stage (Usd.Stage): The USD stage to validate.

    Returns:
        bool: True if the stage has no authoring errors, False otherwise.
    """
    try:
        stage.GetRootLayer().ExportToString()
        return True
    except Sdf.AuthoringError:
        return False


def batch_reparent_prims(namespace_edit: Sdf.BatchNamespaceEdit, src_paths: list[str], dst_path: str) -> bool:
    """Reparent multiple prims to a new parent path in a single operation.

    Args:
        namespace_edit (Sdf.BatchNamespaceEdit): The batch namespace edit object to add the reparent operations to.
        src_paths (list[str]): The paths of the prims to reparent.
        dst_path (str): The new parent path for the prims.

    Returns:
        bool: True if the reparent operations were added successfully, False otherwise.
    """
    for src_path in src_paths:
        if not Sdf.Path(src_path).IsAbsolutePath():
            print(f"Error: Invalid source path '{src_path}'. Paths must be absolute.")
            return False
    if not Sdf.Path(dst_path).IsAbsolutePath():
        print(f"Error: Invalid destination path '{dst_path}'. Path must be absolute.")
        return False
    for src_path in src_paths:
        src_prim_name = Sdf.Path(src_path).name
        new_path = Sdf.Path(dst_path).AppendChild(src_prim_name)
        namespace_edit.Add(src_path, new_path)
    return True


def batch_remove_prims(stage: Usd.Stage, prim_paths: List[str]) -> None:
    """Remove multiple prims from the USD stage in a single operation.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths to remove.
    """
    edits = Sdf.BatchNamespaceEdit()
    for prim_path in prim_paths:
        path = Sdf.Path(prim_path)
        if not stage.GetPrimAtPath(path):
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        edits.Add(path, Sdf.Path.emptyPath)
    stage.GetRootLayer().Apply(edits)


def batch_rename_prims(stage: Usd.Stage, rename_map: dict[str, str]) -> None:
    """Batch rename prims using a dictionary mapping old paths to new paths.

    Args:
        stage (Usd.Stage): The stage containing the prims to rename.
        rename_map (dict[str, str]): A dictionary mapping old prim paths to new prim paths.
    """
    batch_edit = Sdf.BatchNamespaceEdit()
    for old_path, new_path in rename_map.items():
        if not stage.GetPrimAtPath(old_path):
            raise ValueError(f"Prim at path {old_path} does not exist in the stage.")
        if stage.GetPrimAtPath(new_path):
            raise ValueError(f"Prim at path {new_path} already exists in the stage.")
        batch_edit.Add(Sdf.Path(old_path), Sdf.Path(new_path))
    stage.GetRootLayer().Apply(batch_edit)


def batch_transform_prims(stage: Usd.Stage, prim_paths: List[str], translation: Tuple[float, float, float]) -> None:
    """Batch transform multiple prims with the same translation.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths to transform.
        translation (Tuple[float, float, float]): The translation to apply to each prim.
    """
    if not prim_paths:
        return
    with Sdf.ChangeBlock():
        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
                continue
            xformable = UsdGeom.Xformable(prim)
            if not xformable:
                print(f"Warning: Prim at path {prim_path} is not transformable. Skipping.")
                continue
            translate_op = add_translate_op(xformable)
            translate_op.Set(Gf.Vec3d(translation))


def batch_modify_prims(stage: Usd.Stage, prim_paths: List[str], new_values: List[Tuple[float, float, float]]):
    """Modify multiple prims' translate attribute in a single batch.

    Args:
        stage (Usd.Stage): The stage to modify prims on.
        prim_paths (List[str]): List of prim paths to modify.
        new_values (List[Tuple[float, float, float]]): List of new translate values,
            must be same length as prim_paths.

    Raises:
        ValueError: If prim_paths and new_values are not the same length.
    """
    if len(prim_paths) != len(new_values):
        raise ValueError("prim_paths and new_values must be the same length.")
    with Sdf.ChangeBlock():
        for prim_path, new_value in zip(prim_paths, new_values):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim:
                continue
            xformable = UsdGeom.Xformable(prim)
            if not xformable:
                continue
            translate_op = add_translate_op(xformable)
            translate_op.Set(Gf.Vec3d(new_value))


def batch_update_material_assignments(stage: Usd.Stage, material_assignments: Dict[str, str]):
    """Update material assignments for multiple prims in a single changeblock.

    Args:
        stage (Usd.Stage): The stage to update material assignments on.
        material_assignments (Dict[str, str]): A dictionary mapping prim paths to material paths.
    """
    if not stage:
        raise ValueError("Invalid stage provided.")
    if not material_assignments:
        return
    with Sdf.ChangeBlock():
        for prim_path, material_path in material_assignments.items():
            prim = stage.GetPrimAtPath(prim_path)
            if not prim:
                print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
                continue
            if not prim.IsA(UsdGeom.Mesh):
                print(f"Warning: Prim at path {prim_path} is not a Mesh. Skipping.")
                continue
            material_prim = stage.GetPrimAtPath(material_path)
            if not material_prim:
                print(f"Warning: Material prim at path {material_path} does not exist. Skipping.")
                continue
            UsdShade.MaterialBindingAPI(prim).Bind(material_prim)


def bulk_update_and_cleanup(layer: Sdf.Layer, prim_paths: List[str], field_key: str, field_value: object):
    """Update the specified field on multiple prims and clean up any empty specs.

    Args:
        layer (Sdf.Layer): The layer to update.
        prim_paths (List[str]): A list of prim paths to update.
        field_key (str): The field key to update on each prim.
        field_value (object): The value to set for the specified field key.
    """
    with Sdf.CleanupEnabler():
        for prim_path in prim_paths:
            prim_spec = Sdf.CreatePrimInLayer(layer, prim_path)
            if not prim_spec:
                print(f"Warning: Failed to create prim spec at path {prim_path}. Skipping.")
                continue
            prim_spec.SetInfo(field_key, field_value)


def clean_and_reassign_material(stage: Usd.Stage, prim_path: str, material_path: str):
    """
    Clean any inert material binding on the prim and reassign a new material.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to clean and reassign material.
        material_path (str): The path to the new material to assign.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path: {prim_path}")
    binding_api = UsdShade.MaterialBindingAPI(prim)
    if binding_api.GetDirectBindingRel().GetTargets():
        binding_api.UnbindDirectBinding()
        with Sdf.CleanupEnabler():
            binding_api.GetDirectBindingRel().ClearTargets(removeSpec=True)
    material_prim = UsdShade.Material(stage.GetPrimAtPath(material_path))
    if not material_prim:
        raise ValueError(f"No material prim found at path: {material_path}")
    UsdShade.MaterialBindingAPI(prim).Bind(material_prim)


def get_all_textures(prim_spec: Sdf.PrimSpec) -> List[Sdf.AssetPath]:
    """
    Get all textures referenced by the given prim spec and its descendants.

    Args:
        prim_spec (Sdf.PrimSpec): The prim spec to search for textures.

    Returns:
        List[Sdf.AssetPath]: A list of unique asset paths for all textures found.
    """
    textures = set()
    if not prim_spec:
        return []
    for prop_spec in prim_spec.properties:
        if isinstance(prop_spec, Sdf.AttributeSpec):
            attr_spec = prop_spec
            if attr_spec.typeName == Sdf.ValueTypeNames.Asset:
                default_value = attr_spec.default
                if default_value:
                    textures.add(default_value)
    for child_spec in prim_spec.nameChildren:
        child_textures = get_all_textures(child_spec)
        textures.update(child_textures)
    return list(textures)


def create_test_stage() -> Usd.Stage:
    """Create a test stage with some textures."""
    stage = Usd.Stage.CreateInMemory()
    prim1 = stage.DefinePrim("/Prim1")
    prim1.CreateAttribute("diffuseColor", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("Textures/Prim1/diffuse.png"))
    prim2 = stage.DefinePrim("/Prim1/Prim2")
    prim2.CreateAttribute("normalMap", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("Textures/Prim2/normal.png"))
    stage.DefinePrim("/Prim3")
    return stage


def apply_dimensionless_unit_to_prims(stage: Usd.Stage, prim_paths: List[str], unit: str) -> None:
    """Apply a dimensionless unit to a list of prims.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (List[str]): A list of prim paths to apply the unit to.
        unit (str): The dimensionless unit to apply as a string.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            print(f"Warning: Prim at path {prim_path} is not transformable. Skipping.")
            continue
        translate_op = add_translate_op(xformable)
        translate_op.GetAttr().SetCustomDataByKey("my_dimensionless_unit", unit)


def check_layer_muted(layer: Sdf.Layer) -> bool:
    """Check if the layer is muted.

    Args:
        layer (Sdf.Layer): The layer to check.

    Returns:
        bool: True if the layer is muted, False otherwise.
    """
    if not layer:
        raise ValueError("Invalid layer")
    is_muted = layer.IsMuted()
    return is_muted


def apply_updates(layer: Sdf.Layer, updates: list) -> bool:
    """Apply a list of updates to the Sdf layer.

    Args:
        layer (Sdf.Layer): The Sdf layer to update.
        updates (list): A list of tuples representing the updates to apply.
                        Each tuple should contain the update type, path, and data.
                        Update types:
                        - "set": Set an attribute value. Data should be (attribute_name, attribute_value).
                        - "set_field": Set a field value. Data should be (field_name, field_value).
                        - "clear": Clear an attribute. Data should be the attribute_name.
                        - "remove": Remove a prim. Data should be None.
                        - "reparent": Reparent a prim. Data should be the new parent path.

    Returns:
        bool: True if the updates were applied successfully, False otherwise.
    """
    if not layer:
        return False
    try:
        with Sdf.ChangeBlock():
            for update_type, path, data in updates:
                spec = layer.GetPrimAtPath(path)
                if not spec:
                    continue
                if update_type == "set":
                    (attribute_name, attribute_value) = data
                    if Sdf.FieldKeys.IsValidAttributeKey(attribute_name):
                        spec.SetInfo(attribute_name, attribute_value)
                    else:
                        print(f"Invalid attribute name: {attribute_name}")
                elif update_type == "set_field":
                    (field_name, field_value) = data
                    spec.SetField(field_name, field_value)
                elif update_type == "clear":
                    attribute_name = data
                    if Sdf.FieldKeys.IsValidAttributeKey(attribute_name) and spec.HasInfo(attribute_name):
                        spec.ClearInfo(attribute_name)
                elif update_type == "remove":
                    layer.RemovePrim(path)
                elif update_type == "reparent":
                    new_parent_path = data
                    layer.ReparentPrim(path, new_parent_path)
        return True
    except Exception as e:
        print(f"Error applying updates: {e}")
        return False


class FastUpdate:

    def __init__(self, path=None, value=None):
        self.path = path
        self.value = value


def apply_fast_updates(fast_updates: List[FastUpdate]) -> None:
    """Apply a list of fast updates to a stage.

    Args:
        fast_updates (List[FastUpdate]): A list of FastUpdate objects.

    Raises:
        ValueError: If any of the FastUpdate objects have invalid paths or values.
    """
    if not fast_updates:
        return
    for update in fast_updates:
        if not update.path:
            raise ValueError("Invalid path in FastUpdate object.")
        if update.value is None:
            raise ValueError("Invalid value in FastUpdate object.")


def collect_prim_updates(updates: List[Sdf.Path]) -> Dict[Sdf.Path, Any]:
    """Collects prim updates from a list of Sdf.Path objects.

    Args:
        updates (List[Sdf.Path]): A list of Sdf.Path objects representing prim paths.

    Returns:
        Dict[Sdf.Path, Any]: A dictionary mapping prim paths to their updated values.
    """
    prim_updates: Dict[Sdf.Path, Any] = {}
    for path in updates:
        if path.IsPrimPath():
            prim_updates[path] = None
    return prim_updates


def get_primary_file_extension(file_format: Sdf.FileFormat) -> str:
    """Get the primary file extension for a given file format."""
    if not file_format:
        raise ValueError("Invalid file format.")
    primary_extension = file_format.primaryFileExtension
    if not primary_extension:
        raise ValueError("File format does not have a primary extension.")
    return primary_extension


def is_file_format_expired(file_format: Sdf.FileFormat) -> bool:
    """Check if a file format object has expired."""
    if not file_format:
        raise ValueError("Invalid file format object")
    return file_format.expired


def update_material_parameters(material: UsdShade.Material, parameter_dict: Dict[str, Any]) -> None:
    """Update the parameters of a material with the given dictionary.

    Args:
        material (UsdShade.Material): The material to update.
        parameter_dict (Dict[str, Any]): A dictionary mapping parameter names to their new values.

    Raises:
        ValueError: If the material is invalid or any parameter name is not found on the material.
    """
    if not material:
        raise ValueError("Invalid material")
    for param_name, param_value in parameter_dict.items():
        shader_input = material.GetInput(param_name)
        if not shader_input:
            raise ValueError(f"Parameter '{param_name}' not found on material")
        shader_input.Set(param_value)


def get_file_format_target(file_format: Sdf.FileFormat) -> str:
    """Get the target for a given file format.

    Args:
        file_format (Sdf.FileFormat): The file format to get the target for.

    Returns:
        str: The target for the given file format.

    Raises:
        ValueError: If the file format is invalid or expired.
    """
    if not file_format:
        raise ValueError("Invalid file format.")
    if file_format.expired:
        raise ValueError("File format has expired.")
    target = file_format.target
    return target


def create_and_assign_material(stage: Usd.Stage, prim_path: str, material_name: str) -> UsdShade.Material:
    """Create a new material and assign it to the specified prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to assign the material to.
        material_name (str): The name of the material to create.

    Returns:
        UsdShade.Material: The created material prim.

    Raises:
        ValueError: If the specified prim does not exist or is not a mesh.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    if not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"Prim at path {prim_path} is not a mesh.")
    material_path = f"{prim_path}/{material_name}"
    material = UsdShade.Material.Define(stage, material_path)
    shader_path = f"{material_path}/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((1.0, 0.0, 0.0))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(prim).Bind(material)
    return material


def batch_update_prim_attributes(stage: Usd.Stage, prim_attr_dict: Dict[str, Dict[str, Any]]) -> None:
    """Updates attributes on multiple prims in a single batch operation.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_attr_dict (Dict[str, Dict[str, Any]]): A dictionary mapping prim paths to a dictionary of attribute names and values.

    Raises:
        ValueError: If any of the prims or attributes are invalid.
    """
    for prim_path, attr_dict in prim_attr_dict.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        for attr_name, attr_value in attr_dict.items():
            attribute = prim.GetAttribute(attr_name)
            if not attribute.IsDefined():
                raise ValueError(f"Attribute {attr_name} does not exist on prim {prim_path}.")
            try:
                attribute.Set(attr_value)
            except Exception as e:
                raise ValueError(f"Failed to set attribute {attr_name} on prim {prim_path}: {str(e)}")


def validate_and_load_file(file_path: str, file_format: Sdf.FileFormat) -> Optional[Sdf.Layer]:
    """
    Validate and load a file using the specified file format.

    Args:
        file_path (str): The path to the file to validate and load.
        file_format (Sdf.FileFormat): The file format to use for validation and loading.

    Returns:
        Optional[Sdf.Layer]: The loaded layer if successful, None otherwise.
    """
    if not file_format.CanRead(file_path):
        print(f"Error: File format {file_format.formatId} cannot read the file {file_path}")
        return None
    file_extension = Sdf.FileFormat.GetFileExtension(file_path)
    if not file_format.IsSupportedExtension(file_extension):
        print(f"Error: File extension {file_extension} is not supported by file format {file_format.formatId}")
        return None
    try:
        layer = Sdf.Layer.FindOrOpen(file_path)
        if not layer:
            print(f"Error: Failed to load file {file_path} using file format {file_format.formatId}")
            return None
    except Tf.ErrorException as e:
        print(
            f"Error: Exception occurred while loading file {file_path} using file format {file_format.formatId}: {str(e)}"
        )
        return None
    return layer


def get_all_file_extensions() -> set[str]:
    """Get all file extensions supported by registered file formats."""
    file_formats = Sdf.FileFormat.FindAllFileFormatExtensions()
    extensions = set()
    for ext in file_formats:
        extensions.add(ext)
    return extensions


def find_file_format_by_extension(path: str, target: Optional[str] = None) -> Optional[Sdf.FileFormat]:
    """
    Find the file format instance that supports the extension for the given path.

    If a format with a matching extension is not found, this returns None.
    An extension may be handled by multiple file formats, but each with a different target.
    If no target is specified, the file format that is registered as the primary plugin will be returned.
    Otherwise, the file format whose target matches the provided target will be returned.

    Args:
        path (str): The file path to find the file format for.
        target (Optional[str]): The target file format. Defaults to None.

    Returns:
        Optional[Sdf.FileFormat]: The file format instance that supports the extension, or None if not found.
    """
    extension = Sdf.FileFormat.GetFileExtension(path)
    if target is None:
        file_format = Sdf.FileFormat.FindByExtension(path, {})
    else:
        file_format = Sdf.FileFormat.FindByExtension(path, {"target": target})
    return file_format if file_format else None


def is_valid_file_format(file_format: Sdf.FileFormat, file_path: str) -> bool:
    """Check if the given file format can read the specified file path."""
    if not file_format:
        return False
    file_extension = Sdf.FileFormat.GetFileExtension(file_path)
    if not file_format.IsSupportedExtension(file_extension):
        return False
    if not file_format.CanRead(file_path):
        return False
    return True


def list_supported_file_formats() -> list[Sdf.FileFormat]:
    """Returns a list of all supported file formats."""
    extensions = Sdf.FileFormat.FindAllFileFormatExtensions()
    supported_formats = []
    for extension in extensions:
        file_format = Sdf.FileFormat.FindByExtension(extension, {})
        if not file_format:
            continue
        supported_formats.append(file_format)
    return supported_formats


def get_file_format_id(file_path: str) -> str:
    """
    Get the file format identifier for the given file path.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The file format identifier.

    Raises:
        ValueError: If no file format is found for the given file path.
    """
    file_extension = Sdf.FileFormat.GetFileExtension(file_path)
    file_format = Sdf.FileFormat.FindByExtension(file_path, {})
    if not file_format:
        raise ValueError(f"No file format found for file path: {file_path}")
    return file_format.formatId


def add_file_reference(
    stage: Usd.Stage,
    prim_path: str,
    file_path: str,
    position: Gf.Vec3d = Gf.Vec3d(0, 0, 0),
    scale: Gf.Vec3f = Gf.Vec3f(1, 1, 1),
) -> Usd.Prim:
    """Add a reference to an external USD file at the specified prim path.

    Args:
        stage (Usd.Stage): The stage to add the reference to.
        prim_path (str): The path of the prim to add the reference to.
        file_path (str): The path to the external USD file to reference.
        position (Gf.Vec3d): The position of the referenced prim. Defaults to (0, 0, 0).
        scale (Gf.Vec3f): The scale of the referenced prim. Defaults to (1, 1, 1).

    Returns:
        Usd.Prim: The referenced prim.

    Raises:
        ValueError: If the prim path is invalid or the file path does not exist.
    """
    if not Sdf.Path.IsValidPathString(prim_path):
        raise ValueError(f"Invalid prim path: {prim_path}")
    if not Sdf.Layer.FindOrOpen(file_path):
        raise ValueError(f"File does not exist: {file_path}")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(file_path)
    UsdGeom.XformCommonAPI(prim).SetTranslate(position)
    UsdGeom.XformCommonAPI(prim).SetScale(scale)
    return prim


def validate_and_save_stage(stage: Usd.Stage, file_path: str, file_format: str = "usda") -> bool:
    """Validate and save the given stage to a file.

    Args:
        stage (Usd.Stage): The USD stage to validate and save.
        file_path (str): The file path to save the stage to.
        file_format (str, optional): The file format to save as. Defaults to "usda".

    Returns:
        bool: True if the stage was saved successfully, False otherwise.
    """
    if file_format not in Sdf.FileFormat.FindAllFileFormatExtensions():
        print(f"Unsupported file format: {file_format}")
        return False
    try:
        stage.GetRootLayer().Export(file_path, args={"format": file_format})
        print(f"Stage saved successfully to {file_path}")
        return True
    except Tf.ErrorException as e:
        print(f"Error saving stage: {e}")
        return False


def batch_convert_and_save_stages(input_dir: str, output_dir: str, output_format: str = "usda") -> List[str]:
    """
    Convert all USD stages in the input directory to the specified output format and save them in the output directory.

    Args:
        input_dir (str): The directory containing the input USD stages.
        output_dir (str): The directory where the converted stages will be saved.
        output_format (str, optional): The desired output format for the stages. Defaults to "usda".

    Returns:
        List[str]: A list of file paths of the successfully converted stages.
    """
    if not os.path.isdir(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return []
    os.makedirs(output_dir, exist_ok=True)
    usd_files = [f for f in os.listdir(input_dir) if f.endswith((".usd", ".usda", ".usdc"))]
    converted_stages = []
    for usd_file in usd_files:
        input_path = os.path.join(input_dir, usd_file)
        output_path = os.path.join(output_dir, os.path.splitext(usd_file)[0] + "." + output_format)
        try:
            stage = Usd.Stage.Open(input_path)
            stage.Export(output_path)
            converted_stages.append(output_path)
        except Tf.ErrorException as e:
            print(f"Error converting {input_path}: {str(e)}")
    return converted_stages


def merge_int64_lists(list1: Sdf.Int64ListOp, list2: Sdf.Int64ListOp) -> Sdf.Int64ListOp:
    """Merge two Int64ListOp objects into a new Int64ListOp.

    Args:
        list1 (Sdf.Int64ListOp): The first list to merge.
        list2 (Sdf.Int64ListOp): The second list to merge.

    Returns:
        Sdf.Int64ListOp: A new Int64ListOp containing the merged items.
    """
    merged_list = Sdf.Int64ListOp()
    explicit_items1 = list1.GetAddedOrExplicitItems()
    explicit_items2 = list2.GetAddedOrExplicitItems()
    merged_explicit_items = explicit_items1 + explicit_items2
    for item in merged_explicit_items:
        merged_list.appendedItems.append(item)
    merged_list.ClearAndMakeExplicit()
    return merged_list


def filter_uint_list_by_predicate(uint_list_op: Sdf.UIntListOp, predicate: Callable[[int], bool]) -> Sdf.UIntListOp:
    """Filter a UIntListOp by a predicate function.

    Args:
        uint_list_op (Sdf.UIntListOp): The input UIntListOp to filter.
        predicate (Callable[[int], bool]): The predicate function to apply.
            Should take an unsigned int and return a bool.

    Returns:
        Sdf.UIntListOp: A new UIntListOp with the filtered values.
    """
    filtered_list_op = Sdf.UIntListOp.CreateExplicit()
    for item in uint_list_op.GetAddedOrExplicitItems():
        if predicate(item):
            filtered_list_op.appendedItems.append(item)
    return filtered_list_op


def is_even(x: int) -> bool:
    return x % 2 == 0


def clear_int64_lists(int64_list_op: Sdf.Int64ListOp) -> None:
    """Clear the Int64ListOp, removing all items."""
    if not int64_list_op:
        raise ValueError("Invalid Int64ListOp provided.")
    if not int64_list_op.GetAddedOrExplicitItems():
        return
    int64_list_op.Clear()
    if int64_list_op.GetAddedOrExplicitItems():
        raise RuntimeError("Failed to clear the Int64ListOp.")


def remove_duplicate_int64_items(int64_list_op: Sdf.Int64ListOp) -> Sdf.Int64ListOp:
    """Remove duplicate items from an Int64ListOp while preserving order.

    Args:
        int64_list_op (Sdf.Int64ListOp): The input Int64ListOp.

    Returns:
        Sdf.Int64ListOp: A new Int64ListOp with duplicate items removed.
    """
    ordered_items = int64_list_op.orderedItems
    result_list_op = Sdf.Int64ListOp.Create()
    unique_items = set()
    for item in ordered_items:
        if item not in unique_items:
            unique_items.add(item)
            result_list_op.appendedItems.append(item)
    return result_list_op


def add_int64_item_to_all(int64_list_op: Sdf.Int64ListOp, item: int) -> None:
    """Add an integer item to all operation lists in an Int64ListOp.

    Args:
        int64_list_op (Sdf.Int64ListOp): The Int64ListOp to modify.
        item (int): The integer item to add.

    Raises:
        TypeError: If int64_list_op is not an instance of Sdf.Int64ListOp.
        TypeError: If item is not an integer.
    """
    if not isinstance(int64_list_op, Sdf.Int64ListOp):
        raise TypeError("int64_list_op must be an instance of Sdf.Int64ListOp")
    if not isinstance(item, int):
        raise TypeError("item must be an integer")
    if int64_list_op.addedItems is None:
        int64_list_op.addedItems = [item]
    else:
        int64_list_op.addedItems.append(item)
    if int64_list_op.prependedItems is None:
        int64_list_op.prependedItems = [item]
    else:
        int64_list_op.prependedItems.append(item)
    if int64_list_op.appendedItems is None:
        int64_list_op.appendedItems = [item]
    else:
        int64_list_op.appendedItems.append(item)
    if int64_list_op.orderedItems is None:
        int64_list_op.orderedItems = [item]
    else:
        int64_list_op.orderedItems.append(item)


def prepend_int64_items(list_op: Sdf.Int64ListOp, items: Sequence[int]) -> None:
    """Prepend a list of integer items to the list operation.

    Args:
        list_op (Sdf.Int64ListOp): The list operation to modify.
        items (Sequence[int]): The items to prepend to the list operation.

    Raises:
        TypeError: If list_op is not an instance of Sdf.Int64ListOp.
        TypeError: If any item in items is not an integer.
    """
    if not isinstance(list_op, Sdf.Int64ListOp):
        raise TypeError(f"Expected Sdf.Int64ListOp, got {type(list_op)}")
    for item in items:
        if not isinstance(item, int):
            raise TypeError(f"Expected int, got {type(item)}")
    for item in reversed(items):
        list_op.prependedItems.insert(0, item)


def delete_int64_items(int64_list_op: Sdf.Int64ListOp, items_to_delete: List[int]) -> Sdf.Int64ListOp:
    """Delete items from an Sdf.Int64ListOp.

    Args:
        int64_list_op (Sdf.Int64ListOp): The input Int64ListOp to delete items from.
        items_to_delete (List[int]): The list of items to delete.

    Returns:
        Sdf.Int64ListOp: The updated Int64ListOp with items deleted.
    """
    updated_list_op = Sdf.Int64ListOp()
    updated_list_op.appendedItems = int64_list_op.appendedItems
    updated_list_op.prependedItems = int64_list_op.prependedItems
    updated_list_op.deletedItems = int64_list_op.deletedItems
    for item in items_to_delete:
        if item not in updated_list_op.deletedItems:
            updated_list_op.deletedItems.append(item)
    updated_list_op.appendedItems = [item for item in updated_list_op.appendedItems if item not in items_to_delete]
    updated_list_op.prependedItems = [item for item in updated_list_op.prependedItems if item not in items_to_delete]
    return updated_list_op


def set_ordered_int64_items(int64_list_op: Sdf.Int64ListOp, items: List[int]) -> None:
    """Set the ordered items in an Int64ListOp.

    This function clears any existing items and sets the ordered items to the given list.

    Args:
        int64_list_op (Sdf.Int64ListOp): The Int64ListOp to modify.
        items (List[int]): The list of integers to set as the ordered items.

    Raises:
        TypeError: If int64_list_op is not an instance of Sdf.Int64ListOp.
        TypeError: If items is not a list of integers.
    """
    if not isinstance(int64_list_op, Sdf.Int64ListOp):
        raise TypeError("int64_list_op must be an instance of Sdf.Int64ListOp")
    if not isinstance(items, list) or not all((isinstance(item, int) for item in items)):
        raise TypeError("items must be a list of integers")
    int64_list_op.Clear()
    for item in items:
        int64_list_op.appendedItems.append(item)


def copy_int64_list(src_list_op: Sdf.Int64ListOp) -> Sdf.Int64ListOp:
    """Create a copy of an Sdf.Int64ListOp."""
    dst_list_op = Sdf.Int64ListOp()
    if src_list_op.isExplicit:
        dst_list_op = Sdf.Int64ListOp.CreateExplicit(src_list_op.explicitItems)
    else:
        for item in src_list_op.addedItems:
            dst_list_op.addedItems.append(item)
        for item in src_list_op.prependedItems:
            dst_list_op.prependedItems.append(item)
        for item in src_list_op.appendedItems:
            dst_list_op.appendedItems.append(item)
        for item in src_list_op.deletedItems:
            dst_list_op.deletedItems.append(item)
        for item in src_list_op.orderedItems:
            dst_list_op.orderedItems.append(item)
    return dst_list_op


def reverse_int64_list(int64_list_op: Sdf.Int64ListOp) -> Sdf.Int64ListOp:
    """Reverses the order of items in an Sdf.Int64ListOp.

    Args:
        int64_list_op (Sdf.Int64ListOp): The input Int64ListOp to reverse.

    Returns:
        Sdf.Int64ListOp: A new Int64ListOp with the items in reversed order.
    """
    reversed_list_op = Sdf.Int64ListOp()
    if int64_list_op.isExplicit:
        for item in reversed(int64_list_op.explicitItems):
            reversed_list_op.appendedItems.append(item)
    else:
        items = []
        int64_list_op.ApplyOperations(items)
        for item in reversed(items):
            reversed_list_op.appendedItems.append(item)
    return reversed_list_op


def assign_int64_list_from_prim(prim: Usd.Prim, attribute_name: str, int64_list: Vt.Int64Array):
    """Assign a Vt.Int64Array to a prim attribute.

    Args:
        prim (Usd.Prim): The prim to assign the Int64Array to.
        attribute_name (str): The name of the attribute to assign the Int64Array to.
        int64_list (Vt.Int64Array): The Int64Array to assign.
    """
    attribute = prim.GetAttribute(attribute_name)
    if not attribute:
        attribute = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Int64Array)
    attribute.Set(int64_list)


def get_int_list_operations(int_list_op: Sdf.IntListOp) -> Tuple[List[int], List[int], List[int], List[int], bool]:
    """Get the various operations and state of an Sdf.IntListOp.

    Args:
        int_list_op (Sdf.IntListOp): The IntListOp to query.

    Returns:
        Tuple[List[int], List[int], List[int], List[int], bool]:
            - The explicitly set items.
            - The added items.
            - The prepended items.
            - The appended items.
            - Whether the IntListOp is explicit.
    """
    explicit_items = int_list_op.GetAddedOrExplicitItems()
    added_items = int_list_op.addedItems if int_list_op.addedItems else []
    prepended_items = int_list_op.prependedItems if int_list_op.prependedItems else []
    appended_items = int_list_op.appendedItems if int_list_op.appendedItems else []
    is_explicit = int_list_op.isExplicit
    return (explicit_items, added_items, prepended_items, appended_items, is_explicit)


def create_int_list_op(
    explicit_items: Sdf.IntListOp = None,
    prepended_items: Sdf.IntListOp = None,
    appended_items: Sdf.IntListOp = None,
    deleted_items: Sdf.IntListOp = None,
) -> Sdf.IntListOp:
    """Create an Sdf.IntListOp with the specified items.

    Args:
        explicit_items (Sdf.IntListOp, optional): Explicitly set items. Defaults to None.
        prepended_items (Sdf.IntListOp, optional): Items to prepend. Defaults to None.
        appended_items (Sdf.IntListOp, optional): Items to append. Defaults to None.
        deleted_items (Sdf.IntListOp, optional): Items to delete. Defaults to None.

    Returns:
        Sdf.IntListOp: The created IntListOp.
    """
    int_list_op = Sdf.IntListOp()
    if explicit_items is not None:
        int_list_op.explicitItems = explicit_items
    if prepended_items is not None:
        for item in prepended_items:
            int_list_op.prependedItems.append(item)
    if appended_items is not None:
        for item in appended_items:
            int_list_op.appendedItems.append(item)
    if deleted_items is not None:
        for item in deleted_items:
            int_list_op.deletedItems.append(item)
    return int_list_op


def combine_int_lists(list_op_a: Sdf.IntListOp, list_op_b: Sdf.IntListOp) -> Sdf.IntListOp:
    """Combine two IntListOp objects into a single IntListOp.

    This function takes two IntListOp objects and combines their operations
    into a single IntListOp. The resulting IntListOp will have the combined
    explicit items, added items, prepended items, appended items, and deleted items
    from both input IntListOps.

    Args:
        list_op_a (Sdf.IntListOp): The first IntListOp to combine.
        list_op_b (Sdf.IntListOp): The second IntListOp to combine.

    Returns:
        Sdf.IntListOp: A new IntListOp with the combined operations.
    """
    combined_list_op = Sdf.IntListOp()
    combined_list_op.explicitItems = list(set(list_op_a.explicitItems + list_op_b.explicitItems))
    combined_list_op.addedItems = list(set(list_op_a.addedItems + list_op_b.addedItems))
    combined_list_op.prependedItems = list_op_a.prependedItems + list_op_b.prependedItems
    combined_list_op.appendedItems = list_op_a.appendedItems + list_op_b.appendedItems
    combined_list_op.deletedItems = list(set(list_op_a.deletedItems + list_op_b.deletedItems))
    return combined_list_op


def set_int_list(int_list_op: Sdf.IntListOp, items: List[int], explicit: bool = False) -> None:
    """Set the items of an IntListOp.

    Args:
        int_list_op (Sdf.IntListOp): The IntListOp to modify.
        items (List[int]): The list of integers to set.
        explicit (bool, optional): Whether to set the items explicitly. Defaults to False.
    """
    int_list_op.Clear()
    if explicit:
        for item in items:
            int_list_op.addedItems.append(item)
    else:
        int_list_op.explicitItems = items


def append_to_int_list(int_list_op: Sdf.IntListOp, value: int) -> None:
    """Append a value to an IntListOp.

    Args:
        int_list_op (Sdf.IntListOp): The IntListOp to append to.
        value (int): The integer value to append.
    """
    if not int_list_op:
        raise ValueError("Invalid IntListOp. Cannot append value.")
    if int_list_op.HasItem(value):
        return
    appended_items = int_list_op.appendedItems
    if appended_items is None:
        appended_items = [value]
    else:
        appended_items.append(value)
    int_list_op.appendedItems = appended_items


def prepend_to_int_list(int_list_op: Sdf.IntListOp, value: int) -> None:
    """Prepend a value to the IntListOp.

    Args:
        int_list_op (Sdf.IntListOp): The IntListOp to prepend to.
        value (int): The integer value to prepend.
    """
    if not isinstance(int_list_op, Sdf.IntListOp):
        raise TypeError("Expected an Sdf.IntListOp")
    if not isinstance(value, int):
        raise TypeError("Expected an integer value")
    if int_list_op.HasItem(value):
        explicit_items = int_list_op.GetAddedOrExplicitItems()
        explicit_items.remove(value)
        int_list_op = Sdf.IntListOp.CreateExplicit(explicit_items)
    prepended_items = int_list_op.prependedItems or []
    prepended_items.insert(0, value)
    new_int_list_op = Sdf.IntListOp()
    new_int_list_op.prependedItems = prepended_items
    new_int_list_op.appendedItems = int_list_op.appendedItems
    new_int_list_op.deletedItems = int_list_op.deletedItems
    int_list_op.ApplyOperations(new_int_list_op)


def delete_from_int_list(int_list_op: Sdf.IntListOp, items_to_delete: List[int]) -> None:
    """Delete items from an IntListOp.

    Args:
        int_list_op (Sdf.IntListOp): The IntListOp to delete items from.
        items_to_delete (List[int]): The list of integers to delete.
    """
    if int_list_op.isExplicit:
        explicit_items = int_list_op.explicitItems
        new_items = [item for item in explicit_items if item not in items_to_delete]
        int_list_op.ClearAndMakeExplicit()
        for item in new_items:
            int_list_op.addedItems.append(item)
    else:
        for item in items_to_delete:
            if item not in int_list_op.deletedItems:
                int_list_op.deletedItems.append(item)


def clear_and_make_explicit_int_list(int_list_op: Sdf.IntListOp) -> None:
    """Clear the IntListOp and make it explicit.

    Args:
        int_list_op (Sdf.IntListOp): The IntListOp to clear and make explicit.
    """
    if not isinstance(int_list_op, Sdf.IntListOp):
        raise TypeError("Input must be an instance of Sdf.IntListOp")
    int_list_op.Clear()
    int_list_op.ClearAndMakeExplicit()


def apply_int_operations(
    int_list_op: Sdf.IntListOp,
    explicit_items: List[int],
    added_items: List[int],
    prepended_items: List[int],
    appended_items: List[int],
    deleted_items: List[int],
) -> List[int]:
    """Apply a series of list operations to an integer list.

    Args:
        int_list_op (Sdf.IntListOp): The IntListOp object to apply operations to.
        explicit_items (List[int]): The list of explicit items to set.
        added_items (List[int]): The list of items to add.
        prepended_items (List[int]): The list of items to prepend.
        appended_items (List[int]): The list of items to append.
        deleted_items (List[int]): The list of items to delete.

    Returns:
        List[int]: The resulting list after applying all operations.
    """
    int_list_op.ClearAndMakeExplicit()
    for item in explicit_items:
        int_list_op.explicitItems.append(item)
    for item in added_items:
        int_list_op.addedItems.append(item)
    for item in prepended_items:
        int_list_op.prependedItems.append(item)
    for item in appended_items:
        int_list_op.appendedItems.append(item)
    for item in deleted_items:
        int_list_op.deletedItems.append(item)
    result = Sdf.IntListOp()
    int_list_op.ApplyOperations(result)
    return list(result.GetAddedOrExplicitItems())


def clear_int_attribute(prim: Usd.Prim, attribute_name: str) -> bool:
    """Clear the value of an integer array attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to clear the attribute on.
        attribute_name (str): The name of the integer array attribute to clear.

    Returns:
        bool: True if the attribute was successfully cleared, False otherwise.
    """
    if not prim.IsValid():
        return False
    if not prim.HasAttribute(attribute_name):
        return False
    attr = prim.GetAttribute(attribute_name)
    if attr.GetTypeName() != Sdf.ValueTypeNames.IntArray:
        return False
    attr.Set([])
    return True


def get_added_or_explicit_int_items(int_list_op: Sdf.IntListOp) -> List[int]:
    """Get the added or explicit integer items from an IntListOp.

    Args:
        int_list_op (Sdf.IntListOp): The IntListOp to retrieve items from.

    Returns:
        List[int]: The list of added or explicit integer items.
    """
    if int_list_op.isExplicit:
        return list(int_list_op.explicitItems)
    else:
        added_items = []
        if int_list_op.prependedItems:
            added_items.extend(int_list_op.prependedItems)
        if int_list_op.appendedItems:
            added_items.extend(int_list_op.appendedItems)
        return added_items


def reorder_int_list(int_list_op: Sdf.IntListOp, item: int, new_index: int) -> None:
    """Reorder an item in an IntListOp to a new index.

    Args:
        int_list_op (Sdf.IntListOp): The IntListOp to modify.
        item (int): The item to reorder.
        new_index (int): The new index to move the item to.

    Raises:
        ValueError: If the item is not in the IntListOp or the new index is out of range.
    """
    explicit_items = int_list_op.explicitItems
    if item not in explicit_items:
        raise ValueError(f"Item {item} does not exist in the IntListOp.")
    if new_index < 0 or new_index >= len(explicit_items):
        raise ValueError(f"Invalid new index {new_index}. Index out of range.")
    explicit_items.remove(item)
    explicit_items.insert(new_index, item)
    int_list_op.ClearAndMakeExplicit()
    for item in explicit_items:
        int_list_op.addedItems.append(item)


def get_layer_display_name(layer: Sdf.Layer) -> str:
    """Get the display name for a layer.

    The display name is the base filename of the identifier.
    If the layer has no identifier, return the empty string.
    """
    identifier = layer.identifier
    if not identifier:
        return ""
    components = identifier.split("/")
    display_name = components[-1]
    display_name = display_name.split(".")[0]
    return display_name


def get_time_samples_for_path(layer: Sdf.Layer, prim_path: str) -> set[float]:
    """Get the time samples for a given prim path in a layer.

    Args:
        layer (Sdf.Layer): The layer to query.
        prim_path (str): The path to the prim to get time samples for.

    Returns:
        set[float]: The time samples for the prim path, or an empty set if
        the prim path is not valid or has no time samples.
    """
    prim_spec = layer.GetPrimAtPath(prim_path)
    if not prim_spec:
        return set()
    attr_names = prim_spec.attributes.keys()
    time_samples = set()
    for attr_name in attr_names:
        attr_spec = prim_spec.attributes[attr_name]
        time_samples.update(attr_spec.layer.ListTimeSamplesForPath(attr_spec.path))
    return time_samples


def clear_layer_color_management_system(layer: Sdf.Layer) -> None:
    """Clear the colorManagementSystem metadata from the given layer."""
    if not layer:
        raise ValueError("Invalid layer")
    if layer.HasColorManagementSystem():
        layer.ClearColorManagementSystem()


def clear_layer_default_prim(layer: Sdf.Layer) -> None:
    """Clear the default prim metadata for the given layer."""
    if not layer:
        raise ValueError("Invalid layer")
    if not layer.HasDefaultPrim():
        return
    layer.ClearDefaultPrim()


def get_layer_pseudo_root(layer: Sdf.Layer) -> Sdf.PseudoRootSpec:
    """Get the pseudo-root of the given layer.

    Args:
        layer (Sdf.Layer): The layer to get the pseudo-root from.

    Returns:
        Sdf.PseudoRootSpec: The pseudo-root spec of the layer.
    """
    pseudo_root = layer.pseudoRoot
    if not pseudo_root:
        raise ValueError(f"Layer {layer.identifier} has no valid pseudo-root.")
    return pseudo_root


def get_layer_resolved_path(layer: Sdf.Layer) -> str:
    """Returns the resolved path for the given layer.

    Args:
        layer (Sdf.Layer): The layer to get the resolved path for.

    Returns:
        str: The resolved path of the layer.

    Raises:
        ValueError: If the given layer is invalid.
    """
    if not layer:
        raise ValueError("Invalid layer")
    identifier = layer.identifier
    if layer.anonymous:
        return identifier
    resolved_path = layer.resolvedPath
    if not resolved_path:
        resolved_path = identifier
    return resolved_path


def has_layer_custom_data(layer: Sdf.Layer) -> bool:
    """Check if a layer has custom data authored."""
    custom_data = layer.customLayerData
    if custom_data:
        return True
    else:
        return False


def has_layer_frame_precision(layer: Sdf.Layer) -> bool:
    """Check if the given layer has a frame precision opinion."""
    if not layer:
        raise ValueError("Invalid layer")
    frame_precision = layer.framePrecision
    if frame_precision is not None:
        return True
    else:
        return False


def has_layer_frames_per_second(layer: Sdf.Layer) -> bool:
    """Check if the given layer has a frames per second opinion."""
    if not layer:
        raise ValueError("Invalid layer")
    fps = layer.framesPerSecond
    if fps > 0 or fps == -1:
        return True
    else:
        return False


def list_layer_custom_data_keys(layer: Sdf.Layer) -> list[str]:
    """List the keys in the layer's custom data dictionary."""
    if not layer:
        raise ValueError("Invalid layer")
    if not layer.HasCustomLayerData():
        return []
    custom_data = layer.customLayerData
    return list(custom_data.keys())


def update_layer_external_reference(layer: Sdf.Layer, old_asset_path: str, new_asset_path: str) -> bool:
    """
    Update the asset path of an external reference in the given layer.

    If new_asset_path is supplied, update any occurrence of old_asset_path
    to new_asset_path. If new_asset_path is not provided (None), remove any
    occurrence of old_asset_path.

    Args:
        layer (Sdf.Layer): The layer to update.
        old_asset_path (str): The old asset path to update/remove.
        new_asset_path (str): The new asset path to update to, or None to remove.

    Returns:
        bool: True if the layer was modified, False otherwise.
    """
    if not layer:
        raise ValueError("Invalid layer provided")
    external_refs = layer.GetExternalReferences()
    if old_asset_path not in external_refs:
        return False
    if new_asset_path is None:
        layer.UpdateExternalReference(old_asset_path, "")
    else:
        layer.UpdateExternalReference(old_asset_path, new_asset_path)
    return True


def create_layer_with_default_time_codes(
    layer_path: str, start_time_code: float, end_time_code: float, time_codes_per_second: float
) -> Usd.Stage:
    """Create a new layer with default time code settings.

    Args:
        layer_path (str): The path to save the layer to.
        start_time_code (float): The start time code for the layer.
        end_time_code (float): The end time code for the layer.
        time_codes_per_second (float): The number of time codes per second.

    Returns:
        Usd.Stage: The stage with the new layer.
    """
    layer = Sdf.Layer.CreateNew(layer_path)
    layer.startTimeCode = start_time_code
    layer.endTimeCode = end_time_code
    layer.timeCodesPerSecond = time_codes_per_second
    stage = Usd.Stage.Open(layer)
    return stage


def set_layer_color_configuration(layer: Sdf.Layer, color_config_path: str) -> None:
    """Set the color configuration metadata for the given layer.

    Args:
        layer (Sdf.Layer): The layer to set the color configuration on.
        color_config_path (str): The asset path to the color configuration to set.

    Raises:
        ValueError: If the given layer is invalid.
    """
    if not layer:
        raise ValueError("Invalid layer")
    layer.colorConfiguration = color_config_path
    if not color_config_path:
        layer.ClearColorConfiguration()


def get_layer_identifier(layer: Sdf.Layer) -> str:
    """Get the identifier for a given layer.

    Args:
        layer (Sdf.Layer): The layer to get the identifier from.

    Returns:
        str: The identifier of the layer.

    Raises:
        ValueError: If the input layer is invalid.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    identifier = layer.identifier
    return identifier


def set_layer_frame_precision(layer: Sdf.Layer, precision: int) -> None:
    """Set the frame precision for a given layer.

    Args:
        layer (Sdf.Layer): The layer to set the frame precision for.
        precision (int): The frame precision value to set.

    Raises:
        ValueError: If the provided precision is negative.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    if precision < 0:
        raise ValueError("Frame precision must be a non-negative integer.")
    layer.framePrecision = precision


def set_layer_frames_per_second(layer: Sdf.Layer, frames_per_second: float) -> None:
    """Set the framesPerSecond metadata value for a given layer.

    Args:
        layer (Sdf.Layer): The layer to set the frames per second on.
        frames_per_second (float): The frames per second value to set.

    Raises:
        ValueError: If the frames_per_second is not greater than 0.
    """
    if frames_per_second <= 0:
        raise ValueError("framesPerSecond must be greater than 0")
    layer.framesPerSecond = frames_per_second
    layer.timeCodesPerSecond = frames_per_second


def set_layer_time_codes(
    layer: Sdf.Layer, start_time: float, end_time: float, time_codes_per_second: float, frame_precision: int
) -> None:
    """Set the time code attributes on a given layer.

    Args:
        layer (Sdf.Layer): The layer to set the time code attributes on.
        start_time (float): The start time code value.
        end_time (float): The end time code value.
        time_codes_per_second (float): The time codes per second (a.k.a. FPS).
        frame_precision (int): The frame precision, i.e. number of digits in the frame count.

    Raises:
        ValueError: If any of the input values are invalid.
    """
    if start_time > end_time:
        raise ValueError("start_time must be less than or equal to end_time")
    if time_codes_per_second <= 0:
        raise ValueError("time_codes_per_second must be greater than 0")
    if frame_precision < 0:
        raise ValueError("frame_precision must be greater than or equal to 0")
    layer.startTimeCode = start_time
    layer.endTimeCode = end_time
    layer.timeCodesPerSecond = time_codes_per_second
    layer.framePrecision = frame_precision


def remove_inert_scene_description(layer: Sdf.Layer):
    """
    Removes all scene description in the given layer that does not affect the scene.

    This method walks the layer namespace hierarchy and removes any prims
    that are not contributing any opinions.
    """
    root_prim = layer.pseudoRoot

    def is_inert(prim: Sdf.PrimSpec) -> bool:
        """Check if a prim is inert (has no properties or metadata)."""
        if prim.properties:
            return False
        if prim.ListInfoKeys():
            return False
        return True

    def remove_inert_children(prim: Sdf.PrimSpec):
        """Recursively remove inert children from the given prim."""
        for child in prim.nameChildren:
            if is_inert(child):
                prim.RemoveNameChild(child)
            else:
                remove_inert_children(child)

    remove_inert_children(root_prim)


@staticmethod
def RemoveFromMutedLayers(mutedPath: str) -> None:
    """Remove the specified path from the muted layers set."""
    mutedLayers = Sdf.Layer.GetMutedLayers()
    if mutedPath in mutedLayers:
        mutedLayers.remove(mutedPath)
        Sdf.Layer.SetMutedLayers(mutedLayers)
    else:
        pass


def clear_layer_end_time_code(layer: Sdf.Layer) -> None:
    """Clear the endTimeCode opinion for the given layer."""
    if layer.HasEndTimeCode():
        layer.ClearEndTimeCode()
    else:
        pass


def clear_layer_frame_precision(layer: Sdf.Layer) -> None:
    """Clear the framePrecision metadata from the given layer."""
    if layer.HasFramePrecision():
        layer.ClearFramePrecision()
    else:
        pass


def clear_layer_frames_per_second(layer: Sdf.Layer) -> None:
    """Clear the framesPerSecond opinion from the given layer."""
    if not layer:
        raise ValueError("Invalid layer")
    if not layer.HasFramesPerSecond():
        return
    layer.ClearFramesPerSecond()


def clear_layer_owner(layer: Sdf.Layer):
    """Clear the owner opinion from the given layer."""
    if not layer:
        raise ValueError("Invalid layer")
    if layer.HasOwner():
        layer.ClearOwner()
    else:
        pass


def clear_layer_session_owner(layer: Sdf.Layer) -> None:
    """Clear the layer's session owner, if set."""
    if layer.HasSessionOwner():
        layer.ClearSessionOwner()
        print(f"Cleared session owner for layer: {layer.identifier}")
    else:
        print(f"No session owner set for layer: {layer.identifier}")


def clear_layer_start_time_code(layer: Sdf.Layer) -> None:
    """Clear the startTimeCode opinion of the given layer."""
    if layer.HasStartTimeCode():
        layer.ClearStartTimeCode()
    else:
        pass


def clear_layer_time_codes_per_second(layer: Sdf.Layer) -> None:
    """Clear the timeCodesPerSecond opinion from the given layer."""
    if layer.HasTimeCodesPerSecond():
        layer.ClearTimeCodesPerSecond()
    else:
        pass


def erase_time_sample(layer: Sdf.Layer, path: Sdf.Path, time: float) -> None:
    """Erase a time sample from the given path in the layer.

    Args:
        layer (Sdf.Layer): The layer to erase the time sample from.
        path (Sdf.Path): The path to the attribute to erase the time sample from.
        time (float): The time at which to erase the sample.

    Raises:
        ValueError: If the given path is not a valid attribute path.
    """
    if not path.IsPropertyPath():
        raise ValueError(f"Path {path} is not a valid attribute path")
    if layer.QueryTimeSample(path, time):
        layer.EraseTimeSample(path, time)


def get_layer_file_format(layer: Sdf.Layer) -> Sdf.FileFormat:
    """Get the file format of a given layer.

    Args:
        layer (Sdf.Layer): The layer to get the file format for.

    Returns:
        Sdf.FileFormat: The file format of the given layer.
    """
    file_format = layer.GetFileFormat()
    if not file_format:
        raise ValueError(f"Layer {layer.identifier} has no valid file format.")
    return file_format


def get_loaded_layers() -> List[Sdf.Layer]:
    """Return a list of all currently loaded layers."""
    layer_registry = Sdf.Layer.GetLoadedLayers()
    loaded_layers = list(layer_registry)
    return loaded_layers


def test_get_loaded_layers():
    layer1 = Sdf.Layer.CreateAnonymous()
    stage1 = Usd.Stage.Open(layer1)
    stage1.DefinePrim("/TestPrim1")
    stage1.DefinePrim("/TestPrim2")
    layer2 = Sdf.Layer.CreateAnonymous()
    stage2 = Usd.Stage.Open(layer2)
    stage2.DefinePrim("/TestPrim3")
    stage2.DefinePrim("/TestPrim4")
    loaded_layers = get_loaded_layers()
    assert layer1 in loaded_layers
    assert layer2 in loaded_layers
    print([l.identifier for l in loaded_layers])


def get_layer_owner(layer: Sdf.Layer) -> str:
    """Get the owner of the given layer.

    Args:
        layer (Sdf.Layer): The layer to get the owner for.

    Returns:
        str: The owner of the layer, or an empty string if no owner is set.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    owner = layer.owner
    return owner if owner else ""


def get_layer_permission_to_save(layer: Sdf.Layer) -> bool:
    """Get the permission to save for a given layer.

    Args:
        layer (Sdf.Layer): The layer to check permission for.

    Returns:
        bool: True if the layer has permission to save, False otherwise.
    """
    if not layer:
        raise ValueError("Invalid layer")
    has_permission = layer.permissionToSave
    return has_permission


def get_layer_time_codes_per_second(layer: Sdf.Layer) -> float:
    """Get the timeCodesPerSecond value for a given layer.

    Args:
        layer (Sdf.Layer): The layer to get the timeCodesPerSecond value from.

    Returns:
        float: The timeCodesPerSecond value of the layer, or 24.0 if not set.
    """
    if layer.HasTimeCodesPerSecond():
        return layer.timeCodesPerSecond
    else:
        return 24.0


def has_layer_color_management_system(layer: Sdf.Layer) -> bool:
    """Check if a color management system is set on the given layer.

    Args:
        layer (Sdf.Layer): The layer to check for a color management system.

    Returns:
        bool: True if a color management system is set, False otherwise.
    """
    cms = layer.colorManagementSystem
    if cms:
        return True
    else:
        return False


def has_layer_end_time_code(layer: Sdf.Layer) -> bool:
    """Check if the given layer has an endTimeCode opinion."""
    return layer.HasEndTimeCode()


def has_layer_owner(layer: Sdf.Layer) -> bool:
    """Check if the given layer has an owner opinion."""
    return layer.HasOwner()


def has_layer_start_time_code(layer: Sdf.Layer) -> bool:
    """Check if the given layer has a startTimeCode opinion."""
    return layer.HasStartTimeCode()


def has_layer_time_codes_per_second(layer: Sdf.Layer) -> bool:
    """Check if a layer has timeCodesPerSecond opinion."""
    if not layer:
        raise ValueError("Invalid layer")
    time_codes_per_second = layer.timeCodesPerSecond
    if time_codes_per_second is not None:
        return True
    else:
        return False


def reload_layers(layers: Sequence[Sdf.Layer], force: bool = False) -> bool:
    """
    Reloads the specified layers.

    Args:
        layers (Sequence[Sdf.Layer]): The layers to reload.
        force (bool, optional): If True, force the reload even if the layer hasn't changed on disk. Defaults to False.

    Returns:
        bool: True if all layers were successfully reloaded, False otherwise.
    """
    all_reloaded = True
    for layer in layers:
        if not layer.Reload(force):
            all_reloaded = False
            continue
    return all_reloaded


def set_layer_custom_data(layer: Sdf.Layer, key: str, value: Any):
    """Set a custom data entry on the given layer.

    Args:
        layer (Sdf.Layer): The layer to set the custom data on.
        key (str): The key for the custom data entry.
        value (Any): The value to set for the custom data entry.
    """
    if not layer:
        raise ValueError("Invalid layer")
    if not key or not isinstance(key, str):
        raise ValueError("Invalid key")
    custom_data = layer.customLayerData
    if not custom_data:
        custom_data = {}
    custom_data[key] = value
    layer.customLayerData = custom_data


def get_layer_custom_data(layer: Sdf.Layer, key: str) -> Any:
    """Retrieves custom data with the given key from the layer.

    Args:
        layer (Sdf.Layer): The layer to retrieve custom data from.
        key (str): The key associated with the custom data.

    Returns:
        Any: The custom data value if found, None otherwise.
    """
    if not layer.HasCustomLayerData():
        return None
    custom_data = layer.customLayerData
    if key not in custom_data:
        return None
    return custom_data[key]


def remove_layer_custom_data(layer: Sdf.Layer, key: str) -> bool:
    """Remove a custom data entry from the layer by key.

    Args:
        layer (Sdf.Layer): The layer to remove custom data from.
        key (str): The key of the custom data to remove.

    Returns:
        bool: True if the key was found and removed, False otherwise.
    """
    if not layer.HasCustomLayerData():
        return False
    custom_data = layer.customLayerData
    if key not in custom_data:
        return False
    del custom_data[key]
    layer.customLayerData = custom_data
    return True


def get_layer_documentation(layer: Sdf.Layer) -> str:
    """Get the documentation string for a layer.

    Args:
        layer (Sdf.Layer): The layer to retrieve documentation from.

    Returns:
        str: The documentation string for the layer.
    """
    doc = layer.documentation
    return doc


def get_layer_frame_precision(layer: Sdf.Layer) -> int:
    """Get the frame precision for a given layer.

    Args:
        layer (Sdf.Layer): The layer to get the frame precision from.

    Returns:
        int: The frame precision value. If not set, returns 0.
    """
    if layer.HasFramePrecision():
        return layer.framePrecision
    else:
        return 0


def get_layer_color_configuration(layer: Sdf.Layer) -> str:
    """Get the color configuration metadata from the given layer.

    Args:
        layer (Sdf.Layer): The layer to retrieve the color configuration from.

    Returns:
        str: The color configuration asset path, or an empty string if not set.
    """
    if layer.HasColorConfiguration():
        color_config = layer.GetColorConfiguration()
        return color_config
    else:
        return ""


def get_layer_color_management_system(layer: Sdf.Layer) -> str:
    """Get the colorManagementSystem metadata from the given layer.

    Args:
        layer (Sdf.Layer): The layer to retrieve colorManagementSystem from.

    Returns:
        str: The colorManagementSystem value if authored, otherwise an empty string.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    if layer.HasColorManagementSystem():
        return layer.colorManagementSystem
    else:
        return ""


def get_layer_session_owner(layer: Sdf.Layer) -> str:
    """Get the session owner of the given layer.

    Args:
        layer (Sdf.Layer): The layer to get the session owner for.

    Returns:
        str: The session owner of the layer, or an empty string if not set.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    session_owner = layer.sessionOwner
    return session_owner if session_owner else ""


def create_new_layer_with_format(identifier: str, file_format: str) -> Sdf.Layer:
    """Create a new layer with the given identifier and file format."""
    if not identifier:
        raise ValueError("Identifier cannot be empty.")
    if not file_format:
        raise ValueError("File format cannot be None.")
    format_obj = Sdf.FileFormat.FindByExtension(file_format)
    if not format_obj:
        raise ValueError(f"Invalid file format: {file_format}")
    layer = Sdf.Layer.CreateNew(identifier, args={"format": format_obj})
    if not layer:
        raise RuntimeError(f"Failed to create layer with identifier '{identifier}' and format '{format_obj}'.")
    return layer


def get_attribute_at_path(layer: Sdf.Layer, path: Sdf.Path) -> Sdf.AttributeSpec:
    """
    Get the attribute spec at the given path in the specified layer.

    Args:
        layer (Sdf.Layer): The layer to query.
        path (Sdf.Path): The path to the attribute.

    Returns:
        Sdf.AttributeSpec: The attribute spec at the given path, or None if not found.
    """
    (prim_path, attr_name) = (path.GetPrimPath(), path.name)
    prim_spec = layer.GetPrimAtPath(prim_path)
    if not prim_spec:
        return None
    attr_spec = prim_spec.attributes.get(attr_name)
    return attr_spec


def clear_layer_custom_data(layer: Sdf.Layer) -> None:
    """Clears the custom data dictionary associated with the given layer."""
    if layer.HasCustomLayerData():
        empty_custom_data = {}
        layer.customLayerData = empty_custom_data
    else:
        pass


def has_layer_default_prim(layer: Sdf.Layer) -> bool:
    """Check if a layer has a default prim specified.

    Args:
        layer (Sdf.Layer): The layer to check for a default prim.

    Returns:
        bool: True if the layer has a default prim, False otherwise.
    """
    root_prim = layer.pseudoRoot
    if root_prim.HasInfo("defaultPrim"):
        return True
    else:
        return False


def is_anonymous_layer_identifier(identifier: str) -> bool:
    """Returns true if the identifier is an anonymous layer unique identifier."""
    if not identifier.startswith("anon:"):
        return False
    parts = identifier.split(":")
    if len(parts) < 2:
        return False
    try:
        uuid.UUID(parts[1])
    except ValueError:
        return False
    return True


def get_layer_comment(layer: Sdf.Layer) -> str:
    """Get the comment for a given layer.

    Args:
        layer (Sdf.Layer): The layer to get the comment from.

    Returns:
        str: The comment of the layer, or an empty string if no comment is set.
    """
    if not layer:
        raise ValueError("Invalid layer")
    comment = layer.comment
    return comment if comment else ""


def export_layer_to_file(layer: Sdf.Layer, file_path: str, comment: str = "", args: dict = None) -> bool:
    """
    Exports the given layer to a file on disk.

    Args:
        layer (Sdf.Layer): The layer to export.
        file_path (str): The file path to export the layer to.
        comment (str, optional): The comment to add to the exported layer. Defaults to "".
        args (dict, optional): Additional file format arguments. Defaults to None.

    Returns:
        bool: True if the export was successful, False otherwise.
    """
    if not layer:
        return False
    if not file_path:
        return False
    try:
        file_format_args = None
        if args:
            file_format_args = Sdf.FileFormatArguments(args)
        success = layer.Export(file_path, comment, file_format_args)
        return success
    except Exception as e:
        print(f"Error exporting layer: {str(e)}")
        return False


def reload_layer_with_force(layer: Sdf.Layer, force: bool = False) -> bool:
    """
    Reloads the given layer from its persistent representation.

    Args:
        layer (Sdf.Layer): The layer to reload.
        force (bool, optional): If True, forces the layer to be reloaded from disk
            regardless of whether it has changed. Defaults to False.

    Returns:
        bool: True if the layer was successfully reloaded, False otherwise.
    """
    if not layer:
        print(f"Error: Invalid layer.")
        return False
    if layer.identifier.isspace() or layer.identifier == "":
        print(f"Error: Layer has an empty identifier.")
        return False
    try:
        result = layer.Reload(force)
        return result
    except Exception as e:
        print(f"Error: Failed to reload layer: {str(e)}")
        return False


def set_layer_permission_to_edit(layer: Sdf.Layer, allow: bool) -> None:
    """Sets permission to edit the given layer.

    Args:
        layer (Sdf.Layer): The layer to set edit permission for.
        allow (bool): Whether to allow or disallow editing.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    layer.SetPermissionToEdit(allow)
    if layer.permissionToEdit != allow:
        raise RuntimeError("Failed to set permission to edit on the layer.")


def set_layer_time_sample(layer: Sdf.Layer, prim_path: str, attr_name: str, time: float, value: Any):
    """Sets a time sample on an attribute in the given layer.

    Args:
        layer (Sdf.Layer): The layer to set the time sample in.
        prim_path (str): The path to the prim containing the attribute.
        attr_name (str): The name of the attribute to set the time sample on.
        time (float): The time at which to set the sample.
        value (Any): The value to set at the given time.
    """
    prim_spec = layer.GetPrimAtPath(prim_path)
    if not prim_spec:
        raise ValueError(f"Prim at path {prim_path} does not exist in the layer.")
    attr_spec = prim_spec.properties.get(attr_name)
    if not attr_spec:
        attr_spec = Sdf.AttributeSpec(
            prim_spec, attr_name, Sdf.ValueTypeNames.Double, variability=Sdf.VariabilityVarying
        )
    layer.SetTimeSample(attr_spec.path, time, value)


def get_layer_asset_name(layer: Sdf.Layer) -> str:
    """Get the asset name associated with the given layer.

    Args:
        layer (Sdf.Layer): The layer to get the asset name from.

    Returns:
        str: The asset name of the layer, or an empty string if not set.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    asset_name = layer.GetDisplayName()
    return asset_name


def export_layer_to_string(layer: Sdf.Layer) -> str:
    """Export a layer to a string.

    Args:
        layer (Sdf.Layer): The layer to export.

    Returns:
        str: The exported layer as a string.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    layer_string = layer.ExportToString()
    if not layer_string:
        raise RuntimeError("Failed to export the layer to a string.")
    return layer_string


def get_layer_external_asset_dependencies(layer: Sdf.Layer) -> set:
    """Returns a set of resolved paths to all external asset dependencies the layer needs to generate its contents."""
    dependencies = set()
    for prim_spec in layer.rootPrims:

        def traverse(spec):
            if spec.HasField("references"):
                for ref in spec.referenceList.GetAddedOrExplicitItems():
                    dependencies.add(layer.ComputeAbsolutePath(ref.assetPath))
            if spec.HasField("payload"):
                for payload in spec.payloadList.GetAddedOrExplicitItems():
                    dependencies.add(layer.ComputeAbsolutePath(payload.assetPath))
            for child_spec in spec.nameChildren:
                traverse(child_spec)

        traverse(prim_spec)
    return dependencies


def get_layer_muted_state(layer: Sdf.Layer) -> bool:
    """Get the muted state of the given layer.

    Args:
        layer (Sdf.Layer): The layer to check the muted state of.

    Returns:
        bool: True if the layer is muted, False otherwise.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    is_muted = layer.IsMuted()
    return is_muted


def has_layer_session_owner(layer: Sdf.Layer) -> bool:
    """Check if the given layer has a session owner."""
    if not layer:
        raise ValueError("Invalid layer provided.")
    return layer.HasSessionOwner()


def import_layer_from_string(layer: Sdf.Layer, string: str) -> bool:
    """
    Import the contents of the given string into the layer.

    Args:
        layer (Sdf.Layer): The layer to import the string contents into.
        string (str): The string containing the layer contents to import.

    Returns:
        bool: True if the import was successful, False otherwise.
    """
    if not layer:
        return False
    if not string:
        return False
    anonymous_layer = Sdf.Layer.CreateAnonymous()
    success = anonymous_layer.ImportFromString(string)
    if success:
        layer.TransferContent(anonymous_layer)
    return success


def does_layer_stream_data(layer: Sdf.Layer) -> bool:
    """Check if the given layer streams data from its serialized data store."""
    if not layer:
        raise ValueError("Invalid layer provided.")
    streams_data = layer.StreamsData()
    return streams_data


def set_layer_documentation(layer: Sdf.Layer, documentation: str) -> None:
    """Set the documentation string for a given layer.

    Args:
        layer (Sdf.Layer): The layer to set documentation on.
        documentation (str): The documentation string to set.

    Raises:
        ValueError: If the given layer is invalid.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    layer.documentation = documentation


def set_layer_sub_layer_paths(layer: Sdf.Layer, sub_layer_paths: List[str]) -> None:
    """Set the sub-layer paths for a given layer.

    Args:
        layer (Sdf.Layer): The layer to set sub-layer paths on.
        sub_layer_paths (List[str]): The list of sub-layer paths to set.

    Raises:
        ValueError: If the input layer is invalid.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    layer.subLayerPaths = sub_layer_paths
    print(f"Set sub-layer paths on layer: {layer.identifier}")


def find_layers_matching_criteria(
    criteria: dict, layers: Optional[List[Sdf.Layer]] = None
) -> List[Tuple[Sdf.Layer, dict]]:
    """
    Find layers matching the given criteria.

    The criteria dictionary can contain the following keys:
        - hasOwnedSubLayers: bool
        - muted: bool

    If layers is None, all loaded layers will be searched.
    Returns a list of tuples containing the matching layer and the
    criteria it matched.
    """
    if layers is None:
        layers = Sdf.Layer.GetLoadedLayers()
    valid_criteria = ["hasOwnedSubLayers", "muted"]
    invalid_criteria = [c for c in criteria if c not in valid_criteria]
    if invalid_criteria:
        raise ValueError(f"Invalid criteria keys: {invalid_criteria}")

    def match_criteria(layer: Sdf.Layer) -> Optional[dict]:
        matched = {}
        for key, value in criteria.items():
            if key == "hasOwnedSubLayers":
                if layer.hasOwnedSubLayers != value:
                    return None
                matched[key] = value
            elif key == "muted":
                if layer.IsMuted() != value:
                    return None
                matched[key] = value
        return matched

    matches = []
    for layer in layers:
        matched_criteria = match_criteria(layer)
        if matched_criteria:
            matches.append((layer, matched_criteria))
    return matches


def set_layer_owner(layer: Sdf.Layer, owner: str) -> bool:
    """Set the owner of the given layer.

    Args:
        layer (Sdf.Layer): The layer to set the owner on.
        owner (str): The owner to set.

    Returns:
        bool: True if setting the owner succeeded, False otherwise.
    """
    if not layer:
        print(f"Invalid layer")
        return False
    if not layer.permissionToEdit:
        print(f"No permission to edit layer {layer.identifier}")
        return False
    layer.owner = owner
    result = layer.Save()
    if not result:
        print(f"Failed to save layer {layer.identifier}")
        return False
    return True


def get_layer_permission_to_edit(layer: Sdf.Layer) -> bool:
    """Get the permission to edit for a given layer.

    Args:
        layer (Sdf.Layer): The layer to check edit permission for.

    Returns:
        bool: True if the layer is permitted to be edited, False otherwise.
    """
    if not layer:
        return False
    return layer.permissionToEdit


def set_layer_permission_to_save(layer: Sdf.Layer, allow: bool) -> None:
    """Set the permission to save for the given layer.

    Args:
        layer (Sdf.Layer): The layer to set the permission for.
        allow (bool): Whether to allow or disallow saving the layer.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    layer.SetPermissionToSave(allow)
    if layer.permissionToSave != allow:
        raise RuntimeError("Failed to set permission to save on the layer.")


def get_layer_external_references(layer: Sdf.Layer) -> set[str]:
    """
    Returns a set of resolved paths to all external asset dependencies the
    layer needs to generate its contents.

    These are additional asset dependencies that are determined by the
    layer's file format and will be consulted during Reload() when
    determining if the layer needs to be reloaded. This specifically does
    not include dependencies related to composition, i.e. this will not
    include assets from references, payloads, and sublayers.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    external_refs = layer.GetExternalReferences()
    resolved_refs = {layer.ComputeAbsolutePath(ref) for ref in external_refs}
    return resolved_refs


def create_anonymous_layer_with_tag(tag: str, file_format: Sdf.FileFormat, args: dict = None) -> Sdf.Layer:
    """Create an anonymous layer with a tag, file format, and optional format arguments.

    Args:
        tag (str): The tag to use for the anonymous layer.
        file_format (Sdf.FileFormat): The file format to use for the anonymous layer.
        args (dict, optional): Additional file format arguments. Defaults to None.

    Returns:
        Sdf.Layer: The created anonymous layer.
    """
    if not tag:
        raise ValueError("Tag cannot be empty.")
    if not file_format:
        raise ValueError("File format cannot be None.")
    layer = Sdf.Layer.CreateAnonymous(tag, file_format, args or {})
    if not layer:
        raise RuntimeError("Failed to create anonymous layer.")
    return layer


def get_layer_bracketing_time_samples(layer: Sdf.Layer, path: Sdf.Path, time: float) -> Tuple[float, float]:
    """
    Get the bracketing time samples for a given time in a layer at a specific path.

    Args:
        layer (Sdf.Layer): The layer to query for bracketing time samples.
        path (Sdf.Path): The path to query for time samples.
        time (float): The time value to find bracketing samples for.

    Returns:
        Tuple[float, float]: A tuple containing the lower and upper bracketing time samples.
                             If no samples exist, returns (None, None).
    """
    path_times = layer.ListTimeSamplesForPath(path)
    if not path_times:
        return (None, None)
    lower_bracket = None
    upper_bracket = None
    for sample_time in path_times:
        if sample_time <= time:
            lower_bracket = sample_time
        else:
            upper_bracket = sample_time
            break
    return (lower_bracket, upper_bracket)


def save_layer_with_force(layer: Sdf.Layer, force: bool = False) -> bool:
    """Save the given layer to disk with optional force flag.

    Args:
        layer (Sdf.Layer): The layer to save.
        force (bool, optional): If True, force the save even if the layer is not dirty. Defaults to False.

    Returns:
        bool: True if the save was successful, False otherwise.
    """
    if not layer:
        return False
    file_path = layer.realPath
    if not file_path:
        return False
    if not layer.dirty and (not force):
        return True
    try:
        success = layer.Save(force=force)
        return success
    except Exception as e:
        print(f"Error saving layer: {e}")
        return False


def get_layer_asset_info(layer: Sdf.Layer) -> Dict[str, Any]:
    """Get the asset info for a given layer.

    Args:
        layer (Sdf.Layer): The layer to retrieve asset info from.

    Returns:
        Dict[str, Any]: A dictionary containing the layer's asset info.
    """
    asset_info = layer.GetAssetInfo()
    if not asset_info:
        return {}
    asset_info_dict = asset_info.GetDictionary()
    if not asset_info_dict:
        return {}
    return asset_info_dict


def apply_namespace_edits(layer: Sdf.Layer, namespace_edits: Sdf.BatchNamespaceEdit) -> bool:
    """Apply a batch of namespace edits to the given layer.

    Args:
        layer (Sdf.Layer): The layer to apply the edits to.
        namespace_edits (Sdf.BatchNamespaceEdit): The batch of namespace edits to apply.

    Returns:
        bool: True if the edits were applied successfully, False otherwise.
    """
    result = layer.Apply(namespace_edits)
    return result


def get_layer_root_prim_order(layer: Sdf.Layer) -> list[str]:
    """Get the root prim order for a given layer.

    Args:
        layer (Sdf.Layer): The layer to get the root prim order from.

    Returns:
        list[str]: The list of root prim names in the order specified by the layer.
    """
    root_prim_order = layer.pseudoRoot.GetInfo("primOrder")
    if root_prim_order is None:
        return []
    root_prim_order_list = list(root_prim_order)
    return root_prim_order_list


def is_layer_detached(layer: Sdf.Layer) -> bool:
    """Check if the given layer is detached.

    Args:
        layer (Sdf.Layer): The layer to check.

    Returns:
        bool: True if the layer is detached, False otherwise.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    identifier = layer.identifier
    is_detached = Sdf.Layer.IsIncludedByDetachedLayerRules(identifier)
    return is_detached


def add_to_muted_layers(muted_path: str):
    """Add the specified path to the muted layers set."""
    if not isinstance(muted_path, str) or not muted_path:
        raise ValueError("muted_path must be a non-empty string")
    muted_layer = Sdf.Layer.Find(muted_path)
    if not muted_layer:
        raise ValueError(f"Layer not found at path: {muted_path}")
    if muted_layer.IsMuted():
        return
    muted_layer.SetMuted(True)


def find_layer(layer_path: str) -> Optional[Sdf.Layer]:
    """
    Find and return the layer with the given path if it exists.

    Args:
        layer_path (str): The path of the layer to find.

    Returns:
        Optional[Sdf.Layer]: The found layer or None if it doesn't exist.
    """
    if not layer_path:
        return None
    layer = Sdf.Layer.Find(layer_path)
    return layer


def list_all_layer_time_samples(layer: Sdf.Layer) -> Set[float]:
    """List all time samples in the given layer."""
    time_samples: Set[float] = set()

    def traverse(path: Sdf.Path):
        spec = layer.GetObjectAtPath(path)
        if spec:
            if spec.typeName == "PrimSpec":
                for prop_name in spec.GetPropertyNames():
                    prop_path = path.AppendProperty(prop_name)
                    traverse(prop_path)
            elif spec.typeName in ["AttributeSpec", "RelationshipSpec"]:
                num_samples = spec.layer.GetNumTimeSamplesForPath(spec.path)
                if num_samples > 0:
                    for time_sample in spec.layer.ListTimeSamplesForPath(spec.path):
                        time_samples.add(time_sample)

    traverse(Sdf.Path.absoluteRootPath)
    return time_samples


def get_layer_version(layer: Sdf.Layer) -> str:
    """Get the version of the given layer.

    Args:
        layer (Sdf.Layer): The layer to get the version from.

    Returns:
        str: The version of the layer, or an empty string if no version is set.
    """
    version = layer.customLayerData.get("version", "")
    if version:
        return version
    else:
        return ""


def include_all_rules() -> bool:
    """Include all rules in the detached layer."""
    rules = Sdf.Layer.DetachedLayerRules()
    if rules.IncludedAll():
        return True
    rules.IncludeAll()
    if not rules.IncludedAll():
        raise RuntimeError("Failed to include all rules in the detached layer.")
    return True


def exclude_rules_by_name(rules: Sdf.Layer.DetachedLayerRules, rule_names: Union[str, List[str]]):
    """Exclude the specified rules by name.

    Args:
        rules (Sdf.Layer.DetachedLayerRules): The DetachedLayerRules instance.
        rule_names (Union[str, List[str]]): The name or list of names of the rules to exclude.

    Returns:
        None
    """
    if isinstance(rule_names, str):
        rule_names = [rule_names]
    rules_to_exclude = []
    for rule_name in rule_names:
        if not isinstance(rule_name, str):
            raise TypeError(f"Rule name must be a string, got {type(rule_name)}")
        if rules.IsIncluded(rule_name):
            rules_to_exclude.append(rule_name)
    if rules_to_exclude:
        rules.Exclude(rules_to_exclude)


def apply_layer_offset_to_prim(prim: Usd.Prim, offset: Sdf.LayerOffset):
    """Apply a layer offset to a prim.

    Args:
        prim (Usd.Prim): The prim to apply the offset to.
        offset (Sdf.LayerOffset): The layer offset to apply.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    translate_op = None
    scale_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
    if scale_op:
        old_scale = scale_op.Get()
        new_scale = Gf.Vec3f(*(s * offset.scale for s in old_scale)) if old_scale else Gf.Vec3f(offset.scale)
        scale_op.Set(new_scale)
    else:
        add_scale_op(xformable).Set(Gf.Vec3f(offset.scale))
    if translate_op:
        old_translation = translate_op.Get()
        new_translation = (
            Gf.Vec3f(*(t + offset.offset for t in old_translation)) if old_translation else Gf.Vec3f(offset.offset)
        )
        translate_op.Set(new_translation)
    else:
        add_translate_op(xformable).Set(Gf.Vec3f(offset.offset))


def check_expired_layers_in_tree(layer_tree: Sdf.LayerTree) -> bool:
    """
    Check if any layer in the given layer tree is expired.

    Args:
        layer_tree (Sdf.LayerTree): The layer tree to check for expired layers.

    Returns:
        bool: True if any layer in the tree is expired, False otherwise.
    """
    if layer_tree.expired:
        return True
    for child_tree in layer_tree.childTrees:
        if check_expired_layers_in_tree(child_tree):
            return True
    return False


def convert_scene_units(stage: Usd.Stage, target_unit: str, scale_metersPerUnit: float = 1.0) -> None:
    """Convert the linear units of a USD stage to the specified target unit.

    Args:
        stage (Usd.Stage): The USD stage to convert the units of.
        target_unit (str): The target unit to convert to (e.g., "cm", "m", "ft").
        scale_metersPerUnit (float, optional): Custom conversion scale if target_unit is not a standard unit.
                                               Defaults to 1.0.

    Raises:
        ValueError: If the stage is invalid or the target unit is not supported.
    """
    if not stage:
        raise ValueError("Invalid USD stage.")
    current_metersPerUnit = UsdGeom.GetStageMetersPerUnit(stage)
    if target_unit == "mm":
        conversion_factor = current_metersPerUnit * 1000
    elif target_unit == "cm":
        conversion_factor = current_metersPerUnit * 100
    elif target_unit == "m":
        conversion_factor = current_metersPerUnit
    elif target_unit == "km":
        conversion_factor = current_metersPerUnit * 0.001
    elif target_unit == "in":
        conversion_factor = current_metersPerUnit * 39.3701
    elif target_unit == "ft":
        conversion_factor = current_metersPerUnit * 3.28084
    elif target_unit == "yd":
        conversion_factor = current_metersPerUnit * 1.09361
    else:
        conversion_factor = current_metersPerUnit * scale_metersPerUnit
    UsdGeom.SetStageMetersPerUnit(stage, conversion_factor)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)


def merge_list_op_values(
    stronger_value: Optional[Sdf.LayerOffset], weaker_value: Optional[Sdf.LayerOffset], op_type: Sdf.ListOpType
) -> Optional[Sdf.LayerOffset]:
    """Merge two Sdf.LayerOffset values based on the specified Sdf.ListOpType.

    Args:
        stronger_value (Optional[Sdf.LayerOffset]): The stronger value to merge.
        weaker_value (Optional[Sdf.LayerOffset]): The weaker value to merge.
        op_type (Sdf.ListOpType): The list operation type to apply during the merge.

    Returns:
        Optional[Sdf.LayerOffset]: The merged result, or None if both inputs are None.
    """
    if stronger_value is None and weaker_value is None:
        return None
    if stronger_value is None:
        return weaker_value
    if weaker_value is None:
        return stronger_value
    if op_type == Sdf.ListOpTypeExplicit:
        return stronger_value
    elif op_type == Sdf.ListOpTypeAdded:
        return Sdf.LayerOffset(stronger_value.offset + weaker_value.offset, stronger_value.scale * weaker_value.scale)
    elif op_type == Sdf.ListOpTypeAppended:
        return Sdf.LayerOffset(
            stronger_value.offset + stronger_value.scale * weaker_value.offset,
            stronger_value.scale * weaker_value.scale,
        )
    elif op_type == Sdf.ListOpTypePrepended:
        return Sdf.LayerOffset(
            weaker_value.offset + weaker_value.scale * stronger_value.offset, weaker_value.scale * stronger_value.scale
        )
    else:
        raise ValueError(f"Unsupported list operation type: {op_type}")


def bulk_rename_prims(stage: Usd.Stage, prim_paths: List[str], new_names: List[str]) -> None:
    """Bulk rename prims given their paths and new names.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths to rename.
        new_names (List[str]): List of new names for the prims.

    Raises:
        ValueError: If the number of prim paths and new names don't match.
    """
    if len(prim_paths) != len(new_names):
        raise ValueError("Number of prim paths and new names must match.")
    edits = Sdf.BatchNamespaceEdit()
    for prim_path, new_name in zip(prim_paths, new_names):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        edit = Sdf.NamespaceEdit.Rename(prim_path, new_name)
        edits.Add(edit)
    stage.GetRootLayer().Apply(edits)


def rename_and_reparent_prim(
    stage: Usd.Stage, current_path: str, new_parent_path: str, new_name: str, index: int = -1
) -> bool:
    """Renames and reparents a prim in a USD stage.

    Args:
        stage (Usd.Stage): The USD stage.
        current_path (str): The current path of the prim.
        new_parent_path (str): The new parent path for the prim.
        new_name (str): The new name for the prim.
        index (int, optional): The index to insert the prim at under the new parent. Defaults to -1 (the end).

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    prim = stage.GetPrimAtPath(current_path)
    if not prim.IsValid():
        print(f"Error: Prim at path {current_path} does not exist.")
        return False
    new_parent_prim = stage.GetPrimAtPath(new_parent_path)
    if not new_parent_prim.IsValid():
        print(f"Error: New parent prim at path {new_parent_path} does not exist.")
        return False
    edit = Sdf.NamespaceEdit.ReparentAndRename(Sdf.Path(current_path), Sdf.Path(new_parent_path), new_name, index)
    batch_edit = Sdf.BatchNamespaceEdit()
    batch_edit.Add(edit)
    stage.GetRootLayer().Apply(batch_edit)
    return True


def reparent_and_reorder_prim(stage: Usd.Stage, prim_path: str, new_parent_path: str, name: str, index: int) -> None:
    """Reparent and reorder a prim in a USD stage.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path of the prim to reparent and reorder.
        new_parent_path (str): The path of the new parent prim.
        name (str): The new name for the prim.
        index (int): The new index for the prim under the new parent.

    Raises:
        ValueError: If the prim or new parent prim does not exist.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    new_parent_prim = stage.GetPrimAtPath(new_parent_path)
    if not new_parent_prim:
        raise ValueError(f"New parent prim at path {new_parent_path} does not exist.")
    edit = Sdf.NamespaceEdit()
    edit.ReparentAndRename(prim.GetPath(), new_parent_prim.GetPath(), name, index)
    batch_edit = Sdf.BatchNamespaceEdit()
    batch_edit.Add(edit)
    stage.GetRootLayer().Apply(batch_edit)


def get_namespace_edit_details(edit_detail: Sdf.NamespaceEditDetail) -> Optional[Sdf.NamespaceEdit]:
    """Get the namespace edit from a NamespaceEditDetail.

    Args:
        edit_detail (Sdf.NamespaceEditDetail): The NamespaceEditDetail to get the edit from.

    Returns:
        Optional[Sdf.NamespaceEdit]: The namespace edit if available, otherwise None.
    """
    if edit_detail.edit is not None:
        return edit_detail.edit
    else:
        return None


@staticmethod
def GetValueFromName(name: str) -> Optional["Sdf.NamespaceEditDetail.Result"]:
    """Get the Result enum value corresponding to the given name.

    Args:
        name (str): The name of the Result enum value.

    Returns:
        Optional[Sdf.NamespaceEditDetail.Result]: The corresponding Result enum value,
        or None if the name does not match any value in the Result enum.
    """
    if name == "Error":
        return Sdf.NamespaceEditDetail.Result.Error
    elif name == "Unbatched":
        return Sdf.NamespaceEditDetail.Result.Unbatched
    elif name == "Okay":
        return Sdf.NamespaceEditDetail.Result.Okay
    else:
        return None


def get_notices_of_type(notices: List[Tf.Notice], notice_type: type) -> List[Tf.Notice]:
    """Return a list of notices that match the given notice type."""
    if not isinstance(notices, list):
        raise TypeError("notices must be a list")
    if not isinstance(notice_type, type):
        raise TypeError("notice_type must be a type")
    filtered_notices = [notice for notice in notices if isinstance(notice, notice_type)]
    return filtered_notices


class TestNotice(Tf.Notice):
    pass


class AnotherNotice(Tf.Notice):
    pass


def search_and_filter_prims(stage: Usd.Stage, filter_func: Callable[[Usd.Prim], bool]) -> List[Usd.Prim]:
    """
    Search for prims in the stage and return those that match the filter function.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        filter_func (Callable[[Usd.Prim], bool]): A function that takes a Usd.Prim as input and returns a boolean indicating whether to include the prim in the result.

    Returns:
        List[Usd.Prim]: A list of prims that match the filter function.
    """
    matching_prims: List[Usd.Prim] = []
    for prim in stage.Traverse():
        if prim.IsValid():
            if filter_func(prim):
                matching_prims.append(prim)
    return matching_prims


def is_mesh(prim: Usd.Prim) -> bool:
    """Filter function to check if a prim is a mesh."""
    return prim.IsA(UsdGeom.Mesh)


def monitor_layer_reloads(layers: List[Sdf.Layer]) -> Dict[Sdf.Layer, int]:
    """Monitor layer reloads and return a dictionary of layer to reload count.

    Args:
        layers (List[Sdf.Layer]): The layers to monitor for reloads.

    Returns:
        Dict[Sdf.Layer, int]: A dictionary mapping each layer to the number of times it was reloaded.
    """
    reload_counts: Dict[Sdf.Layer, int] = {layer: 0 for layer in layers}

    def _on_layer_reloaded(notice, sender):
        reload_counts[sender] += 1

    for layer in layers:
        Tf.Notice.Register(Sdf.Notice.LayerDidReloadContent, _on_layer_reloaded, layer)
    return reload_counts


def monitor_layer_content_replacement(layer: Sdf.Layer, notice: Sdf.Notice.LayerDidReplaceContent) -> None:
    """Monitor when a layer's content is replaced and print the old and new identifier.

    Args:
        layer (Sdf.Layer): The layer whose content was replaced.
        notice (Sdf.Notice.LayerDidReplaceContent): The notice containing details about the content replacement.
    """
    old_identifier = notice.GetOldIdentifier()
    new_identifier = notice.GetNewIdentifier()
    print(f"Layer content replaced for {layer.identifier}")
    print(f"Old identifier: {old_identifier}")
    print(f"New identifier: {new_identifier}")


class MockNotice(Sdf.Notice.LayerDidReplaceContent):

    def __init__(self, old_id, new_id):
        self._oldId = old_id
        self._newId = new_id

    def GetOldIdentifier(self):
        return self._oldId

    def GetNewIdentifier(self):
        return self._newId


def replace_layer_content(layer: Sdf.Layer, new_content: str) -> None:
    """Replace the content of a layer with new content.

    Args:
        layer (Sdf.Layer): The layer to replace the content of.
        new_content (str): The new content to set for the layer.

    Raises:
        ValueError: If the layer is invalid or the new content is empty.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    if not new_content:
        raise ValueError("New content cannot be empty.")
    layer.ImportFromString(new_content)
    layer.Save()


def track_layer_dirtiness(notice: Sdf.Notice, dirty_layers: List[Sdf.Layer]) -> None:
    """Track the dirtiness of layers in a stage.

    Args:
        notice (Sdf.Notice): The notice containing information about the layer dirtiness change.
        dirty_layers (List[Sdf.Layer]): A list to store the dirty layers.
    """
    if isinstance(notice, Sdf.Notice.LayerDirtinessChanged):
        layer = notice.GetLayer()
        if layer.IsDirty():
            dirty_layers.append(layer)


def notice_wrapper(notice, sender):
    track_layer_dirtiness(notice, dirty_layers)


def isolate_dirty_layers(notice: Sdf.Notice) -> List[Sdf.Layer]:
    """Isolate the dirty layers from a notice.

    Args:
        notice (Sdf.Notice): The notice to process.

    Returns:
        List[Sdf.Layer]: The list of dirty layers.
    """
    dirty_layers = []
    if isinstance(notice, Sdf.Notice.LayerDirtinessChanged):
        affected_layer = notice.GetLayer()
        if affected_layer.IsDirty():
            dirty_layers.append(affected_layer)
    return dirty_layers


def notice_listener(notice, sender):
    result = isolate_dirty_layers(notice)
    print(result)


def handle_layer_identifier_change(notice: Sdf.Notice.LayerIdentifierDidChange) -> Optional[str]:
    """Handle a layer identifier change notice.

    Args:
        notice (Sdf.Notice.LayerIdentifierDidChange): The notice object containing the change information.

    Returns:
        Optional[str]: The new identifier if available, otherwise None.
    """
    if notice.newIdentifier is not None:
        new_identifier = notice.newIdentifier
        return new_identifier
    else:
        old_identifier = notice.oldIdentifier
        return None


class MockNotice:

    def __init__(self, old_identifier, new_identifier):
        self.oldIdentifier = old_identifier
        self.newIdentifier = new_identifier


def monitor_layer_info_changes(layer: Sdf.Layer) -> None:
    """Monitor changes to a layer's info.

    Args:
        layer (Sdf.Layer): The layer to monitor for info changes.
    """

    def on_layer_info_changed(notice: Sdf.Notice, sender: Sdf.Layer):
        if sender == layer:
            print(f"Layer info changed for {layer.identifier}")
            print(f"  New layer info: {layer.pseudoRoot.GetInfo()}")

    Tf.Notice.Register(Sdf.Notice.LayerInfoDidChange, on_layer_info_changed, layer)
    layer._info_listener = on_layer_info_changed


def handle_layer_muteness_change(notice) -> None:
    """Handle layer muteness change notice.

    Args:
        notice: Notice object containing layer muteness change details.
    """
    layer_path = notice.layerPath
    if not layer_path:
        return
    was_muted = notice.wasMuted
    if was_muted is None:
        return
    layer = Sdf.Layer.Find(layer_path)
    if not layer:
        return
    is_muted = layer.IsMuted()
    if was_muted == is_muted:
        return
    if is_muted:
        print(f"Layer '{layer_path}' is now muted.")
    else:
        print(f"Layer '{layer_path}' is now unmuted.")


class MockNotice:

    def __init__(self, layer_path, was_muted):
        self.layerPath = layer_path
        self.wasMuted = was_muted


class LayerMutenessChangedNotice:

    def __init__(self, layer_path: str, was_muted: bool):
        self.layerPath = layer_path
        self.wasMuted = was_muted


def update_muted_layers_references(notice: LayerMutenessChangedNotice, layer_muted_dict: Dict[str, bool]) -> None:
    """
    Update the dictionary of muted layers based on the received LayerMutenessChanged notice.

    Args:
        notice (LayerMutenessChangedNotice): The notice containing information about the layer muteness change.
        layer_muted_dict (Dict[str, bool]): A dictionary mapping layer paths to their muteness state.

    Returns:
        None
    """
    layer_path = notice.layerPath
    if layer_path is None or not isinstance(layer_path, str):
        raise ValueError("Invalid layer path in the notice.")
    was_muted = notice.wasMuted
    if was_muted is None or not isinstance(was_muted, bool):
        raise ValueError("Invalid previous muteness state in the notice.")
    layer_muted_dict[layer_path] = not was_muted


def get_changed_layer_paths(layers: List[Sdf.Layer]) -> List[str]:
    """Get the paths of the changed layers.

    Args:
        layers (List[Sdf.Layer]): The list of changed layers.

    Returns:
        List[str]: The paths of the changed layers.
    """
    if not layers:
        return []
    changed_layer_paths = [layer.identifier for layer in layers]
    return changed_layer_paths


def find_prims_with_changes(notice: Sdf.Notice.LayersDidChangeSentPerLayer) -> List[Sdf.Path]:
    """
    Find the prim paths that have changes based on the given LayersDidChangeSentPerLayer notice.

    Args:
        notice (Sdf.Notice.LayersDidChangeSentPerLayer): The notice containing information about layer changes.

    Returns:
        List[Sdf.Path]: A list of prim paths that have changes.
    """
    changed_prim_paths = []
    layers = notice.GetLayers()
    for layer in layers:
        pseudo_root = layer.pseudoRoot
        prim_specs = pseudo_root.nameChildren
        changed_prim_paths.extend([Sdf.Path(prim_spec.path) for prim_spec in prim_specs])
    return changed_prim_paths


class MockLayersDidChangeSentPerLayer:

    def GetLayers(self):
        return [stage.GetRootLayer()]


def align_prims_to_grid(stage: Usd.Stage, grid_size: float = 10.0):
    """Align all prims on the stage to a uniform grid.

    Args:
        stage (Usd.Stage): The stage containing the prims to align.
        grid_size (float, optional): The size of the grid cells. Defaults to 10.0.
    """
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        xformable = UsdGeom.Xformable(prim)
        translate_op = xformable.GetOrderedXformOps()[-1]
        if translate_op.GetOpType() != UsdGeom.XformOp.TypeTranslate:
            continue
        translation = translate_op.Get()
        rounded_translation = Gf.Vec3d(
            round(translation[0] / grid_size) * grid_size,
            round(translation[1] / grid_size) * grid_size,
            round(translation[2] / grid_size) * grid_size,
        )
        translate_op.Set(rounded_translation)


def create_prim(stage, prim_type, path):
    prim = stage.DefinePrim(path, prim_type)
    xform = UsdGeom.Xform(prim)
    add_translate_op(xform).Set(Gf.Vec3d(1.23, 4.56, 7.89))
    return prim


def find_prims_with_transform(stage: Usd.Stage, prim_path: str, transform_type: str) -> list[Usd.Prim]:
    """
    Find all prims under the given prim path that have the specified transform type.

    Args:
        stage (Usd.Stage): The USD stage to search.
        prim_path (str): The path of the prim to start the search from.
        transform_type (str): The type of transform to search for (e.g., "translate", "rotate", "scale").

    Returns:
        list[Usd.Prim]: A list of prims that have the specified transform type.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    matching_prims = []
    for descendant_prim in Usd.PrimRange(prim):
        xformable = UsdGeom.Xformable(descendant_prim)
        if xformable:
            xform_ops = xformable.GetOrderedXformOps()
            for op in xform_ops:
                if op.GetOpType().displayName.lower() == transform_type.lower():
                    matching_prims.append(descendant_prim)
                    break
    return matching_prims


def find_prims_with_material(stage: Usd.Stage, material_path: str) -> list[Usd.Prim]:
    """Find all prims on the stage that have a specific material bound.

    Args:
        stage (Usd.Stage): The USD stage to search.
        material_path (str): The path to the material to search for.

    Returns:
        list[Usd.Prim]: A list of prims that have the specified material bound.
    """
    if not stage or not stage.GetPrimAtPath(material_path):
        raise ValueError(f"Invalid stage or material path: {material_path}")
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsA(UsdShade.Material):
        raise ValueError(f"Prim at path {material_path} is not a Material")
    prims_with_material = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        binding_api = UsdShade.MaterialBindingAPI(prim)
        if binding_api.GetDirectBindingRel().GetTargets():
            bound_material_path = binding_api.GetDirectBindingRel().GetTargets()[0]
            if bound_material_path == material_prim.GetPath():
                prims_with_material.append(prim)
    return prims_with_material


def batch_apply_variant(stage, prim_paths, variant_set, variant):
    """Apply a variant selection to a list of prims.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (list[Sdf.Path]): A list of prim paths to apply the variant selection to.
        variant_set (str): The name of the variant set.
        variant (str): The name of the variant to select.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print("Warning: Prim at path {} does not exist. Skipping.".format(prim_path))
            continue
        if not prim.HasVariantSets():
            print("Warning: Prim at path {} does not have any variant sets. Skipping.".format(prim_path))
            continue
        var_sets = prim.GetVariantSets()
        if not var_sets.HasVariantSet(variant_set):
            print("Warning: Prim at path {} does not have variant set '{}'. Skipping.".format(prim_path, variant_set))
            continue
        var_set = var_sets.GetVariantSet(variant_set)
        if not var_set.HasVariant(variant):
            print(
                "Warning: Variant set '{}' on prim at path {} does not have variant '{}'. Skipping.".format(
                    variant_set, prim_path, variant
                )
            )
            continue
        var_set.SetVariantSelection(variant)


def batch_delete_prims(stage: Usd.Stage, prim_paths: list[Sdf.Path]) -> None:
    """Delete multiple prims from a USD stage.

    Args:
        stage (Usd.Stage): The USD stage to delete prims from.
        prim_paths (list[Sdf.Path]): A list of prim paths to delete.

    Raises:
        ValueError: If any of the prim paths are invalid or do not exist on the stage.
    """
    with Sdf.ChangeBlock():
        for prim_path in prim_paths:
            if not Sdf.Path.IsValidPathString(str(prim_path)):
                raise ValueError(f"Invalid prim path: {prim_path}")
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise ValueError(f"Prim does not exist: {prim_path}")
            stage.RemovePrim(prim_path)


def test_batch_delete_prims():
    stage = Usd.Stage.CreateInMemory()
    prim_paths = [Sdf.Path("/World"), Sdf.Path("/World/Prim1"), Sdf.Path("/World/Prim2"), Sdf.Path("/World/Prim3")]
    for prim_path in prim_paths:
        stage.DefinePrim(prim_path)
    delete_paths = [Sdf.Path("/World/Prim1"), Sdf.Path("/World/Prim2")]
    batch_delete_prims(stage, delete_paths)
    for prim_path in delete_paths:
        assert not stage.GetPrimAtPath(prim_path)
    remaining_paths = [Sdf.Path("/World"), Sdf.Path("/World/Prim3")]
    for prim_path in remaining_paths:
        assert stage.GetPrimAtPath(prim_path)
    print(stage.GetRootLayer().ExportToString())


def find_prims_with_attributes(
    stage: Usd.Stage, attribute_names: List[str], prim_path: Sdf.Path = Sdf.Path.absoluteRootPath
) -> List[Sdf.Path]:
    """
    Recursively find all prims under the given prim path that have all the specified attributes.

    Args:
        stage (Usd.Stage): The USD stage to search.
        attribute_names (List[str]): The list of attribute names to match.
        prim_path (Sdf.Path, optional): The starting prim path for the search. Defaults to the absolute root path.

    Returns:
        List[Sdf.Path]: A list of prim paths that have all the specified attributes.
    """
    matching_prims: List[Sdf.Path] = []
    prim: Usd.Prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return matching_prims
    has_all_attributes: bool = True
    for attr_name in attribute_names:
        if not prim.HasAttribute(attr_name):
            has_all_attributes = False
            break
    if has_all_attributes:
        matching_prims.append(prim.GetPath())
    for child in prim.GetChildren():
        matching_prims.extend(find_prims_with_attributes(stage, attribute_names, child.GetPath()))
    return matching_prims


def find_prims_with_variants(stage: Usd.Stage) -> list[Usd.Prim]:
    """Find all prims in the stage that have variant sets defined.

    Args:
        stage (Usd.Stage): The USD stage to search for prims with variants.

    Returns:
        list[Usd.Prim]: A list of prims that have variant sets defined.
    """
    prims_with_variants: list[Usd.Prim] = []
    for prim in stage.TraverseAll():
        if prim.HasVariantSets():
            prims_with_variants.append(prim)
    return prims_with_variants


def create_material_with_textures(
    stage: Usd.Stage, material_path: str, diffuse_texture_path: str, normal_texture_path: str
) -> UsdShade.Material:
    """Create a material with diffuse and normal textures.

    Args:
        stage (Usd.Stage): The USD stage to create the material on.
        material_path (str): The path where the material should be created.
        diffuse_texture_path (str): The file path to the diffuse texture.
        normal_texture_path (str): The file path to the normal texture.

    Returns:
        UsdShade.Material: The created material prim.
    """
    if not Sdf.Path(material_path).IsPrimPath():
        raise ValueError(f"Invalid material path: {material_path}")
    material: UsdShade.Material = UsdShade.Material.Define(stage, material_path)
    shader: UsdShade.Shader = UsdShade.Shader.Define(stage, material_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    diffuse_texture: UsdShade.Shader = UsdShade.Shader.Define(stage, material_path + "/DiffuseTexture")
    diffuse_texture.CreateIdAttr("UsdUVTexture")
    diffuse_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(diffuse_texture_path)
    diffuse_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    normal_texture: UsdShade.Shader = UsdShade.Shader.Define(stage, material_path + "/NormalTexture")
    normal_texture.CreateIdAttr("UsdUVTexture")
    normal_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(normal_texture_path)
    normal_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        diffuse_texture.ConnectableAPI(), "rgb"
    )
    shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(normal_texture.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def find_prims_with_payloads(stage: Usd.Stage) -> list[Sdf.Path]:
    """Find all prims on the given stage that have payloads."""
    prims_with_payloads: list[Sdf.Path] = []
    for prim in stage.TraverseAll():
        if prim.HasPayload():
            prims_with_payloads.append(prim.GetPath())
    return prims_with_payloads


def batch_remove_payloads(stage: Usd.Stage, prim_paths: list[Sdf.Path]) -> int:
    """Remove payloads from the specified prims in batch.

    Args:
        stage (Usd.Stage): The USD stage to remove payloads from.
        prim_paths (list[Sdf.Path]): A list of prim paths to remove payloads from.

    Returns:
        int: The number of prims that had their payloads removed.
    """
    num_removed = 0
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        if prim.HasPayload():
            prim.SetPayload(Sdf.Payload())
            num_removed += 1
    return num_removed


def batch_create_sdf_paths(path_strings: List[str]) -> List[Sdf.Path]:
    """Create a list of Sdf.Path objects from a list of path strings.

    Args:
        path_strings (List[str]): A list of path strings.

    Returns:
        List[Sdf.Path]: A list of Sdf.Path objects.
    """
    sdf_paths: List[Sdf.Path] = []
    for path_string in path_strings:
        if not path_string:
            continue
        sdf_path = Sdf.Path(path_string)
        if sdf_path.isEmpty:
            continue
        sdf_paths.append(sdf_path)
    return sdf_paths


def resolve_relative_paths(anchor_path: Sdf.Path, relative_paths: list[Sdf.Path]) -> list[Sdf.Path]:
    """Resolve relative paths to absolute paths using the given anchor path.

    Args:
        anchor_path (Sdf.Path): The absolute path to use as the anchor for resolving relative paths.
        relative_paths (list[Sdf.Path]): A list of relative paths to resolve.

    Returns:
        list[Sdf.Path]: A list of absolute paths resolved from the relative paths.

    Raises:
        ValueError: If the anchor path is not an absolute path.
    """
    if not anchor_path.IsAbsolutePath():
        raise ValueError("Anchor path must be an absolute path.")
    resolved_paths = []
    for relative_path in relative_paths:
        if relative_path.IsAbsolutePath():
            resolved_paths.append(relative_path)
        else:
            resolved_path = relative_path.MakeAbsolutePath(anchor_path)
            resolved_paths.append(resolved_path)
    return resolved_paths


def batch_get_parent_paths(paths: List[Sdf.Path]) -> List[Sdf.Path]:
    """
    Given a list of paths, return a list of their parent paths.

    If a path is EmptyPath or AbsoluteRootPath, its parent path is EmptyPath.

    Args:
        paths (List[Sdf.Path]): A list of paths to get parent paths for.

    Returns:
        List[Sdf.Path]: A list of parent paths corresponding to the input paths.
    """
    parent_paths = []
    for path in paths:
        if path == Sdf.Path.emptyPath or path == Sdf.Path.absoluteRootPath:
            parent_paths.append(Sdf.Path.emptyPath)
        else:
            parent_path = path.GetParentPath()
            parent_paths.append(parent_path)
    return parent_paths


def create_hierarchical_prim(stage: Usd.Stage, prim_path: Sdf.Path) -> Usd.Prim:
    """
    Create a hierarchy of prims given a prim path.

    Args:
        stage (Usd.Stage): The stage to create the prims on.
        prim_path (Sdf.Path): The path of the prim to create.

    Returns:
        Usd.Prim: The created prim.

    Raises:
        Tf.ErrorException: If the prim path is not a valid prim path.
    """
    if not Sdf.Path.IsValidPathString(str(prim_path)):
        raise Tf.ErrorException(f"Invalid prim path: {prim_path}")
    parent_path = prim_path.GetParentPath()
    prim_name = prim_path.name
    if not parent_path.isEmpty and (not stage.GetPrimAtPath(parent_path)):
        create_hierarchical_prim(stage, parent_path)
    prim = stage.DefinePrim(prim_path)
    return prim


def replace_prim_and_property(path: Sdf.Path, new_prim_name: str, new_property_name: str) -> Sdf.Path:
    """
    Replace the prim and property names in a path.

    If the path is a prim path, only the prim name is replaced.
    If the path is a property path, both the prim and property names are replaced.
    If the path is not a prim or property path, an exception is raised.

    Args:
        path (Sdf.Path): The input path.
        new_prim_name (str): The new prim name.
        new_property_name (str): The new property name.

    Returns:
        Sdf.Path: The updated path with replaced prim and property names.

    Raises:
        ValueError: If the input path is not a prim or property path.
    """
    if path.IsAbsoluteRootOrPrimPath():
        parent_path = path.GetParentPath()
        return parent_path.AppendChild(new_prim_name)
    elif path.IsPrimPropertyPath():
        prim_path = path.GetPrimPath()
        new_prim_path = replace_prim_and_property(prim_path, new_prim_name, "")
        return new_prim_path.AppendProperty(new_property_name)
    else:
        raise ValueError(f"Invalid path type for replacement: {path}")


def batch_set_prim_transforms(stage: Usd.Stage, prim_paths: List[str], translations: List[Gf.Vec3d]) -> None:
    """Set translations on a batch of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths.
        translations (List[Gf.Vec3d]): List of translation vectors.

    Raises:
        ValueError: If the lengths of prim_paths and translations don't match.
    """
    if len(prim_paths) != len(translations):
        raise ValueError("Length of prim_paths and translations must match.")
    for prim_path, translation in zip(prim_paths, translations):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            print(f"Warning: Prim '{prim_path}' does not exist. Skipping.")
            continue
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            print(f"Warning: Prim '{prim_path}' is not transformable. Skipping.")
            continue
        translate_op = add_translate_op(xformable)
        translate_op.Set(translation)


def collect_prim_hierarchy_paths(prim: Usd.Prim, paths: List[Sdf.Path], include_root: bool = True) -> None:
    """Recursively collect paths for all prims in a hierarchy.

    Args:
        prim (Usd.Prim): The root prim to start collection from.
        paths (List[Sdf.Path]): The list to add collected paths to.
        include_root (bool, optional): Whether to include the root prim path. Defaults to True.
    """
    if not prim.IsValid():
        raise ValueError("Input prim is not valid.")
    if include_root or prim.GetPath() != Sdf.Path.absoluteRootPath:
        paths.append(prim.GetPath())
    for child in prim.GetChildren():
        collect_prim_hierarchy_paths(child, paths, include_root=False)


def batch_translate_prims(
    stage: Usd.Stage, prim_paths: List[Sdf.Path], translation: Tuple[float, float, float]
) -> None:
    """Batch translate multiple prims.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (List[Sdf.Path]): The paths of the prims to translate.
        translation (Tuple[float, float, float]): The translation to apply.
    """
    if len(translation) != 3:
        raise ValueError("Translation must be a tuple of 3 floats.")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            print(f"Warning: Prim at path {prim_path} is not transformable. Skipping.")
            continue
        translate_op = add_translate_op(xformable)
        translate_op.Set(Gf.Vec3d(translation))


def batch_set_prim_visibilities(stage: Usd.Stage, prim_path_visibility_map: Dict[Sdf.Path, bool]):
    """Sets the visibility for multiple prims in a single transaction.

    Args:
        stage (Usd.Stage): The stage to set visibilities on.
        prim_path_visibility_map (Dict[Sdf.Path, bool]): A dictionary mapping prim paths to their desired visibility state.
    """
    with Sdf.ChangeBlock():
        for prim_path, visibility in prim_path_visibility_map.items():
            if not Sdf.Path.IsValidPathString(str(prim_path)):
                raise ValueError(f"Invalid prim path: {prim_path}")
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise ValueError(f"Prim at path {prim_path} does not exist.")
            imageable = UsdGeom.Imageable(prim)
            if not imageable:
                raise ValueError(f"Prim at path {prim_path} is not imageable.")
            imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if visibility else UsdGeom.Tokens.invisible)


def batch_assign_textures_to_material(
    stage: Usd.Stage,
    material_prim_path: str,
    texture_file_paths: Dict[str, str],
    texture_filtering: str = "auto",
    texture_wrapping: str = "repeat",
) -> None:
    """Batch assign textures to a material.

    Args:
        stage (Usd.Stage): The USD stage.
        material_prim_path (str): The path to the material prim.
        texture_file_paths (Dict[str, str]): A dictionary mapping texture names to file paths.
        texture_filtering (str, optional): The texture filtering mode. Defaults to "auto".
        texture_wrapping (str, optional): The texture wrapping mode. Defaults to "repeat".

    Raises:
        ValueError: If the material prim does not exist or is not a valid material.
    """
    material_prim = stage.GetPrimAtPath(material_prim_path)
    if not material_prim.IsValid():
        raise ValueError(f"Material prim at path {material_prim_path} does not exist.")
    material = UsdShade.Material(material_prim)
    if not material:
        raise ValueError(f"Prim at path {material_prim_path} is not a valid material.")
    for texture_name, texture_file_path in texture_file_paths.items():
        shader_name = f"{texture_name}_shader"
        shader_prim = UsdShade.Shader.Define(stage, material_prim.GetPath().AppendChild(shader_name))
        shader_prim.CreateIdAttr("UsdUVTexture")
        shader_prim.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_file_path)
        shader_prim.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            material.CreateInput(f"{texture_name}Coords", Sdf.ValueTypeNames.Float2)
        )
        shader_prim.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set((0.0, 0.0, 0.0, 1.0))
        shader_prim.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set(texture_wrapping)
        shader_prim.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set(texture_wrapping)
        if texture_filtering == "auto":
            shader_prim.CreateInput("useMetadata", Sdf.ValueTypeNames.Bool).Set(True)
        else:
            shader_prim.CreateInput("useMetadata", Sdf.ValueTypeNames.Bool).Set(False)
            shader_prim.CreateInput("filter", Sdf.ValueTypeNames.Token).Set(texture_filtering)
        material.CreateInput(texture_name, Sdf.ValueTypeNames.Color3f).ConnectToSource(
            shader_prim.ConnectableAPI(), "rgb"
        )


def batch_add_keyframes_to_prims(
    stage: Usd.Stage, prim_paths: List[str], attribute_name: str, time_samples: List[Tuple[float, float]]
):
    """
    Batch add keyframes to multiple prims in a single call.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths to add keyframes to.
        attribute_name (str): The name of the attribute to set keyframes on.
        time_samples (List[Tuple[float, float]]): A list of (time, value) tuples representing the keyframes.
    """
    if not stage:
        raise ValueError("Invalid stage.")
    if not prim_paths or not time_samples:
        return
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        attribute = prim.GetAttribute(attribute_name)
        if not attribute:
            attribute = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Float)
        for time, value in time_samples:
            attribute.Set(value, Usd.TimeCode(time))


def batch_remove_keyframes_from_prims(stage: Usd.Stage, prim_paths: List[str], attr_name: str) -> None:
    """Remove keyframes from attributes on multiple prims.

    Args:
        stage (Usd.Stage): The stage to modify.
        prim_paths (List[str]): List of prim paths to remove keyframes from.
        attr_name (str): The name of the attribute to remove keyframes from.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            print(f"Warning: Prim {prim_path} does not exist. Skipping...")
            continue
        attr = prim.GetAttribute(attr_name)
        if not attr:
            print(f"Warning: Attribute {attr_name} does not exist on prim {prim_path}. Skipping...")
            continue
        attr.Clear()


def batch_append_prim_paths(stage: Usd.Stage, root_path: str, prim_names: List[str], prim_type: str) -> List[Sdf.Path]:
    """Batch append prims to a stage under a given root path.

    Args:
        stage (Usd.Stage): The stage to add the prims to.
        root_path (str): The root path where the prims will be appended.
        prim_names (List[str]): A list of prim names to create.
        prim_type (str): The type of the prims to create.

    Returns:
        List[Sdf.Path]: A list of the created prim paths.
    """
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim:
        raise ValueError(f"Root prim at path {root_path} does not exist.")
    prim_paths = []
    for prim_name in prim_names:
        prim_path = root_prim.GetPath().AppendChild(prim_name)
        if stage.GetPrimAtPath(prim_path):
            prim_paths.append(prim_path)
            continue
        new_prim = stage.DefinePrim(prim_path, prim_type)
        if new_prim:
            prim_paths.append(new_prim.GetPath())
        else:
            raise RuntimeError(f"Failed to create prim at path {prim_path}")
    return prim_paths


def batch_replace_prefixes(
    paths: List[Sdf.Path], old_prefixes: List[Sdf.Path], new_prefixes: List[Sdf.Path]
) -> List[Sdf.Path]:
    """Replace multiple path prefixes in one pass.

    For each path in paths, replace the longest prefix that matches a key in
    old_prefixes dict with the corresponding value in new_prefixes dict.
    Return a list of paths with replaced prefixes.

    Args:
        paths (List[Sdf.Path]): The list of paths to replace prefixes in.
        old_prefixes (List[Sdf.Path]): A list of old prefixes to replace.
        new_prefixes (List[Sdf.Path]): A list of new replacement prefixes.

    Returns:
        List[Sdf.Path]: A new list with the replaced paths.
    """
    if len(old_prefixes) != len(new_prefixes):
        raise ValueError("old_prefixes and new_prefixes must have the same length.")
    result = []
    for path in paths:
        max_prefix_len = -1
        max_prefix_idx = -1
        for idx, old_prefix in enumerate(old_prefixes):
            if path.HasPrefix(old_prefix):
                prefix_len = len(old_prefix.pathString)
                if prefix_len > max_prefix_len:
                    max_prefix_len = prefix_len
                    max_prefix_idx = idx
        if max_prefix_idx != -1:
            new_path = path.ReplacePrefix(
                old_prefixes[max_prefix_idx], new_prefixes[max_prefix_idx], fixTargetPaths=True
            )
            result.append(new_path)
        else:
            result.append(path)
    return result


def batch_join_identifiers(lhs_vec: Sequence[str], rhs_vec: Sequence[str]) -> list[str]:
    """Combine pairs of strings into new strings using SdfPath.JoinIdentifier.

    Args:
        lhs_vec (Sequence[str]): A sequence of left-hand side strings.
        rhs_vec (Sequence[str]): A sequence of right-hand side strings.

    Returns:
        list[str]: A list of joined strings.

    Raises:
        ValueError: If the input sequences are not the same length.
    """
    if len(lhs_vec) != len(rhs_vec):
        raise ValueError("Input sequences must be the same length.")
    result = []
    for lhs, rhs in zip(lhs_vec, rhs_vec):
        joined = Sdf.Path.JoinIdentifier(lhs, rhs)
        result.append(joined)
    return result


def batch_get_absolute_paths(paths: List[Sdf.Path], anchor: Sdf.Path) -> List[Sdf.Path]:
    """
    Given a list of paths, return a list of absolute paths using the provided anchor path.

    Args:
        paths (List[Sdf.Path]): List of paths to convert to absolute paths.
        anchor (Sdf.Path): The anchor path to use as the basis for relative paths.

    Returns:
        List[Sdf.Path]: List of absolute paths.
    """
    if not anchor.IsAbsolutePath():
        raise ValueError("Anchor path must be an absolute path.")
    absolute_paths = []
    for path in paths:
        if path.IsAbsolutePath():
            absolute_paths.append(path)
        else:
            absolute_path = path.MakeAbsolutePath(anchor)
            absolute_paths.append(absolute_path)
    return absolute_paths


def batch_get_relative_paths(paths: List[Sdf.Path]) -> List[Sdf.Path]:
    """
    Given a list of absolute paths, return a list of relative paths.

    The relative paths are constructed such that each is relative to
    the longest common prefix of all paths. If there is no common
    prefix, the original absolute paths are returned.
    """
    if not paths:
        return []
    common_prefix = paths[0]
    for path in paths[1:]:
        common_prefix = common_prefix.GetCommonPrefix(path)
    if common_prefix == Sdf.Path.emptyPath:
        return paths
    relative_paths = []
    for path in paths:
        if path.HasPrefix(common_prefix):
            relative_path = path.MakeRelativePath(common_prefix)
        else:
            relative_path = path
        relative_paths.append(relative_path)
    return relative_paths


def batch_check_valid_identifiers(identifiers: List[str]) -> List[Tuple[str, bool]]:
    """
    Check if a list of identifiers are valid.

    Args:
        identifiers (List[str]): List of identifiers to check.

    Returns:
        List[Tuple[str, bool]]: List of tuples containing the identifier and its validity status.
    """
    results = []
    for identifier in identifiers:
        try:
            is_valid = Sdf.Path.IsValidIdentifier(identifier)
            results.append((identifier, is_valid))
        except Exception as e:
            results.append((identifier, False))
    return results


def batch_check_valid_namespaced_identifiers(names: List[str]) -> List[bool]:
    """Check if a list of names are valid namespaced identifiers.

    Args:
        names (List[str]): A list of names to check.

    Returns:
        List[bool]: A list of booleans indicating if each name is a valid namespaced identifier.
    """
    results = []
    for name in names:
        tokens = Sdf.Path.TokenizeIdentifier(name)
        is_valid = False
        if tokens:
            stripped_name = Sdf.Path.StripNamespace(name)
            if Sdf.Path.IsValidIdentifier(stripped_name):
                is_valid = True
        results.append(is_valid)
    return results


def batch_strip_all_variant_selections(paths: Sequence[Sdf.Path]) -> list[Sdf.Path]:
    """Strip all variant selections from a sequence of paths.

    Args:
        paths (Sequence[Sdf.Path]): The input sequence of paths.

    Returns:
        list[Sdf.Path]: A new list with variant selections stripped from each path.
    """
    stripped_paths: list[Sdf.Path] = []
    for path in paths:
        stripped_path: Sdf.Path = path.StripAllVariantSelections()
        stripped_paths.append(stripped_path)
    return stripped_paths


def batch_strip_namespaces(names: List[str]) -> List[str]:
    """Strip namespaces from a list of names.

    Args:
        names (List[str]): A list of names to strip namespaces from.

    Returns:
        List[str]: A list of names with namespaces stripped.
    """
    stripped_names: List[str] = []
    for name in names:
        tokens: List[str] = Sdf.Path.TokenizeIdentifier(name)
        if not tokens:
            stripped_names.append(name)
        else:
            stripped_names.append(tokens[-1])
    return stripped_names


def batch_tokenize_identifiers(names: List[str]) -> List[List[str]]:
    """
    Tokenizes a list of identifiers by the namespace delimiter.

    Args:
        names (List[str]): List of identifiers to tokenize.

    Returns:
        List[List[str]]: List of tokenized identifiers. If an identifier is not a valid
            namespaced identifier, an empty list is returned for that identifier.
    """
    results = []
    for name in names:
        if Sdf.Path.IsValidIdentifier(name):
            tokens = name.split(":")
            results.append(tokens)
        else:
            results.append([])
    return results


def find_prims_with_property(stage: Usd.Stage, property_name: str) -> list[Usd.Prim]:
    """
    Find all prims on the given stage that have a specific property.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        property_name (str): The name of the property to search for.

    Returns:
        list[Usd.Prim]: A list of prims that have the specified property.
    """
    prims_with_property = []
    for prim in stage.Traverse():
        if prim.HasProperty(property_name):
            prims_with_property.append(prim)
    return prims_with_property


def reassign_materials(stage: Usd.Stage, material_mapping: dict[Sdf.Path, Sdf.Path]) -> None:
    """Reassign materials on a USD stage based on a material mapping dictionary.

    Args:
        stage (Usd.Stage): The USD stage to reassign materials on.
        material_mapping (dict[Sdf.Path, Sdf.Path]): A dictionary mapping old material paths to new material paths.
    """
    for prim in stage.Traverse():
        if UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel().GetTargets():
            old_material_path = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel().GetTargets()[0]
            if old_material_path in material_mapping:
                new_material_path = material_mapping[old_material_path]
                new_material_prim = stage.GetPrimAtPath(new_material_path)
                if not new_material_prim.IsValid():
                    raise ValueError(f"Invalid material path: {new_material_path}")
                new_material = UsdShade.Material(new_material_prim)
                UsdShade.MaterialBindingAPI(prim).Bind(new_material)


def batch_update_prim_properties(stage: Usd.Stage, updates: Dict[Sdf.Path, Dict[str, object]]) -> None:
    """Batch update properties on multiple prims in a stage.

    Args:
        stage (Usd.Stage): The stage to update prims on.
        updates (Dict[Sdf.Path, Dict[str, object]]): A dictionary mapping prim paths to
            a dictionary of property names and values to set.

    Raises:
        ValueError: If any of the prim paths are invalid or the properties fail to author.
    """
    for prim_path, props in updates.items():
        if not Sdf.Path(str(prim_path)).IsPrimPath():
            raise ValueError(f"Invalid prim path: {prim_path}")
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found at path: {prim_path}")
        for prop_name, prop_value in props.items():
            if isinstance(prop_value, bool):
                attr_type = Sdf.ValueTypeNames.Bool
            elif isinstance(prop_value, int):
                attr_type = Sdf.ValueTypeNames.Int
            elif isinstance(prop_value, float):
                attr_type = Sdf.ValueTypeNames.Float
            elif isinstance(prop_value, str):
                attr_type = Sdf.ValueTypeNames.String
            else:
                raise ValueError(f"Unsupported property value type: {type(prop_value)}")
            attr = prim.CreateAttribute(prop_name, attr_type)
            if not attr.IsValid():
                raise ValueError(f"Failed to create attribute {prop_name} on prim {prim_path}")
            if not attr.Set(prop_value):
                raise ValueError(f"Failed to set attribute {prop_name} to {prop_value} on prim {prim_path}")


def extract_subtree_as_new_layer(stage: Usd.Stage, prim_path: str, new_layer_identifier: str) -> Usd.Stage:
    """Extract a subtree from a USD stage and create a new stage with that subtree.

    Args:
        stage (Usd.Stage): The input USD stage.
        prim_path (str): The path to the root prim of the subtree to extract.
        new_layer_identifier (str): The identifier to use for the new layer.

    Returns:
        Usd.Stage: A new USD stage containing the extracted subtree.
    """
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")
    new_layer = Sdf.Layer.CreateNew(new_layer_identifier)
    new_stage = Usd.Stage.Open(new_layer)

    def copy_subtree(src_prim: Usd.Prim, dst_prim: Usd.Prim):
        for attr in src_prim.GetAttributes():
            if attr.HasValue():
                dst_attr = dst_prim.CreateAttribute(attr.GetName(), attr.GetTypeName())
                dst_attr.Set(attr.Get())
        for rel in src_prim.GetRelationships():
            dst_rel = dst_prim.CreateRelationship(rel.GetName())
            dst_rel.SetTargets(rel.GetTargets())
        for child in src_prim.GetChildren():
            dst_child = dst_prim.CreateChild(child.GetName(), child.GetTypeName())
            copy_subtree(child, dst_child)

    dst_root_prim = new_stage.DefinePrim(root_prim.GetPath(), root_prim.GetTypeName())
    copy_subtree(root_prim, dst_root_prim)
    return new_stage


def create_animated_prim(
    stage: Usd.Stage,
    prim_type: str,
    prim_path: str,
    reference: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    time_samples: Optional[Dict[float, Dict[str, Any]]] = None,
) -> Usd.Prim:
    """Create a new prim with optional attribute values and time samples.

    Args:
        stage (Usd.Stage): The USD stage to create the prim on.
        prim_type (str): The type of the prim to create (e.g., "Sphere").
        prim_path (str): The path where the prim will be created.
        reference (Optional[str]): The path to the referenced USD file, if any.
        attributes (Optional[Dict[str, Any]]): Attribute names and their default values.
        time_samples (Optional[Dict[float, Dict[str, Any]]]): Time sample values per frame.

    Returns:
        Usd.Prim: The newly created prim.
    """
    if not stage:
        raise ValueError("Stage is invalid.")
    if not prim_type:
        raise ValueError("Prim type is required.")
    if not prim_path:
        raise ValueError("Prim path is required.")
    existing_prim = stage.GetPrimAtPath(prim_path)
    if existing_prim.IsValid():
        return existing_prim
    prim = stage.DefinePrim(prim_path, prim_type)
    if reference:
        prim.GetReferences().AddReference(reference)
    if attributes:
        for attr_name, attr_value in attributes.items():
            attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Find(type(attr_value).__name__))
            attr.Set(attr_value)
    if time_samples:
        for frame, attribute_values in time_samples.items():
            for attr_name, attr_value in attribute_values.items():
                attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Find(type(attr_value).__name__))
                attr.Set(attr_value, Usd.TimeCode(frame))
    return prim


def duplicate_subtree(stage: Usd.Stage, src_path: Sdf.Path, dest_path: Sdf.Path) -> Usd.Prim:
    """Duplicate a subtree from one path to another within the same stage.

    Args:
        stage (Usd.Stage): The USD stage.
        src_path (Sdf.Path): The source path of the subtree to duplicate.
        dest_path (Sdf.Path): The destination path for the duplicated subtree.

    Returns:
        Usd.Prim: The prim at the root of the newly duplicated subtree.

    Raises:
        ValueError: If the source prim does not exist or if the destination
            path is not a valid prim path.
    """
    src_prim = stage.GetPrimAtPath(src_path)
    if not src_prim.IsValid():
        raise ValueError(f"Source prim at path {src_path} does not exist.")
    if not Sdf.Path.IsValidPathString(str(dest_path)):
        raise ValueError(f"Destination path {dest_path} is not a valid prim path.")
    dest_prim = stage.DefinePrim(dest_path)
    for child in src_prim.GetChildren():
        child_name = child.GetName()
        dest_child_path = dest_prim.GetPath().AppendChild(child_name)
        duplicate_subtree(stage, child.GetPath(), dest_child_path)
    attrs = src_prim.GetAttributes()
    rels = src_prim.GetRelationships()
    for attr in attrs:
        dest_prim.CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
    for rel in rels:
        dest_rel = dest_prim.CreateRelationship(rel.GetName())
        targets = rel.GetTargets()
        for target in targets:
            dest_rel.AddTarget(target)
    return dest_prim


def batch_create_materials(
    stage: Usd.Stage, material_paths: list[Sdf.Path], material_type: str = "UsdPreviewSurface"
) -> list[UsdShade.Material]:
    """Create multiple materials on a stage at given paths.

    Args:
        stage (Usd.Stage): The USD stage to create materials on.
        material_paths (list[Sdf.Path]): A list of paths where materials should be created.
        material_type (str, optional): The type of material to create. Defaults to "UsdPreviewSurface".

    Returns:
        list[UsdShade.Material]: A list of created UsdShade.Material prims.
    """
    materials: list[UsdShade.Material] = []
    for material_path in material_paths:
        if not material_path.IsAbsolutePath():
            raise ValueError(f"Material path must be an absolute path: {material_path}")
        existing_prim = stage.GetPrimAtPath(material_path)
        if existing_prim.IsValid():
            raise ValueError(f"A prim already exists at path: {material_path}")
        material_prim = UsdShade.Material.Define(stage, material_path)
        shader_prim = UsdShade.Shader.Define(stage, material_path.AppendChild("Shader"))
        shader_prim.CreateIdAttr().Set(material_type)
        material_prim.CreateSurfaceOutput().ConnectToSource(shader_prim.ConnectableAPI(), "surface")
        materials.append(material_prim)
    return materials


def batch_strip_variant_selections(paths: List[Sdf.Path]) -> List[Sdf.Path]:
    """
    Strip variant selections from a list of paths.

    Args:
        paths (List[Sdf.Path]): The list of paths to strip variant selections from.

    Returns:
        List[Sdf.Path]: A new list of paths with all variant selections stripped.
    """
    stripped_paths: List[Sdf.Path] = []
    for path in paths:
        if path.isEmpty:
            continue
        stripped_path = path.StripAllVariantSelections()
        stripped_paths.append(stripped_path)
    return stripped_paths


def reparent_prim(stage: Usd.Stage, prim_path: Sdf.Path, new_parent_path: Sdf.Path) -> bool:
    """Reparent a prim to a new parent prim.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_path (Sdf.Path): The path of the prim to reparent.
        new_parent_path (Sdf.Path): The path of the new parent prim.

    Returns:
        bool: True if the reparenting succeeded, False otherwise.
    """
    prim = stage.GetPrimAtPath(prim_path)
    new_parent = stage.GetPrimAtPath(new_parent_path)
    if not prim.IsValid() or not new_parent.IsValid():
        return False
    if new_parent_path.HasPrefix(prim_path):
        return False
    old_parent_path = prim_path.GetParentPath()
    old_parent = stage.GetPrimAtPath(old_parent_path)
    if not old_parent.IsValid():
        return False
    stage.RemovePrim(prim_path)
    new_prim_name = prim_path.name
    new_prim_path = new_parent_path.AppendChild(new_prim_name)
    stage.DefinePrim(new_prim_path)
    return True


def merge_prim_properties(src_prim: Usd.Prim, dst_prim: Usd.Prim, skip_properties: List[str] = None) -> None:
    """Merge properties from src_prim to dst_prim.

    Args:
        src_prim (Usd.Prim): The source prim to merge properties from.
        dst_prim (Usd.Prim): The destination prim to merge properties to.
        skip_properties (List[str], optional): A list of property names to skip merging. Defaults to None.
    """
    if not src_prim.IsValid():
        raise ValueError("Source prim is not valid.")
    if not dst_prim.IsValid():
        raise ValueError("Destination prim is not valid.")
    if skip_properties is None:
        skip_properties = []
    for prop in src_prim.GetProperties():
        prop_name = prop.GetName()
        if prop_name in skip_properties:
            continue
        prop_value = prop.Get()
        dst_attr = dst_prim.GetAttribute(prop_name)
        if not dst_attr:
            dst_attr = dst_prim.CreateAttribute(prop_name, prop.GetTypeName())
        dst_attr.Set(prop_value)


def batch_set_prim_activation(stage: Usd.Stage, prim_paths: List[str], active: bool) -> None:
    """Set the active state for multiple prims in a single batch operation.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths to set activation state for.
        active (bool): The activation state to set for the prims.

    Raises:
        ValueError: If any of the prim paths are invalid.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")
        prim.SetActive(active)


def reorder_prims(stage: Usd.Stage, parent_path: str, prim_names: List[str]) -> None:
    """Reorder the child prims under the specified parent prim.

    Args:
        stage (Usd.Stage): The stage containing the prims to reorder.
        parent_path (str): The path of the parent prim.
        prim_names (List[str]): The desired order of the child prim names.
    """
    parent_prim = stage.GetPrimAtPath(parent_path)
    if not parent_prim.IsValid():
        raise ValueError(f"Parent prim at path {parent_path} does not exist.")
    current_prim_names = [prim.GetName() for prim in parent_prim.GetChildren()]
    if set(current_prim_names) != set(prim_names):
        raise ValueError("The specified prim names do not match the current child prims.")
    prim_indices = {name: index for (index, name) in enumerate(prim_names)}
    sorted_prim_names = sorted(current_prim_names, key=lambda name: prim_indices[name])
    parent_prim.SetMetadata("primOrder", sorted_prim_names)


def batch_set_prim_colors(stage: Usd.Stage, color_map: dict) -> None:
    """Set the display color of multiple prims in a single traversal.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        color_map (dict): A dictionary mapping prim paths to color tuples.
    """
    for prim in stage.TraverseAll():
        prim_path = prim.GetPath()
        if str(prim_path) in color_map:
            color = color_map[str(prim_path)]
            binding_api = UsdShade.MaterialBindingAPI(prim)
            material = binding_api.ComputeBoundMaterial()[0]
            if material:
                material.CreateInput("displayColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(color))
            else:
                print(f"Warning: No material bound to prim at path {prim_path}. Skipping.")


def get_hierarchy_paths(prim_path: Sdf.Path) -> List[Sdf.Path]:
    """Get a list of paths for all ancestors of the given prim path.

    Args:
        prim_path (Sdf.Path): The path to the prim to get ancestors for.

    Returns:
        List[Sdf.Path]: A list of paths for all ancestors of the given prim path.
    """
    if not prim_path.IsAbsolutePath():
        raise ValueError(f"Input path {prim_path} is not an absolute path.")
    ancestors_range = prim_path.GetAncestorsRange()
    ancestor_paths = []
    for ancestor_path in ancestors_range:
        ancestor_paths.append(ancestor_path)
    return ancestor_paths


def filter_path_array_by_prim_type(stage: Usd.Stage, path_array: Sdf.PathArray, type_name: str) -> Sdf.PathArray:
    """Filter a PathArray to only include paths to prims of a specific type.

    Args:
        stage (Usd.Stage): The USD stage to query.
        path_array (Sdf.PathArray): The array of paths to filter.
        type_name (str): The prim type to filter by.

    Returns:
        Sdf.PathArray: A new PathArray containing only paths to prims of the specified type.
    """
    filtered_paths = Sdf.PathArray()
    for path in path_array:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and prim.IsA(type_name):
            filtered_paths.append(path)
    return filtered_paths


def get_material_assignments_from_path_array(path_array: Sdf.PathArray) -> Dict[str, Sdf.PathArray]:
    """
    Given a Sdf.PathArray, returns a dictionary mapping each material prim path to the
    paths of the geometry prims it is bound to.
    """
    material_assignments: Dict[str, Sdf.PathArray] = {}
    for path in path_array:
        if not path.IsPropertyPath():
            continue
        prim_path = path.GetPrimPath()
        prop_name = path.name
        if prop_name == "material:binding":
            target_paths = path_array.GetTargetPathsForProperty(path)
            for target_path in target_paths:
                if target_path not in material_assignments:
                    material_assignments[target_path] = Sdf.PathArray()
                material_assignments[target_path].append(prim_path)
    return material_assignments


def merge_path_arrays(path_arrays: List[Sdf.PathArray]) -> Sdf.PathArray:
    """Merge multiple PathArrays into a single PathArray, removing duplicates."""
    unique_paths: Set[Sdf.Path] = set()
    for path_array in path_arrays:
        for path in path_array:
            unique_paths.add(path)
    merged_array: Sdf.PathArray = Sdf.PathArray(sorted(unique_paths))
    return merged_array


def create_prims_from_path_array(stage: Usd.Stage, path_array: Sdf.PathArray) -> None:
    """Create prims on the given stage from an array of paths.

    Args:
        stage (Usd.Stage): The USD stage to create the prims on.
        path_array (Sdf.PathArray): An array of prim paths to create.
    """
    for path in path_array:
        if not path.IsPrimPath():
            raise ValueError(f"Path {path} is not a valid prim path.")
        if stage.GetPrimAtPath(path):
            continue
        parent_path = path.GetParentPath()
        if parent_path != Sdf.Path.absoluteRootPath:
            create_prims_from_path_array(stage, Sdf.PathArray([parent_path]))
        stage.DefinePrim(path)


def get_common_ancestor_path(path_array: Sdf.PathArray) -> Sdf.Path:
    """
    Returns the longest common prefix path of all paths in the array.

    Returns an empty path if the array is empty.
    """
    if len(path_array) == 0:
        return Sdf.Path()
    common_prefix = path_array[0]
    for path in path_array[1:]:
        while not path.HasPrefix(common_prefix):
            common_prefix = common_prefix.GetParentPath()
            if common_prefix.isEmpty:
                return Sdf.Path()
    return common_prefix


def check_item_in_path_list(path_list_op: Sdf.PathListOp, item: Sdf.Path) -> bool:
    """Check if an item exists in a PathListOp.

    Args:
        path_list_op (Sdf.PathListOp): The PathListOp to check.
        item (Sdf.Path): The item to check for existence.

    Returns:
        bool: True if the item exists in the PathListOp, False otherwise.
    """
    if path_list_op.isExplicit:
        return item in path_list_op.explicitItems
    else:
        return item in path_list_op.addedItems or (
            item in path_list_op.orderedItems and item not in path_list_op.deletedItems
        )


def extract_path_list_info(
    path_list_op: Sdf.PathListOp,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], bool]:
    """Extract information from a Sdf.PathListOp.

    Args:
        path_list_op (Sdf.PathListOp): The PathListOp to extract information from.

    Returns:
        Tuple[List[str], List[str], List[str], List[str], List[str], bool]:
            A tuple containing the following:
            - explicitItems: List of explicit items.
            - addedItems: List of added items.
            - prependedItems: List of prepended items.
            - appendedItems: List of appended items.
            - deletedItems: List of deleted items.
            - isExplicit: Whether the PathListOp is explicit.
    """
    explicitItems = [str(path) for path in path_list_op.explicitItems]
    addedItems = [str(path) for path in path_list_op.addedItems]
    prependedItems = [str(path) for path in path_list_op.prependedItems]
    appendedItems = [str(path) for path in path_list_op.appendedItems]
    deletedItems = [str(path) for path in path_list_op.deletedItems]
    isExplicit = path_list_op.isExplicit
    return (explicitItems, addedItems, prependedItems, appendedItems, deletedItems, isExplicit)


def get_added_or_explicit_paths(string_list_op: Sdf.StringListOp) -> List[str]:
    """Get the added or explicit paths from a StringListOp.

    Args:
        string_list_op (Sdf.StringListOp): The StringListOp to get paths from.

    Returns:
        List[str]: The list of added or explicit paths.
    """
    if string_list_op.isExplicit:
        return list(string_list_op.explicitItems)
    else:
        return list(string_list_op.addedItems)


def remove_deleted_paths(path_list_op: Sdf.PathListOp) -> Sdf.PathListOp:
    """Remove deleted paths from a PathListOp.

    Args:
        path_list_op (Sdf.PathListOp): The input PathListOp to remove deleted paths from.

    Returns:
        Sdf.PathListOp: A new PathListOp with deleted paths removed.
    """
    result = Sdf.PathListOp.CreateExplicit()
    for explicit_path in path_list_op.explicitItems:
        if not path_list_op.HasItem(explicit_path):
            result.explicitItems.append(explicit_path)
    return result


def clear_path_operations(path_list_op: Sdf.PathListOp) -> None:
    """Clear the operations on a PathListOp.

    Args:
        path_list_op (Sdf.PathListOp): The PathListOp to clear.
    """
    if isinstance(path_list_op, Sdf.PathListOp):
        path_list_op.ClearAndMakeExplicit()
    else:
        raise TypeError(f"Unsupported input type: {type(path_list_op)}")


def get_ordered_paths(path_list_op: Sdf.PathListOp) -> List[Sdf.Path]:
    """Get the ordered list of paths from a PathListOp.

    Args:
        path_list_op (Sdf.PathListOp): The input PathListOp.

    Returns:
        List[Sdf.Path]: The ordered list of paths.
    """
    if path_list_op.isExplicit:
        return list(path_list_op.explicitItems)
    else:
        result_path_list_op = Sdf.PathListOp()
        result_path_list_op.ApplyOperations(path_list_op)
        return list(result_path_list_op.orderedItems)


def toggle_explicit_mode(path_list_op: Sdf.PathListOp) -> None:
    """Toggle the explicit mode of the PathListOp.

    If the PathListOp is currently in explicit mode, it will be converted to
    non-explicit mode. If it is currently in non-explicit mode, it will be
    converted to explicit mode.

    Args:
        path_list_op: The PathListOp to toggle the explicit mode on.

    Raises:
        RuntimeError: If the PathListOp contains any added, prepended, appended,
            or deleted items, which would be lost in the conversion.
    """
    if (
        path_list_op.addedItems
        or path_list_op.prependedItems
        or path_list_op.appendedItems
        or path_list_op.deletedItems
    ):
        raise RuntimeError(
            "Cannot toggle explicit mode on a PathListOp with added, prepended, appended or deleted items."
        )
    if path_list_op.isExplicit:
        path_list_op.Clear()
    else:
        path_list_op.ClearAndMakeExplicit()


def apply_path_operations(path_list_op: Sdf.PathListOp, prim_path: str) -> None:
    """Apply path list operations to a given prim path.

    Args:
        path_list_op (Sdf.PathListOp): The path list operation to apply.
        prim_path (str): The prim path to apply the operations to.

    Raises:
        ValueError: If the prim path is invalid.
    """
    if not Sdf.Path.IsValidPathString(prim_path):
        raise ValueError(f"Invalid prim path: {prim_path}")
    path = Sdf.Path(prim_path)
    path_list_op.addedItems.append(path)
    result_path_list_op = Sdf.PathListOp()
    path_list_op.ApplyOperations(result_path_list_op)
    if result_path_list_op.HasItem(path):
        print(f"Prim path '{prim_path}' was added or already present in the explicit items.")
    else:
        print(f"Prim path '{prim_path}' was not added to the explicit items.")


def merge_path_lists(list1: Sdf.PathListOp, list2: Sdf.PathListOp) -> Sdf.PathListOp:
    """Merges two PathListOp objects into a single PathListOp.

    Args:
        list1 (Sdf.PathListOp): The first PathListOp object to merge.
        list2 (Sdf.PathListOp): The second PathListOp object to merge.

    Returns:
        Sdf.PathListOp: A new PathListOp object containing the merged paths.
    """
    merged_list = Sdf.PathListOp()
    for path in list1.explicitItems:
        merged_list.explicitItems.append(path)
    for path in list1.addedItems:
        merged_list.addedItems.append(path)
    for path in list2.explicitItems:
        if path not in merged_list.explicitItems:
            merged_list.explicitItems.append(path)
    for path in list2.addedItems:
        if path not in merged_list.addedItems:
            merged_list.addedItems.append(path)
    for path in list1.deletedItems:
        if path in merged_list.explicitItems:
            merged_list.explicitItems.remove(path)
        if path in merged_list.addedItems:
            merged_list.addedItems.remove(path)
    for path in list2.deletedItems:
        if path in merged_list.explicitItems:
            merged_list.explicitItems.remove(path)
        if path in merged_list.addedItems:
            merged_list.addedItems.remove(path)
    return merged_list


def consolidate_path_operations(path_list_op: Sdf.PathListOp) -> Sdf.PathListOp:
    """Consolidate the path list operations into a single explicit list.

    This function takes a PathListOp object and consolidates all the individual
    add, remove, and reorder operations into a single explicit list of paths.

    Args:
        path_list_op (Sdf.PathListOp): The input PathListOp to consolidate.

    Returns:
        Sdf.PathListOp: A new PathListOp with the consolidated explicit path list.
    """
    consolidated_op = Sdf.PathListOp()
    consolidated_op.ApplyOperations(path_list_op)
    return consolidated_op


def modify_path_list(
    path_list_op: Sdf.PathListOp, paths_to_add: List[Sdf.Path], paths_to_remove: List[Sdf.Path]
) -> Sdf.PathListOp:
    """Modifies a PathListOp by adding and removing paths.

    Args:
        path_list_op (Sdf.PathListOp): The PathListOp to modify.
        paths_to_add (List[Sdf.Path]): A list of paths to add to the PathListOp.
        paths_to_remove (List[Sdf.Path]): A list of paths to remove from the PathListOp.

    Returns:
        Sdf.PathListOp: The modified PathListOp.
    """
    modified_path_list_op = Sdf.PathListOp()
    modified_path_list_op.explicitItems = path_list_op.explicitItems.copy()
    for path in paths_to_add:
        if not path.IsAbsolutePath():
            raise ValueError(f"Path {path} is not an absolute path.")
        modified_path_list_op.appendedItems.append(path)
    for path in paths_to_remove:
        if not path.IsAbsolutePath():
            raise ValueError(f"Path {path} is not an absolute path.")
        modified_path_list_op.deletedItems.append(path)
    result = Sdf.PathListOp.ApplyOperations(modified_path_list_op, path_list_op)
    return result


def get_payload_info(payload: Sdf.Payload) -> Tuple[str, str, Sdf.LayerOffset]:
    """Get the asset path, prim path, and layer offset from a payload."""
    if not isinstance(payload, Sdf.Payload):
        raise TypeError("Argument 'payload' must be an instance of Sdf.Payload")
    asset_path = payload.assetPath
    prim_path = payload.primPath.pathString
    layer_offset = payload.layerOffset
    return (asset_path, prim_path, layer_offset)


def copy_payload_list(source: Sdf.PayloadListOp, dest: Optional[Sdf.PayloadListOp] = None) -> Sdf.PayloadListOp:
    """Copy the payload list from source to dest.

    Args:
        source (Sdf.PayloadListOp): The source payload list to copy from.
        dest (Optional[Sdf.PayloadListOp]): The destination payload list to copy to. If None, a new PayloadListOp is created.

    Returns:
        Sdf.PayloadListOp: The destination payload list with the copied items.
    """
    if dest is None:
        dest = Sdf.PayloadListOp()
    dest.Clear()
    for item in source.explicitItems:
        dest.addedItems.append(item)
    for item in source.addedItems:
        dest.addedItems.append(item)
    for item in source.prependedItems:
        dest.prependedItems.append(item)
    for item in source.appendedItems:
        dest.appendedItems.append(item)
    for item in source.deletedItems:
        dest.deletedItems.append(item)
    return dest


def replace_payload(
    payload_list_op: Sdf.PayloadListOp, old_payload: Sdf.Payload, new_payload: Sdf.Payload
) -> Sdf.PayloadListOp:
    """Replace an existing payload in the PayloadListOp with a new payload.

    Args:
        payload_list_op (Sdf.PayloadListOp): The PayloadListOp to modify.
        old_payload (Sdf.Payload): The payload to replace.
        new_payload (Sdf.Payload): The new payload to use as replacement.

    Returns:
        Sdf.PayloadListOp: The modified PayloadListOp with the replaced payload.
    """
    if not payload_list_op.HasItem(old_payload):
        raise ValueError("The old payload does not exist in the PayloadListOp.")
    new_payload_list_op = Sdf.PayloadListOp()
    for payload in payload_list_op.GetAddedOrExplicitItems():
        if payload == old_payload:
            new_payload_list_op.appendedItems.append(new_payload)
        else:
            new_payload_list_op.appendedItems.append(payload)
    return new_payload_list_op


def prepend_payloads(payload_list_op: Sdf.PayloadListOp, payloads: List[Sdf.Payload]) -> None:
    """Prepend payloads to the front of the list of explicit payloads.

    Args:
        payload_list_op (Sdf.PayloadListOp): The PayloadListOp to modify.
        payloads (List[Sdf.Payload]): The list of payloads to prepend.
    """
    if not payload_list_op.isExplicit:
        payload_list_op.ClearAndMakeExplicit()
    explicit_items = payload_list_op.explicitItems
    explicit_items = payloads + explicit_items
    payload_list_op.explicitItems = explicit_items


def list_all_payloads(payload_list_op: Sdf.PayloadListOp) -> List[Sdf.Payload]:
    """Return a list of all payloads in the PayloadListOp.

    This includes added, prepended, appended, and explicit payloads.
    Deleted payloads are excluded.

    Args:
        payload_list_op (Sdf.PayloadListOp): The PayloadListOp to get payloads from.

    Returns:
        List[Sdf.Payload]: A list of all payloads in the PayloadListOp.
    """
    explicit_payloads = payload_list_op.GetAddedOrExplicitItems()
    all_payloads = []
    all_payloads.extend(payload_list_op.prependedItems or [])
    all_payloads.extend(explicit_payloads)
    all_payloads.extend(payload_list_op.appendedItems or [])
    all_payloads = [Sdf.Payload(p) if isinstance(p, str) else p for p in all_payloads]
    return all_payloads


def merge_payloads(payload_list_op: Sdf.PayloadListOp, payloads_to_add: List[Sdf.Payload]) -> None:
    """Merges a list of payloads into an existing PayloadListOp.

    Args:
        payload_list_op (Sdf.PayloadListOp): The PayloadListOp to merge payloads into.
        payloads_to_add (List[Sdf.Payload]): The list of payloads to add.
    """
    if not payload_list_op:
        raise ValueError("Invalid PayloadListOp. Cannot merge payloads.")
    payloads_to_add_op = Sdf.PayloadListOp()
    for payload in payloads_to_add:
        if not payload_list_op.HasItem(payload):
            payloads_to_add_op.appendedItems.append(payload)
        else:
            continue
    payload_list_op.ApplyOperations(payloads_to_add_op)


def analyze_payloads(
    payload_list_op: Sdf.PayloadListOp,
) -> Tuple[List[Sdf.Payload], List[Sdf.Payload], List[Sdf.Payload], List[Sdf.Payload], bool]:
    """Analyze a PayloadListOp and return its various properties.

    Args:
        payload_list_op (Sdf.PayloadListOp): The PayloadListOp to analyze.

    Returns:
        A tuple containing:
            - added_items (List[Sdf.Payload]): The added payload items.
            - prepended_items (List[Sdf.Payload]): The prepended payload items.
            - appended_items (List[Sdf.Payload]): The appended payload items.
            - deleted_items (List[Sdf.Payload]): The deleted payload items.
            - is_explicit (bool): Whether the PayloadListOp is explicit.
    """
    added_items = payload_list_op.GetAddedOrExplicitItems()
    prepended_items = payload_list_op.prependedItems or []
    appended_items = payload_list_op.appendedItems or []
    deleted_items = payload_list_op.deletedItems or []
    is_explicit = payload_list_op.isExplicit or False
    return (added_items, prepended_items, appended_items, deleted_items, is_explicit)


def synchronize_payloads(payload_list_op: Sdf.PayloadListOp, new_payloads: List[Sdf.Payload]) -> Sdf.PayloadListOp:
    """Synchronize the payload list op with the new payloads.

    This will clear any existing payloads and set the new payloads as the explicit items.

    Args:
        payload_list_op (Sdf.PayloadListOp): The payload list op to synchronize.
        new_payloads (List[Sdf.Payload]): The new payloads to set as explicit items.

    Returns:
        Sdf.PayloadListOp: The synchronized payload list op.
    """
    synchronized_op = Sdf.PayloadListOp()
    synchronized_op.explicitItems = new_payloads
    payload_list_op.ApplyOperations(synchronized_op)
    return payload_list_op


def make_payloads_explicit(payload_list_op: Sdf.PayloadListOp) -> None:
    """Make all payloads in the PayloadListOp explicit."""
    if payload_list_op.isExplicit:
        return
    added_or_explicit_items = payload_list_op.GetAddedOrExplicitItems()
    payload_list_op.ClearAndMakeExplicit()
    for item in added_or_explicit_items:
        payload_list_op.explicitItems.append(item)


def check_payload_existence(payload_list_op: Sdf.PayloadListOp, payload: Sdf.Payload) -> bool:
    """Check if a payload exists in the PayloadListOp.

    Args:
        payload_list_op (Sdf.PayloadListOp): The PayloadListOp to check.
        payload (Sdf.Payload): The payload to check for existence.

    Returns:
        bool: True if the payload exists in the PayloadListOp, False otherwise.
    """
    if payload_list_op.isExplicit:
        return payload in payload_list_op.explicitItems
    else:
        return payload in payload_list_op.addedItems


def set_permissions_recursively(prim: Sdf.PrimSpec, permission: Optional[str]) -> None:
    """Set the permission for a prim and all its children recursively.

    Args:
        prim (Sdf.PrimSpec): The prim to set the permission for.
        permission (Optional[str]): The permission to set. Valid values are "public", "private", or None.
    """
    if permission is None:
        prim.permission = Sdf.PermissionPublic
    elif permission == "public":
        prim.permission = Sdf.PermissionPublic
    elif permission == "private":
        prim.permission = Sdf.PermissionPrivate
    else:
        raise ValueError(f"Invalid permission value: {permission}")
    for child_prim in prim.nameChildren:
        set_permissions_recursively(child_prim, permission)


def create_hierarchy(stage: Usd.Stage, prim_paths: list[str], prim_type: str = "Xform") -> list[Usd.Prim]:
    """Create a hierarchy of prims given a list of prim paths.

    Args:
        stage (Usd.Stage): The stage to create the prims on.
        prim_paths (list[str]): A list of prim paths to create.
        prim_type (str, optional): The type of prims to create. Defaults to "Xform".

    Returns:
        list[Usd.Prim]: A list of the created prims.
    """
    if not stage:
        raise ValueError("Invalid stage")
    if not prim_paths:
        raise ValueError("No prim paths provided")
    prims = []
    for path in prim_paths:
        if not path.startswith("/"):
            path = "/" + path
        existing_prim = stage.GetPrimAtPath(path)
        if existing_prim:
            prims.append(existing_prim)
            continue
        prim = stage.DefinePrim(path, prim_type)
        if not prim:
            raise RuntimeError(f"Failed to create prim at path {path}")
        prims.append(prim)
    return prims


def get_all_materials(stage: Usd.Stage) -> list[UsdShade.Material]:
    """Get all materials in the stage.

    Args:
        stage (Usd.Stage): The stage to search for materials.

    Returns:
        list[UsdShade.Material]: A list of all materials in the stage.
    """
    materials: list[UsdShade.Material] = []
    for prim in stage.TraverseAll():
        if prim.IsA(UsdShade.Material):
            material = UsdShade.Material(prim)
            materials.append(material)
    return materials


def replicate_hierarchy_with_modifications(
    source_prim: Sdf.PrimSpec, dest_prim: Sdf.PrimSpec, reference: bool = False, instanceable: bool = False
) -> None:
    """
    Recursively copy prim hierarchy from source to destination with optional modifications.

    Args:
        source_prim (Sdf.PrimSpec): The source prim to copy from.
        dest_prim (Sdf.PrimSpec): The destination prim to copy to.
        reference (bool): If True, add a reference to the source prim instead of copying.
        instanceable (bool): If True, make the copied prims instanceable.
    """
    if not source_prim:
        raise ValueError("Invalid source prim")
    if not dest_prim:
        raise ValueError("Invalid destination prim")
    if reference:
        dest_prim.referenceList.Add(source_prim.path)
        return
    dest_prim.specifier = source_prim.specifier
    dest_prim.typeName = source_prim.typeName
    if instanceable:
        dest_prim.instanceable = True
    for child_name in source_prim.nameChildren.keys():
        source_child = source_prim.GetPrimAtPath(child_name)
        dest_child = Sdf.PrimSpec(dest_prim, child_name, source_child.specifier, source_child.typeName)
        replicate_hierarchy_with_modifications(source_child, dest_child, reference, instanceable)


def animate_prims(
    stage: Usd.Stage, prim_paths: List[str], attribute_name: str, time_samples: List[float], attribute_values: List[Any]
) -> None:
    """Animate the specified attribute on a list of prims over time.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of paths to the prims to animate.
        attribute_name (str): Name of the attribute to animate on each prim.
        time_samples (List[float]): List of time samples for the animation.
        attribute_values (List[Any]): List of attribute values corresponding to each time sample.

    Raises:
        ValueError: If the lengths of time_samples and attribute_values do not match,
                    or if a specified prim or attribute does not exist.
    """
    if len(time_samples) != len(attribute_values):
        raise ValueError("Length of time_samples and attribute_values must match.")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        attribute = prim.GetAttribute(attribute_name)
        if not attribute.IsValid():
            raise ValueError(f"Attribute '{attribute_name}' does not exist on prim at path {prim_path}.")
        for time_sample, attribute_value in zip(time_samples, attribute_values):
            attribute.Set(attribute_value, time_sample)


def create_animatable_parameters(prim: Usd.Prim, attribute_names: List[str]) -> None:
    """Create animatable float attributes on a prim.

    Args:
        prim (Usd.Prim): The prim to create attributes on.
        attribute_names (List[str]): A list of attribute names to create.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    for attr_name in attribute_names:
        if prim.HasAttribute(attr_name):
            print(f"Warning: Attribute '{attr_name}' already exists on prim {prim.GetPath()}. Skipping creation.")
            continue
        attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
        attr.SetCustom(True)
        attr.SetVariability(Sdf.VariabilityVarying)


def configure_shadow_settings(
    prim_spec: Sdf.PrimSpec, enable_shadows: bool = True, shadow_color: Gf.Vec4f = Gf.Vec4f(0, 0, 0, 1)
) -> None:
    """Configure shadow settings for a prim.

    Args:
        prim_spec (Sdf.PrimSpec): The prim spec to configure shadow settings for.
        enable_shadows (bool, optional): Whether to enable shadows. Defaults to True.
        shadow_color (Gf.Vec4f, optional): The color of the shadows. Defaults to Gf.Vec4f(0, 0, 0, 1).
    """
    if not prim_spec:
        raise ValueError("Invalid prim spec.")
    enable_attr = prim_spec.attributes.get("enableShadows")
    if not enable_attr:
        enable_attr = Sdf.AttributeSpec(prim_spec, "enableShadows", Sdf.ValueTypeNames.Bool)
    enable_attr.default = enable_shadows
    color_attr = prim_spec.attributes.get("shadowColor")
    if not color_attr:
        color_attr = Sdf.AttributeSpec(prim_spec, "shadowColor", Sdf.ValueTypeNames.Color4f)
    color_attr.default = shadow_color


def setup_environment_lighting(stage: Usd.Stage, dome_light_path: str = "/World/DomeLight") -> UsdLux.DomeLight:
    """Create a dome light for environment lighting in the given USD stage.

    Args:
        stage (Usd.Stage): The USD stage to add the dome light to.
        dome_light_path (str, optional): The path for the dome light prim. Defaults to "/World/DomeLight".

    Returns:
        UsdLux.DomeLight: The created dome light prim.
    """
    if stage.GetPrimAtPath(dome_light_path):
        return UsdLux.DomeLight(stage.GetPrimAtPath(dome_light_path))
    dome_light = UsdLux.DomeLight.Define(stage, dome_light_path)
    dome_light.CreateTextureFileAttr("domeLight.exr")
    dome_light.CreateIntensityAttr(1000.0)
    dome_light.CreateExposureAttr(0.01)
    return dome_light


def set_active_status_recursive(prim_spec: Sdf.PrimSpec, active: bool):
    """Recursively set the active status for a prim and all its children."""
    prim_spec.active = active
    for child_name in prim_spec.nameChildren.keys():
        child_prim = prim_spec.GetPrimAtPath(prim_spec.path.AppendChild(child_name))
        if child_prim:
            set_active_status_recursive(child_prim, active)


def consolidate_animations(stage: Usd.Stage, prim_paths: List[Sdf.Path]) -> None:
    """
    Consolidate animations from multiple prims into a single prim.

    This function takes a list of prim paths, finds all animated attributes
    on those prims, and copies the animation data to a new prim at the path
    "/ConsolidatedAnimation". The source animation data is then deleted.

    Args:
        stage (Usd.Stage): The USD stage to operate on.
        prim_paths (List[Sdf.Path]): A list of paths to prims to consolidate animations from.

    Raises:
        ValueError: If any of the provided prim paths are invalid.
    """
    dest_prim = stage.GetPrimAtPath("/ConsolidatedAnimation")
    if not dest_prim:
        dest_prim = stage.DefinePrim("/ConsolidatedAnimation", "Xform")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            raise ValueError(f"Invalid prim path: {prim_path}")
        for prop in prim.GetAuthoredProperties():
            if prop.HasValue():
                continue
            dest_prop = dest_prim.GetProperty(prop.GetName())
            if not dest_prop:
                dest_prop = dest_prim.CreateProperty(prop.GetName(), prop.GetTypeName())
            dest_prop.Set(prop.Get(Usd.TimeCode.Default()))
            prop.ClearDefault()


def create_light_probe(stage: Usd.Stage, prim_path: str, probe_type: str = "spherical") -> UsdLux.DomeLight:
    """Create a light probe at the specified prim path.

    Args:
        stage (Usd.Stage): The USD stage to create the light probe on.
        prim_path (str): The path where the light probe should be created.
        probe_type (str, optional): The type of light probe to create. Can be "spherical" or "planar". Defaults to "spherical".

    Returns:
        UsdLux.DomeLight: The created light probe prim.

    Raises:
        ValueError: If the specified probe_type is not supported.
    """
    if probe_type not in ["spherical", "planar"]:
        raise ValueError(f"Unsupported probe type: {probe_type}. Supported types are 'spherical' and 'planar'.")
    light_probe_prim = UsdLux.DomeLight.Define(stage, Sdf.Path(prim_path))
    if probe_type == "spherical":
        light_probe_prim.CreateTextureFormatAttr().Set("latlong")
    else:
        light_probe_prim.CreateTextureFormatAttr().Set("mirrorball")
    return light_probe_prim


def create_dynamic_simulation(stage: Usd.Stage, prim_path: str) -> UsdPhysics.Scene:
    """Create a dynamic simulation for a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
    if not rigid_body_api:
        raise ValueError(f"Prim at path {prim_path} is not a valid UsdPhysics.RigidBodyAPI.")
    scene_path = Sdf.Path(prim_path).GetParentPath().AppendChild("PhysicsScene")
    physics_scene = UsdPhysics.Scene.Define(stage, scene_path)
    rigid_body_api.CreateSimulationOwnerRel().SetTargets([physics_scene.GetPath()])
    return physics_scene


def reorder_children_by_type(prim_spec: Sdf.PrimSpec, type_order: List[str]) -> None:
    """
    Reorders the children of the given prim spec based on the provided type order.

    Args:
        prim_spec (Sdf.PrimSpec): The prim spec whose children should be reordered.
        type_order (List[str]): The desired order of types. Children will be ordered based on their type's position in this list.

    Raises:
        ValueError: If the prim spec is invalid.
    """
    if not prim_spec:
        raise ValueError("Invalid prim spec.")
    children_names = prim_spec.nameChildren.keys()
    children_by_type = {}
    for child_name in children_names:
        child_spec = prim_spec.nameChildren[child_name]
        type_name = child_spec.typeName
        if type_name not in children_by_type:
            children_by_type[type_name] = []
        children_by_type[type_name].append(child_name)
    reordered_children = []
    for type_name in type_order:
        if type_name in children_by_type:
            reordered_children.extend(children_by_type[type_name])
            del children_by_type[type_name]
    for remaining_children in children_by_type.values():
        reordered_children.extend(remaining_children)
    prim_spec.nameChildrenOrder = reordered_children


def assign_material_to_hierarchy(stage: Usd.Stage, material_path: str, prim_path: str) -> None:
    """Assign a material to a prim and its descendants.

    Args:
        stage (Usd.Stage): The USD stage.
        material_path (str): The path to the material prim.
        prim_path (str): The path to the prim to assign the material to.

    Raises:
        ValueError: If the material prim or the target prim do not exist.
    """
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise ValueError(f"Material prim at path {material_path} does not exist.")
    target_prim = stage.GetPrimAtPath(prim_path)
    if not target_prim.IsValid():
        raise ValueError(f"Target prim at path {prim_path} does not exist.")
    for prim in Usd.PrimRange(target_prim):
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            UsdShade.MaterialBindingAPI(mesh).Bind(material_prim)


def replicate_prims_with_offset(
    stage: Usd.Stage, prim_paths: List[str], offset: Tuple[float, float, float], copies: int
) -> List[Usd.Prim]:
    """Replicate prims with an offset.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The paths of the prims to replicate.
        offset (Tuple[float, float, float]): The offset to apply to each copy.
        copies (int): The number of copies to create.

    Returns:
        List[Usd.Prim]: The created prim copies.
    """
    created_prims = []
    if copies < 1:
        raise ValueError("Number of copies must be greater than 0.")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        parent_path = prim.GetPath().GetParentPath()
        prim_name = prim.GetName()
        for i in range(copies):
            new_prim_path = parent_path.AppendPath(f"{prim_name}_copy_{i + 1}")
            new_prim = stage.DefinePrim(new_prim_path, prim.GetTypeName())
            add_translate_op(UsdGeom.Xform(new_prim)).Set(Gf.Vec3d(offset) * (i + 1))
            new_prim.GetReferences().AddInternalReference(prim.GetPath())
            created_prims.append(new_prim)
    return created_prims


def get_prim_hierarchy_statistics(prim: Usd.Prim) -> Dict[str, int]:
    """Get statistics about the hierarchy under a given prim.

    Args:
        prim (Usd.Prim): The prim to analyze.

    Returns:
        Dict[str, int]: A dictionary with keys "totalPrims", "activePrims",
            "inactivePrims", "pureOverPrims", "instancedPrims" and values
            representing the count of each type of prim under the given prim.
    """
    total_prims = 0
    active_prims = 0
    inactive_prims = 0
    pure_over_prims = 0
    instanced_prims = 0
    for descendant_prim in Usd.PrimRange(prim):
        total_prims += 1
        if descendant_prim.IsActive():
            active_prims += 1
        else:
            inactive_prims += 1
        if descendant_prim.IsDefined():
            pure_over_prims += 1
        if descendant_prim.IsInstance():
            instanced_prims += 1
    return {
        "totalPrims": total_prims,
        "activePrims": active_prims,
        "inactivePrims": inactive_prims,
        "pureOverPrims": pure_over_prims,
        "instancedPrims": instanced_prims,
    }


def validate_hierarchy_structure(prim_spec: Sdf.PrimSpec) -> bool:
    """
    Validate the hierarchy structure of a prim spec.

    This function checks if the prim spec has a valid hierarchy structure
    by verifying that it has a parent and that its name is not empty.

    Args:
        prim_spec (Sdf.PrimSpec): The prim spec to validate.

    Returns:
        bool: True if the hierarchy structure is valid, False otherwise.
    """
    if not prim_spec.nameParent:
        return False
    if not prim_spec.name:
        return False
    for child_name in prim_spec.nameChildren:
        child_spec = prim_spec.nameChildren[child_name]
        if not validate_hierarchy_structure(child_spec):
            return False
    return True


def remove_unused_prims(prim_spec: Sdf.PrimSpec, keep_empty_ancestors: bool = False) -> None:
    """Recursively remove all unused prims under the given prim spec.

    An unused prim is a prim that has no properties, attributes, relationships
    or variants set and no children that are used.

    Args:
        prim_spec (Sdf.PrimSpec): The prim spec to start the removal process from.
        keep_empty_ancestors (bool): If True, ancestors of used prims will be kept even if they are empty.
                                     If False, all empty ancestors will be removed. Defaults to False.
    """
    if _is_prim_used(prim_spec):
        for child_name in list(prim_spec.nameChildren.keys()):
            remove_unused_prims(prim_spec.nameChildren[child_name], keep_empty_ancestors)
    else:
        for child_name in list(prim_spec.nameChildren.keys()):
            prim_spec.nameChildren.remove(child_name)
        if not keep_empty_ancestors:
            _remove_prim_spec(prim_spec)


def _is_prim_used(prim_spec: Sdf.PrimSpec) -> bool:
    """Check if a prim spec is used.

    A prim spec is considered used if it has any properties, attributes,
    relationships, variants set, or if any of its children are used.

    Args:
        prim_spec (Sdf.PrimSpec): The prim spec to check.

    Returns:
        bool: True if the prim spec is used, False otherwise.
    """
    if prim_spec.properties:
        return True
    if prim_spec.attributes:
        return True
    if prim_spec.relationships:
        return True
    if prim_spec.variantSets:
        return True
    for child_spec in prim_spec.nameChildren.values():
        if _is_prim_used(child_spec):
            return True
    return False


def _remove_prim_spec(prim_spec: Sdf.PrimSpec) -> None:
    """Remove a prim spec from its parent.

    Args:
        prim_spec (Sdf.PrimSpec): The prim spec to remove.
    """
    parent_spec = prim_spec.nameParent
    if parent_spec:
        del parent_spec.nameChildren[prim_spec.name]


def create_shadow_catchers(stage: Usd.Stage, prim_paths: List[str]) -> List[Usd.Prim]:
    """Create shadow catcher prims at the given paths."""
    shadow_catchers: List[Usd.Prim] = []
    for prim_path in prim_paths:
        prim = stage.DefinePrim(prim_path, "Mesh")
        if not prim.IsValid():
            raise ValueError(f"Failed to create prim at path {prim_path}")
        prim.CreateAttribute("usdLux:shadowCatcher", Sdf.ValueTypeNames.Bool).Set(True)
        shadow_catchers.append(prim)
    return shadow_catchers


def setup_occlusion_culling(prim: Usd.Prim, purpose: str = "default") -> None:
    """Set up occlusion culling for a prim."""
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    valid_purposes = ["default", "render", "proxy"]
    if purpose not in valid_purposes:
        raise ValueError(f"Invalid purpose '{purpose}'. Must be one of {valid_purposes}.")
    attr_name = "occlusion"
    if not prim.HasAttribute(attr_name):
        prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Token, True)
    prim.GetAttribute(attr_name).Set(purpose)


def retarget_animation(src_prim: Usd.Prim, dst_prim: Usd.Prim) -> bool:
    """Retarget animation from one prim to another.

    Args:
        src_prim (Usd.Prim): The source prim with the animation.
        dst_prim (Usd.Prim): The destination prim to apply the animation.

    Returns:
        bool: True if the animation was successfully retargeted, False otherwise.
    """
    if not src_prim.IsValid():
        print(f"Source prim {src_prim.GetPath()} is not a valid prim.")
        return False
    if not dst_prim.IsValid():
        print(f"Destination prim {dst_prim.GetPath()} is not a valid prim.")
        return False
    src_translate_attr = src_prim.GetAttribute("xformOp:translate")
    src_rotate_attr = src_prim.GetAttribute("xformOp:rotateXYZ")
    src_scale_attr = src_prim.GetAttribute("xformOp:scale")
    if not src_translate_attr.IsValid() or not src_rotate_attr.IsValid() or (not src_scale_attr.IsValid()):
        print(f"Source prim {src_prim.GetPath()} does not have the required transform attributes.")
        return False
    dst_translate_attr = dst_prim.GetAttribute("xformOp:translate")
    dst_rotate_attr = dst_prim.GetAttribute("xformOp:rotateXYZ")
    dst_scale_attr = dst_prim.GetAttribute("xformOp:scale")
    if not dst_translate_attr.IsValid():
        dst_translate_attr = dst_prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
    if not dst_rotate_attr.IsValid():
        dst_rotate_attr = dst_prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Double3)
    if not dst_scale_attr.IsValid():
        dst_scale_attr = dst_prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3)
    time_samples = src_translate_attr.GetTimeSamples()
    for time_sample in time_samples:
        translation = src_translate_attr.Get(time_sample)
        rotation = src_rotate_attr.Get(time_sample)
        scale = src_scale_attr.Get(time_sample)
        dst_translate_attr.Set(translation, time_sample)
        dst_rotate_attr.Set(rotation, time_sample)
        dst_scale_attr.Set(scale, time_sample)
    return True


def setup_physics_properties(
    prim: Usd.Prim, mass: float, density: float, physics_api_version: str = "PhysicsMassAPI"
) -> None:
    """Set up physics properties on a prim.

    Args:
        prim (Usd.Prim): The prim to set up physics properties on.
        mass (float): The mass of the prim.
        density (float): The density of the prim.
        physics_api_version (str, optional): The physics API version to use. Defaults to "PhysicsMassAPI".
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if physics_api_version == "PhysicsMassAPI":
        physics_api = UsdPhysics.MassAPI.Apply(prim)
    else:
        raise ValueError(f"Unsupported physics API version: {physics_api_version}")
    if mass > 0:
        physics_api.CreateMassAttr().Set(mass)
    else:
        raise ValueError(f"Invalid mass value: {mass}. Mass must be greater than zero.")
    if density > 0:
        physics_api.CreateDensityAttr().Set(density)
    else:
        raise ValueError(f"Invalid density value: {density}. Density must be greater than zero.")
    UsdPhysics.CollisionAPI.Apply(prim)


def create_deformation_cages(
    stage: Usd.Stage, prim_path: str, cage_count: int = 1, cage_size: float = 1.0
) -> List[Usd.Prim]:
    """Create deformation cages around a prim.

    Args:
        stage (Usd.Stage): The stage to create the cages on.
        prim_path (str): The path to the prim to create the cages around.
        cage_count (int, optional): The number of cages to create. Defaults to 1.
        cage_size (float, optional): The size of the cages. Defaults to 1.0.

    Returns:
        List[Usd.Prim]: A list of the created cage prims.
    """
    if cage_count < 1:
        raise ValueError("cage_count must be greater than or equal to 1")
    if cage_size <= 0:
        raise ValueError("cage_size must be greater than 0")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist")
    bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
    if bbox.GetRange().IsEmpty():
        raise ValueError(f"Prim at path {prim_path} does not have a valid bounding box")
    cages = []
    for i in range(cage_count):
        size = bbox.ComputeAlignedRange().GetSize() * (1.0 + (i + 1) * cage_size)
        cage_path = f"{prim_path}/Cage_{i}"
        cage_prim = UsdGeom.Cube.Define(stage, cage_path).GetPrim()
        add_scale_op(UsdGeom.Xformable(cage_prim)).Set(size)
        cages.append(cage_prim)
    return cages


def create_rigged_characters(stage: Usd.Stage, names: List[str], root_path: str = "/World") -> List[Usd.Prim]:
    """Creates multiple rigged characters under a common root prim.

    Args:
        stage (Usd.Stage): The USD stage to create the characters in.
        names (List[str]): A list of names for the characters.
        root_path (str, optional): The path of the root prim to create characters under.
            Defaults to "/World".

    Returns:
        List[Usd.Prim]: A list of the created character prims.
    """
    if not Sdf.Path(root_path).IsPrimPath():
        raise ValueError(f"Invalid root prim path: {root_path}")
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim:
        root_prim = stage.DefinePrim(root_path, "Xform")
    character_prims = []
    for name in names:
        character_path = root_prim.GetPath().AppendChild(name)
        character_prim = stage.DefinePrim(character_path, "Scope")
        character_prims.append(character_prim)
        skel_path = character_path.AppendChild("Skel")
        skel_prim = UsdSkel.Skeleton.Define(stage, skel_path)
        mesh_path = character_path.AppendChild("Mesh")
        mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)
        UsdSkel.BindingAPI.Apply(mesh_prim.GetPrim())
        binding_api = UsdSkel.BindingAPI(mesh_prim)
        binding_api.CreateSkeletonRel().SetTargets([skel_path])
    return character_prims


def create_rigid_body_simulation(
    stage: Usd.Stage, prim_path: str, gravity: Gf.Vec3f = Gf.Vec3f(0.0, -9.81, 0.0)
) -> UsdPhysics.RigidBodyAPI:
    """Create a rigid body simulation for a prim.

    Args:
        stage (Usd.Stage): The USD stage to create the simulation on.
        prim_path (str): The path of the prim to create the simulation for.
        gravity (Gf.Vec3f, optional): The gravity vector for the simulation. Defaults to Gf.Vec3f(0.0, -9.81, 0.0).

    Raises:
        ValueError: If the prim at the given path does not exist or is not a valid rigid body.

    Returns:
        UsdPhysics.RigidBodyAPI: The rigid body API for the prim.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    if not UsdPhysics.RigidBodyAPI.CanApply(prim):
        raise ValueError(f"Prim at path {prim_path} is not a valid rigid body.")
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body_api.CreateSimulationOwnerRel().SetTargets(["/physicsScene"])
    return rigid_body_api


def find_prims_by_pattern(stage: Usd.Stage, pattern: str) -> List[Usd.Prim]:
    """
    Find all prims on the stage that match the given pattern.

    Args:
        stage (Usd.Stage): The stage to search for prims.
        pattern (str): The pattern to match against prim paths.

    Returns:
        List[Usd.Prim]: A list of prims that match the pattern.
    """
    matching_prims = []
    for prim in stage.TraverseAll():
        if Sdf.Path.IsValidPathString(pattern):
            prim_path = prim.GetPath()
            if prim_path.HasPrefix(Sdf.Path(pattern)) or prim_path == Sdf.Path(pattern):
                matching_prims.append(prim)
        elif pattern in str(prim.GetPath()):
            matching_prims.append(prim)
    valid_prims = [prim for prim in matching_prims if prim.IsValid()]
    return valid_prims


def find_and_replace_material(stage: Usd.Stage, search_material: str, replace_material: str) -> int:
    """
    Find and replace material bindings on all prims in the given stage.

    Args:
        stage (Usd.Stage): The USD stage to search for material bindings.
        search_material (str): The material path to search for.
        replace_material (str): The material path to replace with.

    Returns:
        int: The number of material bindings replaced.
    """
    num_replacements = 0
    for prim in stage.TraverseAll():
        binding_api = UsdShade.MaterialBindingAPI(prim)
        direct_binding_rel = binding_api.GetDirectBindingRel()
        if direct_binding_rel.GetTargets():
            bound_material_path = direct_binding_rel.GetTargets()[0]
            if str(bound_material_path) == search_material:
                direct_binding_rel.SetTargets([replace_material])
                num_replacements += 1
    return num_replacements


def align_prims(stage: Usd.Stage, source_prim_path: str, dest_prim_path: str) -> bool:
    """Align one prim to another by copying its transformation properties."""
    source_prim = stage.GetPrimAtPath(source_prim_path)
    dest_prim = stage.GetPrimAtPath(dest_prim_path)
    if not source_prim.IsValid() or not dest_prim.IsValid():
        raise ValueError("One or both prim paths are invalid.")
    source_xformable = UsdGeom.Xformable(source_prim)
    dest_xformable = UsdGeom.Xformable(dest_prim)
    if not source_xformable or not dest_xformable:
        raise ValueError("One or both prims are not Xformable.")
    translation = [
        op.Get() for op in source_xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    rotation = [
        op.Get() for op in source_xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ
    ]
    scale = [op.Get() for op in source_xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
    if translation:
        add_translate_op(dest_xformable).Set(translation[0])
    if rotation:
        add_rotate_xyz_op(dest_xformable).Set(rotation[0])
    if scale:
        add_scale_op(dest_xformable).Set(scale[0])
    return True


def create_lod_variants(prim: Usd.Prim, lod_names: List[str]) -> Usd.VariantSet:
    """Create a LOD variant set on the given prim with the specified LOD names.

    Args:
        prim (Usd.Prim): The prim to create the LOD variant set on.
        lod_names (List[str]): The names of the LOD variants to create.

    Returns:
        Usd.VariantSet: The created LOD variant set.

    Raises:
        ValueError: If the prim is not valid or if the list of LOD names is empty.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if not lod_names:
        raise ValueError("List of LOD names is empty.")
    lod_varset = prim.GetVariantSets().AddVariantSet("LOD")
    for lod_name in lod_names:
        lod_varset.AddVariant(lod_name)
        lod_varset.SetVariantSelection(lod_name)
        edit_target = lod_varset.GetVariantEditTarget()
        with Usd.EditContext(prim.GetStage(), edit_target):
            pass
    lod_varset.SetVariantSelection(lod_names[0])
    return lod_varset


def create_hair_simulation(
    stage: Usd.Stage, hair_prim_path: str, rest_points: List[Gf.Vec3f], num_curves: int
) -> UsdSkel.Root:
    """Create a hair simulation setup in USD.

    Args:
        stage (Usd.Stage): The USD stage to create the hair simulation on.
        hair_prim_path (str): The path where the hair prim will be created.
        rest_points (List[Gf.Vec3f]): The rest points for the hair curves.
        num_curves (int): The number of hair curves to create.

    Returns:
        UsdSkel.Root: The created Root prim for the hair simulation.
    """
    if not Sdf.Path(hair_prim_path).IsPrimPath():
        raise ValueError(f"Invalid hair prim path: {hair_prim_path}")
    hair_prim = UsdGeom.BasisCurves.Define(stage, hair_prim_path)
    hair_prim.CreateTypeAttr().Set(UsdGeom.Tokens.cubic)
    hair_prim.CreateBasisAttr().Set(UsdGeom.Tokens.bspline)
    vertex_counts = [len(rest_points) // num_curves] * num_curves
    hair_prim.CreateCurveVertexCountsAttr().Set(vertex_counts)
    hair_prim.CreatePointsAttr().Set(rest_points)
    root = UsdSkel.Root.Define(stage, hair_prim_path)
    skeleton = UsdSkel.Skeleton.Define(stage, f"{hair_prim_path}/Skeleton")
    binding_api = UsdSkel.BindingAPI(hair_prim)
    binding_api.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    return root


def setup_wind_simulation(
    stage: Usd.Stage, wind_prim_path: str, wind_speed: float = 1.0, wind_angle: float = 0.0
) -> UsdGeom.Xform:
    """Set up a wind simulation for a USD stage using an Xform prim.

    Args:
        stage (Usd.Stage): The USD stage to set up the wind simulation on.
        wind_prim_path (str): The path where the wind prim will be created.
        wind_speed (float, optional): The speed of the wind. Defaults to 1.0.
        wind_angle (float, optional): The angle of the wind in degrees. Defaults to 0.0.

    Returns:
        UsdGeom.Xform: The created wind prim.
    """
    wind_prim = UsdGeom.Xform.Define(stage, wind_prim_path)
    wind_speed_attr = wind_prim.GetPrim().CreateAttribute("windSpeed", Sdf.ValueTypeNames.Float)
    wind_speed_attr.Set(wind_speed)
    wind_angle_attr = wind_prim.GetPrim().CreateAttribute("windAngle", Sdf.ValueTypeNames.Float)
    wind_angle_attr.Set(wind_angle)
    wind_rotation = Gf.Rotation(Gf.Vec3d(0, 1, 0), wind_angle)
    add_rotate_xyz_op(wind_prim).Set(wind_rotation.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis()))
    return wind_prim


def blend_shapes(
    stage: Usd.Stage, prim_path: str, blend_shape_names: List[str], blend_shape_weights: List[float]
) -> None:
    """Blend shapes on a Mesh primitive using UsdSkel.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): Path to the Mesh primitive.
        blend_shape_names (List[str]): List of blend shape names.
        blend_shape_weights (List[float]): List of blend shape weights.

    Raises:
        ValueError: If the prim is not a valid Mesh or if the number of names and weights don't match.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"Prim at path {prim_path} is not a valid Mesh.")
    if len(blend_shape_names) != len(blend_shape_weights):
        raise ValueError("Number of blend shape names and weights must match.")
    blend_shape_api = UsdSkel.BlendShape(prim)
    for name, weight in zip(blend_shape_names, blend_shape_weights):
        inbetween = blend_shape_api.CreateInbetween(name)
        inbetween.SetWeight(weight)


def create_light_rig(
    stage: Usd.Stage,
    root_path: str = "/LightRig",
    light_types: List[str] = ["DistantLight", "DomeLight", "SphereLight"],
    positions: List[Tuple[float, float, float]] = [(0, 10, 0), (0, 0, 0), (5, 5, 5)],
    intensities: List[float] = [1500.0, 1.0, 5.0],
) -> Usd.Prim:
    """Create a light rig with specified light types, positions and intensities.

    Args:
        stage (Usd.Stage): The USD stage to create the light rig on.
        root_path (str): The root path for the light rig prim. Defaults to "/LightRig".
        light_types (List[str]): List of light types to create. Defaults to ["DistantLight", "DomeLight", "SphereLight"].
        positions (List[Tuple[float, float, float]]): List of positions for each light. Defaults to [(0, 10, 0), (0, 0, 0), (5, 5, 5)].
        intensities (List[float]): List of intensities for each light. Defaults to [1500.0, 1.0, 5.0].

    Returns:
        Usd.Prim: The created light rig prim.

    Raises:
        ValueError: If the number of light types, positions, and intensities do not match.
    """
    if len(light_types) != len(positions) or len(light_types) != len(intensities):
        raise ValueError("Number of light types, positions, and intensities must match.")
    light_rig_prim = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    for light_type, position, intensity in zip(light_types, positions, intensities):
        light_prim = stage.DefinePrim(light_rig_prim.GetPath().AppendChild(light_type), light_type)
        xformable = UsdGeom.Xformable(light_prim)
        xform_op = add_translate_op(xformable)
        xform_op.Set(Gf.Vec3f(position))
        light_prim.CreateAttribute("intensity", Sdf.ValueTypeNames.Float).Set(intensity)
    return light_rig_prim


def configure_particle_system(
    prim: UsdGeom.PointInstancer,
    accel: Tuple[float, float, float],
    drag: float,
    mass_range: Tuple[float, float],
    life_range: Tuple[float, float],
    rate: float,
) -> None:
    """Configure a particle system prim with the given parameters."""
    if not prim:
        raise ValueError(f"Invalid PointInstancer prim.")
    if drag < 0:
        raise ValueError("Drag must be non-negative.")
    if mass_range[0] > mass_range[1]:
        raise ValueError("Mass range minimum must be less than or equal to maximum.")
    if life_range[0] > life_range[1]:
        raise ValueError("Life range minimum must be less than or equal to maximum.")
    if rate <= 0:
        raise ValueError("Rate must be positive.")
    accel_attr = prim.GetPrim().CreateAttribute("accel", Sdf.ValueTypeNames.Float3)
    accel_attr.Set(Gf.Vec3f(accel[0], accel[1], accel[2]))
    drag_attr = prim.GetPrim().CreateAttribute("drag", Sdf.ValueTypeNames.Float)
    drag_attr.Set(drag)
    mass_attr = prim.GetPrim().CreateAttribute("massRange", Sdf.ValueTypeNames.Float2)
    mass_attr.Set(Gf.Vec2f(mass_range[0], mass_range[1]))
    life_attr = prim.GetPrim().CreateAttribute("lifeRange", Sdf.ValueTypeNames.Float2)
    life_attr.Set(Gf.Vec2f(life_range[0], life_range[1]))
    rate_attr = prim.GetPrim().CreateAttribute("rate", Sdf.ValueTypeNames.Float)
    rate_attr.Set(rate)


def find_prims_with_custom_data(stage: Usd.Stage, custom_data_key: str) -> List[Usd.Prim]:
    """Find all prims on the stage that have a specific custom data key set.

    Args:
        stage (Usd.Stage): The USD stage to search.
        custom_data_key (str): The custom data key to search for.

    Returns:
        List[Usd.Prim]: A list of prims that have the specified custom data key set.
    """
    prims_with_custom_data: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        if prim.HasCustomDataKey(custom_data_key):
            prims_with_custom_data.append(prim)
        for prop in prim.GetProperties():
            if prop.HasCustomDataKey(custom_data_key):
                prims_with_custom_data.append(prim)
                break
    return prims_with_custom_data


def find_prims_with_custom_attributes(stage: Usd.Stage) -> List[Usd.Prim]:
    """
    Find all prims on the given stage that have custom attributes.

    Args:
        stage (Usd.Stage): The USD stage to search for prims with custom attributes.

    Returns:
        List[Usd.Prim]: A list of prims that have custom attributes.
    """
    prims_with_custom_attrs = []
    for prim in stage.TraverseAll():
        for prop in prim.GetProperties():
            if prop.IsCustom():
                prims_with_custom_attrs.append(prim)
                break
    return prims_with_custom_attrs


def set_property_comment(prop_spec: Sdf.PropertySpec, comment: str):
    """Set the comment for a property spec.

    Args:
        prop_spec (Sdf.PropertySpec): The property spec to set the comment on.
        comment (str): The comment to set.
    """
    if not prop_spec:
        raise ValueError("Invalid property spec")
    prop_spec.comment = comment
    if prop_spec.comment != comment:
        raise RuntimeError("Failed to set comment on property spec")


def get_property_variability(property_spec: Sdf.PropertySpec) -> Sdf.Variability:
    """Get the variability of a property spec.

    Args:
        property_spec (Sdf.PropertySpec): The property spec to get the variability of.

    Returns:
        Sdf.Variability: The variability of the property spec.
    """
    if not property_spec:
        raise ValueError("Invalid property spec")
    variability = property_spec.variability
    return variability


def get_property_custom_data(property_spec: Sdf.PropertySpec, key: str) -> Any:
    """
    Retrieves the value associated with a given key in the custom data dictionary of a property spec.

    Args:
        property_spec (Sdf.PropertySpec): The property spec to retrieve the custom data from.
        key (str): The key to retrieve the value for.

    Returns:
        Any: The value associated with the given key, or None if the key is not found.
    """
    if property_spec.customData:
        custom_data = property_spec.customData
        if key in custom_data:
            return custom_data[key]
    return None


def clear_all_default_values(prop_spec: Sdf.PropertySpec) -> None:
    """Clear all default values for a given property spec."""
    if not prop_spec:
        raise ValueError("Invalid property spec")
    if prop_spec.HasDefaultValue():
        prop_spec.ClearDefaultValue()
    if prop_spec.HasInfo("composite"):
        composite_name = prop_spec.GetInfo("composite")
        composite_spec = Sdf.Layer.Find(composite_name)
        if composite_spec:
            clear_all_default_values(composite_spec)


def set_property_display_group(prop_spec: Sdf.PropertySpec, display_group: str) -> None:
    """Set the display group for a property spec.

    Args:
        prop_spec (Sdf.PropertySpec): The property spec to set the display group for.
        display_group (str): The display group to set.

    Raises:
        TypeError: If prop_spec is not an instance of Sdf.PropertySpec.
        ValueError: If display_group is not a valid display group name.
    """
    if not isinstance(prop_spec, Sdf.PropertySpec):
        raise TypeError(f"Expected an instance of Sdf.PropertySpec, got {type(prop_spec)}")
    if not display_group or not isinstance(display_group, str):
        raise ValueError(f"Expected a non-empty string for display_group, got {display_group}")
    prop_spec.displayGroup = display_group


def delete_property(prim: Usd.Prim, property_name: str) -> bool:
    """Delete a property from a prim.

    Args:
        prim (Usd.Prim): The prim to delete the property from.
        property_name (str): The name of the property to delete.

    Returns:
        bool: True if the property was successfully deleted, False otherwise.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if not prim.HasProperty(property_name):
        print(f"Property '{property_name}' does not exist on prim '{prim.GetPath()}'")
        return False
    success = prim.RemoveProperty(property_name)
    return success


def set_property_asset_info(prop_spec: Sdf.PropertySpec, asset_info: dict) -> None:
    """Sets the asset info for a property spec.

    Args:
        prop_spec (Sdf.PropertySpec): The property spec to set the asset info on.
        asset_info (dict): The asset info dictionary to set.

    Raises:
        TypeError: If prop_spec is not an Sdf.PropertySpec or asset_info is not a dictionary.
    """
    if not isinstance(prop_spec, Sdf.PropertySpec):
        raise TypeError("prop_spec must be an instance of Sdf.PropertySpec")
    if not isinstance(asset_info, dict):
        raise TypeError("asset_info must be a dictionary")
    prop_spec.assetInfo = asset_info


def set_property_custom_data(prop_spec: Sdf.PropertySpec, custom_data: Dict[str, Any]) -> None:
    """Set custom data on a property spec.

    Args:
        prop_spec (Sdf.PropertySpec): The property spec to set custom data on.
        custom_data (Dict[str, Any]): The custom data dictionary to set.
    """
    if not prop_spec:
        raise ValueError("Invalid property spec.")
    current_custom_data = prop_spec.customData
    new_custom_data = dict(current_custom_data)
    new_custom_data.update(custom_data)
    prop_spec.customData = new_custom_data


def analyze_scene_hierarchy(pseudo_root_spec: Sdf.PseudoRootSpec) -> Dict[str, List[str]]:
    """
    Analyze the scene hierarchy and return a dictionary mapping each prim path to its child prim paths.

    Args:
        pseudo_root_spec (Sdf.PseudoRootSpec): The pseudo-root spec of the USD layer.

    Returns:
        Dict[str, List[str]]: A dictionary mapping each prim path to a list of its child prim paths.
    """
    hierarchy: Dict[str, List[str]] = {}
    for prim_spec in pseudo_root_spec.nameChildren:
        prim_path: str = prim_spec.path.pathString
        child_paths: List[str] = []
        for child_spec in prim_spec.nameChildren:
            child_path: str = child_spec.path.pathString
            child_paths.append(child_path)
        hierarchy[prim_path] = child_paths
    return hierarchy


def update_reference_layer_offset(reference: Sdf.Reference, layer_offset: Sdf.LayerOffset) -> None:
    """Update the layer offset of a reference.

    Args:
        reference (Sdf.Reference): The reference to update.
        layer_offset (Sdf.LayerOffset): The new layer offset to set.
    """
    if not reference:
        raise ValueError("Invalid reference")
    if not layer_offset:
        raise ValueError("Invalid layer offset")
    updated_reference = Sdf.Reference(reference.assetPath, reference.primPath, layer_offset, reference.customData)
    reference = updated_reference


def list_references_for_prim(prim: Usd.Prim) -> List[Sdf.Reference]:
    """
    Return a list of Sdf.Reference objects for the given prim.

    Args:
        prim (Usd.Prim): The prim to get references for.

    Returns:
        List[Sdf.Reference]: A list of Sdf.Reference objects. Empty list if no references or invalid prim.
    """
    if not prim.IsValid():
        print(f"Warning: Invalid prim: {prim}")
        return []
    prim_spec = prim.GetPrimStack()[0]
    references = []
    for ref in prim_spec.referenceList.GetAddedOrExplicitItems():
        layer = Sdf.Layer.Find(ref.assetPath)
        if not layer:
            continue
        references.append(ref)
    return references


def get_all_references(reference_list_op: Sdf.ReferenceListOp) -> List[Sdf.Reference]:
    """Get all references from a ReferenceListOp, including explicit and added items."""
    all_references: List[Sdf.Reference] = []
    if reference_list_op.isExplicit:
        all_references.extend(reference_list_op.explicitItems)
    else:
        all_references.extend(reference_list_op.addedItems)
        all_references = reference_list_op.prependedItems + all_references + reference_list_op.appendedItems
    all_references = [ref for ref in all_references if ref not in reference_list_op.deletedItems]
    return all_references


def append_reference(reference_list_op: Sdf.ReferenceListOp, reference: Sdf.Reference) -> None:
    """Append a reference to the ReferenceListOp.

    Args:
        reference_list_op (Sdf.ReferenceListOp): The ReferenceListOp to append to.
        reference (Sdf.Reference): The reference to append.
    """
    if reference_list_op.HasItem(reference):
        return
    new_items: List[Sdf.Reference] = list(reference_list_op.GetAddedOrExplicitItems())
    new_items.append(reference)
    reference_list_op.ClearAndMakeExplicit()
    for item in new_items:
        reference_list_op.addedItems.append(item)


def get_explicit_references(ref_list_op: Sdf.ReferenceListOp) -> List[Sdf.Reference]:
    """Returns the explicit references in a ReferenceListOp.

    Args:
        ref_list_op (Sdf.ReferenceListOp): The ReferenceListOp to get explicit references from.

    Returns:
        List[Sdf.Reference]: A list of explicit Sdf.Reference objects.
    """
    if not ref_list_op:
        return []
    if ref_list_op.isExplicit:
        return list(ref_list_op.explicitItems)
    else:
        return list(ref_list_op.GetAddedOrExplicitItems())


def get_ordered_references(reference_list_op: Sdf.ReferenceListOp) -> List[Sdf.Reference]:
    """Get the ordered list of references from a ReferenceListOp.

    Args:
        reference_list_op (Sdf.ReferenceListOp): The ReferenceListOp to get the references from.

    Returns:
        List[Sdf.Reference]: The ordered list of references.
    """
    if reference_list_op.isExplicit:
        return list(reference_list_op.explicitItems)
    else:
        ordered_references = []
        ordered_references.extend(reference_list_op.prependedItems)
        ordered_references.extend(reference_list_op.addedItems)
        ordered_references.extend(reference_list_op.appendedItems)
        return ordered_references


def prepend_reference(reference_list_op: Sdf.ReferenceListOp, reference: Sdf.Reference) -> None:
    """Prepend a reference to the ReferenceListOp.

    Args:
        reference_list_op (Sdf.ReferenceListOp): The ReferenceListOp to modify.
        reference (Sdf.Reference): The reference to prepend.
    """
    if not reference_list_op:
        raise ValueError("Invalid ReferenceListOp")
    if not reference:
        raise ValueError("Invalid reference")
    prepended_items = reference_list_op.prependedItems
    if prepended_items is None:
        reference_list_op.prependedItems = [reference]
    else:
        reference_list_op.prependedItems = [reference] + prepended_items


def delete_reference(reference_list_op: Sdf.ReferenceListOp, reference: Sdf.Reference) -> bool:
    """Delete a reference from a ReferenceListOp.

    Args:
        reference_list_op (Sdf.ReferenceListOp): The ReferenceListOp to delete the reference from.
        reference (Sdf.Reference): The reference to delete.

    Returns:
        bool: True if the reference was deleted, False otherwise.
    """
    if not reference_list_op.HasItem(reference):
        return False
    new_list = Sdf.ReferenceListOp()
    for item in reference_list_op.GetAddedOrExplicitItems():
        if item != reference:
            new_list.appendedItems.append(item)
    reference_list_op.Clear()
    reference_list_op.appendedItems = new_list.appendedItems
    return True


def clear_references(reference_list_op: Sdf.ReferenceListOp) -> None:
    """Clears all references from the given ReferenceListOp.

    Args:
        reference_list_op (Sdf.ReferenceListOp): The ReferenceListOp to clear.

    Raises:
        TypeError: If the input is not a ReferenceListOp.
    """
    if isinstance(reference_list_op, Sdf.ReferenceListOp):
        reference_list_op.Clear()
    else:
        raise TypeError("Input must be a ReferenceListOp.")


def create_and_add_reference(reference_list_op: Sdf.ReferenceListOp, reference_path: str, append=True) -> bool:
    """Create a reference and add it to the ReferenceListOp.

    Args:
        reference_list_op (Sdf.ReferenceListOp): The ReferenceListOp to add the reference to.
        reference_path (str): The path to the referenced layer.
        append (bool, optional): If True, the reference is appended to the end of the list.
                                  If False, the reference is prepended to the beginning of the list.
                                  Defaults to True.

    Returns:
        bool: True if the reference was successfully added, False otherwise.
    """
    if not reference_path:
        return False
    new_reference = Sdf.Reference(reference_path)
    if reference_list_op.HasItem(new_reference):
        return False
    if append:
        reference_list_op.appendedItems.append(new_reference)
    else:
        reference_list_op.prependedItems.append(new_reference)
    return True


def set_explicit_references(reference_list_op: Sdf.ReferenceListOp, references: List[Sdf.Reference]) -> None:
    """Set the explicit references in a ReferenceListOp.

    Args:
        reference_list_op (Sdf.ReferenceListOp): The ReferenceListOp to modify.
        references (List[Sdf.Reference]): The list of references to set as explicit.

    Raises:
        ValueError: If the input reference_list_op is not valid.
    """
    if not reference_list_op:
        raise ValueError("Invalid ReferenceListOp. Cannot set explicit references.")
    reference_list_op.ClearAndMakeExplicit()
    for reference in references:
        reference_list_op.addedItems.append(reference)


def has_reference(self, assetPath: Sdf.AssetPath) -> bool:
    """Check if the ReferenceListOp contains a reference to the specified asset path.

    Args:
        assetPath (Sdf.AssetPath): The asset path to check for.

    Returns:
        bool: True if the ReferenceListOp contains a reference to the specified asset path, False otherwise.
    """
    if any((item.assetPath == assetPath for item in self.explicitItems)):
        return True
    if any((item.assetPath == assetPath for item in self.addedItems)):
        return True
    if any((item.assetPath == assetPath for item in self.prependedItems)):
        return True
    if any((item.assetPath == assetPath for item in self.appendedItems)):
        return True
    return False


def get_relationship_targets(rel_spec: Sdf.RelationshipSpec) -> List[Sdf.Path]:
    """Get the target paths of a relationship spec.

    Args:
        rel_spec (Sdf.RelationshipSpec): The relationship spec to query.

    Returns:
        List[Sdf.Path]: The target paths of the relationship spec.
    """
    if not rel_spec:
        raise ValueError("Invalid relationship spec")
    target_path_list = rel_spec.targetPathList
    if target_path_list.isExplicit:
        return list(target_path_list.explicitItems)
    else:
        resolved_paths = Sdf.PathListOp()
        for item in target_path_list.addedItems + target_path_list.prependedItems:
            resolved_paths.appendedItems.append(item)
        for item in target_path_list.deletedItems:
            resolved_paths.deletedItems.append(item)
        for item in target_path_list.orderedItems:
            resolved_paths.appendedItems.append(item)
        return list(resolved_paths.ApplyOperations([]))


def remove_relationship_target(
    relationship_spec: Sdf.RelationshipSpec, target_path: Sdf.Path, preserve_target_order: bool = False
) -> None:
    """Remove a target path from a relationship spec.

    Args:
        relationship_spec (Sdf.RelationshipSpec): The relationship spec to remove the target path from.
        target_path (Sdf.Path): The target path to remove.
        preserve_target_order (bool, optional): Whether to preserve the order of the remaining targets. Defaults to False.
    """
    if not relationship_spec:
        raise ValueError("Invalid relationship spec.")
    target_path_list_editor = relationship_spec.targetPathList
    if not target_path_list_editor.ContainsItemEdit(target_path):
        raise ValueError(f"Target path {target_path} does not exist in the relationship.")
    relationship_spec.RemoveTargetPath(target_path, preserve_target_order)


def filter_prims_by_metadata(stage: Usd.Stage, key: str, value: str) -> List[Usd.Prim]:
    """
    Filter prims in the stage by a metadata key-value pair.

    Args:
        stage (Usd.Stage): The USD stage to filter prims from.
        key (str): The metadata key to filter by.
        value (str): The metadata value to filter by.

    Returns:
        List[Usd.Prim]: A list of prims that match the metadata key-value pair.
    """
    matching_prims: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        if prim.HasCustomDataKey(key):
            prim_value = prim.GetCustomDataByKey(key)
            if prim_value == value:
                matching_prims.append(prim)
    return matching_prims


def remove_metadata_keys(spec: Sdf.Spec, keys_to_remove: List[str]) -> None:
    """Remove the specified metadata keys from the given spec.

    Args:
        spec (Sdf.Spec): The spec to remove metadata keys from.
        keys_to_remove (List[str]): The list of metadata keys to remove.

    Raises:
        ValueError: If the spec is invalid.
    """
    if not spec:
        raise ValueError("Invalid spec. Cannot remove metadata keys.")
    all_keys = spec.ListInfoKeys()
    for key in keys_to_remove:
        if key in all_keys:
            spec.ClearInfo(key)


def transfer_metadata(source_spec: Sdf.Spec, dest_spec: Sdf.Spec, skip_keys: Optional[List[str]] = None):
    """Transfer metadata from source_spec to dest_spec, skipping keys if provided."""
    info_keys = source_spec.GetMetaDataInfoKeys()
    for key in info_keys:
        if skip_keys and key in skip_keys:
            continue
        value = source_spec.GetInfo(key)
        dest_spec.SetInfo(key, value)


def extract_metadata_by_keys(spec: Sdf.Spec, keys: List[str]) -> Dict[str, Sdf.ValueBlock]:
    """
    Extracts metadata values from a given Sdf.Spec based on the provided keys.

    Args:
        spec (Sdf.Spec): The Sdf.Spec to extract metadata from.
        keys (List[str]): The list of metadata keys to extract.

    Returns:
        Dict[str, Sdf.ValueBlock]: A dictionary mapping metadata keys to their corresponding values.
    """
    metadata = {}
    for key in keys:
        if spec.HasInfo(key):
            value = spec.GetInfo(key)
            metadata[key] = value
        else:
            metadata[key] = None
    return metadata


def compare_metadata(spec1: Usd.Prim, spec2: Usd.Prim) -> Dict[str, Tuple[Any, Any]]:
    """Compare metadata between two Usd prims and return a dictionary of differences.

    Args:
        spec1 (Usd.Prim): The first prim to compare.
        spec2 (Usd.Prim): The second prim to compare.

    Returns:
        Dict[str, Tuple[Any, Any]]: A dictionary where keys are metadata names and
        values are tuples of the form (value1, value2) representing the different
        values for the metadata in spec1 and spec2 respectively. Only metadata with
        differing values are included in the dictionary.
    """
    keys1 = spec1.GetAllMetadata().keys()
    keys2 = spec2.GetAllMetadata().keys()
    all_keys = set(keys1) | set(keys2)
    diffs = {}
    for key in all_keys:
        if spec1.HasMetadata(key) and spec2.HasMetadata(key):
            value1 = spec1.GetMetadata(key)
            value2 = spec2.GetMetadata(key)
            if value1 != value2:
                diffs[key] = (value1, value2)
        elif spec1.HasMetadata(key):
            diffs[key] = (spec1.GetMetadata(key), None)
        else:
            diffs[key] = (None, spec2.GetMetadata(key))
    return diffs


def list_prims_with_empty_metadata(stage: Usd.Stage) -> List[Sdf.Path]:
    """Return a list of prim paths that have no metadata."""
    empty_metadata_prims: List[Sdf.Path] = []
    for prim in stage.Traverse():
        prim_spec: Sdf.PrimSpec = stage.GetRootLayer().GetPrimAtPath(prim.GetPath())
        info_keys: List[str] = prim_spec.ListInfoKeys()
        if len(info_keys) == 0:
            empty_metadata_prims.append(prim.GetPath())
    return empty_metadata_prims


def copy_prim_with_metadata(stage: Usd.Stage, source_prim_path: str, dest_prim_path: str) -> Usd.Prim:
    """Copy a prim and its metadata to a new path."""
    source_prim = stage.GetPrimAtPath(source_prim_path)
    if not source_prim.IsValid():
        raise ValueError(f"Source prim at path {source_prim_path} does not exist.")
    dest_parent_path = Sdf.Path(dest_prim_path).GetParentPath()
    dest_parent_prim = stage.GetPrimAtPath(dest_parent_path)
    if not dest_parent_prim.IsValid():
        raise ValueError(f"Destination parent prim at path {dest_parent_path} does not exist.")
    dest_prim = stage.DefinePrim(dest_prim_path)
    reference = Sdf.Reference("", source_prim.GetPath())
    dest_prim.GetReferences().AddReference(reference)
    for key, value in source_prim.GetAllMetadata().items():
        if Tf.Type.FindByName(key) is None:
            continue
        dest_prim.SetMetadata(key, value)
    return dest_prim


def check_metadata_keys(prim: Usd.Prim, expected_keys: List[str]) -> bool:
    """Check if the given prim has all the expected metadata keys.

    Args:
        prim (Usd.Prim): The prim to check metadata keys on.
        expected_keys (List[str]): The list of expected metadata keys.

    Returns:
        bool: True if the prim has all the expected keys, False otherwise.
    """
    metadata_keys = prim.GetAllMetadata().keys()
    for key in expected_keys:
        if key not in metadata_keys:
            return False
    return True


def analyze_and_modify_spec_type(spec_type_name: str) -> Tuple[Sdf.SpecType, bool]:
    """
    Analyze and modify the given spec type name.

    Args:
        spec_type_name (str): The name of the spec type to analyze and modify.

    Returns:
        Tuple[Sdf.SpecType, bool]: A tuple containing the analyzed spec type and a boolean
        indicating if the spec type was modified.
    """
    spec_type = Sdf.SpecType.GetValueFromName(spec_type_name)
    if spec_type == Sdf.SpecTypeUnknown:
        raise ValueError(f"Invalid spec type name: {spec_type_name}")
    modify = False
    if spec_type == Sdf.SpecTypeAttribute:
        spec_type = Sdf.SpecTypeRelationship
        modify = True
    elif spec_type == Sdf.SpecTypePrim:
        spec_type = Sdf.SpecTypeClass
        modify = True
    return (spec_type, modify)


def create_and_set_prim_specifier(stage: Usd.Stage, prim_path: str, specifier_name: str) -> Usd.Prim:
    """Create a new prim with the given specifier if it doesn't exist.

    Args:
        stage (Usd.Stage): The stage to create the prim on.
        prim_path (str): The path of the prim to create or get.
        specifier_name (str): The name of the specifier to set on the prim.

    Returns:
        Usd.Prim: The created or retrieved prim with the specifier set.

    Raises:
        ValueError: If the specifier name is invalid.
    """
    valid_specifiers = {"def": Sdf.SpecifierDef, "over": Sdf.SpecifierOver, "class": Sdf.SpecifierClass}
    specifier = valid_specifiers.get(specifier_name)
    if specifier is None:
        raise ValueError(f"Invalid specifier name: {specifier_name}")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        prim = stage.DefinePrim(prim_path)
    prim.SetSpecifier(specifier)
    return prim


def reorder_prim_paths(paths: List[str], new_order: List[int]) -> List[str]:
    """Reorder a list of prim paths based on a list of new indices.

    Args:
        paths (List[str]): The list of prim paths to reorder.
        new_order (List[int]): The list of new indices specifying the desired order.

    Returns:
        List[str]: The reordered list of prim paths.
    """
    if len(paths) != len(new_order):
        raise ValueError("The lengths of paths and new_order must be the same.")
    if set(new_order) != set(range(len(paths))):
        raise ValueError("new_order must contain indices from 0 to len(paths) - 1.")
    op = Sdf.StringListOp()
    op.explicitItems = paths
    reordered_paths = [None] * len(paths)
    for i, index in enumerate(new_order):
        reordered_paths[i] = paths[index]
    op.explicitItems = reordered_paths
    return list(op.explicitItems)


def apply_operations_to_prim_paths(prim_paths: List[str], operations: Sdf.StringListOp) -> List[str]:
    """Apply string list operations to a list of prim paths.

    Args:
        prim_paths (List[str]): The list of prim paths to apply the operations to.
        operations (Sdf.StringListOp): The string list operations to apply.

    Returns:
        List[str]: The resulting list of prim paths after applying the operations.
    """
    result = prim_paths.copy()
    result = operations.orderedItems if operations.orderedItems is not None else result
    if operations.prependedItems is not None:
        result = operations.prependedItems + result
    if operations.appendedItems is not None:
        result.extend(operations.appendedItems)
    if operations.deletedItems is not None:
        result = [path for path in result if path not in operations.deletedItems]
    if operations.isExplicit:
        result = operations.GetAddedOrExplicitItems()
    return result


def has_item_in_operations(string_list_op: Sdf.StringListOp, item: str) -> bool:
    """Check if an item exists in any of the StringListOp operations.

    Args:
        string_list_op (Sdf.StringListOp): The StringListOp to check.
        item (str): The item to search for.

    Returns:
        bool: True if the item exists in any operation, False otherwise.
    """
    if string_list_op.isExplicit and item in string_list_op.explicitItems:
        return True
    if item in string_list_op.addedItems:
        return True
    if item in string_list_op.prependedItems:
        return True
    if item in string_list_op.appendedItems:
        return True
    if item in string_list_op.orderedItems:
        return True
    return False


def clear_and_make_prim_path_list_explicit(prim_path_list_op: Sdf.StringListOp, prim_paths: List[str]) -> None:
    """Clear the prim path list op and make it explicit with the given prim paths.

    Args:
        prim_path_list_op (Sdf.StringListOp): The prim path list op to modify.
        prim_paths (List[str]): The list of prim paths to set explicitly.

    Raises:
        ValueError: If the input prim_path_list_op is already explicit.
    """
    if prim_path_list_op.isExplicit:
        raise ValueError("The input prim path list op is already explicit.")
    prim_path_list_op.ClearAndMakeExplicit()
    prim_path_list_op.explicitItems = prim_paths


def create_and_apply_operations(
    explicit_items: List[str],
    added_items: List[str],
    prepended_items: List[str],
    appended_items: List[str],
    deleted_items: List[str],
) -> Sdf.StringListOp:
    """Create a StringListOp with the given lists and apply the operations.

    Args:
        explicit_items (List[str]): List of explicit items.
        added_items (List[str]): List of items to add.
        prepended_items (List[str]): List of items to prepend.
        appended_items (List[str]): List of items to append.
        deleted_items (List[str]): List of items to delete.

    Returns:
        Sdf.StringListOp: The resulting StringListOp after applying the operations.
    """
    string_list_op = Sdf.StringListOp.CreateExplicit(explicit_items)
    operations = Sdf.StringListOp()
    for item in added_items:
        operations.addedItems.append(item)
    for item in prepended_items:
        operations.prependedItems.append(item)
    for item in appended_items:
        operations.appendedItems.append(item)
    for item in deleted_items:
        operations.deletedItems.append(item)
    string_list_op.ApplyOperations(operations)
    return string_list_op


def append_items_to_prim_paths(prim_paths: list[Sdf.Path], items_to_append: Sdf.StringListOp) -> list[Sdf.Path]:
    """Append items from a StringListOp to a list of prim paths.

    Args:
        prim_paths (list[Sdf.Path]): A list of prim paths.
        items_to_append (Sdf.StringListOp): A StringListOp containing items to append.

    Returns:
        list[Sdf.Path]: A new list of prim paths with the items appended.
    """
    updated_prim_paths = []
    for prim_path in prim_paths:
        if not prim_path.IsAbsolutePath():
            raise ValueError(f"Invalid prim path: {prim_path}")
        for item in items_to_append.appendedItems:
            new_path = prim_path.AppendPath(item)
            updated_prim_paths.append(new_path)
    return updated_prim_paths


def delete_items_from_prim_paths(op: Sdf.StringListOp, prim_paths: List[str]) -> None:
    """Delete items from a StringListOp that match the given prim paths.

    Args:
        op (Sdf.StringListOp): The StringListOp to modify.
        prim_paths (List[str]): The list of prim paths to delete from the StringListOp.
    """
    for prim_path in prim_paths:
        if op.HasItem(prim_path):
            if op.isExplicit:
                explicit_items = list(op.explicitItems)
                explicit_items.remove(prim_path)
                op.explicitItems = explicit_items
            else:
                deleted_items = list(op.deletedItems)
                deleted_items.append(prim_path)
                op.deletedItems = deleted_items


def resample_animation(stage: Usd.Stage, prim_path: str, time_samples: List[Usd.TimeCode]) -> bool:
    """Resample the animation for a prim at the given time samples.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path of the prim to resample.
        time_samples (List[Usd.TimeCode]): The time samples to resample at.

    Returns:
        bool: True if the resampling was successful, False otherwise.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if not translate_ops:
        return False
    translate_op = translate_ops[0]
    attr = translate_op.GetAttr()
    for time_sample in time_samples:
        value = attr.Get(time_sample)
        if value is not None:
            attr.Set(value, time_sample)
        else:
            attr.Set(Gf.Vec3d(0, 0, 0), time_sample)
    return True


def merge_timecode_arrays(arrays: List[Sdf.TimeCodeArray]) -> Sdf.TimeCodeArray:
    """Merge multiple TimeCodeArrays into a single TimeCodeArray.

    Args:
        arrays (List[Sdf.TimeCodeArray]): A list of TimeCodeArrays to merge.

    Returns:
        Sdf.TimeCodeArray: A new TimeCodeArray containing all the time codes from the input arrays.
    """
    unique_timecodes = set()
    for array in arrays:
        for timecode in array:
            unique_timecodes.add(timecode)
    sorted_timecodes = sorted(list(unique_timecodes))
    merged_array = Sdf.TimeCodeArray(sorted_timecodes)
    return merged_array


def resample_animations(stage: Usd.Stage, prim_path: str, time_samples: Sdf.TimeCodeArray) -> bool:
    """Resample the animation on a prim at the given time samples.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to resample.
        time_samples (Sdf.TimeCodeArray): The time samples to resample at.

    Returns:
        bool: True if the resampling was successful, False otherwise.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    translate_op = None
    rotate_op = None
    scale_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            rotate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
    if translate_op:
        attr = translate_op.GetAttr()
        if attr.GetNumTimeSamples() > 0:
            attr.Clear()
            for time_code in time_samples:
                value = translate_op.Get(time_code)
                if value is not None:
                    attr.Set(value, time_code)
    if rotate_op:
        attr = rotate_op.GetAttr()
        if attr.GetNumTimeSamples() > 0:
            attr.Clear()
            for time_code in time_samples:
                value = rotate_op.Get(time_code)
                if value is not None:
                    attr.Set(value, time_code)
    if scale_op:
        attr = scale_op.GetAttr()
        if attr.GetNumTimeSamples() > 0:
            attr.Clear()
            for time_code in time_samples:
                value = scale_op.Get(time_code)
                if value is not None:
                    attr.Set(value, time_code)
    return True


def generate_timecode_snapshot(timecode_array: Sdf.TimeCodeArray, index: int) -> Sdf.TimeCode:
    """
    Generate a timecode snapshot from a TimeCodeArray at the specified index.

    Args:
        timecode_array (Sdf.TimeCodeArray): The input TimeCodeArray.
        index (int): The index of the desired timecode snapshot.

    Returns:
        Sdf.TimeCode: The timecode snapshot at the specified index.

    Raises:
        IndexError: If the provided index is out of bounds.
    """
    if index < 0 or index >= len(timecode_array):
        raise IndexError(f"Index {index} is out of bounds for TimeCodeArray of length {len(timecode_array)}")
    timecode_snapshot = timecode_array[index]
    return timecode_snapshot


def bake_transforms_at_times(
    prim: Usd.Prim, time_codes: Sdf.TimeCodeArray
) -> Tuple[List[Gf.Matrix4d], List[Usd.TimeCode]]:
    """
    Bakes the local transform of a prim at specified time codes.

    Args:
        prim (Usd.Prim): The prim to bake transforms for.
        time_codes (Sdf.TimeCodeArray): The time codes to bake transforms at.

    Returns:
        A tuple containing two lists:
        - A list of Gf.Matrix4d objects representing the baked local transforms.
        - A list of Usd.TimeCode objects representing the corresponding time codes.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if not time_codes:
        raise ValueError("Empty time_codes array")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim '{prim.GetPath()}' is not transformable")
    baked_xforms = []
    baked_times = []
    for time_code in time_codes:
        usd_time_code = Usd.TimeCode(time_code)
        local_xform = xformable.GetLocalTransformation(usd_time_code)
        baked_xforms.append(local_xform)
        baked_times.append(usd_time_code)
    return (baked_xforms, baked_times)


def retime_animations(stage: Usd.Stage, time_codes: Sdf.TimeCodeArray, scale: float = 1.0, offset: float = 0.0):
    """Retimes the animation on a USD stage.

    Args:
        stage (Usd.Stage): The USD stage to retime animations on.
        time_codes (Sdf.TimeCodeArray): The time codes to retime.
        scale (float, optional): The scale factor to apply to the time codes. Defaults to 1.0.
        offset (float, optional): The offset to apply to the time codes. Defaults to 0.0.

    Returns:
        None
    """
    if not stage:
        raise ValueError("Invalid USD stage.")
    if not time_codes:
        raise ValueError("Time code array is empty.")
    for prim in stage.TraverseAll():
        for attr in prim.GetAttributes():
            if attr.ValueMightBeTimeVarying():
                existing_time_samples = attr.GetTimeSamples()
                remapped_time_samples = {}
                for time_code in existing_time_samples:
                    remapped_time_code = time_code * scale + offset
                    remapped_time_samples[Usd.TimeCode(remapped_time_code)] = attr.Get(time_code)
                for time_code, value in remapped_time_samples.items():
                    attr.Set(value, time_code)


def analyze_token_operations(token_list_op: Sdf.TokenListOp) -> List[str]:
    """Analyze the TokenListOp and return a list of strings describing its state."""
    result = []
    if token_list_op.isExplicit:
        result.append("TokenListOp is explicit")
    else:
        result.append("TokenListOp is not explicit")
    explicit_items = token_list_op.explicitItems
    if explicit_items:
        result.append(f"Explicit items: {', '.join(explicit_items)}")
    added_items = token_list_op.addedItems
    if added_items:
        result.append(f"Added items: {', '.join(added_items)}")
    prepended_items = token_list_op.prependedItems
    if prepended_items:
        result.append(f"Prepended items: {', '.join(prepended_items)}")
    appended_items = token_list_op.appendedItems
    if appended_items:
        result.append(f"Appended items: {', '.join(appended_items)}")
    deleted_items = token_list_op.deletedItems
    if deleted_items:
        result.append(f"Deleted items: {', '.join(deleted_items)}")
    ordered_items = token_list_op.orderedItems
    if ordered_items:
        result.append(f"Ordered items: {', '.join(ordered_items)}")
    return result


def merge_token_lists(list1: Sdf.TokenListOp, list2: Sdf.TokenListOp) -> Sdf.TokenListOp:
    """Merges two TokenListOp objects into a single TokenListOp.

    Args:
        list1 (Sdf.TokenListOp): The first TokenListOp to merge.
        list2 (Sdf.TokenListOp): The second TokenListOp to merge.

    Returns:
        Sdf.TokenListOp: A new TokenListOp containing the merged items.
    """
    merged_list = Sdf.TokenListOp()
    if list1.isExplicit or list2.isExplicit:
        merged_list = Sdf.TokenListOp.CreateExplicit()
        for item in list1.explicitItems:
            merged_list.explicitItems.append(item)
        for item in list2.explicitItems:
            if item not in merged_list.explicitItems:
                merged_list.explicitItems.append(item)
    else:
        for item in list1.prependedItems:
            merged_list.prependedItems.append(item)
        for item in list2.prependedItems:
            if item not in merged_list.prependedItems:
                merged_list.prependedItems.append(item)
        for item in list1.appendedItems:
            merged_list.appendedItems.append(item)
        for item in list2.appendedItems:
            if item not in merged_list.appendedItems:
                merged_list.appendedItems.append(item)
        for item in list1.addedItems:
            merged_list.addedItems.append(item)
        for item in list2.addedItems:
            if item not in merged_list.addedItems:
                merged_list.addedItems.append(item)
        for item in list1.deletedItems:
            if item in merged_list.orderedItems:
                merged_list.deletedItems.append(item)
        for item in list2.deletedItems:
            if item in merged_list.orderedItems and item not in merged_list.deletedItems:
                merged_list.deletedItems.append(item)
    return merged_list


def clear_token_operations(token_list_op: Sdf.TokenListOp) -> None:
    """Clear all token list operations and make the list explicit."""
    if not isinstance(token_list_op, Sdf.TokenListOp):
        raise TypeError("Expected an Sdf.TokenListOp object.")
    token_list_op.ClearAndMakeExplicit()
    if not token_list_op.isExplicit:
        raise RuntimeError("Failed to make the TokenListOp explicit.")
    if token_list_op.explicitItems:
        raise RuntimeError("TokenListOp is not empty after clearing.")


def get_added_or_explicit_tokens(token_list_op: Sdf.TokenListOp) -> Sdf.TokenListOp:
    """
    Get the added or explicit tokens from a TokenListOp.

    Args:
        token_list_op (Sdf.TokenListOp): The input TokenListOp.

    Returns:
        Sdf.TokenListOp: A new TokenListOp containing only the added or explicit tokens.
    """
    result_op = Sdf.TokenListOp()
    if token_list_op.isExplicit:
        result_op.explicitItems = token_list_op.explicitItems
    else:
        added_items = token_list_op.addedItems
        if added_items:
            result_op.explicitItems = added_items
        else:
            result_op.Clear()
    return result_op


def apply_token_operations(
    token_list_op: Sdf.TokenListOp,
    explicit_tokens: List[str] = None,
    added_tokens: List[str] = None,
    prepended_tokens: List[str] = None,
    appended_tokens: List[str] = None,
    deleted_tokens: List[str] = None,
) -> Sdf.TokenListOp:
    """Apply a series of list operation edits to the given TokenListOp.

    Args:
        token_list_op (Sdf.TokenListOp): The TokenListOp to edit.
        explicit_tokens (List[str], optional): Tokens to set explicitly.
        added_tokens (List[str], optional): Tokens to add.
        prepended_tokens (List[str], optional): Tokens to prepend.
        appended_tokens (List[str], optional): Tokens to append.
        deleted_tokens (List[str], optional): Tokens to delete.

    Returns:
        Sdf.TokenListOp: The modified TokenListOp.
    """
    if explicit_tokens:
        token_list_op.ClearAndMakeExplicit()
        for token in explicit_tokens:
            token_list_op.explicitItems.append(token)
    if deleted_tokens:
        for token in deleted_tokens:
            if token in token_list_op.GetAddedOrExplicitItems():
                token_list_op.deletedItems.append(token)
    if prepended_tokens:
        for token in prepended_tokens:
            token_list_op.prependedItems.append(token)
    if appended_tokens:
        for token in appended_tokens:
            token_list_op.appendedItems.append(token)
    if added_tokens:
        for token in added_tokens:
            if token not in token_list_op.GetAddedOrExplicitItems():
                token_list_op.addedItems.append(token)
    return token_list_op


def filter_tokens(op: Sdf.TokenListOp, filter_list: List[str]) -> List[str]:
    """Filter the items in a TokenListOp based on a filter list.

    Args:
        op (Sdf.TokenListOp): The TokenListOp to filter.
        filter_list (List[str]): The list of items to filter by.

    Returns:
        List[str]: The filtered list of items.
    """
    explicit_items = op.GetAddedOrExplicitItems()
    filtered_items = [item for item in explicit_items if item in filter_list]
    return filtered_items


def create_explicit_token_list(tokens: List[str]) -> Sdf.TokenListOp:
    """Create an explicit Sdf.TokenListOp from a list of tokens."""
    token_list_op = Sdf.TokenListOp()
    if not tokens:
        return token_list_op
    token_list_op.Clear()
    explicit_list = Sdf.TokenListOp.CreateExplicit(tokens)
    return explicit_list


def append_tokens(op: Sdf.TokenListOp, tokens: Sequence[str]) -> None:
    """Append tokens to the TokenListOp.

    Args:
        op (Sdf.TokenListOp): The TokenListOp to modify.
        tokens (Sequence[str]): The tokens to append.
    """
    if not isinstance(op, Sdf.TokenListOp):
        raise TypeError("Invalid input: op must be of type Sdf.TokenListOp")
    if not isinstance(tokens, Sequence):
        raise TypeError("Invalid input: tokens must be a sequence")
    for token in tokens:
        if not isinstance(token, str):
            raise TypeError("Invalid input: all elements in tokens must be of type str")
    for token in tokens:
        op.appendedItems.append(token)


def delete_tokens(op: Sdf.TokenListOp, tokens_to_delete: List[str]) -> Sdf.TokenListOp:
    """Remove the specified tokens from the TokenListOp.

    Args:
        op (Sdf.TokenListOp): The original TokenListOp.
        tokens_to_delete (List[str]): The list of tokens to remove.

    Returns:
        Sdf.TokenListOp: A new TokenListOp with the specified tokens removed.
    """
    new_op = Sdf.TokenListOp()
    new_op.explicitItems = op.explicitItems[:]
    for token in tokens_to_delete:
        if token in new_op.explicitItems:
            new_op.explicitItems.remove(token)
        else:
            new_op.deletedItems.append(token)
    return new_op


def get_ordered_tokens(token_list_op: Sdf.TokenListOp) -> Sdf.TokenListOp:
    """Get the ordered tokens from a TokenListOp.

    Args:
        token_list_op (Sdf.TokenListOp): The TokenListOp to get the ordered tokens from.

    Returns:
        Sdf.TokenListOp: The ordered tokens as a new TokenListOp.
    """
    if token_list_op.isExplicit:
        return Sdf.TokenListOp.CreateExplicit(token_list_op.explicitItems)
    else:
        prepended_op = Sdf.TokenListOp()
        prepended_op.prependedItems = token_list_op.prependedItems
        appended_op = Sdf.TokenListOp()
        appended_op.appendedItems = token_list_op.appendedItems
        result = Sdf.TokenListOp.ApplyOperations(prepended_op, appended_op)
        return Sdf.TokenListOp.Create(result.explicitItems)


def prepend_tokens(token_list_op: Sdf.TokenListOp, tokens: Sequence[str]) -> None:
    """Prepend tokens to the TokenListOp.

    Args:
        token_list_op (Sdf.TokenListOp): The TokenListOp to modify.
        tokens (Sequence[str]): The tokens to prepend.

    Raises:
        ValueError: If the input TokenListOp is explicit.
    """
    if token_list_op.isExplicit:
        raise ValueError("Cannot prepend tokens to an explicit TokenListOp.")
    if not isinstance(tokens, list):
        tokens = list(tokens)
    prepended_items = token_list_op.prependedItems or []
    token_list_op.prependedItems = tokens + prepended_items


def apply_uint64_operations(list_op: Sdf.UInt64ListOp, items: List[int]) -> List[int]:
    """
    Apply the list editing operations in the given Sdf.UInt64ListOp to the provided list of items.

    Args:
        list_op (Sdf.UInt64ListOp): The list operation to apply.
        items (List[int]): The list of items to apply the operations to.

    Returns:
        List[int]: The resulting list after applying the operations.
    """
    result = items.copy()
    if list_op.isExplicit:
        result = list(list_op.explicitItems)
    else:
        for item in list_op.deletedItems:
            if item in result:
                result.remove(item)
        result = list(list_op.prependedItems) + result
        result.extend(list_op.appendedItems)
        for item in list_op.addedItems:
            if item not in result:
                result.append(item)
    return result


def clear_uint64_list(uint64_list_op: Sdf.UInt64ListOp) -> None:
    """Clear the UInt64ListOp, removing all items."""
    if not isinstance(uint64_list_op, Sdf.UInt64ListOp):
        raise TypeError("Input must be an instance of Sdf.UInt64ListOp")
    uint64_list_op.Clear()
    if uint64_list_op.GetAddedOrExplicitItems():
        raise RuntimeError("Failed to clear the UInt64ListOp")


def add_explicit_uint64_items(list_op: Sdf.UInt64ListOp, items: Sequence[int]) -> None:
    """Add explicit uint64 items to a UInt64ListOp.

    Args:
        list_op (Sdf.UInt64ListOp): The UInt64ListOp to add items to.
        items (Sequence[int]): The items to add as explicit items.
    """
    if not isinstance(list_op, Sdf.UInt64ListOp):
        raise TypeError("list_op must be an instance of Sdf.UInt64ListOp")
    uint64_items = [int(item) for item in items]
    for item in uint64_items:
        if not list_op.HasItem(item):
            list_op.addedItems.append(item)
        else:
            if item in list_op.deletedItems:
                list_op.deletedItems.remove(item)
            if item not in list_op.explicitItems:
                list_op.explicitItems.append(item)
    del list_op.prependedItems[:]
    del list_op.appendedItems[:]


def get_added_or_explicit_uint64_items(uint64_list_op: Sdf.UInt64ListOp) -> List[int]:
    """Get the added or explicit items from a UInt64ListOp.

    Args:
        uint64_list_op (Sdf.UInt64ListOp): The UInt64ListOp to retrieve items from.

    Returns:
        List[int]: The list of added or explicit items.
    """
    if uint64_list_op.isExplicit:
        return list(uint64_list_op.explicitItems)
    else:
        return list(uint64_list_op.addedItems)


def check_uint64_item_existence(list_op: Sdf.UInt64ListOp, item: int) -> bool:
    """Check if a uint64 item exists in the list operation.

    Args:
        list_op (Sdf.UInt64ListOp): The list operation to check.
        item (int): The uint64 item to check for existence.

    Returns:
        bool: True if the item exists in the list operation, False otherwise.
    """
    if list_op.isExplicit:
        return list_op.HasItem(item)
    else:
        added_items = list_op.addedItems
        appended_items = list_op.appendedItems
        if added_items and item in added_items:
            return True
        if appended_items and item in appended_items:
            return True
        prepended_items = list_op.prependedItems
        if prepended_items and item in prepended_items:
            return True
        deleted_items = list_op.deletedItems
        if deleted_items and item in deleted_items:
            return False
    return False


def append_uint64_items(list_op: Sdf.UInt64ListOp, items: Sequence[int]) -> None:
    """Append uint64 items to the list op.

    Args:
        list_op (Sdf.UInt64ListOp): The list op to append items to.
        items (Sequence[int]): The items to append.

    Raises:
        TypeError: If any of the items are not integers.
        ValueError: If any of the items are negative.
    """
    if not all((isinstance(item, int) for item in items)):
        raise TypeError("All items must be integers.")
    if any((item < 0 for item in items)):
        raise ValueError("Items must be non-negative.")
    uint64_items = [int(item) for item in items]
    for item in uint64_items:
        list_op.appendedItems.append(item)


def retrieve_ordered_uint64_items(uint64_list_op: Sdf.UInt64ListOp) -> List[int]:
    """Retrieve the ordered items from a UInt64ListOp.

    Args:
        uint64_list_op (Sdf.UInt64ListOp): The UInt64ListOp to retrieve items from.

    Returns:
        List[int]: The ordered items as a list of integers.
    """
    if uint64_list_op.isExplicit:
        return list(uint64_list_op.explicitItems)
    else:
        ordered_items = []
        ordered_items.extend(uint64_list_op.prependedItems)
        if uint64_list_op.explicitItems:
            ordered_items.extend(uint64_list_op.explicitItems)
        ordered_items.extend(uint64_list_op.appendedItems)
        return ordered_items


def convert_to_explicit_uint64(uint64_list_op: Sdf.UInt64ListOp) -> Sdf.UInt64ListOp:
    """Convert a UInt64ListOp to an explicit list operation.

    Args:
        uint64_list_op (Sdf.UInt64ListOp): The input UInt64ListOp to convert.

    Returns:
        Sdf.UInt64ListOp: A new UInt64ListOp with the explicit items set.
    """
    if uint64_list_op.isExplicit:
        return uint64_list_op
    explicit_list_op = Sdf.UInt64ListOp.CreateExplicit()
    ordered_items = uint64_list_op.orderedItems
    for item in ordered_items:
        explicit_list_op.addedItems.append(item)
    return explicit_list_op


def manage_uint64_list_operations(uint64_list_op: Sdf.UInt64ListOp) -> List[int]:
    """Manage UInt64ListOp operations and return the final list."""
    result_list_op = Sdf.UInt64ListOp()
    result_list_op.ClearAndMakeExplicit()
    result_list_op.appendedItems = [10, 20, 30]
    result_list_op.prependedItems = [1, 2, 3]
    result_list_op.addedItems = [15]
    result_list_op.deletedItems = [20]
    Sdf.UInt64ListOp.ApplyOperations(uint64_list_op, result_list_op)
    if uint64_list_op.HasItem(15):
        print("Item 15 exists in the list")
    else:
        print("Item 15 does not exist in the list")
    final_list = uint64_list_op.GetAddedOrExplicitItems()
    return final_list


def prepend_uint64_items(uint64_list_op: Sdf.UInt64ListOp, items: Sequence[int]) -> None:
    """Prepend the given items to the list.

    If the list is not explicit, the given items are prepended to the
    existing prepended items. If the list is explicit, the items become
    the new explicit items.

    Args:
        uint64_list_op (Sdf.UInt64ListOp): The UInt64ListOp to modify.
        items (Sequence[int]): The items to prepend to the list.

    Raises:
        TypeError: If any of the given items are not integers.
    """
    for item in items:
        if not isinstance(item, int):
            raise TypeError(f"Expected int, got {type(item)}")
    prepended_items_op = Sdf.UInt64ListOp()
    prepended_items_op.prependedItems = items
    if uint64_list_op.isExplicit:
        uint64_list_op.Clear()
    uint64_list_op.ApplyOperations(prepended_items_op)


def clear_uint_list(uint_list_op: Sdf.UIntListOp) -> None:
    """Clear the UIntListOp, removing all items."""
    if not uint_list_op:
        raise ValueError("Invalid UIntListOp provided.")
    uint_list_op.Clear()
    if uint_list_op.GetAddedOrExplicitItems():
        raise RuntimeError("Failed to clear the UIntListOp.")


def create_uint_list_op(
    added_items: List[int] = None,
    prepended_items: List[int] = None,
    appended_items: List[int] = None,
    deleted_items: List[int] = None,
    ordered_items: List[int] = None,
) -> Sdf.UIntListOp:
    """Create a UIntListOp with the specified items.

    Args:
        added_items (List[int], optional): Items to add to the list. Defaults to None.
        prepended_items (List[int], optional): Items to prepend to the list. Defaults to None.
        appended_items (List[int], optional): Items to append to the list. Defaults to None.
        deleted_items (List[int], optional): Items to delete from the list. Defaults to None.
        ordered_items (List[int], optional): Explicitly ordered items in the list. Defaults to None.

    Returns:
        Sdf.UIntListOp: The created UIntListOp.
    """
    uint_list_op = Sdf.UIntListOp()
    if added_items:
        for item in added_items:
            uint_list_op.addedItems.append(item)
    if prepended_items:
        for item in prepended_items:
            uint_list_op.prependedItems.append(item)
    if appended_items:
        for item in appended_items:
            uint_list_op.appendedItems.append(item)
    if deleted_items:
        for item in deleted_items:
            uint_list_op.deletedItems.append(item)
    if ordered_items:
        uint_list_op.explicitItems = ordered_items
    return uint_list_op


def combine_uint_list_ops(ops: List[Sdf.UIntListOp]) -> Sdf.UIntListOp:
    """Combine multiple UIntListOp into a single UIntListOp.

    Args:
        ops (List[Sdf.UIntListOp]): List of UIntListOp to combine.

    Returns:
        Sdf.UIntListOp: Combined UIntListOp.
    """
    if not ops:
        return Sdf.UIntListOp()
    combined_op = Sdf.UIntListOp()
    combined_op.explicitItems = ops[0].explicitItems
    for op in ops[1:]:
        for item in op.explicitItems:
            if item not in combined_op.explicitItems:
                combined_op.explicitItems.append(item)
        for item in op.addedItems:
            if item not in combined_op.explicitItems:
                combined_op.addedItems.append(item)
        for item in op.deletedItems:
            if item in combined_op.explicitItems:
                combined_op.explicitItems.remove(item)
            if item in combined_op.addedItems:
                combined_op.addedItems.remove(item)
        combined_op.prependedItems.extend(op.prependedItems)
        combined_op.appendedItems.extend(op.appendedItems)
    return combined_op


def apply_operations_to_uint_list(uint_list_op: Sdf.UIntListOp, items: List[int]) -> List[int]:
    """Apply the list editing operations in uint_list_op to the provided list of items.

    Args:
        uint_list_op (Sdf.UIntListOp): The UIntListOp containing the edit operations.
        items (List[int]): The list of items to apply the operations to.

    Returns:
        List[int]: The resulting list after applying the edit operations.
    """
    result = items.copy()
    if uint_list_op.isExplicit:
        result = uint_list_op.explicitItems
    for item in uint_list_op.deletedItems:
        if item in result:
            result.remove(item)
    for item in uint_list_op.addedItems:
        if item not in result:
            result.append(item)
    result = uint_list_op.prependedItems + result
    result.extend(uint_list_op.appendedItems)
    return result


def get_added_or_explicit_items(self) -> List[int]:
    """Get the list of added or explicit items in the UIntListOp.

    Returns:
        List[int]: The list of added or explicit items.
    """
    if self.isExplicit:
        return self.explicitItems
    else:
        return self.addedItems


@staticmethod
def has_item(uint_list_op: Sdf.UIntListOp, item: int) -> bool:
    """Check if the UIntListOp contains a specific item.

    Args:
        uint_list_op (Sdf.UIntListOp): The UIntListOp to check.
        item (int): The item to search for.

    Returns:
        bool: True if the item is found in the UIntListOp, False otherwise.
    """
    if uint_list_op.isExplicit:
        return item in uint_list_op.explicitItems
    else:
        if item in uint_list_op.addedItems:
            return True
        if item in uint_list_op.appendedItems:
            return True
        if item in uint_list_op.prependedItems:
            return True
        if item in uint_list_op.deletedItems:
            return False
        return False


def get_ordered_items(uint_list_op: Sdf.UIntListOp) -> List[int]:
    """Get the ordered list of items from a UIntListOp."""
    if uint_list_op.isExplicit:
        return list(uint_list_op.explicitItems)
    else:
        ordered_items = []
        ordered_items.extend(uint_list_op.prependedItems)
        ordered_items.extend(uint_list_op.GetAddedOrExplicitItems())
        ordered_items.extend(uint_list_op.appendedItems)
        ordered_items = [item for item in ordered_items if item not in uint_list_op.deletedItems]
        return ordered_items


def prepend_items_to_uint_list(uint_list_op: Sdf.UIntListOp, items: Sequence[int]) -> None:
    """Prepend items to the front of a UIntListOp.

    Args:
        uint_list_op (Sdf.UIntListOp): The UIntListOp to modify.
        items (Sequence[int]): The items to prepend.
    """
    if uint_list_op.isExplicit:
        explicit_items = uint_list_op.explicitItems
        explicit_items[0:0] = items
    else:
        prepended_items = uint_list_op.prependedItems
        if prepended_items is None:
            uint_list_op.prependedItems = items
        else:
            prepended_items[0:0] = items


def delete_items_from_uint_list(uint_list_op: Sdf.UIntListOp, items_to_delete: Sequence[int]) -> None:
    """Delete items from a UIntListOp.

    Args:
        uint_list_op (Sdf.UIntListOp): The UIntListOp to delete items from.
        items_to_delete (Sequence[int]): The items to delete from the UIntListOp.
    """
    if not uint_list_op:
        raise ValueError("uint_list_op is not a valid UIntListOp")
    if not items_to_delete:
        return
    delete_set = set(items_to_delete)
    explicit_items = uint_list_op.GetAddedOrExplicitItems()
    new_explicit_items = [item for item in explicit_items if item not in delete_set]
    uint_list_op.ClearAndMakeExplicit()
    for item in new_explicit_items:
        uint_list_op.appendedItems.append(item)


def migrate_unregistered_value(
    source_spec: Sdf.Spec, source_field_name: str, dest_spec: Sdf.Spec, dest_field_name: str
) -> bool:
    """
    Moves an unregistered metadata value from one spec to another.

    Args:
        source_spec (Sdf.Spec): The spec containing the unregistered value.
        source_field_name (str): The name of the field containing the unregistered value.
        dest_spec (Sdf.Spec): The spec to move the unregistered value to.
        dest_field_name (str): The name of the field to move the unregistered value to.

    Returns:
        bool: True if the unregistered value was successfully moved, False otherwise.
    """
    if not source_spec.HasInfo(source_field_name):
        return False
    unregistered_value = source_spec.GetInfo(source_field_name)
    if not isinstance(unregistered_value, Sdf.UnregisteredValue):
        return False
    dest_spec.SetInfo(dest_field_name, unregistered_value)
    source_spec.ClearInfo(source_field_name)
    return True


def remove_item_from_operations(op: Sdf.UnregisteredValueListOp, item_to_remove: Sdf.UnregisteredValue) -> bool:
    """Remove an item from the list operation.

    Args:
        op (Sdf.UnregisteredValueListOp): The list operation to modify.
        item_to_remove (Sdf.UnregisteredValue): The item to remove from the list operation.

    Returns:
        bool: True if the item was removed, False otherwise.
    """
    if not op.HasItem(item_to_remove):
        return False
    new_op = Sdf.UnregisteredValueListOp.CreateExplicit()
    for item in op.GetAddedOrExplicitItems():
        if item != item_to_remove:
            new_op.addedItems.append(item)
    op.ClearAndMakeExplicit()
    for item in new_op.addedItems:
        op.addedItems.append(item)
    return True


def get_added_or_explicit_items_from_prims(prims: List[Usd.Prim], attribute_name: str) -> List:
    """Get the added or explicit items from a list of prims for a given attribute name.

    Args:
        prims (List[Usd.Prim]): A list of Usd.Prim objects.
        attribute_name (str): The name of the attribute to retrieve the items from.

    Returns:
        List: A list containing the added or explicit items from the prims' attributes.
    """
    items = []
    for prim in prims:
        if not prim.IsValid():
            continue
        attribute = prim.GetAttribute(attribute_name)
        if not attribute.IsValid():
            continue
        value = attribute.Get()
        if value is None:
            continue
        if isinstance(value, Sdf.UnregisteredValueListOp):
            added_or_explicit_items = value.GetAddedOrExplicitItems()
            items.extend(added_or_explicit_items)
        else:
            items.extend(value)
    return items


def check_item_in_operations(list_op: Sdf.UnregisteredValueListOp, item: Any) -> bool:
    """Check if an item exists in any of the list operations.

    Args:
        list_op (Sdf.UnregisteredValueListOp): The list operation object to check.
        item (Any): The item to check for existence.

    Returns:
        bool: True if the item exists in any of the list operations, False otherwise.
    """
    item_str = str(item)
    if list_op.isExplicit and any((str(val) == item_str for val in list_op.explicitItems)):
        return True
    if list_op.addedItems and any((str(val) == item_str for val in list_op.addedItems)):
        return True
    if list_op.prependedItems and any((str(val) == item_str for val in list_op.prependedItems)):
        return True
    if list_op.appendedItems and any((str(val) == item_str for val in list_op.appendedItems)):
        return True
    if list_op.orderedItems and any((str(val) == item_str for val in list_op.orderedItems)):
        return True
    return False


def get_all_items_from_operations(op: Sdf.UnregisteredValueListOp) -> List[Sdf.UnregisteredValue]:
    """Get all items from the UnregisteredValueListOp, combining items from various sources."""
    all_items = []
    if op.isExplicit:
        all_items.extend(op.explicitItems)
    else:
        if op.addedItems:
            all_items.extend(op.addedItems)
        if op.prependedItems:
            all_items = op.prependedItems + all_items
        if op.appendedItems:
            all_items.extend(op.appendedItems)
        if op.orderedItems:
            ordered_indexes = [all_items.index(item) for item in op.orderedItems if item in all_items]
            ordered_indexes.sort()
            all_items = [all_items[index] for index in ordered_indexes]
        if op.deletedItems:
            all_items = [item for item in all_items if item not in op.deletedItems]
    return all_items


def block_default_and_time_samples(stage: Usd.Stage, prim_path: str, attribute_name: str) -> None:
    """Block the default value and time samples for an attribute on a prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.
        attribute_name (str): The name of the attribute to block.

    Raises:
        ValueError: If the prim or attribute does not exist.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    attribute = prim.GetAttribute(attribute_name)
    if not attribute.IsValid():
        raise ValueError(f"Attribute {attribute_name} does not exist on prim {prim_path}.")
    attribute.Set(Sdf.ValueBlock())
    time_samples = attribute.GetTimeSamples()
    for time in time_samples:
        attribute.Set(Sdf.ValueBlock(), time)


def clear_attribute_value(attribute: Usd.Attribute, time_code: Usd.TimeCode = Usd.TimeCode.Default()):
    """Clears the authored value of an attribute at a given time code.

    Args:
        attribute (Usd.Attribute): The attribute to clear the value for.
        time_code (Usd.TimeCode, optional): The time code at which to clear the value.
            Defaults to Usd.TimeCode.Default() which represents the default value.

    Raises:
        ValueError: If the provided attribute is not valid.
    """
    if not attribute.IsValid():
        raise ValueError("Provided attribute is not valid.")
    if time_code == Usd.TimeCode.Default():
        attribute.ClearDefault()
    else:
        attribute.ClearAtTime(time_code)
        if attribute.GetNumTimeSamples() == 0 and (not attribute.HasAuthoredValueOpinion()):
            attribute.Clear()


def copy_prim_with_role(
    source_stage: Usd.Stage,
    source_prim_path: str,
    dest_stage: Usd.Stage,
    dest_prim_path: str,
    role: Optional[Sdf.ValueRoleNames] = None,
) -> Optional[Usd.Prim]:
    """Copy a prim from one stage to another, optionally filtering by value role.

    Args:
        source_stage (Usd.Stage): The stage containing the prim to copy.
        source_prim_path (str): The path of the prim to copy.
        dest_stage (Usd.Stage): The stage to copy the prim to.
        dest_prim_path (str): The path where the prim should be copied to.
        role (Optional[Sdf.ValueRoleNames], optional): The value role to filter properties by. Defaults to None.

    Returns:
        Optional[Usd.Prim]: The copied prim in the destination stage, or None if the copy failed.
    """
    source_prim = source_stage.GetPrimAtPath(source_prim_path)
    if not source_prim.IsValid():
        print(f"Source prim at path {source_prim_path} does not exist.")
        return None
    dest_prim = dest_stage.GetPrimAtPath(dest_prim_path)
    if not dest_prim.IsValid():
        dest_prim = dest_stage.DefinePrim(dest_prim_path, source_prim.GetTypeName())
    for prop in source_prim.GetAuthoredProperties():
        if role is not None and prop.GetNamespace() != role:
            continue
        dest_prop = dest_prim.GetProperty(prop.GetName())
        if not dest_prop.IsValid():
            dest_prop = dest_prim.CreateProperty(prop.GetName(), prop.GetTypeName())
        dest_prop.Set(prop.Get())
    return dest_prim


def transfer_role_data(source_prim: Usd.Prim, dest_prim: Usd.Prim, role_name: Sdf.ValueRoleNames) -> bool:
    """Transfer role data from source prim to destination prim.

    Args:
        source_prim (Usd.Prim): The source prim to transfer data from.
        dest_prim (Usd.Prim): The destination prim to transfer data to.
        role_name (Sdf.ValueRoleNames): The role name to transfer.

    Returns:
        bool: True if the transfer was successful, False otherwise.
    """
    if not source_prim.IsValid() or not dest_prim.IsValid():
        return False
    source_attr = source_prim.GetAttribute(role_name)
    if not source_attr.IsValid():
        return False
    value = source_attr.Get()
    if value is None:
        return False
    dest_attr = dest_prim.CreateAttribute(role_name, source_attr.GetTypeName())
    success = dest_attr.Set(value)
    return success


def change_role_name(role_name: str, new_role_name: str) -> bool:
    """Change the name of a registered value role.

    Args:
        role_name (str): The current name of the value role.
        new_role_name (str): The new name for the value role.

    Returns:
        bool: True if the role name was successfully changed, False otherwise.
    """
    if not role_name in Sdf.ValueRoleNames.__dict__.values():
        return False
    if new_role_name in Sdf.ValueRoleNames.__dict__.values():
        return False
    role_value = getattr(Sdf.ValueRoleNames, role_name)
    setattr(Sdf.ValueRoleNames, new_role_name, role_value)
    delattr(Sdf.ValueRoleNames, role_name)
    return True


def set_prim_color_by_role(prim: Usd.Prim, role: str, color: Gf.Vec3f) -> None:
    """Set the displayColor of a prim based on the specified role.

    Args:
        prim (Usd.Prim): The prim to set the displayColor for.
        role (str): The role to use for setting the displayColor.
        color (Gf.Vec3f): The color value to set.

    Raises:
        ValueError: If the prim is not valid or if the role is not supported.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    gprim = UsdGeom.Gprim(prim)
    if not gprim:
        raise ValueError(f"Prim {prim.GetPath()} is not a Gprim")
    if role not in [Sdf.ValueRoleNames.Color, "color"]:
        raise ValueError(f"Unsupported role: {role}")
    color_attr = gprim.GetDisplayColorAttr()
    if not color_attr:
        raise ValueError(f"Failed to get displayColor attribute for role: {role}")
    color_attr.Set([color])


def unify_value_type_roles(value_type_roles: Dict[str, str]) -> Dict[str, str]:
    """
    Given a dictionary mapping value type names to roles, return a dictionary
    with the same keys but with the roles unified across all aliases of each
    value type name.

    If a value type name has no role assigned, it will not be included in the
    output dictionary.

    If two aliases for the same value type have conflicting roles, a coding
    error is raised.
    """
    unified_roles: Dict[Sdf.ValueTypeName, str] = {}
    for value_type_str, role in value_type_roles.items():
        value_type = Sdf.ValueTypeNames.Find(value_type_str)
        if not value_type:
            continue
        if value_type not in unified_roles:
            unified_roles[value_type] = role
        elif unified_roles[value_type] != role:
            raise Tf.ErrorException(
                f"Conflicting roles '{unified_roles[value_type]}' and '{role}' assigned to value type {value_type}"
            )
    result: Dict[str, str] = {}
    for value_type, role in unified_roles.items():
        for alias in value_type.aliasesAsStrings:
            result[alias] = role
    return result


def find_prims_by_value_type(stage: Usd.Stage, value_type: Sdf.ValueTypeName) -> List[Usd.Prim]:
    """
    Find all prims on the stage that have an attribute with the specified value type.

    Args:
        stage (Usd.Stage): The USD stage to search.
        value_type (Sdf.ValueTypeName): The value type to search for.

    Returns:
        List[Usd.Prim]: A list of prims that have an attribute with the specified value type.
    """
    prims: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() == value_type:
                prims.append(prim)
                break
    return prims


def summarize_stage_value_types(stage: Usd.Stage) -> Dict[str, int]:
    """
    Summarize the value types used in a USD stage.

    Args:
        stage (Usd.Stage): The USD stage to analyze.

    Returns:
        Dict[str, int]: A dictionary mapping value type names to their count.
    """
    value_type_counts: Dict[str, int] = {}
    for prim in stage.TraverseAll():
        for attr in prim.GetAttributes():
            type_name = attr.GetTypeName().cppTypeName
            value_type_counts[type_name] = value_type_counts.get(type_name, 0) + 1
    return value_type_counts


def analyze_value_type_composition(value_type: Sdf.ValueTypeName) -> Dict[str, Any]:
    """Analyze the composition of a value type and return a dictionary of its properties."""
    properties: Dict[str, Any] = {}
    if not value_type:
        properties["is_valid"] = False
        return properties
    properties["is_valid"] = True
    properties["type_name"] = str(value_type)
    properties["cpp_type_name"] = value_type.cppTypeName
    properties["role"] = value_type.role
    properties["tf_type"] = value_type.type
    properties["is_scalar"] = value_type.isScalar
    properties["is_array"] = value_type.isArray
    properties["scalar_type"] = str(value_type.scalarType)
    properties["array_type"] = str(value_type.arrayType)
    properties["default_value"] = value_type.defaultValue
    properties["default_unit"] = value_type.defaultUnit
    properties["aliases"] = value_type.aliasesAsStrings
    return properties


def convert_scalar_to_array(value_type: Sdf.ValueTypeName) -> Sdf.ValueTypeName:
    """Convert a scalar value type to its array equivalent.

    Args:
        value_type (Sdf.ValueTypeName): The scalar value type to convert.

    Returns:
        Sdf.ValueTypeName: The array equivalent of the input value type.
                           If the input is already an array type or is invalid,
                           the input value type is returned unchanged.
    """
    if value_type.isScalar:
        array_type = value_type.arrayType
        if array_type != Sdf.ValueTypeName():
            return array_type
    return value_type


def list_value_type_aliases(value_type_name: Sdf.ValueTypeName) -> List[str]:
    """Get the list of aliases for a given value type name."""
    if not value_type_name:
        raise ValueError("Invalid value type name.")
    aliases_tuple = value_type_name.aliasesAsStrings
    if not aliases_tuple:
        return []
    return list(aliases_tuple)


def transfer_default_values(src_attr: Usd.Attribute, dst_attr: Usd.Attribute):
    """Transfer default value from source attribute to destination attribute if possible."""
    if not src_attr.IsValid():
        raise ValueError("Source attribute is invalid.")
    if not dst_attr.IsValid():
        raise ValueError("Destination attribute is invalid.")
    if not src_attr.HasFallbackValue():
        return
    default_value = src_attr.Get()
    src_type_name = src_attr.GetTypeName()
    dst_type_name = dst_attr.GetTypeName()
    if src_type_name != dst_type_name:
        if not Sdf.ValueTypeRegistry.CanCastToTypeName(src_type_name, dst_type_name):
            return
        default_value = Sdf.ValueTypeRegistry.CastToTypeName(default_value, dst_type_name)
    dst_attr.Set(default_value)


def create_custom_attribute(
    prim: Usd.Prim, attribute_name: str, type_name: Sdf.ValueTypeName, default_value=None
) -> Usd.Attribute:
    """Create a custom attribute on a prim with a specific type and optional default value."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if prim.HasAttribute(attribute_name):
        raise ValueError(f"Attribute '{attribute_name}' already exists on prim {prim.GetPath()}")
    attribute = prim.CreateAttribute(attribute_name, type_name)
    if default_value is not None:
        attribute.Set(default_value)
    return attribute


def transfer_variability(source: Sdf.Variability, target: Sdf.Variability) -> Sdf.Variability:
    """Transfer variability from source to target.

    If the source variability is more varying than the target, the target will
    be upgraded to be at least as varying as the source.

    Args:
        source (Sdf.Variability): The source variability.
        target (Sdf.Variability): The target variability.

    Returns:
        Sdf.Variability: The resulting variability after the transfer.
    """
    if source == Sdf.VariabilityVarying:
        return Sdf.VariabilityVarying
    elif source == Sdf.VariabilityUniform and target == Sdf.VariabilityUniform:
        return Sdf.VariabilityUniform
    elif source == Sdf.VariabilityUniform and target == Sdf.VariabilityUniform:
        return Sdf.VariabilityUniform
    else:
        return target


def remove_variant_from_set(variant_set: Sdf.VariantSetSpec, variant_name: str) -> None:
    """Remove a variant from a variant set.

    Args:
        variant_set (Sdf.VariantSetSpec): The variant set to remove the variant from.
        variant_name (str): The name of the variant to remove.

    Raises:
        ValueError: If the variant set does not contain a variant with the specified name.
    """
    if variant_name not in variant_set.variants:
        raise ValueError(f"Variant '{variant_name}' does not exist in the variant set.")
    variant_set.RemoveVariant(variant_set.variants[variant_name])
    if variant_set.defaultVariantName == variant_name:
        variant_set.defaultVariantName = ""


def get_variant_set_variants(variant_set: Sdf.VariantSetSpec) -> List[str]:
    """Get a list of variant names in a variant set.

    Args:
        variant_set (Sdf.VariantSetSpec): The variant set to query.

    Returns:
        List[str]: A list of variant names in the variant set.
    """
    if not variant_set:
        raise ValueError("Invalid variant set.")
    variants = variant_set.variantList
    variant_names = [variant.name for variant in variants]
    return variant_names


def apply_variant_set(prim: Usd.Prim, variant_set_name: str, variant_name: str):
    """Apply a variant from a variant set to a prim.

    Args:
        prim (Usd.Prim): The prim to apply the variant to.
        variant_set_name (str): The name of the variant set.
        variant_name (str): The name of the variant to apply.

    Raises:
        ValueError: If the prim is not valid, the variant set does not exist,
            or the variant does not exist in the variant set.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    variant_set = prim.GetVariantSet(variant_set_name)
    if not variant_set.IsValid():
        raise ValueError(f"Variant set {variant_set_name} does not exist on prim {prim.GetPath()}.")
    if not variant_set.GetVariantNames().__contains__(variant_name):
        raise ValueError(
            f"Variant {variant_name} does not exist in variant set {variant_set_name} on prim {prim.GetPath()}."
        )
    variant_set.SetVariantSelection(variant_name)


def merge_variants(source_variant: Sdf.VariantSpec, target_variant: Sdf.VariantSpec) -> None:
    """Merge the contents of source_variant into target_variant."""
    target_prim_spec = target_variant.primSpec
    for prop_name, prop_spec in source_variant.primSpec.properties.items():
        if prop_name in target_prim_spec.properties:
            target_prop_spec = target_prim_spec.properties[prop_name]
            target_prop_spec.CopyFrom(prop_spec)
        else:
            target_prop_spec = Sdf.PropertySpec(target_prim_spec, prop_name, prop_spec.typeName)
            target_prop_spec.CopyFrom(prop_spec)
    for vset_name, source_vset in source_variant.variantSets.items():
        if vset_name in target_variant.variantSets:
            target_vset = target_variant.variantSets[vset_name]
        else:
            target_vset = Sdf.VariantSetSpec(target_prim_spec, vset_name)
        for v_name in source_vset.variants:
            source_v_spec = source_vset.variants[v_name]
            if v_name in target_vset.variants:
                target_v_spec = target_vset.variants[v_name]
                merge_variants(source_v_spec, target_v_spec)
            else:
                target_v_spec = Sdf.VariantSpec(target_vset, v_name)
                target_v_spec.CopyFrom(source_v_spec)


def switch_to_variant(variantSet: Sdf.VariantSetSpec, variant_name: str) -> None:
    """Switch the variant selection to the specified variant."""
    if not variantSet:
        raise ValueError("Invalid variant set")
    variants = variantSet.GetVariantNames()
    if variant_name not in variants:
        raise ValueError(f"Variant '{variant_name}' does not exist in the variant set")
    variantSet.SetVariantSelection(variant_name)
