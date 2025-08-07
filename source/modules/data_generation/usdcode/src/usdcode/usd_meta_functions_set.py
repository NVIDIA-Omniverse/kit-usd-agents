## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from math import atan2, cos, pi, sin, sqrt, ceil, degrees, asin
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import random
import bisect
import re

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

from .metafunction_modules.add_op import *
from .usd_meta_functions_get import *


def set_translate(stage: Usd.Stage, prim_path: str, translation: Tuple[float, float, float]) -> None:
    """Set the translation for a prim. Animation and keyframes are not supported. Don't set time codes here."""
    new_translation = Gf.Vec3d(*translation)

    if isinstance(prim_path, Usd.Prim):
        # Sometimes it passes a prim instead of a path
        prim = prim_path
    else:
        prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim at path '{prim_path}' does not exist.")

    xformable = UsdGeom.Xformable(prim)

    # Check if there's an existing translate xformOp
    xform_ops = xformable.GetOrderedXformOps()
    translate_op = None
    for op in xform_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break

    # If no existing translate xformOp, add one
    if not translate_op:
        translate_op = xformable.AddTranslateOp()

    # Set the new translation value
    translate_op.Set(new_translation)

    # Ensure the xformOp order is correct
    if not translate_op in xform_ops:
        new_order = xform_ops + [translate_op]
        xformable.SetXformOpOrder(new_order)


def set_scale(stage: Usd.Stage, prim_path: str, scale: Tuple[float, float, float]) -> None:
    """Set the scale for a prim. Animation and keyframes are not supported. Don't set time codes here."""
    new_scale = Gf.Vec3d(*scale)

    if isinstance(prim_path, Usd.Prim):
        # Sometimes it passes a prim instead of a path
        prim = prim_path
    else:
        prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim at path '{prim_path}' does not exist.")

    xformable = UsdGeom.Xformable(prim)

    # Check if there's an existing scale xformOp
    xform_ops = xformable.GetOrderedXformOps()
    scale_op = None
    for op in xform_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break

    # If no existing scale xformOp, add one
    if not scale_op:
        scale_op = xformable.AddScaleOp()

    # Set the new scale value
    scale_op.Set(new_scale)

    # Ensure the xformOp order is correct
    if not scale_op in xform_ops:
        new_order = xform_ops + [scale_op]
        xformable.SetXformOpOrder(new_order)


def set_rotate(stage: Usd.Stage, prim_path: str, rotation: Tuple[float, float, float]) -> None:
    """Set the rotation for a prim. Animation and keyframes are not supported. Don't set time codes here."""
    new_rotation = Gf.Vec3d(*rotation)

    if isinstance(prim_path, Usd.Prim):
        # Sometimes it passes a prim instead of a path
        prim = prim_path
    else:
        prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim at path '{prim_path}' does not exist.")

    xformable = UsdGeom.Xformable(prim)

    # Check if there's an existing rotate xformOp
    xform_ops = xformable.GetOrderedXformOps()
    rotate_op = None
    for op in xform_ops:
        if op.GetOpType() in [
            UsdGeom.XformOp.TypeRotateXYZ,
            UsdGeom.XformOp.TypeRotateXZY,
            UsdGeom.XformOp.TypeRotateYXZ,
            UsdGeom.XformOp.TypeRotateYZX,
            UsdGeom.XformOp.TypeRotateZXY,
            UsdGeom.XformOp.TypeRotateZYX,
        ]:
            rotate_op = op
            break

    # If no existing rotate xformOp, add one (default to XYZ order)
    if not rotate_op:
        rotate_op = xformable.AddRotateXYZOp()

    # Set the new rotation value
    rotate_op.Set(new_rotation)

    # Ensure the xformOp order is correct
    if not rotate_op in xform_ops:
        new_order = xform_ops + [rotate_op]
        xformable.SetXformOpOrder(new_order)


def rotate_around_axis(stage: Usd.Stage, prim_path: str, axis: Tuple[float, float, float], angle_deg: float):
    """
    Rotate a prim around an axis.

    Args:
        stage (Usd.Stage): The USD stage where the prim is located.
        prim_path (str): The path to the prim to rotate.
        axis (Tuple[float, float, float]): The axis to rotate around.
        angle_deg (float): The angle in degrees to rotate.
    """

    def get_rotation_matrix(x, y, z):
        x *= pi / 180.0
        y *= pi / 180.0
        z *= pi / 180.0

        result = Gf.Matrix4d(1.0)
        if x != 0.0:
            c = cos(x)
            s = sin(x)
            result = result * Gf.Matrix4d(1.0, 0.0, 0.0, 0.0, 0.0, c, s, 0.0, 0.0, -s, c, 0.0, 0.0, 0.0, 0.0, 1.0)
        if y != 0.0:
            c = cos(y)
            s = sin(y)
            result = result * Gf.Matrix4d(c, 0.0, -s, 0.0, 0.0, 1.0, 0.0, 0.0, s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0)
        if z != 0.0:
            c = cos(z)
            s = sin(z)
            result = result * Gf.Matrix4d(c, s, 0.0, 0.0, -s, c, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        return result

    if isinstance(prim_path, Usd.Prim):
        # Sometimes it passes a prim instead of a path
        prim = prim_path
    else:
        prim = stage.GetPrimAtPath(prim_path)

    begin_rotation = get_rotate(stage, prim)
    begin_rotation_matrix = get_rotation_matrix(begin_rotation[0], begin_rotation[1], begin_rotation[2])

    # convert to radian
    angle = angle_deg / 180.0 * pi

    # calculate rotation matrix around axis for angle in radian
    matrix = [
        cos(angle) + axis[0] ** 2 * (1 - cos(angle)),
        axis[0] * axis[1] * (1 - cos(angle)) + axis[2] * sin(angle),
        axis[0] * axis[2] * (1 - cos(angle)) - axis[1] * sin(angle),
        0,
    ]
    matrix += [
        axis[0] * axis[1] * (1 - cos(angle)) - axis[2] * sin(angle),
        cos(angle) + axis[1] ** 2 * (1 - cos(angle)),
        axis[1] * axis[2] * (1 - cos(angle)) + axis[0] * sin(angle),
        0,
    ]
    matrix += [
        axis[0] * axis[2] * (1 - cos(angle)) + axis[1] * sin(angle),
        axis[1] * axis[2] * (1 - cos(angle)) - axis[0] * sin(angle),
        cos(angle) + axis[2] ** 2 * (1 - cos(angle)),
        0,
    ]

    matrix += [0, 0, 0, 1]
    matrix = Gf.Matrix4d(*matrix)

    rotate = begin_rotation_matrix * matrix

    # decompose back to x y z euler angle
    sy = sqrt(rotate[0][0] ** 2 + rotate[0][1] ** 2)
    is_singular = sy < 10**-6
    if not is_singular:
        z = atan2(rotate[0][1], rotate[0][0])
        y = atan2(-rotate[0][2], sy)
        x = atan2(rotate[1][2], rotate[2][2])
    else:
        z = atan2(-rotate[2][1], rotate[1][1])
        y = atan2(-rotate[0][2], sy)
        x = 0

    x = x / pi * 180.0
    y = y / pi * 180.0
    z = z / pi * 180.0

    set_rotate(stage, prim, (x, y, z))


def get_next_free_path(stage: Usd.Stage, path: Union[str, Sdf.Path], prepend_default_prim: bool = True) -> str:
    """
    Internal helper to generate unique prim paths by incrementing a numerical suffix until a free path is found.

    This function ensures that the generated path:
    1. Is a valid USD path
    2. Has a proper absolute path format
    3. Respects the stage's default prim if specified
    4. Does not conflict with existing prims on the stage

    Args:
        stage (Usd.Stage): The USD stage where the new prim path will be created.
        path (Union[str, Sdf.Path]): The initial path for the new prim. If it's a string,
            it should be a valid USD path.
        prepend_default_prim (bool, optional): If True and the stage has a default prim,
            the default prim's path will be prepended to the generated path if the provided
            path is not already under the default prim. Defaults to True.

    Returns:
        str: A unique path string that does not correspond to an existing prim on the stage.
            If the initial path exists, a numerical suffix will be added or incremented
            (e.g., "path" -> "path_01" -> "path_02").

    Raises:
        ValueError: If the provided path is not a valid USD path string.

    Example:
        >>> get_next_free_path(stage, "/Sphere")
        '/Sphere'
        >>> # After creating a prim at /Sphere
        >>> get_next_free_path(stage, "/Sphere")
        '/Sphere_01'
    """
    # Convert input path to string if it's a Sdf.Path
    path_str = str(path)

    if not Sdf.Path.IsValidPathString(path_str):
        raise ValueError(f"{path_str} is not a valid path")

    # Ensure absolute path
    path_sdf = Sdf.Path(path_str).MakeAbsolutePath(Sdf.Path.absoluteRootPath)
    path_str = str(path_sdf)

    # Handle default prim prefix
    if prepend_default_prim and stage.GetDefaultPrim():
        default_prim = stage.GetDefaultPrim()
        if default_prim and not (
            Sdf.Path(path_str).HasPrefix(default_prim.GetPath()) and path_str != str(default_prim.GetPath())
        ):
            path_str = str(Sdf.Path(path_str).ReplacePrefix(Sdf.Path.absoluteRootPath, default_prim.GetPath()))

    def increment_path(p: str) -> str:
        match = re.search(r"_(\d+)$", p)
        if match:
            new_num = int(match.group(1)) + 1
            return re.sub(r"_(\d+)$", f"_{new_num:02d}", p)
        return p + "_01"

    while stage.GetPrimAtPath(path_str):
        path_str = increment_path(path_str)

    return path_str


def create_prim(
    stage: Usd.Stage, prim_type: str, prim_path: Optional[str] = None, **attributes: Dict[str, Any]
) -> Usd.Prim:
    """
    Create a primitive on the stage. Always use it to create a new prim. It automatically orients the prim based on the stage up axis.

    Args:
        stage (Usd.Stage): The USD stage to create the prim on.
        prim_type (str): The type of the prim to create.
        prim_path (Optional[str]): The path where the prim should be created.
        **attributes: Keyword arguments for prim attributes.

    Returns:
        Usd.Prim: The created prim.
    """

    def _create_default_xform(prim: Usd.Prim):
        xformable = UsdGeom.Xformable(prim)
        xform_ops = xformable.GetOrderedXformOps()
        op_types = {op.GetOpType() for op in xform_ops}
        if UsdGeom.XformOp.TypeTranslate not in op_types:
            xformable.AddTranslateOp()
        if UsdGeom.XformOp.TypeRotateXYZ not in op_types:
            xformable.AddRotateXYZOp()
        if UsdGeom.XformOp.TypeScale not in op_types:
            xformable.AddScaleOp()

    def _create_light_extra(prim: Usd.Prim):
        if (hasattr(UsdLux, "LightAPI") and prim.HasAPI(UsdLux.LightAPI)) or (
            hasattr(UsdLux, "Light") and prim.IsA(UsdLux.Light)
        ):
            light_api = UsdLux.ShapingAPI.Apply(prim)
            light_api.CreateShapingConeAngleAttr(180)
            light_api.CreateShapingConeSoftnessAttr()
            light_api.CreateShapingFocusAttr()
            light_api.CreateShapingFocusTintAttr()
            light_api.CreateShapingIesFileAttr()

            up_axis = UsdGeom.GetStageUpAxis(stage)
            if up_axis == "Z":
                set_rotate(stage, prim.GetPath().pathString, (90, 0, 90))

    def _set_refinement_level(prim: Usd.Prim):
        if (
            prim.IsA(UsdGeom.Cylinder)
            or prim.IsA(UsdGeom.Capsule)
            or prim.IsA(UsdGeom.Cone)
            or prim.IsA(UsdGeom.Sphere)
        ):
            prim.CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool, True).Set(True)
            prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int, True).Set(2)

    if not prim_path:
        prim_path = get_next_free_path(stage, f"/{prim_type}", True)

    # Ensure parent prims exist
    parent_path = Sdf.Path(prim_path).GetParentPath()
    if not parent_path.IsAbsoluteRootPath() and not stage.GetPrimAtPath(parent_path):
        stage.DefinePrim(parent_path, "Xform")

    # Create the prim
    prim = stage.DefinePrim(prim_path, prim_type)

    # Set attributes
    with Sdf.ChangeBlock():
        for attr, value in attributes.items():
            value_type = Sdf.ValueTypeNames.Find(type(value).__name__)
            if value_type:
                prim.CreateAttribute(attr, value_type).Set(value)
            else:
                raise ValueError(f"Unsupported attribute type: {type(value).__name__}")

    # Handle specific prim types
    if prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule) or prim.IsA(UsdGeom.Cone):
        prim.GetAttribute("axis").Set(UsdGeom.GetStageUpAxis(stage))

    if prim.IsA(UsdGeom.Xformable):
        _create_default_xform(prim)

    # Set extent if not already provided
    if "extent" not in attributes:
        extent_attr = prim.GetAttribute("extent")
        if extent_attr:
            bounds = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim), Usd.TimeCode.Default())
            if bounds is not None:
                extent_attr.Set(bounds)

    _create_light_extra(prim)
    _set_refinement_level(prim)

    return prim


def copy_prim(stage: Usd.Stage, path_from: str, path_to: Optional[str] = None) -> Usd.Prim:
    """Copy a primitive to a new path."""
    source_prim = stage.GetPrimAtPath(path_from)
    if not source_prim:
        raise ValueError(f"Source prim {path_from} not found")

    if not path_to:
        path_to = Sdf.Path(path_from).GetParentPath().AppendChild(Sdf.Path(path_from).name + "_copy")
        while stage.GetPrimAtPath(path_to):
            path_to = Sdf.Path(path_to).GetParentPath().AppendChild(Sdf.Path(path_to).name + "_copy")

    # Get the layer stack in strength order
    layer_stack = stage.GetLayerStack()

    # Find the layer that contains the source prim
    source_layer = None
    for layer in layer_stack:
        if layer.GetPrimAtPath(path_from):
            source_layer = layer
            break

    if not source_layer:
        raise ValueError(f"Could not find source prim {path_from} in any layer")

    # Get the current edit target
    edit_target = stage.GetEditTarget()
    edit_layer = edit_target.GetLayer()

    # Create the new prim in the edit layer
    target_prim = create_prim(stage, source_prim.GetTypeName(), path_to)
    path_to = target_prim.GetPath()
    # Sdf.CreatePrimInLayer(edit_layer, path_to)

    # Copy the spec from the source layer to the edit layer
    Sdf.CopySpec(source_layer, path_from, edit_layer, path_to)

    return target_prim


def change_property(stage: Usd.Stage, prop_path: str, value: Any) -> Union[Usd.Attribute, Usd.Relationship, None]:
    """Change the value of a property."""
    sdf_path = Sdf.Path(prop_path)
    prim_path = sdf_path.GetPrimPath()
    prop_name = sdf_path.name

    if not prim_path or not prop_name:
        raise ValueError(f"Invalid property path: {prop_path}")

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return None

    prop = prim.GetProperty(prop_name)
    if not prop:
        return None

    if isinstance(prop, Usd.Attribute):
        if value is None:
            prop.Clear()
        else:
            prop.Set(value)
    elif isinstance(prop, Usd.Relationship):
        if value is None:
            prop.ClearTargets(True)
        else:
            prop.SetTargets(value)
    else:
        return None

    return prop


def create_usd_attribute(
    stage: Usd.Stage,
    prim_path: str,
    attr_name: str,
    attr_type: Sdf.ValueTypeName,
    attr_value: Optional[Any] = None,
    custom: bool = True,
    variability: Sdf.Variability = Sdf.VariabilityVarying,
) -> Optional[Usd.Attribute]:
    """Create a new USD attribute for a primitive."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return None

    attr = prim.CreateAttribute(name=attr_name, typeName=attr_type, custom=custom, variability=variability)

    if attr.IsValid():
        if attr_value is not None:
            attr.Set(attr_value)
        return attr

    return None


def add_reference(
    stage: Usd.Stage,
    prim_path: str,
    asset_path: str,
    reference_prim_path: Optional[Union[str, Sdf.Path]] = None,
    position: Union[Usd.ListPosition, str] = Usd.ListPositionBackOfPrependList,
) -> None:
    """
    Add a reference to a prim.

    Args:
        stage (Usd.Stage): The USD stage containing the prim.
        prim_path (str): The path to the prim to add the reference to.
        asset_path (str): The path to the asset to reference.
        reference_prim_path (Optional[Union[str, Sdf.Path]]): The prim path within the asset to reference.
        position (Union[Usd.ListPosition, str]): Where to add the reference in the list of references.
            Can be a Usd.ListPosition or a string ('frontOfPrependList', 'backOfPrependList',
            'frontOfAppendList', 'backOfAppendList').
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim at path '{prim_path}' does not exist.")

    # Handle position argument
    if isinstance(position, str):
        position_mapping = {
            "frontOfPrependList": Usd.ListPositionFrontOfPrependList,
            "backOfPrependList": Usd.ListPositionBackOfPrependList,
            "frontOfAppendList": Usd.ListPositionFrontOfAppendList,
            "backOfAppendList": Usd.ListPositionBackOfAppendList,
        }
        position_value = position_mapping.get(position)
        if position_value is None:
            raise ValueError(
                f"Invalid position string: '{position}'. Valid options are "
                "'frontOfPrependList', 'backOfPrependList', 'frontOfAppendList', 'backOfAppendList'."
            )
        position = position_value
    elif not isinstance(position, Usd.ListPosition):
        raise TypeError("Position must be a Usd.ListPosition or a valid position string.")

    reference = Sdf.Reference(asset_path, reference_prim_path)

    references = prim.GetReferences()
    if references:
        references.AddReference(reference, position)


def create_reference(
    stage: Usd.Stage,
    prim_path: str,
    asset_path: str,
    reference_prim_path: Optional[Union[str, Sdf.Path]] = None,
    instanceable: bool = False,
) -> Usd.Prim:
    """
    Create a new prim at the given path that references the specified asset.

    Fast way to create the reference in the stage:
    ```python
    usdcode.create_reference(stage, "/World/Target", "", "/World/Source")
    usdcode.set_translate(stage, "/World/Target", (60, 0 ,0))
    ```

    Args:
        stage (Usd.Stage): The USD stage to create the prim on.
        prim_path (str): The path where the prim should be created.
        asset_path (str): The path to the asset to reference.
        reference_prim_path (Optional[Union[str, Sdf.Path]]): The prim path within the asset to reference.
        instanceable (bool): Whether the prim should be instanceable.

    Returns:
        Usd.Prim: The created prim.
    """
    # Ensure parent prims exist
    parent_path = Sdf.Path(prim_path).GetParentPath()
    if not stage.GetPrimAtPath(parent_path):
        stage.DefinePrim(parent_path, "Xform")

    # Create the prim
    prim = stage.DefinePrim(prim_path)
    if not prim:
        raise ValueError(f"Failed to create prim at '{prim_path}'.")

    # Create the reference
    if reference_prim_path:
        reference = Sdf.Reference(asset_path, reference_prim_path)
    else:
        reference = Sdf.Reference(asset_path)
    prim.GetReferences().AddReference(reference)

    # Set instanceable if needed
    prim.SetInstanceable(instanceable)

    return prim


def create_payload(
    stage: Usd.Stage,
    prim_path: str,
    asset_path: str,
    asset_prim_path: Optional[str] = None,
    instanceable: bool = True
) -> Usd.Prim:
    """
    Create a new prim at the given path that references the specified asset as a payload.

    Payloads are similar to references but are loaded on demand, which can improve performance
    for large scenes. This function creates a prim and sets up a payload to the specified asset.

    Args:
        stage (Usd.Stage): The USD stage to create the prim on.
        prim_path (str): The path where the prim should be created.
        asset_path (str): The path to the asset to reference as a payload.
        asset_prim_path (Optional[str]): The prim path within the asset to reference.
        instanceable (bool): Whether the prim should be instanceable. Default is True.

    Returns:
        Usd.Prim: The created prim.
    """

    def _ensure_parents_exist(stage: Usd.Stage, path: Sdf.Path):
        """Ensure that all parent prims exist."""
        if isinstance(path, str):
            path = Sdf.Path(path)

        if not path or path == Sdf.Path.absoluteRootPath:
            return

        _ensure_parents_exist(stage, path.GetParentPath())

        if not stage.GetPrimAtPath(path):
            stage.DefinePrim(path, "Xform")

    def _create_default_xform(prim: Usd.Prim):
        xformable = UsdGeom.Xformable(prim)
        xformable.AddTranslateOp()
        xformable.AddRotateXYZOp()
        xformable.AddScaleOp()

    # Ensure parent prims exist
    parent_path = Sdf.Path(prim_path).GetParentPath()
    _ensure_parents_exist(stage, parent_path)

    # Check if the prim already exists
    prim = stage.GetPrimAtPath(prim_path)
    if prim:
        # If the prim exists and already has a payload, don't modify it
        if prim.HasPayload():
            print(f"Prim at '{prim_path}' already exists with a payload. Skipping creation.")
            return prim
    else:
        # Create the prim if it doesn't exist
        prim = stage.DefinePrim(prim_path, "Xform")

    if not prim:
        raise ValueError(f"Failed to create or get prim at '{prim_path}'")

    if prim.IsA(UsdGeom.Xformable):
        _create_default_xform(prim)

    # Set the payload
    if asset_prim_path:
        payload = Sdf.Payload(asset_path, asset_prim_path)
    else:
        payload = Sdf.Payload(asset_path)
    prim.SetPayload(payload)

    # Check if the asset has a default prim that is an Xform
    default_prim_is_xform = False
    ref_stage = None
    if asset_path:
        anonymous_layer = Sdf.Layer.CreateAnonymous()
        asset_layer = Sdf.Layer.FindOrOpen(asset_path)
        if asset_layer:
            ref_stage = Usd.Stage.Open(asset_layer, anonymous_layer)
            default_prim = ref_stage.GetDefaultPrim()
            if default_prim and default_prim.IsA(UsdGeom.Xform):
                default_prim_is_xform = True
            elif not default_prim:
                # print("The payload doesn't have a default prim set, which means that some prims may be missing.")
                pass

    # Set instanceable if needed
    if instanceable and default_prim_is_xform:
        prim.SetInstanceable(True)

    # Handle rotation if needed
    if default_prim_is_xform and ref_stage:
        curr_up = UsdGeom.GetStageUpAxis(stage)
        ref_up = UsdGeom.GetStageUpAxis(ref_stage)

        if ref_up != curr_up:
            if ref_up == "Y" and curr_up == "Z":
                set_rotate(stage, prim_path, (90, 0, 90))
            elif ref_up == "Z" and curr_up == "Y":
                set_rotate(stage, prim_path, (-90, -90, 0))

    return prim


def remove_prim(stage: Usd.Stage, prim_path: str) -> None:
    """Remove a prim from the stage."""
    path = Sdf.Path(prim_path)

    if path == Sdf.Path.absoluteRootPath:
        raise ValueError("Cannot remove root path")

    prim = stage.GetPrimAtPath(path)
    if not prim:
        return

    # Get all layers where this prim might exist
    layer_stack = stage.GetLayerStack(includeSessionLayers=True)

    edit_target = stage.GetEditTarget()
    with Sdf.ChangeBlock():
        # Remove from all layers in the stack
        for layer in layer_stack:
            stage.SetEditTarget(Usd.EditTarget(layer))
            if prim.IsActive():
                prim.SetActive(False)
            stage.RemovePrim(path)

    # Restore original edit target
    stage.SetEditTarget(edit_target)


def create_material(stage: Usd.Stage, material_path: str, **shader_params: Dict[str, Any]) -> UsdShade.Material:
    """
    Create a new MDL Material with customizable shader parameters using OmniSurface.

    Args:
        stage (Usd.Stage): The USD stage to create the material on.
        material_path (str): The path where the material should be created.
        **shader_params: Keyword arguments for shader parameters.

    Possible shader parameters:
        coat_weight (float): Coat layer weight. Range: [0, 1]
        coat_roughness (float): Coat layer roughness. Range: [0, 1]
        diffuse_reflection_color (tuple): Diffuse reflection color. Format: (r, g, b)
        displacement (tuple): Displacement amount. Format: (x, y, z)
        emission_color (tuple): Emission color. Format: (r, g, b)
        ior (float): Index of refraction.
        metalness (float): Metalness amount. Range: [0, 1]
        normal (tuple): Normal vector. Format: (x, y, z)
        opacity (float): Opacity. Range: [0, 1]
        roughness (float): Surface roughness. Range: [0, 1]
        specular_reflection_color (tuple): Specular reflection color. Format: (r, g, b)

    Returns:
        UsdShade.Material: The created material.
    """
    try:
        import omni.usd

        use_mdl = True
    except ImportError:
        use_mdl = False

    # Helper function to create parent prims
    def _create_parent_prims(stage: Usd.Stage, path: Sdf.Path):
        """Recursively create parent prims if they don't exist."""
        if not path or path == Sdf.Path.absoluteRootPath:
            return

        _create_parent_prims(stage, path.GetParentPath())

        if not stage.GetPrimAtPath(path):
            stage.DefinePrim(path, "Scope")

    # Ensure the parent prims exist
    parent_path = Sdf.Path(material_path).GetParentPath()
    _create_parent_prims(stage, parent_path)

    # Generate a unique material path if the given material_path is already taken
    material_path = get_next_free_path(stage, material_path, False)

    # Create the material prim
    mat_prim = stage.DefinePrim(material_path, "Material")
    material = UsdShade.Material.Get(stage, mat_prim.GetPath())

    if use_mdl:
        # Create the shader prim
        shader_path = f"{material_path}/Shader"
        shader_prim = stage.DefinePrim(shader_path, "Shader")
        shader = UsdShade.Shader.Get(stage, shader_prim.GetPath())

        if not shader:
            raise RuntimeError(f"Failed to create shader at {shader_path}")

        # Set MDL attributes
        shader_out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        shader_out.SetRenderType("material")
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader_out)
        material.CreateVolumeOutput("mdl").ConnectToSource(shader_out)
        material.CreateDisplacementOutput("mdl").ConnectToSource(shader_out)
        shader.CreateIdAttr("OmniSurface")

        shader.GetImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath("OmniSurface.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniSurface", "mdl")

        # Default shader parameter values
        default_params = {
            "coat_weight": 0.0,
            "coat_roughness": 0.01,
            "diffuse_reflection_color": (0.18, 0.18, 0.18),
            "displacement": (0.0, 0.0, 0.0),
            "emission_color": (0, 0, 0),
            "ior": 1.5,
            "metalness": 0.0,
            "normal": (0, 0, 1),
            "opacity": 1.0,
            "roughness": 0.5,
            "specular_reflection_color": (0, 0, 0),
        }

        # Set up the shader inputs with default values or provided values
        for param, default_value in default_params.items():
            value = shader_params.get(param, default_value)
            value_type = Sdf.ValueTypeNames.Float if isinstance(value, float) else Sdf.ValueTypeNames.Color3f
            shader.CreateInput(param, value_type).Set(value)

    else:
        # Create UsdPreviewSurface shader
        shader_path = f"{material_path}/PreviewSurface"
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")

        # Create outputs and connections
        surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(surface_output)

        # Default shader parameter values (keeping the same as MDL for consistency)
        default_params = {
            "coat_weight": 0.0,
            "coat_roughness": 0.01,
            "diffuse_reflection_color": (0.18, 0.18, 0.18),
            "displacement": (0.0, 0.0, 0.0),
            "emission_color": (0, 0, 0),
            "ior": 1.5,
            "metalness": 0.0,
            "normal": (0, 0, 1),
            "opacity": 1.0,
            "roughness": 0.5,
            "specular_reflection_color": (0, 0, 0),
        }

        # Map parameters to UsdPreviewSurface inputs
        param_mapping = {
            "diffuse_reflection_color": ("diffuseColor", Sdf.ValueTypeNames.Color3f),
            "emission_color": ("emissiveColor", Sdf.ValueTypeNames.Color3f),
            "metalness": ("metallic", Sdf.ValueTypeNames.Float),
            "opacity": ("opacity", Sdf.ValueTypeNames.Float),
            "roughness": ("roughness", Sdf.ValueTypeNames.Float),
            "ior": ("ior", Sdf.ValueTypeNames.Float),
            "normal": ("normal", Sdf.ValueTypeNames.Normal3f),
        }

        # Set up the shader inputs with default values
        used_params = set()
        for param, default_value in default_params.items():
            if param not in shader_params:
                value = default_value
                if param in param_mapping:
                    preview_name, value_type = param_mapping[param]
                    shader.CreateInput(preview_name, value_type).Set(value)
                    used_params.add(preview_name)

        # Handle all provided parameters
        for param, value in shader_params.items():
            value_type = Sdf.ValueTypeNames.Float if isinstance(value, float) else Sdf.ValueTypeNames.Color3f
            if param in param_mapping:
                preview_name, value_type = param_mapping[param]
                if preview_name not in used_params:
                    shader.CreateInput(preview_name, value_type).Set(value)
                    used_params.add(preview_name)
            else:
                if param not in used_params:
                    shader.CreateInput(param, value_type).Set(value)
                    used_params.add(param)

    return material


def assign_material(stage: Usd.Stage, prim_path: str, material_path: str):
    """Assign a material to a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        print(f"Prim not found at path: {prim_path}")
        return

    material = UsdShade.Material.Get(stage, material_path)
    if not material:
        print(f"Material not found at path: {material_path}")
        return

    # Apply the MaterialBindingAPI to the prim
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)

    # Bind the material
    binding_api.Bind(material, bindingStrength=UsdShade.Tokens.weakerThanDescendants)


def align_objects(
    stage: Usd.Stage,
    source_path: str,
    target_path: str,
    axes: List[str] = ["x", "y", "z"],
    alignment_type: str = "min_to_min",
    purpose: List[str] = ["default"],
) -> None:
    """
    Align one object to another along specified axes.

    This function doesn't stack objects and it doesn't consider the intersections. It only aligns the objects based on their bounding boxes.

    To stack objects, use the `stack_objects` function.

    Args:
        source_path (str): Path of the object to be aligned.
        target_path (str): Path of the object to align to.
        axes (List[str]): List of axes to align. Default is ['x', 'y', 'z'].
        alignment_type (str): Type of alignment. Options are:
            'min_to_min', 'max_to_max', 'center_to_center',
            'min_to_max', 'max_to_min', 'min_to_center', 'max_to_center'.
            Default is 'min_to_min'.
        purpose (List[str]): List of purposes to include when computing bounds.
            Default is ["default"].
    """
    source_prim = stage.GetPrimAtPath(source_path)
    target_prim = stage.GetPrimAtPath(target_path)

    if not source_prim or not target_prim:
        raise ValueError("Source or target prim not found.")

    # Compute bounding boxes with specified purposes
    source_bbox = get_bbox_world(stage, source_path)
    target_bbox = get_bbox_world(stage, target_path)

    source_min = source_bbox.GetMin()
    source_max = source_bbox.GetMax()
    target_min = target_bbox.GetMin()
    target_max = target_bbox.GetMax()

    source_center = (source_min + source_max) / 2
    target_center = (target_min + target_max) / 2

    translation = Gf.Vec3d(0, 0, 0)

    axis_map = {"x": 0, "y": 1, "z": 2}

    for axis in axes:
        if axis not in axis_map:
            raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'.")

        i = axis_map[axis]

        if alignment_type == "min_to_min":
            translation[i] = target_min[i] - source_min[i]
        elif alignment_type == "max_to_max":
            translation[i] = target_max[i] - source_max[i]
        elif alignment_type == "center_to_center":
            translation[i] = target_center[i] - source_center[i]
        elif alignment_type == "min_to_max":
            translation[i] = target_max[i] - source_min[i]
        elif alignment_type == "max_to_min":
            translation[i] = target_min[i] - source_max[i]
        elif alignment_type == "min_to_center":
            translation[i] = target_center[i] - source_min[i]
        elif alignment_type == "max_to_center":
            translation[i] = target_center[i] - source_max[i]
        else:
            raise ValueError(f"Invalid alignment type: {alignment_type}")

    # Apply the translation
    current_translation = get_translate(stage, source_path)
    new_translation = (
        current_translation[0] + translation[0],
        current_translation[1] + translation[1],
        current_translation[2] + translation[2],
    )
    set_translate(stage, source_path, new_translation)


def set_selection(selected_prim_paths: List[str]):
    try:
        import omni.usd
    except ImportError:
        raise ImportError("omni.usd is not installed. Selection is not supported.")

    usd_context = omni.usd.get_context()
    selection = usd_context.get_selection()
    selection.set_selected_prim_paths([str(s) for s in selected_prim_paths])


def scatter_prims(
    stage: Usd.Stage,
    point_instancer_path: str,
    prototype_path: str,
    surface_path: str,
    num_instances: int,
    random_seed: Optional[int] = None,
    scale: Union[float, Tuple[float, float]] = 1.0,
    rotation: Union[float, Tuple[float, float]] = 0.0,
    translation: Union[
        Tuple[float, float, float],
        Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    ] = (0.0, 0.0, 0.0),
    align_to_normals: float = 1.0,
    max_normal_deviation_angle: float = 360.0,
    **kwargs,
) -> Optional[Usd.Prim]:
    """
    Scatter prototypes over a surface using a PointInstancer.

    This function randomly places instances of a prototype prim onto the geometry of a given surface prim.
    The scattering is controlled by parameters such as number of instances, scale, rotation, translation,
    and alignment to the surface normals.

    Always use this function to randomly place any number of instances on a surface.
    Never copy the same prim multiple times to randomly place them on a surface.

    Args:
        stage (Usd.Stage): The USD stage where the PointInstancer will be created.
        point_instancer_path (str): The path in the stage where the PointInstancer prim will be created.
        prototype_path (str): The path to the prototype prim to be instanced.
        surface_path (str): The path to the surface prim where instances will be scattered.
        num_instances (int): The number of instances to scatter.
        random_seed (Optional[int]): An optional seed for the random number generator.
        scale (float or Tuple[float, float]): Uniform scale or a (min, max) tuple for random scaling.
        rotation (float or Tuple[float, float]): Rotation in degrees around the UP axis, or a (min, max) tuple for random rotation.
        translation (Tuple[float, float, float] or Tuple[Tuple[float, float, float], Tuple[float, float, float]]): Fixed translation or a range ((xmin, ymin, zmin), (xmax, ymax, zmax)) for random translation.
        align_to_normals (float): A value between 0.0 and 1.0 indicating how much to align instances to the surface normals.
        max_normal_deviation_angle (float): Maximum allowed angle in degrees between surface normal and up vector.
            Instances with normals deviating more than this angle will be skipped. Default is 360.0.
            Use 5.0 to place objects only on horizontal surfaces.
        **kwargs: Additional keyword arguments.

    Returns:
        Optional[Usd.Prim]: The created PointInstancer prim if successful, or None if failed.

    Example:
        ```python
        # Paths to the prototype, surface, and where to create the PointInstancer
        prototype_path = "/World/Cube"
        surface_path = "/World/Plane"
        point_instancer_path = "/World/CubeInstances"

        # Scatter 100 instances of the prototype onto the surface
        usdcode.scatter_prims(
            stage=stage,
            point_instancer_path=point_instancer_path,
            prototype_path=prototype_path,
            surface_path=surface_path,
            num_instances=100,
            random_seed=42,
            scale=(0.5, 1.5),
            rotation=(0, 360),
            translation=((0, 0, 0), (0, 0.25, 0)),
            align_to_normals=1.0,
            max_normal_deviation_angle=45.0
        )
        ```
    """

    # -------------- Helper Vector Math --------------
    def vec_add(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def vec_sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def vec_scale(a, s):
        return (a[0] * s, a[1] * s, a[2] * s)

    def vec_dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def vec_cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def vec_length(a):
        return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

    def vec_normalize(a):
        length = vec_length(a)
        if abs(length) < 1e-12:
            return (0.0, 0.0, 0.0)
        return (a[0] / length, a[1] / length, a[2] / length)

    # -------------- Helper Quaternion Logic --------------
    def quat_slerp(q1, q2, t):
        """Spherical linear interpolation between two Gf.Quatf quaternions."""
        dot_val = Gf.Dot(q1.GetImaginary(), q2.GetImaginary()) + q1.GetReal() * q2.GetReal()
        if dot_val < 0.0:
            q2 = -q2
            dot_val = -dot_val

        DOT_THRESHOLD = 0.9995
        if dot_val > DOT_THRESHOLD:
            # Linear interpolation
            result = Gf.Quatf(
                q1.GetReal() + t * (q2.GetReal() - q1.GetReal()),
                q1.GetImaginary() + t * (q2.GetImaginary() - q1.GetImaginary()),
            )
            result.Normalize()
            return result

        theta_0 = math.acos(dot_val)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)

        s0 = math.cos(theta) - dot_val * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        result = (q1 * s0) + (q2 * s1)
        result.Normalize()
        return result

    def quat_from_two_vectors(v0, v1):
        """Compute quaternion that rotates v0 to v1."""
        n0 = vec_normalize(v0)
        n1 = vec_normalize(v1)

        dot_v = vec_dot(n0, n1)
        if dot_v >= 1.0:
            # Vectors are the same
            return Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        elif dot_v <= -1.0:
            # Vectors are opposite; find an orthogonal axis
            axis = vec_cross((1.0, 0.0, 0.0), n0)
            if vec_length(axis) < 1e-6:
                axis = vec_cross((0.0, 1.0, 0.0), n0)
            axis = vec_normalize(axis)
            return Gf.Quatf(0.0, axis[0], axis[1], axis[2])
        else:
            s = math.sqrt((1.0 + dot_v) * 2.0)
            invs = 1.0 / s
            c = vec_cross(n0, n1)
            quat = Gf.Quatf(
                s * 0.5,
                c[0] * invs,
                c[1] * invs,
                c[2] * invs,
            )
            quat.Normalize()
            return quat

    # -------------- Prototype Flattening --------------
    def create_flattened_prototype(stage: Usd.Stage, prototype_path: str) -> Optional[str]:
        """
        Create a single flattened prototype at the origin with zeroed transforms,
        marked as instanceable and referencing the original prototype.
        Place the flattened prototype under the '/Prototypes' scope.
        """
        proto_prim = stage.GetPrimAtPath(prototype_path)
        if not proto_prim:
            print(f"Prototype prim not found at path: {prototype_path}")
            return None

        # Create '/Prototypes' scope if it doesn't exist
        prototypes_scope = stage.GetPrimAtPath("/Prototypes")
        if not prototypes_scope:
            prototypes_scope = UsdGeom.Scope.Define(stage, "/Prototypes")
            UsdGeom.Imageable(prototypes_scope).MakeInvisible()

        # Define the flattened prototype path under '/Prototypes'
        proto_name = prototype_path.strip("/").split("/")[-1]
        flattened_proto_path = f"/Prototypes/{proto_name}"

        # Remove any existing prim at the flattened proto path
        stage.RemovePrim(flattened_proto_path)

        if proto_prim.IsA(UsdGeom.Xform):
            # The prototype is an Xform; define the flattened prim as an Xform
            flattened_prim = UsdGeom.Xform.Define(stage, flattened_proto_path)
            flattened_prim.GetPrim().GetReferences().AddReference(stage.GetRootLayer().identifier, prototype_path)
        else:
            # The prototype is not an Xform; wrap it in an Xform
            flattened_prim = UsdGeom.Xform.Define(stage, flattened_proto_path)
            proto_type_name = proto_prim.GetTypeName()
            child_prim_path = flattened_proto_path + "/" + proto_name
            child_prim = stage.DefinePrim(child_prim_path, proto_type_name)
            child_prim.GetReferences().AddReference(stage.GetRootLayer().identifier, prototype_path)
            # Zero out transforms on the child prim if Xformable
            child_xformable = UsdGeom.Xformable(child_prim)
            if child_xformable:
                child_xformable.ClearXformOpOrder()

                translate_op = add_translate_op(child_xformable)
                if translate_op:
                    translate_op.Set(get_vec3_type_for_presision(translate_op)(0, 0, 0))
                rotate_op = add_rotate_xyz_op(child_xformable)
                if rotate_op:
                    rotate_op.Set(get_vec3_type_for_presision(rotate_op)(0, 0, 0))
                scale_op = add_scale_op(child_xformable)
                if scale_op:
                    scale_op.Set(get_vec3_type_for_presision(scale_op)(1, 1, 1))
                xforms = [op for op in [translate_op, rotate_op, scale_op] if op]
                if xforms:
                    child_xformable.SetXformOpOrder(xforms)

        # Set the flattened prim instanceable
        flattened_prim.GetPrim().SetInstanceable(True)

        # Zero out transformations on the flattened prim
        xformable = flattened_prim
        xformable.ClearXformOpOrder()

        translate_op = add_translate_op(xformable)
        if translate_op:
            translate_op.Set(get_vec3_type_for_presision(translate_op)(0, 0, 0))
        rotate_op = add_rotate_xyz_op(xformable)
        if rotate_op:
            rotate_op.Set(get_vec3_type_for_presision(rotate_op)(0, 0, 0))
        scale_op = add_scale_op(xformable)
        if scale_op:
            scale_op.Set(get_vec3_type_for_presision(scale_op)(1, 1, 1))
        xforms = [op for op in [translate_op, rotate_op, scale_op] if op]
        if xforms:
            xformable.SetXformOpOrder(xforms)

        return flattened_proto_path

    # -------------- Get Triangles, Normals, Areas --------------
    def get_mesh_triangles(mesh_prim: Usd.Prim, xform_cache: UsdGeom.XformCache):
        """
        Get the triangles, normals, and areas from a mesh prim, applying transformations.
        Returns three lists:
            triangles_list: List of (p0, p1, p2) where each p is a 3-tuple.
            normals_list: List of per-triangle normal as a 3-tuple.
            areas_list: List of per-triangle float areas.
        """
        mesh = UsdGeom.Mesh(mesh_prim)

        points_attr = mesh.GetPointsAttr()
        if not points_attr:
            return [], [], []

        points = points_attr.Get()
        if not points:
            return [], [], []

        transform = xform_cache.GetLocalToWorldTransform(mesh_prim)
        transformed_points = []
        for p in points:
            # Transform p (Gf.Vec3f) using transform
            p_world = transform.Transform(p)
            transformed_points.append((p_world[0], p_world[1], p_world[2]))

        faceVertexCounts = mesh.GetFaceVertexCountsAttr().Get()
        faceVertexIndices = mesh.GetFaceVertexIndicesAttr().Get()
        if not faceVertexCounts or not faceVertexIndices:
            return [], [], []

        triangles_out = []
        normals_out = []
        areas_out = []

        idx = 0
        for count in faceVertexCounts:
            c = count
            if c < 3:
                idx += c
                continue
            face_indices = faceVertexIndices[idx : idx + c]
            idx += c

            # Triangulate the face by creating a fan from the first vertex
            for i in range(1, c - 1):
                i0 = face_indices[0]
                i1 = face_indices[i]
                i2 = face_indices[i + 1]
                p0 = transformed_points[i0]
                p1 = transformed_points[i1]
                p2 = transformed_points[i2]
                triangles_out.append((p0, p1, p2))

        # Compute normals and areas
        for tri in triangles_out:
            p0, p1, p2 = tri
            e1 = vec_sub(p1, p0)
            e2 = vec_sub(p2, p0)
            cross_v = vec_cross(e1, e2)
            area = 0.5 * vec_length(cross_v)
            norm_v = vec_normalize(cross_v)
            normals_out.append(norm_v)
            areas_out.append(area)

        return triangles_out, normals_out, areas_out

    # -------------------- Start of scatter logic ----------------------
    if random_seed is not None:
        random.seed(random_seed)

    surface_prim = stage.GetPrimAtPath(surface_path)
    if not surface_prim:
        raise ValueError(f"Surface prim not found at path: {surface_path}")

    # Collect mesh prims under surface_prim
    mesh_prims = []
    for prim in Usd.PrimRange(surface_prim):
        if prim.IsA(UsdGeom.Mesh):
            mesh_prims.append(prim)
    if not mesh_prims:
        raise ValueError(f"No mesh prims found under surface path: {surface_path}")

    xform_cache = UsdGeom.XformCache()

    # Collect all triangles, normals, areas
    all_triangles = []
    all_normals = []
    all_areas = []

    for mprim in mesh_prims:
        tlist, nlist, alist = get_mesh_triangles(mprim, xform_cache)
        all_triangles.extend(tlist)
        all_normals.extend(nlist)
        all_areas.extend(alist)

    if not all_triangles:
        raise ValueError("No triangles found on the given surfaces.")

    # Build a CDF from areas
    cdf = []
    running_sum = 0.0
    for ar in all_areas:
        running_sum += ar
        cdf.append(running_sum)
    total_area = running_sum

    # Generate random numbers, find triangle indices using bisect
    rand_nums = [random.random() for _ in range(num_instances)]
    triangle_indices = [bisect.bisect_left(cdf, r * total_area) for r in rand_nums]

    # Generate random barycentric coords
    us = [random.random() for _ in range(num_instances)]
    vs = [random.random() for _ in range(num_instances)]
    # If u+v > 1, reflect
    for i in range(num_instances):
        if us[i] + vs[i] > 1.0:
            us[i] = 1.0 - us[i]
            vs[i] = 1.0 - vs[i]

    # Compute final points from barycentric
    points_list = []
    selected_normals = []
    for i in range(num_instances):
        tri_index = triangle_indices[i]
        p0, p1, p2 = all_triangles[tri_index]
        n0 = all_normals[tri_index]
        u = us[i]
        v = vs[i]
        w = 1.0 - (u + v)

        # Weighted point
        px = p0[0] * u + p1[0] * v + p2[0] * w
        py = p0[1] * u + p1[1] * v + p2[1] * w
        pz = p0[2] * u + p1[2] * v + p2[2] * w
        points_list.append((px, py, pz))
        selected_normals.append(n0)  # all_normals[tri_index]

    # Apply translation
    if isinstance(translation[0], (tuple, list)):
        # Random translation range
        (xmin, ymin, zmin), (xmax, ymax, zmax) = translation
        for i in range(num_instances):
            tx = random.uniform(xmin, xmax)
            ty = random.uniform(ymin, ymax)
            tz = random.uniform(zmin, zmax)
            points_list[i] = vec_add(points_list[i], (tx, ty, tz))
    else:
        # Fixed translation
        tx, ty, tz = translation
        for i in range(num_instances):
            points_list[i] = vec_add(points_list[i], (tx, ty, tz))

    # Determine stage up axis
    up_axis_token = UsdGeom.GetStageUpAxis(stage)
    if up_axis_token == UsdGeom.Tokens.y:
        up_vector = (0.0, 1.0, 0.0)
    elif up_axis_token == UsdGeom.Tokens.z:
        up_vector = (0.0, 0.0, 1.0)
    else:
        # Default to Y-up
        up_vector = (0.0, 1.0, 0.0)

    up_vector_norm = vec_normalize(up_vector)

    # Filter out large normal deviations
    # Angle between normal and up
    valid_points = []
    valid_normals = []
    angle_limit = max_normal_deviation_angle

    for i in range(num_instances):
        nrml = selected_normals[i]
        dotval = vec_dot(vec_normalize(nrml), up_vector_norm)
        dotval = min(max(dotval, -1.0), 1.0)
        angle_deg = math.degrees(math.acos(dotval))
        if angle_deg <= angle_limit:
            valid_points.append(points_list[i])
            valid_normals.append(nrml)

    num_valid = len(valid_points)
    if num_valid == 0:
        raise ValueError("No valid instances after applying normal deviation angle filter.")

    # Prepare orientation quaternions
    final_quats = []
    al = max(0.0, min(1.0, align_to_normals))

    for i in range(num_valid):
        nvec = valid_normals[i]
        base_quat = quat_from_two_vectors(up_vector, nvec)
        if al < 1.0:
            identity_q = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
            q_slerp = quat_slerp(identity_q, base_quat, al)
            final_quats.append(q_slerp)
        else:
            final_quats.append(base_quat)

    # Apply random rotation around up axis
    if isinstance(rotation, (tuple, list)):
        rot_min, rot_max = rotation
        random_rots = [random.uniform(rot_min, rot_max) for _ in range(num_valid)]
    else:
        random_rots = [rotation for _ in range(num_valid)]

    up_n = vec_normalize(up_vector)
    for i in range(num_valid):
        angle_rad = math.radians(random_rots[i])
        sin_half = math.sin(angle_rad * 0.5)
        cos_half = math.cos(angle_rad * 0.5)
        # Rotation about up axis
        axis = up_n
        rot_quat = Gf.Quatf(
            cos_half,
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half,
        )
        final_quats[i] = rot_quat * final_quats[i]

    quath_list = []
    for q in final_quats:
        re = float(q.GetReal())
        im = q.GetImaginary()
        quath_list.append(Gf.Quath(re, float(im[0]), float(im[1]), float(im[2])))
    quath_array = Vt.QuathArray(quath_list)

    # Prepare scales
    if isinstance(scale, (tuple, list)):
        smin, smax = scale
        random_scales = [random.uniform(smin, smax) for _ in range(num_valid)]
    else:
        random_scales = [scale for _ in range(num_valid)]
    scales_vec = [Gf.Vec3f(s, s, s) for s in random_scales]

    # ProtoIndices (all zero for single prototype)
    protoIndices = [0] * num_valid

    # Convert final positions
    gf_positions = [Gf.Vec3f(p[0], p[1], p[2]) for p in valid_points]

    # Ensure parent prims exist
    def _ensure_parents_exist(stage: Usd.Stage, path: Sdf.Path):
        if not path or path == Sdf.Path.absoluteRootPath:
            return
        _ensure_parents_exist(stage, path.GetParentPath())
        if not stage.GetPrimAtPath(path):
            stage.DefinePrim(path, "Xform")

    _ensure_parents_exist(stage, Sdf.Path(point_instancer_path).GetParentPath())

    # Create PointInstancer
    point_instancer_prim = stage.DefinePrim(point_instancer_path, "PointInstancer")
    point_instancer = UsdGeom.PointInstancer(point_instancer_prim)

    point_instancer.CreatePositionsAttr().Set(gf_positions)
    point_instancer.CreateOrientationsAttr().Set(quath_array)
    point_instancer.CreateScalesAttr().Set(scales_vec)
    point_instancer.CreateProtoIndicesAttr().Set(protoIndices)

    # Flatten the prototype once and set prototypes rel
    flattened_prototype_path = create_flattened_prototype(stage, prototype_path)
    if not flattened_prototype_path:
        raise ValueError("Failed to create flattened prototype.")
    prototypes_rel = point_instancer.CreatePrototypesRel()
    prototypes_rel.AddTarget(flattened_prototype_path)

    return point_instancer_prim


def stack_objects(stage: Usd.Stage, base_object_path: str, objects_to_stack_paths: List[str]) -> None:
    """
    Stacks objects on top of a base object and each other without intersections,
    optimizing placement based on base dimensions. The function operates with
    bounding boxes, considering the up vector of the scene. Objects can be
    rotated around the up axis to fit optimally.

    Use this function to stack multiple objects on top of each other without intersections like in real world.
    Stacked objects consider each others' dimensions and are placed optimally to avoid intersections.

    Don't use `align_objects` to stack multiple objects on top of each other. Use this function instead.

    Args:
        stage (Usd.Stage): The USD stage to operate on.
        base_object_path (str): The path to the base object to stack on top of.
        objects_to_stack_paths (List[str]): List of paths to the objects to stack.
    """

    # Helper class for available area
    class StackArea:
        def __init__(self, source_prim, min_point, max_point, height, rotated):
            self.source_prim = source_prim  # The prim on which this area is located
            self.min = min_point  # Gf.Vec2d (x, z) min corner of the area
            self.max = max_point  # Gf.Vec2d (x, z) max corner of the area
            self.ocup = min_point  # Record the area occupied by other object
            self.height = height  # The height at which this area is located (value along up axis)
            self.rotated = rotated  # Whether the object is rotated
            self.children = []  # List of children areas

        def __repr__(self):
            return f"StackArea({self.min}, {self.max}, {self.ocup}, {self.height})"

        @property
        def width(self):
            return self.max[0] - self.min[0]

        @property
        def depth(self):
            return self.max[1] - self.min[1]

        def can_place(self, obj_width, obj_depth):
            def can_fit(obj_width, obj_depth, rect):
                """Check if the object can fit in the given rectangle."""
                width = rect[2] - rect[0]
                depth = rect[3] - rect[1]
                if obj_width <= width and obj_depth <= depth:
                    return [rect[0], rect[1], rect[0] + obj_width, rect[1] + obj_depth]

            sub_rects_available_space = [
                [self.min[0], self.ocup[1], self.max[0], self.max[1]],
                [self.ocup[0], self.min[1], self.max[0], self.max[1]],
            ]

            max_area = self.width * self.depth
            best_area = None
            best_placement = None
            for rect in sub_rects_available_space:
                placed = can_fit(obj_width, obj_depth, rect)
                if not placed:
                    continue

                new_occup = [max(self.ocup[0], placed[2]), max(self.ocup[1], placed[3])]
                new_occup_area = (new_occup[0] - self.min[0]) * (new_occup[1] - self.min[1])
                rest_area = max_area - new_occup_area

                if rest_area > 0:
                    if not best_area or rest_area < best_area:
                        best_area = rest_area
                        best_placement = placed

            return best_area, best_placement

        def add_child(self, source_prim, min_point, max_point, height, rotated):
            child = StackArea(source_prim, min_point, max_point, height, rotated)
            self.ocup = [max(self.ocup[0], child.max[0]), max(self.ocup[1], child.max[1])]
            self.children.append(child)
            return child

    if not objects_to_stack_paths:
        raise ValueError("No objects to stack.")

    # Get the up axis
    up_axis_token = UsdGeom.GetStageUpAxis(stage)
    if up_axis_token == UsdGeom.Tokens.y:
        up_axis = Gf.Vec3d(0, 1, 0)
        base_plane_axes = (0, 2)  # X and Z axes
        up_axis_index = 1
    elif up_axis_token == UsdGeom.Tokens.z:
        up_axis = Gf.Vec3d(0, 0, 1)
        base_plane_axes = (0, 1)  # X and Y axes
        up_axis_index = 2
    else:
        raise ValueError("Unsupported up axis: {}".format(up_axis_token))

    # Get base object's bounding box
    base_bbox = get_bbox_world(stage, base_object_path)
    if base_bbox.IsEmpty():
        raise ValueError("Base object bounding box is empty.")

    base_min = base_bbox.GetMin()
    base_max = base_bbox.GetMax()
    base_top_height = base_max[up_axis_index]
    base_bottom_height = base_min[up_axis_index]

    # Get base plane coordinates
    base_xmin = base_min[base_plane_axes[0]]
    base_xmax = base_max[base_plane_axes[0]]
    base_zmin = base_min[base_plane_axes[1]]
    base_zmax = base_max[base_plane_axes[1]]

    base_available_area = StackArea(
        stage.GetPrimAtPath(base_object_path),
        Gf.Vec2d(base_xmin, base_zmin),
        Gf.Vec2d(base_xmax, base_zmax),
        base_top_height,
        False,
    )

    # For each object to stack, get its bounding box and possible rotations
    objects_info = []

    for obj_path in objects_to_stack_paths:
        if obj_path == base_object_path:
            continue

        obj_prim = stage.GetPrimAtPath(obj_path)
        if not obj_prim:
            raise ValueError(f"Object '{obj_path}' not found.")

        # Get object's bounding box in world space
        obj_bbox_world = get_bbox_world(stage, obj_path)
        if obj_bbox_world.IsEmpty():
            print(f"Object '{obj_path}' has empty bounding box. Skipping.")
            continue

        # Get object's size in world space
        obj_size = obj_bbox_world.GetSize()

        width = obj_size[base_plane_axes[0]]
        depth = obj_size[base_plane_axes[1]]
        height = obj_size[up_axis_index]

        objects_info.append(
            {
                "path": obj_path,
                "prim": obj_prim,
                "width": width,
                "depth": depth,
                "height": height,
            }
        )

    # Sort objects by the largest area (largest first)
    objects_info.sort(key=lambda x: -x["width"] * x["depth"])

    # Initialize the list of available areas
    available_areas = [base_available_area]

    # Place the objects
    for obj_info in objects_info:
        placed = False
        obj_width = obj_info["width"]
        obj_depth = obj_info["depth"]
        obj_height = obj_info["height"]
        obj_path = obj_info["path"]
        obj_prim = obj_info["prim"]

        # Try to place the object on any available area
        for area in available_areas:

            area_0, placement_0 = area.can_place(obj_width, obj_depth)
            area_90, placement_90 = area.can_place(obj_depth, obj_width)

            if area_0 and area_90:
                if area_0 >= area_90:
                    placement = placement_0
                    rotated = False
                else:
                    placement = placement_90
                    rotated = True
            elif area_0:
                placement = placement_0
                rotated = False
            elif area_90:
                placement = placement_90
                rotated = True
            else:
                continue

            placed = True
            child = area.add_child(
                obj_prim,
                Gf.Vec2d(placement[0], placement[1]),
                Gf.Vec2d(placement[2], placement[3]),
                area.height + obj_height,
                rotated,
            )
            available_areas.append(child)
            break

        if not placed:
            for area in available_areas:
                if area.children:
                    continue

                is_area_width_biggest = area.width > area.depth
                is_obj_width_biggest = obj_width > obj_depth

                if is_area_width_biggest != is_obj_width_biggest:
                    obj_width, obj_depth = obj_depth, obj_width
                    rotated = True
                else:
                    rotated = False

                # Center hanging objects
                area_center_x = (area.min[0] + area.max[0]) / 2.0
                area_center_z = (area.min[1] + area.max[1]) / 2.0
                placement_min_x = area_center_x - obj_width / 2.0
                placement_min_z = area_center_z - obj_depth / 2.0
                placement = [
                    placement_min_x,
                    placement_min_z,
                    placement_min_x + obj_width,
                    placement_min_z + obj_depth,
                ]
                placed = True
                child = area.add_child(
                    obj_prim,
                    Gf.Vec2d(placement[0], placement[1]),
                    Gf.Vec2d(placement[2], placement[3]),
                    area.height + obj_height,
                    rotated,
                )
                available_areas.append(child)
                break

    # Now place everything according to the map
    for area in available_areas:
        prim = area.source_prim

        if area.rotated:
            rotate_around_axis(stage, prim, up_axis, 90)

        bbox = get_bbox_world(stage, prim)
        delta_position = [0.0, 0.0, 0.0]
        delta_position[base_plane_axes[0]] = area.min[0] - bbox.GetMin()[base_plane_axes[0]]
        delta_position[base_plane_axes[1]] = area.min[1] - bbox.GetMin()[base_plane_axes[1]]
        delta_position[up_axis_index] = area.height - bbox.GetMax()[up_axis_index]

        if all(d == 0 for d in delta_position):
            continue

        translate = get_translate(stage, prim)
        translate = (
            translate[0] + delta_position[0],
            translate[1] + delta_position[1],
            translate[2] + delta_position[2],
        )

        set_translate(stage, prim, translate)


def add_attr_variants(
    stage: Usd.Stage,
    prim: Union[Usd.Prim, str],
    variant_set_name: str,
    variants_dict: Dict,
    attr_type_dict: Optional[Dict] = None,
):
    """
    Adds variants to a variant set on the specified prim, setting attribute values in each variant.

    This function simplifies the process of creating a variant set, adding variants to it, and assigning
    attribute values within each variant.

    Parameters:
        stage (Usd.Stage): The USD stage to operate on.
        prim (Union[Usd.Prim, str]): The USD prim on which to create or modify the variant set.
        variant_set_name (str): The name of the variant set to create or modify.
        variants_dict (dict): A dictionary mapping variant names to dictionaries of attribute names and values.
            Each key is a variant name, and each value is a dictionary mapping attribute names to values.
        attr_type_dict (dict, optional): A dictionary mapping attribute names to Sdf.ValueTypeName.
            If an attribute does not exist on the prim and its type cannot be inferred, this dictionary
            provides the typeName to use when creating the attribute.

    Example:
        # Define the variants and their attribute values
        variants = {
            "red": {"primvars:displayColor": [(1.0, 0.0, 0.0)]},
            "green": {"primvars:displayColor": [(0.0, 1.0, 0.0)]},
            "blue": {"primvars:displayColor": [(0.0, 0.0, 1.0)]}
        }

        # Optionally, define attribute types
        attr_types = {"primvars:displayColor": Sdf.ValueTypeNames.Color3fArray}

        # Add the variants to the prim
        usdcode.add_attr_variants(stage, prim, "shadingVariant", variants, attr_type_dict=attr_types)

    Returns:
        None
    """

    def infer_type_name_from_value(value):
        """
        Infers the Sdf.ValueTypeName for a given value.

        Parameters:
            value: The value from which to infer the type.

        Returns:
            Sdf.ValueTypeName or None if the type cannot be inferred.
        """
        if isinstance(value, bool):
            return Sdf.ValueTypeNames.Bool
        elif isinstance(value, int):
            return Sdf.ValueTypeNames.Int
        elif isinstance(value, float):
            return Sdf.ValueTypeNames.Float
        elif isinstance(value, str):
            return Sdf.ValueTypeNames.String
        elif isinstance(value, tuple):
            # Check for float tuples
            if all(isinstance(v, float) for v in value):
                if len(value) == 2:
                    return Sdf.ValueTypeNames.Float2
                elif len(value) == 3:
                    return Sdf.ValueTypeNames.Float3
                elif len(value) == 4:
                    return Sdf.ValueTypeNames.Float4
        elif isinstance(value, list):
            if len(value) == 0:
                return None  # Cannot infer type from empty list
            elem_type = infer_type_name_from_value(value[0])
            if elem_type:
                # Map scalar types to array types
                array_type_name = elem_type.type.pythonType.__name__ + "Array"
                array_type = getattr(Sdf.ValueTypeNames, array_type_name, None)
                return array_type
        return None

    if isinstance(prim, str) or isinstance(prim, Sdf.Path):
        prim = stage.GetPrimAtPath(prim)

    variant_sets = prim.GetVariantSets()
    vset = variant_sets.GetVariantSet(variant_set_name)
    if not vset:
        vset = variant_sets.AddVariantSet(variant_set_name)

    # Add all variants
    for variant_name in variants_dict.keys():
        vset.AddVariant(variant_name)

    # For each variant, set the variant selection and author the attribute values
    for variant_name, attributes in variants_dict.items():
        vset.SetVariantSelection(variant_name)
        with vset.GetVariantEditContext():
            for attr_name, attr_value in attributes.items():
                attr = prim.GetAttribute(attr_name)
                if not attr:
                    # Attribute does not exist, attempt to create it
                    if attr_type_dict and attr_name in attr_type_dict:
                        type_name = attr_type_dict[attr_name]
                        attr = prim.CreateAttribute(attr_name, type_name)
                    else:
                        # Attempt to infer typeName from value
                        type_name = infer_type_name_from_value(attr_value)
                        if type_name:
                            attr = prim.CreateAttribute(attr_name, type_name)
                        else:
                            raise RuntimeError(
                                f"Attribute '{attr_name}' does not exist on prim '{prim.GetPath()}', "
                                "and typeName cannot be inferred. Provide 'attr_type_dict' for this attribute."
                            )
                attr.Set(attr_value)
    # Optionally, reset variant selection
    vset.ClearVariantSelection()


def add_rel_variants(stage: Usd.Stage, prim: Union[Usd.Prim, str], variant_set_name: str, variants_dict: Dict):
    """
    Adds relationship variants to a variant set on the specified prim.

    This function simplifies the process of creating a variant set, adding variants to it,
    and assigning relationships within each variant.

    Parameters:
        prim (Usd.Prim): The USD prim on which to create or modify the variant set.
        variant_set_name (str): The name of the variant set to create or modify.
        variants_dict (dict): A dictionary mapping variant names to dictionaries of relationship
            names and target paths. Each key is a variant name, and each value is a dictionary
            mapping relationship names to target paths or lists of target paths.

    Example:
        # Define the variants and their relationship targets
        variants = {
            "MaterialA": {"material:binding": Sdf.Path("/Materials/MaterialA")},
            "MaterialB": {"material:binding": Sdf.Path("/Materials/MaterialB")},
        }

        # Add the relationship variants to the prim
        add_rel_variants(prim, "materialVariant", variants)

    Returns:
        None
    """
    if isinstance(prim, str) or isinstance(prim, Sdf.Path):
        prim = stage.GetPrimAtPath(prim)

    variant_sets = prim.GetVariantSets()
    vset = variant_sets.GetVariantSet(variant_set_name)
    if not vset:
        vset = variant_sets.AddVariantSet(variant_set_name)

    # Add all variants
    for variant_name in variants_dict.keys():
        vset.AddVariant(variant_name)

    # For each variant, set the variant selection and author the relationships
    for variant_name, relationships in variants_dict.items():
        vset.SetVariantSelection(variant_name)
        with vset.GetVariantEditContext():
            for rel_name, target_paths in relationships.items():
                rel = prim.GetRelationship(rel_name)
                if not rel:
                    # Relationship does not exist, create it
                    rel = prim.CreateRelationship(rel_name)
                else:
                    # Clear existing targets
                    rel.ClearTargets(True)
                # Add the target(s)
                if isinstance(target_paths, (Sdf.Path, str)):
                    rel.AddTarget(Sdf.Path(target_paths))
                elif isinstance(target_paths, list):
                    for target_path in target_paths:
                        rel.AddTarget(Sdf.Path(target_path))
                else:
                    raise TypeError(f"Invalid type for target_paths: {type(target_paths)}")
    # Optionally, reset variant selection
    vset.ClearVariantSelection()


def add_geo_variants(stage: Usd.Stage, prim: Union[str, Usd.Prim], variant_set_name: str, variants_dict: Dict):
    """
    Adds geometry variants to a variant set on the specified prim.

    This function allows you to create different geometry or sub-prim structures
    within each variant of a variant set on a prim.

    Parameters:
        stage (Usd.Stage): The USD stage to operate on.
        prim (Union[Usd.Prim, str]): The USD prim on which to create or modify the variant set.
        variant_set_name (str): The name of the variant set to create or modify.
        variants_dict (dict): A dictionary mapping variant names to functions that
            define the geometry or subgraph for that variant. Each function takes
            a single argument, the USD stage, and is responsible for creating the
            geometry under the prim.

    Example:
        # Define functions that create geometry for each variant
        def create_sphere(stage, prim_path):
            usdcode.create_prim(stage, "Sphere", f"{prim_path}/Sphere")

        def create_cube(stage, prim_path):
            usdcode.create_prim(stage, "Cube", f"{prim_path}/Cube")

        # Map variant names to geometry creation functions
        variants = {
            "Sphere": create_sphere,
            "Cube": create_cube,
        }

        # Add the geometry variants to the prim
        usdcode.add_geo_variants(stage, prim, "geoVariant", variants)

    Returns:
        None
    """
    if isinstance(prim, str) or isinstance(prim, Sdf.Path):
        prim = stage.GetPrimAtPath(prim)

    variant_sets = prim.GetVariantSets()
    vset = variant_sets.GetVariantSet(variant_set_name)
    if not vset:
        vset = variant_sets.AddVariantSet(variant_set_name)

    # Add all variants
    for variant_name in variants_dict.keys():
        vset.AddVariant(variant_name)

    # For each variant, set the variant selection and author the geometry
    for variant_name, create_geometry_fn in variants_dict.items():
        vset.SetVariantSelection(variant_name)
        with vset.GetVariantEditContext():
            # Clear existing children under the prim for this variant
            for child in prim.GetAllChildren():
                prim.RemoveChild(child)
            # Call the user-provided function to create geometry under the prim
            create_geometry_fn(prim.GetStage(), prim.GetPath())
    # Optionally, reset variant selection
    vset.ClearVariantSelection()


def select_variant(stage: Usd.Stage, prim: Union[Usd.Prim, str], variant_set_name: str, variant_name: str):
    """
    Sets the variant selection for a given variant set on the specified prim.

    Parameters:
        stage (Usd.Stage): The USD stage to operate on.
        prim (Union[Usd.Prim, str]): The USD prim on which to set the variant selection.
        variant_set_name (str): The name of the variant set.
        variant_name (str): The name of the variant to select.

    Returns:
        None
    """
    if isinstance(prim, str) or isinstance(prim, Sdf.Path):
        prim = stage.GetPrimAtPath(prim)

    variant_sets = prim.GetVariantSets()
    variant_set = variant_sets.GetVariantSet(variant_set_name)
    if variant_set:
        variant_set.SetVariantSelection(variant_name)
    else:
        raise ValueError(f"Variant set '{variant_set_name}' does not exist on prim '{prim.GetPath()}'.")


def remove_variant(stage: Usd.Stage, prim: Union[Usd.Prim, str], variant_set_name: str, variant_name: str):
    """
    Removes a variant from a variant set on the specified prim.

    Parameters:
        prim (Usd.Prim): The USD prim from which to remove the variant.
        variant_set_name (str): The name of the variant set.
        variant_name (str): The name of the variant to remove.

    Returns:
        None
    """
    if isinstance(prim, str) or isinstance(prim, Sdf.Path):
        prim = stage.GetPrimAtPath(prim)

    stage = prim.GetStage()
    # Iterate over all layers in the stage's local LayerStack
    for layer in stage.GetLayerStack():
        if not layer.permissionToEdit:
            continue  # Skip read-only layers
        prim_spec = layer.GetPrimAtPath(prim.GetPath())
        if not prim_spec:
            continue  # Prim not defined in this layer
        # Access the variant set spec
        variant_set_spec = prim_spec.variantSets.get(variant_set_name)
        if not variant_set_spec:
            continue  # Variant set not defined in this layer
        # Remove the variant if it exists
        if variant_name in variant_set_spec.variants:
            variant_set_spec.RemoveVariant(variant_set_spec.variants[variant_name])


def clear_variant_selection(stage: Usd.Stage, prim: Union[Usd.Prim, str], variant_set_name: str):
    """
    Clears the variant selection for a given variant set on the specified prim.

    Parameters:
        stage (Usd.Stage): The USD stage to operate on.
        prim (Union[Usd.Prim, str]): The USD prim on which to clear the variant selection.
        variant_set_name (str): The name of the variant set.

    Returns:
        None
    """
    if isinstance(prim, str) or isinstance(prim, Sdf.Path):
        prim = stage.GetPrimAtPath(prim)

    variant_sets = prim.GetVariantSets()
    variant_set = variant_sets.GetVariantSet(variant_set_name)
    if variant_set:
        variant_set.ClearVariantSelection()
    else:
        raise ValueError(f"Variant set '{variant_set_name}' does not exist on prim '{prim.GetPath()}'.")


def delete_variant_set(stage: Usd.Stage, prim: Union[Usd.Prim, str], variant_set_name: str):
    """
    Deletes a variant set from the specified prim.

    Parameters:
        stage (Usd.Stage): The USD stage to operate on.
        prim (Union[Usd.Prim, str]): The USD prim from which to remove the variant set.
        variant_set_name (str): The name of the variant set to remove.

    Returns:
        None
    """
    if isinstance(prim, str) or isinstance(prim, Sdf.Path):
        prim = stage.GetPrimAtPath(prim)
    elif not stage:
        stage = prim.GetStage()

    # Iterate over all layers in the stage's local LayerStack
    for layer in stage.GetLayerStack():
        if not layer.permissionToEdit:
            continue  # Skip read-only layers
        prim_spec = layer.GetPrimAtPath(prim.GetPath())
        if not prim_spec:
            continue  # Prim not defined in this layer
        # Check if the variant set exists
        if variant_set_name in prim_spec.variantSets:
            del prim_spec.variantSets[variant_set_name]


def construct_room(
    stage: Usd.Stage,
    parent_prim_path: str,
    room_dimensions: Tuple[float, float, float],
    floor_prim_path: str,
    wall_prim_path: str,
    ceiling_prim_path: str,
) -> None:
    """
    Creates a room by tiling floor, wall, and ceiling prims.

    Instead of scaling individual prims, this function creates a room by placing multiple copies
    of your floor, wall, and ceiling prims in a grid pattern. This preserves the original
    materials, textures, and proportions of your architectural elements.

    Example:
        If you have:
        - A 1x1m floor tile at "/Assets/FloorTile"
        - A 2x3m wall panel at "/Assets/WallPanel"
        - A 1x1m ceiling tile at "/Assets/CeilingTile"

        You can create a 5x4x3m room with:
        >>> usdcode.construct_room(
        ...     stage,
        ...     "/World/Room",
        ...     (5.0, 4.0, 3.0),  # width=5m, depth=4m, height=3m
        ...     "/Assets/FloorTile",
        ...     "/Assets/WallPanel",
        ...     "/Assets/CeilingTile"
        ... )

        This will automatically:
        - Tile the floor with 1x1m tiles to cover 5x4m
        - Place wall panels around the perimeter
        - Tile the ceiling with 1x1m tiles at 3m height

    Args:
        stage (Usd.Stage): The USD stage where the room will be created.
        parent_prim_path (str): The path where the room assembly will be created.
        room_dimensions (Tuple[float, float, float]): Desired room size in (width, depth, height).
        floor_prim_path (str): Reference to the floor tile/piece to be repeated.
        wall_prim_path (str): Reference to the wall panel/piece to be repeated.
        ceiling_prim_path (str): Reference to the ceiling tile/piece to be repeated.
    """
    # Get stage's up axis
    up_axis = UsdGeom.GetStageUpAxis(stage)
    up_axis_index = {"X": 0, "Y": 1, "Z": 2}[up_axis]

    # Define base plane axes based on up axis
    if up_axis == "Y":
        base_plane_axes = (0, 2)  # X and Z for floor plane
    elif up_axis == "Z":
        base_plane_axes = (0, 1)  # X and Y for floor plane
    else:  # X
        base_plane_axes = (1, 2)  # Y and Z for floor plane

    # Ensure parent prim exists and is empty
    parent_prim = stage.GetPrimAtPath(parent_prim_path)
    if not parent_prim:
        parent_prim = stage.DefinePrim(parent_prim_path, "Xform")
    else:
        # Remove all existing children
        for child in parent_prim.GetChildren():
            remove_prim(stage, str(child.GetPath()))

    room_width, room_depth, room_height = room_dimensions

    # Helper function to get bounding box size and offset
    def get_bbox_size_and_offset(prim_path: str):
        bbox = get_bbox_world(stage, prim_path)
        size = bbox.GetSize()
        # Get the prim's position
        prim_pos = get_translate(stage, prim_path)
        # Calculate offset from pivot (prim position) to bbox center
        bbox_center = bbox.GetMidpoint()
        offset = [c - p for c, p in zip(bbox_center, prim_pos)]
        return size, offset

    # Get tile dimensions and offsets
    floor_dims, floor_offset = get_bbox_size_and_offset(floor_prim_path)
    wall_dims, wall_offset = get_bbox_size_and_offset(wall_prim_path)
    ceiling_dims, ceiling_offset = get_bbox_size_and_offset(ceiling_prim_path)

    # Calculate number of floor tiles needed
    num_floor_tiles_x = ceil(room_width / floor_dims[base_plane_axes[0]]) + 1
    num_floor_tiles_z = ceil(room_depth / floor_dims[base_plane_axes[1]]) + 1

    # Ensure even number of floor tiles for symmetric placement
    num_floor_tiles_x += num_floor_tiles_x % 2 == 0
    num_floor_tiles_z += num_floor_tiles_z % 2 == 0

    # Calculate number of ceiling tiles needed
    num_ceiling_tiles_x = ceil(room_width / ceiling_dims[base_plane_axes[0]]) + 1
    num_ceiling_tiles_z = ceil(room_depth / ceiling_dims[base_plane_axes[1]]) + 1

    # Ensure even number of ceiling tiles for symmetric placement
    num_ceiling_tiles_x += num_ceiling_tiles_x % 2 == 0
    num_ceiling_tiles_z += num_ceiling_tiles_z % 2 == 0

    # Create floor tiles
    for i in range(num_floor_tiles_x):
        for j in range(num_floor_tiles_z):
            tile_prim_path = f"{parent_prim_path}/Floor_{i}_{j}"
            create_reference(stage, tile_prim_path, "", floor_prim_path)

            # Calculate position
            pos = [0, 0, 0]
            pos[base_plane_axes[0]] = (i - num_floor_tiles_x / 2 + 0.5) * floor_dims[base_plane_axes[0]]
            pos[base_plane_axes[1]] = (j - num_floor_tiles_z / 2 + 0.5) * floor_dims[base_plane_axes[1]]
            # Adjust position by the offset
            pos = [p - o for p, o in zip(pos, floor_offset)]
            set_translate(stage, tile_prim_path, tuple(pos))

    # Create ceiling tiles using ceiling-specific calculations
    for i in range(num_ceiling_tiles_x):
        for j in range(num_ceiling_tiles_z):
            tile_prim_path = f"{parent_prim_path}/Ceiling_{i}_{j}"
            create_reference(stage, tile_prim_path, "", ceiling_prim_path)

            # Calculate position
            pos = [0, 0, 0]
            pos[base_plane_axes[0]] = (i - num_ceiling_tiles_x / 2 + 0.5) * ceiling_dims[base_plane_axes[0]]
            pos[base_plane_axes[1]] = (j - num_ceiling_tiles_z / 2 + 0.5) * ceiling_dims[base_plane_axes[1]]
            pos[up_axis_index] = room_height - (ceiling_dims[up_axis_index] / 2)
            # Adjust position by the offset
            pos = [p - o for p, o in zip(pos, ceiling_offset)]
            set_translate(stage, tile_prim_path, tuple(pos))

    # For walls, find dimensions by size (sort by actual dimension values)
    wall_dims_with_indices = [(dim, i) for i, dim in enumerate(wall_dims)]
    wall_dims_sorted = sorted(wall_dims_with_indices)  # Sort by dimension value

    wall_thickness_index = wall_dims_sorted[0][1]  # Index of smallest dimension (thickness)
    wall_width_index = wall_dims_sorted[1][1]  # Index of middle dimension (width for spacing)
    wall_height_index = wall_dims_sorted[2][1]  # Index of largest dimension (height)

    wall_thickness = wall_dims[wall_thickness_index]
    wall_height = wall_dims[wall_height_index]
    wall_width = wall_dims[wall_width_index]  # This is the dimension we'll use for spacing

    # Calculate number of wall tiles needed using wall_width for horizontal spacing
    num_wall_tiles_y = ceil(room_height / wall_height)
    num_wall_x = ceil(room_width / wall_width) + 1  # Add 1 for complete coverage
    num_wall_z = ceil(room_depth / wall_width) + 1  # Add 1 for complete coverage

    # Calculate the starting height to ensure walls reach from floor to ceiling
    wall_start_height = wall_height / 2  # Start at half height to align bottom with floor
    remaining_height = room_height - (num_wall_tiles_y * wall_height)
    if remaining_height < 0:
        wall_start_height += remaining_height / 2

    # Calculate rotations needed for walls
    def get_wall_rotation(is_front_back: bool) -> Tuple[float, float, float]:
        # Start with no rotation
        rotation = [0, 0, 0]

        if up_axis == "Z":
            # First, rotate so the thickness is along the correct axis
            if wall_thickness_index == 0:  # X is thickness
                if is_front_back:
                    rotation[2] = 90  # Rotate 90 around Z for front/back
                else:
                    rotation[2] = 0  # No Z rotation for side walls
            elif wall_thickness_index == 1:  # Y is thickness
                if is_front_back:
                    rotation[2] = 0  # No Z rotation needed for front/back
                else:
                    rotation[2] = -90  # Rotate -90 around Z for side walls
            elif wall_thickness_index == 2:  # Z is thickness
                rotation[0] = 90  # Rotate 90 around X to make Z horizontal
                if is_front_back:
                    rotation[2] = 90  # Additional rotation for front/back
                else:
                    rotation[2] = 0  # No additional rotation for side walls

            # Then ensure height is along Z axis
            if wall_height_index == 0:  # Height is along X
                rotation[1] = -90
            elif wall_height_index == 1:  # Height is along Y
                rotation[0] = 90

        else:  # Y is up (existing logic)
            # First, rotate so the thickness is along the correct axis
            if wall_thickness_index == 0:  # X is thickness
                if is_front_back:
                    rotation[1] = 0  # No Y rotation needed for front/back
                else:
                    rotation[1] = -90  # Rotate -90 around Y for side walls
            elif wall_thickness_index == 1:  # Y is thickness
                rotation[0] = 90  # Rotate around X to point Y along Z
                if not is_front_back:
                    rotation[1] = -90  # Additional rotation for side walls
            elif wall_thickness_index == 2:  # Z is thickness
                if is_front_back:
                    rotation[1] = 90  # Rotate 90 around Y for front/back
                else:
                    rotation[1] = 0  # No rotation needed for side walls

            # Then ensure height is along up axis
            if wall_height_index != up_axis_index:
                if wall_height_index == 0:  # Height is along X
                    rotation[2] = 90
                elif wall_height_index == 2:  # Height is along Z
                    rotation[0] = -90

        return tuple(rotation)

    # Create walls along X axis (front and back)
    front_back_rotation = get_wall_rotation(True)
    for is_front in [True, False]:
        for i in range(num_wall_x):
            for j in range(num_wall_tiles_y):
                z_pos = room_depth / 2 if is_front else -room_depth / 2
                tile_prim_path = f"{parent_prim_path}/Wall_X_{i}_{j}_{'front' if is_front else 'back'}"
                create_reference(stage, tile_prim_path, "", wall_prim_path)

                # Calculate position using wall_width for spacing
                pos = [0, 0, 0]
                pos[base_plane_axes[0]] = (i - num_wall_x / 2 + 0.5) * wall_width
                pos[up_axis_index] = wall_start_height + (j * wall_height)
                pos[base_plane_axes[1]] = z_pos
                # Adjust position by the offset
                pos = [p - o for p, o in zip(pos, wall_offset)]
                set_translate(stage, tile_prim_path, tuple(pos))

                # Apply rotation
                set_rotate(stage, tile_prim_path, front_back_rotation)

    # Create walls along Z axis (left and right)
    side_rotation = get_wall_rotation(False)
    for is_right in [True, False]:
        for i in range(num_wall_z):
            for j in range(num_wall_tiles_y):
                x_pos = room_width / 2 if is_right else -room_width / 2
                tile_prim_path = f"{parent_prim_path}/Wall_Z_{i}_{j}_{'right' if is_right else 'left'}"
                create_reference(stage, tile_prim_path, "", wall_prim_path)

                # Calculate position using wall_width for spacing
                pos = [0, 0, 0]
                pos[base_plane_axes[0]] = x_pos
                pos[up_axis_index] = wall_start_height + (j * wall_height)
                pos[base_plane_axes[1]] = (i - num_wall_z / 2 + 0.5) * wall_width
                # Adjust position by the offset
                pos = [p - o for p, o in zip(pos, wall_offset)]
                set_translate(stage, tile_prim_path, tuple(pos))

                # Apply rotation
                set_rotate(stage, tile_prim_path, side_rotation)


def construct_parallel_arrays(
    stage: Usd.Stage,
    parent_prim_path: str,
    source_prim_path: str,
    num_arrays: int,
    length_prims: int,
    height_prims: int,
    distance_between_arrays: float,
) -> None:
    """
    Creates parallel arrays of prims by arranging multiple copies in a grid pattern.

    Perfect for creating warehouse storage rows by arranging multiple copies of a source prim.

    This function aligns the source prim so that its longest side is along the array length,
    and arranges multiple instances to create parallel rows with specified spacing.

    Args:
        stage (Usd.Stage): The USD stage where the arrays will be created.
        parent_prim_path (str): The path where the array assembly will be created.
        source_prim_path (str): Reference to the source prim to be repeated.
        num_arrays (int): The number of parallel arrays to create.
        length_prims (int): The number of prims to make the length of each array.
        height_prims (int): The number of prims to make the height of each array.
        distance_between_arrays (float): The distance between arrays, measured edge to edge.

    Example:
        >>> usdcode.construct_parallel_arrays(
        ...     stage,
        ...     "/World/Arrays",
        ...     "/Assets/SourcePrim",
        ...     num_arrays=5,
        ...     length_prims=10,
        ...     height_prims=3,
        ...     distance_between_arrays=2.0
        ... )
    """
    # Get stage's up axis
    up_axis = UsdGeom.GetStageUpAxis(stage)
    up_axis_index = {"X": 0, "Y": 1, "Z": 2}[up_axis]

    # Define horizontal axes based on up axis
    horizontal_axes = [0, 1, 2]
    horizontal_axes.remove(up_axis_index)
    axis_length = horizontal_axes[0]  # We'll align aisle length along this axis
    axis_width = horizontal_axes[1]  # Aisle width (spacing between aisles) along this axis

    # Ensure parent prim exists
    parent_prim = stage.GetPrimAtPath(parent_prim_path)
    if not parent_prim:
        parent_prim = stage.DefinePrim(parent_prim_path, "Xform")
    else:
        # Remove all existing children
        for child in parent_prim.GetChildren():
            remove_prim(stage, str(child.GetPath()))

    # Get source prim dimensions and pivot offset
    def get_bbox_size_and_offset(prim_path: str):
        bbox = get_bbox_world(stage, prim_path)
        size = bbox.GetSize()
        # Get the prim's position
        prim_pos = get_translate(stage, prim_path)
        # Calculate offset from pivot (prim position) to bbox center
        bbox_center = bbox.GetMidpoint()
        offset = [c - p for c, p in zip(bbox_center, prim_pos)]
        return size, offset

    source_dims, source_offset = get_bbox_size_and_offset(source_prim_path)

    # Determine the longest side (excluding up axis)
    size_without_up = [source_dims[i] for i in horizontal_axes]
    longest_side_index = size_without_up.index(max(size_without_up))
    # Map back to original axis
    longest_axis = horizontal_axes[longest_side_index]

    # Determine if rotation is needed
    # We want to align the longest side with the aisle length axis
    rotation_angle = 0
    if longest_axis != axis_length:
        # Need to rotate around up axis by 90 degrees
        rotation_angle = 90  # Degrees

    # Prepare the rotated source prim
    temp_prim_path = parent_prim_path + "/TempSource"
    create_reference(stage, temp_prim_path, "", source_prim_path)
    if rotation_angle != 0:
        # Create up axis vector [0,0,0] with 1 at the up axis index
        up_axis_vector = [0, 0, 0]
        up_axis_vector[up_axis_index] = 1
        rotate_around_axis(stage, temp_prim_path, up_axis_vector, rotation_angle)
    # Get the dimensions after potential rotation
    rotated_dims, rotated_offset = get_bbox_size_and_offset(temp_prim_path)
    # Remove the temporary prim
    remove_prim(stage, temp_prim_path)

    # Now we can use rotated_dims for placement calculations
    # Calculate total width of an aisle (edge to edge)
    aisle_width = rotated_dims[axis_width]
    # Calculate the starting position for aisles
    total_aisles_width = num_arrays * aisle_width + (num_arrays - 1) * distance_between_arrays
    start_offset = -total_aisles_width / 2 + aisle_width / 2

    # Calculate the starting height for stacking
    unit_height = rotated_dims[up_axis_index]
    remaining_height = (height_prims * unit_height) - unit_height * height_prims
    start_height = unit_height / 2 + remaining_height / 2

    # Create aisles
    for aisle_num in range(num_arrays):
        # Position along the width axis
        aisle_pos = start_offset + aisle_num * (aisle_width + distance_between_arrays)
        for length_idx in range(length_prims):
            for height_idx in range(height_prims):
                prim_name = f"Aisle_{aisle_num}_L{length_idx}_H{height_idx}"
                prim_path = f"{parent_prim_path}/{prim_name}"
                create_reference(stage, prim_path, "", source_prim_path)

                # Apply rotation if needed
                if rotation_angle != 0:
                    rotate_around_axis(stage, prim_path, up_axis_vector, rotation_angle)

                # Calculate position
                pos = [0, 0, 0]
                # Position along the length axis
                pos[axis_length] = (length_idx - length_prims / 2 + 0.5) * rotated_dims[axis_length]
                # Position along the width axis
                pos[axis_width] = aisle_pos
                # Position along the up axis
                pos[up_axis_index] = start_height + height_idx * unit_height
                # Adjust for pivot offset
                pos = [p - o for p, o in zip(pos, rotated_offset)]
                set_translate(stage, prim_path, tuple(pos))


def create_light(
    stage: Usd.Stage,
    prim_path: str,
    light_type: str,
    intensity: float,
    color: Tuple[float, float, float],
    direction: Tuple[float, float, float],  # Direction vector the light is pointing
    position: Tuple[float, float, float],
    size: Union[Tuple[float, float], float],  # (width, height) for RectLight, radius for SphereLight
) -> None:
    """
    Creates a light of specified type with given parameters.

    Args:
        stage (Usd.Stage): The USD stage where the light will be created.
        prim_path (str): Path where to create the light.
        light_type (str): Type of light, either "rect" or "sphere".
        intensity (float): Light intensity.
        color (Tuple[float, float, float]): Light color in RGB.
        direction (Tuple[float, float, float]): Direction vector the light is pointing.
        position (Tuple[float, float, float]): Position of the light.
        size (Union[Tuple[float, float], float]): For RectLight: (width, height), for SphereLight: radius.

    Example:
        # Create a red rectangular area light pointing down
        >>> usdcode.create_light(
        ...     stage,
        ...     "/World/Lights/AreaLight1",
        ...     "rect",
        ...     intensity=1E+3,
        ...     color=(1.0, 0.0, 0.0),
        ...     direction=(0, 0, -1),  # Points straight down
        ...     position=(0, 0, 5),
        ...     size=(3.3, 3.5)
        ... )

    Typical created light looks like this:
        def RectLight "RectLight_0"
        {
            float3[] extent = [(-25, -25, 0), (25, 25, 0)]
            color3f inputs:color = (1, 1, 1)
            float inputs:height = 50
            float inputs:intensity = 1000
            float inputs:width = 50
            float3 xformOp:rotateXYZ = (-90, -180, 0)
            double3 xformOp:translate = (0, 200, 0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
        }
    """
    # Validate light type
    if light_type not in ["rect", "sphere"]:
        raise ValueError("light_type must be either 'rect' or 'sphere'")

    # Create the light prim
    light_type_map = {"rect": "RectLight", "sphere": "SphereLight"}
    light_prim = stage.DefinePrim(prim_path, light_type_map[light_type])

    # Wrap with correct schema
    if light_type == "rect":
        light = UsdLux.RectLight(light_prim)
    else:  # sphere
        light = UsdLux.SphereLight(light_prim)

    # Set common attributes
    light.CreateIntensityAttr().Set(intensity)
    light.CreateColorAttr().Set(Gf.Vec3f(*color))

    # Convert direction vector to rotation
    dir_vec = Gf.Vec3d(*direction).GetNormalized()
    # Invert the direction as the light points in the opposite direction
    dir_vec = -dir_vec
    # Calculate rotation angles
    # First, calculate yaw (around Y axis)
    yaw = degrees(atan2(dir_vec[0], dir_vec[2]))
    # Then calculate pitch (around X axis)
    pitch = degrees(-asin(dir_vec[1]))
    # Roll is 0 as we don't care about roll for lights
    roll = 0

    # Set transform using our helper functions
    set_translate(stage, prim_path, position)
    set_rotate(stage, prim_path, (pitch, yaw, roll))

    # Set type-specific attributes
    if light_type == "rect":
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError("size for RectLight must be a tuple of (width, height)")
        width, height = size
        light.CreateWidthAttr().Set(width)
        light.CreateHeightAttr().Set(height)
        # Set extent based on width and height
        half_width = width / 2
        half_height = height / 2
        light.CreateExtentAttr().Set([(-half_width, -half_height, 0), (half_width, half_height, 0)])
    else:  # sphere
        if not isinstance(size, (int, float)):
            raise ValueError("size for SphereLight must be a single number (radius)")
        radius = size
        light.CreateRadiusAttr().Set(radius)
        extent = radius  # Match the example's extent scale
        light.CreateExtentAttr().Set([(-extent, -extent, -extent), (extent, extent, extent)])
