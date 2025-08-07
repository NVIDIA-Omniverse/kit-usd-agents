## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import usdcode
from pxr import Sdf, Usd, UsdGeom


def create_camera_from_view(stage: Usd.Stage, path: Sdf.Path | str = None) -> str:
    """
    Creates a new camera at the specified path using the current camera's view.
    This is the best function to create a new camera from the current view.

    Args:
        path: The path where the new camera should be created. If None, a default path will be used.

    Returns:
        str: The path to the newly created camera.

    Example:
        usdcode.create_camera_from_view(stage)
    """
    if not stage:
        raise RuntimeError("No stage available")

    # Get current
    current_camera_path = usdcode.get_current_camera_path()
    if not current_camera_path:
        raise RuntimeError("No active camera found")

    # Get current camera prim
    current_camera_prim = stage.GetPrimAtPath(current_camera_path)
    if not current_camera_prim:
        raise RuntimeError("Failed to get current camera prim")

    # Get the camera's transform
    xform = UsdGeom.Xformable(current_camera_prim)
    if not xform:
        raise RuntimeError("Failed to get camera transform")

    time = Usd.TimeCode.Default()
    world_transform = xform.ComputeLocalToWorldTransform(time)

    # Create path for new camera if not provided
    if path is None:
        path = usdcode.get_next_free_path(stage, "/Camera", True)

    # Convert string path to SdfPath if needed
    if isinstance(path, str):
        path = Sdf.Path(path)

    # Create new camera prim
    new_camera_prim = UsdGeom.Camera.Define(stage, path)
    if not new_camera_prim:
        raise RuntimeError(f"Failed to create new camera at {path}")

    # Apply transform from current camera
    new_xform = UsdGeom.Xformable(new_camera_prim.GetPrim())

    # Get existing xform operations
    xform_ops = new_xform.GetOrderedXformOps()

    # Look for an existing transform operation
    transform_op = None
    for op in xform_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            transform_op = op
            break

    # If no transform operation exists, add one
    if not transform_op:
        # Clear existing transforms to start fresh
        new_xform.ClearXformOpOrder()
        transform_op = new_xform.AddTransformOp()

    # Set the transform value
    transform_op.Set(world_transform)

    # Ensure the operation is in the xform order if we're using an existing one
    if transform_op not in xform_ops and xform_ops:
        new_order = xform_ops + [transform_op]
        new_xform.SetXformOpOrder(new_order)

    # Copy camera parameters (focal length, fStop, etc.) if needed
    camera_schema = UsdGeom.Camera(current_camera_prim)
    new_camera_schema = UsdGeom.Camera(new_camera_prim.GetPrim())

    # Copy essential camera attributes
    attributes_to_copy = [
        "focalLength",
        "horizontalAperture",
        "verticalAperture",
        "horizontalApertureOffset",
        "verticalApertureOffset",
        "fStop",
        "focusDistance",
        "clippingRange",
    ]

    for attr_name in attributes_to_copy:
        src_attr = camera_schema.GetPrim().GetAttribute(attr_name)
        if src_attr and src_attr.HasValue():
            dst_attr = new_camera_schema.GetPrim().GetAttribute(attr_name)
            if dst_attr:
                dst_attr.Set(src_attr.Get())

    return str(path)
