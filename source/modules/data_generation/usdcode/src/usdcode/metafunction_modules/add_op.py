## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from pxr import Gf, UsdGeom


def get_vec3_type_for_presision(op):
    if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        return Gf.Vec3d
    return Gf.Vec3f


def add_translate_op(xformable, *args, **kwargs):
    """Add a translate op to the xformable if it doesn't exist."""
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    try:
        return xformable.AddTranslateOp(*args, **kwargs)
    except Exception as e:
        return None


def add_rotate_xyz_op(xformable, *args, **kwargs):
    """Add a rotateXYZ op to the xformable if it doesn't exist."""
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            return op
    try:
        return xformable.AddRotateXYZOp(*args, **kwargs)
    except Exception as e:
        return None


def add_scale_op(xformable, *args, **kwargs):
    """Add a scale op to the xformable if it doesn't exist."""
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            return op
    try:
        return xformable.AddScaleOp(*args, **kwargs)
    except Exception as e:
        return None


def add_orient_op(xformable, *args, **kwargs):
    """Add an orient op to the xformable if it doesn't exist."""
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            return op
    try:
        return xformable.AddOrientOp(*args, **kwargs)
    except Exception as e:
        return None
