## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import math
from math import acos, radians, sin, tan
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdShade, UsdSkel, Vt

from .add_op import *


def combine_bboxes(bbox_a: Gf.Range3f, bbox_b: Gf.Range3f) -> Gf.Range3f:
    """Combine two bounding boxes into a single bounding box that encloses both."""
    if bbox_a.IsEmpty():
        return bbox_b
    if bbox_b.IsEmpty():
        return bbox_a
    min_a = bbox_a.GetMin()
    max_a = bbox_a.GetMax()
    min_b = bbox_b.GetMin()
    max_b = bbox_b.GetMax()
    combined_min = Gf.Vec3f(min(min_a[0], min_b[0]), min(min_a[1], min_b[1]), min(min_a[2], min_b[2]))
    combined_max = Gf.Vec3f(max(max_a[0], max_b[0]), max(max_a[1], max_b[1]), max(max_a[2], max_b[2]))
    combined_bbox = Gf.Range3f()
    combined_bbox.SetMin(combined_min)
    combined_bbox.SetMax(combined_max)
    return combined_bbox


def combine_all_bboxes(bboxes: List[Gf.BBox3d]) -> Gf.BBox3d:
    """
    Combines multiple bounding boxes into a single bounding box.

    Args:
        bboxes (List[Gf.BBox3d]): A list of bounding boxes to combine.

    Returns:
        Gf.BBox3d: The combined bounding box.

    Raises:
        ValueError: If the input list is empty.
    """
    if not bboxes:
        raise ValueError("The input list of bounding boxes is empty.")
    combined_bbox = Gf.BBox3d(bboxes[0])
    for bbox in bboxes[1:]:
        combined_bbox = Gf.BBox3d.Combine(combined_bbox, bbox)
    return combined_bbox


def compute_bbox_centroid(bbox: Gf.BBox3d) -> Gf.Vec3d:
    """
    Computes the centroid of the given bounding box.

    Args:
        bbox (Gf.BBox3d): The bounding box to compute the centroid for.

    Returns:
        Gf.Vec3d: The centroid of the bounding box.
    """
    if bbox.GetRange().IsEmpty():
        return Gf.Vec3d(0, 0, 0)
    centroid = bbox.ComputeCentroid()
    return centroid


def get_bbox_volume(bbox: Gf.Range3d) -> float:
    """Calculate the volume of a 3D bounding box.

    Args:
        bbox (Gf.Range3d): The input bounding box.

    Returns:
        float: The volume of the bounding box.

    Raises:
        ValueError: If the bounding box is empty or invalid.
    """
    if bbox.IsEmpty():
        raise ValueError("The bounding box is empty.")
    size = bbox.GetSize()
    if not all((dim > 0 for dim in size)):
        raise ValueError("The bounding box has invalid dimensions.")
    volume = size[0] * size[1] * size[2]
    return volume


def set_bbox_matrix(bbox: Gf.BBox3d, matrix: Gf.Matrix4d) -> None:
    """Set the matrix of a BBox3d.

    Args:
        bbox (Gf.BBox3d): The bounding box to set the matrix on.
        matrix (Gf.Matrix4d): The matrix to set on the bounding box.
    """
    if not isinstance(bbox, Gf.BBox3d):
        raise TypeError("Input bbox must be of type Gf.BBox3d")
    if not isinstance(matrix, Gf.Matrix4d):
        raise TypeError("Input matrix must be of type Gf.Matrix4d")
    bbox.SetMatrix(matrix)


def set_bbox_range(bbox: Gf.BBox3d, range: Gf.Range3d) -> None:
    """Set the range of an axis-aligned bounding box.

    This function sets the range of the given bounding box without modifying its transformation matrix.

    Args:
        bbox (Gf.BBox3d): The bounding box to modify.
        range (Gf.Range3d): The new range to set for the bounding box.

    Raises:
        TypeError: If the input arguments are not of the expected types.
    """
    if not isinstance(bbox, Gf.BBox3d):
        raise TypeError(f"Expected bbox to be of type Gf.BBox3d, got {type(bbox)}")
    if not isinstance(range, Gf.Range3d):
        raise TypeError(f"Expected range to be of type Gf.Range3d, got {type(range)}")
    bbox.SetRange(range)


def set_zero_area_primitives_flag(bbox: Gf.BBox3d, has_zero_area: bool) -> None:
    """Set the zero-area primitives flag on a BBox3d.

    Args:
        bbox (Gf.BBox3d): The bounding box to set the flag on.
        has_zero_area (bool): The value to set the flag to.
    """
    if not isinstance(bbox, Gf.BBox3d):
        raise TypeError(f"Expected a Gf.BBox3d, got {type(bbox)}")
    bbox.SetHasZeroAreaPrimitives(has_zero_area)


def get_bbox_with_children(prim: Usd.Prim) -> Gf.BBox3d:
    """
    Compute the bounding box of a prim and all its children.

    Args:
        prim (Usd.Prim): The prim to compute the bounding box for.

    Returns:
        Gf.BBox3d: The combined bounding box of the prim and its children.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    prim_bbox = UsdGeom.Boundable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
    combined_bbox = Gf.BBox3d()
    combined_bbox = Gf.BBox3d.Combine(combined_bbox, prim_bbox)
    for child_prim in prim.GetChildren():
        child_bbox = get_bbox_with_children(child_prim)
        combined_bbox = Gf.BBox3d.Combine(combined_bbox, child_bbox)
    return combined_bbox


def compute_combined_centroid(b1: Gf.BBox3d, b2: Gf.BBox3d) -> Gf.Vec3d:
    """
    Computes the centroid of the combined bounding box of two input bounding boxes.

    Args:
        b1 (Gf.BBox3d): The first bounding box.
        b2 (Gf.BBox3d): The second bounding box.

    Returns:
        Gf.Vec3d: The centroid of the combined bounding box.
    """
    if b1.GetRange().IsEmpty() and b2.GetRange().IsEmpty():
        raise ValueError("Both input bounding boxes are empty.")
    elif b1.GetRange().IsEmpty():
        return b2.ComputeCentroid()
    elif b2.GetRange().IsEmpty():
        return b1.ComputeCentroid()
    combined_bbox = Gf.BBox3d.Combine(b1, b2)
    return combined_bbox.ComputeCentroid()


def get_combined_bbox_volume(bbox1: Gf.BBox3d, bbox2: Gf.BBox3d) -> float:
    """
    Compute the volume of the combined bounding box of two BBox3d objects.

    Args:
        bbox1 (Gf.BBox3d): The first bounding box.
        bbox2 (Gf.BBox3d): The second bounding box.

    Returns:
        float: The volume of the combined bounding box.
    """
    if bbox1.HasZeroAreaPrimitives() or bbox2.HasZeroAreaPrimitives():
        combined_bbox = Gf.BBox3d.Combine(bbox1, bbox2)
    else:
        range1 = bbox1.ComputeAlignedRange()
        range2 = bbox2.ComputeAlignedRange()
        combined_range = Gf.Range3d()
        combined_range.UnionWith(range1)
        combined_range.UnionWith(range2)
        combined_bbox = Gf.BBox3d(combined_range, Gf.Matrix4d())
    return combined_bbox.GetVolume()


def fit_prim_to_bbox(prim: Usd.Prim, bbox: Gf.BBox3d):
    """
    Fit a prim to a given bounding box by applying a transform.

    Args:
        prim (Usd.Prim): The prim to fit to the bounding box.
        bbox (Gf.BBox3d): The target bounding box.

    Raises:
        ValueError: If the prim is not transformable or has no defined extent.
    """
    if not prim.IsA(UsdGeom.Xformable):
        raise ValueError("Prim is not transformable")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    prim_bbox = cache.ComputeWorldBound(prim)
    if not prim_bbox.GetRange().IsEmpty():
        prim_range = prim_bbox.ComputeAlignedRange()
        prim_size = prim_range.GetSize()
        target_range = bbox.ComputeAlignedRange()
        target_size = target_range.GetSize()
        scale_factors = [target_size[i] / prim_size[i] if prim_size[i] != 0 else 1 for i in range(3)]
        scale = Gf.Vec3f(*scale_factors)
        prim_centroid = prim_bbox.ComputeCentroid()
        target_centroid = bbox.ComputeCentroid()
        translate = target_centroid - prim_centroid
        xformable = UsdGeom.Xformable(prim)
        add_scale_op(xformable).Set(scale)
        add_translate_op(xformable).Set(translate)
    else:
        raise ValueError("Prim has no defined extent")


def align_bbox_to_world(bbox: Gf.BBox3d) -> Gf.BBox3d:
    """Align the bounding box to the world coordinate system.

    This function takes a possibly transformed bounding box and returns a new
    bounding box that is axis-aligned in world space. The resulting box will
    fully contain the original box.

    Args:
        bbox (Gf.BBox3d): The input bounding box to align.

    Returns:
        Gf.BBox3d: The axis-aligned bounding box in world space.
    """
    if bbox.GetRange().IsEmpty():
        return Gf.BBox3d()
    local_corners = [bbox.GetRange().GetCorner(i) for i in range(8)]
    world_corners = [bbox.GetMatrix().Transform(c) for c in local_corners]
    world_range = Gf.Range3d()
    for corner in world_corners:
        world_range.UnionWith(corner)
    aligned_bbox = Gf.BBox3d(world_range, Gf.Matrix4d(1))
    aligned_bbox.SetHasZeroAreaPrimitives(bbox.HasZeroAreaPrimitives())
    return aligned_bbox


def compute_aligned_bbox(bbox: Gf.BBox3d) -> Gf.Range3d:
    """Compute the axis-aligned bounding box from an arbitrarily oriented BBox3d."""
    if bbox.GetRange().IsEmpty():
        raise ValueError("Input BBox3d is empty.")
    matrix = bbox.GetMatrix()
    range_3d = bbox.GetRange()
    aligned_bbox = Gf.BBox3d(range_3d, Gf.Matrix4d())
    aligned_bbox.Transform(matrix)
    aligned_range = aligned_bbox.ComputeAlignedRange()
    return aligned_range


def transform_bbox(bbox: Gf.Range3f, mat: Gf.Matrix4d) -> Gf.Range3f:
    """
    Transform a bounding box by a matrix.

    Args:
        bbox (Gf.Range3f): The input bounding box.
        mat (Gf.Matrix4d): The transformation matrix.

    Returns:
        Gf.Range3f: The transformed bounding box.
    """
    if bbox.IsEmpty():
        return Gf.Range3f()
    transformed_bbox = Gf.Range3f()
    for i in range(8):
        corner = bbox.GetCorner(i)
        transformed_corner = mat.Transform(corner)
        transformed_bbox.UnionWith(transformed_corner)
    return transformed_bbox


def set_combined_bbox_matrix(b1: Gf.BBox3d, b2: Gf.BBox3d) -> Gf.BBox3d:
    """
    Combines two bboxes and sets the matrix of the result to the matrix
    that produces the smaller of the two resulting boxes.

    Args:
        b1 (Gf.BBox3d): First bounding box.
        b2 (Gf.BBox3d): Second bounding box.

    Returns:
        Gf.BBox3d: Combined bounding box with optimized matrix.
    """
    combined_b1 = Gf.BBox3d.Combine(b1, b2)
    combined_b2 = Gf.BBox3d.Combine(b2, b1)
    if combined_b1.GetVolume() < combined_b2.GetVolume():
        combined_b1.SetMatrix(b1.GetMatrix())
        return combined_b1
    else:
        combined_b2.SetMatrix(b2.GetMatrix())
        return combined_b2


def get_inverse_bbox_matrix(bbox: Gf.BBox3d) -> Gf.Matrix4d:
    """Get the inverse matrix of a bounding box.

    Args:
        bbox (Gf.BBox3d): The bounding box to get the inverse matrix of.

    Returns:
        Gf.Matrix4d: The inverse matrix of the bounding box.
    """
    if bbox.GetRange().IsEmpty():
        raise ValueError("The bounding box is empty.")
    matrix = bbox.GetMatrix()
    try:
        inverse_matrix = matrix.GetInverse()
        return inverse_matrix
    except Gf.UsdCpp.ArithmeticError:
        return Gf.Matrix4d(1.0)


def get_transformed_bbox(bbox: Gf.BBox3d) -> Gf.BBox3d:
    """
    Get the transformed bounding box by applying the stored transformation matrix.

    Args:
        bbox (Gf.BBox3d): The input bounding box.

    Returns:
        Gf.BBox3d: The transformed bounding box.
    """
    if bbox.GetRange().IsEmpty():
        raise ValueError("Bounding box range is empty.")
    matrix = bbox.GetMatrix()
    if matrix.GetDeterminant() == 0:
        raise ValueError("Transformation matrix is not invertible.")
    transformed_bbox = Gf.BBox3d(bbox.GetRange())
    transformed_bbox.Transform(matrix)
    return transformed_bbox


def adjust_camera_clipping_planes(camera: Gf.Camera, near: float, far: float) -> None:
    """Adjust the clipping planes of a camera.

    Args:
        camera (Gf.Camera): The camera to adjust the clipping planes of.
        near (float): The near clipping plane distance.
        far (float): The far clipping plane distance.

    Raises:
        ValueError: If the near clipping plane is not less than the far clipping plane.
    """
    if near >= far:
        raise ValueError("Near clipping plane must be less than far clipping plane.")
    camera.clippingRange = Gf.Range1f(near, far)


def apply_post_processing_to_camera(
    camera: Gf.Camera, fstop: float, focus_distance: float, focal_length: float
) -> None:
    """Apply post-processing settings to a Gf.Camera.

    Args:
        camera (Gf.Camera): The camera to modify.
        fstop (float): The f-stop value for depth of field.
        focus_distance (float): The focus distance in world units.
        focal_length (float): The focal length in tenths of a world unit.

    Raises:
        ValueError: If any of the input parameters are invalid.
    """
    if fstop <= 0:
        raise ValueError("fstop must be greater than 0.")
    if focus_distance <= 0:
        raise ValueError("focus_distance must be greater than 0.")
    if focal_length <= 0:
        raise ValueError("focal_length must be greater than 0.")
    camera.fStop = fstop
    camera.focusDistance = focus_distance
    camera.focalLength = focal_length
    near = camera.clippingRange.min
    far = camera.clippingRange.max
    if focus_distance < near or focus_distance > far:
        near = min(near, focus_distance * 0.8)
        far = max(far, focus_distance * 1.2)
        camera.clippingRange = Gf.Range1f(near, far)


def match_camera_fov(
    camera: Gf.Camera, fov: float, direction: Gf.Camera.FOVDirection = Gf.Camera.FOVHorizontal
) -> None:
    """
    Match the camera's field of view to the given value.

    Args:
        camera (Gf.Camera): The camera to modify.
        fov (float): The desired field of view in degrees.
        direction (Gf.Camera.FOVDirection): The direction of the field of view (horizontal or vertical).
    """
    aspect_ratio = camera.aspectRatio
    camera.SetPerspectiveFromAspectRatioAndFieldOfView(
        aspectRatio=aspect_ratio, fieldOfView=fov, direction=direction, horizontalAperture=camera.horizontalAperture
    )


def get_camera_focal_length(camera: Gf.Camera) -> float:
    """Get the focal length of a Gf.Camera.

    Args:
        camera (Gf.Camera): The input camera.

    Returns:
        float: The focal length of the camera in tenths of world units.
    """
    if not isinstance(camera, Gf.Camera):
        raise TypeError("Input must be a Gf.Camera")
    focal_length = camera.focalLength
    return focal_length


def set_camera_focal_length(camera: Gf.Camera, focal_length: float) -> None:
    """Set the focal length of a Gf.Camera.

    Args:
        camera (Gf.Camera): The camera to modify.
        focal_length (float): The new focal length in tenths of world units.
    """
    if focal_length <= 0:
        raise ValueError(f"Focal length must be positive, got {focal_length}")
    camera.focalLength = focal_length
    aspect_ratio = camera.horizontalAperture / camera.verticalAperture
    fov_direction = Gf.Camera.FOVVertical
    camera.SetPerspectiveFromAspectRatioAndFieldOfView(
        aspect_ratio, camera.GetFieldOfView(fov_direction), fov_direction, camera.horizontalAperture
    )


def set_camera_focus_distance(camera: Gf.Camera, focus_distance: float) -> None:
    """Set the focus distance of a camera in world units.

    Args:
        camera (Gf.Camera): The camera to set the focus distance on.
        focus_distance (float): The focus distance in world units.

    Raises:
        TypeError: If camera is not a Gf.Camera.
        ValueError: If focus_distance is negative.
    """
    if not isinstance(camera, Gf.Camera):
        raise TypeError("Expected a Gf.Camera object")
    if focus_distance < 0:
        raise ValueError("Focus distance cannot be negative")
    camera.focusDistance = focus_distance


def get_camera_aperture(camera: Gf.Camera) -> Tuple[float, float]:
    """Get the horizontal and vertical aperture of a camera.

    Args:
        camera (Gf.Camera): The camera to get the aperture from.

    Returns:
        Tuple[float, float]: A tuple containing the horizontal and vertical aperture.
    """
    horizontal_aperture = camera.horizontalAperture
    vertical_aperture = camera.verticalAperture
    return (horizontal_aperture, vertical_aperture)


def set_camera_aperture(camera: Gf.Camera, horizontal_aperture: float, vertical_aperture: float) -> None:
    """
    Set the horizontal and vertical aperture of a camera.

    Args:
        camera (Gf.Camera): The camera to modify.
        horizontal_aperture (float): The horizontal aperture in tenths of a world unit.
        vertical_aperture (float): The vertical aperture in tenths of a world unit.

    Raises:
        ValueError: If either aperture value is less than or equal to zero.
    """
    if horizontal_aperture <= 0 or vertical_aperture <= 0:
        raise ValueError("Aperture values must be greater than zero.")
    camera.horizontalAperture = horizontal_aperture
    camera.verticalAperture = vertical_aperture


def adjust_camera_f_stop(camera: Gf.Camera, f_stop: float) -> None:
    """Adjust the f-stop of a camera.

    Parameters:
        camera (Gf.Camera): The camera to adjust.
        f_stop (float): The new f-stop value.

    Raises:
        ValueError: If the provided f_stop is not positive.
    """
    if f_stop <= 0:
        raise ValueError("F-stop must be a positive value.")
    camera.fStop = f_stop


def get_camera_view_matrix(camera: Gf.Camera) -> Gf.Matrix4d:
    """Returns the view matrix for the given camera.

    The view matrix is the inverse of the camera's transform matrix.
    """
    transform_matrix = camera.transform
    if not transform_matrix.GetDeterminant():
        raise ValueError("Camera transform matrix is not invertible.")
    view_matrix = transform_matrix.GetInverse()
    return view_matrix


def set_camera_perspective(
    camera: Gf.Camera,
    aspect_ratio: float,
    field_of_view: float,
    direction: Gf.Camera.FOVDirection = Gf.Camera.FOVVertical,
    horizontal_aperture: float = 2.4,
) -> None:
    """
    Set the camera's perspective projection parameters.

    Args:
        camera (Gf.Camera): The camera to set the perspective projection for.
        aspect_ratio (float): The aspect ratio of the camera's projection.
        field_of_view (float): The field of view in degrees.
        direction (Gf.Camera.FOVDirection, optional): The direction of the field of view.
            Defaults to Gf.Camera.FOVVertical.
        horizontal_aperture (float, optional): The horizontal aperture size in cm. Defaults to 2.4.

    Raises:
        ValueError: If the aspect ratio or field of view is not positive.
    """
    if aspect_ratio <= 0:
        raise ValueError("Aspect ratio must be positive.")
    if field_of_view <= 0:
        raise ValueError("Field of view must be positive.")
    camera.SetPerspectiveFromAspectRatioAndFieldOfView(
        aspectRatio=aspect_ratio, fieldOfView=field_of_view, direction=direction, horizontalAperture=horizontal_aperture
    )


def set_camera_transform_from_look_at(
    camera: Gf.Camera, look_at_point: Gf.Vec3d, look_from_point: Gf.Vec3d, up_vector: Gf.Vec3d = Gf.Vec3d(0, 1, 0)
):
    """Set camera transform based on look-at and look-from points.

    Args:
        camera (Gf.Camera): Camera to set transform for.
        look_at_point (Gf.Vec3d): Point the camera should look at.
        look_from_point (Gf.Vec3d): Point the camera should look from.
        up_vector (Gf.Vec3d): Up vector for the camera. Defaults to (0, 1, 0).
    """
    forward = look_at_point - look_from_point
    forward.Normalize()
    right = Gf.Cross(forward, up_vector)
    right.Normalize()
    up = Gf.Cross(right, forward)
    rotation_matrix = Gf.Matrix3d(
        right[0], up[0], -forward[0], right[1], up[1], -forward[1], right[2], up[2], -forward[2]
    )
    transform_matrix = Gf.Matrix4d().SetIdentity()
    transform_matrix.SetRotate(rotation_matrix)
    transform_matrix.SetTranslateOnly(look_from_point)
    camera.transform = transform_matrix


def calculate_camera_frustum(camera: Gf.Camera) -> Gf.Frustum:
    """Calculate and return the camera frustum.

    Args:
        camera (Gf.Camera): The input camera.

    Returns:
        Gf.Frustum: The computed camera frustum.
    """
    if not camera:
        raise ValueError("Invalid camera provided.")
    projection = camera.projection
    if projection not in [Gf.Camera.Perspective, Gf.Camera.Orthographic]:
        raise ValueError(f"Unsupported projection type: {projection}")
    transform = camera.transform
    clipping_range = camera.clippingRange
    near = clipping_range.min
    far = clipping_range.max
    frustum = Gf.Frustum()
    if projection == Gf.Camera.Perspective:
        fov = camera.GetFieldOfView(Gf.Camera.FOVHorizontal)
        aspect_ratio = camera.aspectRatio
        frustum.SetPerspective(fov, True, aspect_ratio, near, far)
    else:
        horizontal_aperture = camera.horizontalAperture
        vertical_aperture = camera.verticalAperture
        aspect_ratio = horizontal_aperture / vertical_aperture
        frustum.SetOrthographic(vertical_aperture, aspect_ratio, near, far)
    frustum.Transform(transform)
    return frustum


def configure_camera_fov(
    camera: UsdGeom.Camera, fov: float, direction: Gf.Camera.FOVDirection = Gf.Camera.FOVHorizontal
) -> None:
    """Configure the field of view for a camera.

    Args:
        camera (UsdGeom.Camera): The camera to configure.
        fov (float): The field of view value in degrees.
        direction (Gf.Camera.FOVDirection, optional): The direction of the FOV. Defaults to Gf.Camera.FOVHorizontal.

    Raises:
        ValueError: If the camera is not valid or the FOV is not in the valid range (0, 180).
    """
    if not camera:
        raise ValueError("Invalid camera")
    if fov <= 0 or fov >= 180:
        raise ValueError(f"Invalid FOV value: {fov}. Must be between 0 and 180 degrees.")
    camera.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
    if direction == Gf.Camera.FOVHorizontal:
        camera.GetHorizontalApertureAttr().Set(fov)
    elif direction == Gf.Camera.FOVVertical:
        camera.GetVerticalApertureAttr().Set(fov)


def set_camera_projection(camera: UsdGeom.Camera, projection_type: str) -> None:
    """Set the projection type for a USD camera.

    Args:
        camera (UsdGeom.Camera): The USD camera prim.
        projection_type (str): The projection type to set. Valid values are "perspective" or "orthographic".

    Raises:
        ValueError: If the camera prim is not valid or if an invalid projection type is provided.
    """
    if not camera.GetPrim().IsValid():
        raise ValueError("Invalid camera prim.")
    projection_attr = camera.GetProjectionAttr()
    if projection_type not in ["perspective", "orthographic"]:
        raise ValueError(f"Invalid projection type: {projection_type}")
    projection_attr.Set(projection_type)


def apply_dual_quat_transform_to_prim(prim: Usd.Prim, dual_quat: Gf.DualQuatd):
    """Apply a dual quaternion transformation to a USD prim.

    Args:
        prim (Usd.Prim): The USD prim to transform.
        dual_quat (Gf.DualQuatd): The dual quaternion representing the transformation.

    Raises:
        ValueError: If the input prim is not valid or not transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim '{prim.GetPath()}' is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim '{prim.GetPath()}' is not transformable.")
    rotation = dual_quat.GetReal()
    translation = dual_quat.GetTranslation()
    rot_op = xformable.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ)
    rot_op.Set(Gf.Vec3d(rotation.GetImaginary()))
    trans_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTranslate)
    trans_op.Set(translation)


def extract_translation_from_dual_quat(dual_quat: Gf.DualQuatd) -> Gf.Vec3d:
    """Extract the translation component from a dual quaternion.

    Args:
        dual_quat (Gf.DualQuatd): The input dual quaternion.

    Returns:
        Gf.Vec3d: The translation component of the dual quaternion.
    """
    real_quat = dual_quat.GetReal()
    dual_quat = dual_quat.GetDual()
    inv_real_quat = real_quat.GetInverse()
    translation_quat = dual_quat * inv_real_quat
    translation_quat *= 2.0
    return Gf.Vec3d(translation_quat.GetImaginary())


def transform_points_using_dual_quat(dual_quat: Gf.DualQuatd, points: List[Gf.Vec3d]) -> List[Gf.Vec3d]:
    """
    Transform a list of points using a dual quaternion.

    Args:
        dual_quat (Gf.DualQuatd): The dual quaternion to use for the transformation.
        points (List[Gf.Vec3d]): The list of points to transform.

    Returns:
        List[Gf.Vec3d]: The list of transformed points.
    """
    length = dual_quat.GetLength()
    if not Gf.IsClose(length[0], 1.0, 1e-06) or not Gf.IsClose(length[1], 0.0, 1e-06):
        dual_quat = dual_quat.GetNormalized(1e-06)
    transformed_points = [dual_quat.Transform(point) for point in points]
    return transformed_points


def copy_dual_quat_transform(src_transform: Gf.DualQuatd, dst_transform: Gf.DualQuatd) -> None:
    """Copy the transformation from one DualQuatd to another.

    Args:
        src_transform (Gf.DualQuatd): The source transformation to copy from.
        dst_transform (Gf.DualQuatd): The destination transformation to copy to.
    """
    if not isinstance(src_transform, Gf.DualQuatd) or not isinstance(dst_transform, Gf.DualQuatd):
        raise TypeError("Both src_transform and dst_transform must be of type Gf.DualQuatd.")
    src_real = src_transform.GetReal()
    src_dual = src_transform.GetDual()
    dst_transform.SetReal(src_real)
    dst_transform.SetDual(src_dual)


def apply_cumulative_dual_quat_transforms(transforms: List[Gf.DualQuatd], point: Gf.Vec3d) -> Gf.Vec3d:
    """
    Apply a list of dual quaternion transforms to a 3D point in a cumulative manner.

    Args:
        transforms (List[Gf.DualQuatd]): A list of dual quaternion transforms to apply.
        point (Gf.Vec3d): The 3D point to transform.

    Returns:
        Gf.Vec3d: The transformed 3D point.
    """
    if not transforms:
        return point
    cumulative_transform = Gf.DualQuatd()
    for transform in transforms:
        cumulative_transform *= transform
    cumulative_transform.Normalize()
    transformed_point = cumulative_transform.Transform(point)
    return transformed_point


def compute_dual_quat_length(dual_quat: Gf.DualQuatd) -> Tuple[float, float]:
    """Compute the geometric length of a dual quaternion.

    Args:
        dual_quat (Gf.DualQuatd): The input dual quaternion.

    Returns:
        Tuple[float, float]: The geometric length of the dual quaternion.
    """
    length = dual_quat.GetLength()
    (real_length, dual_length) = length
    return (real_length, dual_length)


def blend_prims_dual_quat_transforms(stage: Usd.Stage, prim_paths: List[str], weights: List[float]) -> Gf.DualQuatd:
    """Blends the transforms of multiple prims using dual quaternion skinning.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The paths of the prims to blend.
        weights (List[float]): The weights for each prim transform. Should be the same length as prim_paths.

    Returns:
        Gf.DualQuatd: The blended dual quaternion transform.

    Raises:
        ValueError: If the number of weights does not match the number of prim paths.
    """
    if len(prim_paths) != len(weights):
        raise ValueError("Number of weights must match number of prim paths.")
    blended_dual_quat = Gf.DualQuatd.GetIdentity()
    for prim_path, weight in zip(prim_paths, weights):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        xformable = UsdGeom.Xformable(prim)
        transform_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        transform_dual_quat = Gf.DualQuatd(
            Gf.Quatd(transform_matrix.ExtractRotationQuat()), Gf.Quatd(0, transform_matrix.ExtractTranslation())
        )
        weighted_dual_quat = transform_dual_quat * weight
        blended_dual_quat += weighted_dual_quat
    blended_dual_quat = blended_dual_quat.GetNormalized()
    return blended_dual_quat


def get_prim_dual_quat_transform(stage: Usd.Stage, prim_path: str) -> Gf.DualQuatd:
    """Get the transform of a prim as a dual quaternion.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path of the prim.

    Returns:
        Gf.DualQuatd: The transform of the prim as a dual quaternion.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    local_transform = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    (translation, rotation, scale) = UsdSkel.DecomposeTransform(local_transform)
    real_quat = Gf.Quatd(rotation)
    dual_quat = Gf.DualQuatd()
    dual_quat.SetReal(real_quat)
    dual_quat.SetTranslation(Gf.Vec3d(translation))
    return dual_quat


def set_prim_translation_using_dual_quat(prim: Usd.Prim, translation: Gf.Vec3d) -> None:
    """
    Set the translation of a prim using a dual quaternion.

    Args:
        prim (Usd.Prim): The prim to set the translation for.
        translation (Gf.Vec3d): The translation to set.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    dual_quat = Gf.DualQuatd()
    dual_quat.SetTranslation(translation)
    xform_op = add_translate_op(xformable)
    xform_op.Set(translation)


def get_dual_quatf_transform(translation: Gf.Vec3f, rotation: Gf.Quatf) -> Gf.DualQuatf:
    """
    Create a DualQuatf from a translation vector and rotation quaternion.

    Args:
        translation (Gf.Vec3f): The translation vector.
        rotation (Gf.Quatf): The rotation quaternion.

    Returns:
        Gf.DualQuatf: The dual quaternion representing the transformation.
    """
    rotation_norm = Gf.Quatf(rotation).GetNormalized()
    dual_quat = Gf.DualQuatf()
    dual_quat.SetReal(rotation_norm)
    dual_vec = Gf.Quatf(0, translation[0], translation[1], translation[2])
    dual_part = 0.5 * (dual_vec * rotation_norm)
    dual_quat.SetDual(dual_part)
    return dual_quat


def set_prim_rigid_transform(stage: Usd.Stage, prim_path: str, transform: Gf.DualQuatf) -> None:
    """Set the rigid transform for a prim using a DualQuatf.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path of the prim to set the transform for.
        transform (Gf.DualQuatf): The dual quaternion representing the rigid transform.

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    xformable.ClearXformOpOrder()
    translation = transform.GetTranslation()
    add_translate_op(xformable).Set(translation)
    rotation = transform.GetReal()
    add_orient_op(xformable).Set(rotation)


def extract_translation_from_dual_quatf(dual_quat: Gf.DualQuatf) -> Gf.Vec3f:
    """Extract the translation component from a dual quaternion.

    Args:
        dual_quat (Gf.DualQuatf): The input dual quaternion.

    Returns:
        Gf.Vec3f: The translation component of the dual quaternion.
    """
    real_quat = dual_quat.GetReal()
    dual_quat = dual_quat.GetDual()
    translation = 2.0 * (
        dual_quat.GetImaginary() * real_quat.GetReal() - real_quat.GetImaginary() * dual_quat.GetReal()
    )
    return translation


def extract_rotation_from_dual_quatf(dual_quat: Gf.DualQuatf) -> Gf.Quatf:
    """Extract the rotation quaternion from a dual quaternion.

    Args:
        dual_quat (Gf.DualQuatf): The input dual quaternion.

    Returns:
        Gf.Quatf: The rotation quaternion extracted from the dual quaternion.
    """
    normalized_dual_quat = dual_quat.GetNormalized(eps=1e-06)
    rotation_quat = normalized_dual_quat.GetReal()
    return rotation_quat


def set_dual_quatf_from_transform(transform: Gf.Matrix4d) -> Gf.DualQuatf:
    """
    Set a DualQuatf from a transform matrix.

    Args:
        transform (Gf.Matrix4d): The transform matrix to set the DualQuatf from.

    Returns:
        Gf.DualQuatf: The DualQuatf representing the transform.
    """
    rotation = Gf.Quatf(transform.ExtractRotationQuat())
    translation = Gf.Vec3f(transform.ExtractTranslation())
    dual_quat = Gf.DualQuatf()
    dual_quat.SetReal(rotation)
    dual_quat.SetTranslation(translation)
    return dual_quat


def dual_quatf_to_matrix(dual_quat: Gf.DualQuatf) -> Gf.Matrix4d:
    """Converts a DualQuatf to a Matrix4d."""
    real_quat = dual_quat.GetReal()
    dual_quat = dual_quat.GetDual()
    rot_matrix = Gf.Matrix3f(Gf.Quatf(real_quat))
    translation = Gf.Vec3f(
        2.0
        * (
            -dual_quat.GetReal() * real_quat.GetImaginary()[0]
            + real_quat.GetReal() * dual_quat.GetImaginary()[0]
            - real_quat.GetImaginary()[1] * dual_quat.GetImaginary()[2]
            + real_quat.GetImaginary()[2] * dual_quat.GetImaginary()[1]
        ),
        2.0
        * (
            -dual_quat.GetReal() * real_quat.GetImaginary()[1]
            + real_quat.GetReal() * dual_quat.GetImaginary()[1]
            + real_quat.GetImaginary()[0] * dual_quat.GetImaginary()[2]
            - real_quat.GetImaginary()[2] * dual_quat.GetImaginary()[0]
        ),
        2.0
        * (
            -dual_quat.GetReal() * real_quat.GetImaginary()[2]
            + real_quat.GetReal() * dual_quat.GetImaginary()[2]
            - real_quat.GetImaginary()[0] * dual_quat.GetImaginary()[1]
            + real_quat.GetImaginary()[1] * dual_quat.GetImaginary()[0]
        ),
    )
    matrix = Gf.Matrix4d(
        rot_matrix[0][0],
        rot_matrix[0][1],
        rot_matrix[0][2],
        0,
        rot_matrix[1][0],
        rot_matrix[1][1],
        rot_matrix[1][2],
        0,
        rot_matrix[2][0],
        rot_matrix[2][1],
        rot_matrix[2][2],
        0,
        translation[0],
        translation[1],
        translation[2],
        1,
    )
    return matrix


def matrix_to_dual_quatf(matrix: Gf.Matrix4d) -> Gf.DualQuatf:
    """
    Convert a 4x4 matrix to a dual quaternion.

    Args:
        matrix (Gf.Matrix4d): The input 4x4 matrix.

    Returns:
        Gf.DualQuatf: The resulting dual quaternion.
    """
    rotation = matrix.ExtractRotationQuat()
    translation = Gf.Vec3f(matrix.ExtractTranslation())
    real_quat = Gf.Quatf(rotation)
    dual_quat = Gf.Quatf(
        0.5
        * (
            translation[0] * real_quat.GetImaginary()[0]
            + translation[1] * real_quat.GetImaginary()[1]
            + translation[2] * real_quat.GetImaginary()[2]
        ),
        0.5 * translation[0] * real_quat.GetReal(),
        0.5 * translation[1] * real_quat.GetReal(),
        0.5 * translation[2] * real_quat.GetReal(),
    )
    return Gf.DualQuatf(real_quat, dual_quat)


def apply_dual_quatf_to_prim(prim: Usd.Prim, dual_quat: Gf.DualQuatf) -> None:
    """Apply a dual quaternion transformation to a USD prim.

    Args:
        prim (Usd.Prim): The USD prim to apply the transformation to.
        dual_quat (Gf.DualQuatf): The dual quaternion representing the transformation.

    Raises:
        ValueError: If the provided prim is not a valid transformable prim.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim} is not transformable")
    rotation = dual_quat.GetReal()
    translation = dual_quat.GetTranslation()
    euler_angles = Gf.Rotation(rotation).Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
    add_rotate_xyz_op(xformable).Set(euler_angles)
    add_translate_op(xformable).Set(translation)


def interpolate_dual_quatf(dq1: Gf.DualQuatf, dq2: Gf.DualQuatf, weight: float) -> Gf.DualQuatf:
    """Interpolate between two dual quaternions.

    Args:
        dq1 (Gf.DualQuatf): The first dual quaternion.
        dq2 (Gf.DualQuatf): The second dual quaternion.
        weight (float): The interpolation weight in the range [0, 1].

    Returns:
        Gf.DualQuatf: The interpolated dual quaternion.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"Interpolation weight must be in the range [0, 1]. Got {weight}.")
    real_dot = Gf.Dot(dq1.real, dq2.real)
    if real_dot < 0:
        dq1 = Gf.DualQuatf(-dq1.real, -dq1.dual)
        real_dot = -real_dot
    if real_dot < 0.9995:
        angle = acos(Gf.Clamp(real_dot, -1, 1))
        angle_sin = sin(angle)
        scale0 = sin((1.0 - weight) * angle) / angle_sin
        scale1 = sin(weight * angle) / angle_sin
    else:
        scale0 = 1.0 - weight
        scale1 = weight
    result = Gf.DualQuatf(scale0 * dq1.real + scale1 * dq2.real, scale0 * dq1.dual + scale1 * dq2.dual)
    result.Normalize(eps=1e-06)
    return result


def blend_dual_quatf_transforms(a: Gf.DualQuatf, b: Gf.DualQuatf, t: float) -> Gf.DualQuatf:
    """Blend two dual quaternion transforms by parameter t.

    The blending is done using spherical linear interpolation (SLERP) on the real
    part and linear interpolation on the dual part.

    Args:
        a (Gf.DualQuatf): The first dual quaternion transform.
        b (Gf.DualQuatf): The second dual quaternion transform.
        t (float): The blending parameter in the range [0, 1].

    Returns:
        Gf.DualQuatf: The blended dual quaternion transform.
    """
    if t < 0 or t > 1:
        raise ValueError(f"Blending parameter t must be in the range [0, 1], got {t}")
    real_part = Gf.Slerp(t, a.real, b.real)
    dual_part = Gf.Quatf(
        Gf.Lerp(t, a.dual.GetReal(), b.dual.GetReal()), Gf.Lerp(t, a.dual.GetImaginary(), b.dual.GetImaginary())
    )
    result = Gf.DualQuatf()
    result.SetReal(real_part)
    result.SetDual(dual_part)
    return result


def conjugate_prim_dual_quatf(prim: Usd.Prim) -> Gf.DualQuatf:
    """
    Conjugate the dual quaternion of a prim.

    Args:
        prim (Usd.Prim): The prim to conjugate the dual quaternion of.

    Returns:
        Gf.DualQuatf: The conjugated dual quaternion.

    Raises:
        ValueError: If the prim is not valid or does not have a dual quaternion.
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError("Prim does not have an xformable")
    local_transform = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    rotation = Gf.Quatf(local_transform.ExtractRotationQuat())
    translation = Gf.Vec3f(local_transform.ExtractTranslation())
    dual_quat = Gf.DualQuatf(rotation, translation)
    conjugated_dual_quat = dual_quat.GetConjugate()
    return conjugated_dual_quat


def inverse_prim_dual_quatf(stage: Usd.Stage, prim_path: str) -> Tuple[Gf.DualQuatf, bool]:
    """
    Compute the inverse dual quaternion of a prim's transform.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path of the prim.

    Returns:
        A tuple containing:
        - The inverse dual quaternion (Gf.DualQuatf)
        - A boolean indicating if the prim had a valid transform.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return (Gf.DualQuatf(1.0), False)
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return (Gf.DualQuatf(1.0), False)
    transform_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rotation = Gf.Quatf(transform_matrix.ExtractRotationQuat())
    translation = Gf.Vec3f(transform_matrix.ExtractTranslation())
    real_part = rotation
    dual_part = Gf.Quatf(0.5 * (Gf.Quatf(0, translation) * rotation))
    dual_quat = Gf.DualQuatf(real_part, dual_part)
    inverse_dual_quat = dual_quat.GetInverse()
    return (inverse_dual_quat, True)


def bake_dual_quatf_animation(
    stage: Usd.Stage,
    prim_path: str,
    attribute_name: str,
    time_samples: Sequence[float],
    dual_quatf_samples: Sequence[Gf.DualQuatf],
) -> None:
    """Bake a DualQuatf animation onto a prim attribute.

    Args:
        stage (Usd.Stage): The USD stage to write the animation to.
        prim_path (str): The path to the prim to write the animation to.
        attribute_name (str): The name of the attribute to write the animation to.
        time_samples (Sequence[float]): A sequence of time samples for the animation.
        dual_quatf_samples (Sequence[Gf.DualQuatf]): A sequence of DualQuatf samples for the animation.
    """
    if len(time_samples) != len(dual_quatf_samples):
        raise ValueError("time_samples and dual_quatf_samples must have the same length")
    if len(time_samples) == 0:
        raise ValueError("time_samples and dual_quatf_samples cannot be empty")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim not found at path: {prim_path}")
    if not prim.HasAttribute(attribute_name):
        prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Quatf)
    attr = prim.GetAttribute(attribute_name)
    if not attr:
        raise ValueError(f"Attribute {attribute_name} not found on prim {prim_path}")
    attr.Clear()
    for time, value in zip(time_samples, dual_quatf_samples):
        attr.Set(value.real, Usd.TimeCode(time))


def copy_prim_transform_to_another(source_prim: Usd.Prim, target_prim: Usd.Prim):
    """Copy the transform from one prim to another."""
    if not source_prim.IsValid():
        raise ValueError("Source prim is not valid.")
    if not target_prim.IsValid():
        raise ValueError("Target prim is not valid.")
    source_xformable = UsdGeom.Xformable(source_prim)
    target_xformable = UsdGeom.Xformable(target_prim)
    if not source_xformable:
        raise ValueError("Source prim is not transformable.")
    if not target_xformable:
        raise ValueError("Target prim is not transformable.")
    source_xform_ops = source_xformable.GetOrderedXformOps()
    target_xformable.ClearXformOpOrder()
    for op in source_xform_ops:
        op_type = op.GetOpType()
        op_value = op.Get()
        target_op = target_xformable.AddXformOp(op_type)
        if op_value is not None:
            target_op.Set(op_value)


def extract_translation_from_dual_quath(dual_quath: Gf.DualQuath) -> Gf.Vec3h:
    """Extract the translation component from a dual quaternion.

    Args:
        dual_quath (Gf.DualQuath): The input dual quaternion.

    Returns:
        Gf.Vec3h: The translation component of the dual quaternion.
    """
    if not isinstance(dual_quath, Gf.DualQuath):
        raise TypeError("Input must be a Gf.DualQuath")
    translation = dual_quath.GetTranslation()
    return translation


def transform_point_with_prim_dual_quath(stage: Usd.Stage, prim_path: str, point: Gf.Vec3f) -> Gf.Vec3f:
    """
    Transform a point using the dual quaternion of a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        point (Gf.Vec3f): The point to transform.

    Returns:
        Gf.Vec3f: The transformed point.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim not found at path: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable")
    transform_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rotation = transform_matrix.ExtractRotationQuat()
    translation = transform_matrix.ExtractTranslation()
    dual_quath = Gf.DualQuath()
    dual_quath.SetReal(Gf.Quath(rotation))
    dual_quath.SetTranslation(Gf.Vec3h(translation))
    point_h = Gf.Vec3h(point)
    transformed_point_h = dual_quath.Transform(point_h)
    transformed_point = Gf.Vec3f(transformed_point_h)
    return transformed_point


def blend_prims_transformations(stage: Usd.Stage, prim_paths: List[str], weights: List[float]) -> Gf.Matrix4d:
    """Blend the transformations of multiple prims using weights.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The paths to the prims to blend.
        weights (List[float]): The weights for each prim transformation.

    Returns:
        Gf.Matrix4d: The blended transformation matrix.
    """
    if len(prim_paths) != len(weights):
        raise ValueError("Number of prim paths and weights must match.")
    if len(prim_paths) == 0:
        return Gf.Matrix4d()
    weight_sum = sum(weights)
    if weight_sum == 0:
        weights = [1.0 / len(weights)] * len(weights)
    else:
        weights = [w / weight_sum for w in weights]
    result = Gf.Matrix4d()
    for prim_path, weight in zip(prim_paths, weights):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            raise ValueError(f"Prim not found at path: {prim_path}")
        xform = UsdGeom.Xformable(prim)
        if not xform:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        result += transform * weight
    return result


def inverse_transform_of_prim(stage: Usd.Stage, prim_path: str) -> Gf.Matrix4d:
    """
    Get the inverse transform of a prim as a Gf.Matrix4d.

    Parameters:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.

    Returns:
        Gf.Matrix4d: The inverse transform of the prim.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    local_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return local_transform.GetInverse()


def create_dual_quath_from_transform(transform: Gf.Matrix4d) -> Gf.DualQuath:
    """Create a dual quaternion from a transform matrix.

    Args:
        transform (Gf.Matrix4d): The input transform matrix.

    Returns:
        Gf.DualQuath: The resulting dual quaternion.
    """
    rotation = transform.ExtractRotation()
    translation = transform.ExtractTranslation()
    real_quath = Gf.Quath(rotation.GetQuat())
    dual_quath = Gf.DualQuath(real_quath)
    dual_quath.SetTranslation(Gf.Vec3h(translation))
    return dual_quath


def extract_rotation_from_dual_quath(dual_quath: Gf.DualQuath) -> Gf.Quatf:
    """Extract the rotation component from a dual quaternion.

    Args:
        dual_quath (Gf.DualQuath): The input dual quaternion.

    Returns:
        Gf.Quatf: The rotation component as a regular quaternion.
    """
    real_quath = dual_quath.GetReal()
    real_quath = real_quath.GetNormalized()
    real_quatf = Gf.Quatf(
        real_quath.GetReal(), real_quath.GetImaginary()[0], real_quath.GetImaginary()[1], real_quath.GetImaginary()[2]
    )
    return real_quatf


def create_frustum_from_camera(
    camera_position: Gf.Vec3d, camera_rotation: Gf.Rotation, window: Gf.Range2d, near_far_range: Gf.Range1d
) -> Gf.Frustum:
    """Create a Gf.Frustum from camera parameters.

    Args:
        camera_position (Gf.Vec3d): The position of the camera in world space.
        camera_rotation (Gf.Rotation): The rotation of the camera in world space.
        window (Gf.Range2d): The window rectangle defining the view frustum.
        near_far_range (Gf.Range1d): The near and far clipping planes of the view frustum.

    Returns:
        Gf.Frustum: The created frustum.
    """
    frustum = Gf.Frustum()
    frustum.SetPosition(camera_position)
    frustum.SetRotation(camera_rotation)
    frustum.SetWindow(window)
    frustum.SetNearFar(near_far_range)
    return frustum


def set_frustum_rotation(frustum: Gf.Frustum, rotation: Gf.Rotation) -> None:
    """Set the rotation of a Frustum.

    Args:
        frustum (Gf.Frustum): The Frustum to modify.
        rotation (Gf.Rotation): The new rotation to set.
    """
    if not isinstance(frustum, Gf.Frustum):
        raise TypeError("Input 'frustum' must be of type Gf.Frustum")
    if not isinstance(rotation, Gf.Rotation):
        raise TypeError("Input 'rotation' must be of type Gf.Rotation")
    frustum.SetRotation(rotation)


def get_frustum_up_vector(frustum: Gf.Frustum) -> Gf.Vec3d:
    """Get the normalized world-space up vector of the frustum."""
    if not isinstance(frustum, Gf.Frustum):
        raise TypeError("Invalid input type. Expected Gf.Frustum.")
    up_vector = frustum.ComputeUpVector()
    return up_vector


def get_frustum_fov(frustum: Gf.Frustum) -> float:
    """Get the horizontal field of view of a frustum in degrees.

    If the frustum is not perspective, return 0.0.
    """
    if frustum.GetProjectionType() != Gf.Frustum.Perspective:
        return 0.0
    persp_data = frustum.GetPerspective()
    if persp_data is None:
        raise ValueError("Frustum is perspective but GetPerspective returned None.")
    (fov, aspect_ratio, near, far) = persp_data
    return fov


def narrow_frustum_to_point(frustum: Gf.Frustum, world_point: Gf.Vec3d, size: float = 0.1) -> Gf.Frustum:
    """
    Narrows the given frustum to a small window around a target point.

    Args:
        frustum (Gf.Frustum): The input frustum to narrow.
        world_point (Gf.Vec3d): The target point in world space.
        size (float): The size of the window around the target point. Default is 0.1.

    Returns:
        Gf.Frustum: The narrowed frustum.
    """
    view_dir = frustum.ComputeViewDirection()
    view_to_point = world_point - frustum.GetPosition()
    distance = Gf.Dot(view_to_point, view_dir)
    half_size = size / 2.0
    window = Gf.Range2d(Gf.Vec2d(-half_size, -half_size), Gf.Vec2d(half_size, half_size))
    narrowed = Gf.Frustum(frustum)
    narrowed.SetWindow(window)
    narrowed.SetNearFar(Gf.Range1d(distance, distance + 1.0))
    return narrowed


def get_frustum_projection_type(frustum: Gf.Frustum) -> Gf.Frustum.ProjectionType:
    """Get the projection type of a Gf.Frustum.

    Args:
        frustum (Gf.Frustum): The frustum to get the projection type from.

    Returns:
        Gf.Frustum.ProjectionType: The projection type of the frustum.
    """
    projection_type: Gf.Frustum.ProjectionType = frustum.GetProjectionType()
    return projection_type


def get_frustum_reference_plane_depth(frustum: Union[Gf.Frustum, None]) -> float:
    """Get the depth of the reference plane of a frustum.

    Args:
        frustum (Union[Gf.Frustum, None]): The input frustum. Can be None.

    Returns:
        float: The depth of the reference plane. Returns 0.0 if frustum is None.
    """
    if frustum is None:
        return 0.0
    depth = frustum.GetReferencePlaneDepth()
    return depth


def compute_frustum_view_volume_intersection(frustum: Gf.Frustum, bbox: Gf.BBox3d, vpMat: Gf.Matrix4d) -> bool:
    """Compute the intersection of a frustum's view volume with a bounding box.

    Args:
        frustum (Gf.Frustum): The frustum representing the view volume.
        bbox (Gf.BBox3d): The bounding box to test for intersection.
        vpMat (Gf.Matrix4d): The view-projection matrix.

    Returns:
        bool: True if the bounding box intersects the view volume, False otherwise.
    """
    if frustum.Intersects(bbox):
        return True
    if Gf.Frustum.IntersectsViewVolume(bbox, vpMat):
        return True
    return False


def set_frustum_view_matrix(frustum: Gf.Frustum, view_matrix: Gf.Matrix4d) -> None:
    """Set the view matrix of a frustum.

    Args:
        frustum (Gf.Frustum): The frustum to modify.
        view_matrix (Gf.Matrix4d): The view matrix to set on the frustum.
    """
    position = view_matrix.ExtractTranslation()
    rotation = view_matrix.ExtractRotation()
    frustum.SetPosition(position)
    frustum.SetRotation(rotation)


def compute_frustum_corners(frustum: Gf.Frustum) -> Tuple[Gf.Vec3d, ...]:
    """
    Compute the world-space corners of the frustum.

    Args:
        frustum (Gf.Frustum): The input frustum.

    Returns:
        Tuple[Gf.Vec3d, ...]: A tuple of 8 Vec3d points representing the corners of the frustum.
    """
    near_far = frustum.GetNearFar()
    near = near_far.min
    far = near_far.max
    corners_near = frustum.ComputeCornersAtDistance(near)
    corners_far = frustum.ComputeCornersAtDistance(far)
    corners = (
        corners_near[0],
        corners_near[1],
        corners_near[2],
        corners_near[3],
        corners_far[0],
        corners_far[1],
        corners_far[2],
        corners_far[3],
    )
    return corners


def set_frustum_perspective(frustum: Gf.Frustum, fov: float, aspect_ratio: float, near: float, far: float):
    """Set the frustum to a perspective projection."""
    if fov <= 0.0 or fov >= 180.0:
        raise ValueError("FOV must be greater than 0 and less than 180 degrees.")
    if aspect_ratio <= 0.0:
        raise ValueError("Aspect ratio must be greater than 0.")
    if near <= 0.0 or far <= 0.0:
        raise ValueError("Near and far distances must be greater than 0.")
    if near >= far:
        raise ValueError("Near distance must be less than far distance.")
    frustum.SetProjectionType(Gf.Frustum.Perspective)
    frustum.SetPerspective(fov, True, aspect_ratio, near, far)


def get_frustum_near_far_range(frustum: Gf.Frustum) -> Gf.Range1d:
    """Get the near/far range of a GfFrustum.

    Args:
        frustum (Gf.Frustum): The frustum to get the near/far range from.

    Returns:
        Gf.Range1d: The near/far range of the frustum.
    """
    near_far_range: Gf.Range1d = frustum.GetNearFar()
    if near_far_range.min > near_far_range.max:
        near_far_range = Gf.Range1d(near_far_range.max, near_far_range.min)
    return near_far_range


def set_frustum_near_far_range(frustum: Gf.Frustum, near: float, far: float) -> None:
    """Set the near and far range of a frustum.

    Args:
        frustum (Gf.Frustum): The frustum to modify.
        near (float): The near plane distance.
        far (float): The far plane distance.

    Raises:
        ValueError: If near or far is non-positive, or near >= far.
    """
    if near <= 0 or far <= 0:
        raise ValueError("Near and far must be positive.")
    if near >= far:
        raise ValueError("Near must be less than far.")
    old_range = frustum.GetNearFar()
    near_ratio = near / old_range.GetMin()
    far_ratio = far / old_range.GetMax()
    window = frustum.GetWindow()
    new_window = Gf.Range2d(Gf.Vec2d(window.GetMin()) * near_ratio, Gf.Vec2d(window.GetMax()) * far_ratio)
    frustum.SetWindow(new_window)
    frustum.SetNearFar(Gf.Range1d(near, far))


def get_frustum_position(frustum: Gf.Frustum) -> Gf.Vec3d:
    """Get the position of the frustum in world space.

    Args:
        frustum (Gf.Frustum): The frustum to get the position from.

    Returns:
        Gf.Vec3d: The position of the frustum in world space.
    """
    if not isinstance(frustum, Gf.Frustum):
        raise TypeError("Input must be a Gf.Frustum")
    position = frustum.GetPosition()
    return position


def set_frustum_position(frustum: Gf.Frustum, position: Gf.Vec3d) -> None:
    """Set the position of the frustum in world space.

    Args:
        frustum (Gf.Frustum): The frustum to modify.
        position (Gf.Vec3d): The new position in world space.
    """
    if not isinstance(frustum, Gf.Frustum):
        raise TypeError("Input frustum must be of type Gf.Frustum")
    if not isinstance(position, Gf.Vec3d):
        raise TypeError("Input position must be of type Gf.Vec3d")
    frustum.SetPosition(position)


def get_frustum_rotation(frustum: Gf.Frustum) -> Gf.Rotation:
    """Get the rotation of a frustum.

    Args:
        frustum (Gf.Frustum): The input frustum.

    Returns:
        Gf.Rotation: The rotation of the frustum.
    """
    if not isinstance(frustum, Gf.Frustum):
        raise TypeError("Input must be a Gf.Frustum")
    rotation = frustum.GetRotation()
    return rotation


def get_frustum_view_direction(frustum: Gf.Frustum) -> Gf.Vec3d:
    """Get the view direction of a frustum.

    Args:
        frustum (Gf.Frustum): The input frustum.

    Returns:
        Gf.Vec3d: The normalized view direction vector.
    """
    if not frustum:
        raise ValueError("Invalid frustum.")
    view_dir = frustum.ComputeViewDirection()
    view_dir.Normalize()
    return view_dir


def compute_frustum_look_at_point(frustum: Gf.Frustum) -> Gf.Vec3d:
    """Computes and returns the world-space look-at point from the frustum's position, rotation, and view distance."""
    position: Gf.Vec3d = frustum.GetPosition()
    rotation: Gf.Rotation = frustum.GetRotation()
    view_distance: float = frustum.GetViewDistance()
    view_direction: Gf.Vec3d = rotation.TransformDir(Gf.Vec3d(0, 0, -1))
    look_at_point: Gf.Vec3d = position + view_direction * view_distance
    return look_at_point


def compute_frustum_pick_ray(frustum: Gf.Frustum, screen_pos: Gf.Vec2d) -> Gf.Ray:
    """Compute a world space pick ray from a screen space position.

    Args:
        frustum (Gf.Frustum): The view frustum.
        screen_pos (Gf.Vec2d): The screen space position in normalized coordinates (0 to 1).

    Returns:
        Gf.Ray: The computed world space pick ray.
    """
    clamped_pos = Gf.Vec2d(max(0.0, min(1.0, screen_pos[0])), max(0.0, min(1.0, screen_pos[1])))
    view_pos = Gf.Vec3d(clamped_pos[0] * 2.0 - 1.0, clamped_pos[1] * 2.0 - 1.0, -1.0)
    inv_view_matrix = frustum.ComputeViewInverse()
    world_pos = inv_view_matrix.Transform(view_pos)
    pick_ray = Gf.Ray(frustum.GetPosition(), world_pos - frustum.GetPosition())
    return pick_ray


def get_frustum_window(frustum: Gf.Frustum) -> Gf.Range2d:
    """Get the window rectangle of a frustum.

    Args:
        frustum (Gf.Frustum): The input frustum.

    Returns:
        Gf.Range2d: The window rectangle in the reference plane.
    """
    if not frustum:
        raise ValueError("Invalid frustum.")
    window = frustum.GetWindow()
    return window


def set_frustum_window(frustum: Gf.Frustum, window: Gf.Range2d) -> None:
    """Set the window rectangle of the frustum.

    Args:
        frustum (Gf.Frustum): The frustum to modify.
        window (Gf.Range2d): The new window rectangle.
    """
    if window.IsEmpty():
        raise ValueError("The input window range is empty.")
    if window.GetMin()[0] >= window.GetMax()[0] or window.GetMin()[1] >= window.GetMax()[1]:
        raise ValueError("The minimum values of the window range must be less than the maximum values.")
    frustum.SetWindow(window)


def compute_frustum_view_frame(frustum: Gf.Frustum) -> Tuple[Gf.Vec3d, Gf.Vec3d, Gf.Vec3d]:
    """Compute the view frame (side, up, view) of a Gf.Frustum.

    Args:
        frustum (Gf.Frustum): The frustum to compute the view frame for.

    Returns:
        Tuple[Gf.Vec3d, Gf.Vec3d, Gf.Vec3d]: The side, up, and view vectors of the view frame.
    """
    view_dir: Gf.Vec3d = frustum.ComputeViewDirection()
    up_vec: Gf.Vec3d = frustum.ComputeUpVector()
    side_vec: Gf.Vec3d = Gf.Cross(view_dir, up_vec)
    return (side_vec, up_vec, view_dir)


def compute_frustum_corners_at_distance(frustum: Gf.Frustum, distance: float) -> List[Gf.Vec3d]:
    """
    Computes the world-space corners of the frustum at a specific distance from the apex.

    Args:
        frustum (Gf.Frustum): The view frustum.
        distance (float): The distance from the frustum apex.

    Returns:
        List[Gf.Vec3d]: The world-space corners of the frustum at the given distance,
            ordered as: Left bottom, Right bottom, Left top, Right top.

    Raises:
        ValueError: If the distance is not within the frustum's near/far range.
    """
    near_far_range = frustum.GetNearFar()
    if distance < near_far_range.min or distance > near_far_range.max:
        raise ValueError(f"Distance {distance} is outside the frustum's near/far range.")
    window = frustum.GetWindow()
    (left, right) = (window.GetMin()[0], window.GetMax()[0])
    (bottom, top) = (window.GetMin()[1], window.GetMax()[1])
    position = frustum.GetPosition()
    rotation = frustum.GetRotation()
    scale_factor = distance / frustum.GetReferencePlaneDepth()
    scaled_left = left * scale_factor
    scaled_right = right * scale_factor
    scaled_bottom = bottom * scale_factor
    scaled_top = top * scale_factor
    local_corners = [
        Gf.Vec3d(scaled_left, scaled_bottom, -distance),
        Gf.Vec3d(scaled_right, scaled_bottom, -distance),
        Gf.Vec3d(scaled_left, scaled_top, -distance),
        Gf.Vec3d(scaled_right, scaled_top, -distance),
    ]
    world_corners = [position + rotation.TransformDir(corner) for corner in local_corners]
    return world_corners


def set_frustum_view_distance(frustum: Gf.Frustum, view_distance: float) -> None:
    """Set the view distance of a frustum.

    Args:
        frustum (Gf.Frustum): The frustum to modify.
        view_distance (float): The new view distance.
    """
    if not isinstance(frustum, Gf.Frustum):
        raise TypeError("Input frustum must be of type Gf.Frustum")
    if view_distance < 0:
        raise ValueError("View distance must be non-negative")
    frustum.SetViewDistance(view_distance)


def get_frustum_view_distance(frustum: Gf.Frustum) -> float:
    """Get the view distance of a Frustum."""
    if not isinstance(frustum, Gf.Frustum):
        raise TypeError("Input must be a Gf.Frustum.")
    view_distance = frustum.GetViewDistance()
    return view_distance


def frustum_intersects_triangle(frustum: Gf.Frustum, p0: Gf.Vec3d, p1: Gf.Vec3d, p2: Gf.Vec3d) -> bool:
    """
    Check if a triangle intersects a frustum.

    Args:
        frustum (Gf.Frustum): The frustum to check against.
        p0 (Gf.Vec3d): The first point of the triangle.
        p1 (Gf.Vec3d): The second point of the triangle.
        p2 (Gf.Vec3d): The third point of the triangle.

    Returns:
        bool: True if the triangle intersects the frustum, False otherwise.
    """
    if frustum.Intersects(p0) or frustum.Intersects(p1) or frustum.Intersects(p2):
        return True
    edges = [(p0, p1), (p1, p2), (p2, p0)]
    for edge in edges:
        if frustum.Intersects(edge[0], edge[1]):
            return True
    corners = frustum.ComputeCorners()
    for i in range(4):
        plane = Gf.Plane(corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4])
        if plane.Intersect(p0, p1, p2):
            return True
    return False


def compute_frustum_side_vector(frustum: Gf.Frustum) -> Gf.Vec3d:
    """Computes the side vector of the given frustum.

    The side vector is the normalized cross product of the view direction
    and the up vector.

    Args:
        frustum (Gf.Frustum): The frustum to compute the side vector for.

    Returns:
        Gf.Vec3d: The computed side vector.
    """
    view_dir = frustum.ComputeViewDirection()
    up_vec = frustum.ComputeUpVector()
    side_vec = Gf.Cross(view_dir, up_vec)
    side_vec.Normalize()
    return side_vec


def set_frustum_external_parameters(
    frustum: Gf.Frustum, position: Gf.Vec3d, rotation: Gf.Rotation, view_distance: float
) -> None:
    """Set the external parameters of a frustum.

    External parameters are position, rotation, and view distance.

    Args:
        frustum (Gf.Frustum): The frustum to modify.
        position (Gf.Vec3d): The new position of the frustum in world space.
        rotation (Gf.Rotation): The new orientation of the frustum in world space.
        view_distance (float): The new view distance of the frustum.

    Raises:
        ValueError: If the provided view_distance is not greater than or equal to 0.
    """
    if view_distance < 0:
        raise ValueError("view_distance must be greater than or equal to 0.")
    frustum.SetPosition(position)
    frustum.SetRotation(rotation)
    frustum.SetViewDistance(view_distance)


def get_frustum_external_parameters(frustum: Gf.Frustum) -> Tuple[Gf.Vec3d, Gf.Rotation, float]:
    """Get the external parameters of a frustum.

    Returns:
        A tuple containing the position, rotation, and view distance.
    """
    position: Gf.Vec3d = frustum.GetPosition()
    rotation: Gf.Rotation = frustum.GetRotation()
    view_distance: float = frustum.GetViewDistance()
    return (position, rotation, view_distance)


def compute_frustum_screen_space_bbox(frustum: Gf.Frustum, window_size: Gf.Vec2i) -> Gf.Range2f:
    """
    Computes the screen space bounding box of the given frustum.

    Args:
        frustum (Gf.Frustum): The frustum to compute the screen space bounding box for.
        window_size (Gf.Vec2i): The size of the window in pixels.

    Returns:
        Gf.Range2f: The screen space bounding box of the frustum.
    """
    corners = frustum.ComputeCorners()
    screen_space_corners = []
    for corner in corners:
        ndc_corner = Gf.Vec3f(corner)
        ndc_corner[0] /= corner[2]
        ndc_corner[1] /= corner[2]
        screen_space_corner = Gf.Vec2f(
            (ndc_corner[0] + 1.0) * 0.5 * window_size[0], (1.0 - ndc_corner[1]) * 0.5 * window_size[1]
        )
        screen_space_corners.append(screen_space_corner)
    bbox = Gf.Range2f()
    for corner in screen_space_corners:
        bbox.UnionWith(corner)
    return bbox


def transform_frustum_view_matrix(frustum: Gf.Frustum, matrix: Gf.Matrix4d) -> Gf.Matrix4d:
    """
    Transforms the view matrix of a frustum by the given matrix.

    The view matrix is computed from the transformed position and rotation of the frustum.
    The scale, shear, and projection components of the matrix are ignored.

    Args:
        frustum (Gf.Frustum): The input frustum to transform.
        matrix (Gf.Matrix4d): The transformation matrix to apply.

    Returns:
        Gf.Matrix4d: The transformed view matrix.
    """
    transformed_frustum = frustum.Transform(matrix)
    transformed_position = transformed_frustum.GetPosition()
    transformed_rotation = transformed_frustum.GetRotation()
    transformed_view_matrix = Gf.Matrix4d()
    transformed_view_matrix.SetLookAt(
        transformed_position,
        transformed_position + transformed_rotation.TransformDir(Gf.Vec3d(0, 0, -1)),
        transformed_rotation.TransformDir(Gf.Vec3d(0, 1, 0)),
    )
    return transformed_view_matrix


def compute_frustum_view_matrix(frustum: Gf.Frustum) -> Gf.Matrix4d:
    """Compute the view matrix from a GfFrustum.

    Args:
        frustum (Gf.Frustum): The input frustum.

    Returns:
        Gf.Matrix4d: The computed view matrix.
    """
    if not isinstance(frustum, Gf.Frustum):
        raise TypeError("Input must be a Gf.Frustum")
    position = frustum.GetPosition()
    rotation = frustum.GetRotation()
    view_dir = rotation.TransformDir(Gf.Vec3d(0, 0, -1))
    up_vec = rotation.TransformDir(Gf.Vec3d(0, 1, 0))
    side_vec = Gf.Cross(view_dir, up_vec)
    view_matrix = Gf.Matrix4d(
        side_vec[0],
        up_vec[0],
        -view_dir[0],
        0,
        side_vec[1],
        up_vec[1],
        -view_dir[1],
        0,
        side_vec[2],
        up_vec[2],
        -view_dir[2],
        0,
        -Gf.Dot(side_vec, position),
        -Gf.Dot(up_vec, position),
        Gf.Dot(view_dir, position),
        1,
    )
    return view_matrix


def set_frustum_projection_type(frustum: Gf.Frustum, projection_type: Gf.Frustum.ProjectionType) -> None:
    """Set the projection type of a frustum.

    Args:
        frustum (Gf.Frustum): The frustum to modify.
        projection_type (Gf.Frustum.ProjectionType): The projection type to set.

    Raises:
        ValueError: If the provided projection type is invalid.
    """
    if not isinstance(projection_type, Gf.Frustum.ProjectionType):
        raise ValueError(f"Invalid projection type: {projection_type}")
    frustum.SetProjectionType(projection_type)


def compute_frustum_screen_space_corners(frustum: Gf.Frustum) -> Gf.Range2f:
    """Compute the screen space corners of a frustum.

    Args:
        frustum (Gf.Frustum): The frustum to compute the corners for.

    Returns:
        Gf.Range2f: The screen space corners of the frustum.
    """
    window = frustum.GetWindow()
    min_point = window.GetMin()
    max_point = window.GetMax()
    width = max_point[0] - min_point[0]
    height = max_point[1] - min_point[1]
    aspect_ratio = width / height
    left = -aspect_ratio
    right = aspect_ratio
    bottom = -1.0
    top = 1.0
    return Gf.Range2f(Gf.Vec2f(left, bottom), Gf.Vec2f(right, top))


def set_frustum_orthographic(
    frustum: Gf.Frustum, left: float, right: float, bottom: float, top: float, near: float, far: float
) -> None:
    """Set the frustum to an orthographic projection.

    Args:
        frustum (Gf.Frustum): The frustum to modify.
        left (float): The left coordinate of the viewing volume.
        right (float): The right coordinate of the viewing volume.
        bottom (float): The bottom coordinate of the viewing volume.
        top (float): The top coordinate of the viewing volume.
        near (float): The near clipping plane distance.
        far (float): The far clipping plane distance.
    """
    if left >= right:
        raise ValueError("Left must be less than right.")
    if bottom >= top:
        raise ValueError("Bottom must be less than top.")
    if near >= far:
        raise ValueError("Near must be less than far.")
    frustum.SetProjectionType(Gf.Frustum.Orthographic)
    window = Gf.Range2d(Gf.Vec2d(left, bottom), Gf.Vec2d(right, top))
    frustum.SetWindow(window)
    frustum.SetNearFar(Gf.Range1d(near, far))


def compute_frustum_aspect_ratio(frustum: Gf.Frustum) -> float:
    """Compute the aspect ratio of the given frustum.

    The aspect ratio is defined as the width of the window divided by the height.
    If the height is zero or negative, this returns 0.
    """
    window: Gf.Range2d = frustum.GetWindow()
    width: float = window.GetSize()[0]
    height: float = window.GetSize()[1]
    if height <= 0:
        return 0.0
    return width / height


def compute_frustum_view_inverse(frustum: Gf.Frustum) -> Gf.Matrix4d:
    """Compute the inverse view matrix for a given frustum.

    Args:
        frustum (Gf.Frustum): The input frustum.

    Returns:
        Gf.Matrix4d: The inverse view matrix.
    """
    position = frustum.GetPosition()
    rotation = frustum.GetRotation()
    view_dir = rotation.TransformDir(Gf.Vec3d(0, 0, -1))
    up_vec = rotation.TransformDir(Gf.Vec3d(0, 1, 0))
    side_vec = Gf.Cross(view_dir, up_vec)
    view_matrix = Gf.Matrix4d(
        side_vec[0],
        up_vec[0],
        -view_dir[0],
        0,
        side_vec[1],
        up_vec[1],
        -view_dir[1],
        0,
        side_vec[2],
        up_vec[2],
        -view_dir[2],
        0,
        -Gf.Dot(side_vec, position),
        -Gf.Dot(up_vec, position),
        Gf.Dot(view_dir, position),
        1,
    )
    return view_matrix.GetInverse()


def get_frustum_perspective_parameters(frustum: Gf.Frustum) -> Optional[Tuple[float, float, float, float]]:
    """Get the perspective parameters of a frustum.

    Args:
        frustum (Gf.Frustum): The frustum to get the perspective parameters from.

    Returns:
        Optional[Tuple[float, float, float, float]]: A tuple containing the field of view (in degrees),
            the aspect ratio, the near distance, and the far distance.
            Returns None if the frustum is not a perspective projection.
    """
    if frustum.GetProjectionType() != Gf.Frustum.Perspective:
        return None
    perspective_data = frustum.GetPerspective()
    if perspective_data is None:
        return None
    (field_of_view, aspect_ratio, near_distance, far_distance) = perspective_data
    return (field_of_view, aspect_ratio, near_distance, far_distance)


def compute_frustum_corners_for_distance(frustum: Gf.Frustum, distance: float) -> List[Gf.Vec3d]:
    """
    Compute the world-space corners of the frustum at a specific distance from the apex.

    Args:
        frustum (Gf.Frustum): The view frustum.
        distance (float): The distance from the apex.

    Returns:
        List[Gf.Vec3d]: The world-space corners of the frustum at the specified distance.
    """
    if distance <= 0:
        raise ValueError("Distance must be greater than zero.")
    window = frustum.window
    view_matrix = frustum.ComputeViewMatrix()
    inv_view_matrix = frustum.ComputeViewInverse()
    scaled_window = window * (distance / frustum.nearFar.max)
    corners_view = [
        Gf.Vec3d(scaled_window.min[0], scaled_window.min[1], -distance),
        Gf.Vec3d(scaled_window.max[0], scaled_window.min[1], -distance),
        Gf.Vec3d(scaled_window.min[0], scaled_window.max[1], -distance),
        Gf.Vec3d(scaled_window.max[0], scaled_window.max[1], -distance),
    ]
    corners_world = [inv_view_matrix.Transform(corner) for corner in corners_view]
    return corners_world


def transform_frustum(frustum: Gf.Frustum, matrix: Gf.Matrix4d) -> Gf.Frustum:
    """
    Transforms the given frustum by the given matrix.

    The transformation matrix is applied as follows:
    1. The position and the direction vector are transformed with the given matrix.
    2. The length of the new direction vector is used to rescale the near and far plane and the view distance.
    3. The points that define the reference plane are transformed by the matrix.

    This method ensures that the frustum will not be sheared or perspective-projected.

    Note: The transformed frustum does not preserve scales very well. Do not use this function to transform
    a frustum that is to be used for precise operations such as intersection testing.

    Args:
        frustum (Gf.Frustum): The input frustum to be transformed.
        matrix (Gf.Matrix4d): The transformation matrix.

    Returns:
        Gf.Frustum: The transformed frustum.
    """
    position = frustum.GetPosition()
    rotation = frustum.GetRotation()
    direction = rotation.TransformDir(Gf.Vec3d(0, 0, -1))
    new_position = matrix.Transform(position)
    new_direction = matrix.TransformDir(direction)
    scale_factor = new_direction.GetLength() / direction.GetLength()
    near_far = frustum.GetNearFar()
    new_near_far = Gf.Range1d(near_far.GetMin() * scale_factor, near_far.GetMax() * scale_factor)
    new_view_distance = frustum.GetViewDistance() * scale_factor
    window = frustum.GetWindow()
    new_window = Gf.Range2d(
        Gf.Vec2d(
            matrix.Transform(Gf.Vec3d(window.GetMin()[0], window.GetMin()[1], 0))[0],
            matrix.Transform(Gf.Vec3d(window.GetMin()[0], window.GetMin()[1], 0))[1],
        ),
        Gf.Vec2d(
            matrix.Transform(Gf.Vec3d(window.GetMax()[0], window.GetMax()[1], 0))[0],
            matrix.Transform(Gf.Vec3d(window.GetMax()[0], window.GetMax()[1], 0))[1],
        ),
    )
    new_frustum = Gf.Frustum()
    new_frustum.SetPosition(new_position)
    new_frustum.SetRotation(Gf.Rotation(new_direction, Gf.Vec3d(0, 1, 0)))
    new_frustum.SetNearFar(new_near_far)
    new_frustum.SetViewDistance(new_view_distance)
    new_frustum.SetWindow(new_window)
    new_frustum.SetProjectionType(frustum.GetProjectionType())
    return new_frustum


def compute_frustum_projection_matrix(frustum: Gf.Frustum) -> Gf.Matrix4d:
    """Compute the projection matrix for a given frustum.

    Args:
        frustum (Gf.Frustum): The frustum to compute the projection matrix for.

    Returns:
        Gf.Matrix4d: The computed projection matrix.

    Raises:
        ValueError: If the frustum has an unsupported projection type.
    """
    proj_type = frustum.GetProjectionType()
    if proj_type == Gf.Frustum.Orthographic:
        (left, right, bottom, top, near, far) = [0.0] * 6
        if not frustum.GetOrthographic(left, right, bottom, top, near, far):
            raise ValueError("Failed to get orthographic projection parameters.")
        return Gf.Matrix4d(
            2.0 / (right - left),
            0.0,
            0.0,
            -(right + left) / (right - left),
            0.0,
            2.0 / (top - bottom),
            0.0,
            -(top + bottom) / (top - bottom),
            0.0,
            0.0,
            -2.0 / (far - near),
            -(far + near) / (far - near),
            0.0,
            0.0,
            0.0,
            1.0,
        )
    elif proj_type == Gf.Frustum.Perspective:
        (fov, aspect_ratio, near, far) = frustum.GetPerspective()
        tan_half_fov = math.tan(math.radians(fov * 0.5))
        x_scale = 1.0 / (aspect_ratio * tan_half_fov)
        y_scale = 1.0 / tan_half_fov
        A = (far + near) / (near - far)
        B = 2.0 * far * near / (near - far)
        return Gf.Matrix4d(x_scale, 0.0, 0.0, 0.0, 0.0, y_scale, 0.0, 0.0, 0.0, 0.0, A, B, 0.0, 0.0, -1.0, 0.0)
    else:
        raise ValueError(f"Unsupported projection type: {proj_type}")


def get_frustum_orthographic_parameters(
    frustum: Gf.Frustum,
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Get the orthographic parameters for a frustum.

    Args:
        frustum (Gf.Frustum): The frustum to get orthographic parameters from.

    Returns:
        Optional[Tuple[float, float, float, float, float, float]]: The orthographic parameters
        (left, right, bottom, top, near, far) if the frustum is orthographic, None otherwise.
    """
    if frustum.GetProjectionType() != Gf.Frustum.Orthographic:
        return None
    (left, right, bottom, top, near, far) = (
        frustum.GetWindow().GetMin()[0],
        frustum.GetWindow().GetMax()[0],
        frustum.GetWindow().GetMin()[1],
        frustum.GetWindow().GetMax()[1],
        frustum.GetNearFar().GetMin(),
        frustum.GetNearFar().GetMax(),
    )
    return (left, right, bottom, top, near, far)


def set_frustum_projection_matrix(frustum: Gf.Frustum) -> Gf.Matrix4d:
    """
    Set the projection matrix of a frustum and return the resulting matrix.

    Args:
        frustum (Gf.Frustum): The frustum to set the projection matrix for.

    Returns:
        Gf.Matrix4d: The resulting projection matrix.
    """
    if not frustum:
        raise ValueError("Invalid frustum provided.")
    projection_type = frustum.GetProjectionType()
    projection_matrix = Gf.Matrix4d()
    if projection_type == Gf.Frustum.Orthographic:
        (left, right, bottom, top, near, far) = frustum.GetOrthographic()
        projection_matrix.SetIdentity()
        projection_matrix[0, 0] = 2.0 / (right - left)
        projection_matrix[1, 1] = 2.0 / (top - bottom)
        projection_matrix[2, 2] = -2.0 / (far - near)
        projection_matrix[3, 0] = -(right + left) / (right - left)
        projection_matrix[3, 1] = -(top + bottom) / (top - bottom)
        projection_matrix[3, 2] = -(far + near) / (far - near)
    elif projection_type == Gf.Frustum.Perspective:
        (fov, aspect_ratio, near, far) = frustum.GetPerspective()
        tan_half_fov = tan(radians(fov) / 2.0)
        projection_matrix.SetIdentity()
        projection_matrix[0, 0] = 1.0 / (aspect_ratio * tan_half_fov)
        projection_matrix[1, 1] = 1.0 / tan_half_fov
        projection_matrix[2, 2] = -(far + near) / (far - near)
        projection_matrix[2, 3] = -1.0
        projection_matrix[3, 2] = -(2.0 * far * near) / (far - near)
        projection_matrix[3, 3] = 0.0
    else:
        raise ValueError(f"Unsupported projection type: {projection_type}")
    return projection_matrix


def set_orthographic_projection(
    camera: UsdGeom.Camera, near: float, far: float, left: float, right: float, bottom: float, top: float
) -> None:
    """Set the camera's projection to orthographic with the given parameters.

    Args:
        camera (UsdGeom.Camera): The camera to set the projection for.
        near (float): The near clipping plane distance.
        far (float): The far clipping plane distance.
        left (float): The left boundary of the viewing frustum.
        right (float): The right boundary of the viewing frustum.
        bottom (float): The bottom boundary of the viewing frustum.
        top (float): The top boundary of the viewing frustum.
    """
    camera.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(near, far))
    camera.GetHorizontalApertureAttr().Set(right - left)
    camera.GetVerticalApertureAttr().Set(top - bottom)
    horizontal_offset = (left + right) / 2
    vertical_offset = (bottom + top) / 2
    camera.GetHorizontalApertureOffsetAttr().Set(horizontal_offset)
    camera.GetVerticalApertureOffsetAttr().Set(vertical_offset)


class MultiInterval:

    def __init__(self):
        self.intervals = []

    def Add(self, interval):
        self.intervals.append(interval)

    def IsEmpty(self):
        return len(self.intervals) == 0

    def __iter__(self):
        return iter(self.intervals)


def merge_intervals(intervals: MultiInterval) -> MultiInterval:
    """Merge overlapping intervals in a MultiInterval."""
    merged = MultiInterval()
    if intervals.IsEmpty():
        return merged
    interval_list = sorted(intervals.intervals, key=lambda i: i.min)
    current_min = interval_list[0].min
    current_max = interval_list[0].max
    for interval in interval_list[1:]:
        if current_max >= interval.min - 1:
            current_max = max(current_max, interval.max)
        else:
            merged.Add(Gf.Interval(current_min, current_max))
            current_min = interval.min
            current_max = interval.max
    merged.Add(Gf.Interval(current_min, current_max))
    return merged


def find_gaps_in_intervals(intervals: Gf.MultiInterval) -> Gf.MultiInterval:
    """
    Find the gaps (complement) in a given set of intervals.

    Args:
        intervals (Gf.MultiInterval): The input intervals.

    Returns:
        Gf.MultiInterval: The gaps (complement) of the input intervals.
    """
    if intervals.IsEmpty():
        return Gf.MultiInterval.GetFullInterval()
    gaps = intervals.GetComplement()
    return gaps


def subdivide_intervals(intervals: List[Gf.Interval], num_segments: int) -> List[Gf.Interval]:
    """
    Subdivide a list of intervals into a specified number of segments.

    Args:
        intervals (List[Gf.Interval]): The list of intervals to subdivide.
        num_segments (int): The number of segments to subdivide each interval into.

    Returns:
        List[Gf.Interval]: A new list of intervals with each original interval subdivided.
    """
    if num_segments <= 0:
        raise ValueError("num_segments must be a positive integer")
    result = []
    for interval in intervals:
        if interval.IsEmpty():
            continue
        segment_size = interval.GetSize() / num_segments
        segment_start = interval.GetMin()
        for _ in range(num_segments):
            segment_end = segment_start + segment_size
            segment_interval = Gf.Interval()
            segment_interval.SetMin(segment_start)
            segment_interval.SetMax(segment_end)
            result.append(segment_interval)
            segment_start = segment_end
    return result


def get_finite_intervals(intervals: List[Gf.Interval]) -> List[Gf.Interval]:
    """
    Get a list of finite intervals from a list of intervals.

    Args:
        intervals (List[Gf.Interval]): A list of intervals to filter.

    Returns:
        List[Gf.Interval]: A list of finite intervals.
    """
    finite_intervals: List[Gf.Interval] = []
    for interval in intervals:
        if interval.IsFinite():
            finite_intervals.append(interval)
        elif interval.IsMinFinite():
            finite_interval = Gf.Interval(interval.GetMin(), float("inf"))
            finite_intervals.append(finite_interval)
        elif interval.IsMaxFinite():
            finite_interval = Gf.Interval(float("-inf"), interval.GetMax())
            finite_intervals.append(finite_interval)
    return finite_intervals


def align_intervals_to_grid(intervals: Gf.Interval, grid_size: float) -> Gf.Interval:
    """
    Align an interval to a grid of a given size.

    The interval is expanded if necessary so that its minimum and maximum
    values are aligned to the nearest grid lines.

    Args:
        intervals (Gf.Interval): The input interval to align.
        grid_size (float): The size of the grid to align to.

    Returns:
        Gf.Interval: The aligned interval.
    """
    if grid_size <= 0:
        raise ValueError("Grid size must be greater than zero.")
    min_val = intervals.GetMin()
    max_val = intervals.GetMax()
    aligned_min = min_val // grid_size * grid_size
    aligned_max = (max_val + grid_size - 1) // grid_size * grid_size
    aligned_interval = Gf.Interval()
    aligned_interval.SetMin(aligned_min)
    aligned_interval.SetMax(aligned_max)
    return aligned_interval


def get_interval_bounds(interval: Gf.Interval) -> Tuple[float, float]:
    """Get the minimum and maximum bounds of an interval.

    Args:
        interval (Gf.Interval): The interval to query.

    Returns:
        Tuple[float, float]: A tuple containing the minimum and maximum bounds.
    """
    if interval.IsEmpty():
        raise ValueError("Cannot get bounds of an empty interval.")
    min_value = interval.GetMin()
    max_value = interval.GetMax()
    return (min_value, max_value)


def interpolate_intervals(interval_a: Gf.Interval, interval_b: Gf.Interval, alpha: float) -> Gf.Interval:
    """Linearly interpolate between two intervals.

    Args:
        interval_a (Gf.Interval): The first interval.
        interval_b (Gf.Interval): The second interval.
        alpha (float): The interpolation factor in the range [0, 1].

    Returns:
        Gf.Interval: The interpolated interval.
    """
    if interval_a.IsEmpty() or interval_b.IsEmpty():
        raise ValueError("Cannot interpolate empty intervals.")
    alpha = min(max(alpha, 0.0), 1.0)
    min_value = Gf.Lerp(interval_a.GetMin(), interval_b.GetMin(), alpha)
    max_value = Gf.Lerp(interval_a.GetMax(), interval_b.GetMax(), alpha)
    min_closed = interval_a.IsMinClosed() if alpha < 1.0 else interval_b.IsMinClosed()
    max_closed = interval_a.IsMaxClosed() if alpha > 0.0 else interval_b.IsMaxClosed()
    return Gf.Interval(min_value, max_value, min_closed, max_closed)


def scale_interval(interval: Gf.Interval, scale: float) -> Gf.Interval:
    """Scale an interval by a given factor.

    Args:
        interval (Gf.Interval): The input interval to scale.
        scale (float): The scaling factor.

    Returns:
        Gf.Interval: The scaled interval.
    """
    if interval.IsEmpty():
        raise ValueError("Cannot scale an empty interval.")
    min_value = interval.GetMin()
    max_value = interval.GetMax()
    scaled_min = min_value * scale
    scaled_max = max_value * scale
    scaled_interval = Gf.Interval()
    scaled_interval.SetMin(scaled_min, interval.IsMinClosed())
    scaled_interval.SetMax(scaled_max, interval.IsMaxClosed())
    return scaled_interval


def get_intervals_for_time_range(start_time: float, end_time: float, stride: float) -> List[Gf.Interval]:
    """
    Get a list of Gf.Interval objects that span a given time range.

    Args:
        start_time (float): The start time of the range.
        end_time (float): The end time of the range.
        stride (float): The stride between each interval.

    Returns:
        List[Gf.Interval]: A list of Gf.Interval objects spanning the time range.
    """
    intervals = []
    current_time = start_time
    if start_time >= end_time:
        raise ValueError("start_time must be less than end_time")
    if stride <= 0:
        raise ValueError("stride must be positive")
    while current_time < end_time:
        interval_end = min(current_time + stride, end_time)
        interval = Gf.Interval(current_time, interval_end, True, False)
        intervals.append(interval)
        current_time += stride
    return intervals


def union_of_intervals(intervals: List[Gf.Interval]) -> Gf.Interval:
    """
    Compute the union of a list of intervals.

    Args:
        intervals (List[Gf.Interval]): List of intervals to compute the union of.

    Returns:
        Gf.Interval: The union of the input intervals.
    """
    if not intervals:
        return Gf.Interval()
    result = Gf.Interval()
    for interval in intervals:
        if interval.IsEmpty():
            continue
        if result.IsEmpty():
            result = interval
        else:
            result_min = min(result.GetMin(), interval.GetMin())
            result_max = max(result.GetMax(), interval.GetMax())
            min_closed = result.IsMinClosed() or interval.IsMinClosed()
            max_closed = result.IsMaxClosed() or interval.IsMaxClosed()
            result = Gf.Interval(result_min, result_max, min_closed, max_closed)
    return result


def create_intervals_for_materials(stage: Usd.Stage, material_paths: List[str]) -> List[Gf.Interval]:
    """Create a list of intervals for a list of material paths."""
    intervals = []
    for material_path in material_paths:
        material_prim = stage.GetPrimAtPath(material_path)
        if not material_prim.IsValid():
            raise ValueError(f"Material prim at path {material_path} does not exist.")
        material = UsdShade.Material(material_prim)
        if not material:
            raise ValueError(f"Prim at path {material_path} is not a valid UsdShade.Material.")
        roughness_attr = material.GetInput("roughness")
        if not roughness_attr:
            interval = Gf.Interval(0.0, 1.0, True, True)
        else:
            roughness_value = roughness_attr.Get()
            if roughness_value is None:
                interval = Gf.Interval(0.0, 1.0, True, True)
            else:
                interval = Gf.Interval(roughness_value, roughness_value, True, True)
        intervals.append(interval)
    return intervals


def intersection_of_intervals(a: Gf.Interval, b: Gf.Interval) -> Gf.Interval:
    """
    Returns the intersection of two intervals.

    If the intervals do not intersect, an empty interval is returned.
    """
    min_val = max(a.GetMin(), b.GetMin())
    max_val = min(a.GetMax(), b.GetMax())
    if min_val > max_val:
        return Gf.Interval()
    min_closed = a.IsMinClosed() and b.IsMinClosed() and (min_val != a.GetMin()) and (min_val != b.GetMin())
    max_closed = a.IsMaxClosed() and b.IsMaxClosed() and (max_val != a.GetMax()) and (max_val != b.GetMax())
    intersection = Gf.Interval()
    intersection.SetMin(min_val, min_closed)
    intersection.SetMax(max_val, max_closed)
    return intersection


def aggregate_intervals_by_layer(layers: List[Sdf.Layer]) -> Dict[Sdf.Layer, Gf.Interval]:
    """
    Aggregates the time code intervals for each layer in the given list of layers.

    Args:
        layers (List[Sdf.Layer]): The list of layers to aggregate intervals for.

    Returns:
        Dict[Sdf.Layer, Gf.Interval]: A dictionary mapping each layer to its aggregated time code interval.
    """
    result = {}
    for layer in layers:
        start_time = layer.startTimeCode
        end_time = layer.endTimeCode
        if start_time is None or end_time is None:
            layer_interval = Gf.Interval()
        else:
            layer_interval = Gf.Interval(start_time, end_time)
        result[layer] = layer_interval
    return result


def set_intervals_for_selection(stage: Usd.Stage, selection: List[str], interval: Gf.Interval) -> None:
    """Set the visibility interval for selected prims."""
    for prim_path in selection:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist.")
            continue
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            print(f"Warning: Prim at path {prim_path} is not imageable.")
            continue
        imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible, Usd.TimeCode(interval.GetMin()))
        imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited, Usd.TimeCode(interval.GetMax()))


def animate_prim_interval(stage: Usd.Stage, prim_path: str, interval: Gf.Interval, translation_distance: float):
    """Animate a prim's translation based on an interval.

    The prim's translation will be animated from the start to the end of the interval,
    moving by the specified translation_distance.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to animate.
        interval (Gf.Interval): The time interval over which to animate the prim.
        translation_distance (float): The total distance to translate the prim.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    translate_op = add_translate_op(xformable)
    interval_size = interval.GetSize()
    if interval_size == 0:
        translation_per_frame = Gf.Vec3d(0, 0, 0)
    else:
        translation_per_frame = Gf.Vec3d(translation_distance / interval_size, 0, 0)
    for frame in range(int(interval.GetMin()), int(interval.GetMax()) + 1):
        timecode = Usd.TimeCode(frame)
        translation = translation_per_frame * (frame - interval.GetMin())
        translate_op.Set(translation, timecode)


def get_prims_by_interval_size(stage: Usd.Stage, min_size: float, max_size: float) -> List[Usd.Prim]:
    """
    Get a list of prims whose bounding box size falls within the given interval.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        min_size (float): The minimum size of the bounding box (inclusive).
        max_size (float): The maximum size of the bounding box (inclusive).

    Returns:
        List[Usd.Prim]: A list of prims whose bounding box size falls within the interval.
    """
    size_interval = Gf.Interval(min_size, max_size)
    prims = []
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        bbox_size = bbox.ComputeAlignedRange().GetSize().GetLength()
        if size_interval.Contains(bbox_size):
            prims.append(prim)
    return prims


def set_prim_interval(
    stage: Usd.Stage, prim_path: str, min_val: float, max_val: float, min_closed: bool = True, max_closed: bool = True
) -> None:
    """Set the interval for a prim's extent."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    bbox_attr = UsdGeom.Boundable(prim).GetExtentAttr()
    if not bbox_attr:
        raise ValueError(f"Prim at path {prim_path} does not have an extent attribute.")
    interval = Gf.Interval()
    interval.SetMin(min_val, min_closed)
    interval.SetMax(max_val, max_closed)
    bbox_attr.Set(Vt.Vec3dArray(2, (Gf.Vec3d(interval.GetMin()), Gf.Vec3d(interval.GetMax()))))


def create_intervals_from_prims(stage: Usd.Stage, prim_paths: List[str]) -> List[Gf.Interval]:
    """Create intervals from the bounding boxes of a list of prims."""
    intervals = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
        if bbox.GetBox().IsEmpty():
            continue
        box = bbox.GetBox()
        min_point = box.GetMin()
        max_point = box.GetMax()
        interval = Gf.Interval(min_point[0], max_point[0])
        intervals.append(interval)
    return intervals


def create_bounding_interval(
    min_value: float, max_value: float, min_closed: bool = True, max_closed: bool = True
) -> Gf.Interval:
    """Create a bounding interval from min and max values.

    Args:
        min_value (float): The minimum value of the interval.
        max_value (float): The maximum value of the interval.
        min_closed (bool, optional): Whether the minimum value is included in the interval. Defaults to True.
        max_closed (bool, optional): Whether the maximum value is included in the interval. Defaults to True.

    Returns:
        Gf.Interval: The created bounding interval.

    Raises:
        ValueError: If min_value is greater than max_value.
    """
    if min_value > max_value:
        raise ValueError("min_value must be less than or equal to max_value")
    interval = Gf.Interval()
    interval.SetMin(min_value, min_closed)
    interval.SetMax(max_value, max_closed)
    return interval


def find_closest_prims_to_line(stage: Usd.Stage, line: Gf.Line, max_distance: float) -> List[Tuple[Usd.Prim, float]]:
    """
    Find the closest prims to a given line within a maximum distance.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        line (Gf.Line): The line to measure distance from.
        max_distance (float): The maximum distance threshold.

    Returns:
        List[Tuple[Usd.Prim, float]]: A list of tuples containing the closest prims and their distances.
    """
    closest_prims = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            bbox_attr = mesh.GetAttribute("extent")
            if not bbox_attr.HasValue():
                continue
            bounds = bbox_attr.Get()
            center = (Gf.Vec3d(bounds[0]) + Gf.Vec3d(bounds[1])) / 2.0
            closest_point = line.FindClosestPoint(center, None)
            distance = (center - closest_point).GetLength()
            if distance <= max_distance:
                closest_prims.append((prim, distance))
    return closest_prims


def place_prims_on_line(stage: Usd.Stage, line: Gf.Line, num_prims: int, prim_type: str, prim_name_prefix: str) -> None:
    """Places prims of specified type along a line at equal intervals.

    Args:
        stage (Usd.Stage): The USD stage to create the prims on.
        line (Gf.Line): The line along which to place the prims.
        num_prims (int): The number of prims to create.
        prim_type (str): The type of prim to create, e.g., 'Sphere', 'Cube', etc.
        prim_name_prefix (str): The prefix for the prim names. Prims will be named as {prim_name_prefix}_0, {prim_name_prefix}_1, etc.

    Raises:
        ValueError: If num_prims is less than 1.
    """
    if num_prims < 1:
        raise ValueError("num_prims must be greater than 0.")
    interval = 1.0 / (num_prims - 1) if num_prims > 1 else 0
    for i in range(num_prims):
        t = i * interval
        position = line.GetPoint(t)
        prim_path = f"/World/{prim_name_prefix}_{i}"
        prim = stage.DefinePrim(prim_path, prim_type)
        UsdGeom.XformCommonAPI(prim).SetTranslate(position)


def create_prim(stage: Usd.Stage, prim_type: str, prim_path: str) -> Usd.Prim:
    """Create a new prim at the given path with the specified type."""
    return stage.DefinePrim(prim_path, prim_type)


def get_translate(stage: Usd.Stage, prim_path: str) -> Tuple[float, float, float]:
    """Get the translation for a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if not translate_ops:
        return (0.0, 0.0, 0.0)
    translate_op = translate_ops[0]
    translation = translate_op.Get(Usd.TimeCode.Default())
    if translation is None:
        return (0.0, 0.0, 0.0)
    return (translation[0], translation[1], translation[2])


def align_prims_along_line(stage: Usd.Stage, line_seg: Gf.LineSeg, prim_paths: List[str], spacing: float = 1.0):
    """
    Aligns a list of prims along a line segment with uniform spacing.

    Args:
        stage (Usd.Stage): The USD stage.
        line_seg (Gf.LineSeg): The line segment to align prims along.
        prim_paths (List[str]): A list of prim paths to align.
        spacing (float): The spacing between prims. Defaults to 1.0.

    Raises:
        ValueError: If any of the prim paths are invalid or not transformable.
    """
    if not prim_paths:
        return
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")
        if not prim.IsA(UsdGeom.Xformable):
            raise ValueError(f"Prim is not transformable: {prim_path}")
    line_length = line_seg.GetLength()
    total_spacing = (len(prim_paths) - 1) * spacing
    start_pos = line_seg.GetPoint(0)
    end_pos = line_seg.GetPoint(1)
    if total_spacing > line_length:
        print(f"Warning: Spacing ({total_spacing}) exceeds line length ({line_length}). Prims will overlap.")
    for i, prim_path in enumerate(prim_paths):
        t = i * spacing / line_length
        pos = start_pos + (end_pos - start_pos) * t
        prim = stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        add_translate_op(xformable).Set(Gf.Vec3d(pos))


def get_line_segment_length(line_seg: Gf.LineSeg) -> float:
    """Get the length of a line segment.

    Args:
        line_seg (Gf.LineSeg): The line segment to calculate the length of.

    Returns:
        float: The length of the line segment.
    """
    if not isinstance(line_seg, Gf.LineSeg):
        raise TypeError("Invalid input. Expected a Gf.LineSeg.")
    length = line_seg.GetLength()
    return length


def get_normalized_direction_of_line(line_seg: Gf.LineSeg) -> Gf.Vec3d:
    """Get the normalized direction of a line segment.

    Args:
        line_seg (Gf.LineSeg): The line segment to get the direction of.

    Returns:
        Gf.Vec3d: The normalized direction of the line segment.
    """
    direction = line_seg.GetDirection()
    if direction == Gf.Vec3d(0, 0, 0):
        raise ValueError("Line segment has zero length and no defined direction.")
    return direction


def calculate_closest_points_between_lines(line1: Gf.LineSeg, line2: Gf.LineSeg) -> Tuple[Gf.Vec3d, Gf.Vec3d]:
    """
    Calculate the closest points between two line segments.

    Args:
        line1 (Gf.LineSeg): The first line segment.
        line2 (Gf.LineSeg): The second line segment.

    Returns:
        Tuple[Gf.Vec3d, Gf.Vec3d]: A tuple containing the closest points on line1 and line2 respectively.
    """
    dir1 = line1.GetDirection()
    dir2 = line2.GetDirection()
    start1 = line1.GetPoint(0)
    start2 = line2.GetPoint(0)
    startDiff = start2 - start1
    a = Gf.Dot(dir1, dir1)
    b = Gf.Dot(dir1, dir2)
    c = Gf.Dot(dir2, dir2)
    d = Gf.Dot(dir1, startDiff)
    e = Gf.Dot(dir2, startDiff)
    det = a * c - b * b
    if abs(det) < 1e-06:
        t1 = 0
        t2 = d / b if abs(b) > 1e-06 else 0
    else:
        t1 = (b * e - c * d) / det
        t2 = (a * e - b * d) / det
    t1 = max(0, min(1, t1))
    t2 = max(0, min(1, t2))
    closestPoint1 = line1.GetPoint(t1)
    closestPoint2 = line2.GetPoint(t2)
    return (closestPoint1, closestPoint2)


def find_closest_point_on_line(line: Gf.LineSeg, point: Gf.Vec3d) -> Gf.Vec3d:
    """
    Find the closest point on a line segment to a given point.

    Args:
        line (Gf.LineSeg): The line segment.
        point (Gf.Vec3d): The point to find the closest point to.

    Returns:
        Gf.Vec3d: The closest point on the line segment to the given point.
    """
    start_point = line.GetPoint(0)
    end_point = line.GetPoint(1)
    start_to_point = point - start_point
    direction = end_point - start_point
    projection = Gf.Dot(start_to_point, direction) / Gf.Dot(direction, direction)
    clamped_projection = Gf.Clamp(projection, 0, 1)
    closest_point = start_point + clamped_projection * direction
    return closest_point


def interpolate_matrix2d(start: Gf.Matrix2d, end: Gf.Matrix2d, alpha: float) -> Gf.Matrix2d:
    """Linearly interpolate between two 2D matrices.

    Args:
        start (Gf.Matrix2d): The start matrix.
        end (Gf.Matrix2d): The end matrix.
        alpha (float): The interpolation factor in the range [0.0, 1.0].

    Returns:
        Gf.Matrix2d: The interpolated matrix.
    """
    if alpha < 0 or alpha > 1:
        raise ValueError(f"Interpolation factor alpha must be in range [0.0, 1.0], got {alpha}")
    if alpha == 0:
        return Gf.Matrix2d(start)
    if alpha == 1:
        return Gf.Matrix2d(end)
    result = Gf.Matrix2d()
    for i in range(2):
        for j in range(2):
            s = start[i, j]
            e = end[i, j]
            result[i, j] = s + alpha * (e - s)
    return result


def transform_point(matrix: Gf.Matrix4d, point: Gf.Vec3d) -> Gf.Vec3d:
    """Transform a point by a 4x4 matrix.

    Args:
        matrix (Gf.Matrix4d): The transformation matrix.
        point (Gf.Vec3d): The point to transform.

    Returns:
        Gf.Vec3d: The transformed point.
    """
    transformed_point = matrix.Transform(point)
    return transformed_point


def scale_matrix2d(scale: Gf.Vec2d) -> Gf.Matrix2d:
    """Create a 2D scale matrix from a 2D scale vector.

    Args:
        scale (Gf.Vec2d): The 2D scale vector.

    Returns:
        Gf.Matrix2d: The resulting 2D scale matrix.
    """
    matrix = Gf.Matrix2d(1)
    matrix[0, 0] = scale[0]
    matrix[1, 1] = scale[1]
    return matrix


def translate_matrix2d(matrix: Gf.Matrix2d, translation: Gf.Vec2d) -> Gf.Matrix2d:
    """
    Translates the given matrix by the specified translation vector.

    Args:
        matrix (Gf.Matrix2d): The input matrix to be translated.
        translation (Gf.Vec2d): The translation vector.

    Returns:
        Gf.Matrix2d: The translated matrix.
    """
    result = Gf.Matrix2d()
    result.SetIdentity()
    result.SetColumn(2, translation)
    result = matrix * result
    return result


def create_identity_matrix2d() -> Gf.Matrix2d:
    """Create and return a 2x2 identity matrix."""
    matrix = Gf.Matrix2d()
    matrix.SetIdentity()
    return matrix


def transform_vector(matrix: Gf.Matrix4f, vector: Gf.Vec3f) -> Gf.Vec3f:
    """
    Transform a 3D vector by a 4x4 matrix.

    Args:
        matrix (Gf.Matrix4f): The transformation matrix.
        vector (Gf.Vec3f): The vector to transform.

    Returns:
        Gf.Vec3f: The transformed vector.
    """
    transformed_vector = matrix.TransformDir(vector)
    return transformed_vector


def rotate_matrix2d(angle_degrees: float) -> Gf.Matrix2d:
    """Create a 2D rotation matrix from an angle in degrees."""
    angle_radians = math.radians(angle_degrees)
    matrix = Gf.Matrix2d()
    matrix.SetIdentity()
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)
    matrix.SetRow(0, Gf.Vec2d(cos_angle, -sin_angle))
    matrix.SetRow(1, Gf.Vec2d(sin_angle, cos_angle))
    return matrix


def combine_transformations(transform1: Gf.Matrix2d, transform2: Gf.Matrix2d) -> Gf.Matrix2d:
    """Combine two 2D transformation matrices.

    Args:
        transform1 (Gf.Matrix2d): The first transformation matrix.
        transform2 (Gf.Matrix2d): The second transformation matrix.

    Returns:
        Gf.Matrix2d: The combined transformation matrix.
    """
    if not isinstance(transform1, Gf.Matrix2d) or not isinstance(transform2, Gf.Matrix2d):
        raise TypeError("Input matrices must be of type Gf.Matrix2d")
    combined_matrix = transform1 * transform2
    return combined_matrix


def get_matrix2d_properties(matrix: Gf.Matrix2d) -> dict:
    """Get various properties of a Gf.Matrix2d.

    Args:
        matrix (Gf.Matrix2d): The input matrix.

    Returns:
        dict: A dictionary containing the matrix properties.
    """
    determinant = matrix.GetDeterminant()
    inverse = matrix.GetInverse()
    transpose = matrix.GetTranspose()
    rows = [matrix.GetRow(i) for i in range(2)]
    columns = [matrix.GetColumn(i) for i in range(2)]
    properties = {
        "determinant": determinant,
        "inverse": inverse,
        "transpose": transpose,
        "rows": rows,
        "columns": columns,
    }
    return properties


def apply_inverse_transform(matrix: Gf.Matrix2d, point: Gf.Vec2d) -> Gf.Vec2d:
    """
    Apply the inverse of a 2D transformation matrix to a 2D point.

    Args:
        matrix (Gf.Matrix2d): The 2D transformation matrix.
        point (Gf.Vec2d): The 2D point to transform.

    Returns:
        Gf.Vec2d: The transformed 2D point.
    """
    det = matrix.GetDeterminant()
    if Gf.IsClose(det, 0.0, 1e-06):
        raise ValueError("The matrix is not invertible.")
    inverse_matrix = matrix.GetInverse()
    transformed_point = inverse_matrix * point
    return transformed_point


def get_matrix_columns(matrix: Gf.Matrix2f) -> List[Gf.Vec2f]:
    """Get the columns of a Matrix2f as a list of Vec2f."""
    if not isinstance(matrix, Gf.Matrix2f):
        raise TypeError("Input must be a Gf.Matrix2f")
    num_columns = 2
    columns = []
    for i in range(num_columns):
        column = matrix.GetColumn(i)
        columns.append(column)
    return columns


def set_matrix_from_rows(row1: Gf.Vec2f, row2: Gf.Vec2f) -> Gf.Matrix2f:
    """Set a Gf.Matrix2f from two row vectors.

    Args:
        row1 (Gf.Vec2f): The first row vector.
        row2 (Gf.Vec2f): The second row vector.

    Returns:
        Gf.Matrix2f: The constructed matrix.
    """
    matrix = Gf.Matrix2f(1.0)
    matrix.SetRow(0, row1)
    matrix.SetRow(1, row2)
    return matrix


def set_matrix_from_columns(matrix: Gf.Matrix2f, col0: Gf.Vec2f, col1: Gf.Vec2f) -> None:
    """Set the matrix from two column vectors.

    Args:
        matrix (Gf.Matrix2f): The matrix to set.
        col0 (Gf.Vec2f): The first column vector.
        col1 (Gf.Vec2f): The second column vector.

    Raises:
        TypeError: If any of the arguments are not of the expected type.
    """
    if not isinstance(matrix, Gf.Matrix2f):
        raise TypeError("matrix must be of type Gf.Matrix2f")
    if not isinstance(col0, Gf.Vec2f) or not isinstance(col1, Gf.Vec2f):
        raise TypeError("col0 and col1 must be of type Gf.Vec2f")
    matrix.SetColumn(0, col0)
    matrix.SetColumn(1, col1)


def get_matrix_rows(matrix: Gf.Matrix2f) -> Tuple[Gf.Vec2f, Gf.Vec2f]:
    """Get the rows of a 2x2 matrix as a tuple of Vec2f.

    Args:
        matrix (Gf.Matrix2f): The input 2x2 matrix.

    Returns:
        Tuple[Gf.Vec2f, Gf.Vec2f]: A tuple containing the two rows of the matrix as Vec2f.
    """
    row0: Gf.Vec2f = matrix.GetRow(0)
    row1: Gf.Vec2f = matrix.GetRow(1)
    return (row0, row1)


def transpose_matrix(matrix: Gf.Matrix4f) -> Gf.Matrix4f:
    """Transpose the given 4x4 matrix.

    Args:
        matrix (Gf.Matrix4f): The input 4x4 matrix to transpose.

    Returns:
        Gf.Matrix4f: The transposed matrix.
    """
    transposed = matrix.GetTranspose()
    return transposed


def multiply_matrices(matrix1: Gf.Matrix2f, matrix2: Gf.Matrix2f) -> Gf.Matrix2f:
    """
    Multiply two 2x2 matrices and return the result.

    Args:
        matrix1 (Gf.Matrix2f): The first matrix.
        matrix2 (Gf.Matrix2f): The second matrix.

    Returns:
        Gf.Matrix2f: The result of the matrix multiplication.
    """
    result = Gf.Matrix2f()
    for i in range(2):
        for j in range(2):
            element = 0.0
            for k in range(2):
                element += matrix1[i, k] * matrix2[k, j]
            result[i, j] = element
    return result


def set_matrix_identity(matrix: Gf.Matrix2f) -> Gf.Matrix2f:
    """
    Set the given matrix to the identity matrix.

    Args:
        matrix (Gf.Matrix2f): The input matrix to set to identity.

    Returns:
        Gf.Matrix2f: The modified matrix set to identity.
    """
    if not isinstance(matrix, Gf.Matrix2f):
        raise TypeError("Input matrix must be of type Gf.Matrix2f")
    matrix.SetIdentity()
    return matrix


def set_matrix_zero(matrix: Gf.Matrix2f) -> Gf.Matrix2f:
    """Set all elements of a Gf.Matrix2f to zero.

    Args:
        matrix (Gf.Matrix2f): The input matrix to modify.

    Returns:
        Gf.Matrix2f: The modified matrix with all elements set to zero.
    """
    result = matrix.SetZero()
    return result


def invert_matrix(matrix: Gf.Matrix4d, eps: float = 1e-10) -> Gf.Matrix4d:
    """Invert a 4x4 matrix.

    Args:
        matrix (Gf.Matrix4d): The input matrix to invert.
        eps (float, optional): The epsilon value for checking singularity. Defaults to 1e-10.

    Returns:
        Gf.Matrix4d: The inverted matrix, or the identity matrix if the input is singular.
    """
    det = matrix.GetDeterminant()
    if abs(det) < eps:
        return Gf.Matrix4d(1.0)
    inverse = matrix.GetInverse()
    return inverse


def to_scale_translation(matrix: Gf.Matrix2f) -> Tuple[Gf.Vec2f, Gf.Vec2f]:
    """
    Decomposes a 2x2 matrix into a scale and translation.

    Args:
        matrix (Gf.Matrix2f): The input 2x2 matrix to decompose.

    Returns:
        Tuple[Gf.Vec2f, Gf.Vec2f]: A tuple containing the scale and translation.
            - scale (Gf.Vec2f): The scale component of the matrix.
            - translation (Gf.Vec2f): The translation component of the matrix.

    Raises:
        ValueError: If the input matrix is not a valid 2x2 matrix.
    """
    if matrix.dimension[0] != 2 or matrix.dimension[1] != 2:
        raise ValueError("Input matrix must be a 2x2 matrix")
    scale = Gf.Vec2f(matrix[0][0], matrix[1][1])
    translation = Gf.Vec2f(matrix[0][1], matrix[1][0])
    return (scale, translation)


def from_scale_translation(scale: Gf.Vec2f, translation: Gf.Vec2f) -> Gf.Matrix2f:
    """Create a matrix from scale and translation.

    Args:
        scale (Gf.Vec2f): The scale vector.
        translation (Gf.Vec2f): The translation vector.

    Returns:
        Gf.Matrix2f: The resulting matrix.
    """
    mat = Gf.Matrix2f(1.0)
    mat[0, 0] = scale[0]
    mat[1, 1] = scale[1]
    mat[0, 1] = translation[0]
    mat[1, 0] = translation[1]
    return mat


def is_singular(self, eps: float = 1e-06) -> bool:
    """Check if the matrix is singular.

    A matrix is considered singular if its determinant is less than
    or equal to the specified epsilon value.

    Parameters
    ----------
    eps : float, optional
        The epsilon value for determining singularity. Default is 1e-6.

    Returns
    -------
    bool
        True if the matrix is singular, False otherwise.
    """
    det = self.GetDeterminant()
    if abs(det) <= eps:
        return True
    else:
        return False


def determinant_and_inverse(matrix: Gf.Matrix2f) -> Tuple[float, Gf.Matrix2f]:
    """
    Compute the determinant and inverse of a 2x2 matrix.

    Args:
        matrix (Gf.Matrix2f): The input 2x2 matrix.

    Returns:
        Tuple[float, Gf.Matrix2f]: A tuple containing the determinant and inverse matrix.
                                    If the matrix is singular, the inverse will be the
                                    identity matrix scaled by float_max.
    """
    determinant = matrix.GetDeterminant()
    inverse = matrix.GetInverse()
    return (determinant, inverse)


def set_zero_for_prim_matrix(stage: Usd.Stage, prim_path: str) -> None:
    """Sets the matrix of a prim to the zero matrix.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Raises:
        ValueError: If the prim is not valid or transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    transform_op = xformable.GetOrderedXformOps()[-1]
    if transform_op.GetOpType() != UsdGeom.XformOp.TypeTransform:
        transform_op = xformable.AddTransformOp()
    zero_matrix = Gf.Matrix4d(0)
    transform_op.Set(zero_matrix)


def rotate_prim_matrix(prim: Usd.Prim, rotation_matrix: Gf.Matrix3d, time_code: Usd.TimeCode = Usd.TimeCode.Default()):
    """
    Rotates a prim using a rotation matrix.

    Args:
        prim (Usd.Prim): The prim to rotate.
        rotation_matrix (Gf.Matrix3d): The rotation matrix to apply.
        time_code (Usd.TimeCode, optional): The time code at which to set the rotation. Defaults to Usd.TimeCode.Default().

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim '{prim.GetPath()}' is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim '{prim.GetPath()}' is not transformable.")
    rotation_op = None
    xform_ops = xformable.GetOrderedXformOps()
    for op in xform_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            rotation_op = op
            break
    if rotation_op is None:
        rotation_op = add_rotate_xyz_op(xformable, UsdGeom.XformOp.PrecisionFloat, "")
    rotation_op.Set(
        Gf.Vec3d(rotation_matrix.ExtractRotation().Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())),
        time_code,
    )


def compute_handedness_of_prim(prim: Usd.Prim) -> float:
    """Compute the handedness of a prim.

    Args:
        prim (Usd.Prim): The prim to compute the handedness for.

    Returns:
        float: 1.0 if right-handed, -1.0 if left-handed, 0.0 if singular.

    Raises:
        ValueError: If the prim is not transformable.
    """
    if not prim.IsA(UsdGeom.Xformable):
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    xformable = UsdGeom.Xformable(prim)
    transform_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rotation_matrix = Gf.Matrix3d(transform_matrix.ExtractRotationMatrix())
    return rotation_matrix.GetHandedness()


def get_row_from_prim_matrix(stage: Usd.Stage, prim_path: str, row_index: int) -> Optional[Gf.Vec3d]:
    """Get a row from the transformation matrix of a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        row_index (int): The index of the row to retrieve (0, 1, or 2).

    Returns:
        Optional[Gf.Vec3d]: The row vector if the prim has a transform matrix, None otherwise.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable")
    matrix = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    if not matrix:
        return None
    if row_index < 0 or row_index > 2:
        raise ValueError(f"Invalid row index {row_index}. Must be 0, 1, or 2.")
    row = matrix.GetRow(row_index)
    return row


def set_column_in_prim_matrix(prim: Usd.Prim, col_index: int, vec: Gf.Vec3d):
    """Set a column in the transform matrix of a prim.

    Args:
        prim (Usd.Prim): The prim to set the matrix column for.
        col_index (int): The index of the column to set (0, 1, or 2).
        vec (Gf.Vec3d): The vector to set as the column.

    Raises:
        ValueError: If the prim is not valid or not transformable, or if the column index is out of range.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    if col_index < 0 or col_index > 2:
        raise ValueError(f"Column index {col_index} is out of range. Must be 0, 1, or 2.")
    transform_attr = xformable.GetXformOpOrderAttr()
    transform_matrix = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    transform_matrix.SetColumn(col_index, Gf.Vec4d(vec[0], vec[1], vec[2], 0 if col_index < 3 else 1))
    xform_op = xformable.MakeMatrixXform()
    xform_op.Set(transform_matrix, Usd.TimeCode.Default())
    transform_attr.Set(Vt.TokenArray([xform_op.GetOpName()]))


def extract_scaling_from_prim_matrix(prim: Usd.Prim) -> Gf.Vec3f:
    """Extracts the scaling from a prim's local transformation matrix.

    Args:
        prim (Usd.Prim): The prim to extract the scaling from.

    Returns:
        Gf.Vec3f: The scaling vector extracted from the prim's matrix.

    Raises:
        ValueError: If the prim is not transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    matrix = xformable.GetLocalTransformation()
    scale_x = Gf.Vec3d(matrix.GetColumn(0)[:3]).GetLength()
    scale_y = Gf.Vec3d(matrix.GetColumn(1)[:3]).GetLength()
    scale_z = Gf.Vec3d(matrix.GetColumn(2)[:3]).GetLength()
    return Gf.Vec3f(scale_x, scale_y, scale_z)


def create_orthonormal_prim_matrix(stage: Usd.Stage, prim_path: str, matrix: Gf.Matrix3d) -> None:
    """Create an orthonormal matrix attribute on a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path of the prim to create the attribute on.
        matrix (Gf.Matrix3d): The input matrix to orthonormalize and set on the prim.

    Raises:
        ValueError: If the prim at the given path is not valid or transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    ortho_matrix = matrix.GetOrthonormalized(True)
    prim.CreateAttribute("orthonormalMatrix", Sdf.ValueTypeNames.Matrix3d).Set(ortho_matrix)


def extract_rotation_from_prim(prim: Usd.Prim) -> Gf.Matrix3d:
    """Extract the rotation from a prim as a Matrix3d."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    rotation_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() in (
            UsdGeom.XformOp.TypeRotateXYZ,
            UsdGeom.XformOp.TypeRotateXZY,
            UsdGeom.XformOp.TypeRotateYXZ,
            UsdGeom.XformOp.TypeRotateYZX,
            UsdGeom.XformOp.TypeRotateZXY,
            UsdGeom.XformOp.TypeRotateZYX,
        ):
            rotation_op = op
            break
    if rotation_op is None:
        return Gf.Matrix3d(1)
    rotation_vec3f = rotation_op.Get(Usd.TimeCode.Default())
    rotation_vec3d = Gf.Vec3d(rotation_vec3f)
    rotation_matrix = Gf.Matrix3d(1)
    rotation = (
        Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation_vec3d[0])
        * Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation_vec3d[1])
        * Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation_vec3d[2])
    )
    rotation_matrix.SetRotate(rotation)
    return rotation_matrix


def compute_prim_matrix_determinant(stage: Usd.Stage, prim_path: str) -> float:
    """Compute the determinant of a prim's transformation matrix.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path of the prim.

    Returns:
        float: The determinant of the prim's transformation matrix.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    matrix = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    matrix3d = Gf.Matrix3d(
        matrix[0, 0],
        matrix[0, 1],
        matrix[0, 2],
        matrix[1, 0],
        matrix[1, 1],
        matrix[1, 2],
        matrix[2, 0],
        matrix[2, 1],
        matrix[2, 2],
    )
    return matrix3d.GetDeterminant()


def get_column_from_prim_matrix(stage: Usd.Stage, prim_path: str, column_index: int) -> Gf.Vec3d:
    """Get a column from a prim's transformation matrix."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    matrix = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    if column_index < 0 or column_index > 2:
        raise ValueError(f"Invalid column index {column_index}. Must be between 0 and 2.")
    column = matrix.GetColumn(column_index)
    return column


def inverse_prim_matrix(
    stage: Usd.Stage, prim_path: str, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> Gf.Matrix4d:
    """
    Computes the inverse of the local transformation matrix for a given prim at a specific time.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path of the prim.
        time_code (Usd.TimeCode, optional): The time code at which to compute the inverse matrix. Defaults to Usd.TimeCode.Default().

    Returns:
        Gf.Matrix4d: The inverse of the local transformation matrix.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    local_matrix = xformable.GetLocalTransformation(time_code)
    det = local_matrix.GetDeterminant()
    epsilon = 1e-06
    if abs(det) > epsilon:
        inverse_matrix = local_matrix.GetInverse()
    else:
        inverse_matrix = Gf.Matrix4d(1)
    return inverse_matrix


def transpose_prim_matrix(prim: Usd.Prim) -> Gf.Matrix3f:
    """Transpose the local transformation matrix of a prim.

    Args:
        prim (Usd.Prim): The prim to get the matrix from.

    Returns:
        Gf.Matrix3f: The transposed matrix.

    Raises:
        ValueError: If the prim is not valid or has no transform ops.
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError("Prim has no transform ops")
    transform_matrix = xformable.GetLocalTransformation()
    rotation_matrix = transform_matrix.ExtractRotationMatrix()
    transposed_matrix = rotation_matrix.GetTranspose()
    return transposed_matrix


def set_diagonal_in_prim_matrix(prim: UsdGeom.Xformable, diagonal: Gf.Vec3d) -> None:
    """Set the diagonal values of a prim's matrix.

    Args:
        prim (UsdGeom.Xformable): The prim to set the diagonal values for.
        diagonal (Gf.Vec3d): The diagonal values to set.

    Raises:
        ValueError: If the prim is not transformable.
    """
    if not prim:
        raise ValueError("Prim is not valid.")
    transform_matrix = prim.GetLocalTransformation()
    new_matrix = Gf.Matrix4d(diagonal[0], 0, 0, 0, 0, diagonal[1], 0, 0, 0, 0, diagonal[2], 0, 0, 0, 0, 1)
    combined_matrix = new_matrix * transform_matrix
    transform_op = prim.MakeMatrixXform()
    transform_op.Set(combined_matrix)


def set_row_in_prim_matrix(stage: Usd.Stage, prim_path: str, row_index: int, row_values: Gf.Vec4d):
    """Set a row of a prim's matrix from a Vec4d.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.
        row_index (int): The index of the row to set (0, 1, 2, or 3).
        row_values (Gf.Vec4d): The Vec4d containing the values for the row.

    Raises:
        ValueError: If the prim is not valid or transformable, or if the row index is invalid.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    matrix = xformable.GetLocalTransformation()
    if row_index < 0 or row_index > 3:
        raise ValueError(f"Invalid row index {row_index}. Must be 0, 1, 2, or 3.")
    matrix.SetRow(row_index, row_values)
    transform_op = xformable.MakeMatrixXform()
    transform_op.Set(matrix)


def get_prim_matrix_column(prim: Union[Usd.Prim, UsdGeom.Xformable], column_index: int) -> Gf.Vec3f:
    """Get a column of the local transform matrix of a prim.

    Args:
        prim (Union[Usd.Prim, UsdGeom.Xformable]): The prim or xformable to get the matrix column from.
        column_index (int): The index of the column to retrieve (0, 1, or 2).

    Returns:
        Gf.Vec3f: The column vector from the local transform matrix.

    Raises:
        ValueError: If the input prim is not valid or not transformable, or if the column index is out of range.
    """
    if isinstance(prim, Usd.Prim):
        if not prim.IsValid():
            raise ValueError(f"Prim {prim} is not valid.")
        xformable = UsdGeom.Xformable(prim)
    else:
        xformable = prim
    if not xformable:
        raise ValueError(f"Prim {prim} is not transformable.")
    if column_index < 0 or column_index > 2:
        raise ValueError(f"Column index {column_index} is out of range. Must be 0, 1, or 2.")
    local_matrix = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    column = local_matrix.GetColumn(column_index)
    return column


def concatenate_transformations(transforms: List[Gf.Matrix3f]) -> Gf.Matrix3f:
    """Concatenate a list of Matrix3f transformations.

    Args:
        transforms (List[Gf.Matrix3f]): List of transformation matrices.

    Returns:
        Gf.Matrix3f: The concatenated transformation matrix.
    """
    if not transforms:
        return Gf.Matrix3f(1)
    result = transforms[0]
    for i in range(1, len(transforms)):
        result *= transforms[i]
    return result


def decompose_matrix_to_components(matrix: Gf.Matrix3f) -> Tuple[Gf.Vec3f, Gf.Rotation, Gf.Vec3f]:
    """
    Decomposes a 3x3 matrix into translation, rotation, and scale components.

    Args:
        matrix (Gf.Matrix3f): The input 3x3 matrix to decompose.

    Returns:
        Tuple[Gf.Vec3f, Gf.Rotation, Gf.Vec3f]: A tuple containing the translation, rotation, and scale components.

    Raises:
        ValueError: If the input matrix is not a valid 3x3 matrix.
    """
    if not isinstance(matrix, Gf.Matrix3f):
        raise ValueError("Input matrix must be a Gf.Matrix3f.")
    translation = Gf.Vec3f(0.0)
    rotation = matrix.ExtractRotation()
    scale = Gf.Vec3f(
        Gf.Vec3f(matrix[0][0], matrix[0][1], matrix[0][2]).GetLength(),
        Gf.Vec3f(matrix[1][0], matrix[1][1], matrix[1][2]).GetLength(),
        Gf.Vec3f(matrix[2][0], matrix[2][1], matrix[2][2]).GetLength(),
    )
    return (translation, rotation, scale)


def modify_matrix_column(matrix: Gf.Matrix3f, column_index: int, new_column: Gf.Vec3f) -> Gf.Matrix3f:
    """
    Modifies a specific column of a Gf.Matrix3f with a new Gf.Vec3f.

    Args:
        matrix (Gf.Matrix3f): The input matrix to modify.
        column_index (int): The index of the column to modify (0, 1, or 2).
        new_column (Gf.Vec3f): The new column vector to replace the existing column.

    Returns:
        Gf.Matrix3f: The modified matrix with the specified column replaced.

    Raises:
        ValueError: If the column_index is outside the valid range [0, 2].
    """
    if column_index < 0 or column_index > 2:
        raise ValueError(f"Invalid column_index. Expected 0, 1, or 2, but got {column_index}.")
    modified_matrix = Gf.Matrix3f(matrix)
    modified_matrix.SetColumn(column_index, new_column)
    return modified_matrix


def modify_matrix_row(matrix: Gf.Matrix3f, row_index: int, new_row: Gf.Vec3f) -> Gf.Matrix3f:
    """Modify a specific row of a Matrix3f with a new Vec3f.

    Args:
        matrix (Gf.Matrix3f): The input matrix to modify.
        row_index (int): The index of the row to replace (0, 1, or 2).
        new_row (Gf.Vec3f): The new vector to use as the row.

    Returns:
        Gf.Matrix3f: The modified matrix with the specified row replaced.

    Raises:
        IndexError: If the provided row_index is outside the valid range [0, 2].
    """
    if row_index < 0 or row_index > 2:
        raise IndexError(f"Invalid row index: {row_index}. Must be between 0 and 2.")
    modified_matrix = Gf.Matrix3f(matrix)
    modified_matrix.SetRow(row_index, new_row)
    return modified_matrix


def apply_diagonal_matrix(matrix: Gf.Matrix3f, diagonal: Gf.Vec3f) -> Gf.Matrix3f:
    """Apply a diagonal matrix to the input matrix.

    The diagonal matrix is constructed from the given diagonal vector.
    This is equivalent to multiplying the input matrix by a diagonal
    matrix with the given diagonal entries.

    Args:
        matrix (Gf.Matrix3f): The input matrix.
        diagonal (Gf.Vec3f): The diagonal entries for the diagonal matrix.

    Returns:
        Gf.Matrix3f: The resulting matrix after applying the diagonal matrix.
    """
    result = Gf.Matrix3f(matrix)
    for i in range(3):
        column = result.GetColumn(i)
        result.SetColumn(i, column * diagonal[i])
    return result


def set_prim_rotation(prim: Usd.Prim, rotation: Gf.Quath) -> None:
    """Set the rotation for a prim using a GfQuath.

    Args:
        prim (Usd.Prim): The prim to set the rotation for.
        rotation (Gf.Quath): The rotation quaternion to set.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    rotation_op = add_rotate_xyz_op(xformable, UsdGeom.XformOp.PrecisionFloat, "rotation")
    normalized_rotation = rotation.GetNormalized()
    rotation_op.Set(normalized_rotation.GetImaginary(), Usd.TimeCode.Default())


def set_prim_scale(stage: Usd.Stage, prim_path: str, scale: Gf.Vec3f) -> None:
    """Set the scale for a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path of the prim to set the scale for.
        scale (Gf.Vec3f): The scale to set.

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    scale_op = add_scale_op(xformable)
    scale_op.Set(scale)


def zero_out_prim_matrix(stage: Usd.Stage, prim_path: str) -> None:
    """Zero out the transform matrix of a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    transform_ops = xformable.GetOrderedXformOps()
    if not transform_ops:
        transform_op = xformable.AddTransformOp()
    else:
        transform_op = transform_ops[-1]
    if transform_op.GetOpType() != UsdGeom.XformOp.TypeTransform:
        transform_op = xformable.AddTransformOp()
    transform_op.Set(Gf.Matrix4d(1.0))


def extract_and_apply_rotation(matrix: Gf.Matrix3f, vec: Gf.Vec3f) -> Gf.Vec3f:
    """
    Extracts the rotation from a matrix and applies it to a vector.

    Args:
        matrix (Gf.Matrix3f): The input matrix to extract rotation from.
        vec (Gf.Vec3f): The vector to apply the rotation to.

    Returns:
        Gf.Vec3f: The rotated vector.
    """
    rotation: Gf.Rotation = matrix.ExtractRotation()
    rotated_vec: Gf.Vec3f = rotation.TransformDir(vec)
    return rotated_vec


def set_prim_to_identity(prim: Usd.Prim) -> None:
    """Set the transform of a prim to the identity matrix."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    transform_ops = xformable.GetOrderedXformOps()
    if not transform_ops:
        xformable.AddTransformOp()
        transform_ops = xformable.GetOrderedXformOps()
    transform_op = transform_ops[-1]
    if transform_op.GetOpType() != UsdGeom.XformOp.TypeTransform:
        raise ValueError(f"Last transform op of prim {prim.GetPath()} is not a matrix op.")
    transform_op.Set(Gf.Matrix4d(1))


def check_handedness_of_prim(prim: Usd.Prim) -> str:
    """Check the handedness of a prim's transformation matrix.

    Args:
        prim (Usd.Prim): The prim to check the handedness of.

    Returns:
        str: "left" if the prim's transformation is left-handed,
             "right" if the prim's transformation is right-handed,
             "singular" if the prim's transformation matrix is singular,
             "no transform" if the prim has no transformation matrix.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return "no transform"
    matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rotation_matrix = Gf.Matrix3f(
        matrix[0, 0],
        matrix[0, 1],
        matrix[0, 2],
        matrix[1, 0],
        matrix[1, 1],
        matrix[1, 2],
        matrix[2, 0],
        matrix[2, 1],
        matrix[2, 2],
    )
    handedness = rotation_matrix.GetHandedness()
    if handedness == 1.0:
        return "right"
    elif handedness == -1.0:
        return "left"
    else:
        return "singular"


def get_prim_inverse_matrix(prim: Usd.Prim) -> Gf.Matrix3f:
    """Get the inverse of the local transformation matrix for a prim."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim} is not transformable.")
    matrix = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    matrix3d = Gf.Matrix3f(
        matrix[0, 0],
        matrix[0, 1],
        matrix[0, 2],
        matrix[1, 0],
        matrix[1, 1],
        matrix[1, 2],
        matrix[2, 0],
        matrix[2, 1],
        matrix[2, 2],
    )
    det = matrix3d.GetDeterminant()
    eps = 1e-06
    if abs(det) <= eps:
        return Gf.Matrix3f(1)
    return matrix3d.GetInverse()


def normalize_prim_matrix(stage: Usd.Stage, prim_path: str, eps: float = 1e-06) -> bool:
    """
    Normalize the local transformation matrix of a prim to be orthonormal.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path of the prim to normalize.
        eps (float): The epsilon value for checking matrix singularity.

    Returns:
        bool: True if the normalization was successful, False if the matrix is singular.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim not found at path: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable")
    matrix = xformable.GetLocalTransformation()
    rotation_matrix = Gf.Matrix3f(
        matrix[0, 0],
        matrix[0, 1],
        matrix[0, 2],
        matrix[1, 0],
        matrix[1, 1],
        matrix[1, 2],
        matrix[2, 0],
        matrix[2, 1],
        matrix[2, 2],
    )
    det = rotation_matrix.GetDeterminant()
    if abs(det) <= eps:
        return False
    orthonormalized_matrix = rotation_matrix.GetOrthonormalized(issueWarning=True)
    new_matrix = Gf.Matrix4d(
        orthonormalized_matrix[0, 0],
        orthonormalized_matrix[0, 1],
        orthonormalized_matrix[0, 2],
        0,
        orthonormalized_matrix[1, 0],
        orthonormalized_matrix[1, 1],
        orthonormalized_matrix[1, 2],
        0,
        orthonormalized_matrix[2, 0],
        orthonormalized_matrix[2, 1],
        orthonormalized_matrix[2, 2],
        0,
        matrix[3, 0],
        matrix[3, 1],
        matrix[3, 2],
        matrix[3, 3],
    )
    xform_op = xformable.AddTransformOp()
    xform_op.Set(new_matrix)
    return True


def copy_matrix_to_prim(stage: Usd.Stage, prim_path: str, matrix: Gf.Matrix3f):
    """Copy a Gf.Matrix3f to a prim's transform.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        matrix (Gf.Matrix3f): The matrix to copy.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable")
    matrix_4d = Gf.Matrix4d(
        matrix[0, 0],
        matrix[0, 1],
        matrix[0, 2],
        0,
        matrix[1, 0],
        matrix[1, 1],
        matrix[1, 2],
        0,
        matrix[2, 0],
        matrix[2, 1],
        matrix[2, 2],
        0,
        0,
        0,
        0,
        1,
    )
    transform_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTransform)
    transform_op.Set(matrix_4d)


def get_prim_matrix(stage: Usd.Stage, prim_path: str) -> Gf.Matrix4d:
    """Get the local transformation matrix for a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Returns:
        Gf.Matrix4d: The local transformation matrix of the prim.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    matrix = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    return matrix


def get_prim_rotation(stage: Usd.Stage, prim_path: str) -> Gf.Matrix4f:
    """Get the rotation for a prim as a Matrix4f."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    transform_matrix = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    rotation_quat = transform_matrix.ExtractRotationQuat()
    rotation_matrix = Gf.Matrix4f()
    rotation_matrix.SetRotate(Gf.Rotation(rotation_quat))
    return rotation_matrix


def get_prim_scale(stage: Usd.Stage, prim_path: str) -> Tuple[float, float, float]:
    """Get the scale for a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    scale_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
    if not scale_ops:
        return (1.0, 1.0, 1.0)
    scale_op = scale_ops[0]
    scale = scale_op.Get(Usd.TimeCode.Default())
    if scale is None:
        return (1.0, 1.0, 1.0)
    return (scale[0], scale[1], scale[2])


def create_rotation_matrix(rotation_quat: Gf.Quatd) -> Gf.Matrix4d:
    """Create a rotation matrix from a quaternion.

    Args:
        rotation_quat (Gf.Quatd): The rotation quaternion.

    Returns:
        Gf.Matrix4d: The rotation matrix.
    """
    if not rotation_quat.GetLength():
        raise ValueError("Invalid rotation quaternion. Length is zero.")
    rotation_quat = rotation_quat.GetNormalized()
    rotation_matrix = Gf.Matrix4d()
    rotation_matrix.SetRotate(Gf.Rotation(rotation_quat))
    return rotation_matrix


def create_translation_matrix(translate: Gf.Vec3d) -> Gf.Matrix4d:
    """Create a translation matrix from a Vec3d.

    Args:
        translate (Gf.Vec3d): The translation vector.

    Returns:
        Gf.Matrix4d: The translation matrix.
    """
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTranslateOnly(translate)
    return matrix


def create_scale_matrix(scale_factor: Gf.Vec3d) -> Gf.Matrix4d:
    """Create a scale matrix from a Vec3d scale factor.

    Args:
        scale_factor (Gf.Vec3d): The scale factor vector.

    Returns:
        Gf.Matrix4d: The resulting scale matrix.
    """
    matrix = Gf.Matrix4d(1.0)
    matrix[0, 0] = scale_factor[0]
    matrix[1, 1] = scale_factor[1]
    matrix[2, 2] = scale_factor[2]
    return matrix


def get_prim_hierarchy_matrices(stage: Usd.Stage) -> Dict[str, Gf.Matrix4d]:
    """
    Recursively traverse the prim hierarchy and compute the local transform
    matrix for each prim, storing it in a dictionary.

    Args:
        stage (Usd.Stage): The USD stage to traverse.

    Returns:
        Dict[str, Gf.Matrix4d]: A dictionary mapping prim paths to their
        local transform matrices.
    """
    prim_matrices = {}

    def traverse_hierarchy(prim: Usd.Prim, parent_matrix: Gf.Matrix4d):
        """
        Recursive helper function to traverse the prim hierarchy.

        Args:
            prim (Usd.Prim): The current prim being processed.
            parent_matrix (Gf.Matrix4d): The accumulated parent matrix.
        """
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            local_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        else:
            local_matrix = Gf.Matrix4d(1)
        final_matrix = local_matrix * parent_matrix
        prim_matrices[prim.GetPath().pathString] = final_matrix
        for child_prim in prim.GetChildren():
            traverse_hierarchy(child_prim, final_matrix)

    pseudoroot = stage.GetPseudoRoot()
    traverse_hierarchy(pseudoroot, Gf.Matrix4d(1))
    return prim_matrices


def set_prim_hierarchy_matrices(stage: Usd.Stage, root_prim_path: str, matrix: Gf.Matrix4d) -> None:
    """
    Recursively set the local transformation matrix for a hierarchy of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        root_prim_path (str): The path to the root prim of the hierarchy.
        matrix (Gf.Matrix4d): The transformation matrix to set.
    """
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid():
        raise ValueError(f"Prim at path {root_prim_path} does not exist.")
    for prim in Usd.PrimRange(root_prim):
        if prim.GetPath() != root_prim_path:
            xformable = UsdGeom.Xformable(prim)
            if xformable:
                xformable.MakeMatrixXform().Set(matrix)
            else:
                print(f"Warning: Prim at path {prim.GetPath()} is not transformable.")


def get_prim_relative_matrix(stage: Usd.Stage, prim_path: str, parent_path: str) -> Gf.Matrix4d:
    """
    Get the relative transformation matrix of a prim with respect to its parent.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path of the prim.
        parent_path (str): The path of the parent prim.

    Returns:
        Gf.Matrix4d: The relative transformation matrix.

    Raises:
        ValueError: If the prim or parent prim is invalid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    parent_prim = stage.GetPrimAtPath(parent_path)
    if not parent_prim.IsValid():
        raise ValueError(f"Parent prim at path {parent_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    parent_xformable = UsdGeom.Xformable(parent_prim)
    if not parent_xformable:
        raise ValueError(f"Parent prim at path {parent_path} is not transformable.")
    local_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    parent_matrix = parent_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    relative_matrix = parent_matrix.GetInverse() * local_matrix
    return relative_matrix


def extract_prim_hierarchy_transforms(stage: Usd.Stage) -> Dict[str, Gf.Matrix4d]:
    """
    Extracts the local transform of each prim in the stage hierarchy.

    Args:
        stage (Usd.Stage): The USD stage to extract transforms from.

    Returns:
        A dictionary mapping prim paths to their local transform matrix.
    """
    prim_transforms = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        prim_path = prim.GetPath().pathString
        xformable = UsdGeom.Xformable(prim)
        local_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        prim_transforms[prim_path] = local_transform
    return prim_transforms


def transform_prim_to_world(prim: Usd.Prim) -> Gf.Matrix4d:
    """
    Compute the world space transformation matrix for a given prim.

    Args:
        prim (Usd.Prim): The prim to compute the world space transformation for.

    Returns:
        Gf.Matrix4d: The world space transformation matrix.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim} is not transformable")
    local_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    parent = prim.GetParent()
    while parent.IsValid():
        xformable = UsdGeom.Xformable(parent)
        if xformable:
            parent_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            local_matrix = local_matrix * parent_matrix
        parent = parent.GetParent()
    return local_matrix


def transform_prim_to_local(stage: Usd.Stage, prim_path: str, matrix: Gf.Matrix4d) -> None:
    """Transform a prim to its local space using the given matrix.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to transform.
        matrix (Gf.Matrix4d): The transformation matrix.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    xformable.MakeMatrixXform().Set(matrix)


def create_look_at_matrix(eye, center, up) -> Gf.Matrix4d:
    """Create a look-at matrix.

    Args:
        eye (Gf.Vec3d): The eye position.
        center (Gf.Vec3d): The center position.
        up (Gf.Vec3d): The up vector.

    Returns:
        Gf.Matrix4d: The look-at matrix.
    """
    if not isinstance(eye, Gf.Vec3d) or not isinstance(center, Gf.Vec3d) or (not isinstance(up, Gf.Vec3d)):
        raise TypeError("eye, center, and up must be Gf.Vec3d")
    z_axis = (eye - center).GetNormalized()
    x_axis = Gf.Cross(up, z_axis).GetNormalized()
    y_axis = Gf.Cross(z_axis, x_axis)
    matrix = Gf.Matrix4d(
        x_axis[0],
        x_axis[1],
        x_axis[2],
        0,
        y_axis[0],
        y_axis[1],
        y_axis[2],
        0,
        z_axis[0],
        z_axis[1],
        z_axis[2],
        0,
        eye[0],
        eye[1],
        eye[2],
        1,
    )
    return matrix


def extract_translation(matrix: Gf.Matrix4d) -> Gf.Vec3d:
    """Extract the translation component from a 4x4 matrix.

    Args:
        matrix (Gf.Matrix4d): The input 4x4 matrix.

    Returns:
        Gf.Vec3d: The translation component of the matrix.
    """
    tx = matrix.GetRow3(3)[0]
    ty = matrix.GetRow3(3)[1]
    tz = matrix.GetRow3(3)[2]
    translation = Gf.Vec3d(tx, ty, tz)
    return translation


def extract_scale(matrix: Gf.Matrix4d) -> Gf.Vec3d:
    """Extract the scale components from a 4x4 matrix.

    Args:
        matrix (Gf.Matrix4d): The input matrix to extract scale from.

    Returns:
        Gf.Vec3d: The scale components as a Vec3d.
    """
    upper_3x3 = Gf.Matrix3d(
        matrix[0, 0],
        matrix[0, 1],
        matrix[0, 2],
        matrix[1, 0],
        matrix[1, 1],
        matrix[1, 2],
        matrix[2, 0],
        matrix[2, 1],
        matrix[2, 2],
    )
    scale_x = Gf.Vec3d(upper_3x3[0, 0], upper_3x3[0, 1], upper_3x3[0, 2]).GetLength()
    scale_y = Gf.Vec3d(upper_3x3[1, 0], upper_3x3[1, 1], upper_3x3[1, 2]).GetLength()
    scale_z = Gf.Vec3d(upper_3x3[2, 0], upper_3x3[2, 1], upper_3x3[2, 2]).GetLength()
    return Gf.Vec3d(scale_x, scale_y, scale_z)


def combine_transforms(transform_a: Gf.Matrix4d, transform_b: Gf.Matrix4d) -> Gf.Matrix4d:
    """Combine two transform matrices into a single transform matrix.

    Args:
        transform_a (Gf.Matrix4d): The first transform matrix.
        transform_b (Gf.Matrix4d): The second transform matrix.

    Returns:
        Gf.Matrix4d: The combined transform matrix.
    """
    if not isinstance(transform_a, Gf.Matrix4d) or not isinstance(transform_b, Gf.Matrix4d):
        raise TypeError("Input matrices must be of type Gf.Matrix4d")
    combined_transform = transform_a * transform_b
    return combined_transform


def compute_prim_distance(stage: Usd.Stage, prim1_path: str, prim2_path: str) -> float:
    """Compute the distance between two prims in world space.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim1_path (str): The path of the first prim.
        prim2_path (str): The path of the second prim.

    Returns:
        float: The distance between the two prims in world space.
    """
    prim1 = stage.GetPrimAtPath(prim1_path)
    prim2 = stage.GetPrimAtPath(prim2_path)
    if not prim1.IsValid() or not prim2.IsValid():
        raise ValueError("One or both prims are invalid.")
    xformable1 = UsdGeom.Xformable(prim1)
    xformable2 = UsdGeom.Xformable(prim2)
    if not xformable1 or not xformable2:
        raise ValueError("One or both prims are not transformable.")
    matrix1 = xformable1.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    matrix2 = xformable2.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translate1 = matrix1.ExtractTranslation()
    translate2 = matrix2.ExtractTranslation()
    distance = (translate2 - translate1).GetLength()
    return distance


def get_world_matrix(stage: Usd.Stage, prim_path: str) -> Gf.Matrix4d:
    """Get the world transformation matrix for a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Returns:
        Gf.Matrix4d: The world transformation matrix.

    Raises:
        ValueError: If the prim is not valid.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if xformable:
        local_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    else:
        local_matrix = Gf.Matrix4d(1)
    parent_prim = prim.GetParent()
    if parent_prim.IsValid():
        parent_matrix = get_world_matrix(stage, parent_prim.GetPath())
    else:
        parent_matrix = Gf.Matrix4d(1)
    world_matrix = local_matrix * parent_matrix
    return world_matrix


def get_prim_translation(prim: Usd.Prim) -> Gf.Vec3f:
    """Get the translation of a prim.

    Args:
        prim (Usd.Prim): The prim to get the translation for.

    Returns:
        Gf.Vec3f: The translation of the prim. Returns (0, 0, 0) if no translation is set.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim '{prim.GetPath()}' is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim '{prim.GetPath()}' is not transformable.")
    xform_ops = xformable.GetOrderedXformOps()
    for op in xform_ops:
        if op.GetOpName() == "translate":
            return op.Get(Usd.TimeCode.Default()) or Gf.Vec3f(0, 0, 0)
    return Gf.Vec3f(0, 0, 0)


def set_prim_translation(stage: Usd.Stage, prim_path: str, translation: Gf.Vec3f) -> None:
    """Set the translation for a prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.
        translation (Gf.Vec3f): The translation to set.

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    translate_op = add_translate_op(xformable)
    translate_op.Set(translation)


def compute_local_to_world_matrix(prim: Usd.Prim) -> Gf.Matrix4d:
    """Compute the local to world transform matrix for a prim.

    Args:
        prim (Usd.Prim): The input prim.

    Returns:
        Gf.Matrix4d: The local to world transform matrix.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return Gf.Matrix4d(1.0)
    local_transform = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    parent_prim = prim.GetParent()
    if parent_prim.IsValid():
        parent_matrix = compute_local_to_world_matrix(parent_prim)
        return local_transform * parent_matrix
    else:
        return local_transform


def orthonormalize_matrix(matrix: Gf.Matrix4f, issue_warning: bool = True) -> bool:
    """
    Orthonormalize the given matrix in place.

    This is an iterative method that is much more stable than the previous
    cross/cross method. If the iterative method does not converge, a
    warning is issued.

    Args:
        matrix (Gf.Matrix4f): The matrix to orthonormalize.
        issue_warning (bool): If True, a warning will be issued if the iteration
            does not converge. Default is True.

    Returns:
        bool: True if the iteration converged, False otherwise.
    """
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationMatrix()
    max_iterations = 10
    epsilon = 1e-06
    for _ in range(max_iterations):
        x_axis = Gf.Vec3f(rotation[0, 0], rotation[0, 1], rotation[0, 2])
        y_axis = Gf.Vec3f(rotation[1, 0], rotation[1, 1], rotation[1, 2])
        z_axis = Gf.Vec3f(rotation[2, 0], rotation[2, 1], rotation[2, 2])
        error = abs(x_axis.GetLength() - 1) + abs(y_axis.GetLength() - 1) + abs(z_axis.GetLength() - 1)
        if error < epsilon:
            break
        x_axis.Normalize()
        y_axis = (y_axis - x_axis * x_axis.Dot(y_axis)).GetNormalized()
        z_axis = x_axis.Cross(y_axis)
        rotation.SetRow(0, Gf.Vec3f(x_axis))
        rotation.SetRow(1, Gf.Vec3f(y_axis))
        rotation.SetRow(2, Gf.Vec3f(z_axis))
    else:
        if issue_warning:
            print("Warning: orthonormalize_matrix did not converge.")
        return False
    matrix.SetRotateOnly(rotation)
    matrix.SetTranslateOnly(translation)
    return True


def set_prim_relative_matrix(stage: Usd.Stage, prim_path: str, relative_to_path: str, matrix: Gf.Matrix4d) -> None:
    """Set the local transform matrix of a prim relative to another prim.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_path (str): The path of the prim to set the matrix on.
        relative_to_path (str): The path of the prim to set the matrix relative to.
        matrix (Gf.Matrix4d): The matrix to set on the prim.

    Raises:
        ValueError: If either prim is not valid or transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    relative_to_prim = stage.GetPrimAtPath(relative_to_path)
    if not relative_to_prim.IsValid():
        raise ValueError(f"Prim at path {relative_to_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    relative_to_xformable = UsdGeom.Xformable(relative_to_prim)
    if not relative_to_xformable:
        raise ValueError(f"Prim at path {relative_to_path} is not transformable.")
    relative_matrix = (
        relative_to_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        * matrix
        * xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    )
    transform_op = xformable.MakeMatrixXform()
    transform_op.Set(relative_matrix)


def get_prim_local_bbox(stage: Usd.Stage, prim_path: str) -> Gf.Range3f:
    """Get the local bounding box for a prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.

    Returns:
        Gf.Range3f: The local bounding box of the prim.

    Raises:
        ValueError: If the prim is not valid or does not have a defined bounding box.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    bbox_attr = UsdGeom.Boundable(prim).GetExtentAttr()
    if not bbox_attr.HasValue():
        raise ValueError(f"Prim at path {prim_path} does not have a defined bounding box.")
    bbox = bbox_attr.Get()
    return bbox


def compute_hierarchy_average_transform(stage: Usd.Stage, prim_path: str) -> Optional[Gf.Matrix4d]:
    """
    Compute the average transform for a prim and its descendants.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Returns:
        Optional[Gf.Matrix4d]: The average transform, or None if no transforms are found.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    transforms = []
    for descendant in Usd.PrimRange(prim):
        if descendant.IsA(UsdGeom.Xformable):
            transform = UsdGeom.Xformable(descendant).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            transforms.append(transform)
    if not transforms:
        return None
    avg_transform = Gf.Matrix4d(0)
    for transform in transforms:
        avg_transform += transform
    avg_transform *= 1.0 / len(transforms)
    return avg_transform


def extract_shear(matrix: Gf.Matrix4d) -> Tuple[float, float, float]:
    """Extract the shear components from a 4x4 matrix.

    Args:
        matrix (Gf.Matrix4d): The input matrix.

    Returns:
        Tuple[float, float, float]: The shear components (shear_xy, shear_xz, shear_yz).
    """
    rotation_matrix = matrix.ExtractRotationMatrix()
    inverse_rotation = rotation_matrix.GetInverse()
    scale_x = Gf.Vec3d(matrix[0][0], matrix[0][1], matrix[0][2]).GetLength()
    scale_y = Gf.Vec3d(matrix[1][0], matrix[1][1], matrix[1][2]).GetLength()
    scale_z = Gf.Vec3d(matrix[2][0], matrix[2][1], matrix[2][2]).GetLength()
    shear_matrix = (
        matrix
        * Gf.Matrix4d(
            1.0 / scale_x, 0.0, 0.0, 0.0, 0.0, 1.0 / scale_y, 0.0, 0.0, 0.0, 0.0, 1.0 / scale_z, 0.0, 0.0, 0.0, 0.0, 1.0
        )
        * Gf.Matrix4d(
            inverse_rotation[0][0],
            inverse_rotation[0][1],
            inverse_rotation[0][2],
            0.0,
            inverse_rotation[1][0],
            inverse_rotation[1][1],
            inverse_rotation[1][2],
            0.0,
            inverse_rotation[2][0],
            inverse_rotation[2][1],
            inverse_rotation[2][2],
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    )
    shear_xy = shear_matrix.GetRow3(0)[1]
    shear_xz = shear_matrix.GetRow3(0)[2]
    shear_yz = shear_matrix.GetRow3(1)[2]
    return (shear_xy, shear_xz, shear_yz)


def set_prim_matrix(stage: Usd.Stage, prim_path: str, matrix: Gf.Matrix4d) -> None:
    """Set the local transform matrix for a prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path of the prim to set the transform for.
        matrix (Gf.Matrix4d): The transformation matrix to set.

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    transform_op = xformable.AddTransformOp()
    transform_op.Set(matrix)


def apply_hierarchical_transform(stage: Usd.Stage, prim_path: str, matrix: Gf.Matrix4d) -> None:
    """Apply a transform matrix to a prim and its descendants hierarchically.

    Args:
        stage (Usd.Stage): The stage containing the prim hierarchy.
        prim_path (str): The path of the prim to apply the transform to.
        matrix (Gf.Matrix4d): The transformation matrix to apply.

    Raises:
        ValueError: If the prim at the given path is not valid or transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    xformable.AddTransformOp().Set(matrix)
    for child in prim.GetChildren():
        child_path = child.GetPath().pathString
        apply_hierarchical_transform(stage, child_path, matrix)


def get_bounding_box(stage: Usd.Stage, prim_path: str) -> Gf.Range3d:
    """Get the bounding box for a prim in world space."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    bounding_box = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
    if bounding_box.GetRange().IsEmpty():
        return Gf.Range3d()
    return bounding_box.GetRange()


def get_prim_world_bbox(prim: Usd.Prim, time_code: Usd.TimeCode = Usd.TimeCode.Default()) -> Gf.Range3d:
    """
    Get the world-space bounding box for a prim at a specific time.

    Args:
        prim (Usd.Prim): The prim to compute the bounding box for.
        time_code (Usd.TimeCode): The time code at which to compute the bounding box.
            Defaults to Default time.

    Returns:
        Gf.Range3d: The computed world-space bounding box.

    Raises:
        ValueError: If the input prim is not valid or does not have a defined extent.
    """
    if not prim.IsValid():
        raise ValueError("Input prim is not valid.")
    bbox_cache = UsdGeom.BBoxCache(time_code, includedPurposes=[UsdGeom.Tokens.default_])
    local_bbox = bbox_cache.ComputeLocalBound(prim)
    if local_bbox.GetRange().IsEmpty():
        raise ValueError("Input prim does not have a defined extent.")
    xform_cache = UsdGeom.XformCache(time_code)
    world_transform = xform_cache.GetLocalToWorldTransform(prim)
    world_bbox = local_bbox.ComputeAlignedRange()
    world_bbox.SetMin(world_transform.Transform(world_bbox.GetMin()))
    world_bbox.SetMax(world_transform.Transform(world_bbox.GetMax()))
    return world_bbox


def extract_prim_euler_rotation(prim: Usd.Prim) -> Gf.Vec3d:
    """Extract the euler rotation in degrees from a prim's transform matrix.

    Args:
        prim (Usd.Prim): The prim to extract rotation from.

    Returns:
        Gf.Vec3d: The euler rotation in degrees (x, y, z).
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    matrix = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    rotation_quat = matrix.ExtractRotationQuat()
    rotation_euler = Gf.Rotation(rotation_quat).Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
    rotation_degrees = Gf.Vec3d(
        Gf.RadiansToDegrees(rotation_euler[0]),
        Gf.RadiansToDegrees(rotation_euler[1]),
        Gf.RadiansToDegrees(rotation_euler[2]),
    )
    return rotation_degrees


def set_prim_euler_rotation(
    stage: Usd.Stage, prim_path: str, rotation_euler: Gf.Vec3f, rotation_order: str = "XYZ"
) -> None:
    """Set the rotation for a prim using Euler angles.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        rotation_euler (Gf.Vec3f): The Euler angles (in radians) for the rotation.
        rotation_order (str, optional): The order of the Euler rotations (e.g., "XYZ", "YZX", etc.). Defaults to "XYZ".

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            xformable.RemoveXformOp(op)
    rotation_op = add_rotate_xyz_op(xformable, UsdGeom.XformOp.PrecisionFloat, rotation_order)
    rotation_op.Set(rotation_euler, Usd.TimeCode.Default())


def extract_rotation(matrix: Gf.Matrix4d) -> Gf.Quatd:
    """Extract the rotation from a 4x4 matrix as a quaternion.

    Args:
        matrix (Gf.Matrix4d): The input 4x4 matrix.

    Returns:
        Gf.Quatd: The extracted rotation as a quaternion.
    """
    ortho_matrix = matrix.GetOrthonormalized(issueWarning=True)
    rotation_quat = ortho_matrix.ExtractRotationQuat()
    return rotation_quat


def apply_shear_to_prim(prim: Usd.Prim, shear_matrix: Gf.Matrix4d) -> None:
    """Apply a shear transformation to a prim.

    Args:
        prim (Usd.Prim): The prim to apply the shear to.
        shear_matrix (Gf.Matrix4d): The shear matrix to apply.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim '{prim.GetPath()}' is not transformable")
    transform_matrix = xformable.GetLocalTransformation()
    new_transform_matrix = transform_matrix * shear_matrix
    xform_op = xformable.AddTransformOp()
    xform_op.Set(new_transform_matrix)


def get_prim_world_transform(stage: Usd.Stage, prim_path: str) -> Gf.Matrix4d:
    """
    Get the world transformation matrix for a prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.

    Returns:
        Gf.Matrix4d: The world transformation matrix for the prim.

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    local_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    parent_transform = Gf.Matrix4d(1)
    parent_prim = prim.GetParent()
    while parent_prim.IsValid():
        parent_xformable = UsdGeom.Xformable(parent_prim)
        if parent_xformable:
            parent_local_transform = parent_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            parent_transform *= parent_local_transform
        parent_prim = parent_prim.GetParent()
    world_transform = local_transform * parent_transform
    return world_transform


def compute_transform_difference(matrix1: Gf.Matrix4f, matrix2: Gf.Matrix4f) -> Gf.Matrix4f:
    """
    Compute the difference between two transform matrices.

    The difference is computed as: matrix2 * inverse(matrix1)
    This represents the transform needed to go from matrix1 to matrix2.

    Args:
        matrix1 (Gf.Matrix4f): The first matrix.
        matrix2 (Gf.Matrix4f): The second matrix.

    Returns:
        Gf.Matrix4f: The difference matrix.
    """
    if matrix1.GetDeterminant() == 0.0 or matrix2.GetDeterminant() == 0.0:
        raise ValueError("One or both input matrices are singular.")
    inverse_matrix1 = matrix1.GetInverse()
    difference_matrix = matrix2 * inverse_matrix1
    return difference_matrix


def get_root_transform(stage: Usd.Stage) -> Gf.Matrix4d:
    """
    Get the root transform matrix of the stage.

    Args:
        stage (Usd.Stage): The USD stage.

    Returns:
        Gf.Matrix4d: The root transform matrix.
    """
    root_prim = stage.GetPseudoRoot()
    if not root_prim.IsValid():
        raise ValueError("Invalid stage. The stage has no root prim.")
    xformable = UsdGeom.Xformable(root_prim)
    if not xformable:
        return Gf.Matrix4d(1)
    transform_matrix = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    return transform_matrix


def extract_rotation_from_matrix(matrix: Gf.Matrix4f) -> Gf.Matrix4f:
    """Extract the rotation component from a 4x4 matrix.

    Args:
        matrix (Gf.Matrix4f): Input 4x4 matrix.

    Returns:
        Gf.Matrix4f: 4x4 matrix representing the rotation component.
    """
    if not matrix:
        raise ValueError("Invalid input matrix")
    rotation_matrix = Gf.Matrix4f()
    for i in range(3):
        for j in range(3):
            rotation_matrix[i, j] = matrix[i, j]
    rotation_matrix.Orthonormalize(True)
    return rotation_matrix


def extract_translation_from_matrix(matrix: Gf.Matrix4f) -> Gf.Vec3f:
    """Extract the translation component from a 4x4 matrix.

    Args:
        matrix (Gf.Matrix4f): The input 4x4 matrix.

    Returns:
        Gf.Vec3f: The translation component of the matrix.
    """
    if not isinstance(matrix, Gf.Matrix4f):
        raise TypeError("Input must be a Gf.Matrix4f")
    translation = Gf.Vec3f(matrix[3, 0], matrix[3, 1], matrix[3, 2])
    return translation


def set_rotation_in_matrix(matrix: Gf.Matrix4f, rotation: Gf.Matrix4f) -> Gf.Matrix4f:
    """Set the rotation part of a matrix.

    Args:
        matrix (Gf.Matrix4f): The matrix to set the rotation in.
        rotation (Gf.Matrix4f): The rotation matrix to set.

    Returns:
        Gf.Matrix4f: The updated matrix with the new rotation.
    """
    translation = matrix.ExtractTranslation()
    result = Gf.Matrix4f(
        rotation[0, 0],
        rotation[0, 1],
        rotation[0, 2],
        0,
        rotation[1, 0],
        rotation[1, 1],
        rotation[1, 2],
        0,
        rotation[2, 0],
        rotation[2, 1],
        rotation[2, 2],
        0,
        translation[0],
        translation[1],
        translation[2],
        1,
    )
    return result


def set_scale_in_matrix(matrix: Gf.Matrix4f, scale: Gf.Vec3f) -> Gf.Matrix4f:
    """Set the scale components of a 4x4 matrix.

    Args:
        matrix (Gf.Matrix4f): The input 4x4 matrix.
        scale (Gf.Vec3f): The scale vector (sx, sy, sz).

    Returns:
        Gf.Matrix4f: The modified matrix with the scale components set.
    """
    result = Gf.Matrix4f(matrix)
    result[0, 0] = scale[0]
    result[1, 1] = scale[1]
    result[2, 2] = scale[2]
    return result


def set_translation_in_matrix(matrix: Gf.Matrix4f, translation: Gf.Vec3f) -> Gf.Matrix4f:
    """Set the translation component of a 4x4 matrix.

    Args:
        matrix (Gf.Matrix4f): The input 4x4 matrix.
        translation (Gf.Vec3f): The translation vector to set.

    Returns:
        Gf.Matrix4f: The updated matrix with the new translation.
    """
    if not isinstance(matrix, Gf.Matrix4f):
        raise TypeError("Input matrix must be of type Gf.Matrix4f")
    if not isinstance(translation, Gf.Vec3f):
        raise TypeError("Input translation must be of type Gf.Vec3f")
    result = Gf.Matrix4f(matrix)
    result.SetTranslateOnly(translation)
    return result


def get_handness_of_matrix(matrix: Gf.Matrix4f) -> float:
    """Returns the sign of the determinant of the upper 3x3 matrix.

    1 for a right-handed matrix, -1 for a left-handed matrix,
    and 0 for a singular matrix.
    """
    upper_3x3 = Gf.Matrix3f(
        matrix[0, 0],
        matrix[0, 1],
        matrix[0, 2],
        matrix[1, 0],
        matrix[1, 1],
        matrix[1, 2],
        matrix[2, 0],
        matrix[2, 1],
        matrix[2, 2],
    )
    det = upper_3x3.GetDeterminant()
    if det > 0:
        return 1.0
    elif det < 0:
        return -1.0
    else:
        return 0.0


def get_matrix_row(matrix: Gf.Matrix4f, row_index: int) -> Gf.Vec4f:
    """Get a row of a Matrix4f as a Vec4f.

    Args:
        matrix (Gf.Matrix4f): The input matrix.
        row_index (int): The index of the row to retrieve (0-3).

    Returns:
        Gf.Vec4f: The row vector.

    Raises:
        IndexError: If row_index is outside the valid range of 0 to 3.
    """
    if row_index < 0 or row_index > 3:
        raise IndexError(f"Invalid row index {row_index}. Must be 0, 1, 2 or 3.")
    row = matrix.GetRow(row_index)
    return row


def set_matrix_row(matrix: Gf.Matrix4f, row_index: int, row_values: Gf.Vec4f) -> None:
    """Set the values of a specific row in a 4x4 matrix.

    Args:
        matrix (Gf.Matrix4f): The matrix to modify.
        row_index (int): The index of the row to set (0-3).
        row_values (Gf.Vec4f): The new values for the row.

    Raises:
        ValueError: If the row index is out of range.
    """
    if row_index < 0 or row_index > 3:
        raise ValueError(f"Invalid row index: {row_index}. Must be between 0 and 3.")
    for col in range(4):
        matrix[row_index, col] = row_values[col]


def transform_affine(matrix: Gf.Matrix4f, vec: Gf.Vec3f) -> Gf.Vec3f:
    """
    Transforms a 3D vector by a 4x4 affine transformation matrix.

    Args:
        matrix (Gf.Matrix4f): The 4x4 affine transformation matrix.
        vec (Gf.Vec3f): The 3D vector to transform.

    Returns:
        Gf.Vec3f: The transformed 3D vector.
    """
    vec4 = Gf.Vec4f(vec[0], vec[1], vec[2], 1.0)
    transformed_vec4 = matrix * vec4
    transformed_vec = Gf.Vec3f(transformed_vec4[0], transformed_vec4[1], transformed_vec4[2])
    return transformed_vec


def set_matrix_to_identity(matrix: Gf.Matrix4f) -> Gf.Matrix4f:
    """
    Set the given matrix to the identity matrix.

    Args:
        matrix (Gf.Matrix4f): The input matrix to be set to identity.

    Returns:
        Gf.Matrix4f: The modified matrix set to identity.
    """
    if not isinstance(matrix, Gf.Matrix4f):
        raise TypeError("Input matrix must be of type Gf.Matrix4f")
    matrix.SetIdentity()
    return matrix


def set_matrix_translation_only(matrix: Gf.Matrix4f, translation: Gf.Vec3f) -> None:
    """Set the translation component of a matrix without affecting other components.

    Args:
        matrix (Gf.Matrix4f): The input matrix to modify.
        translation (Gf.Vec3f): The translation vector to set.
    """
    (m00, m01, m02, _) = matrix.GetRow(0)
    (m10, m11, m12, _) = matrix.GetRow(1)
    (m20, m21, m22, _) = matrix.GetRow(2)
    matrix.SetRow(3, Gf.Vec4f(translation[0], translation[1], translation[2], 1))
    matrix.SetRow(0, Gf.Vec4f(m00, m01, m02, 0))
    matrix.SetRow(1, Gf.Vec4f(m10, m11, m12, 0))
    matrix.SetRow(2, Gf.Vec4f(m20, m21, m22, 0))


def recompose_matrix(r: Gf.Matrix4f, s: Gf.Vec3f, u: Gf.Matrix4f, t: Gf.Vec3f, p: Gf.Matrix4f) -> Gf.Matrix4f:
    """Recompose a matrix from its factored components.

    The matrix is recomposed as: M = r * s * -r * u * t * p

    Args:
        r (Gf.Matrix4f): Rotation matrix.
        s (Gf.Vec3f): Scale vector.
        u (Gf.Matrix4f): Rotation matrix with shear information.
        t (Gf.Vec3f): Translation vector.
        p (Gf.Matrix4f): Projection matrix.

    Returns:
        Gf.Matrix4f: The recomposed matrix.
    """
    scale_matrix = Gf.Matrix4f(1.0).SetScale(s)
    translate_matrix = Gf.Matrix4f(1.0).SetTranslate(t)
    r_inv = r.GetTranspose()
    matrix = r * scale_matrix * r_inv * u * translate_matrix * p
    return matrix


def get_matrix_determinant(matrix: Gf.Matrix4f) -> float:
    """Get the determinant of a Matrix4f.

    Args:
        matrix (Gf.Matrix4f): Input matrix to calculate the determinant of.

    Returns:
        float: Determinant value of the input matrix.
    """
    determinant = matrix.GetDeterminant()
    return determinant


def get_matrix_column(matrix: Gf.Matrix4f, column_index: int) -> Gf.Vec4f:
    """Get a column of a Matrix4f as a Vec4f.

    Args:
        matrix (Gf.Matrix4f): The input matrix.
        column_index (int): The index of the column to retrieve (0-3).

    Returns:
        Gf.Vec4f: The column vector at the specified index.

    Raises:
        IndexError: If the column_index is out of bounds (not in range 0-3).
    """
    if column_index < 0 or column_index > 3:
        raise IndexError(f"Column index {column_index} is out of bounds. Must be in range 0-3.")
    column = matrix.GetColumn(column_index)
    return column


def set_matrix_column(matrix: Gf.Matrix4f, col_index: int, value: Gf.Vec4f) -> None:
    """Set a column of a Matrix4f to a Vec4f value.

    Args:
        matrix (Gf.Matrix4f): The matrix to modify.
        col_index (int): The index of the column to set (0-3).
        value (Gf.Vec4f): The Vec4f value to set the column to.

    Raises:
        IndexError: If col_index is outside the valid range [0, 3].
        TypeError: If value is not a Gf.Vec4f.
    """
    if col_index < 0 or col_index > 3:
        raise IndexError(f"Invalid column index {col_index}. Must be in range [0, 3].")
    if not isinstance(value, Gf.Vec4f):
        raise TypeError(f"Invalid value type {type(value)}. Expected Gf.Vec4f.")
    matrix[col_index] = tuple(value)


def set_matrix_to_zero() -> Gf.Matrix4f:
    """Set the matrix to zero."""
    matrix = Gf.Matrix4f()
    for i in range(4):
        for j in range(4):
            matrix[i, j] = 0.0
    return matrix


def get_prim_local_transform(prim: Usd.Prim) -> Gf.Matrix4d:
    """
    Get the local transformation matrix for a prim.

    Args:
        prim (Usd.Prim): The prim to get the local transformation matrix for.

    Returns:
        Gf.Matrix4d: The local transformation matrix of the prim.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    if not prim.IsA(UsdGeom.Xformable):
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    xformable = UsdGeom.Xformable(prim)
    local_transform = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    return local_transform


def set_prim_transform_hierarchy(stage: Usd.Stage, prim_path: str, matrix: Gf.Matrix4d):
    """Set the local transform of a prim and all its descendants.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        matrix (Gf.Matrix4d): The transformation matrix to apply.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    transform_op = xformable.AddTransformOp()
    transform_op.Set(matrix)
    for child in prim.GetChildren():
        child_xformable = UsdGeom.Xformable(child)
        if child_xformable:
            child_transform_op = child_xformable.AddTransformOp()
            child_transform_op.Set(matrix)


def compute_prim_local_bbox(prim: Usd.Prim) -> Gf.Range3f:
    """Compute the local bounding box for a prim.

    Args:
        prim (Usd.Prim): The prim to compute the bounding box for.

    Returns:
        Gf.Range3f: The local bounding box of the prim.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid")
    bboxCache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bbox = bboxCache.ComputeLocalBound(prim)
    if bbox.GetRange().IsEmpty():
        return Gf.Range3f()
    bbox_range = bbox.GetRange()
    min_point = Gf.Vec3f(bbox_range.GetMin())
    max_point = Gf.Vec3f(bbox_range.GetMax())
    return Gf.Range3f(min_point, max_point)


def set_prim_transform_components(
    stage: Usd.Stage, prim_path: str, translation: Gf.Vec3f, rotation: Gf.Vec3f, scale: Gf.Vec3f
):
    """Sets the transform components (translation, rotation, scale) of a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path of the prim to set the transform components for.
        translation (Gf.Vec3f): The translation vector.
        rotation (Gf.Vec3f): The rotation vector (in degrees).
        scale (Gf.Vec3f): The scale vector.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    if not xform:
        raise ValueError(f"Prim at path {prim_path} is not transformable")
    translate_op = add_translate_op(xform)
    translate_op.Set(translation)
    rotate_op = add_rotate_xyz_op(xform)
    rotation_rad = Gf.Vec3f(
        Gf.DegreesToRadians(rotation[0]), Gf.DegreesToRadians(rotation[1]), Gf.DegreesToRadians(rotation[2])
    )
    rotate_op.Set(rotation_rad)
    scale_op = add_scale_op(xform)
    scale_op.Set(scale)


def extract_scale_from_matrix(matrix: Gf.Matrix4f) -> Gf.Vec3f:
    """Extract the scale from a 4x4 transformation matrix.

    Args:
        matrix (Gf.Matrix4f): The input 4x4 transformation matrix.

    Returns:
        Gf.Vec3f: The scale extracted from the matrix.
    """
    mat3 = Gf.Matrix3f(
        matrix[0, 0],
        matrix[0, 1],
        matrix[0, 2],
        matrix[1, 0],
        matrix[1, 1],
        matrix[1, 2],
        matrix[2, 0],
        matrix[2, 1],
        matrix[2, 2],
    )
    scale_x = Gf.Vec3f(mat3.GetRow(0)).GetLength()
    scale_y = Gf.Vec3f(mat3.GetRow(1)).GetLength()
    scale_z = Gf.Vec3f(mat3.GetRow(2)).GetLength()
    return Gf.Vec3f(scale_x, scale_y, scale_z)


def transform_direction(matrix: Gf.Matrix4f, direction: Gf.Vec3f) -> Gf.Vec3f:
    """Transform a direction vector by a matrix.

    Args:
        matrix (Gf.Matrix4f): The transformation matrix.
        direction (Gf.Vec3f): The direction vector to transform.

    Returns:
        Gf.Vec3f: The transformed direction vector.
    """
    if not matrix.IsRightHanded() or not matrix.HasOrthogonalRows3():
        raise ValueError("The matrix must be orthonormal and right-handed.")
    rotation_matrix = matrix.ExtractRotationMatrix()
    rotation_matrix_4f = Gf.Matrix4f(
        rotation_matrix[0, 0],
        rotation_matrix[0, 1],
        rotation_matrix[0, 2],
        0.0,
        rotation_matrix[1, 0],
        rotation_matrix[1, 1],
        rotation_matrix[1, 2],
        0.0,
        rotation_matrix[2, 0],
        rotation_matrix[2, 1],
        rotation_matrix[2, 2],
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    transformed_direction = rotation_matrix_4f.TransformDir(direction)
    return transformed_direction


def extract_prim_transform_components(stage: Usd.Stage, prim_path: str) -> Tuple[Gf.Vec3f, Gf.Quatf, Gf.Vec3f]:
    """Extract translation, rotation, and scale components from a prim's transform.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Returns:
        A tuple of (translation, rotation, scale).

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    transform_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = transform_matrix.ExtractTranslation()
    rotation = transform_matrix.ExtractRotationQuat()
    scale = Gf.Vec3f(1.0)
    xform_ops = xformable.GetOrderedXformOps()
    for xform_op in xform_ops:
        if xform_op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale = xform_op.Get()
            break
    return (translation, rotation, scale)


def inverse_matrix(matrix: Gf.Matrix4f, det: Optional[float] = None, eps: float = 1e-05) -> Gf.Matrix4f:
    """
    Returns the inverse of the matrix, or FLT_MAX * SetIdentity() if the matrix is singular.

    The matrix is considered singular if the determinant is less than or equal to eps.
    If det is not None, it is set to the determinant of the matrix.

    Parameters:
        matrix (Gf.Matrix4f): The input matrix to invert.
        det (Optional[float]): If provided, will be set to the determinant of the matrix.
        eps (float): The epsilon value to check for matrix singularity. Default is 0.00001.

    Returns:
        Gf.Matrix4f: The inverse of the input matrix, or FLT_MAX * SetIdentity() if the matrix is singular.
    """
    determinant = matrix.GetDeterminant()
    if det is not None:
        det = determinant
    if abs(determinant) <= eps:
        return Gf.Matrix4f().SetIdentity() * float("inf")
    inverse = matrix.GetInverse(determinant, eps)
    return inverse


def remove_scale_shear_from_matrix(matrix: Gf.Matrix4f) -> Gf.Matrix4f:
    """
    Remove any scaling or shearing from a matrix, leaving only rotation and translation.

    Args:
        matrix (Gf.Matrix4f): Input matrix to remove scale and shear from.

    Returns:
        Gf.Matrix4f: Matrix with scale and shear removed, containing only rotation and translation.
    """
    result = matrix.RemoveScaleShear()
    return result


def apply_transform_to_prim(prim: Usd.Prim, transform: Gf.Transform) -> None:
    """Applies a Gf.Transform to a USD prim.

    Args:
        prim (Usd.Prim): The prim to apply the transform to.
        transform (Gf.Transform): The transform to apply.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    translation = transform.GetTranslation()
    add_translate_op(xformable).Set(translation)
    rotation = transform.GetRotation()
    add_rotate_xyz_op(xformable).Set(rotation.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis()))
    scale = transform.GetScale()
    add_scale_op(xformable).Set(scale)
    pivot_position = transform.GetPivotPosition()
    pivot_orientation = transform.GetPivotOrientation()
    xformable.AddTransformOp(UsdGeom.XformOp.PrecisionFloat, "pivot").Set(
        Gf.Matrix4d().SetTranslate(pivot_position) * Gf.Matrix4d().SetRotate(pivot_orientation)
    )


def get_prim_transform_hierarchy(prim: Usd.Prim) -> Dict[str, Gf.Matrix4d]:
    """
    Get the transform hierarchy for a prim and its descendants.

    Args:
        prim (Usd.Prim): The root prim to start the hierarchy from.

    Returns:
        Dict[str, Gf.Matrix4d]: A dictionary mapping prim paths to their world space transform.
    """
    result = {}
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    stack = [(prim, Gf.Matrix4d())]
    while stack:
        (curr_prim, curr_transform) = stack.pop()
        xformable = UsdGeom.Xformable(curr_prim)
        if xformable:
            local_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            curr_transform = curr_transform * local_transform
        result[str(curr_prim.GetPath())] = curr_transform
        for child_prim in curr_prim.GetChildren():
            stack.append((child_prim, Gf.Matrix4d(curr_transform)))
    return result


def set_prim_world_transform(stage: Usd.Stage, prim_path: str, matrix: Gf.Matrix4d) -> None:
    """Set the world transform for a prim using a Gf.Matrix4d.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        matrix (Gf.Matrix4d): The transformation matrix.

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    xformable.ClearXformOpOrder()
    transform_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTransform)
    transform_op.Set(matrix)


def compute_prim_world_bbox(prim: Usd.Prim, time_code: Usd.TimeCode = Usd.TimeCode.Default()) -> Gf.Range3d:
    """
    Compute the world-space bounding box for a prim at a given time code.

    Args:
        prim (Usd.Prim): The input prim.
        time_code (Usd.TimeCode): The time code at which to compute the bounding box.
            Defaults to Usd.TimeCode.Default().

    Returns:
        Gf.Range3d: The computed world-space bounding box.

    Raises:
        ValueError: If the input prim is not valid or does not have a defined extent.
    """
    if not prim.IsValid():
        raise ValueError("Input prim is not valid.")
    local_bound = UsdGeom.Imageable(prim).ComputeLocalBound(time_code, purpose1="default")
    if local_bound.GetRange().IsEmpty():
        raise ValueError("Prim does not have a defined extent.")
    world_transform = (
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code)
        if prim.IsA(UsdGeom.Xformable)
        else Gf.Matrix4d(1.0)
    )
    local_range = local_bound.GetRange()
    world_bbox = Gf.Range3d(
        world_transform.Transform(local_range.GetMin()), world_transform.Transform(local_range.GetMax())
    )
    return world_bbox


def analyze_scene_intervals(stage: Usd.Stage) -> Gf.MultiInterval:
    """Analyze the time intervals where prims are visible in the scene.

    Args:
        stage (Usd.Stage): The USD stage to analyze.

    Returns:
        Gf.MultiInterval: The combined time intervals where prims are visible.
    """
    combined_interval = Gf.MultiInterval()
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            visibility_attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
            if visibility_attr.HasAuthoredValue():
                time_samples = visibility_attr.GetTimeSamples()
                for time_sample in time_samples:
                    visibility_value = visibility_attr.Get(time_sample)
                    if visibility_value == UsdGeom.Tokens.inherited:
                        combined_interval.ArithmeticAdd(Gf.Interval(time_sample, time_sample))
    return combined_interval


def intersect_prim_intervals(interval1: Gf.MultiInterval, interval2: Gf.MultiInterval) -> Gf.MultiInterval:
    """Intersect two MultiIntervals and return the result.

    Args:
        interval1 (Gf.MultiInterval): First interval to intersect.
        interval2 (Gf.MultiInterval): Second interval to intersect.

    Returns:
        Gf.MultiInterval: The intersection of the two input intervals.
    """
    if interval1.IsEmpty() or interval2.IsEmpty():
        return Gf.MultiInterval()
    result = Gf.MultiInterval()
    result.Intersect(interval1)
    result.Intersect(interval2)
    return result


def get_full_scene_interval(stage: Usd.Stage) -> Gf.MultiInterval:
    """
    Get the full scene interval for a given USD stage.

    Args:
        stage (Usd.Stage): The USD stage to get the full scene interval for.

    Returns:
        Gf.MultiInterval: The full scene interval.
    """
    root_layer = stage.GetRootLayer()
    start_time_code = root_layer.startTimeCode
    end_time_code = root_layer.endTimeCode
    interval = Gf.Interval(start_time_code, end_time_code)
    multi_interval = Gf.MultiInterval()
    multi_interval.Add(interval)
    return multi_interval


def get_complement_intervals(multi_interval: Gf.MultiInterval) -> Gf.MultiInterval:
    """
    Get the complement of the given multi-interval.

    Args:
        multi_interval (Gf.MultiInterval): The input multi-interval.

    Returns:
        Gf.MultiInterval: The complement of the input multi-interval.
    """
    if multi_interval.IsEmpty():
        return Gf.MultiInterval.GetFullInterval()
    complement = multi_interval.GetComplement()
    return complement


def clear_intervals(multi_interval: Gf.MultiInterval) -> None:
    """Clear all intervals from the given MultiInterval."""
    if not isinstance(multi_interval, Gf.MultiInterval):
        raise TypeError("Input must be a Gf.MultiInterval object.")
    multi_interval.Clear()


def get_prim_interval_size(stage: Usd.Stage, prim_path: str, time_code: Usd.TimeCode = Usd.TimeCode.Default()) -> int:
    """Get the number of intervals for a prim's active time samples."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    attr = prim.GetAttribute("xformOp:transform")
    if not attr:
        raise ValueError(f"Prim at path {prim_path} does not have an xformOp:transform attribute.")
    time_samples = attr.GetTimeSamples()
    multi_interval = Gf.MultiInterval()
    for time_sample in time_samples:
        multi_interval.Add(Gf.Interval(time_sample, time_sample))
    return multi_interval.GetSize()


def set_prim_intervals(stage: Usd.Stage, prim_path: str, intervals: Gf.MultiInterval) -> None:
    """Set the visibility intervals for a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        intervals (Gf.MultiInterval): The intervals to set.

    Raises:
        ValueError: If the prim does not exist or is not an Imageable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        raise ValueError(f"Prim at path {prim_path} is not an Imageable.")
    for interval in intervals:
        imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible, Usd.TimeCode(interval.GetMin()))
        imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited, Usd.TimeCode(interval.GetMax()))


def remove_prim_intervals(stage: Usd.Stage, prim_path: str, interval: Gf.Interval) -> bool:
    """Remove the specified interval from the prim's active time samples.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.
        interval (Gf.Interval): The interval to remove.

    Returns:
        bool: True if the interval was removed, False otherwise.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    active_time_samples = prim.GetAttribute("active").Get()
    if not active_time_samples:
        return False
    multi_interval = Gf.MultiInterval()
    for time_sample in active_time_samples:
        multi_interval.Add(Gf.Interval(time_sample, time_sample))
    multi_interval.Remove(interval)
    updated_time_samples = [i.GetMin() for i in multi_interval]
    prim.CreateAttribute("active", Sdf.ValueTypeNames.FloatArray).Set(updated_time_samples)
    return True


def arithmetic_add_intervals(multi_interval: Gf.MultiInterval, interval: Gf.Interval) -> None:
    """
    Uses the given interval to extend the multi-interval in the interval arithmetic sense.

    Parameters:
        multi_interval (Gf.MultiInterval): The multi-interval to extend.
        interval (Gf.Interval): The interval to add to the multi-interval.
    """
    if interval.IsEmpty():
        return
    result = Gf.MultiInterval()
    for i in range(multi_interval.GetSize()):
        current_interval = multi_interval.GetBounds()
        new_interval = current_interval + interval
        result.Add(new_interval)
        multi_interval.Remove(current_interval)
    for i in range(result.GetSize()):
        multi_interval.Add(result.GetBounds())


def align_prim_intervals(stage: Usd.Stage, prim_path: str, target_interval: Gf.Interval) -> bool:
    """Aligns the time samples of a prim's attributes to the target interval.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to align.
        target_interval (Gf.Interval): The target interval to align to.

    Returns:
        bool: True if successful, False otherwise.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    attr_intervals = Gf.MultiInterval()
    for attr in prim.GetAttributes():
        time_samples = attr.GetTimeSamples()
        for time_sample in time_samples:
            attr_intervals.Add(Gf.Interval(time_sample, time_sample))
    if attr_intervals.IsEmpty():
        return True
    for interval in attr_intervals:
        if interval.GetMin() < target_interval.GetMin() or interval.GetMax() > target_interval.GetMax():
            for attr in prim.GetAttributes():
                attr.ClearAtTime(interval.GetMin())
                attr.ClearAtTime(interval.GetMax())
    return True


def optimize_time_samples(attr: Usd.Attribute, time_samples: Sequence[Tuple[float, Any]]) -> List[Tuple[float, Any]]:
    """Optimize the given time samples by removing redundant samples.

    Args:
        attr (Usd.Attribute): The attribute to optimize time samples for.
        time_samples (Sequence[Tuple[float, Any]]): The time samples to optimize.

    Returns:
        List[Tuple[float, Any]]: The optimized time samples.
    """
    if not attr.IsValid():
        raise ValueError("Invalid attribute.")
    if not time_samples:
        return []
    sorted_samples = sorted(time_samples, key=lambda x: x[0])
    optimized_samples = [sorted_samples[0]]
    attr_time_samples = attr.GetTimeSamples()
    for sample in sorted_samples[1:]:
        (time, value) = sample
        (prev_time, prev_value) = optimized_samples[-1]
        if value != prev_value:
            optimized_samples.append(sample)
        else:
            time_diff = time - prev_time
            num_samples = len([t for t in attr_time_samples if prev_time < t < time])
            if time_diff > num_samples:
                optimized_samples.append(sample)
    return optimized_samples


def calculate_distance_between_prims(stage: Usd.Stage, prim_path1: str, prim_path2: str) -> float:
    """
    Calculate the distance between two prims in the scene.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path1 (str): The path to the first prim.
        prim_path2 (str): The path to the second prim.

    Returns:
        float: The distance between the two prims.

    Raises:
        ValueError: If either prim is invalid or not transformable.
    """
    prim1 = stage.GetPrimAtPath(prim_path1)
    prim2 = stage.GetPrimAtPath(prim_path2)
    if not prim1.IsValid() or not prim2.IsValid():
        raise ValueError("One or both prims are invalid.")
    xformable1 = UsdGeom.Xformable(prim1)
    xformable2 = UsdGeom.Xformable(prim2)
    if not xformable1 or not xformable2:
        raise ValueError("One or both prims are not transformable.")
    position1 = xformable1.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    position2 = xformable2.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    distance = Gf.Vec3d(position1 - position2).GetLength()
    return distance


def calculate_prim_normal_to_plane(stage: Usd.Stage, prim_path: str, plane: Gf.Plane) -> Gf.Vec3d:
    """Calculate the normal vector from a prim to a plane.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.
        plane (Gf.Plane): The plane to calculate the normal to.

    Returns:
        Gf.Vec3d: The normal vector from the prim to the plane.

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    prim_position = world_transform.ExtractTranslation()
    projected_position = plane.Project(prim_position)
    normal = projected_position - prim_position
    normal.Normalize()
    return normal


def filter_prims_by_plane_distance(stage: Usd.Stage, plane: Gf.Plane, max_distance: float) -> List[Usd.Prim]:
    """
    Filter prims in a USD stage based on their distance from a plane.

    Args:
        stage (Usd.Stage): The USD stage to filter prims from.
        plane (Gf.Plane): The plane to measure distance from.
        max_distance (float): The maximum distance a prim can be from the plane to be included.

    Returns:
        List[Usd.Prim]: A list of prims that are within the specified distance from the plane.
    """
    prims = []
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
        center = bbox.ComputeCentroid()
        distance = plane.GetDistance(Gf.Vec3d(center))
        if abs(distance) <= max_distance:
            prims.append(prim)
    return prims


def set_prim_plane_equation(prim: Usd.Prim, plane: Gf.Plane) -> None:
    """Set the plane equation attribute on a prim."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    plane_attr = prim.GetAttribute("planeEquation")
    if not plane_attr:
        plane_attr = prim.CreateAttribute("planeEquation", Sdf.ValueTypeNames.Float4)
    plane_eqn = plane.GetEquation()
    plane_attr.Set(Gf.Vec4f(plane_eqn[0], plane_eqn[1], plane_eqn[2], plane_eqn[3]))


def reorient_prims_to_positive_halfspace(stage: Usd.Stage, plane: Gf.Plane):
    """
    Reorients all prims on the stage to the positive halfspace of the given plane.

    Args:
        stage (Usd.Stage): The stage containing the prims to reorient.
        plane (Gf.Plane): The plane used to determine the positive halfspace.
    """
    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Xformable):
            xform = UsdGeom.Xformable(prim)
            transform_matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = transform_matrix.ExtractTranslation()
            if not plane.IntersectsPositiveHalfSpace(Gf.Vec3d(translation)):
                scale_op = None
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        scale_op = op
                        break
                if scale_op:
                    current_scale = scale_op.Get()
                    scale_op.Set(Gf.Vec3d(-current_scale[0], -current_scale[1], -current_scale[2]))
                else:
                    add_scale_op(xform).Set(Gf.Vec3d(-1, -1, -1))


def calculate_distance_from_prims_to_origin(stage: Usd.Stage, prim_paths: List[str]) -> List[float]:
    """Calculate the distance from each prim to the origin using the Gf.Plane class.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths.

    Returns:
        List[float]: List of distances from each prim to the origin.
    """
    distances = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")
        bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
        if bbox.GetRange().IsEmpty():
            raise ValueError(f"Prim {prim_path} has an empty bounding box")
        center = bbox.ComputeCentroid()
        plane = Gf.Plane(Gf.Vec3d(0, 0, 1), center)
        distance = plane.GetDistanceFromOrigin()
        distances.append(distance)
    return distances


def compute_intersection_of_prims(stage: Usd.Stage, prim_path_a: str, prim_path_b: str) -> Gf.Rect2i:
    """Compute the intersection of the bounding boxes of two prims.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_path_a (str): The path to the first prim.
        prim_path_b (str): The path to the second prim.

    Returns:
        Gf.Rect2i: The intersection of the bounding boxes of the two prims.

    Raises:
        ValueError: If either prim is invalid or does not have a bounding box.
    """
    prim_a = stage.GetPrimAtPath(prim_path_a)
    prim_b = stage.GetPrimAtPath(prim_path_b)
    if not prim_a.IsValid() or not prim_b.IsValid():
        raise ValueError("One or both prims are invalid.")
    bbox_a = UsdGeom.Imageable(prim_a).ComputeWorldBound(Usd.TimeCode.Default(), "default")
    bbox_b = UsdGeom.Imageable(prim_b).ComputeWorldBound(Usd.TimeCode.Default(), "default")
    if bbox_a.GetRange().IsEmpty() or bbox_b.GetRange().IsEmpty():
        raise ValueError("One or both prims do not have a valid bounding box.")
    min_a = Gf.Vec2i(int(bbox_a.GetRange().GetMin()[0]), int(bbox_a.GetRange().GetMin()[1]))
    max_a = Gf.Vec2i(int(bbox_a.GetRange().GetMax()[0]), int(bbox_a.GetRange().GetMax()[1]))
    min_b = Gf.Vec2i(int(bbox_b.GetRange().GetMin()[0]), int(bbox_b.GetRange().GetMin()[1]))
    max_b = Gf.Vec2i(int(bbox_b.GetRange().GetMax()[0]), int(bbox_b.GetRange().GetMax()[1]))
    rect_a = Gf.Rect2i(min_a, max_a)
    rect_b = Gf.Rect2i(min_b, max_b)
    intersection = rect_a.GetIntersection(rect_b)
    return intersection


def create_prim(stage, prim_type, path):
    prim = stage.DefinePrim(path, prim_type)
    return prim


def transform_prim_by_plane(prim: Usd.Prim, plane: Gf.Plane) -> None:
    """Transform a prim by a given plane.

    Args:
        prim (Usd.Prim): The prim to transform.
        plane (Gf.Plane): The plane to transform the prim by.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim '{prim.GetPath()}' is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim '{prim.GetPath()}' is not transformable.")
    local_transform = xformable.GetLocalTransformation()
    plane_transform = Gf.Matrix4d(
        Gf.Rotation(Gf.Vec3d(0, 0, 1), plane.normal), Gf.Vec3d(0, 0, plane.distanceFromOrigin)
    )
    new_transform = local_transform * plane_transform
    xform_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTransform)
    xform_op.Set(new_transform)


def calculate_prim_projection_on_plane(stage: Usd.Stage, prim_path: str, plane: Gf.Plane) -> Tuple[Gf.Vec3d, bool]:
    """
    Calculate the projection of a prim's bounding box center onto a plane.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        plane (Gf.Plane): The plane to project onto.

    Returns:
        Tuple[Gf.Vec3d, bool]: A tuple containing the projected point and a boolean indicating if the prim is in the positive half-space of the plane.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")
    bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
    if bbox.GetRange().IsEmpty():
        raise ValueError(f"Prim {prim_path} has an empty bounding box")
    center_point = bbox.ComputeCentroid()
    projected_point = plane.Project(center_point)
    is_positive_side = plane.IntersectsPositiveHalfSpace(center_point)
    return (projected_point, is_positive_side)


def intersect_prims_with_plane(stage: Usd.Stage, plane: Gf.Plane) -> List[Usd.Prim]:
    """
    Finds all prims on the stage that intersect with the given plane.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        plane (Gf.Plane): The plane to test for intersection.

    Returns:
        List[Usd.Prim]: A list of prims that intersect with the plane.
    """
    intersecting_prims: List[Usd.Prim] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        imageable = UsdGeom.Imageable(prim)
        purpose = UsdGeom.Tokens.default_
        bbox = imageable.ComputeLocalBound(Usd.TimeCode.Default(), purpose)
        range_ = bbox.GetRange()
        if plane.IntersectsPositiveHalfSpace(range_.GetMin()) != plane.IntersectsPositiveHalfSpace(range_.GetMax()):
            intersecting_prims.append(prim)
    return intersecting_prims


def get_prim_plane_equation(stage: Usd.Stage, prim_path: str) -> Gf.Vec4d:
    """Get the plane equation for a prim of type UsdGeomGprim.

    The plane equation is represented as a Gf.Vec4d where the first three
    components are the normal vector and the fourth component is the distance
    from the origin.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Returns:
        Gf.Vec4d: The plane equation.

    Raises:
        ValueError: If the prim is not valid or not of type UsdGeomMesh.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        raise ValueError(f"Prim at path {prim_path} is not of type UsdGeomMesh.")
    points = mesh.GetPointsAttr().Get()
    centroid = Gf.Vec3d()
    for pt in points:
        centroid += Gf.Vec3d(pt)
    centroid /= len(points)
    v1 = Gf.Vec3d(points[1] - points[0])
    v2 = Gf.Vec3d(points[-1] - points[0])
    normal = Gf.Cross(v1, v2).GetNormalized()
    distance = Gf.Dot(normal, centroid)
    return Gf.Vec4d(normal[0], normal[1], normal[2], distance)


def project_prims_to_plane(stage: Usd.Stage, plane: Gf.Plane, prim_paths: List[str]) -> None:
    """Projects the given prims onto the specified plane.

    Args:
        stage (Usd.Stage): The USD stage.
        plane (Gf.Plane): The plane to project the prims onto.
        prim_paths (List[str]): The paths of the prims to project.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        translate_ops = xformable.GetOrderedXformOps()
        translate_op = None
        for op in translate_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if not translate_op:
            continue
        translation = translate_op.Get()
        if translation is None:
            continue
        point = Gf.Vec3d(translation[0], translation[1], translation[2])
        projected_point = plane.Project(point)
        translate_op.Set(Gf.Vec3d(projected_point))


def set_identity_rotation_if_small(quat: Gf.Quatd, eps: float = 1e-06) -> None:
    """Set the quaternion to identity rotation if its length is smaller than eps.

    If the length of the quaternion is smaller than eps, set it to the identity
    rotation quaternion (1, 0, 0, 0). Otherwise, leave it unchanged.

    Parameters
    ----------
    quat : Gf.Quatd
        The quaternion to potentially set to identity rotation.
    eps : float, optional
        The threshold length below which the quaternion is considered small, by default 1e-6.
    """
    length = quat.GetLength()
    if length < eps:
        quat.SetReal(1.0)
        quat.SetImaginary(0.0, 0.0, 0.0)


def apply_rotation_to_prims(stage: Usd.Stage, prim_paths: List[str], rotation: Gf.Quatd) -> None:
    """Apply a rotation to a list of prims.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (List[str]): A list of prim paths to apply the rotation to.
        rotation (Gf.Quatd): The rotation quaternion to apply.

    Raises:
        ValueError: If any of the prim paths are invalid or not transformable.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        rotation_op = add_rotate_xyz_op(xformable, UsdGeom.XformOp.PrecisionFloat, "")
        euler_angles = Gf.Rotation(rotation).Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
        rotation_op.Set(euler_angles)


def accumulate_rotations(rotations: List[Gf.Quatd]) -> Gf.Quatd:
    """
    Accumulate a list of rotations into a single rotation.

    Args:
        rotations (List[Gf.Quatd]): List of rotations to accumulate.

    Returns:
        Gf.Quatd: The accumulated rotation.
    """
    if not rotations:
        return Gf.Quatd(1, Gf.Vec3d(0, 0, 0))
    accumulated_rotation = rotations[0]
    for rotation in rotations[1:]:
        accumulated_rotation *= rotation
    accumulated_rotation.Normalize()
    return accumulated_rotation


def set_imaginary_rotation_for_prims(stage: Usd.Stage, prim_paths: List[str], rotation: Gf.Quatd) -> None:
    """Set the imaginary rotation for a list of prims using a quaternion.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths to set the rotation for.
        rotation (Gf.Quatd): The quaternion rotation to set.

    Raises:
        ValueError: If any of the prim paths are invalid or not transformable.
    """
    rotation = rotation.GetNormalized()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        add_rotate_xyz_op(xformable).Set(rotation.GetImaginary(), Usd.TimeCode.Default())


def reset_rotation_to_zero(quat: Gf.Quatd) -> Gf.Quatd:
    """Reset the rotation represented by the quaternion to zero/identity.

    Parameters
    ----------
    quat : Gf.Quatd
        The quaternion to reset.

    Returns
    -------
    Gf.Quatd
        The quaternion with rotation reset to zero/identity.
    """
    if not isinstance(quat, Gf.Quatd):
        raise TypeError("Input must be a Gf.Quatd")
    reset_quat = Gf.Quatd(quat.GetReal(), Gf.Vec3d(0, 0, 0))
    reset_quat.Normalize()
    return reset_quat


def find_prim_with_smallest_rotation(stage: Usd.Stage) -> Tuple[Usd.Prim, float]:
    """
    Find the prim on the given stage with the smallest rotation magnitude.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.

    Returns:
        Tuple[Usd.Prim, float]: The prim with the smallest rotation magnitude and the magnitude value.
                                 If no prims have rotations, returns (None, 0.0).
    """
    smallest_rotation_prim = None
    smallest_rotation_magnitude = float("inf")
    for prim in stage.TraverseAll():
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            continue
        rotation_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ]
        if not rotation_ops:
            continue
        rotation_op = rotation_ops[0]
        rotation = rotation_op.Get(Usd.TimeCode.Default())
        rotation_quat = Gf.Quatd()
        rotation_quat.SetImaginary(Gf.Vec3d(rotation))
        rotation_magnitude = rotation_quat.GetLength()
        if rotation_magnitude < smallest_rotation_magnitude:
            smallest_rotation_prim = prim
            smallest_rotation_magnitude = rotation_magnitude
    return (smallest_rotation_prim, smallest_rotation_magnitude)


def apply_inverse_rotation(quat: Gf.Quatd, vec: Gf.Vec3d) -> Gf.Vec3d:
    """
    Apply the inverse rotation represented by the quaternion to the given vector.

    Parameters:
        quat (Gf.Quatd): The quaternion representing the rotation.
        vec (Gf.Vec3d): The vector to rotate.

    Returns:
        Gf.Vec3d: The rotated vector.
    """
    if not Gf.IsClose(quat.GetLength(), 1.0, 1e-06):
        quat = Gf.Quatd(quat).GetNormalized(1e-06)
    inv_quat = quat.GetInverse()
    rotated_vec = inv_quat.Transform(vec)
    return rotated_vec


def blend_rotations(a: Gf.Quatd, b: Gf.Quatd, t: float) -> Gf.Quatd:
    """Blend between two quaternions.

    Args:
        a (Gf.Quatd): First quaternion.
        b (Gf.Quatd): Second quaternion.
        t (float): Blending factor between 0 and 1.

    Returns:
        Gf.Quatd: Blended quaternion.
    """
    a = a.GetNormalized()
    b = b.GetNormalized()
    if t < 0.0 or t > 1.0:
        raise ValueError("Blending factor must be between 0 and 1.")
    if Gf.IsClose(a.GetReal(), -b.GetReal(), 1e-05) and Gf.IsClose(a.GetImaginary(), -b.GetImaginary(), 1e-05):
        angle = Gf.Acos(a.GetReal())
        axis = a.GetImaginary().GetNormalized()
        blend_angle = (1.0 - t) * -angle + t * angle
        return Gf.Quatd(Gf.Cos(blend_angle), axis * Gf.Sin(blend_angle))
    else:
        return ((1.0 - t) * a + t * b).GetNormalized()


def get_prim_rotation_length(stage: Usd.Stage, prim_path: str) -> float:
    """Get the length of the rotation quaternion for a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    rotation_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ]
    if not rotation_ops:
        return Gf.Quatd.GetIdentity().GetLength()
    rotation_op = rotation_ops[0]
    rotation_attr = rotation_op.GetAttr()
    if not rotation_attr.HasValue():
        return Gf.Quatd.GetIdentity().GetLength()
    rotation_vec3d = rotation_attr.Get()
    rotation_quat = (
        Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation_vec3d[0]).GetQuaternion()
        * Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation_vec3d[1]).GetQuaternion()
        * Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation_vec3d[2]).GetQuaternion()
    )
    return rotation_quat.GetLength()


def align_prims_rotation(source_prim: UsdGeom.Xformable, target_prim: UsdGeom.Xformable) -> None:
    """
    Align the rotation of the target prim to match the rotation of the source prim.

    Args:
        source_prim (UsdGeom.Xformable): The source prim whose rotation will be used as reference.
        target_prim (UsdGeom.Xformable): The target prim whose rotation will be aligned to the source prim.

    Raises:
        ValueError: If either the source or target prim is not a valid Xformable prim.
    """
    if not source_prim:
        raise ValueError("Invalid source prim. It must be a valid Xformable.")
    if not target_prim:
        raise ValueError("Invalid target prim. It must be a valid Xformable.")
    source_rotation_op = source_prim.GetOrderedXformOps()
    source_rotation = Gf.Vec3f(0, 0, 0)
    for op in source_rotation_op:
        if op.GetOpName() == "rotateXYZ":
            source_rotation = op.Get()
            break
    target_rotation_op = add_rotate_xyz_op(target_prim)
    target_rotation_op.Set(source_rotation)


def distribute_rotation_across_prims(stage: Usd.Stage, prims: List[str], rotation: Gf.Quatf) -> None:
    """Distribute a rotation quaternion across multiple prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prims (List[str]): List of prim paths.
        rotation (Gf.Quatf): The rotation quaternion to distribute.

    Raises:
        ValueError: If any prim path is invalid or not transformable.
    """
    rotation.Normalize()
    for prim_path in prims:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        add_orient_op(xformable).Set(rotation)


def get_prims_with_inverted_rotation(stage: Usd.Stage, root_prim_path: str) -> List[Usd.Prim]:
    """
    Return a list of prims under the given root prim path that have an inverted rotation.

    An inverted rotation is defined as a rotation where the real component of the
    quaternion is negative.

    Args:
        stage (Usd.Stage): The USD stage to query.
        root_prim_path (str): The path to the root prim to start the search from.

    Returns:
        List[Usd.Prim]: A list of prims with inverted rotations.
    """
    prims_with_inverted_rotation = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_prim_path)):
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            rotation_ops = [
                op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ
            ]
            if rotation_ops:
                rotation_op = rotation_ops[0]
                rotation_vec = rotation_op.Get(Usd.TimeCode.Default())
                rotation_quat = (
                    Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation_vec[0])
                    * Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation_vec[1])
                    * Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation_vec[2])
                )
                quat = rotation_quat.GetQuat()
                if quat.GetReal() < 0:
                    prims_with_inverted_rotation.append(prim)
    return prims_with_inverted_rotation


def blend_quaternions(q1: Gf.Quath, q2: Gf.Quath, bias: float) -> Gf.Quath:
    """Blend two quaternions using spherical linear interpolation (SLERP).

    Args:
        q1 (Gf.Quath): The first quaternion.
        q2 (Gf.Quath): The second quaternion.
        bias (float): The bias value between 0 and 1. 0 favors q1, 1 favors q2.

    Returns:
        Gf.Quath: The blended quaternion.
    """
    if not isinstance(q1, Gf.Quath) or not isinstance(q2, Gf.Quath):
        raise TypeError("Input quaternions must be of type Gf.Quath")
    if bias < 0 or bias > 1:
        raise ValueError("Bias value must be between 0 and 1")
    if q1 == Gf.Quath.GetIdentity():
        return q2
    if q2 == Gf.Quath.GetIdentity():
        return q1
    cos_angle = Gf.Dot(q1.GetImaginary(), q2.GetImaginary()) + q1.GetReal() * q2.GetReal()
    if cos_angle < 0:
        q2 = -q2
        cos_angle = -cos_angle
    if cos_angle > 0.9999:
        result = (1 - bias) * q1 + bias * q2
        return Gf.Quath(result.GetReal(), result.GetImaginary())
    angle = Gf.Acos(cos_angle)
    sin_angle = Gf.Sqrt(1 - cos_angle * cos_angle)
    factor1 = Gf.Sin((1 - bias) * angle) / sin_angle
    factor2 = Gf.Sin(bias * angle) / sin_angle
    result = factor1 * q1 + factor2 * q2
    return Gf.Quath(result.GetReal(), result.GetImaginary()).GetNormalized()


def convert_euler_to_quaternion(euler_angles: Gf.Vec3d, rotation_order: str = "XYZ") -> Gf.Quaternion:
    """
    Convert Euler angles to a quaternion.

    Args:
        euler_angles (Gf.Vec3d): The Euler angles in radians (x, y, z).
        rotation_order (str): The order of the rotations (default: "XYZ").

    Returns:
        Gf.Quaternion: The quaternion representing the rotation.
    """
    if rotation_order not in ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]:
        raise ValueError(f"Invalid rotation order: {rotation_order}")
    (x, y, z) = euler_angles
    half_x = x * 0.5
    half_y = y * 0.5
    half_z = z * 0.5
    cx = math.cos(half_x)
    cy = math.cos(half_y)
    cz = math.cos(half_z)
    sx = math.sin(half_x)
    sy = math.sin(half_y)
    sz = math.sin(half_z)
    if rotation_order == "XYZ":
        qw = cx * cy * cz + sx * sy * sz
        qx = sx * cy * cz - cx * sy * sz
        qy = cx * sy * cz + sx * cy * sz
        qz = cx * cy * sz - sx * sy * cz
    elif rotation_order == "XZY":
        qw = cx * cy * cz - sx * sy * sz
        qx = sx * cy * cz + cx * sy * sz
        qy = cx * sy * cz + sx * cy * sz
        qz = cx * cy * sz - sx * sy * cz
    elif rotation_order == "YXZ":
        qw = cx * cy * cz + sx * sy * sz
        qx = sx * cy * cz - cx * sy * sz
        qy = cx * sy * cz - sx * cy * sz
        qz = cx * cy * sz + sx * sy * cz
    elif rotation_order == "YZX":
        qw = cx * cy * cz - sx * sy * sz
        qx = sx * cy * cz + cx * sy * sz
        qy = cx * sy * cz + sx * cy * sz
        qz = cx * cy * sz - sx * sy * cz
    elif rotation_order == "ZXY":
        qw = cx * cy * cz - sx * sy * sz
        qx = sx * cy * cz + cx * sy * sz
        qy = cx * sy * cz - sx * cy * sz
        qz = cx * cy * sz + sx * sy * cz
    elif rotation_order == "ZYX":
        qw = cx * cy * cz + sx * sy * sz
        qx = sx * cy * cz - cx * sy * sz
        qy = cx * sy * cz + sx * cy * sz
        qz = cx * cy * sz - sx * sy * cz
    return Gf.Quaternion(qw, Gf.Vec3d(qx, qy, qz))


def create_quaternion_keyframe_animation(prim: Usd.Prim, attribute_name: str, keyframes: Dict[float, Gf.Quatf]) -> None:
    """Create a quaternion keyframe animation on a prim attribute.

    Args:
        prim (Usd.Prim): The prim to create the animation on.
        attribute_name (str): The name of the attribute to animate.
        keyframes (Dict[float, Gf.Quatf]): A dictionary of time values (float) and corresponding quaternion keyframe values.

    Raises:
        ValueError: If the prim is not valid or the attribute name is empty.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if not attribute_name:
        raise ValueError("Empty attribute name.")
    if not prim.HasAttribute(attribute_name):
        prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Quatf)
    attr = prim.GetAttribute(attribute_name)
    for time, quat_value in keyframes.items():
        attr.Set(time=time, value=quat_value)


def convert_quaternion_to_euler(quat: Gf.Quath) -> Gf.Vec3f:
    """Convert a quaternion to Euler angles (in radians).

    The rotation order is ZYX (Yaw, Pitch, Roll).
    """
    quat = quat.GetNormalized()
    qw = quat.GetReal()
    (qx, qy, qz) = quat.GetImaginary()
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return Gf.Vec3f(roll, pitch, yaw)


def chain_quaternion_transformations(quaternions: List[Gf.Quaternion]) -> Gf.Quaternion:
    """
    Chains a list of quaternion transformations together.

    Args:
        quaternions (List[Gf.Quaternion]): A list of quaternions to chain.

    Returns:
        Gf.Quaternion: The resulting quaternion from chaining the input quaternions.
    """
    if not quaternions:
        return Gf.Quaternion()
    result = Gf.Quaternion(quaternions[0].GetReal(), quaternions[0].GetImaginary())
    for i in range(1, len(quaternions)):
        result *= quaternions[i]
    return result


def get_prim_rotation_in_world_space(stage: Usd.Stage, prim_path: str) -> Gf.Quatf:
    """Get the rotation of a prim in world space.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Returns:
        Gf.Quatf: The rotation of the prim in world space.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    local_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rotation = local_transform.ExtractRotationQuat()
    return rotation


def combine_rotations(first: Gf.Quatf, second: Gf.Quatf) -> Gf.Quatf:
    """
    Combine two rotation quaternions into a single quaternion.

    The resulting quaternion represents the rotation obtained by
    applying the first rotation followed by the second rotation.

    Parameters:
        first (Gf.Quatf): The first rotation quaternion.
        second (Gf.Quatf): The second rotation quaternion.

    Returns:
        Gf.Quatf: The combined rotation quaternion.
    """
    if not isinstance(first, Gf.Quatf) or not isinstance(second, Gf.Quatf):
        raise TypeError("Input parameters must be of type Gf.Quatf")
    first = first.GetNormalized(eps=1e-06)
    second = second.GetNormalized(eps=1e-06)
    combined = second * first
    combined = combined.GetNormalized(eps=1e-06)
    return combined


def extract_real_and_imaginary_components(quat: Gf.Quatf) -> Tuple[float, Gf.Vec3f]:
    """Extract the real and imaginary components of a quaternion.

    Args:
        quat (Gf.Quatf): The input quaternion.

    Returns:
        Tuple[float, Gf.Vec3f]: A tuple containing the real component and the imaginary component as a Gf.Vec3f.
    """
    real = quat.GetReal()
    imaginary = quat.GetImaginary()
    return (real, imaginary)


def rotate_prim_around_point(prim: UsdGeom.Xformable, rotation_quat: Gf.Quatf, pivot_point: Gf.Vec3f):
    """Rotate a prim around a pivot point by a given quaternion rotation.

    Args:
        prim (UsdGeom.Xformable): The prim to rotate.
        rotation_quat (Gf.Quatf): The rotation quaternion.
        pivot_point (Gf.Vec3f): The pivot point to rotate around.
    """
    if not prim:
        raise ValueError("Invalid prim")
    pivot_to_origin = Gf.Vec3f(0, 0, 0) - pivot_point
    prim_translate_op = add_translate_op(prim, UsdGeom.XformOp.PrecisionFloat, "pivot_to_origin")
    prim_translate_op.Set(pivot_to_origin)
    rotation_op = add_orient_op(prim, UsdGeom.XformOp.PrecisionFloat, "rotation")
    rotation_op.Set(rotation_quat)
    post_rotate_translate_op = add_translate_op(prim, UsdGeom.XformOp.PrecisionFloat, "origin_to_pivot")
    post_rotate_translate_op.Set(-pivot_to_origin)


def mirror_prim_rotation(stage: Usd.Stage, prim_path: str, mirror_axis: Gf.Vec3f) -> None:
    """Mirror the rotation of a prim across the given mirror axis.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim to mirror the rotation for.
        mirror_axis (Gf.Vec3f): The axis to mirror the rotation across.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    rotation_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ]
    if not rotation_ops:
        return
    rotation_op = rotation_ops[0]
    rotation_attr = rotation_op.GetAttr()
    rotation_value = rotation_attr.Get()
    rotation_quat = Gf.Quatf()
    rotation_quat.SetReal(0)
    rotation_quat.SetImaginary(Gf.Vec3f(rotation_value))
    mirror_quat = Gf.Quatf(1, mirror_axis[0], mirror_axis[1], mirror_axis[2])
    mirrored_quat = mirror_quat * rotation_quat * mirror_quat.GetInverse()
    mirrored_rotation = mirrored_quat.GetImaginary()
    rotation_attr.Set(mirrored_rotation)


def set_prim_rotation_from_matrix(prim: Usd.Prim, rotation_matrix: Gf.Matrix4d):
    """Set the rotation of a prim using a rotation matrix.

    Args:
        prim (Usd.Prim): The prim to set the rotation for.
        rotation_matrix (Gf.Matrix4d): The rotation matrix to set.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    rotation = rotation_matrix.ExtractRotation()
    rotation_quat = Gf.Quatf(rotation.GetQuat())
    xform_op = xformable.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ)
    xform_op.Set(rotation_quat.GetImaginary())


def apply_quaternion_rotation(point: Gf.Vec3h, rotation: Gf.Quath) -> Gf.Vec3h:
    """
    Apply a quaternion rotation to a 3D point.

    Args:
        point (Gf.Vec3h): The point to rotate.
        rotation (Gf.Quath): The quaternion representing the rotation.

    Returns:
        Gf.Vec3h: The rotated point.
    """
    if not Gf.IsClose(rotation.GetLength(), 1.0, 1e-05):
        rotation = Gf.Quath(rotation).GetNormalized()
    rotated_point = rotation.Transform(point)
    return rotated_point


def apply_rotation_to_point(rotation: Gf.Quath, point: Gf.Vec3h) -> Gf.Vec3h:
    """Apply a rotation quaternion to a 3D point.

    Args:
        rotation (Gf.Quath): The rotation quaternion.
        point (Gf.Vec3h): The 3D point to rotate.

    Returns:
        Gf.Vec3h: The rotated point.
    """
    if not Gf.IsClose(rotation.GetLength(), 1.0, 1e-05):
        rotation = Gf.Quath(rotation).GetNormalized(1e-05)
    rotated_point = rotation.Transform(point)
    return rotated_point


def rotate_prims_around_common_center(stage: Usd.Stage, prim_paths: List[str], degrees: float, axis: Gf.Vec3d) -> None:
    """
    Rotate a list of prims around their common center by a given angle in degrees around a given axis.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths to rotate.
        degrees (float): The rotation angle in degrees.
        axis (Gf.Vec3d): The rotation axis.
    """
    if not prim_paths:
        raise ValueError("No prim paths provided.")
    radians = degrees * (3.141592653589793 / 180.0)
    rotation = Gf.Rotation(axis, radians)
    center = Gf.Vec3d(0, 0, 0)
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        if translate_ops:
            translate_op = translate_ops[0]
            translation = translate_op.Get(Usd.TimeCode.Default())
            if translation is not None:
                center += Gf.Vec3d(translation)
    center /= len(prim_paths)
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        old_translate_op = add_translate_op(xformable, opSuffix="old")
        old_translate_op.Set(center * -1)
        rotate_op = add_rotate_xyz_op(xformable, opSuffix="rotate")
        rotate_op.Set(rotation.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis()))
        new_translate_op = add_translate_op(xformable, opSuffix="new")
        new_translate_op.Set(center)


def slerp_quaternions(q1: Gf.Quath, q2: Gf.Quath, t: float) -> Gf.Quath:
    """Spherical linear interpolation between two quaternions.

    Args:
        q1 (Gf.Quath): First quaternion.
        q2 (Gf.Quath): Second quaternion.
        t (float): Interpolation parameter in the range [0, 1].

    Returns:
        Gf.Quath: Interpolated quaternion.
    """
    if t < 0.0 or t > 1.0:
        raise ValueError("Interpolation parameter t must be in the range [0, 1].")
    if not Gf.IsClose(q1.GetLength(), 1.0, 1e-05) or not Gf.IsClose(q2.GetLength(), 1.0, 1e-05):
        raise ValueError("Input quaternions must be normalized.")
    if t == 0.0:
        return q1
    if t == 1.0:
        return q2
    cos_angle = Gf.Dot(q1.GetImaginary(), q2.GetImaginary()) + q1.GetReal() * q2.GetReal()
    if cos_angle < 0.0:
        q2 = -q2
        cos_angle = -cos_angle
    angle = math.degrees(math.acos(min(1.0, max(-1.0, cos_angle))))
    sin_angle = math.sqrt(1.0 - cos_angle * cos_angle)
    if Gf.IsClose(sin_angle, 0.0, 1e-05):
        return (1.0 - t) * q1 + t * q2
    a = math.sin((1.0 - t) * math.radians(angle)) / sin_angle
    b = math.sin(t * math.radians(angle)) / sin_angle
    return a * q1 + b * q2


def set_prim_rotation_from_quaternion(prim: Usd.Prim, quaternion: Gf.Quath) -> None:
    """Set the rotation of a prim using a quaternion.

    Args:
        prim (Usd.Prim): The prim to set the rotation on.
        quaternion (Gf.Quath): The quaternion representing the rotation.

    Raises:
        ValueError: If the prim is not valid or transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    ops = xformable.GetOrderedXformOps()
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ or op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            xformable.RemoveXformOp(op)
    rotate_op = add_rotate_xyz_op(xformable, UsdGeom.XformOp.PrecisionFloat, "")
    rotation_euler = Gf.Rotation(quaternion).Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
    rotate_op.Set(rotation_euler, Usd.TimeCode.Default())


def compute_quaternion_from_two_vectors(v1: Gf.Vec3d, v2: Gf.Vec3d) -> Gf.Quatd:
    """Compute the quaternion that rotates v1 to v2."""
    v1 = v1.GetNormalized()
    v2 = v2.GetNormalized()
    cross = Gf.Cross(v1, v2)
    dot = Gf.Dot(v1, v2)
    if Gf.IsClose(dot, -1.0, 1e-06):
        x = Gf.Vec3d(1, 0, 0)
        if Gf.IsClose(Gf.Abs(Gf.Dot(v1, x)), 1.0, 1e-06):
            x = Gf.Vec3d(0, 1, 0)
        perp = Gf.Cross(v1, x).GetNormalized()
        return Gf.Quatd(0, perp[0], perp[1], perp[2])
    q_real = 1 + dot
    q_imag = cross
    q = Gf.Quatd(q_real, q_imag[0], q_imag[1], q_imag[2])
    q = q.GetNormalized()
    return q


def rotate_prim_around_axis(stage: Usd.Stage, prim_path: str, axis: Gf.Vec3d, angle_degrees: float) -> None:
    """
    Rotate a prim around an arbitrary axis by a given angle in degrees.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to rotate.
        axis (Gf.Vec3d): The axis to rotate around.
        angle_degrees (float): The rotation angle in degrees.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    axis.Normalize()
    angle_radians = Gf.DegreesToRadians(angle_degrees)
    rotation_matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(axis, angle_radians))
    local_transform = xformable.GetLocalTransformation()
    new_transform = local_transform * rotation_matrix
    xform_op = xformable.MakeMatrixXform()
    xform_op.Set(new_transform)


def align_prim_to_normal(prim: UsdGeom.Gprim, normal: Gf.Vec3f) -> None:
    """Align a prim to a normal vector.

    Args:
        prim (UsdGeom.Gprim): The prim to align.
        normal (Gf.Vec3f): The normal vector to align to.
    """
    if not prim or not prim.IsA(UsdGeom.Gprim):
        raise ValueError("Invalid prim")
    if Gf.IsClose(normal.GetLength(), 0.0, 1e-06):
        raise ValueError("Normal vector must have non-zero length")
    normal = normal.GetNormalized()
    xformable = UsdGeom.Xformable(prim)
    rotation_op = None
    if xformable.GetXformOpOrderAttr().Get():
        rotation_op = xformable.GetOrderedXformOps()[-1]
    if not rotation_op or rotation_op.GetOpType() != UsdGeom.XformOp.TypeRotateXYZ:
        rotation_op = add_rotate_xyz_op(xformable)
    up = Gf.Vec3d(0, 1, 0)
    rotation = Gf.Rotation(Gf.Vec3d(up), Gf.Vec3d(normal))
    euler_angles = rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
    rotation_op.Set(Gf.Vec3f(euler_angles))


def get_ranges_intersection(range1: Gf.Range1d, range2: Gf.Range1d) -> Gf.Range1d:
    """
    Get the intersection of two Gf.Range1d instances.

    Args:
        range1 (Gf.Range1d): The first range.
        range2 (Gf.Range1d): The second range.

    Returns:
        Gf.Range1d: The intersection of the two ranges.
    """
    if range1.IsEmpty() or range2.IsEmpty():
        result = Gf.Range1d()
        result.SetEmpty()
        return result
    min_value = max(range1.GetMin(), range2.GetMin())
    max_value = min(range1.GetMax(), range2.GetMax())
    if min_value <= max_value:
        result = Gf.Range1d(min_value, max_value)
    else:
        result = Gf.Range1d()
        result.SetEmpty()
    return result


def merge_adjacent_ranges(ranges: List[Gf.Range1d]) -> List[Gf.Range1d]:
    """Merge adjacent ranges in a list of ranges.

    Args:
        ranges (List[Gf.Range1d]): List of ranges to merge.

    Returns:
        List[Gf.Range1d]: List of merged ranges.
    """
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda r: r.GetMin())
    merged_ranges = [sorted_ranges[0]]
    for curr_range in sorted_ranges[1:]:
        prev_range = merged_ranges[-1]
        if curr_range.GetMin() <= prev_range.GetMax():
            merged_range = Gf.Range1d()
            merged_range.SetMin(min(prev_range.GetMin(), curr_range.GetMin()))
            merged_range.SetMax(max(prev_range.GetMax(), curr_range.GetMax()))
            merged_ranges[-1] = merged_range
        else:
            merged_ranges.append(curr_range)
    return merged_ranges


def create_prim_with_range(
    stage: Usd.Stage, prim_type: str, prim_path: str, range_min: float, range_max: float
) -> Usd.Prim:
    """
    Create a new prim with a custom range attribute.

    Args:
        stage (Usd.Stage): The stage to create the prim on.
        prim_type (str): The type of the prim to create (e.g., "Cube", "Sphere").
        prim_path (str): The path where the prim should be created.
        range_min (float): The minimum value of the range attribute.
        range_max (float): The maximum value of the range attribute.

    Returns:
        Usd.Prim: The newly created prim with the custom range attribute.

    Raises:
        ValueError: If the prim type is not supported or if range_min is greater than range_max.
    """
    if prim_type not in ["Cube", "Sphere", "Cylinder", "Cone"]:
        raise ValueError(f"Unsupported prim type: {prim_type}")
    if range_min > range_max:
        raise ValueError(f"range_min ({range_min}) must be less than or equal to range_max ({range_max})")
    prim = stage.DefinePrim(prim_path, prim_type)
    range_attr = prim.CreateAttribute("custom_range", Sdf.ValueTypeNames.Float2)
    range_attr.Set(Gf.Vec2f(range_min, range_max))
    return prim


def scale_ranges(ranges: Gf.Range1d, scale: float) -> Gf.Range1d:
    """Scale a Gf.Range1d by a given scale factor.

    Args:
        ranges (Gf.Range1d): The range to scale.
        scale (float): The scale factor to apply.

    Returns:
        Gf.Range1d: The scaled range.
    """
    if ranges.IsEmpty():
        return Gf.Range1d()
    min_val = ranges.GetMin()
    max_val = ranges.GetMax()
    scaled_min = min_val * scale
    scaled_max = max_val * scale
    scaled_range = Gf.Range1d(scaled_min, scaled_max)
    return scaled_range


def find_extreme_ranges(ranges: List[Gf.Range1d]) -> Tuple[Gf.Range1d, Gf.Range1d]:
    """
    Find the ranges with the minimum and maximum midpoints from a list of ranges.

    Args:
        ranges (List[Gf.Range1d]): A list of Gf.Range1d objects.

    Returns:
        Tuple[Gf.Range1d, Gf.Range1d]: A tuple containing the range with the minimum midpoint
            and the range with the maximum midpoint.

    Raises:
        ValueError: If the input list is empty.
    """
    if not ranges:
        raise ValueError("Input list of ranges cannot be empty.")
    min_midpoint_range = ranges[0]
    max_midpoint_range = ranges[0]
    for range_ in ranges[1:]:
        if range_.GetMidpoint() < min_midpoint_range.GetMidpoint():
            min_midpoint_range = range_
        if range_.GetMidpoint() > max_midpoint_range.GetMidpoint():
            max_midpoint_range = range_
    return (min_midpoint_range, max_midpoint_range)


def filter_ranges_by_size(ranges: List[Gf.Range1d], min_size: float) -> List[Gf.Range1d]:
    """Filter a list of Gf.Range1d objects by a minimum size.

    Args:
        ranges (List[Gf.Range1d]): The list of ranges to filter.
        min_size (float): The minimum size for a range to be included.

    Returns:
        List[Gf.Range1d]: A new list containing only the ranges with size >= min_size.
    """
    filtered_ranges = []
    for range in ranges:
        if range.IsEmpty():
            continue
        size = range.GetSize()
        if size >= min_size:
            filtered_ranges.append(range)
    return filtered_ranges


def get_ranges_midpoints(ranges: List[Gf.Range1d]) -> List[float]:
    """
    Get the midpoints of a list of Gf.Range1d objects.

    Args:
        ranges (List[Gf.Range1d]): A list of Gf.Range1d objects.

    Returns:
        List[float]: A list of midpoints, one for each input range.
    """
    midpoints = []
    for range_ in ranges:
        if range_.IsEmpty():
            midpoints.append(None)
        else:
            midpoint = range_.GetMidpoint()
            midpoints.append(midpoint)
    return midpoints


def set_prim_attribute_to_range(prim: Usd.Prim, attr_name: str, range_val: Gf.Range1d):
    """Set a prim's attribute value to a Gf.Range1d.

    Args:
        prim (Usd.Prim): The prim to set the attribute on.
        attr_name (str): The name of the attribute to set.
        range_val (Gf.Range1d): The range value to set the attribute to.

    Raises:
        ValueError: If the prim is not valid or the attribute name is empty.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if not attr_name:
        raise ValueError("Empty attribute name.")
    attr = prim.GetAttribute(attr_name)
    if not attr:
        attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float2)
    attr.Set(Gf.Vec2f(range_val.min, range_val.max))


def expand_ranges_to_include_value(ranges: Gf.Range1d, value: float) -> None:
    """Expands the given ranges to include the specified value.

    If the value is already within the ranges, the ranges remain unchanged.
    If the value is outside the ranges, the corresponding range is extended
    to include the value.

    Args:
        ranges (Gf.Range1d): The ranges to expand.
        value (float): The value to include in the ranges.
    """
    if value < ranges.GetMin():
        ranges.SetMin(value)
    elif value > ranges.GetMax():
        ranges.SetMax(value)


def get_ranges_union(ranges: Sequence[Gf.Range1d]) -> Gf.Range1d:
    """
    Compute the union of a sequence of ranges.

    Args:
        ranges (Sequence[Gf.Range1d]): The sequence of ranges to union.

    Returns:
        Gf.Range1d: The union of all the ranges.
    """
    if not ranges:
        return Gf.Range1d()
    result = Gf.Range1d()
    for range_ in ranges:
        if range_.IsEmpty():
            continue
        result.UnionWith(range_)
    return result


def adjust_ranges_for_prims(stage: Usd.Stage, prims: List[Usd.Prim], time_code=Usd.TimeCode.Default()) -> Gf.Range3d:
    """Adjust the bounding box ranges for a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prims (List[Usd.Prim]): List of prims to compute the bounding box for.
        time_code (Usd.TimeCode, optional): The time code at which to compute the bounding box.
            Defaults to Default time code.

    Returns:
        Gf.Range3d: The adjusted bounding box range.
    """
    bbox_range = Gf.Range3d()
    for prim in prims:
        if not prim.IsValid():
            continue
        model_api = UsdGeom.ModelAPI(prim)
        prim_bbox = model_api.ComputeWorldBound(time_code, ["default"])
        if prim_bbox.GetRange().IsEmpty():
            continue
        bbox_range = bbox_range.UnionWith(prim_bbox.GetRange())
    return bbox_range


def normalize_ranges(ranges: Sequence[Gf.Range1d]) -> Sequence[Gf.Range1d]:
    """Normalize a sequence of ranges to [0, 1] based on their combined extents.

    Args:
        ranges (Sequence[Gf.Range1d]): The input ranges to normalize.

    Returns:
        Sequence[Gf.Range1d]: The normalized ranges.
    """
    if not ranges:
        return []
    min_val = min((r.GetMin() for r in ranges))
    max_val = max((r.GetMax() for r in ranges))
    if min_val == max_val:
        return [Gf.Range1d(0, 1) for _ in ranges]
    scale = 1.0 / (max_val - min_val)
    result = []
    for r in ranges:
        if r.IsEmpty():
            result.append(Gf.Range1d())
        else:
            norm_min = (r.GetMin() - min_val) * scale
            norm_max = (r.GetMax() - min_val) * scale
            result.append(Gf.Range1d(norm_min, norm_max))
    return result


def split_range_at_value(range_to_split: Gf.Range1d, split_value: float) -> Tuple[Gf.Range1d, Gf.Range1d]:
    """Split a Gf.Range1d into two ranges at the specified value.

    Args:
        range_to_split (Gf.Range1d): The range to split.
        split_value (float): The value at which to split the range.

    Returns:
        Tuple[Gf.Range1d, Gf.Range1d]: A tuple containing the two split ranges.
            If the split value is outside the input range, one of the returned
            ranges will be empty.

    Raises:
        ValueError: If the input range is empty.
    """
    if range_to_split.IsEmpty():
        raise ValueError("Cannot split an empty range.")
    range1 = Gf.Range1d(range_to_split.GetMin(), range_to_split.GetMax())
    range2 = Gf.Range1d(range_to_split.GetMin(), range_to_split.GetMax())
    range1.SetMax(split_value)
    range2.SetMin(split_value)
    if split_value <= range_to_split.GetMin():
        range1.SetEmpty()
    elif split_value >= range_to_split.GetMax():
        range2.SetEmpty()
    return (range1, range2)


def get_range_distances_from_point(range1d: Gf.Range1d, point: float) -> Tuple[float, float]:
    """
    Get the distances from a point to the min and max of a Gf.Range1d.

    Args:
        range1d (Gf.Range1d): The range to calculate distances from.
        point (float): The point to calculate distances to.

    Returns:
        Tuple[float, float]: A tuple containing the distance from the point to the min and max of the range.
    """
    min_dist = abs(point - range1d.GetMin())
    max_dist = abs(point - range1d.GetMax())
    return (min_dist, max_dist)


def filter_prims_by_range(stage: Usd.Stage, prim_paths: List[str], range_: Gf.Range1d) -> List[str]:
    """
    Filter a list of prim paths based on whether their bounding box intersects the given range.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The list of prim paths to filter.
        range_ (Gf.Range1d): The range to check for intersection.

    Returns:
        List[str]: The list of prim paths whose bounding boxes intersect the given range.
    """
    filtered_paths = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bounds = bbox_cache.ComputeWorldBound(prim)
        if bounds.GetRange().IsEmpty():
            continue
        y_range = Gf.Range1d(bounds.GetRange().GetMin()[1], bounds.GetRange().GetMax()[1])
        if not Gf.Range1d.GetIntersection(y_range, range_).IsEmpty():
            filtered_paths.append(prim_path)
    return filtered_paths


def get_prim_bbox_size(stage: Usd.Stage, prim_path: str) -> Gf.Range3f:
    """Get the bounding box size for a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Returns:
        Gf.Range3f: The bounding box size of the prim.

    Raises:
        ValueError: If the prim is not valid or does not have a defined bounding box.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    if not bbox.GetRange().IsEmpty():
        return bbox.ComputeAlignedRange().GetSize()
    else:
        raise ValueError(f"Prim at path {prim_path} does not have a defined bounding box.")


def match_prim_bbox_size(
    stage: Usd.Stage, prim_path: str, target_size: float, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> None:
    """
    Match the bounding box size of a prim to a target size by scaling it uniformly.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to modify.
        target_size (float): The target size for the prim's bounding box.
        time_code (Usd.TimeCode, optional): The time code at which to evaluate the bounding box. Defaults to Default().
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path '{prim_path}'")
    bbox_cache = UsdGeom.BBoxCache(time_code, includedPurposes=[UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    if not bbox.GetRange().IsEmpty():
        bbox_size = max(bbox.GetRange().GetSize())
        scale_factor = target_size / bbox_size
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            add_scale_op(xformable).Set(Gf.Vec3f(scale_factor))
        else:
            raise ValueError(f"Prim at path '{prim_path}' is not xformable")
    else:
        raise ValueError(f"Prim at path '{prim_path}' has an empty bounding box")


def get_intersection_of_prim_bboxes(stage: Usd.Stage, prim_paths: List[str]) -> Gf.Range3f:
    """
    Get the intersection of the bounding boxes of a list of prims.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (List[str]): A list of prim paths.

    Returns:
        Gf.Range3f: The intersection of the bounding boxes.
    """
    if not prim_paths:
        raise ValueError("No prim paths provided.")
    prim = stage.GetPrimAtPath(prim_paths[0])
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_paths[0]}")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    intersection_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    for prim_path in prim_paths[1:]:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")
        prim_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        intersection_range = Gf.Range3f(
            Gf.Vec3f(
                max(intersection_range.GetMin()[0], prim_range.GetMin()[0]),
                max(intersection_range.GetMin()[1], prim_range.GetMin()[1]),
                max(intersection_range.GetMin()[2], prim_range.GetMin()[2]),
            ),
            Gf.Vec3f(
                min(intersection_range.GetMax()[0], prim_range.GetMax()[0]),
                min(intersection_range.GetMax()[1], prim_range.GetMax()[1]),
                min(intersection_range.GetMax()[2], prim_range.GetMax()[2]),
            ),
        )
    return intersection_range


def get_distance_squared_to_prim_bbox(stage: Usd.Stage, prim_path: str, point: Gf.Vec3d) -> float:
    """
    Calculate the squared distance from a point to the bounding box of a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        point (Gf.Vec3d): The point to calculate the distance from.

    Returns:
        float: The squared distance from the point to the prim's bounding box.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path: {prim_path}")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bound = bbox_cache.ComputeWorldBound(prim)
    range = bound.ComputeAlignedRange()
    dist_squared = range.GetDistanceSquared(point)
    return dist_squared


def get_prims_by_bbox_size(stage: Usd.Stage, min_size: float, max_size: float) -> List[Usd.Prim]:
    """
    Get a list of prims whose bounding box size falls within the specified range.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        min_size (float): The minimum bounding box size (inclusive).
        max_size (float): The maximum bounding box size (inclusive).

    Returns:
        List[Usd.Prim]: A list of prims whose bounding box size falls within the specified range.
    """
    if not stage:
        raise ValueError("Invalid stage")
    if min_size < 0 or max_size < 0 or min_size > max_size:
        raise ValueError("Invalid size range")
    prims = []
    for prim in stage.TraverseAll():
        bbox = UsdGeom.Imageable(prim).ComputeLocalBound(Usd.TimeCode.Default(), purpose1="default")
        if bbox.GetRange().IsEmpty():
            continue
        bbox_size = max(bbox.ComputeAlignedRange().GetSize())
        if min_size <= bbox_size <= max_size:
            prims.append(prim)
    return prims


def set_prim_bbox_to_empty(prim: Usd.Prim) -> None:
    """Set the bounding box of a prim to an empty range."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    bbox_attr = UsdGeom.Boundable(prim).GetExtentAttr()
    if not bbox_attr.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} does not have a bounding box attribute.")
    empty_range = Gf.Range3f()
    empty_range.SetEmpty()
    empty_extent = (Gf.Vec3f(empty_range.GetMin()), Gf.Vec3f(empty_range.GetMax()))
    bbox_attr.Set(empty_extent)


def get_prim_bbox_midpoint(prim: Usd.Prim) -> Gf.Vec3f:
    """Get the midpoint of a prim's bounding box in world space.

    Args:
        prim (Usd.Prim): The input prim.

    Returns:
        Gf.Vec3f: The midpoint of the prim's bounding box in world space.

    Raises:
        ValueError: If the input prim is not valid or doesn't have a defined bounding box.
    """
    if not prim.IsValid():
        raise ValueError("Input prim is not valid")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bound = bbox_cache.ComputeWorldBound(prim)
    range = bound.ComputeAlignedRange()
    if not range.IsEmpty():
        midpoint = range.GetMidpoint()
        return midpoint
    else:
        raise ValueError("Prim does not have a defined bounding box")


def align_prim_to_bbox(stage: Usd.Stage, prim_path: str, bbox: Gf.Range3f) -> None:
    """Align a prim to the center of a bounding box."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim not found at path: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable")
    bbox_center = bbox.GetMidpoint()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    prim_bbox = bbox_cache.ComputeLocalBound(prim)
    prim_center = prim_bbox.GetRange().GetMidpoint()
    prim_center_vec3f = Gf.Vec3f(prim_center[0], prim_center[1], prim_center[2])
    translation = bbox_center - prim_center_vec3f
    translate_op = xformable.GetOrderedXformOps()[0] if xformable.GetOrderedXformOps() else add_translate_op(xformable)
    translate_op.Set(translation)


def set_prim_bbox_max(prim: Usd.Prim, max_value: Gf.Vec3f) -> None:
    """Set the maximum extent of the bounding box for a prim."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    bbox_attr = UsdGeom.Boundable(prim).GetExtentAttr()
    if not bbox_attr.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} does not have a bounding box attribute.")
    old_extent = bbox_attr.Get()
    if old_extent is None or len(old_extent) == 0:
        new_extent = Gf.Range3f(Gf.Vec3f(), max_value)
    else:
        new_extent = Gf.Range3f(old_extent[0], max_value)
    bbox_attr.Set(Vt.Vec3fArray([new_extent.GetMin(), new_extent.GetMax()]))


def set_prim_bbox_min(prim: Usd.Prim, min_point: Gf.Vec3f) -> None:
    """Set the minimum point of the bounding box for a prim."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    bbox_attr = UsdGeom.Boundable(prim).GetExtentAttr()
    if not bbox_attr.HasAuthoredValue():
        raise ValueError(f"Bounding box attribute not authored for prim {prim.GetPath()}.")
    bbox = bbox_attr.Get()
    bbox = Gf.Range3f(Gf.Vec3f(min_point), Gf.Vec3f(bbox[1]))
    bbox_attr.Set([bbox.min, bbox.max])


def extend_prim_bbox_to_include_value(prim: Usd.Prim, value: float):
    """Extend the bounding box of a prim to include a given value."""
    bbox_attr = UsdGeom.Boundable(prim).GetExtentAttr()
    if not bbox_attr or not bbox_attr.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} does not have a valid bounding box attribute.")
    bbox = bbox_attr.Get()
    if not bbox:
        bbox = Gf.Range3f()
    else:
        bbox = Gf.Range3f(Gf.Vec3f(bbox[0]), Gf.Vec3f(bbox[1]))
    bbox.UnionWith(Gf.Vec3f(value, value, value))
    bbox_attr.Set(Vt.Vec3fArray([bbox.GetMin(), bbox.GetMax()]))


def is_prim_within_bbox(prim: Usd.Prim, bbox: Gf.Range3d) -> bool:
    """Check if a prim is within a given bounding box.

    Args:
        prim (Usd.Prim): The prim to check.
        bbox (Gf.Range3d): The bounding box to check against.

    Returns:
        bool: True if the prim is within the bounding box, False otherwise.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    prim_bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
    if prim_bbox.GetRange().IsEmpty():
        return False
    else:
        return bbox.Contains(prim_bbox.ComputeAlignedRange())


def split_bbox_into_quadrants(bbox: Gf.Range2d) -> Tuple[Gf.Range2d, Gf.Range2d, Gf.Range2d, Gf.Range2d]:
    """Split a 2D bounding box into its four quadrants.

    Args:
        bbox (Gf.Range2d): The bounding box to split.

    Returns:
        Tuple[Gf.Range2d, Gf.Range2d, Gf.Range2d, Gf.Range2d]: The four quadrants (SW, SE, NW, NE).
    """
    if bbox.IsEmpty():
        raise ValueError("Cannot split an empty bounding box.")
    min_point = bbox.GetMin()
    max_point = bbox.GetMax()
    midpoint = bbox.GetMidpoint()
    sw_quadrant = Gf.Range2d(min_point, midpoint)
    se_quadrant = Gf.Range2d(Gf.Vec2d(midpoint[0], min_point[1]), Gf.Vec2d(max_point[0], midpoint[1]))
    nw_quadrant = Gf.Range2d(Gf.Vec2d(min_point[0], midpoint[1]), Gf.Vec2d(midpoint[0], max_point[1]))
    ne_quadrant = Gf.Range2d(midpoint, max_point)
    return (sw_quadrant, se_quadrant, nw_quadrant, ne_quadrant)


def get_prims_in_bbox_quadrants(stage: Usd.Stage, bounding_box: Gf.Range2d) -> Dict[int, List[Usd.Prim]]:
    """
    Get a dictionary mapping quadrant indices to lists of prims whose bounding boxes lie within that quadrant.

    Args:
        stage (Usd.Stage): The USD stage to query prims from.
        bounding_box (Gf.Range2d): The overall bounding box to divide into quadrants.

    Returns:
        Dict[int, List[Usd.Prim]]: A dictionary mapping quadrant indices (0-3) to lists of prims.
    """
    quadrant_prims: Dict[int, List[Usd.Prim]] = {i: [] for i in range(4)}
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        prim_bbox = bbox_cache.ComputeWorldBound(prim)
        if prim_bbox.GetRange().IsEmpty():
            continue
        prim_min = Gf.Vec2d(prim_bbox.GetRange().GetMin()[0], prim_bbox.GetRange().GetMin()[2])
        prim_max = Gf.Vec2d(prim_bbox.GetRange().GetMax()[0], prim_bbox.GetRange().GetMax()[2])
        prim_range = Gf.Range2d(prim_min, prim_max)
        for i in range(4):
            quadrant_bbox = Gf.Range2d(bounding_box.GetMin(), bounding_box.GetMax())
            midpoint = quadrant_bbox.GetMidpoint()
            if i == 0:
                quadrant_bbox.SetMax(midpoint)
            elif i == 1:
                quadrant_bbox.SetMin(Gf.Vec2d(midpoint[0], quadrant_bbox.GetMin()[1]))
                quadrant_bbox.SetMax(Gf.Vec2d(quadrant_bbox.GetMax()[0], midpoint[1]))
            elif i == 2:
                quadrant_bbox.SetMin(Gf.Vec2d(quadrant_bbox.GetMin()[0], midpoint[1]))
                quadrant_bbox.SetMax(Gf.Vec2d(midpoint[0], quadrant_bbox.GetMax()[1]))
            elif i == 3:
                quadrant_bbox.SetMin(midpoint)
            if quadrant_bbox.Contains(prim_range):
                quadrant_prims[i].append(prim)
    return quadrant_prims


def set_bbox_for_prim(stage: Usd.Stage, prim_path: str, bbox_min: Gf.Vec3f, bbox_max: Gf.Vec3f):
    """Set the bounding box for a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        bbox_min (Gf.Vec3f): The minimum point of the bounding box.
        bbox_max (Gf.Vec3f): The maximum point of the bounding box.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path: {prim_path}")
    bbox_attr = UsdGeom.Boundable(prim).GetExtentAttr()
    if not bbox_attr:
        raise ValueError(f"Prim at path {prim_path} does not have a bounding box attribute")
    bbox_extent = Vt.Vec3fArray(2)
    bbox_extent[0] = bbox_min
    bbox_extent[1] = bbox_max
    bbox_attr.Set(bbox_extent)


def get_local_bbox_union(stage: Usd.Stage, prim_paths: List[str]) -> Gf.Range3d:
    """
    Compute the union of the local bounding boxes of the prims at the given paths.

    Parameters:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths.

    Returns:
        Gf.Range3d: The union of the local bounding boxes.
    """
    bbox_union = Gf.Range3d()
    bbox_union.SetEmpty()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        prim_bbox = UsdGeom.Imageable(prim).ComputeLocalBound(0.0, "default")
        if prim_bbox.GetRange().IsEmpty():
            continue
        bbox_union = bbox_union.UnionWith(prim_bbox.GetRange())
    return bbox_union


def scale_prims_to_bbox(stage: Usd.Stage, prim_paths: List[str], bbox: Gf.Range3d):
    """Scale prims to fit within a bounding box.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The paths to the prims to scale.
        bbox (Gf.Range3d): The target bounding box.

    Raises:
        ValueError: If any prim path is invalid or not transformable.
    """
    union_bbox = Gf.Range3d()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        prim_bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
        union_bbox = Gf.Range3d.GetUnion(union_bbox, Gf.Range3d(prim_bbox.GetBox()))
    union_size = union_bbox.GetSize()
    bbox_size = bbox.GetSize()
    scale_factor = min(bbox_size[0] / union_size[0], bbox_size[1] / union_size[1], bbox_size[2] / union_size[2])
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        scale_op = add_scale_op(xformable)
        scale_op.Set(Gf.Vec3f(scale_factor))


def expand_bbox_to_include_prims(stage: Usd.Stage, bbox: Gf.Range3d, prim_paths: List[str]) -> Gf.Range3d:
    """Expand a bounding box to include the bounding boxes of a list of prims.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        bbox (Gf.Range3d): The initial bounding box to expand.
        prim_paths (List[str]): A list of prim paths to include in the bounding box.

    Returns:
        Gf.Range3d: The expanded bounding box.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            continue
        prim_bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
        if prim_bbox.GetRange().IsEmpty():
            continue
        if bbox.IsEmpty():
            bbox = prim_bbox.GetRange()
        else:
            bbox = Gf.Range3d(
                Gf.Vec3d(
                    min(bbox.GetMin()[0], prim_bbox.GetRange().GetMin()[0]),
                    min(bbox.GetMin()[1], prim_bbox.GetRange().GetMin()[1]),
                    min(bbox.GetMin()[2], prim_bbox.GetRange().GetMin()[2]),
                ),
                Gf.Vec3d(
                    max(bbox.GetMax()[0], prim_bbox.GetRange().GetMax()[0]),
                    max(bbox.GetMax()[1], prim_bbox.GetRange().GetMax()[1]),
                    max(bbox.GetMax()[2], prim_bbox.GetRange().GetMax()[2]),
                ),
            )
    return bbox


def get_world_bbox_union(stage: Usd.Stage, prim_paths: List[str]) -> Gf.Range3d:
    """
    Compute the world space bounding box union of a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths.

    Returns:
        Gf.Range3d: The world space bounding box union of the prims.

    Raises:
        ValueError: If all prims are invalid.
    """
    bbox_union = Gf.Range3d()
    bbox_union.SetEmpty()
    valid_prims = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            valid_prims.append(prim)
    if not valid_prims:
        raise ValueError("All prims are invalid")
    for prim in valid_prims:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        prim_bbox = bbox_cache.ComputeWorldBound(prim)
        if not prim_bbox.GetRange().IsEmpty():
            bbox_union = Gf.Range3d.GetUnion(bbox_union, Gf.Range3d(prim_bbox.GetRange()))
    return bbox_union


def move_prims_to_bbox_center(stage: Usd.Stage, prim_paths: List[str]) -> None:
    """Move the specified prims to the center of their bounding box."""
    if not prim_paths:
        return
    bbox = Gf.Range3d()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        prim_bbox = UsdGeom.Imageable(prim).ComputeWorldBound(
            Usd.TimeCode.Default(),
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        )
        bbox.UnionWith(prim_bbox.GetRange())
    if bbox.IsEmpty():
        return
    center = bbox.GetMidpoint()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        xformable = UsdGeom.Xformable(prim)
        old_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        old_translation = old_xform.ExtractTranslation()
        new_translation = old_translation - center
        translate_op = xformable.GetOrderedXformOps()[0]
        translate_op.Set(new_translation)


def get_prims_outside_bbox(stage: Usd.Stage, bbox: Gf.Range3f) -> List[Usd.Prim]:
    """
    Returns a list of prims on the stage that are outside the given bounding box.

    Args:
        stage (Usd.Stage): The USD stage to query.
        bbox (Gf.Range3f): The bounding box to check against.

    Returns:
        List[Usd.Prim]: A list of prims that are outside the bounding box.
    """
    prims_outside_bbox = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            extent = mesh.ComputeExtent(Usd.TimeCode.Default())
            if not extent.Intersects(bbox):
                prims_outside_bbox.append(prim)
    return prims_outside_bbox


def translate_prim_to_fit_bbox(stage: Usd.Stage, prim_path: str, bbox_range: Gf.Range3f):
    """Translate a prim to fit within a bounding box range.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to translate.
        bbox_range (Gf.Range3f): The bounding box range to fit the prim within.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path: {prim_path}")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    prim_bbox = bbox_cache.ComputeWorldBound(prim)
    (prim_min, prim_max) = (prim_bbox.GetRange().GetMin(), prim_bbox.GetRange().GetMax())
    (range_min, range_max) = (bbox_range.GetMin(), bbox_range.GetMax())
    translate_offset = Gf.Vec3f(range_min[0] - prim_min[0], range_min[1] - prim_min[1], range_min[2] - prim_min[2])
    xformable = UsdGeom.Xformable(prim)
    add_translate_op(xformable).Set(translate_offset)


def split_prim_bbox(prim: Usd.Prim, quadrant: int) -> Gf.Range3f:
    """Split the bounding box of a prim into quadrants and return the specified quadrant.

    Args:
        prim (Usd.Prim): The prim to get the bounding box from.
        quadrant (int): The quadrant index (0-7) to return.

    Returns:
        Gf.Range3f: The bounding box quadrant.

    Raises:
        ValueError: If the prim is not valid or does not have a bounding box.
        IndexError: If the quadrant index is out of range (0-7).
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bounds = bbox_cache.ComputeWorldBound(prim)
    if bounds.GetRange().IsEmpty():
        raise ValueError("Prim does not have a bounding box")
    (min_pt, max_pt) = (bounds.GetRange().GetMin(), bounds.GetRange().GetMax())
    mid_pt = bounds.ComputeCentroid()
    quadrants = [
        Gf.Range3f(Gf.Vec3f(min_pt[0], min_pt[1], min_pt[2]), Gf.Vec3f(mid_pt[0], mid_pt[1], mid_pt[2])),
        Gf.Range3f(Gf.Vec3f(mid_pt[0], min_pt[1], min_pt[2]), Gf.Vec3f(max_pt[0], mid_pt[1], mid_pt[2])),
        Gf.Range3f(Gf.Vec3f(min_pt[0], mid_pt[1], min_pt[2]), Gf.Vec3f(mid_pt[0], max_pt[1], mid_pt[2])),
        Gf.Range3f(Gf.Vec3f(mid_pt[0], mid_pt[1], min_pt[2]), Gf.Vec3f(max_pt[0], max_pt[1], mid_pt[2])),
        Gf.Range3f(Gf.Vec3f(min_pt[0], min_pt[1], mid_pt[2]), Gf.Vec3f(mid_pt[0], mid_pt[1], max_pt[2])),
        Gf.Range3f(Gf.Vec3f(mid_pt[0], min_pt[1], mid_pt[2]), Gf.Vec3f(max_pt[0], mid_pt[1], max_pt[2])),
        Gf.Range3f(Gf.Vec3f(min_pt[0], mid_pt[1], mid_pt[2]), Gf.Vec3f(mid_pt[0], max_pt[1], max_pt[2])),
        Gf.Range3f(Gf.Vec3f(mid_pt[0], mid_pt[1], mid_pt[2]), Gf.Vec3f(max_pt[0], max_pt[1], max_pt[2])),
    ]
    if 0 <= quadrant < 8:
        return quadrants[quadrant]
    else:
        raise IndexError("Quadrant index out of range (0-7)")


def find_largest_bbox_prim(stage: Usd.Stage) -> Usd.Prim:
    """Find the prim with the largest bounding box on the stage.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.

    Returns:
        Usd.Prim: The prim with the largest bounding box, or an invalid prim if no prims have a bounding box.
    """
    largest_bbox_prim: Usd.Prim = Usd.Prim()
    largest_bbox_range: Gf.Range3d = Gf.Range3d()
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
            bbox = bbox_cache.ComputeWorldBound(prim)
            if bbox.GetRange().IsEmpty():
                continue
            if bbox.GetRange().GetSize() > largest_bbox_range.GetSize():
                largest_bbox_prim = prim
                largest_bbox_range = bbox.GetRange()
    return largest_bbox_prim


def is_bbox_contained(bbox1: Gf.Range2f, bbox2: Gf.Range2f) -> bool:
    """
    Check if bbox1 is fully contained within bbox2.

    Args:
        bbox1 (Gf.Range2f): The first bounding box.
        bbox2 (Gf.Range2f): The second bounding box.

    Returns:
        bool: True if bbox1 is fully contained within bbox2, False otherwise.
    """
    if bbox1.IsEmpty() or bbox2.IsEmpty():
        return False
    if not (bbox1.GetMin()[0] >= bbox2.GetMin()[0] and bbox1.GetMin()[1] >= bbox2.GetMin()[1]):
        return False
    if not (bbox1.GetMax()[0] <= bbox2.GetMax()[0] and bbox1.GetMax()[1] <= bbox2.GetMax()[1]):
        return False
    return True


def set_prim_bbox(prim: Usd.Prim, bbox: Gf.Range3f) -> None:
    """Set the bounding box for a prim."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    extent_attr = prim.GetAttribute("extent")
    if not extent_attr.IsValid():
        extent_attr = prim.CreateAttribute("extent", Sdf.ValueTypeNames.Float3Array)
    extent_attr.Set([bbox.GetMin(), bbox.GetMax()])


def calculate_prim_bbox_volume(prim: Usd.Prim, purpose=UsdGeom.Tokens.default_) -> float:
    """Calculate the volume of a prim's bounding box.

    Args:
        prim (Usd.Prim): The prim to calculate the bounding box volume for.
        purpose (UsdGeom.Tokens): The purpose to use when computing the bounding box.

    Returns:
        float: The volume of the prim's bounding box. Returns 0 if no bounding box.
    """
    bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose)
    box_range = bbox.ComputeAlignedRange()
    if box_range.IsEmpty():
        return 0.0
    dimensions = box_range.GetSize()
    volume = dimensions[0] * dimensions[1] * dimensions[2]
    return volume


def grow_prim_bbox(prim: Usd.Prim, pad_amount: float) -> Gf.Range3d:
    """Grow the bounding box of a prim by a specified amount.

    Args:
        prim (Usd.Prim): The prim to get the bounding box for.
        pad_amount (float): The amount to pad the bounding box by.

    Returns:
        Gf.Range3d: The grown bounding box.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    orig_bbox = bbox_cache.ComputeWorldBound(prim)
    if orig_bbox.GetRange().IsEmpty():
        raise ValueError(f"Prim {prim.GetPath()} has an empty bounding box.")
    min_pt = orig_bbox.GetRange().GetMin()
    max_pt = orig_bbox.GetRange().GetMax()
    padded_min_pt = min_pt - Gf.Vec3d(pad_amount)
    padded_max_pt = max_pt + Gf.Vec3d(pad_amount)
    padded_bbox = Gf.Range3d(padded_min_pt, padded_max_pt)
    return padded_bbox


def scale_prim_to_fit_bbox(prim: Usd.Prim, bbox: Gf.Range3f) -> None:
    """Scale a prim to fit within a given bounding box."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    prim_bbox = UsdGeom.Imageable(prim).ComputeLocalBound(Usd.TimeCode.Default(), "default")
    if prim_bbox.GetRange().IsEmpty():
        raise ValueError(f"Prim {prim.GetPath()} has an empty bounding box.")
    prim_range = prim_bbox.GetRange()
    prim_size = prim_range.GetSize()
    bbox_size = bbox.GetSize()
    scale_factors = Gf.Vec3f(bbox_size[0] / prim_size[0], bbox_size[1] / prim_size[1], bbox_size[2] / prim_size[2])
    min_scale = min(scale_factors)
    xform = UsdGeom.Xformable(prim)
    scale_op = add_scale_op(xform)
    scale_op.Set(Gf.Vec3f(min_scale))


def merge_prim_bboxes(stage: Usd.Stage, prim_paths: List[str]) -> Gf.Range3f:
    """
    Merge the bounding boxes of multiple prims into a single bounding box.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths to merge bounding boxes for.

    Returns:
        Gf.Range3f: The merged bounding box.

    Raises:
        ValueError: If any of the prim paths are invalid or have no bounding box.
    """
    merged_bbox = Gf.Range3f()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        prim_bbox = bbox_cache.ComputeWorldBound(prim)
        if prim_bbox.GetRange().IsEmpty():
            raise ValueError(f"Prim {prim_path} has an empty bounding box")
        prim_range = Gf.Range3f(Gf.Vec3f(prim_bbox.GetRange().GetMin()), Gf.Vec3f(prim_bbox.GetRange().GetMax()))
        merged_bbox = Gf.Range3f.GetUnion(merged_bbox, prim_range)
    return merged_bbox


def combine_bounding_boxes(box1: Gf.Range3d, box2: Gf.Range3d) -> Gf.Range3d:
    """Combine two bounding boxes into a single bounding box."""
    if box1.IsEmpty() and box2.IsEmpty():
        return Gf.Range3d()
    elif box1.IsEmpty():
        return box2
    elif box2.IsEmpty():
        return box1
    box1_min = box1.GetMin()
    box1_max = box1.GetMax()
    box2_min = box2.GetMin()
    box2_max = box2.GetMax()
    combined_min = Gf.Vec3d(min(box1_min[0], box2_min[0]), min(box1_min[1], box2_min[1]), min(box1_min[2], box2_min[2]))
    combined_max = Gf.Vec3d(max(box1_max[0], box2_max[0]), max(box1_max[1], box2_max[1]), max(box1_max[2], box2_max[2]))
    combined_box = Gf.Range3d(combined_min, combined_max)
    return combined_box


def transform_bounding_box(bounding_box: Gf.Range3d, matrix: Gf.Matrix4d) -> Gf.Range3d:
    """
    Transform a bounding box by a matrix.

    Parameters:
        bounding_box (Gf.Range3d): The bounding box to transform.
        matrix (Gf.Matrix4d): The transformation matrix.

    Returns:
        Gf.Range3d: The transformed bounding box.
    """
    if bounding_box.IsEmpty():
        return bounding_box
    corners = [bounding_box.GetCorner(i) for i in range(8)]
    transformed_corners = [matrix.Transform(corner) for corner in corners]
    transformed_bbox = Gf.Range3d()
    for corner in transformed_corners:
        transformed_bbox.UnionWith(corner)
    return transformed_bbox


def expand_bounding_box_to_include(bbox: Gf.Range3d, point: Gf.Vec3d) -> Gf.Range3d:
    """Expand the bounding box to include the given point.

    If the bounding box is empty, set it to a zero-sized box at the point.

    Parameters:
        bbox (Gf.Range3d): The bounding box to expand.
        point (Gf.Vec3d): The point to include in the bounding box.

    Returns:
        Gf.Range3d: The expanded bounding box.
    """
    if bbox.IsEmpty():
        bbox.SetMin(point)
        bbox.SetMax(point)
    else:
        min_pt = bbox.GetMin()
        max_pt = bbox.GetMax()
        min_pt[0] = min(min_pt[0], point[0])
        min_pt[1] = min(min_pt[1], point[1])
        min_pt[2] = min(min_pt[2], point[2])
        max_pt[0] = max(max_pt[0], point[0])
        max_pt[1] = max(max_pt[1], point[1])
        max_pt[2] = max(max_pt[2], point[2])
        bbox.SetMin(min_pt)
        bbox.SetMax(max_pt)
    return bbox


def get_bounding_box_intersection(bbox1: Gf.Range3d, bbox2: Gf.Range3d) -> Gf.Range3d:
    """
    Compute the intersection of two bounding boxes.

    Args:
        bbox1 (Gf.Range3d): The first bounding box.
        bbox2 (Gf.Range3d): The second bounding box.

    Returns:
        Gf.Range3d: The intersection of the two bounding boxes.
    """
    min1 = bbox1.GetMin()
    max1 = bbox1.GetMax()
    min2 = bbox2.GetMin()
    max2 = bbox2.GetMax()
    intersection_min = Gf.Vec3d(max(min1[0], min2[0]), max(min1[1], min2[1]), max(min1[2], min2[2]))
    intersection_max = Gf.Vec3d(min(max1[0], max2[0]), min(max1[1], max2[1]), min(max1[2], max2[2]))
    if (
        intersection_min[0] <= intersection_max[0]
        and intersection_min[1] <= intersection_max[1]
        and (intersection_min[2] <= intersection_max[2])
    ):
        return Gf.Range3d(intersection_min, intersection_max)
    else:
        return Gf.Range3d()


def get_bounding_box_union(bbox1: Gf.Range3d, bbox2: Gf.Range3d) -> Gf.Range3d:
    """
    Compute the union of two bounding boxes.

    Args:
        bbox1 (Gf.Range3d): The first bounding box.
        bbox2 (Gf.Range3d): The second bounding box.

    Returns:
        Gf.Range3d: The union of the two bounding boxes.
    """
    if bbox1.IsEmpty() and bbox2.IsEmpty():
        return Gf.Range3d()
    elif bbox1.IsEmpty():
        return bbox2
    elif bbox2.IsEmpty():
        return bbox1
    union_min = Gf.Vec3d(
        min(bbox1.GetMin()[0], bbox2.GetMin()[0]),
        min(bbox1.GetMin()[1], bbox2.GetMin()[1]),
        min(bbox1.GetMin()[2], bbox2.GetMin()[2]),
    )
    union_max = Gf.Vec3d(
        max(bbox1.GetMax()[0], bbox2.GetMax()[0]),
        max(bbox1.GetMax()[1], bbox2.GetMax()[1]),
        max(bbox1.GetMax()[2], bbox2.GetMax()[2]),
    )
    union_bbox = Gf.Range3d(union_min, union_max)
    return union_bbox


def get_bounding_box_center(bounding_box: Gf.Range3d) -> Gf.Vec3d:
    """Get the center point of a bounding box.

    Args:
        bounding_box (Gf.Range3d): The bounding box to get the center of.

    Returns:
        Gf.Vec3d: The center point of the bounding box.
    """
    if bounding_box.IsEmpty():
        return Gf.Vec3d(0, 0, 0)
    min_point = bounding_box.GetMin()
    max_point = bounding_box.GetMax()
    center = (min_point + max_point) / 2
    return center


def find_prims_within_distance(stage: Usd.Stage, ref_point: Gf.Vec3d, max_distance: float) -> List[Usd.Prim]:
    """
    Find all prims on the stage within a certain distance of a reference point.

    Args:
        stage (Usd.Stage): The USD stage to search.
        ref_point (Gf.Vec3d): The reference point to measure distance from.
        max_distance (float): The maximum distance threshold.

    Returns:
        List[Usd.Prim]: A list of prims within the specified distance.
    """
    prims_within_distance: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Boundable):
            continue
        bbox = UsdGeom.Boundable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
        range3d = bbox.ComputeAlignedRange()
        if range3d.GetDistanceSquared(ref_point) <= max_distance**2:
            prims_within_distance.append(prim)
    return prims_within_distance


def merge_bounding_boxes(box1: Gf.Range3d, box2: Gf.Range3d) -> Gf.Range3d:
    """Merge two bounding boxes into a single bounding box."""
    if box1.IsEmpty() and box2.IsEmpty():
        return Gf.Range3d()
    elif box1.IsEmpty():
        return box2
    elif box2.IsEmpty():
        return box1
    box1_min = box1.GetMin()
    box1_max = box1.GetMax()
    box2_min = box2.GetMin()
    box2_max = box2.GetMax()
    merged_min = Gf.Vec3d(min(box1_min[0], box2_min[0]), min(box1_min[1], box2_min[1]), min(box1_min[2], box2_min[2]))
    merged_max = Gf.Vec3d(max(box1_max[0], box2_max[0]), max(box1_max[1], box2_max[1]), max(box1_max[2], box2_max[2]))
    merged_box = Gf.Range3d(merged_min, merged_max)
    return merged_box


def offset_bounding_box(bbox: Gf.Range3d, offset: float) -> Gf.Range3d:
    """Offset the bounding box by the given offset value.

    Args:
        bbox (Gf.Range3d): The input bounding box.
        offset (float): The offset value to apply to the bounding box.

    Returns:
        Gf.Range3d: The offset bounding box.
    """
    if bbox.IsEmpty():
        return bbox
    min_point = bbox.GetMin()
    max_point = bbox.GetMax()
    offset_min = Gf.Vec3d(min_point[0] - offset, min_point[1] - offset, min_point[2] - offset)
    offset_max = Gf.Vec3d(max_point[0] + offset, max_point[1] + offset, max_point[2] + offset)
    offset_bbox = Gf.Range3d(offset_min, offset_max)
    return offset_bbox


def get_bounding_box_octants(bounding_box: Gf.Range3d) -> List[Gf.Range3d]:
    """Get the octants of a bounding box.

    Args:
        bounding_box (Gf.Range3d): The bounding box to get octants for.

    Returns:
        List[Gf.Range3d]: A list of 8 Gf.Range3d objects representing the octants.
    """
    if bounding_box.IsEmpty():
        raise ValueError("Cannot get octants of an empty bounding box.")
    octants = []
    for i in range(8):
        octant = bounding_box.GetOctant(i)
        octants.append(octant)
    return octants


def scale_bounding_box(range3d: Gf.Range3d, scale: float) -> Gf.Range3d:
    """Scale a bounding box represented by a Gf.Range3d.

    Args:
        range3d (Gf.Range3d): The bounding box to scale.
        scale (float): The scale factor to apply to the bounding box.

    Returns:
        Gf.Range3d: The scaled bounding box.
    """
    if range3d.IsEmpty():
        return range3d
    min_point = range3d.GetMin()
    max_point = range3d.GetMax()
    center = (min_point + max_point) * 0.5
    half_size = (max_point - min_point) * 0.5
    scaled_half_size = half_size * scale
    scaled_min = center - scaled_half_size
    scaled_max = center + scaled_half_size
    scaled_range = Gf.Range3d(scaled_min, scaled_max)
    return scaled_range


def get_bounding_box_corners(bounding_box: Gf.Range3d) -> Tuple[Gf.Vec3d, ...]:
    """
    Get the corners of a bounding box.

    The corners are returned in the following order:
    LDB, RDB, LUB, RUB, LDF, RDF, LUF, RUF
    (L/R = left/right, D/U = down/up, B/F = back/front)

    Parameters:
        bounding_box (Gf.Range3d): The bounding box to get the corners of.

    Returns:
        Tuple[Gf.Vec3d, ...]: A tuple of 8 Vec3d objects representing the corners.
    """
    if bounding_box.IsEmpty():
        raise ValueError("Cannot get corners of an empty bounding box.")
    corners = tuple((bounding_box.GetCorner(i) for i in range(8)))
    return corners


def find_prims_within_bounding_box(stage: Usd.Stage, bounding_box: Gf.Range3d) -> List[Usd.Prim]:
    """
    Find all prims on the stage whose bounding box intersects with the given bounding box.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        bounding_box (Gf.Range3d): The bounding box to check for intersection.

    Returns:
        List[Usd.Prim]: A list of prims whose bounding box intersects with the given bounding box.
    """
    prims_in_bounds: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        if UsdGeom.Imageable(prim):
            prim_bounds: Gf.Range3d = (
                UsdGeom.Imageable(prim)
                .ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
                .ComputeAlignedRange()
            )
            if prim_bounds.IntersectWith(bounding_box).IsEmpty() == False:
                prims_in_bounds.append(prim)
    return prims_in_bounds


def get_largest_prim_bounding_box(stage: Usd.Stage) -> Gf.Range3d:
    """
    Get the largest bounding box of all prims on the stage.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.

    Returns:
        Gf.Range3d: The largest bounding box of all prims on the stage.
                    Returns an empty range if no valid prims are found.
    """
    largest_bbox = Gf.Range3d()
    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Boundable):
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
            prim_bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if largest_bbox.IsEmpty():
                largest_bbox = prim_bbox
            else:
                largest_bbox = Gf.Range3d.GetUnion(largest_bbox, prim_bbox)
    return largest_bbox


def align_prims_within_bounding_box(stage: Usd.Stage, prim_paths: List[str], bounding_box: Gf.Range3d):
    """
    Aligns the given prims within the specified bounding box.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The paths of the prims to align.
        bounding_box (Gf.Range3d): The bounding box to align the prims within.
    """
    bbox_center = bounding_box.GetMidpoint()
    bbox_size = bounding_box.GetSize()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        prim_bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
        prim_center = prim_bbox.ComputeCentroid()
        prim_range = Gf.Range3d(prim_bbox.GetRange())
        prim_size = prim_range.GetSize()
        scale_factors = [bbox_size[i] / prim_size[i] for i in range(3)]
        scale_factor = min(scale_factors)
        scale_op = add_scale_op(xformable)
        scale_op.Set(Gf.Vec3f(scale_factor))
        translate_op = add_translate_op(xformable)
        translate_op.Set(bbox_center - prim_center * scale_factor)


def get_intersection_of_bboxes(a: Gf.Range3f, b: Gf.Range3f) -> Gf.Range3f:
    """
    Get the intersection of two bounding boxes.

    Args:
        a (Gf.Range3f): The first bounding box.
        b (Gf.Range3f): The second bounding box.

    Returns:
        Gf.Range3f: The intersection of the two bounding boxes.
    """
    (a_min, a_max) = (a.GetMin(), a.GetMax())
    (b_min, b_max) = (b.GetMin(), b.GetMax())
    intersection_min = Gf.Vec3f(max(a_min[0], b_min[0]), max(a_min[1], b_min[1]), max(a_min[2], b_min[2]))
    intersection_max = Gf.Vec3f(min(a_max[0], b_max[0]), min(a_max[1], b_max[1]), min(a_max[2], b_max[2]))
    intersection_bbox = Gf.Range3f(intersection_min, intersection_max)
    if intersection_bbox.IsEmpty():
        return Gf.Range3f()
    return intersection_bbox


def get_bbox_corners(bbox: Gf.Range3f) -> List[Gf.Vec3f]:
    """
    Returns the 8 corner points of a 3D bounding box.

    Args:
        bbox (Gf.Range3f): The bounding box to get the corners from.

    Returns:
        List[Gf.Vec3f]: A list of 8 Gf.Vec3f points representing the corners
        of the bounding box in the following order:
        [(-x, -y, -z), (+x, -y, -z), (-x, +y, -z), (+x, +y, -z),
         (-x, -y, +z), (+x, -y, +z), (-x, +y, +z), (+x, +y, +z)]
    """
    min_pt = bbox.GetMin()
    max_pt = bbox.GetMax()
    corners = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = min_pt[0] if i == 0 else max_pt[0]
                y = min_pt[1] if j == 0 else max_pt[1]
                z = min_pt[2] if k == 0 else max_pt[2]
                corners.append(Gf.Vec3f(x, y, z))
    return corners


def is_bbox_empty(bbox: Gf.Range3f) -> bool:
    """Check if a bounding box is empty.

    A bounding box is considered empty if any component of the min is greater
    than the corresponding component of the max.

    Args:
        bbox (Gf.Range3f): The bounding box to check.

    Returns:
        bool: True if the bounding box is empty, False otherwise.
    """
    if any((bbox.GetMin()[i] > bbox.GetMax()[i] for i in range(3))):
        return True
    return False


def expand_bbox_by_bbox(bbox_to_expand: Gf.Range3f, bbox_to_add: Gf.Range3f) -> Gf.Range3f:
    """Expand a bounding box by another bounding box.

    If bbox_to_expand is empty, it will be set to bbox_to_add.

    Args:
        bbox_to_expand (Gf.Range3f): The bounding box to expand.
        bbox_to_add (Gf.Range3f): The bounding box to add to bbox_to_expand.

    Returns:
        Gf.Range3f: The expanded bounding box.
    """
    if bbox_to_expand.IsEmpty():
        bbox_to_expand = Gf.Range3f(bbox_to_add)
    else:
        bbox_to_expand.UnionWith(bbox_to_add)
    return bbox_to_expand


def align_bbox_to_grid(bbox: Gf.Range3f, grid_size: float) -> Gf.Range3f:
    """
    Align the bounding box to the nearest grid lines.

    Args:
        bbox (Gf.Range3f): The input bounding box to align.
        grid_size (float): The size of the grid cells.

    Returns:
        Gf.Range3f: The aligned bounding box.
    """
    min_point = bbox.GetMin()
    max_point = bbox.GetMax()
    aligned_min = Gf.Vec3f(
        grid_size * (min_point[0] // grid_size),
        grid_size * (min_point[1] // grid_size),
        grid_size * (min_point[2] // grid_size),
    )
    aligned_max = Gf.Vec3f(
        grid_size * ((max_point[0] + grid_size - 1e-06) // grid_size),
        grid_size * ((max_point[1] + grid_size - 1e-06) // grid_size),
        grid_size * ((max_point[2] + grid_size - 1e-06) // grid_size),
    )
    aligned_bbox = Gf.Range3f(aligned_min, aligned_max)
    return aligned_bbox


def get_union_of_bboxes(bboxes: List[Gf.Range3f]) -> Gf.Range3f:
    """Get the union bounding box that encompasses all provided bboxes.

    Args:
        bboxes (List[Gf.Range3f]): List of bounding boxes to union.

    Returns:
        Gf.Range3f: The union bounding box.

    Raises:
        ValueError: If the input list of bboxes is empty.
    """
    if not bboxes:
        raise ValueError("Input list of bboxes is empty")
    union_bbox = Gf.Range3f(bboxes[0])
    for bbox in bboxes[1:]:
        union_bbox.UnionWith(bbox)
    return union_bbox


def is_point_inside_bbox(bbox: Gf.Range3f, point: Gf.Vec3f) -> bool:
    """Check if a point is inside a bounding box.

    Args:
        bbox (Gf.Range3f): The bounding box to check against.
        point (Gf.Vec3f): The point to check.

    Returns:
        bool: True if the point is inside the bounding box, False otherwise.
    """
    if bbox.IsEmpty():
        return False
    if (
        point[0] < bbox.GetMin()[0]
        or point[0] > bbox.GetMax()[0]
        or point[1] < bbox.GetMin()[1]
        or (point[1] > bbox.GetMax()[1])
        or (point[2] < bbox.GetMin()[2])
        or (point[2] > bbox.GetMax()[2])
    ):
        return False
    return True


def get_distance_squared_to_bbox(point: Gf.Vec3f, bbox: Gf.Range3f) -> float:
    """
    Compute the squared distance from a 3D point to a 3D bounding box.

    Args:
        point (Gf.Vec3f): The 3D point.
        bbox (Gf.Range3f): The 3D bounding box.

    Returns:
        float: The squared distance from the point to the bounding box.
    """
    dist_squared = 0.0
    for i in range(3):
        v = point[i]
        if v < bbox.GetMin()[i]:
            dist_squared += (bbox.GetMin()[i] - v) ** 2
        elif v > bbox.GetMax()[i]:
            dist_squared += (v - bbox.GetMax()[i]) ** 2
        else:
            pass
    return dist_squared


def expand_bbox_by_point(bbox: Gf.Range3f, point: Gf.Vec3f) -> Gf.Range3f:
    """Expand the bounding box to include the given point.

    If the bounding box is empty, set it to a zero-sized box at the point.

    Parameters:
        bbox (Gf.Range3f): The bounding box to expand.
        point (Gf.Vec3f): The point to include in the bounding box.

    Returns:
        Gf.Range3f: The expanded bounding box.
    """
    if bbox.IsEmpty():
        bbox.SetMin(point)
        bbox.SetMax(point)
    else:
        min_pt = bbox.GetMin()
        max_pt = bbox.GetMax()
        min_pt[0] = min(min_pt[0], point[0])
        min_pt[1] = min(min_pt[1], point[1])
        min_pt[2] = min(min_pt[2], point[2])
        max_pt[0] = max(max_pt[0], point[0])
        max_pt[1] = max(max_pt[1], point[1])
        max_pt[2] = max(max_pt[2], point[2])
        bbox.SetMin(min_pt)
        bbox.SetMax(max_pt)
    return bbox


def get_octants_of_bbox(bbox: Gf.Range3f) -> Tuple[Gf.Range3f, ...]:
    """Get the octants of a bounding box.

    Args:
        bbox (Gf.Range3f): The bounding box to get octants for.

    Returns:
        Tuple[Gf.Range3f, ...]: A tuple of 8 Gf.Range3f objects representing the octants.
    """
    if bbox.IsEmpty():
        raise ValueError("Cannot get octants of an empty bounding box.")
    min_point = bbox.GetMin()
    max_point = bbox.GetMax()
    midpoint = bbox.GetMidpoint()
    octants = (
        Gf.Range3f(min_point, midpoint),
        Gf.Range3f(Gf.Vec3f(midpoint[0], min_point[1], min_point[2]), Gf.Vec3f(max_point[0], midpoint[1], midpoint[2])),
        Gf.Range3f(Gf.Vec3f(min_point[0], midpoint[1], min_point[2]), Gf.Vec3f(midpoint[0], max_point[1], midpoint[2])),
        Gf.Range3f(Gf.Vec3f(midpoint[0], midpoint[1], min_point[2]), Gf.Vec3f(max_point[0], max_point[1], midpoint[2])),
        Gf.Range3f(Gf.Vec3f(min_point[0], min_point[1], midpoint[2]), Gf.Vec3f(midpoint[0], midpoint[1], max_point[2])),
        Gf.Range3f(Gf.Vec3f(midpoint[0], min_point[1], midpoint[2]), Gf.Vec3f(max_point[0], midpoint[1], max_point[2])),
        Gf.Range3f(Gf.Vec3f(min_point[0], midpoint[1], midpoint[2]), Gf.Vec3f(midpoint[0], max_point[1], max_point[2])),
        Gf.Range3f(midpoint, max_point),
    )
    return octants


def get_bboxes_of_prims(stage: Usd.Stage, prim_paths: List[str]) -> List[Tuple[Gf.Vec3f, Gf.Vec3f]]:
    """
    Get the bounding boxes of a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths.

    Returns:
        List[Tuple[Gf.Vec3f, Gf.Vec3f]]: A list of bounding boxes, each represented as a tuple of (min, max) points.
    """
    bboxes = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
        if bbox.GetRange().IsEmpty():
            print(f"Warning: Prim at path {prim_path} has an empty bounding box. Skipping.")
            continue
        (min_point, max_point) = (bbox.GetRange().GetMin(), bbox.GetRange().GetMax())
        bboxes.append((min_point, max_point))
    return bboxes


def get_bbox_size(stage: Usd.Stage, prim_path: str) -> Gf.Vec3f:
    """
    Get the size of the bounding box for a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Returns:
        Gf.Vec3f: The size of the bounding box.

    Raises:
        ValueError: If the prim is not valid or does not have a bounding box.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    bbox = bbox_cache.ComputeWorldBound(prim)
    if bbox.GetRange().IsEmpty():
        raise ValueError(f"Prim at path {prim_path} does not have a valid bounding box.")
    bbox_size = bbox.GetRange().GetSize()
    return bbox_size


def get_minimum_enclosing_bbox(prims: List[Usd.Prim]) -> Gf.Range3f:
    """
    Compute the minimum enclosing bounding box for a list of prims.

    Args:
        prims (List[Usd.Prim]): The list of prims to compute the bounding box for.

    Returns:
        Gf.Range3f: The minimum enclosing bounding box.

    Raises:
        ValueError: If the input prim list is empty.
    """
    if not prims:
        raise ValueError("Input prim list cannot be empty")
    bbox = UsdGeom.Imageable(prims[0]).ComputeLocalBound(Usd.TimeCode.Default(), "default").ComputeAlignedRange()
    for prim in prims[1:]:
        prim_bbox = UsdGeom.Imageable(prim).ComputeLocalBound(Usd.TimeCode.Default(), "default").ComputeAlignedRange()
        bbox.UnionWith(prim_bbox)
    return bbox


def create_prims_on_ray(stage: Usd.Stage, ray: Gf.Ray, prim_count: int, prim_type: str = "Sphere") -> None:
    """Create prims along a ray.

    Args:
        stage (Usd.Stage): The USD stage to create the prims on.
        ray (Gf.Ray): The ray along which to create the prims.
        prim_count (int): The number of prims to create.
        prim_type (str, optional): The type of prim to create. Defaults to "Sphere".

    Raises:
        ValueError: If prim_count is less than 1.
    """
    if prim_count < 1:
        raise ValueError("prim_count must be greater than or equal to 1.")
    ray_length = ray.direction.GetLength()
    distance_between_prims = ray_length / (prim_count - 1) if prim_count > 1 else 0
    for i in range(prim_count):
        distance = i * distance_between_prims
        position = ray.GetPoint(distance)
        prim_path = f"/Prim_{i}"
        prim = stage.DefinePrim(prim_path, prim_type)
        xform = UsdGeom.Xform(prim)
        add_translate_op(xform).Set(Gf.Vec3d(position))


def calculate_ray_bounding_box(ray: Gf.Ray, max_distance: float) -> Tuple[Gf.Vec3d, Gf.Vec3d]:
    """
    Calculate the bounding box of a ray segment.

    Args:
        ray (Gf.Ray): The ray to calculate the bounding box for.
        max_distance (float): The maximum distance along the ray to consider.

    Returns:
        Tuple[Gf.Vec3d, Gf.Vec3d]: A tuple containing the minimum and maximum points of the bounding box.
    """
    start_point = ray.startPoint
    end_point = ray.GetPoint(max_distance)
    min_point = Gf.Vec3d(start_point)
    max_point = Gf.Vec3d(start_point)
    for i in range(3):
        if end_point[i] < min_point[i]:
            min_point[i] = end_point[i]
        if end_point[i] > max_point[i]:
            max_point[i] = end_point[i]
    return (min_point, max_point)


def create_material_along_ray(
    stage: Usd.Stage, ray: Gf.Ray, color: Gf.Vec3f, distance: float = 1.0, material_name: str = "MyMaterial"
) -> UsdShade.Material:
    """Create a material at a point along a ray.

    Args:
        stage (Usd.Stage): The USD stage to create the material in.
        ray (Gf.Ray): The ray to use for positioning the material.
        color (Gf.Vec3f): The color to assign to the material.
        distance (float, optional): The distance along the ray to position the material. Defaults to 1.0.
        material_name (str, optional): The name to give the material. Defaults to "MyMaterial".

    Returns:
        UsdShade.Material: The created material prim.
    """
    point = ray.GetPoint(distance)
    material_path = Sdf.Path(f"/Materials/{material_name}")
    material = UsdShade.Material.Define(stage, material_path)
    diffuse_shader = UsdShade.Shader.Define(stage, material_path.AppendChild("DiffuseShader"))
    diffuse_shader.CreateIdAttr("UsdPreviewSurface")
    diffuse_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    material.CreateSurfaceOutput().ConnectToSource(diffuse_shader.ConnectableAPI(), "surface")
    sphere_path = material_path.AppendChild("Sphere")
    sphere = UsdGeom.Sphere.Define(stage, sphere_path)
    sphere.CreateRadiusAttr(1.0)
    add_translate_op(UsdGeom.Xform(sphere)).Set(point)
    return material


def select_prims_along_ray(stage: Usd.Stage, ray: Gf.Ray, max_distance: float) -> List[Usd.Prim]:
    """
    Select all prims along a ray up to a maximum distance.

    Args:
        stage (Usd.Stage): The USD stage to query.
        ray (Gf.Ray): The ray to trace.
        max_distance (float): The maximum distance to trace the ray.

    Returns:
        List[Usd.Prim]: A list of prims that intersect the ray, sorted by distance.
    """
    meshes = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(prim)
    hit_prims = []
    for mesh_prim in meshes:
        mesh = UsdGeom.Mesh(mesh_prim)
        (hit_point, hit_distance, _) = mesh.GetPointAtDistance(ray, max_distance)
        if hit_point is not None:
            hit_prims.append((mesh_prim, hit_distance))
    hit_prims.sort(key=lambda x: x[1])
    result_prims = [prim for (prim, dist) in hit_prims]
    return result_prims


def align_prims_along_ray(stage: Usd.Stage, prim_paths: List[str], ray: Gf.Ray) -> None:
    """
    Aligns a list of prims along a ray.

    The prims are positioned at equal intervals along the ray based on their
    order in the input list. The first prim is placed at the ray's start point,
    and the last prim is placed at the ray's end point.

    Parameters:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (List[str]): A list of paths to the prims to align.
        ray (Gf.Ray): The ray along which to align the prims.

    Raises:
        ValueError: If the stage is invalid, any prim path is invalid,
                    or if the list of prim paths is empty.
    """
    if not stage:
        raise ValueError("Invalid stage")
    if not prim_paths:
        raise ValueError("Empty list of prim paths")
    num_prims = len(prim_paths)
    ray_length = (ray.startPoint - ray.GetPoint(1)).GetLength()
    interval = ray_length / (num_prims - 1) if num_prims > 1 else 0
    for i, prim_path in enumerate(prim_paths):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            raise ValueError(f"Invalid prim path: {prim_path}")
        xform = UsdGeom.Xformable(prim)
        if not xform:
            raise ValueError(f"Prim at path {prim_path} is not transformable")
        point_on_ray = ray.GetPoint(i * interval / ray_length)
        add_translate_op(xform).Set(point_on_ray)


def find_closest_prim_along_ray(
    stage: Usd.Stage, ray: Gf.Ray, predicate: Optional[callable] = None
) -> Tuple[Usd.Prim, float]:
    """
    Find the closest prim along a ray in a USD stage.

    Args:
        stage (Usd.Stage): The USD stage to search.
        ray (Gf.Ray): The ray to cast.
        predicate (callable, optional): Optional predicate function to filter prims. Defaults to None.

    Returns:
        Tuple[Usd.Prim, float]: A tuple containing the closest prim and the distance along the ray.
                                Returns (None, float('inf')) if no prim is found.
    """
    closest_prim = None
    closest_distance = float("inf")
    for prim in stage.Traverse():
        if predicate and (not predicate(prim)):
            continue
        bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
        if not bbox.GetRange().IsEmpty():
            range_ = bbox.ComputeAlignedRange()
            (hit, enter_dist, _) = ray.Intersect(range_)
            if hit and enter_dist < closest_distance:
                closest_prim = prim
                closest_distance = enter_dist
    return (closest_prim, closest_distance)


def normalize_all_bounding_boxes(bounding_boxes: List[Gf.Rect2i]) -> List[Gf.Rect2i]:
    """Normalize a list of bounding boxes.

    Args:
        bounding_boxes (List[Gf.Rect2i]): A list of bounding boxes to normalize.

    Returns:
        List[Gf.Rect2i]: A new list with all bounding boxes normalized.
    """
    if not isinstance(bounding_boxes, list):
        raise TypeError("Input must be a list of Gf.Rect2i")
    result = []
    for bbox in bounding_boxes:
        if not isinstance(bbox, Gf.Rect2i):
            raise TypeError(f"Element {bbox} is not of type Gf.Rect2i")
        normalized_bbox = bbox.GetNormalized()
        result.append(normalized_bbox)
    return result


def find_invalid_prims(stage: Usd.Stage) -> List[Usd.Prim]:
    """Find all invalid prims in the given stage.

    An invalid prim is one that has no defined typename or one that
    fails validation.

    Parameters
    ----------
    stage : Usd.Stage
        The stage to search for invalid prims.

    Returns
    -------
    List[Usd.Prim]
        A list of invalid prims found in the stage.
    """
    invalid_prims: List[Usd.Prim] = []
    for prim in stage.Traverse():
        if not prim.HasDefiningSpecifier():
            invalid_prims.append(prim)
            continue
        if not prim.IsValid():
            invalid_prims.append(prim)
    return invalid_prims


def set_bounding_box_min_max(bounding_box: Gf.Rect2i, min_values: Gf.Vec2i, max_values: Gf.Vec2i):
    """Set the min and max values of a bounding box.

    Args:
        bounding_box (Gf.Rect2i): The bounding box to update.
        min_values (Gf.Vec2i): The new min values.
        max_values (Gf.Vec2i): The new max values.

    Raises:
        ValueError: If min_values are greater than max_values.
    """
    if min_values[0] > max_values[0] or min_values[1] > max_values[1]:
        raise ValueError("Min values must be less than or equal to max values.")
    bounding_box.SetMin(min_values)
    bounding_box.SetMax(max_values)


def get_bounding_box_diagonal(bbox: Gf.Rect2i) -> float:
    """Calculate the diagonal length of a 2D bounding box.

    Args:
        bbox (Gf.Rect2i): The input 2D bounding box.

    Returns:
        float: The diagonal length of the bounding box.

    Raises:
        ValueError: If the input bounding box is empty or invalid.
    """
    if not bbox.IsValid():
        raise ValueError("Input bounding box is invalid.")
    if bbox.IsEmpty():
        raise ValueError("Input bounding box is empty.")
    min_point = bbox.GetMin()
    max_point = bbox.GetMax()
    width = max_point[0] - min_point[0]
    height = max_point[1] - min_point[1]
    diagonal = (width**2 + height**2) ** 0.5
    return diagonal


def find_null_prims(stage: Usd.Stage) -> List[Usd.Prim]:
    """Find all null prims on the given stage.

    A null prim is defined as a prim with no properties, attributes, or relationships.
    """
    null_prims: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        if prim.GetProperties():
            continue
        if prim.GetAttributes():
            continue
        if prim.GetRelationships():
            continue
        null_prims.append(prim)
    return null_prims


def expand_bounding_boxes(boxes: List[Gf.Rect2i], expansion_size: int) -> List[Gf.Rect2i]:
    """
    Expand a list of bounding boxes by a specified expansion size.

    Args:
        boxes (List[Gf.Rect2i]): A list of bounding boxes to expand.
        expansion_size (int): The size by which to expand each bounding box.

    Returns:
        List[Gf.Rect2i]: A new list of expanded bounding boxes.
    """
    expanded_boxes = []
    for box in boxes:
        expanded_box = Gf.Rect2i(box)
        expanded_box.min = expanded_box.min - Gf.Vec2i(expansion_size, expansion_size)
        expanded_box.max = expanded_box.max + Gf.Vec2i(expansion_size, expansion_size)
        expanded_boxes.append(expanded_box)
    return expanded_boxes


def shrink_bounding_boxes(bboxes: List[Gf.Rect2i], shrink_pixels: int) -> List[Gf.Rect2i]:
    """
    Shrink a list of bounding boxes by a specified number of pixels on each side.

    Args:
        bboxes (List[Gf.Rect2i]): List of bounding boxes to shrink.
        shrink_pixels (int): Number of pixels to shrink on each side.

    Returns:
        List[Gf.Rect2i]: List of shrunken bounding boxes.
    """
    shrunken_bboxes = []
    for bbox in bboxes:
        if not bbox.IsValid():
            continue
        new_min_x = bbox.GetMinX() + shrink_pixels
        new_min_y = bbox.GetMinY() + shrink_pixels
        new_max_x = bbox.GetMaxX() - shrink_pixels
        new_max_y = bbox.GetMaxY() - shrink_pixels
        if new_min_x <= new_max_x and new_min_y <= new_max_y:
            shrunken_bbox = Gf.Rect2i(Gf.Vec2i(new_min_x, new_min_y), Gf.Vec2i(new_max_x, new_max_y))
            shrunken_bboxes.append(shrunken_bbox)
    return shrunken_bboxes


def group_prims_by_bounding_box(
    stage: Usd.Stage, prim_paths: List[str], threshold: float
) -> Dict[Tuple[int, int], List[str]]:
    """
    Group prims based on their bounding box proximity.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The list of prim paths to group.
        threshold (float): The maximum distance between bounding boxes to consider them as part of the same group.

    Returns:
        Dict[Tuple[int, int], List[str]]: A dictionary mapping grid coordinates to a list of prim paths in that group.
    """
    groups: Dict[Tuple[int, int], List[str]] = {}
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid() and prim.GetAttribute("extent").IsValid():
            extent = prim.GetAttribute("extent").Get()
            if extent:
                center = (extent[0] + extent[1]) * 0.5
                grid_x = int(center[0] / threshold)
                grid_y = int(center[2] / threshold)
                if (grid_x, grid_y) not in groups:
                    groups[grid_x, grid_y] = []
                groups[grid_x, grid_y].append(prim_path)
    return groups


def calculate_total_bounding_box_area(bounding_boxes: List[Gf.Rect2i]) -> int:
    """Calculate the total area of a list of bounding boxes.

    Args:
        bounding_boxes (List[Gf.Rect2i]): A list of bounding boxes.

    Returns:
        int: The total area of the bounding boxes.
    """
    if not bounding_boxes:
        return 0
    total_area = 0
    for bbox in bounding_boxes:
        if not bbox.IsValid():
            continue
        area = bbox.GetArea()
        total_area += area
    return total_area


def compute_prim_area(stage: Usd.Stage, prim_path: str) -> float:
    """Compute the area of a prim's bounding box."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    bbox_range = bbox.ComputeAlignedRange()
    if bbox_range.IsEmpty():
        return 0.0
    min_point = bbox_range.GetMin()
    max_point = bbox_range.GetMax()
    dimensions = max_point - min_point
    dimensions_vec2i = Gf.Vec2i(int(dimensions[0]), int(dimensions[1]))
    rect = Gf.Rect2i(Gf.Vec2i(0, 0), dimensions_vec2i)
    return rect.GetArea()


def set_prim_bounding_box(prim: Usd.Prim, extent: Gf.Rect2i):
    """Set the bounding box of a prim to the given extent.

    Args:
        prim (Usd.Prim): The prim to set the bounding box for.
        extent (Gf.Rect2i): The bounding box extent.

    Raises:
        ValueError: If the prim is not valid or not a UsdGeom.Boundable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    boundable = UsdGeom.Boundable(prim)
    if not boundable:
        raise ValueError(f"Prim {prim.GetPath()} is not boundable.")
    min_pt = Gf.Vec3f(extent.GetMinX(), extent.GetMinY(), 0)
    max_pt = Gf.Vec3f(extent.GetMaxX(), extent.GetMaxY(), 0)
    extent_attr = boundable.CreateExtentAttr()
    extent_attr.Set([min_pt, max_pt])


def find_empty_prims(stage: Usd.Stage) -> List[Usd.Prim]:
    """Find all empty prims in the given stage.

    An empty prim is defined as a prim that has no properties, attributes,
    relationships, or child prims.

    Args:
        stage (Usd.Stage): The stage to search for empty prims.

    Returns:
        List[Usd.Prim]: A list of empty prims found in the stage.
    """
    empty_prims: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        if prim.GetPropertyNames():
            continue
        if prim.GetAttributes():
            continue
        if prim.GetRelationships():
            continue
        if prim.GetChildren():
            continue
        empty_prims.append(prim)
    return empty_prims


def calculate_average_bounding_box_size(stage: Usd.Stage, prim_paths: List[str]) -> Gf.Vec2f:
    """
    Calculate the average bounding box size for a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The list of prim paths.

    Returns:
        Gf.Vec2f: The average bounding box size (width, height).
    """
    if not prim_paths:
        raise ValueError("Prim paths list is empty.")
    total_width = 0
    total_height = 0
    valid_prim_count = 0
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        bounding_box = UsdGeom.Imageable(prim).ComputeLocalBound(Usd.TimeCode.Default(), "default")
        if bounding_box.GetRange().IsEmpty():
            continue
        min_point = bounding_box.GetRange().GetMin()
        max_point = bounding_box.GetRange().GetMax()
        width = max_point[0] - min_point[0]
        height = max_point[1] - min_point[1]
        total_width += width
        total_height += height
        valid_prim_count += 1
    if valid_prim_count == 0:
        raise ValueError("No valid prims found.")
    average_width = total_width / valid_prim_count
    average_height = total_height / valid_prim_count
    return Gf.Vec2f(average_width, average_height)


def get_prim_bounding_box_dimensions(stage: Usd.Stage, prim_path: str) -> Gf.Vec3f:
    """
    Get the dimensions of the bounding box of a prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.

    Returns:
        Gf.Vec3f: The dimensions of the bounding box as (width, height, depth).
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    bounding_box = UsdGeom.Imageable(prim).ComputeLocalBound(Usd.TimeCode.Default(), "default")
    if bounding_box.GetRange().IsEmpty():
        return Gf.Vec3f(0, 0, 0)
    range = bounding_box.ComputeAlignedRange()
    min = range.GetMin()
    max = range.GetMax()
    dimensions = Gf.Vec3f(max[0] - min[0], max[1] - min[1], max[2] - min[2])
    return dimensions


def get_prim_center(prim: Usd.Prim) -> Tuple[float, float, float]:
    """
    Get the center point of a prim's bounding box in local space.

    Args:
        prim (Usd.Prim): The prim to get the center point for.

    Returns:
        Tuple[float, float, float]: The center point of the prim's bounding box.

    Raises:
        ValueError: If the prim is not valid or does not have a defined extent.
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid.")
    bbox = UsdGeom.Imageable(prim).ComputeLocalBound(Usd.TimeCode.Default(), purpose1="default")
    if bbox.GetRange().IsEmpty():
        raise ValueError("Prim does not have a defined extent.")
    box_range = bbox.GetRange()
    min_point = box_range.GetMin()
    max_point = box_range.GetMax()
    center_x = (min_point[0] + max_point[0]) / 2
    center_y = (min_point[1] + max_point[1]) / 2
    center_z = (min_point[2] + max_point[2]) / 2
    return (center_x, center_y, center_z)


def translate_prims(stage: Usd.Stage, prim_paths: List[str], translation: Gf.Vec3d) -> None:
    """Translate multiple prims by a given translation vector.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths to translate.
        translation (Gf.Vec3d): The translation vector.

    Raises:
        ValueError: If any prim path is invalid or not transformable.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        translate_op = add_translate_op(xformable)
        translate_op.Set(translation)


def get_translation(prim):
    xformable = UsdGeom.Xformable(prim)
    translate_op = xformable.GetOrderedXformOps()[0]
    return translate_op.Get()


def snap_prim_to_grid(prim: Usd.Prim, grid_size: Gf.Vec3d) -> None:
    """Snap a prim's translation to a grid.

    Args:
        prim (Usd.Prim): The prim to snap to the grid.
        grid_size (Gf.Vec3d): The size of the grid cells.

    Raises:
        ValueError: If the prim is not transformable.
    """
    if not prim.IsA(UsdGeom.Xformable):
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    xformable = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if not translate_ops:
        translate_op = add_translate_op(xformable)
        translate_op.Set(Gf.Vec3d(0, 0, 0))
    else:
        translate_op = translate_ops[0]
    translation = translate_op.Get()
    snapped_translation = Gf.Vec3d(
        round(translation[0] / grid_size[0]) * grid_size[0],
        round(translation[1] / grid_size[1]) * grid_size[1],
        round(translation[2] / grid_size[2]) * grid_size[2],
    )
    translate_op.Set(snapped_translation)


def set_default_bounding_box(prim: Usd.Prim, bounding_box: Gf.Rect2i):
    """Set the default bounding box for a prim.

    Args:
        prim (Usd.Prim): The prim to set the bounding box on.
        bounding_box (Gf.Rect2i): The bounding box to set.

    Raises:
        ValueError: If the prim is not valid or not a UsdGeom.Boundable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    boundable = UsdGeom.Boundable(prim)
    if not boundable:
        raise ValueError(f"Prim {prim.GetPath()} is not a UsdGeom.Boundable.")
    min_corner = bounding_box.GetMin()
    max_corner = bounding_box.GetMax()
    extents = Gf.Range3f(Gf.Vec3f(min_corner[0], min_corner[1], 0), Gf.Vec3f(max_corner[0], max_corner[1], 0))
    extent_array = Vt.Vec3fArray(2)
    extent_array[0] = extents.GetMin()
    extent_array[1] = extents.GetMax()
    boundable.CreateExtentAttr().Set(extent_array, Usd.TimeCode.Default())


def find_prims_by_bounding_box_size(
    stage: Usd.Stage, min_size: Tuple[float, float, float], max_size: Tuple[float, float, float]
) -> List[Usd.Prim]:
    """
    Find all prims on the stage whose bounding box size falls within the specified range.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        min_size (Tuple[float, float, float]): The minimum bounding box size (width, height, depth).
        max_size (Tuple[float, float, float]): The maximum bounding box size (width, height, depth).

    Returns:
        List[Usd.Prim]: A list of prims whose bounding box size falls within the specified range.
    """
    if any((dim < 0 for dim in min_size)) or any((dim < 0 for dim in max_size)):
        raise ValueError("Bounding box dimensions cannot be negative.")
    min_vec = Gf.Vec3d(*min_size)
    max_vec = Gf.Vec3d(*max_size)
    matching_prims = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Imageable):
            continue
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        bounds = bbox_cache.ComputeWorldBound(prim)
        box_size = bounds.ComputeAlignedBox().max - bounds.ComputeAlignedBox().min
        if all((min_vec[i] <= box_size[i] <= max_vec[i] for i in range(3))):
            matching_prims.append(prim)
    return matching_prims


def get_prims_within_radius(stage: Usd.Stage, center: Gf.Vec3d, radius: float) -> List[Usd.Prim]:
    """
    Get all prims within a given radius from a center point.

    Args:
        stage (Usd.Stage): The USD stage to query.
        center (Gf.Vec3d): The center point to measure distance from.
        radius (float): The maximum distance a prim can be from the center to be included.

    Returns:
        List[Usd.Prim]: A list of prims within the specified radius from the center point.
    """
    prims_in_radius = []
    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Imageable):
            bounds = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
            range = bounds.ComputeAlignedRange()
            if range.IsEmpty():
                continue
            bbox_center = range.GetMidpoint()
            distance = (Gf.Vec3d(bbox_center) - center).GetLength()
            if distance <= radius:
                prims_in_radius.append(prim)
    return prims_in_radius


def move_prim_to_origin(stage: Usd.Stage, prim_path: str) -> None:
    """Move the specified prim to the origin (0, 0, 0)."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    translate_ops = xformable.GetOrderedXformOps()
    translate_op = None
    for op in translate_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op:
        current_translation = translate_op.Get()
    else:
        current_translation = Gf.Vec3d(0, 0, 0)
    new_translation = Gf.Vec3d(0, 0, 0) - current_translation
    if translate_op:
        translate_op.Set(new_translation)
    else:
        add_translate_op(xformable).Set(new_translation)


def align_prims_to_center(stage: Usd.Stage, prim_paths: List[str]) -> None:
    """Align the given prims to the center of their bounding box."""
    if not prim_paths:
        return
    bounds = Gf.Range3d()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        prim_bounds = bbox_cache.ComputeWorldBound(prim)
        if prim_bounds.GetRange().IsEmpty():
            continue
        bounds.UnionWith(prim_bounds.GetRange())
    if bounds.IsEmpty():
        return
    center = bounds.GetMidpoint()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        xformable = UsdGeom.Xformable(prim)
        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        prim_position = world_transform.Transform(Gf.Vec3d(0, 0, 0))
        translation = center - prim_position
        translate_op = xformable.GetOrderedXformOps()[0]
        if translate_op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op.Set(translation)
        else:
            add_translate_op(xformable).Set(translation)


def find_prims_by_bounding_box_area(stage: Usd.Stage, min_area: float) -> List[Usd.Prim]:
    """Find all prims on the stage whose bounding box area is greater than or equal to min_area."""
    prims = []
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Imageable):
            continue
        purpose = UsdGeom.Tokens.default_
        bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose)
        range = bbox.ComputeAlignedRange()
        size = range.GetSize()
        area = size[0] * size[1]
        if area >= min_area:
            prims.append(prim)
    return prims


def set_prim_rotation_by_quaternion(stage: Usd.Stage, prim_path: str, quaternion: Gf.Quatd) -> None:
    """Set the rotation of a prim using a quaternion.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        quaternion (Gf.Quatd): The quaternion representing the rotation.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    rotation = Gf.Rotation(quaternion)
    add_rotate_xyz_op(xformable).Set(rotation.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis()))


def get_prim_rotation_as_quaternion(stage: Usd.Stage, prim_path: str) -> Gf.Quatd:
    """Get the rotation for a prim as a quaternion."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    rotation_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ]
    if not rotation_ops:
        return Gf.Quatd(1.0)
    rotation_op = rotation_ops[0]
    rotation_attr = rotation_op.GetAttr()
    if not rotation_attr.HasValue():
        return Gf.Quatd(1.0)
    rotation_vec3d = rotation_attr.Get()
    rotation_rad = Gf.Vec3d(
        Gf.DegreesToRadians(rotation_vec3d[0]),
        Gf.DegreesToRadians(rotation_vec3d[1]),
        Gf.DegreesToRadians(rotation_vec3d[2]),
    )
    rotation = (
        Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation_rad[0])
        * Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation_rad[1])
        * Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation_rad[2])
    )
    return rotation.GetQuat()


def set_prim_rotation_to_identity(stage: Usd.Stage, prim_path: str) -> bool:
    """Set the rotation of a prim to the identity rotation.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Returns:
        bool: True if the rotation was set successfully, False otherwise.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return False
    ops = xformable.GetOrderedXformOps()
    rotation_op = None
    for op in ops:
        if op.GetOpName().startswith("xformOp:rotateXYZ"):
            rotation_op = op
            break
    if not rotation_op:
        rotation_op = add_rotate_xyz_op(xformable)
    identity_rotation = Gf.Vec3f(0, 0, 0)
    rotation_op.Set(identity_rotation)
    return True


def align_hierarchical_rotations(stage: Usd.Stage, root_path: str, target_rotation: Gf.Rotation) -> None:
    """
    Aligns the local rotations of prims in a hierarchy to a target rotation.

    Args:
        stage (Usd.Stage): The USD stage.
        root_path (str): The path to the root prim of the hierarchy.
        target_rotation (Gf.Rotation): The target rotation to align to.
    """
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise ValueError(f"Invalid root prim path: {root_path}")
    for prim in Usd.PrimRange(root_prim):
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            continue
        rotate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ]
        rotation_op = rotate_ops[0] if rotate_ops else add_rotate_xyz_op(xformable)
        current_rotation = rotation_op.Get()
        if current_rotation is None:
            current_rotation = Gf.Vec3d(0, 0, 0)
        current_gf_rotation = (
            Gf.Rotation(Gf.Vec3d(1, 0, 0), current_rotation[0])
            * Gf.Rotation(Gf.Vec3d(0, 1, 0), current_rotation[1])
            * Gf.Rotation(Gf.Vec3d(0, 0, 1), current_rotation[2])
        )
        aligned_rotation = current_gf_rotation * target_rotation
        new_rotation_xyz = aligned_rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
        rotation_op.Set(new_rotation_xyz)


def get_aggregate_rotation(stage: Usd.Stage, prim_path: str) -> Optional[Gf.Rotation]:
    """
    Get the aggregate rotation for a prim.

    This function computes the aggregate rotation by combining all the
    rotation transformations applied to the prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Returns:
        Optional[Gf.Rotation]: The aggregate rotation, or None if the prim has no rotation.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    rotation_ops = [
        op
        for op in xformable.GetOrderedXformOps()
        if op.GetOpType()
        in (
            UsdGeom.XformOp.TypeRotateX,
            UsdGeom.XformOp.TypeRotateY,
            UsdGeom.XformOp.TypeRotateZ,
            UsdGeom.XformOp.TypeRotateXYZ,
            UsdGeom.XformOp.TypeRotateXZY,
            UsdGeom.XformOp.TypeRotateYXZ,
            UsdGeom.XformOp.TypeRotateYZX,
            UsdGeom.XformOp.TypeRotateZXY,
            UsdGeom.XformOp.TypeRotateZYX,
        )
    ]
    if not rotation_ops:
        return None
    aggregate_rotation = Gf.Rotation()
    for op in rotation_ops:
        rotation_attr = op.GetAttr()
        if rotation_attr.HasValue():
            rotation_value = rotation_attr.Get()
            op_type = op.GetOpType()
            if op_type == UsdGeom.XformOp.TypeRotateX:
                rotation = Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation_value)
            elif op_type == UsdGeom.XformOp.TypeRotateY:
                rotation = Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation_value)
            elif op_type == UsdGeom.XformOp.TypeRotateZ:
                rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation_value)
            else:
                rotation = Gf.Rotation(Gf.Vec3d(rotation_value))
            aggregate_rotation *= rotation
    return aggregate_rotation


def set_prim_rotation_to_align_with_vector(stage: Usd.Stage, prim_path: str, target_vector: Gf.Vec3d):
    """Set the rotation of a prim to align with a target vector.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        target_vector (Gf.Vec3d): The target vector to align with.

    Raises:
        ValueError: If the prim is not valid or transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    rotation_ops = xformable.GetOrderedXformOps()
    rotation_op = None
    for op in rotation_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            rotation_op = op
            break
    if not rotation_op:
        rotation_op = xformable.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ)
    old_rotation_euler = rotation_op.Get()
    if not old_rotation_euler:
        old_rotation_euler = Gf.Vec3d(0, 0, 0)
    old_rotation = (
        Gf.Rotation(Gf.Vec3d(1, 0, 0), old_rotation_euler[0])
        * Gf.Rotation(Gf.Vec3d(0, 1, 0), old_rotation_euler[1])
        * Gf.Rotation(Gf.Vec3d(0, 0, 1), old_rotation_euler[2])
    )
    old_quat = old_rotation.GetQuat()
    default_up_vector = Gf.Vec3d(0, 1, 0)
    rotation = Gf.Rotation()
    rotation.SetRotateInto(default_up_vector, target_vector)
    new_quat = rotation.GetQuat() * old_quat
    new_rotation = Gf.Rotation(new_quat)
    new_rotation_euler = new_rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
    rotation_op.Set(new_rotation_euler)


def decompose_prim_rotation(
    prim: Usd.Prim, tw_axis: Gf.Vec3d, fb_axis: Gf.Vec3d, lr_axis: Gf.Vec3d
) -> Tuple[float, float, float, float]:
    """
    Decompose the rotation of a prim into Euler angles about the given axes.

    Args:
        prim (Usd.Prim): The prim to get the rotation from.
        tw_axis (Gf.Vec3d): The twist axis.
        fb_axis (Gf.Vec3d): The front-back axis.
        lr_axis (Gf.Vec3d): The left-right axis.

    Returns:
        Tuple[float, float, float, float]: The twist, front-back, left-right, and swing angles in degrees.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable")
    rotation_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ]
    if not rotation_ops:
        return (0.0, 0.0, 0.0, 0.0)
    rotation_op = rotation_ops[0]
    rotation = rotation_op.Get(Usd.TimeCode.Default())
    if rotation is None:
        return (0.0, 0.0, 0.0, 0.0)
    rotation_matrix = Gf.Matrix4d()
    rotation_matrix.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), math.radians(rotation[0])))
    rotation_matrix.SetRotateOnly(
        Gf.Rotation(Gf.Vec3d(0, 1, 0), math.radians(rotation[1]))
        * Gf.Rotation(Gf.Vec3d(0, 0, 1), math.radians(rotation[2]))
    )
    theta_tw = 0.0
    theta_fb = 0.0
    theta_lr = 0.0
    theta_sw = 0.0
    handedness = 1.0
    Gf.Rotation.DecomposeRotation(
        rotation_matrix, tw_axis, fb_axis, lr_axis, handedness, theta_tw, theta_fb, theta_lr, theta_sw, False, 0
    )
    theta_tw = math.degrees(theta_tw)
    theta_fb = math.degrees(theta_fb)
    theta_lr = math.degrees(theta_lr)
    theta_sw = math.degrees(theta_sw)
    return (theta_tw, theta_fb, theta_lr, theta_sw)


def rotate_prim_onto_projection(stage: Usd.Stage, prim_path: str, v1: Gf.Vec3d, v2: Gf.Vec3d, axis: Gf.Vec3d) -> None:
    """
    Rotate a prim to align one vector with another projected onto a plane.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): Path to the prim to rotate.
        v1 (Gf.Vec3d): The source vector.
        v2 (Gf.Vec3d): The destination vector.
        axis (Gf.Vec3d): The normal vector of the plane to project onto.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    v1_projected = v1 - Gf.Dot(v1, axis) * axis
    v2_projected = v2 - Gf.Dot(v2, axis) * axis
    v1_projected = v1_projected.GetNormalized()
    v2_projected = v2_projected.GetNormalized()
    rotation = Gf.Rotation()
    rotation.SetRotateInto(v1_projected, v2_projected)
    rotate_op = xformable.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ)
    rotate_op.Set(rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)), Usd.TimeCode.Default())


def match_prim_rotation_to_hint(prim: Usd.Prim, target_rotation: Gf.Quatf) -> None:
    """
    Match the rotation of a prim to a target rotation hint.

    The rotation will be set to the closest rotation to the target hint.

    Args:
        prim (Usd.Prim): The prim to set the rotation on.
        target_rotation (Gf.Quatf): The target rotation hint as a quaternion.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    rotation_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            rotation_op = op
            break
    if rotation_op:
        old_rotation_euler = rotation_op.Get()
        old_rotation = Gf.Rotation(
            Gf.Vec3d(
                Gf.DegreesToRadians(old_rotation_euler[0]),
                Gf.DegreesToRadians(old_rotation_euler[1]),
                Gf.DegreesToRadians(old_rotation_euler[2]),
            ),
            Gf.Vec3d(1, 0, 0),
        )
    else:
        old_rotation = Gf.Rotation()
    old_rotation_quat = old_rotation.GetQuat()
    target_tw = 0
    target_fb = 0
    target_lr = 0
    target_sw = 0
    Gf.Rotation.MatchClosestEulerRotation(
        target_tw,
        target_fb,
        target_lr,
        target_sw,
        old_rotation_quat.GetImaginary()[0],
        old_rotation_quat.GetImaginary()[1],
        old_rotation_quat.GetImaginary()[2],
        old_rotation_quat.GetReal(),
    )
    new_rotation_euler = Gf.Vec3f(
        Gf.RadiansToDegrees(target_tw), Gf.RadiansToDegrees(target_fb), Gf.RadiansToDegrees(target_lr)
    )
    if not rotation_op:
        rotation_op = add_rotate_xyz_op(xformable)
    rotation_op.Set(new_rotation_euler)


def batch_set_rotation_by_axis_angle(
    stage: Usd.Stage, prim_paths: List[str], axis_angles: List[Tuple[Gf.Vec3d, float]]
) -> None:
    """
    Set the rotation for multiple prims using axis-angle representation.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths.
        axis_angles (List[Tuple[Gf.Vec3d, float]]): List of axis-angle pairs, where each pair is a tuple of (axis, angle_in_degrees).

    Raises:
        ValueError: If the number of prim paths and axis-angle pairs do not match.
    """
    if len(prim_paths) != len(axis_angles):
        raise ValueError("Number of prim paths and axis-angle pairs must match.")
    for prim_path, (axis, angle) in zip(prim_paths, axis_angles):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            continue
        rotation = Gf.Rotation(Gf.Vec3d(axis), angle)
        euler_angles = Gf.Vec3f(rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)))
        UsdGeom.XformCommonAPI(xformable).SetRotate(euler_angles, UsdGeom.XformCommonAPI.RotationOrderXYZ)


def batch_set_rotation_by_quaternion(stage: Usd.Stage, prim_quat_pairs: List[Tuple[str, Gf.Quatf]]) -> None:
    """Set rotation for multiple prims using quaternions.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_quat_pairs (List[Tuple[str, Gf.Quatf]]): A list of tuples, each containing a prim path and a quaternion.
    """
    for prim_path, quat in prim_quat_pairs:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        rotation_op = add_orient_op(xformable)
        rotation_op.Set(quat)


def align_prim_rotation(stage: Usd.Stage, prim_path: str, target_dir: Gf.Vec3d) -> None:
    """Align the rotation of a prim to a target direction.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to align.
        target_dir (Gf.Vec3d): The target direction to align the prim to.

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    xform_ops = xformable.GetOrderedXformOps()
    rotation_op = None
    for op in xform_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            rotation_op = op
            break
    if not rotation_op:
        rotation_op = xformable.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ)
    current_rotation = rotation_op.Get()
    current_rotation = (
        Gf.Rotation(Gf.Vec3d(1, 0, 0), current_rotation[0])
        * Gf.Rotation(Gf.Vec3d(0, 1, 0), current_rotation[1])
        * Gf.Rotation(Gf.Vec3d(0, 0, 1), current_rotation[2])
    )
    current_dir = current_rotation.TransformDir(Gf.Vec3d(0, 0, 1))
    align_rotation = Gf.Rotation()
    align_rotation.SetRotateInto(current_dir, target_dir)
    new_rotation = current_rotation * align_rotation
    new_euler_rotation = new_rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
    rotation_op.Set(new_euler_rotation)


def set_prim_rotation_by_axis_angle(prim: Usd.Prim, axis: Gf.Vec3d, angle: float) -> None:
    """Set the rotation for a prim using axis-angle representation.

    Args:
        prim (Usd.Prim): The prim to set the rotation for.
        axis (Gf.Vec3d): The axis of rotation.
        angle (float): The angle of rotation in degrees.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim '{prim.GetPath()}' is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim '{prim.GetPath()}' is not transformable.")
    rotation = Gf.Rotation()
    rotation.SetAxisAngle(axis, angle)
    euler_angles = rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
    euler_angles = Gf.Vec3f(math.degrees(euler_angles[0]), math.degrees(euler_angles[1]), math.degrees(euler_angles[2]))
    add_rotate_xyz_op(xformable).Set(euler_angles, Usd.TimeCode.Default())


def transform_dir_by_prim_rotation(stage: Usd.Stage, prim_path: str, dir: Gf.Vec3d) -> Gf.Vec3d:
    """Transform a direction vector by the rotation of a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        dir (Gf.Vec3d): The direction vector to transform.

    Returns:
        Gf.Vec3d: The transformed direction vector.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim not found at path: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    rotation_ops = [
        op
        for op in xformable.GetOrderedXformOps()
        if op.GetOpType() in [UsdGeom.XformOp.TypeRotateX, UsdGeom.XformOp.TypeRotateY, UsdGeom.XformOp.TypeRotateZ]
    ]
    if not rotation_ops:
        return dir
    rotation = Gf.Rotation()
    for op in rotation_ops:
        rotation *= op.Get()
    transformed_dir = rotation.TransformDir(dir)
    return transformed_dir


def decompose_rotation_matrix(
    rot: Gf.Matrix4d,
    TwAxis: Gf.Vec3d,
    FBAxis: Gf.Vec3d,
    LRAxis: Gf.Vec3d,
    handedness: float,
    thetaTw: float,
    thetaFB: float,
    thetaLR: float,
    thetaSw: float,
    useHint: bool,
    swShift: float,
) -> None:
    """
    Decompose a rotation matrix into Tw, FB, LR, Sw angles.

    Args:
        rot (Gf.Matrix4d): The input rotation matrix.
        TwAxis (Gf.Vec3d): The twist axis.
        FBAxis (Gf.Vec3d): The front/back axis.
        LRAxis (Gf.Vec3d): The left/right axis.
        handedness (float): The handedness of the coordinate system.
        thetaTw (float): The output twist angle in radians.
        thetaFB (float): The output front/back angle in radians.
        thetaLR (float): The output left/right angle in radians.
        thetaSw (float): The output swing angle in radians.
        useHint (bool): Whether to use the input angles as a hint.
        swShift (float): The shift applied to the swing angle.
    """
    if (
        not Gf.IsClose(Gf.Dot(TwAxis, FBAxis), 0.0, 1e-06)
        or not Gf.IsClose(Gf.Dot(TwAxis, LRAxis), 0.0, 1e-06)
        or (not Gf.IsClose(Gf.Dot(FBAxis, LRAxis), 0.0, 1e-06))
    ):
        raise ValueError("Input axes are not orthogonal.")
    TwAxis = TwAxis.GetNormalized()
    FBAxis = FBAxis.GetNormalized()
    LRAxis = LRAxis.GetNormalized()
    rotation = Gf.Rotation()
    rotation.SetRotateInto(Gf.Vec3d(1, 0, 0), TwAxis)
    (thetaFB, thetaLR, thetaTw) = rotation.Decompose(TwAxis, FBAxis, LRAxis)
    thetaSw = thetaFB * handedness
    if useHint:
        Gf.Rotation.MatchClosestEulerRotation(thetaTw, thetaFB, thetaLR, thetaSw, thetaTw, thetaFB, thetaLR, thetaSw)
    thetaSw += swShift
    thetaTw = thetaTw
    thetaFB = thetaFB
    thetaLR = thetaLR
    thetaSw = thetaSw


def blend_prim_rotations(stage: Usd.Stage, prim_path_a: str, prim_path_b: str, weight: float = 0.5) -> Gf.Rotation:
    """Blend the rotations of two prims using spherical linear interpolation.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path_a (str): The path to the first prim.
        prim_path_b (str): The path to the second prim.
        weight (float, optional): The blending weight between 0 and 1. Defaults to 0.5.

    Raises:
        ValueError: If the prims are not valid or transformable, or if the weight is out of range.

    Returns:
        Gf.Rotation: The blended rotation.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"Weight must be between 0 and 1, got {weight}.")
    prim_a = stage.GetPrimAtPath(prim_path_a)
    prim_b = stage.GetPrimAtPath(prim_path_b)
    if not prim_a.IsValid() or not prim_b.IsValid():
        raise ValueError(f"One or both prims are not valid: {prim_path_a}, {prim_path_b}")
    xformable_a = UsdGeom.Xformable(prim_a)
    xformable_b = UsdGeom.Xformable(prim_b)
    if not xformable_a or not xformable_b:
        raise ValueError(f"One or both prims are not transformable: {prim_path_a}, {prim_path_b}")
    rotation_ops_a = [op for op in xformable_a.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ]
    rotation_ops_b = [op for op in xformable_b.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ]
    rotation_a = Gf.Vec3d(0, 0, 0) if not rotation_ops_a else Gf.Vec3d(*rotation_ops_a[0].Get(Usd.TimeCode.Default()))
    rotation_b = Gf.Vec3d(0, 0, 0) if not rotation_ops_b else Gf.Vec3d(*rotation_ops_b[0].Get(Usd.TimeCode.Default()))
    quat_a = (
        Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation_a[0])
        * Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation_a[1])
        * Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation_a[2])
    )
    quat_b = (
        Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation_b[0])
        * Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation_b[1])
        * Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation_b[2])
    )
    blended_quat = Gf.Slerp(weight, quat_a.GetQuat(), quat_b.GetQuat())
    blended_rotation = Gf.Rotation(blended_quat)
    return blended_rotation


def compute_union_size(size1: Gf.Size2, size2: Gf.Size2) -> Gf.Size2:
    """Compute the union size of two Size2 objects."""
    if size1 == Gf.Size2() and size2 == Gf.Size2():
        return Gf.Size2()
    elif size1 == Gf.Size2():
        return Gf.Size2(size2)
    elif size2 == Gf.Size2():
        return Gf.Size2(size1)
    max_width = max(size1[0], size2[0])
    max_height = max(size1[1], size2[1])
    return Gf.Size2(max_width, max_height)


def calculate_combined_size(sizes: List[Gf.Size3]) -> Gf.Size3:
    """
    Calculate the combined size from a list of Gf.Size3 objects.

    Args:
        sizes (List[Gf.Size3]): A list of Gf.Size3 objects.

    Returns:
        Gf.Size3: The combined size.
    """
    if not sizes:
        return Gf.Size3(0, 0, 0)
    combined_size = Gf.Size3(sizes[0])
    for size in sizes[1:]:
        combined_size[0] += size[0]
        combined_size[1] += size[1]
        combined_size[2] += size[2]
    return combined_size


def get_prim_transform(stage: Usd.Stage, prim_path: str) -> Gf.Matrix4d:
    """
    Get the local transformation matrix for a prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.

    Returns:
        Gf.Matrix4d: The local transformation matrix of the prim.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    transform_matrix = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    return transform_matrix


def set_prim_transform(stage: Usd.Stage, prim_path: str, transform: Gf.Transform) -> None:
    """Set the local transform for a prim using a Gf.Transform.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.
        transform (Gf.Transform): The transform to set on the prim.

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    translation = transform.GetTranslation()
    rotation = transform.GetRotation()
    scale = transform.GetScale()
    pivot = transform.GetPivotPosition()
    xformable.ClearXformOpOrder()
    add_translate_op(xformable).Set(translation)
    add_rotate_xyz_op(xformable).Set(rotation.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis()))
    add_scale_op(xformable).Set(scale)
    add_translate_op(xformable, UsdGeom.XformOp.PrecisionFloat, "pivot").Set(pivot)


def reset_prim_transform(prim: Usd.Prim) -> None:
    """Reset the transform of a prim to identity.

    Args:
        prim (Usd.Prim): The prim to reset the transform for.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    transform_op = xformable.MakeMatrixXform()
    transform_op.Set(Gf.Matrix4d(1))


def get_world_transform(stage: Usd.Stage, prim_path: str) -> Gf.Matrix4d:
    """
    Get the world transformation matrix for a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Returns:
        Gf.Matrix4d: The world transformation matrix.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    local_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    parent_prim = prim.GetParent()
    if parent_prim.IsValid():
        parent_xformable = UsdGeom.Xformable(parent_prim)
        if parent_xformable:
            parent_transform = parent_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            world_transform = local_transform * parent_transform
        else:
            world_transform = local_transform
    else:
        world_transform = local_transform
    return world_transform


def align_prims_to_transform(stage: Usd.Stage, prims: List[Usd.Prim], transform: Gf.Transform) -> None:
    """
    Aligns the given prims to the specified transform.

    Args:
        stage (Usd.Stage): The USD stage.
        prims (List[Usd.Prim]): The list of prims to align.
        transform (Gf.Transform): The target transform to align the prims to.
    """
    for prim in prims:
        if not prim.IsValid():
            print(f"Skipping invalid prim: {prim.GetPath()}")
            continue
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            print(f"Skipping non-transformable prim: {prim.GetPath()}")
            continue
        xformable.ClearXformOpOrder()
        matrix = transform.GetMatrix()
        xformOp = xformable.AddXformOp(UsdGeom.XformOp.TypeTransform)
        xformOp.Set(matrix)


def set_transform_uniform_scale(transform: Gf.Transform, scale: float) -> None:
    """Set the scale component of a Gf.Transform to a uniform value.

    Args:
        transform (Gf.Transform): The transform to modify.
        scale (float): The uniform scale value to set.

    Raises:
        TypeError: If the input transform is not of type Gf.Transform.
        ValueError: If the input scale is not a positive number.
    """
    if not isinstance(transform, Gf.Transform):
        raise TypeError(f"Expected input of type Gf.Transform, got {type(transform)}")
    if scale <= 0:
        raise ValueError(f"Scale must be a positive number, got {scale}")
    current_scale = transform.GetScale()
    scale_ratios = Gf.Vec3d(scale / current_scale[0], scale / current_scale[1], scale / current_scale[2])
    transform.SetScale(Gf.Vec3d(scale, scale, scale))
    pivot_position = transform.GetPivotPosition()
    adjusted_pivot_position = Gf.Vec3d(
        pivot_position[0] * scale_ratios[0], pivot_position[1] * scale_ratios[1], pivot_position[2] * scale_ratios[2]
    )
    transform.SetPivotPosition(adjusted_pivot_position)


def compute_relative_transform(child_transform: Gf.Matrix4d, parent_transform: Gf.Matrix4d) -> Gf.Matrix4d:
    """
    Computes the relative transform of a child object with respect to its parent.

    Args:
        child_transform (Gf.Matrix4d): The transform of the child object in world space.
        parent_transform (Gf.Matrix4d): The transform of the parent object in world space.

    Returns:
        Gf.Matrix4d: The relative transform of the child object with respect to its parent.
    """
    if not child_transform or not parent_transform:
        raise ValueError("Invalid input transforms")
    inverse_parent_transform = parent_transform.GetInverse()
    relative_transform = child_transform * inverse_parent_transform
    return relative_transform


def extract_transform_components(transform: Gf.Transform) -> Dict[str, Any]:
    """Extract the individual components of a Gf.Transform.

    Args:
        transform (Gf.Transform): The input transform to extract components from.

    Returns:
        Dict[str, Any]: A dictionary containing the extracted components.
    """
    components = {}
    translation = transform.GetTranslation()
    components["translation"] = (translation[0], translation[1], translation[2])
    rotation = transform.GetRotation()
    quat = rotation.GetQuat()
    components["rotation"] = (quat.GetReal(), quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2])
    scale = transform.GetScale()
    components["scale"] = (scale[0], scale[1], scale[2])
    pivot_position = transform.GetPivotPosition()
    components["pivot_position"] = (pivot_position[0], pivot_position[1], pivot_position[2])
    pivot_orientation = transform.GetPivotOrientation()
    quat = pivot_orientation.GetQuat()
    components["pivot_orientation"] = (
        quat.GetReal(),
        quat.GetImaginary()[0],
        quat.GetImaginary()[1],
        quat.GetImaginary()[2],
    )
    return components


def get_prim_pivot_position(prim: Usd.Prim) -> Gf.Vec3d:
    """
    Get the pivot position for a given prim.

    Args:
        prim (Usd.Prim): The prim to get the pivot position for.

    Returns:
        Gf.Vec3d: The pivot position of the prim.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    transform_ops = xformable.GetOrderedXformOps()
    pivot_position_op = None
    for op in transform_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            pivot_position_op = op
            break
    if pivot_position_op is None:
        return Gf.Vec3d(0, 0, 0)
    transform = pivot_position_op.Get()
    return transform.ExtractTranslation()


def transform_prim_hierarchy(stage: Usd.Stage, root_prim_path: str, transform: Gf.Matrix4d):
    """
    Recursively apply a transformation matrix to a prim and its descendants.

    Args:
        stage (Usd.Stage): The USD stage.
        root_prim_path (str): The path to the root prim of the hierarchy to transform.
        transform (Gf.Matrix4d): The transformation matrix to apply.
    """
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid():
        raise ValueError(f"Prim at path {root_prim_path} does not exist.")
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsActive():
            continue
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            xformable.AddTransformOp().Set(transform)


def mirror_prim_transform(stage: Usd.Stage, prim_path: str, axis: str = "X") -> None:
    """Mirror the transformation of a prim along a specified axis.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to mirror.
        axis (str): The axis to mirror along. Must be one of "X", "Y", or "Z". Defaults to "X".

    Raises:
        ValueError: If the prim does not exist or is not transformable, or if an invalid axis is specified.
    """
    if axis not in ["X", "Y", "Z"]:
        raise ValueError(f"Invalid axis '{axis}'. Must be one of 'X', 'Y', or 'Z'.")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    transform_matrix = xformable.GetLocalTransformation()
    transform = Gf.Transform()
    transform.SetMatrix(transform_matrix)
    translation = transform.GetTranslation()
    if axis == "X":
        translation[0] *= -1
    elif axis == "Y":
        translation[1] *= -1
    else:
        translation[2] *= -1
    transform.SetTranslation(translation)
    rotation = transform.GetRotation()
    rotation_quat = rotation.GetQuat()
    rotation_quat.SetReal(-rotation_quat.GetReal())
    if axis == "X":
        rotation_quat.SetImaginary(
            Gf.Vec3d(-rotation_quat.GetImaginary()[0], rotation_quat.GetImaginary()[1], rotation_quat.GetImaginary()[2])
        )
    elif axis == "Y":
        rotation_quat.SetImaginary(
            Gf.Vec3d(rotation_quat.GetImaginary()[0], -rotation_quat.GetImaginary()[1], rotation_quat.GetImaginary()[2])
        )
    else:
        rotation_quat.SetImaginary(
            Gf.Vec3d(rotation_quat.GetImaginary()[0], rotation_quat.GetImaginary()[1], -rotation_quat.GetImaginary()[2])
        )
    rotation = Gf.Rotation(rotation_quat)
    transform.SetRotation(rotation)
    scale = transform.GetScale()
    if axis == "X":
        scale[0] *= -1
    elif axis == "Y":
        scale[1] *= -1
    else:
        scale[2] *= -1
    transform.SetScale(scale)
    xform_ops = xformable.GetOrderedXformOps()
    for op in xform_ops:
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate:
            op.Set(transform.GetTranslation())
        elif op_type == UsdGeom.XformOp.TypeRotateXYZ:
            op.Set(transform.GetRotation().Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)))
        elif op_type == UsdGeom.XformOp.TypeScale:
            op.Set(transform.GetScale())


def transform_prims_by_matrix(stage: Usd.Stage, prim_paths: List[str], matrix: Gf.Matrix4d) -> None:
    """Transform multiple prims by a given matrix.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths to transform.
        matrix (Gf.Matrix4d): The transformation matrix.

    Raises:
        ValueError: If any prim path is invalid or not transformable.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        local_xform = xformable.GetLocalTransformation()
        new_xform = local_xform * matrix
        xform_op = xformable.AddTransformOp()
        xform_op.Set(new_xform)


def project_prim_onto_plane(
    stage: Usd.Stage, prim_path: str, plane_normal: Gf.Vec3d, plane_point: Gf.Vec3d
) -> Gf.Vec3d:
    """Projects a prim onto a plane defined by a normal and a point.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to project.
        plane_normal (Gf.Vec3d): The normal vector of the plane.
        plane_point (Gf.Vec3d): A point on the plane.

    Returns:
        Gf.Vec3d: The projected point on the plane.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim not found at path: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    transform_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    prim_position = transform_matrix.ExtractTranslation()
    prim_to_plane = prim_position - plane_point
    projected_vec = prim_to_plane.GetProjection(plane_normal)
    projected_point = prim_position - projected_vec
    return projected_point


def snap_prims_to_grid(stage: Usd.Stage, grid_size: Gf.Vec2d) -> None:
    """Snap the XZ translation of all prims on the stage to align with a 2D grid.

    Args:
        stage (Usd.Stage): The USD stage to operate on.
        grid_size (Gf.Vec2d): The size of the grid cells in the XZ plane.

    Raises:
        ValueError: If the grid size is not positive in both dimensions.
    """
    if grid_size[0] <= 0 or grid_size[1] <= 0:
        raise ValueError("Grid size must be positive in both dimensions.")
    for prim in stage.Traverse():
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            continue
        translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        if not translate_ops:
            continue
        translate_op = translate_ops[0]
        translation = translate_op.Get(Usd.TimeCode.Default())
        if translation is None:
            continue
        snapped_x = round(translation[0] / grid_size[0]) * grid_size[0]
        snapped_z = round(translation[2] / grid_size[1]) * grid_size[1]
        if snapped_x != translation[0] or snapped_z != translation[2]:
            snapped_translation = Gf.Vec3d(snapped_x, translation[1], snapped_z)
            translate_op.Set(snapped_translation)


def distribute_prims_along_axis(stage: Usd.Stage, prim_paths: List[str], axis: Gf.Vec2d, spacing: float) -> None:
    """Distribute prims along a specified 2D axis with equal spacing.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths to distribute.
        axis (Gf.Vec2d): The normalized 2D axis vector along which to distribute the prims.
        spacing (float): The spacing distance between each prim.

    Raises:
        ValueError: If the provided axis is not normalized.
    """
    if not Gf.IsClose(axis.GetLength(), 1.0, 1e-06):
        raise ValueError("The provided axis must be normalized.")
    total_distance = spacing * (len(prim_paths) - 1)
    start_pos = -axis * total_distance * 0.5
    for i, prim_path in enumerate(prim_paths):
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid() and prim.IsA(UsdGeom.Xformable):
            pos = start_pos + axis * spacing * i
            UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(pos[0], 0, pos[1]))


def calculate_prims_bounding_box(stage: Usd.Stage, prim_paths: List[str]) -> Tuple[Gf.Vec3d, Gf.Vec3d]:
    """Calculate the bounding box of a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The list of prim paths.

    Returns:
        Tuple[Gf.Vec3d, Gf.Vec3d]: The minimum and maximum points of the bounding box.

    Raises:
        ValueError: If any of the prim paths are invalid or if no valid bounding box is found.
    """
    min_point = Gf.Vec3d(float("inf"), float("inf"), float("inf"))
    max_point = Gf.Vec3d(float("-inf"), float("-inf"), float("-inf"))
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")
        bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
        if bbox.GetRange().IsEmpty():
            continue
        min_point = Gf.Vec3d(
            min(min_point[0], bbox.GetRange().GetMin()[0]),
            min(min_point[1], bbox.GetRange().GetMin()[1]),
            min(min_point[2], bbox.GetRange().GetMin()[2]),
        )
        max_point = Gf.Vec3d(
            max(max_point[0], bbox.GetRange().GetMax()[0]),
            max(max_point[1], bbox.GetRange().GetMax()[1]),
            max(max_point[2], bbox.GetRange().GetMax()[2]),
        )
    if min_point == Gf.Vec3d(float("inf"), float("inf"), float("inf")) or max_point == Gf.Vec3d(
        float("-inf"), float("-inf"), float("-inf")
    ):
        raise ValueError("No valid bounding box found for the given prim paths.")
    return (min_point, max_point)


def align_prims_to_plane(stage: Usd.Stage, prim_paths: List[str], plane_normal: Gf.Vec3d) -> None:
    """Align a list of prims to a given plane defined by its normal vector.

    The prims are rotated such that their local Y-axis aligns with the plane's normal.

    Parameters:
        stage (Usd.Stage): The USD stage containing the prims.
        prim_paths (List[str]): A list of paths to the prims to be aligned.
        plane_normal (Gf.Vec3d): The normal vector of the plane.

    Raises:
        ValueError: If any of the prim paths are invalid or not transformable.
    """
    plane_normal = plane_normal.GetNormalized()
    up_vector = Gf.Vec3d.YAxis()
    rotation = Gf.Rotation(up_vector, plane_normal)
    rotation_angles = rotation.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        rotate_op = add_rotate_xyz_op(xformable)
        rotate_op.Set(rotation_angles)


def transform_prim_to_origin(stage: Usd.Stage, prim_path: str):
    """Transform a prim to the origin.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to transform.

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    translate_ops = xformable.GetOrderedXformOps()
    translate_op = None
    for op in translate_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op:
        current_translation = translate_op.Get()
    else:
        current_translation = Gf.Vec3d(0, 0, 0)
    origin_translation = -current_translation
    if translate_op:
        translate_op.Set(origin_translation)
    else:
        add_translate_op(xformable).Set(origin_translation)


def normalize_prim_scale(prim: Usd.Prim, eps: float = 1e-05) -> Gf.Vec3d:
    """Normalize the scale of a prim.

    Args:
        prim (Usd.Prim): The prim to normalize the scale of.
        eps (float, optional): Epsilon value for normalization. Defaults to 1e-5.

    Returns:
        Gf.Vec3d: The original scale of the prim before normalization.
    """
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim.GetPath()} is not transformable.")
    ops = xformable.GetOrderedXformOps()
    scale_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
    if not scale_op:
        return Gf.Vec3d(1.0)
    orig_scale = scale_op.Get()
    length = orig_scale.GetLength()
    if length < eps:
        new_scale = Gf.Vec3d(1.0)
    else:
        new_scale = orig_scale / length
    scale_op.Set(new_scale)
    return orig_scale


def compute_orthogonal_complement_of_prim_translation(stage: Usd.Stage, prim_path: str) -> Gf.Vec2f:
    """
    Computes the orthogonal complement of the translation of a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Returns:
        Gf.Vec2f: The orthogonal complement of the prim's translation in 2D.

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if not translate_ops:
        translation = Gf.Vec3d(0.0, 0.0, 0.0)
    else:
        translate_op = translate_ops[0]
        translation = translate_op.Get(Usd.TimeCode.Default())
        if translation is None:
            translation = Gf.Vec3d(0.0, 0.0, 0.0)
    translation_vec2f = Gf.Vec2f(translation[0], translation[1])
    return translation_vec2f.GetComplement(Gf.Vec2f.XAxis())


def normalize_all_prims_translations(stage: Usd.Stage) -> List[str]:
    """Normalize the translation of all prims on the stage.

    This function traverses all prims on the given stage and normalizes
    their translation attribute using the Gf.Vec2f.GetNormalized() method.
    It returns a list of the prim paths whose translation was normalized.
    """
    normalized_prims = []
    for prim in stage.TraverseAll():
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            continue
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op:
            old_translation = translate_op.Get(Usd.TimeCode.Default())
            if old_translation is None:
                old_translation = Gf.Vec3d(0, 0, 0)
            new_translation = Gf.Vec2f(old_translation[0], old_translation[1]).GetNormalized(1e-05)
            translate_op.Set(Gf.Vec3d(new_translation[0], new_translation[1], old_translation[2]))
            normalized_prims.append(str(prim.GetPath()))
    return normalized_prims


def get_prim_normalized_translation(stage: Usd.Stage, prim_path: str) -> Gf.Vec2f:
    """
    Get the normalized 2D translation for a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Returns:
        Gf.Vec2f: The normalized 2D translation vector.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    translate_ops = xformable.GetOrderedXformOps()
    translation = Gf.Vec3f(0.0, 0.0, 0.0)
    for op in translate_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translation = op.Get()
            break
    translation_2d = Gf.Vec2f(translation[0], translation[1])
    normalized_translation = translation_2d.GetNormalized()
    return normalized_translation


def project_prim_translation_to_vector(stage: Usd.Stage, prim_path: str, vector: Gf.Vec2f) -> Gf.Vec2f:
    """Project the translation of a prim onto a 2D vector.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path of the prim to get the translation from.
        vector (Gf.Vec2f): The 2D vector to project the translation onto.

    Returns:
        Gf.Vec2f: The projection of the prim's translation onto the vector.

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if not translate_ops:
        return Gf.Vec2f(0.0)
    translate_3d = translate_ops[0].Get()
    translate_2d = Gf.Vec2f(translate_3d[0], translate_3d[1])
    projection = translate_2d.GetProjection(vector)
    return projection


def scale_prim_translation(stage: Usd.Stage, prim_path: str, scale: Gf.Vec2f) -> None:
    """Scale the translation of a prim by a Vec2f scale factor."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable")
    translate_ops = xformable.GetOrderedXformOps()
    translate_op = None
    for op in translate_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if not translate_op:
        raise ValueError(f"Prim at path {prim_path} does not have a translate op")
    old_translation = translate_op.Get()
    if not old_translation:
        old_translation = Gf.Vec3d(0, 0, 0)
    new_translation = Gf.Vec3d(old_translation[0] * scale[0], old_translation[1] * scale[1], old_translation[2])
    translate_op.Set(new_translation)


def transform_prim_translation_to_unit_vector(stage: Usd.Stage, prim_path: str) -> Gf.Vec2f:
    """
    Transform the translation of a prim to a unit vector in the XY plane.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.

    Returns:
        Gf.Vec2f: The unit vector representing the direction of the prim's translation in the XY plane.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if not translate_ops:
        return Gf.Vec2f(0.0, 0.0)
    translation_op = translate_ops[0]
    translation = translation_op.Get()
    if translation is None:
        return Gf.Vec2f(0.0, 0.0)
    vec2f = Gf.Vec2f(translation[0], translation[1])
    unit_vector = vec2f.GetNormalized()
    return unit_vector


def align_prims_to_vector_projection(stage: Usd.Stage, prim_paths: List[str], target_vector: Gf.Vec2f) -> None:
    """
    Align the given prims to the projection of the target vector onto the XZ plane.

    Args:
        stage (Usd.Stage): The USD stage containing the prims.
        prim_paths (List[str]): The paths of the prims to align.
        target_vector (Gf.Vec2f): The target vector to project onto the XZ plane.

    Raises:
        ValueError: If any of the prim paths are invalid or not transformable.
    """
    target_vector.Normalize()
    projected_vector = Gf.Vec3d(target_vector[0], 0, target_vector[1])
    align_rotation = Gf.Rotation(Gf.Vec3d(0, 1, 0), projected_vector)
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        xformable.AddRotateZOp().Set(align_rotation.GetAngle())


def apply_transformations(prim: Usd.Prim, translation: Gf.Vec3f, rotation: Gf.Vec3f, scale: Gf.Vec3f) -> None:
    """Apply translation, rotation, and scale transformations to a given prim.

    Args:
        prim (Usd.Prim): The prim to apply transformations to.
        translation (Gf.Vec3f): The translation vector.
        rotation (Gf.Vec3f): The rotation angles in degrees (X, Y, Z).
        scale (Gf.Vec3f): The scale factors (X, Y, Z).
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable")
    add_translate_op(xformable).Set(translation)
    rotation_rad = Gf.Vec3f(
        Gf.DegreesToRadians(rotation[0]), Gf.DegreesToRadians(rotation[1]), Gf.DegreesToRadians(rotation[2])
    )
    add_rotate_xyz_op(xformable).Set(rotation_rad)
    add_scale_op(xformable).Set(scale)


def compute_scene_bounds(stage: Usd.Stage) -> Gf.Range3d:
    """Compute the bounds of a USD scene.

    Args:
        stage (Usd.Stage): The USD stage to compute bounds for.

    Returns:
        Gf.Range3d: The computed bounds of the scene.
    """
    bounds = Gf.Range3d()
    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Boundable):
            boundable = UsdGeom.Boundable(prim)
            extent_attr = boundable.GetExtentAttr()
            extent = extent_attr.Get(Usd.TimeCode.Default())
            if not extent:
                continue
            prim_range = Gf.Range3d(Gf.Vec3d(extent[0]), Gf.Vec3d(extent[1]))
            bounds.UnionWith(prim_range)
    return bounds


def transfer_materials_between_prims(source_prim: Usd.Prim, dest_prim: Usd.Prim):
    """Transfer material bindings from one prim to another.

    This function copies the material bindings from the source prim to the
    destination prim. If the destination prim already has material bindings,
    they will be overwritten.

    Args:
        source_prim (Usd.Prim): The prim to copy material bindings from.
        dest_prim (Usd.Prim): The prim to copy material bindings to.

    Raises:
        ValueError: If either the source or destination prim is invalid.
    """
    if not source_prim.IsValid():
        raise ValueError("Source prim is invalid.")
    if not dest_prim.IsValid():
        raise ValueError("Destination prim is invalid.")
    source_binding_api = UsdShade.MaterialBindingAPI(source_prim)
    if source_binding_api.GetDirectBindingRel().GetTargets():
        dest_binding_api = UsdShade.MaterialBindingAPI(dest_prim)
        dest_binding_api.UnbindAllBindings()
        direct_binding_rel = source_binding_api.GetDirectBindingRel()
        for target in direct_binding_rel.GetTargets():
            material_prim = stage.GetPrimAtPath(target)
            if UsdShade.Material(material_prim):
                dest_binding_api.Bind(UsdShade.Material(material_prim))


def distribute_prims_evenly(prims: List[Usd.Prim], center: Gf.Vec2f, radius: float) -> None:
    """
    Distributes the given list of prims evenly in a circle around the center.

    Args:
        prims (List[Usd.Prim]): List of prims to distribute.
        center (Gf.Vec2f): Center point of the distribution circle.
        radius (float): Radius of the distribution circle.
    """
    num_prims = len(prims)
    if num_prims < 1:
        return
    angle_increment = 2 * math.pi / num_prims
    for i, prim in enumerate(prims):
        angle = i * angle_increment
        offset = Gf.Vec2f(math.cos(angle), math.sin(angle)) * radius
        xformable = UsdGeom.Xformable(prim)
        add_translate_op(xformable).Set(Gf.Vec3f(center[0] + offset[0], 0, center[1] + offset[1]))


def align_prim_to_axis(stage: Usd.Stage, prim_path: str, axis: Gf.Vec2i) -> None:
    """Align a prim to a given axis.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to align.
        axis (Gf.Vec2i): The axis to align the prim to.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    rotation_ops = xformable.GetOrderedXformOps()
    rotation_op = None
    for op in rotation_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            rotation_op = op
            break
    if not rotation_op:
        rotation_op = xformable.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ)
    current_rotation = rotation_op.Get()
    if not current_rotation:
        current_rotation = Gf.Vec3f(0.0)
    target_rotation = Gf.Vec3f(0.0)
    if axis == Gf.Vec2i(1, 0):
        target_rotation = Gf.Vec3f(0.0, 0.0, 90.0)
    elif axis == Gf.Vec2i(0, 1):
        target_rotation = Gf.Vec3f(0.0, 0.0, 0.0)
    else:
        raise ValueError(f"Invalid axis: {axis}")
    rotation_op.Set(target_rotation)


def create_unit_vector_prim(stage: Usd.Stage, prim_path: str, vector: Gf.Vec2i) -> Usd.Prim:
    """Create a prim with a unit vector attribute.

    Args:
        stage (Usd.Stage): The stage to create the prim on.
        prim_path (str): The path to create the prim at.
        vector (Gf.Vec2i): The unit vector to set as an attribute.

    Returns:
        Usd.Prim: The created prim.
    """
    if vector not in [Gf.Vec2i.XAxis(), Gf.Vec2i.YAxis()]:
        raise ValueError(f"Invalid unit vector: {vector}")
    prim = stage.DefinePrim(prim_path, "Xform")
    if not prim.IsValid():
        raise RuntimeError(f"Failed to create prim at path: {prim_path}")
    attr = prim.CreateAttribute("unitVector", Sdf.ValueTypeNames.Int2)
    if not attr.Set(vector):
        raise RuntimeError(f"Failed to set unit vector attribute on prim: {prim.GetPath()}")
    return prim


def scale_prim_to_match_unit_vector(stage: Usd.Stage, prim_path: str, unit_vector: Gf.Vec2i) -> bool:
    """Scale a prim to match the length of a unit vector.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to scale.
        unit_vector (Gf.Vec2i): The unit vector to match the scale.

    Returns:
        bool: True if the prim was successfully scaled, False otherwise.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Error: Prim at path {prim_path} does not exist.")
        return False
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        print(f"Error: Prim at path {prim_path} is not transformable.")
        return False
    length = Gf.GetLength(Gf.Vec2f(unit_vector))
    add_scale_op(xformable).Set(Gf.Vec3f(length))
    return True


def calculate_dot_product_between_prims(stage: Usd.Stage, prim_a_path: str, prim_b_path: str) -> int:
    """
    Calculate the dot product between the translation vectors of two prims.

    Args:
        stage (Usd.Stage): The USD stage containing the prims.
        prim_a_path (str): The path to the first prim.
        prim_b_path (str): The path to the second prim.

    Returns:
        int: The dot product between the translation vectors of the two prims.

    Raises:
        ValueError: If either prim is not valid or not transformable.
    """
    prim_a = stage.GetPrimAtPath(prim_a_path)
    prim_b = stage.GetPrimAtPath(prim_b_path)
    if not prim_a.IsValid() or not prim_b.IsValid():
        raise ValueError(f"One or both prims ({prim_a_path}, {prim_b_path}) are not valid.")
    xformable_a = UsdGeom.Xformable(prim_a)
    xformable_b = UsdGeom.Xformable(prim_b)
    if not xformable_a or not xformable_b:
        raise ValueError(f"One or both prims ({prim_a_path}, {prim_b_path}) are not transformable.")
    translation_a = Gf.Vec3f(0)
    translation_b = Gf.Vec3f(0)
    for op in xformable_a.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translation_a = op.Get()
            break
    for op in xformable_b.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translation_b = op.Get()
            break
    vec2i_a = Gf.Vec2i(int(translation_a[0]), int(translation_a[1]))
    vec2i_b = Gf.Vec2i(int(translation_b[0]), int(translation_b[1]))
    return vec2i_a[0] * vec2i_b[0] + vec2i_a[1] * vec2i_b[1]


def compute_orthonormal_basis(v1: Gf.Vec3h, v2: Gf.Vec3h, eps: float = 1e-05) -> Tuple[Gf.Vec3h, Gf.Vec3h, Gf.Vec3h]:
    """
    Computes an orthonormal basis given two non-zero, non-parallel vectors.

    Args:
        v1 (Gf.Vec3h): The first vector.
        v2 (Gf.Vec3h): The second vector.
        eps (float, optional): The epsilon value for checking vector length. Defaults to 1e-5.

    Returns:
        Tuple[Gf.Vec3h, Gf.Vec3h, Gf.Vec3h]: The computed orthonormal basis (v1, v2, v3).

    Raises:
        ValueError: If v1 or v2 is a zero vector or if v1 and v2 are parallel.
    """
    if v1.GetLength() < eps:
        raise ValueError("v1 is a zero vector")
    if v2.GetLength() < eps:
        raise ValueError("v2 is a zero vector")
    v1 = v1.GetNormalized(eps)
    v3 = Gf.Cross(v1, v2)
    if v3.GetLength() < eps:
        raise ValueError("v1 and v2 are parallel")
    v3 = v3.GetNormalized(eps)
    v2 = Gf.Cross(v3, v1)
    return (v1, v2, v3)


def project_point_onto_prim(stage: Usd.Stage, prim_path: str, point: Gf.Vec3d) -> Gf.Vec3d:
    """Project a point onto the surface of a prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.
        point (Gf.Vec3d): The point to project.

    Returns:
        Gf.Vec3d: The projected point on the prim surface.

    Raises:
        ValueError: If the prim is not valid or does not have a defined volume.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")
    bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
    if not bbox.GetRange().IsEmpty():
        center = bbox.ComputeCentroid()
        extents = bbox.GetRange().GetSize() * 0.5
        direction = point - center
        distances = Gf.Vec3d(
            direction[0] / extents[0] if extents[0] != 0 else 0,
            direction[1] / extents[1] if extents[1] != 0 else 0,
            direction[2] / extents[2] if extents[2] != 0 else 0,
        )
        max_distance = max(abs(distances[0]), abs(distances[1]), abs(distances[2]))
        clamped_distances = (
            Gf.Vec3d(
                extents[0] if distances[0] > 0 else -extents[0],
                extents[1] if distances[1] > 0 else -extents[1],
                extents[2] if distances[2] > 0 else -extents[2],
            )
            if max_distance > 1.0
            else distances * extents
        )
        projected_point = center + clamped_distances
        return projected_point
    else:
        raise ValueError(f"Prim at path {prim_path} does not have a defined volume.")


def calculate_prim_distance(stage: Usd.Stage, prim_path1: str, prim_path2: str) -> float:
    """
    Calculate the Euclidean distance between two prims in a USD stage.

    Args:
        stage (Usd.Stage): The USD stage containing the prims.
        prim_path1 (str): The path to the first prim.
        prim_path2 (str): The path to the second prim.

    Returns:
        float: The Euclidean distance between the two prims.

    Raises:
        ValueError: If either prim path is invalid or not transformable.
    """
    prim1 = stage.GetPrimAtPath(prim_path1)
    prim2 = stage.GetPrimAtPath(prim_path2)
    if not prim1.IsValid() or not prim2.IsValid():
        raise ValueError(f"One or both prim paths are invalid: {prim_path1}, {prim_path2}")
    xformable1 = UsdGeom.Xformable(prim1)
    xformable2 = UsdGeom.Xformable(prim2)
    if not xformable1 or not xformable2:
        raise ValueError(f"One or both prims are not transformable: {prim_path1}, {prim_path2}")
    matrix1 = xformable1.GetLocalTransformation()
    matrix2 = xformable2.GetLocalTransformation()
    translation1 = matrix1.ExtractTranslation()
    translation2 = matrix2.ExtractTranslation()
    distance = (translation2 - translation1).GetLength()
    return distance


def compute_prim_bounding_box(prim: Usd.Prim, time_code: Usd.TimeCode = Usd.TimeCode.Default()) -> Gf.Range3d:
    """Compute the bounding box for a prim at a given time code.

    Args:
        prim (Usd.Prim): The prim to compute the bounding box for.
        time_code (Usd.TimeCode, optional): The time code to compute the bounding box at. Defaults to Usd.TimeCode.Default().

    Returns:
        Gf.Range3d: The computed bounding box.

    Raises:
        ValueError: If the prim is not valid or does not have a defined bounding box.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim '{prim.GetPath()}' is not valid.")
    bbox_cache = UsdGeom.BBoxCache(time_code, includedPurposes=[UsdGeom.Tokens.default_])
    prim_bbox = bbox_cache.ComputeWorldBound(prim)
    if prim_bbox.GetRange().IsEmpty():
        raise ValueError(f"Prim '{prim.GetPath()}' does not have a defined bounding box.")
    return prim_bbox.ComputeAlignedRange()


def scale_prim_along_axis(stage: Usd.Stage, prim_path: str, axis: Gf.Vec3d, scale_factor: float) -> None:
    """Scale a prim along a specific axis.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to be scaled.
        axis (Gf.Vec3d): The axis along which to scale the prim.
        scale_factor (float): The scale factor to apply.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    axis.Normalize()
    scale_matrix = Gf.Matrix4d(
        axis[0] * scale_factor, 0, 0, 0, 0, axis[1] * scale_factor, 0, 0, 0, 0, axis[2] * scale_factor, 0, 0, 0, 0, 1
    )
    xform_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTransform)
    xform_op.Set(scale_matrix)


def translate_prim_along_vector(prim: Usd.Prim, vector: Gf.Vec3d, time_code: Usd.TimeCode = Usd.TimeCode.Default()):
    """Translate a prim along a vector.

    Args:
        prim (Usd.Prim): The prim to translate.
        vector (Gf.Vec3d): The vector to translate along.
        time_code (Usd.TimeCode, optional): The time code to set the translation at. Defaults to Default time code.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim} is not transformable.")
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_op = translate_ops[0]
        old_translation = translate_op.Get(time_code)
        if old_translation is None:
            old_translation = Gf.Vec3d(0, 0, 0)
    else:
        old_translation = Gf.Vec3d(0, 0, 0)
        translate_op = add_translate_op(xformable)
    new_translation = old_translation + vector
    translate_op.Set(new_translation, time_code)


def find_prims_within_radius(stage: Usd.Stage, center: Gf.Vec3d, radius: float) -> List[Usd.Prim]:
    """
    Find all prims within a given radius from a center point.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        center (Gf.Vec3d): The center point to measure distance from.
        radius (float): The maximum distance from the center point to include prims.

    Returns:
        List[Usd.Prim]: A list of prims within the specified radius from the center point.
    """
    prims_within_radius = []
    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Xformable):
            xformable = UsdGeom.Xformable(prim)
            translate_op = xformable.GetOrderedXformOps()[0]
            if translate_op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                prim_translation = translate_op.Get()
                prim_position = Gf.Vec3d(prim_translation)
                distance = (prim_position - center).GetLength()
                if distance <= radius:
                    prims_within_radius.append(prim)
    return prims_within_radius


def calculate_prim_normal(stage: Usd.Stage, prim_path: str) -> Gf.Vec3d:
    """Calculate the normal vector for a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    if not prim.IsA(UsdGeom.Gprim):
        raise ValueError(f"Prim at path {prim_path} is not a UsdGeomGprim.")
    if prim.IsA(UsdGeom.Sphere):
        normal = Gf.Vec3d(0, 1, 0)
    elif prim.IsA(UsdGeom.Cube):
        normal = Gf.Vec3d(0, 1, 0)
    elif prim.IsA(UsdGeom.Cylinder):
        normal = Gf.Vec3d(0, 1, 0)
    elif prim.IsA(UsdGeom.Cone):
        normal = Gf.Vec3d(0, 1, 0)
    elif prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        if mesh.GetNormalsAttr().HasAuthoredValue():
            normals = mesh.GetNormalsAttr().Get()
            if normals:
                normal = Gf.Vec3d(normals[0])
            else:
                raise ValueError(f"Mesh prim at path {prim_path} has an empty normals array.")
        else:
            face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
            points = mesh.GetPointsAttr().Get()
            if face_vertex_indices and points:
                face = face_vertex_indices[:3]
                p0 = Gf.Vec3d(points[face[0]])
                p1 = Gf.Vec3d(points[face[1]])
                p2 = Gf.Vec3d(points[face[2]])
                edge1 = p1 - p0
                edge2 = p2 - p0
                normal = Gf.Vec3d(Gf.Cross(edge1, edge2))
            else:
                raise ValueError(f"Mesh prim at path {prim_path} does not have face vertex indices or points.")
    else:
        raise ValueError(f"Unsupported prim type: {prim.GetTypeName()}")
    normal.Normalize()
    return normal


def distribute_prims_along_curve(
    curve_points: List[Gf.Vec3f], prim_paths: List[Sdf.Path], up_vector: Gf.Vec3f = Gf.Vec3f(0, 1, 0)
) -> Usd.Stage:
    """Distribute prims along a curve defined by a list of points.

    Args:
        curve_points (List[Gf.Vec3f]): A list of points defining the curve.
        prim_paths (List[Sdf.Path]): A list of prim paths to distribute along the curve.
        up_vector (Gf.Vec3f, optional): The up vector for orienting the prims. Defaults to Gf.Vec3f(0, 1, 0).

    Returns:
        Usd.Stage: The stage containing the distributed prims.
    """
    if len(curve_points) < 2:
        raise ValueError("At least two points are required to define a curve.")
    if len(prim_paths) < 1:
        raise ValueError("At least one prim path is required.")
    stage = Usd.Stage.CreateInMemory()
    total_length = 0.0
    for i in range(len(curve_points) - 1):
        total_length += (curve_points[i + 1] - curve_points[i]).GetLength()
    distance = 0.0
    segment_start = curve_points[0]
    for i in range(len(curve_points) - 1):
        segment_end = curve_points[i + 1]
        segment_length = (segment_end - segment_start).GetLength()
        while distance / total_length < (i + 1) / (len(curve_points) - 1):
            t = (distance - total_length * i / (len(curve_points) - 1)) / segment_length
            position = segment_start + (segment_end - segment_start) * t
            if len(prim_paths) > 0:
                prim_path = prim_paths.pop(0)
                prim = stage.DefinePrim(prim_path)
                add_translate_op(UsdGeom.Xform(prim)).Set(position)
                if i < len(curve_points) - 2:
                    tangent = curve_points[i + 1] - curve_points[i]
                    normal = Gf.Cross(tangent, up_vector).GetNormalized()
                    add_rotate_xyz_op(UsdGeom.Xform(prim)).Set(
                        Gf.Rotation(Gf.Vec3d(normal), Gf.Vec3d(tangent)).Decompose(
                            Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)
                        )
                    )
            distance += total_length / (len(prim_paths) + 1)
        segment_start = segment_end
    return stage


def create_orthogonal_basis(v1: Gf.Vec3f, v2: Gf.Vec3f, eps: float = 1e-06) -> Tuple[Gf.Vec3f, Gf.Vec3f, Gf.Vec3f]:
    """Create an orthogonal basis from two input vectors.

    This function takes two input vectors v1 and v2, and creates an orthogonal basis
    consisting of three mutually orthogonal unit vectors. The first output vector is
    the normalized version of v1. The second output vector is orthogonal to the first,
    and the third output vector is the cross product of the first two.

    Args:
        v1 (Gf.Vec3f): The first input vector.
        v2 (Gf.Vec3f): The second input vector.
        eps (float): Epsilon value for normalization and orthogonalization.

    Returns:
        Tuple[Gf.Vec3f, Gf.Vec3f, Gf.Vec3f]: A tuple of three mutually orthogonal unit vectors.
    """
    v1_length = v1.Normalize(eps)
    if v1_length < eps:
        return (Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 0, 1))
    v2 = v2 - v1 * Gf.Dot(v1, v2)
    v2_length = v2.Normalize(eps)
    if v2_length < eps:
        v2 = v1.GetOrthogonal()
    v3 = Gf.Cross(v1, v2)
    return (v1, v2, v3)


def find_orthogonal_complement_vectors(v: Gf.Vec3f, eps: float = 1e-06) -> Tuple[Gf.Vec3f, Gf.Vec3f]:
    """
    Find two orthogonal complement vectors to the given vector.

    The returned vectors v1 and v2 are unit vectors such that v, v1, and v2 are
    mutually orthogonal. If the length of v is smaller than eps, then v1 and v2
    will have magnitude 1/eps.

    Parameters:
        v (Gf.Vec3f): The input vector.
        eps (float): The epsilon value for checking vector length. Default is 1e-6.

    Returns:
        Tuple[Gf.Vec3f, Gf.Vec3f]: Two orthogonal complement vectors to v.
    """
    length = v.GetLength()
    if length < eps:
        v = v * (1.0 / eps)
    else:
        v = v * (1.0 / length)
    if abs(v[0]) < abs(v[1]):
        if abs(v[0]) < abs(v[2]):
            v1 = Gf.Vec3f(0, -v[2], v[1])
        else:
            v1 = Gf.Vec3f(-v[1], v[0], 0)
    elif abs(v[1]) < abs(v[2]):
        v1 = Gf.Vec3f(v[2], 0, -v[0])
    else:
        v1 = Gf.Vec3f(-v[1], v[0], 0)
    v1 = v1.GetNormalized(eps)
    v2 = Gf.Cross(v, v1).GetNormalized(eps)
    return (v1, v2)


def normalize_prim_vectors(stage: Usd.Stage, prim_path: str, eps: float = 1e-05):
    """Normalize all vector-valued attributes on a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        eps (float, optional): Epsilon value for normalization. Defaults to 1e-5.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path: {prim_path}")
    for attr in prim.GetAttributes():
        if attr.GetTypeName() in (Sdf.ValueTypeNames.Float2, Sdf.ValueTypeNames.Float3, Sdf.ValueTypeNames.Float4):
            vec = attr.Get()
            if vec is not None:
                if len(vec) == 2:
                    vec4f = Gf.Vec4f(vec[0], vec[1], 0.0, 0.0)
                elif len(vec) == 3:
                    vec4f = Gf.Vec4f(vec[0], vec[1], vec[2], 0.0)
                else:
                    vec4f = Gf.Vec4f(vec[0], vec[1], vec[2], vec[3])
                length = vec4f.Normalize(eps)
                if len(vec) == 2:
                    attr.Set(Gf.Vec2f(vec4f[0], vec4f[1]))
                elif len(vec) == 3:
                    attr.Set(Gf.Vec3f(vec4f[0], vec4f[1], vec4f[2]))
                else:
                    attr.Set(vec4f)


def align_prims_to_vector(stage: Usd.Stage, prim_paths: List[str], vector: Gf.Vec4d) -> None:
    """Align a list of prims to a given vector.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths to align.
        vector (Gf.Vec4d): The vector to align the prims to.

    Raises:
        ValueError: If any of the prim paths are invalid or not transformable.
    """
    vector.Normalize()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        rotation_order = xformable.GetXformOpOrderAttr().Get()
        rotation_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                rotation_op = op
                break
        if rotation_op is None:
            rotation_op = add_rotate_xyz_op(xformable, opSuffix="align")
        rotation = rotation_op.Get()
        if rotation is None:
            rotation = Gf.Vec3d(0, 0, 0)
        prim_rotation = (
            Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation[0])
            * Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation[1])
            * Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation[2])
        )
        target_rotation = Gf.Rotation(Gf.Vec3d(0, 1, 0), Gf.Vec3d(vector[0], vector[1], vector[2]))
        new_rotation = prim_rotation * target_rotation
        rotation_op.Set(new_rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)))


def distribute_prims_in_grid(stage: Usd.Stage, prim_name: str, grid_size: Tuple[int, int], grid_spacing: float) -> None:
    """Distribute multiple prims in a grid pattern.

    Args:
        stage (Usd.Stage): The USD stage to create the prims on.
        prim_name (str): The name of the prim to create.
        grid_size (Tuple[int, int]): The number of prims in the grid (rows, columns).
        grid_spacing (float): The spacing between the prims in the grid.
    """
    if grid_size[0] <= 0 or grid_size[1] <= 0:
        raise ValueError("Grid size must be positive.")
    if grid_spacing <= 0:
        raise ValueError("Grid spacing must be positive.")
    parent_xform = UsdGeom.Xform.Define(stage, Sdf.Path("/Grid"))
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            position = Gf.Vec3d(i * grid_spacing, 0, j * grid_spacing)
            prim_path = f"/Grid/Sphere_{i}_{j}"
            prim = stage.DefinePrim(prim_path, prim_name)
            UsdGeom.XformCommonAPI(prim).SetTranslate(position)


def project_prim_to_plane(stage: Usd.Stage, prim_path: str, plane_normal: Gf.Vec3f, plane_point: Gf.Vec3f) -> Gf.Vec3f:
    """Project a prim's pivot point onto a plane defined by a normal and a point on the plane.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to project.
        plane_normal (Gf.Vec3f): The normal vector of the plane.
        plane_point (Gf.Vec3f): A point on the plane.

    Returns:
        Gf.Vec3f: The projected point on the plane.

    Raises:
        ValueError: If the prim is not valid or transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    transform_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pivot_point = Gf.Vec3f(transform_matrix.ExtractTranslation())
    pivot_to_plane = pivot_point - plane_point
    projection = pivot_to_plane.GetProjection(plane_normal)
    projected_point = pivot_point - projection
    return projected_point


def set_prim_orthogonal_frame(prim: Usd.Prim, x_dir: Gf.Vec3f, y_dir: Gf.Vec3f, z_dir: Gf.Vec3f) -> None:
    """Sets the orientation of a prim using an orthogonal frame defined by x, y, z direction vectors.

    The vectors will be orthogonalized and normalized to form a valid frame.
    If the vectors are close to colinear, the function may fail to converge.

    Args:
        prim (Usd.Prim): The prim to set the orientation for.
        x_dir (Gf.Vec3f): The X direction vector.
        y_dir (Gf.Vec3f): The Y direction vector.
        z_dir (Gf.Vec3f): The Z direction vector.

    Raises:
        ValueError: If the prim is not valid or not transformable.
        ArithmeticError: If the vectors fail to orthogonalize due to being colinear.
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError("Prim is not transformable")
    eps = 1e-06
    normalize = True
    converged = Gf.Vec3f.OrthogonalizeBasis(x_dir, y_dir, z_dir, normalize, eps)
    if not converged:
        raise ArithmeticError("Failed to orthogonalize frame vectors")
    rotation = Gf.Matrix3d(x_dir[0], x_dir[1], x_dir[2], y_dir[0], y_dir[1], y_dir[2], z_dir[0], z_dir[1], z_dir[2])
    transform = Gf.Matrix4d()
    transform.SetRotate(rotation)
    xform_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTransform)
    xform_op.Set(transform)


def calculate_prim_projection(stage: Usd.Stage, prim_path: str, projection_vector: Gf.Vec3d) -> Gf.Vec3d:
    """Calculate the projection of a prim's transform onto a vector.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        projection_vector (Gf.Vec3d): The vector to project onto.

    Returns:
        Gf.Vec3d: The projected vector.

    Raises:
        ValueError: If the prim is not valid or transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    local_transform = xformable.GetLocalTransformation()
    translation = local_transform.ExtractTranslation()
    projected_vector = translation.GetProjection(projection_vector)
    return projected_vector


def compute_dot_product(vec1: Gf.Vec3h, vec2: Gf.Vec3h) -> float:
    """
    Compute the dot product of two Vec3h vectors.

    Args:
        vec1 (Gf.Vec3h): The first vector.
        vec2 (Gf.Vec3h): The second vector.

    Returns:
        float: The dot product of the two vectors.
    """
    if not isinstance(vec1, Gf.Vec3h) or not isinstance(vec2, Gf.Vec3h):
        raise TypeError("Input vectors must be of type Gf.Vec3h")
    dot_product = vec1 * vec2
    return float(dot_product)


def compute_cross_product(a: Gf.Vec3h, b: Gf.Vec3h) -> Gf.Vec3h:
    """
    Compute the cross product of two Vec3h vectors.

    Args:
        a (Gf.Vec3h): The first vector.
        b (Gf.Vec3h): The second vector.

    Returns:
        Gf.Vec3h: The cross product of a and b.
    """
    if a.GetLength() == 0 or b.GetLength() == 0:
        return Gf.Vec3h(0, 0, 0)
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return Gf.Vec3h(x, y, z)


def compute_prim_length(prim: Usd.Prim) -> float:
    """Compute the length of a prim based on its bounding box.

    Args:
        prim (Usd.Prim): The input prim.

    Returns:
        float: The length of the prim. Returns 0.0 if the prim is not valid or has no bounding box.
    """
    if not prim.IsValid():
        print(f"Warning: Prim {prim.GetPath()} is not valid.")
        return 0.0
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bound = bbox_cache.ComputeWorldBound(prim)
    if not bound.IsValid():
        print(f"Warning: Prim {prim.GetPath()} has no valid bounding box.")
        return 0.0
    (min_pt, max_pt) = (bound.GetRange().GetMin(), bound.GetRange().GetMax())
    diagonal = max_pt - min_pt
    length = diagonal.GetLength()
    return length


def create_unit_vector_along_axis(axis_index: int) -> Gf.Vec3h:
    """Create a unit vector along the specified axis.

    Args:
        axis_index (int): The index of the axis (0 for X, 1 for Y, 2 for Z).

    Returns:
        Gf.Vec3h: The unit vector along the specified axis.

    Raises:
        ValueError: If the axis_index is not 0, 1, or 2.
    """
    if axis_index == 0:
        return Gf.Vec3h.XAxis()
    elif axis_index == 1:
        return Gf.Vec3h.YAxis()
    elif axis_index == 2:
        return Gf.Vec3h.ZAxis()
    else:
        raise ValueError(f"Invalid axis_index: {axis_index}. Must be 0, 1, or 2.")


def get_orthogonal_complement(vec: Gf.Vec3h, b: Gf.Vec3h) -> Gf.Vec3h:
    """
    Returns the orthogonal complement of the projection of vec onto b.

    This is equivalent to: vec - vec.GetProjection(b)

    Parameters:
        vec (Gf.Vec3h): The input vector.
        b (Gf.Vec3h): The vector to project onto.

    Returns:
        Gf.Vec3h: The orthogonal complement vector.
    """
    if vec.GetLength() == 0 or b.GetLength() == 0:
        return Gf.Vec3h(0, 0, 0)
    projection = vec.GetProjection(b)
    orthogonal_complement = vec - projection
    return orthogonal_complement


def orthogonalize_basis_vectors(
    tx: Gf.Vec3h, ty: Gf.Vec3h, tz: Gf.Vec3h, normalize: bool = True, eps: float = 1e-06
) -> bool:
    """
    Orthogonalize and optionally normalize a set of basis vectors.

    Args:
        tx (Gf.Vec3h): The first basis vector.
        ty (Gf.Vec3h): The second basis vector.
        tz (Gf.Vec3h): The third basis vector.
        normalize (bool): Whether to normalize the basis vectors. Defaults to True.
        eps (float): The epsilon value for convergence. Defaults to 1e-6.

    Returns:
        bool: True if the solution converged, False otherwise.
    """
    max_iterations = 10
    iteration = 0
    converged = False
    while iteration < max_iterations and (not converged):
        ty = ty - ty.GetProjection(tx)
        tz = tz - tz.GetProjection(tx) - tz.GetProjection(ty)
        tx = tx - tx.GetProjection(ty) - tx.GetProjection(tz)
        dot_xy = abs(tx.GetDot(ty))
        dot_xz = abs(tx.GetDot(tz))
        dot_yz = abs(ty.GetDot(tz))
        converged = dot_xy < eps and dot_xz < eps and (dot_yz < eps)
        iteration += 1
    if normalize:
        tx.Normalize(eps)
        ty.Normalize(eps)
        tz.Normalize(eps)
    return converged


def get_vector_projection(v1: Gf.Vec3h, v2: Gf.Vec3h) -> Gf.Vec3h:
    """
    Get the projection of v1 onto v2.

    Args:
        v1 (Gf.Vec3h): The vector to project.
        v2 (Gf.Vec3h): The vector to project onto.

    Returns:
        Gf.Vec3h: The projection of v1 onto v2.

    Raises:
        ValueError: If v2 is a zero vector.
    """
    if v2.GetLength() < 1e-06:
        raise ValueError("Cannot project onto a zero vector.")
    dot_product = Gf.Dot(v1, v2)
    v2_length_squared = v2.GetLength() ** 2
    projection_scalar = dot_product / v2_length_squared
    projection_vector = v2 * projection_scalar
    return projection_vector


def normalize_prim_orientation(prim: Usd.Prim) -> None:
    """Normalize the orientation of a prim to align with the world axes.

    This function adjusts the rotation of the prim so that its X-axis aligns with the
    world X-axis, its Y-axis aligns with the world Y-axis, and its Z-axis aligns with
    the world Z-axis.

    Args:
        prim (Usd.Prim): The prim whose orientation needs to be normalized.

    Raises:
        ValueError: If the provided prim is not valid or not transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    rotation_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ]
    if not rotation_ops:
        return
    rotation_op = rotation_ops[0]
    rotation = rotation_op.Get()
    prim_orientation = Gf.Rotation(Gf.Vec3d(rotation), Gf.Vec3d(0, 0, 1))
    target_orientation = Gf.Rotation(Gf.Vec3d(0, 0, 1), 0)
    correction_rotation = target_orientation * prim_orientation.GetInverse()
    new_rotation = correction_rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
    rotation_op.Set(Gf.Vec3f(new_rotation))


def align_prim_to_another(stage: Usd.Stage, source_prim_path: str, target_prim_path: str) -> None:
    """Aligns the source prim to the target prim's orientation in world space."""
    source_prim = stage.GetPrimAtPath(source_prim_path)
    if not source_prim.IsValid():
        raise ValueError(f"Source prim at path {source_prim_path} does not exist.")
    target_prim = stage.GetPrimAtPath(target_prim_path)
    if not target_prim.IsValid():
        raise ValueError(f"Target prim at path {target_prim_path} does not exist.")
    source_xformable = UsdGeom.Xformable(source_prim)
    if not source_xformable:
        raise ValueError(f"Source prim at path {source_prim_path} is not transformable.")
    target_xformable = UsdGeom.Xformable(target_prim)
    if not target_xformable:
        raise ValueError(f"Target prim at path {target_prim_path} is not transformable.")
    source_matrix = source_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    target_matrix = target_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rotation_matrix = target_matrix.ExtractRotationMatrix() * source_matrix.ExtractRotationMatrix().GetInverse()
    rotation_euler = rotation_matrix.ExtractRotation().Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())
    rotation_op = source_xformable.GetOrderedXformOps()[0]
    if rotation_op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
        rotation_op.Set(rotation_euler)
    else:
        add_rotate_xyz_op(source_xformable).Set(rotation_euler)


def match_prim_scale_along_axis(stage: Usd.Stage, prim_path: str, target_scale: Gf.Vec3i) -> None:
    """
    Match the scale of a prim along the specified axis.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to modify.
        target_scale (Gf.Vec3i): The target scale along each axis (X, Y, Z).

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    scale_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
    if scale_ops:
        scale_op = scale_ops[0]
    else:
        scale_op = add_scale_op(xformable)
    old_scale = scale_op.Get(Usd.TimeCode.Default())
    new_scale = Gf.Vec3f(old_scale) if old_scale else Gf.Vec3f(1.0)
    for i in range(3):
        if target_scale[i] != 0:
            new_scale[i] = float(target_scale[i])
    scale_op.Set(new_scale)


def compute_point_along_axis(axis: Gf.Vec3i, distance: int) -> Gf.Vec3i:
    """Compute a point along the given axis at the specified distance.

    Args:
        axis (Gf.Vec3i): The axis along which to compute the point.
        distance (int): The distance along the axis to compute the point.

    Returns:
        Gf.Vec3i: The computed point along the axis.

    Raises:
        ValueError: If the provided axis is not a unit vector.
    """
    if axis not in [Gf.Vec3i.XAxis(), Gf.Vec3i.YAxis(), Gf.Vec3i.ZAxis()]:
        raise ValueError("The provided axis must be a unit vector.")
    point = axis * distance
    return point


def scatter_points_along_axis(start: Gf.Vec3i, end: Gf.Vec3i, count: int) -> List[Gf.Vec3i]:
    """Scatter points evenly along an axis defined by start and end points.

    Args:
        start (Gf.Vec3i): The start point of the axis.
        end (Gf.Vec3i): The end point of the axis.
        count (int): The number of points to scatter along the axis (inclusive of start and end).

    Returns:
        List[Gf.Vec3i]: A list of evenly scattered points along the axis.

    Raises:
        ValueError: If count is less than 2 or start and end points are the same.
    """
    if count < 2:
        raise ValueError("Count must be at least 2.")
    if start == end:
        raise ValueError("Start and end points cannot be the same.")
    step = Gf.Vec3i((end - start) / (count - 1))
    scattered_points = [start + i * step for i in range(count)]
    return scattered_points


def compute_average_position(positions: List[Gf.Vec4d]) -> Gf.Vec4d:
    """
    Compute the average position from a list of Vec4d positions.

    Args:
        positions (List[Gf.Vec4d]): List of positions.

    Returns:
        Gf.Vec4d: The average position.

    Raises:
        ValueError: If the input list is empty.
    """
    if not positions:
        raise ValueError("Input list of positions is empty.")
    sum_position = Gf.Vec4d(0, 0, 0, 0)
    for position in positions:
        sum_position += position
    num_positions = len(positions)
    average_position = sum_position / num_positions
    return average_position


def create_unit_vectors() -> Dict[str, Gf.Vec4d]:
    """Create a dictionary of unit vectors along each axis.

    Returns:
        Dict[str, Gf.Vec4d]: A dictionary mapping axis names to unit vectors.
    """
    unit_vectors = {"x": Gf.Vec4d.XAxis(), "y": Gf.Vec4d.YAxis(), "z": Gf.Vec4d.ZAxis(), "w": Gf.Vec4d.WAxis()}
    return unit_vectors


def calculate_vector_projections(v1: Gf.Vec4d, v2: Gf.Vec4d) -> Tuple[Gf.Vec4d, Gf.Vec4d]:
    """Calculate the projection and complement of v1 onto v2.

    Args:
        v1 (Gf.Vec4d): The vector to project.
        v2 (Gf.Vec4d): The vector to project onto.

    Returns:
        Tuple[Gf.Vec4d, Gf.Vec4d]: A tuple containing the projection and complement vectors.
    """
    if v1.GetLength() == 0 or v2.GetLength() == 0:
        raise ValueError("Cannot project onto or from a zero vector.")
    projection = v2.GetProjection(v1)
    complement = v1 - projection
    return (projection, complement)


def compute_bounding_box_volume(min_point: Gf.Vec4d, max_point: Gf.Vec4d) -> float:
    """Compute the volume of an axis-aligned bounding box.

    Args:
        min_point (Gf.Vec4d): The minimum point of the bounding box.
        max_point (Gf.Vec4d): The maximum point of the bounding box.

    Returns:
        float: The volume of the bounding box.

    Raises:
        ValueError: If the minimum point is not less than or equal to the maximum point.
    """
    if not all((min_point[i] <= max_point[i] for i in range(3))):
        raise ValueError("Minimum point must be less than or equal to maximum point.")
    dimensions = max_point - min_point
    volume = dimensions[0] * dimensions[1] * dimensions[2]
    return volume


def calculate_dot_product(a: Gf.Vec4d, b: Gf.Vec4d) -> float:
    """
    Calculate the dot product of two Vec4d vectors.

    Args:
        a (Gf.Vec4d): The first vector.
        b (Gf.Vec4d): The second vector.

    Returns:
        float: The dot product of the two vectors.
    """
    if not isinstance(a, Gf.Vec4d) or not isinstance(b, Gf.Vec4d):
        raise TypeError("Both arguments must be of type Gf.Vec4d")
    dot_product = a.GetDot(b)
    return dot_product


def compute_vector_complement(a: Gf.Vec4d, b: Gf.Vec4d) -> Gf.Vec4d:
    """
    Compute the orthogonal complement of the projection of vector a onto vector b.

    The orthogonal complement is defined as: a - a.GetProjection(b)

    Args:
        a (Gf.Vec4d): The input vector.
        b (Gf.Vec4d): The vector to project onto.

    Returns:
        Gf.Vec4d: The orthogonal complement vector.
    """
    eps = 1e-06
    if a.GetLength() < eps or b.GetLength() < eps:
        raise ValueError("Input vectors must be non-zero.")
    projection = a.GetProjection(b)
    complement = a - projection
    return complement


def align_prims_along_axes(stage: Usd.Stage, prim_paths: List[str], alignment_vector: Gf.Vec4d) -> None:
    """
    Align a list of prims along the given vector.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths to align.
        alignment_vector (Gf.Vec4d): The vector to align the prims along.

    Raises:
        ValueError: If any of the prim paths are invalid or not transformable.
    """
    alignment_vector = alignment_vector.GetNormalized(1e-05)
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        rotation_op = add_rotate_xyz_op(xformable, UsdGeom.XformOp.PrecisionFloat, "rotation")
        target_rotation = Gf.Rotation(
            Gf.Vec3d(0, 1, 0), Gf.Vec3d(alignment_vector[0], alignment_vector[1], alignment_vector[2])
        )
        euler_angles = target_rotation.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())
        rotation_op.Set(Gf.Vec3d(euler_angles[0], euler_angles[1], euler_angles[2]))


def align_prims_by_vec4d(
    stage: Usd.Stage, source_prim_path: str, target_prim_path: str, up_vector: Gf.Vec4d = Gf.Vec4d(0, 1, 0, 0)
) -> None:
    """Aligns the source prim to the target prim using the provided up vector.

    Args:
        stage (Usd.Stage): The USD stage.
        source_prim_path (str): The path of the source prim to align.
        target_prim_path (str): The path of the target prim to align to.
        up_vector (Gf.Vec4d, optional): The up vector to use for alignment. Defaults to Gf.Vec4d(0, 1, 0, 0).

    Raises:
        ValueError: If either the source or target prim is invalid or not transformable.
    """
    source_prim = stage.GetPrimAtPath(source_prim_path)
    target_prim = stage.GetPrimAtPath(target_prim_path)
    if not source_prim.IsValid() or not target_prim.IsValid():
        raise ValueError("Source or target prim is invalid.")
    source_xformable = UsdGeom.Xformable(source_prim)
    target_xformable = UsdGeom.Xformable(target_prim)
    if not source_xformable or not target_xformable:
        raise ValueError("Source or target prim is not transformable.")
    source_transform = source_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    target_transform = target_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    source_pos = source_transform.ExtractTranslation()
    target_pos = target_transform.ExtractTranslation()
    direction = target_pos - source_pos
    direction.Normalize()
    rotation = Gf.Matrix4d()
    rotation.SetRotate(
        Gf.Rotation(
            Gf.Vec3d(up_vector[0], up_vector[1], up_vector[2]), Gf.Vec3d(direction[0], direction[1], direction[2])
        )
    )
    transform_op = source_xformable.AddXformOp(UsdGeom.XformOp.TypeTransform)
    transform_op.Set(rotation)


def compute_distance_between_prims(stage: Usd.Stage, prim_a_path: str, prim_b_path: str) -> float:
    """Compute the Euclidean distance between two prims.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_a_path (str): The path to the first prim.
        prim_b_path (str): The path to the second prim.

    Returns:
        float: The Euclidean distance between the two prims.

    Raises:
        ValueError: If either prim does not exist or is not transformable.
    """
    prim_a = stage.GetPrimAtPath(prim_a_path)
    prim_b = stage.GetPrimAtPath(prim_b_path)
    if not prim_a.IsValid() or not prim_b.IsValid():
        raise ValueError(f"One or both prims do not exist: {prim_a_path}, {prim_b_path}")
    xformable_a = UsdGeom.Xformable(prim_a)
    xformable_b = UsdGeom.Xformable(prim_b)
    if not xformable_a or not xformable_b:
        raise ValueError(f"One or both prims are not transformable: {prim_a_path}, {prim_b_path}")
    translation_a = xformable_a.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    translation_b = xformable_b.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    distance = (translation_b - translation_a).GetLength()
    return distance


def project_prim_onto_vector(stage: Usd.Stage, prim_path: str, vector: Gf.Vec4f) -> Gf.Vec4f:
    """Project a prim's world space position onto a vector.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.
        vector (Gf.Vec4f): The vector to project onto.

    Returns:
        Gf.Vec4f: The projected vector.

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    position = world_transform.ExtractTranslation()
    position_vec4f = Gf.Vec4f(position[0], position[1], position[2], 1.0)
    projected_vec4f = vector.GetProjection(position_vec4f)
    return projected_vec4f


def compute_orthogonal_complement(vec: Gf.Vec4f, basis: Gf.Vec4f) -> Gf.Vec4f:
    """
    Compute the orthogonal complement of the projection of vec onto basis.

    The orthogonal complement is defined as:
    vec - vec.GetProjection(basis)

    Args:
        vec (Gf.Vec4f): The input vector.
        basis (Gf.Vec4f): The basis vector.

    Returns:
        Gf.Vec4f: The orthogonal complement vector.
    """
    if not isinstance(vec, Gf.Vec4f) or not isinstance(basis, Gf.Vec4f):
        raise TypeError("Input vectors must be of type Gf.Vec4f")
    projection = vec.GetProjection(basis)
    orthogonal_complement = vec - projection
    return orthogonal_complement


def compute_bounding_box_for_prims(stage: Usd.Stage, prim_paths: List[str]) -> Tuple[Gf.Vec4f, Gf.Vec4f]:
    """Compute the bounding box for a list of prims.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (List[str]): A list of prim paths to compute the bounding box for.

    Returns:
        Tuple[Gf.Vec4f, Gf.Vec4f]: A tuple containing the minimum and maximum points of the bounding box.
    """
    min_point = Gf.Vec4f(float("inf"), float("inf"), float("inf"), 1.0)
    max_point = Gf.Vec4f(float("-inf"), float("-inf"), float("-inf"), 1.0)
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        bbox = UsdGeom.Imageable(prim).ComputeLocalBound(Usd.TimeCode.Default(), "default")
        if bbox.GetRange().IsEmpty():
            continue
        min_point = Gf.Vec4f(
            min(min_point[0], bbox.GetRange().GetMin()[0]),
            min(min_point[1], bbox.GetRange().GetMin()[1]),
            min(min_point[2], bbox.GetRange().GetMin()[2]),
            1.0,
        )
        max_point = Gf.Vec4f(
            max(max_point[0], bbox.GetRange().GetMax()[0]),
            max(max_point[1], bbox.GetRange().GetMax()[1]),
            max(max_point[2], bbox.GetRange().GetMax()[2]),
            1.0,
        )
    if min_point == Gf.Vec4f(float("inf"), float("inf"), float("inf"), 1.0):
        min_point = Gf.Vec4f(0.0, 0.0, 0.0, 1.0)
        max_point = Gf.Vec4f(0.0, 0.0, 0.0, 1.0)
    return (min_point, max_point)


def distribute_prims_along_vector(stage: Usd.Stage, prim_paths: List[str], vector: Gf.Vec3f, spacing: float) -> None:
    """Distribute prims along a vector with specified spacing.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths to distribute.
        vector (Gf.Vec3f): The vector along which to distribute the prims.
        spacing (float): The spacing between each prim.
    """
    vector.Normalize()
    vector *= spacing
    for i, prim_path in enumerate(prim_paths):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            raise ValueError(f"Prim not found at path: {prim_path}")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        translation = vector * i
        add_translate_op(xformable).Set(Gf.Vec3d(translation))


def set_prims_to_origin(stage: Usd.Stage, prim_paths: List[str]) -> None:
    """Set the translation of the specified prims to the origin (0, 0, 0).

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths to set the translation for.

    Raises:
        ValueError: If any of the specified prims are invalid or not transformable.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        if not translate_ops:
            translate_op = add_translate_op(xformable)
        else:
            translate_op = translate_ops[0]
        translate_op.Set(Gf.Vec3d(0, 0, 0))


def compute_centroid_of_prims(stage: Usd.Stage, prim_paths: List[str]) -> Optional[Gf.Vec4f]:
    """
    Compute the centroid of a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths.

    Returns:
        Optional[Gf.Vec4f]: The centroid of the prims, or None if no valid prims are found.
    """
    centroid = Gf.Vec4f(0, 0, 0, 0)
    total_weight = 0.0
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
        if bbox.GetRange().IsEmpty():
            continue
        bbox_centroid = bbox.ComputeCentroid()
        bbox_range = bbox.GetRange()
        bbox_volume = bbox_range.GetSize()[0] * bbox_range.GetSize()[1] * bbox_range.GetSize()[2]
        centroid += Gf.Vec4f(bbox_centroid[0], bbox_centroid[1], bbox_centroid[2], 1.0) * bbox_volume
        total_weight += bbox_volume
    if total_weight == 0.0:
        return None
    centroid /= total_weight
    return Gf.Vec4f(centroid[0], centroid[1], centroid[2], 1.0)


def set_vector_attributes(prim: Usd.Prim, attr_name: str, value: Gf.Vec4h) -> None:
    """Set a Vec4h attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to set the attribute on.
        attr_name (str): The name of the attribute to set.
        value (Gf.Vec4h): The Vec4h value to set on the attribute.

    Raises:
        ValueError: If the prim is not valid.
        TypeError: If the value is not of type Gf.Vec4h.
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid")
    if not isinstance(value, Gf.Vec4h):
        raise TypeError(f"Value must be of type Gf.Vec4h, got {type(value)}")
    if not prim.HasAttribute(attr_name):
        prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Half4)
    attribute = prim.GetAttribute(attr_name)
    attribute.Set(value)


def get_prim_vector_projection(a: Gf.Vec4h, b: Gf.Vec4h) -> Gf.Vec4h:
    """
    Get the projection of vector a onto vector b.

    Args:
        a (Gf.Vec4h): The vector to project.
        b (Gf.Vec4h): The vector to project onto.

    Returns:
        Gf.Vec4h: The projection of a onto b.
    """
    if a.GetLength() == 0 or b.GetLength() == 0:
        return Gf.Vec4h(0, 0, 0, 0)
    dot_product = Gf.Dot(a, b)
    b_length_squared = b.GetLength() ** 2
    projection_length = dot_product / b_length_squared
    projection = b * projection_length
    return projection


def calculate_vector_complement(a: Gf.Vec4h, b: Gf.Vec4h) -> Gf.Vec4h:
    """
    Calculate the orthogonal complement of the projection of vector a onto vector b.

    Args:
        a (Gf.Vec4h): The vector to project.
        b (Gf.Vec4h): The vector to project onto.

    Returns:
        Gf.Vec4h: The orthogonal complement of the projection.
    """
    if a.GetLength() == 0 or b.GetLength() == 0:
        return Gf.Vec4h(0, 0, 0, 0)
    projection = a.GetProjection(b)
    complement = a - projection
    return complement


def create_unit_vectors_along_axes() -> Tuple[Gf.Vec4h, Gf.Vec4h, Gf.Vec4h, Gf.Vec4h]:
    """Create unit vectors along the X, Y, Z, and W axes.

    Returns:
        Tuple[Gf.Vec4h, Gf.Vec4h, Gf.Vec4h, Gf.Vec4h]: Unit vectors along the X, Y, Z, and W axes.
    """
    x_unit = Gf.Vec4h.XAxis()
    y_unit = Gf.Vec4h.YAxis()
    z_unit = Gf.Vec4h.ZAxis()
    w_unit = Gf.Vec4h.WAxis()
    return (x_unit, y_unit, z_unit, w_unit)


def get_prim_vector_length(stage: Usd.Stage, prim_path: str, vector_attr_name: str) -> float:
    """Get the length of a vector attribute on a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        vector_attr_name (str): The name of the vector attribute.

    Returns:
        float: The length of the vector attribute.

    Raises:
        ValueError: If the prim or attribute does not exist.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    attr = prim.GetAttribute(vector_attr_name)
    if not attr.IsValid():
        raise ValueError(f"Attribute {vector_attr_name} does not exist on prim {prim_path}.")
    value = attr.Get()
    if value is None:
        return 0.0
    vec4h = Gf.Vec4h(value)
    length = vec4h.GetLength()
    return float(length)


def calculate_and_set_vector_complement(vec: Gf.Vec4h, b: Gf.Vec4h) -> Gf.Vec4h:
    """
    Calculate the orthogonal complement of the projection of vec onto b.

    The result is stored in vec.

    Args:
        vec (Gf.Vec4h): The vector to calculate the complement for.
        b (Gf.Vec4h): The vector to project onto.

    Returns:
        Gf.Vec4h: The modified vec containing the orthogonal complement.
    """
    projection = vec.GetProjection(b)
    complement = vec - projection
    vec[0] = complement[0]
    vec[1] = complement[1]
    vec[2] = complement[2]
    vec[3] = complement[3]
    return vec


def interpolate_vectors(a: Gf.Vec4h, b: Gf.Vec4h, t: float) -> Gf.Vec4h:
    """Linearly interpolate between two vectors.

    Args:
        a (Gf.Vec4h): The first vector.
        b (Gf.Vec4h): The second vector.
        t (float): The interpolation factor between 0 and 1.

    Returns:
        Gf.Vec4h: The interpolated vector.
    """
    t = max(0.0, min(t, 1.0))
    result = a + (b - a) * t
    return result


def normalize_and_project_vector(v: Gf.Vec4h, onto: Gf.Vec4h, eps: float = 1e-06) -> Tuple[Gf.Vec4h, float]:
    """
    Normalize the input vector 'v' and project it onto the vector 'onto'.

    Args:
        v (Gf.Vec4h): The vector to normalize and project.
        onto (Gf.Vec4h): The vector to project onto.
        eps (float): The epsilon value for normalization (default: 1e-6).

    Returns:
        Tuple[Gf.Vec4h, float]: A tuple containing the normalized and projected vector,
                                and the length of the original vector before normalization.
    """
    length: float = v.Normalize(eps)
    projected: Gf.Vec4h = v.GetProjection(onto)
    return (projected, length)


def calculate_vector_dot_product(vec1: Gf.Vec4h, vec2: Gf.Vec4h) -> float:
    """
    Calculate the dot product of two Vec4h vectors.

    Args:
        vec1 (Gf.Vec4h): The first vector.
        vec2 (Gf.Vec4h): The second vector.

    Returns:
        float: The dot product of the two vectors.
    """
    if not isinstance(vec1, Gf.Vec4h) or not isinstance(vec2, Gf.Vec4h):
        raise TypeError("Both inputs must be of type Gf.Vec4h.")
    dot_product = vec1.GetDot(vec2)
    return float(dot_product)


def compute_combined_dot_product(vec1: Gf.Vec4i, vec2: Gf.Vec4i) -> int:
    """Compute the combined dot product of two Vec4i vectors.

    The combined dot product is the sum of the element-wise products.

    Args:
        vec1 (Gf.Vec4i): The first vector.
        vec2 (Gf.Vec4i): The second vector.

    Returns:
        int: The combined dot product.
    """
    if not isinstance(vec1, Gf.Vec4i) or not isinstance(vec2, Gf.Vec4i):
        raise TypeError("Input vectors must be of type Gf.Vec4i")
    products = [v1 * v2 for (v1, v2) in zip(vec1, vec2)]
    combined_dot_product = sum(products)
    return combined_dot_product


def filter_prims_by_vector(stage: Usd.Stage, vector: Gf.Vec4i) -> List[Usd.Prim]:
    """
    Filter prims on a stage by a vector condition.

    Args:
        stage (Usd.Stage): The USD stage to filter prims from.
        vector (Gf.Vec4i): The vector condition to filter prims by.

    Returns:
        List[Usd.Prim]: A list of prims that match the vector condition.
    """
    matching_prims: List[Usd.Prim] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xformable):
            xformable = UsdGeom.Xformable(prim)
            transform = xformable.GetLocalTransformation()
            translation = transform.ExtractTranslation()
            prim_vector = Gf.Vec4i(int(translation[0]), int(translation[1]), int(translation[2]), 0)
            if prim_vector == vector:
                matching_prims.append(prim)
    return matching_prims


def apply_vector_transformation(vector: Gf.Vec4i, transformation: Gf.Matrix4d) -> Gf.Vec4i:
    """
    Apply a transformation matrix to a 4D integer vector.

    Args:
        vector (Gf.Vec4i): The input 4D integer vector.
        transformation (Gf.Matrix4d): The 4x4 transformation matrix.

    Returns:
        Gf.Vec4i: The transformed 4D integer vector.
    """
    if not isinstance(vector, Gf.Vec4i):
        raise TypeError("Input vector must be of type Gf.Vec4i")
    if not isinstance(transformation, Gf.Matrix4d):
        raise TypeError("Transformation matrix must be of type Gf.Matrix4d")
    vector_double = Gf.Vec4d(vector)
    transformed_vector_double = transformation * vector_double
    transformed_vector_int = Gf.Vec4i(
        int(round(transformed_vector_double[0])),
        int(round(transformed_vector_double[1])),
        int(round(transformed_vector_double[2])),
        int(round(transformed_vector_double[3])),
    )
    return transformed_vector_int


def set_prims_to_unit_vectors(stage: Usd.Stage, prim_paths: List[str]):
    """Set the scale of each prim to a unit vector based on its index.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (List[str]): The paths of the prims to set.

    Raises:
        ValueError: If a prim path is invalid or the prim is not transformable.
    """
    for i, prim_path in enumerate(prim_paths):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        if i == 0:
            unit_vector = Gf.Vec4i.XAxis()
        elif i == 1:
            unit_vector = Gf.Vec4i.YAxis()
        elif i == 2:
            unit_vector = Gf.Vec4i.ZAxis()
        else:
            unit_vector = Gf.Vec4i.WAxis()
        scale_op = add_scale_op(xformable)
        scale_op.Set(Gf.Vec3f(unit_vector[0], unit_vector[1], unit_vector[2]))


def animate_prim_along_path(
    stage: Usd.Stage,
    prim_path: str,
    positions: List[Tuple[float, float, float]],
    orientations: List[Gf.Quatf],
    scales: List[Tuple[float, float, float]],
    time_codes: List[Usd.TimeCode],
) -> None:
    """Animate a prim along a path defined by positions, orientations, scales, and time codes.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to animate.
        positions (List[Tuple[float, float, float]]): A list of positions along the path.
        orientations (List[Gf.Quatf]): A list of orientations along the path.
        scales (List[Tuple[float, float, float]]): A list of scales along the path.
        time_codes (List[Usd.TimeCode]): A list of time codes corresponding to each position, orientation, and scale.

    Raises:
        ValueError: If the lengths of positions, orientations, scales, and time_codes are not equal.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    if len(positions) != len(orientations) or len(positions) != len(scales) or len(positions) != len(time_codes):
        raise ValueError("Lengths of positions, orientations, scales, and time_codes must be equal.")
    translate_op = add_translate_op(xformable)
    orient_op = add_orient_op(xformable)
    scale_op = add_scale_op(xformable)
    for i in range(len(positions)):
        translate_op.Set(Gf.Vec3d(positions[i]), time_codes[i])
        orient_op.Set(orientations[i], time_codes[i])
        scale_op.Set(Gf.Vec3f(scales[i]), time_codes[i])
