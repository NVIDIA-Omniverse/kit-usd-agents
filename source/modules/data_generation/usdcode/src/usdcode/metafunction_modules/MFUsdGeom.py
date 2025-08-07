## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import math
import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from pxr import Gf, Kind, Sdf, Tf, Usd, UsdGeom, UsdShade, Vt

from .add_op import *


def get_bbox_for_selected_prims(stage: Usd.Stage, prim_paths: List[str]) -> Tuple[Gf.Vec3d, Gf.Vec3d]:
    """
    Compute the world bounding box for a list of prim paths.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths.

    Returns:
        Tuple[Gf.Vec3d, Gf.Vec3d]: The min and max points of the bounding box.

    Raises:
        ValueError: If any of the prim paths are invalid.
    """
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bbox = Gf.BBox3d()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")
        prim_bbox = bbox_cache.ComputeWorldBound(prim)
        bbox = Gf.BBox3d.Combine(bbox, prim_bbox)
    if bbox.GetRange().IsEmpty():
        return None
    else:
        return (bbox.GetRange().GetMin(), bbox.GetRange().GetMax())


def get_animated_prims_bbox(
    stage: Usd.Stage, bboxCache: UsdGeom.BBoxCache, prim_paths: List[Sdf.Path]
) -> Dict[Sdf.Path, Gf.BBox3d]:
    """
    Compute the bounding box for each prim at the specified paths and return a dictionary mapping prim paths to their bounding boxes.
    Only include prims that have time-sampled transforms.

    Args:
        stage (Usd.Stage): The USD stage.
        bboxCache (UsdGeom.BBoxCache): The bounding box cache to use for computations.
        prim_paths (List[Sdf.Path]): The list of prim paths to compute bounding boxes for.

    Returns:
        Dict[Sdf.Path, Gf.BBox3d]: A dictionary mapping prim paths to their computed bounding boxes.
                                    Only prims with time-sampled transforms are included.
    """
    result = {}
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        xformable = UsdGeom.Xformable(prim)
        if not xformable or not xformable.TransformMightBeTimeVarying():
            continue
        bbox = bboxCache.ComputeWorldBound(prim)
        result[prim_path] = bbox
    return result


def get_bbox_for_unloaded_prims(stage: Usd.Stage, prim_paths: List[str]) -> Union[Gf.BBox3d, None]:
    """
    Compute the bounding box for a list of unloaded prims using their extent hints.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths.

    Returns:
        Union[Gf.BBox3d, None]: The computed bounding box, or None if no valid extent hints are found.
    """
    bbox = Gf.BBox3d()
    has_valid_extent = False
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        model_api = UsdGeom.ModelAPI(prim)
        extent_attr = model_api.GetExtentsHintAttr()
        if extent_attr:
            extent = extent_attr.Get()
            if extent:
                bbox = Gf.BBox3d(bbox).UnionWith(Gf.Range3d(extent[0], extent[1]))
                has_valid_extent = True
    if has_valid_extent:
        return bbox
    else:
        return None


def get_point_instances_bbox(
    stage: Usd.Stage,
    point_instancer_path: str,
    instance_ids: List[int],
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
) -> List[Gf.BBox3d]:
    """
    Compute the bounding boxes of specified instances of a UsdGeomPointInstancer.

    Args:
        stage (Usd.Stage): The USD stage.
        point_instancer_path (str): The path to the UsdGeomPointInstancer prim.
        instance_ids (List[int]): The list of instance IDs to compute bounding boxes for.
        time_code (Usd.TimeCode): The time code at which to compute the bounding boxes (default: Default()).

    Returns:
        List[Gf.BBox3d]: The list of computed bounding boxes for the specified instances.
    """
    point_instancer_prim = stage.GetPrimAtPath(point_instancer_path)
    if not point_instancer_prim.IsValid():
        raise ValueError(f"Invalid prim at path: {point_instancer_path}")
    point_instancer = UsdGeom.PointInstancer(point_instancer_prim)
    if not point_instancer:
        raise ValueError(f"Prim at path {point_instancer_path} is not a valid UsdGeomPointInstancer")
    bbox_cache = UsdGeom.BBoxCache(time_code, includedPurposes=[UsdGeom.Tokens.default_])
    bboxes = []
    for instance_id in instance_ids:
        bbox = bbox_cache.ComputePointInstanceWorldBound(point_instancer, instance_id)
        bboxes.append(bbox)
    return bboxes


def test_get_point_instances_bbox():
    stage = Usd.Stage.CreateInMemory()
    point_instancer_path = "/PointInstancer"
    point_instancer = UsdGeom.PointInstancer.Define(stage, point_instancer_path)
    positions = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    indices = [0, 0, 0, 0]
    point_instancer.CreatePositionsAttr().Set(positions)
    point_instancer.CreateProtoIndicesAttr().Set(indices)
    prototype_path = "/Prototype"
    prototype = UsdGeom.Cube.Define(stage, prototype_path)
    prototype.GetSizeAttr().Set(0.5)
    point_instancer.CreatePrototypesRel().SetTargets([prototype.GetPath()])
    instance_ids = [0, 2]
    bboxes = get_point_instances_bbox(stage, point_instancer_path, instance_ids)
    print(bboxes)


def get_aggregate_world_bbox(
    prim: Usd.Prim,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
    purpose1: UsdGeom.Tokens = UsdGeom.Tokens.default_,
    purpose2: UsdGeom.Tokens = UsdGeom.Tokens.default_,
    purpose3: UsdGeom.Tokens = UsdGeom.Tokens.default_,
    purpose4: UsdGeom.Tokens = UsdGeom.Tokens.default_,
) -> Gf.BBox3d:
    """
    Compute the aggregate (untransformed) bounding box for a prim and its descendants.

    Args:
        prim (Usd.Prim): The root prim to compute the bounding box for.
        time (Usd.TimeCode): The time at which to compute the bounding box. Defaults to Default.
        purpose1 (UsdGeom.Tokens): The first purpose to include in the computation. Defaults to default_.
        purpose2 (UsdGeom.Tokens): The second purpose to include in the computation. Defaults to default_.
        purpose3 (UsdGeom.Tokens): The third purpose to include in the computation. Defaults to default_.
        purpose4 (UsdGeom.Tokens): The fourth purpose to include in the computation. Defaults to default_.

    Returns:
        Gf.BBox3d: The computed aggregate bounding box in local space of the prim.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim")
    bbox_cache = UsdGeom.BBoxCache(time, includedPurposes=[purpose1, purpose2, purpose3, purpose4])
    bbox = bbox_cache.ComputeUntransformedBound(prim)
    return bbox


def get_aggregate_local_bbox(
    prim: Usd.Prim, bbox_cache: UsdGeom.BBoxCache, xform_cache: UsdGeom.XformCache
) -> Gf.BBox3d:
    """
    Compute the aggregate local bounding box for a prim, considering all its
    children and descendants.

    Args:
        prim (Usd.Prim): The prim to compute the bounding box for.
        bbox_cache (UsdGeom.BBoxCache): The BBoxCache instance to use for computations.
        xform_cache (UsdGeom.XformCache): The XformCache instance to use for computations.

    Returns:
        Gf.BBox3d: The computed aggregate local bounding box.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim")
    bbox = Gf.BBox3d()
    for child in prim.GetChildren():
        child_bbox = bbox_cache.ComputeLocalBound(child)
        child_xform = xform_cache.GetLocalToWorldTransform(child)
        child_bbox.Transform(child_xform)
        bbox = Gf.BBox3d.Combine(bbox, child_bbox)
    return bbox


def filter_prims_by_bbox_size(
    stage: Usd.Stage, min_size: Tuple[float, float, float], max_size: Tuple[float, float, float]
) -> List[Usd.Prim]:
    """
    Filter prims on the stage by their bounding box size.

    Args:
        stage (Usd.Stage): The USD stage to filter prims from.
        min_size (Tuple[float, float, float]): The minimum size of the bounding box (x, y, z).
        max_size (Tuple[float, float, float]): The maximum size of the bounding box (x, y, z).

    Returns:
        List[Usd.Prim]: A list of prims whose bounding box size falls within the specified range.
    """
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    filtered_prims = []
    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Boundable):
            bbox = bbox_cache.ComputeWorldBound(prim)
            size = bbox.ComputeAlignedRange().GetSize()
            if all((min_size[i] <= size[i] <= max_size[i] for i in range(3))):
                filtered_prims.append(prim)
    return filtered_prims


def set_transform_order(prim: UsdGeom.Xformable, xform_op_names: List[str]) -> bool:
    """Sets the order of transform operations on a prim.

    Args:
        prim (UsdGeom.Xformable): The prim to set the transform order on.
        xform_op_names (List[str]): The list of transform operation names in the desired order.

    Returns:
        bool: True if the transform order was set successfully, False otherwise.
    """
    current_ops = prim.GetOrderedXformOps()
    reordered_ops = []
    for op_name in xform_op_names:
        matching_ops = [op for op in current_ops if op.GetName() == op_name]
        if not matching_ops:
            print(f"Transform operation '{op_name}' not found on the prim.")
            return False
        reordered_ops.append(matching_ops[0])
    try:
        prim.SetXformOpOrder(reordered_ops, resetXformStack=False)
    except Exception as e:
        print(f"Error setting transform order: {str(e)}")
        return False
    return True


def create_prim(stage, prim_type, prim_path):
    prim = stage.DefinePrim(prim_path, prim_type)
    xform = UsdGeom.Xform(prim)
    return xform


def compute_combined_bbox(stage: Usd.Stage, prim_paths: list[str]) -> Gf.BBox3d:
    """Compute the combined bounding box for a list of prims."""
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    combined_bbox = Gf.BBox3d()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        prim_bbox = bbox_cache.ComputeWorldBound(prim)
        combined_bbox = Gf.BBox3d.Combine(combined_bbox, prim_bbox)
    return combined_bbox


def compute_difference_between_world_and_local_bbox(cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> Gf.Vec3d:
    """Compute the difference between the world and local bounding box of a prim."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim} is not valid.")
    world_bbox = cache.ComputeWorldBound(prim)
    local_bbox = cache.ComputeLocalBound(prim)
    world_bbox_size = world_bbox.ComputeAlignedRange().GetSize()
    local_bbox_size = local_bbox.ComputeAlignedRange().GetSize()
    bbox_diff = world_bbox_size - local_bbox_size
    return bbox_diff


def compute_world_bounding_boxes(
    stage: Usd.Stage, prim_paths: List[str], time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> List[Gf.BBox3d]:
    """
    Compute the world bounding boxes for a list of prim paths at a specific time code.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The list of prim paths.
        time_code (Usd.TimeCode, optional): The time code at which to compute the bounding boxes. Defaults to Usd.TimeCode.Default().

    Returns:
        List[Gf.BBox3d]: The list of computed world bounding boxes.
    """
    bbox_cache = UsdGeom.BBoxCache(time_code, includedPurposes=[UsdGeom.Tokens.default_])
    world_bboxes = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        try:
            world_bbox = bbox_cache.ComputeWorldBound(prim)
            world_bboxes.append(world_bbox)
        except RuntimeError:
            pass
    return world_bboxes


def compute_bbox_difference(bbox_cache: UsdGeom.BBoxCache, prim1: Usd.Prim, prim2: Usd.Prim) -> Gf.Range3d:
    """
    Compute the difference between the bounding boxes of two prims.

    Args:
        bbox_cache (UsdGeom.BBoxCache): The BBoxCache instance to use for computing bounds.
        prim1 (Usd.Prim): The first prim.
        prim2 (Usd.Prim): The second prim.

    Returns:
        Gf.Range3d: The difference between the bounding boxes of the two prims.

    Raises:
        ValueError: If either prim is invalid or not boundable.
    """
    if not prim1.IsValid() or not prim2.IsValid():
        raise ValueError("One or both prims are invalid.")
    if not prim1.IsA(UsdGeom.Boundable) or not prim2.IsA(UsdGeom.Boundable):
        raise ValueError("One or both prims are not boundable.")
    bbox1 = bbox_cache.ComputeWorldBound(prim1)
    bbox2 = bbox_cache.ComputeWorldBound(prim2)
    bbox_diff = Gf.Range3d()
    bbox_diff.UnionWith(bbox1.ComputeAlignedRange())
    bbox_diff.UnionWith(bbox2.ComputeAlignedRange())
    return bbox_diff


def compute_bbox_for_visible_prims(stage: Usd.Stage, purposes: List[UsdGeom.Tokens]) -> Gf.BBox3d:
    """Compute the bounding box for all visible prims on the given stage.

    Args:
        stage (Usd.Stage): The USD stage to compute the bounding box for.
        purposes (List[UsdGeom.Tokens]): The purposes to consider when computing the bounding box.

    Returns:
        Gf.BBox3d: The computed bounding box.
    """
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    root_prim = stage.GetPseudoRoot()
    bbox = Gf.BBox3d()
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsActive() or not prim.IsLoaded():
            continue
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        if imageable.GetPurposeAttr().Get() not in purposes:
            continue
        prim_bbox = bbox_cache.ComputeWorldBound(prim)
        bbox = Gf.BBox3d.Combine(bbox, prim_bbox)
    return bbox


def get_bbox_for_all_purposes(prim: Usd.Prim, time_code: Usd.TimeCode = Usd.TimeCode.Default()) -> Gf.BBox3d:
    """
    Compute the bounding box for a prim considering all purposes.

    Args:
        prim (Usd.Prim): The prim to compute the bounding box for.
        time_code (Usd.TimeCode, optional): The time code at which to compute the bounding box.
            Defaults to Usd.TimeCode.Default().

    Returns:
        Gf.BBox3d: The computed bounding box.

    Raises:
        ValueError: If the provided prim is not valid.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim provided.")
    bbox_cache = UsdGeom.BBoxCache(time_code, includedPurposes=[])
    all_purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide]
    bbox_cache.SetIncludedPurposes(all_purposes)
    bbox = bbox_cache.ComputeWorldBound(prim)
    return bbox


def compute_local_bounding_boxes(stage: Usd.Stage, prim_paths: list[str]) -> Dict[str, Gf.BBox3d]:
    """Compute the local bounding box for each prim path in the given list.

    Args:
        stage (Usd.Stage): The USD stage to compute bounding boxes for.
        prim_paths (list[str]): A list of prim paths to compute bounding boxes for.

    Returns:
        Dict[str, Gf.BBox3d]: A dictionary mapping prim paths to their local bounding boxes.
    """
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bboxes = {}
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        bbox = bbox_cache.ComputeLocalBound(prim)
        bboxes[prim_path] = bbox
    return bboxes


def update_bbox_cache_for_time_range(
    bbox_cache: UsdGeom.BBoxCache, prim: Usd.Prim, start_time: Usd.TimeCode, end_time: Usd.TimeCode, step_size: float
) -> None:
    """Update the BBoxCache for a prim over a range of times.

    Args:
        bbox_cache (UsdGeom.BBoxCache): The BBoxCache to update.
        prim (Usd.Prim): The prim to compute the bounding box for.
        start_time (Usd.TimeCode): The start time of the range.
        end_time (Usd.TimeCode): The end time of the range.
        step_size (float): The step size for sampling times.

    Raises:
        ValueError: If the prim is not valid or if the start time is greater than the end time.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if start_time > end_time:
        raise ValueError("Start time must be less than or equal to end time.")
    step_size = abs(step_size)
    current_time = start_time.GetValue()
    end_time_value = end_time.GetValue()
    while current_time <= end_time_value:
        bbox_cache.SetTime(Usd.TimeCode(current_time))
        bbox_cache.ComputeWorldBound(prim)
        current_time += step_size


def set_time_and_purposes_for_bbox_cache(
    bbox_cache: UsdGeom.BBoxCache, time: Usd.TimeCode, purposes: List[Usd.Tokens] = None
) -> None:
    """Set the time and purposes for a BBoxCache.

    Args:
        bbox_cache (UsdGeom.BBoxCache): The BBoxCache instance to update.
        time (Usd.TimeCode): The time to set for the BBoxCache.
        purposes (List[Usd.Tokens], optional): The purposes to set for the BBoxCache.
            If None, uses the default purposes. Defaults to None.
    """
    bbox_cache.SetTime(time)
    if purposes is not None:
        bbox_cache.SetIncludedPurposes(purposes)


def update_bbox_cache_for_prims(
    prims: List[Usd.Prim], time: Usd.TimeCode, purposes: List[UsdGeom.Tokens], cache: UsdGeom.BBoxCache
) -> None:
    """Updates the bounding box cache for the given list of prims.

    Args:
        prims (List[Usd.Prim]): The list of prims to update the cache for.
        time (Usd.TimeCode): The time at which to compute the bounding boxes.
        purposes (List[UsdGeom.Tokens]): The list of purposes to include in the computation.
        cache (UsdGeom.BBoxCache): The cache to update.
    """
    if not prims:
        return
    if not cache:
        raise ValueError("Invalid cache object")
    cache.SetTime(time)
    cache.SetIncludedPurposes(purposes)
    for prim in prims:
        if not prim.IsValid():
            continue
        if not prim.IsA(UsdGeom.Imageable):
            continue
        try:
            cache.ComputeWorldBound(prim)
        except Exception as e:
            print(f"Error computing bbox for prim {prim.GetPath()}: {str(e)}")


def create_test_cube(stage, path):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    return cube.GetPrim()


def compute_bbox_for_prim_and_descendants(
    prim: Usd.Prim,
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
    purposes_to_include: Sequence[UsdGeom.Tokens] = (UsdGeom.Tokens.default_,),
) -> Gf.BBox3d:
    """Recursively compute the bounding box for a prim and its descendants.

    This function computes the bounding box for a prim and its descendants at the specified
    time code, considering only the prims with the specified purposes.

    Args:
        prim (Usd.Prim): The root prim to compute the bounding box for.
        time_code (Usd.TimeCode): The time code at which to compute the bounding box.
            Defaults to Usd.TimeCode.Default().
        purposes_to_include (Sequence[UsdGeom.Tokens]): The purposes to include when computing
            the bounding box. Defaults to (UsdGeom.Tokens.default_,).

    Returns:
        Gf.BBox3d: The computed bounding box.

    Raises:
        ValueError: If the input prim is not valid.
    """
    if not prim.IsValid():
        raise ValueError("Input prim is not valid")
    bbox_cache = UsdGeom.BBoxCache(time_code, includedPurposes=purposes_to_include)
    bbox = bbox_cache.ComputeWorldBound(prim)
    return bbox


def retarget_basis_curve(source_curve: UsdGeom.BasisCurves, target_curve: UsdGeom.BasisCurves):
    """
    Retarget the vertex count, wrap mode, and basis from the source curve to the target curve.

    Args:
        source_curve (UsdGeom.BasisCurves): The source curve to retarget from.
        target_curve (UsdGeom.BasisCurves): The target curve to retarget to.
    """
    if not source_curve or not source_curve.GetPrim().IsValid():
        raise ValueError("Invalid source curve.")
    if not target_curve or not target_curve.GetPrim().IsValid():
        raise ValueError("Invalid target curve.")
    src_curve_vert_counts_attr = source_curve.GetCurveVertexCountsAttr()
    if src_curve_vert_counts_attr.HasAuthoredValue():
        src_curve_vert_counts = src_curve_vert_counts_attr.Get()
        target_curve.GetCurveVertexCountsAttr().Set(src_curve_vert_counts)
    else:
        target_curve.GetCurveVertexCountsAttr().Clear()
    src_wrap_attr = source_curve.GetWrapAttr()
    if src_wrap_attr.HasAuthoredValue():
        src_wrap = src_wrap_attr.Get()
        target_curve.GetWrapAttr().Set(src_wrap)
    else:
        target_curve.GetWrapAttr().Clear()
    src_basis_attr = source_curve.GetBasisAttr()
    if src_basis_attr.HasAuthoredValue():
        src_basis = src_basis_attr.Get()
        target_curve.GetBasisAttr().Set(src_basis)
    else:
        target_curve.GetBasisAttr().Clear()


def add_velocity_to_basis_curve(
    curve: UsdGeom.BasisCurves, velocities: Vt.Vec3fArray, time_code: Usd.TimeCode = Usd.TimeCode.Default()
):
    """Add velocity attribute to a BasisCurve.

    Args:
        curve (UsdGeom.BasisCurves): The BasisCurve to add velocity to.
        velocities (Vt.Vec3fArray): The velocity values to set.
        time_code (Usd.TimeCode, optional): The timecode to set the value at. Defaults to Default.

    Raises:
        ValueError: If the curve is not a valid BasisCurve.
        ValueError: If the number of velocities does not match the number of points.
    """
    if not curve or not isinstance(curve, UsdGeom.BasisCurves):
        raise ValueError("Invalid BasisCurve")
    points = curve.GetPointsAttr().Get(time_code)
    if not points or len(points) != len(velocities):
        raise ValueError("Number of velocities does not match number of points")
    vel_attr = curve.GetPrim().CreateAttribute("velocities", Sdf.ValueTypeNames.Vector3fArray)
    vel_attr.Set(velocities, time_code)


def compute_basis_curve_length(curve: UsdGeom.BasisCurves, time_code: Usd.TimeCode = Usd.TimeCode.Default()) -> float:
    """Compute the total length of a BasisCurves prim.

    Args:
        curve (UsdGeom.BasisCurves): The BasisCurves prim to compute the length for.
        time_code (Usd.TimeCode, optional): The time code at which to compute the length. Defaults to Usd.TimeCode.Default().

    Returns:
        float: The total length of the curve.
    """
    points_attr = curve.GetPointsAttr()
    if not points_attr.HasValue():
        raise ValueError("The curve has no points defined.")
    points = points_attr.Get(time_code)
    curve_vertex_counts_attr = curve.GetCurveVertexCountsAttr()
    if not curve_vertex_counts_attr.HasValue():
        raise ValueError("The curve has no curve vertex counts defined.")
    curve_vertex_counts = curve_vertex_counts_attr.Get(time_code)
    curve_basis = curve.GetBasisAttr().Get(time_code)
    curve_type = curve.GetTypeAttr().Get(time_code)
    total_length = 0.0
    vertex_index = 0
    for vertex_count in curve_vertex_counts:
        if curve_type == UsdGeom.Tokens.linear:
            for i in range(vertex_index, vertex_index + vertex_count - 1):
                total_length += Gf.Vec3d(points[i] - points[i + 1]).GetLength()
        elif curve_type == UsdGeom.Tokens.cubic:
            if curve_basis == UsdGeom.Tokens.bezier:
                for i in range(vertex_index, vertex_index + vertex_count - 1, 3):
                    (p0, p1, p2, p3) = (points[i], points[i + 1], points[i + 2], points[i + 3])
                    total_length += Gf.Vec3d(p0 - p1).GetLength()
                    total_length += Gf.Vec3d(p1 - p2).GetLength()
                    total_length += Gf.Vec3d(p2 - p3).GetLength()
            elif curve_basis in [UsdGeom.Tokens.bspline, UsdGeom.Tokens.catmullRom]:
                for i in range(vertex_index, vertex_index + vertex_count - 3):
                    (p0, p1, p2, p3) = (points[i], points[i + 1], points[i + 2], points[i + 3])
                    total_length += Gf.Vec3d(p0 - p1).GetLength()
                    total_length += Gf.Vec3d(p1 - p2).GetLength()
                    total_length += Gf.Vec3d(p2 - p3).GetLength()
            else:
                raise ValueError(f"Unsupported curve basis: {curve_basis}")
        else:
            raise ValueError(f"Unsupported curve type: {curve_type}")
        vertex_index += vertex_count
    return total_length


def merge_basis_curves(stage: Usd.Stage, prim_paths: Sequence[str], merged_prim_name: str) -> UsdGeom.BasisCurves:
    """
    Merge multiple BasisCurves prims into a single BasisCurves prim.

    Args:
        stage (Usd.Stage): The stage containing the BasisCurves prims.
        prim_paths (Sequence[str]): The paths of the BasisCurves prims to merge.
        merged_prim_name (str): The name of the new merged BasisCurves prim.

    Returns:
        UsdGeom.BasisCurves: The merged BasisCurves prim.

    Raises:
        ValueError: If any of the input prims are not valid BasisCurves prims.
    """
    merged_prim = UsdGeom.BasisCurves.Define(stage, f"/World/{merged_prim_name}")
    merged_points = []
    merged_widths = []
    merged_curve_vertex_counts = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.BasisCurves):
            raise ValueError(f"Prim at path {prim_path} is not a valid BasisCurves prim.")
        basis_curves = UsdGeom.BasisCurves(prim)
        points = basis_curves.GetPointsAttr().Get()
        widths = basis_curves.GetWidthsAttr().Get()
        curve_vertex_counts = basis_curves.GetCurveVertexCountsAttr().Get()
        merged_points.extend(points)
        merged_widths.extend(widths)
        merged_curve_vertex_counts.extend(curve_vertex_counts)
    merged_prim.GetPointsAttr().Set(merged_points)
    merged_prim.GetWidthsAttr().Set(merged_widths)
    merged_prim.GetCurveVertexCountsAttr().Set(merged_curve_vertex_counts)
    return merged_prim


def create_basis_curves(stage, prim_path, points, widths, curve_vertex_counts):
    prim = UsdGeom.BasisCurves.Define(stage, prim_path)
    prim.GetPointsAttr().Set(points)
    prim.GetWidthsAttr().Set(widths)
    prim.GetCurveVertexCountsAttr().Set(curve_vertex_counts)
    return prim


def smooth_basis_curve(
    stage: Usd.Stage, prim_path: str, curve_points: list[Gf.Vec3f], wrap: str = "nonperiodic", basis: str = "bspline"
) -> UsdGeom.BasisCurves:
    """Create a smooth basis curve from a list of points.

    Args:
        stage (Usd.Stage): The stage to create the curve on.
        prim_path (str): The path where the curve will be created.
        curve_points (list[Gf.Vec3f]): The points defining the curve.
        wrap (str, optional): The wrap mode of the curve. Defaults to "nonperiodic".
        basis (str, optional): The basis type of the curve. Defaults to "bspline".

    Raises:
        ValueError: If wrap or basis is not a valid value, or if insufficient points are provided.

    Returns:
        UsdGeom.BasisCurves: The created basis curve prim.
    """
    if wrap not in ["nonperiodic", "periodic", "pinned"]:
        raise ValueError(f"Invalid wrap mode '{wrap}'. Must be 'nonperiodic', 'periodic', or 'pinned'.")
    if basis not in ["bezier", "bspline", "catmullRom"]:
        raise ValueError(f"Invalid basis type '{basis}'. Must be 'bezier', 'bspline', or 'catmullRom'.")
    if len(curve_points) < 4:
        raise ValueError(f"At least 4 points are required for a cubic {basis} curve. Got {len(curve_points)}.")
    curve = UsdGeom.BasisCurves.Define(stage, prim_path)
    curve.CreateWrapAttr().Set(wrap)
    curve.CreateBasisAttr().Set(basis)
    curve.CreateTypeAttr().Set(UsdGeom.Tokens.cubic)
    curve.CreateCurveVertexCountsAttr().Set([len(curve_points)])
    curve.CreatePointsAttr().Set(Vt.Vec3fArray(curve_points))
    return curve


def set_basis_curve_color(
    curve_prim: UsdGeom.BasisCurves, color: Gf.Vec3f, time_code: Usd.TimeCode = Usd.TimeCode.Default()
):
    """Set the display color for a BasisCurves prim.

    Args:
        curve_prim (UsdGeom.BasisCurves): The BasisCurves prim.
        color (Gf.Vec3f): The RGB color to set.
        time_code (Usd.TimeCode): The time code at which to set the color. Defaults to Default.
    """
    if not curve_prim or not isinstance(curve_prim, UsdGeom.BasisCurves):
        raise ValueError("Invalid BasisCurves prim")
    display_color_attr = curve_prim.GetDisplayColorAttr()
    if not isinstance(color, Gf.Vec3f):
        raise TypeError("Color must be a Gf.Vec3f")
    try:
        display_color_attr.Set(Vt.Vec3fArray([color]), time_code)
    except Tf.ErrorException as e:
        print(f"Error setting display color: {e}")
        raise


def create_ribbon_from_basis_curve(
    stage: Usd.Stage, basis_curve_path: str, ribbon_path: str, ribbon_width: float
) -> UsdGeom.Mesh:
    """
    Create a ribbon mesh from a basis curve.

    Args:
        stage (Usd.Stage): The USD stage.
        basis_curve_path (str): The path to the basis curve prim.
        ribbon_path (str): The path where the ribbon mesh will be created.
        ribbon_width (float): The width of the ribbon.

    Returns:
        UsdGeom.Mesh: The created ribbon mesh prim.
    """
    basis_curve_prim = stage.GetPrimAtPath(basis_curve_path)
    if not basis_curve_prim.IsValid():
        raise ValueError(f"Basis curve prim at path {basis_curve_path} does not exist.")
    if not basis_curve_prim.IsA(UsdGeom.BasisCurves):
        raise TypeError(f"Prim at path {basis_curve_path} is not of type BasisCurves.")
    points_attr = basis_curve_prim.GetAttribute("points")
    widths_attr = basis_curve_prim.GetAttribute("widths")
    if not points_attr.IsValid():
        raise ValueError(f"Basis curve prim at path {basis_curve_path} does not have a 'points' attribute.")
    if not widths_attr.IsValid():
        raise ValueError(f"Basis curve prim at path {basis_curve_path} does not have a 'widths' attribute.")
    points = points_attr.Get(Usd.TimeCode.Default())
    widths = widths_attr.Get(Usd.TimeCode.Default())
    if len(points) != len(widths):
        raise ValueError("Number of points and widths do not match for the basis curve.")
    ribbon_mesh = UsdGeom.Mesh.Define(stage, ribbon_path)
    vertices = []
    face_vertex_counts = []
    face_vertex_indices = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        w0 = widths[i] * ribbon_width
        w1 = widths[i + 1] * ribbon_width
        v0 = p0 + Gf.Vec3f(-w0 / 2, 0, 0)
        v1 = p0 + Gf.Vec3f(w0 / 2, 0, 0)
        v2 = p1 + Gf.Vec3f(w1 / 2, 0, 0)
        v3 = p1 + Gf.Vec3f(-w1 / 2, 0, 0)
        vertices.extend([v0, v1, v2, v3])
        face_vertex_counts.append(4)
        face_vertex_indices.extend([i * 4 + j for j in range(4)])
    ribbon_mesh.CreatePointsAttr().Set(vertices)
    ribbon_mesh.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
    ribbon_mesh.CreateFaceVertexIndicesAttr().Set(face_vertex_indices)
    return ribbon_mesh


def transfer_basis_curve_attributes(src_prim: UsdGeom.BasisCurves, dst_prim: UsdGeom.BasisCurves):
    """Transfer attributes from one BasisCurves prim to another."""
    src_type_attr = src_prim.GetTypeAttr()
    src_basis_attr = src_prim.GetBasisAttr()
    src_wrap_attr = src_prim.GetWrapAttr()
    if not src_type_attr.IsValid():
        raise ValueError("Source prim is missing 'type' attribute")
    if not src_basis_attr.IsValid():
        raise ValueError("Source prim is missing 'basis' attribute")
    if not src_wrap_attr.IsValid():
        raise ValueError("Source prim is missing 'wrap' attribute")
    src_type = src_type_attr.Get()
    src_basis = src_basis_attr.Get()
    src_wrap = src_wrap_attr.Get()
    dst_type_attr = dst_prim.CreateTypeAttr()
    dst_basis_attr = dst_prim.CreateBasisAttr()
    dst_wrap_attr = dst_prim.CreateWrapAttr()
    dst_type_attr.Set(src_type)
    dst_basis_attr.Set(src_basis)
    dst_wrap_attr.Set(src_wrap)


def create_cubic_basis_curve(
    stage: Usd.Stage,
    prim_path: str,
    points: List[Gf.Vec3f],
    widths: Optional[List[float]] = None,
    wrap: str = "nonperiodic",
    basis: str = "bezier",
) -> UsdGeom.BasisCurves:
    """Create a cubic basis curve prim with the given points, widths, wrap mode and basis.

    Args:
        stage (Usd.Stage): The stage to create the curve prim on.
        prim_path (str): The path where the curve prim should be created.
        points (List[Gf.Vec3f]): The points defining the curve.
        widths (Optional[List[float]]): The widths at each point. If None, a constant width of 1.0 is used.
        wrap (str): The wrap mode for the curve. Must be "nonperiodic", "periodic", or "pinned".
        basis (str): The basis for cubic curve interpolation. Must be "bezier", "bspline", or "catmullRom".

    Returns:
        UsdGeom.BasisCurves: The created BasisCurves prim.

    Raises:
        ValueError: If an invalid wrap mode or basis is provided, or if not enough points are provided.
    """
    if wrap not in ["nonperiodic", "periodic", "pinned"]:
        raise ValueError(f'Invalid wrap mode "{wrap}". Must be "nonperiodic", "periodic", or "pinned".')
    if basis not in ["bezier", "bspline", "catmullRom"]:
        raise ValueError(f'Invalid basis "{basis}". Must be "bezier", "bspline", or "catmullRom".')
    if len(points) < 4:
        raise ValueError(f"At least 4 points are required for a cubic curve, got {len(points)}")
    curves = UsdGeom.BasisCurves.Define(stage, prim_path)
    curves.CreateTypeAttr().Set("cubic")
    curves.CreateWrapAttr().Set(wrap)
    curves.CreateBasisAttr().Set(basis)
    curves.CreatePointsAttr().Set(points)
    curves.CreateCurveVertexCountsAttr().Set([len(points)])
    if widths is None:
        widths = [1.0] * len(points)
    curves.CreateWidthsAttr().Set(widths)
    return curves


def animate_basis_curve(
    stage: Usd.Stage,
    prim_path: str,
    points: List[Gf.Vec3f],
    widths: List[float],
    wrap: str = "nonperiodic",
    basis: str = "bezier",
    time_range: Tuple[float, float] = (0.0, 1.0),
    time_samples: int = 10,
) -> None:
    """Animate a basis curve over time with changing points and widths.

    Args:
        stage (Usd.Stage): The stage to create the curve in.
        prim_path (str): The path where the curve will be created.
        points (List[Gf.Vec3f]): The points defining the curve.
        widths (List[float]): The widths at each point.
        wrap (str): The wrapping mode of the curve. Defaults to "nonperiodic".
        basis (str): The basis type of the curve. Defaults to "bezier".
        time_range (Tuple[float, float]): The range of time to animate over. Defaults to (0, 1).
        time_samples (int): The number of time samples over the range. Defaults to 10.

    Raises:
        ValueError: If wrap or basis is not a valid value, or time range is invalid.
    """
    if wrap not in ["nonperiodic", "periodic", "pinned"]:
        raise ValueError(f'Invalid wrap mode "{wrap}"')
    if basis not in ["bezier", "bspline", "catmullRom"]:
        raise ValueError(f'Invalid basis "{basis}"')
    if time_range[1] <= time_range[0]:
        raise ValueError(f"Invalid time range {time_range}")
    curve = UsdGeom.BasisCurves.Define(stage, prim_path)
    curve.CreateWrapAttr().Set(wrap)
    curve.CreateBasisAttr().Set(basis)
    curve.CreateTypeAttr().Set("cubic")
    times = [
        Usd.TimeCode(t)
        for t in [time_range[0] + i * (time_range[1] - time_range[0]) / (time_samples - 1) for i in range(time_samples)]
    ]
    for time in times:
        curve.CreatePointsAttr().Set(points, time)
        curve.CreateWidthsAttr().Set(widths, time)
        points = [p + Gf.Vec3f(0.1, 0.1, 0.1) for p in points]
        widths = [w + 0.1 for w in widths]


def propagate_extent_changes(boundable: UsdGeom.Boundable, time: Usd.TimeCode) -> None:
    """Propagate the extent change of a boundable prim to its ancestors."""
    extent = boundable.GetExtentAttr().Get(time)
    if not extent:
        if not UsdGeom.Boundable.ComputeExtentFromPlugins(boundable, time, Gf.Matrix4d(1), extent):
            return
    parent = boundable.GetPrim().GetParent()
    while parent:
        parent_boundable = UsdGeom.Boundable(parent)
        if parent_boundable:
            parent_extent = parent_boundable.GetExtentAttr().Get(time)
            if parent_extent:
                Gf.Range3d(parent_extent).UnionWith(Gf.Range3d(extent))
                parent_boundable.GetExtentAttr().Set(parent_extent, time)
            else:
                parent_boundable.GetExtentAttr().Set(extent, time)
        parent = parent.GetParent()


def compare_cached_and_computed_extent(boundable: UsdGeom.Boundable, time_code: Usd.TimeCode) -> bool:
    """
    Compare the cached extent of a Boundable prim with the computed extent.

    Args:
        boundable (UsdGeom.Boundable): The Boundable prim to compare extents for.
        time_code (Usd.TimeCode): The time code at which to compare the extents.

    Returns:
        bool: True if the cached and computed extents are equal, False otherwise.
    """
    if not boundable or not boundable.GetPrim().IsValid():
        raise ValueError("Invalid Boundable prim")
    extent_attr = boundable.GetExtentAttr()
    if extent_attr.HasValue():
        cached_extent = extent_attr.Get(time_code)
    else:
        return False
    computed_extent = Gf.Vec3fArray(2)
    if not UsdGeom.Boundable.ComputeExtentFromPlugins(boundable, time_code, Gf.Matrix4d(1), computed_extent):
        return False
    is_equal = cached_extent == computed_extent
    return is_equal


def aggregate_extents_by_material(
    prim: Usd.Prim, time: Usd.TimeCode = Usd.TimeCode.Default()
) -> Dict[Sdf.Path, Gf.Range3d]:
    """Aggregate the extents of a prim and its descendants by material binding.

    This function traverses the prim hierarchy starting from the given prim and
    aggregates the extents of all prims that have a material binding. The extents
    are stored in a dictionary where the keys are the material paths and the values
    are the aggregated extents for each material.

    Args:
        prim (Usd.Prim): The root prim to start the traversal from.
        time (Usd.TimeCode): The time at which to compute the extents. Defaults to Usd.TimeCode.Default().

    Returns:
        Dict[Sdf.Path, Gf.Range3d]: A dictionary mapping material paths to their aggregated extents.
    """
    extents_by_material: Dict[Sdf.Path, Gf.Range3d] = {}
    for descendant in Usd.PrimRange(prim):
        if not descendant.IsA(UsdGeom.Boundable):
            continue
        material_binding = UsdShade.MaterialBindingAPI(descendant)
        direct_binding_targets = material_binding.GetDirectBindingRel().GetTargets()
        if not direct_binding_targets:
            continue
        material_path = direct_binding_targets[0]
        boundable = UsdGeom.Boundable(descendant)
        extent = boundable.ComputeExtent(time)
        if len(extent) == 0:
            continue
        extent_range = Gf.Range3d(Gf.Vec3d(extent[0]), Gf.Vec3d(extent[1]))
        if material_path in extents_by_material:
            extents_by_material[material_path] = extents_by_material[material_path].UnionWith(extent_range)
        else:
            extents_by_material[material_path] = extent_range
    return extents_by_material


def get_camera_view_matrix(
    stage: Usd.Stage, camera_path: str, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> Gf.Matrix4d:
    """
    Get the view matrix for a camera at a specific time.

    Args:
        stage (Usd.Stage): The USD stage.
        camera_path (str): The path to the camera prim.
        time_code (Usd.TimeCode): The time code at which to get the view matrix. Defaults to Default time.

    Returns:
        Gf.Matrix4d: The view matrix of the camera.

    Raises:
        ValueError: If the camera prim is not valid or if it is not a UsdGeomCamera.
    """
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim.IsValid():
        raise ValueError(f"Invalid camera prim path: {camera_path}")
    usd_camera = UsdGeom.Camera(camera_prim)
    if not usd_camera:
        raise ValueError(f"Prim at path {camera_path} is not a valid UsdGeomCamera")
    local_to_world_matrix = usd_camera.ComputeLocalToWorldTransform(time_code)
    view_matrix = local_to_world_matrix.GetInverse()
    return view_matrix


def set_camera_focal_length(
    camera: UsdGeom.Camera, focal_length: float, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> None:
    """Set the focal length of a USD camera.

    Args:
        camera (UsdGeom.Camera): The camera to set the focal length on.
        focal_length (float): The focal length value to set, in millimeters.
        time_code (Usd.TimeCode, optional): The time code to set the value at. Defaults to Default().

    Raises:
        ValueError: If the input camera is not a valid UsdGeom.Camera.
    """
    if not camera or not isinstance(camera, UsdGeom.Camera):
        raise ValueError("Invalid camera object")
    focal_length_attr = camera.GetFocalLengthAttr()
    focal_length_attr.Set(focal_length, time_code)


def set_camera_clipping_planes(
    camera: UsdGeom.Camera, clipping_planes: List[Gf.Vec4f], time_code: Usd.TimeCode = Usd.TimeCode.Default()
):
    """Set the clipping planes for a USD camera.

    Args:
        camera (UsdGeom.Camera): The camera to set clipping planes on.
        clipping_planes (List[Gf.Vec4f]): The list of clipping planes as Vec4f.
        time_code (Usd.TimeCode, optional): The time code to set the value at. Defaults to Default time.
    """
    clipping_planes_attr = camera.GetClippingPlanesAttr()
    if not clipping_planes_attr:
        clipping_planes_attr = camera.CreateClippingPlanesAttr([], Usd.TimeCode.Default())
    clipping_planes_attr.Set(clipping_planes, time_code)


def animate_camera_transform(
    stage: Usd.Stage,
    camera_path: str,
    translate_values: List[Tuple[float, float, float]],
    rotate_values: List[Tuple[float, float, float]],
    timecodes: List[Usd.TimeCode],
) -> None:
    """Animate the transformation of a camera over time.

    Args:
        stage (Usd.Stage): The USD stage.
        camera_path (str): The path to the camera prim.
        translate_values (List[Tuple[float, float, float]]): List of translation values.
        rotate_values (List[Tuple[float, float, float]]): List of rotation values in degrees.
        timecodes (List[Usd.TimeCode]): Corresponding timecodes for each transform value.

    Raises:
        ValueError: If the input lists have different lengths.
    """
    if len(translate_values) != len(rotate_values) or len(translate_values) != len(timecodes):
        raise ValueError("Length of input lists must be equal.")

    camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))

    # Clear any existing transform ops
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()

    # Create new transform ops for translation and rotation
    translate_op = add_translate_op(xform)
    rotate_op = add_rotate_xyz_op(xform)

    # Set the transform values at each timecode
    for i in range(len(timecodes)):
        translate_op.Set(time=timecodes[i], value=Gf.Vec3d(*translate_values[i]))
        rotate_op.Set(time=timecodes[i], value=Gf.Vec3f(*rotate_values[i]))


def convert_gf_camera_to_usd(
    gf_camera: Gf.Camera, usd_camera: UsdGeom.Camera, time_code: Usd.TimeCode = Usd.TimeCode.Default()
):
    """
    Convert a GfCamera object to a UsdGeomCamera prim.

    Args:
        gf_camera (Gf.Camera): The input GfCamera object.
        usd_camera (UsdGeom.Camera): The UsdGeomCamera prim to update.
        time_code (Usd.TimeCode): The time code at which to set the camera attributes. Defaults to Default time.
    """
    projection = (
        UsdGeom.Tokens.perspective if gf_camera.projection == Gf.Camera.Perspective else UsdGeom.Tokens.orthographic
    )
    usd_camera.CreateProjectionAttr().Set(projection, time_code)
    usd_camera.CreateHorizontalApertureAttr().Set(gf_camera.horizontalAperture, time_code)
    usd_camera.CreateHorizontalApertureOffsetAttr().Set(gf_camera.horizontalApertureOffset, time_code)
    usd_camera.CreateVerticalApertureAttr().Set(gf_camera.verticalAperture, time_code)
    usd_camera.CreateVerticalApertureOffsetAttr().Set(gf_camera.verticalApertureOffset, time_code)
    usd_camera.CreateFocalLengthAttr().Set(gf_camera.focalLength, time_code)
    usd_camera.CreateClippingRangeAttr().Set(
        Gf.Vec2f(gf_camera.clippingRange.min, gf_camera.clippingRange.max), time_code
    )
    usd_camera.CreateClippingPlanesAttr().Set(gf_camera.clippingPlanes, time_code)
    usd_camera.CreateFStopAttr().Set(gf_camera.fStop, time_code)
    usd_camera.CreateFocusDistanceAttr().Set(gf_camera.focusDistance, time_code)
    usd_camera.SetFromCamera(gf_camera, time_code)


def set_camera_shutter_times(camera: UsdGeom.Camera, shutter_open: float, shutter_close: float) -> None:
    """Set the shutter open and close times for a USD camera.

    Args:
        camera (UsdGeom.Camera): The camera prim to set the shutter times on.
        shutter_open (float): The shutter open time in UsdTimeCode units relative to the current frame.
                              Negative value means shutter opens before current frame time.
        shutter_close (float): The shutter close time in UsdTimeCode units relative to the current frame.
                               Must be greater than or equal to the shutter open time.

    Raises:
        ValueError: If the camera prim is not valid or if shutter_close < shutter_open.
    """
    if not camera.GetPrim().IsValid():
        raise ValueError("Provided camera prim is not valid")
    if shutter_close < shutter_open:
        raise ValueError(f"shutter_close ({shutter_close}) must be >= shutter_open ({shutter_open})")
    shutter_open_attr = camera.GetShutterOpenAttr()
    shutter_close_attr = camera.GetShutterCloseAttr()
    shutter_open_attr.Set(shutter_open)
    shutter_close_attr.Set(shutter_close)


def batch_update_camera_attributes(
    camera: UsdGeom.Camera, attributes: Dict[str, Any], time_code: Usd.TimeCode = Usd.TimeCode.Default()
):
    """
    Update multiple attributes of a UsdGeom.Camera in a single transaction.

    Args:
        camera (UsdGeom.Camera): The camera to update attributes for.
        attributes (Dict[str, Any]): A dictionary mapping attribute names to their new values.
        time_code (Usd.TimeCode, optional): The time code at which to set the attribute values. Defaults to Default time code.
    """
    if not camera or not isinstance(camera, UsdGeom.Camera):
        raise ValueError("Invalid UsdGeom.Camera object provided.")
    attr_dict = {}
    for attr_name, attr_value in attributes.items():
        attr = camera.GetPrim().GetAttribute(attr_name)
        if not attr.IsDefined():
            if isinstance(attr_value, float):
                attr = camera.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
            elif isinstance(attr_value, Gf.Vec2f):
                attr = camera.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float2)
            elif isinstance(attr_value, Gf.Vec3f):
                attr = camera.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float3)
            elif isinstance(attr_value, Gf.Vec4f):
                attr = camera.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float4)
            elif isinstance(attr_value, list) and attr_value and isinstance(attr_value[0], Gf.Vec4f):
                attr = camera.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float4Array)
            elif isinstance(attr_value, str):
                attr = camera.CreateAttribute(attr_name, Sdf.ValueTypeNames.String)
            elif isinstance(attr_value, Sdf.AssetPath):
                attr = camera.CreateAttribute(attr_name, Sdf.ValueTypeNames.Asset)
            elif isinstance(attr_value, bool):
                attr = camera.CreateAttribute(attr_name, Sdf.ValueTypeNames.Bool)
            else:
                raise ValueError(f"Unsupported attribute value type for attribute '{attr_name}': {type(attr_value)}")
        attr_dict[attr_name] = attr
    with Sdf.ChangeBlock():
        for attr_name, attr in attr_dict.items():
            attr_value = attributes[attr_name]
            attr.Set(attr_value, time_code)


def set_camera_dof_parameters(camera: UsdGeom.Camera, fstop: float, focus_distance: float, focal_length: float):
    """Set depth of field parameters on a USD camera.

    Args:
        camera (UsdGeom.Camera): The camera to set DOF parameters on.
        fstop (float): The f-stop value (lens aperture). Set to 0 to disable DOF.
        focus_distance (float): The focus distance in world units.
        focal_length (float): The focal length in millimeters.

    Raises:
        ValueError: If the input camera is not a valid UsdGeom.Camera prim.
    """
    if not camera or not camera.GetPrim().IsA(UsdGeom.Camera):
        raise ValueError("Invalid camera prim")
    fstop_attr = camera.GetFStopAttr()
    if fstop_attr:
        fstop_attr.Set(fstop)
    else:
        camera.CreateFStopAttr().Set(fstop)
    focus_dist_attr = camera.GetFocusDistanceAttr()
    if focus_dist_attr:
        focus_dist_attr.Set(focus_distance)
    else:
        camera.CreateFocusDistanceAttr().Set(focus_distance)
    focal_len_attr = camera.GetFocalLengthAttr()
    if focal_len_attr:
        focal_len_attr.Set(focal_length)
    else:
        camera.CreateFocalLengthAttr().Set(focal_length)


def get_camera_aperture_size(
    camera: UsdGeom.Camera, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> Tuple[float, float]:
    """Get the horizontal and vertical aperture size of a camera.

    Args:
        camera (UsdGeom.Camera): The camera to get the aperture size from.
        time_code (Usd.TimeCode): The time code to evaluate the attributes at. Defaults to Default.

    Returns:
        Tuple[float, float]: A tuple with (horizontal_aperture, vertical_aperture) in generic scene units.
    """
    if not camera or not camera.GetPrim().IsA(UsdGeom.Camera):
        raise ValueError("Invalid camera object")
    horizontal_aperture = 0.0
    vertical_aperture = 0.0
    if camera.GetHorizontalApertureAttr().HasValue():
        horizontal_aperture = camera.GetHorizontalApertureAttr().Get(time_code)
    if camera.GetVerticalApertureAttr().HasValue():
        vertical_aperture = camera.GetVerticalApertureAttr().Get(time_code)
    return (horizontal_aperture, vertical_aperture)


def set_camera_shutter_open_close(camera: UsdGeom.Camera, shutter_open: float, shutter_close: float) -> None:
    """Set the shutter open and close attributes on a USD camera.

    Args:
        camera (UsdGeom.Camera): The camera to set the shutter attributes on.
        shutter_open (float): The shutter open time in frames relative to the current frame.
                              Negative values open the shutter before the current frame.
        shutter_close (float): The shutter close time in frames relative to the current frame.
                               Must be greater than or equal to shutter_open.

    Raises:
        ValueError: If shutter_close is less than shutter_open.
    """
    if shutter_close < shutter_open:
        raise ValueError(
            f"shutter_close ({shutter_close}) must be greater than or equal to shutter_open ({shutter_open})"
        )
    shutter_open_attr = camera.GetShutterOpenAttr()
    shutter_close_attr = camera.GetShutterCloseAttr()
    shutter_open_attr.Set(shutter_open)
    shutter_close_attr.Set(shutter_close)


def animate_camera_focal_length(stage: Usd.Stage, camera_path: str, focal_lengths: Dict[Usd.TimeCode, float]):
    """Animate the focal length of a camera over time.

    Args:
        stage (Usd.Stage): The USD stage.
        camera_path (str): The path to the camera prim.
        focal_lengths (Dict[Usd.TimeCode, float]): A dictionary mapping time codes to focal length values.
    """
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim:
        raise ValueError(f"No camera prim found at path: {camera_path}")
    camera = UsdGeom.Camera(camera_prim)
    if not camera:
        raise ValueError(f"Prim at path {camera_path} is not a valid camera")
    focal_length_attr = camera.GetFocalLengthAttr()
    if not focal_length_attr:
        raise ValueError(f"Camera at path {camera_path} does not have a focalLength attribute")
    for time_code, focal_length in focal_lengths.items():
        focal_length_attr.Set(focal_length, time_code)


def set_camera_stereo_role(camera_prim: UsdGeom.Camera, role: str) -> None:
    """Set the stereo role for a camera prim.

    Args:
        camera_prim (UsdGeom.Camera): The camera prim to set the stereo role for.
        role (str): The stereo role to set. Must be one of 'mono', 'left', or 'right'.

    Raises:
        ValueError: If the provided role is not one of the allowed values.
    """
    allowed_roles = ["mono", "left", "right"]
    if role not in allowed_roles:
        raise ValueError(f"Invalid stereo role '{role}'. Must be one of {allowed_roles}")
    stereo_role_attr = camera_prim.GetStereoRoleAttr()
    stereo_role_attr.Set(role)


def copy_camera_parameters(
    src_camera: UsdGeom.Camera, dst_camera: UsdGeom.Camera, time_code: Usd.TimeCode = Usd.TimeCode.Default()
):
    """Copy camera parameters from source to destination camera at a specific time.

    Parameters:
        src_camera (UsdGeom.Camera): The source camera to copy parameters from.
        dst_camera (UsdGeom.Camera): The destination camera to copy parameters to.
        time_code (Usd.TimeCode): The time code to copy the parameters at. Defaults to Default time.
    """
    if not src_camera.GetPrim().IsValid():
        raise ValueError("Source camera prim is not valid.")
    if not dst_camera.GetPrim().IsValid():
        raise ValueError("Destination camera prim is not valid.")
    dst_camera.GetProjectionAttr().Set(src_camera.GetProjectionAttr().Get(time_code), time_code)
    dst_camera.GetHorizontalApertureAttr().Set(src_camera.GetHorizontalApertureAttr().Get(time_code), time_code)
    dst_camera.GetVerticalApertureAttr().Set(src_camera.GetVerticalApertureAttr().Get(time_code), time_code)
    dst_camera.GetHorizontalApertureOffsetAttr().Set(
        src_camera.GetHorizontalApertureOffsetAttr().Get(time_code), time_code
    )
    dst_camera.GetVerticalApertureOffsetAttr().Set(src_camera.GetVerticalApertureOffsetAttr().Get(time_code), time_code)
    dst_camera.GetFocalLengthAttr().Set(src_camera.GetFocalLengthAttr().Get(time_code), time_code)
    dst_camera.GetClippingRangeAttr().Set(src_camera.GetClippingRangeAttr().Get(time_code), time_code)
    dst_camera.GetClippingPlanesAttr().Set(src_camera.GetClippingPlanesAttr().Get(time_code), time_code)
    dst_camera.GetFStopAttr().Set(src_camera.GetFStopAttr().Get(time_code), time_code)
    dst_camera.GetFocusDistanceAttr().Set(src_camera.GetFocusDistanceAttr().Get(time_code), time_code)


def set_camera_resolution(
    camera: UsdGeom.Camera, resolution: Tuple[float, float], time_code: Usd.TimeCode = Usd.TimeCode.Default()
):
    """Sets the horizontal and vertical aperture to achieve the desired pixel resolution"""
    horizontal_aperture = resolution[0] / 10.0
    vertical_aperture = resolution[1] / 10.0
    camera.CreateHorizontalApertureAttr().Set(horizontal_aperture, time_code)
    camera.CreateVerticalApertureAttr().Set(vertical_aperture, time_code)


def set_camera_aperture(camera: UsdGeom.Camera, horizontal_aperture: float, vertical_aperture: float) -> None:
    """Set the horizontal and vertical aperture of a USD camera.

    Args:
        camera (UsdGeom.Camera): The USD camera prim to modify.
        horizontal_aperture (float): The horizontal aperture value in tenths of scene units.
        vertical_aperture (float): The vertical aperture value in tenths of scene units.

    Raises:
        ValueError: If the input camera prim is not a valid UsdGeom.Camera.
    """
    if not camera or not isinstance(camera, UsdGeom.Camera):
        raise ValueError("Invalid UsdGeom.Camera prim")
    camera.CreateHorizontalApertureAttr().Set(horizontal_aperture)
    camera.CreateVerticalApertureAttr().Set(vertical_aperture)


def reset_camera_transform(camera: UsdGeom.Camera, time_code: Usd.TimeCode = Usd.TimeCode.Default()):
    """Reset the camera's transform to the default identity transform.

    Args:
        camera (UsdGeom.Camera): The camera prim to reset the transform on.
        time_code (Usd.TimeCode, optional): The time code to set the transform at. Defaults to Default time code.
    """
    transform_attr = camera.GetPrim().GetAttribute("xformOp:transform")
    if not transform_attr.IsDefined():
        UsdGeom.XformCommonAPI(camera).SetXformVectors(
            translation=Gf.Vec3d(0, 0, 0),
            rotation=Gf.Vec3f(0, 0, 0),
            scale=Gf.Vec3f(1, 1, 1),
            pivot=Gf.Vec3f(0, 0, 0),
            rotationOrder=UsdGeom.XformCommonAPI.RotationOrderXYZ,
            time=time_code,
        )
    else:
        transform_attr.Clear()
    camera.GetOrderedXformOps()
    xform_op_order_attr = camera.GetPrim().GetAttribute("xformOpOrder")
    xform_op_order_attr.Clear()
    xform_op_order_attr.Set(["xformOp:transform"])


def create_stereo_camera_pair(
    stage: Usd.Stage, path_base: str, interocular_distance: float = 0.0635
) -> Tuple[UsdGeom.Camera, UsdGeom.Camera]:
    """
    Create a stereo camera pair on the given stage.

    Args:
        stage (Usd.Stage): The stage to create the cameras on.
        path_base (str): The base path for the camera prims, e.g., "/World/Cameras".
        interocular_distance (float): The distance between the left and right cameras. Defaults to 0.0635 (6.35 cm).

    Returns:
        Tuple[UsdGeom.Camera, UsdGeom.Camera]: The left and right camera prims.
    """
    if not Sdf.Path(path_base).IsAbsolutePath():
        raise ValueError(f"Path '{path_base}' is not an absolute path.")
    left_cam_path = f"{path_base}/LeftCamera"
    left_camera = UsdGeom.Camera.Define(stage, left_cam_path)
    left_camera.CreateStereoRoleAttr().Set(UsdGeom.Tokens.left)
    right_cam_path = f"{path_base}/RightCamera"
    right_camera = UsdGeom.Camera.Define(stage, right_cam_path)
    right_camera.CreateStereoRoleAttr().Set(UsdGeom.Tokens.right)
    half_dist = interocular_distance / 2.0
    UsdGeom.XformCommonAPI(left_camera).SetTranslate((-half_dist, 0, 0))
    UsdGeom.XformCommonAPI(right_camera).SetTranslate((half_dist, 0, 0))
    return (left_camera, right_camera)


def remove_camera_clipping_plane(camera: UsdGeom.Camera, index: int) -> bool:
    """Remove a clipping plane from a camera at the specified index.

    Args:
        camera (UsdGeom.Camera): The camera to remove the clipping plane from.
        index (int): The index of the clipping plane to remove.

    Returns:
        bool: True if the clipping plane was removed, False otherwise.
    """
    clipping_planes_attr = camera.GetClippingPlanesAttr()
    if not clipping_planes_attr.IsValid():
        return False
    clipping_planes = clipping_planes_attr.Get()
    if index < 0 or index >= len(clipping_planes):
        return False
    updated_clipping_planes = clipping_planes[:index] + clipping_planes[index + 1 :]
    clipping_planes_attr.Set(updated_clipping_planes)
    return True


def get_camera_clipping_range(
    camera: UsdGeom.Camera, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> Tuple[float, float]:
    """
    Get the near and far clipping plane distances for a USD camera.

    Args:
        camera (UsdGeom.Camera): The USD camera prim.
        time_code (Usd.TimeCode): The time code at which to retrieve the value. Defaults to Default.

    Returns:
        Tuple[float, float]: A tuple with (near, far) clipping plane distances.
    """
    clipping_range_attr = camera.GetClippingRangeAttr()
    if clipping_range_attr and clipping_range_attr.HasAuthoredValue():
        clipping_range = clipping_range_attr.Get(time_code)
        return (clipping_range[0], clipping_range[1])
    else:
        return (1.0, 1000000.0)


def set_camera_orientation(
    camera: UsdGeom.Camera, rotation_euler: Tuple[float, float, float], time_code: Usd.TimeCode = Usd.TimeCode.Default()
):
    """Set the orientation of a USD camera using Euler angles.

    Args:
        camera (UsdGeom.Camera): The USD camera prim.
        rotation_euler (Tuple[float, float, float]): The rotation in degrees as Euler angles (X, Y, Z).
        time_code (Usd.TimeCode): The time code for the orientation. Defaults to Default time code.
    """
    rotation_euler_rad = tuple((math.radians(angle) for angle in rotation_euler))
    rotation_matrix = Gf.Matrix3d(
        Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation_euler_rad[0])
        * Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation_euler_rad[1])
        * Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation_euler_rad[2])
    )
    transform_matrix = Gf.Matrix4d().SetIdentity()
    transform_matrix.SetRotateOnly(rotation_matrix)
    camera_prim = camera.GetPrim()
    xformable = UsdGeom.Xformable(camera_prim)
    xformable.ClearXformOpOrder()
    xform_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTransform)
    xform_op.Set(transform_matrix, time_code)


def add_camera_clipping_plane(
    camera: UsdGeom.Camera, plane: Gf.Vec4d, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> None:
    """Add a clipping plane to a USD camera.

    Args:
        camera (UsdGeom.Camera): The camera to add the clipping plane to.
        plane (Gf.Vec4d): The clipping plane to add, encoded as (a, b, c, d) where
            the plane is defined by a*x + b*y + c*z + d*1 < 0.
        time_code (Usd.TimeCode, optional): The time code to set the value at.
            Defaults to Default() (current time).
    """
    if not camera or not isinstance(camera, UsdGeom.Camera):
        raise ValueError("Invalid camera object provided.")
    clipping_planes_attr = camera.GetClippingPlanesAttr()
    clipping_planes = clipping_planes_attr.Get(time_code)
    if clipping_planes is None:
        clipping_planes = Vt.Vec4fArray([Gf.Vec4f(plane)])
    else:
        clipping_planes_list = list(clipping_planes)
        clipping_planes_list.append(Gf.Vec4f(plane))
        clipping_planes = Vt.Vec4fArray(clipping_planes_list)
    clipping_planes_attr.Set(clipping_planes, time_code)


def create_camera_with_defaults(stage: Usd.Stage, prim_path: str) -> UsdGeom.Camera:
    """Create a camera prim with default settings.

    Args:
        stage (Usd.Stage): The USD stage to create the camera in.
        prim_path (str): The path where the camera prim should be created.

    Returns:
        UsdGeom.Camera: The created camera prim.
    """
    camera = UsdGeom.Camera.Define(stage, Sdf.Path(prim_path))
    camera.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
    camera.CreateHorizontalApertureAttr().Set(20.955)
    camera.CreateVerticalApertureAttr().Set(15.2908)
    camera.CreateFocalLengthAttr().Set(50.0)
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(1, 1000000))
    camera.CreateFStopAttr().Set(0.0)
    camera.CreateFocusDistanceAttr().Set(0.0)
    return camera


def set_camera_intrinsics(
    usd_camera: UsdGeom.Camera, focal_length: float, horizontal_aperture: float, vertical_aperture: float
) -> None:
    """Set the camera intrinsics for a USD camera.

    Args:
        usd_camera (UsdGeom.Camera): The USD camera prim to set intrinsics for.
        focal_length (float): The focal length in millimeters.
        horizontal_aperture (float): The horizontal aperture in millimeters.
        vertical_aperture (float): The vertical aperture in millimeters.
    """
    if not usd_camera:
        raise ValueError("Invalid UsdGeom.Camera provided.")
    focal_length_attr = usd_camera.GetFocalLengthAttr()
    if focal_length_attr:
        focal_length_value = focal_length * 10.0
        focal_length_attr.Set(focal_length_value)
    else:
        print(f"Warning: Failed to set focal length for camera {usd_camera.GetPath()}")
    horizontal_aperture_attr = usd_camera.GetHorizontalApertureAttr()
    if horizontal_aperture_attr:
        horizontal_aperture_value = horizontal_aperture * 10.0
        horizontal_aperture_attr.Set(horizontal_aperture_value)
    else:
        print(f"Warning: Failed to set horizontal aperture for camera {usd_camera.GetPath()}")
    vertical_aperture_attr = usd_camera.GetVerticalApertureAttr()
    if vertical_aperture_attr:
        vertical_aperture_value = vertical_aperture * 10.0
        vertical_aperture_attr.Set(vertical_aperture_value)
    else:
        print(f"Warning: Failed to set vertical aperture for camera {usd_camera.GetPath()}")


def toggle_camera_stereo_mode(camera: UsdGeom.Camera) -> None:
    """Toggle the camera's stereo role between mono and left/right stereo pair."""
    stereo_role_attr = camera.GetStereoRoleAttr()
    if not stereo_role_attr.IsValid():
        stereo_role_attr = camera.CreateStereoRoleAttr(Sdf.ValueTypeNames.Token, True)
        stereo_role_attr.Set(UsdGeom.Tokens.mono)
    current_role = stereo_role_attr.Get()
    if current_role == UsdGeom.Tokens.mono:
        stereo_role_attr.Set(UsdGeom.Tokens.left)
        stage = camera.GetPrim().GetStage()
        camera_parent_path = camera.GetPath().GetParentPath()
        camera_name = camera.GetPrim().GetName()
        right_camera_path = camera_parent_path.AppendChild(f"{camera_name}_right")
        right_camera = UsdGeom.Camera.Define(stage, right_camera_path)
        right_camera.GetStereoRoleAttr().Set(UsdGeom.Tokens.right)
    else:
        stereo_role_attr.Set(UsdGeom.Tokens.mono)
        stage = camera.GetPrim().GetStage()
        camera_parent_path = camera.GetPath().GetParentPath()
        camera_name = camera.GetPrim().GetName()
        right_camera_name = f"{camera_name}_right"
        right_camera_path = camera_parent_path.AppendChild(right_camera_name)
        right_camera_prim = stage.GetPrimAtPath(right_camera_path)
        if right_camera_prim.IsValid():
            stage.RemovePrim(right_camera_path)


def configure_camera_for_render(
    camera: UsdGeom.Camera,
    aperture: Tuple[float, float] = (20.955, 15.2908),
    translate: Gf.Vec3d = Gf.Vec3d(0.0),
    rotate: Gf.Vec3f = Gf.Vec3f(0.0),
    focal_length: float = 50.0,
    focus_distance: float = 0.0,
    f_stop: float = 0.0,
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Configure a USD camera for rendering by setting common attributes.

    Args:
        camera (UsdGeom.Camera): The camera prim to configure.
        aperture (Tuple[float, float], optional): Horizontal and vertical aperture. Defaults to (20.955, 15.2908).
        translate (Gf.Vec3d, optional): Camera translation. Defaults to Gf.Vec3d(0.0).
        rotate (Gf.Vec3f, optional): Camera rotation in degrees Defaults to Gf.Vec3f(0.0).
        focal_length (float, optional): Focal length in mm. Defaults to 50.0.
        focus_distance (float, optional): Focus distance. Defaults to 0.0 (no focus).
        f_stop (float, optional): F-stop value for depth of field. Defaults to 0.0 (no depth of field).
        time_code (Usd.TimeCode, optional): Time code to set the attributes at. Defaults to Default time.

    Raises:
        ValueError: If the input camera prim is not a valid UsdGeom.Camera.
    """
    if not camera or not isinstance(camera, UsdGeom.Camera):
        raise ValueError("Invalid camera prim provided.")
    camera.CreateHorizontalApertureAttr().Set(aperture[0], time_code)
    camera.CreateVerticalApertureAttr().Set(aperture[1], time_code)
    UsdGeom.XformCommonAPI(camera).SetTranslate(translate, time_code)
    UsdGeom.XformCommonAPI(camera).SetRotate(rotate, UsdGeom.XformCommonAPI.RotationOrderXYZ, time_code)
    camera.CreateFocalLengthAttr().Set(focal_length, time_code)
    if focus_distance > 0.0:
        camera.CreateFocusDistanceAttr().Set(focus_distance, time_code)
    if f_stop > 0.0:
        camera.CreateFStopAttr().Set(f_stop, time_code)


def convert_camera_to_gf_camera(
    usd_camera: UsdGeom.Camera, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> Gf.Camera:
    """Convert a USD camera to a GfCamera object at the specified time code."""
    gf_camera = Gf.Camera()
    projection = usd_camera.GetProjectionAttr().Get(time_code)
    if projection == UsdGeom.Tokens.perspective:
        gf_camera.projection = Gf.Camera.Perspective
    elif projection == UsdGeom.Tokens.orthographic:
        gf_camera.projection = Gf.Camera.Orthographic
    else:
        raise ValueError(f"Unsupported projection type: {projection}")
    gf_camera.horizontalAperture = usd_camera.GetHorizontalApertureAttr().Get(time_code)
    gf_camera.verticalAperture = usd_camera.GetVerticalApertureAttr().Get(time_code)
    gf_camera.horizontalApertureOffset = usd_camera.GetHorizontalApertureOffsetAttr().Get(time_code)
    gf_camera.verticalApertureOffset = usd_camera.GetVerticalApertureOffsetAttr().Get(time_code)
    gf_camera.focalLength = usd_camera.GetFocalLengthAttr().Get(time_code)
    clipping_range = usd_camera.GetClippingRangeAttr().Get(time_code)
    gf_camera.clippingRange = Gf.Range1f(clipping_range[0], clipping_range[1])
    clipping_planes = usd_camera.GetClippingPlanesAttr().Get(time_code)
    gf_camera.clippingPlanes = [Gf.Vec4f(p[0], p[1], p[2], p[3]) for p in clipping_planes]
    gf_camera.fStop = usd_camera.GetFStopAttr().Get(time_code)
    gf_camera.focusDistance = usd_camera.GetFocusDistanceAttr().Get(time_code)
    transform_attr = usd_camera.GetPrim().GetAttribute("xformOp:transform")
    if transform_attr.IsValid():
        transform_matrix = Gf.Matrix4d(transform_attr.Get(time_code))
        gf_camera.transform = transform_matrix
    return gf_camera


def align_camera_look_at(camera_prim: UsdGeom.Camera, look_at_point: Gf.Vec3d, up_vector: Gf.Vec3d = Gf.Vec3d(0, 1, 0)):
    """Aligns a camera prim to look at a specific point.

    Args:
        camera_prim (UsdGeom.Camera): The camera prim to align.
        look_at_point (Gf.Vec3d): The point in world space to look at.
        up_vector (Gf.Vec3d): The up vector of the camera. Defaults to (0, 1, 0).

    Returns:
        None
    """
    camera_xform = UsdGeom.Xformable(camera_prim)
    camera_transform = camera_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    camera_position = camera_transform.ExtractTranslation()
    look_at_matrix = Gf.Matrix4d()
    look_at_matrix.SetLookAt(camera_position, look_at_point, up_vector)
    camera_xform_op = camera_xform.AddTransformOp()
    camera_xform_op.Set(look_at_matrix, Usd.TimeCode.Default())


def get_capsule_attributes(capsule: UsdGeom.Capsule) -> Tuple[str, Tuple[float, float, float], float, float]:
    """
    Get the attribute values of a UsdGeom.Capsule.

    Args:
        capsule (UsdGeom.Capsule): The capsule to retrieve attributes from.

    Returns:
        A tuple containing the following attribute values:
        - axis (str): The axis along which the spine of the capsule is aligned.
        - height (float): The size of the capsule's spine along the specified axis.
        - radius (float): The radius of the capsule.
        - extent (Tuple[float, float, float]): The extent of the capsule.
    """
    axis_attr = capsule.GetAxisAttr()
    if axis_attr.HasAuthoredValue():
        axis = axis_attr.Get()
    else:
        axis = axis_attr.GetDefaultValue()
    height_attr = capsule.GetHeightAttr()
    if height_attr.HasAuthoredValue():
        height = height_attr.Get()
    else:
        height = height_attr.GetDefaultValue()
    radius_attr = capsule.GetRadiusAttr()
    if radius_attr.HasAuthoredValue():
        radius = radius_attr.Get()
    else:
        radius = radius_attr.GetDefaultValue()
    extent_attr = capsule.GetExtentAttr()
    if extent_attr.HasAuthoredValue():
        extent = extent_attr.Get()
    else:
        extent = extent_attr.GetDefaultValue()
    return (axis, height, radius, tuple(extent[0]))


def duplicate_capsule(stage: Usd.Stage, src_prim_path: str, dst_prim_path: str) -> UsdGeom.Capsule:
    """
    Duplicate a UsdGeomCapsule prim within the same stage.

    Args:
        stage (Usd.Stage): The stage containing the source capsule prim.
        src_prim_path (str): The path of the source capsule prim.
        dst_prim_path (str): The path where the duplicated capsule prim will be created.

    Returns:
        UsdGeom.Capsule: The duplicated capsule prim.

    Raises:
        ValueError: If the source prim is not a valid UsdGeomCapsule.
    """
    src_prim = stage.GetPrimAtPath(src_prim_path)
    if not src_prim.IsValid() or not UsdGeom.Capsule(src_prim):
        raise ValueError(f"Prim at path {src_prim_path} is not a valid UsdGeomCapsule.")
    dst_capsule = UsdGeom.Capsule.Define(stage, dst_prim_path)
    src_capsule = UsdGeom.Capsule(src_prim)
    if src_capsule.GetAxisAttr().HasAuthoredValue():
        dst_capsule.GetAxisAttr().Set(src_capsule.GetAxisAttr().Get())
    if src_capsule.GetHeightAttr().HasAuthoredValue():
        dst_capsule.GetHeightAttr().Set(src_capsule.GetHeightAttr().Get())
    if src_capsule.GetRadiusAttr().HasAuthoredValue():
        dst_capsule.GetRadiusAttr().Set(src_capsule.GetRadiusAttr().Get())
    if src_capsule.GetExtentAttr().HasAuthoredValue():
        dst_capsule.GetExtentAttr().Set(src_capsule.GetExtentAttr().Get())
    return dst_capsule


def scale_capsule(capsule: UsdGeom.Capsule, scale: Gf.Vec3f) -> None:
    """Scale a capsule prim by a given scale factor.

    Args:
        capsule (UsdGeom.Capsule): The capsule prim to scale.
        scale (Gf.Vec3f): The scale factor to apply.
    """
    if not capsule or not capsule.GetPrim().IsValid():
        raise ValueError("Invalid capsule prim.")
    radius_attr = capsule.GetRadiusAttr()
    height_attr = capsule.GetHeightAttr()
    radius = radius_attr.Get()
    height = height_attr.Get()
    new_radius = radius * scale[0]
    new_height = height * scale[1]
    radius_attr.Set(new_radius)
    height_attr.Set(new_height)
    half_height = new_height * 0.5
    min_extent = Gf.Vec3f(-new_radius, -new_radius - half_height, -new_radius)
    max_extent = Gf.Vec3f(new_radius, new_radius + half_height, new_radius)
    capsule.CreateExtentAttr().Set([min_extent, max_extent])


def animate_capsule(
    stage: Usd.Stage, prim_path: str, num_frames: int, radius_scale: float, height_scale: float
) -> None:
    """Animate a capsule primitive over a number of frames.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the capsule primitive.
        num_frames (int): The number of frames to animate.
        radius_scale (float): The maximum scale factor for the radius.
        height_scale (float): The maximum scale factor for the height.

    Raises:
        ValueError: If the prim at the given path is not a Capsule.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"No prim found at path: {prim_path}")
    capsule = UsdGeom.Capsule(prim)
    if not capsule:
        raise ValueError(f"Prim at path {prim_path} is not a Capsule")
    radius_attr = capsule.GetRadiusAttr()
    height_attr = capsule.GetHeightAttr()
    for frame in range(num_frames):
        t = float(frame) / (num_frames - 1)
        radius = 1.0 + (radius_scale - 1.0) * t
        radius_attr.Set(radius, Usd.TimeCode(frame))
        height = 1.0 + (height_scale - 1.0) * t
        height_attr.Set(height, Usd.TimeCode(frame))


def create_capsule(
    stage: Usd.Stage, prim_path: str, height: float = 1.0, radius: float = 0.5, axis: str = "Z"
) -> UsdGeom.Capsule:
    """Create a capsule prim at the given path on the stage.

    Args:
        stage (Usd.Stage): The stage to create the capsule on.
        prim_path (str): The path where the capsule prim should be created.
        height (float, optional): The height of the capsule's cylinder section. Defaults to 1.0.
        radius (float, optional): The radius of the capsule. Defaults to 0.5.
        axis (str, optional): The primary axis of the capsule ("X", "Y", or "Z"). Defaults to "Z".

    Returns:
        UsdGeom.Capsule: The created capsule prim.
    """
    if not stage:
        raise ValueError("Invalid stage")
    if not prim_path:
        raise ValueError("Invalid prim path")
    if axis not in ["X", "Y", "Z"]:
        raise ValueError(f"Invalid axis '{axis}', must be 'X', 'Y', or 'Z'")
    capsule = UsdGeom.Capsule.Define(stage, prim_path)
    capsule.CreateHeightAttr().Set(height)
    capsule.CreateRadiusAttr().Set(radius)
    capsule.CreateAxisAttr().Set(axis)
    half_height = height / 2.0
    min_extent = (-radius, -radius, -half_height)
    max_extent = (radius, radius, half_height)
    if axis == "Z":
        pass
    elif axis == "Y":
        min_extent = (-radius, -half_height, -radius)
        max_extent = (radius, half_height, radius)
    elif axis == "X":
        min_extent = (-half_height, -radius, -radius)
        max_extent = (half_height, radius, radius)
    capsule.CreateExtentAttr().Set([Gf.Vec3f(min_extent), Gf.Vec3f(max_extent)])
    return capsule


def align_capsules(stage: Usd.Stage, capsule_paths: List[str], spacing: float = 2.0) -> None:
    """Align capsules along the X-axis with specified spacing.

    Args:
        stage (Usd.Stage): The USD stage.
        capsule_paths (List[str]): A list of paths to capsule prims.
        spacing (float, optional): The spacing between capsules. Defaults to 2.0.
    """
    if not capsule_paths:
        return
    first_capsule = UsdGeom.Capsule(stage.GetPrimAtPath(capsule_paths[0]))
    if not first_capsule:
        raise ValueError(f"Prim at path {capsule_paths[0]} is not a valid Capsule.")
    radius = first_capsule.GetRadiusAttr().Get()
    for i, capsule_path in enumerate(capsule_paths):
        capsule = UsdGeom.Capsule(stage.GetPrimAtPath(capsule_path))
        if not capsule:
            raise ValueError(f"Prim at path {capsule_path} is not a valid Capsule.")
        capsule.CreateAxisAttr().Set("X")
        x_coord = i * (spacing + 2 * radius)
        UsdGeom.XformCommonAPI(capsule).SetTranslate((x_coord, 0, 0))


def find_capsules_by_attribute(stage: Usd.Stage, attribute_name: str, attribute_value: float) -> List[UsdGeom.Capsule]:
    """
    Find all capsule prims on the given stage that have a specific attribute value.

    Args:
        stage (Usd.Stage): The stage to search for capsules.
        attribute_name (str): The name of the attribute to check.
        attribute_value (float): The value to match for the attribute.

    Returns:
        List[UsdGeom.Capsule]: A list of capsule prims that match the criteria.
    """
    capsules = []
    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Capsule):
            capsule = UsdGeom.Capsule(prim)
            if prim.HasAttribute(attribute_name):
                attribute = prim.GetAttribute(attribute_name)
                if attribute.Get() == attribute_value:
                    capsules.append(capsule)
    return capsules


def translate_capsule(capsule: UsdGeom.Capsule, translation: Tuple[float, float, float]) -> None:
    """Translate a Capsule prim.

    Args:
        capsule (UsdGeom.Capsule): The Capsule prim to translate.
        translation (Tuple[float, float, float]): The translation vector (x, y, z).

    Raises:
        ValueError: If the input capsule prim is not valid.
    """
    if not capsule.GetPrim().IsValid():
        raise ValueError("Input Capsule prim is not valid.")
    xformable = UsdGeom.Xformable(capsule)
    if not xformable:
        raise ValueError("Input Capsule prim is not transformable.")
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if not translate_op:
        translate_op = add_translate_op(xformable)
    translate_op.Set(Gf.Vec3d(translation))


def set_capsule_attributes(capsule: UsdGeom.Capsule, height: float, radius: float, axis: str = "Z") -> None:
    """Set the attributes of a UsdGeom.Capsule.

    Args:
        capsule (UsdGeom.Capsule): The capsule prim to set attributes on.
        height (float): The height of the cylinder portion of the capsule.
        radius (float): The radius of the capsule.
        axis (str, optional): The axis along which the spine of the capsule is aligned.
            Must be one of "X", "Y", or "Z". Defaults to "Z".

    Raises:
        ValueError: If the provided axis is not one of "X", "Y", or "Z".
    """
    if axis not in ["X", "Y", "Z"]:
        raise ValueError(f"Invalid axis '{axis}'. Must be one of 'X', 'Y', or 'Z'.")
    height_attr = capsule.CreateHeightAttr()
    height_attr.Set(height)
    radius_attr = capsule.CreateRadiusAttr()
    radius_attr.Set(radius)
    axis_attr = capsule.CreateAxisAttr()
    axis_attr.Set(axis)
    half_height = height / 2.0
    if axis == "X":
        extent = ((-half_height, -radius, -radius), (half_height, radius, radius))
    elif axis == "Y":
        extent = ((-radius, -half_height, -radius), (radius, half_height, radius))
    else:
        extent = ((-radius, -radius, -half_height), (radius, radius, half_height))
    extent_attr = capsule.CreateExtentAttr()
    extent_attr.Set([Gf.Vec3f(extent[0]), Gf.Vec3f(extent[1])])


def create_cone_with_attributes(
    stage: Usd.Stage, prim_path: str, height: float, radius: float, axis: str = "Z"
) -> UsdGeom.Cone:
    """
    Create a UsdGeom.Cone prim with specified attributes.

    Args:
        stage (Usd.Stage): The USD stage to create the cone on.
        prim_path (str): The path where the cone prim should be created.
        height (float): The height of the cone.
        radius (float): The radius of the cone.
        axis (str, optional): The axis along which the spine of the cone is aligned. Defaults to "Z".

    Returns:
        UsdGeom.Cone: The created cone prim.
    """
    if not stage:
        raise ValueError("Invalid stage")
    if not prim_path:
        raise ValueError("Invalid prim path")
    cone = UsdGeom.Cone.Define(stage, prim_path)
    cone.CreateHeightAttr(height)
    cone.CreateRadiusAttr(radius)
    cone.CreateAxisAttr(axis)
    half_height = height / 2
    min_extent = Gf.Vec3f(-radius, -half_height, -radius)
    max_extent = Gf.Vec3f(radius, half_height, radius)
    cone.CreateExtentAttr([min_extent, max_extent])
    return cone


def copy_cone_attributes(source_cone: UsdGeom.Cone, dest_cone: UsdGeom.Cone) -> None:
    """Copy attributes from one UsdGeom.Cone to another.

    This function copies the attribute values from the source cone to the
    destination cone. If an attribute is not authored on the source, it will
    not be copied to the destination.

    Args:
        source_cone (UsdGeom.Cone): The source cone to copy attributes from.
        dest_cone (UsdGeom.Cone): The destination cone to copy attributes to.
    """
    if source_cone.GetAxisAttr().HasAuthoredValue():
        dest_cone.GetAxisAttr().Set(source_cone.GetAxisAttr().Get())
    if source_cone.GetHeightAttr().HasAuthoredValue():
        dest_cone.GetHeightAttr().Set(source_cone.GetHeightAttr().Get())
    if source_cone.GetRadiusAttr().HasAuthoredValue():
        dest_cone.GetRadiusAttr().Set(source_cone.GetRadiusAttr().Get())
    if source_cone.GetExtentAttr().HasAuthoredValue():
        dest_cone.GetExtentAttr().Set(source_cone.GetExtentAttr().Get())


def get_cone_bounding_box(cone: UsdGeom.Cone) -> Gf.Range3d:
    """
    Compute the bounding box for a UsdGeomCone.

    Args:
        cone (UsdGeom.Cone): The cone to compute the bounding box for.

    Returns:
        Gf.Range3d: The computed bounding box.
    """
    if not cone or not cone.GetPrim().IsValid():
        raise ValueError("Invalid UsdGeomCone.")
    height = cone.GetHeightAttr().Get()
    radius = cone.GetRadiusAttr().Get()
    axis = cone.GetAxisAttr().Get()
    if axis == UsdGeom.Tokens.x:
        lower = Gf.Vec3d(0, -radius, -radius)
        upper = Gf.Vec3d(height, radius, radius)
    elif axis == UsdGeom.Tokens.y:
        lower = Gf.Vec3d(-radius, 0, -radius)
        upper = Gf.Vec3d(radius, height, radius)
    elif axis == UsdGeom.Tokens.z:
        lower = Gf.Vec3d(-radius, -radius, 0)
        upper = Gf.Vec3d(radius, radius, height)
    else:
        raise ValueError(f"Invalid axis value: {axis}")
    return Gf.Range3d(lower, upper)


def attach_material_to_cone(stage: Usd.Stage, cone_path: str, material_path: str):
    """Attach a material to a cone prim.

    Args:
        stage (Usd.Stage): The USD stage.
        cone_path (str): The path to the cone prim.
        material_path (str): The path to the material prim.

    Raises:
        ValueError: If the cone or material prim is not found or invalid.
    """
    cone_prim = stage.GetPrimAtPath(cone_path)
    if not cone_prim.IsValid():
        raise ValueError(f"Cone prim at path {cone_path} does not exist.")
    if not UsdGeom.Cone(cone_prim):
        raise ValueError(f"Prim at path {cone_path} is not a UsdGeomCone.")
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise ValueError(f"Material prim at path {material_path} does not exist.")
    if not UsdShade.Material(material_prim):
        raise ValueError(f"Prim at path {material_path} is not a UsdShadeMaterial.")
    material = UsdShade.Material(material_prim)
    UsdShade.MaterialBindingAPI(cone_prim).Bind(material)


def duplicate_cone(stage: Usd.Stage, cone_path: str, dest_path: str) -> UsdGeom.Cone:
    """
    Duplicates a UsdGeomCone prim to a new path in the stage.

    Args:
        stage (Usd.Stage): The USD stage.
        cone_path (str): The path of the cone prim to duplicate.
        dest_path (str): The destination path for the duplicated cone prim.

    Returns:
        UsdGeom.Cone: The duplicated cone prim.

    Raises:
        ValueError: If the source cone prim is not valid or if the destination path is not valid.
    """
    source_cone = UsdGeom.Cone(stage.GetPrimAtPath(cone_path))
    if not source_cone:
        raise ValueError(f"Invalid cone prim at path: {cone_path}")
    dest_prim = stage.GetPrimAtPath(dest_path)
    if not dest_prim:
        dest_prim = stage.DefinePrim(dest_path)
    dest_cone = UsdGeom.Cone.Define(stage, dest_path)
    if source_cone.GetHeightAttr().HasValue():
        dest_cone.GetHeightAttr().Set(source_cone.GetHeightAttr().Get())
    if source_cone.GetRadiusAttr().HasValue():
        dest_cone.GetRadiusAttr().Set(source_cone.GetRadiusAttr().Get())
    if source_cone.GetAxisAttr().HasValue():
        dest_cone.GetAxisAttr().Set(source_cone.GetAxisAttr().Get())
    if source_cone.GetExtentAttr().HasValue():
        dest_cone.GetExtentAttr().Set(source_cone.GetExtentAttr().Get())
    return dest_cone


def animate_cone_translation(
    stage: Usd.Stage, cone_path: str, translations: List[Tuple[float, float, float]], time_codes: List[Usd.TimeCode]
) -> None:
    """Animate the translation of a cone over time.

    Args:
        stage (Usd.Stage): The USD stage.
        cone_path (str): The path to the cone prim.
        translations (List[Tuple[float, float, float]]): A list of translation values (x, y, z) for each time code.
        time_codes (List[Usd.TimeCode]): A list of time codes corresponding to each translation value.

    Raises:
        ValueError: If the cone prim is not found or if the number of translations and time codes do not match.
    """
    cone_prim = stage.GetPrimAtPath(cone_path)
    if not cone_prim.IsValid():
        raise ValueError(f"Cone prim not found at path: {cone_path}")
    if len(translations) != len(time_codes):
        raise ValueError("Number of translations and time codes must match.")
    xformable = UsdGeom.Xformable(cone_prim)
    translate_op = add_translate_op(xformable)
    for translation, time_code in zip(translations, time_codes):
        translate_op.Set(Gf.Vec3d(translation), time_code)


def analyze_cone_intersection(stage: Usd.Stage, cone_path: str, target_path: str) -> Tuple[bool, Gf.Vec3d]:
    """
    Analyze the intersection between a cone and a target prim.

    Args:
        stage (Usd.Stage): The USD stage.
        cone_path (str): The path to the cone prim.
        target_path (str): The path to the target prim.

    Returns:
        Tuple[bool, Gf.Vec3d]: A tuple containing:
            - A boolean indicating whether the cone intersects the target prim.
            - The intersection point in local space of the cone, or (0, 0, 0) if no intersection.
    """
    cone_prim = stage.GetPrimAtPath(cone_path)
    target_prim = stage.GetPrimAtPath(target_path)
    if not cone_prim.IsValid():
        raise ValueError(f"Cone prim at path {cone_path} does not exist.")
    if not target_prim.IsValid():
        raise ValueError(f"Target prim at path {target_path} does not exist.")
    cone = UsdGeom.Cone(cone_prim)
    if not cone:
        raise ValueError(f"Prim at path {cone_path} is not a valid cone.")
    target = UsdGeom.Imageable(target_prim)
    if not target:
        raise ValueError(f"Prim at path {target_path} is not a valid imageable prim.")
    height = cone.GetHeightAttr().Get()
    radius = cone.GetRadiusAttr().Get()
    axis = cone.GetAxisAttr().Get()
    cone_xform = UsdGeom.Xformable(cone_prim)
    cone_transform = cone_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    target_xform = UsdGeom.Xformable(target_prim)
    target_transform = target_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    target_boundable = UsdGeom.Boundable(target_prim)
    if target_boundable:
        target_extent = target_boundable.ComputeExtentFromPlugins(Usd.TimeCode.Default(), target_transform)
        if not target_extent.IsEmpty():
            target_center = target_extent.ComputeCentroid()
        else:
            target_center = Gf.Vec3d(0, 0, 0)
    else:
        target_center = Gf.Vec3d(0, 0, 0)
    target_center_local = target_transform.GetInverse().Transform(target_center)
    cone_apex = Gf.Vec3d(0, 0, height / 2) if axis == "Z" else Gf.Vec3d(0, height / 2, 0)
    cone_apex_world = cone_transform.Transform(cone_apex)
    intersection_vec = target_center - cone_apex_world
    intersection_dist = intersection_vec.GetLength()
    intersects = intersection_dist <= height / 2
    intersection_point_local = Gf.Vec3d(0, 0, 0)
    if intersects:
        intersection_point_world = cone_apex_world + intersection_vec.GetNormalized() * intersection_dist
        intersection_point_local = cone_transform.GetInverse().Transform(intersection_point_world)
    return (intersects, intersection_point_local)


def convert_cone_to_mesh(cone_prim: UsdGeom.Cone) -> UsdGeom.Mesh:
    """Convert a UsdGeomCone to a UsdGeomMesh.

    Args:
        cone_prim (UsdGeom.Cone): The input cone prim to be converted.

    Returns:
        UsdGeom.Mesh: The converted mesh prim.

    Raises:
        ValueError: If the input prim is not a valid UsdGeomCone.
    """
    if not cone_prim or not cone_prim.GetPrim().IsValid():
        raise ValueError("Invalid UsdGeomCone prim.")
    height = cone_prim.GetHeightAttr().Get()
    radius = cone_prim.GetRadiusAttr().Get()
    axis = cone_prim.GetAxisAttr().Get()
    mesh_prim = UsdGeom.Mesh.Define(cone_prim.GetPrim().GetStage(), cone_prim.GetPath())
    points = [Gf.Vec3f(0, 0, 0), Gf.Vec3f(0, height, 0)]
    for i in range(4):
        angle = i * (2 * math.pi / 4)
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        points.append(Gf.Vec3f(x, 0, z))
    mesh_prim.CreatePointsAttr().Set(points)
    vertexCounts = [5, 3, 3, 3, 3]
    mesh_prim.CreateFaceVertexCountsAttr().Set(vertexCounts)
    vertexIndices = [0, 1, 2, 3, 4, 0, 2, 1, 0, 3, 1, 0, 4, 1, 0, 1, 5]
    mesh_prim.CreateFaceVertexIndicesAttr().Set(vertexIndices)
    if axis != "Z":
        xform = UsdGeom.Xform(mesh_prim)
        if axis == "X":
            xform.AddRotateYOp().Set(90)
        elif axis == "Y":
            xform.AddRotateXOp().Set(-90)
    return mesh_prim


def test_convert_cone_to_mesh():
    stage = Usd.Stage.CreateInMemory()
    cone_path = "/Cone"
    cone_prim = UsdGeom.Cone.Define(stage, cone_path)
    cone_prim.GetHeightAttr().Set(5)
    cone_prim.GetRadiusAttr().Set(3)
    cone_prim.GetAxisAttr().Set("Y")
    mesh_prim = convert_cone_to_mesh(cone_prim)
    assert mesh_prim.GetPrim().IsValid()
    assert mesh_prim.GetPointsAttr().Get() is not None
    assert mesh_prim.GetFaceVertexCountsAttr().Get() is not None
    assert mesh_prim.GetFaceVertexIndicesAttr().Get() is not None
    print(mesh_prim.GetPrim().GetStage().GetRootLayer().ExportToString())


def get_all_constraint_targets(model_api: UsdGeom.ModelAPI) -> List[UsdGeom.ConstraintTarget]:
    """
    Returns all the constraint targets belonging to the model.

    Only valid constraint targets in the "constraintTargets" namespace are
    returned by this method.
    """
    prim = model_api.GetPrim()
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    constraint_targets = []
    for prop in prim.GetPropertiesInNamespace("constraintTargets"):
        if isinstance(prop, Usd.Attribute):
            constraint_target = UsdGeom.ConstraintTarget(prop)
            if constraint_target.GetAttr().IsValid():
                constraint_targets.append(constraint_target)
    return constraint_targets


def validate_constraint_targets(prim: Usd.Prim) -> List[UsdGeom.ConstraintTarget]:
    """Validate and return all constraint targets on a prim."""
    attrs = prim.GetAttributes()
    valid_targets = []
    for attr in attrs:
        if UsdGeom.ConstraintTarget.IsValid(attr):
            target = UsdGeom.ConstraintTarget(attr)
            identifier = target.GetIdentifier()
            if not identifier:
                print(f"Warning: Constraint target {attr.GetName()} has an empty identifier.")
                continue
            value = target.Get()
            if not value or not all((len(row) == 4 for row in value)):
                print(f"Warning: Constraint target {attr.GetName()} has an invalid matrix value.")
                continue
            valid_targets.append(target)
    return valid_targets


def set_constraint_targets_relative_to_model(stage: Usd.Stage, constraint_target_paths: List[str]) -> None:
    """
    Sets the constraint targets to be relative to the model's local space.

    Args:
        stage (Usd.Stage): The USD stage.
        constraint_target_paths (List[str]): A list of paths to constraint targets.

    Raises:
        ValueError: If any of the constraint target paths are invalid.
    """
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        raise ValueError("The stage has no default prim.")
    root_xformable = UsdGeom.Xformable(root_prim)
    root_transform = root_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    inverse_root_transform = root_transform.GetInverse()
    for constraint_target_path in constraint_target_paths:
        constraint_target_attr = stage.GetAttributeAtPath(constraint_target_path)
        if not constraint_target_attr:
            raise ValueError(f"Invalid constraint target path: {constraint_target_path}")
        constraint_target = UsdGeom.ConstraintTarget(constraint_target_attr)
        constraint_target_value = Gf.Matrix4d()
        constraint_target.Get(constraint_target_value, Usd.TimeCode.Default())
        relative_constraint_target_value = constraint_target_value * inverse_root_transform
        constraint_target.Set(relative_constraint_target_value, Usd.TimeCode.Default())


def set_constraint_target_transform(
    constraint_target: UsdGeom.ConstraintTarget,
    transform: Gf.Matrix4d,
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Set the transform of a constraint target.

    Args:
        constraint_target (UsdGeom.ConstraintTarget): The constraint target to set the transform for.
        transform (Gf.Matrix4d): The transform matrix to set.
        time_code (Usd.TimeCode, optional): The time code to set the value at. Defaults to Default time code.

    Raises:
        ValueError: If the constraint target is not valid.
    """
    if not constraint_target.GetAttr().IsDefined():
        raise ValueError("Constraint target is not defined.")
    success = constraint_target.Set(transform, time_code)
    if not success:
        raise ValueError("Failed to set constraint target transform.")


def copy_constraint_target(
    src_constraint_target: UsdGeom.ConstraintTarget, dst_prim: Usd.Prim, dst_attr_name: str
) -> UsdGeom.ConstraintTarget:
    """
    Copies a constraint target to a new attribute on a destination prim.

    Args:
        src_constraint_target (UsdGeom.ConstraintTarget): The source constraint target to copy.
        dst_prim (Usd.Prim): The destination prim to create the new constraint target attribute on.
        dst_attr_name (str): The name of the new constraint target attribute on the destination prim.

    Returns:
        UsdGeom.ConstraintTarget: The newly created constraint target on the destination prim.

    Raises:
        ValueError: If the source constraint target is invalid or if the destination attribute already exists.
    """
    if not src_constraint_target.GetAttr().IsDefined():
        raise ValueError("Source constraint target is not defined.")
    dst_attr = dst_prim.GetAttribute(dst_attr_name)
    if dst_attr.IsDefined():
        raise ValueError(f"Destination attribute '{dst_attr_name}' already exists on prim '{dst_prim.GetPath()}'.")
    src_value = Gf.Matrix4d()
    src_constraint_target.Get(src_value, Usd.TimeCode.Default())
    src_identifier = src_constraint_target.GetIdentifier()
    dst_attr = dst_prim.CreateAttribute(dst_attr_name, Sdf.ValueTypeNames.Matrix4d)
    dst_constraint_target = UsdGeom.ConstraintTarget(dst_attr)
    dst_constraint_target.Set(src_value, Usd.TimeCode.Default())
    dst_constraint_target.SetIdentifier(src_identifier)
    return dst_constraint_target


def find_cubes_by_size_range(stage: Usd.Stage, min_size: float, max_size: float) -> List[UsdGeom.Cube]:
    """
    Find all UsdGeomCube prims on the given stage whose size is within the specified range.

    Args:
        stage (Usd.Stage): The USD stage to search for cubes.
        min_size (float): The minimum size (inclusive) of the cubes to find.
        max_size (float): The maximum size (exclusive) of the cubes to find.

    Returns:
        List[UsdGeom.Cube]: A list of UsdGeomCube prims whose size is within the specified range.
    """
    if min_size < 0:
        raise ValueError("min_size must be non-negative")
    if max_size <= min_size:
        raise ValueError("max_size must be greater than min_size")
    cubes = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Cube):
            cube = UsdGeom.Cube(prim)
            size_attr = cube.GetSizeAttr()
            if size_attr.HasAuthoredValue():
                size = size_attr.Get()
                if min_size <= size < max_size:
                    cubes.append(cube)
    return cubes


def create_and_transform_cube(
    stage: Usd.Stage, cube_path: str, translation: Gf.Vec3f, rotation: Gf.Vec3f, scale: Gf.Vec3f
) -> UsdGeom.Cube:
    """Create a cube prim and apply transformations to it.

    Args:
        stage (Usd.Stage): The USD stage to create the cube on.
        cube_path (str): The path where the cube prim should be created.
        translation (Gf.Vec3f): The translation to apply to the cube.
        rotation (Gf.Vec3f): The rotation to apply to the cube (in degrees).
        scale (Gf.Vec3f): The scale to apply to the cube.

    Returns:
        UsdGeom.Cube: The created cube prim.

    Raises:
        ValueError: If the cube prim cannot be created or transformed.
    """
    cube = UsdGeom.Cube.Define(stage, cube_path)
    if not cube:
        raise ValueError(f"Failed to create cube prim at path {cube_path}")
    xformable = UsdGeom.Xformable(cube)
    if not xformable:
        raise ValueError(f"Cube prim at path {cube_path} is not transformable")
    translate_op = add_translate_op(xformable)
    translate_op.Set(translation)
    rotate_op = add_rotate_xyz_op(xformable)
    rotation_degrees = Gf.Vec3f(
        Gf.RadiansToDegrees(rotation[0]), Gf.RadiansToDegrees(rotation[1]), Gf.RadiansToDegrees(rotation[2])
    )
    rotate_op.Set(rotation_degrees)
    scale_op = add_scale_op(xformable)
    scale_op.Set(scale)
    return cube


def animate_cube_size(cube_prim: UsdGeom.Cube, start_size: float, end_size: float, num_frames: int) -> None:
    """Animate the size of a cube prim over a specified number of frames.

    Args:
        cube_prim (UsdGeom.Cube): The cube prim to animate.
        start_size (float): The starting size of the cube.
        end_size (float): The ending size of the cube.
        num_frames (int): The number of frames to animate over.
    """
    if not cube_prim or not cube_prim.GetPrim().IsValid():
        raise ValueError("Invalid cube prim.")
    if start_size <= 0 or end_size <= 0:
        raise ValueError("Size values must be positive.")
    if num_frames <= 0:
        raise ValueError("Number of frames must be positive.")
    size_attr = cube_prim.GetSizeAttr()
    size_increment = (end_size - start_size) / (num_frames - 1)
    for frame in range(num_frames):
        frame_time = frame
        size_value = start_size + size_increment * frame
        size_attr.Set(size_value, Usd.TimeCode(frame_time))


def merge_cubes(stage: Usd.Stage, prim_paths: List[str], merge_path: str) -> Usd.Prim:
    """Merge multiple cube prims into a single cube prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The paths of the cube prims to merge.
        merge_path (str): The path where the merged cube prim will be created.

    Returns:
        Usd.Prim: The merged cube prim.

    Raises:
        ValueError: If any of the input prim paths are invalid or not of type Cube.
    """
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {path} does not exist.")
        if not prim.IsA(UsdGeom.Cube):
            raise ValueError(f"Prim at path {path} is not of type Cube.")
    merged_cube = UsdGeom.Cube.Define(stage, merge_path)
    bbox = Gf.BBox3d()
    for path in prim_paths:
        cube = UsdGeom.Cube(stage.GetPrimAtPath(path))
        local_bbox = cube.ComputeLocalBound(0, "default")
        bbox = Gf.BBox3d.Combine(bbox, local_bbox)
    merged_cube.CreateExtentAttr([bbox.ComputeAlignedRange().GetMin(), bbox.ComputeAlignedRange().GetMax()], True)
    return merged_cube.GetPrim()


def copy_cube_with_transform(
    stage: Usd.Stage, source_cube_path: str, dest_cube_path: str, transform: Gf.Matrix4d
) -> UsdGeom.Cube:
    """Copy a cube prim and apply a transform to the copy.

    Args:
        stage (Usd.Stage): The stage containing the source cube.
        source_cube_path (str): The path of the source cube prim.
        dest_cube_path (str): The path where the transformed cube copy should be created.
        transform (Gf.Matrix4d): The transformation matrix to apply to the cube copy.

    Returns:
        UsdGeom.Cube: The newly created transformed cube prim.

    Raises:
        ValueError: If the source cube prim does not exist or is not a valid UsdGeomCube.
    """
    source_cube = UsdGeom.Cube(stage.GetPrimAtPath(source_cube_path))
    if not source_cube:
        raise ValueError(f"Prim at path {source_cube_path} is not a valid UsdGeomCube.")
    dest_cube = UsdGeom.Cube.Define(stage, dest_cube_path)
    source_prim = source_cube.GetPrim()
    dest_prim = dest_cube.GetPrim()
    for attr in source_prim.GetAttributes():
        if attr.HasValue():
            attr_name = attr.GetName()
            attr_value = attr.Get()
            dest_prim.CreateAttribute(attr_name, attr.GetTypeName()).Set(attr_value)
    dest_cube_xform = UsdGeom.Xformable(dest_cube)
    dest_cube_xform.MakeMatrixXform().Set(transform)
    return dest_cube


def create_hermite_curve(
    stage: Usd.Stage, curve_path: str, points: Vt.Vec3fArray, tangents: Vt.Vec3fArray, curve_vertex_counts: Vt.IntArray
) -> UsdGeom.HermiteCurves:
    """Create a HermiteCurves prim with the given points, tangents, and curve vertex counts.

    Args:
        stage (Usd.Stage): The stage to create the curve on.
        curve_path (str): The path to create the curve at.
        points (Vt.Vec3fArray): The points of the curve.
        tangents (Vt.Vec3fArray): The tangents of the curve. Must be the same length as points.
        curve_vertex_counts (Vt.IntArray): The number of vertices for each curve in the batch.

    Returns:
        UsdGeom.HermiteCurves: The created HermiteCurves prim.

    Raises:
        ValueError: If the lengths of points and tangents do not match, or if the sum of curve_vertex_counts does not equal the length of points.
    """
    if len(points) != len(tangents):
        raise ValueError("Points and tangents must have the same length.")
    if sum(curve_vertex_counts) != len(points):
        raise ValueError("Sum of curve_vertex_counts must equal the number of points.")
    curves = UsdGeom.HermiteCurves.Define(stage, curve_path)
    curves.GetPointsAttr().Set(points)
    curves.GetTangentsAttr().Set(tangents)
    curves.GetCurveVertexCountsAttr().Set(curve_vertex_counts)
    return curves


def merge_hermite_curves(stage: Usd.Stage, curves_paths: Tuple[str], dst_path: str) -> UsdGeom.HermiteCurves:
    """Merge multiple hermite curves into a single hermite curve.

    Args:
        stage (Usd.Stage): The USD stage.
        curves_paths (Tuple[str]): The paths to the input hermite curves.
        dst_path (str): The path for the output merged hermite curve.

    Returns:
        UsdGeom.HermiteCurves: The merged hermite curve.

    Raises:
        ValueError: If any input curve is invalid or not a HermiteCurves prim.
    """
    for curve_path in curves_paths:
        prim = stage.GetPrimAtPath(curve_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {curve_path} does not exist.")
        if not prim.IsA(UsdGeom.HermiteCurves):
            raise ValueError(f"Prim at path {curve_path} is not a HermiteCurves.")
    merged_curve = UsdGeom.HermiteCurves.Define(stage, dst_path)
    points = []
    tangents = []
    curve_vertex_counts = []
    for curve_path in curves_paths:
        curve = UsdGeom.HermiteCurves(stage.GetPrimAtPath(curve_path))
        points.extend(curve.GetPointsAttr().Get())
        tangents.extend(curve.GetTangentsAttr().Get())
        curve_vertex_counts.append(len(curve.GetPointsAttr().Get()))
    merged_curve.CreatePointsAttr(points)
    merged_curve.CreateTangentsAttr(tangents)
    merged_curve.CreateCurveVertexCountsAttr(curve_vertex_counts)
    return merged_curve


def assign_hermite_curve_width(
    curves: UsdGeom.HermiteCurves, width: Union[float, Vt.FloatArray], time_code: Usd.TimeCode = Usd.TimeCode.Default()
):
    """Assign width to a HermiteCurves prim.

    Args:
        curves (UsdGeom.HermiteCurves): The HermiteCurves prim to assign width to.
        width (Union[float, Vt.FloatArray]): The width value(s) to assign. Can be a single float or a FloatArray.
        time_code (Usd.TimeCode, optional): The time code at which to set the value. Defaults to Default.

    Raises:
        ValueError: If the input curves prim is not a valid UsdGeom.HermiteCurves prim.
        TypeError: If the input width is not a float or Vt.FloatArray.
    """
    if not curves or not isinstance(curves, UsdGeom.HermiteCurves):
        raise ValueError("Input 'curves' must be a valid UsdGeom.HermiteCurves prim.")
    if isinstance(width, float):
        width_array = Vt.FloatArray([width])
        curves.GetWidthsAttr().Set(width_array, time_code)
    elif isinstance(width, Vt.FloatArray):
        curve_vertices = curves.GetCurveVertexCountsAttr().Get(time_code)
        if curve_vertices and len(width) == sum(curve_vertices):
            curves.GetWidthsAttr().Set(width, time_code)
        else:
            raise ValueError("Length of input 'width' FloatArray does not match the number of curve vertices.")
    else:
        raise TypeError("Input 'width' must be a float or Vt.FloatArray.")


def analyze_hermite_curve_tangents(curve: UsdGeom.HermiteCurves) -> float:
    """
    Analyze the tangents of a hermite curve and return the average tangent length.

    Args:
        curve (UsdGeom.HermiteCurves): The hermite curve to analyze.

    Returns:
        float: The average length of the curve's tangents.

    Raises:
        ValueError: If the curve has no points or tangents defined.
    """
    points = curve.GetPointsAttr().Get()
    tangents = curve.GetTangentsAttr().Get()
    if not points or not tangents:
        raise ValueError("Hermite curve must have points and tangents defined.")
    if len(points) != len(tangents):
        raise ValueError("Number of points and tangents must be equal.")
    total_length = 0.0
    for tangent in tangents:
        length = Gf.Vec3f(tangent).GetLength()
        total_length += length
    avg_length = total_length / len(tangents)
    return avg_length


def retarget_hermite_curve(source_prim: UsdGeom.HermiteCurves, target_prim: UsdGeom.HermiteCurves):
    """
    Retarget the points and tangents from a source HermiteCurves prim to a target HermiteCurves prim.

    Args:
        source_prim (UsdGeom.HermiteCurves): The source HermiteCurves prim to retarget from.
        target_prim (UsdGeom.HermiteCurves): The target HermiteCurves prim to retarget to.

    Raises:
        ValueError: If the source or target prim is not a valid HermiteCurves prim.
    """
    if not source_prim:
        raise ValueError("Invalid source HermiteCurves prim")
    if not target_prim:
        raise ValueError("Invalid target HermiteCurves prim")
    source_points_attr = source_prim.GetPointsAttr()
    source_tangents_attr = source_prim.GetTangentsAttr()
    target_points_attr = target_prim.GetPointsAttr()
    target_tangents_attr = target_prim.GetTangentsAttr()
    if source_points_attr.HasValue():
        target_points_attr.Set(source_points_attr.Get())
    if source_tangents_attr.HasValue():
        target_tangents_attr.Set(source_tangents_attr.Get())


def get_interpolated_curve_points(
    points: List[Tuple[float, float, float]], tangents: List[Tuple[float, float, float]], num_points: int
) -> List[Tuple[float, float, float]]:
    """
    Interpolate points along a Hermite curve defined by control points and tangents.

    Args:
        points (List[Tuple[float, float, float]]): Control points of the curve.
        tangents (List[Tuple[float, float, float]]): Tangent vectors at each control point.
        num_points (int): Number of points to interpolate along the curve.

    Returns:
        List[Tuple[float, float, float]]: Interpolated points along the curve.
    """
    if len(points) != len(tangents):
        raise ValueError("Points and tangents must have the same length.")
    if len(points) < 2:
        raise ValueError("At least two control points are required.")
    if num_points <= 0:
        raise ValueError("Number of points to interpolate must be greater than zero.")

    # Generate t values
    t_values = [i / (num_points - 1) for i in range(num_points)]
    interpolated_points = [(0.0, 0.0, 0.0)] * num_points

    for i in range(len(points) - 1):
        p0, p1 = points[i], points[i + 1]
        t0, t1 = tangents[i], tangents[i + 1]

        # Calculate segment t values
        segment_start = i / (len(points) - 1)
        segment_end = (i + 1) / (len(points) - 1)

        start_index = int(i * num_points / (len(points) - 1))
        end_index = int((i + 1) * num_points / (len(points) - 1))

        for idx in range(start_index, end_index):
            t = (t_values[idx] - segment_start) * (len(points) - 1)

            # Calculate Hermite basis functions
            h00 = (1 + 2 * t) * (1 - t) ** 2
            h10 = t * (1 - t) ** 2
            h01 = t**2 * (3 - 2 * t)
            h11 = t**2 * (t - 1)

            # Calculate interpolated point components
            x = h00 * p0[0] + h10 * t0[0] + h01 * p1[0] + h11 * t1[0]
            y = h00 * p0[1] + h10 * t0[1] + h01 * p1[1] + h11 * t1[1]
            z = h00 * p0[2] + h10 * t0[2] + h01 * p1[2] + h11 * t1[2]

            interpolated_points[idx] = (x, y, z)

    return interpolated_points


def is_curve_empty(point_and_tangent_arrays: UsdGeom.HermiteCurves.PointAndTangentArrays) -> bool:
    """Check if the HermiteCurves PointAndTangentArrays is empty.

    Args:
        point_and_tangent_arrays (UsdGeom.HermiteCurves.PointAndTangentArrays): The PointAndTangentArrays object to check.

    Returns:
        bool: True if the PointAndTangentArrays is empty, False otherwise.
    """
    is_empty = point_and_tangent_arrays.IsEmpty()
    return is_empty


def convert_curve_to_uniform(curve: UsdGeom.HermiteCurves, count: int) -> UsdGeom.BasisCurves:
    """Convert a HermiteCurves prim to a BasisCurves prim with uniform knots.

    Args:
        curve (UsdGeom.HermiteCurves): The input HermiteCurves prim.
        count (int): The number of vertices in the output BasisCurves prim.

    Returns:
        UsdGeom.BasisCurves: The output BasisCurves prim.
    """
    if not curve.GetPrim().IsValid():
        raise ValueError("Invalid input HermiteCurves prim.")
    if count <= 0:
        raise ValueError("Count must be a positive integer.")
    points_attr = curve.GetPointsAttr()
    tangents_attr = curve.GetTangentsAttr()
    if not points_attr.IsValid() or not tangents_attr.IsValid():
        raise ValueError("HermiteCurves prim is missing points or tangents.")
    points = points_attr.Get()
    tangents = tangents_attr.Get()
    stage = curve.GetPrim().GetStage()
    basis_curves_path = curve.GetPrim().GetPath().AppendPath("UniformCurve")
    basis_curves = UsdGeom.BasisCurves.Define(stage, basis_curves_path)
    basis_curves.CreatePointsAttr(points)
    basis_curves.CreateCurveVertexCountsAttr([count])
    basis_curves.CreateBasisAttr(UsdGeom.Tokens.bspline)
    basis_curves.CreateTypeAttr(UsdGeom.Tokens.cubic)
    basis_curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    return basis_curves


def split_curve_points_and_tangents(
    point_and_tangent_arrays: UsdGeom.HermiteCurves.PointAndTangentArrays,
) -> Tuple[Vt.Vec3fArray, Vt.Vec3fArray]:
    """Split the interleaved points and tangents into separate arrays.

    Args:
        point_and_tangent_arrays (UsdGeom.HermiteCurves.PointAndTangentArrays): The interleaved points and tangents.

    Returns:
        Tuple[Vt.Vec3fArray, Vt.Vec3fArray]: A tuple containing the separated points and tangents arrays.
    """
    if point_and_tangent_arrays.IsEmpty():
        return (Vt.Vec3fArray(), Vt.Vec3fArray())
    interleaved_array = point_and_tangent_arrays.Interleave()
    num_elements = len(interleaved_array)
    if num_elements % 2 != 0:
        raise ValueError("Invalid interleaved array. Number of elements must be even.")
    num_points = num_elements // 2
    points = Vt.Vec3fArray(interleaved_array[::2][:num_points])
    tangents = Vt.Vec3fArray(interleaved_array[1::2][:num_points])
    return (points, tangents)


def set_curve_points_and_tangents(
    curve: UsdGeom.HermiteCurves, points: List[Gf.Vec3f], tangents: List[Gf.Vec3f]
) -> None:
    """Set the points and tangents for a Hermite curve.

    Args:
        curve (UsdGeom.HermiteCurves): The Hermite curve to set points and tangents for.
        points (List[Gf.Vec3f]): The points of the curve.
        tangents (List[Gf.Vec3f]): The tangents of the curve.

    Raises:
        ValueError: If the number of points and tangents do not match.
    """
    if len(points) != len(tangents):
        raise ValueError("Number of points and tangents must match.")
    points_array = Vt.Vec3fArray(points)
    tangents_array = Vt.Vec3fArray(tangents)
    curve.GetPointsAttr().Set(points_array)
    curve.GetTangentsAttr().Set(tangents_array)


def create_curves_with_vertex_counts(
    stage: Usd.Stage,
    prim_path: str,
    curve_vertex_counts: Vt.IntArray,
    points: Vt.Vec3fArray,
    widths: Vt.FloatArray = None,
) -> UsdGeom.BasisCurves:
    """Create a BasisCurves prim with the given vertex counts, points, and optional widths.

    Args:
        stage (Usd.Stage): The stage to create the prim on.
        prim_path (str): The path where the prim will be created.
        curve_vertex_counts (Vt.IntArray): The number of vertices for each curve.
        points (Vt.Vec3fArray): The points for the curves.
        widths (Vt.FloatArray, optional): The widths for the curves. Defaults to None.

    Returns:
        UsdGeom.BasisCurves: The created BasisCurves prim.

    Raises:
        ValueError: If the number of points does not match the sum of curve_vertex_counts.
    """
    if len(points) != sum(curve_vertex_counts):
        raise ValueError("Number of points must match sum of curve_vertex_counts")
    curves = UsdGeom.BasisCurves.Define(stage, Sdf.Path(prim_path))
    curves.CreateCurveVertexCountsAttr(curve_vertex_counts)
    curves.CreatePointsAttr(points)
    if widths is not None:
        curves.CreateWidthsAttr(widths)
        curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    return curves


def compute_curves_extent(
    points: List[Gf.Vec3f], widths: List[float], transform: Optional[Gf.Matrix4d] = None
) -> Tuple[Gf.Vec3f, Gf.Vec3f]:
    """
    Compute the extent for the curves defined by points and widths.

    Args:
        points (List[Gf.Vec3f]): List of curve points.
        widths (List[float]): List of curve widths.
        transform (Optional[Gf.Matrix4d]): Optional transform matrix to apply.

    Returns:
        Tuple[Gf.Vec3f, Gf.Vec3f]: The minimum and maximum extent of the curves.
    """
    if not points:
        raise ValueError("Points list is empty.")
    if len(points) != len(widths):
        raise ValueError("Points and widths lists must have the same length.")
    min_extent = Gf.Vec3f(points[0])
    max_extent = Gf.Vec3f(points[0])
    for point, width in zip(points, widths):
        if transform:
            point = transform.Transform(point)
        min_extent = Gf.Vec3f(
            min(min_extent[0], point[0] - width * 0.5),
            min(min_extent[1], point[1] - width * 0.5),
            min(min_extent[2], point[2] - width * 0.5),
        )
        max_extent = Gf.Vec3f(
            max(max_extent[0], point[0] + width * 0.5),
            max(max_extent[1], point[1] + width * 0.5),
            max(max_extent[2], point[2] + width * 0.5),
        )
    return (min_extent, max_extent)


def set_curve_widths(
    curve: UsdGeom.Curves, widths: Vt.FloatArray, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> None:
    """Set the widths attribute on a UsdGeomCurves primitive.

    Args:
        curve (UsdGeom.Curves): The curves primitive to set the widths on.
        widths (Vt.FloatArray): The width values to set.
        time_code (Usd.TimeCode, optional): The time code to set the value at. Defaults to Default.

    Raises:
        ValueError: If the curve prim is not valid or if the widths array is empty.
    """
    if not curve.GetPrim().IsValid():
        raise ValueError("The given curve prim is not valid.")
    if not len(widths):
        raise ValueError("The widths array must not be empty.")
    widths_attr = curve.GetWidthsAttr()
    if not widths_attr:
        widths_attr = curve.CreateWidthsAttr()
    widths_attr.Set(widths, time_code)
    curve.SetWidthsInterpolation(UsdGeom.Tokens.vertex)


def animate_curve_points(curve: UsdGeom.Curves, points: Sequence[Gf.Vec3f], time_samples: Sequence[float]) -> None:
    """Animate the points of a UsdGeom.Curves prim over time.

    Args:
        curve (UsdGeom.Curves): The curve prim to animate.
        points (Sequence[Gf.Vec3f]): The points to set at each time sample.
        time_samples (Sequence[float]): The time samples to set the points at.

    Raises:
        ValueError: If the curve prim is not valid, the number of time samples doesn't match
                    the number of point arrays, or the lengths of the point arrays differ.
    """
    if not curve:
        raise ValueError("Curve prim is not valid")
    if len(points) != len(time_samples):
        raise ValueError(
            f"Number of point arrays ({len(points)}) does not match number of time samples ({len(time_samples)})"
        )
    points_attr = curve.GetPointsAttr()
    if not points_attr:
        raise ValueError("Failed to get points attribute")
    num_points = len(points[0])
    for i, pts in enumerate(points):
        if len(pts) != num_points:
            raise ValueError(f"Length of points array at index {i} does not match the first array")
    for pts, time in zip(points, time_samples):
        points_attr.Set(pts, Usd.TimeCode(time))


def create_curve_from_points(
    stage: Usd.Stage, curve_path: str, points: Vt.Vec3fArray, widths: Vt.FloatArray = None, order: int = 4
) -> UsdGeom.NurbsCurves:
    """Create a NURBS curve from a set of points.

    Args:
        stage (Usd.Stage): The USD stage to create the curve in.
        curve_path (str): The path where the curve should be created.
        points (Vt.Vec3fArray): The points defining the curve.
        widths (Vt.FloatArray, optional): The widths of the curve at each point. Defaults to None.
        order (int, optional): The order of the NURBS curve. Defaults to 4.

    Returns:
        UsdGeom.NurbsCurves: The created NURBS curve prim.
    """
    if not stage:
        raise ValueError("Invalid stage")
    if not curve_path:
        raise ValueError("Invalid curve path")
    if not points or len(points) < 4:
        raise ValueError("At least 4 points are required to create a NURBS curve")
    if widths and len(widths) != len(points):
        raise ValueError("Number of widths must match number of points")
    if order < 2:
        raise ValueError("Order must be at least 2")
    curve = UsdGeom.NurbsCurves.Define(stage, curve_path)
    curve.GetPointsAttr().Set(points)
    if widths:
        curve.GetWidthsAttr().Set(widths)
    curve.GetOrderAttr().Set([order])
    num_verts = len(points) - order
    knots = [0] * (order - 1) + list(range(1, num_verts)) + [num_verts] * (order - 1)
    curve.GetKnotsAttr().Set(knots)
    curve.GetRangesAttr().Set([Gf.Vec2d(0, num_verts)])
    return curve


def set_curve_vertex_counts(curves: UsdGeom.Curves, curve_vertex_counts: Sequence[int]) -> None:
    """Set the curve vertex counts for a Curves prim.

    Args:
        curves (UsdGeom.Curves): The Curves prim to set the curve vertex counts for.
        curve_vertex_counts (Sequence[int]): The curve vertex counts to set.

    Raises:
        ValueError: If the Curves prim is not valid.
        TypeError: If curve_vertex_counts is not a sequence of integers.
    """
    if not curves.GetPrim().IsValid():
        raise ValueError("Curves prim is not valid")
    if not all((isinstance(count, int) for count in curve_vertex_counts)):
        raise TypeError("curve_vertex_counts must be a sequence of integers")
    curve_vertex_counts_attr = curves.GetCurveVertexCountsAttr()
    curve_vertex_counts_attr.Set(curve_vertex_counts)


def get_curve_points(curves_prim: UsdGeom.Curves) -> List[Tuple[Gf.Vec3f]]:
    """
    Returns a list of points for each curve in the Curves prim.

    Args:
        curves_prim (UsdGeom.Curves): The Curves prim to get points from.

    Returns:
        List[Tuple[Gf.Vec3f]]: A list of tuples, where each tuple contains the points for one curve.
    """
    if not curves_prim.GetPrim().IsValid():
        raise ValueError("Invalid Curves prim")
    points_attr = curves_prim.GetPointsAttr()
    if not points_attr.IsValid():
        raise ValueError("Points attribute is not valid")
    curve_vertex_counts_attr = curves_prim.GetCurveVertexCountsAttr()
    if not curve_vertex_counts_attr.IsValid():
        raise ValueError("Curve vertex counts attribute is not valid")
    points = points_attr.Get(Usd.TimeCode.Default())
    curve_vertex_counts = curve_vertex_counts_attr.Get(Usd.TimeCode.Default())
    result = []
    start_index = 0
    for vertex_count in curve_vertex_counts:
        end_index = start_index + vertex_count
        curve_points = points[start_index:end_index]
        result.append(tuple(curve_points))
        start_index = end_index
    return result


def merge_multiple_curves(
    stage: Usd.Stage, curve_paths: List[Sdf.Path], merged_curve_path: Sdf.Path
) -> UsdGeom.BasisCurves:
    """Merge multiple curves into a single curve primitive.

    Args:
        stage (Usd.Stage): The USD stage.
        curve_paths (List[Sdf.Path]): A list of paths to the curve primitives to merge.
        merged_curve_path (Sdf.Path): The path for the merged curve primitive.

    Returns:
        UsdGeom.BasisCurves: The merged curve primitive.

    Raises:
        ValueError: If any of the input curve primitives are invalid or if the merged curve path already exists.
    """
    if stage.GetPrimAtPath(merged_curve_path):
        raise ValueError(f"Merged curve path {merged_curve_path} already exists.")
    merged_curve = UsdGeom.BasisCurves.Define(stage, merged_curve_path)
    merged_points = []
    merged_widths = []
    merged_vertex_counts = []
    for curve_path in curve_paths:
        curve = UsdGeom.BasisCurves(stage.GetPrimAtPath(curve_path))
        if not curve:
            raise ValueError(f"Invalid curve primitive at path {curve_path}.")
        points = curve.GetPointsAttr().Get()
        widths = curve.GetWidthsAttr().Get()
        vertex_counts = curve.GetCurveVertexCountsAttr().Get()
        merged_points.extend(points)
        merged_widths.extend(widths)
        merged_vertex_counts.extend(vertex_counts)
    merged_curve.GetPointsAttr().Set(merged_points)
    merged_curve.GetWidthsAttr().Set(merged_widths)
    merged_curve.GetCurveVertexCountsAttr().Set(merged_vertex_counts)
    return merged_curve


def modify_curve_widths_based_on_distance(
    curves_prim: UsdGeom.Curves, distances: List[float], time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> None:
    """Modify the widths of a Curves prim based on a list of distances.

    The widths are modified in place on the Curves prim. The number of widths
    should match the number of distances provided. If the number of distances
    is less than the number of widths, the remaining widths are unchanged.
    If the number of distances is greater than the number of widths, the extra
    distances are ignored.

    Args:
        curves_prim (UsdGeom.Curves): The Curves prim to modify.
        distances (List[float]): The distances to set the widths to.
        time_code (Usd.TimeCode): The time code to set the widths at. Defaults to Default.

    Raises:
        ValueError: If the curves prim is not valid or if the widths attribute does not exist.
    """
    if not curves_prim.GetPrim().IsValid():
        raise ValueError("The provided Curves prim is not valid.")
    widths_attr = curves_prim.GetWidthsAttr()
    if not widths_attr.HasValue():
        raise ValueError("The widths attribute does not exist on the Curves prim.")
    widths = widths_attr.Get(time_code)
    for i in range(min(len(widths), len(distances))):
        widths[i] = distances[i]
    widths_attr.Set(widths, time_code)


def set_cylinder_attributes(cylinder: UsdGeom.Cylinder, height: float, radius: float, axis: str = "Z") -> None:
    """Set the attributes of a UsdGeom.Cylinder.

    Args:
        cylinder (UsdGeom.Cylinder): The cylinder prim to set attributes on.
        height (float): The height of the cylinder.
        radius (float): The radius of the cylinder.
        axis (str, optional): The axis along which the cylinder is oriented. Defaults to "Z".

    Raises:
        ValueError: If the input cylinder is not a valid UsdGeom.Cylinder prim.
    """
    if not cylinder or not isinstance(cylinder, UsdGeom.Cylinder):
        raise ValueError("Invalid UsdGeom.Cylinder prim provided.")
    height_attr = cylinder.GetHeightAttr()
    if height_attr:
        height_attr.Set(height)
    else:
        cylinder.CreateHeightAttr(height)
    radius_attr = cylinder.GetRadiusAttr()
    if radius_attr:
        radius_attr.Set(radius)
    else:
        cylinder.CreateRadiusAttr(radius)
    axis_attr = cylinder.GetAxisAttr()
    if axis_attr:
        axis_attr.Set(axis)
    else:
        cylinder.CreateAxisAttr(axis)
    half_height = height / 2
    min_extent = Gf.Vec3f(-radius, -half_height, -radius)
    max_extent = Gf.Vec3f(radius, half_height, radius)
    extent_attr = cylinder.GetExtentAttr()
    if extent_attr:
        extent_attr.Set([min_extent, max_extent])
    else:
        cylinder.CreateExtentAttr([min_extent, max_extent])


def get_cylinder_volume(cylinder: UsdGeom.Cylinder) -> float:
    """Calculate the volume of a cylinder prim.

    Args:
        cylinder (UsdGeom.Cylinder): The cylinder prim to calculate the volume for.

    Returns:
        float: The volume of the cylinder.

    Raises:
        ValueError: If the input prim is not a valid cylinder.
    """
    if not cylinder or not cylinder.GetPrim().IsValid():
        raise ValueError("Invalid cylinder prim.")
    radius_attr = cylinder.GetRadiusAttr()
    height_attr = cylinder.GetHeightAttr()
    if not radius_attr.HasAuthoredValue() or not height_attr.HasAuthoredValue():
        raise ValueError("Radius and height attributes must be authored.")
    radius = radius_attr.Get()
    height = height_attr.Get()
    volume = math.pi * radius**2 * height
    return volume


def attach_material_to_cylinder(stage: Usd.Stage, cylinder_path: str, material_path: str) -> None:
    """Attaches a material to a cylinder prim.

    Args:
        stage (Usd.Stage): The USD stage.
        cylinder_path (str): The path to the cylinder prim.
        material_path (str): The path to the material prim.

    Raises:
        ValueError: If the cylinder prim or material prim does not exist.
    """
    cylinder_prim = stage.GetPrimAtPath(cylinder_path)
    if not cylinder_prim.IsValid():
        raise ValueError(f"Cylinder prim at path {cylinder_path} does not exist.")
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise ValueError(f"Material prim at path {material_path} does not exist.")
    cylinder_mesh = UsdGeom.Cylinder(cylinder_prim)
    if not cylinder_mesh:
        raise ValueError(f"Prim at path {cylinder_path} is not a valid Cylinder.")
    material = UsdShade.Material(material_prim)
    if not material:
        raise ValueError(f"Prim at path {material_path} is not a valid Material.")
    UsdShade.MaterialBindingAPI(cylinder_prim).Bind(material)


def animate_cylinder_height(stage: Usd.Stage, cylinder_path: str, height_values: list, time_samples: list) -> None:
    """Animate the height attribute of a UsdGeomCylinder prim over time.

    Args:
        stage (Usd.Stage): The USD stage.
        cylinder_path (str): The path to the UsdGeomCylinder prim.
        height_values (list): The height values to set at each time sample.
        time_samples (list): The time samples corresponding to each height value.

    Raises:
        ValueError: If the cylinder prim is not found or is not a UsdGeomCylinder.
        ValueError: If the number of height values does not match the number of time samples.
    """
    cylinder_prim = stage.GetPrimAtPath(cylinder_path)
    if not cylinder_prim.IsValid():
        raise ValueError(f"Prim not found at path: {cylinder_path}")
    if not UsdGeom.Cylinder(cylinder_prim):
        raise ValueError(f"Prim at path {cylinder_path} is not a UsdGeomCylinder")
    if len(height_values) != len(time_samples):
        raise ValueError("Number of height values does not match the number of time samples")
    height_attr = UsdGeom.Cylinder(cylinder_prim).GetHeightAttr()
    for height, time in zip(height_values, time_samples):
        height_attr.Set(height, time)


def get_cylinder_extent(cylinder: UsdGeom.Cylinder) -> Tuple[Gf.Vec3f, Gf.Vec3f]:
    """
    Get the extent of a USD Cylinder.

    Args:
        cylinder (UsdGeom.Cylinder): The cylinder to get the extent for.

    Returns:
        Tuple[Gf.Vec3f, Gf.Vec3f]: A tuple with the min and max extent.
    """
    if not cylinder:
        raise ValueError("Invalid cylinder")
    extent_attr = cylinder.GetExtentAttr()
    if not extent_attr:
        extent = UsdGeom.Cylinder.GetExtentAttr().GetDefaultValue()
    else:
        extent = extent_attr.Get()
    if not extent or len(extent) != 2:
        raise ValueError("Invalid extent")
    return (extent[0], extent[1])


def align_cylinders(stage: Usd.Stage, cylinder_paths: List[str], target_path: str) -> None:
    """
    Align the cylinders to the target prim's orientation.

    Args:
        stage (Usd.Stage): The USD stage.
        cylinder_paths (List[str]): The paths to the cylinder prims.
        target_path (str): The path to the target prim.

    Raises:
        ValueError: If any of the provided paths are invalid or not of the correct type.
    """
    target_prim = stage.GetPrimAtPath(target_path)
    if not target_prim.IsValid():
        raise ValueError(f"Target prim at path {target_path} does not exist.")
    target_xformable = UsdGeom.Xformable(target_prim)
    if not target_xformable:
        raise ValueError(f"Target prim at path {target_path} is not transformable.")
    target_rotation = None
    for op in target_xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            target_rotation = op.Get()
            break
    if target_rotation is None:
        raise ValueError(f"Target prim at path {target_path} does not have a RotateXYZ transform operation.")
    for cylinder_path in cylinder_paths:
        cylinder_prim = stage.GetPrimAtPath(cylinder_path)
        if not cylinder_prim.IsValid():
            raise ValueError(f"Cylinder prim at path {cylinder_path} does not exist.")
        cylinder = UsdGeom.Cylinder(cylinder_prim)
        if not cylinder:
            raise ValueError(f"Prim at path {cylinder_path} is not a Cylinder.")
        cylinder_xformable = UsdGeom.Xformable(cylinder_prim)
        if not cylinder_xformable:
            raise ValueError(f"Cylinder prim at path {cylinder_path} is not transformable.")
        add_rotate_xyz_op(cylinder_xformable).Set(target_rotation)


def duplicate_cylinder(stage: Usd.Stage, prim_path: str, new_prim_path: str) -> UsdGeom.Cylinder:
    """Duplicate a cylinder prim at a new path."""
    original_cylinder = UsdGeom.Cylinder(stage.GetPrimAtPath(prim_path))
    if not original_cylinder:
        raise ValueError(f"Prim at path {prim_path} is not a valid Cylinder.")
    new_cylinder = UsdGeom.Cylinder.Define(stage, new_prim_path)
    for attr in original_cylinder.GetPrim().GetAuthoredAttributes():
        attr_name = attr.GetName()
        attr_value = attr.Get()
        new_cylinder.GetPrim().CreateAttribute(attr_name, attr.GetTypeName()).Set(attr_value)
    return new_cylinder


def merge_cylinders(stage: Usd.Stage, parent_path: str, cylinder_paths: List[str]) -> Usd.Prim:
    """Merge multiple cylinders into a single mesh.

    Args:
        stage (Usd.Stage): The USD stage.
        parent_path (str): The path of the parent prim for the merged mesh.
        cylinder_paths (List[str]): A list of paths to the cylinder prims to merge.

    Returns:
        Usd.Prim: The merged mesh prim.
    """
    merged_prim = UsdGeom.Mesh.Define(stage, parent_path)
    points = []
    for cylinder_path in cylinder_paths:
        cylinder_prim = stage.GetPrimAtPath(cylinder_path)
        if not cylinder_prim.IsValid():
            raise ValueError(f"Cylinder prim at path {cylinder_path} does not exist.")
        cylinder = UsdGeom.Cylinder(cylinder_prim)
        height = cylinder.GetHeightAttr().Get()
        radius = cylinder.GetRadiusAttr().Get()
        transform = UsdGeom.Xformable(cylinder_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        num_segments = 16
        for i in range(num_segments + 1):
            angle = i * (360.0 / num_segments)
            x = radius * math.cos(math.radians(angle))
            z = radius * math.sin(math.radians(angle))
            bottom_point = Gf.Vec3f(x, -height / 2, z)
            bottom_point = transform.Transform(bottom_point)
            points.append(bottom_point)
            top_point = Gf.Vec3f(x, height / 2, z)
            top_point = transform.Transform(top_point)
            points.append(top_point)
    merged_prim.CreatePointsAttr().Set(points)
    extent = UsdGeom.Boundable.ComputeExtentFromPlugins(merged_prim, Usd.TimeCode.Default())
    merged_prim.GetExtentAttr().Set(extent)
    return merged_prim


def set_mesh_face_vertex_indices(mesh: UsdGeom.Mesh, face_vertex_indices: Vt.IntArray) -> None:
    """Set the faceVertexIndices attribute on a mesh.

    Args:
        mesh (UsdGeom.Mesh): The mesh prim to set the faceVertexIndices on.
        face_vertex_indices (Vt.IntArray): The face vertex indices to set.

    Raises:
        ValueError: If the mesh prim is not valid or if the faceVertexIndices
                    array is not the same size as the existing faceVertexCounts.
    """
    if not mesh:
        raise ValueError("Mesh prim is not valid")
    face_vertex_counts_attr = mesh.GetFaceVertexCountsAttr()
    if not face_vertex_counts_attr:
        raise ValueError("Mesh prim does not have a faceVertexCounts attribute")
    face_vertex_counts = face_vertex_counts_attr.Get()
    if not face_vertex_counts:
        raise ValueError("Mesh prim does not have a faceVertexCounts value")
    expected_num_indices = sum(face_vertex_counts)
    if len(face_vertex_indices) != expected_num_indices:
        raise ValueError(
            f"Number of face vertex indices ({len(face_vertex_indices)}) does not match expected number based on face vertex counts ({expected_num_indices})"
        )
    mesh.CreateFaceVertexIndicesAttr().Set(face_vertex_indices)


def compute_mesh_normals(mesh: UsdGeom.Mesh) -> Vt.Vec3fArray:
    """Compute vertex normals for a UsdGeomMesh."""
    points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
    if not points or not face_vertex_counts or (not face_vertex_indices):
        raise ValueError("Mesh is missing required attributes.")
    num_points = len(points)
    normals = [Gf.Vec3f(0, 0, 0)] * num_points
    face_index = 0
    vertex_index = 0
    for vertex_count in face_vertex_counts:
        face_indices = face_vertex_indices[vertex_index : vertex_index + vertex_count]
        face_normal = Gf.Vec3f(0, 0, 0)
        for i in range(vertex_count):
            current_index = face_indices[i]
            next_index = face_indices[(i + 1) % vertex_count]
            current_pos = Gf.Vec3f(*points[current_index])
            next_pos = Gf.Vec3f(*points[next_index])
            edge = next_pos - current_pos
            face_normal += Gf.Cross(current_pos, edge)
        face_normal.Normalize()
        for index in face_indices:
            normals[index] += face_normal
        face_index += 1
        vertex_index += vertex_count
    for i in range(num_points):
        normals[i].Normalize()
    return normals


def merge_meshes(stage: Usd.Stage, prim_paths: List[str], merge_prim_path: str) -> UsdGeom.Mesh:
    """Merge multiple mesh prims into a single mesh prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths to merge.
        merge_prim_path (str): Prim path for the merged mesh.

    Returns:
        UsdGeom.Mesh: The merged mesh prim.
    """
    merged_mesh = UsdGeom.Mesh.Define(stage, merge_prim_path)
    merged_points = []
    merged_face_vertex_counts = []
    merged_face_vertex_indices = []
    for prim_path in prim_paths:
        mesh_prim = UsdGeom.Mesh(stage.GetPrimAtPath(prim_path))
        if not mesh_prim:
            raise ValueError(f"Prim at path {prim_path} is not a valid Mesh.")
        points = mesh_prim.GetPointsAttr().Get()
        face_vertex_counts = mesh_prim.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh_prim.GetFaceVertexIndicesAttr().Get()
        merged_points.extend(points)
        merged_face_vertex_counts.extend(face_vertex_counts)
        offset = len(merged_points) - len(points)
        merged_face_vertex_indices.extend([index + offset for index in face_vertex_indices])
    merged_mesh.GetPointsAttr().Set(merged_points)
    merged_mesh.GetFaceVertexCountsAttr().Set(merged_face_vertex_counts)
    merged_mesh.GetFaceVertexIndicesAttr().Set(merged_face_vertex_indices)
    return merged_mesh


def extract_mesh_edges(faceVertexIndices: Vt.IntArray, faceVertexCounts: Vt.IntArray) -> List[Tuple[int, int]]:
    """Extract unique edges from a mesh topology.

    Args:
        faceVertexIndices (Vt.IntArray): Flat list of the vertex indices for each face.
        faceVertexCounts (Vt.IntArray): Number of vertices for each face.

    Returns:
        List[Tuple[int, int]]: List of unique edges, each represented as a tuple of two vertex indices.
    """
    edges = set()
    face_start_index = 0
    for face_vertex_count in faceVertexCounts:
        for i in range(face_vertex_count):
            v1 = faceVertexIndices[face_start_index + i]
            v2 = faceVertexIndices[face_start_index + (i + 1) % face_vertex_count]
            edge = (min(v1, v2), max(v1, v2))
            edges.add(edge)
        face_start_index += face_vertex_count
    return list(edges)


def convert_mesh_to_triangles(mesh: UsdGeom.Mesh) -> Tuple[List[int], List[int]]:
    """Convert a mesh to a list of triangle face vertex indices.

    Args:
        mesh (UsdGeom.Mesh): The input mesh to convert.

    Returns:
        A tuple containing:
            - face_vertex_counts (List[int]): The number of vertices for each face (all 3).
            - face_vertex_indices (List[int]): The vertex indices for each triangle face.
    """
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
    if all((count == 3 for count in face_vertex_counts)):
        return (face_vertex_counts, face_vertex_indices)
    tri_face_vertex_counts = []
    tri_face_vertex_indices = []
    index_offset = 0
    for count in face_vertex_counts:
        for i in range(1, count - 1):
            tri_face_vertex_counts.append(3)
            tri_face_vertex_indices.extend(
                [
                    face_vertex_indices[index_offset],
                    face_vertex_indices[index_offset + i],
                    face_vertex_indices[index_offset + i + 1],
                ]
            )
        index_offset += count
    return (tri_face_vertex_counts, tri_face_vertex_indices)


def smooth_mesh(mesh: UsdGeom.Mesh, smooth_iterations: int = 1) -> None:
    """Smooth the vertices of a polygonal mesh using Laplacian smoothing.

    Args:
        mesh (UsdGeom.Mesh): The polygonal mesh to smooth.
        smooth_iterations (int): The number of smoothing iterations to apply. Defaults to 1.

    Raises:
        ValueError: If the input mesh is not a valid polygonal mesh.
    """
    if not mesh:
        raise ValueError("Invalid mesh provided.")
    if mesh.GetSubdivisionSchemeAttr().Get() != "none":
        raise ValueError("Smoothing only supported for polygonal meshes.")
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not counts or (not indices):
        raise ValueError("Mesh is missing required attributes.")
    num_points = len(points)
    valence = [0] * num_points
    for face_vertex_count in counts:
        for i in range(face_vertex_count):
            valence[indices[i]] += 1
    for _ in range(smooth_iterations):
        new_points = [Gf.Vec3f(0, 0, 0)] * num_points
        offset = 0
        for face_vertex_count in counts:
            for i in range(face_vertex_count):
                curr_idx = indices[offset + i]
                next_idx = indices[offset + (i + 1) % face_vertex_count]
                new_points[curr_idx] += points[next_idx]
            offset += face_vertex_count
        for i in range(num_points):
            if valence[i] > 0:
                new_points[i] /= valence[i]
        points = new_points
    mesh.GetPointsAttr().Set(points)


def create_subdiv_mesh(
    stage: Usd.Stage,
    mesh_path: str,
    positions: Vt.Vec3fArray,
    face_vertex_counts: Vt.IntArray,
    face_vertex_indices: Vt.IntArray,
) -> UsdGeom.Mesh:
    """Create a subdivision mesh prim.

    Args:
        stage (Usd.Stage): The USD stage to create the mesh on.
        mesh_path (str): The path where the mesh prim will be created.
        positions (Vt.Vec3fArray): The vertex positions for the mesh.
        face_vertex_counts (Vt.IntArray): The number of vertices per face.
        face_vertex_indices (Vt.IntArray): The vertex indices for each face.

    Returns:
        UsdGeom.Mesh: The created mesh prim.
    """
    (is_valid, reason) = UsdGeom.Mesh.ValidateTopology(face_vertex_indices, face_vertex_counts, len(positions))
    if not is_valid:
        raise ValueError(f"Invalid mesh topology: {reason}")
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr().Set(positions)
    mesh.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr().Set(face_vertex_indices)
    mesh.CreateSubdivisionSchemeAttr().Set("catmullClark")
    return mesh


def get_mesh_vertex_positions(mesh: UsdGeom.Mesh, time_code: Usd.TimeCode = Usd.TimeCode.Default()) -> Vt.Vec3fArray:
    """
    Get the vertex positions of a mesh at a specific time.

    Args:
        mesh (UsdGeom.Mesh): The mesh to get vertex positions from.
        time_code (Usd.TimeCode): The time code to query the vertex positions at.
            Defaults to Default (current time).

    Returns:
        Vt.Vec3fArray: The vertex positions of the mesh.
    """
    if not mesh:
        raise ValueError("Invalid mesh")
    points_attr = mesh.GetPointsAttr()
    if not points_attr.HasValue():
        return Vt.Vec3fArray()
    vertex_positions = points_attr.Get(time_code)
    return vertex_positions


def decimate_mesh(mesh: UsdGeom.Mesh, target_face_count: int) -> UsdGeom.Mesh:
    """Decimate a mesh to a target face count.

    Args:
        mesh (UsdGeom.Mesh): The input mesh to decimate.
        target_face_count (int): The target number of faces after decimation.

    Returns:
        UsdGeom.Mesh: The decimated mesh.
    """
    if not mesh:
        raise ValueError("Input mesh is invalid.")
    if target_face_count <= 0:
        raise ValueError("Target face count must be greater than zero.")
    original_face_count = len(mesh.GetFaceVertexCountsAttr().Get())
    if target_face_count >= original_face_count:
        return mesh
    decimation_ratio = target_face_count / original_face_count
    decimated_mesh = UsdGeom.Mesh.Define(
        mesh.GetPrim().GetStage(), mesh.GetPrim().GetPath().AppendChild("DecimatedMesh")
    )
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
    decimated_face_vertex_counts = [
        count for (i, count) in enumerate(face_vertex_counts) if i / len(face_vertex_counts) <= decimation_ratio
    ]
    decimated_mesh.GetFaceVertexCountsAttr().Set(decimated_face_vertex_counts)
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
    decimated_face_vertex_indices = face_vertex_indices[: sum(decimated_face_vertex_counts)]
    decimated_mesh.GetFaceVertexIndicesAttr().Set(decimated_face_vertex_indices)
    decimated_mesh.GetPointsAttr().Set(mesh.GetPointsAttr().Get())
    return decimated_mesh


def calculate_mesh_volume(mesh: UsdGeom.Mesh) -> float:
    """Calculate the volume of a mesh.

    Args:
        mesh (UsdGeom.Mesh): The input mesh.

    Returns:
        float: The calculated volume of the mesh.

    Raises:
        ValueError: If the input mesh is not a valid UsdGeom.Mesh object or missing required attributes.
    """
    if not mesh or not isinstance(mesh, UsdGeom.Mesh):
        raise ValueError("Invalid UsdGeom.Mesh object provided.")
    points = mesh.GetPointsAttr().Get()
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
    if points is None or face_vertex_counts is None or face_vertex_indices is None:
        raise ValueError("Mesh is missing required attributes.")
    volume = 0.0
    current_index = 0
    for face_vertex_count in face_vertex_counts:
        face_indices = face_vertex_indices[current_index : current_index + face_vertex_count]
        face_vertices = [Gf.Vec3f(points[index]) for index in face_indices]
        face_volume = 0.0
        for i in range(1, len(face_vertices) - 1):
            cross_product = Gf.Cross(face_vertices[i] - face_vertices[0], 
                                   face_vertices[i + 1] - face_vertices[0])
            dot_product = Gf.Dot(cross_product, face_vertices[0])
            face_volume += dot_product
        volume += face_volume
        current_index += face_vertex_count
    volume /= 6.0
    return abs(volume)


def apply_transform_to_mesh(stage: Usd.Stage, mesh_path: str, transform: Gf.Matrix4d) -> None:
    """Apply a transformation matrix to a mesh.

    Args:
        stage (Usd.Stage): The USD stage.
        mesh_path (str): The path to the mesh prim.
        transform (Gf.Matrix4d): The transformation matrix to apply.
    """
    mesh_prim = stage.GetPrimAtPath(mesh_path)
    if not mesh_prim:
        raise ValueError(f"No prim found at path: {mesh_path}")
    mesh = UsdGeom.Mesh(mesh_prim)
    if not mesh:
        raise ValueError(f"Prim at path {mesh_path} is not a valid Mesh")
    points_attr = mesh.GetPointsAttr()
    if not points_attr:
        raise ValueError(f"Mesh at path {mesh_path} does not have a points attribute")
    points = points_attr.Get()
    transformed_points = [transform.Transform(point) for point in points]
    points_attr.Set(transformed_points)


def export_mesh_as_obj(mesh: UsdGeom.Mesh, file_path: str) -> bool:
    """Exports a USD Mesh prim to an OBJ file.

    Args:
        mesh (UsdGeom.Mesh): The USD Mesh prim to export.
        file_path (str): The file path to save the OBJ file.

    Returns:
        bool: True if the export is successful, False otherwise.
    """
    if not mesh or not mesh.GetPrim().IsValid():
        return False
    points = mesh.GetPointsAttr().Get()
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not face_vertex_counts or (not face_vertex_indices):
        return False
    with open(file_path, "w") as file:
        for point in points:
            file.write(f"v {point[0]} {point[1]} {point[2]}\n")
        face_index = 0
        for face_vertex_count in face_vertex_counts:
            face_indices = face_vertex_indices[face_index : face_index + face_vertex_count]
            face_index += face_vertex_count
            face_str = " ".join((str(index + 1) for index in face_indices))
            file.write(f"f {face_str}\n")
    return True


def set_mesh_subdivision_scheme(mesh: UsdGeom.Mesh, scheme: str):
    """Set the subdivision scheme for a mesh.

    Args:
        mesh (UsdGeom.Mesh): The mesh to set the subdivision scheme for.
        scheme (str): The subdivision scheme to set. Must be one of "catmullClark", "loop", "bilinear", or "none".

    Raises:
        ValueError: If the mesh is not a valid UsdGeom.Mesh, or if the scheme is not one of the allowed values.
    """
    if not mesh or not isinstance(mesh, UsdGeom.Mesh):
        raise ValueError("Invalid UsdGeom.Mesh object")
    allowed_schemes = ["catmullClark", "loop", "bilinear", "none"]
    if scheme not in allowed_schemes:
        raise ValueError(f"Invalid subdivision scheme '{scheme}'. Must be one of {allowed_schemes}")
    subdivision_scheme_attr = mesh.GetSubdivisionSchemeAttr()
    subdivision_scheme_attr.Set(scheme)


def set_mesh_face_vertex_counts(mesh: UsdGeom.Mesh, face_vertex_counts: Sequence[int]) -> None:
    """Set the face vertex counts for a mesh.

    Args:
        mesh (UsdGeom.Mesh): The mesh to set the face vertex counts on.
        face_vertex_counts (Sequence[int]): The face vertex counts to set.

    Raises:
        ValueError: If the mesh is not valid.
        ValueError: If the face vertex counts are not valid.
    """
    if not mesh:
        raise ValueError("Mesh is not valid")
    if not face_vertex_counts:
        raise ValueError("Face vertex counts are empty")
    if any((count < 3 for count in face_vertex_counts)):
        raise ValueError("Face vertex counts must be at least 3")
    counts_attr = mesh.GetFaceVertexCountsAttr()
    if not counts_attr:
        counts_attr = mesh.CreateFaceVertexCountsAttr()
    counts_attr.Set(face_vertex_counts)


def weld_mesh_vertices(mesh: UsdGeom.Mesh, points_to_weld: Dict[int, int], weld_tolerance: float = 1e-05) -> bool:
    """
    Weld vertices of a mesh that are within a certain distance.

    Args:
        mesh (UsdGeom.Mesh): The mesh to modify.
        points_to_weld (Dict[int, int]): Dictionary mapping point indices to be welded. Key is the point to remove, value is the point to weld to.
        weld_tolerance (float): The maximum distance between two points for them to be considered coincident. Default is 1e-5.

    Returns:
        bool: True if any vertices were welded, False otherwise.
    """
    points = mesh.GetPointsAttr().Get()
    old_to_new_indices = {}
    for old_index, new_index in points_to_weld.items():
        old_point = points[old_index]
        new_point = points[new_index]
        if Gf.Vec3d(old_point).GetLength() - Gf.Vec3d(new_point).GetLength() > weld_tolerance:
            continue
        old_to_new_indices[old_index] = new_index
    if not old_to_new_indices:
        return False
    new_face_vertex_counts = []
    new_face_vertex_indices = []
    for count, start_index in zip(
        mesh.GetFaceVertexCountsAttr().Get(),
        range(0, len(mesh.GetFaceVertexIndicesAttr().Get()), mesh.GetFaceVertexCountsAttr().Get()[0]),
    ):
        indices = mesh.GetFaceVertexIndicesAttr().Get()[start_index : start_index + count]
        new_indices = [old_to_new_indices.get(i, i) for i in indices]
        new_face_vertex_counts.append(len(new_indices))
        new_face_vertex_indices.extend(new_indices)
    new_points = [p for (i, p) in enumerate(points) if i not in old_to_new_indices]
    mesh.GetPointsAttr().Set(new_points)
    mesh.GetFaceVertexCountsAttr().Set(new_face_vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(new_face_vertex_indices)
    return True


def smooth_mesh_normals(mesh: UsdGeom.Mesh, smoothing_iterations: int = 1) -> None:
    """Smooth the normals of a mesh using a simple iterative averaging algorithm.

    Args:
        mesh (UsdGeom.Mesh): The mesh to smooth the normals of.
        smoothing_iterations (int): The number of smoothing iterations to perform. Defaults to 1.

    Raises:
        ValueError: If the mesh is not a triangle mesh.
    """
    if not mesh:
        raise ValueError("Mesh is invalid")
    counts = mesh.GetFaceVertexCountsAttr().Get()
    if any((count != 3 for count in counts)):
        raise ValueError("Mesh must be a triangle mesh")
    points = mesh.GetPointsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    point_to_face = [[] for _ in range(len(points))]
    for face_idx, i in enumerate(range(0, len(indices), 3)):
        for j in range(3):
            point_to_face[indices[i + j]].append(face_idx)
    point_to_point = [set() for _ in range(len(points))]
    for face_indices in point_to_face:
        for i in range(len(face_indices)):
            for j in range(i + 1, len(face_indices)):
                (idx1, idx2) = (indices[face_indices[i] * 3], indices[face_indices[j] * 3])
                point_to_point[idx1].add(idx2)
                point_to_point[idx2].add(idx1)
    normals = [Gf.Vec3f(0, 0, 0)] * len(points)
    for face_idx in range(len(counts)):
        i = face_idx * 3
        (p0, p1, p2) = (
            Gf.Vec3f(points[indices[i]]),
            Gf.Vec3f(points[indices[i + 1]]),
            Gf.Vec3f(points[indices[i + 2]]),
        )
        face_normal = Gf.Cross(p1 - p0, p2 - p0).GetNormalized()
        for j in range(3):
            normals[indices[i + j]] += face_normal
    for i in range(len(normals)):
        normals[i].Normalize()
    for _ in range(smoothing_iterations):
        new_normals = [Gf.Vec3f(0, 0, 0)] * len(points)
        for i in range(len(points)):
            normal_sum = normals[i]
            for j in point_to_point[i]:
                normal_sum += normals[j]
            new_normals[i] = normal_sum.GetNormalized()
        normals = new_normals
    mesh.CreateNormalsAttr(normals)


def extract_mesh_face_indices(mesh: UsdGeom.Mesh) -> List[List[int]]:
    """Extract the face vertex indices for each face of a mesh.

    Args:
        mesh (UsdGeom.Mesh): The input mesh.

    Returns:
        List[List[int]]: A list of lists, where each inner list contains the vertex
        indices for a single face.
    """
    counts_attr = mesh.GetFaceVertexCountsAttr()
    if not counts_attr.HasValue():
        raise ValueError("Mesh does not have face vertex counts.")
    indices_attr = mesh.GetFaceVertexIndicesAttr()
    if not indices_attr.HasValue():
        raise ValueError("Mesh does not have face vertex indices.")
    counts = counts_attr.Get()
    indices = indices_attr.Get()
    num_points = len(mesh.GetPointsAttr().Get())
    (valid, reason) = UsdGeom.Mesh.ValidateTopology(indices, counts, num_points)
    if not valid:
        raise ValueError(f"Mesh has invalid topology: {reason}")
    face_indices = []
    index = 0
    for count in counts:
        face = indices[index : index + count]
        face_indices.append(face)
        index += count
    return face_indices


def set_mesh_vertex_positions(mesh: UsdGeom.Mesh, positions: Sequence[Gf.Vec3f]) -> None:
    """Set the vertex positions for a mesh.

    Args:
        mesh (UsdGeom.Mesh): The mesh to set vertex positions for.
        positions (Sequence[Gf.Vec3f]): The vertex positions to set.

    Raises:
        ValueError: If the mesh is not valid.
    """
    if not mesh or not mesh.GetPrim().IsValid():
        raise ValueError("Mesh is not valid.")
    points_attr = mesh.GetPointsAttr()
    if not points_attr:
        points_attr = mesh.CreatePointsAttr()
    points_attr.Set(positions)


def add_mesh_creases(
    mesh: UsdGeom.Mesh, crease_indices: List[int], crease_lengths: List[int], crease_sharpnesses: List[float]
) -> None:
    """Add creases to a USD Mesh.

    Args:
        mesh (UsdGeom.Mesh): The mesh to add creases to.
        crease_indices (List[int]): The vertex indices of the creases.
        crease_lengths (List[int]): The length of each crease.
        crease_sharpnesses (List[float]): The sharpness value for each crease.
    """
    if len(crease_indices) != sum(crease_lengths):
        raise ValueError("Length of crease_indices must match sum of crease_lengths")
    if len(crease_lengths) != len(crease_sharpnesses):
        raise ValueError("Length of crease_lengths must match length of crease_sharpnesses")
    crease_indices_attr = mesh.GetCreaseIndicesAttr()
    if not crease_indices_attr:
        crease_indices_attr = mesh.CreateCreaseIndicesAttr()
    crease_lengths_attr = mesh.GetCreaseLengthsAttr()
    if not crease_lengths_attr:
        crease_lengths_attr = mesh.CreateCreaseLengthsAttr()
    crease_sharpness_attr = mesh.GetCreaseSharpnessesAttr()
    if not crease_sharpness_attr:
        crease_sharpness_attr = mesh.CreateCreaseSharpnessesAttr()
    crease_indices_attr.Set(crease_indices_attr.Get() + Vt.IntArray(crease_indices))
    crease_lengths_attr.Set(crease_lengths_attr.Get() + Vt.IntArray(crease_lengths))
    crease_sharpness_attr.Set(crease_sharpness_attr.Get() + Vt.FloatArray(crease_sharpnesses))


def add_mesh_holes(mesh: UsdGeom.Mesh, hole_face_indices: List[int]) -> None:
    """Adds hole faces to a UsdGeom.Mesh.

    Args:
        mesh (UsdGeom.Mesh): The mesh to add holes to.
        hole_face_indices (List[int]): A list of face indices to mark as holes.
    """
    hole_indices_attr = mesh.GetHoleIndicesAttr()
    if hole_indices_attr.IsValid():
        existing_hole_indices = hole_indices_attr.Get()
        all_hole_indices = existing_hole_indices + Vt.IntArray(hole_face_indices)
    else:
        all_hole_indices = Vt.IntArray(hole_face_indices)
    mesh.CreateHoleIndicesAttr(all_hole_indices)


def transfer_uvs(source_mesh: UsdGeom.Mesh, target_mesh: UsdGeom.Mesh) -> None:
    """Transfer UV coordinates from source mesh to target mesh.

    Args:
        source_mesh (UsdGeom.Mesh): The source mesh to transfer UVs from.
        target_mesh (UsdGeom.Mesh): The target mesh to transfer UVs to.

    Raises:
        ValueError: If source or target mesh is not a valid mesh prim.
        RuntimeError: If source mesh has no UV coordinates.
    """
    if not source_mesh.GetPrim().IsValid() or not source_mesh.GetPrim().IsA(UsdGeom.Mesh):
        raise ValueError("Source prim is not a valid Mesh.")
    if not target_mesh.GetPrim().IsValid() or not target_mesh.GetPrim().IsA(UsdGeom.Mesh):
        raise ValueError("Target prim is not a valid Mesh.")
    source_primvars = UsdGeom.PrimvarsAPI(source_mesh)
    uv_primvar = source_primvars.GetPrimvar("st")
    if not uv_primvar or not uv_primvar.HasValue():
        raise RuntimeError("Source mesh has no UV coordinates.")
    interpolation = uv_primvar.GetInterpolation()
    uv_values = uv_primvar.Get(Usd.TimeCode.Default())
    target_primvars = UsdGeom.PrimvarsAPI(target_mesh)
    target_uv_primvar = target_primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, interpolation)
    target_uv_primvar.Set(uv_values)


def create_mesh_from_points(
    stage: Usd.Stage,
    prim_path: str,
    points: Vt.Vec3fArray,
    face_vertex_counts: Vt.IntArray,
    face_vertex_indices: Vt.IntArray,
) -> UsdGeom.Mesh:
    """Create a new mesh from points, face vertex counts, and face vertex indices.

    Args:
        stage (Usd.Stage): The USD stage to create the mesh on.
        prim_path (str): The path where the mesh prim should be created.
        points (Vt.Vec3fArray): The array of points defining the mesh vertices.
        face_vertex_counts (Vt.IntArray): The number of vertices for each face.
        face_vertex_indices (Vt.IntArray): The vertex indices for each face, flattened.

    Returns:
        UsdGeom.Mesh: The newly created mesh prim.

    Raises:
        ValueError: If the topology is invalid or the prim path is invalid.
    """
    if not UsdGeom.Mesh.ValidateTopology(face_vertex_indices, face_vertex_counts, len(points)):
        raise ValueError("Invalid mesh topology")
    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(prim_path))
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
    return mesh


def calculate_mesh_surface_area(mesh: UsdGeom.Mesh) -> float:
    """Calculate the total surface area of a mesh.

    Args:
        mesh (UsdGeom.Mesh): The input mesh.

    Returns:
        float: The total surface area of the mesh.
    """
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not face_vertex_counts or not face_vertex_indices:
        return 0.0
    points = mesh.GetPointsAttr().Get()
    total_area = 0.0
    index_offset = 0
    for face_vertex_count in face_vertex_counts:
        face_indices = face_vertex_indices[index_offset : index_offset + face_vertex_count]
        face_area = 0.0
        for i in range(face_vertex_count - 2):
            v0 = points[face_indices[0]]
            v1 = points[face_indices[i + 1]]
            v2 = points[face_indices[i + 2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = Gf.Cross(edge1, edge2)
            triangle_area = 0.5 * Gf.GetLength(cross)
            face_area += triangle_area
        total_area += face_area
        index_offset += face_vertex_count
    return total_area


def assign_material_to_mesh(mesh_prim: UsdGeom.Mesh, material_path: str):
    """Assign a material to a mesh prim.

    Args:
        mesh_prim (UsdGeom.Mesh): The mesh prim to assign the material to.
        material_path (str): The path to the material prim.

    Raises:
        ValueError: If the mesh_prim is not a valid UsdGeom.Mesh prim.
        ValueError: If the material path does not exist.
    """
    if not mesh_prim:
        raise ValueError(f"Invalid mesh prim: {mesh_prim}")
    stage = mesh_prim.GetPrim().GetStage()
    if not stage.GetPrimAtPath(material_path):
        raise ValueError(f"Material path does not exist: {material_path}")
    binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
    binding_api.Bind(UsdShade.Material(stage.GetPrimAtPath(material_path)))


def create_nurbs_curves(
    stage: Usd.Stage,
    prim_path: str,
    points: Vt.Vec3fArray,
    order: int,
    knots: Vt.DoubleArray,
    widths: Vt.FloatArray = None,
    ranges: Vt.Vec2dArray = None,
) -> UsdGeom.NurbsCurves:
    """Create a NurbsCurve prim with the given attributes.

    Args:
        stage (Usd.Stage): The stage to create the prim on.
        prim_path (str): The path to create the prim at.
        points (Vt.Vec3fArray): The control points for the curve.
        order (int): The order of the curve.
        knots (Vt.DoubleArray): The knot vector for the curve.
        widths (Vt.FloatArray, optional): The widths at each control point. Defaults to None.
        ranges (Vt.Vec2dArray, optional): The ranges for each curve segment. Defaults to None.

    Returns:
        UsdGeom.NurbsCurves: The created NurbsCurves prim.
    """
    nurbs_curves = UsdGeom.NurbsCurves.Define(stage, prim_path)
    nurbs_curves.GetPointsAttr().Set(points)
    nurbs_curves.GetOrderAttr().Set([order])
    nurbs_curves.GetKnotsAttr().Set(knots)
    if widths is not None:
        nurbs_curves.GetWidthsAttr().Set(widths)
    if ranges is not None:
        nurbs_curves.GetRangesAttr().Set(ranges)
    return nurbs_curves


def get_nurbs_curve_points(nurbs_curve: UsdGeom.NurbsCurves) -> List[Tuple[float, float, float]]:
    """
    Get the control points of a NURBS curve.

    Args:
        nurbs_curve (UsdGeom.NurbsCurves): The NURBS curve to retrieve points from.

    Returns:
        List[Tuple[float, float, float]]: A list of control points as tuples (x, y, z).
    """
    if not nurbs_curve or not isinstance(nurbs_curve, UsdGeom.NurbsCurves):
        raise ValueError("Invalid NURBS curve provided.")
    points_attr = nurbs_curve.GetPointsAttr()
    if not points_attr.HasValue():
        raise ValueError("NURBS curve does not have any control points.")
    points = points_attr.Get()
    return [(p[0], p[1], p[2]) for p in points]


def get_nurbs_curve_knots(prim: UsdGeom.NurbsCurves) -> List[float]:
    """Get the knot values of a NurbsCurve prim.

    Args:
        prim (UsdGeom.NurbsCurves): The NurbsCurve prim to get knots from.

    Returns:
        List[float]: The knot values of the NurbsCurve.

    Raises:
        ValueError: If the input prim is not a valid NurbsCurves prim.
    """
    if not prim or not isinstance(prim, UsdGeom.NurbsCurves):
        raise ValueError("Input prim must be a valid UsdGeom.NurbsCurves.")
    knots_attr = prim.GetKnotsAttr()
    if not knots_attr or not knots_attr.HasAuthoredValue():
        return []
    knots = knots_attr.Get()
    return list(knots)


def set_nurbs_curve_points(nurbs_curve: UsdGeom.NurbsCurves, points: Sequence[Sequence[float]]) -> None:
    """Set the points for a NURBS curve.

    Args:
        nurbs_curve (UsdGeom.NurbsCurves): The NURBS curve to set points for.
        points (Sequence[Sequence[float]]): A sequence of points, each point is a sequence of 3 floats (x, y, z).

    Raises:
        ValueError: If the input points are not a sequence of sequences of 3 floats.
    """
    if not all((len(point) == 3 and all((isinstance(coord, float) for coord in point)) for point in points)):
        raise ValueError("Input points must be a sequence of sequences of 3 floats.")
    point_array = Vt.Vec3fArray([Gf.Vec3f(*point) for point in points])
    nurbs_curve.GetPointsAttr().Set(point_array)


def set_nurbs_curve_knots(nurbs_curve: UsdGeom.NurbsCurves, knots: Vt.DoubleArray) -> None:
    """Set the knot vector for a NURBS curve.

    Args:
        nurbs_curve (UsdGeom.NurbsCurves): The NURBS curve to set knots for.
        knots (Vt.DoubleArray): The knot vector to set.

    Raises:
        ValueError: If the input curve is not a valid NURBS curve prim.
        ValueError: If the provided knot vector is not valid or incompatible with the curve.
    """
    if not nurbs_curve or not isinstance(nurbs_curve, UsdGeom.NurbsCurves):
        raise ValueError("Invalid NURBS curve prim.")
    num_curves = len(nurbs_curve.GetCurveVertexCountsAttr().Get())
    order = nurbs_curve.GetOrderAttr().Get()
    if not isinstance(knots, Vt.DoubleArray):
        raise ValueError("Knots must be a Vt.DoubleArray.")
    if len(knots) != sum(order) + sum(nurbs_curve.GetCurveVertexCountsAttr().Get()):
        raise ValueError("Invalid number of knot values for the curve.")
    if any((knots[i] > knots[i + 1] for i in range(len(knots) - 1))):
        raise ValueError("Knot values must be non-decreasing.")
    knots_attr = nurbs_curve.GetKnotsAttr()
    knots_attr.Set(knots)


def get_nurbs_curve_weights(nurbs_curves: UsdGeom.NurbsCurves) -> List[float]:
    """Get the weights of the control points of a NURBS curve.

    Args:
        nurbs_curves (UsdGeom.NurbsCurves): The NURBS curve prim.

    Returns:
        List[float]: The weights of the control points. If the curve is not rational,
            returns an empty list.
    """
    if not nurbs_curves or not isinstance(nurbs_curves, UsdGeom.NurbsCurves):
        raise ValueError("Invalid NurbsCurves prim.")
    point_weights_attr = nurbs_curves.GetPointWeightsAttr()
    if not point_weights_attr or not point_weights_attr.HasAuthoredValue():
        return []
    weights = point_weights_attr.Get()
    return list(weights)


def set_nurbs_curve_weights(nurbs_curves: UsdGeom.NurbsCurves, weights: Sequence[float]) -> None:
    """Set the weights for a NurbsCurves prim.

    Args:
        nurbs_curves (UsdGeom.NurbsCurves): The NurbsCurves prim to set weights for.
        weights (Sequence[float]): The weights to set on the prim.

    Raises:
        ValueError: If the input prim is not a valid NurbsCurves prim or if the number
                    of weights does not match the number of points in the curve.
    """
    if not nurbs_curves or not isinstance(nurbs_curves, UsdGeom.NurbsCurves):
        raise ValueError("Input prim must be a valid UsdGeom.NurbsCurves.")
    points_attr = nurbs_curves.GetPointsAttr()
    if not points_attr.HasAuthoredValue():
        raise ValueError("NurbsCurves prim must have authored points.")
    num_points = len(points_attr.Get())
    if len(weights) != num_points:
        raise ValueError(f"Number of weights ({len(weights)}) must match number of points ({num_points}).")
    weights_attr = nurbs_curves.CreatePointWeightsAttr()
    weights_attr.Set(weights)


def align_nurbs_curves(curves: UsdGeom.NurbsCurves, target_points: Sequence[Gf.Vec3f]) -> None:
    """
    Aligns the control points of a NurbsCurves prim to a target set of points.

    The number of points in target_points must match the number of points in the NurbsCurves prim.
    If the numbers don't match, the function will raise a ValueError.

    Args:
        curves (UsdGeom.NurbsCurves): The NurbsCurves prim to align.
        target_points (Sequence[Gf.Vec3f]): The target points to align the curve to.

    Raises:
        ValueError: If the number of target points does not match the number of points in the curve.
    """
    points_attr = curves.GetPointsAttr()
    if not points_attr.IsValid():
        raise ValueError("The NurbsCurves prim does not have a valid points attribute.")
    curve_points = points_attr.Get()
    if len(curve_points) != len(target_points):
        raise ValueError(
            f"Number of target points ({len(target_points)}) does not match the number of curve points ({len(curve_points)})."
        )
    points_attr.Set(Vt.Vec3fArray(target_points))


def merge_nurbs_curves(curves: Sequence[UsdGeom.NurbsCurves]) -> UsdGeom.NurbsCurves:
    """Merge multiple NurbsCurves prims into a single NurbsCurves prim.

    The merged NurbsCurves prim will have the topology and attributes combined from the input curves.

    Args:
        curves (Sequence[UsdGeom.NurbsCurves]): A sequence of NurbsCurves prims to merge.

    Returns:
        UsdGeom.NurbsCurves: The merged NurbsCurves prim.

    Raises:
        ValueError: If the input curves have inconsistent attributes that prevent merging.
    """
    if not curves:
        raise ValueError("At least one NurbsCurves prim must be provided.")
    merged_prim = curves[0].GetPrim()
    points = []
    orders = []
    knots = []
    ranges = []
    widths = []
    point_weights = []
    for curve in curves:
        points.extend(curve.GetPointsAttr().Get())
        orders.extend(curve.GetOrderAttr().Get())
        knots.extend(curve.GetKnotsAttr().Get())
        ranges.extend(curve.GetRangesAttr().Get())
        if curve.GetWidthsAttr().HasAuthoredValue():
            widths.extend(curve.GetWidthsAttr().Get())
        if curve.GetPointWeightsAttr().HasAuthoredValue():
            point_weights.extend(curve.GetPointWeightsAttr().Get())
    merged_curve = UsdGeom.NurbsCurves(merged_prim)
    merged_curve.GetPointsAttr().Set(points)
    merged_curve.GetOrderAttr().Set(orders)
    merged_curve.GetKnotsAttr().Set(knots)
    merged_curve.GetRangesAttr().Set(ranges)
    if widths:
        merged_curve.GetWidthsAttr().Set(widths)
    if point_weights:
        merged_curve.GetPointWeightsAttr().Set(point_weights)
    return merged_curve


def animate_nurbs_curve(
    curve: UsdGeom.NurbsCurves,
    points: List[Tuple[float, float, float]],
    time_range: Tuple[float, float],
    translate: Tuple[float, float, float] = (0, 0, 0),
) -> None:
    """Animates a NURBS curve by setting control point positions over a time range.

    Args:
        curve (UsdGeom.NurbsCurves): The NURBS curve to animate.
        points (List[Tuple[float, float, float]]): The control point positions for each frame.
        time_range (Tuple[float, float]): The start and end times for the animation.
        translate (Tuple[float, float, float], optional): Additional translation to apply to the curve. Defaults to (0, 0, 0).

    Raises:
        ValueError: If the number of points does not match the number of frames in the time range.
    """
    num_frames = int(time_range[1] - time_range[0] + 1)
    if len(points) != num_frames:
        raise ValueError(f"Number of points ({len(points)}) does not match the number of frames ({num_frames}).")
    points_attr = curve.GetPointsAttr()
    for i, point in enumerate(points):
        frame = time_range[0] + i
        points_attr.Set(Vt.Vec3fArray(point), Usd.TimeCode(frame))
    if translate != (0, 0, 0):
        translate_op = add_translate_op(curve)
        for frame in range(time_range[0], time_range[1] + 1):
            translate_op.Set(Gf.Vec3f(translate), Usd.TimeCode(frame))


def convert_nurbs_to_basis(nurbs_curves: UsdGeom.NurbsCurves) -> UsdGeom.BasisCurves:
    """
    Convert a NURBS curve to a Basis curve.

    Args:
        nurbs_curves (UsdGeom.NurbsCurves): The input NURBS curve.

    Returns:
        UsdGeom.BasisCurves: The converted Basis curve.
    """
    points = nurbs_curves.GetPointsAttr().Get()
    counts = nurbs_curves.GetCurveVertexCountsAttr().Get()
    widths = nurbs_curves.GetWidthsAttr().Get() if nurbs_curves.GetWidthsAttr().HasValue() else None
    order = nurbs_curves.GetOrderAttr().Get()
    if not points or not counts or (not order):
        raise ValueError("Invalid NURBS curve data")
    basis_curves = UsdGeom.BasisCurves.Define(nurbs_curves.GetPrim().GetStage(), nurbs_curves.GetPath())
    basis_curves.GetPointsAttr().Set(points)
    basis_curves.GetCurveVertexCountsAttr().Set(counts)
    if widths:
        basis_curves.GetWidthsAttr().Set(widths)
    basis = "bezier"
    wrap = "nonperiodic"
    basis_curves.GetBasisAttr().Set(basis)
    basis_curves.GetWrapAttr().Set(wrap)
    basis_curves.GetTypeAttr().Set("cubic")
    adjusted_counts = [count - (o - 1) for (count, o) in zip(counts, order)]
    basis_curves.GetCurveVertexCountsAttr().Set(adjusted_counts)
    return basis_curves


def optimize_nurbs_curve(nurbs_curves: UsdGeom.NurbsCurves) -> bool:
    """Optimize a NurbsCurves prim by removing redundant knots and control points.

    Args:
        nurbs_curves (UsdGeom.NurbsCurves): The NurbsCurves prim to optimize.

    Returns:
        bool: True if the curve was optimized, False otherwise.
    """
    if not nurbs_curves or not isinstance(nurbs_curves, UsdGeom.NurbsCurves):
        raise ValueError("Invalid NurbsCurves prim.")
    knots = nurbs_curves.GetKnotsAttr().Get()
    points = nurbs_curves.GetPointsAttr().Get()
    if not knots or not points or len(knots) < 4 or (len(points) < 4):
        return False
    order = nurbs_curves.GetOrderAttr().Get()
    if not order or order[0] < 2 or order[0] > len(knots) - len(points) + 1:
        return False
    unique_knots = list(set(knots))
    unique_knots.sort()
    optimized_points = list(points[: order[0]])
    for i in range(order[0], len(points)):
        if points[i] != points[i - 1]:
            optimized_points.append(points[i])
    optimized_points.extend(points[-order[0] :])
    if len(unique_knots) == len(knots) and len(optimized_points) == len(points):
        return False
    nurbs_curves.GetKnotsAttr().Set(unique_knots)
    nurbs_curves.GetPointsAttr().Set(optimized_points)
    return True


def duplicate_nurbs_patch(stage: Usd.Stage, src_prim_path: str, dst_prim_path: str) -> UsdGeom.NurbsPatch:
    """Duplicate a NurbsPatch prim.

    Args:
        stage (Usd.Stage): The USD stage.
        src_prim_path (str): The path to the source NurbsPatch prim.
        dst_prim_path (str): The path to the destination NurbsPatch prim.

    Returns:
        UsdGeom.NurbsPatch: The duplicated NurbsPatch prim.

    Raises:
        ValueError: If the source prim is not a valid NurbsPatch.
    """
    src_prim = stage.GetPrimAtPath(src_prim_path)
    if not src_prim.IsValid() or not src_prim.IsA(UsdGeom.NurbsPatch):
        raise ValueError(f"Prim at path {src_prim_path} is not a valid NurbsPatch.")
    src_nurbs_patch = UsdGeom.NurbsPatch(src_prim)
    dst_nurbs_patch = UsdGeom.NurbsPatch.Define(stage, dst_prim_path)
    attrs_to_copy = [
        "uVertexCount",
        "vVertexCount",
        "uOrder",
        "vOrder",
        "uForm",
        "vForm",
        "uKnots",
        "vKnots",
        "uRange",
        "vRange",
        "points",
        "trimCurve:counts",
        "trimCurve:orders",
        "trimCurve:knots",
        "trimCurve:ranges",
        "trimCurve:points",
        "trimCurve:vertexCounts",
        "pointWeights",
    ]
    for attr_name in attrs_to_copy:
        src_attr = src_nurbs_patch.GetPrim().GetAttribute(attr_name)
        if src_attr.IsValid():
            attr_value = src_attr.Get()
            if attr_value is not None:
                dst_nurbs_patch.GetPrim().CreateAttribute(attr_name, src_attr.GetTypeName()).Set(attr_value)
    return dst_nurbs_patch


def remove_trim_curves_from_nurbs_patch(nurbs_patch: UsdGeom.NurbsPatch) -> None:
    """Remove trim curves from a NURBS patch.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The NURBS patch to remove trim curves from.

    Raises:
        ValueError: If the input prim is not a valid UsdGeomNurbsPatch.
    """
    if not nurbs_patch or not nurbs_patch.GetPrim().IsValid():
        raise ValueError("Invalid UsdGeomNurbsPatch")
    if nurbs_patch.GetTrimCurveCountsAttr().HasAuthoredValue():
        nurbs_patch.GetTrimCurveCountsAttr().Clear()
    if nurbs_patch.GetTrimCurveOrdersAttr().HasAuthoredValue():
        nurbs_patch.GetTrimCurveOrdersAttr().Clear()
    if nurbs_patch.GetTrimCurveVertexCountsAttr().HasAuthoredValue():
        nurbs_patch.GetTrimCurveVertexCountsAttr().Clear()
    if nurbs_patch.GetTrimCurveKnotsAttr().HasAuthoredValue():
        nurbs_patch.GetTrimCurveKnotsAttr().Clear()
    if nurbs_patch.GetTrimCurveRangesAttr().HasAuthoredValue():
        nurbs_patch.GetTrimCurveRangesAttr().Clear()
    if nurbs_patch.GetTrimCurvePointsAttr().HasAuthoredValue():
        nurbs_patch.GetTrimCurvePointsAttr().Clear()


def get_nurbs_patch_order(stage: Usd.Stage, prim_path: str) -> Tuple[int, int]:
    """Get the U and V order of a NurbsPatch prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the NurbsPatch prim.

    Returns:
        Tuple[int, int]: A tuple containing the U and V order of the NurbsPatch.

    Raises:
        ValueError: If the prim at the given path is not a valid NurbsPatch.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    nurbs_patch = UsdGeom.NurbsPatch(prim)
    if not nurbs_patch:
        raise ValueError(f"Prim at path {prim_path} is not a valid NurbsPatch.")
    u_order_attr = nurbs_patch.GetUOrderAttr()
    v_order_attr = nurbs_patch.GetVOrderAttr()
    if not u_order_attr.HasValue() or not v_order_attr.HasValue():
        raise ValueError(f"NurbsPatch at path {prim_path} is missing U or V order attributes.")
    u_order = u_order_attr.Get()
    v_order = v_order_attr.Get()
    return (u_order, v_order)


def add_trim_curve_to_nurbs_patch(
    nurbs_patch: UsdGeom.NurbsPatch,
    trim_curve_points: List[Gf.Vec3d],
    trim_curve_knots: List[float],
    trim_curve_ranges: List[Gf.Vec2d],
    trim_curve_orders: List[int],
    trim_curve_vertex_counts: List[int],
) -> None:
    """Add a trim curve to a NURBS patch.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The NURBS patch to add the trim curve to.
        trim_curve_points (List[Gf.Vec3d]): The points of the trim curve.
        trim_curve_knots (List[float]): The knots of the trim curve.
        trim_curve_ranges (List[Gf.Vec2d]): The ranges of the trim curve.
        trim_curve_orders (List[int]): The orders of the trim curve.
        trim_curve_vertex_counts (List[int]): The vertex counts of the trim curve.
    """
    existing_counts = nurbs_patch.GetTrimCurveCountsAttr().Get()
    existing_orders = nurbs_patch.GetTrimCurveOrdersAttr().Get()
    existing_knots = nurbs_patch.GetTrimCurveKnotsAttr().Get()
    existing_ranges = nurbs_patch.GetTrimCurveRangesAttr().Get()
    existing_points = nurbs_patch.GetTrimCurvePointsAttr().Get()
    existing_vertex_counts = nurbs_patch.GetTrimCurveVertexCountsAttr().Get()
    if existing_counts is None:
        existing_counts = []
    if existing_orders is None:
        existing_orders = []
    if existing_knots is None:
        existing_knots = []
    if existing_ranges is None:
        existing_ranges = []
    if existing_points is None:
        existing_points = []
    if existing_vertex_counts is None:
        existing_vertex_counts = []
    existing_counts.append(len(trim_curve_orders))
    existing_orders.extend(trim_curve_orders)
    existing_knots.extend(trim_curve_knots)
    existing_ranges.extend(trim_curve_ranges)
    existing_points.extend(trim_curve_points)
    existing_vertex_counts.extend(trim_curve_vertex_counts)
    nurbs_patch.GetTrimCurveCountsAttr().Set(existing_counts)
    nurbs_patch.GetTrimCurveOrdersAttr().Set(existing_orders)
    nurbs_patch.GetTrimCurveKnotsAttr().Set(existing_knots)
    nurbs_patch.GetTrimCurveRangesAttr().Set(existing_ranges)
    nurbs_patch.GetTrimCurvePointsAttr().Set(existing_points)
    nurbs_patch.GetTrimCurveVertexCountsAttr().Set(existing_vertex_counts)


def get_nurbs_patch_u_knots(nurbs_patch: UsdGeom.NurbsPatch) -> Vt.DoubleArray:
    """
    Get the U knot vector of a NurbsPatch.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The NurbsPatch to get the U knot vector from.

    Returns:
        Vt.DoubleArray: The U knot vector of the NurbsPatch.

    Raises:
        ValueError: If the input prim is not a valid NurbsPatch.
    """
    if not nurbs_patch or not isinstance(nurbs_patch, UsdGeom.NurbsPatch):
        raise ValueError("Input prim is not a valid NurbsPatch")
    u_knots_attr = nurbs_patch.GetUKnotsAttr()
    if not u_knots_attr or not u_knots_attr.HasValue():
        return Vt.DoubleArray()
    u_knots = u_knots_attr.Get()
    return u_knots


def get_nurbs_patch_v_knots(nurbs_patch: UsdGeom.NurbsPatch) -> Vt.DoubleArray:
    """
    Get the V knot vector of a NurbsPatch.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The NurbsPatch prim.

    Returns:
        Vt.DoubleArray: The V knot vector.
    """
    if not nurbs_patch or not nurbs_patch.GetPrim().IsValid():
        raise ValueError("Invalid NurbsPatch prim.")
    v_knots_attr = nurbs_patch.GetVKnotsAttr()
    if not v_knots_attr.HasAuthoredValue():
        return Vt.DoubleArray()
    v_knots = v_knots_attr.Get()
    if not v_knots:
        return Vt.DoubleArray()
    return v_knots


def get_nurbs_patch_control_points(nurbs_patch: UsdGeom.NurbsPatch) -> Tuple[Vt.Vec3fArray, bool]:
    """Get the control points of a UsdGeom.NurbsPatch.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The NurbsPatch to get the control points from.

    Returns:
        A tuple with two elements:
        - Vt.Vec3fArray: The control points of the NurbsPatch.
        - bool: True if the control points are in homogeneous (4D) space, False if they are in Euclidean (3D) space.

    Raises:
        ValueError: If the input prim is not a valid UsdGeom.NurbsPatch.
    """
    if not nurbs_patch or not isinstance(nurbs_patch, UsdGeom.NurbsPatch):
        raise ValueError("Input prim must be a valid UsdGeom.NurbsPatch")
    points_attr = nurbs_patch.GetPointsAttr()
    if not points_attr or not points_attr.HasValue():
        return (Vt.Vec3fArray(), False)
    points = points_attr.Get()
    is_homogeneous = len(points[0]) == 4
    if is_homogeneous:
        euclidean_points = Vt.Vec3fArray(len(points))
        for i, pt in enumerate(points):
            euclidean_points[i] = Gf.Vec3f(pt[0] / pt[3], pt[1] / pt[3], pt[2] / pt[3])
        return (euclidean_points, True)
    else:
        return (Vt.Vec3fArray(points), False)


def set_nurbs_patch_u_knots(nurbs_patch: UsdGeom.NurbsPatch, u_knots: Sequence[float]) -> None:
    """Set the u knots for a NurbsPatch.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The NurbsPatch to set the u knots for.
        u_knots (Sequence[float]): The u knots to set.

    Raises:
        ValueError: If the number of knots is not equal to uVertexCount + uOrder.
        ValueError: If the knots are not monotonically increasing.
    """
    u_vertex_count = nurbs_patch.GetUVertexCountAttr().Get()
    u_order = nurbs_patch.GetUOrderAttr().Get()
    if len(u_knots) != u_vertex_count + u_order:
        raise ValueError(
            f"Number of u knots ({len(u_knots)}) must be equal to uVertexCount ({u_vertex_count}) + uOrder ({u_order})."
        )
    if any((u_knots[i] > u_knots[i + 1] for i in range(len(u_knots) - 1))):
        raise ValueError("U knots must be monotonically increasing.")
    nurbs_patch.CreateUKnotsAttr().Set(u_knots)
    u_min = u_knots[u_order - 1]
    u_max = u_knots[-u_order]
    nurbs_patch.CreateURangeAttr().Set(Gf.Vec2d(u_min, u_max))


def set_nurbs_patch_control_points(nurbs_patch: UsdGeom.NurbsPatch, points: List[Gf.Vec3f]) -> None:
    """Set the control points for a NurbsPatch.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The NurbsPatch prim to set the control points for.
        points (List[Gf.Vec3f]): The list of control points to set.

    Raises:
        ValueError: If the number of control points does not match the expected number based on uVertexCount and vVertexCount.
    """
    u_vertex_count = nurbs_patch.GetUVertexCountAttr().Get()
    v_vertex_count = nurbs_patch.GetVVertexCountAttr().Get()
    expected_count = u_vertex_count * v_vertex_count
    if len(points) != expected_count:
        raise ValueError(f"Expected {expected_count} control points, but got {len(points)}")
    nurbs_patch.GetPointsAttr().Set(points)


def set_nurbs_patch_weights(nurbs_patch: UsdGeom.NurbsPatch, weights: Sequence[float]) -> None:
    """Set the weights for the control points of a NurbsPatch.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The NurbsPatch to set weights for.
        weights (Sequence[float]): The weights to set, must match the number of points.

    Raises:
        ValueError: If the number of weights does not match the number of points.
    """
    points_attr = nurbs_patch.GetPointsAttr()
    if not points_attr.IsValid():
        raise ValueError("NurbsPatch has no points attribute")
    num_points = len(points_attr.Get())
    if len(weights) != num_points:
        raise ValueError(f"Number of weights ({len(weights)}) does not match number of points ({num_points})")
    weights_attr = nurbs_patch.GetPointWeightsAttr()
    if not weights_attr.IsValid():
        weights_attr = nurbs_patch.CreatePointWeightsAttr()
    weights_attr.Set(weights)


def set_nurbs_patch_v_range(nurbs_patch: UsdGeom.NurbsPatch, v_range: Tuple[float, float]) -> None:
    """Set the V range for a NurbsPatch.

    The V range represents the minimum and maximum parametric values over which the surface is defined in the V direction.
    The range must be within the knot range defined by vKnots attribute.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The NurbsPatch prim to set the V range for.
        v_range (Tuple[float, float]): The V range as a tuple of (min, max) values.

    Raises:
        ValueError: If the provided V range is invalid or outside the valid knot range.
    """
    v_knots = nurbs_patch.GetVKnotsAttr().Get()
    v_order = nurbs_patch.GetVOrderAttr().Get()
    if v_order is None or v_knots is None:
        raise ValueError("NurbsPatch must have vOrder and vKnots attributes authored.")
    min_valid_knot = v_knots[v_order - 1]
    max_valid_knot = v_knots[-1]
    if v_range[0] >= v_range[1]:
        raise ValueError(f"Invalid V range: {v_range}. Min must be less than max.")
    if v_range[0] < min_valid_knot or v_range[1] > max_valid_knot:
        raise ValueError(f"V range {v_range} is outside the valid knot range [{min_valid_knot}, {max_valid_knot}].")
    v_range_attr = (
        nurbs_patch.CreateVRangeAttr() if nurbs_patch.GetVRangeAttr() is None else nurbs_patch.GetVRangeAttr()
    )
    v_range_attr.Set(Gf.Vec2d(v_range[0], v_range[1]))


def get_nurbs_patch_u_range(nurbs_patch: UsdGeom.NurbsPatch) -> Tuple[float, float]:
    """Get the parametric range in the U direction for a NurbsPatch.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The NurbsPatch prim.

    Returns:
        Tuple[float, float]: The minimum and maximum U parametric range as a tuple.
            Returns (0.0, 0.0) if the uRange attribute is not authored or has no value.
    """
    u_range_attr = nurbs_patch.GetURangeAttr()
    if u_range_attr.HasAuthoredValue():
        u_range = u_range_attr.Get()
        if u_range is not None:
            if u_range[0] < u_range[1]:
                return (u_range[0], u_range[1])
            else:
                raise ValueError("Invalid uRange: minimum must be less than maximum.")
    return (0.0, 0.0)


def get_nurbs_patch_v_range(nurbs_patch: UsdGeom.NurbsPatch) -> Tuple[float, float]:
    """Get the V range of a NurbsPatch.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The NurbsPatch prim.

    Returns:
        Tuple[float, float]: The minimum and maximum V values of the NurbsPatch.

    Raises:
        ValueError: If the provided prim is not a valid NurbsPatch.
    """
    if not nurbs_patch or not isinstance(nurbs_patch, UsdGeom.NurbsPatch):
        raise ValueError("Invalid NurbsPatch prim.")
    v_range_attr = nurbs_patch.GetVRangeAttr()
    if not v_range_attr or not v_range_attr.HasValue():
        v_knots = nurbs_patch.GetVKnotsAttr().Get()
        v_order = nurbs_patch.GetVOrderAttr().Get()
        if v_knots and v_order:
            min_v = v_knots[v_order - 1]
            max_v = v_knots[-1]
            return (min_v, max_v)
        else:
            return (0.0, 1.0)
    v_range = v_range_attr.Get()
    return (v_range[0], v_range[1])


def set_nurbs_patch_order(nurbs_patch: UsdGeom.NurbsPatch, u_order: int, v_order: int) -> None:
    """Set the order of a NurbsPatch in the U and V directions."""
    if u_order < 1:
        raise ValueError("U order must be greater than or equal to 1")
    if v_order < 1:
        raise ValueError("V order must be greater than or equal to 1")
    u_order_attr = nurbs_patch.GetUOrderAttr()
    v_order_attr = nurbs_patch.GetVOrderAttr()
    u_order_attr.Set(u_order)
    v_order_attr.Set(v_order)
    u_knots_attr = nurbs_patch.GetUKnotsAttr()
    v_knots_attr = nurbs_patch.GetVKnotsAttr()
    u_vertex_count = nurbs_patch.GetUVertexCountAttr().Get()
    v_vertex_count = nurbs_patch.GetVVertexCountAttr().Get()
    u_knots_attr.Clear()
    v_knots_attr.Clear()
    u_knots_attr.Set([0.0] * (u_vertex_count + u_order))
    v_knots_attr.Set([0.0] * (v_vertex_count + v_order))


def transform_nurbs_patch(nurbs_patch: UsdGeom.NurbsPatch, transform_matrix: Gf.Matrix4d) -> None:
    """Transform the points of a NurbsPatch by a given transformation matrix.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The NurbsPatch to transform.
        transform_matrix (Gf.Matrix4d): The transformation matrix to apply.
    """
    points_attr = nurbs_patch.GetPointsAttr()
    if not points_attr.IsValid():
        raise ValueError("The NurbsPatch does not have a valid points attribute.")
    points = points_attr.Get()
    if not points:
        raise ValueError("The NurbsPatch does not have any points.")
    transformed_points = [transform_matrix.Transform(Gf.Vec3d(p)) for p in points]
    points_attr.Set(transformed_points)
    UsdGeom.Boundable(nurbs_patch).ComputeExtent(Usd.TimeCode.Default())


def create_nurbs_patch_from_points(
    stage: Usd.Stage,
    prim_path: str,
    points: List[Gf.Vec3f],
    u_range: Gf.Vec2d,
    v_range: Gf.Vec2d,
    u_order: int = 3,
    v_order: int = 3,
    u_knots: Optional[Vt.DoubleArray] = None,
    v_knots: Optional[Vt.DoubleArray] = None,
) -> UsdGeom.NurbsPatch:
    """Create a NurbsPatch from a list of points.

    Args:
        stage (Usd.Stage): The stage to create the NurbsPatch on.
        prim_path (str): The path to create the NurbsPatch at.
        points (List[Gf.Vec3f]): The points to create the NurbsPatch from.
        u_range (Gf.Vec2d): The range of the u parameter.
        v_range (Gf.Vec2d): The range of the v parameter.
        u_order (int, optional): The order of the u parameter. Defaults to 3.
        v_order (int, optional): The order of the v parameter. Defaults to 3.
        u_knots (Optional[Vt.DoubleArray], optional): The knots of the u parameter. Defaults to None.
        v_knots (Optional[Vt.DoubleArray], optional): The knots of the v parameter. Defaults to None.

    Returns:
        UsdGeom.NurbsPatch: The created NurbsPatch.
    """
    if len(points) == 0:
        raise ValueError("At least one point must be provided")
    if u_range[0] >= u_range[1]:
        raise ValueError("u_range[0] must be less than u_range[1]")
    if v_range[0] >= v_range[1]:
        raise ValueError("v_range[0] must be less than v_range[1]")
    if u_order < 1:
        raise ValueError("u_order must be greater than 0")
    if v_order < 1:
        raise ValueError("v_order must be greater than 0")
    nurbs_patch = UsdGeom.NurbsPatch.Define(stage, prim_path)
    nurbs_patch.GetPointsAttr().Set(points)
    nurbs_patch.GetURangeAttr().Set(u_range)
    nurbs_patch.GetVRangeAttr().Set(v_range)
    nurbs_patch.GetUOrderAttr().Set(u_order)
    nurbs_patch.GetVOrderAttr().Set(v_order)
    u_vertex_count = int(len(points) ** 0.5)
    v_vertex_count = u_vertex_count
    nurbs_patch.GetUVertexCountAttr().Set(u_vertex_count)
    nurbs_patch.GetVVertexCountAttr().Set(v_vertex_count)
    if u_knots:
        nurbs_patch.GetUKnotsAttr().Set(u_knots)
    if v_knots:
        nurbs_patch.GetVKnotsAttr().Set(v_knots)
    return nurbs_patch


def reverse_nurbs_patch_direction(nurbs_patch: UsdGeom.NurbsPatch, direction: str) -> None:
    """Reverse the control points, knots, and ranges of a NURBS patch in the specified direction.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The NURBS patch to modify.
        direction (str): The direction to reverse, either "u" or "v".

    Raises:
        ValueError: If the specified direction is not "u" or "v".
    """
    if direction not in ["u", "v"]:
        raise ValueError(f'Invalid direction "{direction}". Must be either "u" or "v".')
    points = nurbs_patch.GetPointsAttr().Get()
    if points:
        if direction == "u":
            points = points[::-1]
        else:
            num_u = nurbs_patch.GetUVertexCountAttr().Get()
            points = [points[i::num_u] for i in range(num_u)][::-1]
            points = [p for pts in zip(*points) for p in pts]
        nurbs_patch.GetPointsAttr().Set(points)
    knots_attr = nurbs_patch.GetUKnotsAttr() if direction == "u" else nurbs_patch.GetVKnotsAttr()
    knots = knots_attr.Get()
    if knots:
        knots = knots[::-1]
        (knot_min, knot_max) = (knots[0], knots[-1])
        knots = [knot_max + knot_min - k for k in knots]
        knots_attr.Set(knots)
    range_attr = nurbs_patch.GetURangeAttr() if direction == "u" else nurbs_patch.GetVRangeAttr()
    (range_min, range_max) = range_attr.Get()
    range_attr.Set(Gf.Vec2d(knot_max + knot_min - range_max, knot_max + knot_min - range_min))


def create_nurbs_patch_with_attributes(
    stage: Usd.Stage,
    prim_path: str,
    points: Vt.Vec3fArray,
    u_range: Gf.Vec2d,
    v_range: Gf.Vec2d,
    u_order: int,
    v_order: int,
    u_knots: Vt.DoubleArray,
    v_knots: Vt.DoubleArray,
    u_form: str,
    v_form: str,
    point_weights: Vt.DoubleArray = None,
) -> UsdGeom.NurbsPatch:
    """Create a UsdGeom.NurbsPatch with provided attributes.

    Args:
        stage (Usd.Stage): The USD stage to create the NurbsPatch on.
        prim_path (str): The path where the NurbsPatch should be created.
        points (Vt.Vec3fArray): The control points for the NurbsPatch.
        u_range (Gf.Vec2d): The parametric range in the U direction.
        v_range (Gf.Vec2d): The parametric range in the V direction.
        u_order (int): The order of the NurbsPatch in the U direction.
        v_order (int): The order of the NurbsPatch in the V direction.
        u_knots (Vt.DoubleArray): The knot vector in the U direction.
        v_knots (Vt.DoubleArray): The knot vector in the V direction.
        u_form (str): The form of the NurbsPatch in the U direction. Must be "open", "closed", or "periodic".
        v_form (str): The form of the NurbsPatch in the V direction. Must be "open", "closed", or "periodic".
        point_weights (Vt.DoubleArray, optional): The weights for the control points. Defaults to None.

    Returns:
        UsdGeom.NurbsPatch: The created NurbsPatch prim.
    """
    if u_form not in ["open", "closed", "periodic"]:
        raise ValueError(f"Invalid u_form: {u_form}. Must be 'open', 'closed', or 'periodic'.")
    if v_form not in ["open", "closed", "periodic"]:
        raise ValueError(f"Invalid v_form: {v_form}. Must be 'open', 'closed', or 'periodic'.")
    u_vertex_count = len(u_knots) - u_order
    v_vertex_count = len(v_knots) - v_order
    expected_points = u_vertex_count * v_vertex_count
    if len(points) != expected_points:
        raise ValueError(
            f"Number of control points ({len(points)}) must match knot vectors and orders (expected {expected_points})."
        )
    if point_weights is not None and len(point_weights) != len(points):
        raise ValueError("If provided, point_weights must have the same length as points.")
    nurbs_patch = UsdGeom.NurbsPatch.Define(stage, prim_path)
    nurbs_patch.GetPointsAttr().Set(points)
    nurbs_patch.GetURangeAttr().Set(u_range)
    nurbs_patch.GetVRangeAttr().Set(v_range)
    nurbs_patch.GetUOrderAttr().Set(u_order)
    nurbs_patch.GetVOrderAttr().Set(v_order)
    nurbs_patch.GetUKnotsAttr().Set(u_knots)
    nurbs_patch.GetVKnotsAttr().Set(v_knots)
    nurbs_patch.GetUVertexCountAttr().Set(u_vertex_count)
    nurbs_patch.GetVVertexCountAttr().Set(v_vertex_count)
    nurbs_patch.GetUFormAttr().Set(u_form)
    nurbs_patch.GetVFormAttr().Set(v_form)
    if point_weights is not None:
        nurbs_patch.GetPointWeightsAttr().Set(point_weights)
    return nurbs_patch


def merge_nurbs_patches(
    patches: List[UsdGeom.NurbsPatch], merge_points_tolerance: float = 1e-05
) -> Optional[UsdGeom.NurbsPatch]:
    """Merge a list of NURBS patches into a single patch if possible.

    The patches must have the same u and v orders, knots, and ranges. The patches
    must also form a contiguous grid of points - points along shared edges must match
    within the specified tolerance.

    Args:
        patches (List[UsdGeom.NurbsPatch]): The input patches to merge.
        merge_points_tolerance (float): The tolerance for comparing points along shared patch edges.

    Returns:
        Optional[UsdGeom.NurbsPatch]: The merged NURBS patch, or None if the patches could not be merged.
    """
    if not patches:
        return None
    u_order = patches[0].GetUOrderAttr().Get()
    v_order = patches[0].GetVOrderAttr().Get()
    u_knots = patches[0].GetUKnotsAttr().Get()
    v_knots = patches[0].GetVKnotsAttr().Get()
    u_range = patches[0].GetURangeAttr().Get()
    v_range = patches[0].GetVRangeAttr().Get()
    for patch in patches[1:]:
        if (
            patch.GetUOrderAttr().Get() != u_order
            or patch.GetVOrderAttr().Get() != v_order
            or patch.GetUKnotsAttr().Get() != u_knots
            or (patch.GetVKnotsAttr().Get() != v_knots)
            or (patch.GetURangeAttr().Get() != u_range)
            or (patch.GetVRangeAttr().Get() != v_range)
        ):
            return None
    num_patches_u = 1
    num_patches_v = 1
    for patch in patches[1:]:
        patch_pos = patch.GetPrim().GetPath().name
        patch_pos = patch_pos.split("_")
        if len(patch_pos) != 2 or not patch_pos[0].isdigit() or (not patch_pos[1].isdigit()):
            return None
        u = int(patch_pos[0])
        v = int(patch_pos[1])
        num_patches_u = max(num_patches_u, u + 1)
        num_patches_v = max(num_patches_v, v + 1)
    patch_u_size = patches[0].GetUVertexCountAttr().Get()
    patch_v_size = patches[0].GetVVertexCountAttr().Get()
    u_size = (patch_u_size - 1) * num_patches_u + 1
    v_size = (patch_v_size - 1) * num_patches_v + 1
    merged_points = [Gf.Vec3f()] * (u_size * v_size)
    merged_weights: List[float] = []
    if patches[0].GetPointWeightsAttr().HasAuthoredValue():
        merged_weights = [0.0] * (u_size * v_size)
    for u in range(num_patches_u):
        for v in range(num_patches_v):
            patch_name = f"{u}_{v}"
            patch = next((p for p in patches if p.GetPrim().GetPath().name == patch_name), None)
            if not patch:
                return None
            (points, weights) = (patch.GetPointsAttr().Get(), patch.GetPointWeightsAttr().Get())
            for patch_v in range(patch_v_size):
                for patch_u in range(patch_u_size):
                    point_id = patch_v * patch_u_size + patch_u
                    u_pos = u * (patch_u_size - 1) + patch_u
                    v_pos = v * (patch_v_size - 1) + patch_v
                    merged_point_id = v_pos * u_size + u_pos
                    merged_points[merged_point_id] = Gf.Vec3f(points[point_id])
                    if merged_weights:
                        merged_weights[merged_point_id] = weights[point_id]
    merged_prim = (
        patches[0]
        .GetPrim()
        .GetStage()
        .DefinePrim(patches[0].GetPath().GetParentPath().AppendChild("merged"), "NurbsPatch")
    )
    merged_patch = UsdGeom.NurbsPatch(merged_prim)
    merged_patch.CreatePointsAttr(merged_points)
    if merged_weights:
        merged_patch.CreatePointWeightsAttr(merged_weights)
    merged_patch.CreateUVertexCountAttr(u_size)
    merged_patch.CreateVVertexCountAttr(v_size)
    merged_patch.GetUOrderAttr().Set(u_order)
    merged_patch.GetVOrderAttr().Set(v_order)
    merged_patch.GetUKnotsAttr().Set(u_knots)
    merged_patch.GetVKnotsAttr().Set(v_knots)
    merged_patch.GetURangeAttr().Set(u_range)
    merged_patch.GetVRangeAttr().Set(v_range)
    return merged_patch


def assign_material_to_nurbs_patch(stage: Usd.Stage, nurbs_patch_path: str, material_path: str) -> None:
    """Assign a material to a NurbsPatch prim.

    Args:
        stage (Usd.Stage): The USD stage.
        nurbs_patch_path (str): The path to the NurbsPatch prim.
        material_path (str): The path to the Material prim.

    Raises:
        ValueError: If the NurbsPatch or Material prim does not exist.
    """
    nurbs_patch_prim = stage.GetPrimAtPath(nurbs_patch_path)
    if not nurbs_patch_prim.IsValid():
        raise ValueError(f"NurbsPatch prim at path {nurbs_patch_path} does not exist.")
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise ValueError(f"Material prim at path {material_path} does not exist.")
    nurbs_patch = UsdGeom.NurbsPatch(nurbs_patch_prim)
    if not nurbs_patch:
        raise ValueError(f"Prim at path {nurbs_patch_path} is not a valid NurbsPatch.")
    material = UsdShade.Material(material_prim)
    if not material:
        raise ValueError(f"Prim at path {material_path} is not a valid Material.")
    UsdShade.MaterialBindingAPI(nurbs_patch_prim).Bind(material)


def mirror_nurbs_patch(nurbs_patch: UsdGeom.NurbsPatch, mirror_axis: str) -> UsdGeom.NurbsPatch:
    """
    Mirror a NurbsPatch across a specified axis.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The input NurbsPatch to be mirrored.
        mirror_axis (str): The axis across which to mirror the patch. Must be 'x', 'y', or 'z'.

    Returns:
        UsdGeom.NurbsPatch: The mirrored NurbsPatch.

    Raises:
        ValueError: If the specified mirror_axis is not 'x', 'y', or 'z'.
    """
    if mirror_axis not in ["x", "y", "z"]:
        raise ValueError(f"Invalid mirror_axis: {mirror_axis}. Must be 'x', 'y', or 'z'.")
    points_attr = nurbs_patch.GetPointsAttr()
    if not points_attr.IsValid():
        raise ValueError("The input NurbsPatch has no points attribute.")
    points = points_attr.Get()
    axis_index = {"x": 0, "y": 1, "z": 2}[mirror_axis]
    mirrored_points = []
    for point in points:
        mirrored_point = Gf.Vec3d(point)
        mirrored_point[axis_index] *= -1
        mirrored_points.append(mirrored_point)
    mirrored_patch = UsdGeom.NurbsPatch.Define(
        nurbs_patch.GetPrim().GetStage(), nurbs_patch.GetPrim().GetPath().AppendChild("mirrored")
    )
    mirrored_patch.CreatePointsAttr(mirrored_points)
    for attr_name in ["uForm", "vForm", "uOrder", "vOrder"]:
        attr = nurbs_patch.GetPrim().GetAttribute(attr_name)
        if attr.IsValid():
            mirrored_patch.GetPrim().CreateAttribute(attr_name, attr.GetTypeName()).Set(attr.Get())
    for attr_name in ["uKnots", "vKnots"]:
        attr = nurbs_patch.GetPrim().GetAttribute(attr_name)
        if attr.IsValid():
            mirrored_patch.GetPrim().CreateAttribute(attr_name, attr.GetTypeName()).Set(attr.Get())
    for attr_name in ["uRange", "vRange"]:
        attr = nurbs_patch.GetPrim().GetAttribute(attr_name)
        if attr.IsValid():
            mirrored_patch.GetPrim().CreateAttribute(attr_name, attr.GetTypeName()).Set(Gf.Vec2d(attr.Get()))
    return mirrored_patch


def set_nurbs_patch_v_knots(nurbs_patch: UsdGeom.NurbsPatch, v_knots: Sequence[float]) -> None:
    """Set the V knot vector for a NurbsPatch.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The NurbsPatch prim to set the knots for.
        v_knots (Sequence[float]): The knot vector to set as a sequence of floats.

    Raises:
        ValueError: If the nurbs_patch prim is not a valid NurbsPatch prim.
        ValueError: If the length of v_knots is not equal to vVertexCount + vOrder.
        ValueError: If the v_knots are not monotonically increasing.
    """
    if not nurbs_patch or not isinstance(nurbs_patch, UsdGeom.NurbsPatch):
        raise ValueError("Invalid NurbsPatch prim")
    v_vertex_count = nurbs_patch.GetVVertexCountAttr().Get()
    v_order = nurbs_patch.GetVOrderAttr().Get()
    if len(v_knots) != v_vertex_count + v_order:
        raise ValueError(
            f"Length of v_knots ({len(v_knots)}) must be equal to vVertexCount ({v_vertex_count}) + vOrder ({v_order})"
        )
    if any((v_knots[i] > v_knots[i + 1] for i in range(len(v_knots) - 1))):
        raise ValueError("v_knots must be monotonically increasing")
    nurbs_patch.GetVKnotsAttr().Set(v_knots)


def set_nurbs_patch_u_range(prim: UsdGeom.NurbsPatch, u_range: Tuple[float, float]) -> None:
    """Set the u range for a NURBS patch prim.

    Args:
        prim (UsdGeom.NurbsPatch): The NURBS patch prim.
        u_range (Tuple[float, float]): The u range as a tuple of (min, max) values.

    Raises:
        ValueError: If the prim is not a valid NURBS patch or the range is invalid.
    """
    if not prim or not isinstance(prim, UsdGeom.NurbsPatch):
        raise ValueError("Invalid NURBS patch prim")
    (u_min, u_max) = u_range
    if u_min >= u_max:
        raise ValueError("Invalid u range. u_min must be less than u_max.")
    u_knots_attr = prim.GetUKnotsAttr()
    if not u_knots_attr:
        raise ValueError("U knots attribute not found on the prim")
    u_order_attr = prim.GetUOrderAttr()
    if not u_order_attr:
        raise ValueError("U order attribute not found on the prim")
    u_order = u_order_attr.Get()
    u_knots = u_knots_attr.Get()
    if u_min < u_knots[u_order - 1] or u_max > u_knots[-1]:
        raise ValueError("U range is outside the valid knot range")
    u_range_attr = prim.CreateURangeAttr()
    u_range_attr.Set(Gf.Vec2d(u_min, u_max))


def loft_curves_to_nurbs_patch(
    curves: List[Tuple[Gf.Vec3d, ...]], nu: int, uorder: int, nv: int, vorder: int
) -> UsdGeom.NurbsPatch:
    """
    Loft a list of curves into a NURBS patch.

    Args:
        curves (List[Tuple[Gf.Vec3d, ...]]): List of curves, each curve is a tuple of Gf.Vec3d points.
        nu (int): Number of vertices in the U direction.
        uorder (int): Order in the U direction.
        nv (int): Number of vertices in the V direction.
        vorder (int): Order in the V direction.

    Returns:
        UsdGeom.NurbsPatch: The resulting NURBS patch.
    """
    if not curves:
        raise ValueError("No curves provided for lofting.")
    if nu < 1 or nv < 1:
        raise ValueError("Invalid number of vertices. Both nu and nv must be greater than or equal to 1.")
    if uorder < 2 or vorder < 2:
        raise ValueError("Invalid order. Both uorder and vorder must be greater than or equal to 2.")
    if nu < uorder or nv < vorder:
        raise ValueError("Number of vertices must be greater than or equal to the order in both U and V directions.")
    stage = Usd.Stage.CreateInMemory()
    patch = UsdGeom.NurbsPatch.Define(stage, "/NurbsPatch")
    patch.CreateUVertexCountAttr().Set(nu)
    patch.CreateVVertexCountAttr().Set(nv)
    patch.CreateUOrderAttr().Set(uorder)
    patch.CreateVOrderAttr().Set(vorder)
    uknots = [float(i) / (nu - uorder + 1) for i in range(nu + uorder)]
    vknots = [float(i) / (nv - vorder + 1) for i in range(nv + vorder)]
    patch.CreateUKnotsAttr().Set(uknots)
    patch.CreateVKnotsAttr().Set(vknots)
    points = []
    for v in range(nv):
        for u in range(nu):
            curve_index = int(v / (nv - 1) * (len(curves) - 1))
            point_index = int(u / (nu - 1) * (len(curves[curve_index]) - 1))
            points.append(Gf.Vec3f(*curves[curve_index][point_index]))
    patch.CreatePointsAttr().Set(points)
    return patch


def convert_nurbs_patch_to_subdiv(nurbs_prim: Usd.Prim) -> UsdGeom.Mesh:
    """Convert a NurbsPatch prim to a Mesh prim using Catmull-Clark subdivision."""
    if not nurbs_prim or not nurbs_prim.IsA(UsdGeom.NurbsPatch):
        raise ValueError("Invalid NurbsPatch prim")
    nurbs_mesh = UsdGeom.NurbsPatch(nurbs_prim)
    points = nurbs_mesh.GetPointsAttr().Get()
    u_range = nurbs_mesh.GetURangeAttr().Get()
    v_range = nurbs_mesh.GetVRangeAttr().Get()
    u_vertex_count = nurbs_mesh.GetUVertexCountAttr().Get()
    v_vertex_count = nurbs_mesh.GetVVertexCountAttr().Get()
    if not points or not u_range or (not v_range) or (not u_vertex_count) or (not v_vertex_count):
        raise ValueError("Missing required NurbsPatch attributes")
    stage = nurbs_prim.GetStage()
    mesh_path = nurbs_prim.GetPath().AppendChild("Subdiv")
    mesh_prim = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh_prim.CreatePointsAttr(points)
    mesh_prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.catmullClark)
    mesh_prim.CreateFaceVertexCountsAttr([4] * (u_vertex_count - 1) * (v_vertex_count - 1))
    mesh_prim.CreateFaceVertexIndicesAttr(
        [i + j * u_vertex_count for i in range(u_vertex_count - 1) for j in range(v_vertex_count - 1)]
        + [i + 1 + j * u_vertex_count for i in range(u_vertex_count - 1) for j in range(v_vertex_count - 1)]
        + [i + 1 + (j + 1) * u_vertex_count for i in range(u_vertex_count - 1) for j in range(v_vertex_count - 1)]
        + [i + (j + 1) * u_vertex_count for i in range(u_vertex_count - 1) for j in range(v_vertex_count - 1)]
    )
    mesh_prim.CreateExtentAttr(
        Vt.Vec3fArray(2, (Gf.Vec3f(u_range[0], v_range[0], 0), Gf.Vec3f(u_range[1], v_range[1], 0)))
    )
    return mesh_prim


def align_nurbs_patch_to_surface(nurbs_patch: UsdGeom.NurbsPatch, surface: UsdGeom.Imageable) -> None:
    """Align a NurbsPatch to a surface by copying the rotation and scale.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The NurbsPatch to align.
        surface (UsdGeom.Imageable): The surface to align the NurbsPatch to.

    Raises:
        ValueError: If the input prims are not valid or transformable.
    """
    if not nurbs_patch.GetPrim().IsValid():
        raise ValueError("Input NurbsPatch is not a valid prim.")
    if not surface.GetPrim().IsValid():
        raise ValueError("Input surface is not a valid prim.")
    nurbs_xform = UsdGeom.Xformable(nurbs_patch.GetPrim())
    surface_xform = UsdGeom.Xformable(surface.GetPrim())
    if not nurbs_xform:
        raise ValueError("Input NurbsPatch is not transformable.")
    if not surface_xform:
        raise ValueError("Input surface is not transformable.")
    surface_rotation = Gf.Vec3d(0, 0, 0)
    surface_scale = Gf.Vec3d(1, 1, 1)
    for op in surface_xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            surface_rotation = op.Get()
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
            surface_scale = op.Get()
    add_rotate_xyz_op(nurbs_xform).Set(surface_rotation)
    add_scale_op(nurbs_xform).Set(surface_scale)


def get_nurbs_patch_weights(nurbs_patch: UsdGeom.NurbsPatch) -> Tuple[bool, List[float]]:
    """
    Get the weights for a NurbsPatch prim.

    Args:
        nurbs_patch (UsdGeom.NurbsPatch): The NurbsPatch prim.

    Returns:
        A tuple containing a boolean indicating if the weights are authored,
        and a list of floats representing the weight values.
        If weights are not authored, returns (False, []).
    """
    if nurbs_patch.GetPointWeightsAttr().HasAuthoredValue():
        weights = nurbs_patch.GetPointWeightsAttr().Get()
        return (True, weights)
    else:
        return (False, [])


def set_plane_dimensions(plane: UsdGeom.Plane, width: float, length: float) -> None:
    """Set the width and length dimensions of a UsdGeomPlane.

    Args:
        plane (UsdGeom.Plane): The plane to set dimensions on.
        width (float): The width of the plane.
        length (float): The length of the plane.
    """
    if not plane:
        raise ValueError("Invalid UsdGeomPlane object")
    if width <= 0:
        raise ValueError("Width must be greater than zero")
    if length <= 0:
        raise ValueError("Length must be greater than zero")
    width_attr = plane.CreateWidthAttr()
    width_attr.Set(width)
    length_attr = plane.CreateLengthAttr()
    length_attr.Set(length)
    half_width = width / 2.0
    half_length = length / 2.0
    extent = [(-half_width, -half_length, 0), (half_width, half_length, 0)]
    extent_attr = plane.CreateExtentAttr()
    extent_attr.Set(extent)


def align_plane_to_axis(plane: UsdGeom.Plane, axis: str = "Z") -> None:
    """Align a UsdGeomPlane to a specified axis.

    Args:
        plane (UsdGeom.Plane): The plane to align.
        axis (str): The axis to align the plane to. Must be "X", "Y", or "Z". Defaults to "Z".

    Raises:
        ValueError: If the provided axis is not "X", "Y", or "Z".
    """
    if axis not in ["X", "Y", "Z"]:
        raise ValueError(f"Invalid axis '{axis}'. Must be 'X', 'Y', or 'Z'.")
    plane_axis_attr = plane.GetAxisAttr()
    if plane_axis_attr:
        plane_axis_attr.Set(axis)
    else:
        plane_axis_attr = plane.CreateAxisAttr(axis, writeSparsely=False)
    if axis == "Z":
        extent = [(-1, -1, 0), (1, 1, 0)]
    elif axis == "X":
        extent = [(0, -1, -1), (0, 1, 1)]
    else:
        extent = [(-1, 0, -1), (1, 0, 1)]
    plane_extent_attr = plane.GetExtentAttr()
    if plane_extent_attr:
        plane_extent_attr.Set(extent)
    else:
        plane_extent_attr = plane.CreateExtentAttr(extent, writeSparsely=False)


def get_plane_extent(plane: UsdGeom.Plane) -> Gf.Range3d:
    """
    Get the extent of a UsdGeomPlane.

    If the plane has authored extent, return that value.
    Otherwise, compute the extent based on the plane's width and length.
    """
    extent_attr = plane.GetExtentAttr()
    if extent_attr.HasAuthoredValue():
        extent = extent_attr.Get()
        return Gf.Range3d(Gf.Vec3d(extent[0]), Gf.Vec3d(extent[1]))
    else:
        width = plane.GetWidthAttr().Get()
        length = plane.GetLengthAttr().Get()
        half_width = width / 2.0
        half_length = length / 2.0
        axis = plane.GetAxisAttr().Get()
        if axis == UsdGeom.Tokens.x:
            min_point = Gf.Vec3d(0, -half_width, -half_length)
            max_point = Gf.Vec3d(0, half_width, half_length)
        elif axis == UsdGeom.Tokens.y:
            min_point = Gf.Vec3d(-half_width, 0, -half_length)
            max_point = Gf.Vec3d(half_width, 0, half_length)
        else:
            min_point = Gf.Vec3d(-half_width, -half_length, 0)
            max_point = Gf.Vec3d(half_width, half_length, 0)
        return Gf.Range3d(min_point, max_point)


def assign_material_to_plane(plane: UsdGeom.Plane, material: UsdShade.Material) -> None:
    """Assign a material to a UsdGeomPlane.

    Args:
        plane (UsdGeom.Plane): The plane prim to assign the material to.
        material (UsdShade.Material): The material prim to assign.

    Raises:
        ValueError: If the plane or material prim is not valid.
    """
    if not plane.GetPrim().IsValid():
        raise ValueError("The provided plane prim is not valid.")
    if not material.GetPrim().IsValid():
        raise ValueError("The provided material prim is not valid.")
    binding_api = UsdShade.MaterialBindingAPI(plane)
    binding_api.Bind(material)


def set_plane_axis(plane: UsdGeom.Plane, axis: str) -> None:
    """Set the axis of a UsdGeomPlane.

    Args:
        plane (UsdGeom.Plane): The plane to set the axis on.
        axis (str): The axis to set. Must be 'X', 'Y', or 'Z'.

    Raises:
        ValueError: If the provided axis is not 'X', 'Y', or 'Z'.
    """
    if axis not in ["X", "Y", "Z"]:
        raise ValueError(f"Invalid axis '{axis}'. Must be 'X', 'Y', or 'Z'.")
    axis_attr = plane.GetAxisAttr()
    axis_attr.Set(axis)
    if axis == "X":
        extent = Vt.Vec3fArray([(0, -1, -1), (0, 1, 1)])
    elif axis == "Y":
        extent = Vt.Vec3fArray([(-1, 0, -1), (1, 0, 1)])
    else:
        extent = Vt.Vec3fArray([(-1, -1, 0), (1, 1, 0)])
    plane.CreateExtentAttr(extent, True)


def find_planes_by_axis(stage: Usd.Stage, axis: str) -> List[UsdGeom.Plane]:
    """Find all UsdGeomPlane prims on the stage with the specified axis.

    Args:
        stage (Usd.Stage): The stage to search for planes.
        axis (str): The axis to match. Must be 'X', 'Y', or 'Z'.

    Returns:
        List[UsdGeom.Plane]: A list of all matching UsdGeomPlane prims.

    Raises:
        ValueError: If the provided axis is not 'X', 'Y', or 'Z'.
    """
    if axis not in ["X", "Y", "Z"]:
        raise ValueError(f"Invalid axis '{axis}'. Must be 'X', 'Y', or 'Z'.")
    plane_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Plane)]
    matching_planes = []
    for plane_prim in plane_prims:
        plane = UsdGeom.Plane(plane_prim)
        if plane.GetAxisAttr().Get() == axis:
            matching_planes.append(plane)
    return matching_planes


def toggle_plane_double_sided(plane: UsdGeom.Plane) -> bool:
    """Toggle the double-sided attribute of a UsdGeomPlane.

    Args:
        plane (UsdGeom.Plane): The plane to toggle the double-sided attribute on.

    Returns:
        bool: The new state of the double-sided attribute.
    """
    double_sided_attr = plane.GetDoubleSidedAttr()
    if not double_sided_attr.IsValid():
        raise ValueError("The double-sided attribute is not valid on this plane.")
    current_value = double_sided_attr.Get()
    new_value = not current_value
    double_sided_attr.Set(new_value)
    return new_value


def get_plane_axis(plane: UsdGeom.Plane) -> str:
    """Get the axis of a USD plane prim.

    Args:
        plane (UsdGeom.Plane): The plane prim to get the axis of.

    Returns:
        str: The axis of the plane ('X', 'Y', or 'Z').

    Raises:
        ValueError: If the input prim is not a valid UsdGeom.Plane.
    """
    if not plane or not isinstance(plane, UsdGeom.Plane):
        raise ValueError("Invalid UsdGeom.Plane prim.")
    axis_attr = plane.GetAxisAttr()
    if axis_attr.IsDefined() and axis_attr.HasValue():
        axis = axis_attr.Get()
        return str(axis)
    else:
        return "Z"


def merge_planes(stage: Usd.Stage, prim_paths: Sequence[str], merged_prim_name: str = "Merged_Plane") -> UsdGeom.Plane:
    """Merge multiple plane prims into a single plane prim.

    The merged plane will have the union of the extents of the input planes.
    The merged plane will be placed at the centroid of the input planes.

    Args:
        stage (Usd.Stage): The stage containing the planes to merge.
        prim_paths (Sequence[str]): The paths of the planes to merge.
        merged_prim_name (str): The name to give the merged plane prim.

    Returns:
        UsdGeom.Plane: The merged plane prim.

    Raises:
        ValueError: If any of the input prim paths are invalid or not planes.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")
        if not prim.IsA(UsdGeom.Plane):
            raise ValueError(f"Prim at path {prim_path} is not a plane")
    extents = []
    centroids = []
    for prim_path in prim_paths:
        plane = UsdGeom.Plane(stage.GetPrimAtPath(prim_path))
        extent = plane.GetExtentAttr().Get()
        extents.append(extent)
        centroid = Gf.Vec3f((extent[0] + extent[1]) / 2.0)
        centroids.append(centroid)
    merged_extent_min = Gf.Vec3f(
        min((e[0][0] for e in extents)), min((e[0][1] for e in extents)), min((e[0][2] for e in extents))
    )
    merged_extent_max = Gf.Vec3f(
        max((e[1][0] for e in extents)), max((e[1][1] for e in extents)), max((e[1][2] for e in extents))
    )
    merged_extent = (merged_extent_min, merged_extent_max)
    merged_centroid = Gf.Vec3f(
        sum((c[0] for c in centroids)) / len(centroids),
        sum((c[1] for c in centroids)) / len(centroids),
        sum((c[2] for c in centroids)) / len(centroids),
    )
    merged_prim_path = f"/World/{merged_prim_name}"
    merged_plane = UsdGeom.Plane.Define(stage, merged_prim_path)
    merged_plane.GetExtentAttr().Set(merged_extent)
    add_translate_op(UsdGeom.Xformable(merged_plane)).Set(merged_centroid)
    return merged_plane


def create_plane(
    stage: Usd.Stage, prim_path: str, length: float = 2.0, width: float = 2.0, axis: str = "Z"
) -> UsdGeom.Plane:
    """Create a plane primitive at the specified prim path.

    Args:
        stage (Usd.Stage): The USD stage to create the plane on.
        prim_path (str): The path where the plane prim should be created.
        length (float, optional): The length of the plane. Defaults to 2.0.
        width (float, optional): The width of the plane. Defaults to 2.0.
        axis (str, optional): The axis along which the plane is aligned. Must be "X", "Y", or "Z". Defaults to "Z".

    Returns:
        UsdGeom.Plane: The created plane prim.

    Raises:
        ValueError: If the provided axis is not "X", "Y", or "Z".
    """
    if axis not in ["X", "Y", "Z"]:
        raise ValueError(f"Invalid axis '{axis}'. Must be 'X', 'Y', or 'Z'.")
    plane: UsdGeom.Plane = UsdGeom.Plane.Define(stage, Sdf.Path(prim_path))
    plane.GetLengthAttr().Set(length)
    plane.GetWidthAttr().Set(width)
    plane.GetAxisAttr().Set(axis)
    half_length = length / 2.0
    half_width = width / 2.0
    extent = [Gf.Vec3f(-half_width, -half_length, 0), Gf.Vec3f(half_width, half_length, 0)]
    plane.GetExtentAttr().Set(extent)
    return plane


def duplicate_plane(stage: Usd.Stage, plane_path: str, new_plane_path: str) -> UsdGeom.Plane:
    """Duplicate a UsdGeomPlane prim.

    Args:
        stage (Usd.Stage): The stage containing the plane prim.
        plane_path (str): The path of the plane prim to duplicate.
        new_plane_path (str): The path for the new duplicated plane prim.

    Returns:
        UsdGeom.Plane: The new duplicated plane prim.

    Raises:
        ValueError: If the plane prim does not exist or is not a valid UsdGeomPlane.
    """
    plane = UsdGeom.Plane(stage.GetPrimAtPath(plane_path))
    if not plane:
        raise ValueError(f"Prim at path {plane_path} is not a valid UsdGeomPlane.")
    new_plane = UsdGeom.Plane.Define(stage, new_plane_path)
    attrs_to_copy = ["axis", "doubleSided", "extent", "length", "width"]
    for attr_name in attrs_to_copy:
        attr = plane.GetPrim().GetAttribute(attr_name)
        if attr.HasValue():
            attr_value = attr.Get()
            new_plane.GetPrim().CreateAttribute(attr_name, attr.GetTypeName()).Set(attr_value)
    return new_plane


def import_point_cloud_from_file(
    stage: Usd.Stage, prim_path: str, file_path: str, scale: float = 1.0
) -> UsdGeom.Points:
    """Import point cloud data from a file into a USD Points prim.

    Args:
        stage (Usd.Stage): The USD stage to create the points prim in.
        prim_path (str): The path where the points prim will be created.
        file_path (str): The path to the file containing the point cloud data.
        scale (float, optional): Scale factor to apply to the point positions. Defaults to 1.0.

    Returns:
        UsdGeom.Points: The created points prim.
    """
    if not stage:
        raise ValueError("Invalid stage")
    if not prim_path:
        raise ValueError("Invalid prim path")
    if not file_path:
        raise ValueError("Invalid file path")
    points_prim = UsdGeom.Points.Define(stage, prim_path)
    if not points_prim:
        raise RuntimeError(f"Failed to create points prim at {prim_path}")
    points = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                (x, y, z) = map(float, line.split())
                points.append(Gf.Vec3f(x * scale, y * scale, z * scale))
    except IOError as e:
        raise RuntimeError(f"Failed to read point cloud data from {file_path}: {str(e)}") from e
    points_attr = points_prim.CreatePointsAttr()
    points_attr.Set(points)
    return points_prim


def create_point_cloud(
    stage: Usd.Stage, prim_path: str, points: List[Gf.Vec3f], widths: List[float] = None, ids: List[int] = None
) -> UsdGeom.Points:
    """Create a UsdGeomPoints prim with given points, widths, and ids.

    Args:
        stage (Usd.Stage): The stage to create the point cloud on.
        prim_path (str): The path where the point cloud prim will be created.
        points (List[Gf.Vec3f]): The list of point positions.
        widths (List[float], optional): The list of widths for each point. Defaults to None.
        ids (List[int], optional): The list of ids for each point. Defaults to None.

    Raises:
        ValueError: If the lengths of points, widths, and ids do not match.

    Returns:
        UsdGeom.Points: The created point cloud prim.
    """
    points_prim = UsdGeom.Points.Define(stage, prim_path)
    points_attr = points_prim.CreatePointsAttr()
    points_attr.Set(points)
    if widths is not None:
        if len(widths) != len(points):
            raise ValueError("Number of widths must match number of points.")
        widths_attr = points_prim.CreateWidthsAttr()
        widths_attr.Set(widths)
    if ids is not None:
        if len(ids) != len(points):
            raise ValueError("Number of ids must match number of points.")
        ids_attr = points_prim.CreateIdsAttr()
        ids_attr.Set(ids)
    return points_prim


def merge_point_clouds(
    points1: Vt.Vec3fArray,
    points2: Vt.Vec3fArray,
    ids1: Vt.IntArray = None,
    ids2: Vt.IntArray = None,
    widths1: Vt.FloatArray = None,
    widths2: Vt.FloatArray = None,
) -> Tuple[Vt.Vec3fArray, Vt.IntArray, Vt.FloatArray]:
    """Merge two point clouds into a single point cloud.

    Args:
        points1 (Vt.Vec3fArray): The points of the first point cloud.
        points2 (Vt.Vec3fArray): The points of the second point cloud.
        ids1 (Vt.IntArray, optional): The ids of the first point cloud. Defaults to None.
        ids2 (Vt.IntArray, optional): The ids of the second point cloud. Defaults to None.
        widths1 (Vt.FloatArray, optional): The widths of the first point cloud. Defaults to None.
        widths2 (Vt.FloatArray, optional): The widths of the second point cloud. Defaults to None.

    Returns:
        Tuple[Vt.Vec3fArray, Vt.IntArray, Vt.FloatArray]: The merged points, ids, and widths.
    """
    if not points1 or not points2:
        raise ValueError("Input point clouds cannot be empty.")
    if ids1 and len(ids1) != len(points1):
        raise ValueError("ids1 must have the same length as points1.")
    if ids2 and len(ids2) != len(points2):
        raise ValueError("ids2 must have the same length as points2.")
    if widths1 and len(widths1) != len(points1):
        raise ValueError("widths1 must have the same length as points1.")
    if widths2 and len(widths2) != len(points2):
        raise ValueError("widths2 must have the same length as points2.")
    merged_points = points1 + points2
    if ids1 and ids2:
        merged_ids = ids1 + ids2
    else:
        merged_ids = Vt.IntArray(range(len(merged_points)))
    if widths1 and widths2:
        merged_widths = widths1 + widths2
    else:
        merged_widths = Vt.FloatArray([1.0] * len(merged_points))
    return (merged_points, merged_ids, merged_widths)


def filter_points_by_id(points: UsdGeom.Points, ids: List[int]) -> Tuple[List[Gf.Vec3f], List[float]]:
    """Filter points by a list of ids.

    Args:
        points (UsdGeom.Points): The points prim to filter.
        ids (List[int]): The list of ids to filter by.

    Returns:
        Tuple[List[Gf.Vec3f], List[float]]: A tuple containing the filtered points and widths.
    """
    points_attr = points.GetPointsAttr()
    widths_attr = points.GetWidthsAttr()
    ids_attr = points.GetIdsAttr()
    if not ids_attr.HasValue():
        raise ValueError("Points prim does not have an ids attribute.")
    points_data = points_attr.Get()
    widths_data = widths_attr.Get()
    ids_data = ids_attr.Get()
    if len(points_data) != len(widths_data) or len(points_data) != len(ids_data):
        raise ValueError("Points, widths, and ids arrays must be the same length.")
    filtered_points = []
    filtered_widths = []
    for i, point_id in enumerate(ids_data):
        if point_id in ids:
            filtered_points.append(points_data[i])
            filtered_widths.append(widths_data[i])
    return (filtered_points, filtered_widths)


def export_point_cloud_to_file(
    stage: Usd.Stage,
    prim_path: str,
    file_path: str,
    points: Sequence[Gf.Vec3f],
    widths: Optional[Sequence[float]] = None,
    ids: Optional[Sequence[int]] = None,
) -> bool:
    """Export a point cloud to a USD file.

    Args:
        stage (Usd.Stage): The USD stage to create the point cloud in.
        prim_path (str): The path where the point cloud prim will be created.
        file_path (str): The file path to save the USD file to.
        points (Sequence[Gf.Vec3f]): The points of the point cloud.
        widths (Optional[Sequence[float]]): The widths of the points. Defaults to None.
        ids (Optional[Sequence[int]]): The ids of the points. Defaults to None.

    Returns:
        bool: True if the export was successful, False otherwise.
    """
    if not stage:
        raise ValueError("Invalid stage")
    if not prim_path:
        raise ValueError("Invalid prim path")
    if not file_path:
        raise ValueError("Invalid file path")
    if not points:
        raise ValueError("No points provided")
    if widths and len(widths) != len(points):
        raise ValueError("Number of widths must match number of points")
    if ids and len(ids) != len(points):
        raise ValueError("Number of ids must match number of points")
    prim = stage.DefinePrim(prim_path, "Points")
    points_attr = prim.CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray)
    points_attr.Set(points)
    if widths:
        widths_attr = prim.CreateAttribute("widths", Sdf.ValueTypeNames.FloatArray)
        widths_attr.Set(widths)
        UsdGeom.Points(prim).SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    if ids:
        ids_attr = prim.CreateAttribute("ids", Sdf.ValueTypeNames.Int64Array)
        ids_attr.Set(ids)
    try:
        stage.GetRootLayer().Export(file_path)
        return True
    except Exception as e:
        print(f"Error saving USD file: {e}")
        return False


def get_point_cloud_extent(
    points: Vt.Vec3fArray, widths: Vt.FloatArray, transform: Gf.Matrix4d = Gf.Matrix4d(1)
) -> Gf.Range3f:
    """
    Compute the extent for the point cloud defined by points and widths.

    Args:
        points (Vt.Vec3fArray): Array of point positions.
        widths (Vt.FloatArray): Array of point widths.
        transform (Gf.Matrix4d): Optional transformation matrix to apply to the points.

    Returns:
        Gf.Range3f: The computed extent of the point cloud.

    Raises:
        ValueError: If the lengths of points and widths arrays do not match.
    """
    if len(points) != len(widths):
        raise ValueError("Points and widths arrays must have the same length.")
    extent = Gf.Range3f()
    for point, width in zip(points, widths):
        transformed_point = transform.Transform(Gf.Vec3f(point))
        point_bbox = Gf.Range3f(transformed_point - Gf.Vec3f(width / 2), transformed_point + Gf.Vec3f(width / 2))
        extent.UnionWith(point_bbox)
    return extent


def add_point_attributes(points_prim: UsdGeom.Points, ids: Vt.Int64Array, widths: Vt.FloatArray) -> None:
    """Add ids and widths attributes to a UsdGeom.Points prim.

    Args:
        points_prim (UsdGeom.Points): The points prim to add attributes to.
        ids (Vt.Int64Array): The ids array. Should be the same length as the points array.
        widths (Vt.FloatArray): The widths array. Should be the same length as the points array.

    Raises:
        ValueError: If the ids or widths arrays are not the same length as the points array.
    """
    num_points = len(points_prim.GetPointsAttr().Get())
    if len(ids) != num_points:
        raise ValueError("The ids array must be the same length as the points array.")
    if len(widths) != num_points:
        raise ValueError("The widths array must be the same length as the points array.")
    ids_attr = points_prim.GetIdsAttr()
    if not ids_attr:
        ids_attr = points_prim.CreateIdsAttr()
    ids_attr.Set(ids)
    widths_attr = points_prim.GetWidthsAttr()
    if not widths_attr:
        widths_attr = points_prim.CreateWidthsAttr()
    widths_attr.Set(widths)
    points_prim.SetWidthsInterpolation(UsdGeom.Tokens.vertex)


def update_point_cloud(
    points: UsdGeom.Points,
    points_data: List[Tuple[float, float, float]],
    ids_data: List[int] = None,
    widths_data: List[float] = None,
) -> None:
    """Update the point cloud with new points, ids, and widths data.

    Args:
        points (UsdGeom.Points): The point cloud prim to update.
        points_data (List[Tuple[float, float, float]]): The new point positions.
        ids_data (List[int], optional): The new point ids. If provided, must be the same length as points_data. Defaults to None.
        widths_data (List[float], optional): The new point widths. If provided, must be the same length as points_data. Defaults to None.

    Raises:
        ValueError: If ids_data or widths_data are provided but have a different length than points_data.
    """
    if not points:
        raise ValueError("Invalid UsdGeom.Points prim")
    points.GetPointsAttr().Set(points_data)
    if ids_data is not None:
        if len(ids_data) != len(points_data):
            raise ValueError("ids_data must be the same length as points_data")
        points.CreateIdsAttr().Set(ids_data)
    if widths_data is not None:
        if len(widths_data) != len(points_data):
            raise ValueError("widths_data must be the same length as points_data")
        points.CreateWidthsAttr().Set(widths_data)


def animate_point_cloud(
    points: UsdGeom.Points, positions: List[Tuple[float, float, float]], time_samples: List[float]
) -> None:
    """Animate a point cloud by setting point positions at different time samples.

    Args:
        points (UsdGeom.Points): The point cloud prim to animate.
        positions (List[Tuple[float, float, float]]): A list of point positions for each time sample.
        time_samples (List[float]): A list of time samples corresponding to the positions.

    Raises:
        ValueError: If the number of time samples does not match the number of position lists.
    """
    if len(positions) != len(time_samples):
        raise ValueError("Number of time samples does not match the number of position lists.")
    points_attr = points.GetPointsAttr()
    for i, time_sample in enumerate(time_samples):
        points_attr.Set(positions[i], Usd.TimeCode(time_sample))


def copy_point_cloud(
    source_prim: UsdGeom.Points, dest_prim: UsdGeom.Points, time_code: Usd.TimeCode = Usd.TimeCode.Default()
):
    """Copy point cloud data from source to destination prim.

    Args:
        source_prim (UsdGeom.Points): The source point cloud prim.
        dest_prim (UsdGeom.Points): The destination point cloud prim.
        time_code (Usd.TimeCode, optional): The time code at which to copy the data. Defaults to Default time code.
    """
    if not source_prim.GetPrim().IsValid():
        raise ValueError("Source prim is not valid.")
    if not dest_prim.GetPrim().IsValid():
        raise ValueError("Destination prim is not valid.")
    points_attr = source_prim.GetPointsAttr()
    if points_attr.IsValid():
        point_positions = points_attr.Get(time_code)
        if point_positions is not None:
            dest_prim.CreatePointsAttr().Set(point_positions, time_code)
    widths_attr = source_prim.GetWidthsAttr()
    if widths_attr.IsValid():
        widths = widths_attr.Get(time_code)
        if widths is not None:
            dest_widths_attr = dest_prim.CreateWidthsAttr()
            dest_widths_attr.Set(widths, time_code)
            widths_interp = source_prim.GetWidthsInterpolation()
            if widths_interp != "":
                dest_prim.SetWidthsInterpolation(widths_interp)
    ids_attr = source_prim.GetIdsAttr()
    if ids_attr.IsValid():
        ids = ids_attr.Get(time_code)
        if ids is not None:
            dest_prim.CreateIdsAttr().Set(ids, time_code)
    source_primvar_api = UsdGeom.PrimvarsAPI(source_prim)
    primvars = source_primvar_api.GetPrimvars()
    dest_primvar_api = UsdGeom.PrimvarsAPI(dest_prim)
    for primvar in primvars:
        primvar_attr = primvar.GetAttr()
        if primvar_attr.IsValid():
            primvar_value = primvar_attr.Get(time_code)
            if primvar_value is not None:
                interpolation = primvar.GetInterpolation()
                dest_primvar_attr = dest_primvar_api.CreatePrimvar(
                    primvar.GetName(), primvar_attr.GetTypeName(), interpolation
                ).GetAttr()
                dest_primvar_attr.Set(primvar_value, time_code)


def set_point_cloud_interpolation(points: UsdGeom.Points, interpolation: str) -> bool:
    """Set the interpolation for the points attribute of a UsdGeomPoints prim.

    Args:
        points (UsdGeom.Points): The UsdGeomPoints prim.
        interpolation (str): The interpolation type.

    Returns:
        bool: True if the interpolation was set successfully, False otherwise.
    """
    if not points.GetPrim().IsValid():
        print(f"Error: Invalid prim '{points.GetPrim().GetPath()}'")
        return False
    points_attr = points.GetPointsAttr()
    if not points_attr.IsValid():
        print(f"Error: Failed to get points attribute for prim '{points.GetPrim().GetPath()}'")
        return False
    try:
        success = points_attr.SetMetadata(UsdGeom.Tokens.interpolation, interpolation)
        if not success:
            print(f"Error: Failed to set interpolation to '{interpolation}' for prim '{points.GetPrim().GetPath()}'")
            return False
    except Tf.ErrorException as e:
        print(f"Error: {e}")
        return False
    return True


def transform_points(
    points: List[Tuple[float, float, float]], transform_matrix: Gf.Matrix4d
) -> List[Tuple[float, float, float]]:
    """Transform a list of points using a transform matrix.

    Args:
        points (List[Tuple[float, float, float]]): List of points as tuples (x, y, z).
        transform_matrix (Gf.Matrix4d): Transform matrix to apply to the points.

    Returns:
        List[Tuple[float, float, float]]: List of transformed points as tuples (x, y, z).
    """
    points_vec3f = [Gf.Vec3f(p[0], p[1], p[2]) for p in points]
    matrix_4f = Gf.Matrix4f(transform_matrix)
    transformed_points = [matrix_4f.Transform(p) for p in points_vec3f]
    transformed_points_tuples = [(p[0], p[1], p[2]) for p in transformed_points]
    return transformed_points_tuples


def create_point_based_prim(
    stage: Usd.Stage,
    prim_type: str,
    prim_path: str,
    points: Vt.Vec3fArray,
    normals: Vt.Vec3fArray = None,
    velocities: Vt.Vec3fArray = None,
    accelerations: Vt.Vec3fArray = None,
) -> UsdGeom.PointBased:
    """Create a PointBased prim with the given type, path, and geometry attributes."""
    if not stage:
        raise ValueError("Invalid stage")
    if not prim_type or not prim_path:
        raise ValueError("Prim type and path cannot be empty")
    if not points:
        raise ValueError("Points cannot be empty")
    prim = stage.DefinePrim(prim_path, prim_type)
    if not prim:
        raise RuntimeError(f"Failed to create prim at path {prim_path}")
    point_based = UsdGeom.PointBased(prim)
    if not point_based:
        raise RuntimeError(f"Failed to create PointBased schema for prim at path {prim_path}")
    point_based.CreatePointsAttr().Set(points)
    if normals:
        point_based.CreateNormalsAttr().Set(normals)
    if velocities:
        point_based.CreateVelocitiesAttr().Set(velocities)
    if accelerations:
        point_based.CreateAccelerationsAttr().Set(accelerations)
    return point_based


def add_velocity_to_points(
    points: Vt.Vec3fArray, velocities: Vt.Vec3fArray, time_code: Usd.TimeCode, time_offset: float
) -> Vt.Vec3fArray:
    """
    Add velocity to points to get the points at a future time.

    Args:
        points (Vt.Vec3fArray): The initial points.
        velocities (Vt.Vec3fArray): The velocities corresponding to the points.
        time_code (Usd.TimeCode): The base time code.
        time_offset (float): The offset into the future to compute new positions for in seconds.

    Returns:
        Vt.Vec3fArray: The points offset by velocity * time_offset.
    """
    if len(points) != len(velocities):
        raise ValueError("Points and velocities arrays must be the same length.")
    if time_offset < 0:
        raise ValueError("Time offset must be non-negative.")
    new_points = Vt.Vec3fArray(len(points))
    for i in range(len(points)):
        offset = velocities[i] * time_offset
        new_points[i] = points[i] + offset
    return new_points


def compute_motion_blur(
    points: Vt.Vec3fArray, velocities: Vt.Vec3fArray, time: float, time_codes_per_second: float
) -> Vt.Vec3fArray:
    """
    Compute motion blur points based on positions, velocities, and time.

    Args:
        points (Vt.Vec3fArray): Array of point positions.
        velocities (Vt.Vec3fArray): Array of point velocities.
        time (float): Time value for motion blur computation.
        time_codes_per_second (float): Number of time codes per second.

    Returns:
        Vt.Vec3fArray: Array of motion-blurred point positions.
    """
    if len(points) != len(velocities):
        raise ValueError("Points and velocities arrays must have the same length.")
    time_in_seconds = time / time_codes_per_second
    motion_blur_points = []
    for point, velocity in zip(points, velocities):
        motion_blur_point = point + velocity * time_in_seconds
        motion_blur_points.append(motion_blur_point)
    return Vt.Vec3fArray(motion_blur_points)


def simulate_physics_on_points(
    points: Vt.Vec3fArray,
    velocities: Vt.Vec3fArray,
    mass: float,
    time_step: float,
    num_steps: int,
    force_func: Callable[[Vt.Vec3fArray], Vt.Vec3fArray],
) -> Tuple[Vt.Vec3fArray, Vt.Vec3fArray]:
    """Simulate physics on a set of points.

    Args:
        points (Vt.Vec3fArray): The initial positions of the points.
        velocities (Vt.Vec3fArray): The initial velocities of the points.
        mass (float): The mass of each point.
        time_step (float): The time step for the simulation.
        num_steps (int): The number of simulation steps to perform.
        force_func (Callable[[Vt.Vec3fArray], Vt.Vec3fArray]): Function to compute forces on points.

    Returns:
        Tuple[Vt.Vec3fArray, Vt.Vec3fArray]: Final positions and velocities after simulation.
    """
    if len(points) != len(velocities):
        raise ValueError("Points and velocities arrays must have the same length.")
    for _ in range(num_steps):
        forces = force_func(points)
        for i in range(len(points)):
            acceleration = forces[i] / mass
            velocities[i] += acceleration * time_step
        for i in range(len(points)):
            points[i] += velocities[i] * time_step
    return (points, velocities)


def gravity_force(points: Vt.Vec3fArray) -> Vt.Vec3fArray:
    """Compute gravity force on points."""
    forces = []
    for _ in range(len(points)):
        force = Gf.Vec3f(0, -9.8, 0)
        forces.append(force)
    return Vt.Vec3fArray(forces)


def find_point_based_prims_by_velocity(stage: Usd.Stage, min_velocity: float) -> List[Usd.Prim]:
    """
    Find all PointBased prims on the stage that have a velocity magnitude greater than or equal to min_velocity.

    Args:
        stage (Usd.Stage): The stage to search for PointBased prims.
        min_velocity (float): The minimum velocity magnitude to filter by.

    Returns:
        List[Usd.Prim]: A list of PointBased prims with velocity magnitude greater than or equal to min_velocity.
    """
    matching_prims: List[Usd.Prim] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.PointBased):
            velocities_attr = UsdGeom.PointBased(prim).GetVelocitiesAttr()
            if velocities_attr.HasValue():
                velocities = velocities_attr.Get()
                if velocities:
                    magnitudes = [v.GetLength() for v in velocities]
                    if any((mag >= min_velocity for mag in magnitudes)):
                        matching_prims.append(prim)
    return matching_prims


def sample_point_based_prim(stage: Usd.Stage, prim_path: str, time_code: Usd.TimeCode) -> Vt.Vec3fArray:
    """Sample the points attribute of a PointBased prim at a specific time code.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the PointBased prim.
        time_code (Usd.TimeCode): The time code at which to sample the points.

    Returns:
        Vt.Vec3fArray: The sampled points.

    Raises:
        ValueError: If the prim is not valid or is not a PointBased prim.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    point_based = UsdGeom.PointBased(prim)
    if not point_based:
        raise ValueError(f"Prim at path {prim_path} is not a PointBased prim.")
    points_attr = point_based.GetPointsAttr()
    if not points_attr.HasValue():
        return Vt.Vec3fArray()
    points = points_attr.Get(time_code)
    return points


def merge_point_based_prims(
    stage: Usd.Stage, prim_paths: List[str], merged_prim_name: str = "merged_prim"
) -> UsdGeom.PointBased:
    """Merge multiple point-based prims into a single prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of paths to the prims to be merged.
        merged_prim_name (str, optional): The name of the merged prim. Defaults to "merged_prim".

    Returns:
        UsdGeom.PointBased: The merged point-based prim.
    """
    prims = [stage.GetPrimAtPath(path) for path in prim_paths]
    if not all((prim.IsValid() for prim in prims)):
        raise ValueError("One or more input prims are invalid.")
    if not all((prim.IsA(UsdGeom.PointBased) for prim in prims)):
        raise TypeError("All input prims must be point-based.")
    merged_prim = stage.DefinePrim(f"/{merged_prim_name}", "Points")
    merged_point_based = UsdGeom.Points(merged_prim)
    merged_points = []
    merged_velocities = []
    merged_accelerations = []
    for prim in prims:
        point_based = UsdGeom.PointBased(prim)
        points = point_based.GetPointsAttr().Get(Usd.TimeCode.Default())
        velocities = point_based.GetVelocitiesAttr().Get(Usd.TimeCode.Default())
        accelerations = point_based.GetAccelerationsAttr().Get(Usd.TimeCode.Default())
        if points:
            merged_points.extend(points)
        if velocities:
            merged_velocities.extend(velocities)
        if accelerations:
            merged_accelerations.extend(accelerations)
    merged_point_based.CreatePointsAttr(merged_points)
    if merged_velocities:
        merged_point_based.CreateVelocitiesAttr(merged_velocities)
    if merged_accelerations:
        merged_point_based.CreateAccelerationsAttr(merged_accelerations)
    return merged_point_based


def align_point_based_prims(stage: Usd.Stage, prim_paths: list[str], target_positions: list[Gf.Vec3f]) -> None:
    """Align multiple point-based prims to target positions.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (list[str]): The paths to the prims to align.
        target_positions (list[Gf.Vec3f]): The target positions to align the prims to.

    Raises:
        ValueError: If the number of prim paths and target positions do not match.
    """
    if len(prim_paths) != len(target_positions):
        raise ValueError("Number of prim paths and target positions must match.")
    for prim_path, target_pos in zip(prim_paths, target_positions):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        if prim.IsA(UsdGeom.PointInstancer):
            point_instancer = UsdGeom.PointInstancer(prim)
            positions_attr = point_instancer.GetPositionsAttr()
            positions = positions_attr.Get(Usd.TimeCode.Default())
            if positions is None:
                raise ValueError(f"Prim at path {prim_path} has no positions.")
            centroid = Gf.Vec3f(0, 0, 0)
            for position in positions:
                centroid += position
            centroid /= len(positions)
            translation = target_pos - centroid
            new_positions = [position + translation for position in positions]
            positions_attr.Set(new_positions, Usd.TimeCode.Default())
        elif prim.IsA(UsdGeom.PointBased):
            point_based = UsdGeom.PointBased(prim)
            points_attr = point_based.GetPointsAttr()
            points = points_attr.Get(Usd.TimeCode.Default())
            if points is None:
                raise ValueError(f"Prim at path {prim_path} has no points.")
            centroid = Gf.Vec3f(0, 0, 0)
            for point in points:
                centroid += point
            centroid /= len(points)
            translation = target_pos - centroid
            new_points = [point + translation for point in points]
            points_attr.Set(new_points, Usd.TimeCode.Default())
        else:
            raise ValueError(f"Prim at path {prim_path} is not a supported PointBased prim.")


def set_point_normals_based_on_geometry(prim: UsdGeom.PointBased) -> None:
    """Set the normals attribute on a PointBased prim based on its geometry."""
    if not prim.GetPrim().IsValid():
        raise ValueError("Invalid prim")
    points_attr = prim.GetPointsAttr()
    if not points_attr.HasValue():
        raise ValueError("Prim has no points attribute")
    points = points_attr.Get()
    num_points = len(points)
    normals_attr = prim.GetNormalsAttr()
    if normals_attr.HasValue() and len(normals_attr.Get()) == num_points:
        return
    normals = []
    for i in range(num_points):
        normals.append(Gf.Vec3f(0, 1, 0))
    normals_attr.Set(normals)
    prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)


def interpolate_points_between_times(points: Vt.Vec3fArray, times: Vt.Vec2dArray, time: float) -> Vt.Vec3fArray:
    """
    Interpolate point positions between two times.

    Args:
        points (Vt.Vec3fArray): Array of points with positions at two different times.
        times (Vt.Vec2dArray): Array of two time values corresponding to the point positions.
        time (float): The time at which to interpolate the point positions.

    Returns:
        Vt.Vec3fArray: Interpolated point positions at the specified time.
    """
    if len(points) != len(times):
        raise ValueError("points and times arrays must have the same length.")
    if len(times) != 2:
        raise ValueError("times array must have exactly two elements.")
    if time < times[0][0] or time > times[1][0]:
        raise ValueError("Interpolation time must be within the range of the provided times.")
    t = (time - times[0][0]) / (times[1][0] - times[0][0])
    interpolated_points = []
    for i in range(len(points[0])):
        interpolated_points.append(Gf.Lerp(t, points[0][i], points[1][i]))
    return Vt.Vec3fArray(interpolated_points)


def set_gprim_opacity(
    gprim: UsdGeom.Gprim, opacity: Gf.Vec3f, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> None:
    """Set the opacity for a Gprim.

    Args:
        gprim (UsdGeom.Gprim): The Gprim to set the opacity for.
        opacity (Gf.Vec3f): The opacity value to set as a Vec3f.
        time_code (Usd.TimeCode, optional): The time code to set the value at. Defaults to Default().
    """
    if not gprim:
        raise ValueError("Invalid Gprim.")
    display_opacity_attr = gprim.GetDisplayOpacityAttr()
    if not display_opacity_attr:
        display_opacity_attr = gprim.CreateDisplayOpacityAttr()
    display_opacity_attr.Set([opacity[0], opacity[1], opacity[2]], time_code)


def batch_toggle_double_sided(stage: Usd.Stage, prim_paths: List[str]) -> None:
    """Toggle the doubleSided attribute for a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths to toggle doubleSided for.

    Raises:
        ValueError: If any prim path is invalid or not a Gprim.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        gprim = UsdGeom.Gprim(prim)
        if not gprim:
            raise ValueError(f"Prim at path {prim_path} is not a Gprim.")
        doubleSided_attr = gprim.GetDoubleSidedAttr()
        if doubleSided_attr.HasAuthoredValue():
            current_value = doubleSided_attr.Get()
            doubleSided_attr.Set(not current_value)
        else:
            doubleSided_attr.Set(True)


def toggle_double_sided(prim: Usd.Prim) -> bool:
    """Toggle the doubleSided attribute of a Gprim.

    If the prim is not a Gprim, raise an error.
    If doubleSided is True, set it to False. If False, set it to True.
    Return the new value of doubleSided.
    """
    if not prim.IsA(UsdGeom.Gprim):
        raise ValueError(f"Prim at path {prim.GetPath()} is not a Gprim.")
    gprim = UsdGeom.Gprim(prim)
    double_sided_attr = gprim.GetDoubleSidedAttr()
    if not double_sided_attr.HasAuthoredValue():
        double_sided_attr.Set(True)
        return True
    current_value = double_sided_attr.Get()
    new_value = not current_value
    double_sided_attr.Set(new_value)
    return new_value


def batch_create_gprims(stage: Usd.Stage, prim_paths: List[str], prim_type: str = "Cube") -> List[UsdGeom.Gprim]:
    """Create multiple Gprims of the same type in a single operation.

    Args:
        stage (Usd.Stage): The stage to create the prims on.
        prim_paths (List[str]): A list of prim paths where the Gprims should be created.
        prim_type (str, optional): The type of Gprim to create. Defaults to "Cube".

    Returns:
        List[UsdGeom.Gprim]: A list of the created Gprims.
    """
    if not stage:
        raise ValueError("Invalid stage")
    if not prim_paths:
        raise ValueError("No prim paths provided")
    gprims = []
    for prim_path in prim_paths:
        full_path = Sdf.Path(prim_path)
        if stage.GetPrimAtPath(full_path):
            continue
        prim = stage.DefinePrim(full_path, prim_type)
        gprim = UsdGeom.Gprim(prim)
        gprims.append(gprim)
    return gprims


def copy_gprim_display_attributes(src_prim: UsdGeom.Gprim, dst_prim: UsdGeom.Gprim):
    """Copy display attributes from one Gprim to another."""
    src_display_color_attr = src_prim.GetDisplayColorAttr()
    if src_display_color_attr.HasAuthoredValue():
        src_display_color = src_display_color_attr.Get()
        dst_prim.GetDisplayColorAttr().Set(src_display_color)
    else:
        dst_prim.GetDisplayColorAttr().Clear()
    src_display_opacity_attr = src_prim.GetDisplayOpacityAttr()
    if src_display_opacity_attr.HasAuthoredValue():
        src_display_opacity = src_display_opacity_attr.Get()
        dst_prim.GetDisplayOpacityAttr().Set(src_display_opacity)
    else:
        dst_prim.GetDisplayOpacityAttr().Clear()
    src_double_sided_attr = src_prim.GetDoubleSidedAttr()
    if src_double_sided_attr.HasAuthoredValue():
        src_double_sided = src_double_sided_attr.Get()
        dst_prim.GetDoubleSidedAttr().Set(src_double_sided)
    else:
        dst_prim.GetDoubleSidedAttr().Clear()
    src_orientation_attr = src_prim.GetOrientationAttr()
    if src_orientation_attr.HasAuthoredValue():
        src_orientation = src_orientation_attr.Get()
        dst_prim.GetOrientationAttr().Set(src_orientation)
    else:
        dst_prim.GetOrientationAttr().Clear()


def align_gprim_orientations(stage: Usd.Stage):
    """Align the orientation of all Gprims on the given stage to 'rightHanded'."""
    prims = stage.TraverseAll()
    for prim in prims:
        if prim.IsA(UsdGeom.Gprim):
            gprim = UsdGeom.Gprim(prim)
            orientation_attr = gprim.GetOrientationAttr()
            if orientation_attr.IsAuthored():
                orientation_attr.Set(UsdGeom.Tokens.rightHanded)
            else:
                gprim.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)


def scale_gprim_colors(gprim: UsdGeom.Gprim, scale_factor: float) -> None:
    """Scale the displayColor and displayOpacity of a Gprim by a scale factor.

    Args:
        gprim (UsdGeom.Gprim): The Gprim to modify.
        scale_factor (float): The scale factor to apply to the color and opacity values.

    Raises:
        ValueError: If the input Gprim is not valid.
    """
    if not gprim or not gprim.GetPrim().IsValid():
        raise ValueError("Invalid Gprim provided.")
    color_attr = gprim.GetDisplayColorAttr()
    opacity_attr = gprim.GetDisplayOpacityAttr()
    if color_attr.HasAuthoredValue():
        color_values = color_attr.Get()
        scaled_colors = [Gf.Vec3f(c[0] * scale_factor, c[1] * scale_factor, c[2] * scale_factor) for c in color_values]
        color_attr.Set(scaled_colors)
    if opacity_attr.HasAuthoredValue():
        opacity_values = opacity_attr.Get()
        scaled_opacities = [o * scale_factor for o in opacity_values]
        opacity_attr.Set(scaled_opacities)


def batch_set_gprim_attributes(
    stage: Usd.Stage,
    prim_paths: List[str],
    double_sided: Optional[bool] = None,
    orientation: Optional[str] = None,
    display_opacity: Optional[List[float]] = None,
    display_color: Optional[List[Gf.Vec3f]] = None,
) -> None:
    """Set Gprim attributes in batch for multiple prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths to set attributes for.
        double_sided (Optional[bool]): Value for doubleSided attribute. If None, attribute is not set.
        orientation (Optional[str]): Value for orientation attribute. Must be "rightHanded" or "leftHanded". If None, attribute is not set.
        display_opacity (Optional[List[float]]): Value for displayOpacity attribute. If None, attribute is not set.
        display_color (Optional[List[Gf.Vec3f]]): Value for displayColor attribute. If None, attribute is not set.
    """
    if orientation and orientation not in ["rightHanded", "leftHanded"]:
        raise ValueError(f'Invalid orientation value "{orientation}". Must be "rightHanded" or "leftHanded".')
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            raise ValueError(f'Prim not found at path "{prim_path}"')
        gprim = UsdGeom.Gprim(prim)
        if not gprim:
            raise ValueError(f'Prim at path "{prim_path}" is not a Gprim')
        if double_sided is not None:
            gprim.CreateDoubleSidedAttr(double_sided)
        if orientation:
            gprim.CreateOrientationAttr(orientation)
        if display_opacity:
            gprim.CreateDisplayOpacityAttr(display_opacity)
        if display_color:
            gprim.CreateDisplayColorAttr(display_color)


def set_gprim_color_over_time(
    prim: UsdGeom.Gprim, color_attr_name: str, color_values: List[Tuple[Gf.Vec3f, Usd.TimeCode]]
) -> None:
    """Set color values for a Gprim over time.

    Args:
        prim (UsdGeom.Gprim): The Gprim to set the color values on.
        color_attr_name (str): The name of the color attribute to set (e.g., "primvars:displayColor").
        color_values (List[Tuple[Gf.Vec3f, Usd.TimeCode]]): A list of tuples containing the color value and timecode.

    Raises:
        ValueError: If the prim is not a valid Gprim.
        AttributeError: If the specified color attribute does not exist on the prim.
    """
    if not prim or not isinstance(prim, UsdGeom.Gprim):
        raise ValueError("Invalid Gprim provided.")
    color_attr = prim.GetPrim().GetAttribute(color_attr_name)
    if not color_attr:
        raise AttributeError(f"Color attribute '{color_attr_name}' does not exist on the prim.")
    for color, timecode in color_values:
        color_attr.Set(Vt.Vec3fArray([color]), timecode)


def merge_display_colors(prim: UsdGeom.Gprim, source_prim: UsdGeom.Gprim) -> None:
    """Merges display colors from source_prim onto prim.

    If prim already has authored display colors, they will be overridden by
    the display colors from source_prim.

    Args:
        prim (UsdGeom.Gprim): The prim to merge display colors onto.
        source_prim (UsdGeom.Gprim): The prim to merge display colors from.

    Raises:
        ValueError: If either prim or source_prim is not a valid UsdGeomGprim.
    """
    if not prim:
        raise ValueError(f"Prim {prim.GetPrim().GetPath()} is not a valid UsdGeomGprim")
    if not source_prim:
        raise ValueError(f"Source prim {source_prim.GetPrim().GetPath()} is not a valid UsdGeomGprim")
    source_display_color_attr = source_prim.GetDisplayColorAttr()
    if source_display_color_attr.HasAuthoredValue():
        source_display_color = source_display_color_attr.Get()
        prim.GetDisplayColorAttr().Set(source_display_color)
    else:
        prim.GetDisplayColorAttr().Block()


def test_merge_display_colors():
    stage = Usd.Stage.CreateInMemory()
    source_prim = UsdGeom.Sphere.Define(stage, "/source_sphere")
    source_prim.GetDisplayColorAttr().Set([(1, 0, 0)])
    target_prim = UsdGeom.Sphere.Define(stage, "/target_sphere")
    target_prim.GetDisplayColorAttr().Set([(0, 1, 0)])
    merge_display_colors(target_prim, source_prim)
    merged_display_color = target_prim.GetDisplayColorAttr().Get()
    assert merged_display_color == [(1, 0, 0)]
    source_prim2 = UsdGeom.Sphere.Define(stage, "/source_sphere2")
    merge_display_colors(target_prim, source_prim2)
    assert not target_prim.GetDisplayColorAttr().HasAuthoredValue()
    print("All tests passed!")


def create_gprim_hierarchy(stage: Usd.Stage, root_path: str, num_levels: int) -> None:
    """Create a hierarchy of Xforms with a Mesh at each leaf node."""
    if not stage:
        raise ValueError("Stage is invalid")
    if not root_path:
        raise ValueError("Root path is empty")
    if num_levels <= 0:
        raise ValueError("Number of levels must be positive")

    def create_level(parent_path: str, level: int):
        xform_path = parent_path + f"/Xform_{level}"
        xform = UsdGeom.Xform.Define(stage, xform_path)
        if level < num_levels:
            create_level(xform_path, level + 1)
        else:
            mesh_path = xform_path + "/Mesh"
            UsdGeom.Mesh.Define(stage, mesh_path)

    root_xform = UsdGeom.Xform.Define(stage, root_path)
    create_level(root_path, 1)


def create_gprim_grid(stage: Usd.Stage, grid_size: Tuple[int, int], prim_type: str, prim_name_prefix: str) -> None:
    """Create a grid of gprims on the given stage.

    Args:
        stage (Usd.Stage): The USD stage to create the grid on.
        grid_size (Tuple[int, int]): The number of rows and columns in the grid.
        prim_type (str): The type of gprim to create (e.g., "Sphere", "Cube").
        prim_name_prefix (str): The prefix for the prim names.
    """
    if grid_size[0] <= 0 or grid_size[1] <= 0:
        raise ValueError("Grid size must be positive.")
    prim_cls = getattr(UsdGeom, prim_type, None)
    if prim_cls is None:
        raise ValueError(f"Invalid prim type: {prim_type}")
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            prim_path = f"/World/{prim_name_prefix}_{i}_{j}"
            prim = prim_cls.Define(stage, Sdf.Path(prim_path))
            translation = Gf.Vec3d(i, 0, j)
            add_translate_op(UsdGeom.Xformable(prim)).Set(translation)


def create_gprim_with_material(stage: Usd.Stage, prim_path: str, gprim_type: str, material_path: str) -> UsdGeom.Gprim:
    """Create a Gprim with a material binding.

    Args:
        stage (Usd.Stage): The USD stage to create the prim on.
        prim_path (str): The path where the Gprim will be created.
        gprim_type (str): The type of Gprim to create (e.g., "Sphere", "Cube", "Cylinder").
        material_path (str): The path to the material to bind to the Gprim.

    Returns:
        UsdGeom.Gprim: The created Gprim prim.

    Raises:
        ValueError: If an invalid Gprim type is provided or if the material prim does not exist.
    """
    valid_types = ["Sphere", "Cube", "Cylinder", "Cone", "Capsule"]
    if gprim_type not in valid_types:
        raise ValueError(f"Invalid Gprim type '{gprim_type}'. Must be one of: {', '.join(valid_types)}")
    gprim = None
    if gprim_type == "Sphere":
        gprim = UsdGeom.Sphere.Define(stage, Sdf.Path(prim_path))
    elif gprim_type == "Cube":
        gprim = UsdGeom.Cube.Define(stage, Sdf.Path(prim_path))
    elif gprim_type == "Cylinder":
        gprim = UsdGeom.Cylinder.Define(stage, Sdf.Path(prim_path))
    elif gprim_type == "Cone":
        gprim = UsdGeom.Cone.Define(stage, Sdf.Path(prim_path))
    elif gprim_type == "Capsule":
        gprim = UsdGeom.Capsule.Define(stage, Sdf.Path(prim_path))
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    if not material:
        raise ValueError(f"Material prim at path '{material_path}' does not exist or is not a valid Material.")
    UsdShade.MaterialBindingAPI(gprim).Bind(material)
    return gprim


def get_purpose_attributes(prim: UsdGeom.Imageable) -> List[Usd.Attribute]:
    """
    Get the purpose attributes for a UsdGeomImageable prim.

    Args:
        prim (UsdGeom.Imageable): The imageable prim to get purpose attributes for.

    Returns:
        List[Usd.Attribute]: A list of purpose attributes on the prim.
    """
    purpose_attrs = []
    purpose_attr = prim.GetPurposeAttr()
    if purpose_attr.HasAuthoredValue():
        purpose_attrs.append(purpose_attr)
    vis_api = UsdGeom.VisibilityAPI(prim)
    if vis_api:
        guide_vis_attr = vis_api.GetGuideVisibilityAttr()
        if guide_vis_attr.HasAuthoredValue():
            purpose_attrs.append(guide_vis_attr)
        proxy_vis_attr = vis_api.GetProxyVisibilityAttr()
        if proxy_vis_attr.HasAuthoredValue():
            purpose_attrs.append(proxy_vis_attr)
        render_vis_attr = vis_api.GetRenderVisibilityAttr()
        if render_vis_attr.HasAuthoredValue():
            purpose_attrs.append(render_vis_attr)
    return purpose_attrs


def compute_and_cache_transforms(stage: Usd.Stage, time: Usd.TimeCode) -> Dict[str, Gf.Matrix4d]:
    """Compute and cache the local-to-world transforms for all prims on the stage.

    Args:
        stage (Usd.Stage): The USD stage to compute transforms for.
        time (Usd.TimeCode): The time at which to compute the transforms.

    Returns:
        Dict[str, Gf.Matrix4d]: A dictionary mapping prim paths to their computed local-to-world transforms.
    """
    xform_cache = UsdGeom.XformCache(time)
    transforms = {}
    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Xformable):
            prim_path = str(prim.GetPath())
            try:
                local_to_world = xform_cache.GetLocalToWorldTransform(prim)
            except Exception as e:
                print(f"Error computing transform for prim {prim_path}: {str(e)}")
                continue
            transforms[prim_path] = local_to_world
    return transforms


def compute_hierarchical_visibility(prim: Usd.Prim, time: Usd.TimeCode) -> str:
    """Compute the hierarchical visibility for a prim at a given time.

    This function recursively checks the visibility of the prim and its ancestors.
    If any ancestor is invisible, the prim is considered invisible.

    Args:
        prim (Usd.Prim): The prim to compute visibility for.
        time (Usd.TimeCode): The time at which to compute visibility.

    Returns:
        str: The computed visibility, either 'inherited' or 'invisible'.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        raise ValueError(f"Prim {prim} is not imageable")
    vis_attr = imageable.GetVisibilityAttr()
    if vis_attr.HasAuthoredValue():
        return vis_attr.Get(time)
    ancestor = prim.GetParent()
    while ancestor.IsValid():
        ancestor_imageable = UsdGeom.Imageable(ancestor)
        if not ancestor_imageable:
            ancestor = ancestor.GetParent()
            continue
        ancestor_vis_attr = ancestor_imageable.GetVisibilityAttr()
        if ancestor_vis_attr.HasAuthoredValue():
            ancestor_vis = ancestor_vis_attr.Get(time)
            if ancestor_vis == "invisible":
                return "invisible"
            else:
                ancestor = ancestor.GetParent()
        else:
            ancestor = ancestor.GetParent()
    return "inherited"


def batch_set_visibility(
    stage: Usd.Stage, prim_paths: List[str], visibility: str, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> None:
    """
    Set the visibility for a list of prims at a specific time.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (List[str]): A list of prim paths to set visibility for.
        visibility (str): The visibility value to set. Must be either "inherited" or "invisible".
        time_code (Usd.TimeCode, optional): The time code at which to set the visibility. Defaults to Default time code.

    Raises:
        ValueError: If an invalid visibility value is provided or if any of the prim paths are invalid.
    """
    if visibility not in ["inherited", "invisible"]:
        raise ValueError(f"Invalid visibility value '{visibility}'. Must be 'inherited' or 'invisible'.")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path '{prim_path}' does not exist.")
        imageable = UsdGeom.Imageable(prim)
        imageable.GetVisibilityAttr().Set(visibility, time_code)


def compute_proxy_prims(prim: Usd.Prim) -> List[Tuple[Usd.Prim, Usd.Prim]]:
    """
    Compute the proxy prims for the given prim and its descendants.

    Args:
        prim (Usd.Prim): The prim to compute proxy prims for.

    Returns:
        List[Tuple[Usd.Prim, Usd.Prim]]: A list of tuples, where each tuple contains
            the proxy prim and the corresponding render prim.
    """
    proxy_prims = []
    if not prim.IsValid():
        return proxy_prims
    for descendant in Usd.PrimRange(prim):
        if UsdGeom.Imageable(descendant):
            proxy_prim_rel = UsdGeom.Imageable(descendant).GetProxyPrimRel()
            if proxy_prim_rel and proxy_prim_rel.GetTargets():
                proxy_prim_target = proxy_prim_rel.GetTargets()[0]
                proxy_prim = descendant.GetStage().GetPrimAtPath(proxy_prim_target)
                if proxy_prim.IsValid():
                    proxy_prims.append((proxy_prim, descendant))
    return proxy_prims


def adjust_visibility_based_on_hierarchy(prim: Usd.Prim, time: Usd.TimeCode):
    """Adjust visibility of the prim based on its hierarchy."""
    if not prim.IsA(UsdGeom.Imageable):
        return
    imageable = UsdGeom.Imageable(prim)
    visibility = imageable.ComputeVisibility(time)
    if visibility == UsdGeom.Tokens.inherited:
        return
    if visibility == UsdGeom.Tokens.invisible:
        imageable.MakeInvisible(time)
    else:
        imageable.MakeVisible(time)
        for child in prim.GetChildren():
            adjust_visibility_based_on_hierarchy(child, time)


def create_test_stage() -> Usd.Stage:
    """Create a test stage with a hierarchy of prims."""
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/A")
    UsdGeom.Xform.Define(stage, "/World/A/B")
    UsdGeom.Xform.Define(stage, "/World/A/C")
    UsdGeom.Xform.Define(stage, "/World/A/C/D")
    return stage


def set_proxy_relationships(stage: Usd.Stage, render_prim_path: str, proxy_prim_path: str) -> None:
    """Set the renderProxy relationship between a render prim and its proxy prim."""
    render_prim = stage.GetPrimAtPath(render_prim_path)
    if not render_prim.IsValid():
        raise ValueError(f"Render prim at path {render_prim_path} does not exist.")
    proxy_prim = stage.GetPrimAtPath(proxy_prim_path)
    if not proxy_prim.IsValid():
        raise ValueError(f"Proxy prim at path {proxy_prim_path} does not exist.")
    render_proxy_rel = render_prim.GetRelationship("renderProxy")
    if not render_proxy_rel:
        render_proxy_rel = render_prim.CreateRelationship("renderProxy")
    render_proxy_rel.SetTargets([proxy_prim.GetPath()])
    proxy_prim_rel = proxy_prim.GetRelationship("proxyPrim")
    if not proxy_prim_rel:
        proxy_prim_rel = proxy_prim.CreateRelationship("proxyPrim")
    proxy_prim_rel.SetTargets([render_prim.GetPath()])


def compute_local_bounds_with_hierarchy(prim: Usd.Prim, purposes: List[str]) -> Gf.BBox3d:
    """
    Compute the local bounds of a prim and its children.

    Args:
        prim (Usd.Prim): The root prim to compute bounds for.
        purposes (List[str]): The purposes to consider when computing bounds.

    Returns:
        Gf.BBox3d: The computed local bounds.
    """
    if not purposes:
        raise ValueError("At least one purpose must be provided.")
    bbox = Gf.BBox3d()
    for descendant in Usd.PrimRange(prim):
        imageable = UsdGeom.Imageable(descendant)
        if not imageable:
            continue
        local_bound = imageable.ComputeLocalBound(Usd.TimeCode.Default(), *purposes)
        bbox = Gf.BBox3d.Combine(bbox, local_bound)
    return bbox


def batch_set_purpose(prims: List[Usd.Prim], purpose: str) -> None:
    """Set the purpose attribute on a list of prims.

    Args:
        prims (List[Usd.Prim]): The list of prims to set the purpose on.
        purpose (str): The purpose value to set. Must be one of the allowed values.

    Raises:
        ValueError: If an invalid purpose value is provided.
    """
    allowed_purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide]
    if purpose not in allowed_purposes:
        raise ValueError(f"Invalid purpose value '{purpose}'. Must be one of {allowed_purposes}")
    for prim in prims:
        if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
            imageable = UsdGeom.Imageable(prim)
            purpose_attr = imageable.GetPurposeAttr()
            purpose_attr.Set(purpose)


def create_test_prims(stage):
    """Create test prims for the batch_set_purpose function."""
    world_prim = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world_prim)
    stage.DefinePrim("/World/Prim1", "Cube")
    stage.DefinePrim("/World/Prim2", "Sphere")
    stage.DefinePrim("/World/Prim3", "Cylinder")
    return [stage.GetPrimAtPath(p) for p in ["/World/Prim1", "/World/Prim2", "/World/Prim3"]]


def compute_world_bounds_for_purposes(prim: UsdGeom.Imageable, time: Usd.TimeCode, *purposes: str) -> Gf.BBox3d:
    """Compute the world space bounds of a prim for the specified purposes.

    Args:
        prim (UsdGeom.Imageable): The imageable prim to compute bounds for.
        time (Usd.TimeCode): The time at which to compute the bounds.
        *purposes (str): One or more purpose values to consider when computing the bounds.

    Returns:
        Gf.BBox3d: The world space bounds of the prim for the specified purposes.

    Raises:
        ValueError: If no purposes are specified or if the prim is invalid.
    """
    if not prim.GetPrim().IsValid():
        raise ValueError("Cannot compute bounds for an invalid prim.")
    if not purposes:
        raise ValueError("Must specify at least one purpose value.")
    bounds = prim.ComputeWorldBound(time, *purposes)
    return bounds


def compute_transform_and_bound_with_hierarchy(prim: Usd.Prim, time: Usd.TimeCode) -> Tuple[Gf.Matrix4d, Gf.BBox3d]:
    """
    Recursively compute the local-to-world transform and bound for a prim and its descendants.

    Args:
        prim (Usd.Prim): The prim to compute the transform and bound for.
        time (Usd.TimeCode): The time at which to compute the transform and bound.

    Returns:
        A tuple containing the local-to-world transform matrix and the world-space bound.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        raise ValueError(f"Prim {prim.GetPath()} is not an Imageable")
    local_transform = imageable.ComputeLocalToWorldTransform(time)
    bound = Gf.BBox3d()
    for child in prim.GetChildren():
        (child_transform, child_bound) = compute_transform_and_bound_with_hierarchy(child, time)
        if not child_bound.GetRange().IsEmpty():
            child_bound.Transform(child_transform)
            bound = Gf.BBox3d.Combine(bound, child_bound)
    if imageable.GetPrim().IsA(UsdGeom.Boundable):
        local_bound = UsdGeom.Boundable(imageable).ComputeLocalBound(time, "default")
        if not local_bound.GetRange().IsEmpty():
            local_bound.Transform(local_transform)
            bound = Gf.BBox3d.Combine(bound, local_bound)
    return (local_transform, bound)


def create_test_scene():
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/root")
    child1 = UsdGeom.Xform.Define(stage, "/root/child1")
    add_translate_op(child1).Set((1, 0, 0))
    grandchild1 = UsdGeom.Cube.Define(stage, "/root/child1/grandchild1")
    grandchild1.GetSizeAttr().Set(2)
    child2 = UsdGeom.Xform.Define(stage, "/root/child2")
    add_translate_op(child2).Set((0, 1, 0))
    grandchild2 = UsdGeom.Sphere.Define(stage, "/root/child2/grandchild2")
    grandchild2.GetRadiusAttr().Set(1.5)
    return stage


def compute_and_cache_world_bounds(stage: Usd.Stage, purposes: List[str], time: Usd.TimeCode) -> Gf.Range3d:
    """
    Compute and cache the world bounds for a given stage, purposes, and time.

    Args:
        stage (Usd.Stage): The USD stage to compute bounds for.
        purposes (List[str]): The purposes to consider when computing bounds.
        time (Usd.TimeCode): The time at which to compute the bounds.

    Returns:
        Gf.Range3d: The computed world bounds.
    """
    bbox_cache = UsdGeom.BBoxCache(time, includedPurposes=purposes)
    root_prim = stage.GetPseudoRoot()
    bounds = bbox_cache.ComputeWorldBound(root_prim)
    if bounds.GetRange().IsEmpty():
        return Gf.Range3d()
    return bounds.ComputeAlignedRange()


def test_compute_and_cache_world_bounds():
    stage = Usd.Stage.CreateInMemory()
    sphere = UsdGeom.Sphere.Define(stage, "/sphere")
    cube = UsdGeom.Cube.Define(stage, "/cube")
    add_translate_op(cube).Set((5, 0, 0))
    default_purpose = UsdGeom.Tokens.default_
    default_bounds = compute_and_cache_world_bounds(stage, [default_purpose], Usd.TimeCode.Default())
    print(f"Default bounds: {default_bounds}")
    render_purpose = UsdGeom.Tokens.render
    render_bounds = compute_and_cache_world_bounds(stage, [render_purpose], Usd.TimeCode.Default())
    print(f"Render bounds: {render_bounds}")


def get_bound_and_transform(prim: UsdGeom.Gprim, time_code: Usd.TimeCode) -> Tuple[Gf.BBox3d, Gf.Matrix4d]:
    """
    Get the bound and transform for a given prim at a specific time.

    Args:
        prim (UsdGeom.Gprim): The prim to get the bound and transform for.
        time_code (Usd.TimeCode): The time code to evaluate the bound and transform at.

    Returns:
        Tuple[Gf.BBox3d, Gf.Matrix4d]: A tuple containing the bound and transform.
    """
    if not prim.GetPrim().IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPrim().GetPath()}")
    bound = prim.ComputeLocalBound(time_code, purpose1="default")
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time_code)
    return (bound, transform)


def compute_visibility_over_time(prim: Usd.Prim, time_range: Tuple[float, float], num_samples: int) -> List[str]:
    """Compute the visibility of a prim over a range of time samples.

    Args:
        prim (Usd.Prim): The prim to compute visibility for.
        time_range (Tuple[float, float]): The range of time to sample over (start, end).
        num_samples (int): The number of samples to take over the time range.

    Returns:
        List[str]: The visibility values at each time sample.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if num_samples < 2:
        raise ValueError("Must have at least 2 samples.")
    (start_time, end_time) = time_range
    if end_time <= start_time:
        raise ValueError("End time must be greater than start time.")
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        raise ValueError("Prim is not an Imageable.")
    
    step = (end_time - start_time) / (num_samples - 1)
    time_samples = [Usd.TimeCode(start_time + i * step) for i in range(num_samples)]
    
    visibilities = []
    for time in time_samples:
        visibility = imageable.ComputeVisibility(time)
        visibilities.append(visibility)
    return visibilities


def compute_combined_transform_and_bound(
    prim: UsdGeom.Imageable, time_code: Usd.TimeCode
) -> Tuple[Gf.Matrix4d, Gf.BBox3d]:
    """
    Compute the combined transform and bound of an imageable prim.

    Args:
        prim (UsdGeom.Imageable): The input imageable prim.
        time_code (Usd.TimeCode): The time code at which to compute the transform and bound.

    Returns:
        Tuple[Gf.Matrix4d, Gf.BBox3d]: A tuple containing the combined transform matrix and
        the bound in world space.

    Raises:
        ValueError: If the input prim is not a valid imageable prim.
    """
    if not prim:
        raise ValueError("Input prim is not a valid imageable prim.")
    usd_prim = prim.GetPrim()
    local_transform = UsdGeom.Xformable(usd_prim).ComputeLocalToWorldTransform(time_code)
    untransformed_bound = UsdGeom.Boundable(usd_prim).ComputeUntransformedBound(time_code, purpose1="default")
    world_bound = Gf.BBox3d()
    if untransformed_bound.GetRange().IsEmpty():
        world_bound = Gf.BBox3d()
    else:
        world_bound = untransformed_bound.Transform(local_transform)
    return (local_transform, world_bound)


def create_test_prim(stage):
    """Create a test prim with a cube geometry."""
    prim_path = "/Root"
    prim = stage.DefinePrim(prim_path, "Xform")
    mesh = UsdGeom.Cube.Define(stage, f"{prim_path}/Cube")
    mesh.GetSizeAttr().Set(10)
    return UsdGeom.Imageable(prim)


def adjust_purpose_based_on_hierarchy(prim: Usd.Prim) -> None:
    """Adjust the purpose of the prim based on its hierarchy."""
    if not prim.GetParent():
        return
    parent_imageable = UsdGeom.Imageable(prim.GetParent())
    if parent_imageable.GetPurposeAttr().Get() == UsdGeom.Tokens.guide:
        imageable = UsdGeom.Imageable(prim)
        imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
        return
    if parent_imageable.GetPurposeAttr().Get() == UsdGeom.Tokens.proxy:
        imageable = UsdGeom.Imageable(prim)
        imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.proxy)
        return
    if parent_imageable.GetPurposeAttr().Get() == UsdGeom.Tokens.render:
        return
    if parent_imageable.GetPurposeAttr().Get() == UsdGeom.Tokens.default_:
        return


def toggle_visibility(prim: UsdGeom.Imageable, time_code: Usd.TimeCode = Usd.TimeCode.Default()):
    """Toggle the visibility of a prim at a specific time code.

    If the prim is invisible, it will be made visible. If it is visible, it will be made invisible.
    """
    if not prim.GetPrim().IsValid():
        raise ValueError("Invalid prim.")
    vis_attr = prim.GetVisibilityAttr()
    if vis_attr.HasAuthoredValue():
        vis = vis_attr.Get(time_code)
        if vis == UsdGeom.Tokens.invisible:
            prim.MakeVisible(time_code)
        else:
            prim.MakeInvisible(time_code)
    else:
        prim.MakeInvisible(time_code)


def filter_prims_by_purpose(prims: List[Usd.Prim], purpose: UsdGeom.Tokens) -> List[Usd.Prim]:
    """Filter a list of prims by their purpose.

    Args:
        prims (List[Usd.Prim]): A list of prims to filter.
        purpose (UsdGeom.Tokens): The purpose to filter by.

    Returns:
        List[Usd.Prim]: A list of prims that match the specified purpose.
    """
    filtered_prims: List[Usd.Prim] = []
    for prim in prims:
        if not prim.IsValid():
            continue
        prim_purpose = UsdGeom.Imageable(prim).GetPurposeAttr().Get()
        if prim_purpose == purpose:
            filtered_prims.append(prim)
    return filtered_prims


def gather_purpose_info(prim: Usd.Prim) -> Optional[UsdGeom.Imageable.PurposeInfo]:
    """Gather purpose info for a prim if it is imageable.

    Args:
        prim (Usd.Prim): The prim to gather purpose info for.

    Returns:
        Optional[UsdGeom.Imageable.PurposeInfo]: The purpose info if the prim is imageable, None otherwise.
    """
    if not prim.IsValid():
        return None
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        return None
    purpose = imageable.ComputePurpose()
    inheritable_purpose = UsdGeom.Imageable.PurposeInfo.GetInheritablePurpose()
    is_inheritable = purpose == inheritable_purpose
    purpose_info = UsdGeom.Imageable.PurposeInfo()
    purpose_info.purpose = purpose
    purpose_info.isInheritable = is_inheritable
    return purpose_info


def convert_linear_units(src_units: UsdGeom.LinearUnits, dst_units: UsdGeom.LinearUnits, value: float) -> float:
    """Convert a value from one linear unit to another.

    Args:
        src_units (UsdGeom.LinearUnits): The source linear units.
        dst_units (UsdGeom.LinearUnits): The destination linear units.
        value (float): The value to convert.

    Returns:
        float: The converted value in the destination units.
    """
    if src_units == UsdGeom.LinearUnits.meters:
        meters_value = value
    elif src_units == UsdGeom.LinearUnits.centimeters:
        meters_value = value * 0.01
    elif src_units == UsdGeom.LinearUnits.feet:
        meters_value = value * 0.3048
    elif src_units == UsdGeom.LinearUnits.inches:
        meters_value = value * 0.0254
    elif src_units == UsdGeom.LinearUnits.yards:
        meters_value = value * 0.9144
    else:
        raise ValueError(f"Unsupported source unit: {src_units}")
    if dst_units == UsdGeom.LinearUnits.meters:
        return meters_value
    elif dst_units == UsdGeom.LinearUnits.centimeters:
        return meters_value / 0.01
    elif dst_units == UsdGeom.LinearUnits.feet:
        return meters_value / 0.3048
    elif dst_units == UsdGeom.LinearUnits.inches:
        return meters_value / 0.0254
    elif dst_units == UsdGeom.LinearUnits.yards:
        return meters_value / 0.9144
    else:
        raise ValueError(f"Unsupported destination unit: {dst_units}")


def validate_linear_units_consistency(stage: Usd.Stage) -> bool:
    """Validate that the stage has consistent linear units."""
    root_layer = stage.GetRootLayer()
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        default_prim = stage.GetPseudoRoot()
    meters_per_unit_attr = default_prim.GetAttribute("metersPerUnit")
    if not meters_per_unit_attr.IsValid():
        return True
    meters_per_unit = meters_per_unit_attr.Get()
    for prim in Usd.PrimRange(default_prim):
        if not prim.HasAuthoredReferences():
            continue
        for reference in prim.GetReferences():
            referenced_layer = reference.GetAssetPath()
            referenced_stage = Usd.Stage.Open(referenced_layer)
            referenced_meters_per_unit_attr = referenced_stage.GetDefaultPrim().GetAttribute("metersPerUnit")
            if not referenced_meters_per_unit_attr.IsValid():
                continue
            referenced_meters_per_unit = referenced_meters_per_unit_attr.Get()
            if not Gf.IsClose(meters_per_unit, referenced_meters_per_unit, 1e-06):
                return False
    return True


def set_stage_linear_units(stage: Usd.Stage, linear_units: UsdGeom.LinearUnits) -> None:
    """Set the linear units of a USD stage.

    Args:
        stage (Usd.Stage): The USD stage to set the linear units on.
        linear_units (UsdGeom.LinearUnits): The linear units to set.

    Raises:
        ValueError: If the stage is invalid.
    """
    if not stage:
        raise ValueError("Invalid stage. Please provide a valid USD stage.")
    UsdGeom.SetStageMetersPerUnit(stage, linear_units)
    stage.Save()


def get_prim_linear_units(prim: Usd.Prim) -> Optional[UsdGeom.LinearUnits]:
    """Get the linear units for a prim.

    Args:
        prim (Usd.Prim): The prim to get the linear units for.

    Returns:
        Optional[UsdGeom.LinearUnits]: The linear units for the prim, or None if not set.
    """
    meters_per_unit_attr = prim.GetAttribute("metersPerUnit")
    if meters_per_unit_attr.IsValid() and meters_per_unit_attr.HasValue():
        meters_per_unit = meters_per_unit_attr.Get()
        for units_name in dir(UsdGeom.LinearUnits):
            units = getattr(UsdGeom.LinearUnits, units_name)
            if isinstance(units, UsdGeom.LinearUnits):
                if UsdGeom.ConvertLinearToMeters(1.0, units) == meters_per_unit:
                    return units
    return None


def get_model_draw_mode(prim: Usd.Prim, parent_draw_mode: str = "") -> str:
    """Get the effective model:drawMode of this prim."""
    drawMode = prim.GetAttribute("model:drawMode")
    if drawMode and drawMode.IsValid():
        authored_draw_mode = drawMode.Get(Usd.TimeCode.Default())
        if authored_draw_mode and authored_draw_mode != UsdGeom.Tokens.inherited:
            return authored_draw_mode
    if parent_draw_mode:
        return parent_draw_mode
    parent = prim.GetParent()
    while parent:
        parent_draw_mode = get_model_draw_mode(parent)
        if parent_draw_mode:
            return parent_draw_mode
        parent = parent.GetParent()
    return UsdGeom.Tokens.default_


def set_model_draw_mode(prim: Usd.Prim, draw_mode: str):
    """Set the model:drawMode attribute on a prim."""
    prim.CreateAttribute("model:drawMode", Sdf.ValueTypeNames.Token).Set(draw_mode)


def get_model_apply_draw_mode(prim: Usd.Prim) -> bool:
    """Get the value of the model:applyDrawMode attribute for a prim."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    model_api = UsdGeom.ModelAPI(prim)
    if not model_api:
        raise ValueError(f"Prim {prim.GetPath()} does not have the ModelAPI applied.")
    apply_draw_mode_attr = model_api.GetModelApplyDrawModeAttr()
    if not apply_draw_mode_attr:
        return False
    apply_draw_mode = apply_draw_mode_attr.Get(Usd.TimeCode.Default())
    if apply_draw_mode is None:
        return False
    return apply_draw_mode


def compute_hierarchical_extents(model_prim: Usd.Prim, time: Usd.TimeCode = Usd.TimeCode.Default()) -> Gf.Range3d:
    """Computes the hierarchical extents of a model prim.

    Args:
        model_prim (Usd.Prim): The model prim to compute extents for.
        time (Usd.TimeCode, optional): The time at which to compute extents. Defaults to Default time.

    Returns:
        Gf.Range3d: The computed hierarchical extents.

    Raises:
        ValueError: If the input prim is not a valid model prim.
    """
    if not model_prim.IsValid() or not model_prim.IsModel():
        raise ValueError(f"Input prim '{model_prim.GetPath()}' is not a valid model prim.")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    range_3d = bbox_cache.ComputeUntransformedBound(model_prim).ComputeAlignedBox()
    if range_3d.IsEmpty():
        return Gf.Range3d()
    return range_3d


def test_compute_hierarchical_extents():
    stage = Usd.Stage.CreateInMemory()
    model_prim = stage.DefinePrim("/Model", "Xform")
    model_prim.SetMetadata("kind", "component")
    sphere_prim = stage.DefinePrim("/Model/Sphere", "Sphere")
    sphere_prim.GetAttribute("radius").Set(2.0)
    cube_prim = stage.DefinePrim("/Model/Cube", "Cube")
    cube_prim.GetAttribute("size").Set(4.0)
    extents = compute_hierarchical_extents(model_prim)
    print(extents)


def batch_get_draw_mode(stage: Usd.Stage, prim_paths: List[str]) -> Dict[str, str]:
    """
    Get the computed model:drawMode for a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths.

    Returns:
        Dict[str, str]: Dictionary mapping prim paths to their computed model:drawMode.
    """
    results = {}
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        model_api = UsdGeom.ModelAPI(prim)
        if not model_api:
            continue
        draw_mode = model_api.ComputeModelDrawMode("default")
        results[prim_path] = draw_mode
    return results


def toggle_model_apply_draw_mode(prim: Usd.Prim) -> bool:
    """Toggle the model:applyDrawMode attribute on the given prim."""
    model_api = UsdGeom.ModelAPI(prim)
    if not model_api:
        raise ValueError(f"Prim at path {prim.GetPath()} does not have UsdGeomModelAPI applied.")
    apply_draw_mode_attr = model_api.GetModelApplyDrawModeAttr()
    if not apply_draw_mode_attr:
        apply_draw_mode_attr = model_api.CreateModelApplyDrawModeAttr(False, True)
    current_value = apply_draw_mode_attr.Get()
    new_value = not current_value
    success = apply_draw_mode_attr.Set(new_value)
    return success


def get_draw_mode_color(model_api: UsdGeom.ModelAPI) -> Gf.Vec3f:
    """Get the draw mode color for a ModelAPI prim."""
    attr = model_api.GetModelDrawModeColorAttr()
    if not attr.IsValid():
        return Gf.Vec3f(0.18, 0.18, 0.18)
    authored_value = attr.Get()
    if authored_value is not None:
        return authored_value
    else:
        return attr.GetFallbackValue()


def batch_set_draw_mode(stage: Usd.Stage, prim_paths: List[str], draw_mode: str) -> None:
    """Batch set the model:drawMode attribute for multiple prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths to set the draw mode for.
        draw_mode (str): The draw mode to set. Must be one of the allowed values.

    Raises:
        ValueError: If an invalid draw mode is provided.
    """
    allowed_draw_modes = ["origin", "bounds", "cards", "default", "inherited"]
    if draw_mode not in allowed_draw_modes:
        raise ValueError(f"Invalid draw mode '{draw_mode}'. Must be one of {allowed_draw_modes}.")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        model_api = UsdGeom.ModelAPI(prim)
        model_api.CreateModelDrawModeAttr(draw_mode)


def get_constraint_target_names(model_api: UsdGeom.ModelAPI) -> List[str]:
    """
    Get the names of all constraint targets on the given ModelAPI prim.

    Args:
        model_api (UsdGeom.ModelAPI): The ModelAPI prim to query.

    Returns:
        List[str]: The names of all constraint targets on the prim.
    """
    constraint_targets = model_api.GetConstraintTargets()
    constraint_target_names = [target.GetName() for target in constraint_targets if target.IsValid()]
    return constraint_target_names


def set_draw_mode_color(prim: Usd.Prim, color: Tuple[float, float, float]) -> None:
    """Set the draw mode color for a prim.

    Args:
        prim (Usd.Prim): The prim to set the draw mode color for.
        color (Tuple[float, float, float]): The RGB color as a tuple of 3 floats.

    Raises:
        ValueError: If the input prim is not valid or does not have a ModelAPI applied.
    """
    if not prim.IsValid():
        raise ValueError("Input prim is not valid")
    model_api = UsdGeom.ModelAPI(prim)
    if not model_api:
        raise ValueError("Prim does not have a ModelAPI applied")
    draw_mode_color_attr = model_api.GetModelDrawModeColorAttr()
    draw_mode_color_attr.Set(Gf.Vec3f(color))


def compute_draw_mode_for_hierarchy(prim: Usd.Prim, parentDrawMode: str = "") -> str:
    """Compute the effective model:drawMode for the given prim hierarchy."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    model_api = UsdGeom.ModelAPI(prim)
    if model_api.GetModelDrawModeAttr().HasAuthoredValue():
        return model_api.GetModelDrawModeAttr().Get()
    elif parentDrawMode:
        return parentDrawMode
    else:
        parent_prim = prim.GetParent()
        while parent_prim.IsValid():
            parent_model_api = UsdGeom.ModelAPI(parent_prim)
            if parent_model_api.GetModelDrawModeAttr().HasAuthoredValue():
                return parent_model_api.GetModelDrawModeAttr().Get()
            parent_prim = parent_prim.GetParent()
    return "default"


def set_draw_mode_for_model_hierarchy(prim: Usd.Prim, draw_mode: str) -> None:
    """Set the model:drawMode for a model hierarchy.

    Args:
        prim (Usd.Prim): The root prim of the model hierarchy.
        draw_mode (str): The draw mode to set ("origin", "bounds", "cards", "default").

    Raises:
        ValueError: If the input prim is not a valid model hierarchy root.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    model = Usd.ModelAPI(prim)
    if not model:
        raise ValueError(f"Prim {prim.GetPath()} is not a model")
    model_attr = UsdGeom.Tokens.modelDrawMode
    prim.CreateAttribute(model_attr, Sdf.ValueTypeNames.Token).Set(draw_mode)
    for child in prim.GetAllChildren():
        if child.IsA(UsdGeom.Imageable):
            apply_attr = UsdGeom.Tokens.modelApplyDrawMode
            child.CreateAttribute(apply_attr, Sdf.ValueTypeNames.Bool).Set(True)


def remove_model_card_textures(prim: Usd.Prim) -> None:
    """Remove all model card texture attributes from the given prim."""
    model_api = UsdGeom.ModelAPI(prim)
    if not prim.IsValid() or not model_api:
        raise ValueError(f"Prim {prim.GetPath()} is not valid or does not have a ModelAPI applied.")
    attrs = [
        model_api.GetModelCardTextureXPosAttr(),
        model_api.GetModelCardTextureXNegAttr(),
        model_api.GetModelCardTextureYPosAttr(),
        model_api.GetModelCardTextureYNegAttr(),
        model_api.GetModelCardTextureZPosAttr(),
        model_api.GetModelCardTextureZNegAttr(),
    ]
    for attr in attrs:
        if attr.HasAuthoredValue():
            attr.Clear()


def create_card_textures(prim: Usd.Prim, texture_paths: Dict[str, str]) -> None:
    """Create card textures on a prim using the ModelAPI schema.

    Args:
        prim (Usd.Prim): The prim to add the card textures to.
        texture_paths (Dict[str, str]): A dictionary mapping card texture
            attributes to file paths. Valid keys are 'XNeg', 'XPos', 'YNeg',
            'YPos', 'ZNeg', 'ZPos'.

    Raises:
        ValueError: If prim is not a valid prim.
        ValueError: If any of the texture paths are invalid.
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid")
    model = UsdGeom.ModelAPI(prim)
    for key, path in texture_paths.items():
        if not Sdf.Layer.IsAnonymousLayerIdentifier(path):
            try:
                Sdf.AssetPath(path)
            except Sdf.ErrorException:
                raise ValueError(f"Invalid texture path: {path}")
        if key == "XNeg":
            attr = model.CreateModelCardTextureXNegAttr()
        elif key == "XPos":
            attr = model.CreateModelCardTextureXPosAttr()
        elif key == "YNeg":
            attr = model.CreateModelCardTextureYNegAttr()
        elif key == "YPos":
            attr = model.CreateModelCardTextureYPosAttr()
        elif key == "ZNeg":
            attr = model.CreateModelCardTextureZNegAttr()
        elif key == "ZPos":
            attr = model.CreateModelCardTextureZPosAttr()
        else:
            raise ValueError(f"Invalid texture key: {key}")
        attr.Set(path)


def apply_draw_mode_to_hierarchy(prim: Usd.Prim, draw_mode: str) -> None:
    """Apply the given draw mode to the prim and its descendants."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    model_api = UsdGeom.ModelAPI(prim)
    model_api.CreateModelDrawModeAttr(draw_mode)
    model_api.CreateModelApplyDrawModeAttr(True)
    for child in prim.GetChildren():
        apply_draw_mode_to_hierarchy(child, draw_mode)


def print_draw_mode_info(prim):
    model_api = UsdGeom.ModelAPI(prim)
    draw_mode = model_api.GetModelDrawModeAttr().Get()
    apply_draw_mode = model_api.GetModelApplyDrawModeAttr().Get()
    print(f"Prim: {prim.GetPath()}")
    print(f"  model:drawMode = {draw_mode}")
    print(f"  model:applyDrawMode = {apply_draw_mode}")


def get_model_card_textures(prim: Usd.Prim) -> Dict[str, str]:
    """Get the model card textures for a prim.

    Args:
        prim (Usd.Prim): The prim to get the model card textures for.

    Returns:
        Dict[str, str]: A dictionary mapping the card texture orientation to the texture path.

    Raises:
        ValueError: If the prim is not a valid UsdGeomModelAPI prim.
    """
    model_api = UsdGeom.ModelAPI(prim)
    if not model_api:
        raise ValueError(f"Prim {prim.GetPath()} is not a valid UsdGeomModelAPI prim.")
    card_texture_attrs = {
        "x_neg": model_api.GetModelCardTextureXNegAttr(),
        "x_pos": model_api.GetModelCardTextureXPosAttr(),
        "y_neg": model_api.GetModelCardTextureYNegAttr(),
        "y_pos": model_api.GetModelCardTextureYPosAttr(),
        "z_neg": model_api.GetModelCardTextureZNegAttr(),
        "z_pos": model_api.GetModelCardTextureZPosAttr(),
    }
    card_textures = {}
    for orientation, attr in card_texture_attrs.items():
        if attr.HasValue():
            texture_path = attr.Get().path
            card_textures[orientation] = texture_path
    return card_textures


def apply_draw_mode_hierarchy(prim: Usd.Prim, parent_draw_mode: str = "") -> None:
    """Recursively apply the model:drawMode to the prim hierarchy.

    If a prim has a non-empty model:drawMode, it will be applied to all its
    descendants until another descendant with a non-empty model:drawMode is
    encountered.

    Args:
        prim (Usd.Prim): The starting prim to apply the draw mode.
        parent_draw_mode (str, optional): The draw mode of the parent prim.
            Defaults to empty string.
    """
    model_api = UsdGeom.ModelAPI(prim)
    draw_mode = model_api.ComputeModelDrawMode(parent_draw_mode)
    if draw_mode and draw_mode != "inherited":
        model_api.CreateModelApplyDrawModeAttr().Set(True)
    else:
        model_api.CreateModelApplyDrawModeAttr().Set(False)
    for child in prim.GetChildren():
        apply_draw_mode_hierarchy(child, draw_mode)


def test_apply_draw_mode_hierarchy():
    stage = Usd.Stage.CreateInMemory()
    root_prim = stage.DefinePrim("/Root", "Xform")
    child1_prim = stage.DefinePrim("/Root/Child1", "Xform")
    child2_prim = stage.DefinePrim("/Root/Child2", "Xform")
    grandchild_prim = stage.DefinePrim("/Root/Child1/Grandchild", "Xform")
    UsdGeom.ModelAPI(root_prim).CreateModelDrawModeAttr().Set("cards")
    UsdGeom.ModelAPI(child2_prim).CreateModelDrawModeAttr().Set("bounds")
    UsdGeom.ModelAPI(grandchild_prim).CreateModelDrawModeAttr().Set("origin")
    apply_draw_mode_hierarchy(root_prim)
    assert UsdGeom.ModelAPI(root_prim).GetModelApplyDrawModeAttr().Get() == True
    assert UsdGeom.ModelAPI(child1_prim).GetModelApplyDrawModeAttr().Get() == True
    assert UsdGeom.ModelAPI(child2_prim).GetModelApplyDrawModeAttr().Get() == True
    assert UsdGeom.ModelAPI(grandchild_prim).GetModelApplyDrawModeAttr().Get() == True


def get_model_card_geometry(prim: Usd.Prim) -> str:
    """Get the model:cardGeometry attribute value for a prim."""
    model_api = UsdGeom.ModelAPI(prim)
    if model_api.GetModelCardGeometryAttr().HasAuthoredValue():
        card_geometry = model_api.GetModelCardGeometryAttr().Get()
        allowed_values = ["cross", "box", "fromTexture"]
        if card_geometry in allowed_values:
            return card_geometry
        else:
            print(f"Warning: Invalid model:cardGeometry value '{card_geometry}', using default 'cross'")
            return "cross"
    else:
        return "cross"


def compute_hierarchical_draw_modes(prim: Usd.Prim) -> Dict[Usd.Prim, str]:
    """Compute the hierarchical draw modes for a prim and its descendants."""
    result: Dict[Usd.Prim, str] = {}
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")

    def _traverse(current_prim: Usd.Prim, parent_draw_mode: str):
        model_api = UsdGeom.ModelAPI(current_prim)
        draw_mode: str = model_api.ComputeModelDrawMode(parent_draw_mode)
        result[current_prim] = draw_mode
        for child in current_prim.GetChildren():
            _traverse(child, draw_mode)

    _traverse(prim, "")
    return result


def set_model_card_geometry_mode(prim: Usd.Prim, mode: str) -> None:
    """Set the model:cardGeometry attribute on the given prim to the specified mode.

    Args:
        prim (Usd.Prim): The prim to set the model:cardGeometry attribute on.
        mode (str): The model:cardGeometry mode to set. Must be one of "cross", "box", or "fromTexture".

    Raises:
        ValueError: If the given prim is not valid or if an invalid mode is specified.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if mode not in ["cross", "box", "fromTexture"]:
        raise ValueError(f"Invalid model:cardGeometry mode '{mode}'. Must be one of 'cross', 'box', or 'fromTexture'.")
    model_api = UsdGeom.ModelAPI(prim)
    card_geometry_attr = model_api.GetModelCardGeometryAttr()
    if not card_geometry_attr:
        card_geometry_attr = model_api.CreateModelCardGeometryAttr()
    card_geometry_attr.Set(mode)


def create_motion_api_attributes(prim: Usd.Prim) -> None:
    """Create motion API attributes on the given prim."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if not UsdGeom.MotionAPI.CanApply(prim):
        raise ValueError(f"MotionAPI schema cannot be applied to prim: {prim}")
    motion_api = UsdGeom.MotionAPI.Apply(prim)
    blur_scale_attr = motion_api.CreateMotionBlurScaleAttr(1.0, writeSparsely=False)
    blur_scale_attr.Set(0.5)
    sample_count_attr = motion_api.CreateNonlinearSampleCountAttr(3, writeSparsely=False)
    sample_count_attr.Set(5)
    velocity_scale_attr = motion_api.CreateVelocityScaleAttr(1.0, writeSparsely=False)
    velocity_scale_attr.Set(2.0)


def apply_and_set_motion_api_attributes(
    prim: Usd.Prim, blur_scale: float, nonlinear_sample_count: int, velocity_scale: float
) -> None:
    """Apply UsdGeomMotionAPI to a prim and set its attributes.

    Args:
        prim (Usd.Prim): The prim to apply the UsdGeomMotionAPI to.
        blur_scale (float): The value to set for the motion:blurScale attribute.
        nonlinear_sample_count (int): The value to set for the motion:nonlinearSampleCount attribute.
        velocity_scale (float): The value to set for the motion:velocityScale attribute.

    Raises:
        ValueError: If the provided prim is not valid.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim provided.")
    motion_api = UsdGeom.MotionAPI.Apply(prim)
    blur_scale_attr = motion_api.CreateMotionBlurScaleAttr(blur_scale, writeSparsely=True)
    nonlinear_sample_count_attr = motion_api.CreateNonlinearSampleCountAttr(nonlinear_sample_count, writeSparsely=True)
    velocity_scale_attr = motion_api.CreateVelocityScaleAttr(velocity_scale, writeSparsely=True)


def compute_inherited_motion_blur_scale(prim: Usd.Prim, time_code: Usd.TimeCode = Usd.TimeCode.Default()) -> float:
    """
    Compute the inherited value of motion:blurScale at the given time code.

    This function traverses the prim's ancestors in namespace until it finds an authored value
    for the motion:blurScale attribute. If no authored value is found, it returns the default value of 1.0.

    Parameters:
        prim (Usd.Prim): The prim to compute the inherited motion:blurScale value for.
        time_code (Usd.TimeCode): The time code at which to compute the value. Defaults to Default.

    Returns:
        float: The inherited motion:blurScale value, or 1.0 if no authored value is found.
    """
    motion_api = UsdGeom.MotionAPI(prim)
    if motion_api.GetMotionBlurScaleAttr().HasAuthoredValue():
        return motion_api.GetMotionBlurScaleAttr().Get(time_code)
    ancestor_prim = prim.GetParent()
    while ancestor_prim.IsValid():
        ancestor_motion_api = UsdGeom.MotionAPI(ancestor_prim)
        if ancestor_motion_api.GetMotionBlurScaleAttr().HasAuthoredValue():
            return ancestor_motion_api.GetMotionBlurScaleAttr().Get(time_code)
        ancestor_prim = ancestor_prim.GetParent()
    return 1.0


def set_nonlinear_sample_count(prim: Usd.Prim, sample_count: int, time_code: Usd.TimeCode = Usd.TimeCode.Default()):
    """Set the nonlinear sample count for a prim.

    Args:
        prim (Usd.Prim): The prim to set the nonlinear sample count for.
        sample_count (int): The number of samples to set.
        time_code (Usd.TimeCode, optional): The time code to set the value at. Defaults to Default time code.

    Raises:
        ValueError: If the prim is not valid or if the sample count is less than 1.
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid")
    if sample_count < 1:
        raise ValueError("Sample count must be greater than or equal to 1")
    motion_api = UsdGeom.MotionAPI(prim)
    sample_count_attr = motion_api.GetNonlinearSampleCountAttr()
    if not sample_count_attr:
        sample_count_attr = motion_api.CreateNonlinearSampleCountAttr()
    sample_count_attr.Set(sample_count, time_code)


def set_velocity_scale(prim: Usd.Prim, velocity_scale: float, time_code: Usd.TimeCode = Usd.TimeCode.Default()):
    """Set the velocity scale value for a prim with MotionAPI applied.

    Args:
        prim (Usd.Prim): The prim to set the velocity scale on.
        velocity_scale (float): The velocity scale value to set.
        time_code (Usd.TimeCode, optional): The time code to set the value at. Defaults to Default time code.

    Raises:
        ValueError: If the prim is not valid or does not have MotionAPI applied.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    motion_api = UsdGeom.MotionAPI(prim)
    if not motion_api:
        raise ValueError(f"Prim {prim.GetPath()} does not have MotionAPI applied.")
    velocity_scale_attr = motion_api.GetVelocityScaleAttr()
    if not velocity_scale_attr:
        velocity_scale_attr = motion_api.CreateVelocityScaleAttr()
    velocity_scale_attr.Set(velocity_scale, time_code)


def compute_velocity_scale_for_prims(prims: List[Usd.Prim], time: Usd.TimeCode) -> List[float]:
    """
    Computes the inherited value of motion:velocityScale for each given prim at the specified time.

    Args:
        prims (List[Usd.Prim]): The list of prims to compute velocity scale for.
        time (Usd.TimeCode): The time at which to compute the velocity scale.

    Returns:
        List[float]: The computed velocity scale for each prim in the same order as the input prims.
    """
    velocity_scales = []
    for prim in prims:
        if not prim.IsValid():
            velocity_scales.append(None)
            continue
        motion_api = UsdGeom.MotionAPI(prim)
        if not motion_api:
            velocity_scales.append(None)
            continue
        velocity_scale = motion_api.ComputeVelocityScale(time)
        velocity_scales.append(velocity_scale)
    return velocity_scales


def reset_motion_api_attributes_to_defaults(prim: Usd.Prim):
    """Reset all MotionAPI attributes on the given prim to their default values."""
    motion_api = UsdGeom.MotionAPI(prim)
    if not motion_api:
        return
    blur_scale_attr = motion_api.GetMotionBlurScaleAttr()
    if blur_scale_attr.IsAuthored():
        blur_scale_attr.Clear()
        blur_scale_attr.Set(1.0)
    nonlinear_sample_count_attr = motion_api.GetNonlinearSampleCountAttr()
    if nonlinear_sample_count_attr.IsAuthored():
        nonlinear_sample_count_attr.Clear()
        nonlinear_sample_count_attr.Set(3)
    velocity_scale_attr = motion_api.GetVelocityScaleAttr()
    if velocity_scale_attr.IsAuthored():
        velocity_scale_attr.Clear()
        velocity_scale_attr.Set(1.0)


def set_motion_blur_scale(prim: Usd.Prim, blur_scale: float, time_code: Usd.TimeCode = Usd.TimeCode.Default()) -> None:
    """Set the motion:blurScale attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to set the motion:blurScale attribute on.
        blur_scale (float): The value to set for motion:blurScale.
        time_code (Usd.TimeCode, optional): The time code to set the value at. Defaults to Default.

    Raises:
        ValueError: If the prim is not valid.
        ValueError: If the blur_scale is negative.
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid.")
    if blur_scale < 0:
        raise ValueError("blur_scale must be non-negative.")
    motion_api = UsdGeom.MotionAPI.Apply(prim)
    blur_scale_attr = motion_api.GetMotionBlurScaleAttr()
    blur_scale_attr.Set(blur_scale, time_code)


def get_nonlinear_sample_count_for_prims(
    stage: Usd.Stage, prim_paths: List[str], time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> Dict[str, int]:
    """
    Get the nonlinear sample count for a list of prims at a specific time code.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths.
        time_code (Usd.TimeCode, optional): The time code at which to retrieve the nonlinear sample count. Defaults to Usd.TimeCode.Default().

    Returns:
        Dict[str, int]: A dictionary mapping prim paths to their nonlinear sample count.
    """
    result = {}
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        motion_api = UsdGeom.MotionAPI(prim)
        if not motion_api:
            result[prim_path] = 3
        else:
            nonlinear_sample_count = motion_api.ComputeNonlinearSampleCount(time_code)
            result[prim_path] = nonlinear_sample_count
    return result


def copy_motion_api_attributes(source_prim: Usd.Prim, dest_prim: Usd.Prim) -> None:
    """Copy the motion API attributes from source_prim to dest_prim."""
    if not UsdGeom.MotionAPI(source_prim).GetPrim():
        return
    motion_api = UsdGeom.MotionAPI.Apply(dest_prim)
    source_motion_api = UsdGeom.MotionAPI(source_prim)
    if source_motion_api.GetMotionBlurScaleAttr().HasAuthoredValue():
        motion_blur_scale = source_motion_api.GetMotionBlurScaleAttr().Get()
        motion_api.CreateMotionBlurScaleAttr(motion_blur_scale, True)
    if source_motion_api.GetNonlinearSampleCountAttr().HasAuthoredValue():
        nonlinear_sample_count = source_motion_api.GetNonlinearSampleCountAttr().Get()
        motion_api.CreateNonlinearSampleCountAttr(nonlinear_sample_count, True)
    if source_motion_api.GetVelocityScaleAttr().HasAuthoredValue():
        velocity_scale = source_motion_api.GetVelocityScaleAttr().Get()
        motion_api.CreateVelocityScaleAttr(velocity_scale, True)


def remove_motion_api_from_prims(stage: Usd.Stage, prim_paths: List[str]) -> None:
    """Remove the MotionAPI from the specified prims.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (List[str]): The paths of the prims to remove the MotionAPI from.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        if not prim.HasAPI(UsdGeom.MotionAPI):
            print(f"Warning: Prim at path {prim_path} does not have the MotionAPI applied. Skipping.")
            continue
        prim.RemoveAPI(UsdGeom.MotionAPI)
        print(f"Removed MotionAPI from prim at path {prim_path}")


def deactivate_and_vis_ids(
    instancer: UsdGeom.PointInstancer, deactivate_ids: List[int], vis_ids: List[int], time_code: Usd.TimeCode
):
    """Deactivate and make visible a set of instances in a UsdGeomPointInstancer.

    Args:
        instancer (UsdGeom.PointInstancer): The point instancer prim.
        deactivate_ids (List[int]): List of instance IDs to deactivate.
        vis_ids (List[int]): List of instance IDs to make visible.
        time_code (Usd.TimeCode): The time code at which to perform the operation.

    Returns:
        bool: True if the operation succeeded, False otherwise.
    """
    if not instancer or not instancer.GetPrim().IsValid():
        return False
    if not instancer.DeactivateIds(deactivate_ids):
        return False
    if not instancer.VisIds(vis_ids, time_code):
        return False
    return True


def create_prototype_hierarchy(
    stage: Usd.Stage, prototype_paths: list[str], instancer_path: str
) -> UsdGeom.PointInstancer:
    """Create a prototype hierarchy for a PointInstancer.

    Args:
        stage (Usd.Stage): The stage to create the prototypes on.
        prototype_paths (list[str]): A list of paths to the prototype prims.
        instancer_path (str): The path to the PointInstancer prim.

    Returns:
        UsdGeom.PointInstancer: The created PointInstancer prim.
    """
    instancer = UsdGeom.PointInstancer.Define(stage, instancer_path)
    prototypes_rel = instancer.GetPrototypesRel()
    for prototype_path in prototype_paths:
        prototype_prim = stage.GetPrimAtPath(prototype_path)
        if not prototype_prim.IsValid():
            raise ValueError(f"Prototype prim at path {prototype_path} does not exist.")
        prototypes_rel.AddTarget(prototype_path)
    return instancer


def add_prototype_to_instancer(instancer_prim: Usd.Prim, prototype_prim: Usd.Prim) -> bool:
    """Add a prototype prim to a PointInstancer's prototypes relationship.

    Args:
        instancer_prim (Usd.Prim): The PointInstancer prim.
        prototype_prim (Usd.Prim): The prototype prim to add.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not instancer_prim.IsValid():
        return False
    if not UsdGeom.PointInstancer(instancer_prim):
        return False
    if not prototype_prim.IsValid():
        return False
    prototypes_rel = UsdGeom.PointInstancer(instancer_prim).GetPrototypesRel()
    targets = prototypes_rel.GetTargets()
    if prototype_prim.GetPath() in targets:
        return True
    targets.append(prototype_prim.GetPath())
    success = prototypes_rel.SetTargets(targets)
    return success


def duplicate_instancer(source_instancer: UsdGeom.PointInstancer, dest_instancer_path: str) -> UsdGeom.PointInstancer:
    """Duplicates a PointInstancer prim to a new path.

    Args:
        source_instancer (UsdGeom.PointInstancer): The source PointInstancer to duplicate.
        dest_instancer_path (str): The path for the new duplicated PointInstancer prim.

    Returns:
        UsdGeom.PointInstancer: The newly created PointInstancer prim.
    """
    if not source_instancer or not source_instancer.GetPrim().IsValid():
        raise ValueError("Invalid source PointInstancer.")
    stage = source_instancer.GetPrim().GetStage()
    dest_path = Sdf.Path(dest_instancer_path)
    if not dest_path.IsAbsolutePath() or not dest_path.IsPrimPath():
        raise ValueError("Invalid destination path. Must be an absolute prim path.")
    existing_prim = stage.GetPrimAtPath(dest_path)
    if existing_prim:
        raise ValueError(f"A prim already exists at path: {dest_instancer_path}")
    dest_instancer = UsdGeom.PointInstancer.Define(stage, dest_path)
    for attr in source_instancer.GetPrim().GetAttributes():
        attr_name = attr.GetName()
        attr_value = attr.Get()
        if attr_value is not None:
            dest_instancer.GetPrim().CreateAttribute(attr_name, attr.GetTypeName()).Set(attr_value)
    for rel in source_instancer.GetPrim().GetRelationships():
        rel_name = rel.GetName()
        rel_targets = rel.GetTargets()
        if rel_targets:
            dest_instancer.GetPrim().CreateRelationship(rel_name).SetTargets(rel_targets)
    return dest_instancer


def animate_instance_scales(
    point_instancer: UsdGeom.PointInstancer, time_samples: list[Usd.TimeCode], scale_values: list[list[Gf.Vec3f]]
) -> bool:
    """Animates the scales of instances in a UsdGeomPointInstancer.

    Args:
        point_instancer (UsdGeom.PointInstancer): The PointInstancer prim.
        time_samples (list[Usd.TimeCode]): A list of time samples.
        scale_values (list[list[Gf.Vec3f]]): A list of scale values per instance per time sample.

    Returns:
        bool: True if setting the scales was successful, False otherwise.
    """
    if not point_instancer:
        print("Error: Invalid PointInstancer")
        return False
    scales_attr = point_instancer.GetScalesAttr()
    if not scales_attr:
        print("Error: Could not get scales attribute")
        return False
    if len(time_samples) != len(scale_values):
        print("Error: Number of time samples does not match number of scale value lists")
        return False
    for i in range(len(time_samples)):
        time_code = time_samples[i]
        scales = scale_values[i]
        if len(scales) != point_instancer.GetInstanceCount(time_code):
            print(f"Error: Number of scales at time {time_code} does not match number of instances")
            return False
        if not scales_attr.Set(scales, time_code):
            print(f"Error: Could not set scales at time {time_code}")
            return False
    return True


def apply_instance_mask(mask: List[bool], array: List[float]) -> List[float]:
    """Apply a mask to an array, returning a new array with only the elements where the mask is True.

    Args:
        mask (List[bool]): A list of boolean values indicating which elements to keep.
        array (List[float]): The input array to be masked.

    Returns:
        List[float]: A new array containing only the elements where the mask is True.

    Raises:
        ValueError: If the length of the mask and array are not the same.
    """
    if len(mask) != len(array):
        raise ValueError("Mask and array must have the same length.")
    masked_array = [elem for (include, elem) in zip(mask, array) if include]
    return masked_array


def get_instance_velocities(
    point_instancer: UsdGeom.PointInstancer, time_code: Usd.TimeCode
) -> Tuple[Vt.Vec3fArray, bool]:
    """
    Get the instance velocities for a PointInstancer at a specific time.

    Args:
        point_instancer (UsdGeom.PointInstancer): The PointInstancer to get velocities for.
        time_code (Usd.TimeCode): The time at which to retrieve the velocities.

    Returns:
        A tuple containing:
        - velocities (Vt.Vec3fArray): The instance velocities. If no velocities are authored, returns an empty array.
        - velocities_authored (bool): True if velocities are authored on the PointInstancer, False otherwise.
    """
    velocities_attr = point_instancer.GetVelocitiesAttr()
    if velocities_attr.HasValue():
        velocities = velocities_attr.Get(time_code)
        if not isinstance(velocities, Vt.Vec3fArray):
            raise ValueError("Velocities attribute must be a Vec3fArray")
        return (velocities, True)
    else:
        return (Vt.Vec3fArray(), False)


def set_instance_scales(
    point_instancer: UsdGeom.PointInstancer, scales: List[Gf.Vec3f], time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> None:
    """Set the scales for each instance in a PointInstancer.

    Args:
        point_instancer (UsdGeom.PointInstancer): The PointInstancer prim.
        scales (List[Gf.Vec3f]): A list of scale vectors, one for each instance.
        time_code (Usd.TimeCode): The time code at which to set the scales. Defaults to Default.

    Raises:
        ValueError: If the length of scales does not match the number of instances.
    """
    scales_attr = point_instancer.GetScalesAttr()
    num_instances = len(point_instancer.GetProtoIndicesAttr().Get(time_code))
    if len(scales) != num_instances:
        raise ValueError(f"Number of scales ({len(scales)}) does not match number of instances ({num_instances})")
    scales_attr.Set(scales, time_code)


def get_instance_transforms(
    pointInstancer: UsdGeom.PointInstancer, time: Usd.TimeCode
) -> Tuple[bool, List[Gf.Matrix4d]]:
    """
    Get the instance transforms for a UsdGeomPointInstancer at a specific time.

    Args:
        pointInstancer (UsdGeom.PointInstancer): The point instancer to get the transforms for.
        time (Usd.TimeCode): The time to evaluate the transforms at.

    Returns:
        A tuple with two elements:
        - bool: True if the transforms were computed successfully, False otherwise.
        - List[Gf.Matrix4d]: The instance transforms. Empty if unsuccessful.
    """
    if not pointInstancer or not pointInstancer.GetPrim().IsValid():
        return (False, [])
    prototypes_rel = pointInstancer.GetPrototypesRel()
    if not prototypes_rel.GetTargets():
        return (False, [])
    proto_indices = pointInstancer.GetProtoIndicesAttr().Get(time)
    if not proto_indices:
        return (False, [])
    xforms = Vt.Matrix4dArray()
    try:
        pointInstancer.ComputeInstanceTransformsAtTime(xforms, time, time)
    except Exception as e:
        return (False, [])
    return (True, list(xforms))


def set_instance_velocities(
    point_instancer: UsdGeom.PointInstancer,
    velocities: list[Gf.Vec3f],
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Set per-instance velocities for a PointInstancer.

    Args:
        point_instancer (UsdGeom.PointInstancer): The PointInstancer prim.
        velocities (list[Gf.Vec3f]): The velocity values to set for each instance.
        time_code (Usd.TimeCode, optional): The time code at which to set the velocities.
            Defaults to Usd.TimeCode.Default().

    Raises:
        ValueError: If the length of velocities does not match the number of instances.
    """
    velocities_attr = point_instancer.GetVelocitiesAttr()
    num_instances = len(point_instancer.GetProtoIndicesAttr().Get(time_code))
    if len(velocities) != num_instances:
        raise ValueError(
            f"Number of velocities ({len(velocities)}) does not match the number of instances ({num_instances}) at time code {time_code}"
        )
    velocities_attr.Set(velocities, time_code)


def get_instance_ids(pointInstancer: UsdGeom.PointInstancer, time: Usd.TimeCode) -> Vt.IntArray:
    """Get the instance IDs for a PointInstancer at a specific time."""
    ids_attr = pointInstancer.GetIdsAttr()
    if not ids_attr.HasValue():
        return Vt.IntArray()
    ids = ids_attr.Get(time)
    if ids is None:
        return Vt.IntArray()
    return ids


def set_instance_ids(
    point_instancer: UsdGeom.PointInstancer, ids: list[int], time_code: Usd.TimeCode = Usd.TimeCode.Default()
):
    """Set the ids attribute on a PointInstancer.

    The ids array should be the same length as the protoIndices array,
    specifying the id of each instance.

    Args:
        point_instancer (UsdGeom.PointInstancer): The PointInstancer to set the ids on.
        ids (list[int]): The ids to set.
        time_code (Usd.TimeCode, optional): The time code to set the value at. Defaults to Default time.
    """
    ids_attr = point_instancer.GetIdsAttr()
    if not ids_attr:
        ids_attr = point_instancer.CreateIdsAttr()
    proto_indices = point_instancer.GetProtoIndicesAttr().Get(time_code)
    if not proto_indices:
        raise ValueError("No protoIndices defined on the PointInstancer.")
    if len(ids) != len(proto_indices):
        raise ValueError(f"Length of ids ({len(ids)}) must match length of protoIndices ({len(proto_indices)}).")
    ids_attr.Set(ids, time_code)


def get_instance_accelerations(
    point_instancer: UsdGeom.PointInstancer, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> Vt.Vec3fArray:
    """
    Get the accelerations for each instance in a PointInstancer at a specific time.

    Args:
        point_instancer (UsdGeom.PointInstancer): The PointInstancer to get the accelerations from.
        time_code (Usd.TimeCode): The time code to get the accelerations at. Defaults to Default time.

    Returns:
        Vt.Vec3fArray: The accelerations for each instance, or an empty array if no accelerations are authored.
    """
    accelerations_attr = point_instancer.GetAccelerationsAttr()
    if accelerations_attr.HasAuthoredValue():
        accelerations = accelerations_attr.Get(time_code)
        if accelerations is not None:
            return accelerations
    return Vt.Vec3fArray()


def set_instance_accelerations(
    point_instancer: UsdGeom.PointInstancer,
    accelerations: List[Gf.Vec3f],
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Set the accelerations for each instance in a PointInstancer.

    Args:
        point_instancer (UsdGeom.PointInstancer): The PointInstancer prim.
        accelerations (List[Gf.Vec3f]): A list of acceleration vectors, one per instance.
        time_code (Usd.TimeCode, optional): The timecode at which to set the accelerations. Defaults to Usd.TimeCode.Default().

    Raises:
        ValueError: If the length of accelerations does not match the number of instances.
    """
    accelerations_attr = point_instancer.GetAccelerationsAttr()
    num_instances = len(point_instancer.GetProtoIndicesAttr().Get(time_code))
    if len(accelerations) != num_instances:
        raise ValueError(
            f"Length of accelerations ({len(accelerations)}) does not match the number of instances ({num_instances})"
        )
    accelerations_attr.Set(accelerations, time_code)


def compute_instance_bboxes(point_instancer: UsdGeom.PointInstancer, time_code: Usd.TimeCode) -> Gf.Range3d:
    """Compute the bounding box for each instance in a UsdGeomPointInstancer.

    Args:
        point_instancer (UsdGeom.PointInstancer): The PointInstancer prim.
        time_code (Usd.TimeCode): The time code at which to compute the bounding boxes.

    Returns:
        Gf.Range3d: The combined bounding box of all instances.
    """
    if not point_instancer.GetPrim().IsValid():
        raise ValueError("Invalid PointInstancer prim.")
    prototypes = point_instancer.GetPrototypesRel().GetTargets()
    if not prototypes:
        raise ValueError("No prototypes found for PointInstancer.")
    instance_transforms = []
    point_instancer.ComputeInstanceTransformsAtTime(instance_transforms, time_code, time_code)
    bounds = Gf.Range3d()
    for prototype_index, prototype in enumerate(prototypes):
        prototype_prim = point_instancer.GetStage().GetPrimAtPath(prototype)
        if not prototype_prim.IsValid():
            continue
        prototype_bbox = UsdGeom.Imageable(prototype_prim).ComputeLocalBound(time_code, "default")
        proto_indices = point_instancer.GetProtoIndicesAttr().Get(time_code)
        for transform in (t for (i, t) in enumerate(instance_transforms) if proto_indices[i] == prototype_index):
            bounds.UnionWith(prototype_bbox.ComputeAlignedRange().Transform(transform))
    return bounds


def get_prototype_paths(point_instancer: UsdGeom.PointInstancer) -> list[Sdf.Path]:
    """Get the prototype paths for a point instancer."""
    prototypes_rel = point_instancer.GetPrototypesRel()
    if not prototypes_rel.IsValid():
        return []
    prototype_paths = prototypes_rel.GetTargets()
    return prototype_paths


def set_prototype_paths(point_instancer: UsdGeom.PointInstancer, prototype_paths: list[Sdf.Path]):
    """Set the prototype paths for a PointInstancer.

    Args:
        point_instancer (UsdGeom.PointInstancer): The PointInstancer prim to set prototype paths on.
        prototype_paths (list[Sdf.Path]): A list of paths to the prototype prims.
    """
    prototypes_rel = point_instancer.GetPrototypesRel()
    targets = prototypes_rel.GetTargets()
    if targets:
        prototypes_rel.ClearTargets(removeSpec=False)
    for path in prototype_paths:
        prototypes_rel.AddTarget(path)


def create_instancer_with_prototypes(
    stage: Usd.Stage,
    instancer_path: str,
    prototype_paths: List[str],
    prototype_indices: List[int],
    positions: List[Tuple[float, float, float]],
) -> UsdGeom.PointInstancer:
    """Create a PointInstancer prim with prototypes.

    Args:
        stage (Usd.Stage): The stage to create the PointInstancer on.
        instancer_path (str): The path where the PointInstancer will be created.
        prototype_paths (List[str]): A list of paths to the prototype prims.
        prototype_indices (List[int]): A list of indices corresponding to the prototypes for each instance.
        positions (List[Tuple[float, float, float]]): A list of positions for each instance.

    Returns:
        UsdGeom.PointInstancer: The created PointInstancer prim.

    Raises:
        ValueError: If the number of prototype indices does not match the number of positions.
    """
    instancer = UsdGeom.PointInstancer.Define(stage, Sdf.Path(instancer_path))
    prototypes_rel = instancer.GetPrototypesRel()
    for prototype_path in prototype_paths:
        prototypes_rel.AddTarget(Sdf.Path(prototype_path))
    proto_indices_attr = instancer.GetProtoIndicesAttr()
    if len(prototype_indices) != len(positions):
        raise ValueError("Number of prototype indices does not match the number of positions.")
    proto_indices_attr.Set(prototype_indices)
    instancer.GetPositionsAttr().Set(positions)
    return instancer


def get_instance_scales(prim: Usd.Prim, time_code: Usd.TimeCode = Usd.TimeCode.Default()) -> Vt.Vec3fArray:
    """
    Get the instance scales for a UsdGeomPointInstancer prim at a specific time.

    Args:
        prim (Usd.Prim): The UsdGeomPointInstancer prim.
        time_code (Usd.TimeCode): The time code at which to retrieve the scales. Defaults to Default.

    Returns:
        Vt.Vec3fArray: The instance scales, or an empty array if not authored.

    Raises:
        ValueError: If the input prim is not a valid UsdGeomPointInstancer.
    """
    if not prim.IsValid() or not UsdGeom.PointInstancer(prim):
        raise ValueError(f"Prim '{prim.GetPath()}' is not a valid UsdGeomPointInstancer.")
    scales_attr = UsdGeom.PointInstancer(prim).GetScalesAttr()
    if scales_attr.HasAuthoredValue():
        return scales_attr.Get(time_code)
    else:
        return Vt.Vec3fArray()


def compute_instance_transforms_over_time(xform_arrays, stage, times, prototype_paths, apply_mask=True):
    """Compute instance transforms over time.

    Args:
        xform_arrays (list[Vt.Matrix4dArray]): List of arrays to store computed transforms per time sample.
        stage (Usd.Stage): USD stage.
        times (list[Usd.TimeCode]): List of time samples.
        prototype_paths (list[Sdf.Path]): List of paths to the prototypes.
        apply_mask (bool, optional): Whether to apply masking. Defaults to True.

    Returns:
        bool: True if successful, False otherwise.
    """
    if len(times) != len(xform_arrays):
        return False
    point_instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath("/PointInstancer"))
    if not point_instancer:
        return False
    for i, time in enumerate(times):
        positions = point_instancer.GetPositionsAttr().Get(time)
        scales = point_instancer.GetScalesAttr().Get(time)
        orientations = point_instancer.GetOrientationsAttr().Get(time)
        prototype_indices = point_instancer.GetProtoIndicesAttr().Get(time)
        if not positions or not prototype_indices or len(positions) != len(prototype_indices):
            return False
        xform_arrays[i] = Vt.Matrix4dArray(len(positions))
        for j in range(len(positions)):
            prototype_index = prototype_indices[j]
            if prototype_index < 0 or prototype_index >= len(prototype_paths):
                return False
            prototype_prim = stage.GetPrimAtPath(prototype_paths[prototype_index])
            if not prototype_prim:
                return False
            prototype_xform = UsdGeom.Xformable(prototype_prim).ComputeLocalToWorldTransform(time)
            instance_xform = Gf.Matrix4d(1)
            if scales and len(scales) > j:
                instance_xform.SetScale(scales[j])
            if orientations and len(orientations) > j:
                instance_xform.SetRotate(Gf.Rotation(orientations[j]))
            instance_xform.SetTranslate(positions[j])
            xform_arrays[i][j] = prototype_xform * instance_xform
        if apply_mask:
            mask = point_instancer.ComputeMaskAtTime(time)
            if mask:
                xform_arrays[i] = UsdGeom.PointInstancer.ApplyMaskToArray(xform_arrays[i], mask)
    return True


def set_instance_positions(
    instancer_prim: UsdGeom.PointInstancer,
    positions: List[Tuple[float, float, float]],
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Set the positions of instances in a PointInstancer.

    Args:
        instancer_prim (UsdGeom.PointInstancer): The PointInstancer prim.
        positions (List[Tuple[float, float, float]]): A list of positions as tuples (x, y, z).
        time_code (Usd.TimeCode, optional): The time code at which to set the positions. Defaults to Default time.

    Raises:
        ValueError: If the instancer_prim is not a valid PointInstancer.
        ValueError: If the number of positions does not match the number of instances.
    """
    if not instancer_prim:
        raise ValueError(f"Invalid PointInstancer prim: {instancer_prim}")
    positions_attr = instancer_prim.GetPositionsAttr()
    num_instances = len(instancer_prim.GetProtoIndicesAttr().Get(time_code))
    if len(positions) != num_instances:
        raise ValueError(
            f"Number of positions ({len(positions)}) does not match the number of instances ({num_instances})"
        )
    usd_positions = Vt.Vec3fArray([Gf.Vec3f(p[0], p[1], p[2]) for p in positions])
    positions_attr.Set(usd_positions, time_code)


def set_invisible_ids(point_instancer: UsdGeom.PointInstancer, invisible_ids: List[int], time_code: Usd.TimeCode):
    """Set the invisible ids for a point instancer at a specific time.

    Args:
        point_instancer (UsdGeom.PointInstancer): The point instancer prim.
        invisible_ids (List[int]): The list of invisible ids to set.
        time_code (Usd.TimeCode): The time code to set the invisible ids at.
    """
    invisible_ids_attr = point_instancer.GetInvisibleIdsAttr()
    if not invisible_ids_attr:
        invisible_ids_attr = point_instancer.CreateInvisibleIdsAttr()
    invisible_ids_vt = Vt.Int64Array(invisible_ids)
    invisible_ids_attr.Set(invisible_ids_vt, time_code)


def merge_instancers(
    stage: Usd.Stage, instancer_paths: List[str], merged_instancer_path: str
) -> UsdGeom.PointInstancer:
    """Merge multiple point instancers into a single point instancer."""
    for instancer_path in instancer_paths:
        if not stage.GetPrimAtPath(instancer_path):
            raise ValueError(f"Invalid instancer path: {instancer_path}")
    merged_instancer = UsdGeom.PointInstancer.Define(stage, merged_instancer_path)
    prototypes = []
    proto_indices = []
    positions = []
    orientations = []
    scales = []
    velocities = []
    angular_velocities = []
    accelerations = []
    ids = []
    time_samples = set()
    for instancer_path in instancer_paths:
        instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(instancer_path))
        instancer_prototypes = instancer.GetPrototypesRel().GetTargets()
        proto_offset = len(prototypes)
        prototypes.extend(instancer_prototypes)
        indices_attr = instancer.GetProtoIndicesAttr()
        positions_attr = instancer.GetPositionsAttr()
        orientations_attr = instancer.GetOrientationsAttr()
        scales_attr = instancer.GetScalesAttr()
        velocities_attr = instancer.GetVelocitiesAttr()
        angular_velocities_attr = instancer.GetAngularVelocitiesAttr()
        accelerations_attr = instancer.GetAccelerationsAttr()
        ids_attr = instancer.GetIdsAttr()
        for attr in [
            indices_attr,
            positions_attr,
            orientations_attr,
            scales_attr,
            velocities_attr,
            angular_velocities_attr,
            accelerations_attr,
            ids_attr,
        ]:
            if attr:
                time_samples.update(attr.GetTimeSamples())
        for t in time_samples:
            indices = indices_attr.Get(t) if indices_attr else []
            proto_indices.extend([i + proto_offset for i in indices])
            positions.extend(positions_attr.Get(t) if positions_attr else [])
            orientations.extend(orientations_attr.Get(t) if orientations_attr else [])
            scales.extend(scales_attr.Get(t) if scales_attr else [])
            velocities.extend(velocities_attr.Get(t) if velocities_attr else [])
            angular_velocities.extend(angular_velocities_attr.Get(t) if angular_velocities_attr else [])
            accelerations.extend(accelerations_attr.Get(t) if accelerations_attr else [])
            ids.extend(ids_attr.Get(t) if ids_attr else [])
    merged_instancer.CreatePrototypesRel().SetTargets(prototypes)
    indices_attr = merged_instancer.CreateProtoIndicesAttr()
    positions_attr = merged_instancer.CreatePositionsAttr()
    for t in time_samples:
        indices_attr.Set(proto_indices, t)
        positions_attr.Set(positions, t)
        if orientations:
            merged_instancer.CreateOrientationsAttr().Set(orientations, t)
        if scales:
            merged_instancer.CreateScalesAttr().Set(scales, t)
        if velocities:
            merged_instancer.CreateVelocitiesAttr().Set(velocities, t)
        if angular_velocities:
            merged_instancer.CreateAngularVelocitiesAttr().Set(angular_velocities, t)
        if accelerations:
            merged_instancer.CreateAccelerationsAttr().Set(accelerations, t)
        if ids:
            merged_instancer.CreateIdsAttr().Set(ids, t)
    return merged_instancer


def activate_and_invis_ids(
    instancer: UsdGeom.PointInstancer, time: Usd.TimeCode, activate_ids: Vt.Int64Array, invis_ids: Vt.Int64Array
) -> None:
    """Activate and make invisible specific instances at a given time.

    Args:
        instancer (UsdGeom.PointInstancer): The PointInstancer prim.
        time (Usd.TimeCode): The time at which to activate/invis the instances.
        activate_ids (Vt.Int64Array): The ids of the instances to activate.
        invis_ids (Vt.Int64Array): The ids of the instances to make invisible.
    """
    if not instancer:
        raise ValueError("Invalid PointInstancer prim")
    if activate_ids:
        instancer.ActivateIds(activate_ids)
    if invis_ids:
        instancer.InvisIds(invis_ids, time)


def set_instance_primvar_values(
    instancer_prim: UsdGeom.PointInstancer,
    primvar_name: str,
    values: list,
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
) -> bool:
    """
    Set the values of a primvar on a UsdGeomPointInstancer prim.

    Args:
        instancer_prim (UsdGeom.PointInstancer): The UsdGeomPointInstancer prim.
        primvar_name (str): The name of the primvar to set.
        values (list): The values to set for the primvar.
        time_code (Usd.TimeCode, optional): The time code at which to set the values. Defaults to Usd.TimeCode.Default().

    Returns:
        bool: True if the primvar values were set successfully, False otherwise.
    """
    if not instancer_prim:
        print(f"Error: {instancer_prim.GetPrim().GetPath()} is not a valid UsdGeomPointInstancer prim.")
        return False
    primvar_attr = instancer_prim.GetPrim().GetAttribute(primvar_name)
    if not primvar_attr:
        if not values:
            print(f"Error: No values provided for primvar {primvar_name}.")
            return False
        primvar_type = type(values[0])
        if primvar_type == float:
            primvar_attr = instancer_prim.GetPrim().CreateAttribute(primvar_name, Sdf.ValueTypeNames.FloatArray)
        elif primvar_type == Gf.Vec2f:
            primvar_attr = instancer_prim.GetPrim().CreateAttribute(primvar_name, Sdf.ValueTypeNames.Float2Array)
        elif primvar_type == Gf.Vec3f:
            primvar_attr = instancer_prim.GetPrim().CreateAttribute(primvar_name, Sdf.ValueTypeNames.Float3Array)
        elif primvar_type == Gf.Vec4f:
            primvar_attr = instancer_prim.GetPrim().CreateAttribute(primvar_name, Sdf.ValueTypeNames.Float4Array)
        else:
            print(f"Error: Unsupported primvar type {primvar_type} for primvar {primvar_name}.")
            return False
        primvar_attr.SetMetadata("interpolation", "instance")
    try:
        primvar_attr.Set(values, time_code)
    except Exception as e:
        print(f"Error setting primvar {primvar_name} values at time code {time_code}: {str(e)}")
        return False
    return True


def test_set_instance_primvar_values():
    stage = Usd.Stage.CreateInMemory()
    instancer_prim = UsdGeom.PointInstancer.Define(stage, "/PointInstancer")
    float_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = set_instance_primvar_values(instancer_prim, "float_primvar", float_values)
    print(f"Set float primvar values: {result}")
    vec3f_values = [Gf.Vec3f(1.0, 2.0, 3.0), Gf.Vec3f(4.0, 5.0, 6.0), Gf.Vec3f(7.0, 8.0, 9.0)]
    result = set_instance_primvar_values(instancer_prim, "vec3f_primvar", vec3f_values)
    print(f"Set Vec3f primvar values: {result}")
    float_values_time = [10.0, 20.0, 30.0, 40.0, 50.0]
    result = set_instance_primvar_values(
        instancer_prim, "float_primvar", float_values_time, time_code=Usd.TimeCode(10.0)
    )
    print(f"Set float primvar values at time code 10: {result}")


def consolidate_instance_attributes(instancer: UsdGeom.PointInstancer) -> bool:
    """Consolidate instance attributes on the given UsdGeom.PointInstancer.

    This function will author all instance attributes (positions, velocities,
    accelerations, orientations, angularVelocities, scales, protoIndices, and ids)
    on the UsdGeom.PointInstancer prim itself, in the "instance" namespace.

    Returns:
        True if successful, False otherwise.
    """
    pos_attr = instancer.GetPositionsAttr()
    vel_attr = instancer.GetVelocitiesAttr()
    acc_attr = instancer.GetAccelerationsAttr()
    orient_attr = instancer.GetOrientationsAttr()
    ang_vel_attr = instancer.GetAngularVelocitiesAttr()
    scales_attr = instancer.GetScalesAttr()
    proto_indices_attr = instancer.GetProtoIndicesAttr()
    ids_attr = instancer.GetIdsAttr()
    if not pos_attr.IsValid():
        return False
    pos_attr.SetCustom(True)
    pos_attr.SetDisplayName("instance:positions")
    if vel_attr.IsValid():
        vel_attr.SetCustom(True)
        vel_attr.SetDisplayName("instance:velocities")
    if acc_attr.IsValid():
        acc_attr.SetCustom(True)
        acc_attr.SetDisplayName("instance:accelerations")
    if orient_attr.IsValid():
        orient_attr.SetCustom(True)
        orient_attr.SetDisplayName("instance:orientations")
    if ang_vel_attr.IsValid():
        ang_vel_attr.SetCustom(True)
        ang_vel_attr.SetDisplayName("instance:angularVelocities")
    if scales_attr.IsValid():
        scales_attr.SetCustom(True)
        scales_attr.SetDisplayName("instance:scales")
    proto_indices_attr.SetCustom(True)
    proto_indices_attr.SetDisplayName("instance:protoIndices")
    if ids_attr.IsValid():
        ids_attr.SetCustom(True)
        ids_attr.SetDisplayName("instance:ids")
    return True


def set_instance_orientations(
    point_instancer: UsdGeom.PointInstancer,
    orientations: Sequence[Gf.Quatf],
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
):
    """Set the orientations for each instance in a PointInstancer.

    Args:
        point_instancer (UsdGeom.PointInstancer): The PointInstancer prim.
        orientations (Sequence[Gf.Quatf]): A sequence of quaternions specifying the orientation for each instance.
        time_code (Usd.TimeCode): The time code at which to set the orientations. Defaults to Default time.

    Raises:
        ValueError: If the length of orientations does not match the number of instances in the PointInstancer.
    """
    orientations_attr = point_instancer.GetOrientationsAttr()
    num_instances = len(point_instancer.GetProtoIndicesAttr().Get(time_code))
    if len(orientations) != num_instances:
        raise ValueError(
            f"Number of orientations ({len(orientations)}) does not match the number of instances ({num_instances})."
        )
    gf_orientations = [Gf.Quath(quat) for quat in orientations]
    orientations_attr.Set(gf_orientations, time_code)


def remove_prototype_from_instancer(instancer: UsdGeom.PointInstancer, prototype_path: str) -> bool:
    """Remove a prototype from a PointInstancer.

    Args:
        instancer (UsdGeom.PointInstancer): The PointInstancer to remove the prototype from.
        prototype_path (str): The path of the prototype to remove.

    Returns:
        bool: True if the prototype was removed, False otherwise.
    """
    if not instancer:
        raise ValueError("Invalid PointInstancer")
    prototypes_rel = instancer.GetPrototypesRel()
    if not prototypes_rel:
        print(f"No prototypes found on {instancer.GetPath()}")
        return False
    prototype_target = Sdf.Path(prototype_path)
    prototype_targets = prototypes_rel.GetTargets()
    if prototype_target not in prototype_targets:
        print(f"Prototype {prototype_path} not found in {instancer.GetPath()}")
        return False
    prototype_index = prototype_targets.index(prototype_target)
    prototypes_rel.RemoveTarget(prototype_target)
    indices_attr = instancer.GetProtoIndicesAttr()
    if indices_attr:
        indices = indices_attr.Get()
        indices = [i for i in indices if i != prototype_index]
        indices = [i if i < prototype_index else i - 1 for i in indices]
        indices_attr.Set(indices)
    return True


def transfer_instances(
    source_instancer: UsdGeom.PointInstancer,
    dest_instancer: UsdGeom.PointInstancer,
    source_purpose: int,
    dest_purpose: int,
    time: Usd.TimeCode = Usd.TimeCode.Default(),
) -> bool:
    """Transfer instances from one PointInstancer to another based on the purpose.

    Args:
        source_instancer (UsdGeom.PointInstancer): The source PointInstancer to transfer instances from.
        dest_instancer (UsdGeom.PointInstancer): The destination PointInstancer to transfer instances to.
        source_purpose (int): The purpose value to filter instances from the source PointInstancer.
        dest_purpose (int): The purpose value to assign to the transferred instances on the destination PointInstancer.
        time (Usd.TimeCode): The time at which to transfer the instances. Defaults to Usd.TimeCode.Default().

    Returns:
        bool: True if the transfer was successful, False otherwise.
    """
    if not source_instancer.GetPrim().IsValid() or not dest_instancer.GetPrim().IsValid():
        print("Error: Invalid source or destination PointInstancer.")
        return False
    source_positions = source_instancer.GetPositionsAttr().Get(time)
    source_proto_indices = source_instancer.GetProtoIndicesAttr().Get(time)
    source_ids = source_instancer.GetIdsAttr().Get(time)
    if not source_positions or not source_proto_indices or len(source_positions) != len(source_proto_indices):
        print("Error: Invalid or mismatched attributes on the source PointInstancer.")
        return False
    filtered_positions = []
    filtered_proto_indices = []
    filtered_ids = []
    for i in range(len(source_positions)):
        if source_ids and source_ids[i] == source_purpose:
            filtered_positions.append(source_positions[i])
            filtered_proto_indices.append(source_proto_indices[i])
            filtered_ids.append(dest_purpose)
    if not filtered_positions:
        print("No instances found with the specified source purpose.")
        return True
    dest_positions = dest_instancer.GetPositionsAttr().Get(time)
    dest_proto_indices = dest_instancer.GetProtoIndicesAttr().Get(time)
    dest_ids = dest_instancer.GetIdsAttr().Get(time)
    dest_positions = list(dest_positions) if dest_positions else []
    dest_proto_indices = list(dest_proto_indices) if dest_proto_indices else []
    dest_ids = list(dest_ids) if dest_ids else []
    dest_positions.extend(filtered_positions)
    dest_proto_indices.extend(filtered_proto_indices)
    dest_ids.extend(filtered_ids)
    dest_instancer.GetPositionsAttr().Set(dest_positions, time)
    dest_instancer.GetProtoIndicesAttr().Set(dest_proto_indices, time)
    dest_instancer.GetIdsAttr().Set(dest_ids, time)
    return True


def apply_transforms_to_instances(instancer: UsdGeom.PointInstancer, time: Usd.TimeCode) -> bool:
    """Apply prototype root transforms to each instance.

    This function will compute the transform for each instance by
    combining the instance's transform with the transform of the
    prototype root prim.

    Args:
        instancer (UsdGeom.PointInstancer): The point instancer prim.
        time (Usd.TimeCode): Timecode to evaluate the transforms at.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not instancer or not instancer.GetPrim().IsA(UsdGeom.PointInstancer):
        return False
    prototypes = instancer.GetPrototypesRel().GetTargets()
    if not prototypes:
        return False
    positions = instancer.GetPositionsAttr().Get(time)
    if not positions:
        return False
    indices = instancer.GetProtoIndicesAttr().Get(time)
    if not indices:
        return False
    scales = instancer.GetScalesAttr().Get(time)
    orientations = instancer.GetOrientationsAttr().Get(time)
    new_xforms = []
    for i, proto_index in enumerate(indices):
        if proto_index >= len(prototypes):
            continue
        proto_root = stage.GetPrimAtPath(prototypes[proto_index])
        proto_root_xform = UsdGeom.Xformable(proto_root).ComputeLocalToWorldTransform(time)
        inst_xform = Gf.Matrix4d(1)
        inst_xform.SetTranslateOnly(Gf.Vec3d(positions[i]))
        if orientations and i < len(orientations):
            orientation_matrix = Gf.Matrix4d(1)
            orientation_matrix.SetRotate(orientations[i])
            inst_xform = inst_xform * orientation_matrix
        if scales and i < len(scales):
            scale_matrix = Gf.Matrix4d(1)
            scale_matrix.SetScale(Gf.Vec3d(scales[i]))
            inst_xform = inst_xform * scale_matrix
        new_xform = inst_xform * proto_root_xform
        new_xforms.append(new_xform)
    xform_attr = instancer.GetPrim().GetAttribute("xformOp:transform")
    if not xform_attr:
        xform_attr = instancer.GetPrim().CreateAttribute("xformOp:transform", Sdf.ValueTypeNames.Matrix4dArray)
    xform_attr.Set(new_xforms, time)
    return True


def get_instance_orientations(
    point_instancer: UsdGeom.PointInstancer, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> Vt.QuathArray:
    """
    Get the orientations of instances in a UsdGeomPointInstancer at a specific time.

    Args:
        point_instancer (UsdGeom.PointInstancer): The PointInstancer to get the orientations from.
        time_code (Usd.TimeCode): The time code to query the orientations at. Defaults to Default time.

    Returns:
        Vt.QuathArray: An array of orientations as quaternions, one per instance.

    Raises:
        ValueError: If the 'orientations' attribute is not defined on the PointInstancer.
    """
    orientations_attr = point_instancer.GetOrientationsAttr()
    if not orientations_attr:
        raise ValueError("'orientations' attribute not found on the PointInstancer.")
    orientations = orientations_attr.Get(time_code)
    if orientations is None:
        return Vt.QuathArray()
    return orientations


def apply_point_instancer_mask(
    point_instancer: UsdGeom.PointInstancer, time_code: Optional[Usd.TimeCode] = None
) -> bool:
    """Apply the point instancer mask at the specified time code.

    Args:
        point_instancer (UsdGeom.PointInstancer): The point instancer to apply the mask to.
        time_code (Optional[Usd.TimeCode], optional): The time code to evaluate the mask at. Defaults to None.

    Returns:
        bool: True if the mask should be applied, False otherwise.
    """
    if time_code is None:
        time_code = Usd.TimeCode.Default()
    if not point_instancer.GetPrim().IsValid():
        raise ValueError("Invalid point instancer prim.")
    mask_attr = point_instancer.GetProtoIndicesAttr()
    if not mask_attr.HasValue():
        return False
    mask = point_instancer.ComputeMaskAtTime(time_code)
    if len(mask) == 0:
        return False
    return True


def set_proto_xform_inclusion(point_instancer: UsdGeom.PointInstancer, proto_index: int, inclusion: str) -> bool:
    """Set the proto xform inclusion for a specific prototype.

    Args:
        point_instancer (UsdGeom.PointInstancer): The PointInstancer to set the proto xform inclusion on.
        proto_index (int): The index of the prototype to set the inclusion for.
        inclusion (str): The inclusion value to set. Valid values are "IncludeProtoXform" and "ExcludeProtoXform".

    Returns:
        bool: True if the value was set successfully, False otherwise.
    """
    if not point_instancer:
        return False
    proto_indices_attr = point_instancer.GetProtoIndicesAttr()
    if not proto_indices_attr or not proto_indices_attr.Get():
        return False
    if proto_index < 0 or proto_index >= len(proto_indices_attr.Get()):
        return False
    proto_xform_inclusion_attr = point_instancer.GetProtoXformInclusionAttr()
    new_inclusions = list(proto_xform_inclusion_attr.Get()) if proto_xform_inclusion_attr.Get() else []
    if proto_index >= len(new_inclusions):
        new_inclusions.extend([UsdGeom.Tokens.IncludeProtoXform] * (proto_index - len(new_inclusions) + 1))
    new_inclusions[proto_index] = (
        UsdGeom.Tokens.IncludeProtoXform if inclusion == "IncludeProtoXform" else UsdGeom.Tokens.ExcludeProtoXform
    )
    success = proto_xform_inclusion_attr.Set(new_inclusions)
    return success


def copy_primvar(
    src_primvar: UsdGeom.Primvar,
    dest_prim: Usd.Prim,
    dest_name: str = None,
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
) -> UsdGeom.Primvar:
    """Copy a primvar from one prim to another.

    Args:
        src_primvar (UsdGeom.Primvar): The source primvar to copy.
        dest_prim (Usd.Prim): The destination prim to copy the primvar to.
        dest_name (str, optional): The name to give the new primvar on the destination prim.
            If None, uses the same name as the source primvar. Defaults to None.
        time_code (Usd.TimeCode, optional): The time code to use when copying the primvar value.
            Defaults to Usd.TimeCode.Default().

    Returns:
        UsdGeom.Primvar: The newly created primvar on the destination prim.
    """
    if not src_primvar.IsDefined():
        raise ValueError("Source primvar is not defined")
    primvar_name = dest_name if dest_name else src_primvar.GetPrimvarName()
    full_primvar_name = UsdGeom.Primvar.StripPrimvarsName(primvar_name)
    dest_primvar = UsdGeom.Primvar(dest_prim.CreateAttribute(full_primvar_name, src_primvar.GetTypeName()))
    dest_primvar.SetInterpolation(src_primvar.GetInterpolation())
    if src_primvar.HasAuthoredElementSize():
        dest_primvar.SetElementSize(src_primvar.GetElementSize())
    src_value = src_primvar.Get(time_code)
    if src_value is not None:
        dest_primvar.Set(src_value, time_code)
    if src_primvar.IsIndexed():
        src_indices = src_primvar.GetIndices(time_code)
        if src_indices is not None:
            dest_primvar.SetIndices(src_indices, time_code)
        if src_primvar.GetUnauthoredValuesIndex() != -1:
            dest_primvar.SetUnauthoredValuesIndex(src_primvar.GetUnauthoredValuesIndex())
    return dest_primvar


def get_inherited_primvars(prim: Usd.Prim) -> List[UsdGeom.Primvar]:
    """
    Get the list of inherited primvars for a given prim.

    Args:
        prim (Usd.Prim): The prim to get inherited primvars for.

    Returns:
        List[UsdGeom.Primvar]: A list of inherited Primvar objects.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    parent = prim.GetParent()
    if not parent:
        return []
    parent_primvars = UsdGeom.PrimvarsAPI(parent).GetPrimvars()
    inherited_primvars = [
        primvar for primvar in parent_primvars if primvar.GetInterpolation() == UsdGeom.Tokens.constant
    ]
    ancestor_primvars = get_inherited_primvars(parent)
    inherited_primvars.extend(ancestor_primvars)
    return inherited_primvars


def has_authored_primvar_element_size(primvar: UsdGeom.Primvar) -> bool:
    """Return true if elementSize is authored on the given primvar."""
    attr = primvar.GetAttr()
    if not attr.IsValid():
        raise ValueError("Invalid primvar attribute")
    primvar_name = primvar.GetPrimvarName()
    fallback_value = 1
    authored_value = attr.GetMetadata("elementSize")
    if authored_value is not None:
        return True
    elif attr.HasMetadata("elementSize"):
        return True
    else:
        return False


def get_primvar_with_custom_fallback(prim: Usd.Prim, primvar_name: str, fallback_value: Any) -> Any:
    """
    Get the value of a primvar on a prim, with a custom fallback value.

    Args:
        prim (Usd.Prim): The prim to get the primvar from.
        primvar_name (str): The name of the primvar to get.
        fallback_value (Any): The fallback value to return if the primvar doesn't exist or has no value.

    Returns:
        Any: The value of the primvar, or the fallback value if the primvar doesn't exist or has no value.
    """
    if not prim.IsValid():
        return fallback_value
    primvar_attr = prim.GetAttribute(primvar_name)
    if not primvar_attr.IsDefined():
        return fallback_value
    primvar_value = primvar_attr.Get()
    if primvar_value is None:
        return fallback_value
    return primvar_value


def validate_primvar_interpolation(primvar: UsdGeom.Primvar, allowed_interpolations: list[str]) -> bool:
    """Validate that the interpolation of a primvar is in the list of allowed interpolations.

    Args:
        primvar (UsdGeom.Primvar): The primvar to validate.
        allowed_interpolations (list[str]): The list of allowed interpolations.

    Returns:
        bool: True if the primvar has an allowed interpolation, False otherwise.
    """
    if not primvar.IsDefined():
        raise ValueError("Primvar is not defined")
    interpolation = primvar.GetInterpolation()
    if interpolation not in allowed_interpolations:
        return False
    return True


def split_primvar_name(primvar_name: str) -> Tuple[str, str]:
    """Split a primvar name into its namespace and base name.

    Args:
        primvar_name (str): The full primvar name, e.g., "primvars:foo:bar"

    Returns:
        Tuple[str, str]: A tuple containing the namespace and base name.
                         If there is no namespace, the first element will be an empty string.

    Raises:
        ValueError: If the primvar name is empty.
    """
    if not primvar_name:
        raise ValueError("Primvar name cannot be empty")
    parts = primvar_name.split(":")
    if parts[0] == "primvars":
        parts = parts[1:]
    base_name = parts[-1]
    namespace = ":".join(parts[:-1])
    return (namespace, base_name)


def set_primvar_as_id_target(primvar: UsdGeom.Primvar, target_path: Sdf.Path) -> bool:
    """Set a primvar of String or StringArray type as an ID target."""
    if primvar.GetTypeName() not in [Sdf.ValueTypeNames.String, Sdf.ValueTypeNames.StringArray]:
        raise ValueError("Primvar must be of String or StringArray type to be set as an ID target.")
    success = primvar.SetIdTarget(target_path)
    return success


def get_primvar_data(
    prim: Usd.Prim, primvar_name: str, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> Tuple[bool, Any]:
    """
    Get the data for a primvar on a prim.

    Args:
        prim (Usd.Prim): The prim to get the primvar data from.
        primvar_name (str): The name of the primvar.
        time_code (Usd.TimeCode): The time code to get the data at. Defaults to Default.

    Returns:
        A tuple of (bool, Any):
            - bool: True if the primvar exists and has a value, False otherwise.
            - Any: The value of the primvar if it exists, None otherwise.
    """
    if not prim.IsValid():
        return (False, None)
    primvar_attr = prim.GetAttribute(primvar_name)
    if not primvar_attr.IsDefined():
        return (False, None)
    primvar_value = primvar_attr.Get(time_code)
    if primvar_value is None:
        return (False, None)
    return (True, primvar_value)


def find_primvars_with_interpolation(prim: Usd.Prim, interpolation: str) -> list[UsdGeom.Primvar]:
    """Find all primvars on a prim with a specific interpolation.

    Args:
        prim (Usd.Prim): The prim to search for primvars.
        interpolation (str): The interpolation to match.

    Returns:
        list[UsdGeom.Primvar]: A list of primvars with the specified interpolation.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if not UsdGeom.Primvar.IsValidInterpolation(interpolation):
        raise ValueError(f"Invalid interpolation: {interpolation}")
    primvars_api = UsdGeom.PrimvarsAPI(prim)
    primvars = primvars_api.GetPrimvars()
    matching_primvars = [primvar for primvar in primvars if primvar.GetInterpolation() == interpolation]
    return matching_primvars


def list_all_primvars(prim: Usd.Prim) -> List[Tuple[str, Sdf.ValueTypeName]]:
    """List all primvars on a prim.

    Args:
        prim (Usd.Prim): The prim to query primvars from.

    Returns:
        List[Tuple[str, Sdf.ValueTypeName]]: A list of tuples containing the
        primvar name and value type for each primvar found.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    primvars_api = UsdGeom.PrimvarsAPI(prim)
    primvars = primvars_api.GetPrimvars()
    primvar_info = []
    for primvar in primvars:
        primvar_name = primvar.GetPrimvarName()
        value_type = primvar.GetTypeName()
        primvar_info.append((primvar_name, value_type))
    return primvar_info


def get_primvar_interpolation(prim: Usd.Prim, primvar_name: str) -> str:
    """Get the interpolation for a primvar on a prim.

    Args:
        prim (Usd.Prim): The prim to get the primvar from.
        primvar_name (str): The name of the primvar.

    Returns:
        str: The interpolation of the primvar.

    Raises:
        ValueError: If the prim is not valid or if the primvar does not exist.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    if not UsdGeom.Primvar.IsPrimvar(prim.GetAttribute(primvar_name)):
        raise ValueError(f"Primvar {primvar_name} does not exist on prim {prim.GetPath()}.")
    primvar = UsdGeom.Primvar(prim.GetAttribute(primvar_name))
    interpolation = primvar.GetInterpolation()
    return interpolation


def get_primvar_element_size(primvar: UsdGeom.Primvar) -> int:
    """Get the element size of a primvar.

    Args:
        primvar (UsdGeom.Primvar): The primvar to query.

    Returns:
        int: The element size of the primvar. Returns 1 if not authored.
    """
    if not primvar.IsDefined():
        raise ValueError("Primvar is not defined")
    if primvar.HasAuthoredElementSize():
        element_size = primvar.GetElementSize()
    else:
        element_size = 1
    return element_size


def get_primvar_indices(
    primvar: UsdGeom.Primvar, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> Tuple[bool, List[int]]:
    """
    Get the indices value of a primvar at a specific time code.

    Args:
        primvar (UsdGeom.Primvar): The primvar to get the indices from.
        time_code (Usd.TimeCode): The time code at which to get the indices. Defaults to Default.

    Returns:
        Tuple[bool, List[int]]: A tuple with a boolean indicating if the primvar is indexed,
                                and a list of indices if indexed, otherwise an empty list.
    """
    if not primvar.IsDefined():
        return (False, [])
    is_indexed = primvar.IsIndexed()
    indices = []
    if is_indexed:
        indices_attr = primvar.GetIndicesAttr()
        if indices_attr.HasValue():
            indices = indices_attr.Get(time_code)
        else:
            pass
    else:
        pass
    return (is_indexed, indices)


def create_primvar(
    prim: Usd.Prim,
    primvar_name: str,
    typeName: Sdf.ValueTypeName,
    interpolation: str = "constant",
    elementSize: int = 1,
) -> UsdGeom.Primvar:
    """Create a new Primvar attribute on the given prim.

    Args:
        prim (Usd.Prim): The prim to create the primvar on.
        primvar_name (str): The name of the primvar attribute.
        typeName (Sdf.ValueTypeName): The value type of the primvar.
        interpolation (str, optional): Interpolation type. Defaults to "constant".
        elementSize (int, optional): The elementSize of the primvar. Defaults to 1.

    Returns:
        UsdGeom.Primvar: The created Primvar object.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if not UsdGeom.Primvar.IsValidInterpolation(interpolation):
        raise ValueError(f"Invalid interpolation type: {interpolation}")
    if elementSize < 1:
        raise ValueError(f"elementSize must be greater than or equal to 1, got {elementSize}")
    full_primvar_name = f"primvars:{primvar_name}"
    primvar = UsdGeom.Primvar(prim.CreateAttribute(full_primvar_name, typeName, False))
    primvar.SetInterpolation(interpolation)
    if elementSize > 1:
        primvar.SetElementSize(elementSize)
    return primvar


def get_primvar_declaration_info(primvar: UsdGeom.Primvar) -> Tuple[str, Sdf.ValueTypeName, str, int]:
    """Get the declaration info for a primvar.

    Returns:
        A tuple containing (name, typeName, interpolation, elementSize)
    """
    if not primvar.IsDefined():
        raise ValueError("Primvar is not defined")
    primvar_name = primvar.GetPrimvarName()
    type_name = primvar.GetTypeName()
    interpolation = primvar.GetInterpolation()
    element_size = primvar.GetElementSize()
    return (primvar_name, type_name, interpolation, element_size)


def is_primvar_defined(attr: Usd.Attribute) -> bool:
    """Return true if the given UsdAttribute represents a valid Primvar."""
    if not attr.IsDefined():
        return False
    attr_name = attr.GetName()
    if not attr_name.startswith("primvars:"):
        return False
    primvar_name = UsdGeom.Primvar.StripPrimvarsName(attr_name)
    if not UsdGeom.Primvar.IsValidPrimvarName(primvar_name):
        return False
    return True


def set_primvar_element_size(primvar: UsdGeom.Primvar, element_size: int) -> bool:
    """Set the element size for a primvar.

    Args:
        primvar (UsdGeom.Primvar): The primvar to set the element size on.
        element_size (int): The element size value to set.

    Returns:
        bool: True if setting the element size succeeded, False otherwise.
    """
    if not primvar.IsDefined():
        return False
    if element_size < 1:
        return False
    success = primvar.SetElementSize(element_size)
    return success


def remove_primvar(prim: Usd.Prim, primvar_name: str, quiet: bool = False) -> bool:
    """Remove a primvar from a prim.

    Args:
        prim (Usd.Prim): The prim to remove the primvar from.
        primvar_name (str): The name of the primvar to remove.
        quiet (bool): If True, suppress warning messages.

    Returns:
        bool: True if the primvar was successfully removed, False otherwise.
    """
    if not prim.IsValid():
        if not quiet:
            warnings.warn(f"Prim {prim.GetPath()} is not valid.")
        return False
    primvar_attr = prim.GetAttribute(f"primvars:{primvar_name}")
    if not primvar_attr.IsValid():
        if not quiet:
            warnings.warn(f"Primvar '{primvar_name}' does not exist on prim {prim.GetPath()}.")
        return False
    success = prim.RemoveProperty(primvar_attr.GetName())
    if not success:
        if not quiet:
            warnings.warn(f"Failed to remove primvar '{primvar_name}' from prim {prim.GetPath()}.")
        return False
    return True


def set_primvar_interpolation(prim: Usd.Prim, primvar_name: str, interpolation: str) -> None:
    """Set the interpolation for a primvar on a given prim."""
    primvars_api = UsdGeom.PrimvarsAPI(prim)
    if not primvars_api.HasPrimvar(primvar_name):
        raise ValueError(f"Primvar '{primvar_name}' does not exist on prim '{prim.GetPath()}'")
    primvar = primvars_api.GetPrimvar(primvar_name)
    valid_interpolations = [
        UsdGeom.Tokens.constant,
        UsdGeom.Tokens.uniform,
        UsdGeom.Tokens.varying,
        UsdGeom.Tokens.vertex,
        UsdGeom.Tokens.faceVarying,
    ]
    if interpolation not in valid_interpolations:
        raise ValueError(f"Invalid interpolation '{interpolation}'. Must be one of: {', '.join(valid_interpolations)}")
    primvar.SetInterpolation(interpolation)


def set_primvar_unauthored_values_index(primvar: UsdGeom.Primvar, unauthored_values_index: int) -> bool:
    """Set the index that represents unauthored values in the indices array for a primvar.

    Args:
        primvar (UsdGeom.Primvar): The primvar to set the unauthored values index on.
        unauthored_values_index (int): The index representing unauthored values.

    Returns:
        bool: True if the unauthored values index was set successfully, False otherwise.
    """
    if not primvar.IsDefined():
        return False
    if unauthored_values_index < 0:
        return False
    success = primvar.SetUnauthoredValuesIndex(unauthored_values_index)
    return success


def copy_primvars_between_prims(
    src_prim: Usd.Prim, dest_prim: Usd.Prim, exclude_primvars: Optional[List[str]] = None
) -> None:
    """Copy primvars from one prim to another.

    Args:
        src_prim (Usd.Prim): The source prim to copy primvars from.
        dest_prim (Usd.Prim): The destination prim to copy primvars to.
        exclude_primvars (Optional[List[str]], optional): A list of primvar names to exclude from copying. Defaults to None.
    """
    src_primvars_api = UsdGeom.PrimvarsAPI(src_prim)
    src_primvars = src_primvars_api.GetPrimvars()
    dest_primvars_api = UsdGeom.PrimvarsAPI(dest_prim)
    for src_primvar in src_primvars:
        primvar_name = src_primvar.GetPrimvarName()
        if exclude_primvars and primvar_name in exclude_primvars:
            continue
        src_full_name = src_primvar.GetName()
        src_type_name = src_primvar.GetTypeName()
        src_interpolation = src_primvar.GetInterpolation()
        dest_primvar = dest_primvars_api.CreatePrimvar(primvar_name, src_type_name, src_interpolation)
        src_value = src_primvar.Get()
        if src_value is not None:
            dest_primvar.Set(src_value)
        if src_primvar.IsIndexed():
            src_indices = src_primvar.GetIndices()
            dest_primvar.SetIndices(src_indices)


def get_primvars_with_values(prim: Usd.Prim) -> List[UsdGeom.Primvar]:
    """
    Get all primvars on the given prim that have a value, whether authored or from schema fallback.

    Args:
        prim (Usd.Prim): The prim to get primvars for.

    Returns:
        List[UsdGeom.Primvar]: A list of primvars with values on the prim.
    """
    primvars_api = UsdGeom.PrimvarsAPI(prim)
    if not primvars_api:
        raise ValueError(f"Prim at path {prim.GetPath()} does not have a valid PrimvarsAPI schema.")
    primvars_with_values = primvars_api.GetPrimvarsWithValues()
    return primvars_with_values


def create_indexed_primvar(
    prim: Usd.Prim,
    name: str,
    typeName: Sdf.ValueTypeName,
    value: Any,
    indices: Vt.IntArray,
    interpolation: str = "constant",
    elementSize: int = 1,
) -> UsdGeom.Primvar:
    """Create an indexed primvar on the given prim.

    Args:
        prim (Usd.Prim): The prim to create the primvar on.
        name (str): The name of the primvar.
        typeName (Sdf.ValueTypeName): The value type of the primvar.
        value (Any): The value of the primvar.
        indices (Vt.IntArray): The indices array for the primvar.
        interpolation (str): The interpolation type of the primvar. Defaults to "constant".
        elementSize (int): The element size of the primvar. Defaults to 1.

    Returns:
        UsdGeom.Primvar: The created primvar.
    """
    primvars_api = UsdGeom.PrimvarsAPI(prim)
    primvar = primvars_api.CreateIndexedPrimvar(
        name=name, typeName=typeName, value=value, indices=indices, interpolation=interpolation, elementSize=elementSize
    )
    if not primvar.IsIndexed():
        raise ValueError(f"Failed to create indexed primvar '{name}' on prim '{prim.GetPath()}'")
    return primvar


def find_and_create_missing_primvars(prim: Usd.Prim, primvar_names: List[str]) -> List[UsdGeom.Primvar]:
    """Find or create the specified primvars on the given prim.

    Args:
        prim (Usd.Prim): The prim to search for primvars.
        primvar_names (List[str]): The names of the primvars to find or create.

    Returns:
        List[UsdGeom.Primvar]: The found or created primvars.
    """
    primvars_api = UsdGeom.PrimvarsAPI(prim)
    created_primvars = []
    for primvar_name in primvar_names:
        if primvars_api.HasPrimvar(primvar_name):
            primvar = primvars_api.GetPrimvar(primvar_name)
        else:
            primvar = primvars_api.CreatePrimvar(primvar_name, Sdf.ValueTypeNames.Float, UsdGeom.Tokens.constant, 1)
        created_primvars.append(primvar)
    return created_primvars


def merge_primvars_from_multiple_prims(prims: List[Usd.Prim]) -> Dict[str, UsdGeom.Primvar]:
    """Merges primvars from multiple prims into a single dictionary.

    Args:
        prims (List[Usd.Prim]): The list of prims to merge primvars from.

    Returns:
        Dict[str, UsdGeom.Primvar]: A dictionary mapping primvar names to primvar objects.
    """
    merged_primvars = {}
    for prim in prims:
        if not prim.IsValid():
            continue
        primvars_api = UsdGeom.PrimvarsAPI(prim)
        primvars = primvars_api.GetPrimvars()
        for primvar in primvars:
            primvar_name = primvar.GetPrimvarName()
            if primvar_name in merged_primvars:
                continue
            merged_primvars[primvar_name] = primvar
    return merged_primvars


def create_prim_with_primvars(stage, path, primvars):
    """Helper function to create a prim with primvars."""
    prim = stage.DefinePrim(path)
    primvars_api = UsdGeom.PrimvarsAPI(prim)
    for primvar_name, primvar_value in primvars.items():
        primvar = primvars_api.CreatePrimvar(primvar_name, Sdf.ValueTypeNames.Float)
        primvar.Set(primvar_value)
    return prim


def block_and_remove_primvar(primvars_api: UsdGeom.PrimvarsAPI, primvar_name: str) -> bool:
    """Block and remove a primvar from the prim.

    This function will block the primvar and its associated indices attribute,
    then remove the primvar and indices attribute from the current edit target.

    Args:
        primvars_api (UsdGeom.PrimvarsAPI): The PrimvarsAPI object for the prim.
        primvar_name (str): The name of the primvar to block and remove.

    Returns:
        bool: True if the primvar and indices attribute were successfully blocked and removed,
              False otherwise.
    """
    if not primvars_api.HasPrimvar(primvar_name):
        return False
    try:
        primvars_api.BlockPrimvar(primvar_name)
    except Tf.ErrorException:
        return False
    try:
        removed = primvars_api.RemovePrimvar(primvar_name)
    except Tf.ErrorException:
        return False
    return removed


def remove_all_primvars_from_prim(prim: Usd.Prim) -> None:
    """Remove all primvars from the given prim."""
    primvars_api = UsdGeom.PrimvarsAPI(prim)
    primvars = primvars_api.GetPrimvars()
    for primvar in primvars:
        primvar_name = primvar.GetPrimvarName()
        if not primvars_api.RemovePrimvar(primvar_name):
            print(f"Failed to remove primvar '{primvar_name}' from prim '{prim.GetPath()}'")


def get_all_primvars_with_values(prim: Usd.Prim) -> List[UsdGeom.Primvar]:
    """Get all primvars with authored values or fallback values on the given prim.

    Args:
        prim (Usd.Prim): The prim to retrieve primvars from.

    Returns:
        List[UsdGeom.Primvar]: A list of primvars with values.
    """
    primvars_api = UsdGeom.PrimvarsAPI(prim)
    primvars_with_values = primvars_api.GetPrimvarsWithValues()
    return primvars_with_values


def get_primvar_from_prim(prim: Usd.Prim, primvar_name: str) -> Tuple[bool, UsdGeom.Primvar]:
    """Get a primvar from a prim by name.

    Args:
        prim (Usd.Prim): The prim to get the primvar from.
        primvar_name (str): The name of the primvar to get.

    Returns:
        Tuple[bool, UsdGeom.Primvar]: A tuple with a boolean indicating if the primvar was found
        and the primvar object. If not found, the primvar object will be invalid.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    primvar_api = UsdGeom.PrimvarsAPI(prim)
    if primvar_api.HasPrimvar(primvar_name):
        primvar = primvar_api.GetPrimvar(primvar_name)
        if primvar:
            return (True, primvar)
        else:
            return (False, UsdGeom.Primvar())
    else:
        return (False, UsdGeom.Primvar())


def set_primvar_value_over_time(
    prim: Usd.Prim,
    primvar_name: str,
    interpolation: str,
    value_type: Union[float, Tuple[float, float, float]],
    time_samples: List[Tuple[float, Union[float, Tuple[float, float, float]]]],
) -> None:
    """Sets time-sampled value for a primvar on a given prim.

    Args:
        prim (Usd.Prim): The prim to create the primvar on.
        primvar_name (str): The name of the primvar.
        interpolation (str): Interpolation type for the primvar (e.g., 'constant', 'vertex').
        value_type (Union[float, Tuple[float, float, float]]): Type of the primvar value (e.g., float, float3).
        time_samples (List[Tuple[float, Union[float, Tuple[float, float, float]]]]):
            List of tuples representing time samples, where each tuple contains a time value and a primvar value.

    Raises:
        ValueError: If prim is not valid, primvar_name is empty, or interpolation is not supported.
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid.")
    if not primvar_name:
        raise ValueError("Primvar name cannot be empty.")
    supported_interpolations = ["constant", "uniform", "vertex", "faceVarying"]
    if interpolation not in supported_interpolations:
        raise ValueError(
            f"Unsupported interpolation type: {interpolation}. Supported types are: {', '.join(supported_interpolations)}"
        )
    if isinstance(value_type, float):
        type_name = Sdf.ValueTypeNames.Float
    elif isinstance(value_type, tuple) and len(value_type) == 3:
        type_name = Sdf.ValueTypeNames.Float3
    else:
        raise ValueError("Unsupported value type. Only float and float3 are supported.")
    primvars_api = UsdGeom.PrimvarsAPI(prim)
    primvar = primvars_api.CreatePrimvar(primvar_name, type_name, interpolation)
    for time, value in time_samples:
        primvar.Set(value, Usd.TimeCode(time))


def find_primvars_with_blocked_values(prim: Usd.Prim) -> List[Tuple[UsdGeom.Primvar, Usd.Attribute]]:
    """Find all primvars on the given prim that have blocked values.

    Returns a list of tuples, where each tuple contains the primvar and the
    attribute that is blocking its value.
    """
    blocked_primvars = []
    for attr in prim.GetAttributes():
        primvar = UsdGeom.Primvar(attr)
        if not primvar:
            continue
        if attr.HasAuthoredValue() and attr.IsAuthored() and attr.GetResolveInfo().ValueIsBlocked():
            blocking_attr = attr.GetResolveInfo().GetValueBlockingAttribute()
            blocked_primvars.append((primvar, blocking_attr))
    return blocked_primvars


def get_all_authored_primvars(prim: Usd.Prim) -> List[UsdGeom.Primvar]:
    """Get all authored primvars on a prim.

    Args:
        prim (Usd.Prim): The prim to get primvars from.

    Returns:
        List[UsdGeom.Primvar]: A list of all authored primvars on the prim.
    """
    primvars_api = UsdGeom.PrimvarsAPI(prim)
    authored_primvars = primvars_api.GetAuthoredPrimvars()
    return authored_primvars


def update_primvar_indices(
    primvar: UsdGeom.Primvar, indices: Vt.IntArray, time: Usd.TimeCode = Usd.TimeCode.Default()
) -> bool:
    """Update the indices for a primvar.

    Args:
        primvar (UsdGeom.Primvar): The primvar to update indices for.
        indices (Vt.IntArray): The new indices value to set.
        time (Usd.TimeCode, optional): The time at which to set the value. Defaults to Default.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not primvar:
        raise ValueError(f"Invalid primvar: {primvar.GetName()}")
    if not primvar.IsIndexed():
        raise ValueError(f"Primvar '{primvar.GetName()}' is not indexed")
    indices_attr = primvar.GetIndicesAttr()
    if not indices_attr:
        raise ValueError(f"Indices attribute for primvar '{primvar.GetName()}' is not valid")
    success = indices_attr.Set(indices, time)
    return success


def create_scope_with_children(stage: Usd.Stage, scope_path: str, child_prims: List[Tuple[str, str]]) -> Usd.Prim:
    """Create a UsdGeomScope prim with the given child prims.

    Args:
        stage (Usd.Stage): The USD stage to create the scope on.
        scope_path (str): The path where the scope should be created.
        child_prims (List[Tuple[str, str]]): A list of tuples representing the child prims to create.
                                             Each tuple contains the prim name and type.

    Returns:
        Usd.Prim: The created UsdGeomScope prim.

    Raises:
        ValueError: If the scope path is invalid or any of the child prim types are invalid.
    """
    if not Sdf.Path(scope_path).IsAbsolutePath():
        raise ValueError(f"Invalid scope path: {scope_path}. Path must be an absolute path.")
    scope_prim = UsdGeom.Scope.Define(stage, scope_path).GetPrim()
    for child_name, child_type in child_prims:
        child_path = scope_prim.GetPath().AppendChild(child_name)
        if child_type not in ["Xform", "Cube", "Sphere", "Cylinder", "Cone"]:
            raise ValueError(
                f"Invalid child prim type: {child_type}. Must be one of 'Xform', 'Cube', 'Sphere', 'Cylinder', or 'Cone'."
            )
        stage.DefinePrim(child_path, child_type)
    return scope_prim


def copy_scope_hierarchy(source_stage: Usd.Stage, dest_stage: Usd.Stage, source_scope_path: str, dest_scope_path: str):
    """
    Copy the hierarchy of UsdGeomScope prims from one stage to another.

    Args:
        source_stage (Usd.Stage): The stage containing the source scope hierarchy.
        dest_stage (Usd.Stage): The stage where the scope hierarchy will be copied to.
        source_scope_path (str): The path of the root scope prim in the source stage.
        dest_scope_path (str): The path where the scope hierarchy will be copied to in the destination stage.
    """
    source_scope = UsdGeom.Scope.Get(source_stage, source_scope_path)
    if not source_scope:
        raise ValueError(f"No UsdGeomScope prim found at path '{source_scope_path}' in the source stage.")
    dest_scope = UsdGeom.Scope.Define(dest_stage, dest_scope_path)

    def copy_scope(source_prim, dest_prim):
        for attr in source_prim.GetAttributes():
            if attr.IsAuthored():
                attr_name = attr.GetName()
                attr_value = attr.Get()
                dest_prim.CreateAttribute(attr_name, attr.GetTypeName()).Set(attr_value)
        for child_prim in source_prim.GetChildren():
            if child_prim.IsA(UsdGeom.Scope):
                child_name = child_prim.GetName()
                dest_child_prim = UsdGeom.Scope.Define(
                    dest_stage, dest_prim.GetPath().AppendChild(child_name)
                ).GetPrim()
                copy_scope(child_prim, dest_child_prim)

    copy_scope(source_scope.GetPrim(), dest_scope.GetPrim())


def get_all_prims_in_scope(scope_prim: UsdGeom.Scope) -> List[Usd.Prim]:
    """
    Get all prims under a UsdGeom.Scope prim.

    Args:
        scope_prim (UsdGeom.Scope): The scope prim to search under.

    Returns:
        List[Usd.Prim]: A list of all prims under the scope prim.
    """
    if not scope_prim or not scope_prim.GetPrim().IsA(UsdGeom.Scope):
        raise ValueError("Invalid UsdGeom.Scope prim")
    prim = scope_prim.GetPrim()
    prims = []

    def traverse(p):
        prims.append(p)
        for child in p.GetChildren():
            traverse(child)

    traverse(prim)
    return prims


def reparent_scope_children(source_scope_path: str, dest_scope_path: str, stage: Usd.Stage) -> bool:
    """
    Reparent all children of the source scope to the destination scope.

    Args:
        source_scope_path (str): The path to the source scope prim.
        dest_scope_path (str): The path to the destination scope prim.
        stage (Usd.Stage): The stage containing the scopes.

    Returns:
        bool: True if the reparenting was successful, False otherwise.
    """
    source_scope = UsdGeom.Scope.Get(stage, source_scope_path)
    dest_scope = UsdGeom.Scope.Get(stage, dest_scope_path)
    if not source_scope or not dest_scope:
        print(f"Error: Invalid scope path(s): {source_scope_path}, {dest_scope_path}")
        return False
    children = source_scope.GetPrim().GetChildren()
    for child in children:
        child_path = child.GetPath()
        dest_child_path = dest_scope.GetPath().AppendChild(child.GetName())
        if stage.GetPrimAtPath(dest_child_path):
            print(f"Warning: Destination path already exists: {dest_child_path}. Skipping.")
            continue
        stage.RemovePrim(child_path)
        stage.DefinePrim(dest_child_path)
    return True


def duplicate_sphere(stage: Usd.Stage, sphere_path: str, dup_path: str) -> UsdGeom.Sphere:
    """Duplicate a UsdGeomSphere prim.

    Args:
        stage (Usd.Stage): The stage containing the sphere prim.
        sphere_path (str): The path of the sphere prim to duplicate.
        dup_path (str): The path for the duplicated sphere prim.

    Returns:
        UsdGeom.Sphere: The duplicated sphere prim.

    Raises:
        ValueError: If the input sphere prim is not a valid UsdGeomSphere.
    """
    sphere_prim = stage.GetPrimAtPath(sphere_path)
    if not sphere_prim.IsValid() or not UsdGeom.Sphere(sphere_prim):
        raise ValueError(f"Prim at path {sphere_path} is not a valid UsdGeomSphere.")
    radius = UsdGeom.Sphere(sphere_prim).GetRadiusAttr().Get()
    extent = UsdGeom.Sphere(sphere_prim).GetExtentAttr().Get()
    dup_sphere = UsdGeom.Sphere.Define(stage, dup_path)
    dup_sphere.GetRadiusAttr().Set(radius)
    dup_sphere.GetExtentAttr().Set(extent)
    return dup_sphere


def scale_sphere(sphere: UsdGeom.Sphere, scale: Gf.Vec3f) -> None:
    """Scale a USD Sphere by adjusting its radius and extent.

    Args:
        sphere (UsdGeom.Sphere): The Sphere prim to scale.
        scale (Gf.Vec3f): The scale factor to apply.
    """
    if not sphere or not sphere.GetPrim().IsValid():
        raise ValueError("Invalid Sphere prim.")
    if any((s <= 0 for s in scale)):
        raise ValueError("Scale values must be greater than zero.")
    radius_attr = sphere.GetRadiusAttr()
    extent_attr = sphere.GetExtentAttr()
    radius = radius_attr.Get()
    extent = extent_attr.Get()
    new_radius = radius * max(scale)
    radius_attr.Set(new_radius)
    new_extent = [(e[0] * scale[0], e[1] * scale[1], e[2] * scale[2]) for e in extent]
    extent_attr.Set(new_extent)


def find_spheres_by_radius(stage: Usd.Stage, radius: float, tolerance: float = 0.001) -> List[Usd.Prim]:
    """Find all UsdGeomSphere prims on the stage with a specific radius.

    Args:
        stage (Usd.Stage): The USD stage to search.
        radius (float): The target radius value.
        tolerance (float, optional): The tolerance for comparing radii. Defaults to 0.001.

    Returns:
        List[Usd.Prim]: A list of UsdGeomSphere prims with the specified radius.
    """
    sphere_prims: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Sphere):
            sphere_prims.append(prim)
    result_prims: List[Usd.Prim] = []
    for sphere_prim in sphere_prims:
        sphere = UsdGeom.Sphere(sphere_prim)
        prim_radius = sphere.GetRadiusAttr().Get()
        if abs(prim_radius - radius) <= tolerance:
            result_prims.append(sphere_prim)
    return result_prims


def animate_sphere_radius(
    sphere_prim: UsdGeom.Sphere, time_code_start: float, time_code_end: float, radius_start: float, radius_end: float
):
    """Animates the radius of a UsdGeom.Sphere over a given time range.

    Args:
        sphere_prim (UsdGeom.Sphere): The sphere prim to animate.
        time_code_start (float): The starting time code for the animation.
        time_code_end (float): The ending time code for the animation.
        radius_start (float): The starting radius value.
        radius_end (float): The ending radius value.
    """
    if not sphere_prim or not isinstance(sphere_prim, UsdGeom.Sphere):
        raise ValueError("Invalid sphere prim provided.")
    radius_attr = sphere_prim.GetRadiusAttr()
    if not radius_attr:
        raise ValueError("Failed to get radius attribute from the sphere prim.")
    radius_attr.Set(radius_start, Usd.TimeCode(time_code_start))
    radius_attr.Set(radius_end, Usd.TimeCode(time_code_end))


def set_sphere_radius(stage: Usd.Stage, prim_path: str, radius: float):
    """Set the radius for a UsdGeomSphere prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path: {prim_path}")
    sphere = UsdGeom.Sphere(prim)
    if not sphere:
        raise ValueError(f"Prim at path {prim_path} is not a valid UsdGeomSphere")
    radius_attr = sphere.GetRadiusAttr()
    if not radius_attr:
        radius_attr = sphere.CreateRadiusAttr()
    try:
        radius_attr.Set(radius)
    except Exception as e:
        raise ValueError(f"Failed to set radius: {str(e)}")
    extent_attr = sphere.GetExtentAttr()
    if not extent_attr:
        extent_attr = sphere.CreateExtentAttr()
    min_extent = Gf.Vec3f(-radius, -radius, -radius)
    max_extent = Gf.Vec3f(radius, radius, radius)
    try:
        extent_attr.Set([min_extent, max_extent])
    except Exception as e:
        raise ValueError(f"Failed to set extent: {str(e)}")


def align_spheres(stage: Usd.Stage, sphere_a_path: str, sphere_b_path: str) -> None:
    """
    Align two spheres in the scene so that they are tangent to each other.

    Args:
        stage (Usd.Stage): The USD stage containing the spheres.
        sphere_a_path (str): The path to the first sphere prim.
        sphere_b_path (str): The path to the second sphere prim.
    """
    sphere_a = UsdGeom.Sphere(stage.GetPrimAtPath(sphere_a_path))
    sphere_b = UsdGeom.Sphere(stage.GetPrimAtPath(sphere_b_path))
    if not sphere_a or not sphere_b:
        raise ValueError("Invalid sphere prims provided.")
    radius_a = sphere_a.GetRadiusAttr().Get()
    radius_b = sphere_b.GetRadiusAttr().Get()
    distance = radius_a + radius_b
    xform_a = UsdGeom.Xformable(sphere_a)
    translate_op_a = add_translate_op(xform_a)
    translate_a = translate_op_a.Get()
    xform_b = UsdGeom.Xformable(sphere_b)
    translate_op_b = add_translate_op(xform_b)
    if translate_a is None:
        translate_a = Gf.Vec3d(0, 0, 0)
    translate_op_b.Set(translate_a + Gf.Vec3d(distance, 0, 0))


def assign_material_to_subset(subset: UsdGeom.Subset, material: UsdShade.Material) -> None:
    """Assign a material to a GeomSubset.

    Args:
        subset (UsdGeom.Subset): The GeomSubset to assign the material to.
        material (UsdShade.Material): The material to assign to the subset.

    Raises:
        ValueError: If the subset or material is invalid.
    """
    if not subset or not subset.GetPrim().IsValid():
        raise ValueError("Invalid GeomSubset.")
    if not material or not material.GetPrim().IsValid():
        raise ValueError("Invalid material.")
    UsdShade.MaterialBindingAPI(subset.GetPrim()).Bind(material)


def create_cube_subset(stage: Usd.Stage, cube_path: str, subset_name: str, indices: Vt.IntArray) -> UsdGeom.Subset:
    """Create a cube subset with the given name and indices."""
    cube = stage.GetPrimAtPath(cube_path)
    if not cube:
        raise ValueError(f"Cube prim not found at path: {cube_path}")
    subset = UsdGeom.Subset.CreateGeomSubset(
        UsdGeom.Imageable(cube), subset_name, "face", indices, UsdGeom.Tokens.partition, ""
    )
    return subset


def get_subset_family_names(geom: UsdGeom.Imageable) -> Set[str]:
    """
    Get the names of all the families of GeomSubsets defined on the given imageable prim.

    Args:
        geom (UsdGeom.Imageable): The imageable prim to get the subset family names from.

    Returns:
        Set[str]: A set of family names for the GeomSubsets defined on the prim.
    """
    if not geom.GetPrim().IsValid():
        raise ValueError("Invalid prim given as input.")
    children = geom.GetPrim().GetChildren()
    family_names = set()
    for child in children:
        if UsdGeom.Subset.Get(child.GetStage(), child.GetPath()):
            family_attr = UsdGeom.Subset.Get(child.GetStage(), child.GetPath()).GetFamilyNameAttr()
            if family_attr.IsValid():
                family_names.add(family_attr.Get())
    return family_names


def update_subset_indices(subset: UsdGeom.Subset, indices: Vt.IntArray) -> None:
    """Update the indices of a GeomSubset.

    Args:
        subset (UsdGeom.Subset): The GeomSubset to update indices for.
        indices (Vt.IntArray): The new indices to set on the subset.

    Raises:
        ValueError: If the provided subset is not valid.
    """
    if not subset.GetPrim().IsValid():
        raise ValueError("Provided subset is not valid.")
    attr = subset.GetIndicesAttr()
    if not attr.IsValid():
        attr = subset.CreateIndicesAttr()
    attr.Set(indices)


def get_all_subsets_in_family(geom: UsdGeom.Imageable, family_name: str) -> List[UsdGeom.Subset]:
    """
    Get all GeomSubsets belonging to a specific family on a given imageable prim.

    Args:
        geom (UsdGeom.Imageable): The imageable prim to get subsets from.
        family_name (str): The name of the family to filter subsets by.

    Returns:
        List[UsdGeom.Subset]: A list of GeomSubsets belonging to the specified family.
    """
    if not geom.GetPrim().IsValid():
        raise ValueError(f"Invalid prim: {geom.GetPrim().GetPath()}")
    if not isinstance(geom, UsdGeom.Imageable):
        raise TypeError(f"Prim {geom.GetPrim().GetPath()} is not an Imageable")
    all_subsets = UsdGeom.Subset.GetAllGeomSubsets(geom)
    family_subsets = [subset for subset in all_subsets if subset.GetFamilyNameAttr().Get() == family_name]
    return family_subsets


def create_subset(stage, parent_path, subset_name, indices, family_name):
    parent_prim = stage.GetPrimAtPath(parent_path)
    UsdGeom.Subset.CreateGeomSubset(
        UsdGeom.Imageable(parent_prim), subset_name, "face", Vt.IntArray(indices), familyName=family_name
    )


def get_subset_element_type(subset: UsdGeom.Subset) -> str:
    """Get the element type of a GeomSubset prim."""
    if not subset or not subset.GetPrim().IsValid():
        raise ValueError("Invalid GeomSubset prim.")
    element_type_attr = subset.GetElementTypeAttr()
    if not element_type_attr.IsValid():
        raise ValueError("GeomSubset prim is missing the elementType attribute.")
    element_type = element_type_attr.Get()
    if not element_type:
        element_type = UsdGeom.Tokens.face
    return str(element_type)


def create_face_subset(geom: UsdGeom.Imageable, subset_name: str, indices: Vt.IntArray) -> UsdGeom.Subset:
    """Create a face subset on the given geometry prim with the specified name and indices."""
    prim = geom.GetPrim()
    stage = prim.GetStage()
    subset_path = prim.GetPath().AppendChild(subset_name)
    if stage.GetPrimAtPath(subset_path):
        subset = UsdGeom.Subset(stage.GetPrimAtPath(subset_path))
        subset.GetIndicesAttr().Set(indices)
    else:
        subset = UsdGeom.Subset.Define(stage, subset_path)
        subset.CreateElementTypeAttr().Set("face")
        subset.CreateIndicesAttr().Set(indices)
    return subset


def get_unassigned_indices(
    subsets: List[UsdGeom.Subset], element_count: int, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> Vt.IntArray:
    """
    Get the list of indices that are not assigned to any of the GeomSubsets.

    Args:
        subsets (List[UsdGeom.Subset]): List of GeomSubset prims.
        element_count (int): Total number of elements in the array being subdivided.
        time_code (Usd.TimeCode): TimeCode at which to evaluate the indices. Defaults to Default.

    Returns:
        Vt.IntArray: Array of unassigned indices.
    """
    assigned_indices = set()
    for subset in subsets:
        indices_attr = subset.GetIndicesAttr()
        if indices_attr.IsValid() and indices_attr.HasValue():
            indices = indices_attr.Get(time_code)
            assigned_indices.update(indices)
    unassigned_indices = []
    for i in range(element_count):
        if i not in assigned_indices:
            unassigned_indices.append(i)
    return Vt.IntArray(unassigned_indices)


def set_subset_family_type(geom: UsdGeom.Imageable, family_name: str, family_type: str) -> bool:
    """
    Set the family type for a family of GeomSubsets on a given imageable prim.

    Args:
        geom (UsdGeom.Imageable): The imageable prim to set the family type on.
        family_name (str): The name of the family to set the type for.
        family_type (str): The type of the family. Must be one of "partition", "nonOverlapping", or "unrestricted".

    Returns:
        bool: True if the family type was set successfully, False otherwise.
    """
    if not geom:
        print(f"Error: {geom} is not a valid UsdGeom.Imageable prim.")
        return False
    if not UsdGeom.Imageable(geom):
        print(f"Error: {geom} is not a valid UsdGeom.Imageable prim.")
        return False
    if not family_name:
        print("Error: family_name cannot be empty.")
        return False
    if family_type not in ["partition", "nonOverlapping", "unrestricted"]:
        print(
            f"Error: Invalid family_type '{family_type}'. Must be one of 'partition', 'nonOverlapping', or 'unrestricted'."
        )
        return False
    try:
        UsdGeom.Subset.SetFamilyType(geom, family_name, family_type)
    except Exception as e:
        print(f"Error setting family type: {str(e)}")
        return False
    return True


def set_subset_family_name(geom_prim: UsdGeom.Imageable, subset_name: str, family_name: str) -> None:
    """Set the family name for a GeomSubset on a given prim."""
    if not geom_prim.GetPrim().IsValid():
        raise ValueError("Geom prim is not valid.")
    subset_prim = geom_prim.GetPrim().GetStage().GetPrimAtPath(geom_prim.GetPrim().GetPath().AppendChild(subset_name))
    if not subset_prim.IsValid():
        raise ValueError(f"GeomSubset {subset_name} does not exist under prim {geom_prim.GetPrim().GetPath()}")
    subset = UsdGeom.Subset(subset_prim)
    family_name_attr = subset.GetFamilyNameAttr()
    if not family_name_attr:
        family_name_attr = subset.CreateFamilyNameAttr()
    family_name_attr.Set(family_name)


def validate_all_subsets(subsets: List[UsdGeom.Subset], elementCount: int, familyType: str) -> Tuple[bool, str]:
    """
    Validates the data in the given set of GeomSubsets.

    Args:
        subsets (List[UsdGeom.Subset]): The list of GeomSubsets to validate.
        elementCount (int): The total number of elements in the array being subdivided.
        familyType (str): The type of family the subsets belong to.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating the validity of the subsets
                          and a string containing the reason if they're invalid.
    """
    element_types = set((subset.GetElementTypeAttr().Get() for subset in subsets))
    if len(element_types) > 1:
        return (False, "Subsets have different element types")
    all_indices = []
    for subset in subsets:
        indices = subset.GetIndicesAttr().Get()
        all_indices.extend(indices)
    if any((index < 0 or index >= elementCount for index in all_indices)):
        return (False, "Indices out of range")
    if familyType in [UsdGeom.Tokens.partition, UsdGeom.Tokens.nonOverlapping]:
        if len(set(all_indices)) != len(all_indices):
            return (False, "Duplicate indices found")
    if familyType == UsdGeom.Tokens.partition:
        if set(all_indices) != set(range(elementCount)):
            return (False, "Not all elements are covered")
    return (True, "")


def get_subset_indices(subset: UsdGeom.Subset, time_code: Usd.TimeCode = Usd.TimeCode.Default()) -> List[int]:
    """
    Get the indices for a GeomSubset at a specific time.

    Args:
        subset (UsdGeom.Subset): The GeomSubset to retrieve indices from.
        time_code (Usd.TimeCode): The time at which to retrieve the indices. Defaults to Default.

    Returns:
        List[int]: The indices of the GeomSubset at the specified time.

    Raises:
        ValueError: If the provided subset is not a valid UsdGeom.Subset.
    """
    if not subset or not isinstance(subset, UsdGeom.Subset):
        raise ValueError("Provided subset is not a valid UsdGeom.Subset.")
    indices_attr = subset.GetIndicesAttr()
    if not indices_attr:
        return []
    indices = indices_attr.Get(time_code)
    if indices is None:
        return []
    return list(indices)


def remove_subset(geom: UsdGeom.Imageable, subset_name: str) -> bool:
    """Remove a GeomSubset from the given imageable prim.

    Args:
        geom (UsdGeom.Imageable): The imageable prim to remove the subset from.
        subset_name (str): The name of the subset to remove.

    Returns:
        bool: True if the subset was successfully removed, False otherwise.
    """
    if not geom.GetPrim().IsValid():
        return False
    subset_prim = geom.GetPrim().GetChild(subset_name)
    if not subset_prim.IsValid():
        return False
    stage = subset_prim.GetStage()
    stage.RemovePrim(subset_prim.GetPath())
    return True


def merge_subsets(subsets: List[UsdGeom.Subset], merged_subset_name: str) -> UsdGeom.Subset:
    """Merges a list of UsdGeom.Subsets into a single subset.

    Args:
        subsets (List[UsdGeom.Subset]): The list of subsets to merge.
        merged_subset_name (str): The name to give the merged subset.

    Returns:
        UsdGeom.Subset: The merged subset.

    Raises:
        ValueError: If the subsets list is empty or the subsets have inconsistent element types.
    """
    if not subsets:
        raise ValueError("The subsets list is empty.")
    parent_prim = subsets[0].GetPrim().GetParent()
    element_type = subsets[0].GetElementTypeAttr().Get()
    for subset in subsets[1:]:
        if subset.GetPrim().GetParent() != parent_prim:
            raise ValueError("All subsets must have the same parent prim.")
        if subset.GetElementTypeAttr().Get() != element_type:
            raise ValueError("All subsets must have the same element type.")
    merged_indices = []
    for subset in subsets:
        merged_indices.extend(subset.GetIndicesAttr().Get())
    merged_subset = UsdGeom.Subset.CreateUniqueGeomSubset(
        UsdGeom.Imageable(parent_prim), merged_subset_name, element_type, Vt.IntArray(merged_indices), "", ""
    )
    return merged_subset


def split_subset(subset: UsdGeom.Subset, element_indices: List[int]) -> Tuple[UsdGeom.Subset, UsdGeom.Subset]:
    """
    Split a GeomSubset into two subsets based on the provided element indices.

    Args:
        subset (UsdGeom.Subset): The GeomSubset to split.
        element_indices (List[int]): The list of element indices to split out into a new subset.

    Returns:
        Tuple[UsdGeom.Subset, UsdGeom.Subset]: A tuple containing the two resulting subsets after the split.
                                                The first subset is the original subset with elements removed,
                                                and the second subset is a new subset with the split out elements.
    """
    if not subset.GetPrim().IsValid():
        raise ValueError("Input GeomSubset is not valid.")
    original_indices = subset.GetIndicesAttr().Get()
    if not all((0 <= i < len(original_indices) for i in element_indices)):
        raise ValueError("One or more element indices are out of range.")
    geom = UsdGeom.Imageable(subset.GetPrim().GetParent())
    new_subset_name = f"{subset.GetPrim().GetName()}_split"
    new_subset = UsdGeom.Subset.CreateUniqueGeomSubset(
        geom,
        new_subset_name,
        subset.GetElementTypeAttr().Get(),
        Vt.IntArray([original_indices[i] for i in element_indices]),
        subset.GetFamilyNameAttr().Get(),
        UsdGeom.Subset.GetFamilyType(geom, subset.GetFamilyNameAttr().Get()),
    )
    remaining_indices = [i for i in original_indices if i not in [original_indices[j] for j in element_indices]]
    subset.GetIndicesAttr().Set(remaining_indices)
    return (subset, new_subset)


def change_subset_name(stage: Usd.Stage, subset_prim_path: str, new_name: str) -> bool:
    """Change the name of a GeomSubset prim.

    Args:
        stage (Usd.Stage): The stage containing the GeomSubset prim.
        subset_prim_path (str): The path to the GeomSubset prim.
        new_name (str): The new name for the GeomSubset prim.

    Returns:
        bool: True if the rename was successful, False otherwise.
    """
    subset_prim = stage.GetPrimAtPath(subset_prim_path)
    if not subset_prim.IsValid():
        print(f"Error: Invalid prim path: {subset_prim_path}")
        return False
    if not subset_prim.IsA(UsdGeom.Subset):
        print(f"Error: Prim at path {subset_prim_path} is not a GeomSubset")
        return False
    parent_prim = subset_prim.GetParent()
    if not parent_prim.IsValid():
        print(f"Error: GeomSubset prim {subset_prim_path} has no parent")
        return False
    new_subset_path = parent_prim.GetPath().AppendChild(new_name)
    if stage.GetPrimAtPath(new_subset_path):
        print(f"Error: A prim with name {new_name} already exists under {parent_prim.GetPath()}")
        return False
    try:
        subset_prim.SetAssetInfoByKey("name", new_name)
        stage.DefinePrim(new_subset_path)
        stage.RemovePrim(subset_prim.GetPath())
    except Tf.ErrorException as e:
        print(f"Error: Failed to rename prim: {e}")
        return False
    return True


def get_prim_purposes() -> Tuple[str, ...]:
    """Get the standard prim purpose tokens.

    These tokens are useful for categorizing prims by their intended usage. See
    UsdGeom schema for more details.

    Returns:
        Tuple[str, ...]: A tuple of standard purpose token strings.
    """
    purposes = (UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide)
    return purposes


def set_attribute_from_token(prim: Usd.Prim, token: str, value: Any) -> None:
    """Set an attribute on a prim using a token from UsdGeom.Tokens.

    Args:
        prim (Usd.Prim): The prim to set the attribute on.
        token (str): The token identifying the attribute to set.
        value (Any): The value to set for the attribute.

    Raises:
        ValueError: If the prim is not valid or the token is not recognized.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if not hasattr(UsdGeom.Tokens, token):
        raise ValueError(f"Unrecognized token: {token}")
    attr = prim.GetAttribute(getattr(UsdGeom.Tokens, token))
    if not attr.IsValid():
        if token == "visibility":
            value_type = Sdf.ValueTypeNames.Token
        elif token == "purpose":
            value_type = Sdf.ValueTypeNames.Token
        elif token == "subdivisionScheme":
            value_type = Sdf.ValueTypeNames.Token
        else:
            value_type = Sdf.ValueTypeNames.Find(type(value).__name__)
        attr = prim.CreateAttribute(getattr(UsdGeom.Tokens, token), value_type)
    attr.Set(value)


def batch_set_geom_properties(stage: Usd.Stage, prim_paths: List[str], properties: Dict[str, Dict[str, Any]]) -> None:
    """Batch set geometry properties on multiple prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths.
        properties (Dict[str, Dict[str, Any]]): A dictionary of property names and values.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        for prop_name, prop_data in properties.items():
            if not prim.HasAttribute(prop_name):
                attr = prim.CreateAttribute(prop_name, prop_data["type"])
            else:
                attr = prim.GetAttribute(prop_name)
            if not attr.IsValid():
                raise ValueError(f"Failed to get or create attribute {prop_name} on prim at path {prim_path}.")
            if attr.GetTypeName() == Sdf.ValueTypeNames.Float:
                attr.Set(prop_data["value"])
            elif attr.GetTypeName() == Sdf.ValueTypeNames.Double:
                attr.Set(prop_data["value"])
            elif attr.GetTypeName() == Sdf.ValueTypeNames.Float3:
                attr.Set(Gf.Vec3f(*prop_data["value"]))
            elif attr.GetTypeName() == Sdf.ValueTypeNames.Color3f:
                attr.Set(Gf.Vec3f(*prop_data["value"]))
            elif attr.GetTypeName() == Sdf.ValueTypeNames.Color3fArray:
                attr.Set([Gf.Vec3f(*color) for color in prop_data["value"]])
            elif attr.GetTypeName() == Sdf.ValueTypeNames.Token:
                attr.Set(prop_data["value"])
            else:
                raise ValueError(f"Unsupported attribute type {attr.GetTypeName()} for attribute {prop_name}.")


def toggle_visibility_for_purpose(prim: Usd.Prim, purpose: str) -> bool:
    """Toggle the visibility of a prim for a specific purpose.

    Args:
        prim (Usd.Prim): The prim to toggle visibility for.
        purpose (str): The purpose to toggle visibility for ("guide", "proxy", or "render").

    Returns:
        bool: True if the visibility was toggled, False otherwise.
    """
    if not prim.IsValid():
        return False
    visibility_api = UsdGeom.VisibilityAPI(prim)
    if not visibility_api:
        return False
    visibility_attr = visibility_api.GetPurposeVisibilityAttr(purpose)
    if not visibility_attr:
        return False
    current_visibility = visibility_attr.Get()
    if current_visibility == UsdGeom.Tokens.invisible:
        visibility_attr.Set(UsdGeom.Tokens.inherited)
    elif current_visibility == UsdGeom.Tokens.inherited:
        visibility_attr.Set(UsdGeom.Tokens.visible)
    else:
        visibility_attr.Set(UsdGeom.Tokens.invisible)
    return True


def get_combined_visibility(prim: UsdGeom.Imageable, purpose: str) -> str:
    """
    Get the combined visibility for a prim based on its purpose and inherited visibility.

    Args:
        prim (UsdGeom.Imageable): The prim to get the combined visibility for.
        purpose (str): The purpose to consider ('default', 'render', 'proxy', or 'guide').

    Returns:
        str: The combined visibility ('inherited', 'invisible', or 'visible').
    """
    inherited_vis = prim.ComputeVisibility()
    if inherited_vis == "invisible":
        return "invisible"
    if purpose == "default":
        return inherited_vis
    purpose_vis_attr = UsdGeom.VisibilityAPI(prim).GetPurposeVisibilityAttr(purpose)
    if not purpose_vis_attr.IsValid():
        return inherited_vis
    purpose_vis = purpose_vis_attr.Get()
    if purpose_vis is None:
        return inherited_vis
    return purpose_vis


def reset_visibility_to_inherited(prim: Usd.Prim, purpose: Optional[str] = None):
    """Resets the visibility of the given prim to 'inherited' for the specified purpose.

    If no purpose is provided, resets visibility for all purposes.
    """
    vis_api = UsdGeom.VisibilityAPI(prim)
    if not vis_api:
        raise ValueError(f"Prim at path {prim.GetPath()} does not have VisibilityAPI applied.")
    if purpose is None:
        purposes = ["guide", "proxy", "render"]
    else:
        if purpose not in ["guide", "proxy", "render"]:
            raise ValueError(f"Invalid purpose: {purpose}. Must be one of 'guide', 'proxy', or 'render'.")
        purposes = [purpose]
    for purpose in purposes:
        vis_attr = vis_api.GetPurposeVisibilityAttr(purpose)
        if vis_attr.HasAuthoredValue():
            vis_attr.Set("inherited")


def bulk_set_render_visibility(prims: List[Usd.Prim], visibility: str):
    """Set the render visibility for multiple prims in bulk.

    Args:
        prims (List[Usd.Prim]): A list of prims to set the render visibility for.
        visibility (str): The visibility value to set. Must be one of "inherited", "invisible", or "visible".

    Raises:
        ValueError: If an invalid visibility value is provided.
    """
    if visibility not in ["inherited", "invisible", "visible"]:
        raise ValueError(
            f"Invalid visibility value: {visibility}. Must be one of 'inherited', 'invisible', or 'visible'."
        )
    for prim in prims:
        if not prim.IsValid():
            continue
        vis_api = UsdGeom.VisibilityAPI.Apply(prim)
        render_vis_attr = vis_api.GetRenderVisibilityAttr()
        render_vis_attr.Set(visibility)


def create_and_apply_visibility_api(prim: Usd.Prim) -> UsdGeom.VisibilityAPI:
    """Create and apply UsdGeom.VisibilityAPI to a prim.

    Args:
        prim (Usd.Prim): The prim to apply the VisibilityAPI to.

    Returns:
        UsdGeom.VisibilityAPI: The applied VisibilityAPI schema.

    Raises:
        ValueError: If the prim is not a valid UsdGeomImageable prim.
    """
    if not UsdGeom.Imageable(prim):
        raise ValueError(f"Prim {prim.GetPath()} is not a valid UsdGeomImageable.")
    if not UsdGeom.VisibilityAPI.CanApply(prim):
        raise ValueError(f"VisibilityAPI cannot be applied to prim {prim.GetPath()}.")
    visibility_api = UsdGeom.VisibilityAPI.Apply(prim)
    visibility_api.CreateGuideVisibilityAttr(UsdGeom.Tokens.invisible, True)
    visibility_api.CreateProxyVisibilityAttr(UsdGeom.Tokens.inherited, True)
    visibility_api.CreateRenderVisibilityAttr(UsdGeom.Tokens.inherited, True)
    return visibility_api


def filter_prims_by_visibility(stage: Usd.Stage, prim_paths: List[str], visibility: str) -> List[str]:
    """
    Filter a list of prim paths based on their visibility.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The list of prim paths to filter.
        visibility (str): The visibility value to filter by ("visible" or "invisible").

    Returns:
        List[str]: The filtered list of prim paths.
    """
    filtered_paths = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        vis_api = UsdGeom.VisibilityAPI(prim)
        if not vis_api:
            attr = prim.GetAttribute("visibility")
            if attr.IsValid():
                vis_value = attr.Get()
                if vis_value == visibility:
                    filtered_paths.append(prim_path)
            continue
        purpose = prim.GetPurpose()
        if purpose == UsdGeom.Tokens.guide:
            attr = vis_api.GetGuideVisibilityAttr()
        elif purpose == UsdGeom.Tokens.proxy:
            attr = vis_api.GetProxyVisibilityAttr()
        elif purpose == UsdGeom.Tokens.render:
            attr = vis_api.GetRenderVisibilityAttr()
        else:
            attr = prim.GetAttribute("visibility")
        if attr.IsValid():
            vis_value = attr.Get()
            if vis_value == visibility:
                filtered_paths.append(prim_path)
    return filtered_paths


def apply_visibility_to_prim_and_descendants(prim: Usd.Prim, visibility: str):
    """Apply the given visibility to the prim and all its descendants.

    Args:
        prim (Usd.Prim): The prim to apply visibility to.
        visibility (str): The visibility value to apply. Must be one of
            "inherited", "invisible", or "visible".

    Raises:
        ValueError: If the given visibility is not a valid value.
    """
    if visibility not in ["inherited", "invisible", "visible"]:
        raise ValueError(f"Invalid visibility value: {visibility}")
    imageable = UsdGeom.Imageable(prim)
    imageable.CreateVisibilityAttr().Set(visibility)
    for child in prim.GetAllChildren():
        apply_visibility_to_prim_and_descendants(child, visibility)


def get_prims_with_visibility(stage: Usd.Stage, visibility: str) -> List[Usd.Prim]:
    """
    Get all prims on the stage with the specified visibility.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        visibility (str): The visibility value to match ("inherited", "invisible", or "visible").

    Returns:
        List[Usd.Prim]: A list of prims with the specified visibility.
    """
    if visibility not in ["inherited", "invisible", "visible"]:
        raise ValueError(f"Invalid visibility value: {visibility}")
    prims_with_visibility = []
    for prim in stage.TraverseAll():
        if prim.HasAttribute("visibility"):
            visibility_attr = prim.GetAttribute("visibility")
            if visibility_attr.Get() == visibility:
                prims_with_visibility.append(prim)
    return prims_with_visibility


def create_prim_with_visibility(stage, path, visibility):
    prim = stage.DefinePrim(path)
    prim.CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set(visibility)
    return prim


def copy_visibility_attributes(source_prim: UsdGeom.Imageable, dest_prim: UsdGeom.Imageable):
    """Copy visibility attributes from source_prim to dest_prim."""
    if not source_prim or not dest_prim:
        raise ValueError("Both source_prim and dest_prim must be valid UsdGeom.Imageable prims.")
    source_vis_attr = source_prim.GetVisibilityAttr()
    if source_vis_attr.HasAuthoredValue():
        dest_prim.CreateVisibilityAttr().Set(source_vis_attr.Get())
    for purpose in ["guide", "proxy", "render"]:
        source_purpose_vis_attr = UsdGeom.VisibilityAPI(source_prim).GetPurposeVisibilityAttr(purpose)
        if source_purpose_vis_attr.HasAuthoredValue():
            dest_vis_api = UsdGeom.VisibilityAPI(dest_prim)
            dest_purpose_vis_attr = dest_vis_api.GetPurposeVisibilityAttr(purpose)
            if not dest_purpose_vis_attr.HasAuthoredValue():
                if purpose == "guide":
                    dest_purpose_vis_attr = dest_vis_api.CreateGuideVisibilityAttr()
                elif purpose == "proxy":
                    dest_purpose_vis_attr = dest_vis_api.CreateProxyVisibilityAttr()
                elif purpose == "render":
                    dest_purpose_vis_attr = dest_vis_api.CreateRenderVisibilityAttr()
            dest_purpose_vis_attr.Set(source_purpose_vis_attr.Get())


def synchronize_visibility_across_purposes(prim: Usd.Prim) -> None:
    """Synchronize visibility attributes across purposes for a prim.

    If the prim has a defined visibility, it will be propagated to the
    purpose-specific visibility attributes (guide, proxy, render).
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        return
    visibility_attr = imageable.GetVisibilityAttr()
    if not visibility_attr.HasAuthoredValue():
        return
    visibility = visibility_attr.Get()
    for purpose in ["guide", "proxy", "render"]:
        purpose_vis_attr = UsdGeom.VisibilityAPI(prim).GetPurposeVisibilityAttr(purpose)
        if not purpose_vis_attr.HasAuthoredValue():
            purpose_vis_attr.Set(visibility)


def copy_xform_hierarchy(source_stage: Usd.Stage, source_root_path: str, dest_stage: Usd.Stage, dest_root_path: str):
    """
    Recursively copy the Xform hierarchy from one stage to another.

    Args:
        source_stage (Usd.Stage): The source stage to copy from.
        source_root_path (str): The path to the root prim of the hierarchy to copy in the source stage.
        dest_stage (Usd.Stage): The destination stage to copy to.
        dest_root_path (str): The path where the copied hierarchy will be rooted in the destination stage.
    """
    source_prim = source_stage.GetPrimAtPath(source_root_path)
    if not source_prim.IsValid():
        raise ValueError(f"Invalid source prim path: {source_root_path}")
    dest_prim = dest_stage.DefinePrim(dest_root_path)
    if not dest_prim.IsValid():
        raise ValueError(f"Failed to create destination prim at path: {dest_root_path}")
    source_xform = UsdGeom.Xformable(source_prim)
    dest_xform = UsdGeom.Xformable(dest_prim)
    if source_xform and dest_xform:
        for op in source_xform.GetOrderedXformOps():
            op_type = op.GetOpType()
            op_attr = op.GetAttr()
            if op_attr.HasValue():
                value = op_attr.Get()
                dest_xform.AddXformOp(op_type, op_attr.GetName()).Set(value)
    for child in source_prim.GetChildren():
        if child.IsA(UsdGeom.Xform):
            child_name = child.GetName()
            dest_child_path = dest_prim.GetPath().AppendChild(child_name)
            copy_xform_hierarchy(source_stage, child.GetPath(), dest_stage, dest_child_path)


def get_world_transform(stage: Usd.Stage, prim_path: str) -> Gf.Matrix4d:
    """
    Get the world transformation matrix for a prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.

    Returns:
        Gf.Matrix4d: The world transformation matrix for the prim.

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
    while parent_prim.IsValid():
        parent_xformable = UsdGeom.Xformable(parent_prim)
        if parent_xformable:
            parent_transform = parent_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            local_transform = local_transform * parent_transform
        parent_prim = parent_prim.GetParent()
    return local_transform


def create_and_align_xform(
    stage: Usd.Stage, prim_path: str, position: Gf.Vec3f, rotation: Gf.Quatf, scale: Gf.Vec3f
) -> UsdGeom.Xform:
    """Create an Xform prim and set its transform properties.

    Args:
        stage (Usd.Stage): The stage to create the prim on.
        prim_path (str): The path where the prim will be created.
        position (Gf.Vec3f): The position of the prim.
        rotation (Gf.Quatf): The rotation of the prim.
        scale (Gf.Vec3f): The scale of the prim.

    Returns:
        UsdGeom.Xform: The created Xform prim.
    """
    xform = UsdGeom.Xform.Define(stage, prim_path)
    add_translate_op(xform).Set(position)
    add_orient_op(xform).Set(rotation)
    add_scale_op(xform).Set(scale)
    return xform


def bake_transform_to_vertices(stage: Usd.Stage, prim_path: str):
    """Bake the local transformation of a prim into its point values.

    This function takes a USD stage and a prim path, retrieves the Mesh prim at that path,
    and bakes the prim's local transformation into its point values. The original
    transformation is then reset to identity.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the mesh prim.

    Raises:
        ValueError: If the prim is not a valid mesh.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"No prim found at path: {prim_path}")
    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        raise ValueError(f"Prim at path {prim_path} is not a valid Mesh")
    xformable = UsdGeom.Xformable(prim)
    local_transform = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
    transformed_points = [local_transform.Transform(point) for point in points]
    mesh.GetPointsAttr().Set(transformed_points)
    xform_ops = xformable.GetOrderedXformOps()
    for op in xform_ops:
        op.Clear()


def get_transforms_for_prims(stage: Usd.Stage, prim_paths: List[str]) -> Dict[str, Gf.Matrix4d]:
    """
    Get the local-to-world transforms for a list of prims using XformCache.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths.

    Returns:
        Dict[str, Gf.Matrix4d]: A dictionary mapping prim paths to their local-to-world transforms.
    """
    xfCache = UsdGeom.XformCache(0)
    transforms = {}
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping...")
            continue
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            print(f"Warning: Prim at path {prim_path} is not transformable. Skipping...")
            continue
        transform = xfCache.GetLocalToWorldTransform(prim)
        transforms[prim_path] = transform
    return transforms


def get_transform_at_time(prim: Usd.Prim, time_code: Usd.TimeCode, xform_cache: UsdGeom.XformCache) -> Gf.Matrix4d:
    """
    Get the local-to-world transformation matrix for a prim at a specific time.

    Args:
        prim (Usd.Prim): The prim to get the transform for.
        time_code (Usd.TimeCode): The time code to get the transform at.
        xform_cache (UsdGeom.XformCache): The XformCache to use for caching transforms.

    Returns:
        Gf.Matrix4d: The local-to-world transformation matrix for the prim at the specified time.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if not prim.IsA(UsdGeom.Xformable):
        raise ValueError(f"Prim {prim.GetPath()} is not transformable")
    xform_cache.SetTime(time_code)
    transform = xform_cache.GetLocalToWorldTransform(prim)
    return transform


def get_prim_world_position(stage: Usd.Stage, prim_path: str) -> Tuple[float, float, float]:
    """
    Get the world space position of a prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.

    Returns:
        Tuple[float, float, float]: The world space position of the prim.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xform_cache = UsdGeom.XformCache(0.0)
    local_to_world_xform = xform_cache.GetLocalToWorldTransform(prim)
    world_position = local_to_world_xform.ExtractTranslation()
    return (world_position[0], world_position[1], world_position[2])


def reset_and_cache_transform(xform_cache: UsdGeom.XformCache, prim: Usd.Prim) -> Tuple[Gf.Matrix4d, bool]:
    """
    Compute the local transformation matrix for a prim and cache it in the XformCache.

    Args:
        xform_cache (UsdGeom.XformCache): The XformCache instance to use for caching.
        prim (Usd.Prim): The prim to compute the local transformation for.

    Returns:
        Tuple[Gf.Matrix4d, bool]: A tuple containing the local transformation matrix and a boolean
        indicating whether the prim resets the transform stack.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    resets_xform_stack = False
    local_xform = xform_cache.GetLocalTransformation(prim)
    if UsdGeom.XformCommonAPI(prim).GetResetXformStack():
        resets_xform_stack = True
    return (local_xform, resets_xform_stack)


def compute_and_cache_hierarchy_transform(xform_cache: UsdGeom.XformCache, prim: Usd.Prim) -> Gf.Matrix4d:
    """
    Compute and cache the local-to-world transform for the given prim.

    This function uses the XformCache to efficiently compute and cache the
    local-to-world transform for the given prim, considering the entire
    hierarchy up to the root prim.

    Parameters:
        xform_cache (UsdGeom.XformCache): The XformCache instance to use for caching.
        prim (Usd.Prim): The prim to compute the transform for.

    Returns:
        Gf.Matrix4d: The local-to-world transform for the given prim.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if prim.IsPseudoRoot():
        return Gf.Matrix4d(1)
    parent_prim = prim.GetParent()
    parent_xform = compute_and_cache_hierarchy_transform(xform_cache, parent_prim)
    local_xform = xform_cache.GetLocalToWorldTransform(prim)
    prim_xform = local_xform * parent_xform
    return prim_xform


def compute_relative_transform_matrix(
    prim: Usd.Prim, ancestor: Usd.Prim, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> Gf.Matrix4d:
    """
    Compute the relative transform matrix between a prim and its ancestor.

    Args:
        prim (Usd.Prim): The prim to compute the relative transform for.
        ancestor (Usd.Prim): The ancestor prim to compute the relative transform against.
        time_code (Usd.TimeCode): The time code at which to compute the transform. Defaults to Default.

    Returns:
        Gf.Matrix4d: The relative transform matrix between the prim and its ancestor.

    Raises:
        ValueError: If the input prims are not valid or if the ancestor is not an ancestor of the prim.
    """
    if not prim.IsValid() or not ancestor.IsValid():
        raise ValueError("Input prims must be valid.")
    if not prim.GetPath().HasPrefix(ancestor.GetPath()):
        raise ValueError("The 'ancestor' prim must be an ancestor of the 'prim'.")
    xform_cache = UsdGeom.XformCache(time_code)
    prim_local_to_world = xform_cache.GetLocalToWorldTransform(prim)
    ancestor_local_to_world = xform_cache.GetLocalToWorldTransform(ancestor)
    relative_transform = ancestor_local_to_world.GetInverse() * prim_local_to_world
    return relative_transform


def batch_apply_transforms(stage: Usd.Stage, transform_data: List[Tuple[str, Gf.Matrix4d]]):
    """
    Applies a list of transform matrices to corresponding prims on a USD stage.

    Args:
        stage (Usd.Stage): The USD stage to apply transforms to.
        transform_data (List[Tuple[str, Gf.Matrix4d]]): A list of tuples, each containing a prim path
                                                         and its corresponding transform matrix.

    Raises:
        ValueError: If any of the prim paths are invalid or not transformable.
    """
    for prim_path, matrix in transform_data:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        xformable.ClearXformOpOrder()
        transform_op = xformable.AddTransformOp()
        transform_op.Set(matrix)


def extract_transform_hierarchy(stage: Usd.Stage) -> Dict[str, List[str]]:
    """Extract the transform hierarchy from a USD stage.

    Args:
        stage (Usd.Stage): The USD stage to extract the hierarchy from.

    Returns:
        Dict[str, List[str]]: A dictionary where keys are prim paths and values are lists of child prim paths.
    """
    hierarchy: Dict[str, List[str]] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        prim_path = prim.GetPath().pathString
        parent_path = prim.GetPath().GetParentPath().pathString
        if parent_path not in hierarchy:
            hierarchy[parent_path] = []
        hierarchy[parent_path].append(prim_path)
    return hierarchy


def toggle_reset_xform_stack(prim: Usd.Prim) -> bool:
    """Toggle the resetXformStack attribute of a prim.

    Args:
        prim (Usd.Prim): The prim to toggle the resetXformStack attribute on.

    Returns:
        bool: The new value of resetXformStack after toggling.
    """
    xform_api = UsdGeom.XformCommonAPI(prim)
    if not xform_api:
        raise ValueError(f"Prim at path {prim.GetPath()} is not compatible with XformCommonAPI.")
    reset_xform_stack = xform_api.GetResetXformStack()
    new_reset_xform_stack = not reset_xform_stack
    success = xform_api.SetResetXformStack(new_reset_xform_stack)
    if not success:
        raise RuntimeError(
            f"Failed to set resetXformStack to {new_reset_xform_stack} on prim at path {prim.GetPath()}."
        )
    return new_reset_xform_stack


def batch_reset_and_apply_transforms(
    stage: Usd.Stage, prim_paths: List[str], new_transform_matrix: Gf.Matrix4d
) -> None:
    """
    Reset the local transform stack of each prim and set a new transform matrix.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (List[str]): The paths of the prims to modify.
        new_transform_matrix (Gf.Matrix4d): The new transform matrix to apply.

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
        xformable.ClearXformOpOrder()
        xformable.SetResetXformStack(True)
        transform_op = xformable.AddTransformOp()
        transform_op.Set(new_transform_matrix)


def interpolate_transforms(
    xformable: UsdGeom.Xformable, start_time: Usd.TimeCode, end_time: Usd.TimeCode, fraction: float
) -> Tuple[Gf.Vec3d, Gf.Vec3f, Gf.Vec3f, Gf.Vec3f]:
    """Interpolate transform values between two times.

    Args:
        xformable (UsdGeom.Xformable): The xformable prim.
        start_time (Usd.TimeCode): The start time.
        end_time (Usd.TimeCode): The end time.
        fraction (float): The interpolation fraction (0.0 to 1.0).

    Returns:
        Tuple[Gf.Vec3d, Gf.Vec3f, Gf.Vec3f, Gf.Vec3f]: The interpolated translation, rotation, scale, and pivot.
    """
    if not xformable:
        raise ValueError("Invalid xformable prim.")
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("Fraction must be between 0.0 and 1.0.")
    xform_api = UsdGeom.XformCommonAPI(xformable)
    (start_translation, start_rotation, start_scale, start_pivot, _) = xform_api.GetXformVectorsByAccumulation(
        start_time
    )
    (end_translation, end_rotation, end_scale, end_pivot, _) = xform_api.GetXformVectorsByAccumulation(end_time)
    translation = Gf.Lerp(fraction, start_translation, end_translation)
    rotation = Gf.Slerp(fraction, Gf.Quatf(0, *start_rotation), Gf.Quatf(0, *end_rotation)).GetImaginary()
    scale = Gf.Lerp(fraction, start_scale, end_scale)
    pivot = Gf.Lerp(fraction, start_pivot, end_pivot)
    return (translation, rotation, scale, pivot)


def convert_rotation_and_apply_to_prims(
    stage: Usd.Stage,
    rotation: Gf.Vec3f,
    rotation_order: UsdGeom.XformCommonAPI.RotationOrder,
    prim_paths: List[str],
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
):
    """
    Convert a rotation vector to the specified rotation order and apply it to the given prims.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        rotation (Gf.Vec3f): The rotation vector to apply.
        rotation_order (UsdGeom.XformCommonAPI.RotationOrder): The desired rotation order.
        prim_paths (List[str]): The paths of the prims to apply the rotation to.
        time_code (Usd.TimeCode, optional): The time code at which to set the rotation. Defaults to Default.

    Returns:
        bool: True if the rotation was successfully applied to all prims, False otherwise.
    """
    success = True
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Error: Prim at path {prim_path} does not exist.")
            success = False
            continue
        xform_api = UsdGeom.XformCommonAPI(prim)
        if not xform_api.SetRotate(rotation, rotation_order, time_code):
            print(f"Error: Failed to set rotation for prim at path {prim_path}.")
            success = False
    return success


def extract_and_accumulate_transforms(
    xformable: UsdGeom.Xformable, time: Usd.TimeCode
) -> Tuple[Gf.Vec3d, Gf.Vec3f, Gf.Vec3f, Gf.Vec3f]:
    """Extract and accumulate transform values from an xformable prim.

    Args:
        xformable (UsdGeom.Xformable): The xformable prim to extract transforms from.
        time (Usd.TimeCode): The time at which to evaluate the transforms.

    Returns:
        Tuple[Gf.Vec3d, Gf.Vec3f, Gf.Vec3f, Gf.Vec3f]: A tuple containing the accumulated translation, rotation, scale, and pivot values.
    """
    if not xformable:
        raise ValueError("Invalid xformable prim")
    translation = Gf.Vec3d(0, 0, 0)
    rotation = Gf.Vec3f(0, 0, 0)
    scale = Gf.Vec3f(1, 1, 1)
    pivot = Gf.Vec3f(0, 0, 0)
    for op in xformable.GetOrderedXformOps():
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate:
            translation += Gf.Vec3d(op.Get(time))
        elif UsdGeom.XformCommonAPI.CanConvertOpTypeToRotationOrder(op_type):
            rotation += Gf.Vec3f(op.Get(time))
        elif op_type == UsdGeom.XformOp.TypeScale:
            scale_value = op.Get(time)
            scale = Gf.Vec3f(scale[0] * scale_value[0], scale[1] * scale_value[1], scale[2] * scale_value[2])
        elif op_type == UsdGeom.XformOp.TypeTranslate and "pivot" in op.GetName():
            pivot = Gf.Vec3f(op.Get(time))
    return (translation, rotation, scale, pivot)


def convert_and_apply_rotation(
    prim: Usd.Prim,
    rotation: Gf.Vec3f,
    rotation_order: UsdGeom.XformOp.Type,
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
) -> bool:
    """
    Convert a rotation vector and rotation order to the corresponding xformOp type and set the rotation on the prim.

    Args:
        prim (Usd.Prim): The prim to apply the rotation to.
        rotation (Gf.Vec3f): The rotation vector.
        rotation_order (UsdGeom.XformOp.Type): The rotation order.
        time_code (Usd.TimeCode, optional): The time code to set the rotation at. Defaults to Default.

    Returns:
        bool: True if the rotation was successfully applied, False otherwise.
    """
    if not prim.IsValid():
        return False
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return False
    if not UsdGeom.XformCommonAPI.CanConvertOpTypeToRotationOrder(rotation_order):
        return False
    rotate_op = xformable.AddXformOp(rotation_order, UsdGeom.XformOp.PrecisionFloat, "")
    success = rotate_op.Set(rotation, time_code)
    return success


def adjust_transform_with_flags(
    xformable: UsdGeom.Xformable, translation: Gf.Vec3d, rotation: Gf.Vec3f, scale: Gf.Vec3f, flags: int
) -> None:
    """Adjust the transform of a prim using the provided translation, rotation, and scale, considering the specified operation flags.

    Args:
        xformable (UsdGeom.Xformable): The xformable prim to adjust the transform of.
        translation (Gf.Vec3d): The translation to apply.
        rotation (Gf.Vec3f): The rotation to apply in degrees.
        scale (Gf.Vec3f): The scale to apply.
        flags (int): The operation flags specifying which transform components to set.
    """
    xform_api = UsdGeom.XformCommonAPI(xformable)
    if flags & UsdGeom.XformCommonAPI.RotationOrderXYZ:
        xform_api.SetTranslate(translation)
    if flags & UsdGeom.XformCommonAPI.RotationOrderXZY:
        xform_api.SetRotate(rotation, UsdGeom.XformCommonAPI.RotationOrderXYZ)
    if flags & UsdGeom.XformCommonAPI.RotationOrderYXZ:
        xform_api.SetScale(scale)


def get_prim_rotation_order(prim: Usd.Prim) -> Optional[UsdGeom.XformCommonAPI.RotationOrder]:
    """Get the rotation order for a prim.

    Args:
        prim (Usd.Prim): The prim to get the rotation order for.

    Returns:
        Optional[UsdGeom.XformCommonAPI.RotationOrder]: The rotation order of the prim, or None if not set.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable.")
    xform_op_order_attr = xformable.GetXformOpOrderAttr()
    if not xform_op_order_attr.IsAuthored():
        return None
    xform_op_order_tokens = xform_op_order_attr.Get()
    for token in xform_op_order_tokens:
        if token.startswith("xformOp:rotateXYZ"):
            rotation_order_str = token.split(":")[2]
            return UsdGeom.XformCommonAPI.RotationOrder.GetValueFromName(rotation_order_str)
    return None


def copy_transform_ops(source_prim: UsdGeom.Xformable, dest_prim: UsdGeom.Xformable):
    """Copy transform ops from source_prim to dest_prim."""
    if not source_prim:
        raise ValueError("Invalid source prim. Expected a UsdGeom.Xformable.")
    if not dest_prim:
        raise ValueError("Invalid destination prim. Expected a UsdGeom.Xformable.")
    source_ops = source_prim.GetOrderedXformOps()
    for op in source_ops:
        op_attr = op.GetAttr()
        attr_name = op_attr.GetName()
        if dest_prim.GetPrim().HasAttribute(attr_name):
            dest_attr = dest_prim.GetPrim().GetAttribute(attr_name)
            dest_attr.Set(op_attr.Get())
        else:
            dest_attr = dest_prim.GetPrim().CreateAttribute(attr_name, op_attr.GetTypeName())
            dest_attr.Set(op_attr.Get())


def mirror_transform(xform_op: UsdGeom.XformOp, axis: str = "x") -> UsdGeom.XformOp:
    """
    Mirror the transformation represented by the given XformOp along the specified axis.

    Args:
        xform_op (UsdGeom.XformOp): The XformOp to mirror.
        axis (str): The axis to mirror along. Must be one of "x", "y", or "z". Defaults to "x".

    Returns:
        UsdGeom.XformOp: A new XformOp representing the mirrored transformation.

    Raises:
        ValueError: If the specified axis is not one of "x", "y", or "z".
    """
    if axis not in ["x", "y", "z"]:
        raise ValueError(f"Invalid axis '{axis}'. Must be one of 'x', 'y', or 'z'.")
    op_type = xform_op.GetOpType()
    op_value = xform_op.Get(Usd.TimeCode.Default())
    if op_type == UsdGeom.XformOp.TypeTranslate:
        if axis == "x":
            mirrored_value = Gf.Vec3d(-op_value[0], op_value[1], op_value[2])
        elif axis == "y":
            mirrored_value = Gf.Vec3d(op_value[0], -op_value[1], op_value[2])
        else:
            mirrored_value = Gf.Vec3d(op_value[0], op_value[1], -op_value[2])
    elif op_type == UsdGeom.XformOp.TypeScale:
        if axis == "x":
            mirrored_value = Gf.Vec3f(1.0 / op_value[0], op_value[1], op_value[2])
        elif axis == "y":
            mirrored_value = Gf.Vec3f(op_value[0], 1.0 / op_value[1], op_value[2])
        else:
            mirrored_value = Gf.Vec3f(op_value[0], op_value[1], 1.0 / op_value[2])
    elif op_type == UsdGeom.XformOp.TypeRotateX:
        mirrored_value = -op_value
    elif op_type == UsdGeom.XformOp.TypeRotateY:
        mirrored_value = -op_value
    elif op_type == UsdGeom.XformOp.TypeRotateZ:
        mirrored_value = -op_value
    else:
        return xform_op
    mirrored_op = UsdGeom.XformOp(xform_op.GetAttr())
    mirrored_op.Set(mirrored_value, Usd.TimeCode.Default())
    return mirrored_op


def randomize_transform(
    prim: Usd.Prim,
    translate_range: Tuple[float, float] = (-10, 10),
    rotate_range: Tuple[float, float] = (-180, 180),
    scale_range: Tuple[float, float] = (0.1, 5),
) -> None:
    """Randomize the transform of a given prim within specified ranges."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim.GetPath()} is not transformable")
    tx = random.uniform(*translate_range)
    ty = random.uniform(*translate_range)
    tz = random.uniform(*translate_range)
    translate_op = add_translate_op(xformable)
    translate_op.Set((tx, ty, tz))
    rx = random.uniform(*rotate_range)
    ry = random.uniform(*rotate_range)
    rz = random.uniform(*rotate_range)
    rotate_op = add_rotate_xyz_op(xformable)
    rotate_op.Set((rx, ry, rz))
    sx = random.uniform(*scale_range)
    sy = random.uniform(*scale_range)
    sz = random.uniform(*scale_range)
    scale_op = add_scale_op(xformable)
    scale_op.Set((sx, sy, sz))


def set_translation_rotation_scale(
    xformable: UsdGeom.Xformable,
    translation: Gf.Vec3d,
    rotation: Gf.Vec3f,
    scale: Gf.Vec3f,
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
):
    """Set translation, rotation, and scale for a transformable prim.

    Args:
        xformable (UsdGeom.Xformable): The transformable prim.
        translation (Gf.Vec3d): The translation vector.
        rotation (Gf.Vec3f): The rotation vector in degrees (XYZ euler angles).
        scale (Gf.Vec3f): The scale vector.
        time_code (Usd.TimeCode, optional): The time code for the operation. Defaults to Default time code.
    """
    if not xformable:
        raise ValueError("The given prim is not transformable.")
    translation_op = add_translate_op(xformable)
    translation_op.Set(translation, time_code)
    rotation_op = add_rotate_xyz_op(xformable)
    rotation_op.Set(rotation, time_code)
    scale_op = add_scale_op(xformable)
    scale_op.Set(scale, time_code)


def align_prims(stage: Usd.Stage, source_prim_path: str, target_prim_path: str) -> bool:
    """Align source prim to target prim by copying the transform.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        source_prim_path (str): The path of the source prim to align.
        target_prim_path (str): The path of the target prim to align to.

    Returns:
        bool: True if the alignment was successful, False otherwise.
    """
    source_prim = stage.GetPrimAtPath(source_prim_path)
    target_prim = stage.GetPrimAtPath(target_prim_path)
    if not source_prim.IsValid() or not target_prim.IsValid():
        print(f"Error: One or both prims are invalid. Source: {source_prim_path}, Target: {target_prim_path}")
        return False
    source_xformable = UsdGeom.Xformable(source_prim)
    target_xformable = UsdGeom.Xformable(target_prim)
    if not source_xformable or not target_xformable:
        print(f"Error: One or both prims are not transformable. Source: {source_prim_path}, Target: {target_prim_path}")
        return False
    target_xform_ops = target_xformable.GetOrderedXformOps()
    source_xformable.SetXformOpOrder([])
    for op in target_xform_ops:
        op_type = op.GetOpType()
        op_value = op.Get()
        if op_type == UsdGeom.XformOp.TypeTranslate:
            add_translate_op(source_xformable).Set(op_value)
        elif op_type == UsdGeom.XformOp.TypeRotateXYZ:
            add_rotate_xyz_op(source_xformable).Set(op_value)
        elif op_type == UsdGeom.XformOp.TypeRotateXZY:
            source_xformable.AddRotateXZYOp().Set(op_value)
        elif op_type == UsdGeom.XformOp.TypeRotateYXZ:
            source_xformable.AddRotateYXZOp().Set(op_value)
        elif op_type == UsdGeom.XformOp.TypeRotateYZX:
            source_xformable.AddRotateYZXOp().Set(op_value)
        elif op_type == UsdGeom.XformOp.TypeRotateZXY:
            source_xformable.AddRotateZXYOp().Set(op_value)
        elif op_type == UsdGeom.XformOp.TypeRotateZYX:
            source_xformable.AddRotateZYXOp().Set(op_value)
        elif op_type == UsdGeom.XformOp.TypeScale:
            add_scale_op(source_xformable).Set(op_value)
    return True


def flatten_transform_hierarchy(stage: Usd.Stage, root_path: str):
    """Flatten the transform hierarchy starting from the given root prim.

    All child prims with transforms will have their transforms collapsed into
    the root prim and the child prims will be removed.
    """
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise ValueError(f"Prim at path {root_path} does not exist.")
    xformable_prims = []
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Xformable):
            xformable_prims.append(prim)
    local_xforms = {}
    for prim in xformable_prims:
        local_xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        local_xforms[prim.GetPath()] = local_xform
    for prim in reversed(xformable_prims[1:]):
        prim.SetActive(False)
    root_xformable = UsdGeom.Xformable(root_prim)
    accum_xform = Gf.Matrix4d(1.0)
    for xform in local_xforms.values():
        accum_xform *= xform
    for op in root_xformable.GetOrderedXformOps():
        op.GetAttr().Clear()
    root_xformable.AddTransformOp().Set(accum_xform)


def get_inverse_transform(stage: Usd.Stage, prim_path: str) -> Gf.Matrix4d:
    """
    Get the inverse of the local-to-parent transformation for a prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path of the prim.

    Returns:
        Gf.Matrix4d: The inverse of the local-to-parent transformation matrix.

    Raises:
        ValueError: If the prim at the given path does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    xform_ops = xformable.GetOrderedXformOps()
    local_to_parent_xform = Gf.Matrix4d(1.0)
    for op in xform_ops:
        op_matrix = op.GetOpTransform(Usd.TimeCode.Default())
        local_to_parent_xform = local_to_parent_xform * op_matrix
    try:
        inverse_xform = local_to_parent_xform.GetInverse()
    except ArithmeticError:
        raise ValueError(f"Local-to-parent transformation matrix for prim at path {prim_path} is not invertible.")
    return inverse_xform


def animate_transform(
    stage: Usd.Stage, prim_path: str, translations: List[Tuple[float, float, float]], time_codes: List[Usd.TimeCode]
):
    """Animate the translation of a prim over time.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to animate.
        translations (List[Tuple[float, float, float]]): A list of translation values, each a tuple of (x, y, z).
        time_codes (List[Usd.TimeCode]): A list of time codes corresponding to each translation value.

    Raises:
        ValueError: If the prim path is invalid or the prim is not transformable.
        ValueError: If the number of translations and time codes do not match.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    if len(translations) != len(time_codes):
        raise ValueError("Number of translations and time codes must match.")
    translate_op = add_translate_op(xformable)
    for translation, time_code in zip(translations, time_codes):
        translate_op.Set(Gf.Vec3d(translation), time_code)


def set_combined_transform(
    xformable: UsdGeom.Xformable, transform_matrix: Gf.Matrix4d, time_code: Usd.TimeCode = Usd.TimeCode.Default()
):
    """Sets the local transform of an xformable prim to the given matrix.

    This function will clear any existing transform ops and create a single
    transform op to represent the given transform.

    Args:
        xformable (UsdGeom.Xformable): The xformable prim to set the transform on.
        transform_matrix (Gf.Matrix4d): The desired local transform matrix.
        time_code (Usd.TimeCode, optional): The timecode at which to author the transform.
            Defaults to Default time.
    """
    xformable.ClearXformOpOrder()
    try:
        transform_op = xformable.AddTransformOp()
    except Tf.ErrorException as e:
        print(f"Error adding transform op: {e}")
        return
    try:
        transform_op.Set(transform_matrix, time_code)
    except Tf.ErrorException as e:
        print(f"Error setting transform value: {e}")


def clear_transform_ops(xformable: UsdGeom.Xformable) -> bool:
    """Clear all transform ops on the given xformable prim."""
    if not xformable:
        raise ValueError("Invalid xformable prim.")
    xform_op_order = xformable.GetXformOpOrderAttr()
    if xform_op_order.Get() is None or len(xform_op_order.Get()) == 0:
        return True
    success = xform_op_order.Clear()
    if not success:
        raise RuntimeError("Failed to clear xformOpOrder.")
    for op in xformable.GetOrderedXformOps():
        op_attr = xformable.GetPrim().GetAttribute(op.GetName())
        if not op_attr.Clear():
            raise RuntimeError(f"Failed to remove transform op: {op.GetName()}")
    return True


def get_combined_transform(stage: Usd.Stage, prim_path: str) -> Gf.Matrix4d:
    """Get the combined transform for a prim."""
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
            combined_transform = local_transform * parent_transform
        else:
            combined_transform = local_transform
    else:
        combined_transform = local_transform
    return combined_transform


def get_transform_at_time_range(
    prim: UsdGeom.Xformable, time_range: Tuple[float, float], step_size: float = 1.0
) -> List[Tuple[float, Gf.Matrix4d]]:
    """
    Get the transformation matrices for a prim over a given time range.

    Args:
        prim (UsdGeom.Xformable): The prim to get the transformation matrices for.
        time_range (Tuple[float, float]): The start and end times of the range.
        step_size (float, optional): The step size for sampling the time range. Defaults to 1.0.

    Returns:
        List[Tuple[float, Gf.Matrix4d]]: A list of tuples containing the time and the corresponding transformation matrix.
    """
    if not prim:
        raise ValueError("Invalid prim.")
    if not prim.GetPrim().IsValid():
        raise ValueError("Prim is not valid.")
    (start_time, end_time) = time_range
    if start_time > end_time:
        raise ValueError("Invalid time range. Start time must be less than or equal to end time.")
    if step_size <= 0:
        raise ValueError("Step size must be greater than zero.")
    stage = prim.GetPrim().GetStage()
    transforms = []
    current_time = start_time
    while current_time <= end_time:
        time_code = Usd.TimeCode(current_time)
        transform_matrix = prim.ComputeLocalToWorldTransform(time_code)
        transforms.append((current_time, transform_matrix))
        current_time += step_size
    return transforms


def set_transform_keyframes(
    xformable: UsdGeom.Xformable,
    translation: Tuple[float, float, float] = (0, 0, 0),
    rotation: Tuple[float, float, float] = (0, 0, 0),
    scale: Tuple[float, float, float] = (1, 1, 1),
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Set translation, rotation, and scale keyframes on a xformable prim at a specific time.

    Args:
        xformable (UsdGeom.Xformable): The xformable prim to set keyframes on.
        translation (Tuple[float, float, float], optional): Translation values (x, y, z). Defaults to (0, 0, 0).
        rotation (Tuple[float, float, float], optional): Rotation values in degrees (x, y, z). Defaults to (0, 0, 0).
        scale (Tuple[float, float, float], optional): Scale values (x, y, z). Defaults to (1, 1, 1).
        time_code (Usd.TimeCode, optional): The time code to set the keyframes at. Defaults to Usd.TimeCode.Default().

    Raises:
        ValueError: If the input xformable is not a valid UsdGeom.Xformable prim.
    """
    if not xformable or not isinstance(xformable, UsdGeom.Xformable):
        raise ValueError("Invalid xformable prim.")
    translate_ops = xformable.GetOrderedXformOps()
    translate_op = next((op for op in translate_ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    if not translate_op:
        translate_op = add_translate_op(xformable)
    translate_op.Set(Gf.Vec3d(translation), time_code)
    rotate_ops = xformable.GetOrderedXformOps()
    rotate_op = next((op for op in rotate_ops if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ), None)
    if not rotate_op:
        rotate_op = add_rotate_xyz_op(xformable)
    rotate_op.Set(
        Gf.Vec3f(Gf.RadiansToDegrees(rotation[0]), Gf.RadiansToDegrees(rotation[1]), Gf.RadiansToDegrees(rotation[2])),
        time_code,
    )
    scale_ops = xformable.GetOrderedXformOps()
    scale_op = next((op for op in scale_ops if op.GetOpType() == UsdGeom.XformOp.TypeScale), None)
    if not scale_op:
        scale_op = add_scale_op(xformable)
    scale_op.Set(Gf.Vec3f(scale), time_code)


def normalize_transform_hierarchy(stage: Usd.Stage, root_path: str) -> None:
    """Normalize the transform hierarchy under the given root path.

    This function removes any intermediate Xform prims that have only one child
    and no xformOps, and reparents the child directly to the parent of the Xform.

    Args:
        stage (Usd.Stage): The USD stage to operate on.
        root_path (str): The path to the root prim of the hierarchy to normalize.
    """
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise ValueError(f"Invalid root path: {root_path}")
    prims_to_process = [root_prim]
    while prims_to_process:
        prim = prims_to_process.pop()
        if prim.IsA(UsdGeom.Xform) and len(prim.GetChildren()) == 1:
            xform = UsdGeom.Xform(prim)
            if not xform.GetXformOpOrderAttr().Get():
                child = next(iter(prim.GetChildren()))
                parent = prim.GetParent()
                child_path = child.GetPath()
                new_child_path = parent.GetPath().AppendChild(child_path.name)
                child.SetActive(False)
                new_child = stage.DefinePrim(new_child_path, child.GetTypeName())
                for attr in child.GetAttributes():
                    attr_value = attr.Get()
                    if attr_value is not None:
                        new_child.CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr_value)
                for rel in child.GetRelationships():
                    targets = rel.GetTargets()
                    if targets:
                        new_rel = new_child.CreateRelationship(rel.GetName())
                        new_rel.SetTargets(targets)
                stage.RemovePrim(prim.GetPath())
                stage.RemovePrim(child_path)
                prims_to_process.append(new_child)
        else:
            prims_to_process.extend(prim.GetChildren())


def transform_prim_with_precision(
    prim: Usd.Prim, transform_op_type: UsdGeom.XformOp.Type, precision: UsdGeom.XformOp.Precision, value: Gf.Vec3d
):
    """Apply a transformation operation to a prim with a specific precision.

    Args:
        prim (Usd.Prim): The prim to transform.
        transform_op_type (UsdGeom.XformOp.Type): The type of transformation operation.
        precision (UsdGeom.XformOp.Precision): The precision to use for the transformation.
        value (Gf.Vec3d): The value of the transformation.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim '{prim.GetPath()}' is not valid.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim '{prim.GetPath()}' is not transformable.")
    existing_ops = [
        op
        for op in xformable.GetOrderedXformOps()
        if op.GetOpType() == transform_op_type and op.GetPrecision() == precision
    ]
    if existing_ops:
        op = existing_ops[0]
        op.Set(value)
    else:
        xformable.AddXformOp(transform_op_type, precision).Set(value)


def apply_transformations(
    stage: Usd.Stage, prim_path: str, transformations: List[Tuple[UsdGeom.XformOp.Type, Gf.Vec3d]]
):
    """Apply a list of transformations to a prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim to transform.
        transformations (List[Tuple[UsdGeom.XformOp.Type, Gf.Vec3d]]): A list of tuples containing the transformation type and value.

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    for op_type, value in transformations:
        op = xformable.GetOrderedXformOps()[-1] if xformable.GetOrderedXformOps() else None
        if op and op.GetOpType() == op_type:
            op.Set(value)
        else:
            op = xformable.AddXformOp(op_type)
            op.Set(value)


def apply_transform_to_hierarchy(
    stage: Usd.Stage,
    prim_path: str,
    transform: Tuple[
        float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float
    ],
    transform_op_suffix: Optional[str] = None,
) -> None:
    """Apply a transform to a prim and its descendants.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to transform.
        transform (Tuple[float, ...]): The transform matrix as a tuple of 16 floats.
        transform_op_suffix (str, optional): The suffix to use for the transform op. If None, a unique suffix will be generated.

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    if transform_op_suffix is None:
        transform_op_suffix = f"transform_{len(xformable.GetOrderedXformOps())}"
    transform_op = xformable.GetOrderedXformOps()[-1] if xformable.GetOrderedXformOps() else None
    if (
        not transform_op
        or transform_op.GetOpType() != UsdGeom.XformOp.TypeTransform
        or transform_op.GetName() != transform_op_suffix
    ):
        transform_op = xformable.AddXformOp(
            UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionFloat, transform_op_suffix
        )
    transform_op.Set(Gf.Matrix4d(*transform))
    for child in prim.GetAllChildren():
        apply_transform_to_hierarchy(stage, child.GetPath(), transform, transform_op_suffix)


def set_transform_with_constraints(
    prim: Usd.Prim,
    translation: Gf.Vec3d,
    rotation_euler: Gf.Vec3f,
    scale: Gf.Vec3f,
    rotation_order: UsdGeom.XformOp.Type = UsdGeom.XformOp.TypeRotateXYZ,
) -> None:
    """Set the local transform of a prim with rotation order and op name constraints.

    Args:
        prim (Usd.Prim): The prim to set the transform on.
        translation (Gf.Vec3d): The translation to set.
        rotation_euler (Gf.Vec3f): The rotation to set as Euler angles in degrees.
        scale (Gf.Vec3f): The scale to set.
        rotation_order (UsdGeom.XformOp.Type): The rotation order to use. Defaults to RotateXYZ.
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError("Prim is not transformable")
    xformable.ClearXformOpOrder()
    translation_op = add_translate_op(xformable, opSuffix="translate_op")
    translation_op.Set(translation)
    if rotation_order == UsdGeom.XformOp.TypeRotateXYZ:
        rotation_op = add_rotate_xyz_op(xformable, opSuffix="rotate_op")
    elif rotation_order == UsdGeom.XformOp.TypeRotateXZY:
        rotation_op = xformable.AddRotateXZYOp(opSuffix="rotate_op")
    elif rotation_order == UsdGeom.XformOp.TypeRotateYXZ:
        rotation_op = xformable.AddRotateYXZOp(opSuffix="rotate_op")
    elif rotation_order == UsdGeom.XformOp.TypeRotateYZX:
        rotation_op = xformable.AddRotateYZXOp(opSuffix="rotate_op")
    elif rotation_order == UsdGeom.XformOp.TypeRotateZXY:
        rotation_op = xformable.AddRotateZXYOp(opSuffix="rotate_op")
    elif rotation_order == UsdGeom.XformOp.TypeRotateZYX:
        rotation_op = xformable.AddRotateZYXOp(opSuffix="rotate_op")
    else:
        raise ValueError(f"Unsupported rotation order: {rotation_order}")
    rotation_op.Set(rotation_euler)
    scale_op = add_scale_op(xformable, opSuffix="scale_op")
    scale_op.Set(scale)


def extract_and_export_transform(stage: Usd.Stage, prim_path: str, output_file: str) -> Tuple[bool, str]:
    """Extract the local transform of a prim and export it to a file.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path of the prim to extract the transform from.
        output_file (str): The path to the file to export the transform to.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating success or failure,
                          and a message string.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return (False, f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return (False, f"Prim at path {prim_path} is not transformable.")
    transform_matrix = xformable.GetLocalTransformation()
    export_stage = Usd.Stage.CreateNew(output_file)
    root_prim = UsdGeom.Xform.Define(export_stage, "/root")
    transform_attr = root_prim.AddTransformOp()
    transform_attr.Set(Gf.Matrix4d(transform_matrix))
    success = export_stage.GetRootLayer().Export(output_file)
    if not success:
        return (False, f"Failed to export transform to {output_file}.")
    return (True, f"Successfully exported transform to {output_file}.")


def add_custom_scale_op(
    xformable: UsdGeom.Xformable, scale_value: Tuple[float, float, float], op_suffix: str = ""
) -> UsdGeom.XformOp:
    """Add a custom scale op to the xformable.

    Args:
        xformable (UsdGeom.Xformable): The xformable to add the scale op to.
        scale_value (Tuple[float, float, float]): The scale value to set.
        op_suffix (str, optional): The suffix to append to the op name. Defaults to "".

    Returns:
        UsdGeom.XformOp: The created scale op.
    """
    if not xformable:
        raise ValueError("Invalid xformable provided.")
    scale_op = add_scale_op(xformable, precision=UsdGeom.XformOp.PrecisionFloat, opSuffix=op_suffix)
    if not scale_op:
        raise RuntimeError("Failed to create scale op.")
    success = scale_op.Set(Gf.Vec3d(scale_value))
    if not success:
        raise RuntimeError("Failed to set scale value.")
    return scale_op


def remove_transform_op(xformable: UsdGeom.Xformable, op_type: UsdGeom.XformOp.Type, op_suffix: str = "") -> bool:
    """Remove a transform operation from a Xformable prim.

    Args:
        xformable (UsdGeom.Xformable): The Xformable prim to remove the transform op from.
        op_type (UsdGeom.XformOp.Type): The type of the transform op to remove.
        op_suffix (str, optional): The suffix of the transform op to remove. Defaults to "".

    Returns:
        bool: True if the op was successfully removed, False otherwise.
    """
    ops = xformable.GetOrderedXformOps()
    matching_ops = [op for op in ops if op.GetOpType() == op_type and op.GetName().endswith(op_suffix)]
    if not matching_ops:
        return False
    for op in matching_ops:
        ops.remove(op)
    xformable.SetXformOpOrder(ops, xformable.GetResetXformStack())
    return True


def animate_transform_op(
    xformable: UsdGeom.Xformable,
    translation: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    scale: Tuple[float, float, float],
    time_code: Usd.TimeCode,
):
    """Animate the local transformation of a UsdGeomXformable prim using matrix decomposition.

    Args:
        xformable (UsdGeom.Xformable): The xformable prim to animate.
        translation (Tuple[float, float, float]): The translation values (tx, ty, tz).
        rotation (Tuple[float, float, float]): The rotation values in degrees (rx, ry, rz).
        scale (Tuple[float, float, float]): The scale values (sx, sy, sz).
        time_code (Usd.TimeCode): The time code at which to set the transform values.
    """
    if not xformable:
        raise ValueError("The given prim is not transformable.")
    xformable.SetXformOpOrder([], False)
    transform_op = xformable.MakeMatrixXform()
    translation_mat = Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(translation))
    rotation_mat = Gf.Matrix4d(1).SetRotate(
        Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation[0])
        * Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation[1])
        * Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation[2])
    )
    scale_mat = Gf.Matrix4d(1).SetScale(Gf.Vec3d(scale))
    final_mat = scale_mat * rotation_mat * translation_mat
    transform_op.Set(final_mat, time_code)


def reorder_transform_ops(xformable: UsdGeom.Xformable, new_order: List[str]) -> bool:
    """Reorder the transform ops on a Xformable prim.

    Args:
        xformable (UsdGeom.Xformable): The Xformable prim to reorder ops on.
        new_order (List[str]): The new order of ops, specified by op name.

    Returns:
        bool: True if successful, False otherwise.
    """
    current_ops = xformable.GetOrderedXformOps()
    op_dict = {op.GetOpName(): op for op in current_ops}
    for op_name in new_order:
        if op_name not in op_dict:
            print(f"Op {op_name} does not exist on prim {xformable.GetPath()}")
            return False
    reordered_ops = [op_dict[op_name] for op_name in new_order]
    return xformable.SetXformOpOrder(reordered_ops, resetXformStack=False)


def import_and_apply_transform(stage: Usd.Stage, prim_path: str, transform_matrix: Gf.Matrix4d) -> None:
    """Import a transform matrix and apply it to a prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim to apply the transform to.
        transform_matrix (Gf.Matrix4d): The transform matrix to apply.

    Raises:
        ValueError: If the prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    xformable.SetXformOpOrder([], False)
    transform_op = xformable.AddTransformOp()
    transform_op.Set(transform_matrix)


def get_transform_op_info(prim: Usd.Prim) -> Dict[str, Tuple[UsdGeom.XformOp.Type, UsdGeom.XformOp]]:
    """
    Get information about all transform ops on a prim.

    Args:
        prim (Usd.Prim): The prim to query for transform ops.

    Returns:
        Dict[str, Tuple[UsdGeom.XformOp.Type, UsdGeom.XformOp]]:
            A dictionary mapping transform op names to tuples containing the op type and UsdGeom.XformOp object.
            Returns an empty dictionary if the prim is not transformable or has no transform ops.
    """
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        return {}
    ops = xformable.GetOrderedXformOps()
    op_info = {}
    for op in ops:
        op_name = op.GetName()
        op_type = op.GetOpType()
        op_info[op_name] = (op_type, op)
    return op_info


def scale_transform_op(
    xformable: UsdGeom.Xformable, scale: Tuple[float, float, float], op_suffix: str = ""
) -> UsdGeom.XformOp:
    """Apply a scale transformation operation to a UsdGeomXformable prim.

    Args:
        xformable (UsdGeom.Xformable): The prim to apply the scale operation to.
        scale (Tuple[float, float, float]): The scale values in x, y, z.
        op_suffix (str): Optional suffix to append to the scale op name.

    Returns:
        UsdGeom.XformOp: The created scale XformOp.

    Raises:
        ValueError: If the provided prim is not a UsdGeomXformable.
    """
    if not xformable:
        raise ValueError("Provided prim is not a UsdGeomXformable.")
    scale_op = add_scale_op(xformable, UsdGeom.XformOp.PrecisionFloat, op_suffix, False)
    scale_op.Set(Gf.Vec3d(scale))
    return scale_op


def apply_srt_transform(
    xformable: UsdGeom.Xformable,
    translation: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    scale: Tuple[float, float, float],
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
):
    """Apply scale, rotation, and translation transform ops to a UsdGeomXformable prim.

    Args:
        xformable (UsdGeom.Xformable): The xformable prim to apply the transform to.
        translation (Tuple[float, float, float]): The translation as a tuple of 3 floats (x, y, z).
        rotation (Tuple[float, float, float]): The rotation in degrees as a tuple of 3 floats (x, y, z).
        scale (Tuple[float, float, float]): The scale as a tuple of 3 floats (x, y, z).
        time_code (Usd.TimeCode, optional): The time code to set the transform at. Defaults to Default time.

    Raises:
        ValueError: If xformable prim is not valid or not an Xformable.
    """
    if not xformable.GetPrim().IsValid():
        raise ValueError("Xformable prim is not valid")
    if not xformable:
        raise ValueError("Prim is not an Xformable")
    xformable.SetXformOpOrder([], False)
    if any((v != 1 for v in scale)):
        scale_op = add_scale_op(xformable)
        scale_op.Set(Gf.Vec3f(scale), time_code)
    if any((v != 0 for v in rotation)):
        rotate_op = add_rotate_xyz_op(xformable)
        rotate_op.Set(Gf.Vec3f(rotation), time_code)
    if any((v != 0 for v in translation)):
        translate_op = add_translate_op(xformable)
        translate_op.Set(Gf.Vec3f(translation), time_code)


def reset_and_reapply_transform_ops(xformable: UsdGeom.Xformable, ops: List[UsdGeom.XformOp]) -> None:
    """Reset the transform ops on an Xformable prim and reapply the given ops."""
    if not xformable or not xformable.GetPrim().IsValid():
        raise ValueError("Input Xformable is invalid.")
    xformable.SetXformOpOrder([], False)
    for op in ops:
        if not op or not op.GetAttr().IsValid():
            raise ValueError("One or more of the input XformOps is invalid.")
        opType = op.GetOpType()
        opSuffix = op.GetOpName()
        if ":" in opSuffix:
            opSuffix = opSuffix.split(":")[1]
        precision = op.GetPrecision()
        isInverseOp = op.IsInverseOp()
        xformable.AddXformOp(opType, precision, opSuffix, isInverseOp)
        value = op.Get()
        if value is not None:
            attr = xformable.GetPrim().GetAttribute(op.GetOpName())
            if attr:
                attr.Set(value)


def invert_transform_op(op: UsdGeom.XformOp) -> Gf.Matrix4d:
    """Invert the transformation matrix of a transform op."""
    transform = op.GetOpTransform(Usd.TimeCode.Default())
    if isinstance(transform, Gf.Matrix4f):
        transform = Gf.Matrix4d(transform)
    if transform.GetDeterminant() == 0:
        raise ValueError("The transformation matrix is not invertible.")
    inverted_transform = transform.GetInverse()
    return inverted_transform


def add_custom_rotate_op(
    xformable: UsdGeom.Xformable, angle_deg: float, rotate_order: str, op_suffix: str = ""
) -> UsdGeom.XformOp:
    """Add a custom rotation op to the xformable.

    Args:
        xformable (UsdGeom.Xformable): The xformable to add the op to.
        angle_deg (float): The rotation angle in degrees.
        rotate_order (str): The rotation order, e.g., "XYZ", "ZXY", etc.
        op_suffix (str): Optional suffix to append to the op name. Defaults to "".

    Returns:
        UsdGeom.XformOp: The newly created rotation op.

    Raises:
        ValueError: If the rotation order is invalid.
    """
    valid_orders = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]
    if rotate_order not in valid_orders:
        raise ValueError(f"Invalid rotation order '{rotate_order}'. Must be one of {valid_orders}.")
    op_type = getattr(UsdGeom.XformOp, f"TypeRotate{rotate_order}")
    op = xformable.AddXformOp(op_type, UsdGeom.XformOp.PrecisionFloat, op_suffix, False)
    if rotate_order.endswith("X"):
        op.Set(Gf.Vec3f(angle_deg, 0, 0), Usd.TimeCode.Default())
    elif rotate_order.endswith("Y"):
        op.Set(Gf.Vec3f(0, angle_deg, 0), Usd.TimeCode.Default())
    else:
        op.Set(Gf.Vec3f(0, 0, angle_deg), Usd.TimeCode.Default())
    return op


def get_transform_time_samples(stage: Usd.Stage, prim_path: str) -> List[float]:
    """Get all time samples where the transform is authored for a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    ops = xformable.GetOrderedXformOps()
    if not ops:
        return []
    time_samples = []
    for op in ops:
        if op.GetAttr().ValueMightBeTimeVarying():
            op_samples = op.GetTimeSamples()
            time_samples.extend(op_samples)
    time_samples = sorted(set(time_samples))
    return time_samples


def mirror_transform_op(xformable: UsdGeom.Xformable, op_suffix: str, axis: str) -> UsdGeom.XformOp:
    """Mirror an existing transform operation across a specified axis.

    Args:
        xformable (UsdGeom.Xformable): The xformable prim.
        op_suffix (str): The suffix of the transform op to mirror.
        axis (str): The axis to mirror across ("x", "y", or "z").

    Returns:
        UsdGeom.XformOp: The newly created mirrored transform op.

    Raises:
        ValueError: If the specified transform op does not exist or if an invalid axis is provided.
    """
    ops = xformable.GetOrderedXformOps()
    op = next((op for op in ops if op.GetName().endswith(op_suffix)), None)
    if op is None:
        raise ValueError(f"Transform op with suffix '{op_suffix}' does not exist.")
    if axis == "x":
        mirror_matrix = Gf.Matrix4d(-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    elif axis == "y":
        mirror_matrix = Gf.Matrix4d(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    elif axis == "z":
        mirror_matrix = Gf.Matrix4d(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1)
    else:
        raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'.")
    transform_matrix = op.GetOpTransform(Usd.TimeCode.Default())
    mirrored_op = xformable.AddTransformOp()
    mirrored_op.Set(transform_matrix * mirror_matrix)
    return mirrored_op


def apply_transform_and_parent(
    stage: Usd.Stage,
    prim_path: str,
    parent_path: str,
    translation: Tuple[float, float, float] = (0, 0, 0),
    rotation: Tuple[float, float, float] = (0, 0, 0),
    scale: Tuple[float, float, float] = (1, 1, 1),
) -> Usd.Prim:
    """Apply transformation and reparent a prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path of the prim to transform and reparent.
        parent_path (str): The path of the new parent prim.
        translation (Tuple[float, float, float]): Translation values (x, y, z). Defaults to (0, 0, 0).
        rotation (Tuple[float, float, float]): Rotation values in degrees (x, y, z). Defaults to (0, 0, 0).
        scale (Tuple[float, float, float]): Scale values (x, y, z). Defaults to (1, 1, 1).

    Returns:
        Usd.Prim: The transformed and reparented prim.

    Raises:
        ValueError: If the prim or parent prim does not exist or is not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    parent_prim = stage.GetPrimAtPath(parent_path)
    if not parent_prim.IsValid():
        raise ValueError(f"Parent prim at path {parent_path} does not exist.")
    translate_op = add_translate_op(xformable)
    translate_op.Set(Gf.Vec3d(translation))
    rotate_op = add_rotate_xyz_op(xformable)
    rotate_op.Set(Gf.Vec3d(rotation))
    scale_op = add_scale_op(xformable)
    scale_op.Set(Gf.Vec3d(scale))
    prim_name = prim.GetName()
    prim.GetParent().RemoveProperty(prim_name)
    parent_prim.CreateRelationship(prim_name).AddTarget(prim.GetPath())
    return prim


def consolidate_transform_ops(
    xformable: UsdGeom.Xformable, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> UsdGeom.XformOp:
    """Consolidate all transform ops on an xformable into a single matrix4d op."""
    ops = xformable.GetOrderedXformOps()
    if not ops:
        return xformable.MakeMatrixXform()
    transform_matrix = xformable.GetLocalTransformation(ops, time_code)
    for op in ops:
        op.GetAttr().Clear()
    xformable.GetXformOpOrderAttr().Clear()
    matrix_op = xformable.MakeMatrixXform()
    matrix_op.Set(transform_matrix, time_code)
    return matrix_op


def get_combined_local_transform(
    stage: Usd.Stage, prim_path: str, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> Gf.Matrix4d:
    """
    Get the combined local transform for a prim at a specific time.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        time_code (Usd.TimeCode): The time code to evaluate the transform at. Defaults to the default time code.

    Returns:
        Gf.Matrix4d: The combined local transform matrix.

    Raises:
        ValueError: If the prim is not valid or not transformable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    xform_ops = xformable.GetOrderedXformOps()
    combined_transform = Gf.Matrix4d(1.0)
    for op in xform_ops:
        transform_value = op.Get(time_code)
        if transform_value is None:
            continue
        op_type = op.GetOpType()
        if op_type == UsdGeom.XformOp.TypeTranslate:
            translation = Gf.Vec3d(transform_value)
            op_matrix = Gf.Matrix4d().SetTranslate(translation)
        elif op_type == UsdGeom.XformOp.TypeScale:
            scale = Gf.Vec3d(transform_value)
            op_matrix = Gf.Matrix4d().SetScale(scale)
        elif op_type == UsdGeom.XformOp.TypeRotateXYZ:
            rotation = Gf.Vec3d(transform_value)
            op_matrix = Gf.Matrix4d().SetRotate(
                Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation[0])
                * Gf.Rotation(Gf.Vec3d(0, 1, 0), rotation[1])
                * Gf.Rotation(Gf.Vec3d(0, 0, 1), rotation[2])
            )
        elif op_type == UsdGeom.XformOp.TypeTransform:
            op_matrix = Gf.Matrix4d(transform_value)
        else:
            continue
        combined_transform = combined_transform * op_matrix
    return combined_transform


def get_local_transform_at_time(prim: Usd.Prim, time_code: Usd.TimeCode) -> Gf.Matrix4d:
    """
    Get the local transformation matrix for a prim at a specific time.

    Args:
        prim (Usd.Prim): The prim to get the local transformation for.
        time_code (Usd.TimeCode): The time code to get the transformation at.

    Returns:
        Gf.Matrix4d: The local transformation matrix.
    """
    if not prim.IsA(UsdGeom.Xformable):
        raise ValueError(f"Prim {prim.GetPath()} is not an Xformable.")
    xformable = UsdGeom.Xformable(prim)
    affected_by_attrs = any(
        (UsdGeom.Xformable.IsTransformationAffectedByAttrNamed(attr.GetName()) for attr in prim.GetAttributes())
    )
    if not affected_by_attrs:
        return Gf.Matrix4d(1)
    xform_ops = xformable.GetOrderedXformOps()
    is_time_varying = any((op.GetAttr().ValueMightBeTimeVarying() for op in xform_ops))
    if not is_time_varying:
        return xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return xformable.ComputeLocalToWorldTransform(time_code)


def add_custom_translate_op(
    xformable: UsdGeom.Xformable,
    translation: Tuple[float, float, float],
    op_suffix: str = "",
    inverse: bool = False,
    precision: UsdGeom.XformOp.Precision = UsdGeom.XformOp.PrecisionFloat,
) -> UsdGeom.XformOp:
    """Add a custom translation operation to a UsdGeomXformable prim.

    Args:
        xformable (UsdGeom.Xformable): The xformable prim to add the op to.
        translation (Tuple[float, float, float]): The translation values.
        op_suffix (str): Optional suffix for the op name. Defaults to "".
        inverse (bool): Whether this op is an inverse op. Defaults to False.
        precision (UsdGeom.XformOp.Precision): The precision of the op.
            Defaults to UsdGeom.XformOp.PrecisionFloat.

    Returns:
        UsdGeom.XformOp: The created translate op.
    """
    if not xformable:
        raise ValueError("Invalid UsdGeomXformable object")
    translate_op = add_translate_op(xformable, precision, op_suffix, inverse)
    if not inverse:
        translate_op.Set(Gf.Vec3d(*translation))
    return translate_op


def duplicate_transform_ops(
    prim: UsdGeom.Xformable, ops_to_duplicate: List[UsdGeom.XformOp], suffix: str = "_copy"
) -> List[UsdGeom.XformOp]:
    """Duplicate the specified transform ops on the given prim with the provided suffix.

    Args:
        prim (UsdGeom.Xformable): The prim to duplicate the transform ops on.
        ops_to_duplicate (List[UsdGeom.XformOp]): The list of transform ops to duplicate.
        suffix (str, optional): The suffix to append to the duplicated op names. Defaults to "_copy".

    Returns:
        List[UsdGeom.XformOp]: The list of duplicated transform ops.

    Raises:
        ValueError: If the prim is not transformable or if any of the ops to duplicate do not exist on the prim.
    """
    if not prim:
        raise ValueError("The provided prim is not transformable.")
    existing_ops = prim.GetOrderedXformOps()
    for op in ops_to_duplicate:
        if op not in existing_ops:
            raise ValueError(f"Op {op.GetName()} does not exist on the prim.")
    duplicated_ops = []
    for op in ops_to_duplicate:
        op_type = op.GetOpType()
        op_name = op.GetName()
        op_suffix = op_name.split(":")[-1] if ":" in op_name else ""
        new_suffix = op_suffix + suffix
        new_op = prim.AddXformOp(op_type, op.GetPrecision(), new_suffix, False)
        op_value = op.Get()
        new_op.Set(op_value)
        duplicated_ops.append(new_op)
    return duplicated_ops


def merge_transform_ops(xformable: UsdGeom.Xformable, ops_to_merge: List[UsdGeom.XformOp]) -> Optional[UsdGeom.XformOp]:
    """Merge multiple transform ops into a single transform op.

    Args:
        xformable (UsdGeom.Xformable): The xformable prim.
        ops_to_merge (List[UsdGeom.XformOp]): The list of transform ops to merge.

    Returns:
        Optional[UsdGeom.XformOp]: The merged transform op, or None if no ops were merged.
    """
    if not xformable.GetPrim().IsValid():
        raise ValueError("Invalid prim.")
    existing_ops = xformable.GetOrderedXformOps()
    for op in ops_to_merge:
        if op not in existing_ops:
            raise ValueError(f"Op {op.GetOpName()} does not exist on the prim.")
    merged_matrix = Gf.Matrix4d(1)
    for op in ops_to_merge:
        op_matrix = op.GetOpTransform(Usd.TimeCode.Default())
        merged_matrix = op_matrix * merged_matrix
    op_names = [op.GetOpName() for op in ops_to_merge]
    remaining_ops = [op for op in existing_ops if op.GetOpName() not in op_names]
    xformable.SetXformOpOrder(remaining_ops, False)
    if merged_matrix != Gf.Matrix4d(1):
        merged_op = xformable.AddTransformOp()
        merged_op.Set(merged_matrix)
        return merged_op
    return None


def add_reset_xform_stack_op(xformable: UsdGeom.Xformable) -> bool:
    """Add a reset transform stack op to the xformable prim.

    Args:
        xformable (UsdGeom.Xformable): The xformable prim to add the op to.

    Returns:
        bool: True if the op was successfully added, False otherwise.
    """
    if not xformable or not xformable.GetPrim().IsValid():
        return False
    xform_op_order = xformable.GetXformOpOrderAttr()
    if xform_op_order.HasValue():
        ops = xform_op_order.Get()
        if ops and ops[0] == "!resetXformStack!":
            return True
        new_ops = ["!resetXformStack!"] + list(ops)
    else:
        new_ops = ["!resetXformStack!"]
    xform_op_order.Set(new_ops)
    return True
