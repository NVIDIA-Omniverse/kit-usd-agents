## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from typing import Any, Dict, List, Sequence, Tuple

from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils, Vt


def collect_variant_selections(prim: Usd.Prim, registered_variant_set_names: List[str]) -> Dict[str, str]:
    """Collect variant selections for a prim based on registered variant sets.

    Args:
        prim (Usd.Prim): The prim to collect variant selections for.
        registered_variant_set_names (List[str]): List of registered variant set names.

    Returns:
        Dict[str, str]: A dictionary mapping variant set names to selected variants.
    """
    variant_selections: Dict[str, str] = {}
    for variant_set_name in registered_variant_set_names:
        if prim.HasVariantSets():
            variant_sets = prim.GetVariantSets()
            if variant_sets.HasVariantSet(variant_set_name):
                variant_set = variant_sets.GetVariantSet(variant_set_name)
                selected_variant = variant_set.GetVariantSelection()
                if selected_variant:
                    variant_selections[variant_set_name] = selected_variant
    return variant_selections


def apply_variant_selection_to_prims(prims: Sequence[Usd.Prim], variantSetName: str, variantSelection: str) -> None:
    """Apply variant selection to a sequence of prims.

    Args:
        prims (Sequence[Usd.Prim]): The prims to apply the variant selection to.
        variantSetName (str): The name of the variant set to use.
        variantSelection (str): The variant selection to apply.

    Raises:
        ValueError: If the variant set is not registered on any of the prims.
        ValueError: If the variant selection is not available in the variant set.
    """
    for prim in prims:
        if not prim.HasVariantSets() or variantSetName not in prim.GetVariantSets().GetNames():
            raise ValueError(f"Variant set '{variantSetName}' is not registered on prim '{prim.GetPath()}'")
    for prim in prims:
        if variantSelection not in prim.GetVariantSet(variantSetName).GetVariantNames():
            raise ValueError(
                f"Variant selection '{variantSelection}' is not available in variant set '{variantSetName}'"
            )
    for prim in prims:
        variantSet = prim.GetVariantSet(variantSetName)
        variantSet.SetVariantSelection(variantSelection)


def export_scene_with_variant_policy(
    stage: Usd.Stage, output_path: str, policy: UsdUtils.RegisteredVariantSet.SelectionExportPolicy
) -> bool:
    """Export the USD stage with the specified variant selection export policy.

    Args:
        stage (Usd.Stage): The USD stage to export.
        output_path (str): The file path to export the USD stage to.
        policy (UsdUtils.RegisteredVariantSet.SelectionExportPolicy): The variant selection export policy.

    Returns:
        bool: True if the export was successful, False otherwise.
    """
    export_context = Usd.Stage.CreateNew(output_path)
    try:
        UsdUtils.RegisteredVariantSet.ExportVariantSelection(stage, export_context, policy)
        export_success = export_context.Export(output_path)
        return export_success
    except Exception as e:
        print(f"Error exporting stage: {str(e)}")
        return False


def apply_sparse_time_samples(attribute: Usd.Attribute, time_samples: List[Tuple[float, float]]) -> None:
    """Apply sparse time samples to a USD attribute using SparseAttrValueWriter.

    Args:
        attribute (Usd.Attribute): The attribute to apply time samples to.
        time_samples (List[Tuple[float, float]]): A list of (time, value) tuples representing the time samples.

    Raises:
        ValueError: If the attribute is not valid or if the time samples are not in chronological order.
    """
    if not attribute.IsValid():
        raise ValueError("Invalid attribute.")
    if len(time_samples) < 2:
        raise ValueError("At least two time samples are required.")
    default_value = attribute.Get()
    value_writer = UsdUtils.SparseAttrValueWriter(attribute, default_value)
    for i in range(len(time_samples)):
        (time, value) = time_samples[i]
        if i > 0 and time <= time_samples[i - 1][0]:
            raise ValueError("Time samples must be in chronological order.")
        value_writer.SetTimeSample(value, Usd.TimeCode(time))


def optimize_animation_data(stage: Usd.Stage, prim_paths: List[str]):
    """Optimize animation data for a list of prims."""
    value_writer = UsdUtils.SparseValueWriter()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            print(f"Prim at path {prim_path} does not exist. Skipping...")
            continue
        attributes = prim.GetAttributes()
        for attr in attributes:
            if attr.ValueMightBeTimeVarying():
                time_samples = attr.GetTimeSamples()
                default_value = attr.Get()
                if default_value is not None:
                    value_writer.SetAttribute(attr, default_value, Usd.TimeCode.Default())
                for time_sample in time_samples:
                    value = attr.Get(time_sample)
                    value_writer.SetAttribute(attr, value, time_sample)
    attr_value_writers = value_writer.GetSparseAttrValueWriters()
    return len(attr_value_writers)


def batch_set_sparse_attributes(
    value_writer: UsdUtils.SparseValueWriter, attributes: List[Tuple[Usd.Attribute, Any, Usd.TimeCode]]
) -> None:
    """
    Set multiple attribute values sparsely using a UsdUtils.SparseValueWriter.

    Args:
        value_writer (UsdUtils.SparseValueWriter): The sparse value writer to use for setting attribute values.
        attributes (List[Tuple[Usd.Attribute, Any, Usd.TimeCode]]): A list of tuples, each containing an attribute, value, and time code.

    Raises:
        ValueError: If the input list of attributes is empty.
    """
    if not attributes:
        raise ValueError("The list of attributes cannot be empty.")
    for attr, value, time_code in attributes:
        if not attr.IsValid():
            print(f"Warning: Skipping invalid attribute {attr.GetName()}")
            continue
        if attr.GetTypeName().isArray:
            if isinstance(value, Gf.Vec3f):
                value = Vt.Vec3fArray([value])
            elif isinstance(value, Gf.Vec3d):
                value = Vt.Vec3dArray([value])
            elif isinstance(value, Gf.Vec2f):
                value = Vt.Vec2fArray([value])
            elif isinstance(value, Gf.Vec2d):
                value = Vt.Vec2dArray([value])
            elif isinstance(value, float):
                value = Vt.FloatArray([value])
            elif isinstance(value, int):
                value = Vt.IntArray([value])
        success = value_writer.SetAttribute(attr, value, time_code)
        if not success:
            print(f"Warning: Failed to set attribute {attr.GetName()} at time {time_code}")


def cache_stage_with_variants(model_path: str, variant_selections: List[Tuple[str, str]]) -> Usd.Stage:
    """
    Cache a USD stage with variant selections using the UsdUtils.StageCache.

    Args:
        model_path (str): The path to the USD model file.
        variant_selections (List[Tuple[str, str]]): A list of variant selections as pairs (variant set name, variant name).

    Returns:
        Usd.Stage: The cached USD stage with the specified variant selections.
    """
    stage = Usd.Stage.Open(model_path)
    if not stage:
        raise ValueError(f"Failed to open the USD stage at path: {model_path}")
    for variant_set_name, variant_selection in variant_selections:
        default_prim = stage.GetDefaultPrim()
        if default_prim.IsValid():
            variant_set = default_prim.GetVariantSet(variant_set_name)
            if variant_set.IsValid():
                variant_set.SetVariantSelection(variant_selection)
            else:
                raise ValueError(f"Variant set '{variant_set_name}' not found in the USD stage.")
        else:
            raise ValueError("Default prim not found in the USD stage.")
    stage_cache = UsdUtils.StageCache.Get()
    stage_cache.Insert(stage)
    return stage


def retime_animation(stage: Usd.Stage, prim_path: str, time_code_range: UsdUtils.TimeCodeRange) -> List[Usd.TimeCode]:
    """Retimes the animation on a prim to a new time code range.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim with the animation.
        time_code_range (UsdUtils.TimeCodeRange): The new time code range.

    Returns:
        List[Usd.TimeCode]: The list of time codes in the new range.

    Raises:
        ValueError: If the prim path is invalid or the prim is not animatable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")
    if not prim.IsA(UsdGeom.Xformable):
        raise ValueError(f"Prim at path {prim_path} is not animatable")
    anim_attrs = [attr for attr in prim.GetAttributes() if attr.ValueMightBeTimeVarying()]
    new_time_codes = []
    for time_code in time_code_range:
        for attr in anim_attrs:
            attr.Set(attr.Get(time_code), time_code)
        new_time_codes.append(time_code)
    return new_time_codes


def generate_motion_path(stage: Usd.Stage, prim_path: str, time_range: UsdUtils.TimeCodeRange) -> Sdf.Path:
    """Generate a motion path for a prim over a given time range.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to generate the motion path for.
        time_range (UsdUtils.TimeCodeRange): The time range to generate the motion path over.

    Returns:
        Sdf.Path: The path to the generated motion path prim.
    """
    if not stage.GetPrimAtPath(prim_path):
        raise ValueError(f"Invalid prim path: {prim_path}")
    if not time_range.IsValid():
        raise ValueError("Invalid time range")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsA(UsdGeom.Xformable):
        raise ValueError(f"Prim at path {prim_path} is not transformable")
    motion_path_prim_path = prim_path + "_motionPath"
    motion_path_prim = UsdGeom.Xform.Define(stage, motion_path_prim_path).GetPrim()
    point_instancer_path = motion_path_prim_path + "_pointInstancer"
    point_instancer = UsdGeom.PointInstancer.Define(stage, point_instancer_path)
    point_instancer.CreatePrototypesRel().SetTargets([motion_path_prim.GetPath()])
    positions = []
    for time_code in time_range:
        translation = prim.GetAttribute("xformOp:translate").Get(time_code)
        if translation is None:
            translation = Gf.Vec3d(0, 0, 0)
        positions.append(Gf.Vec3f(translation))
    point_instancer.CreatePositionsAttr().Set(positions)
    return motion_path_prim.GetPath()


def sample_animation_with_stride(
    stage: Usd.Stage, prim_path: str, time_code_range: UsdUtils.TimeCodeRange
) -> List[Tuple[Usd.TimeCode, Gf.Matrix4d]]:
    """Sample the local transform of a prim at each time code in the given range.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to sample.
        time_code_range (UsdUtils.TimeCodeRange): The range of time codes to sample.

    Returns:
        List[Tuple[Usd.TimeCode, Gf.Matrix4d]]: A list of tuples containing the time code and local transform at each sampled frame.
    """
    if not time_code_range.IsValid():
        raise ValueError("Invalid time code range.")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim '{prim_path}' not found in the stage.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim '{prim_path}' is not transformable.")
    samples = []
    for time_code in time_code_range:
        local_transform = xformable.ComputeLocalToWorldTransform(time_code)
        samples.append((time_code, local_transform))
    return samples


def remove_animation_in_range(prim: Usd.Prim, start_time: Usd.TimeCode, end_time: Usd.TimeCode) -> None:
    """Remove animation from the prim within the given time code range."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if start_time > end_time:
        raise ValueError("Invalid time code range: start time must be less than or equal to end time")
    attributes: List[Usd.Attribute] = prim.GetAttributes()
    for attr in attributes:
        if attr.ValueMightBeTimeVarying():
            time_samples: List[float] = attr.GetTimeSamples()
            time_codes_to_remove: List[Usd.TimeCode] = []
            for time_code in time_samples:
                if start_time <= Usd.TimeCode(time_code) <= end_time:
                    time_codes_to_remove.append(Usd.TimeCode(time_code))
            for time_code in time_codes_to_remove:
                attr.ClearAtTime(time_code)


def extract_sub_frame_range(
    frame_range: UsdUtils.TimeCodeRange, start_frame: Usd.TimeCode, end_frame: Usd.TimeCode
) -> UsdUtils.TimeCodeRange:
    """Extract a sub-range from a given frame range.

    Args:
        frame_range (UsdUtils.TimeCodeRange): The input frame range.
        start_frame (Usd.TimeCode): The start frame of the sub-range.
        end_frame (Usd.TimeCode): The end frame of the sub-range.

    Returns:
        UsdUtils.TimeCodeRange: The extracted sub-range.
    """
    if not frame_range.IsValid():
        raise ValueError("Invalid input frame range.")
    if start_frame < frame_range.startTimeCode or end_frame > frame_range.endTimeCode:
        raise ValueError("Start or end frame is outside the input frame range.")
    if start_frame > end_frame:
        raise ValueError("Start frame cannot be greater than the end frame.")
    sub_range = UsdUtils.TimeCodeRange(start_frame, end_frame, frame_range.stride)
    return sub_range


def find_time_code_ranges_with_property(
    stage: Usd.Stage, prim_path: str, property_name: str
) -> List[UsdUtils.TimeCodeRange]:
    """
    Find all time code ranges where the specified property has authored values.

    Args:
        stage (Usd.Stage): The USD stage to query.
        prim_path (str): The path to the prim to query.
        property_name (str): The name of the property to check.

    Returns:
        List[UsdUtils.TimeCodeRange]: A list of time code ranges where the property has authored values.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim not found at path: {prim_path}")
    attribute = prim.GetAttribute(property_name)
    if not attribute:
        raise ValueError(f"Attribute '{property_name}' not found on prim at path: {prim_path}")
    time_samples = attribute.GetTimeSamples()
    if not time_samples:
        return []
    time_code_ranges = []
    for i in range(len(time_samples)):
        start_time = time_samples[i]
        if i == len(time_samples) - 1:
            time_code_range = UsdUtils.TimeCodeRange()
            time_code_range = UsdUtils.TimeCodeRange.CreateFromFrameSpec(f"{start_time}")
            time_code_ranges.append(time_code_range)
        else:
            end_time = time_samples[i + 1]
            stride = end_time - start_time
            time_code_range = UsdUtils.TimeCodeRange()
            time_code_range = UsdUtils.TimeCodeRange.CreateFromFrameSpec(f"{start_time}:{end_time}x{stride}")
            time_code_ranges.append(time_code_range)
    return time_code_ranges


def list_all_timecode_range_tokens() -> List[str]:
    """List all available timecode range tokens."""
    token_names = [attr for attr in dir(UsdUtils.TimeCodeRange.Tokens) if not attr.startswith("_")]
    token_names = [name for name in token_names if isinstance(getattr(UsdUtils.TimeCodeRange.Tokens, name), str)]
    return token_names


def get_timecode_range_token_by_name(token_name: str) -> UsdUtils.TimeCodeRange.Tokens:
    """Get the TimeCodeRange token by its name.

    Args:
        token_name (str): The name of the token to retrieve.

    Returns:
        UsdUtils.TimeCodeRange.Tokens: The corresponding token.

    Raises:
        ValueError: If the token name is not valid.
    """
    token_names = [attr for attr in dir(UsdUtils.TimeCodeRange.Tokens) if not attr.startswith("_")]
    if token_name not in token_names:
        raise ValueError(f"Invalid token name: {token_name}. Available tokens: {', '.join(token_names)}")
    token = getattr(UsdUtils.TimeCodeRange.Tokens, token_name)
    return token


def analyze_prim_hierarchy(stage: Usd.Stage) -> Dict[str, Any]:
    """Analyze the prim hierarchy of a USD stage.

    Args:
        stage (Usd.Stage): The USD stage to analyze.

    Returns:
        Dict[str, Any]: A dictionary containing the analysis results.
    """
    total_prim_count = 0
    active_prim_count = 0
    inactive_prim_count = 0
    instanced_prim_count = 0
    instance_count = 0
    master_prim_count = 0
    total_attribute_count = 0
    for prim in stage.TraverseAll():
        total_prim_count += 1
        if prim.IsActive():
            active_prim_count += 1
        else:
            inactive_prim_count += 1
        if prim.IsInstance():
            instanced_prim_count += 1
            instance_count += prim.GetInstanceCount()
        if prim.IsInstanceProxy():
            master_prim_count += 1
        total_attribute_count += len(prim.GetAttributes())
    result = {
        "totalPrimCount": total_prim_count,
        "activePrimCount": active_prim_count,
        "inactivePrimCount": inactive_prim_count,
        "instancedPrimCount": instanced_prim_count,
        "instanceCount": instance_count,
        "masterPrimCount": master_prim_count,
        "totalAttributeCount": total_attribute_count,
        "totalInstanceCount": instance_count,
    }
    return result
