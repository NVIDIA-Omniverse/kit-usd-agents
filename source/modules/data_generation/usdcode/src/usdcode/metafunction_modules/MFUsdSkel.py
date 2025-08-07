## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from typing import Dict, List, Optional, Sequence, Tuple, Union

from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdSkel, Vt


def remap_and_apply_transforms(
    source_xforms: Sequence[Gf.Matrix4d], target_xforms: Sequence[Gf.Matrix4d], anim_mapper: UsdSkel.AnimMapper
) -> List[Gf.Matrix4d]:
    """
    Remaps transforms from source to target using the provided AnimMapper.

    Args:
        source_xforms (Sequence[Gf.Matrix4d]): The source transforms.
        target_xforms (Sequence[Gf.Matrix4d]): The target transforms to apply remapping to.
        anim_mapper (UsdSkel.AnimMapper): The AnimMapper to use for remapping.

    Returns:
        List[Gf.Matrix4d]: The remapped target transforms.
    """
    source_vt = Vt.Matrix4dArray(source_xforms)
    target_vt = Vt.Matrix4dArray(target_xforms)
    if anim_mapper.IsNull():
        raise ValueError("Provided AnimMapper is null.")
    success = anim_mapper.RemapTransforms(source_vt, target_vt, elementSize=1)
    if not success:
        raise ValueError("Remapping failed. Check input data.")
    remapped_target_xforms = list(target_vt)
    return remapped_target_xforms


def batch_remap_animations(
    stage: Usd.Stage,
    anim_prims: List[Usd.Prim],
    mappings: List[Tuple[UsdSkel.AnimMapper, int]],
    output_prims: List[Usd.Prim],
) -> None:
    """
    Remaps a list of animation prims to a list of output prims using the provided mappings.

    Args:
        stage (Usd.Stage): The USD stage.
        anim_prims (List[Usd.Prim]): The list of animation prims.
        mappings (List[Tuple[UsdSkel.AnimMapper, int]]): The list of mappings, each a tuple of (AnimMapper, element size).
        output_prims (List[Usd.Prim]): The list of output prims.

    Raises:
        ValueError: If the number of animation prims, mappings, and output prims do not match.
    """
    if len(anim_prims) != len(mappings) or len(anim_prims) != len(output_prims):
        raise ValueError("Number of animation prims, mappings, and output prims must match.")
    for anim_prim, mapping, output_prim in zip(anim_prims, mappings, output_prims):
        anim = UsdSkel.Animation(anim_prim)
        output = UsdSkel.Animation(output_prim)
        source_value = anim.GetTranslationsAttr().Get(Usd.TimeCode.Default())
        target_value = output.GetTranslationsAttr().Get(Usd.TimeCode.Default())
        if target_value is None:
            target_value = Vt.Vec3fArray()
        (mapper, element_size) = mapping
        mapper.Remap(source_value, target_value, element_size, Gf.Vec3f())
        output.GetTranslationsAttr().Set(target_value, Usd.TimeCode.Default())


def convert_joint_transforms_to_world_space(
    joint_paths: Sequence[str], rest_xforms: Sequence[Gf.Matrix4d], local_xforms: Sequence[Gf.Matrix4d]
) -> Sequence[Gf.Matrix4d]:
    """
    Convert joint transforms from joint-local space to world space.

    Parameters:
        joint_paths (Sequence[str]): The paths of the joints in hierarchical order.
        rest_xforms (Sequence[Gf.Matrix4d]): The rest pose transforms of the joints in world space.
        local_xforms (Sequence[Gf.Matrix4d]): The joint-local transforms of the joints.

    Returns:
        Sequence[Gf.Matrix4d]: The joint transforms in world space.
    """
    if len(joint_paths) != len(rest_xforms) or len(joint_paths) != len(local_xforms):
        raise ValueError("Length of joint_paths, rest_xforms, and local_xforms must be equal.")
    world_xforms = [Gf.Matrix4d(1)] * len(joint_paths)
    for i, joint_path in enumerate(joint_paths):
        parent_path = joint_path.GetParentPath()
        parent_index = joint_paths.index(parent_path) if parent_path in joint_paths else -1
        if parent_index != -1:
            world_xforms[i] = world_xforms[parent_index] * rest_xforms[i] * local_xforms[i]
        else:
            world_xforms[i] = rest_xforms[i] * local_xforms[i]
    return world_xforms


def query_joint_transforms_over_time(
    anim_query: UsdSkel.AnimQuery, time_samples: List[float]
) -> List[Tuple[float, List[Gf.Matrix4d]]]:
    """
    Query the joint transforms from an AnimQuery object at specified time samples.

    Args:
        anim_query (UsdSkel.AnimQuery): The AnimQuery object to query joint transforms from.
        time_samples (List[float]): A list of time samples at which to query the joint transforms.

    Returns:
        List[Tuple[float, List[Gf.Matrix4d]]]: A list of tuples, where each tuple contains a time sample
        and the corresponding list of joint transforms as Gf.Matrix4d objects.
    """
    if not anim_query.GetPrim().IsValid():
        raise ValueError("Invalid AnimQuery object")
    joint_order = anim_query.GetJointOrder()
    result = []
    for time in time_samples:
        xforms = Vt.Matrix4dArray(len(joint_order))
        if not anim_query.ComputeJointLocalTransforms(xforms, Usd.TimeCode(time)):
            raise RuntimeError(f"Failed to compute joint local transforms at time {time}")
        xforms_list = [Gf.Matrix4d(xform) for xform in xforms]
        result.append((time, xforms_list))
    return result


def apply_joint_transforms(
    anim_query: UsdSkel.AnimQuery, root_xform: Gf.Matrix4d, time: Usd.TimeCode
) -> Vt.Matrix4dArray:
    """
    Apply joint transforms from an AnimQuery to a root transform.

    Args:
        anim_query (UsdSkel.AnimQuery): The AnimQuery to retrieve joint transforms from.
        root_xform (Gf.Matrix4d): The root transformation matrix.
        time (Usd.TimeCode): The time at which to retrieve the joint transforms.

    Returns:
        Vt.Matrix4dArray: The transformed joint matrices.
    """
    joint_order = anim_query.GetJointOrder()
    num_joints = len(joint_order)
    local_xforms = Vt.Matrix4dArray(num_joints)
    world_xforms = Vt.Matrix4dArray(num_joints)
    if not anim_query.ComputeJointLocalTransforms(local_xforms, time):
        raise ValueError("Failed to compute joint local transforms.")
    world_xforms[0] = root_xform * local_xforms[0]
    for i in range(1, num_joints):
        parent_id = joint_order.GetParentIndices(i)[0]
        world_xforms[i] = world_xforms[parent_id] * local_xforms[i]
    return world_xforms


def compare_animation_queries(query1: UsdSkel.AnimQuery, query2: UsdSkel.AnimQuery) -> bool:
    """
    Compare two UsdSkel.AnimQuery objects for equality.

    This function compares various properties of the two AnimQuery objects to
    determine if they are equivalent. It checks the following:
    - Prim paths
    - Joint orders
    - Blend shape orders
    - Joint transform and blend shape weight time samples

    Args:
        query1 (UsdSkel.AnimQuery): The first AnimQuery object to compare.
        query2 (UsdSkel.AnimQuery): The second AnimQuery object to compare.

    Returns:
        bool: True if the AnimQuery objects are equivalent, False otherwise.
    """
    if query1.GetPrim().GetPath() != query2.GetPrim().GetPath():
        return False
    if query1.GetJointOrder() != query2.GetJointOrder():
        return False
    if query1.GetBlendShapeOrder() != query2.GetBlendShapeOrder():
        return False
    times1 = Vt.Vec3fArray()
    times2 = Vt.Vec3fArray()
    if query1.GetJointTransformTimeSamples(times1) != query2.GetJointTransformTimeSamples(times2):
        return False
    if times1 != times2:
        return False
    times1 = Vt.FloatArray()
    times2 = Vt.FloatArray()
    if query1.GetBlendShapeWeightTimeSamples(times1) != query2.GetBlendShapeWeightTimeSamples(times2):
        return False
    if times1 != times2:
        return False
    return True


def get_joint_hierarchy(topology: UsdSkel.Topology) -> Dict[int, List[int]]:
    """
    Builds a dictionary representing the joint hierarchy.

    Args:
        topology (UsdSkel.Topology): The topology to query.

    Returns:
        Dict[int, List[int]]: A dictionary where each key is a joint index and the corresponding
        value is a list of child joint indices. Joints with no children have an empty list.
    """
    hierarchy: Dict[int, List[int]] = {}
    num_joints = topology.GetNumJoints()
    for i in range(num_joints):
        hierarchy[i] = []
    for i in range(num_joints):
        parent_index = topology.GetParent(i)
        if parent_index != -1:
            hierarchy[parent_index].append(i)
    return hierarchy


def create_and_assign_rotations(
    anim: UsdSkel.Animation, rotations: List[Gf.Quatf], time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> None:
    """Create the rotations attribute on the animation primitive if it doesn't exist, and set its value at the given time code.

    Args:
        anim (UsdSkel.Animation): The animation primitive.
        rotations (List[Gf.Quatf]): The list of rotation values to assign.
        time_code (Usd.TimeCode, optional): The time code at which to set the value. Defaults to Usd.TimeCode.Default().

    Raises:
        ValueError: If the length of rotations does not match the number of joints in the animation.
    """
    joints_attr = anim.GetJointsAttr()
    if not joints_attr.IsValid():
        raise ValueError("The animation primitive does not have a valid joints attribute.")
    num_joints = len(joints_attr.Get())
    if len(rotations) != num_joints:
        raise ValueError(
            f"The length of rotations ({len(rotations)}) does not match the number of joints ({num_joints})."
        )
    rotations_attr = anim.GetRotationsAttr()
    if not rotations_attr.IsValid():
        rotations_attr = anim.CreateRotationsAttr()
    rotations_attr.Set(rotations, time_code)


def create_and_assign_scales(
    anim: UsdSkel.Animation, scales: Sequence[Gf.Vec3h], time: Usd.TimeCode = Usd.TimeCode.Default()
) -> bool:
    """
    Create the scales attribute on the animation if it doesn't exist, and set the scales value at the specified time.

    Args:
        anim (UsdSkel.Animation): The animation to create the scales attribute on and set the value.
        scales (Sequence[Gf.Vec3h]): The scales values to set on the attribute. Should match the number of joints.
        time (Usd.TimeCode): The time to set the value at. Defaults to Default time.

    Returns:
        bool: True if the scales were set successfully, False otherwise.
    """
    scales_attr = anim.GetScalesAttr()
    if not scales_attr:
        scales_attr = anim.CreateScalesAttr()
    if len(scales) != len(anim.GetJointsAttr().Get()):
        print(f"Number of scales ({len(scales)}) does not match number of joints ({len(anim.GetJointsAttr().Get())})")
        return False
    try:
        scales_attr.Set(scales, time=time)
        return True
    except Exception as e:
        print(f"Error setting scales: {e}")
        return False


def get_animation_transforms(anim: UsdSkel.Animation, time_code: Usd.TimeCode) -> Tuple[bool, Vt.Matrix4dArray]:
    """
    Get the transforms for a UsdSkelAnimation at a specific time code.

    Args:
        anim (UsdSkel.Animation): The animation to query transforms from.
        time_code (Usd.TimeCode): The time code at which to query the transforms.

    Returns:
        Tuple[bool, Vt.Matrix4dArray]: A tuple containing a bool indicating if the query was
        successful and the array of transforms. If unsuccessful, the array will be empty.
    """
    if not anim.GetPrim().IsValid():
        return (False, Vt.Matrix4dArray())
    xforms = Vt.Matrix4dArray()
    result = anim.GetTransforms(xforms, time_code)
    return (result, xforms)


def create_and_assign_blend_shape_weights(
    anim_prim: UsdSkel.Animation, blend_shape_names: List[str], weights: List[float]
) -> None:
    """Create the blendShapeWeights attribute on the given animation primitive and assign the provided weights.

    Args:
        anim_prim (UsdSkel.Animation): The animation primitive to create the blendShapeWeights attribute on.
        blend_shape_names (List[str]): The names of the blend shapes to create weights for.
        weights (List[float]): The weight values to assign to each blend shape.

    Raises:
        ValueError: If the number of blend shape names and weights do not match.
    """
    if len(blend_shape_names) != len(weights):
        raise ValueError("Number of blend shape names and weights must match.")
    blend_shape_tokens = [Sdf.ValueTypeNames.Token] * len(blend_shape_names)
    for i, name in enumerate(blend_shape_names):
        blend_shape_tokens[i] = name
    blend_shapes_attr = anim_prim.CreateBlendShapesAttr()
    blend_shapes_attr.Set(blend_shape_tokens)
    blend_shape_weights_attr = anim_prim.CreateBlendShapeWeightsAttr()
    blend_shape_weights_attr.Set(weights)


def create_and_assign_blend_shapes(prim: UsdSkel.Skeleton, blend_shape_names: List[str]) -> UsdSkel.Animation:
    """Create a UsdSkelAnimation prim with blendshapes and assign it to a skeleton prim."""
    if not prim or not prim.GetPrim().IsValid():
        raise ValueError("Invalid skeleton prim.")
    if not blend_shape_names:
        raise ValueError("Blend shape names list is empty.")
    anim_prim = UsdSkel.Animation.Define(prim.GetPrim().GetStage(), prim.GetPrim().GetPath().AppendChild("Animation"))
    if not anim_prim:
        raise RuntimeError("Failed to create animation prim.")
    anim_prim.CreateBlendShapesAttr(blend_shape_names, True)
    num_blend_shapes = len(blend_shape_names)
    default_weights = [0.0] * num_blend_shapes
    anim_prim.CreateBlendShapeWeightsAttr(default_weights, True)
    binding_api = UsdSkel.BindingAPI(prim)
    binding_api.CreateAnimationSourceRel().AddTarget(anim_prim.GetPath())
    return anim_prim


def create_and_assign_translations(
    anim: UsdSkel.Animation, translations: Sequence[Gf.Vec3f], time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> None:
    """Create the translations attribute on the animation primitive if it doesn't exist, and set the translations.

    Args:
        anim (UsdSkel.Animation): The animation primitive.
        translations (Sequence[Gf.Vec3f]): The translations to set on the attribute.
        time_code (Usd.TimeCode): The time code at which to set the translations. Defaults to Default time code.

    Raises:
        ValueError: If the size of translations doesn't match the number of joints in the animation.
    """
    translations_attr = anim.GetTranslationsAttr()
    if not translations_attr:
        translations_attr = anim.CreateTranslationsAttr()
    num_joints = len(anim.GetJointsAttr().Get())
    if len(translations) != num_joints:
        raise ValueError(
            f"Number of translations ({len(translations)}) must match the number of joints ({num_joints})."
        )
    translations_attr.Set(translations, time_code)


def get_animation_attributes(
    anim: UsdSkel.Animation,
) -> Tuple[List[str], List[str], List[Gf.Quatf], List[Gf.Vec3h], List[Gf.Vec3f], List[float]]:
    """
    Get the animation attributes for a UsdSkelAnimation prim.

    Args:
        anim (UsdSkel.Animation): The input UsdSkelAnimation object.

    Returns:
        A tuple containing the following:
        - joints (List[str]): List of joint names.
        - blend_shapes (List[str]): List of blend shape names.
        - rotations (List[Gf.Quatf]): List of joint rotations.
        - scales (List[Gf.Vec3h]): List of joint scales.
        - translations (List[Gf.Vec3f]): List of joint translations.
        - blend_shape_weights (List[float]): List of blend shape weights.

    Raises:
        ValueError: If the input object is not a valid UsdSkelAnimation.
    """
    if not anim.GetPrim().IsValid():
        raise ValueError(f"Object '{anim.GetPrim().GetPath()}' is not a valid UsdSkelAnimation.")
    joints = anim.GetJointsAttr().Get() if anim.GetJointsAttr().HasValue() else []
    blend_shapes = anim.GetBlendShapesAttr().Get() if anim.GetBlendShapesAttr().HasValue() else []
    rotations = anim.GetRotationsAttr().Get() if anim.GetRotationsAttr().HasValue() else []
    scales = anim.GetScalesAttr().Get() if anim.GetScalesAttr().HasValue() else []
    translations = anim.GetTranslationsAttr().Get() if anim.GetTranslationsAttr().HasValue() else []
    blend_shape_weights = anim.GetBlendShapeWeightsAttr().Get() if anim.GetBlendShapeWeightsAttr().HasValue() else []
    return (joints, blend_shapes, rotations, scales, translations, blend_shape_weights)


def sample_animation_at_time(anim: UsdSkel.Animation, time: Usd.TimeCode) -> Tuple[Vt.Matrix4dArray, Vt.TokenArray]:
    """Sample the animation at a specific time and return the joint transforms and names.

    Args:
        anim (UsdSkel.Animation): The animation to sample.
        time (Usd.TimeCode): The time at which to sample the animation.

    Returns:
        Tuple[Vt.Matrix4dArray, Vt.TokenArray]: A tuple containing the joint transforms and joint names.
    """
    num_joints = len(anim.GetJointsAttr().Get())
    xforms = Vt.Matrix4dArray(num_joints)
    joint_names = anim.GetJointsAttr().Get()
    if num_joints == 0:
        return (xforms, joint_names)
    translations = anim.GetTranslationsAttr().Get(time)
    rotations = anim.GetRotationsAttr().Get(time)
    scales = anim.GetScalesAttr().Get(time)
    for i in range(num_joints):
        translation = translations[i] if translations else Gf.Vec3d(0, 0, 0)
        rotation = rotations[i] if rotations else Gf.Quatf(1, 0, 0, 0)
        scale = scales[i] if scales else Gf.Vec3h(1, 1, 1)
        xform = Gf.Matrix4d()
        xform.SetTranslate(Gf.Vec3d(translation))
        xform.SetRotateOnly(Gf.Rotation(rotation))
        xform.SetScale(Gf.Vec3d(scale))
        xforms[i] = xform
    return (xforms, joint_names)


def copy_animation_prim(source_prim: UsdSkel.Animation, dest_prim: UsdSkel.Animation) -> bool:
    """Copy all animation data from source_prim to dest_prim."""
    if not source_prim.GetPrim().IsValid() or not dest_prim.GetPrim().IsValid():
        raise ValueError("Invalid source or destination prim.")
    attrs_to_copy = ["joints", "translations", "rotations", "scales", "blendShapes", "blendShapeWeights"]
    for attr_name in attrs_to_copy:
        source_attr = source_prim.GetPrim().GetAttribute(attr_name)
        if source_attr.IsValid():
            attr_value = source_attr.Get()
            dest_attr = dest_prim.GetPrim().CreateAttribute(attr_name, source_attr.GetTypeName())
            dest_attr.Set(attr_value)
    return True


def set_animation_transforms(animation: UsdSkel.Animation, xforms: Vt.Matrix4dArray, time: Usd.TimeCode) -> bool:
    """
    Set the transforms for an animation at a specific time.

    Args:
        animation (UsdSkel.Animation): The animation to set transforms for.
        xforms (Vt.Matrix4dArray): The array of transforms to set.
        time (Usd.TimeCode): The time at which to set the transforms.

    Returns:
        bool: True if the transforms were set successfully, False otherwise.
    """
    if not animation.GetPrim().IsValid():
        return False
    num_joints = len(animation.GetJointsAttr().Get())
    if num_joints != len(xforms):
        return False
    translations = []
    rotations = []
    scales = []
    for xform in xforms:
        gf_matrix = Gf.Matrix4d(xform)
        translations.append(gf_matrix.ExtractTranslation())
        rotations.append(gf_matrix.ExtractRotation().GetQuat())
        scales.append(Gf.Vec3f(1.0, 1.0, 1.0))
    try:
        animation.GetTranslationsAttr().Set(translations, time)
        animation.GetRotationsAttr().Set(rotations, time)
        animation.GetScalesAttr().Set(scales, time)
    except Exception as e:
        print(f"Error setting transforms: {e}")
        return False
    return True


def merge_animation_prims(stage: Usd.Stage, src_prim_path: str, dst_prim_path: str) -> bool:
    """
    Merge the data from the source animation prim into the destination animation prim.

    Args:
        stage (Usd.Stage): The stage containing the animation prims.
        src_prim_path (str): The path to the source animation prim.
        dst_prim_path (str): The path to the destination animation prim.

    Returns:
        bool: True if the merge was successful, False otherwise.
    """
    src_prim = stage.GetPrimAtPath(src_prim_path)
    dst_prim = stage.GetPrimAtPath(dst_prim_path)
    if not src_prim.IsValid() or not dst_prim.IsValid():
        print(f"Error: Invalid prim paths. Source: {src_prim_path}, Destination: {dst_prim_path}")
        return False
    if not src_prim.IsA(UsdSkel.Animation) or not dst_prim.IsA(UsdSkel.Animation):
        print(f"Error: Prims are not of type UsdSkelAnimation. Source: {src_prim_path}, Destination: {dst_prim_path}")
        return False
    attrs_to_merge = ["joints", "translations", "rotations", "scales", "blendShapes", "blendShapeWeights"]
    for attr_name in attrs_to_merge:
        src_attr = src_prim.GetAttribute(attr_name)
        if not src_attr.IsValid():
            continue
        dst_attr = dst_prim.GetAttribute(attr_name)
        if not dst_attr.IsValid():
            dst_attr = dst_prim.CreateAttribute(attr_name, src_attr.GetTypeName())
        src_values = src_attr.Get()
        dst_values = dst_attr.Get()
        if src_values is not None and dst_values is not None:
            if isinstance(src_values, list) and isinstance(dst_values, list):
                merged_values = dst_values + src_values
            else:
                merged_values = list(dst_values) + list(src_values)
            dst_attr.Set(merged_values)
        elif src_values is not None:
            dst_attr.Set(src_values)
    return True


def blend_animations(anim_a: UsdSkel.Animation, anim_b: UsdSkel.Animation, amount: float) -> UsdSkel.Animation:
    """
    Blend two UsdSkelAnimation prims together with a given blend amount.

    Args:
        anim_a (UsdSkel.Animation): The first animation to blend.
        anim_b (UsdSkel.Animation): The second animation to blend.
        amount (float): The blend amount between 0 and 1. 0 means fully anim_a, 1 means fully anim_b.

    Returns:
        UsdSkel.Animation: A new animation with the blended values.
    """
    if not anim_a.GetPrim().IsValid() or not anim_b.GetPrim().IsValid():
        raise ValueError("Both input animations must be valid.")
    if not 0 <= amount <= 1:
        raise ValueError("Blend amount must be between 0 and 1.")
    stage = anim_a.GetPrim().GetStage()
    blend_anim = UsdSkel.Animation.Define(stage, "/BlendAnim")
    translations_a = anim_a.GetTranslationsAttr().Get()
    translations_b = anim_b.GetTranslationsAttr().Get()
    blended_translations = [Gf.Lerp(amount, Gf.Vec3f(a), Gf.Vec3f(b)) for (a, b) in zip(translations_a, translations_b)]
    blend_anim.CreateTranslationsAttr(blended_translations)
    rotations_a = anim_a.GetRotationsAttr().Get()
    rotations_b = anim_b.GetRotationsAttr().Get()
    blended_rotations = [Gf.Slerp(amount, a, b) for (a, b) in zip(rotations_a, rotations_b)]
    blend_anim.CreateRotationsAttr(blended_rotations)
    scales_a = anim_a.GetScalesAttr().Get()
    scales_b = anim_b.GetScalesAttr().Get()
    blended_scales = [Gf.Lerp(amount, Gf.Vec3f(a), Gf.Vec3f(b)) for (a, b) in zip(scales_a, scales_b)]
    blend_anim.CreateScalesAttr(blended_scales)
    blend_anim.GetJointsAttr().Set(anim_a.GetJointsAttr().Get())
    blend_anim.GetBlendShapesAttr().Set(anim_a.GetBlendShapesAttr().Get())
    return blend_anim


def define_animation_prim(stage: Usd.Stage, prim_path: Union[str, Sdf.Path]) -> UsdSkel.Animation:
    """Define a new UsdSkelAnimation prim at the given path on the stage.

    If a prim already exists at the given path, it will be modified to be a
    UsdSkelAnimation prim. If no prim exists at the path, a new one will be created.
    Any missing ancestor prims will also be created.

    Parameters:
        stage (Usd.Stage): The stage on which to create the prim.
        prim_path (str or Sdf.Path): The path at which to create the prim.

    Returns:
        UsdSkel.Animation: The newly created or modified UsdSkelAnimation prim.

    Raises:
        ValueError: If the given prim path is not a valid path for the current stage.
    """
    if isinstance(prim_path, str):
        prim_path = Sdf.Path(prim_path)
    if not Sdf.Path.IsValidPathString(str(prim_path)):
        raise ValueError(f"Invalid prim path '{prim_path}' for the current stage.")
    anim_prim = UsdSkel.Animation.Define(stage, prim_path)
    return anim_prim


def test_define_animation_prim():
    stage = Usd.Stage.CreateInMemory()
    anim_prim = define_animation_prim(stage, "/Animations/Run")
    assert anim_prim.GetPath() == Sdf.Path("/Animations/Run")
    assert anim_prim.GetPrim().GetTypeName() == "SkelAnimation"
    assert stage.GetPrimAtPath("/Animations").IsValid()
    anim_prim2 = define_animation_prim(stage, "/Animations")
    assert anim_prim2.GetPath() == Sdf.Path("/Animations")
    assert anim_prim2.GetPrim().GetTypeName() == "SkelAnimation"
    print(anim_prim.GetPath())
    print(anim_prim2.GetPath())


def assign_animation_to_skeleton(stage: Usd.Stage, skeleton_path: str, animation_path: str) -> bool:
    """Assign an animation to a skeleton.

    Args:
        stage (Usd.Stage): The stage containing the skeleton and animation prims.
        skeleton_path (str): The path to the skeleton prim.
        animation_path (str): The path to the animation prim.

    Returns:
        bool: True if the assignment was successful, False otherwise.
    """
    skeleton_prim = stage.GetPrimAtPath(skeleton_path)
    if not skeleton_prim.IsValid():
        print(f"Error: Skeleton prim at path {skeleton_path} does not exist.")
        return False
    animation_prim = stage.GetPrimAtPath(animation_path)
    if not animation_prim.IsValid():
        print(f"Error: Animation prim at path {animation_path} does not exist.")
        return False
    skeleton = UsdSkel.Skeleton(skeleton_prim)
    if not skeleton:
        print(f"Error: Failed to get Skeleton schema for prim at path {skeleton_path}")
        return False
    skel_binding_api = UsdSkel.BindingAPI.Apply(skeleton_prim)
    if not skel_binding_api:
        print(f"Error: Failed to apply BindingAPI to skeleton prim at path {skeleton_path}")
        return False
    anim_source_attr = skel_binding_api.GetAnimationSourceRel()
    try:
        anim_source_attr.SetTargets([animation_prim.GetPath()])
    except Exception as e:
        print(f"Error: Failed to set animation source: {str(e)}")
        return False
    return True


def unbind_skeleton_from_prims(stage: Usd.Stage, skeleton_path: str, prim_paths: List[str]):
    """
    Unbinds a skeleton from a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        skeleton_path (str): The path to the skeleton prim.
        prim_paths (List[str]): The paths to the prims to unbind the skeleton from.
    """
    skeleton_prim = stage.GetPrimAtPath(skeleton_path)
    if not skeleton_prim.IsValid():
        raise ValueError(f"Skeleton prim at path {skeleton_path} does not exist.")
    skeleton = UsdSkel.Skeleton(skeleton_prim)
    if not skeleton:
        raise ValueError(f"Prim at path {skeleton_path} is not a valid skeleton.")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        binding_api = UsdSkel.BindingAPI(prim)
        if not binding_api:
            raise ValueError(f"Prim at path {prim_path} does not have a binding API.")
        binding_api.GetSkeletonRel().ClearTargets(removeSpec=True)


def create_skeleton_rel(prim: Usd.Prim, skeleton_path: str) -> UsdSkel.BindingAPI:
    """Create a skeleton binding relationship on the given prim, pointing to the specified skeleton.

    Args:
        prim (Usd.Prim): The prim to create the skeleton binding on.
        skeleton_path (str): The path to the skeleton prim.

    Returns:
        UsdSkel.BindingAPI: The created BindingAPI object.

    Raises:
        ValueError: If the prim is not valid or if the skeleton path is empty.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if not skeleton_path:
        raise ValueError("Skeleton path cannot be empty.")
    binding_api = UsdSkel.BindingAPI(prim)
    skel_rel = binding_api.GetSkeletonRel()
    skel_rel.SetTargets([skeleton_path])
    return binding_api


def test_create_skeleton_rel():
    stage = Usd.Stage.CreateInMemory()
    skel_path = "/Skeleton"
    skel = UsdSkel.Skeleton.Define(stage, skel_path)
    mesh_path = "/Mesh"
    mesh = stage.DefinePrim(mesh_path, "Mesh")
    binding_api = create_skeleton_rel(mesh, skel_path)
    assert binding_api.GetSkeleton().GetPath() == skel.GetPath()


def create_joint_weights_attr(prim: Usd.Prim, attribute_name: str, default_value: Vt.FloatArray) -> Usd.Attribute:
    """Creates a jointWeights attribute for a prim with given default value.

    Args:
        prim (Usd.Prim): The prim to create the attribute on.
        attribute_name (str): The name of the attribute.
        default_value (Vt.FloatArray): The default value for the attribute.

    Returns:
        Usd.Attribute: The created attribute or existing one if already present.

    Raises:
        ValueError: If the prim is not valid.
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid")
    if prim.HasAttribute(attribute_name):
        attr = prim.GetAttribute(attribute_name)
        if attr.GetTypeName() == Sdf.ValueTypeNames.FloatArray:
            return attr
        else:
            raise ValueError(
                f"Attribute {attribute_name} already exists on prim {prim.GetPath()} with a different type"
            )
    attr: Usd.Attribute = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.FloatArray)
    if default_value is not None:
        attr.Set(default_value)
    return attr


def create_joint_indices_primvar(
    prim: Usd.Prim, constant: bool, elementSize: int, indices: Vt.IntArray
) -> UsdGeom.Primvar:
    """Create a 'jointIndices' primvar on the given prim with specified properties.

    Args:
        prim (Usd.Prim): The prim to create the primvar on.
        constant (bool): Whether the primvar should have constant interpolation.
        elementSize (int): The element size of the primvar.
        indices (Vt.IntArray): The indices data to set on the primvar.

    Returns:
        UsdGeom.Primvar: The created 'jointIndices' primvar.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if elementSize <= 0:
        raise ValueError("Element size must be greater than zero.")
    if len(indices) % elementSize != 0:
        raise ValueError("Indices array size must be divisible by element size.")
    primvar = UsdSkel.BindingAPI(prim).CreateJointIndicesPrimvar(constant, elementSize)
    primvar.Set(indices)
    return primvar


def create_joints_attr(prim: Usd.Prim, joints: List[str], elementSize: int = -1) -> UsdSkel.BindingAPI:
    """Create the joints attribute on the given prim.

    If elementSize is -1, it will be set to len(joints).
    If elementSize is greater than len(joints), the joints will be
    padded with empty strings to reach elementSize.
    If elementSize is less than len(joints), the joints will be
    truncated to elementSize.

    Args:
        prim (Usd.Prim): The prim to create the attribute on.
        joints (List[str]): The list of joint names.
        elementSize (int): The elementSize of the attribute. Defaults to -1.

    Returns:
        UsdSkel.BindingAPI: The UsdSkel.BindingAPI object for the prim.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if elementSize == -1:
        elementSize = len(joints)
    if len(joints) < elementSize:
        joints.extend([""] * (elementSize - len(joints)))
    else:
        joints = joints[:elementSize]
    binding_api = UsdSkel.BindingAPI(prim)
    joints_attr = binding_api.CreateJointsAttr()
    joints_attr.Set(joints)
    return binding_api


def get_geom_bind_transform_attr(prim: Usd.Prim) -> Gf.Matrix4d:
    """
    Get the geomBindTransform attribute value for a prim.

    If the attribute is not defined, an identity matrix is returned.
    If the attribute is defined but has no value, an identity matrix is returned.
    """
    attr = UsdSkel.BindingAPI(prim).GetGeomBindTransformAttr()
    if not attr.IsValid():
        return Gf.Matrix4d(1.0)
    value = attr.Get()
    if value is None:
        return Gf.Matrix4d(1.0)
    return value


def get_joint_indices_attr(prim: Usd.Prim) -> Vt.IntArray:
    """
    Get the jointIndices attribute value for a prim.

    Args:
        prim (Usd.Prim): The prim to retrieve the jointIndices attribute from.

    Returns:
        Vt.IntArray: The jointIndices attribute value, or an empty array if not found.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    binding_api = UsdSkel.BindingAPI(prim)
    if binding_api.GetJointIndicesAttr().HasValue():
        joint_indices = binding_api.GetJointIndicesAttr().Get()
        if not isinstance(joint_indices, Vt.IntArray):
            raise TypeError(f"jointIndices attribute value is not of type Vt.IntArray.")
        return joint_indices
    else:
        return Vt.IntArray()


def get_joint_weights_attr(prim: Usd.Prim) -> Vt.FloatArray:
    """
    Get the jointWeights attribute value for a prim.

    Args:
        prim (Usd.Prim): The prim to get the jointWeights attribute from.

    Returns:
        Vt.FloatArray: The jointWeights attribute value, or an empty array if not defined.

    Raises:
        ValueError: If the prim is not valid.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim '{prim.GetPath()}' is not valid.")
    binding_api = UsdSkel.BindingAPI(prim)
    if binding_api.GetJointWeightsAttr().HasValue():
        joint_weights = binding_api.GetJointWeightsAttr().Get()
        if isinstance(joint_weights, Vt.FloatArray):
            return joint_weights
        else:
            print(
                f"Warning: jointWeights attribute on prim '{prim.GetPath()}' is not a float array. Returning empty array."
            )
            return Vt.FloatArray()
    else:
        print(f"jointWeights attribute not defined on prim '{prim.GetPath()}'. Returning empty array.")
        return Vt.FloatArray()


def get_joints_attr(prim: Usd.Prim) -> Vt.TokenArray:
    """
    Get the value of the skel:joints attribute on the given prim.

    If the attribute is not defined, this returns an empty VtTokenArray.
    If the attribute is defined but has no value, this returns an empty VtTokenArray.
    """
    binding_api = UsdSkel.BindingAPI(prim)
    if binding_api.GetJointsAttr().HasValue():
        return binding_api.GetJointsAttr().Get()
    else:
        return Vt.TokenArray()


def get_skeleton_rel(prim: Usd.Prim) -> Usd.Relationship:
    """
    Get the 'skel:skeleton' relationship on the given prim.

    This relationship identifies the skeleton that should be bound to this prim
    and its descendents that possess a mapping to the skeleton's joints.

    Returns:
        Usd.Relationship: The 'skel:skeleton' relationship, or an invalid
        relationship if it doesn't exist.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    skel_rel = prim.GetRelationship("skel:skeleton")
    return skel_rel


def validate_joint_indices(indices: Sequence[int], num_joints: int) -> bool:
    """
    Validate an array of joint indices.

    This ensures that all indices are in the range [0, num_joints).
    Returns True if the indices are valid, or False otherwise.
    """
    if not indices:
        return False
    if any((index < 0 or index >= num_joints for index in indices)):
        return False
    return True


def create_blend_shape_targets_relation(prim: Usd.Prim, targets: List[Usd.Prim]) -> UsdSkel.BindingAPI:
    """Create a blend shape targets relationship on the given prim.

    Args:
        prim (Usd.Prim): The prim to create the relationship on.
        targets (List[Usd.Prim]): The list of target prims for the blend shape.

    Returns:
        UsdSkel.BindingAPI: The BindingAPI object for the prim.

    Raises:
        ValueError: If the prim is not valid or if the targets list is empty.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if not targets:
        raise ValueError("Empty targets list.")
    binding_api = UsdSkel.BindingAPI(prim)
    blend_shape_targets_rel = binding_api.GetBlendShapeTargetsRel()
    if not blend_shape_targets_rel:
        blend_shape_targets_rel = binding_api.CreateBlendShapeTargetsRel()
    blend_shape_targets_rel.ClearTargets(removeSpec=True)
    for target in targets:
        blend_shape_targets_rel.AddTarget(target.GetPath())
    return binding_api


def create_geom_bind_transform_attr(
    prim: Usd.Prim, default_value: Gf.Matrix4d = Gf.Matrix4d(1), write_sparsely: bool = False
) -> Usd.Attribute:
    """Create the geomBindTransform attribute on the given prim if it doesn't exist.

    Args:
        prim (Usd.Prim): The prim to create the attribute on.
        default_value (Gf.Matrix4d): The default value for the attribute. Defaults to identity matrix.
        write_sparsely (bool): Whether to write the attribute sparsely. Defaults to False.

    Returns:
        Usd.Attribute: The created or existing geomBindTransform attribute.
    """
    if UsdSkel.BindingAPI(prim).GetGeomBindTransformAttr():
        return UsdSkel.BindingAPI(prim).GetGeomBindTransformAttr()
    attr = UsdSkel.BindingAPI(prim).CreateGeomBindTransformAttr(default_value, write_sparsely)
    return attr


def get_blend_shape_targets_rel(prim: Usd.Prim) -> Usd.Relationship:
    """
    Get the blend shape targets relationship for a prim.

    Args:
        prim (Usd.Prim): The prim to get the blend shape targets from.

    Returns:
        Usd.Relationship: The blend shape targets relationship.

    Raises:
        ValueError: If the prim is not valid or does not have a blend shape targets relationship.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    binding_api = UsdSkel.BindingAPI(prim)
    blend_shape_targets_rel = binding_api.GetBlendShapeTargetsRel()
    if not blend_shape_targets_rel.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} does not have a valid blend shape targets relationship.")
    return blend_shape_targets_rel


def get_blend_shapes_attr(binding_api: UsdSkel.BindingAPI) -> Vt.TokenArray:
    """
    Get the blend shapes attribute from a UsdSkel.BindingAPI object.

    Args:
        binding_api (UsdSkel.BindingAPI): The binding API object to get the attribute from.

    Returns:
        Vt.TokenArray: The blend shapes token array value, or an empty array if not set.
    """
    if binding_api.GetBlendShapesAttr().HasValue():
        blend_shapes = binding_api.GetBlendShapesAttr().Get()
        if not isinstance(blend_shapes, Vt.TokenArray):
            blend_shapes = Vt.TokenArray(blend_shapes)
        return blend_shapes
    else:
        return Vt.TokenArray()


def get_skinning_method_attr(prim: Usd.Prim) -> Usd.Attribute:
    """Get the skinning method attribute for a prim."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    binding_api = UsdSkel.BindingAPI(prim)
    if not binding_api:
        raise ValueError(f"Prim {prim.GetPath()} does not have a BindingAPI applied.")
    skinning_method_attr = binding_api.GetSkinningMethodAttr()
    if not skinning_method_attr.IsDefined():
        skinning_method_attr = binding_api.CreateSkinningMethodAttr(UsdSkel.Tokens.classicLinear, True)
    return skinning_method_attr


def apply_skel_binding_api(prim: Usd.Prim) -> UsdSkel.BindingAPI:
    """Apply the UsdSkelBindingAPI to a prim.

    Args:
        prim (Usd.Prim): The prim to apply the API to.

    Returns:
        UsdSkel.BindingAPI: The applied BindingAPI schema object.

    Raises:
        ValueError: If the prim is not valid or the API cannot be applied.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if not UsdSkel.BindingAPI.CanApply(prim):
        raise ValueError("UsdSkelBindingAPI cannot be applied to this prim.")
    binding_api = UsdSkel.BindingAPI.Apply(prim)
    if not binding_api.GetPrim():
        raise ValueError("Failed to apply UsdSkelBindingAPI.")
    return binding_api


def can_apply_skel_binding_api(prim: Usd.Prim) -> bool:
    """Check if the SkelBindingAPI can be applied to the given prim."""
    if not prim.IsValid():
        return False
    prim_type = prim.GetTypeName()
    if prim_type not in ["Mesh", "PointInstancer"]:
        return False
    if UsdSkel.BindingAPI(prim):
        return False
    return True


def create_joint_indices_attr(prim: Usd.Prim, indices: Sequence[int], elementSize: int = -1) -> UsdSkel.BindingAPI:
    """Create the jointIndices attribute on the given prim with the provided indices.

    Args:
        prim (Usd.Prim): The prim to create the jointIndices attribute on.
        indices (Sequence[int]): The joint indices to set.
        elementSize (int): The element size of the indices array. If -1, it will be inferred.

    Returns:
        UsdSkel.BindingAPI: The BindingAPI object for the prim.
    """
    binding_api = UsdSkel.BindingAPI(prim)
    if not binding_api:
        binding_api = UsdSkel.BindingAPI.Apply(prim)
    joint_indices_attr = binding_api.GetJointIndicesAttr()
    if not joint_indices_attr:
        joint_indices_attr = binding_api.CreateJointIndicesAttr()
    if elementSize == -1:
        elementSize = 1
    try:
        joint_indices_attr.Set(indices)
        joint_indices_attr.SetMetadata("elementSize", elementSize)
    except Tf.ErrorException as e:
        print(f"Error setting jointIndices: {e}")
        return None
    return binding_api


def get_inherited_skeleton(prim: Usd.Prim) -> UsdSkel.Skeleton:
    """
    Get the skeleton bound at this prim or one of its ancestors.

    Returns:
        The bound skeleton if found, or an invalid skeleton if no binding is inherited.
    """
    binding_api = UsdSkel.BindingAPI(prim)
    skel_query = (
        UsdSkel.Skeleton.Get(prim.GetStage(), binding_api.GetSkeletonRel().GetTargets()[0])
        if binding_api.GetSkeletonRel().GetTargets()
        else UsdSkel.Skeleton()
    )
    if skel_query:
        return skel_query
    ancestor_prim = prim.GetParent()
    while ancestor_prim.IsValid():
        ancestor_binding_api = UsdSkel.BindingAPI(ancestor_prim)
        skel_query = (
            UsdSkel.Skeleton.Get(ancestor_prim.GetStage(), ancestor_binding_api.GetSkeletonRel().GetTargets()[0])
            if ancestor_binding_api.GetSkeletonRel().GetTargets()
            else UsdSkel.Skeleton()
        )
        if skel_query:
            return skel_query
        ancestor_prim = ancestor_prim.GetParent()
    return UsdSkel.Skeleton()


def get_inherited_animation_source(prim: Usd.Prim) -> Usd.Prim:
    """
    Returns the animation source bound at this prim, or one of its ancestors.

    Args:
        prim (Usd.Prim): The prim to query for the inherited animation source.

    Returns:
        Usd.Prim: The bound animation source prim, or an invalid prim if no
        animation source binding is found.
    """
    if not prim.IsValid():
        raise ValueError("Invalid input prim.")
    while prim.IsValid():
        binding_api = UsdSkel.BindingAPI(prim)
        if binding_api.GetAnimationSourceRel().HasAuthoredTargets():
            targets = binding_api.GetAnimationSourceRel().GetTargets()
            return targets[0]
        prim = prim.GetParent()
    return Usd.Prim()


def create_animation_source_relation(prim: Usd.Prim, target_prim: Usd.Prim) -> UsdSkel.BindingAPI:
    """Create an animation source relationship on the given prim targeting the specified prim.

    Args:
        prim (Usd.Prim): The prim to create the relationship on.
        target_prim (Usd.Prim): The target prim for the relationship.

    Returns:
        UsdSkel.BindingAPI: The BindingAPI for the prim with the created relationship.

    Raises:
        ValueError: If either prim is invalid or if the target prim is not a valid Skeleton.
    """
    if not prim.IsValid():
        raise ValueError("The given prim is not valid.")
    if not target_prim.IsValid():
        raise ValueError("The target prim is not valid.")
    if not UsdSkel.Skeleton(target_prim):
        raise ValueError("The target prim is not a valid Skeleton.")
    binding_api = UsdSkel.BindingAPI(prim)
    anim_source_rel = binding_api.CreateAnimationSourceRel()
    anim_source_rel.SetTargets([target_prim.GetPath()])
    return binding_api


def get_skinning_blend_weights_attr(prim: Usd.Prim) -> Vt.FloatArray:
    """
    Get the skinning blend weights attribute value for a prim.

    Args:
        prim (Usd.Prim): The prim to get the skinning blend weights from.

    Returns:
        Vt.FloatArray: The skinning blend weights value, or an empty array if not defined.

    Raises:
        ValueError: If the prim is not valid.
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid")
    skinning_blend_weights_attr = prim.GetAttribute("primvars:skel:skinningBlendWeights")
    if skinning_blend_weights_attr.HasValue():
        skinning_blend_weights = skinning_blend_weights_attr.Get()
        if not isinstance(skinning_blend_weights, Vt.FloatArray):
            skinning_blend_weights = Vt.FloatArray(skinning_blend_weights)
        return skinning_blend_weights
    else:
        return Vt.FloatArray()


def get_animation_source(prim: Usd.Prim) -> Tuple[bool, Usd.Prim]:
    """
    Convenience method to query the animation source bound on this prim.

    Returns a tuple of (bool, Usd.Prim). The bool is True if an animation
    source binding is defined, and False otherwise. The Usd.Prim is the
    target prim if the bool is True, or an invalid Prim if False.

    This does not resolve inherited animation source bindings.
    """
    binding_api = UsdSkel.BindingAPI(prim)
    anim_source_rel = binding_api.GetAnimationSourceRel()
    if not anim_source_rel or not anim_source_rel.GetTargets():
        return (False, Usd.Prim())
    anim_source_path = anim_source_rel.GetTargets()[0]
    if not anim_source_path:
        return (False, Usd.Prim())
    anim_source_prim = prim.GetStage().GetPrimAtPath(anim_source_path)
    if anim_source_prim.IsValid():
        return (True, anim_source_prim)
    else:
        return (False, Usd.Prim())


def set_blend_shape_offsets(
    blend_shape: UsdSkel.BlendShape, offsets: Sequence[Gf.Vec3f], indices: Optional[Sequence[int]] = None
):
    """Set the offsets and point indices for a blend shape.

    Args:
        blend_shape (UsdSkel.BlendShape): The blend shape to set offsets for.
        offsets (Sequence[Gf.Vec3f]): The offsets to set.
        indices (Optional[Sequence[int]]): The point indices corresponding to the offsets. If None, the offsets
            are assumed to be in the same order as the points in the base mesh. Defaults to None.

    Raises:
        ValueError: If the number of offsets and indices don't match, or if the indices are out of range.
    """
    offsets_attr = blend_shape.GetOffsetsAttr()
    if not offsets_attr:
        offsets_attr = blend_shape.CreateOffsetsAttr()
    offsets_attr.Set(offsets)
    if indices is not None:
        if len(indices) != len(offsets):
            raise ValueError("The number of indices must match the number of offsets.")
        indices_attr = blend_shape.GetPointIndicesAttr()
        if not indices_attr:
            indices_attr = blend_shape.CreatePointIndicesAttr()
        indices_attr.Set(indices)


def set_blend_shape_normal_offsets(blend_shape: UsdSkel.BlendShape, normal_offsets: Sequence[Gf.Vec3f]) -> None:
    """Set the normal offsets for a blend shape.

    Args:
        blend_shape (UsdSkel.BlendShape): The blend shape to set normal offsets for.
        normal_offsets (Sequence[Gf.Vec3f]): The normal offsets to set.

    Raises:
        ValueError: If the blend shape is invalid or the normal offsets are empty.
    """
    if not blend_shape.GetPrim().IsValid():
        raise ValueError("Invalid blend shape.")
    if not normal_offsets:
        raise ValueError("Normal offsets cannot be empty.")
    normal_offsets_attr = blend_shape.GetNormalOffsetsAttr()
    normal_offsets_attr.Set(normal_offsets)


def validate_blend_shape_point_indices(indices: Sequence[int], num_points: int) -> bool:
    """
    Validates a set of point indices for a given point count.

    This ensures that all point indices are in the range [0, num_points).
    Returns True if the indices are valid, False otherwise.

    Args:
        indices (Sequence[int]): The point indices to validate.
        num_points (int): The total number of points.

    Returns:
        bool: True if the indices are valid, False otherwise.
    """
    if not indices:
        return True
    min_index = min(indices)
    max_index = max(indices)
    if min_index < 0 or max_index >= num_points:
        return False
    if len(set(indices)) != len(indices):
        return False
    return True


def get_blend_shape_normal_offsets(blend_shape: UsdSkel.BlendShape) -> Tuple[Vt.Vec3fArray, bool]:
    """
    Get the normal offsets attribute value of a blend shape if it exists.

    Args:
        blend_shape (UsdSkel.BlendShape): The blend shape to get the normal offsets from.

    Returns:
        Tuple[Vt.Vec3fArray, bool]: A tuple containing the normal offsets array and a boolean
        indicating whether the attribute was authored or not. If the attribute doesn't exist
        or is not authored, an empty array is returned.
    """
    if not blend_shape:
        raise ValueError("Invalid blend shape.")
    normal_offsets_attr = blend_shape.GetNormalOffsetsAttr()
    if not normal_offsets_attr:
        return (Vt.Vec3fArray(), False)
    normal_offsets = normal_offsets_attr.Get()
    if normal_offsets is None:
        return (Vt.Vec3fArray(), False)
    return (normal_offsets, True)


def remove_blend_shape(prim: Usd.Prim, blend_shape_name: str) -> bool:
    """Remove a blend shape from a prim.

    Args:
        prim (Usd.Prim): The prim to remove the blend shape from.
        blend_shape_name (str): The name of the blend shape to remove.

    Returns:
        bool: True if the blend shape was successfully removed, False otherwise.
    """
    if not prim.IsValid():
        return False
    if not UsdSkel.BlendShape(prim):
        return False
    blend_shape = UsdSkel.BlendShape(prim)
    if not blend_shape.HasInbetween(blend_shape_name):
        return False
    inbetween = blend_shape.GetInbetween(blend_shape_name)
    inbetween.GetAttr().Clear()
    return True


def get_blend_shape_point_indices(blend_shape: UsdSkel.BlendShape) -> Vt.IntArray:
    """
    Get the point indices for a blend shape.

    Args:
        blend_shape (UsdSkel.BlendShape): The blend shape to get the point indices for.

    Returns:
        Vt.IntArray: The point indices for the blend shape.
    """
    point_indices_attr = blend_shape.GetPointIndicesAttr()
    if not point_indices_attr.IsValid():
        return Vt.IntArray()
    point_indices = point_indices_attr.Get()
    if point_indices is not None:
        return point_indices
    else:
        return Vt.IntArray()


def get_blend_shape_attributes(prim: Usd.Prim) -> Optional[List[UsdSkel.BlendShape]]:
    """
    Get the UsdSkelBlendShape objects for a given prim.

    Args:
        prim (Usd.Prim): The prim to get the blend shape attributes from.

    Returns:
        Optional[List[UsdSkel.BlendShape]]: A list of UsdSkelBlendShape objects if the prim has blend shapes,
                                            None otherwise.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    blend_shape = UsdSkel.BlendShape(prim)
    if not blend_shape:
        return None
    blend_shape_attrs = []
    offsets_attr = blend_shape.GetOffsetsAttr()
    if offsets_attr.HasValue():
        blend_shape_attrs.append(offsets_attr)
    normal_offsets_attr = blend_shape.GetNormalOffsetsAttr()
    if normal_offsets_attr.HasValue():
        blend_shape_attrs.append(normal_offsets_attr)
    point_indices_attr = blend_shape.GetPointIndicesAttr()
    if point_indices_attr.HasValue():
        blend_shape_attrs.append(point_indices_attr)
    return blend_shape_attrs


def get_blend_shape_offsets(blend_shape: UsdSkel.BlendShape) -> Tuple[Vt.Vec3fArray, Vt.IntArray]:
    """
    Get the offsets and point indices of a blend shape.

    Args:
        blend_shape (UsdSkel.BlendShape): The blend shape to retrieve offsets and indices from.

    Returns:
        Tuple[Vt.Vec3fArray, Vt.IntArray]: A tuple containing the offsets and point indices.
            - offsets (Vt.Vec3fArray): The position offsets of the blend shape.
            - indices (Vt.IntArray): The point indices corresponding to the offsets.

    Raises:
        ValueError: If the blend shape is invalid or has no offsets authored.
    """
    if not blend_shape or not blend_shape.GetPrim().IsValid():
        raise ValueError("Invalid blend shape.")
    offsets_attr = blend_shape.GetOffsetsAttr()
    if not offsets_attr.HasValue():
        raise ValueError("Blend shape has no offsets authored.")
    offsets = offsets_attr.Get()
    indices_attr = blend_shape.GetPointIndicesAttr()
    if indices_attr.HasValue():
        indices = indices_attr.Get()
    else:
        indices = Vt.IntArray(range(len(offsets)))
    return (offsets, indices)


def copy_blend_shape(source_prim: Usd.Prim, dest_prim: Usd.Prim) -> None:
    """Copy blend shape data from one prim to another."""
    source_blend_shape = UsdSkel.BlendShape(source_prim)
    if not source_blend_shape:
        raise ValueError(f"Source prim {source_prim.GetPath()} is not a valid BlendShape.")
    dest_blend_shape = UsdSkel.BlendShape(dest_prim)
    if not dest_blend_shape:
        raise ValueError(f"Destination prim {dest_prim.GetPath()} is not a valid BlendShape.")
    source_offsets_attr = source_blend_shape.GetOffsetsAttr()
    if source_offsets_attr.HasValue():
        dest_blend_shape.CreateOffsetsAttr(source_offsets_attr.Get(), source_offsets_attr.IsAuthored())
    source_normal_offsets_attr = source_blend_shape.GetNormalOffsetsAttr()
    if source_normal_offsets_attr.HasValue():
        dest_blend_shape.CreateNormalOffsetsAttr(
            source_normal_offsets_attr.Get(), source_normal_offsets_attr.IsAuthored()
        )
    source_point_indices_attr = source_blend_shape.GetPointIndicesAttr()
    if source_point_indices_attr.HasValue():
        dest_blend_shape.CreatePointIndicesAttr(source_point_indices_attr.Get(), source_point_indices_attr.IsAuthored())
    for source_inbetween_attr in source_prim.GetAttributes():
        if UsdSkel.InbetweenShape.IsInbetween(source_inbetween_attr):
            source_inbetween = UsdSkel.InbetweenShape(source_inbetween_attr)
            dest_inbetween = dest_blend_shape.CreateInbetween(source_inbetween_attr.GetName())
            source_inbetween_offsets_attr = source_inbetween.GetOffsets()
            if source_inbetween_offsets_attr:
                dest_inbetween.SetOffsets(source_inbetween_offsets_attr)
            source_inbetween_normal_offsets_attr = source_inbetween.GetNormalOffsets()
            if source_inbetween_normal_offsets_attr:
                dest_inbetween.SetNormalOffsets(source_inbetween_normal_offsets_attr)


def get_blend_shape_inbetweens(prim: Usd.Prim) -> List[UsdSkel.InbetweenShape]:
    """
    Get all the inbetween shapes for a given blend shape prim.

    Args:
        prim (Usd.Prim): The blend shape prim.

    Returns:
        List[UsdSkel.InbetweenShape]: A list of inbetween shapes.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    blend_shape = UsdSkel.BlendShape(prim)
    if not blend_shape:
        raise ValueError(f"Prim {prim} is not a blend shape")
    inbetweens = blend_shape.GetInbetweens()
    return inbetweens


def add_inbetween_to_blend_shape(
    blend_shape: UsdSkel.BlendShape, inbetween_name: str, weight: float
) -> UsdSkel.InbetweenShape:
    """Add an inbetween shape to a blend shape.

    Args:
        blend_shape (UsdSkel.BlendShape): The blend shape to add the inbetween to.
        inbetween_name (str): The name of the inbetween shape.
        weight (float): The weight of the inbetween shape.

    Returns:
        UsdSkel.InbetweenShape: The created inbetween shape.
    """
    if not blend_shape:
        raise ValueError("Invalid blend shape.")
    if blend_shape.HasInbetween(inbetween_name):
        raise ValueError(f"Inbetween '{inbetween_name}' already exists in the blend shape.")
    inbetween_shape = blend_shape.CreateInbetween(inbetween_name)
    inbetween_shape.SetWeight(weight)
    return inbetween_shape


def get_all_blend_shapes(blend_shape_query: UsdSkel.BlendShapeQuery) -> List[UsdSkel.BlendShape]:
    """
    Get all blend shapes associated with the given BlendShapeQuery.

    Args:
        blend_shape_query (UsdSkel.BlendShapeQuery): The BlendShapeQuery instance.

    Returns:
        List[UsdSkel.BlendShape]: A list of all blend shapes.
    """
    num_blend_shapes = blend_shape_query.GetNumBlendShapes()
    blend_shapes = []
    for i in range(num_blend_shapes):
        blend_shape = blend_shape_query.GetBlendShape(i)
        if blend_shape.IsValid():
            blend_shapes.append(blend_shape)
        else:
            raise ValueError(f"Invalid blend shape at index {i}")
    return blend_shapes


def compute_inbetween_weights(
    primary_weights: List[float], inbetweens: List[Tuple[int, int, float]]
) -> Tuple[List[float], List[int], List[int]]:
    """
    Compute the resolved weights for all sub-shapes, including inbetweens.

    Args:
        primary_weights (List[float]): The weights of the primary shapes.
        inbetweens (List[Tuple[int, int, float]]): List of inbetween tuples (primary1, primary2, weight).

    Returns:
        Tuple[List[float], List[int], List[int]]: sub-shape weights, blend shape indices, sub-shape indices.
    """
    sub_shape_weights = []
    blend_shape_indices = []
    sub_shape_indices = []
    for i, weight in enumerate(primary_weights):
        sub_shape_weights.append(weight)
        blend_shape_indices.append(i)
        sub_shape_indices.append(0)
    for inbetween in inbetweens:
        (primary1, primary2, inbetween_weight) = inbetween
        weight1 = primary_weights[primary1]
        weight2 = primary_weights[primary2]
        weight = weight1 + (weight2 - weight1) * inbetween_weight
        sub_shape_weights.append(weight)
        blend_shape_indices.append(primary1)
        sub_shape_indices.append(len(sub_shape_indices))
    return (sub_shape_weights, blend_shape_indices, sub_shape_indices)


def get_blend_shape_indices(blend_shape_query: UsdSkel.BlendShapeQuery, sub_shape_index: int) -> Tuple[int, int]:
    """
    Get the blend shape index and inbetween shape index for a given sub-shape index.

    Args:
        blend_shape_query (UsdSkel.BlendShapeQuery): The blend shape query object.
        sub_shape_index (int): The index of the sub-shape.

    Returns:
        Tuple[int, int]: A tuple containing the blend shape index and inbetween shape index.
                         If the sub-shape is not an inbetween, the second element will be -1.
    """
    num_sub_shapes = blend_shape_query.GetNumSubShapes()
    if sub_shape_index < 0 or sub_shape_index >= num_sub_shapes:
        raise ValueError(f"Invalid sub-shape index: {sub_shape_index}. Must be between 0 and {num_sub_shapes - 1}.")
    blend_shape_index = blend_shape_query.GetBlendShapeIndex(sub_shape_index)
    inbetween_shape = blend_shape_query.GetInbetween(sub_shape_index)
    inbetween_index = inbetween_shape.GetIndex() if inbetween_shape else -1
    return (blend_shape_index, inbetween_index)


def get_skeleton_animation(stage: Usd.Stage, skel_root_path: str, skel_cache: UsdSkel.Cache) -> UsdSkel.Animation:
    """
    Get the UsdSkel.Animation object for a skeleton root prim.

    Args:
        stage (Usd.Stage): The USD stage.
        skel_root_path (str): The path to the skeleton root prim.
        skel_cache (UsdSkel.Cache): The UsdSkel cache instance.

    Returns:
        UsdSkel.Animation: The animation object for the skeleton.

    Raises:
        ValueError: If the skeleton root prim is not valid or if no animation is found.
    """
    skel_root = stage.GetPrimAtPath(skel_root_path)
    if not skel_root.IsValid():
        raise ValueError(f"Invalid skeleton root prim path: {skel_root_path}")
    if not skel_cache.Populate(UsdSkel.Root(skel_root), Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        raise ValueError(f"Failed to populate cache for skeleton root: {skel_root_path}")
    skel_query = skel_cache.GetSkelQuery(UsdSkel.Skeleton(skel_root))
    if not skel_query:
        raise ValueError(f"No skeleton found at path: {skel_root_path}")
    anim_source_prim = skel_query.GetAnimationSource()
    if not anim_source_prim:
        raise ValueError(f"No animation source found for skeleton at path: {skel_root_path}")
    anim_query = skel_cache.GetAnimQuery(anim_source_prim)
    if not anim_query:
        raise ValueError(f"No animation found for skeleton at path: {skel_root_path}")
    return anim_query.GetAnimation()


def clear_and_repopulate_cache(cache: UsdSkel.Cache, root: UsdSkel.Root, predicate: Usd._PrimFlagsPredicate) -> bool:
    """Clear the cache and repopulate it with skeletal data beneath the given root prim."""
    cache.Clear()
    success = cache.Populate(root, predicate)
    return success


def apply_animation_to_skeleton(cache: UsdSkel.Cache, skel: UsdSkel.Skeleton, anim_prim: Usd.Prim) -> bool:
    """Apply the animation at anim_prim to the given Skeleton using the UsdSkel.Cache."""
    skel_query = cache.GetSkelQuery(skel)
    if not skel_query:
        raise ValueError("Failed to get SkeletonQuery for Skeleton prim.")
    anim_query = cache.GetAnimQuery(anim_prim)
    if not anim_query:
        raise ValueError("Failed to get AnimQuery for animation prim.")
    if (
        not anim_query.JointTransformsMightBeTimeVarying()
        or not skel_query.GetAnimQuery()
        or (not skel_query.GetAnimQuery().JointTransformsMightBeTimeVarying())
    ):
        print("Animation is not compatible with the skeleton or has no time samples.")
        return False
    num_time_samples = anim_query.GetJointTransformTimeSamples().size()
    for i in range(num_time_samples):
        time = anim_query.GetJointTransformTimeSamples()[i]
        xforms = anim_query.ComputeJointLocalTransforms(time)
        skel_query.SetJointLocalTransforms(xforms, time)
    return True


def blendshape_remove_inbetween(blendshape: UsdSkel.BlendShape, inbetween_name: str) -> bool:
    """Remove an inbetween shape from a blendshape by name.

    Args:
        blendshape (UsdSkel.BlendShape): The blendshape to remove the inbetween from.
        inbetween_name (str): The name of the inbetween shape to remove.

    Returns:
        bool: True if the inbetween was successfully removed, False otherwise.
    """
    if not blendshape.GetPrim().IsValid():
        return False
    inbetween_attr = blendshape.GetPrim().GetAttribute(inbetween_name)
    if not inbetween_attr.IsValid():
        return False
    if not UsdSkel.InbetweenShape.IsInbetween(inbetween_attr):
        return False
    success = inbetween_attr.GetPrim().RemoveProperty(inbetween_attr.GetName())
    return success


def blendshape_normalize_weights(weights: Sequence[float]) -> List[float]:
    """Normalize blendshape weights to sum up to 1.0.

    If the total sum of weights is zero, return the original weights.

    Args:
        weights (Sequence[float]): The input weights to normalize.

    Returns:
        List[float]: The normalized weights.
    """
    if not weights:
        return []
    total_weight = sum(weights)
    if total_weight == 0:
        return list(weights)
    normalized_weights = [weight / total_weight for weight in weights]
    return normalized_weights


def blendshape_scale_inbetween_offsets(inbetween: UsdSkel.InbetweenShape, scale: float) -> bool:
    """Scale the offsets of an inbetween shape by a given scale factor.

    Args:
        inbetween (UsdSkel.InbetweenShape): The inbetween shape to modify.
        scale (float): The scale factor to apply to the offsets.

    Returns:
        bool: True if the offsets were successfully scaled, False otherwise.
    """
    if not inbetween.IsDefined():
        return False
    offsets = inbetween.GetOffsets()
    if not offsets:
        return False
    scaled_offsets = Vt.Vec3fArray([offset * scale for offset in offsets])
    if not inbetween.SetOffsets(scaled_offsets):
        return False
    if inbetween.GetNormalOffsetsAttr().IsDefined():
        normal_offsets = inbetween.GetNormalOffsets()
        if not normal_offsets:
            return False
        scaled_normal_offsets = Vt.Vec3fArray([offset * scale for offset in normal_offsets])
        if not inbetween.SetNormalOffsets(scaled_normal_offsets):
            return False
    return True


def get_all_skel_animations(stage: Usd.Stage) -> List[UsdSkel.Animation]:
    """
    Retrieve all Skel Animation prims in the given stage.

    Args:
        stage (Usd.Stage): The USD stage to search for Skel Animation prims.

    Returns:
        List[UsdSkel.Animation]: A list of all Skel Animation prims found in the stage.
    """
    skel_animations = []
    for prim in stage.Traverse():
        if prim.IsA(UsdSkel.Animation):
            skel_animation = UsdSkel.Animation(prim)
            skel_animations.append(skel_animation)
    return skel_animations


def define_or_get_skel_root(stage: Usd.Stage, prim_path: str) -> UsdSkel.Root:
    """
    Define a new UsdSkelRoot prim at the given path if it doesn't exist,
    or return the existing UsdSkelRoot prim if it does.

    Parameters:
        stage (Usd.Stage): The stage to create the prim on.
        prim_path (str): The path where the prim should be created or retrieved.

    Returns:
        UsdSkel.Root: The UsdSkelRoot prim created or retrieved.
    """
    existing_prim = stage.GetPrimAtPath(prim_path)
    if existing_prim.IsValid():
        skel_root = UsdSkel.Root(existing_prim)
        if skel_root.GetPrim().IsValid():
            return skel_root
    skel_root = UsdSkel.Root.Define(stage, prim_path)
    if not skel_root.GetPrim().IsValid():
        raise ValueError(f"Failed to create UsdSkelRoot at path {prim_path}")
    return skel_root


def list_all_skel_roots(stage: Usd.Stage) -> List[UsdSkel.Root]:
    """
    Returns a list of all the UsdSkelRoot prims in the given stage.

    Parameters:
        stage (Usd.Stage): The USD stage to search for skel roots.

    Returns:
        List[UsdSkel.Root]: A list of all the UsdSkelRoot prims in the stage.
    """
    root_prim = stage.GetPseudoRoot()
    skel_roots = []
    for prim in Usd.PrimRange(root_prim):
        skel_root = UsdSkel.Root(prim)
        if skel_root:
            skel_roots.append(skel_root)
    return skel_roots


def create_test_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdSkel.Root.Define(stage, "/SkelRoot1")
    UsdSkel.Root.Define(stage, "/SkelRoot2")
    stage.DefinePrim("/NonSkelRoot")
    UsdSkel.Root.Define(stage, "/Parent/ChildSkelRoot")
    return stage


def create_and_assign_skel_root(stage: Usd.Stage, skel_root_path: str) -> UsdSkel.Root:
    """Create a SkelRoot prim and assign it to the stage.

    Args:
        stage (Usd.Stage): The stage to create the SkelRoot prim on.
        skel_root_path (str): The path where the SkelRoot prim should be created.

    Returns:
        UsdSkel.Root: The created SkelRoot prim.

    Raises:
        ValueError: If the SkelRoot prim cannot be created or assigned.
    """
    if not stage:
        raise ValueError("Invalid stage")
    skel_root = UsdSkel.Root.Define(stage, skel_root_path)
    if not skel_root:
        raise ValueError(f"Failed to create SkelRoot prim at path: {skel_root_path}")
    return skel_root


def move_prim_to_skel_root(prim: Usd.Prim, skel_root: UsdSkel.Root) -> bool:
    """Move a prim to be a child of a UsdSkelRoot prim.

    Args:
        prim (Usd.Prim): The prim to move.
        skel_root (UsdSkel.Root): The skel root prim to move the prim under.

    Returns:
        bool: True if the move was successful, False otherwise.
    """
    if not prim.IsValid() or not skel_root.GetPrim().IsValid():
        return False
    if prim.IsInPrototype() or prim.GetPath().HasPrefix(skel_root.GetPrim().GetPath()):
        return True
    stage = prim.GetStage()
    new_path = skel_root.GetPrim().GetPath().AppendChild(prim.GetName())
    if stage.GetPrimAtPath(new_path).IsValid():
        return False
    try:
        stage.DefinePrim(new_path)
        prim.SetActive(False)
        return True
    except Tf.ErrorException:
        return False


def test_move_prim_to_skel_root():
    stage = Usd.Stage.CreateInMemory()
    skel_root = UsdSkel.Root.Define(stage, "/SkelRoot")
    prim_to_move = stage.DefinePrim("/PrimToMove")
    result = move_prim_to_skel_root(prim_to_move, skel_root)
    print(f"Move result: {result}")
    moved_prim = stage.GetPrimAtPath("/SkelRoot/PrimToMove")
    print(f"Moved prim is valid: {moved_prim.IsValid()}")
    result = move_prim_to_skel_root(moved_prim, skel_root)
    print(f"Move result for prim already under skel root: {result}")


def create_skeleton_with_bind_and_rest_transforms(
    stage: Usd.Stage,
    skel_path: str,
    joint_names: List[str],
    bind_transforms: List[Gf.Matrix4d],
    rest_transforms: List[Gf.Matrix4d],
) -> UsdSkel.Skeleton:
    """
    Create a new UsdSkelSkeleton with given joint names, bind transforms, and rest transforms.

    Args:
        stage (Usd.Stage): The stage to create the Skeleton on.
        skel_path (str): The path where the Skeleton will be created.
        joint_names (List[str]): List of joint names. Must be the same length as bind_transforms and rest_transforms.
        bind_transforms (List[Gf.Matrix4d]): List of bind transforms, one per joint, in world space.
        rest_transforms (List[Gf.Matrix4d]): List of rest transforms, one per joint, in local space.

    Returns:
        UsdSkel.Skeleton: The created Skeleton prim.

    Raises:
        ValueError: If joint_names, bind_transforms, and rest_transforms are not the same length.
    """
    if len(joint_names) != len(bind_transforms) or len(joint_names) != len(rest_transforms):
        raise ValueError("joint_names, bind_transforms, and rest_transforms must be the same length.")
    skel = UsdSkel.Skeleton.Define(stage, skel_path)
    joint_names_attr = skel.CreateJointNamesAttr()
    joint_names_attr.Set(joint_names)
    joint_paths = [f"{skel_path}/joints/{name}" for name in joint_names]
    joints_attr = skel.CreateJointsAttr()
    joints_attr.Set(joint_paths)
    bind_transforms_attr = skel.CreateBindTransformsAttr()
    bind_transforms_attr.Set(bind_transforms)
    rest_transforms_attr = skel.CreateRestTransformsAttr()
    rest_transforms_attr.Set(rest_transforms)
    return skel


def create_test_skeleton():
    stage = Usd.Stage.CreateInMemory()
    joint_names = ["root", "child"]
    bind_transforms = [
        Gf.Matrix4d(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        Gf.Matrix4d(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
    ]
    rest_transforms = [
        Gf.Matrix4d(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        Gf.Matrix4d(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0),
    ]
    skel_path = "/Skeleton"
    skel = create_skeleton_with_bind_and_rest_transforms(
        stage, skel_path, joint_names, bind_transforms, rest_transforms
    )
    return stage


def update_skeleton_joints(skeleton: UsdSkel.Skeleton, joint_paths: List[str]) -> None:
    """Update the joints of a UsdSkel.Skeleton.

    Args:
        skeleton (UsdSkel.Skeleton): The skeleton to update.
        joint_paths (List[str]): A list of joint paths to set on the skeleton.

    Raises:
        ValueError: If the input skeleton is invalid.
    """
    if not skeleton.GetPrim().IsValid():
        raise ValueError("Invalid skeleton prim.")
    joint_names = skeleton.GetJointNamesAttr().Get() or []
    rest_xforms = skeleton.GetRestTransformsAttr().Get() or []
    if joint_names == joint_paths:
        return
    skeleton.GetJointNamesAttr().Set(joint_paths)
    if rest_xforms:
        joint_to_xform = dict(zip(joint_names, rest_xforms))
        new_rest_xforms = [joint_to_xform.get(joint, Gf.Matrix4d(1)) for joint in joint_paths]
        skeleton.GetRestTransformsAttr().Set(new_rest_xforms)


def set_skeleton_joint_world_transforms(skeleton: UsdSkel.Skeleton, joint_world_transforms: List[Gf.Matrix4d]) -> None:
    """Set the world space joint transforms on a UsdSkelSkeleton prim.

    Args:
        skeleton (UsdSkel.Skeleton): The skeleton prim to set joint transforms on.
        joint_world_transforms (List[Gf.Matrix4d]): List of world space joint transforms.

    Raises:
        ValueError: If the number of joint transforms does not match the number of joints in the skeleton.
    """
    joints_attr = skeleton.GetJointsAttr()
    if not joints_attr.IsValid():
        raise ValueError("Skeleton prim is missing the 'joints' attribute.")
    num_skeleton_joints = len(joints_attr.Get())
    if len(joint_world_transforms) != num_skeleton_joints:
        raise ValueError(
            f"Number of joint transforms ({len(joint_world_transforms)}) does not match the number of joints in the skeleton ({num_skeleton_joints})."
        )
    bind_transforms_attr = skeleton.GetBindTransformsAttr()
    if not bind_transforms_attr.IsValid():
        bind_transforms_attr = skeleton.CreateBindTransformsAttr(joint_world_transforms, writeSparsely=False)
    else:
        bind_transforms_attr.Set(joint_world_transforms)


def create_skeleton(stage, path):
    skel = UsdSkel.Skeleton.Define(stage, path)
    skel.CreateJointsAttr(["joint1", "joint2", "joint3"])
    return skel


def reparent_skeleton_joint(skeleton_prim: UsdSkel.Skeleton, joint_path: str, new_parent_path: str) -> bool:
    """Reparent a joint in a skeleton.

    Args:
        skeleton_prim (UsdSkel.Skeleton): The skeleton prim.
        joint_path (str): The path of the joint to reparent.
        new_parent_path (str): The path of the new parent joint.

    Returns:
        bool: True if the joint was successfully reparented, False otherwise.
    """
    joints_attr = skeleton_prim.GetJointsAttr()
    if not joints_attr.IsValid():
        return False
    joint_names = joints_attr.Get()
    if joint_names is None:
        return False
    joint_names = list(joint_names)
    if joint_path not in joint_names or new_parent_path not in joint_names:
        return False
    joint_index = joint_names.index(joint_path)
    new_parent_index = joint_names.index(new_parent_path)
    new_joint_path = f"{new_parent_path}/{joint_path.split('/')[-1]}"
    joint_names[joint_index] = new_joint_path
    joints_attr.Set(joint_names)
    return True


def get_skeleton_joint_world_transforms(skeleton: UsdSkel.Skeleton) -> List[Gf.Matrix4d]:
    """
    Get the world space joint transforms for a skeleton.

    Args:
        skeleton (UsdSkel.Skeleton): The skeleton to get joint transforms for.

    Returns:
        List[Gf.Matrix4d]: The world space joint transforms.
    """
    local_xforms = skeleton.GetRestTransformsAttr().Get()
    if not local_xforms:
        local_xforms = [Gf.Matrix4d(1)] * len(skeleton.GetJointsAttr().Get())
    bind_xforms = skeleton.GetBindTransformsAttr().Get()
    if not bind_xforms:
        bind_xforms = [Gf.Matrix4d(1)] * len(skeleton.GetJointsAttr().Get())
    xforms = []
    for bind, rest in zip(bind_xforms, local_xforms):
        xform = bind * rest
        xforms.append(xform)
    return xforms


def create_skeletal_animation(
    stage: Usd.Stage,
    skeleton_path: str,
    joint_paths: list[str],
    translations: list[Vt.Vec3fArray],
    rotations: list[list[Gf.Quatf]],
) -> UsdSkel.Animation:
    """Create a skeletal animation for a given skeleton and joint paths with translations and rotations.

    Args:
        stage (Usd.Stage): The USD stage to create the animation on.
        skeleton_path (str): The path to the skeleton prim.
        joint_paths (list[str]): The paths to the joints to animate.
        translations (list[Vt.Vec3fArray]): The translations for each joint at each frame.
        rotations (list[list[Gf.Quatf]]): The rotations for each joint at each frame.

    Returns:
        UsdSkel.Animation: The created skeletal animation prim.

    Raises:
        ValueError: If the number of joint paths, translations, and rotations do not match.
    """
    if len(joint_paths) != len(translations) or len(joint_paths) != len(rotations):
        raise ValueError("Number of joint paths, translations, and rotations must match.")
    num_frames = len(translations[0])
    anim_path = skeleton_path + "/Anim"
    anim = UsdSkel.Animation.Define(stage, anim_path)
    anim.CreateJointsAttr().Set(joint_paths)
    translations_attr = anim.CreateTranslationsAttr()
    for i in range(num_frames):
        frame_translations = [Gf.Vec3f(*t[i]) for t in translations]
        translations_attr.Set(frame_translations, Usd.TimeCode(i))
    rotations_attr = anim.CreateRotationsAttr()
    for i in range(num_frames):
        frame_rotations = [r[i] for r in rotations]
        rotations_attr.Set(frame_rotations, Usd.TimeCode(i))
    return anim


def blend_skeleton_animations(
    skeleton: UsdSkel.Skeleton, anim_paths: List[str], anim_weights: List[float]
) -> Vt.Matrix4dArray:
    """
    Blend multiple skeleton animations together based on weights.

    Args:
        skeleton (UsdSkel.Skeleton): The skeleton to blend animations for.
        anim_paths (List[str]): List of paths to SkelAnimation prims.
        anim_weights (List[float]): List of weights for each animation. Should be same length as anim_paths.

    Returns:
        Vt.Matrix4dArray: Blended joint transforms in skeleton's joint order.
    """
    if len(anim_paths) != len(anim_weights):
        raise ValueError("Number of animation paths must match number of weights.")
    joint_paths = skeleton.GetJointsAttr().Get()
    blended_xforms = [Gf.Matrix4d(1.0) for _ in range(len(joint_paths))]
    for anim_path, weight in zip(anim_paths, anim_weights):
        anim_prim = skeleton.GetPrim().GetStage().GetPrimAtPath(anim_path)
        if not anim_prim.IsValid():
            raise ValueError(f"Invalid animation prim path: {anim_path}")
        anim = UsdSkel.Animation(anim_prim)
        anim_xforms = anim.GetTranslationsAttr().Get()
        if len(anim_xforms) != len(joint_paths):
            raise ValueError(f"Animation {anim_path} has {len(anim_xforms)} transforms, expected {len(joint_paths)}")
        for i in range(len(joint_paths)):
            blended_xforms[i].SetTranslateOnly(
                blended_xforms[i].ExtractTranslation() + Gf.Vec3d(anim_xforms[i]) * weight
            )
    return Vt.Matrix4dArray(blended_xforms)


def create_skeleton(stage, path):
    """Create a simple skeleton for testing."""
    skel = UsdSkel.Skeleton.Define(stage, path)
    skel.CreateJointsAttr().Set(["Hip", "Shoulder", "Elbow", "Wrist"])
    return skel


def create_animation(stage, path):
    """Create a simple animation for testing."""
    anim = UsdSkel.Animation.Define(stage, path)
    anim.CreateTranslationsAttr().Set(
        [Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 2.0, 0.0), Gf.Vec3f(0.0, 0.0, 3.0), Gf.Vec3f(4.0, 0.0, 0.0)]
    )
    return anim


def get_joint_bind_transforms(
    skelQuery: UsdSkel.SkeletonQuery, world_space: bool = True
) -> Tuple[bool, List[Gf.Matrix4d]]:
    """
    Get the joint bind transforms from a SkeletonQuery.

    Args:
        skelQuery (UsdSkel.SkeletonQuery): The SkeletonQuery to get the bind transforms from.
        world_space (bool): If True, return the bind transforms in world space, otherwise return them in local space.

    Returns:
        A tuple of (bool, List[Gf.Matrix4d]):
            - bool: True if the bind transforms were successfully retrieved, False otherwise.
            - List[Gf.Matrix4d]: The joint bind transforms.
    """
    if not skelQuery:
        return (False, [])
    topology = skelQuery.GetTopology()
    if not topology:
        return (False, [])
    num_joints = topology.GetNumJoints()
    if num_joints == 0:
        return (False, [])
    xforms = Vt.Matrix4dArray(num_joints)
    if world_space:
        result = skelQuery.GetJointWorldBindTransforms(xforms)
    else:
        skel = UsdSkel.Skeleton(skelQuery.GetPrim())
        if not skel:
            return (False, [])
        attr = skel.GetBindTransformsAttr()
        if not attr:
            return (False, [])
        result = attr.Get(Vt.Matrix4dArray(num_joints), Usd.TimeCode.Default())
        xforms = attr.Get(Vt.Matrix4dArray(num_joints))
    if not result:
        return (False, [])
    return (True, list(xforms))


def get_skinning_transforms(skeleton_query: UsdSkel.SkeletonQuery, time: Usd.TimeCode) -> Tuple[bool, Vt.Matrix4dArray]:
    """
    Compute transforms representing the change in transformation of a joint from its rest pose, in skeleton space.

    Args:
        skeleton_query (UsdSkel.SkeletonQuery): The skeleton query object.
        time (Usd.TimeCode): The time at which to compute the skinning transforms.

    Returns:
        A tuple containing a boolean indicating the success of the operation and the computed skinning transforms.
    """
    num_joints = len(skeleton_query.GetJointOrder())
    skinning_transforms = Vt.Matrix4dArray(num_joints)
    success = skeleton_query.ComputeSkinningTransforms(skinning_transforms, time)
    return (success, skinning_transforms)


def test_get_skinning_transforms():
    stage = Usd.Stage.Open("path/to/your/usd/file.usd")
    skeleton_root_prim = stage.GetPrimAtPath("/Root")
    if skeleton_root_prim and UsdSkel.Root(skeleton_root_prim):
        cache = UsdSkel.Cache()
        cache.Populate(UsdSkel.Root(skeleton_root_prim))
        skeleton_prim = skeleton_root_prim
        skeleton_query = cache.GetSkelQuery(skeleton_prim)
        (success, skinning_transforms) = get_skinning_transforms(skeleton_query, Usd.TimeCode.Default())
        print(f"Success: {success}")
        print(f"Skinning Transforms: {skinning_transforms}")
    else:
        print("No skeleton root prim found in the stage.")


def compute_joint_influences(indices: Vt.IntArray, weights: Vt.FloatArray, time: Usd.TimeCode) -> bool:
    """
    Convenience method for computing joint influences.

    In addition to querying influences, this will also perform validation
    of the basic form of the weight data, although the array contents are
    not validated.

    Args:
        indices (Vt.IntArray): Joint indices.
        weights (Vt.FloatArray): Joint weights.
        time (Usd.TimeCode): Time at which to compute influences.

    Returns:
        bool: True if joint influences were computed successfully, False otherwise.
    """
    if len(indices) != len(weights):
        print("Error: Indices and weights arrays must have the same size.")
        return False
    if any((index < 0 for index in indices)):
        print("Error: Indices must be non-negative.")
        return False
    if any((weight < 0 for weight in weights)):
        print("Error: Weights must be non-negative.")
        return False
    return True


def compute_skinned_points(
    xforms: Vt.Matrix4dArray, points: Vt.Vec3fArray, joint_indices: Vt.IntArray, joint_weights: Vt.FloatArray
) -> Vt.Vec3fArray:
    """
    Compute skinned points using linear blend skinning (LBS).

    Args:
        xforms (Vt.Matrix4dArray): Joint transforms in skeleton space.
        points (Vt.Vec3fArray): Points to skin in skeleton space.
        joint_indices (Vt.IntArray): Joint indices for each point.
        joint_weights (Vt.FloatArray): Joint weights for each point.

    Returns:
        Vt.Vec3fArray: Skinned points in skeleton space.
    """
    if not xforms or not points or (not joint_indices) or (not joint_weights):
        raise ValueError("Invalid input data. All input arrays must be non-empty.")
    if len(joint_indices) != len(joint_weights):
        raise ValueError("Joint indices and weights must have the same length.")
    num_points = len(points)
    if num_points * 4 != len(joint_indices):
        raise ValueError("Number of points does not match the number of joint indices.")
    skinned_points = Vt.Vec3fArray(num_points)
    for i in range(num_points):
        skinned_point = Gf.Vec3f(0, 0, 0)
        for j in range(4):
            joint_index = joint_indices[i * 4 + j]
            joint_weight = joint_weights[i * 4 + j]
            if joint_index < 0 or joint_index >= len(xforms):
                raise ValueError(f"Invalid joint index {joint_index} for point {i}.")
            joint_transform = Gf.Matrix4d(xforms[joint_index])
            transformed_point = joint_transform.Transform(Gf.Vec3d(points[i]))
            skinned_point += Gf.Vec3f(transformed_point) * joint_weight
        skinned_points[i] = skinned_point
    return skinned_points


def get_skinning_time_samples(
    skinning_query: UsdSkel.SkinningQuery, interval: Optional[Gf.Interval] = None
) -> List[float]:
    """
    Get the union of time samples for all properties that affect skinning.

    Args:
        skinning_query (UsdSkel.SkinningQuery): The skinning query object.
        interval (Optional[Gf.Interval]): The interval within which to retrieve time samples.
                                           If None, retrieves all time samples.

    Returns:
        List[float]: The list of time samples affecting skinning.
    """
    times = []
    if interval is None:
        if not skinning_query.GetTimeSamples(times):
            return []
    elif not skinning_query.GetTimeSamplesInInterval(interval, times):
        return []
    return [float(t) for t in times]


def has_blend_shapes(prim: UsdSkel.BindingAPI) -> bool:
    """Returns true if there are blend shapes associated with this prim."""
    attr = prim.GetBlendShapesAttr()
    if not attr.IsValid():
        return False
    blend_shapes = attr.Get()
    if not blend_shapes:
        return False
    rel = prim.GetBlendShapeTargetsRel()
    if not rel.IsValid() or not rel.GetTargets():
        return False
    return True


def get_geom_bind_transform_attribute(prim: Usd.Prim) -> Usd.Attribute:
    """
    Get the geomBindTransform attribute for a prim.

    Args:
        prim (Usd.Prim): The prim to get the geomBindTransform attribute from.

    Returns:
        Usd.Attribute: The geomBindTransform attribute, or None if not found.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    skel_binding_api = UsdSkel.BindingAPI(prim)
    if skel_binding_api.GetGeomBindTransformAttr().IsDefined():
        return skel_binding_api.GetGeomBindTransformAttr()
    else:
        return None


def get_blend_shape_targets_relationship(prim: Usd.Prim) -> Usd.Relationship:
    """
    Get the relationship that targets the blend shape prims.

    Args:
        prim (Usd.Prim): The prim to query for blend shape targets.

    Returns:
        Usd.Relationship: The relationship that targets the blend shape prims.
                          Returns an invalid relationship if no blend shapes are defined.
    """
    if not prim.IsValid():
        return Usd.Relationship()
    binding_api = UsdSkel.BindingAPI(prim)
    blend_shape_targets_rel = binding_api.GetBlendShapeTargetsRel()
    if not blend_shape_targets_rel.IsValid():
        return Usd.Relationship()
    return blend_shape_targets_rel


def bake_skeleton_animation(skeleton_root: UsdSkel.Root, start_time: Usd.TimeCode, end_time: Usd.TimeCode) -> bool:
    """Bake the animation of a skeleton within a given time range.

    Args:
        skeleton_root (UsdSkel.Root): The root prim of the skeleton.
        start_time (Usd.TimeCode): The start time of the range to bake the animation for.
        end_time (Usd.TimeCode): The end time of the range to bake the animation for.

    Returns:
        bool: True if the baking was successful, False otherwise.
    """
    if not skeleton_root.GetPrim().IsValid():
        return False
    animation_source = UsdSkel.AnimQuery(skeleton_root.GetPrim()).GetAnimationSource()
    if not animation_source:
        return False
    skeleton_query = UsdSkel.SkeletonQuery(skeleton_root.GetPrim())
    if not skeleton_query:
        return False
    try:
        UsdSkel.BakeAnimation(skeleton_query, animation_source, start_time, end_time)
        return True
    except Exception as e:
        print(f"Error baking animation: {str(e)}")
        return False


def find_joint_descendants(topology: UsdSkel.Topology, joint_index: int) -> List[int]:
    """
    Find all descendant joints of a given joint in a skeleton topology.

    Args:
        topology (UsdSkel.Topology): The skeleton topology.
        joint_index (int): The index of the joint to find descendants for.

    Returns:
        List[int]: A list of indices of all descendant joints.
    """
    if joint_index < 0 or joint_index >= topology.GetNumJoints():
        raise ValueError(f"Invalid joint index: {joint_index}")
    descendants = []
    for i in range(topology.GetNumJoints()):
        if i == joint_index:
            continue
        parent = i
        while parent != -1 and parent != joint_index:
            parent = topology.GetParent(parent)
        if parent == joint_index:
            descendants.append(i)
    return descendants


def get_root_joints(topology: UsdSkel.Topology) -> List[int]:
    """
    Returns a list of root joint indices in the given topology.

    A joint is considered a root if it has no parent joint.
    """
    num_joints = topology.GetNumJoints()
    root_joints = []
    for i in range(num_joints):
        if topology.IsRoot(i):
            root_joints.append(i)
    return root_joints


def find_joint_ancestors(topology: UsdSkel.Topology, joint_index: int) -> List[int]:
    """
    Find the indices of all ancestors for a given joint in the topology.

    Args:
        topology (UsdSkel.Topology): The skeleton topology.
        joint_index (int): The index of the joint to find ancestors for.

    Returns:
        List[int]: A list of indices representing the ancestors of the joint.
    """
    if joint_index < 0 or joint_index >= topology.GetNumJoints():
        raise ValueError(f"Invalid joint index: {joint_index}")
    ancestors = []
    parent_index = topology.GetParent(joint_index)
    while parent_index != -1:
        ancestors.append(parent_index)
        parent_index = topology.GetParent(parent_index)
    return ancestors
