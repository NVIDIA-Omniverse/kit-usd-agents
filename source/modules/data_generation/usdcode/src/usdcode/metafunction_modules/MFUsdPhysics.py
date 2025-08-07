## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

from .add_op import *


def apply_articulation_root(prim: Usd.Prim) -> UsdPhysics.ArticulationRootAPI:
    """Apply ArticulationRootAPI to a prim."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if not UsdPhysics.ArticulationRootAPI.CanApply(prim):
        raise ValueError(f"ArticulationRootAPI cannot be applied to prim: {prim}")
    articulation_root_api = UsdPhysics.ArticulationRootAPI.Apply(prim)
    if not articulation_root_api.GetPrim() == prim:
        raise ValueError(f"Failed to apply ArticulationRootAPI to prim: {prim}")
    return articulation_root_api


def get_articulation_root_prims(stage: Usd.Stage) -> List[Usd.Prim]:
    """
    Returns a list of all prims on the given stage that have the
    UsdPhysicsArticulationRootAPI applied.
    """
    prims = stage.Traverse()
    articulation_root_prims = []
    for prim in prims:
        if UsdPhysics.ArticulationRootAPI.Get(prim.GetStage(), prim.GetPath()):
            articulation_root_prims.append(prim)
    return articulation_root_prims


def validate_articulation_hierarchy(prim: Usd.Prim) -> bool:
    """Validate that the given prim is a valid articulation hierarchy root."""
    if not prim.IsValid():
        raise ValueError("Invalid prim")
    if not UsdPhysics.ArticulationRootAPI.Get(prim.GetStage(), prim.GetPath()):
        return False
    for child in prim.GetAllChildren():
        if not UsdPhysics.Joint.Get(child.GetStage(), child.GetPath()) and (not UsdPhysics.RigidBodyAPI.Get(child)):
            return False
        if not validate_articulation_hierarchy(child):
            return False
    return True


def toggle_articulation_root(prim: Usd.Prim) -> bool:
    """Toggle the ArticulationRootAPI on a prim.

    If the API is not applied, it will be applied. If it is already applied, it will be removed.

    Args:
        prim (Usd.Prim): The prim to toggle the ArticulationRootAPI on.

    Returns:
        bool: True if the API was applied, False if it was removed.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if not UsdPhysics.ArticulationRootAPI.CanApply(prim):
        raise ValueError(f"ArticulationRootAPI cannot be applied to prim: {prim}")
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        return False
    else:
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        return True


def batch_apply_collision(prims: List[Union[Usd.Prim, UsdGeom.Xformable]], enabled: bool = True) -> None:
    """
    Apply UsdPhysics.CollisionAPI to a batch of prims.

    Args:
        prims (List[Union[Usd.Prim, UsdGeom.Xformable]]): A list of prims or xformables to apply collision to.
        enabled (bool, optional): Whether collision should be enabled. Defaults to True.
    """
    for prim in prims:
        if isinstance(prim, UsdGeom.Xformable):
            prim = prim.GetPrim()
        if not prim.IsValid():
            continue
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        if collision_api.GetCollisionEnabledAttr().IsAuthored():
            collision_api.GetCollisionEnabledAttr().Set(enabled)
        else:
            collision_api.CreateCollisionEnabledAttr(enabled, writeSparsely=True)


def toggle_collision_enabled(prim: Usd.Prim) -> bool:
    """Toggle the collision enabled state of a prim with CollisionAPI applied.

    Args:
        prim (Usd.Prim): The prim to toggle collision enabled state on.

    Returns:
        bool: The new collision enabled state.

    Raises:
        ValueError: If the input prim is not valid or does not have CollisionAPI applied.
    """
    if not prim.IsValid():
        raise ValueError("Input prim is not valid")
    if not UsdPhysics.CollisionAPI.CanApply(prim):
        raise ValueError("CollisionAPI cannot be applied to the input prim")
    collision_api = UsdPhysics.CollisionAPI.Get(prim.GetStage(), prim.GetPath())
    collision_enabled_attr = collision_api.GetCollisionEnabledAttr()
    if collision_enabled_attr.IsValid():
        current_value = collision_enabled_attr.Get()
    else:
        current_value = True
    new_value = not current_value
    if collision_enabled_attr.IsValid():
        collision_enabled_attr.Set(new_value)
    else:
        collision_api.CreateCollisionEnabledAttr(new_value, True)
    return new_value


def assign_simulation_owner(prim: Usd.Prim, physics_scene_path: str) -> None:
    """Assign a prim to a specific physics scene for simulation."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    collision_api = UsdPhysics.CollisionAPI.Get(prim.GetStage(), prim.GetPath())
    if not collision_api:
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    sim_owner_rel = collision_api.GetSimulationOwnerRel()
    if not sim_owner_rel:
        sim_owner_rel = collision_api.CreateSimulationOwnerRel()
    sim_owner_rel.SetTargets([Sdf.Path(physics_scene_path)])


def validate_collision_setup(stage: Usd.Stage, prim_path: str) -> bool:
    """Validate that a prim has a proper collision setup."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path: {prim_path}")
    collision_api = UsdPhysics.CollisionAPI(prim)
    if not collision_api:
        return False
    collision_enabled_attr = collision_api.GetCollisionEnabledAttr()
    if not collision_enabled_attr or not collision_enabled_attr.Get():
        return False
    simulation_owner_rel = collision_api.GetSimulationOwnerRel()
    if simulation_owner_rel:
        simulation_owner_targets = simulation_owner_rel.GetTargets()
        if not simulation_owner_targets:
            return False
        simulation_owner_prim = stage.GetPrimAtPath(simulation_owner_targets[0])
        if not simulation_owner_prim.IsValid():
            return False
    return True


def copy_collision_attributes(source_prim: Usd.Prim, target_prim: Usd.Prim):
    """Copy collision attributes from source_prim to target_prim."""
    if not source_prim.IsValid():
        raise ValueError("Source prim is not valid.")
    if not target_prim.IsValid():
        raise ValueError("Target prim is not valid.")
    if not UsdPhysics.MeshCollisionAPI.CanApply(source_prim):
        raise ValueError("Source prim does not have MeshCollisionAPI applied.")
    if not target_prim.IsA(UsdGeom.Mesh):
        raise ValueError("Target prim is not a UsdGeomMesh.")
    source_collision_api = UsdPhysics.MeshCollisionAPI(source_prim)
    target_collision_api = UsdPhysics.MeshCollisionAPI.Apply(target_prim)
    if source_collision_api.GetApproximationAttr().HasValue():
        approximation = source_collision_api.GetApproximationAttr().Get()
        target_collision_api.CreateApproximationAttr(approximation, True)


def get_all_colliders(stage: Usd.Stage) -> List[UsdPhysics.CollisionAPI]:
    """
    Returns a list of all prims with CollisionAPI applied in the given stage.

    Parameters:
        stage (Usd.Stage): The stage to search for colliders.

    Returns:
        List[UsdPhysics.CollisionAPI]: A list of CollisionAPI objects for all colliders in the stage.
    """
    colliders = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collider = UsdPhysics.CollisionAPI(prim)
            colliders.append(collider)
    return colliders


def remove_collision_from_hierarchy(prim: Usd.Prim) -> None:
    """Remove collision from a prim and its descendants."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    for descendant in Usd.PrimRange(prim):
        collision_api = UsdPhysics.CollisionAPI(descendant)
        if collision_api.GetPrim().IsValid():
            descendant.RemoveAPI(UsdPhysics.CollisionAPI)


def export_collision_group_table(stage: Usd.Stage) -> Dict[str, Set[str]]:
    """Export the collision group table from a USD stage.

    The collision group table is a dictionary where each key is a collision
    group name and the corresponding value is a set of collision group names
    that are filtered for that group.

    Args:
        stage (Usd.Stage): The USD stage to export the collision group table from.

    Returns:
        Dict[str, Set[str]]: The collision group table dictionary.
    """
    collision_groups = [
        UsdPhysics.CollisionGroup.Get(stage, prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetTypeName() == "PhysicsCollisionGroup"
    ]
    collision_group_table = {}
    for collision_group in collision_groups:
        group_name = collision_group.GetPrim().GetName()
        filtered_groups_rel = collision_group.GetFilteredGroupsRel()
        filtered_group_targets = filtered_groups_rel.GetTargets()
        filtered_group_names = {Sdf.Path(targetPath).name for targetPath in filtered_group_targets}
        collision_group_table[group_name] = filtered_group_names
    return collision_group_table


def add_filtered_group(collision_group: UsdPhysics.CollisionGroup, filtered_group: UsdPhysics.CollisionGroup):
    """Add a collision group to the list of filtered groups for this collision group."""
    filtered_groups_rel = collision_group.GetFilteredGroupsRel()
    if not filtered_groups_rel:
        filtered_groups_rel = collision_group.CreateFilteredGroupsRel()
    targets = filtered_groups_rel.GetTargets()
    if filtered_group.GetPath() not in targets:
        targets.append(filtered_group.GetPath())
        filtered_groups_rel.SetTargets(targets)


def remove_filtered_group(
    collision_group: UsdPhysics.CollisionGroup, filtered_group: UsdPhysics.CollisionGroup
) -> None:
    """Remove a collision group from the list of filtered groups of this collision group."""
    filtered_groups_rel = collision_group.GetFilteredGroupsRel()
    if filtered_groups_rel.IsValid():
        targets = filtered_groups_rel.GetTargets()
        try:
            index = targets.index(filtered_group.GetPath())
        except ValueError:
            return
        targets.pop(index)
        filtered_groups_rel.ClearTargets(True)
        filtered_groups_rel.SetTargets(targets)
    else:
        pass


def set_collision_group_invert_filter(collision_group: UsdPhysics.CollisionGroup, invert: bool):
    """Set the invert filter attribute on a collision group.

    Args:
        collision_group (UsdPhysics.CollisionGroup): The collision group to modify.
        invert (bool): The value to set for the invert filter attribute.
    """
    invert_attr = collision_group.GetInvertFilteredGroupsAttr()
    if not invert_attr:
        invert_attr = collision_group.CreateInvertFilteredGroupsAttr(False, False)
    invert_attr.Set(invert)


def merge_collision_groups(stage: Usd.Stage) -> None:
    """Merge collision groups with matching 'physics:mergeGroup' attribute values."""
    collision_groups = stage.Traverse(
        Usd.TraverseInstanceProxies(Usd.PrimIsActive & Usd.PrimIsDefined & ~Usd.PrimIsAbstract)
    )
    collision_groups = [
        UsdPhysics.CollisionGroup(prim) for prim in collision_groups if prim.IsA(UsdPhysics.CollisionGroup)
    ]
    merge_groups: Dict[str, List[UsdPhysics.CollisionGroup]] = {}
    for collision_group in collision_groups:
        merge_group_attr = collision_group.GetMergeGroupNameAttr()
        if merge_group_attr.IsValid():
            merge_group_name = merge_group_attr.Get()
            if merge_group_name:
                merge_groups.setdefault(merge_group_name, []).append(collision_group)
    for merge_group_name, collision_group_list in merge_groups.items():
        if len(collision_group_list) > 1:
            merged_collision_group = collision_group_list[0]
            for collision_group in collision_group_list[1:]:
                colliders_rel = merged_collision_group.GetCollidersCollectionAPI().CreateIncludesRel()
                for collider_path in collision_group.GetCollidersCollectionAPI().GetIncludesRel().GetTargets():
                    colliders_rel.AddTarget(collider_path)
                filtered_groups_rel = collision_group.GetFilteredGroupsRel()
                merged_filtered_groups_rel = merged_collision_group.GetFilteredGroupsRel()
                for filtered_group_path in filtered_groups_rel.GetTargets():
                    merged_filtered_groups_rel.AddTarget(filtered_group_path)
                stage.RemovePrim(collision_group.GetPath())


def get_collision_group_colliders(collision_group: UsdPhysics.CollisionGroup) -> List[UsdPhysics.CollisionAPI]:
    """Get the list of colliders belonging to a collision group."""
    colliders_collection = collision_group.GetCollidersCollectionAPI()
    if not colliders_collection:
        return []
    target_paths = colliders_collection.GetIncludesRel().GetTargets()
    colliders = [
        UsdPhysics.CollisionAPI(collision_group.GetPrim().GetStage().GetPrimAtPath(path)) for path in target_paths
    ]
    valid_colliders = [collider for collider in colliders if collider]
    return valid_colliders


def get_all_collision_groups(stage: Usd.Stage) -> List[UsdPhysics.CollisionGroup]:
    """
    Get all the CollisionGroup prims in the stage.

    Args:
        stage (Usd.Stage): The USD stage to search for CollisionGroups.

    Returns:
        List[UsdPhysics.CollisionGroup]: A list of all CollisionGroup prims in the stage.
    """
    root_prim = stage.GetPseudoRoot()
    collision_group_prims = root_prim.GetFilteredChildren(
        Usd.TraverseInstanceProxies(Usd.PrimIsActive & Usd.PrimDefaultPredicate)
    )
    collision_groups = [
        UsdPhysics.CollisionGroup(prim) for prim in collision_group_prims if prim.IsA(UsdPhysics.CollisionGroup)
    ]
    return collision_groups


def validate_collision_group_filters(stage: Usd.Stage) -> bool:
    """Validate that collision group filters are set up correctly on the given stage.

    This function checks that for each collision group, the filtered groups are mutually
    exclusive, i.e., a group cannot filter another group that also filters it.

    Args:
        stage (Usd.Stage): The stage to validate collision group filters for.

    Returns:
        bool: True if collision group filters are valid, False otherwise.
    """
    collision_group_prims = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.CollisionGroup):
            collision_group_prims.append(prim)
    filtered_groups_dict = {}
    for collision_group_prim in collision_group_prims:
        collision_group = UsdPhysics.CollisionGroup(collision_group_prim)
        filtered_groups_rel = collision_group.GetFilteredGroupsRel()
        filtered_groups = filtered_groups_rel.GetTargets()
        filtered_groups_dict[collision_group.GetPath()] = [group for group in filtered_groups]
    for collision_group_path, filtered_groups in filtered_groups_dict.items():
        for filtered_group_path in filtered_groups:
            if collision_group_path in filtered_groups_dict.get(filtered_group_path, []):
                return False
    return True


def disable_collision_between_groups(group1: str, group2: str) -> None:
    """Disable collision between two collision groups.

    Args:
        group1 (str): The name of the first collision group.
        group2 (str): The name of the second collision group.

    Raises:
        ValueError: If either group1 or group2 is not a valid collision group.
    """
    groups = UsdPhysics.CollisionGroupTable.GetGroups()
    if group1 not in groups or group2 not in groups:
        raise ValueError(f"Invalid collision group(s): {group1}, {group2}")
    UsdPhysics.CollisionGroupTable.SetCollisionEnabled(group1, group2, False)


def create_distance_joint_with_limits(
    stage: Usd.Stage, path: str, body0: str, body1: str, min_distance: float, max_distance: float
) -> UsdPhysics.DistanceJoint:
    """Create a distance joint with minimum and maximum distance limits between two bodies.

    Args:
        stage (Usd.Stage): The USD stage to create the joint on.
        path (str): The path where the joint should be created.
        body0 (str): The path to the first body to be connected by the joint.
        body1 (str): The path to the second body to be connected by the joint.
        min_distance (float): The minimum distance limit for the joint. If negative, no minimum limit is set.
        max_distance (float): The maximum distance limit for the joint. If negative, no maximum limit is set.

    Returns:
        UsdPhysics.DistanceJoint: The created distance joint.

    Raises:
        ValueError: If either body0 or body1 is not a valid prim path.
    """
    if not stage.GetPrimAtPath(body0).IsValid():
        raise ValueError(f"Invalid prim path for body0: {body0}")
    if not stage.GetPrimAtPath(body1).IsValid():
        raise ValueError(f"Invalid prim path for body1: {body1}")
    joint = UsdPhysics.DistanceJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    if min_distance >= 0:
        joint.CreateMinDistanceAttr(min_distance)
    if max_distance >= 0:
        joint.CreateMaxDistanceAttr(max_distance)
    return joint


def animate_distance_joint(
    stage: Usd.Stage,
    joint_path: str,
    min_distances: List[float],
    max_distances: List[float],
    time_codes: List[Usd.TimeCode],
):
    """Animate the min and max distance attributes of a distance joint over time.

    Args:
        stage (Usd.Stage): The USD stage.
        joint_path (str): The path to the distance joint prim.
        min_distances (List[float]): List of minimum distances values to set at each time code.
        max_distances (List[float]): List of maximum distances values to set at each time code.
        time_codes (List[Usd.TimeCode]): List of time codes to set the values at.

    Raises:
        ValueError: If the joint_path is invalid or not a UsdPhysicsDistanceJoint.
        ValueError: If the lengths of min_distances, max_distances, and time_codes don't match.
    """
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid():
        raise ValueError(f"Invalid prim path: {joint_path}")
    distance_joint = UsdPhysics.DistanceJoint(joint_prim)
    if not distance_joint:
        raise ValueError(f"Prim at path {joint_path} is not a UsdPhysicsDistanceJoint")
    if len(min_distances) != len(max_distances) or len(min_distances) != len(time_codes):
        raise ValueError("min_distances, max_distances, and time_codes must have the same length")
    min_distance_attr = distance_joint.GetMinDistanceAttr()
    max_distance_attr = distance_joint.GetMaxDistanceAttr()
    for i in range(len(time_codes)):
        min_distance_attr.Set(min_distances[i], time_codes[i])
        max_distance_attr.Set(max_distances[i], time_codes[i])


def toggle_distance_joint_limit(joint: UsdPhysics.DistanceJoint, min_enabled: bool, max_enabled: bool) -> None:
    """Toggle the minimum and maximum distance limits on a distance joint.

    Args:
        joint (UsdPhysics.DistanceJoint): The distance joint to modify.
        min_enabled (bool): Whether to enable the minimum distance limit.
        max_enabled (bool): Whether to enable the maximum distance limit.
    """
    min_attr = joint.GetMinDistanceAttr()
    max_attr = joint.GetMaxDistanceAttr()
    if min_enabled:
        if min_attr.HasAuthoredValue():
            min_attr.Set(abs(min_attr.Get()))
        else:
            min_attr.Set(1.0)
    else:
        min_attr.Set(-1.0)
    if max_enabled:
        if max_attr.HasAuthoredValue():
            max_attr.Set(abs(max_attr.Get()))
        else:
            max_attr.Set(2.0)
    else:
        max_attr.Set(-1.0)


def batch_update_distance_joint_limits(
    stage: Usd.Stage, joint_paths: List[str], min_max_values: List[Tuple[float, float]]
):
    """
    Update the minimum and maximum distance limits for a batch of distance joints.

    Args:
        stage (Usd.Stage): The USD stage.
        joint_paths (List[str]): A list of paths to distance joint prims.
        min_max_values (List[Tuple[float, float]]): A list of tuples containing the minimum and maximum distance values for each joint.

    Raises:
        ValueError: If the lengths of joint_paths and min_max_values do not match.
    """
    if len(joint_paths) != len(min_max_values):
        raise ValueError("The lengths of joint_paths and min_max_values must match.")
    for joint_path, (min_distance, max_distance) in zip(joint_paths, min_max_values):
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid() or not joint_prim.IsA(UsdPhysics.DistanceJoint):
            continue
        distance_joint = UsdPhysics.DistanceJoint(joint_prim)
        min_attr = distance_joint.GetMinDistanceAttr()
        if min_attr.IsValid():
            min_attr.Set(min_distance)
        max_attr = distance_joint.GetMaxDistanceAttr()
        if max_attr.IsValid():
            max_attr.Set(max_distance)


def set_distance_joint_limits(joint: UsdPhysics.DistanceJoint, min_distance: float, max_distance: float):
    """Set the minimum and maximum distance limits for a distance joint.

    If a limit is negative, it means the joint is not limited in that direction.

    Args:
        joint (UsdPhysics.DistanceJoint): The distance joint to set limits on.
        min_distance (float): The minimum distance limit. Use negative value for no limit.
        max_distance (float): The maximum distance limit. Use negative value for no limit.

    Raises:
        ValueError: If the provided prim is not a valid UsdPhysics.DistanceJoint.
        ValueError: If min_distance is greater than max_distance.
    """
    if not joint:
        raise ValueError("Invalid UsdPhysics.DistanceJoint provided.")
    if min_distance > max_distance:
        raise ValueError("min_distance cannot be greater than max_distance.")
    min_attr = joint.CreateMinDistanceAttr(min_distance, writeSparsely=True)
    max_attr = joint.CreateMaxDistanceAttr(max_distance, writeSparsely=True)
    if not min_attr.IsValid() or not max_attr.IsValid():
        raise ValueError("Failed to create distance limit attributes.")


def get_distance_joint_limits(joint: UsdPhysics.DistanceJoint) -> Tuple[float, float]:
    """Get the minimum and maximum distance limits for a distance joint.

    Args:
        joint (UsdPhysics.DistanceJoint): The distance joint to get limits for.

    Returns:
        Tuple[float, float]: A tuple with (min_distance, max_distance).
            If a limit is not set, its value will be negative.
    """
    min_attr = joint.GetMinDistanceAttr()
    max_attr = joint.GetMaxDistanceAttr()
    min_distance = min_attr.Get() if min_attr.HasAuthoredValue() else -1.0
    max_distance = max_attr.Get() if max_attr.HasAuthoredValue() else -1.0
    return (min_distance, max_distance)


def duplicate_distance_joint(stage: Usd.Stage, joint_path: str, dup_path: str) -> UsdPhysics.DistanceJoint:
    """Duplicate a distance joint.

    Args:
        stage (Usd.Stage): The stage containing the joint to duplicate.
        joint_path (str): The path of the distance joint to duplicate.
        dup_path (str): The path for the duplicated joint.

    Returns:
        UsdPhysics.DistanceJoint: The duplicated distance joint.

    Raises:
        ValueError: If the original joint is not a valid DistanceJoint prim.
    """
    original_joint = UsdPhysics.DistanceJoint.Get(stage, joint_path)
    if not original_joint:
        raise ValueError(f"Prim at path {joint_path} is not a valid DistanceJoint.")
    dup_joint = UsdPhysics.DistanceJoint.Define(stage, dup_path)
    attrs_to_copy = [
        "physics:minDistance",
        "physics:maxDistance",
        "physics:localPos0",
        "physics:localPos1",
        "physics:body0",
        "physics:body1",
        "physics:breakForce",
        "physics:breakTorque",
        "physics:collisionEnabled",
        "physics:excludeFromArticulation",
    ]
    for attr_name in attrs_to_copy:
        attr = original_joint.GetPrim().GetAttribute(attr_name)
        if attr.HasValue():
            attr_value = attr.Get()
            dup_joint.GetPrim().CreateAttribute(attr_name, attr.GetTypeName()).Set(attr_value)
    return dup_joint


def create_drive_with_attrs(
    stage: Usd.Stage,
    prim_path: str,
    drive_name: str,
    stiffness: float,
    damping: float,
    max_force: float,
    target_position: float,
    target_velocity: float,
    drive_type: str = "force",
) -> UsdPhysics.DriveAPI:
    """Create a drive with specified attributes.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path of the prim to apply the drive to.
        drive_name (str): The name of the drive.
        stiffness (float): The stiffness of the drive.
        damping (float): The damping of the drive.
        max_force (float): The maximum force of the drive.
        target_position (float): The target position of the drive.
        target_velocity (float): The target velocity of the drive.
        drive_type (str, optional): The type of the drive. Defaults to "force".

    Returns:
        UsdPhysics.DriveAPI: The created drive API schema.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path: {prim_path}")
    drive_api = UsdPhysics.DriveAPI.Apply(prim, drive_name)
    drive_api.CreateStiffnessAttr(stiffness)
    drive_api.CreateDampingAttr(damping)
    drive_api.CreateMaxForceAttr(max_force)
    drive_api.CreateTargetPositionAttr(target_position)
    drive_api.CreateTargetVelocityAttr(target_velocity)
    drive_api.CreateTypeAttr(drive_type)
    return drive_api


def get_drive_attributes(prim: Usd.Prim, drive_name: str) -> Dict[str, Any]:
    """Get the attributes of a physics drive applied to a prim.

    Args:
        prim (Usd.Prim): The prim to get the drive attributes from.
        drive_name (str): The name of the drive to get the attributes for.

    Returns:
        Dict[str, Any]: A dictionary containing the drive attributes.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    drive_api = UsdPhysics.DriveAPI.Get(prim, drive_name)
    if not drive_api:
        raise ValueError(f"Drive {drive_name} not found on prim {prim.GetPath()}.")
    attributes = {}
    if drive_api.GetDampingAttr().HasValue():
        attributes["damping"] = drive_api.GetDampingAttr().Get()
    if drive_api.GetMaxForceAttr().HasValue():
        attributes["maxForce"] = drive_api.GetMaxForceAttr().Get()
    if drive_api.GetStiffnessAttr().HasValue():
        attributes["stiffness"] = drive_api.GetStiffnessAttr().Get()
    if drive_api.GetTargetPositionAttr().HasValue():
        attributes["targetPosition"] = drive_api.GetTargetPositionAttr().Get()
    if drive_api.GetTargetVelocityAttr().HasValue():
        attributes["targetVelocity"] = drive_api.GetTargetVelocityAttr().Get()
    if drive_api.GetTypeAttr().HasValue():
        attributes["type"] = drive_api.GetTypeAttr().Get()
    return attributes


def set_drive_target_position(prim: Usd.Prim, drive_name: str, target_position: float) -> None:
    """Set the target position for a drive on a physics joint.

    Args:
        prim (Usd.Prim): The joint prim.
        drive_name (str): The name of the drive, e.g., "angular".
        target_position (float): The target position value to set.

    Raises:
        ValueError: If the prim is not a valid joint or if the drive API does not exist.
    """
    if not UsdPhysics.Joint(prim):
        raise ValueError(f"Prim {prim.GetPath()} is not a valid physics joint.")
    drive_api = UsdPhysics.DriveAPI.Get(prim, drive_name)
    if not drive_api:
        raise ValueError(f"Drive API '{drive_name}' does not exist on joint {prim.GetPath()}.")
    target_position_attr = drive_api.GetTargetPositionAttr()
    if not target_position_attr:
        target_position_attr = drive_api.CreateTargetPositionAttr()
    target_position_attr.Set(target_position)


def check_and_apply_drive(
    prim: Usd.Prim,
    drive_type: str,
    target_position: float,
    target_velocity: float,
    stiffness: float,
    damping: float,
    max_force: float,
) -> bool:
    """Check if a prim can have a drive applied and apply it if possible.

    Args:
        prim (Usd.Prim): The prim to apply the drive to.
        drive_type (str): The type of drive, either "linear" or "angular".
        target_position (float): The target position for the drive.
        target_velocity (float): The target velocity for the drive.
        stiffness (float): The stiffness of the drive.
        damping (float): The damping of the drive.
        max_force (float): The maximum force the drive can apply.

    Returns:
        bool: True if the drive was applied successfully, False otherwise.
    """
    if not prim.IsValid():
        print(f"Error: Prim {prim.GetPath()} is not valid.")
        return False
    if not prim.IsA(UsdPhysics.Joint):
        print(f"Error: Prim {prim.GetPath()} is not a joint.")
        return False
    if drive_type not in ["linear", "angular"]:
        print(f"Error: Invalid drive type {drive_type}. Must be 'linear' or 'angular'.")
        return False
    drive_api = UsdPhysics.DriveAPI.Apply(prim, drive_type)
    drive_api.CreateTypeAttr().Set(UsdPhysics.Tokens.force)
    drive_api.CreateTargetPositionAttr().Set(target_position)
    drive_api.CreateTargetVelocityAttr().Set(target_velocity)
    drive_api.CreateStiffnessAttr().Set(stiffness)
    drive_api.CreateDampingAttr().Set(damping)
    drive_api.CreateMaxForceAttr().Set(max_force)
    return True


def bulk_apply_drive(
    stage: Usd.Stage,
    joint_paths: List[Sdf.Path],
    drive_name: str,
    drive_type: str,
    target_position: float,
    target_velocity: float,
    stiffness: float,
    damping: float,
    max_force: float,
) -> None:
    """Apply a drive to multiple joint prims in bulk.

    Args:
        stage (Usd.Stage): The USD stage.
        joint_paths (List[Sdf.Path]): List of paths to joint prims.
        drive_name (str): Name of the drive.
        drive_type (str): Type of the drive (force or acceleration).
        target_position (float): Target position for the drive.
        target_velocity (float): Target velocity for the drive.
        stiffness (float): Stiffness of the drive.
        damping (float): Damping of the drive.
        max_force (float): Maximum force that can be applied by the drive.

    Raises:
        ValueError: If any of the joint paths are invalid or not of type UsdPhysicsJoint.
    """
    if drive_type not in ["force", "acceleration"]:
        raise ValueError(f"Invalid drive type: {drive_type}. Must be 'force' or 'acceleration'.")
    for joint_path in joint_paths:
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid() or not joint_prim.IsA(UsdPhysics.Joint):
            raise ValueError(f"Invalid or non-joint prim at path: {joint_path}")
        drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, drive_name)
        drive_api.CreateTypeAttr(drive_type)
        drive_api.CreateTargetPositionAttr(target_position)
        drive_api.CreateTargetVelocityAttr(target_velocity)
        drive_api.CreateStiffnessAttr(stiffness)
        drive_api.CreateDampingAttr(damping)
        drive_api.CreateMaxForceAttr(max_force)


def apply_drive_to_prim(
    prim: Usd.Prim,
    drive_name: str,
    drive_type: str,
    target_position: float,
    target_velocity: float,
    stiffness: float,
    damping: float,
    max_force: float,
) -> UsdPhysics.DriveAPI:
    """Apply a drive to a prim with the given parameters."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    valid_drive_types = ["force", "acceleration"]
    if drive_type not in valid_drive_types:
        raise ValueError(f"Invalid drive type: {drive_type}. Must be one of {valid_drive_types}")
    drive_api = UsdPhysics.DriveAPI.Apply(prim, drive_name)
    drive_api.CreateTypeAttr(Vt.Token(drive_type))
    drive_api.CreateTargetPositionAttr(target_position)
    drive_api.CreateTargetVelocityAttr(target_velocity)
    drive_api.CreateStiffnessAttr(stiffness)
    drive_api.CreateDampingAttr(damping)
    drive_api.CreateMaxForceAttr(max_force)
    return drive_api


def set_drive_max_force(prim: Usd.Prim, drive_name: str, max_force: float):
    """Set the max force for a drive on a PhysicsJoint prim.

    Args:
        prim (Usd.Prim): The PhysicsJoint prim.
        drive_name (str): The name of the drive (e.g., "angular").
        max_force (float): The max force value to set.

    Raises:
        ValueError: If the prim is not a valid PhysicsJoint or if the drive name is invalid.
        TypeError: If the max_force is not a valid float value.
    """
    if not UsdPhysics.Joint(prim):
        raise ValueError(f"Prim '{prim.GetPath()}' is not a valid PhysicsJoint.")
    valid_drive_names = ["angular", "linear", "transX", "transY", "transZ", "rotX", "rotY", "rotZ"]
    if drive_name not in valid_drive_names:
        raise ValueError(f"Invalid drive name '{drive_name}'. Must be one of: {', '.join(valid_drive_names)}")
    if not isinstance(max_force, float):
        raise TypeError(f"Max force must be a float value, got {type(max_force).__name__}.")
    drive_api = UsdPhysics.DriveAPI.Get(prim, drive_name)
    if not drive_api:
        drive_api = UsdPhysics.DriveAPI.Apply(prim, drive_name)
    drive_api.CreateMaxForceAttr(max_force)


def set_drive_stiffness(prim: Usd.Prim, drive_name: str, stiffness: float) -> None:
    """Set the drive stiffness for a physics drive on a prim.

    Args:
        prim (Usd.Prim): The prim with the physics drive.
        drive_name (str): The name of the drive, e.g., "angular".
        stiffness (float): The stiffness value to set.

    Raises:
        ValueError: If the prim is invalid or doesn't have the specified drive.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    drive_api = UsdPhysics.DriveAPI.Get(prim, drive_name)
    if not drive_api:
        raise ValueError(f"Drive '{drive_name}' not found on prim: {prim.GetPath()}")
    stiffness_attr = drive_api.GetStiffnessAttr()
    stiffness_attr.Set(stiffness)


def set_drive_target_velocity(prim: Usd.Prim, drive_name: str, target_velocity: float) -> None:
    """Set the target velocity for a drive on a physics joint prim.

    Args:
        prim (Usd.Prim): The joint prim.
        drive_name (str): The name of the drive, e.g., "angular".
        target_velocity (float): The target velocity value to set.

    Raises:
        ValueError: If the prim is not a valid joint or the drive name is not valid.
    """
    if not UsdPhysics.Joint(prim):
        raise ValueError(f"Prim {prim.GetPath()} is not a valid joint.")
    drive_api = UsdPhysics.DriveAPI.Get(prim, drive_name)
    if not drive_api:
        raise ValueError(f"Drive {drive_name} does not exist on joint {prim.GetPath()}.")
    target_velocity_attr = drive_api.GetTargetVelocityAttr()
    if target_velocity_attr:
        target_velocity_attr.Set(target_velocity)
    else:
        target_velocity_attr = drive_api.CreateTargetVelocityAttr(target_velocity, writeSparsely=True)


def configure_drive(
    stage: Usd.Stage,
    prim_path: str,
    drive_name: str,
    stiffness: float = 0.0,
    damping: float = 0.0,
    max_force: float = float("inf"),
    target_position: float = 0.0,
    target_velocity: float = 0.0,
    drive_type: str = "force",
) -> UsdPhysics.DriveAPI:
    """Configure a drive for a physics joint.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the joint prim.
        drive_name (str): The name of the drive (e.g., "angular").
        stiffness (float): The stiffness of the drive spring. Defaults to 0.
        damping (float): The damping of the drive. Defaults to 0.
        max_force (float): The maximum force that can be applied. Defaults to inf.
        target_position (float): The target value for position. Defaults to 0.
        target_velocity (float): The target value for velocity. Defaults to 0.
        drive_type (str): The type of drive - "force" or "acceleration". Defaults to "force".

    Returns:
        UsdPhysics.DriveAPI: The applied DriveAPI schema.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path '{prim_path}'")
    drive_api = UsdPhysics.DriveAPI.Apply(prim, drive_name)
    if drive_api.GetStiffnessAttr().Set(stiffness):
        drive_api.GetStiffnessAttr().Set(stiffness)
    if drive_api.GetDampingAttr().Set(damping):
        drive_api.GetDampingAttr().Set(damping)
    if drive_api.GetMaxForceAttr().Set(max_force):
        drive_api.GetMaxForceAttr().Set(max_force)
    if drive_api.GetTargetPositionAttr().Set(target_position):
        drive_api.GetTargetPositionAttr().Set(target_position)
    if drive_api.GetTargetVelocityAttr().Set(target_velocity):
        drive_api.GetTargetVelocityAttr().Set(target_velocity)
    if drive_api.GetTypeAttr().Set(drive_type):
        drive_api.GetTypeAttr().Set(drive_type)
    return drive_api


def find_prim_with_drive(stage: Usd.Stage, prim_path_prefix: str) -> Optional[Tuple[Usd.Prim, UsdPhysics.DriveAPI]]:
    """
    Find the first prim under the given path prefix that has a PhysicsDriveAPI applied.

    Args:
        stage (Usd.Stage): The USD stage to search.
        prim_path_prefix (str): The prefix of the prim path to search under.

    Returns:
        Optional[Tuple[Usd.Prim, UsdPhysics.DriveAPI]]: A tuple containing the found prim and the DriveAPI instance,
        or None if no prim with DriveAPI is found.
    """
    prim = stage.GetPrimAtPath(prim_path_prefix)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path_prefix} does not exist.")
    for descendant_prim in Usd.PrimRange(prim):
        drive_apis = UsdPhysics.DriveAPI.GetAll(descendant_prim)
        if drive_apis:
            return (descendant_prim, drive_apis[0])
    return None


def duplicate_drive_to_other_prim(src_prim: Usd.Prim, dst_prim: Usd.Prim, drive_name: str) -> bool:
    """Duplicate a drive from one prim to another.

    Args:
        src_prim (Usd.Prim): The source prim to copy the drive from.
        dst_prim (Usd.Prim): The destination prim to copy the drive to.
        drive_name (str): The name of the drive to copy.

    Returns:
        bool: True if the drive was successfully copied, False otherwise.
    """
    src_drive_api = UsdPhysics.DriveAPI.Get(src_prim, drive_name)
    if not src_drive_api:
        print(f"Source prim {src_prim.GetPath()} does not have a drive named {drive_name}")
        return False
    dst_drive_api = UsdPhysics.DriveAPI.Get(dst_prim, drive_name)
    if dst_drive_api:
        print(f"Destination prim {dst_prim.GetPath()} already has a drive named {drive_name}")
        return False
    dst_drive_api = UsdPhysics.DriveAPI.Apply(dst_prim, drive_name)
    if src_drive_api.GetTypeAttr().HasAuthoredValue():
        dst_drive_api.CreateTypeAttr(src_drive_api.GetTypeAttr().Get())
    if src_drive_api.GetMaxForceAttr().HasAuthoredValue():
        dst_drive_api.CreateMaxForceAttr(src_drive_api.GetMaxForceAttr().Get())
    if src_drive_api.GetTargetPositionAttr().HasAuthoredValue():
        dst_drive_api.CreateTargetPositionAttr(src_drive_api.GetTargetPositionAttr().Get())
    if src_drive_api.GetTargetVelocityAttr().HasAuthoredValue():
        dst_drive_api.CreateTargetVelocityAttr(src_drive_api.GetTargetVelocityAttr().Get())
    if src_drive_api.GetDampingAttr().HasAuthoredValue():
        dst_drive_api.CreateDampingAttr(src_drive_api.GetDampingAttr().Get())
    if src_drive_api.GetStiffnessAttr().HasAuthoredValue():
        dst_drive_api.CreateStiffnessAttr(src_drive_api.GetStiffnessAttr().Get())
    return True


def sync_drive_attributes(
    drive_api: UsdPhysics.DriveAPI,
    damping: float,
    stiffness: float,
    max_force: float,
    target_position: float,
    target_velocity: float,
):
    """Synchronize the drive attributes on a PhysicsDriveAPI.

    Args:
        drive_api (UsdPhysics.DriveAPI): The PhysicsDriveAPI to sync attributes on.
        damping (float): The damping value to set.
        stiffness (float): The stiffness value to set.
        max_force (float): The maximum force value to set.
        target_position (float): The target position value to set.
        target_velocity (float): The target velocity value to set.
    """
    damping_attr = drive_api.GetDampingAttr()
    stiffness_attr = drive_api.GetStiffnessAttr()
    max_force_attr = drive_api.GetMaxForceAttr()
    target_position_attr = drive_api.GetTargetPositionAttr()
    target_velocity_attr = drive_api.GetTargetVelocityAttr()
    if damping_attr.IsValid():
        damping_attr.Set(damping)
    else:
        print(f"Warning: Damping attribute not found on {drive_api.GetPath()}")
    if stiffness_attr.IsValid():
        stiffness_attr.Set(stiffness)
    else:
        print(f"Warning: Stiffness attribute not found on {drive_api.GetPath()}")
    if max_force_attr.IsValid():
        max_force_attr.Set(max_force)
    else:
        print(f"Warning: MaxForce attribute not found on {drive_api.GetPath()}")
    if target_position_attr.IsValid():
        target_position_attr.Set(target_position)
    else:
        print(f"Warning: TargetPosition attribute not found on {drive_api.GetPath()}")
    if target_velocity_attr.IsValid():
        target_velocity_attr.Set(target_velocity)
    else:
        print(f"Warning: TargetVelocity attribute not found on {drive_api.GetPath()}")


def convert_drive_type(drive_type: str) -> str:
    """Convert a string drive type to a valid UsdPhysics.DriveAPI type value.

    Args:
        drive_type (str): The drive type as a string. Must be either "force" or "acceleration".

    Returns:
        str: The corresponding drive type value.

    Raises:
        ValueError: If the input drive_type is not a valid value.
    """
    if drive_type not in ["force", "acceleration"]:
        raise ValueError(f"Invalid drive type: {drive_type}. Must be either 'force' or 'acceleration'.")
    return drive_type


def set_drive_damping(drive_api: UsdPhysics.DriveAPI, damping: float) -> None:
    """Set the damping value for a DriveAPI."""
    if not drive_api.GetPrim().IsValid():
        raise ValueError("Invalid DriveAPI.")
    damping_attr = drive_api.GetDampingAttr()
    if not damping_attr.IsDefined():
        damping_attr = drive_api.CreateDampingAttr(damping, writeSparsely=True)
    success = damping_attr.Set(damping)
    if not success:
        raise ValueError("Failed to set damping value.")


def remove_drive_from_prim(prim: Usd.Prim, drive_name: str) -> bool:
    """Remove a drive with the given name from the prim.

    Args:
        prim (Usd.Prim): The prim to remove the drive from.
        drive_name (str): The name of the drive to remove.

    Returns:
        bool: True if the drive was successfully removed, False otherwise.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    drive_schema_name = UsdPhysics.DriveAPI.GetSchemaAttributeNames(False, drive_name)
    if not prim.HasAPI(UsdPhysics.DriveAPI, drive_name):
        print(f"Drive '{drive_name}' not found on prim '{prim.GetPath()}'")
        return False
    success = prim.RemoveAPI(UsdPhysics.DriveAPI, drive_name)
    return success


def list_all_drives_on_prim(prim: Usd.Prim) -> List[UsdPhysics.DriveAPI]:
    """
    Return a list of all UsdPhysics.DriveAPI applied to the given prim.

    Args:
        prim (Usd.Prim): The prim to query for drive APIs.

    Returns:
        List[UsdPhysics.DriveAPI]: A list of UsdPhysics.DriveAPI objects.
                                    The list will be empty if no drive APIs are found.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    applied_schemas = prim.GetAppliedSchemas()
    drive_schemas = [
        UsdPhysics.DriveAPI(prim, schema_name)
        for schema_name in applied_schemas
        if schema_name.startswith("PhysicsDriveAPI:")
    ]
    return drive_schemas


def get_filtered_pairs_relationships(prim: Usd.Prim) -> List[Usd.Prim]:
    """
    Get the list of prims that are filtered pairs for the given prim.

    Args:
        prim (Usd.Prim): The prim to get the filtered pairs for.

    Returns:
        List[Usd.Prim]: A list of prims that are filtered pairs for the given prim.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    filtered_pairs_api = UsdPhysics.FilteredPairsAPI(prim)
    if not filtered_pairs_api:
        raise ValueError(f"Prim {prim} does not have a valid FilteredPairsAPI")
    filtered_pairs_rel = filtered_pairs_api.GetFilteredPairsRel()
    if not filtered_pairs_rel.IsValid():
        return []
    filtered_pairs = [prim.GetStage().GetPrimAtPath(target) for target in filtered_pairs_rel.GetTargets()]
    return [prim for prim in filtered_pairs if prim.IsValid()]


def apply_filtered_pairs_api(prim: Usd.Prim, filtered_prims: List[Usd.Prim]) -> UsdPhysics.FilteredPairsAPI:
    """Apply FilteredPairsAPI to a prim and add filtered prims.

    Args:
        prim (Usd.Prim): The prim to apply the FilteredPairsAPI to.
        filtered_prims (List[Usd.Prim]): A list of prims to add as filtered pairs.

    Returns:
        UsdPhysics.FilteredPairsAPI: The applied FilteredPairsAPI object.

    Raises:
        ValueError: If the input prim is not valid.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(prim)
    filtered_pairs_rel = filtered_pairs_api.GetFilteredPairsRel()
    for filtered_prim in filtered_prims:
        filtered_pairs_rel.AddTarget(filtered_prim.GetPath())
    return filtered_pairs_api


def add_filtered_pair(prim: Usd.Prim, filtered_prim: Usd.Prim) -> None:
    """Add a filtered pair to the FilteredPairsAPI on the given prim."""
    if not prim.IsValid():
        raise ValueError("The prim is not valid.")
    if not filtered_prim.IsValid():
        raise ValueError("The filtered prim is not valid.")
    if not UsdPhysics.FilteredPairsAPI.CanApply(prim):
        raise ValueError("The FilteredPairsAPI cannot be applied to the prim.")
    filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(prim)
    filtered_pairs_rel = filtered_pairs_api.GetFilteredPairsRel()
    filtered_pairs_rel.AddTarget(filtered_prim.GetPath())


def remove_filtered_pair(filtered_pairs_api: UsdPhysics.FilteredPairsAPI, prim_path: str) -> bool:
    """Remove a prim path from the filteredPairs relationship.

    Args:
        filtered_pairs_api (UsdPhysics.FilteredPairsAPI): The FilteredPairsAPI object.
        prim_path (str): The prim path to remove from the filteredPairs relationship.

    Returns:
        bool: True if the prim path was successfully removed, False otherwise.
    """
    filtered_pairs_rel = filtered_pairs_api.GetFilteredPairsRel()
    if not filtered_pairs_rel.IsValid():
        return False
    targets = filtered_pairs_rel.GetTargets()
    path_to_remove = Sdf.Path(prim_path)
    if path_to_remove not in targets:
        return False
    targets.remove(path_to_remove)
    filtered_pairs_rel.ClearTargets(True)
    filtered_pairs_rel.SetTargets(targets)
    return True


def clear_filtered_pairs(prim: Usd.Prim) -> None:
    """Clear the filteredPairs relationship on the given prim."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    filtered_pairs_api = UsdPhysics.FilteredPairsAPI(prim)
    if not filtered_pairs_api:
        raise ValueError(f"Prim {prim} does not have the FilteredPairsAPI applied.")
    filtered_pairs_rel = filtered_pairs_api.GetFilteredPairsRel()
    if not filtered_pairs_rel.IsValid():
        return
    filtered_pairs_rel.ClearTargets(removeSpec=True)


def can_apply_filtered_pairs_api(prim: Usd.Prim) -> bool:
    """Check if the FilteredPairsAPI can be applied to the given prim."""
    if not prim.IsValid():
        return False
    if not prim.IsA(UsdGeom.Imageable):
        return False
    if not (
        prim.IsA(UsdPhysics.RigidBodyAPI)
        or prim.IsA(UsdPhysics.CollisionAPI)
        or prim.IsA(UsdPhysics.ArticulationRootAPI)
    ):
        return False
    return True


def get_filtered_pairs(prim: Usd.Prim) -> List[Usd.Prim]:
    """Get the list of prims that are filtered for the given prim."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    filtered_pairs_api = UsdPhysics.FilteredPairsAPI(prim)
    if not filtered_pairs_api:
        raise ValueError(f"FilteredPairsAPI not applied to prim: {prim}")
    filtered_pairs_rel = filtered_pairs_api.GetFilteredPairsRel()
    if not filtered_pairs_rel.IsValid():
        return []
    filtered_prims = []
    for target_path in filtered_pairs_rel.GetTargets():
        target_prim = prim.GetStage().GetPrimAtPath(target_path)
        if target_prim.IsValid():
            filtered_prims.append(target_prim)
    return filtered_prims


def get_fixed_joints(stage: Usd.Stage) -> List[UsdPhysics.FixedJoint]:
    """
    Get all UsdPhysics.FixedJoint prims in the given USD stage.

    Args:
        stage (Usd.Stage): The USD stage to search for fixed joints.

    Returns:
        List[UsdPhysics.FixedJoint]: A list of all UsdPhysics.FixedJoint prims found in the stage.
    """
    fixed_joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.FixedJoint):
            fixed_joints.append(UsdPhysics.FixedJoint(prim))
    return fixed_joints


def connect_fixed_joint_to_bodies(
    stage: Usd.Stage, joint_path: str, body0_path: str, body1_path: str
) -> UsdPhysics.FixedJoint:
    """Connect a fixed joint to two bodies.

    Args:
        stage (Usd.Stage): The USD stage.
        joint_path (str): The path to the fixed joint prim.
        body0_path (str): The path to the first body prim.
        body1_path (str): The path to the second body prim.

    Returns:
        UsdPhysics.FixedJoint: The connected fixed joint prim.

    Raises:
        ValueError: If any of the provided paths are invalid or don't correspond to the expected prim types.
    """
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid():
        raise ValueError(f"Invalid joint path: {joint_path}")
    joint = UsdPhysics.FixedJoint(joint_prim)
    if not joint:
        raise ValueError(f"Prim at path {joint_path} is not a valid UsdPhysicsFixedJoint")
    body0_prim = stage.GetPrimAtPath(body0_path)
    if not body0_prim.IsValid():
        raise ValueError(f"Invalid body path: {body0_path}")
    body1_prim = stage.GetPrimAtPath(body1_path)
    if not body1_prim.IsValid():
        raise ValueError(f"Invalid body path: {body1_path}")
    joint.CreateBody0Rel().SetTargets([body0_prim.GetPath()])
    joint.CreateBody1Rel().SetTargets([body1_prim.GetPath()])
    return joint


def configure_fixed_joint_parameters(stage: Usd.Stage, joint_path: str) -> UsdPhysics.FixedJoint:
    """Configure parameters for a fixed joint prim.

    Args:
        stage (Usd.Stage): The USD stage.
        joint_path (str): The path to the fixed joint prim.

    Returns:
        UsdPhysics.FixedJoint: The configured fixed joint prim.

    Raises:
        ValueError: If the prim at joint_path is not a valid UsdPhysics.FixedJoint.
    """
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid():
        raise ValueError(f"No prim found at path: {joint_path}")
    fixed_joint = UsdPhysics.FixedJoint(joint_prim)
    if not fixed_joint:
        raise ValueError(f"Prim at path {joint_path} is not a valid UsdPhysics.FixedJoint")
    local_pos0 = fixed_joint.CreateLocalPos0Attr()
    local_pos0.Set(Gf.Vec3f(0, 0, 0))
    local_pos1 = fixed_joint.CreateLocalPos1Attr()
    local_pos1.Set(Gf.Vec3f(0, 0, 0))
    body0 = fixed_joint.CreateBody0Rel()
    body0.SetTargets(["/World/PhysicsScene/Cube"])
    body1 = fixed_joint.CreateBody1Rel()
    body1.SetTargets(["/World/PhysicsScene/Sphere"])
    fixed_joint.CreateJointEnabledAttr().Set(True)
    return fixed_joint


def get_joint_break_conditions(joint: UsdPhysics.Joint) -> Tuple[float, float]:
    """Get the break force and break torque for a joint, if authored.

    Args:
        joint (UsdPhysics.Joint): The joint prim to query.

    Returns:
        A tuple of (break_force, break_torque). If either value is not authored,
        it will be returned as float('inf').
    """
    break_force_attr = joint.GetBreakForceAttr()
    if break_force_attr.HasAuthoredValue():
        break_force = break_force_attr.Get()
    else:
        break_force = float("inf")
    break_torque_attr = joint.GetBreakTorqueAttr()
    if break_torque_attr.HasAuthoredValue():
        break_torque = break_torque_attr.Get()
    else:
        break_torque = float("inf")
    return (break_force, break_torque)


def get_joint_enabled_state(joint: UsdPhysics.Joint) -> bool:
    """Get the enabled state of a physics joint.

    Args:
        joint (UsdPhysics.Joint): The joint to query the enabled state of.

    Returns:
        bool: True if the joint is enabled, False otherwise.

    Raises:
        ValueError: If the provided prim is not a valid UsdPhysics.Joint.
    """
    if not joint:
        raise ValueError("Invalid UsdPhysics.Joint prim.")
    enabled_attr = joint.GetJointEnabledAttr()
    if not enabled_attr:
        return True
    return enabled_attr.Get()


def search_joints_by_body(stage: Usd.Stage, body_path: str) -> List[UsdPhysics.Joint]:
    """
    Search for all joints that are connected to the given body.

    Args:
        stage (Usd.Stage): The USD stage to search.
        body_path (str): The path to the body prim.

    Returns:
        List[UsdPhysics.Joint]: A list of joints connected to the body.
    """
    body_prim = stage.GetPrimAtPath(body_path)
    if not body_prim.IsValid():
        raise ValueError(f"Body prim at path {body_path} does not exist.")
    joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            body0_rel = joint.GetBody0Rel()
            body1_rel = joint.GetBody1Rel()
            if body0_rel.GetTargets() == [body_prim.GetPath()] or body1_rel.GetTargets() == [body_prim.GetPath()]:
                joints.append(joint)
    return joints


def create_generic_joint(stage: Usd.Stage, prim_path: str, body0_path: str, body1_path: str) -> UsdPhysics.Joint:
    """Create a generic D6 joint between two bodies.

    Args:
        stage (Usd.Stage): The stage to create the joint on.
        prim_path (str): The path where the joint should be created.
        body0_path (str): The path to the first body to be jointed.
        body1_path (str): The path to the second body to be jointed.

    Returns:
        UsdPhysics.Joint: The created joint.

    Raises:
        ValueError: If the bodies are not valid prims.
    """
    body0 = stage.GetPrimAtPath(body0_path)
    body1 = stage.GetPrimAtPath(body1_path)
    if not body0.IsValid() or not body1.IsValid():
        raise ValueError("One or both of the specified bodies are not valid.")
    if not UsdGeom.Xformable(body0) or not UsdGeom.Xformable(body1):
        raise ValueError("One or both of the specified bodies are not Xformable.")
    joint = UsdPhysics.Joint.Define(stage, Sdf.Path(prim_path))
    joint.CreateBody0Rel().SetTargets([body0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body1.GetPath()])
    joint.CreateJointEnabledAttr(True)
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateExcludeFromArticulationAttr(False)
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    return joint


def batch_create_joints(stage: Usd.Stage, joint_data: List[Tuple[str, str, str]]) -> List[UsdPhysics.Joint]:
    """
    Create multiple joints in a batch operation.

    Args:
        stage (Usd.Stage): The USD stage to create the joints in.
        joint_data (List[Tuple[str, str, str]]): A list of tuples containing joint data:
            - Tuple[0]: Joint prim path
            - Tuple[1]: Path to first body (or empty string for world)
            - Tuple[2]: Path to second body (or empty string for world)

    Returns:
        List[UsdPhysics.Joint]: A list of created joint prims.
    """
    created_joints = []
    for joint_path, body0_path, body1_path in joint_data:
        joint_prim = UsdPhysics.Joint.Define(stage, joint_path)
        if not joint_prim:
            raise RuntimeError(f"Failed to create joint prim at path: {joint_path}")
        if body0_path:
            body0_prim = stage.GetPrimAtPath(body0_path)
            if not body0_prim.IsValid():
                raise ValueError(f"Body prim at path {body0_path} does not exist.")
            joint_prim.CreateBody0Rel().SetTargets([body0_prim.GetPath()])
        if body1_path:
            body1_prim = stage.GetPrimAtPath(body1_path)
            if not body1_prim.IsValid():
                raise ValueError(f"Body prim at path {body1_path} does not exist.")
            joint_prim.CreateBody1Rel().SetTargets([body1_prim.GetPath()])
        created_joints.append(joint_prim)
    return created_joints


def enable_joint_collision(joint: UsdPhysics.Joint, enable: bool = True) -> None:
    """Enable or disable collision between jointed bodies.

    Args:
        joint (UsdPhysics.Joint): The joint to modify.
        enable (bool): Whether to enable or disable collision. Default is True.

    Raises:
        ValueError: If the input joint is invalid.
    """
    if not joint:
        raise ValueError("Invalid joint provided.")
    collision_attr = joint.GetCollisionEnabledAttr()
    if not collision_attr:
        raise ValueError("Joint does not have a collisionEnabled attribute.")
    collision_attr.Set(enable)


def create_fixed_joint(
    stage: Usd.Stage,
    joint_path: str,
    body0_path: str,
    body1_path: str,
    local_pos0: Gf.Vec3f = Gf.Vec3f(0, 0, 0),
    local_rot0: Gf.Quatf = Gf.Quatf(1, 0, 0, 0),
    local_pos1: Gf.Vec3f = Gf.Vec3f(0, 0, 0),
    local_rot1: Gf.Quatf = Gf.Quatf(1, 0, 0, 0),
) -> UsdPhysics.Joint:
    """Create a fixed joint between two bodies.

    Args:
        stage (Usd.Stage): The USD stage to create the joint on.
        joint_path (str): The path where the joint should be created.
        body0_path (str): The path to the first body to be jointed.
        body1_path (str): The path to the second body to be jointed.
        local_pos0 (Gf.Vec3f): The relative position of the joint frame to body0's frame. Defaults to (0, 0, 0).
        local_rot0 (Gf.Quatf): The relative orientation of the joint frame to body0's frame. Defaults to (1, 0, 0, 0).
        local_pos1 (Gf.Vec3f): The relative position of the joint frame to body1's frame. Defaults to (0, 0, 0).
        local_rot1 (Gf.Quatf): The relative orientation of the joint frame to body1's frame. Defaults to (1, 0, 0, 0).

    Returns:
        UsdPhysics.Joint: The created joint prim.

    Raises:
        ValueError: If the bodies are not valid prims or if the joint path already exists.
    """
    body0 = stage.GetPrimAtPath(body0_path)
    body1 = stage.GetPrimAtPath(body1_path)
    if not body0.IsValid() or not body1.IsValid():
        raise ValueError("One or both of the specified bodies are not valid prims.")
    if stage.GetPrimAtPath(joint_path):
        raise ValueError(f"A prim already exists at the specified joint path: {joint_path}")
    joint = UsdPhysics.Joint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([body0_path])
    joint.CreateBody1Rel().SetTargets([body1_path])
    joint.CreateLocalPos0Attr().Set(local_pos0)
    joint.CreateLocalRot0Attr().Set(local_rot0)
    joint.CreateLocalPos1Attr().Set(local_pos1)
    joint.CreateLocalRot1Attr().Set(local_rot1)
    return joint


def reparent_joint(joint: UsdPhysics.Joint, new_parent_prim: Usd.Prim) -> None:
    """Reparent a joint to a new parent prim."""
    if not joint:
        raise ValueError("Invalid joint provided.")
    if not new_parent_prim.IsValid():
        raise ValueError("Invalid new parent prim.")
    current_parent_prim = joint.GetPrim().GetParent()
    if new_parent_prim == current_parent_prim:
        return
    joint_path = joint.GetPath()
    joint_name = joint_path.name
    new_joint_path = new_parent_prim.GetPath().AppendChild(joint_name)
    joint.GetPrim().SetActive(False)
    result = joint.GetPrim().GetStage().DefinePrim(new_joint_path)
    joint = UsdPhysics.Joint(result)
    joint.GetPrim().SetActive(True)


def delete_joint(stage: Usd.Stage, joint_path: str) -> bool:
    """Delete a joint prim from the stage.

    Args:
        stage (Usd.Stage): The USD stage.
        joint_path (str): The path to the joint prim.

    Returns:
        bool: True if the joint was deleted, False otherwise.
    """
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid() or not UsdPhysics.Joint(joint_prim):
        print(f"Joint prim at path '{joint_path}' does not exist or is not a joint.")
        return False
    stage.RemovePrim(joint_path)
    if stage.GetPrimAtPath(joint_path):
        print(f"Failed to delete joint prim at path '{joint_path}'.")
        return False
    return True


def get_joint_collision_state(joint: UsdPhysics.Joint) -> bool:
    """Get the collision enabled state for a joint."""
    if not joint:
        raise ValueError("Invalid joint passed to get_joint_collision_state.")
    collision_attr = joint.GetCollisionEnabledAttr()
    if collision_attr.HasAuthoredValue():
        return collision_attr.Get()
    else:
        return False


def toggle_joint_state(joint: UsdPhysics.Joint) -> bool:
    """Toggle the state of a physics joint.

    Args:
        joint (UsdPhysics.Joint): The joint to toggle the state of.

    Returns:
        bool: The new state of the joint (True for enabled, False for disabled).
    """
    joint_enabled_attr = joint.GetJointEnabledAttr()
    if not joint_enabled_attr:
        joint_enabled_attr = joint.CreateJointEnabledAttr(True, True)
        return True
    current_state = joint_enabled_attr.Get()
    new_state = not current_state
    joint_enabled_attr.Set(new_state)
    return new_state


def create_joint_with_defaults(
    stage: Usd.Stage, parent_path: str, name: str, body0: Usd.Prim, body1: Usd.Prim
) -> UsdPhysics.Joint:
    """Create a joint with default attribute values between two bodies.

    Args:
        stage (Usd.Stage): The stage to create the joint on.
        parent_path (str): The path where the joint should be created.
        name (str): The name of the joint.
        body0 (Usd.Prim): The first body to connect with the joint.
        body1 (Usd.Prim): The second body to connect with the joint.

    Returns:
        UsdPhysics.Joint: The created joint.

    Raises:
        ValueError: If either body0 or body1 is not a valid prim.
    """
    if not body0.IsValid() or not body1.IsValid():
        raise ValueError("Both body0 and body1 must be valid prims.")
    joint_path = f"{parent_path}{name}"
    joint = UsdPhysics.Joint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([body0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body1.GetPath()])
    joint.CreateLocalPos0Attr()
    joint.CreateLocalRot0Attr()
    joint.CreateLocalPos1Attr()
    joint.CreateLocalRot1Attr()
    joint.CreateJointEnabledAttr()
    joint.CreateCollisionEnabledAttr()
    joint.CreateExcludeFromArticulationAttr()
    joint.CreateBreakForceAttr()
    joint.CreateBreakTorqueAttr()
    return joint


def exclude_joint_from_articulation(joint_prim: UsdPhysics.Joint) -> None:
    """Exclude a joint from articulation by setting physics:excludeFromArticulation to True.

    Args:
        joint_prim (UsdPhysics.Joint): The joint prim to exclude from articulation.

    Raises:
        ValueError: If the input prim is not a valid UsdPhysics.Joint.
    """
    if not joint_prim:
        raise ValueError("Invalid joint prim.")
    exclude_attr = joint_prim.GetExcludeFromArticulationAttr()
    if not exclude_attr:
        exclude_attr = joint_prim.CreateExcludeFromArticulationAttr(True, True)
    exclude_attr.Set(True)


def test_exclude_joint_from_articulation():
    stage = Usd.Stage.CreateInMemory()
    joint_path = "/World/MyJoint"
    joint_prim = UsdPhysics.Joint.Define(stage, joint_path)
    exclude_joint_from_articulation(joint_prim)
    assert joint_prim.GetExcludeFromArticulationAttr().Get() == True
    print(stage.GetRootLayer().ExportToString())


def get_joint_local_transform(joint: UsdPhysics.Joint, body_index: int) -> Gf.Matrix4d:
    """
    Get the local transform of a joint relative to one of its connected bodies.

    Args:
        joint (UsdPhysics.Joint): The joint to get the local transform for.
        body_index (int): The index of the body to get the local transform relative to (0 or 1).

    Returns:
        Gf.Matrix4d: The local transform matrix.

    Raises:
        ValueError: If the joint is invalid or if the body index is not 0 or 1.
    """
    if not joint:
        raise ValueError("Invalid joint.")
    if body_index not in [0, 1]:
        raise ValueError("Body index must be 0 or 1.")
    local_pos_attr = joint.GetLocalPos0Attr() if body_index == 0 else joint.GetLocalPos1Attr()
    local_rot_attr = joint.GetLocalRot0Attr() if body_index == 0 else joint.GetLocalRot1Attr()
    local_pos = local_pos_attr.Get()
    local_rot = local_rot_attr.Get()
    local_pos_vec3d = Gf.Vec3d(local_pos)
    translation_matrix = Gf.Matrix4d().SetTranslate(local_pos_vec3d)
    rotation_matrix = Gf.Matrix4d().SetRotate(local_rot)
    local_transform = translation_matrix * rotation_matrix
    return local_transform


def search_joints_by_attributes(stage: Usd.Stage, attributes: Dict[str, Any]) -> List[UsdPhysics.Joint]:
    """
    Search for joints in the stage that match the given attribute values.

    Args:
        stage (Usd.Stage): The USD stage to search for joints.
        attributes (Dict[str, Any]): A dictionary of attribute names and their expected values.
            The keys should be attribute names (e.g., "jointEnabled", "collisionEnabled").
            The values should be the expected values for those attributes.

    Returns:
        List[UsdPhysics.Joint]: A list of joints that match the given attribute values.
    """
    prims = stage.Traverse()
    joints = [UsdPhysics.Joint(prim) for prim in prims if prim.IsA(UsdPhysics.Joint)]
    matching_joints = []
    for joint in joints:
        match = True
        for attr_name, attr_value in attributes.items():
            attr = joint.GetPrim().GetAttribute(attr_name)
            if not attr:
                match = False
                break
            value = attr.Get()
            if value != attr_value:
                match = False
                break
        if match:
            matching_joints.append(joint)
    return matching_joints


def set_joint_break_conditions(joint: UsdPhysics.Joint, break_force: float, break_torque: float) -> None:
    """Set the break force and torque thresholds for a joint.

    If the joint force or torque exceeds these values, the joint will break.

    Args:
        joint (UsdPhysics.Joint): The joint prim to set break conditions on.
        break_force (float): The force threshold at which the joint breaks.
            A value of inf means the joint will never break due to force.
        break_torque (float): The torque threshold at which the joint breaks.
            A value of inf means the joint will never break due to torque.

    Raises:
        ValueError: If the provided prim is not a valid joint.
    """
    if not joint.GetPrim().IsA(UsdPhysics.Joint):
        raise ValueError("Provided prim is not a valid UsdPhysics.Joint")
    break_force_attr = joint.GetBreakForceAttr()
    break_torque_attr = joint.GetBreakTorqueAttr()
    if break_force_attr:
        break_force_attr.Set(break_force)
    else:
        joint.CreateBreakForceAttr(break_force, writeSparsely=True)
    if break_torque_attr:
        break_torque_attr.Set(break_torque)
    else:
        joint.CreateBreakTorqueAttr(break_torque, writeSparsely=True)


def copy_joint_attributes(src_joint: UsdPhysics.Joint, dst_joint: UsdPhysics.Joint) -> None:
    """Copy attributes from one joint to another.

    Args:
        src_joint (UsdPhysics.Joint): The source joint to copy attributes from.
        dst_joint (UsdPhysics.Joint): The destination joint to copy attributes to.
    """
    if not src_joint or not dst_joint:
        raise ValueError("Invalid source or destination joint.")
    joint_attr_names = UsdPhysics.Joint.GetSchemaAttributeNames(includeInherited=False)
    for attr_name in joint_attr_names:
        src_attr = src_joint.GetPrim().GetAttribute(attr_name)
        if src_attr.HasValue():
            attr_value = src_attr.Get()
            dst_attr = dst_joint.GetPrim().CreateAttribute(attr_name, src_attr.GetTypeName())
            dst_attr.Set(attr_value)


def decompose_transform(transform):
    """Decompose a transform into translation and rotation components."""
    translation = transform.ExtractTranslation()
    rotation = Gf.Quatf(transform.ExtractRotationQuat())
    return (translation, rotation)


def set_joint_local_transform(joint: UsdPhysics.Joint, body0_transform: Gf.Matrix4d, body1_transform: Gf.Matrix4d):
    """Set the local transforms for a joint relative to the two bodies it connects.

    Args:
        joint (UsdPhysics.Joint): The joint to set the local transforms for.
        body0_transform (Gf.Matrix4d): The transform of the first body the joint connects.
        body1_transform (Gf.Matrix4d): The transform of the second body the joint connects.
    """
    if not joint:
        raise ValueError("Invalid joint provided.")
    body0_inv_transform = body0_transform.GetInverse()
    body1_inv_transform = body1_transform.GetInverse()
    joint_transform = UsdGeom.Xformable(joint).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    local_transform0 = body0_inv_transform * joint_transform
    local_transform1 = body1_inv_transform * joint_transform
    (local_pos0, local_rot0) = decompose_transform(local_transform0)
    (local_pos1, local_rot1) = decompose_transform(local_transform1)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(local_pos0))
    joint.CreateLocalRot0Attr().Set(local_rot0)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(local_pos1))
    joint.CreateLocalRot1Attr().Set(local_rot1)


def create_d6_joint(stage: Usd.Stage, joint_path: str, body0_path: str, body1_path: str) -> UsdPhysics.Joint:
    """Create a D6 joint between two rigid bodies or one rigid body and world.

    Args:
        stage (Usd.Stage): The stage to create the joint on.
        joint_path (str): The path where the joint will be created.
        body0_path (str): The path to the first rigid body or an empty string to indicate world.
        body1_path (str): The path to the second rigid body or an empty string to indicate world.

    Returns:
        UsdPhysics.Joint: The created D6 joint.

    Raises:
        ValueError: If the joint path is invalid or either body path is invalid.
    """
    if not Sdf.Path(joint_path).IsPrimPath():
        raise ValueError(f"Invalid joint path: {joint_path}")
    joint = UsdPhysics.Joint.Get(stage, joint_path)
    if not joint:
        joint = UsdPhysics.Joint.Define(stage, joint_path)
    if body0_path:
        body0 = stage.GetPrimAtPath(body0_path)
        if not body0.IsValid():
            raise ValueError(f"Invalid body0 path: {body0_path}")
        joint.CreateBody0Rel().SetTargets([body0.GetPath()])
    else:
        joint.CreateBody0Rel().ClearTargets(True)
    if body1_path:
        body1 = stage.GetPrimAtPath(body1_path)
        if not body1.IsValid():
            raise ValueError(f"Invalid body1 path: {body1_path}")
        joint.CreateBody1Rel().SetTargets([body1.GetPath()])
    else:
        joint.CreateBody1Rel().ClearTargets(True)
    return joint


def apply_limit_to_joint(
    joint_prim: Usd.Prim, limit_name: str, low_limit: float, high_limit: float
) -> UsdPhysics.LimitAPI:
    """Apply a limit to a physics joint prim.

    Args:
        joint_prim (Usd.Prim): The joint prim to apply the limit to.
        limit_name (str): The name of the limit (e.g., "transX", "rotY").
        low_limit (float): The lower limit value.
        high_limit (float): The upper limit value.

    Returns:
        UsdPhysics.LimitAPI: The applied limit API schema.

    Raises:
        ValueError: If the joint prim is not a valid UsdPhysics.Joint.
    """
    if not UsdPhysics.Joint(joint_prim):
        raise ValueError(f"Prim '{joint_prim.GetPath()}' is not a valid UsdPhysics.Joint")
    limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, limit_name)
    limit_api.CreateLowAttr(low_limit)
    limit_api.CreateHighAttr(high_limit)
    return limit_api


def can_apply_limit_to_joint(joint_prim: Usd.Prim, limit_name: str) -> bool:
    """Check if a limit with the given name can be applied to the joint prim.

    Args:
        joint_prim (Usd.Prim): The joint prim to check.
        limit_name (str): The name of the limit to check.

    Returns:
        bool: True if the limit can be applied, False otherwise.
    """
    if not UsdPhysics.Joint(joint_prim):
        return False
    valid_limit_names = ["transX", "transY", "transZ", "rotX", "rotY", "rotZ", "distance"]
    if limit_name not in valid_limit_names:
        return False
    limit_api = UsdPhysics.LimitAPI.Get(joint_prim, limit_name)
    if limit_api.GetPrim():
        return False
    return True


def get_joint_limit_range(joint_prim: Usd.Prim, axis: str) -> Tuple[float, float]:
    """
    Get the joint limit range for a given joint prim and axis.

    Args:
        joint_prim (Usd.Prim): The joint prim.
        axis (str): The axis to get the limit range for. Valid values are "transX", "transY", "transZ", "rotX", "rotY", "rotZ", "distance".

    Returns:
        Tuple[float, float]: A tuple containing the low and high limit values.

    Raises:
        ValueError: If the joint prim is not valid or if the specified axis is not valid.
    """
    if not joint_prim.IsValid():
        raise ValueError(f"Invalid joint prim: {joint_prim}")
    joint_api = UsdPhysics.Joint(joint_prim)
    if not joint_api:
        raise ValueError(f"Prim {joint_prim.GetPath()} is not a valid PhysicsJoint")
    valid_axes = ["transX", "transY", "transZ", "rotX", "rotY", "rotZ", "distance"]
    if axis not in valid_axes:
        raise ValueError(f"Invalid axis: {axis}. Valid axes are: {valid_axes}")
    limit_api = UsdPhysics.LimitAPI.Get(joint_prim, axis)
    if not limit_api:
        return (float("-inf"), float("inf"))
    low_limit = limit_api.GetLowAttr().Get()
    high_limit = limit_api.GetHighAttr().Get()
    return (low_limit, high_limit)


def set_joint_limit_range(joint_prim: Usd.Prim, axis_name: str, lower_limit: float, upper_limit: float):
    """Set the joint limit range for a given physics joint prim and axis.

    Args:
        joint_prim (Usd.Prim): The physics joint prim.
        axis_name (str): The name of the axis (e.g., "transX", "rotY").
        lower_limit (float): The lower limit value.
        upper_limit (float): The upper limit value.

    Raises:
        ValueError: If the joint prim is not valid or if the lower limit is greater than the upper limit.
    """
    if not joint_prim.IsValid():
        raise ValueError("Invalid joint prim.")
    if lower_limit > upper_limit:
        raise ValueError("Lower limit must be less than or equal to the upper limit.")
    limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, axis_name)
    lower_attr = limit_api.CreateLowAttr()
    lower_attr.Set(lower_limit)
    upper_attr = limit_api.CreateHighAttr()
    upper_attr.Set(upper_limit)


def validate_joint_limits(joint_prim: Usd.Prim) -> bool:
    """
    Validate the limits of a joint prim.

    Args:
        joint_prim (Usd.Prim): The joint prim to validate.

    Returns:
        bool: True if the joint limits are valid, False otherwise.
    """
    limit_api = UsdPhysics.LimitAPI.Get(joint_prim, "limit")
    if not limit_api:
        return True
    low_attr = limit_api.GetLowAttr()
    high_attr = limit_api.GetHighAttr()
    if not low_attr.IsValid() or not high_attr.IsValid():
        return False
    low_limit = low_attr.Get()
    high_limit = high_attr.Get()
    if low_limit > high_limit:
        return False
    return True


def apply_limits_to_multiple_joints(
    stage: Usd.Stage, joint_paths: List[str], low_limits: List[float], high_limits: List[float], axes: List[str]
) -> None:
    """Apply low and high limits to multiple joint prims along specified axes.

    Args:
        stage (Usd.Stage): The USD stage.
        joint_paths (List[str]): List of paths to joint prims.
        low_limits (List[float]): List of low limit values.
        high_limits (List[float]): List of high limit values.
        axes (List[str]): List of axes to apply limits to (e.g., 'transX', 'rotY').

    Raises:
        ValueError: If the lengths of joint_paths, low_limits, high_limits, and axes do not match.
    """
    if not len(joint_paths) == len(low_limits) == len(high_limits) == len(axes):
        raise ValueError("The lengths of joint_paths, low_limits, high_limits, and axes must match.")
    for joint_path, low_limit, high_limit, axis in zip(joint_paths, low_limits, high_limits, axes):
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid():
            continue
        limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
        limit_api.GetLowAttr().Set(low_limit)
        limit_api.GetHighAttr().Set(high_limit)


def get_limits_for_joint(joint_prim: Usd.Prim) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Get the limit values for a given joint prim.

    Args:
        joint_prim (Usd.Prim): The joint prim to get limits for.

    Returns:
        Optional[Dict[str, Dict[str, float]]]: A dictionary mapping limit types to their low and high values,
            or None if no limits are set or the prim is not a valid joint.
    """
    if not UsdPhysics.Joint(joint_prim):
        return None
    limit_apis = UsdPhysics.LimitAPI.GetAll(joint_prim)
    if not limit_apis:
        return None
    limits = {}
    for limit_api in limit_apis:
        limit_type = limit_api.GetPrim().GetName()
        low_attr = limit_api.GetLowAttr()
        high_attr = limit_api.GetHighAttr()
        if low_attr.HasValue() and high_attr.HasValue():
            low_value = low_attr.Get()
            high_value = high_attr.Get()
            limits[limit_type] = {"low": low_value, "high": high_value}
    return limits if limits else None


def batch_update_joint_limits(stage: Usd.Stage, joint_paths: List[str], limits: List[Tuple[str, float, float]]) -> None:
    """
    Batch update joint limits for multiple joints.

    Args:
        stage (Usd.Stage): The USD stage.
        joint_paths (List[str]): List of paths to joint prims.
        limits (List[Tuple[str, float, float]]): List of tuples specifying limit updates.
            Each tuple contains (limit_type, lower, upper).

    Raises:
        ValueError: If the number of joint paths and limit tuples don't match.
    """
    if len(joint_paths) != len(limits):
        raise ValueError("Number of joint paths and limit tuples must match.")
    for joint_path, limit_tuple in zip(joint_paths, limits):
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid():
            continue
        limit_type = limit_tuple[0]
        limit_api = UsdPhysics.LimitAPI.Get(joint_prim, limit_type)
        if not limit_api:
            limit_api = UsdPhysics.LimitAPI.Apply(joint_prim, limit_type)
        limit_api.GetLowAttr().Set(limit_tuple[1])
        limit_api.GetHighAttr().Set(limit_tuple[2])


def get_mass_properties(prim: Usd.Prim) -> dict:
    """Get the mass properties for a prim with a UsdPhysics.MassAPI applied."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    mass_api = UsdPhysics.MassAPI(prim)
    if not mass_api:
        raise ValueError(f"Prim {prim.GetPath()} does not have a UsdPhysics.MassAPI applied.")
    properties = {}
    mass_attr = mass_api.GetMassAttr()
    if mass_attr.HasAuthoredValue():
        properties["mass"] = mass_attr.Get()
    else:
        properties["mass"] = 0.0
    density_attr = mass_api.GetDensityAttr()
    if density_attr.HasAuthoredValue():
        properties["density"] = density_attr.Get()
    else:
        properties["density"] = 0.0
    center_of_mass_attr = mass_api.GetCenterOfMassAttr()
    if center_of_mass_attr.HasAuthoredValue():
        properties["center_of_mass"] = center_of_mass_attr.Get()
    else:
        properties["center_of_mass"] = Gf.Vec3f(0.0, 0.0, 0.0)
    diagonalized_inertia_attr = mass_api.GetDiagonalInertiaAttr()
    if diagonalized_inertia_attr.HasAuthoredValue():
        properties["diagonalized_inertia"] = diagonalized_inertia_attr.Get()
    else:
        properties["diagonalized_inertia"] = Gf.Vec3f(0.0, 0.0, 0.0)
    principal_axes_attr = mass_api.GetPrincipalAxesAttr()
    if principal_axes_attr.HasAuthoredValue():
        properties["principal_axes"] = principal_axes_attr.Get()
    else:
        properties["principal_axes"] = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    return properties


def get_prims_with_mass_api(stage: Usd.Stage) -> list[Usd.Prim]:
    """
    Get all prims on the stage that have a UsdPhysics.MassAPI applied.

    Args:
        stage (Usd.Stage): The USD stage to search for prims with MassAPI.

    Returns:
        list[Usd.Prim]: A list of prims that have a MassAPI applied.
    """
    prims_with_mass_api = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MassAPI):
            prims_with_mass_api.append(prim)
    return prims_with_mass_api


def find_prims_by_mass_range(stage: Usd.Stage, mass_min: float, mass_max: float) -> List[Usd.Prim]:
    """
    Find all prims on the stage that have a physics:mass attribute value within the specified range.

    Args:
        stage (Usd.Stage): The USD stage to search.
        mass_min (float): The minimum mass value (inclusive).
        mass_max (float): The maximum mass value (inclusive).

    Returns:
        List[Usd.Prim]: A list of prims with mass values within the specified range.
    """
    prims: List[Usd.Prim] = []
    for prim in stage.Traverse():
        mass_api = UsdPhysics.MassAPI(prim)
        if mass_api and prim.HasAPI(UsdPhysics.MassAPI):
            mass_attr = mass_api.GetMassAttr()
            if mass_attr:
                mass = mass_attr.Get()
                if mass_min <= mass <= mass_max:
                    prims.append(prim)
    return prims


def apply_mass_api_to_prims(stage: Usd.Stage, prim_paths: List[str], density: float = 1.0, mass: float = 0.0) -> None:
    """Apply the massAPI to each prim and set the density and mass attributes."""
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        if not UsdPhysics.MassAPI.CanApply(prim):
            print(f"Warning: MassAPI cannot be applied to prim at path {prim_path}. Skipping.")
            continue
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        if density != 0.0:
            mass_api.CreateDensityAttr(density)
        if mass != 0.0:
            mass_api.CreateMassAttr(mass)


def set_mass_properties(
    prim: Usd.Prim,
    mass: float,
    density: float,
    center_of_mass: Gf.Vec3f,
    diag_inertia: Gf.Vec3f,
    principal_axes: Gf.Quatf,
) -> None:
    """Set the mass properties for a prim using the UsdPhysics.MassAPI schema.

    Args:
        prim (Usd.Prim): The prim to set the mass properties for.
        mass (float): The mass value to set. If 0.0, it is ignored.
        density (float): The density value to set. If 0.0, it is ignored.
        center_of_mass (Gf.Vec3f): The center of mass to set.
        diag_inertia (Gf.Vec3f): The diagonal inertia tensor to set. If (0.0, 0.0, 0.0), it is ignored.
        principal_axes (Gf.Quatf): The orientation of the inertia tensor's principal axes.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    if mass != 0.0:
        mass_attr = mass_api.CreateMassAttr(mass)
        mass_attr.Set(mass)
    if density != 0.0:
        density_attr = mass_api.CreateDensityAttr(density)
        density_attr.Set(density)
    center_of_mass_attr = mass_api.CreateCenterOfMassAttr(center_of_mass)
    center_of_mass_attr.Set(center_of_mass)
    if diag_inertia != Gf.Vec3f(0.0):
        diag_inertia_attr = mass_api.CreateDiagonalInertiaAttr(diag_inertia)
        diag_inertia_attr.Set(diag_inertia)
    principal_axes_attr = mass_api.CreatePrincipalAxesAttr(principal_axes)
    principal_axes_attr.Set(principal_axes)


def copy_mass_properties(source_prim: Usd.Prim, dest_prim: Usd.Prim):
    """Copy mass properties from one prim to another."""
    if not UsdPhysics.MassAPI.CanApply(source_prim):
        raise ValueError("Source prim does not have a mass API.")
    source_mass_api = UsdPhysics.MassAPI(source_prim)
    if not UsdPhysics.MassAPI.CanApply(dest_prim):
        raise ValueError("Destination prim cannot have a mass API.")
    dest_mass_api = UsdPhysics.MassAPI.Apply(dest_prim)
    if source_mass_api.GetMassAttr().HasAuthoredValue():
        mass = source_mass_api.GetMassAttr().Get()
        dest_mass_api.CreateMassAttr(mass)
    if source_mass_api.GetDensityAttr().HasAuthoredValue():
        density = source_mass_api.GetDensityAttr().Get()
        dest_mass_api.CreateDensityAttr(density)
    if source_mass_api.GetCenterOfMassAttr().HasAuthoredValue():
        center_of_mass = source_mass_api.GetCenterOfMassAttr().Get()
        dest_mass_api.CreateCenterOfMassAttr(center_of_mass)
    if source_mass_api.GetDiagonalInertiaAttr().HasAuthoredValue():
        diagonal_inertia = source_mass_api.GetDiagonalInertiaAttr().Get()
        dest_mass_api.CreateDiagonalInertiaAttr(diagonal_inertia)
    if source_mass_api.GetPrincipalAxesAttr().HasAuthoredValue():
        principal_axes = source_mass_api.GetPrincipalAxesAttr().Get()
        dest_mass_api.CreatePrincipalAxesAttr(principal_axes)


def create_mass_api_with_defaults(
    prim: Usd.Prim,
    mass: float = 1.0,
    density: float = 1.0,
    center_of_mass: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    diagonalized_inertia: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    principal_axes: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> UsdPhysics.MassAPI:
    """Create a MassAPI on the given prim with default values.

    Args:
        prim (Usd.Prim): The prim to create the MassAPI on.
        mass (float, optional): The mass value. Defaults to 1.0.
        density (float, optional): The density value. Defaults to 1.0.
        center_of_mass (Tuple[float, float, float], optional): The center of mass. Defaults to (0.0, 0.0, 0.0).
        diagonalized_inertia (Tuple[float, float, float], optional): The diagonalized inertia. Defaults to (1.0, 1.0, 1.0).
        principal_axes (Tuple[float, float, float, float], optional): The principal axes. Defaults to (1.0, 0.0, 0.0, 0.0).

    Returns:
        UsdPhysics.MassAPI: The created MassAPI object.

    Raises:
        ValueError: If the prim is not valid or if the MassAPI cannot be applied to the prim.
    """
    if not prim.IsValid():
        raise ValueError("The given prim is not valid.")
    if not UsdPhysics.MassAPI.CanApply(prim):
        raise ValueError("The MassAPI cannot be applied to the given prim.")
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_attr = mass_api.CreateMassAttr(mass, writeSparsely=True)
    mass_attr.Set(mass)
    density_attr = mass_api.CreateDensityAttr(density, writeSparsely=True)
    density_attr.Set(density)
    center_of_mass_attr = mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*center_of_mass), writeSparsely=True)
    center_of_mass_attr.Set(Gf.Vec3f(*center_of_mass))
    diagonalized_inertia_attr = mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonalized_inertia), writeSparsely=True)
    diagonalized_inertia_attr.Set(Gf.Vec3f(*diagonalized_inertia))
    principal_axes_attr = mass_api.CreatePrincipalAxesAttr(Gf.Quatf(*principal_axes), writeSparsely=True)
    principal_axes_attr.Set(Gf.Quatf(*principal_axes))
    return mass_api


def batch_update_mass_properties(
    prims: Sequence[Usd.Prim],
    masses: Sequence[float],
    densities: Sequence[float],
    centers_of_mass: Sequence[Gf.Vec3f],
    inertias: Sequence[Gf.Vec3f],
    principal_axes: Sequence[Gf.Quatf],
) -> int:
    """
    Batch update mass properties on a sequence of prims.

    Args:
        prims (Sequence[Usd.Prim]): The prims to update mass properties on.
        masses (Sequence[float]): The masses to set on the prims.
        densities (Sequence[float]): The densities to set on the prims.
        centers_of_mass (Sequence[Gf.Vec3f]): The centers of mass to set on the prims.
        inertias (Sequence[Gf.Vec3f]): The inertias to set on the prims.
        principal_axes (Sequence[Gf.Quatf]): The principal axes to set on the prims.

    Returns:
        int: The number of prims successfully updated.

    Raises:
        ValueError: If the input sequences are not the same length as the prims sequence.
    """
    if not len(prims) == len(masses) == len(densities) == len(centers_of_mass) == len(inertias) == len(principal_axes):
        raise ValueError("All input sequences must be the same length as the prims sequence.")
    num_updated = 0
    for prim, mass, density, center_of_mass, inertia, principal_axis in zip(
        prims, masses, densities, centers_of_mass, inertias, principal_axes
    ):
        if not UsdPhysics.MassAPI.Get(prim.GetStage(), prim.GetPath()):
            UsdPhysics.MassAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI(prim)
        mass_api.GetMassAttr().Set(mass)
        mass_api.GetDensityAttr().Set(density)
        mass_api.GetCenterOfMassAttr().Set(center_of_mass)
        mass_api.GetDiagonalInertiaAttr().Set(inertia)
        mass_api.GetPrincipalAxesAttr().Set(principal_axis)
        num_updated += 1
    return num_updated


def validate_mass_properties(mass_api: UsdPhysics.MassAPI) -> bool:
    """Validate the mass properties of a MassAPI prim.

    The function checks the following conditions:
    1. At least one of mass, density, or inertia is authored.
    2. If both mass and density are authored, they must be consistent.
    3. If inertia is authored, the principal axes must also be authored.

    Args:
        mass_api (UsdPhysics.MassAPI): The MassAPI prim to validate.

    Returns:
        bool: True if the mass properties are valid, False otherwise.
    """
    has_mass = mass_api.GetMassAttr().HasAuthoredValue()
    has_density = mass_api.GetDensityAttr().HasAuthoredValue()
    has_inertia = mass_api.GetDiagonalInertiaAttr().HasAuthoredValue()
    if not (has_mass or has_density or has_inertia):
        return False
    if has_mass and has_density:
        mass = mass_api.GetMassAttr().Get()
        density = mass_api.GetDensityAttr().Get()
        volume = mass / density if density != 0.0 else 0.0
        if not Gf.IsClose(mass, density * volume, 1e-06):
            return False
    if has_inertia:
        has_axes = mass_api.GetPrincipalAxesAttr().HasAuthoredValue()
        if not has_axes:
            return False
    return True


def scale_mass_properties(mass_api: UsdPhysics.MassAPI, scale: Gf.Vec3f) -> None:
    """Scale the mass properties of a MassAPI prim by a given scale factor.

    Args:
        mass_api (UsdPhysics.MassAPI): The MassAPI prim to scale the properties of.
        scale (Gf.Vec3f): The scale factor to apply to the mass properties.

    Raises:
        ValueError: If the scale factor contains zero or negative values.
    """
    if any((x <= 0 for x in scale)):
        raise ValueError("Scale factor must be positive and non-zero")
    mass = mass_api.GetMassAttr().Get()
    density = mass_api.GetDensityAttr().Get()
    inertia = mass_api.GetDiagonalInertiaAttr().Get()
    center_of_mass = mass_api.GetCenterOfMassAttr().Get()
    if mass is not None:
        mass_api.GetMassAttr().Set(mass * scale[0] * scale[1] * scale[2])
    if density is not None:
        mass_api.GetDensityAttr().Set(density / (scale[0] * scale[1] * scale[2]))
    if inertia is not None:
        scaled_inertia = Gf.Vec3f(
            inertia[0] * (scale[1] * scale[1] + scale[2] * scale[2]),
            inertia[1] * (scale[0] * scale[0] + scale[2] * scale[2]),
            inertia[2] * (scale[0] * scale[0] + scale[1] * scale[1]),
        )
        mass_api.GetDiagonalInertiaAttr().Set(scaled_inertia)
    if center_of_mass is not None:
        scaled_center_of_mass = Gf.Vec3f(
            center_of_mass[0] * scale[0], center_of_mass[1] * scale[1], center_of_mass[2] * scale[2]
        )
        mass_api.GetCenterOfMassAttr().Set(scaled_center_of_mass)


def compute_combined_mass_properties(mass_apis: List[UsdPhysics.MassAPI]) -> Tuple[float, Gf.Vec3f, Gf.Matrix3f]:
    """Compute combined mass properties from a list of MassAPI objects.

    Args:
        mass_apis (List[UsdPhysics.MassAPI]): List of MassAPI objects.

    Returns:
        Tuple[float, Gf.Vec3f, Gf.Matrix3f]: Combined mass, center of mass, and inertia tensor.
    """
    total_mass = 0.0
    total_com = Gf.Vec3f(0)
    total_inertia_tensor = Gf.Matrix3f(0)
    for mass_api in mass_apis:
        mass = mass_api.GetMassAttr().Get()
        com = mass_api.GetCenterOfMassAttr().Get()
        inertia_tensor = Gf.Matrix3f(mass_api.GetDiagonalInertiaAttr().Get())
        if mass <= 0:
            continue
        total_mass += mass
        total_com += mass * com
        r = com - total_com
        total_inertia_tensor += inertia_tensor + mass * (r.GetLength() ** 2 * Gf.Matrix3f(1) - Gf.Matrix3f(r * r))
    if total_mass > 0:
        total_com /= total_mass
    return (total_mass, total_com, total_inertia_tensor)


def reset_mass_properties_to_defaults(mass_api: UsdPhysics.MassAPI):
    """Reset the mass properties of a MassAPI to their default values."""
    mass_attr = mass_api.GetMassAttr()
    density_attr = mass_api.GetDensityAttr()
    inertia_attr = mass_api.GetDiagonalInertiaAttr()
    com_attr = mass_api.GetCenterOfMassAttr()
    axes_attr = mass_api.GetPrincipalAxesAttr()
    if mass_attr.IsValid():
        mass_attr.Clear()
    if density_attr.IsValid():
        density_attr.Clear()
    if inertia_attr.IsValid():
        inertia_attr.Clear()
    if com_attr.IsValid():
        com_attr.Clear()
    if axes_attr.IsValid():
        axes_attr.Clear()


def transfer_mass_properties(src_prim: Usd.Prim, dest_prim: Usd.Prim):
    """Transfer mass properties from one prim to another."""
    src_mass_api = UsdPhysics.MassAPI(src_prim)
    if not src_mass_api:
        raise ValueError(f"Source prim {src_prim.GetPath()} does not have a valid MassAPI.")
    if not UsdPhysics.MassAPI.CanApply(dest_prim):
        raise ValueError(f"Destination prim {dest_prim.GetPath()} cannot have MassAPI applied.")
    dest_mass_api = UsdPhysics.MassAPI.Apply(dest_prim)
    mass_attr = src_mass_api.GetMassAttr()
    if mass_attr.IsValid():
        mass = mass_attr.Get()
        dest_mass_api.CreateMassAttr(mass)
    density_attr = src_mass_api.GetDensityAttr()
    if density_attr.IsValid():
        density = density_attr.Get()
        dest_mass_api.CreateDensityAttr(density)
    center_of_mass_attr = src_mass_api.GetCenterOfMassAttr()
    if center_of_mass_attr.IsValid():
        center_of_mass = center_of_mass_attr.Get()
        dest_mass_api.CreateCenterOfMassAttr(center_of_mass)
    diag_inertia_attr = src_mass_api.GetDiagonalInertiaAttr()
    if diag_inertia_attr.IsValid():
        diag_inertia = diag_inertia_attr.Get()
        dest_mass_api.CreateDiagonalInertiaAttr(diag_inertia)
    principal_axes_attr = src_mass_api.GetPrincipalAxesAttr()
    if principal_axes_attr.IsValid():
        principal_axes = principal_axes_attr.Get()
        dest_mass_api.CreatePrincipalAxesAttr(principal_axes)


def remove_mass_api_from_prims(stage: Usd.Stage, prim_paths: List[str]) -> None:
    """Remove the UsdPhysics.MassAPI from the specified prims."""
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        if not prim.HasAPI(UsdPhysics.MassAPI):
            print(f"Warning: Prim at path {prim_path} does not have MassAPI applied. Skipping.")
            continue
        prim.RemoveAPI(UsdPhysics.MassAPI)
        print(f"Removed MassAPI from prim at path {prim_path}")


def create_mass_api_for_hierarchy(prim: Usd.Prim, mass: float, density: float) -> None:
    """Apply MassAPI to the prim and its children.

    Args:
        prim (Usd.Prim): The root prim of the hierarchy.
        mass (float): The mass value to set on the MassAPI. If 0, it will be ignored.
        density (float): The density value to set on the MassAPI. If 0, it will be ignored.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    if mass != 0:
        mass_attr = mass_api.CreateMassAttr()
        mass_attr.Set(mass)
    if density != 0:
        density_attr = mass_api.CreateDensityAttr()
        density_attr.Set(density)
    for child_prim in prim.GetAllChildren():
        create_mass_api_for_hierarchy(child_prim, mass, density)


def compute_volume(prim: Usd.Prim) -> float:
    """Compute the volume of a prim.

    Args:
        prim (Usd.Prim): The prim to compute the volume for.

    Returns:
        float: The volume of the prim.
    """
    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        return 0.0
    return mesh.ComputeVolume()


def sync_mass_properties_across_hierarchy(prim: Usd.Prim) -> None:
    """Synchronize mass properties across the hierarchy.

    If a prim has a mass, it will be used. Otherwise, the mass will be
    computed from the density and volume of the prim and its children.

    Args:
        prim (Usd.Prim): The prim to synchronize mass properties for.
    """
    mass_api = UsdPhysics.MassAPI(prim)
    if mass_api.GetMassAttr().HasAuthoredValue():
        return
    density = 0.0
    volume = 0.0
    if mass_api.GetDensityAttr().HasAuthoredValue():
        density = mass_api.GetDensityAttr().Get()
    volume = compute_volume(prim)
    for child in prim.GetChildren():
        child_mass_api = UsdPhysics.MassAPI(child)
        if child_mass_api.GetMassAttr().HasAuthoredValue():
            continue
        if child_mass_api.GetDensityAttr().HasAuthoredValue():
            density += child_mass_api.GetDensityAttr().Get()
        child_volume = compute_volume(child)
        volume += child_volume
    mass = density * volume
    mass_api.GetMassAttr().Set(mass)


def convert_mass_units(value: float, from_unit: UsdPhysics.MassUnits, to_unit: UsdPhysics.MassUnits) -> float:
    """Convert a mass value from one unit to another.

    Args:
        value (float): The mass value to convert.
        from_unit (UsdPhysics.MassUnits): The unit of the input value.
        to_unit (UsdPhysics.MassUnits): The unit to convert to.

    Returns:
        float: The converted mass value in the specified unit.
    """
    value_in_kg = value * from_unit
    converted_value = value_in_kg / to_unit
    return converted_value


def get_mass_in_unit(mass_in_kg: Union[int, float], unit: str) -> Union[int, float]:
    """Convert mass from kilograms to the specified unit.

    Args:
        mass_in_kg (Union[int, float]): Mass value in kilograms.
        unit (str): The unit to convert to. Must be a valid attribute of UsdPhysics.MassUnits.

    Returns:
        Union[int, float]: The mass value in the specified unit.

    Raises:
        AttributeError: If the specified unit is not a valid attribute of UsdPhysics.MassUnits.
        TypeError: If the mass_in_kg is not a numeric value.
    """
    if not hasattr(UsdPhysics.MassUnits, unit):
        raise AttributeError(f"Invalid mass unit: {unit}")
    conversion_factor = getattr(UsdPhysics.MassUnits, unit)
    if not isinstance(mass_in_kg, (int, float)):
        raise TypeError(f"Invalid mass value: {mass_in_kg}")
    mass_in_unit = mass_in_kg / conversion_factor
    return mass_in_unit


def set_mass_with_unit(prim: Usd.Prim, mass_value: float, mass_unit: UsdPhysics.MassUnits) -> None:
    """Set the mass of a prim using a specific mass unit.

    Args:
        prim (Usd.Prim): The prim to set the mass on.
        mass_value (float): The mass value to set.
        mass_unit (UsdPhysics.MassUnits): The mass unit to use.

    Raises:
        ValueError: If the prim is not valid or does not have a physics mass API applied.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    physics_mass_api = UsdPhysics.MassAPI(prim)
    if not physics_mass_api:
        physics_mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_in_kg = mass_value * mass_unit
    physics_mass_api.CreateMassAttr().Set(mass_in_kg)


def normalize_mass_units(mass: Union[float, int], unit: str) -> float:
    """Normalize a mass value to kilograms based on the provided unit."""
    unit_conversion_factors = {
        "kilograms": UsdPhysics.MassUnits.kilograms,
        "grams": UsdPhysics.MassUnits.grams,
        "slugs": UsdPhysics.MassUnits.slugs,
    }
    if unit not in unit_conversion_factors:
        raise ValueError(f"Invalid mass unit: {unit}")
    conversion_factor = unit_conversion_factors[unit]
    if mass < 0:
        raise ValueError(f"Invalid mass value: {mass}. Mass must be non-negative.")
    normalized_mass = mass * conversion_factor
    return normalized_mass


def apply_physics_material_api(prim: Usd.Prim) -> UsdPhysics.MaterialAPI:
    """Apply the PhysicsMaterialAPI to a prim.

    Args:
        prim (Usd.Prim): The prim to apply the PhysicsMaterialAPI to.

    Returns:
        UsdPhysics.MaterialAPI: The applied PhysicsMaterialAPI object.

    Raises:
        ValueError: If the prim is not valid or cannot have the PhysicsMaterialAPI applied.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if not UsdPhysics.MaterialAPI.CanApply(prim):
        raise ValueError("PhysicsMaterialAPI cannot be applied to the prim.")
    return UsdPhysics.MaterialAPI.Apply(prim)


def get_material_properties(prim: Usd.Prim) -> dict:
    """Get the material properties for a prim with UsdPhysicsMaterialAPI applied."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    if not UsdPhysics.MaterialAPI.Apply(prim):
        raise ValueError(f"Prim {prim.GetPath()} does not have UsdPhysicsMaterialAPI applied.")
    material_api = UsdPhysics.MaterialAPI(prim)
    density = material_api.GetDensityAttr().Get()
    dynamic_friction = material_api.GetDynamicFrictionAttr().Get()
    restitution = material_api.GetRestitutionAttr().Get()
    static_friction = material_api.GetStaticFrictionAttr().Get()
    return {
        "density": density if density else 0.0,
        "dynamic_friction": dynamic_friction if dynamic_friction else 0.0,
        "restitution": restitution if restitution else 0.0,
        "static_friction": static_friction if static_friction else 0.0,
    }


def set_material_density(material_prim: Usd.Prim, density: float) -> None:
    """Set the density for a material prim.

    Args:
        material_prim (Usd.Prim): The material prim to set the density for.
        density (float): The density value to set.

    Raises:
        ValueError: If the input prim is not a valid material prim.
    """
    if not material_prim.IsValid():
        raise ValueError(f"Invalid prim: {material_prim}")
    if not UsdShade.Material(material_prim):
        raise ValueError(f"Prim {material_prim.GetPath()} is not a material prim")
    material_api = UsdPhysics.MaterialAPI(material_prim)
    density_attr = material_api.GetDensityAttr()
    if not density_attr:
        density_attr = material_api.CreateDensityAttr(density, writeSparsely=True)
    density_attr.Set(density)


def set_material_restitution(material: UsdPhysics.MaterialAPI, restitution: float) -> None:
    """Set the restitution coefficient for a physics material.

    Args:
        material (UsdPhysics.MaterialAPI): The physics material to modify.
        restitution (float): The restitution coefficient to set (unitless).

    Raises:
        ValueError: If the provided material is invalid.
    """
    if not material:
        raise ValueError("Invalid material provided.")
    restitution_attr = material.GetRestitutionAttr()
    if not restitution_attr:
        restitution_attr = material.CreateRestitutionAttr(restitution, writeSparsely=True)
    restitution_attr.Set(restitution)


def copy_material_properties(source_material: UsdShade.Material, dest_material: UsdShade.Material):
    """Copy material properties from source to destination material."""
    source_api = UsdPhysics.MaterialAPI.Get(source_material.GetPrim().GetStage(), source_material.GetPath())
    dest_api = UsdPhysics.MaterialAPI.Get(dest_material.GetPrim().GetStage(), dest_material.GetPath())
    if not source_api.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        raise ValueError("Source material does not have UsdPhysics.MaterialAPI applied.")
    if not dest_api.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        UsdPhysics.MaterialAPI.Apply(dest_material.GetPrim())
        dest_api = UsdPhysics.MaterialAPI.Get(dest_material.GetPrim().GetStage(), dest_material.GetPath())
    attributes_to_copy = ["density", "dynamicFriction", "restitution", "staticFriction"]
    for attr_name in attributes_to_copy:
        source_attr = source_api.GetPrim().GetAttribute(f"physics:{attr_name}")
        if source_attr.IsValid():
            value = source_attr.Get()
            dest_attr = dest_api.GetPrim().CreateAttribute(f"physics:{attr_name}", source_attr.GetTypeName())
            dest_attr.Set(value)


def create_physics_material(
    stage: Usd.Stage,
    material_path: str,
    density: float = 0.0,
    dynamic_friction: float = 0.0,
    restitution: float = 0.0,
    static_friction: float = 0.0,
) -> UsdPhysics.MaterialAPI:
    """Create a physics material on the given stage with specified properties."""
    if not stage:
        raise ValueError("Invalid stage.")
    material = stage.GetPrimAtPath(material_path)
    if not material:
        material = UsdShade.Material.Define(stage, material_path).GetPrim()
    physics_material = UsdPhysics.MaterialAPI.Apply(material)
    if density > 0.0:
        physics_material.CreateDensityAttr(density)
    if dynamic_friction >= 0.0:
        physics_material.CreateDynamicFrictionAttr(dynamic_friction)
    if restitution >= 0.0:
        physics_material.CreateRestitutionAttr(restitution)
    if static_friction >= 0.0:
        physics_material.CreateStaticFrictionAttr(static_friction)
    return physics_material


def validate_material_properties(material: UsdPhysics.MaterialAPI) -> bool:
    """Validate the properties of a physics material."""
    if not material:
        raise ValueError("Invalid material provided.")
    density = material.GetDensityAttr().Get()
    dynamic_friction = material.GetDynamicFrictionAttr().Get()
    restitution = material.GetRestitutionAttr().Get()
    static_friction = material.GetStaticFrictionAttr().Get()
    if density is not None and density < 0:
        return False
    if dynamic_friction is not None and (not 0 <= dynamic_friction <= 1):
        return False
    if restitution is not None and (not 0 <= restitution <= 1):
        return False
    if static_friction is not None and static_friction < 0:
        return False
    return True


def batch_apply_material_api(
    stage: Usd.Stage,
    prim_paths: List[str],
    density: float,
    dynamic_friction: float,
    restitution: float,
    static_friction: float,
) -> List[UsdPhysics.MaterialAPI]:
    """
    Apply UsdPhysics.MaterialAPI to a list of prims and set the material properties.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths to apply the MaterialAPI to.
        density (float): The density value to set.
        dynamic_friction (float): The dynamic friction value to set.
        restitution (float): The restitution value to set.
        static_friction (float): The static friction value to set.

    Returns:
        List[UsdPhysics.MaterialAPI]: A list of MaterialAPI objects for the prims.

    Raises:
        ValueError: If any of the prim paths are invalid or if the MaterialAPI cannot be applied.
    """
    material_apis = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")
        if not UsdPhysics.MaterialAPI.CanApply(prim):
            raise ValueError(f"Cannot apply MaterialAPI to prim: {prim_path}")
        material_api = UsdPhysics.MaterialAPI.Apply(prim)
        material_api.CreateDensityAttr(density)
        material_api.CreateDynamicFrictionAttr(dynamic_friction)
        material_api.CreateRestitutionAttr(restitution)
        material_api.CreateStaticFrictionAttr(static_friction)
        material_apis.append(material_api)
    return material_apis


def export_material_properties(stage: Usd.Stage, prim_path: str) -> Dict[str, float]:
    """Export the material properties for a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    material_api = UsdPhysics.MaterialAPI(prim)
    if not material_api:
        raise ValueError(f"Prim at path {prim_path} does not have the MaterialAPI applied.")
    material_properties = {}
    density_attr = material_api.GetDensityAttr()
    if density_attr.HasAuthoredValue():
        material_properties["density"] = density_attr.Get()
    dynamic_friction_attr = material_api.GetDynamicFrictionAttr()
    if dynamic_friction_attr.HasAuthoredValue():
        material_properties["dynamicFriction"] = dynamic_friction_attr.Get()
    restitution_attr = material_api.GetRestitutionAttr()
    if restitution_attr.HasAuthoredValue():
        material_properties["restitution"] = restitution_attr.Get()
    static_friction_attr = material_api.GetStaticFrictionAttr()
    if static_friction_attr.HasAuthoredValue():
        material_properties["staticFriction"] = static_friction_attr.Get()
    return material_properties


def set_material_friction(material_prim: Usd.Prim, static_friction: float, dynamic_friction: float) -> None:
    """Set the static and dynamic friction coefficients on a material prim.

    Args:
        material_prim (Usd.Prim): The material prim to set the friction coefficients on.
        static_friction (float): The static friction coefficient (unitless).
        dynamic_friction (float): The dynamic friction coefficient (unitless).

    Raises:
        ValueError: If the provided prim is not a valid material prim.
    """
    if not material_prim.IsValid() or not material_prim.IsA(UsdShade.Material):
        raise ValueError(f"Prim '{material_prim.GetPath()}' is not a valid material prim.")
    physics_material_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    static_friction_attr = physics_material_api.CreateStaticFrictionAttr(static_friction, writeSparsely=True)
    static_friction_attr.Set(static_friction)
    dynamic_friction_attr = physics_material_api.CreateDynamicFrictionAttr(dynamic_friction, writeSparsely=True)
    dynamic_friction_attr.Set(dynamic_friction)


def find_prims_with_material_api(stage: Usd.Stage) -> List[Usd.Prim]:
    """
    Find all prims on the given stage that have the UsdPhysics.MaterialAPI applied.

    Args:
        stage (Usd.Stage): The stage to search for prims with the MaterialAPI.

    Returns:
        List[Usd.Prim]: A list of prims that have the UsdPhysics.MaterialAPI applied.
    """
    prims_with_material_api = []
    for prim in stage.TraverseAll():
        if UsdPhysics.MaterialAPI.Get(stage, prim.GetPath()):
            prims_with_material_api.append(prim)
    return prims_with_material_api


def blend_material_properties(
    material_a: UsdPhysics.MaterialAPI, material_b: UsdPhysics.MaterialAPI, amount: float
) -> UsdPhysics.MaterialAPI:
    """Blends two UsdPhysics.MaterialAPI objects and returns a new MaterialAPI object with the blended properties.

    Args:
        material_a (UsdPhysics.MaterialAPI): The first material to blend.
        material_b (UsdPhysics.MaterialAPI): The second material to blend.
        amount (float): The blending factor between 0 and 1. 0 means fully material_a, 1 means fully material_b.

    Returns:
        UsdPhysics.MaterialAPI: A new MaterialAPI object with the blended properties.
    """
    if not material_a or not material_a.GetPrim().IsValid():
        raise ValueError("material_a is not a valid UsdPhysics.MaterialAPI object.")
    if not material_b or not material_b.GetPrim().IsValid():
        raise ValueError("material_b is not a valid UsdPhysics.MaterialAPI object.")
    if amount < 0 or amount > 1:
        raise ValueError("amount must be between 0 and 1.")
    stage = material_a.GetPrim().GetStage()
    blended_material_path = "/BlendedMaterial"
    blended_material_prim = stage.DefinePrim(blended_material_path, "Material")
    blended_material = UsdPhysics.MaterialAPI.Apply(blended_material_prim)
    density_a = material_a.GetDensityAttr().Get()
    density_b = material_b.GetDensityAttr().Get()
    if density_a is not None and density_b is not None:
        blended_density = density_a * (1 - amount) + density_b * amount
        blended_material.CreateDensityAttr(blended_density)
    dynamic_friction_a = material_a.GetDynamicFrictionAttr().Get()
    dynamic_friction_b = material_b.GetDynamicFrictionAttr().Get()
    if dynamic_friction_a is not None and dynamic_friction_b is not None:
        blended_dynamic_friction = dynamic_friction_a * (1 - amount) + dynamic_friction_b * amount
        blended_material.CreateDynamicFrictionAttr(blended_dynamic_friction)
    restitution_a = material_a.GetRestitutionAttr().Get()
    restitution_b = material_b.GetRestitutionAttr().Get()
    if restitution_a is not None and restitution_b is not None:
        blended_restitution = restitution_a * (1 - amount) + restitution_b * amount
        blended_material.CreateRestitutionAttr(blended_restitution)
    static_friction_a = material_a.GetStaticFrictionAttr().Get()
    static_friction_b = material_b.GetStaticFrictionAttr().Get()
    if static_friction_a is not None and static_friction_b is not None:
        blended_static_friction = static_friction_a * (1 - amount) + static_friction_b * amount
        blended_material.CreateStaticFrictionAttr(blended_static_friction)
    return blended_material


def remove_material_api(prim: Usd.Prim) -> bool:
    """Remove the PhysicsMaterialAPI from the given prim."""
    if not prim.IsValid():
        raise ValueError("Invalid prim")
    if not UsdPhysics.MaterialAPI.Get(prim.GetStage(), prim.GetPath()):
        print(f"Prim {prim.GetPath()} does not have PhysicsMaterialAPI applied")
        return False
    success = prim.RemoveAPI(UsdPhysics.MaterialAPI)
    if not success:
        print(f"Failed to remove PhysicsMaterialAPI from prim {prim.GetPath()}")
        return False
    return True


def validate_collision_setup(stage: Usd.Stage, prim_path: str) -> bool:
    """Validate that a prim has a valid collision setup."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path: {prim_path}")
    mesh = UsdGeom.Mesh(prim)
    if not mesh:
        raise ValueError(f"Prim at path {prim_path} is not a Mesh")
    collision_api = UsdPhysics.CollisionAPI(prim)
    if not collision_api:
        return False
    mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)
    if not mesh_collision_api:
        return False
    approximation_attr = mesh_collision_api.GetApproximationAttr()
    if not approximation_attr or not approximation_attr.Get() or approximation_attr.Get() == "none":
        return False
    return True


def create_collision_mesh(stage: Usd.Stage, path: str) -> None:
    """Create a mesh with collision setup for testing."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision_api.CreateApproximationAttr().Set("convexHull")


def transfer_collision_settings(source_prim: Usd.Prim, target_prim: Usd.Prim) -> None:
    """Transfer collision settings from source_prim to target_prim."""
    if not source_prim.IsValid() or not target_prim.IsValid():
        raise ValueError("Invalid source or target prim.")
    if not UsdPhysics.MeshCollisionAPI.CanApply(source_prim):
        raise ValueError("Source prim does not have MeshCollisionAPI applied.")
    if not UsdPhysics.MeshCollisionAPI.CanApply(target_prim):
        UsdPhysics.MeshCollisionAPI.Apply(target_prim)
    source_collision_api = UsdPhysics.MeshCollisionAPI(source_prim)
    target_collision_api = UsdPhysics.MeshCollisionAPI(target_prim)
    if source_collision_api.GetApproximationAttr().HasAuthoredValue():
        approximation = source_collision_api.GetApproximationAttr().Get()
        target_collision_api.CreateApproximationAttr(approximation, True)


def apply_mesh_collision(prim: Usd.Prim, approximation: str = "none") -> UsdPhysics.MeshCollisionAPI:
    """Apply MeshCollisionAPI to a prim and set the approximation attribute.

    Args:
        prim (Usd.Prim): The prim to apply the MeshCollisionAPI to.
        approximation (str, optional): The approximation method to use. Defaults to "none".

    Returns:
        UsdPhysics.MeshCollisionAPI: The applied MeshCollisionAPI schema.

    Raises:
        ValueError: If the prim is not a valid UsdGeomMesh prim or if the approximation is invalid.
    """
    if not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"Prim {prim.GetPath()} is not a valid UsdGeomMesh.")
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    if approximation not in [
        "none",
        "convexDecomposition",
        "convexHull",
        "boundingSphere",
        "boundingCube",
        "meshSimplification",
    ]:
        raise ValueError(f"Invalid approximation method: {approximation}")
    mesh_collision_api.CreateApproximationAttr(approximation, True)
    return mesh_collision_api


def copy_prismatic_joint(stage: Usd.Stage, src_path: str, dst_path: str) -> UsdPhysics.PrismaticJoint:
    """
    Copy a PrismaticJoint prim from source to destination path.

    Args:
        stage (Usd.Stage): The USD stage.
        src_path (str): The source prim path.
        dst_path (str): The destination prim path.

    Returns:
        UsdPhysics.PrismaticJoint: The copied PrismaticJoint prim.

    Raises:
        ValueError: If the source prim is not a valid PrismaticJoint.
    """
    src_prim = stage.GetPrimAtPath(src_path)
    if not src_prim.IsValid() or not UsdPhysics.PrismaticJoint(src_prim):
        raise ValueError(f"Prim at path {src_path} is not a valid PrismaticJoint.")
    dst_prim = UsdPhysics.PrismaticJoint.Define(stage, dst_path)
    src_joint = UsdPhysics.PrismaticJoint(src_prim)
    dst_joint = UsdPhysics.PrismaticJoint(dst_prim)
    if src_joint.GetAxisAttr().HasAuthoredValue():
        dst_joint.CreateAxisAttr(src_joint.GetAxisAttr().Get(), True)
    if src_joint.GetLowerLimitAttr().HasAuthoredValue():
        dst_joint.CreateLowerLimitAttr(src_joint.GetLowerLimitAttr().Get(), True)
    if src_joint.GetUpperLimitAttr().HasAuthoredValue():
        dst_joint.CreateUpperLimitAttr(src_joint.GetUpperLimitAttr().Get(), True)
    return dst_joint


def create_prismatic_joint_with_limits(
    stage: Usd.Stage, joint_path: str, axis: str = "X", lower_limit: float = -1.0, upper_limit: float = 1.0
) -> UsdPhysics.PrismaticJoint:
    """Create a prismatic joint with specified limits.

    Args:
        stage (Usd.Stage): The USD stage to create the joint on.
        joint_path (str): The path where the joint should be created.
        axis (str, optional): The axis along which the joint moves. Must be "X", "Y", or "Z". Defaults to "X".
        lower_limit (float, optional): The lower limit of the joint's movement. Defaults to -1.0.
        upper_limit (float, optional): The upper limit of the joint's movement. Defaults to 1.0.

    Returns:
        UsdPhysics.PrismaticJoint: The created prismatic joint.

    Raises:
        ValueError: If the provided axis is not "X", "Y", or "Z".
    """
    if axis not in ["X", "Y", "Z"]:
        raise ValueError(f'Invalid axis "{axis}". Must be "X", "Y", or "Z".')
    joint = UsdPhysics.PrismaticJoint.Define(stage, Sdf.Path(joint_path))
    axis_attr = joint.CreateAxisAttr()
    axis_attr.Set(axis)
    lower_limit_attr = joint.CreateLowerLimitAttr()
    lower_limit_attr.Set(lower_limit)
    upper_limit_attr = joint.CreateUpperLimitAttr()
    upper_limit_attr.Set(upper_limit)
    return joint


def set_prismatic_joint_limits(joint: UsdPhysics.PrismaticJoint, lower_limit: float, upper_limit: float):
    """Set the lower and upper limits of a prismatic joint.

    Args:
        joint (UsdPhysics.PrismaticJoint): The prismatic joint to set limits for.
        lower_limit (float): The lower limit value. Use -inf for no limit.
        upper_limit (float): The upper limit value. Use inf for no limit.

    Raises:
        ValueError: If the joint is not a valid prismatic joint.
        ValueError: If the lower limit is greater than the upper limit.
    """
    if not joint:
        raise ValueError("Invalid prismatic joint.")
    if lower_limit > upper_limit:
        raise ValueError("Lower limit cannot be greater than the upper limit.")
    lower_limit_attr = joint.CreateLowerLimitAttr() if joint.GetLowerLimitAttr() is None else joint.GetLowerLimitAttr()
    lower_limit_attr.Set(lower_limit)
    upper_limit_attr = joint.CreateUpperLimitAttr() if joint.GetUpperLimitAttr() is None else joint.GetUpperLimitAttr()
    upper_limit_attr.Set(upper_limit)


def get_prismatic_joint_limits(joint: UsdPhysics.PrismaticJoint) -> Tuple[float, float]:
    """Get the lower and upper limits of a prismatic joint.

    Args:
        joint (UsdPhysics.PrismaticJoint): The prismatic joint to query.

    Returns:
        Tuple[float, float]: A tuple containing the lower and upper limits.
    """
    lower_limit_attr = joint.GetLowerLimitAttr()
    if lower_limit_attr.HasAuthoredValue():
        lower_limit = lower_limit_attr.Get()
    else:
        lower_limit = float("-inf")
    upper_limit_attr = joint.GetUpperLimitAttr()
    if upper_limit_attr.HasAuthoredValue():
        upper_limit = upper_limit_attr.Get()
    else:
        upper_limit = float("inf")
    return (lower_limit, upper_limit)


def set_prismatic_joint_defaults(joint: UsdPhysics.PrismaticJoint) -> None:
    """Set the default values for a prismatic joint."""
    if not joint:
        raise ValueError("Invalid prismatic joint")
    axis_attr = joint.GetAxisAttr()
    lower_limit_attr = joint.GetLowerLimitAttr()
    upper_limit_attr = joint.GetUpperLimitAttr()
    if not axis_attr.IsAuthored():
        axis_attr.Set(UsdPhysics.Tokens.x)
    if not lower_limit_attr.IsAuthored():
        lower_limit_attr.Set(-float("inf"))
    if not upper_limit_attr.IsAuthored():
        upper_limit_attr.Set(float("inf"))


def align_prismatic_joint_axes(joint: UsdPhysics.PrismaticJoint, target_axis: Gf.Vec3d) -> None:
    """Align the local axes of a prismatic joint to a target direction.

    The joint's X-axis is aligned to the target direction, while the Y and Z axes
    are arbitrarily chosen to form an orthonormal basis.

    Args:
        joint (UsdPhysics.PrismaticJoint): The prismatic joint to align.
        target_axis (Gf.Vec3d): The target direction to align the X-axis to.

    Raises:
        ValueError: If the target axis is a zero vector.
    """
    target_axis = target_axis.GetNormalized()
    if target_axis == Gf.Vec3d(0, 0, 0):
        raise ValueError("Target axis cannot be a zero vector.")
    xform = UsdGeom.Xformable(joint.GetPrim())
    rotation_op = add_rotate_xyz_op(xform, opSuffix="jointOrientation")
    x_axis = Gf.Vec3d(1, 0, 0)
    rotation = Gf.Rotation(x_axis, target_axis)
    hpr = rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
    rotation_op.Set(Gf.Vec3f(hpr))
    joint_axis = "X" if abs(target_axis[0]) > 0.5 else "Y" if abs(target_axis[1]) > 0.5 else "Z"
    joint.CreateAxisAttr().Set(joint_axis)


def remove_prismatic_joint(stage: Usd.Stage, joint_path: str) -> bool:
    """Remove a prismatic joint prim from the stage.

    Args:
        stage (Usd.Stage): The USD stage.
        joint_path (str): The path to the prismatic joint prim.

    Returns:
        bool: True if the prismatic joint was successfully removed, False otherwise.
    """
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid() or not UsdPhysics.PrismaticJoint(joint_prim):
        print(f"Prim at path {joint_path} is not a valid prismatic joint.")
        return False
    stage.RemovePrim(joint_path)
    if stage.GetPrimAtPath(joint_path).IsValid():
        print(f"Failed to remove prismatic joint at path {joint_path}.")
        return False
    return True


def get_prismatic_joint_axis(joint: UsdPhysics.PrismaticJoint) -> str:
    """Get the axis of a prismatic joint.

    Args:
        joint (UsdPhysics.PrismaticJoint): The prismatic joint prim.

    Returns:
        str: The axis of the prismatic joint.
    """
    axis_attr = joint.GetAxisAttr()
    if axis_attr.IsValid():
        axis = axis_attr.Get()
        if axis in ["X", "Y", "Z"]:
            return axis
        else:
            raise ValueError(f"Invalid axis value: {axis}. Must be one of X, Y, or Z.")
    else:
        return "X"


def create_prismatic_joint_with_defaults(stage: Usd.Stage, path: str) -> UsdPhysics.PrismaticJoint:
    """Create a PrismaticJoint with default attribute values."""
    joint = UsdPhysics.PrismaticJoint.Define(stage, path)
    if not joint:
        raise RuntimeError(f"Failed to create PrismaticJoint at path: {path}")
    axis_attr = joint.CreateAxisAttr()
    if not axis_attr:
        raise RuntimeError(f"Failed to create axis attribute on PrismaticJoint at path: {path}")
    axis_attr.Set("X")
    lower_limit_attr = joint.CreateLowerLimitAttr()
    if not lower_limit_attr:
        raise RuntimeError(f"Failed to create lowerLimit attribute on PrismaticJoint at path: {path}")
    lower_limit_attr.Set(float("-inf"))
    upper_limit_attr = joint.CreateUpperLimitAttr()
    if not upper_limit_attr:
        raise RuntimeError(f"Failed to create upperLimit attribute on PrismaticJoint at path: {path}")
    upper_limit_attr.Set(float("inf"))
    return joint


def animate_prismatic_joint(
    stage: Usd.Stage, joint_path: str, lower_limit: float, upper_limit: float, num_frames: int = 100
) -> None:
    """Animate a prismatic joint between its lower and upper limits.

    Args:
        stage (Usd.Stage): The USD stage.
        joint_path (str): The path to the prismatic joint prim.
        lower_limit (float): The lower limit of the joint's range of motion.
        upper_limit (float): The upper limit of the joint's range of motion.
        num_frames (int): The number of frames to animate over. Defaults to 100.

    Raises:
        ValueError: If the joint prim is not a valid prismatic joint.
    """
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid() or not joint_prim.IsA(UsdPhysics.PrismaticJoint):
        raise ValueError(f"Prim at path {joint_path} is not a valid prismatic joint.")
    prismatic_joint = UsdPhysics.PrismaticJoint(joint_prim)
    if not prismatic_joint.GetLowerLimitAttr().HasAuthoredValue():
        prismatic_joint.CreateLowerLimitAttr(lower_limit)
    if not prismatic_joint.GetUpperLimitAttr().HasAuthoredValue():
        prismatic_joint.CreateUpperLimitAttr(upper_limit)
    for frame in range(num_frames):
        t = frame / (num_frames - 1)
        joint_pos = lower_limit + (upper_limit - lower_limit) * t
        UsdGeom.XformCommonAPI(joint_prim).SetTranslate(Gf.Vec3d(joint_pos, 0, 0), Usd.TimeCode(frame))


def create_revolute_joint(
    stage: Usd.Stage,
    path: str,
    axis: Gf.Vec3f = Gf.Vec3f(1, 0, 0),
    lower_limit: float = -float("inf"),
    upper_limit: float = float("inf"),
) -> UsdPhysics.RevoluteJoint:
    """Create a revolute joint with specified properties.

    Args:
        stage (Usd.Stage): The stage to create the joint on.
        path (str): The path where the joint should be created.
        axis (Gf.Vec3f): The joint axis. Defaults to (1, 0, 0) which is the X-axis.
        lower_limit (float): Lower limit in degrees. Defaults to negative infinity.
        upper_limit (float): Upper limit in degrees. Defaults to positive infinity.

    Returns:
        UsdPhysics.RevoluteJoint: The created revolute joint.
    """
    if not stage:
        raise ValueError("Invalid stage.")
    path = Sdf.Path(path).MakeAbsolutePath("/")
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    if not joint:
        raise RuntimeError(f"Failed to create revolute joint at path: {path}")
    if axis != Gf.Vec3f(1, 0, 0):
        axis.Normalize()
        axis_attr = joint.CreateAxisAttr()
        if axis == Gf.Vec3f(1, 0, 0):
            axis_attr.Set(UsdPhysics.Tokens.x)
        elif axis == Gf.Vec3f(0, 1, 0):
            axis_attr.Set(UsdPhysics.Tokens.y)
        elif axis == Gf.Vec3f(0, 0, 1):
            axis_attr.Set(UsdPhysics.Tokens.z)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be X, Y, or Z.")
    if lower_limit != -float("inf"):
        lower_limit_attr = joint.CreateLowerLimitAttr()
        lower_limit_attr.Set(lower_limit)
    if upper_limit != float("inf"):
        upper_limit_attr = joint.CreateUpperLimitAttr()
        upper_limit_attr.Set(upper_limit)
    return joint


def toggle_revolute_joint_limit(joint: UsdPhysics.RevoluteJoint, lower_limit: float, upper_limit: float) -> None:
    """Toggle the limits of a revolute joint.

    Args:
        joint (UsdPhysics.RevoluteJoint): The revolute joint to modify.
        lower_limit (float): The lower limit value in degrees. Use -inf for no limit.
        upper_limit (float): The upper limit value in degrees. Use inf for no limit.

    Raises:
        ValueError: If the joint is not a valid UsdPhysics.RevoluteJoint prim.
    """
    if not joint:
        raise ValueError("Invalid UsdPhysics.RevoluteJoint prim.")
    lower_limit_attr = joint.GetLowerLimitAttr()
    if lower_limit_attr:
        lower_limit_attr.Set(lower_limit)
    else:
        joint.CreateLowerLimitAttr(lower_limit, writeSparsely=True)
    upper_limit_attr = joint.GetUpperLimitAttr()
    if upper_limit_attr:
        upper_limit_attr.Set(upper_limit)
    else:
        joint.CreateUpperLimitAttr(upper_limit, writeSparsely=True)


def get_revolute_joint_info(joint: UsdPhysics.RevoluteJoint) -> Tuple[str, Gf.Vec3f, float, float]:
    """Get information about a revolute joint.

    Args:
        joint (UsdPhysics.RevoluteJoint): The revolute joint prim.

    Returns:
        Tuple[str, Gf.Vec3f, float, float]: A tuple containing:
            - The joint axis as a string ("X", "Y", or "Z")
            - The joint local pose as a Gf.Vec3f
            - The lower limit in degrees
            - The upper limit in degrees
    """
    axis_attr = joint.GetAxisAttr()
    if axis_attr.IsValid():
        axis = axis_attr.Get()
    else:
        axis = "X"
    local_pose = joint.GetLocalPos0Attr().Get()
    if local_pose is None:
        local_pose = Gf.Vec3f(0, 0, 0)
    lower_limit_attr = joint.GetLowerLimitAttr()
    if lower_limit_attr.IsValid():
        lower_limit = lower_limit_attr.Get()
    else:
        lower_limit = float("-inf")
    upper_limit_attr = joint.GetUpperLimitAttr()
    if upper_limit_attr.IsValid():
        upper_limit = upper_limit_attr.Get()
    else:
        upper_limit = float("inf")
    return (axis, local_pose, lower_limit, upper_limit)


def set_revolute_joint_limits(joint: UsdPhysics.RevoluteJoint, lower_limit: float, upper_limit: float):
    """Set the lower and upper limits of a revolute joint in degrees."""
    if not joint:
        raise ValueError("Invalid UsdPhysics.RevoluteJoint object")
    if lower_limit > upper_limit:
        raise ValueError("Lower limit cannot be greater than the upper limit")
    lower_limit_attr = joint.CreateLowerLimitAttr()
    lower_limit_attr.Set(lower_limit)
    upper_limit_attr = joint.CreateUpperLimitAttr()
    upper_limit_attr.Set(upper_limit)


def get_revolute_joint_limits(joint: UsdPhysics.RevoluteJoint) -> Tuple[float, float]:
    """Get the lower and upper limits of a revolute joint.

    Args:
        joint (UsdPhysics.RevoluteJoint): The revolute joint to get limits from.

    Returns:
        Tuple[float, float]: A tuple with (lower_limit, upper_limit) in degrees.
    """
    if not joint or not isinstance(joint, UsdPhysics.RevoluteJoint):
        raise ValueError("Invalid RevoluteJoint object")
    lower_limit_attr = joint.GetLowerLimitAttr()
    if lower_limit_attr.IsAuthored():
        lower_limit = lower_limit_attr.Get()
    else:
        lower_limit = float("-inf")
    upper_limit_attr = joint.GetUpperLimitAttr()
    if upper_limit_attr.IsAuthored():
        upper_limit = upper_limit_attr.Get()
    else:
        upper_limit = float("inf")
    return (lower_limit, upper_limit)


def delete_revolute_joint(stage: Usd.Stage, joint_path: str) -> bool:
    """Delete a revolute joint prim from the stage."""
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid() or not joint_prim.IsA(UsdPhysics.RevoluteJoint):
        return False
    stage.RemovePrim(joint_path)
    return True


def set_revolute_joint_axis(joint: UsdPhysics.RevoluteJoint, axis: Gf.Vec3f) -> None:
    """Set the axis for a revolute joint.

    Args:
        joint (UsdPhysics.RevoluteJoint): The revolute joint prim.
        axis (Gf.Vec3f): The joint axis vector.

    Raises:
        ValueError: If the joint prim is not valid or if the axis vector is not unit length.
    """
    if not joint:
        raise ValueError("Invalid UsdPhysics.RevoluteJoint prim.")
    axis = Gf.Vec3f(axis).GetNormalized()
    if not Gf.IsClose(axis.GetLength(), 1.0, 1e-06):
        raise ValueError("Joint axis must be a unit vector.")
    abs_axis = Gf.Vec3f(abs(axis[0]), abs(axis[1]), abs(axis[2]))
    max_component = max(abs_axis[0], abs_axis[1], abs_axis[2])
    if Gf.IsClose(max_component, abs_axis[0], 1e-06):
        joint.CreateAxisAttr().Set(UsdPhysics.Tokens.x)
    elif Gf.IsClose(max_component, abs_axis[1], 1e-06):
        joint.CreateAxisAttr().Set(UsdPhysics.Tokens.y)
    else:
        joint.CreateAxisAttr().Set(UsdPhysics.Tokens.z)


def get_revolute_joint_axis(joint: UsdPhysics.RevoluteJoint) -> str:
    """Get the axis of a revolute joint.

    Args:
        joint (UsdPhysics.RevoluteJoint): The revolute joint prim.

    Returns:
        str: The joint axis token.
    """
    axis_attr = joint.GetAxisAttr()
    if not axis_attr.IsValid():
        return "X"
    axis = axis_attr.Get()
    if axis not in ("X", "Y", "Z"):
        return "X"
    return axis


def set_rigid_body_velocity(stage: Usd.Stage, prim_path: str, linear_velocity: Gf.Vec3f, angular_velocity: Gf.Vec3f):
    """Set the linear and angular velocity for a rigid body prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the rigid body prim.
        linear_velocity (Gf.Vec3f): The linear velocity to set.
        angular_velocity (Gf.Vec3f): The angular velocity to set.

    Raises:
        ValueError: If the prim does not exist or is not a valid rigid body.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
    if not rigid_body_api:
        raise ValueError(f"Prim at path {prim_path} is not a valid rigid body.")
    linear_velocity_attr = rigid_body_api.GetVelocityAttr()
    if linear_velocity_attr:
        linear_velocity_attr.Set(linear_velocity)
    else:
        linear_velocity_attr = rigid_body_api.CreateVelocityAttr()
        linear_velocity_attr.Set(linear_velocity)
    angular_velocity_attr = rigid_body_api.GetAngularVelocityAttr()
    if angular_velocity_attr:
        angular_velocity_attr.Set(angular_velocity)
    else:
        angular_velocity_attr = rigid_body_api.CreateAngularVelocityAttr()
        angular_velocity_attr.Set(angular_velocity)


def get_rigid_body_state(stage: Usd.Stage, prim_path: str) -> Tuple[Gf.Vec3f, Gf.Vec3f, bool]:
    """Get the rigid body state for a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Returns:
        A tuple containing the velocity, angular velocity, and starts asleep state.

    Raises:
        ValueError: If the prim is not valid or does not have a RigidBodyAPI applied.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    rb_api = UsdPhysics.RigidBodyAPI(prim)
    if not rb_api:
        raise ValueError(f"Prim at path {prim_path} does not have a RigidBodyAPI applied.")
    velocity_attr = rb_api.GetVelocityAttr()
    if velocity_attr.HasAuthoredValue():
        velocity = velocity_attr.Get()
    else:
        velocity = Gf.Vec3f(0.0)
    angular_velocity_attr = rb_api.GetAngularVelocityAttr()
    if angular_velocity_attr.HasAuthoredValue():
        angular_velocity = angular_velocity_attr.Get()
    else:
        angular_velocity = Gf.Vec3f(0.0)
    starts_asleep_attr = rb_api.GetStartsAsleepAttr()
    if starts_asleep_attr.HasAuthoredValue():
        starts_asleep = starts_asleep_attr.Get()
    else:
        starts_asleep = False
    return (velocity, angular_velocity, starts_asleep)


def set_rigid_body_enabled(prim: Usd.Prim, enabled: bool) -> None:
    """Set the rigid body enabled state for a prim."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    rigid_body_api = UsdPhysics.RigidBodyAPI.Get(prim.GetStage(), prim.GetPath())
    if not rigid_body_api:
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        if not rigid_body_api:
            raise ValueError(f"Failed to apply RigidBodyAPI schema to prim {prim.GetPath()}.")
    enabled_attr = rigid_body_api.GetRigidBodyEnabledAttr()
    if not enabled_attr:
        enabled_attr = rigid_body_api.CreateRigidBodyEnabledAttr(enabled, True)
    enabled_attr.Set(enabled)


def transfer_physics_properties(src_prim: Usd.Prim, dst_prim: Usd.Prim):
    """Transfer physics properties from one prim to another."""
    src_rigid_body_api = UsdPhysics.RigidBodyAPI.Get(src_prim.GetStage(), src_prim.GetPath())
    if not src_rigid_body_api:
        raise ValueError(f"Source prim {src_prim.GetPath()} does not have a RigidBodyAPI applied.")
    if not dst_prim.IsValid():
        raise ValueError(f"Destination prim {dst_prim.GetPath()} is not valid.")
    dst_rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(dst_prim)
    if src_rigid_body_api.GetRigidBodyEnabledAttr().HasAuthoredValue():
        enabled = src_rigid_body_api.GetRigidBodyEnabledAttr().Get()
        dst_rigid_body_api.CreateRigidBodyEnabledAttr(enabled)
    if src_rigid_body_api.GetKinematicEnabledAttr().HasAuthoredValue():
        kinematic = src_rigid_body_api.GetKinematicEnabledAttr().Get()
        dst_rigid_body_api.CreateKinematicEnabledAttr(kinematic)
    if src_rigid_body_api.GetStartsAsleepAttr().HasAuthoredValue():
        starts_asleep = src_rigid_body_api.GetStartsAsleepAttr().Get()
        dst_rigid_body_api.CreateStartsAsleepAttr(starts_asleep)
    if src_rigid_body_api.GetVelocityAttr().HasAuthoredValue():
        velocity = src_rigid_body_api.GetVelocityAttr().Get()
        dst_rigid_body_api.CreateVelocityAttr(velocity)
    if src_rigid_body_api.GetAngularVelocityAttr().HasAuthoredValue():
        angular_velocity = src_rigid_body_api.GetAngularVelocityAttr().Get()
        dst_rigid_body_api.CreateAngularVelocityAttr(angular_velocity)
    return dst_rigid_body_api


def set_angular_velocity(prim: Usd.Prim, angular_velocity: Gf.Vec3f) -> None:
    """Set the angular velocity for a rigid body prim.

    Args:
        prim (Usd.Prim): The rigid body prim.
        angular_velocity (Gf.Vec3f): The angular velocity to set in degrees per second.

    Raises:
        ValueError: If the prim is not a valid rigid body prim.
    """
    rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
    if not rigid_body_api:
        raise ValueError(f"Prim '{prim.GetPath()}' is not a valid RigidBodyAPI prim.")
    angular_velocity_attr = rigid_body_api.GetAngularVelocityAttr()
    if not angular_velocity_attr:
        angular_velocity_attr = rigid_body_api.CreateAngularVelocityAttr()
    angular_velocity_attr.Set(angular_velocity)


def toggle_kinematic_enabled(rigid_body_api: UsdPhysics.RigidBodyAPI) -> bool:
    """Toggle the kinematic enabled state of a RigidBodyAPI and return the new state."""
    current_state = rigid_body_api.GetKinematicEnabledAttr().Get()
    if current_state is None:
        current_state = False
    new_state = not current_state
    rigid_body_api.CreateKinematicEnabledAttr(new_state)
    return new_state


def batch_apply_rigid_body_api(stage: Usd.Stage, prim_paths: List[str]) -> List[UsdPhysics.RigidBodyAPI]:
    """Apply UsdPhysics.RigidBodyAPI to a list of prim paths.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths to apply the API to.

    Returns:
        List[UsdPhysics.RigidBodyAPI]: List of applied RigidBodyAPI objects.
    """
    applied_apis = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Prim at path {prim_path} does not exist. Skipping.")
            continue
        if not UsdPhysics.RigidBodyAPI.CanApply(prim):
            print(f"Cannot apply RigidBodyAPI to prim at path {prim_path}. Skipping.")
            continue
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        if not rigid_body_api.GetPrim().IsValid():
            print(f"Failed to apply RigidBodyAPI to prim at path {prim_path}. Skipping.")
            continue
        applied_apis.append(rigid_body_api)
    return applied_apis


def batch_set_rigid_body_enabled(prims: List[Usd.Prim], enabled: bool) -> None:
    """Set the rigid body enabled state for a list of prims.

    Args:
        prims (List[Usd.Prim]): List of prims to set the rigid body enabled state for.
        enabled (bool): The enabled state to set.
    """
    for prim in prims:
        if not prim.IsValid():
            continue
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        enabled_attr = rigid_body_api.GetRigidBodyEnabledAttr()
        if enabled_attr.IsValid():
            enabled_attr.Set(enabled)


def export_rigid_body_data(stage: Usd.Stage, prim_path: str) -> Dict[str, Any]:
    """Export rigid body data for a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.

    Returns:
        Dict[str, Any]: A dictionary containing the rigid body data.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path: {prim_path}")
    rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
    if not rigid_body_api:
        raise ValueError(f"Prim at path {prim_path} does not have a RigidBodyAPI applied.")
    data = {}
    if rigid_body_api.GetRigidBodyEnabledAttr().HasAuthoredValue():
        data["rigidBodyEnabled"] = rigid_body_api.GetRigidBodyEnabledAttr().Get()
    else:
        data["rigidBodyEnabled"] = True
    if rigid_body_api.GetKinematicEnabledAttr().HasAuthoredValue():
        data["kinematicEnabled"] = rigid_body_api.GetKinematicEnabledAttr().Get()
    else:
        data["kinematicEnabled"] = False
    if rigid_body_api.GetStartsAsleepAttr().HasAuthoredValue():
        data["startsAsleep"] = rigid_body_api.GetStartsAsleepAttr().Get()
    else:
        data["startsAsleep"] = False
    if rigid_body_api.GetVelocityAttr().HasAuthoredValue():
        data["velocity"] = rigid_body_api.GetVelocityAttr().Get()
    else:
        data["velocity"] = (0.0, 0.0, 0.0)
    if rigid_body_api.GetAngularVelocityAttr().HasAuthoredValue():
        data["angularVelocity"] = rigid_body_api.GetAngularVelocityAttr().Get()
    else:
        data["angularVelocity"] = (0.0, 0.0, 0.0)
    return data


def analyze_rigid_body_collisions(stage: Usd.Stage, collision_distance: float) -> List[Tuple[str, str]]:
    """Analyze rigid body collisions in a USD stage.

    Args:
        stage (Usd.Stage): The USD stage to analyze.
        collision_distance (float): The maximum distance between two bodies to be considered a collision.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the paths of colliding rigid bodies.
    """
    rigid_body_prims = [prim for prim in stage.Traverse() if UsdPhysics.RigidBodyAPI.CanApply(prim)]
    colliding_pairs = []
    for i in range(len(rigid_body_prims)):
        for j in range(i + 1, len(rigid_body_prims)):
            prim1 = rigid_body_prims[i]
            prim2 = rigid_body_prims[j]
            transform1 = UsdGeom.Xformable(prim1).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            transform2 = UsdGeom.Xformable(prim2).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translate1 = transform1.ExtractTranslation()
            translate2 = transform2.ExtractTranslation()
            distance = (translate2 - translate1).GetLength()
            if distance <= collision_distance:
                colliding_pairs.append((str(prim1.GetPath()), str(prim2.GetPath())))
    return colliding_pairs


def create_physics_attributes(prim: Usd.Prim) -> None:
    """Create physics attributes on a prim using UsdPhysics.RigidBodyAPI."""
    if not prim.IsValid():
        raise ValueError(f"Input prim {prim.GetPath()} is not valid.")
    if not prim.IsA(UsdGeom.Xformable):
        raise TypeError(f"Prim {prim.GetPath()} is not of type UsdGeomXformable.")
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)
    rigid_body_api.CreateKinematicEnabledAttr(False)
    rigid_body_api.CreateStartsAsleepAttr(False)
    rigid_body_api.CreateVelocityAttr(Gf.Vec3f(0, 0, 0))
    rigid_body_api.CreateAngularVelocityAttr(Gf.Vec3f(0, 0, 0))


def create_physics_scene(
    stage: Usd.Stage, prim_path: str, gravity_direction: Gf.Vec3f = Gf.Vec3f(0, 0, 0), gravity_magnitude: float = -1.0
) -> UsdPhysics.Scene:
    """Create a physics scene with specified gravity direction and magnitude.

    Args:
        stage (Usd.Stage): The USD stage to create the physics scene on.
        prim_path (str): The path where the physics scene should be created.
        gravity_direction (Gf.Vec3f): The direction of gravity. Defaults to (0, 0, 0) which uses the negative upAxis.
        gravity_magnitude (float): The magnitude of gravity. Defaults to -1.0 which uses earth gravity.

    Returns:
        UsdPhysics.Scene: The created physics scene.

    Raises:
        ValueError: If the prim at the specified path already exists and is not a valid physics scene.
    """
    existing_prim = stage.GetPrimAtPath(prim_path)
    if existing_prim.IsValid():
        physics_scene = UsdPhysics.Scene(existing_prim)
        if not physics_scene.IsValid():
            raise ValueError(f"Prim at path {prim_path} already exists and is not a valid physics scene.")
    else:
        physics_scene = UsdPhysics.Scene.Define(stage, prim_path)
    physics_scene.CreateGravityDirectionAttr(gravity_direction)
    physics_scene.CreateGravityMagnitudeAttr(gravity_magnitude)
    return physics_scene


def apply_rigid_body_api(prim: Usd.Prim) -> UsdPhysics.RigidBodyAPI:
    """Apply RigidBodyAPI to a prim."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if not UsdGeom.Xformable(prim):
        raise ValueError(f"Prim {prim.GetPath()} is not a UsdGeomXformable")
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    if not rigid_body_api:
        raise RuntimeError(f"Failed to apply RigidBodyAPI to prim {prim.GetPath()}")
    return rigid_body_api


def create_simulation_owner_relationship(prim: Usd.Prim, simulation_owner_path: str) -> UsdPhysics.RigidBodyAPI:
    """Create a relationship on the prim to the simulation owner prim.

    Args:
        prim (Usd.Prim): The prim to create the relationship on.
        simulation_owner_path (str): The path to the simulation owner prim.

    Returns:
        UsdPhysics.RigidBodyAPI: The RigidBodyAPI schema with the created relationship.

    Raises:
        ValueError: If the prim is not valid or if the simulation owner prim does not exist.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    stage = prim.GetStage()
    if not stage.GetPrimAtPath(simulation_owner_path):
        raise ValueError(f"Simulation owner prim at path {simulation_owner_path} does not exist.")
    rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
    simulation_owner_rel = rigid_body_api.GetSimulationOwnerRel()
    if not simulation_owner_rel:
        simulation_owner_rel = rigid_body_api.CreateSimulationOwnerRel()
    simulation_owner_rel.AddTarget(simulation_owner_path)
    return rigid_body_api


def remove_rigid_body_api(prim: Usd.Prim) -> bool:
    """Remove the RigidBodyAPI from the given prim if it exists."""
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return False
    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
    return True


def import_rigid_body_data(
    prim: Usd.Prim,
    angular_velocity: Gf.Vec3f = None,
    velocity: Gf.Vec3f = None,
    kinematic_enabled: bool = None,
    starts_asleep: bool = None,
) -> None:
    """Import rigid body data to a USD prim using the UsdPhysics.RigidBodyAPI schema.

    Args:
        prim (Usd.Prim): The USD prim to apply the rigid body data to.
        angular_velocity (Gf.Vec3f, optional): The angular velocity of the rigid body. Defaults to None.
        velocity (Gf.Vec3f, optional): The linear velocity of the rigid body. Defaults to None.
        kinematic_enabled (bool, optional): Whether the rigid body is kinematic. Defaults to None.
        starts_asleep (bool, optional): Whether the rigid body starts asleep. Defaults to None.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    if angular_velocity is not None:
        angular_velocity_attr = rigid_body_api.CreateAngularVelocityAttr()
        angular_velocity_attr.Set(angular_velocity)
    if velocity is not None:
        velocity_attr = rigid_body_api.CreateVelocityAttr()
        velocity_attr.Set(velocity)
    if kinematic_enabled is not None:
        kinematic_enabled_attr = rigid_body_api.CreateKinematicEnabledAttr()
        kinematic_enabled_attr.Set(kinematic_enabled)
    if starts_asleep is not None:
        starts_asleep_attr = rigid_body_api.CreateStartsAsleepAttr()
        starts_asleep_attr.Set(starts_asleep)


def simulate_rigid_body(rigid_body_api: UsdPhysics.RigidBodyAPI, time_code: Usd.TimeCode) -> bool:
    """Simulate the rigid body and update its pose at the given time code.

    Args:
        rigid_body_api (UsdPhysics.RigidBodyAPI): The RigidBodyAPI object to simulate.
        time_code (Usd.TimeCode): The time code at which to simulate the rigid body.

    Returns:
        bool: True if the simulation was successful, False otherwise.
    """
    if not rigid_body_api.GetPrim().IsValid():
        return False
    if not rigid_body_api.GetRigidBodyEnabledAttr().Get(time_code):
        return True
    physics_scene_paths = rigid_body_api.GetSimulationOwnerRel().GetTargets()
    if not physics_scene_paths:
        return False
    physics_scene = rigid_body_api.GetPrim().GetStage().GetPrimAtPath(physics_scene_paths[0])
    if not physics_scene.IsValid():
        return False
    xformable = UsdGeom.Xformable(rigid_body_api.GetPrim())
    initial_transform = xformable.GetLocalTransformation(Usd.TimeCode.Default())
    velocity = rigid_body_api.GetVelocityAttr().Get(time_code)
    angular_velocity = rigid_body_api.GetAngularVelocityAttr().Get(time_code)
    simulated_transform = initial_transform
    simulated_velocity = velocity
    simulated_angular_velocity = angular_velocity
    xform_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTransform)
    xform_op.Set(simulated_transform, time_code)
    rigid_body_api.GetVelocityAttr().Set(simulated_velocity, time_code)
    rigid_body_api.GetAngularVelocityAttr().Set(simulated_angular_velocity, time_code)
    return True


class MassInformation:

    def __init__(self):
        self.centerOfMass = Gf.Vec3f(0)
        self.inertia = Gf.Matrix3f(0)
        self.localPos = Gf.Vec3f(0)
        self.localRot = Gf.Quatf(1)
        self.volume = 0.0


def calculate_combined_mass_information(mass_infos: List[MassInformation]) -> MassInformation:
    """Calculate the combined mass information from a list of MassInformation objects."""
    if not mass_infos:
        raise ValueError("Input list of MassInformation objects is empty.")
    total_mass = sum((mass.volume for mass in mass_infos))
    if total_mass == 0:
        raise ValueError("Total mass of input objects is zero.")
    combined_center_of_mass = Gf.Vec3f(0, 0, 0)
    for mass in mass_infos:
        combined_center_of_mass += mass.volume * mass.localPos
    combined_center_of_mass /= total_mass
    combined_inertia = Gf.Matrix3f(0)
    for mass in mass_infos:
        r = mass.localPos - combined_center_of_mass
        translated_inertia = mass.inertia + mass.volume * (r.GetLength() ** 2 * Gf.Matrix3f(1) - Gf.Matrix3f(r * r))
        combined_inertia += translated_inertia
    combined_mass_info = MassInformation()
    combined_mass_info.volume = total_mass
    combined_mass_info.localPos = combined_center_of_mass
    combined_mass_info.inertia = combined_inertia
    combined_mass_info.centerOfMass = combined_center_of_mass
    combined_mass_info.localRot = Gf.Quatf(1)
    return combined_mass_info


def get_scene_gravity(stage: Usd.Stage) -> Gf.Vec3f:
    """Get the gravity vector for a USD stage."""
    root_prim = stage.GetPseudoRoot()
    scene_prim = UsdPhysics.Scene.Get(stage, root_prim.GetPath())
    if not scene_prim:
        return Gf.Vec3f(0.0, 0.0, 0.0)
    gravity_dir_attr = scene_prim.GetGravityDirectionAttr()
    if gravity_dir_attr.HasAuthoredValue():
        gravity_dir = gravity_dir_attr.Get()
    else:
        gravity_dir = Gf.Vec3f(0.0, -1.0, 0.0)
    gravity_mag_attr = scene_prim.GetGravityMagnitudeAttr()
    if gravity_mag_attr.HasAuthoredValue():
        gravity_mag = gravity_mag_attr.Get()
    else:
        gravity_mag = 981.0
    gravity = gravity_dir.GetNormalized() * gravity_mag
    return gravity


def setup_default_gravity(scene: UsdPhysics.Scene) -> None:
    """Set up default gravity for a physics scene if not already set."""
    if not scene.GetGravityDirectionAttr().IsAuthored():
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
    if not scene.GetGravityMagnitudeAttr().IsAuthored():
        scene.CreateGravityMagnitudeAttr().Set(981.0)


def analyze_physical_properties(stage: Usd.Stage, scene_path: str) -> Dict[str, Any]:
    """Analyze the physical properties of a UsdPhysics.Scene prim.

    Args:
        stage (Usd.Stage): The stage containing the scene prim.
        scene_path (str): The path to the UsdPhysics.Scene prim.

    Returns:
        Dict[str, Any]: A dictionary containing the physical properties.
    """
    scene_prim = stage.GetPrimAtPath(scene_path)
    if not scene_prim.IsValid():
        raise ValueError(f"Invalid scene prim path: {scene_path}")
    scene = UsdPhysics.Scene(scene_prim)
    if not scene:
        raise ValueError(f"Prim at path {scene_path} is not a valid UsdPhysics.Scene")
    result = {}
    gravity_direction_attr = scene.GetGravityDirectionAttr()
    if gravity_direction_attr.HasAuthoredValue():
        result["gravityDirection"] = gravity_direction_attr.Get()
    else:
        result["gravityDirection"] = gravity_direction_attr.GetDefaultValue()
    gravity_magnitude_attr = scene.GetGravityMagnitudeAttr()
    if gravity_magnitude_attr.HasAuthoredValue():
        result["gravityMagnitude"] = gravity_magnitude_attr.Get()
    else:
        result["gravityMagnitude"] = gravity_magnitude_attr.GetDefaultValue()
    return result


def simulate_physics(
    stage: Usd.Stage,
    scene_path: str,
    time_codes: Sequence[Usd.TimeCode],
    gravity_direction: Gf.Vec3f = Gf.Vec3f(0, 0, 0),
    gravity_magnitude: float = -1.0,
) -> None:
    """Simulate physics on a USD stage for given time codes.

    Args:
        stage (Usd.Stage): The USD stage to simulate physics on.
        scene_path (str): The path to the UsdPhysicsScene prim.
        time_codes (Sequence[Usd.TimeCode]): The time codes to simulate physics for.
        gravity_direction (Gf.Vec3f, optional): The gravity direction vector. Defaults to Gf.Vec3f(0, 0, 0).
        gravity_magnitude (float, optional): The gravity magnitude. Defaults to -1.0.

    Raises:
        ValueError: If the physics scene prim is not valid or if no time codes are provided.
    """
    physics_scene = UsdPhysics.Scene.Get(stage, scene_path)
    if not physics_scene:
        raise ValueError(f"Invalid physics scene prim at path: {scene_path}")
    if not time_codes:
        raise ValueError("No time codes provided for physics simulation.")
    physics_scene.CreateGravityDirectionAttr().Set(gravity_direction)
    physics_scene.CreateGravityMagnitudeAttr().Set(gravity_magnitude)
    for time_code in time_codes:
        pass


def reset_physics_scene(stage: Usd.Stage, scene_path: str):
    """Reset the physics scene at the given path."""
    scene_prim = stage.GetPrimAtPath(scene_path)
    if not scene_prim.IsValid():
        raise ValueError(f"Prim at path {scene_path} does not exist.")
    physics_scene = UsdPhysics.Scene(scene_prim)
    if not physics_scene:
        raise ValueError(f"Prim at path {scene_path} is not a valid UsdPhysicsScene.")
    gravity_dir_attr = physics_scene.GetGravityDirectionAttr()
    if gravity_dir_attr:
        gravity_dir_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
    else:
        gravity_dir_attr = physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, 0.0), True)
    gravity_mag_attr = physics_scene.GetGravityMagnitudeAttr()
    if gravity_mag_attr:
        gravity_mag_attr.Set(float("-inf"))
    else:
        gravity_mag_attr = physics_scene.CreateGravityMagnitudeAttr(float("-inf"), True)
    return physics_scene


def create_spherical_joint(
    stage: Usd.Stage, path: str, axis: str = "X", cone_angle_0_limit: float = -1, cone_angle_1_limit: float = -1
) -> UsdPhysics.SphericalJoint:
    """Create a spherical joint.

    Args:
        stage (Usd.Stage): The USD stage.
        path (str): The path of the joint.
        axis (str, optional): The cone limit axis. Defaults to "X".
        cone_angle_0_limit (float, optional): Cone limit from the primary joint axis toward the next axis. Defaults to -1.
        cone_angle_1_limit (float, optional): Cone limit from the primary joint axis toward the second to next axis. Defaults to -1.

    Returns:
        UsdPhysics.SphericalJoint: The created spherical joint.
    """
    if axis not in ["X", "Y", "Z"]:
        raise ValueError(f"Invalid axis value: {axis}. Must be 'X', 'Y', or 'Z'.")
    joint = UsdPhysics.SphericalJoint.Define(stage, Sdf.Path(path))
    axis_attr = joint.CreateAxisAttr()
    axis_attr.Set(axis)
    if cone_angle_0_limit >= 0:
        cone_angle_0_limit_attr = joint.CreateConeAngle0LimitAttr()
        cone_angle_0_limit_attr.Set(cone_angle_0_limit)
    if cone_angle_1_limit >= 0:
        cone_angle_1_limit_attr = joint.CreateConeAngle1LimitAttr()
        cone_angle_1_limit_attr.Set(cone_angle_1_limit)
    return joint


def search_spherical_joints_by_cone_angle(
    stage: Usd.Stage, min_angle: float, max_angle: float
) -> List[UsdPhysics.SphericalJoint]:
    """
    Search for spherical joints in the stage with cone angle limits within the specified range.

    Args:
        stage (Usd.Stage): The USD stage to search.
        min_angle (float): The minimum cone angle limit (inclusive).
        max_angle (float): The maximum cone angle limit (inclusive).

    Returns:
        List[UsdPhysics.SphericalJoint]: A list of spherical joints matching the criteria.
    """
    if min_angle > max_angle:
        raise ValueError("min_angle must be less than or equal to max_angle")
    spherical_joints = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsSphericalJoint":
            spherical_joint = UsdPhysics.SphericalJoint(prim)
            cone_angle_0_attr = spherical_joint.GetConeAngle0LimitAttr()
            cone_angle_1_attr = spherical_joint.GetConeAngle1LimitAttr()
            if cone_angle_0_attr.HasAuthoredValue() and cone_angle_1_attr.HasAuthoredValue():
                cone_angle_0 = cone_angle_0_attr.Get()
                cone_angle_1 = cone_angle_1_attr.Get()
                if min_angle <= cone_angle_0 <= max_angle and min_angle <= cone_angle_1 <= max_angle:
                    spherical_joints.append(spherical_joint)
    return spherical_joints


def create_spherical_joint_with_constraints(
    stage: Usd.Stage,
    joint_path: str,
    parent_path: str,
    child_path: str,
    axis: str = "X",
    cone_angle_0_limit: float = -1.0,
    cone_angle_1_limit: float = -1.0,
) -> UsdPhysics.SphericalJoint:
    """
    Create a spherical joint with cone angle limits between two prims.

    Args:
        stage (Usd.Stage): The USD stage.
        joint_path (str): The path for the joint prim.
        parent_path (str): The path for the parent prim.
        child_path (str): The path for the child prim.
        axis (str, optional): The primary joint axis. Defaults to "X".
        cone_angle_0_limit (float, optional): Cone limit from the primary joint axis toward the next axis. Defaults to -1.0 (not limited).
        cone_angle_1_limit (float, optional): Cone limit from the primary joint axis toward the second to next axis. Defaults to -1.0 (not limited).

    Returns:
        UsdPhysics.SphericalJoint: The created spherical joint.

    Raises:
        ValueError: If the parent or child prim does not exist.
    """
    parent_prim = stage.GetPrimAtPath(parent_path)
    if not parent_prim.IsValid():
        raise ValueError(f"Parent prim at path {parent_path} does not exist.")
    child_prim = stage.GetPrimAtPath(child_path)
    if not child_prim.IsValid():
        raise ValueError(f"Child prim at path {child_path} does not exist.")
    joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([parent_path])
    joint.CreateBody1Rel().SetTargets([child_path])
    joint.CreateAxisAttr().Set(axis)
    joint.CreateConeAngle0LimitAttr().Set(cone_angle_0_limit)
    joint.CreateConeAngle1LimitAttr().Set(cone_angle_1_limit)
    return joint


def update_spherical_joint(
    joint: UsdPhysics.SphericalJoint, axis: str = "X", cone_angle_0: float = -1.0, cone_angle_1: float = -1.0
) -> None:
    """Update the attributes of a spherical joint.

    Args:
        joint (UsdPhysics.SphericalJoint): The joint to update.
        axis (str, optional): The cone limit axis. Must be "X", "Y", or "Z". Defaults to "X".
        cone_angle_0 (float, optional): Cone limit from the primary joint axis toward the next axis.
            Defaults to -1.0 (not limited).
        cone_angle_1 (float, optional): Cone limit from the primary joint axis toward the second to next axis.
            Defaults to -1.0 (not limited).

    Raises:
        ValueError: If the provided joint is not a valid UsdPhysics.SphericalJoint.
        ValueError: If the provided axis is not "X", "Y", or "Z".
    """
    if not joint or not isinstance(joint, UsdPhysics.SphericalJoint):
        raise ValueError("Invalid spherical joint provided.")
    if axis not in ["X", "Y", "Z"]:
        raise ValueError("Invalid axis value. Must be 'X', 'Y', or 'Z'.")
    joint.GetAxisAttr().Set(axis)
    joint.GetConeAngle0LimitAttr().Set(cone_angle_0)
    joint.GetConeAngle1LimitAttr().Set(cone_angle_1)


def batch_update_spherical_joints(
    stage: Usd.Stage, prim_paths: List[str], axis: str, cone_angle_0_limit: float, cone_angle_1_limit: float
) -> None:
    """Update the attributes of multiple UsdPhysics.SphericalJoint prims in a batch.

    Args:
        stage (Usd.Stage): The stage containing the prims to update.
        prim_paths (List[str]): A list of paths to the UsdPhysics.SphericalJoint prims to update.
        axis (str): The cone limit axis. Must be 'X', 'Y', or 'Z'.
        cone_angle_0_limit (float): The cone limit from the primary joint axis toward the next axis, in degrees.
        cone_angle_1_limit (float): The cone limit from the primary joint axis toward the second to next axis, in degrees.
    """
    valid_axes = ["X", "Y", "Z"]
    if axis not in valid_axes:
        raise ValueError(f"Invalid axis value '{axis}'. Must be one of: {valid_axes}")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or not prim.IsA(UsdPhysics.SphericalJoint):
            print(f"Warning: Prim at path '{prim_path}' is not a valid UsdPhysics.SphericalJoint. Skipping.")
            continue
        joint = UsdPhysics.SphericalJoint(prim)
        axis_attr = joint.GetAxisAttr()
        if axis_attr.IsValid():
            axis_attr.Set(axis)
        cone_angle_0_attr = joint.GetConeAngle0LimitAttr()
        if cone_angle_0_attr.IsValid():
            cone_angle_0_attr.Set(cone_angle_0_limit)
        cone_angle_1_attr = joint.GetConeAngle1LimitAttr()
        if cone_angle_1_attr.IsValid():
            cone_angle_1_attr.Set(cone_angle_1_limit)


def copy_spherical_joint(stage: Usd.Stage, src_joint_path: str, dst_joint_path: str) -> UsdPhysics.SphericalJoint:
    """Copy a spherical joint to a new path in the stage.

    Args:
        stage (Usd.Stage): The stage containing the source joint.
        src_joint_path (str): The path of the source joint.
        dst_joint_path (str): The path where the joint will be copied.

    Returns:
        UsdPhysics.SphericalJoint: The newly created spherical joint.

    Raises:
        ValueError: If the source joint is not a valid spherical joint.
    """
    src_joint = UsdPhysics.SphericalJoint.Get(stage, src_joint_path)
    if not src_joint:
        raise ValueError(f"Prim at path {src_joint_path} is not a valid SphericalJoint")
    dst_prim = stage.GetPrimAtPath(dst_joint_path)
    if not dst_prim:
        dst_prim = stage.DefinePrim(dst_joint_path)
    dst_prim.SetTypeName(src_joint.GetPrim().GetTypeName())
    attrs_to_copy = [
        UsdPhysics.SphericalJoint.GetAxisAttr,
        UsdPhysics.SphericalJoint.GetConeAngle0LimitAttr,
        UsdPhysics.SphericalJoint.GetConeAngle1LimitAttr,
    ]
    for attr_func in attrs_to_copy:
        attr = attr_func(src_joint)
        if src_joint.GetPrim().HasAttribute(attr.GetName()):
            value = attr.Get()
            dst_attr = attr_func(UsdPhysics.SphericalJoint(dst_prim))
            dst_attr.Set(value)
    return UsdPhysics.SphericalJoint(dst_prim)


def delete_spherical_joint(stage: Usd.Stage, joint_path: str) -> bool:
    """Delete a spherical joint prim from the stage.

    Args:
        stage (Usd.Stage): The USD stage.
        joint_path (str): The path to the spherical joint prim.

    Returns:
        bool: True if the joint was deleted successfully, False otherwise.
    """
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid():
        print(f"No prim found at path: {joint_path}")
        return False
    if not UsdPhysics.SphericalJoint(joint_prim):
        print(f"Prim at path {joint_path} is not a valid SphericalJoint")
        return False
    stage.RemovePrim(joint_path)
    if stage.GetPrimAtPath(joint_path).IsValid():
        print(f"Failed to delete SphericalJoint prim at path: {joint_path}")
        return False
    return True


def get_spherical_joint_attributes(joint: UsdPhysics.SphericalJoint) -> Tuple[Tf.Enum, float, float]:
    """Get the attributes of a spherical joint.

    Args:
        joint (UsdPhysics.SphericalJoint): The spherical joint prim.

    Returns:
        Tuple[Tf.Enum, float, float]: A tuple containing the axis, cone angle 0 limit, and cone angle 1 limit.
    """
    axis_attr = joint.GetAxisAttr()
    if axis_attr.HasAuthoredValue():
        axis = axis_attr.Get()
    else:
        axis = UsdPhysics.Tokens.X
    cone_angle_0_attr = joint.GetConeAngle0LimitAttr()
    if cone_angle_0_attr.HasAuthoredValue():
        cone_angle_0 = cone_angle_0_attr.Get()
    else:
        cone_angle_0 = -1.0
    cone_angle_1_attr = joint.GetConeAngle1LimitAttr()
    if cone_angle_1_attr.HasAuthoredValue():
        cone_angle_1 = cone_angle_1_attr.Get()
    else:
        cone_angle_1 = -1.0
    return (axis, cone_angle_0, cone_angle_1)


def align_spherical_joint_with_target(joint: UsdPhysics.SphericalJoint, target_xform: Gf.Matrix4d) -> None:
    """
    Align the spherical joint's local space to the target transform.

    Args:
        joint (UsdPhysics.SphericalJoint): The spherical joint to align.
        target_xform (Gf.Matrix4d): The target transform matrix.
    """
    body0_rel = joint.GetBody0Rel()
    if not body0_rel.IsValid():
        raise ValueError("Joint's body0 relationship is not valid.")
    body0_targets = body0_rel.GetTargets()
    if not body0_targets:
        raise ValueError("Joint's body0 relationship has no targets.")
    body0_path = body0_targets[0]
    body0_prim = stage.GetPrimAtPath(body0_path)
    if not body0_prim.IsValid():
        raise ValueError("Joint's body0 prim is not valid.")
    xformable = UsdGeom.Xformable(body0_prim)
    if not xformable:
        raise ValueError("Joint's body0 prim is not transformable.")
    transform_op = xformable.MakeMatrixXform()
    transform_op.Set(target_xform)


def bake_physics_simulation(stage: Usd.Stage, start_time: float, end_time: float, time_samples: List[float]) -> bool:
    """Bake the physics simulation on the given stage from start_time to end_time with the specified time samples.

    Args:
        stage (Usd.Stage): The USD stage with the physics simulation to bake.
        start_time (float): The start time of the simulation.
        end_time (float): The end time of the simulation.
        time_samples (List[float]): The time samples at which to bake the simulation.

    Returns:
        bool: True if the baking succeeded, False otherwise.
    """
    physics_scene = UsdPhysics.Scene.Get(stage, "/physicsScene")
    if not physics_scene:
        print("No physics scene found in the stage.")
        return False
    try:
        physics_scene.GetStartTimeAttr().Set(start_time)
        physics_scene.GetEndTimeAttr().Set(end_time)
    except Tf.ErrorException as e:
        print(f"Error setting simulation time range: {e}")
        return False
    try:
        physics_scene.Simulate(time_samples)
    except Tf.ErrorException as e:
        print(f"Error baking physics simulation: {e}")
        return False
    baked_layer = Sdf.Layer.CreateNew("baked_simulation.usd")
    baked_stage = Usd.Stage.Open(baked_layer.identifier)
    for prim in stage.Traverse():
        if prim.IsActive() and (not prim.IsAbstract()):
            baked_stage.DefinePrim(prim.GetPath())
            for attr in prim.GetAttributes():
                if attr.ValueMightBeTimeVarying():
                    for time_sample in time_samples:
                        attr_value = attr.Get(time_sample)
                        if attr_value is not None:
                            baked_stage.GetAttributeAtPath(attr.GetPath()).Set(attr_value, time_sample)
    baked_stage.GetRootLayer().Save()
    return True


def configure_rigidbody(
    prim: Usd.Prim, density: float = 1000.0, dynamic: bool = True, kinematic: bool = False, start_asleep: bool = False
) -> UsdPhysics.RigidBodyAPI:
    """Configures a prim with RigidBodyAPI and sets common attributes.

    Args:
        prim (Usd.Prim): The prim to apply the RigidBodyAPI to.
        density (float): The density of the rigid body. Defaults to 1000.0.
        dynamic (bool): Whether the body is movable by forces. Defaults to True.
        kinematic (bool): Whether to make the body kinematic. Defaults to False.
        start_asleep (bool): Whether the body should start asleep. Defaults to False.

    Returns:
        UsdPhysics.RigidBodyAPI: The applied RigidBodyAPI schema.
    """
    rigidbody_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigidbody_api.CreateRigidBodyEnabledAttr().Set(dynamic)
    rigidbody_api.CreateKinematicEnabledAttr().Set(kinematic)
    rigidbody_api.CreateStartsAsleepAttr().Set(start_asleep)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateDensityAttr().Set(density)
    return rigidbody_api
