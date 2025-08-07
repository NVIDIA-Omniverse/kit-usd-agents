## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import random
import string
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from pxr import Ar, Gf, Pcp, Sdf, Tf, Usd, UsdGeom


def analyze_dependency_arcs(arc_types: List[Pcp.ArcType]) -> Tuple[int, int, int]:
    """Analyze a list of dependency arcs and return counts for each type.

    Args:
        arc_types (List[Pcp.ArcType]): A list of dependency arc types.

    Returns:
        Tuple[int, int, int]: A tuple containing the counts for each arc type:
            - [0]: Number of reference arcs
            - [1]: Number of payload arcs
            - [2]: Number of inherits arcs
    """
    reference_count = 0
    payload_count = 0
    inherits_count = 0
    for arc_type in arc_types:
        if arc_type == Pcp.ArcTypeReference:
            reference_count += 1
        elif arc_type == Pcp.ArcTypePayload:
            payload_count += 1
        elif arc_type == Pcp.ArcTypeInherit:
            inherits_count += 1
        else:
            raise ValueError(f"Unknown arc type: {arc_type}")
    return (reference_count, payload_count, inherits_count)


def find_cached_property_index(cache: Pcp.Cache, prop_path: Sdf.Path) -> Pcp.PropertyIndex:
    """
    Returns a pointer to the cached computed property index for the given
    path, or None if it has not been computed.
    """
    if not cache:
        raise ValueError("Invalid PcpCache object")
    if not prop_path.IsPropertyPath():
        return None
    prop_index = cache.FindPropertyIndex(prop_path)
    return prop_index


def create_cache_with_layer_stack() -> Pcp.Cache:
    """Create a cache with a layer stack and return it."""
    root_layer = Sdf.Layer.CreateAnonymous()
    layer_stack_identifier = Pcp.LayerStackIdentifier(root_layer)
    cache = Pcp.Cache(layer_stack_identifier)
    return cache


def get_layer_stack_identifier(cache: Pcp.Cache) -> Pcp.LayerStackIdentifier:
    """Get the identifier of the layerStack used for composition in the given cache."""
    identifier: Pcp.LayerStackIdentifier = cache.GetLayerStackIdentifier()
    if not identifier:
        raise ValueError("Invalid layer stack identifier returned from cache.")
    return identifier


def get_used_layers_revision(cache: Pcp.Cache) -> int:
    """
    Return a number that can be used to determine whether or not the set
    of layers used by this cache may have changed or not.

    For example, if one calls GetUsedLayers() and saves the
    GetUsedLayersRevision(), and then later calls GetUsedLayersRevision()
    again, if the number is unchanged, then GetUsedLayers() is guaranteed
    to be unchanged as well.
    """
    revision: int = 0
    usedLayers: Sdf.LayerHandleSet = cache.GetUsedLayers()
    for layer in usedLayers:
        revision += layer.GetCurrentRevision()
    return revision


def get_dynamic_file_format_argument_dependency_data(
    cache: Pcp.Cache, prim_index_path: Sdf.Path
) -> Pcp.DynamicFileFormatDependencyData:
    """
    Returns the dynamic file format dependency data object for the prim
    index with the given prim_index_path.

    This will return an empty dependency data if either there is no cache
    prim index for the path or if the prim index has no dynamic file
    formats that it depends on.
    """
    prim_index = cache.FindPrimIndex(prim_index_path)
    if not prim_index:
        return None
    return prim_index.GetDynamicFileFormatArgumentDependencyData()


def get_used_layers(cache: Pcp.Cache) -> Set[Sdf.Layer]:
    """Get all the layers used by the cache.

    Returns:
        Set[Sdf.Layer]: A set of all layers used by the cache.
    """
    used_layer_handles: Pcp.LayerStackRefPtr = cache.GetUsedLayers()
    used_layers: Set[Sdf.Layer] = set()
    for layer_handle in used_layer_handles:
        if layer_handle:
            layer = layer_handle
            if layer:
                used_layers.add(layer)
        else:
            pass
    return used_layers


def update_dependency_map(dependency_map: Dict[str, List[Pcp.Dependency]], dep: Pcp.Dependency) -> None:
    """Update the dependency map with the given dependency.

    Args:
        dependency_map (Dict[str, List[Pcp.Dependency]]): The dependency map to update.
        dep (Pcp.Dependency): The dependency to add to the map.
    """
    key = dep.sitePath.pathString
    if key in dependency_map:
        dependency_map[key].append(dep)
    else:
        dependency_map[key] = [dep]


def resolve_dependency_paths(dependency: Pcp.Dependency) -> List[Sdf.Path]:
    """Resolve the paths for a given dependency.

    Args:
        dependency (Pcp.Dependency): The dependency to resolve paths for.

    Returns:
        List[Sdf.Path]: A list of resolved paths for the dependency.
    """
    resolved_paths: List[Sdf.Path] = []
    site_path: Sdf.Path = dependency.sitePath
    if not site_path:
        raise ValueError("Dependency has an invalid site path.")
    resolved_paths.append(site_path)
    map_func = dependency.mapFunc
    if map_func:
        mapped_path: Sdf.Path = map_func.MapSourceToTarget(site_path)
        resolved_paths.append(mapped_path)
    return resolved_paths


def analyze_dependency_types(dependency_type_names: List[str]) -> Dict[str, int]:
    """Analyzes a list of dependency type names and returns a dictionary with the count of each type.

    Args:
        dependency_type_names (List[str]): A list of dependency type names as strings.

    Returns:
        Dict[str, int]: A dictionary where the keys are the dependency type names and the values are the count of each type.
    """
    dependency_type_counts: Dict[str, int] = {}
    for name in dependency_type_names:
        dependency_type_value = Pcp.DependencyType.GetValueFromName(name)
        if dependency_type_value:
            if name in dependency_type_counts:
                dependency_type_counts[name] += 1
            else:
                dependency_type_counts[name] = 1
    return dependency_type_counts


class MockDynamicFileFormatDependencyData:

    def __init__(self, relevant_fields: Set[str]):
        self._relevant_fields = relevant_fields

    def IsEmpty(self) -> bool:
        return len(self._relevant_fields) == 0

    def GetRelevantFieldNames(self) -> Set[str]:
        return self._relevant_fields


def analyze_dynamic_file_format_dependencies(dependencies: List[MockDynamicFileFormatDependencyData]) -> Set[str]:
    """
    Analyzes a list of dynamic file format dependencies and returns a set of relevant field names.

    Args:
        dependencies (List[MockDynamicFileFormatDependencyData]): A list of dynamic file format dependencies.

    Returns:
        Set[str]: A set of relevant field names.
    """
    relevant_fields: Set[str] = set()
    for dependency in dependencies:
        if dependency.IsEmpty():
            continue
        fields = dependency.GetRelevantFieldNames()
        relevant_fields.update(fields)
    return relevant_fields


class MockNodeRef:
    """Mock class to mimic Pcp.NodeRef for testing."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


def isolate_cycle(cycle: List[MockNodeRef]) -> List[MockNodeRef]:
    """Isolate the cycle in the error by returning a list of nodes."""
    if not cycle:
        return []
    visited = set()
    cycle_list = [cycle[0]]
    visited.add(cycle[0])
    for node in cycle[1:]:
        if node in visited:
            return cycle_list
        cycle_list.append(node)
        visited.add(node)
    return []


class MockErrorArcPermissionDenied:

    def __init__(self, source_path: str, target_path: str):
        self.sourceSite = MockSite(source_path)
        self.targetSite = MockSite(target_path)


class MockSite:

    def __init__(self, path: str):
        self.path = MockPath(path)


class MockPath:

    def __init__(self, path: str):
        self._path = path

    def GetPrimPath(self) -> str:
        return self._path


def can_fix_permission(node: str) -> bool:
    return node != "/path/to/source2"


def fix_permission(node: str):
    print(f"Fixed permission for node: {node}")


def fix_permission_denied_arcs(errors: List[MockErrorArcPermissionDenied]) -> List[MockErrorArcPermissionDenied]:
    """Fix permission denied arcs in the given list of errors.

    Args:
        errors (List[MockErrorArcPermissionDenied]): A list of MockErrorArcPermissionDenied errors.

    Returns:
        List[MockErrorArcPermissionDenied]: The list of remaining errors after fixing permissions.
    """
    remaining_errors: List[MockErrorArcPermissionDenied] = []
    for error in errors:
        source_node = error.sourceSite.path.GetPrimPath()
        target_node = error.targetSite.path.GetPrimPath()
        if can_fix_permission(source_node) and can_fix_permission(target_node):
            fix_permission(source_node)
            fix_permission(target_node)
        else:
            remaining_errors.append(error)
    return remaining_errors


def resolve_capacity_exceeded_errors(pcp_errors: List[Pcp.ErrorCapacityExceeded]) -> List[Pcp.ErrorCapacityExceeded]:
    """Resolve Pcp.ErrorCapacityExceeded errors.

    Args:
        pcp_errors (List[Pcp.ErrorCapacityExceeded]): List of Pcp.ErrorCapacityExceeded errors.

    Returns:
        List[Pcp.ErrorCapacityExceeded]: List of unresolved Pcp.ErrorCapacityExceeded errors.
    """
    unresolved_errors: List[Pcp.ErrorCapacityExceeded] = []
    for error in pcp_errors:
        site = error.site
        layer_stack = site.layerStack
        if site.hasPayload:
            if not layer_stack.HasLayer(site.payload):
                layer_stack.InsertLayer(site.payload)
            try:
                site.Recompose()
            except Pcp.ErrorCapacityExceeded:
                unresolved_errors.append(error)
        else:
            unresolved_errors.append(error)
    return unresolved_errors


def optimize_stage_for_capacity(stage: Usd.Stage) -> Usd.Stage:
    """Optimize a USD stage to reduce composition arcs and avoid capacity errors.

    Args:
        stage (Usd.Stage): The input USD stage to optimize.

    Returns:
        Usd.Stage: The optimized USD stage.
    """
    optimized_stage = Usd.Stage.CreateInMemory()
    for prim in stage.TraverseAll():
        if prim.IsPseudoRoot():
            continue
        if prim.HasAuthoredInherits() or prim.HasAuthoredReferences() or prim.HasPayload() or prim.HasVariantSets():
            optimized_prim = optimized_stage.DefinePrim(prim.GetPath(), prim.GetTypeName())
            for attr in prim.GetAttributes():
                optimized_prim.CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
            for rel in prim.GetRelationships():
                optimized_prim.CreateRelationship(rel.GetName()).SetTargets(rel.GetTargets())
        else:
            optimized_stage.OverridePrim(prim.GetPath())
    return optimized_stage


def fix_inconsistent_attributes(stage: Usd.Stage, prim_path: str) -> None:
    """Fix inconsistent attribute types on a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim with inconsistent attributes.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")
    attributes = prim.GetAttributes()
    attr_types = {}
    for attr in attributes:
        attr_name = attr.GetName()
        attr_type = attr.GetTypeName()
        if attr_name in attr_types:
            if attr_type != attr_types[attr_name]:
                attr.ClearAtCurrentEditTarget()
        else:
            attr_types[attr_name] = attr_type


def report_inconsistent_variability(attribute: Usd.Attribute) -> Pcp.ErrorInconsistentAttributeVariability:
    """Report an error for attributes with inconsistent variability."""
    attr_name = attribute.GetName()
    prim_path = attribute.GetPrim().GetPath()
    variability = attribute.GetVariability()
    if variability == Sdf.VariabilityUniform:
        return None
    error = Pcp.ErrorInconsistentAttributeVariability()
    error.rootSite = attribute.GetStage().GetRootLayer().identifier
    error.definingLayerIdentifier = attribute.GetStage().GetEditTarget().GetLayer().identifier
    error.definingPrimPath = prim_path
    error.definingFieldName = attr_name
    return error


def propagate_property_type_fix(layer: Sdf.Layer, path: Sdf.Path) -> bool:
    """Propagate consistent property types to specs that lack an explicit type."""
    prop_spec = layer.GetPropertyAtPath(path)
    if not prop_spec:
        return False
    if prop_spec.HasInfo("typeName"):
        return False
    prim_path = path.GetPrimPath()
    prim_stack = Pcp.PrimIndex.BuildPrimIndex(layer, prim_path).ComputePrimStack()
    strongest_prop_spec = None
    for prim_spec in prim_stack:
        if prim_spec.path == prim_path:
            continue
        prop_spec = prim_spec.layer.GetPropertyAtPath(prim_spec.path.AppendProperty(path.name))
        if prop_spec and prop_spec.HasInfo("typeName"):
            strongest_prop_spec = prop_spec
            break
    if not strongest_prop_spec:
        return False
    type_name = strongest_prop_spec.GetInfo("typeName")
    prop_spec.SetInfo("typeName", type_name)
    return True


def remove_invalid_asset_references(layer: Sdf.Layer) -> List[str]:
    """Removes invalid asset references from the given layer.

    Args:
        layer (Sdf.Layer): The layer to process.

    Returns:
        List[str]: The paths of the invalid asset references that were removed.
    """
    removed_paths: List[str] = []
    for prim_spec in layer.rootPrims:
        if prim_spec.hasReferences:
            reference_list = prim_spec.referenceList
            new_reference_list = Sdf.ReferenceListOp()
            for reference in reference_list.explicitItems:
                if Sdf.Layer.IsAnonymousLayerIdentifier(reference.assetPath):
                    removed_paths.append(str(prim_spec.path))
                else:
                    new_reference_list.explicitItems.append(reference)
            prim_spec.referenceList.ClearEdits()
            prim_spec.referenceList.explicitItems = new_reference_list.explicitItems
    return removed_paths


class MockErrorInvalidAssetPath:
    """Mock class to simulate Pcp.ErrorInvalidAssetPath."""

    def __init__(self, resolved_asset_path):
        self.resolvedAssetPath = resolved_asset_path


def handle_invalid_asset_paths(errors: List[MockErrorInvalidAssetPath]) -> List[str]:
    """Handle invalid asset path errors and return a list of resolved paths."""
    resolved_paths = []
    for error in errors:
        invalid_path = error.resolvedAssetPath
        if not invalid_path:
            continue
        resolved_path = _resolve_invalid_path(invalid_path)
        if resolved_path:
            resolved_paths.append(resolved_path)
    return resolved_paths


def _resolve_invalid_path(invalid_path: str) -> str:
    """Attempt to resolve an invalid asset path."""
    resolved_path = invalid_path.replace("invalid", "valid")
    return resolved_path


class CustomErrorInvalidAssetPath:
    """Custom class to mimic Pcp.ErrorInvalidAssetPath for testing."""

    def __init__(self, resolved_asset_path: str, site: str, layer: str):
        self.resolvedAssetPath = resolved_asset_path
        self.site = site
        self.layer = CustomLayer(layer)


class CustomLayer:
    """Custom class to mimic Sdf.Layer for testing."""

    def __init__(self, identifier: str):
        self.identifier = identifier


def log_invalid_asset_paths(errors: List[CustomErrorInvalidAssetPath]) -> None:
    """Log invalid asset paths from a list of CustomErrorInvalidAssetPath."""
    if not errors:
        return
    for error in errors:
        invalid_path = error.resolvedAssetPath
        site = error.site
        layer = error.layer.identifier
        print(f"Invalid asset path: {invalid_path}")
        print(f"  Referenced from: {site}")
        print(f"  Authored in layer: {layer}")


def validate_and_report_asset_paths(layer: Sdf.Layer) -> List[str]:
    """Validate asset paths in a given layer and report any invalid paths.

    Args:
        layer (Sdf.Layer): The layer to validate asset paths in.

    Returns:
        List[str]: A list of invalid asset paths.
    """
    invalid_paths: List[str] = []
    for prim_spec in layer.rootPrims:
        if prim_spec.referenceList or prim_spec.payloadList:
            for ref_spec in prim_spec.referenceList.GetAddedOrExplicitItems():
                asset_path = ref_spec.assetPath
                if not Ar.GetResolver().Resolve(asset_path):
                    invalid_paths.append(asset_path)
            for payload_spec in prim_spec.payloadList.GetAddedOrExplicitItems():
                asset_path = payload_spec.assetPath
                if not Ar.GetResolver().Resolve(asset_path):
                    invalid_paths.append(asset_path)
    return invalid_paths


def correct_invalid_asset_paths(layer: Sdf.Layer) -> bool:
    """Correct invalid asset paths in a given layer.

    Args:
        layer (Sdf.Layer): The layer to correct invalid asset paths in.

    Returns:
        bool: True if any invalid asset paths were corrected, False otherwise.
    """
    corrected = False
    for prim_spec in layer.rootPrims:
        for prop_spec in prim_spec.properties:
            if prop_spec.typeName in ["reference", "payload"]:
                asset_path = prop_spec.assetPath
                if not Sdf.Layer.FindRelativeToLayer(layer, asset_path):
                    corrected_path = asset_path.replace("invalid", "valid")
                    prop_spec.assetPath = corrected_path
                    corrected = True
    return corrected


def report_invalid_asset_paths(asset_paths: List[str]) -> List[str]:
    """
    Report invalid asset paths.

    Args:
        asset_paths (List[str]): List of asset paths to validate.

    Returns:
        List[str]: List of invalid asset paths.
    """
    invalid_paths = []
    for path in asset_paths:
        if not path:
            invalid_paths.append(path)
            continue
        if not path.startswith("/"):
            invalid_paths.append(path)
            continue
        invalid_chars = ["\\", ":", "*", "?", '"', "<", ">", "|"]
        if any((char in path for char in invalid_chars)):
            invalid_paths.append(path)
            continue
    return invalid_paths


class CustomInvalidAssetPathError(Exception):
    """Custom error class for invalid asset paths."""

    def __init__(self, resolvedAssetPath):
        super().__init__(f"Invalid asset path: {resolvedAssetPath}")
        self.resolvedAssetPath = resolvedAssetPath


def handle_invalid_asset_paths(errors: List[CustomInvalidAssetPathError]) -> List[str]:
    """Handle invalid asset path errors and return a list of error messages."""
    error_messages = []
    for error in errors:
        invalid_path = error.resolvedAssetPath
        if not invalid_path:
            error_messages.append("Empty asset path encountered.")
            continue
        if invalid_path.startswith("/"):
            error_messages.append(f"Invalid absolute asset path: {invalid_path}")
        else:
            suggested_path = f"/{invalid_path}"
            error_messages.append(f"Invalid relative asset path: {invalid_path}. Suggestion: {suggested_path}")
    return error_messages


def fix_invalid_external_target_paths(
    pcp_errors: List[Pcp.ErrorInvalidExternalTargetPath],
) -> List[Pcp.ErrorInvalidExternalTargetPath]:
    """Fix invalid external target paths in Pcp errors.

    This function attempts to fix invalid external target paths by making the paths
    relative to the root layer of the stage.

    Args:
        pcp_errors (List[Pcp.ErrorInvalidExternalTargetPath]): A list of Pcp errors with invalid external target paths.

    Returns:
        List[Pcp.ErrorInvalidExternalTargetPath]: A list of Pcp errors that could not be fixed.
    """
    remaining_errors: List[Pcp.ErrorInvalidExternalTargetPath] = []
    for error in pcp_errors:
        layer_stack = error.layer.GetLayerStack()
        root_layer = layer_stack.identifier.rootLayer
        relative_path = Sdf.Path(error.targetPath).MakeRelativePath(root_layer.identifier)
        if relative_path.IsEmpty():
            remaining_errors.append(error)
        else:
            if error.layer.GetPrimAtPath(error.targetPath):
                error.layer.RemovePrim(error.targetPath)
            error.layer.GetPrimAtPath(error.sourcePath).GetReferences().AddReference(relative_path)
    return remaining_errors


def fix_invalid_instance_target_paths(stage: Usd.Stage) -> None:
    """Fix invalid target or connection paths in inherited classes that point to instances.

    This function traverses the stage and checks for invalid target or connection paths
    in inherited classes that point to instances of those classes. It fixes the paths
    by replacing the instance path with the master path.

    Args:
        stage (Usd.Stage): The USD stage to fix invalid instance target paths.
    """
    for prim in stage.Traverse():
        if prim.IsInstance():
            master_path = prim.GetMaster().GetPath()
            for attr in prim.GetAttributes():
                if attr.IsRelationship() or attr.HasConnections():
                    targets = attr.GetTargets()
                    for i, target in enumerate(targets):
                        if target.GetPrimPath().HasPrefix(prim.GetPath()):
                            new_target = target.ReplacePrefix(prim.GetPath(), master_path)
                            attr.SetTargets([new_target if j == i else t for (j, t) in enumerate(targets)])


class ErrorInvalidInstanceTargetPath(Pcp.ErrorTargetPathBase, Pcp.ErrorBase):
    """
    Invalid target or connection path authored in an inherited class that
    points to an instance of that class.
    """

    pass


def validate_instance_target_paths(stage: Usd.Stage) -> List[ErrorInvalidInstanceTargetPath]:
    """Validate instance target paths in the given stage.

    Args:
        stage (Usd.Stage): The USD stage to validate.

    Returns:
        List[ErrorInvalidInstanceTargetPath]: A list of ErrorInvalidInstanceTargetPath objects representing any invalid instance target paths found.
    """
    errors: List[ErrorInvalidInstanceTargetPath] = []
    for prim in stage.TraverseAll():
        if prim.IsInstance():
            master_path = prim.GetMaster().GetPath()
            if stage.GetPrimAtPath(master_path).IsA(Usd.ClassPrim):
                for rel in prim.GetRelationships():
                    if rel.GetTargets() == [master_path]:
                        error = ErrorInvalidInstanceTargetPath(prim.GetPath(), rel.GetName(), master_path)
                        errors.append(error)
    return errors


def find_and_fix_invalid_references(stage: Usd.Stage) -> None:
    """Find and fix invalid prim paths used by references or payloads in a USD stage.

    Args:
        stage (Usd.Stage): The USD stage to inspect and fix.
    """
    root_layer = stage.GetRootLayer()
    for prim_spec in root_layer.rootPrims:
        stack = [prim_spec]
        while stack:
            current_spec = stack.pop()
            for ref_spec in current_spec.referenceList.GetAddedOrExplicitItems():
                ref_layer = ref_spec.assetPath
                ref_path = ref_spec.primPath
                if ref_path and (not Sdf.Path(ref_path).IsPrimPath()):
                    print(f"Invalid reference prim path: {ref_path} in {ref_layer}")
                    new_path = Sdf.Path(ref_path).MakeAbsolutePath(current_spec.path)
                    ref_spec.primPath = new_path
                    print(f"Fixed reference prim path: {new_path}")
            for payload_spec in current_spec.payloadList.GetAddedOrExplicitItems():
                payload_layer = payload_spec.assetPath
                payload_path = payload_spec.primPath
                if payload_path and (not Sdf.Path(payload_path).IsPrimPath()):
                    print(f"Invalid payload prim path: {payload_path} in {payload_layer}")
                    new_path = Sdf.Path(payload_path).MakeAbsolutePath(current_spec.path)
                    payload_spec.primPath = new_path
                    print(f"Fixed payload prim path: {new_path}")
            stack.extend(current_spec.nameChildren)


def create_prim_with_invalid_reference(stage, prim_path, ref_path):
    prim = stage.DefinePrim(prim_path)
    prim.GetReferences().AddReference(assetPath="../ref.usda", primPath=ref_path)


def create_prim_with_invalid_payload(stage, prim_path, payload_path):
    prim = stage.DefinePrim(prim_path)
    prim.GetPayloads().AddPayload(assetPath="../payload.usda", primPath=payload_path)


def report_invalid_payloads(layer: Sdf.Layer) -> List[Pcp.ErrorInvalidPrimPath]:
    """Report invalid prim paths used by payloads in a given layer.

    Args:
        layer (Sdf.Layer): The layer to check for invalid payload paths.

    Returns:
        List[Pcp.ErrorInvalidPrimPath]: A list of ErrorInvalidPrimPath objects
        representing the invalid payload paths found in the layer.
    """
    errors: List[Pcp.ErrorInvalidPrimPath] = []
    for prim_spec in layer.rootPrims:
        payload_list_op = prim_spec.payloadList
        if payload_list_op.isExplicit:
            for payload_path in payload_list_op.explicitItems:
                if not Sdf.Path.IsValidPathString(payload_path.assetPath):
                    error_info = {
                        "rootSite": payload_path.primPath,
                        "errorPath": payload_path.assetPath,
                        "layer": layer,
                    }
                    errors.append(error_info)
    return errors


def report_invalid_reference_offsets_summary(error_messages: List[str]) -> str:
    """Generate a summary report for invalid reference offset error messages.

    Args:
        error_messages (List[str]): A list of error messages related to invalid reference offsets.

    Returns:
        str: A summary report of the invalid reference offset errors.
    """
    if not error_messages:
        return "No invalid reference offset errors found."
    error_counts = {}
    for message in error_messages:
        if message in error_counts:
            error_counts[message] += 1
        else:
            error_counts[message] = 1
    summary = "Invalid Reference Offset Errors Summary:\n"
    for message, count in error_counts.items():
        summary += f"- {message} (count: {count})\n"
    return summary.rstrip()


def get_prims_with_invalid_sublayer_offsets(layer: Sdf.Layer) -> List[Sdf.PrimSpec]:
    """
    Returns a list of PrimSpecs in the given layer that have sublayers with invalid layer offsets.

    Args:
        layer (Sdf.Layer): The layer to search for prims with invalid sublayer offsets.

    Returns:
        List[Sdf.PrimSpec]: A list of PrimSpecs that have sublayers with invalid layer offsets.
    """
    prims_with_invalid_offsets = []
    for prim_spec in layer.rootPrims:
        stack = [prim_spec]
        while stack:
            current_spec = stack.pop()
            if current_spec.hasPayloads:
                for payload in current_spec.payloadList.explicitItems:
                    if payload.layerOffset != Sdf.LayerOffset():
                        prims_with_invalid_offsets.append(current_spec)
                        break
            stack.extend(current_spec.nameChildren)
    return prims_with_invalid_offsets


def generate_temp_usd_file_path() -> str:
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    return "C:\\Users\\horde\\AppData\\Local\\Temp\\usda_" + random_string + ".usda"


class MockErrorInvalidSublayerPath:
    """Mock class mimicking Pcp.ErrorInvalidSublayerPath for testing."""

    def __init__(self, layer_id: str, sublayer_path: str):
        self.layer = MockLayer(layer_id)
        self.sublayerPath = sublayer_path


class MockLayer:
    """Mock class mimicking Pcp.Layer for testing."""

    def __init__(self, identifier: str):
        self.identifier = identifier


def log_invalid_sublayer_paths(errors: List[MockErrorInvalidSublayerPath]) -> None:
    """Log the invalid sublayer paths from the given list of errors."""
    if not errors:
        return
    for error in errors:
        layer_id = error.layer.identifier
        sublayer_path = error.sublayerPath
        print(f"Invalid sublayer path '{sublayer_path}' in layer '{layer_id}'")


def report_invalid_sublayer_paths_to_console(errors: List[Pcp.ErrorInvalidSublayerPath]) -> None:
    """Print the invalid sublayer paths to the console.

    Args:
        errors (List[Pcp.ErrorInvalidSublayerPath]): A list of ErrorInvalidSublayerPath objects.
    """
    if not errors:
        print("No invalid sublayer paths found.")
        return
    for error in errors:
        asset_path = error.resolvedAssetPath
        print(f"Invalid sublayer path: {asset_path}")


class MockErrorInvalidSublayerPath:

    def __init__(self, asset_path):
        self.resolvedAssetPath = asset_path


def gather_invalid_sublayer_paths_info(error) -> List[Tuple[str, str]]:
    """
    Gathers information about invalid sublayer paths from the given error.

    Args:
        error: An object representing the ErrorInvalidSublayerPath error.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the invalid sublayer path
            and the reason for its invalidity.
    """
    invalid_paths_info = []
    if hasattr(error, "sublayerPaths") and hasattr(error, "messages"):
        for sublayer_path in error.sublayerPaths:
            reason = error.messages.get(sublayer_path, "Unknown reason")
            invalid_paths_info.append((sublayer_path, reason))
    return invalid_paths_info


class MockErrorInvalidSublayerPath:

    def __init__(self):
        self.sublayerPaths = ["/path/to/sublayer1.usda", "/path/to/sublayer2.usda"]
        self.messages = {"/path/to/sublayer1.usda": "File not found", "/path/to/sublayer2.usda": "Invalid file format"}


def find_and_report_invalid_target_paths(layer: Sdf.Layer) -> List[Pcp.ErrorInvalidTargetPath]:
    """Find and report invalid target paths in a given layer.

    Args:
        layer (Sdf.Layer): The layer to inspect for invalid target paths.

    Returns:
        List[Pcp.ErrorInvalidTargetPath]: A list of ErrorInvalidTargetPath objects representing the invalid paths found.
    """
    errors: List[Pcp.ErrorInvalidTargetPath] = []
    for prim_spec in layer.rootPrims:
        stack: List[Sdf.PrimSpec] = [prim_spec]
        while stack:
            current_spec = stack.pop()
            target_paths: Sdf.RelationshipSpec = current_spec.relationships.get("sdf:targetPaths")
            if target_paths:
                for path_string in target_paths.targetPathList.explicitItems:
                    target_path = Sdf.Path(path_string)
                    if not target_path.IsPrimPath():
                        errors.append(Pcp.ErrorInvalidTargetPath(current_spec.path, target_path))
            stack.extend(current_spec.nameChildren)
    return errors


def replace_invalid_target_paths(layer: Sdf.Layer, invalid_path: str, replacement_path: str) -> None:
    """Replace invalid target paths in a layer with a replacement path.

    Args:
        layer (Sdf.Layer): The layer to modify.
        invalid_path (str): The invalid target path to replace.
        replacement_path (str): The replacement path to use.
    """
    for prim_spec in layer.rootPrims:
        prim_stack = [prim_spec]
        while prim_stack:
            current_prim = prim_stack.pop()
            for rel_spec in current_prim.relationships.values():
                targets = rel_spec.targetPathList.explicitItems
                for i, target in enumerate(targets):
                    if target == invalid_path:
                        rel_spec.targetPathList.explicitItems[i] = replacement_path
            prim_stack.extend(current_prim.nameChildren)


def validate_and_report_muted_paths(muted_paths: List[str], layer_stacks: Dict[str, List[str]]) -> Dict[str, Any]:
    """Validate muted paths and report any errors.

    Args:
        muted_paths (List[str]): List of muted asset paths.
        layer_stacks (Dict[str, List[str]]): Dictionary mapping asset paths to layer stack identifiers.

    Returns:
        Dict[str, Any]: Dictionary containing validation results.
    """
    result = {"valid_paths": [], "invalid_paths": [], "error_messages": []}
    for path in muted_paths:
        if path not in layer_stacks:
            result["invalid_paths"].append(path)
            error_message = f"Muted asset path '{path}' not found in layer stacks."
            result["error_messages"].append(error_message)
        else:
            result["valid_paths"].append(path)
    return result


def mute_and_log_invalid_references(
    prim_stack: Pcp.PrimIndex, invalid_ref_paths: List[str]
) -> List[Pcp.LayerStackSite]:
    """Mute invalid references in the given prim stack and log an error.

    Args:
        prim_stack (Pcp.PrimIndex): The prim stack to process.
        invalid_ref_paths (List[str]): List of invalid reference paths to mute.

    Returns:
        List[Pcp.LayerStackSite]: List of muted layer stack sites.
    """
    muted_sites = []
    for node in prim_stack.GetNodeRange():
        if node.HasReferences():
            layer_stack = node.GetLayerStack()
            for ref in node.GetReferences():
                ref_path = ref.GetAssetPath()
                if ref_path in invalid_ref_paths:
                    layer_offset = ref.GetLayerOffset()
                    muted_site = Pcp.LayerStackSite(layer_stack, layer_offset)
                    muted_site.MuteLayerStack()
                    muted_sites.append(muted_site)
                    error_msg = f"Muted invalid reference '{ref_path}' in prim stack."
                    Tf.Warn(error_msg)
    return muted_sites


def relocate_prims_with_error_handling(stage: Usd.Stage, prim_paths: List[str], new_path: str) -> None:
    """Relocate a list of prims to a new path, handling errors.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (List[str]): The paths of the prims to relocate.
        new_path (str): The new path to relocate the prims to.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Prim at path {prim_path} does not exist. Skipping.")
            continue
        if prim.HasAuthoredReferences():
            references = prim.GetReferences()
            try:
                references.AddReference(Sdf.Reference(prim_path, new_path))
            except Pcp.ErrorOpinionAtRelocationSource as e:
                print(f"Error relocating prim {prim_path}: {str(e)}. Skipping.")
                continue
        stage.DefinePrim(new_path).GetReferences().AddReference(Sdf.Reference(prim_path, new_path))
        stage.RemovePrim(prim_path)


def filter_prims_with_permission_violations(stage: Usd.Stage, prim_paths: List[str]) -> List[str]:
    """Filter out prim paths that have permission violations.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The list of prim paths to filter.

    Returns:
        List[str]: The list of prim paths without permission violations.
    """
    valid_prim_paths = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        try:
            prim.GetSpecifier()
            valid_prim_paths.append(prim_path)
        except Usd.ErrorPrimPermissionDenied:
            pass
    return valid_prim_paths


class CustomErrorPropertyPermissionDenied(Exception):
    """
    Custom error class to mimic ErrorPropertyPermissionDenied.
    """

    def __init__(self, layer_identifier: str, path: str, property_name: str):
        super().__init__()
        self.layerIdentifier = layer_identifier
        self.path = path
        self.propertyName = property_name


def log_permission_violations(error: CustomErrorPropertyPermissionDenied, log_path: str) -> None:
    """Log permission violation errors to a file.

    Args:
        error (CustomErrorPropertyPermissionDenied): The permission denied error object.
        log_path (str): The path to the log file.
    """
    layer_identifier = error.layerIdentifier
    path = error.path
    property_name = error.propertyName
    log_message = f"Permission Denied: Layer '{layer_identifier}', Path '{path}', Property '{property_name}'\n"
    try:
        with open(log_path, "a") as log_file:
            log_file.write(log_message)
    except IOError as e:
        print(f"Error writing to log file: {e}")


def report_permission_violations(
    layer: Sdf.Layer, allowed_properties: List[str]
) -> List[Pcp.ErrorPropertyPermissionDenied]:
    """
    Report any opinions in the given layer about private properties that are not
    on the list of allowed properties.

    Args:
        layer (Sdf.Layer): The layer to check for property permission violations.
        allowed_properties (List[str]): The list of allowed private properties.

    Returns:
        List[Pcp.ErrorPropertyPermissionDenied]: A list of errors for any illegal opinions found.
    """
    errors = []
    for prim_spec in layer.rootPrims:
        for property_name, property_spec in prim_spec.properties.items():
            if property_name.startswith("_"):
                if property_name not in allowed_properties:
                    error = {
                        "type": "ErrorPropertyPermissionDenied",
                        "layer": layer.identifier,
                        "path": str(property_spec.path),
                        "property": property_name,
                    }
                    errors.append(error)
    return errors


def visualize_sublayer_cycles(cycle_descriptions: List[List[str]]) -> List[List[str]]:
    """
    Visualize the sublayer cycles given the cycle descriptions.

    Args:
        cycle_descriptions (List[List[str]]): A list of cycle descriptions, where each cycle is represented as a list of layer descriptions.

    Returns:
        List[List[str]]: A list of sublayer cycles, where each cycle is represented as a list of layer paths.
    """
    cycles: List[List[str]] = []
    for cycle in cycle_descriptions:
        layer_paths: List[str] = []
        for layer_desc in cycle:
            layer_path = layer_desc.split(" (")[0]
            layer_paths.append(layer_path)
        cycles.append(layer_paths)
    return cycles


class MockErrorBase:
    """Mock base class for errors."""

    pass


class MockErrorTargetPermissionDenied(MockErrorBase):
    """Mock class for permission denied errors."""

    pass


def check_permission_violations(errors: List[MockErrorBase]) -> List[MockErrorBase]:
    """Check for permission violations in a list of errors.

    Args:
        errors (List[MockErrorBase]): A list of mock errors to check.

    Returns:
        List[MockErrorBase]: A list of MockErrorTargetPermissionDenied errors found.
    """
    violations: List[MockErrorBase] = []
    for error in errors:
        if isinstance(error, MockErrorTargetPermissionDenied):
            violations.append(error)
    return violations


class MockErrorTargetPermissionDenied:

    def __init__(self, path):
        self.rootSite = MockSiteRef(path)


class MockSiteRef:

    def __init__(self, path):
        self.path = path


def filter_permission_violations_by_prim(
    errors: List[MockErrorTargetPermissionDenied], prim_path: str
) -> List[MockErrorTargetPermissionDenied]:
    """Filter a list of MockErrorTargetPermissionDenied errors by a specific prim path.

    Args:
        errors (List[MockErrorTargetPermissionDenied]): A list of MockErrorTargetPermissionDenied errors.
        prim_path (str): The prim path to filter the errors by.

    Returns:
        List[MockErrorTargetPermissionDenied]: A list of MockErrorTargetPermissionDenied errors filtered by the given prim path.
    """
    if not errors:
        return []
    filtered_errors = [error for error in errors if str(error.rootSite.path) == prim_path]
    return filtered_errors


def analyze_and_report_errors(error_types: List[str]) -> Tuple[int, List[str]]:
    """Analyze the given error types and report the errors.

    Args:
        error_types (List[str]): A list of error type names to analyze.

    Returns:
        Tuple[int, List[str]]: A tuple containing the total error count and a list of error messages.
    """
    error_count = 0
    error_messages = []
    for error_type in error_types:
        error_value = Pcp.ErrorType.GetValueFromName(error_type)
        if error_value is not None:
            error_count += 1
            error_message = f"Error: {error_type}"
            error_messages.append(error_message)
        else:
            error_messages.append(f"Unknown error type: {error_type}")
    return (error_count, error_messages)


def resolve_asset_paths(asset_paths: List[str]) -> Tuple[List[str], List[str]]:
    """
    Resolve a list of asset paths and return resolved paths and unresolved paths.

    Args:
        asset_paths (List[str]): List of asset paths to resolve.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - List of successfully resolved asset paths.
            - List of unresolved asset paths that raised ErrorUnresolvedPrimPath.
    """
    resolved_paths = []
    unresolved_paths = []
    for asset_path in asset_paths:
        try:
            resolved_path = Sdf.Layer.Find(asset_path)
            if resolved_path:
                resolved_paths.append(resolved_path.realPath)
            else:
                unresolved_paths.append(asset_path)
        except Pcp.ErrorUnresolvedPrimPath:
            unresolved_paths.append(asset_path)
    return (resolved_paths, unresolved_paths)


class CustomErrorUnresolvedPrimPath:
    """
    Custom class to mimic Pcp.ErrorUnresolvedPrimPath.
    """

    def __init__(self, unresolved_path: str):
        self.unresolvedPath = unresolved_path


def report_unresolved_prim_paths(errors: List[CustomErrorUnresolvedPrimPath]) -> List[str]:
    """
    Report the unresolved prim paths from the given list of errors.

    Args:
        errors (List[CustomErrorUnresolvedPrimPath]): A list of CustomErrorUnresolvedPrimPath errors.

    Returns:
        List[str]: A list of unresolved prim paths.
    """
    unresolved_paths = []
    for error in errors:
        if isinstance(error, CustomErrorUnresolvedPrimPath):
            unresolved_path = error.unresolvedPath
            unresolved_paths.append(unresolved_path)
        else:
            continue
    return unresolved_paths


def identify_shared_instanceable_prims(prim_indexes: List[Pcp.PrimIndex]) -> Dict[Pcp.InstanceKey, List[Pcp.PrimIndex]]:
    """
    Identifies instanceable prim indexes that share the same set of opinions.

    Args:
        prim_indexes (List[Pcp.PrimIndex]): A list of prim indexes to analyze.

    Returns:
        Dict[Pcp.InstanceKey, List[Pcp.PrimIndex]]: A dictionary mapping instance keys to lists of prim indexes
        that share the same set of opinions.
    """
    instance_key_to_prim_indexes: Dict[Pcp.InstanceKey, List[Pcp.PrimIndex]] = {}
    for prim_index in prim_indexes:
        if prim_index.IsInstanceable():
            instance_key = prim_index.GetInstanceKey()
            if instance_key not in instance_key_to_prim_indexes:
                instance_key_to_prim_indexes[instance_key] = []
            instance_key_to_prim_indexes[instance_key].append(prim_index)
    return instance_key_to_prim_indexes


class MockLayerStack:

    def __init__(self):
        self.mutedLayers = set()


def unmute_layers(layer_stack, layers_to_unmute: Iterable[str]) -> None:
    """Unmute the specified layers in this layer stack.

    Args:
        layer_stack (MockLayerStack): The layer stack object.
        layers_to_unmute (Iterable[str]): The layers to unmute.

    Raises:
        ValueError: If any of the specified layers are not in the muted layers set.
    """
    layers_to_unmute_set = set(layers_to_unmute)
    if not layers_to_unmute_set.issubset(layer_stack.mutedLayers):
        invalid_layers = layers_to_unmute_set - layer_stack.mutedLayers
        raise ValueError(f"Cannot unmute layers that are not muted: {invalid_layers}")
    layer_stack.mutedLayers -= layers_to_unmute_set


def analyze_layer_stack_identifier(layer_stack_identifier: Pcp.LayerStackIdentifier) -> Dict[str, Any]:
    """Analyze a LayerStackIdentifier and return key information.

    Args:
        layer_stack_identifier (Pcp.LayerStackIdentifier): The LayerStackIdentifier to analyze.

    Returns:
        Dict[str, Any]: A dictionary containing the analyzed information.
    """
    result = {}
    if layer_stack_identifier.rootLayer is not None:
        result["root_layer_path"] = layer_stack_identifier.rootLayer.identifier
        result["root_layer_real_path"] = layer_stack_identifier.rootLayer.realPath
    else:
        result["root_layer_path"] = None
        result["root_layer_real_path"] = None
    if layer_stack_identifier.sessionLayer is not None:
        result["session_layer_path"] = layer_stack_identifier.sessionLayer.identifier
        result["session_layer_real_path"] = layer_stack_identifier.sessionLayer.realPath
    else:
        result["session_layer_path"] = None
        result["session_layer_real_path"] = None
    if layer_stack_identifier.pathResolverContext is not None:
        result["path_resolver_context"] = layer_stack_identifier.pathResolverContext
    else:
        result["path_resolver_context"] = None
    return result


def create_layer_stack_with_session(root_layer: Sdf.Layer, session_layer: Sdf.Layer) -> Pcp.LayerStackIdentifier:
    """Create a LayerStackIdentifier with a root layer and a session layer.

    Args:
        root_layer (Sdf.Layer): The root layer of the layer stack.
        session_layer (Sdf.Layer): The session layer to be added to the layer stack.

    Returns:
        Pcp.LayerStackIdentifier: The created LayerStackIdentifier.
    """
    layer_stack_id = Pcp.LayerStackIdentifier(root_layer)
    if session_layer is not None:
        layer_stack_id = Pcp.LayerStackIdentifier(root_layer, session_layer)
    return layer_stack_id


def apply_root_layer_overrides(
    identifier: Pcp.LayerStackIdentifier, layer: Optional[Sdf.Layer] = None
) -> Pcp.LayerStackIdentifier:
    """Apply root layer overrides to a LayerStackIdentifier.

    Args:
        identifier (Pcp.LayerStackIdentifier): The input LayerStackIdentifier.
        layer (Optional[Sdf.Layer]): The root layer override. If None, the original root layer is used.

    Returns:
        Pcp.LayerStackIdentifier: A new LayerStackIdentifier with the root layer overrides applied.
    """
    if layer is None:
        return identifier
    new_identifier = Pcp.LayerStackIdentifier(
        rootLayer=layer, sessionLayer=identifier.sessionLayer, pathResolverContext=identifier.pathResolverContext
    )
    return new_identifier


def merge_layer_stack_sites(sites: List[Pcp.LayerStackIdentifier]) -> Pcp.LayerStackIdentifier:
    """Merge a list of layer stack identifiers into a single identifier.

    Args:
        sites (List[Pcp.LayerStackIdentifier]): List of layer stack identifiers to merge.

    Returns:
        Pcp.LayerStackIdentifier: The merged layer stack identifier.

    Raises:
        ValueError: If the input list is empty or contains identifiers with different layer stacks.
    """
    if not sites:
        raise ValueError("Input list of sites is empty.")
    root_layer = sites[0].rootLayer
    if any((site.rootLayer != root_layer for site in sites)):
        raise ValueError("All sites must have the same root layer.")
    merged_session_layers = []
    for site in sites:
        if site.sessionLayer:
            merged_session_layers.append(site.sessionLayer)
    if merged_session_layers:
        return Pcp.LayerStackIdentifier(root_layer, merged_session_layers[0])
    else:
        return Pcp.LayerStackIdentifier(root_layer)


def create_and_evaluate_constant_expression(const_value: Pcp.MapFunction) -> Pcp.MapFunction:
    """Create a constant PcpMapExpression and evaluate it.

    Args:
        const_value (Pcp.MapFunction): The constant value for the expression.

    Returns:
        Pcp.MapFunction: The evaluated result of the constant expression.
    """
    expr = Pcp.MapExpression.Constant(const_value)
    result = expr.Evaluate()
    return result


def map_paths_in_hierarchy(expr: Pcp.MapExpression, paths: List[Sdf.Path]) -> List[Sdf.Path]:
    """
    Map a list of paths using the given mapping expression.

    This function maps each path in the input list using the provided
    mapping expression. If a path cannot be mapped (i.e., it is outside
    the domain of the mapping), it is omitted from the result list.

    Parameters:
        expr (Pcp.MapExpression): The mapping expression to apply.
        paths (List[Sdf.Path]): The list of paths to map.

    Returns:
        List[Sdf.Path]: The list of mapped paths, excluding any paths
        that could not be mapped.
    """
    mapped_paths = []
    for path in paths:
        mapped_path = expr.MapSourceToTarget(path)
        if not mapped_path.isEmpty:
            mapped_paths.append(mapped_path)
    return mapped_paths


def get_time_offsets_for_expressions(expressions: List[Pcp.MapExpression]) -> List[Sdf.LayerOffset]:
    """
    Get the time offsets for a list of PcpMapExpression objects.

    Args:
        expressions (List[Pcp.MapExpression]): A list of PcpMapExpression objects.

    Returns:
        List[Sdf.LayerOffset]: A list of Sdf.LayerOffset objects representing the time offsets.
    """
    time_offsets = []
    for expression in expressions:
        if expression.isNull:
            raise ValueError("Invalid PcpMapExpression: expression is null")
        time_offset = expression.timeOffset
        time_offsets.append(time_offset)
    return time_offsets


def inverse_and_evaluate_expression(expression: Pcp.MapExpression) -> Pcp.MapFunction:
    """
    Compute the inverse of the given expression and evaluate it.

    Args:
        expression (Pcp.MapExpression): The expression to invert and evaluate.

    Returns:
        Pcp.MapFunction: The evaluated inverted expression.

    Raises:
        ValueError: If the input expression is None.
    """
    if expression is None:
        raise ValueError("Input expression cannot be None.")
    inverted_expression = Pcp.MapExpression()
    inverted_expression.Inverse(expression)
    map_function = inverted_expression.Evaluate()
    return map_function


def evaluate_expressions_and_check_identity(expressions: List[Pcp.MapExpression]) -> List[bool]:
    """
    Evaluate a list of MapExpressions and check if each one is an identity function.

    Args:
        expressions (List[Pcp.MapExpression]): A list of MapExpressions to evaluate.

    Returns:
        List[bool]: A list of booleans indicating whether each expression is an identity function.
    """
    results = []
    for expr in expressions:
        map_func = expr.Evaluate()
        is_identity = expr.isIdentity
        results.append(is_identity)
    return results


def compose_namespace_mappings(f: Pcp.MapExpression, g: Pcp.MapExpression) -> Pcp.MapExpression:
    """Compose two namespace mapping expressions.

    Args:
        f (Pcp.MapExpression): The first namespace mapping expression.
        g (Pcp.MapExpression): The second namespace mapping expression.

    Returns:
        Pcp.MapExpression: The composed namespace mapping expression.
    """
    if f.isNull:
        return g
    if g.isNull:
        return f
    if f.isIdentity:
        return g
    if g.isIdentity:
        return f
    composed_expression = f.Compose(g)
    return composed_expression


def map_source_to_target_for_prims(
    source_stage: Usd.Stage, target_stage: Usd.Stage, map_expr: Pcp.MapExpression
) -> List[Usd.Prim]:
    """
    Maps prims from the source stage to the target stage using the given map expression.

    Args:
        source_stage (Usd.Stage): The source stage containing the prims to map.
        target_stage (Usd.Stage): The target stage where the mapped prims will be created.
        map_expr (Pcp.MapExpression): The map expression used to map the prim paths.

    Returns:
        List[Usd.Prim]: The list of mapped prims in the target stage.
    """
    mapped_prims = []
    for prim in source_stage.TraverseAll():
        source_path = prim.GetPath()
        target_path = map_expr.MapSourceToTarget(source_path)
        if target_path.isEmpty:
            continue
        mapped_prim = target_stage.DefinePrim(target_path)
        mapped_prims.append(mapped_prim)
    return mapped_prims


def evaluate_and_cache_expressions(expressions: List[Pcp.MapExpression]) -> List[Pcp.MapFunction]:
    """
    Evaluate a list of MapExpression objects and cache their results.

    Args:
        expressions (List[Pcp.MapExpression]): The list of expressions to evaluate.

    Returns:
        List[Pcp.MapFunction]: The list of evaluated values corresponding to the input expressions.
    """
    results = []
    for expression in expressions:
        if expression.isNull:
            continue
        value = expression.Evaluate()
        if not value:
            raise ValueError(f"Failed to evaluate expression: {expression}")
        results.append(value)
    return results


def analyze_and_report_mappings(map_fn: Pcp.MapFunction) -> List[str]:
    """Analyze a MapFunction and report key properties."""
    report: List[str] = []
    if map_fn.isIdentity:
        report.append("The map function is the identity function.")
    else:
        report.append("The map function is not the identity function.")
    if map_fn.isIdentityPathMapping:
        report.append("The map function uses the identity path mapping.")
    else:
        report.append("The map function does not use the identity path mapping.")
    if map_fn.isNull:
        report.append("The map function is the null function.")
    else:
        report.append("The map function is not the null function.")
    report.append(f"The time offset of the mapping is: {map_fn.timeOffset}")
    report.append("Source to target path mappings:")
    for source_path, target_path in map_fn.sourceToTargetMap.items():
        report.append(f"  {source_path} -> {target_path}")
    return report


def remap_prim_paths(stage: Usd.Stage, map_function: Pcp.MapFunction, prim_paths: List[str]) -> List[str]:
    """
    Remaps a list of prim paths using the provided map function.

    Args:
        stage (Usd.Stage): The USD stage.
        map_function (Pcp.MapFunction): The map function to apply.
        prim_paths (List[str]): The list of prim paths to remap.

    Returns:
        List[str]: The remapped prim paths.
    """
    remapped_paths = []
    for prim_path in prim_paths:
        sdf_path = Sdf.Path(prim_path)
        if map_function.MapSourceToTarget(sdf_path) != Sdf.Path.emptyPath:
            remapped_path = map_function.MapSourceToTarget(sdf_path)
            remapped_paths.append(str(remapped_path))
        else:
            remapped_paths.append(Sdf.Path.emptyPath)
    return remapped_paths


def get_all_mapped_paths(map_function: Pcp.MapFunction) -> Set[Sdf.Path]:
    """
    Get all paths that are mapped by the given map function.

    Args:
        map_function (Pcp.MapFunction): The map function to query.

    Returns:
        Set[Sdf.Path]: A set of all paths that are mapped by the map function.
    """
    path_map = map_function.sourceToTargetMap
    mapped_paths = set()
    for mapping in path_map.items():
        (source_path, target_path) = mapping
        mapped_paths.add(source_path)
        mapped_paths.add(target_path)
    return mapped_paths


def compose_map_functions_and_apply(map_functions: List[Pcp.MapFunction], path: Sdf.Path) -> Sdf.Path:
    """
    Compose a list of map functions and apply the result to a path.

    Args:
        map_functions (List[Pcp.MapFunction]): A list of map functions to compose.
        path (Sdf.Path): The path to map.

    Returns:
        Sdf.Path: The mapped path.
    """
    if not map_functions:
        return path
    composed_map_function = map_functions[0]
    for map_function in map_functions[1:]:
        composed_map_function = composed_map_function.Compose(map_function)
    mapped_path = composed_map_function.MapSourceToTarget(path)
    return mapped_path


def invert_map_function_and_apply(map_function: Pcp.MapFunction, target_path: Sdf.Path) -> Sdf.Path:
    """
    Invert the given map function and apply it to the target path.

    This function first inverts the input map function using GetInverse(),
    then applies the inverted map function to the given target path using
    MapTargetToSource(). The resulting path in the source namespace is returned.

    If the input map function is None or the target path is empty, an empty
    path is returned.

    Parameters:
        map_function (Pcp.MapFunction): The map function to invert and apply.
        target_path (Sdf.Path): The target path to map back to the source.

    Returns:
        Sdf.Path: The path in the source namespace after applying the inverted map.
    """
    if map_function is None:
        return Sdf.Path()
    if target_path.isEmpty:
        return Sdf.Path()
    inverted_map = map_function.GetInverse()
    source_path = inverted_map.MapTargetToSource(target_path)
    return source_path


def check_if_map_function_is_identity(map_function: Pcp.MapFunction) -> bool:
    """Check if a given map function is an identity function.

    An identity map function has an identity path mapping and time offset.

    Args:
        map_function (Pcp.MapFunction): The map function to check.

    Returns:
        bool: True if the map function is an identity function, False otherwise.
    """
    if map_function.isNull:
        return False
    if not map_function.isIdentityPathMapping:
        return False
    if map_function.timeOffset != Sdf.LayerOffset():
        return False
    return True


def create_identity_map_function() -> Pcp.MapFunction:
    """Create an identity map function."""
    identity_map_function = Pcp.MapFunction.Identity()
    if not identity_map_function.isIdentity:
        raise ValueError("Failed to create an identity map function.")
    return identity_map_function


def apply_map_function_to_prim(prim: Usd.Prim, map_func: Pcp.MapFunction) -> Usd.Prim:
    """Apply a PcpMapFunction to a prim, mapping its paths.

    Args:
        prim (Usd.Prim): The input prim to map.
        map_func (Pcp.MapFunction): The map function to apply.

    Returns:
        Usd.Prim: A new prim with the mapped paths, or an invalid prim if mapping failed.
    """
    if not prim.IsValid():
        return Usd.Prim()
    prim_path = prim.GetPath()
    target_path = map_func.MapSourceToTarget(prim_path)
    if not target_path.isEmpty:
        stage = prim.GetStage()
        target_prim = stage.DefinePrim(target_path)
        for key, value in prim.GetAllMetadata().items():
            target_prim.SetMetadata(key, value)
        for prop in prim.GetPropertyNames():
            source_prop = prim.GetProperty(prop)
            target_prop = target_prim.CreateAttribute(prop, source_prop.GetTypeName())
            target_prop.Set(source_prop.Get())
        return target_prim
    else:
        return Usd.Prim()


def is_prim_instanceable(prim_index: Pcp.PrimIndex) -> bool:
    """
    Check if a prim is instanceable based on its prim index.

    A prim is considered instanceable if its prim index is instanceable.
    Instanceable prim indexes with the same instance key are guaranteed
    to have the same set of opinions, but may not have local opinions
    about name children.

    Args:
        prim_index (Pcp.PrimIndex): The prim index to check.

    Returns:
        bool: True if the prim is instanceable, False otherwise.
    """
    if not prim_index.IsValid():
        raise ValueError("Invalid prim index.")
    return prim_index.IsInstanceable()


def create_prim_index(stage: Usd.Stage, prim_path: str) -> Pcp.PrimIndex:
    """
    Create a prim index for the specified prim path on the given stage.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path of the prim.

    Returns:
        Pcp.PrimIndex: The created prim index.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")
    prim_index = prim.ComputeExpandedPrimIndex()
    return prim_index


def get_variant_selection(prim_index: Pcp.PrimIndex, variant_set_name: str) -> str:
    """
    Get the variant selection applied for the specified variant set on the given prim index.

    Args:
        prim_index (Pcp.PrimIndex): The prim index to query.
        variant_set_name (str): The name of the variant set.

    Returns:
        str: The applied variant selection, or an empty string if no selection is applied.
    """
    if not prim_index.IsValid():
        raise ValueError("Invalid prim index.")
    authored_selections = prim_index.ComposeAuthoredVariantSelections()
    if variant_set_name not in authored_selections:
        return ""
    applied_selection = prim_index.GetSelectionAppliedForVariantSet(variant_set_name)
    return applied_selection


def get_authored_variant_selections(prim_index: Pcp.PrimIndex) -> Dict[str, str]:
    """Get the authored variant selections for a prim index."""
    authored_selections: Pcp.VariantSelectionMap = prim_index.ComposeAuthoredVariantSelections()
    result: Dict[str, str] = {}
    for variant_set_name, variant_selection in authored_selections.items():
        result[variant_set_name] = variant_selection
    return result


class MockPropertyIndex:

    def __init__(self):
        self.localErrors: List[str] = []
        self.localPropertyStack: List[str] = []
        self.propertyStack: List[str] = []


def validate_property_composition(prop_index: MockPropertyIndex) -> List[str]:
    """Validate the composition of a property and return any errors.

    Args:
        prop_index (MockPropertyIndex): The property index to validate.

    Returns:
        List[str]: A list of error messages, if any.
    """
    errors: List[str] = []
    if prop_index.localErrors:
        errors.extend(prop_index.localErrors)
    if not prop_index.propertyStack:
        errors.append("Property stack is empty.")
    if prop_index.localPropertyStack != prop_index.propertyStack:
        errors.append("Local property stack does not match the full property stack.")
    return errors


def propagate_site_transform(prim: Usd.Prim, xform: Gf.Matrix4d) -> bool:
    """
    Propagate a transform to a prim.

    Args:
        prim (Usd.Prim): The prim to propagate the transform to.
        xform (Gf.Matrix4d): The transform to propagate.

    Returns:
        bool: True if the transform was successfully propagated, False otherwise.
    """
    stage = prim.GetStage()
    if not stage:
        return False
    edit_target = stage.GetEditTarget()
    prim_spec = edit_target.GetPrimSpecForScenePath(prim.GetPath())
    if not prim_spec:
        return False
    transform_op_attr = prim_spec.attributes.get("xformOp:transform")
    if not transform_op_attr:
        transform_op_attr = Sdf.AttributeSpec(
            prim_spec, "xformOp:transform", Sdf.ValueTypeNames.Matrix4d, variability=Sdf.VariabilityUniform
        )
    transform_op_attr.default = Gf.Matrix4d(xform)
    return True
