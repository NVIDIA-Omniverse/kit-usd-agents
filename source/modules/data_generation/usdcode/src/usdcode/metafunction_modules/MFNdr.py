## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import os
from typing import Any, Dict, List, Optional, Tuple

from pxr import Ar, Ndr, Sdf, Usd


def update_expired_plugins(plugin_contexts: List[Ndr.DiscoveryPluginContext]) -> List[Ndr.DiscoveryPluginContext]:
    """Update the list of plugin contexts by removing expired ones.

    Args:
        plugin_contexts (List[Ndr.DiscoveryPluginContext]): List of plugin contexts.

    Returns:
        List[Ndr.DiscoveryPluginContext]: Updated list of plugin contexts with expired ones removed.
    """
    updated_contexts = []
    for context in plugin_contexts:
        if not context.expired:
            updated_contexts.append(context)
        else:
            continue
    return updated_contexts


class MockDiscoveryPluginContext:

    def __init__(self, expired=False):
        self._expired = expired

    @property
    def expired(self):
        return self._expired

    @expired.setter
    def expired(self, value):
        self._expired = value


def validate_discovery_plugins(plugin_context, discovery_types: List[str]) -> List[str]:
    """
    Validate the given discovery types against the source types in the discovery plugin context.

    Args:
        plugin_context: A mock object that mimics the behavior of Ndr.DiscoveryPluginContext.
        discovery_types (List[str]): The list of discovery types to validate.

    Returns:
        List[str]: The list of valid discovery types.
    """
    valid_types = []
    for discovery_type in discovery_types:
        source_type = plugin_context.GetSourceType(discovery_type)
        if source_type:
            valid_types.append(discovery_type)
        else:
            print(f"Warning: No source type found for discovery type '{discovery_type}'")
    return valid_types


class MockDiscoveryPluginContext:

    def GetSourceType(self, discovery_type: str) -> str:
        if discovery_type in ["ShaderNode", "Texture"]:
            return "TestSourceType"
        else:
            return ""


def register_and_validate_plugins(plugin_paths: List[str]) -> List[str]:
    """Register and validate discovery plugins from given paths.

    Args:
        plugin_paths (List[str]): List of paths to discovery plugin files.

    Returns:
        List[str]: List of successfully registered plugin paths.
    """
    registered_plugins = []
    for path in plugin_paths:
        if not os.path.exists(path):
            print(f"Warning: Plugin file not found: {path}")
            continue
        try:
            Ndr.DiscoveryPluginList.append(path)
            registered_plugins.append(path)
        except Exception as e:
            print(f"Error registering plugin: {path}")
            print(f"Error message: {str(e)}")
    valid_plugins = []
    for plugin in registered_plugins:
        try:
            if Ndr.Registry.validate(plugin):
                valid_plugins.append(plugin)
            else:
                print(f"Plugin validation failed: {plugin}")
        except Exception as e:
            print(f"Error validating plugin: {plugin}")
            print(f"Error message: {str(e)}")
    return valid_plugins


def discover_and_resolve_uris(files_discovered: List[Ndr.DiscoveryUri]) -> List[Ndr.DiscoveryUri]:
    """
    Given a list of discovered files, resolve each file's URI and return a new list
    of DiscoveryUri objects with both the original URI and the resolved URI.

    Args:
        files_discovered (List[Ndr.DiscoveryUri]): A list of DiscoveryUri objects containing the discovered file URIs.

    Returns:
        List[Ndr.DiscoveryUri]: A new list of DiscoveryUri objects with the original URI and the resolved URI.
    """
    resolved_uris: List[Ndr.DiscoveryUri] = []
    for discovered_uri in files_discovered:
        uri_str = discovered_uri.uri
        resolved_uri_str = Ar.GetResolver().Resolve(uri_str)
        resolved_uri = Ndr.DiscoveryUri()
        resolved_uri.uri = uri_str
        resolved_uri.resolvedUri = resolved_uri_str
        resolved_uris.append(resolved_uri)
    return resolved_uris


class DiscoveryUri:
    """
    Struct for holding a URI and its resolved URI for a file discovered by
    NdrFsHelpersDiscoverFiles.
    """

    def __init__(self, uri: str = None, resolved_uri: str = None):
        self.uri = uri
        self.resolvedUri = resolved_uri


def replace_references_with_resolved_uris(discovery_uris: List[DiscoveryUri]) -> List[str]:
    """
    Replace the references in the given discovery URIs with their resolved URIs.

    Args:
        discovery_uris (List[DiscoveryUri]): A list of DiscoveryUri objects.

    Returns:
        List[str]: A list of strings with references replaced by resolved URIs.
    """
    resolved_uris: List[str] = []
    for discovery_uri in discovery_uris:
        if discovery_uri.uri is None:
            continue
        if discovery_uri.resolvedUri is None:
            resolved_uris.append(discovery_uri.uri)
        else:
            resolved_uris.append(discovery_uri.resolvedUri)
    return resolved_uris


class MockNode(Ndr.Node):
    """Mock node class for testing."""

    def __init__(self, major: int, minor: int, patch: int):
        self._version = Ndr.Version()
        self._version.major = major
        self._version.minor = minor
        self._version.patch = patch

    def IsValid(self) -> bool:
        return True

    def GetVersion(self) -> Ndr.Version:
        return self._version


def compare_node_versions(node1: Ndr.Node, node2: Ndr.Node) -> Tuple[int, int, int]:
    """Compare the versions of two nodes.

    Returns a tuple of (major_diff, minor_diff, patch_diff), where:
    - major_diff: 1 if node1 has a higher major version, -1 if node2 does, 0 if equal
    - minor_diff: 1 if node1 has a higher minor version, -1 if node2 does, 0 if equal
    - patch_diff: 1 if node1 has a higher patch version, -1 if node2 does, 0 if equal

    If either node is invalid, raises a ValueError.
    """
    if not node1.IsValid() or not node2.IsValid():
        raise ValueError("One or both nodes are invalid.")
    version1 = node1.GetVersion()
    version2 = node2.GetVersion()
    if version1.major > version2.major:
        major_diff = 1
    elif version1.major < version2.major:
        major_diff = -1
    else:
        major_diff = 0
    if version1.minor > version2.minor:
        minor_diff = 1
    elif version1.minor < version2.minor:
        minor_diff = -1
    else:
        minor_diff = 0
    if version1.patch > version2.patch:
        patch_diff = 1
    elif version1.patch < version2.patch:
        patch_diff = -1
    else:
        patch_diff = 0
    return (major_diff, minor_diff, patch_diff)


def group_nodes_by_family(nodes: List[Ndr.Node]) -> Dict[str, List[Ndr.Node]]:
    """Group a list of nodes by their family.

    Args:
        nodes (List[Ndr.Node]): The list of nodes to group.

    Returns:
        Dict[str, List[Ndr.Node]]: A dictionary where the keys are family names
        and the values are lists of nodes belonging to that family.
    """
    family_groups: Dict[str, List[Ndr.Node]] = {}
    for node in nodes:
        family: str = node.GetFamily()
        if family:
            if family not in family_groups:
                family_groups[family] = []
            family_groups[family].append(node)
        else:
            if "" not in family_groups:
                family_groups[""] = []
            family_groups[""].append(node)
    return family_groups


class MockNode:

    def __init__(self, family):
        self._family = family

    def GetFamily(self):
        return self._family


def get_node_info_string_by_property(node: Ndr.Node, property_name: str) -> Optional[str]:
    """
    Get the value of a specific property from a node's info string.

    Args:
        node (Ndr.Node): The node to retrieve the info string from.
        property_name (str): The name of the property to retrieve.

    Returns:
        Optional[str]: The value of the specified property, or None if not found.
    """
    info_string = node.GetInfoString()
    properties = info_string.split(";")
    for prop in properties:
        key_value = prop.split("=")
        if len(key_value) == 2:
            key = key_value[0].strip()
            value = key_value[1].strip()
            if key == property_name:
                return value
    return None


class MockNode(Ndr.Node):

    def __init__(self, info_string):
        self._info_string = info_string

    def GetInfoString(self):
        return self._info_string


def filter_nodes_by_version(nodes: List[Ndr.NodeDiscoveryResult], version: str) -> List[Ndr.NodeDiscoveryResult]:
    """Filter a list of NodeDiscoveryResult objects by version.

    Args:
        nodes (List[Ndr.NodeDiscoveryResult]): List of NodeDiscoveryResult objects to filter.
        version (str): Version string to match against.

    Returns:
        List[Ndr.NodeDiscoveryResult]: Filtered list of NodeDiscoveryResult objects.
    """
    if not nodes:
        return []
    filtered_nodes = [node for node in nodes if str(node.version) == version]
    return filtered_nodes


def validate_node_integrity(node_result: Ndr.NodeDiscoveryResult) -> bool:
    """Validate the integrity of a NodeDiscoveryResult.

    This function checks if all required fields are populated and consistent.
    It returns True if the node is valid, False otherwise.
    """
    if not node_result.identifier:
        return False
    if not node_result.name:
        return False
    if not node_result.family:
        return False
    if not node_result.discoveryType:
        return False
    if not node_result.sourceType:
        return False
    if not node_result.uri:
        return False
    if not node_result.resolvedUri:
        return False
    if not node_result.version:
        return False
    if not node_result.sourceCode and (not node_result.metadata):
        return False
    return True


def get_dynamic_array_properties(properties: List[Ndr.Property]) -> List[Ndr.Property]:
    """
    Return a list of properties that are dynamic arrays.

    Args:
        properties (List[Ndr.Property]): A list of Ndr.Property objects.

    Returns:
        List[Ndr.Property]: A list of Ndr.Property objects that are dynamic arrays.
    """
    dynamic_array_properties = []
    for prop in properties:
        if prop.IsArray():
            if prop.IsDynamicArray():
                dynamic_array_properties.append(prop)
    return dynamic_array_properties


class MockProperty:

    def __init__(self, name, type, isDynamicArray=False):
        self._name = name
        self._type = type
        self._isDynamicArray = isDynamicArray

    def GetName(self):
        return self._name

    def GetType(self):
        return self._type

    def IsArray(self):
        return "[]" in self._type

    def IsDynamicArray(self):
        return self._isDynamicArray


class MockProperty:

    def __init__(self, name: str, type: str, default_value: Any):
        self._name = name
        self._type = type
        self._default_value = default_value

    def GetName(self) -> str:
        return self._name

    def GetType(self) -> str:
        return self._type

    def GetDefaultValue(self) -> Any:
        return self._default_value

    def IsArray(self) -> bool:
        return isinstance(self._default_value, list)

    def GetArraySize(self) -> int:
        if self.IsArray():
            return len(self._default_value)
        return 0


def compare_property_defaults(prop1: MockProperty, prop2: MockProperty) -> bool:
    """Compare the default values of two properties.

    Args:
        prop1 (MockProperty): The first property to compare.
        prop2 (MockProperty): The second property to compare.

    Returns:
        bool: True if the default values are equal, False otherwise.
    """
    default1 = prop1.GetDefaultValue()
    default2 = prop2.GetDefaultValue()
    if default1 is None and default2 is None:
        return True
    elif default1 is None or default2 is None:
        return False
    if prop1.IsArray() and prop2.IsArray():
        if prop1.GetArraySize() != prop2.GetArraySize():
            return False
        for i in range(prop1.GetArraySize()):
            if default1[i] != default2[i]:
                return False
        return True
    else:
        return default1 == default2


def validate_property_connections(properties: List["Mock"]) -> bool:
    """
    Validate that all properties in the given list can be connected to each other.

    Args:
        properties (List[Mock]): The list of mock properties to validate connections for.

    Returns:
        bool: True if all properties can be connected to each other, False otherwise.
    """
    for i in range(len(properties)):
        for j in range(i + 1, len(properties)):
            prop1 = properties[i]
            prop2 = properties[j]
            if not prop1.IsConnectable() or not prop2.IsConnectable():
                return False
            if not prop1.CanConnectTo(prop2) or not prop2.CanConnectTo(prop1):
                return False
    return True


class MockProperty:
    """Mock implementation of Ndr.Property for testing purposes."""

    def __init__(self, name: str, is_connectable: bool):
        self._name = name
        self._is_connectable = is_connectable

    def GetName(self) -> str:
        return self._name

    def IsConnectable(self) -> bool:
        return self._is_connectable


def filter_connectable_properties(properties: List[MockProperty]) -> List[MockProperty]:
    """
    Filter a list of properties to only those that are connectable.

    Args:
        properties (List[MockProperty]): A list of MockProperty objects.

    Returns:
        List[MockProperty]: A list of MockProperty objects that are connectable.
    """
    connectable_properties = [prop for prop in properties if prop.IsConnectable()]
    return connectable_properties


def find_output_properties(node: Ndr.Node) -> List[Ndr.Property]:
    """Find all output properties of an Ndr node.

    Args:
        node (Ndr.Node): The Ndr node to search for output properties.

    Returns:
        List[Ndr.Property]: A list of output properties found on the node.
    """
    properties = node.GetProperties()
    output_properties = [prop for prop in properties if prop.IsOutput()]
    return output_properties


class MockProperty(Ndr.Property):

    def __init__(self, name, type, is_output):
        self._name = name
        self._type = type
        self._is_output = is_output

    def GetName(self):
        return self._name

    def GetType(self):
        return self._type

    def IsOutput(self):
        return self._is_output


class MockNode(Ndr.Node):

    def __init__(self, name):
        self._name = name
        self._properties = []

    def GetName(self):
        return self._name

    def GetProperties(self):
        return self._properties

    def AddProperty(self, prop):
        self._properties.append(prop)


class MockProperty:

    def __init__(self, name: str, type_name: Sdf.ValueTypeName):
        self._name = name
        self._type_name = type_name

    def GetName(self) -> str:
        return self._name

    def GetTypeAsSdfType(self) -> Tuple[Sdf.ValueTypeName, str]:
        return (self._type_name, "")


def convert_property_types(properties: List[MockProperty]) -> List[Tuple[str, str]]:
    """
    Convert a list of MockProperty objects to a list of tuples containing the property name and Sdf type.

    Args:
        properties (List[MockProperty]): List of MockProperty objects to convert.

    Returns:
        List[Tuple[str, str]]: List of tuples containing the property name and Sdf type.
    """
    result = []
    for prop in properties:
        prop_name = prop.GetName()
        sdf_type_indicator = prop.GetTypeAsSdfType()
        if not sdf_type_indicator[1]:
            sdf_type = str(sdf_type_indicator[0])
        else:
            sdf_type = sdf_type_indicator[1]
        result.append((prop_name, sdf_type))
    return result


class MockProperty:

    def __init__(self, name: str, type: str, array_size: int, is_output: bool, is_connectable: bool):
        self._name = name
        self._type = type
        self._array_size = array_size
        self._is_output = is_output
        self._is_connectable = is_connectable

    def IsConnectable(self) -> bool:
        return self._is_connectable

    def GetType(self) -> str:
        return self._type

    def IsArray(self) -> bool:
        return self._array_size > 0

    def GetArraySize(self) -> int:
        return self._array_size

    def IsOutput(self) -> bool:
        return self._is_output

    def CanConnectTo(self, other: Any) -> bool:
        return True


def check_property_compatibility(prop1: MockProperty, prop2: MockProperty) -> bool:
    """Check if two properties are compatible for connection.

    Args:
        prop1 (MockProperty): First property to check.
        prop2 (MockProperty): Second property to check.

    Returns:
        bool: True if properties are compatible, False otherwise.
    """
    if not prop1.IsConnectable() or not prop2.IsConnectable():
        return False
    if prop1.GetType() != prop2.GetType():
        return False
    if prop1.IsArray() and prop2.IsArray():
        if prop1.GetArraySize() != prop2.GetArraySize():
            return False
    if prop1.IsOutput() == prop2.IsOutput():
        return False
    if not prop1.CanConnectTo(prop2) or not prop2.CanConnectTo(prop1):
        return False
    return True


def get_node_names(family: str = "") -> List[str]:
    """
    Get the names of all the nodes that the registry is aware of.

    This will not run the parsing plugins on the nodes that have been
    discovered, so this method is relatively quick. Optionally,
    a "family" name can be specified to only get the names of nodes
    that belong to that family.

    Args:
        family (str): Optional family name to filter nodes by.

    Returns:
        List[str]: List of node names.
    """
    node_names: List[str] = []
    identifiers: NdrIdentifierVec = Registry.GetNodeIdentifiers(family, Ndr.VersionFilterDefaultOnly)
    for identifier in identifiers:
        node_name: str = identifier.GetName()
        node_names.append(node_name)
    return node_names


def get_node_by_identifier(self, identifier: str, sourceTypePriority: list) -> Ndr.Node:
    """
    Get the node with the specified identifier, and an optional sourceTypePriority list
    specifying the set of node SOURCE types that should be searched.

    If no sourceTypePriority is specified, the first encountered node with the specified
    identifier will be returned (first is arbitrary) if found.

    If a sourceTypePriority list is specified, then this will iterate through each source
    type and try to find a node matching by identifier. This is equivalent to calling
    NdrRegistry.GetNodeByIdentifierAndType for each source type until a node is found.

    Nodes of the same identifier but different source type can exist in the registry. If
    a node 'Foo' with source types 'abc' and 'xyz' exist in the registry, and you want to
    make sure the 'abc' version is fetched before the 'xyz' version, the priority list
    would be specified as ['abc', 'xyz']. If the 'abc' version did not exist in the
    registry, then the 'xyz' version would be returned.

    Returns None if a node matching the arguments can't be found.
    """
    if not sourceTypePriority:
        for node in self._nodes.values():
            if node.GetIdentifier() == identifier:
                return node
    else:
        for sourceType in sourceTypePriority:
            for node in self._nodes.values():
                if node.GetIdentifier() == identifier and node.GetSourceType() == sourceType:
                    return node
    return None


class Node:

    def __init__(self, identifier, sourceType):
        self._identifier = identifier
        self._sourceType = sourceType

    def GetIdentifier(self):
        return self._identifier

    def GetSourceType(self):
        return self._sourceType


class Registry:

    def __init__(self):
        self._nodes = {}


def compare_versions(version1: Ndr.Version, version2: Ndr.Version) -> int:
    """Compare two Ndr.Version objects.

    Args:
        version1 (Ndr.Version): The first version to compare.
        version2 (Ndr.Version): The second version to compare.

    Returns:
        int: -1 if version1 < version2, 0 if version1 == version2, 1 if version1 > version2.
    """
    if version1.GetMajor() < version2.GetMajor():
        return -1
    elif version1.GetMajor() > version2.GetMajor():
        return 1
    if version1.GetMinor() < version2.GetMinor():
        return -1
    elif version1.GetMinor() > version2.GetMinor():
        return 1
    return 0


def filter_versions_by_major(versions: List[Ndr.Version], major: int) -> List[Ndr.Version]:
    """Filter a list of versions by major version number.

    Args:
        versions (List[Ndr.Version]): A list of versions to filter.
        major (int): The major version number to filter by.

    Returns:
        List[Ndr.Version]: A new list containing only versions with the specified major version number.
    """
    if not isinstance(versions, list):
        raise TypeError("versions must be a list of Ndr.Version objects")
    if not isinstance(major, int):
        raise TypeError("major must be an integer")
    if major < 0:
        raise ValueError("major must be a non-negative integer")
    filtered_versions = [v for v in versions if v.GetMajor() == major]
    return filtered_versions


def format_version_suffix(version: Ndr.Version) -> str:
    """Format the version as a string suffix."""
    if version.GetMajor() == 0 and version.GetMinor() == 0:
        return ""
    suffix = version.GetStringSuffix()
    if version.IsDefault():
        suffix += "*"
    return suffix


def get_default_version() -> Ndr.Version:
    """Get the default version from Ndr.Version."""
    version = Ndr.Version()
    if version.GetMajor() == 0 and version.GetMinor() == 0:
        return version.GetAsDefault()
    if version.IsDefault():
        return version
    return version.GetAsDefault()


def get_latest_version(versions: List[Ndr.Version]) -> Ndr.Version:
    """
    Get the latest version from a list of versions.

    Args:
        versions (List[Ndr.Version]): List of versions to compare.

    Returns:
        Ndr.Version: The latest version from the list.

    Raises:
        ValueError: If the input list is empty.
    """
    if not versions:
        raise ValueError("Input list of versions is empty.")
    latest_version = versions[0]
    for version in versions:
        if version.GetMajor() > latest_version.GetMajor():
            latest_version = version
        elif version.GetMajor() == latest_version.GetMajor():
            if version.GetMinor() > latest_version.GetMinor():
                latest_version = version
            elif version.GetMinor() == latest_version.GetMinor():
                if version.IsDefault():
                    latest_version = version
    return latest_version


def filter_prims_by_version(stage: Usd.Stage, min_version: Ndr.Version, max_version: Ndr.Version) -> List[Usd.Prim]:
    """Filter prims in a stage by version range.

    Args:
        stage (Usd.Stage): The USD stage to filter prims from.
        min_version (Ndr.Version): The minimum version to include (inclusive).
        max_version (Ndr.Version): The maximum version to include (inclusive).

    Returns:
        List[Usd.Prim]: A list of prims that match the version range.
    """
    prims: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        version_str = prim.GetMetadata("version")
        if version_str is None:
            continue
        version = Ndr.Version.GetValueFromName(version_str)
        if min_version <= version <= max_version:
            prims.append(prim)
    return prims
