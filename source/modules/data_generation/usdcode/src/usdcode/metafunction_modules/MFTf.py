## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import os
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, sys

from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdShade

from .add_op import *


def analyze_call_hierarchy(call_contexts: List[Tf.CallContext]) -> List[str]:
    """Analyze the call hierarchy and return a list of formatted strings.

    Args:
        call_contexts (List[Tf.CallContext]): A list of CallContext objects representing the call hierarchy.

    Returns:
        List[str]: A list of formatted strings representing the call hierarchy.
    """
    hierarchy: List[str] = []
    for context in call_contexts:
        file: str = context.file
        function: str = context.function
        line: int = context.line
        pretty_function: str = context.prettyFunction
        formatted_string: str = f"{file}:{line} - {function} ({pretty_function})"
        hierarchy.append(formatted_string)
    return hierarchy


class MockCallContext:

    def __init__(self, file, function, line, pretty_function):
        self.file = file
        self.function = function
        self.line = line
        self.prettyFunction = pretty_function


def handle_cpp_exception_during_material_assignment(stage: Usd.Stage, prim_path: str, material_path: str) -> bool:
    """Handles CppException during material assignment and returns True if successfully assigned.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to assign the material to.
        material_path (str): The path to the material prim.

    Returns:
        bool: True if the material was successfully assigned, False otherwise.
    """
    try:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return False
        material_prim = stage.GetPrimAtPath(material_path)
        if not material_prim.IsValid():
            return False
        material = UsdShade.Material(material_prim)
        binding_api = UsdShade.MaterialBindingAPI(prim)
        binding_api.Bind(material)
        return True
    except Tf.CppException as e:
        print(f"Caught CppException: {str(e)}")
        return False


def transform_prims_with_error_handling(stage: Usd.Stage, prim_paths: List[str], translation: Gf.Vec3d) -> None:
    """Transform a list of prims by a given translation vector, with error handling.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths to transform.
        translation (Gf.Vec3d): The translation vector to apply.

    Raises:
        ValueError: If any prim path is invalid or not transformable.
        Tf.CppException: If any other USD-related error occurs.
    """
    for prim_path in prim_paths:
        try:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise ValueError(f"Prim at path {prim_path} does not exist.")
            xformable = UsdGeom.Xformable(prim)
            if not xformable:
                raise ValueError(f"Prim at path {prim_path} is not transformable.")
            translate_op = add_translate_op(xformable)
            translate_op.Set(translation)
        except Tf.CppException as e:
            raise Tf.CppException(f"Error transforming prim {prim_path}: {str(e)}") from e


def safe_prim_removal(stage: Usd.Stage, prim_path: str) -> bool:
    """Safely remove a prim from a USD stage.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path of the prim to remove.

    Returns:
        bool: True if the prim was successfully removed, False otherwise.
    """
    if not stage:
        raise ValueError("Invalid stage.")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Prim at path {prim_path} does not exist.")
        return False
    if prim.GetChildren():
        print(f"Cannot remove prim at path {prim_path} because it has children.")
        return False
    try:
        stage.RemovePrim(prim.GetPath())
    except Tf.ErrorException as e:
        print(f"Error removing prim: {e}")
        return False
    if stage.GetPrimAtPath(prim_path).IsValid():
        print(f"Failed to remove prim at path {prim_path}.")
        return False
    return True


def add_reference_with_exception_handling(
    stage: Usd.Stage, prim_path: str, reference_path: str
) -> Optional[Sdf.Reference]:
    """Add a reference to a prim, handling exceptions.

    Args:
        stage (Usd.Stage): The stage to add the reference to.
        prim_path (str): The path of the prim to add the reference to.
        reference_path (str): The path of the referenced layer.

    Returns:
        Optional[Sdf.Reference]: The added reference, or None if an error occurred.
    """
    try:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            raise ValueError(f"Prim not found at path: {prim_path}")
        references = prim.GetReferences()
        added_reference = references.AddReference(assetPath=reference_path)
        return added_reference
    except Tf.ErrorException as e:
        print(f"USD error occurred: {str(e)}")
    except Tf.CppException as e:
        print(f"C++ exception occurred: {str(e)}")
    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")
    return None


def batch_modify_prims_with_exception_handling(
    stage: Usd.Stage, prim_paths: List[str], new_scale_value: Tuple[float, float, float]
) -> List[Tuple[str, str]]:
    """Modify the scale of multiple prims in a batch with exception handling.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths to modify.
        new_scale_value (Tuple[float, float, float]): The new scale value to set.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing the prim path and the result status.
    """
    results = []
    for prim_path in prim_paths:
        try:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise Tf.ErrorException(f"Prim at path {prim_path} does not exist.")
            xformable = UsdGeom.Xformable(prim)
            if not xformable:
                raise Tf.ErrorException(f"Prim at path {prim_path} is not transformable.")
            scale_op = add_scale_op(xformable)
            scale_op.Set(new_scale_value)
            results.append((prim_path, "Success"))
        except Tf.ErrorException as e:
            results.append((prim_path, str(e)))
        except Exception as e:
            results.append((prim_path, f"Unexpected error: {str(e)}"))
    return results


def enable_debug_symbols(pattern: str) -> List[str]:
    """Enable debug symbols matching the given pattern.

    Args:
        pattern (str): The pattern to match debug symbols.

    Returns:
        List[str]: The names of the debug symbols that were enabled.
    """
    if not pattern:
        raise ValueError("Pattern cannot be empty.")
    all_symbols = Tf.Debug.GetDebugSymbolNames()
    matching_symbols = [
        symbol
        for symbol in all_symbols
        if symbol == pattern or (pattern.endswith("*") and symbol.startswith(pattern[:-1]))
    ]
    enabled_symbols = Tf.Debug.SetDebugSymbolsByName(pattern, True)
    return enabled_symbols


def get_all_debug_symbols_info() -> List[Tuple[str, str, bool]]:
    """Get information about all debug symbols.

    Returns:
        A list of tuples containing the debug symbol name, description, and enabled status.
    """
    symbol_names = Tf.Debug.GetDebugSymbolNames()
    symbol_info = []
    for name in symbol_names:
        description = Tf.Debug.GetDebugSymbolDescription(name)
        enabled = Tf.Debug.IsDebugSymbolNameEnabled(name)
        info = (name, description, enabled)
        symbol_info.append(info)
    return symbol_info


def set_debug_output_to_stderr():
    """Set the debug output to stderr."""
    if os.environ.get("TF_DEBUG_OUTPUT_FILE") == "stderr":
        return
    try:
        Tf.Debug.GetDebugSymbolNames()
        Tf.Debug.SetOutputFile(sys.stderr)
    except ValueError as e:
        print(f"Error setting debug output to stderr: {str(e)}")


def toggle_debug_symbols(pattern: str) -> List[str]:
    """Toggle the value of all debug symbols matching the given pattern.

    Args:
        pattern (str): The pattern to match debug symbols against.
            If pattern ends with '*', it matches as a prefix.
            Otherwise, it matches as an exact string.

    Returns:
        List[str]: The names of all debug symbols that were toggled.
    """
    symbol_names = Tf.Debug.GetDebugSymbolNames()
    matching_symbols = []
    for name in symbol_names:
        if pattern.endswith("*"):
            if name.startswith(pattern[:-1]):
                matching_symbols.append(name)
        elif name == pattern:
            matching_symbols.append(name)
    toggled_symbols = []
    for name in matching_symbols:
        current_value = Tf.Debug.IsDebugSymbolNameEnabled(name)
        Tf.Debug.SetDebugSymbolsByName(name, not current_value)
        toggled_symbols.append(name)
    return toggled_symbols


def get_enabled_debug_symbols() -> List[str]:
    """Get a list of all enabled debug symbols."""
    all_symbols = Tf.Debug.GetDebugSymbolNames()
    enabled_symbols = []
    for symbol in all_symbols:
        if Tf.Debug.IsDebugSymbolNameEnabled(symbol):
            enabled_symbols.append(symbol)
    return enabled_symbols


def search_debug_symbols_by_description(description: str) -> List[str]:
    """
    Search for debug symbols by their description.

    Parameters
    ----------
    description : str
        The description or partial description to search for.

    Returns
    -------
    List[str]
        A list of debug symbol names that match the given description.
    """
    all_descriptions = Tf.Debug.GetDebugSymbolDescriptions()
    description_lines = all_descriptions.split("\n")
    matching_lines = [line for line in description_lines if description in line]
    debug_symbol_names = []
    for line in matching_lines:
        parts = line.split(":")
        if len(parts) > 1:
            debug_symbol_names.append(parts[0].strip())
    return debug_symbol_names


def log_diagnostics_for_prim(prim: Usd.Prim):
    """Log diagnostic messages for a given prim."""
    if not prim.IsValid():
        Tf.Warn(f"Prim '{prim.GetPath()}' is not valid.")
        return
    prim_type = prim.GetTypeName()
    print(f"Prim '{prim.GetPath()}' has type '{prim_type}'.")
    attributes = prim.GetAttributes()
    if attributes:
        print(f"Prim '{prim.GetPath()}' has {len(attributes)} attributes:")
        for attr in attributes:
            attr_name = attr.GetName()
            attr_type = attr.GetTypeName()
            print(f"  - Attribute '{attr_name}' of type '{attr_type}'.")
    else:
        print(f"Prim '{prim.GetPath()}' has no attributes.")
    relationships = prim.GetRelationships()
    if relationships:
        print(f"Prim '{prim.GetPath()}' has {len(relationships)} relationships:")
        for rel in relationships:
            rel_name = rel.GetName()
            target_paths = rel.GetTargets()
            print(f"  - Relationship '{rel_name}' with targets: {target_paths}")
    else:
        print(f"Prim '{prim.GetPath()}' has no relationships.")


def get_enum_value_from_fullname(fullname: str) -> Tuple[int, bool]:
    """
    Returns the enumerated value for a fully-qualified name.

    This takes a fully-qualified enumerated value name (e.g., "Season.WINTER")
    and returns the associated value. If there is no such name, this returns -1.
    Since -1 can sometimes be a valid value, a boolean flag is returned to indicate
    whether the name was found or not.

    Parameters
    ----------
    fullname : str
        The fully-qualified enumerated value name.

    Returns
    -------
    Tuple[int, bool]
        A tuple containing the enumerated value and a boolean flag indicating
        whether the name was found or not.
    """
    found_it = False
    (enum_type_name, enum_value_name) = fullname.rsplit(".", 1)
    enum_type = Tf.Type.FindByName(enum_type_name)
    if enum_type is not None:
        enum_value = getattr(enum_type, enum_value_name, None)
        if enum_value is not None:
            found_it = True
            return (int(enum_value), found_it)
    return (-1, found_it)


class CustomError:

    def __init__(self, code: str, message: str):
        self.errorCode = code
        self.errorCodeString = message


def filter_errors_by_code(errors: List[CustomError], code: str) -> List[CustomError]:
    """Filter a list of CustomError objects by the specified error code.

    Args:
        errors (List[CustomError]): The list of CustomError objects to filter.
        code (str): The error code to filter by.

    Returns:
        List[CustomError]: A new list containing only the errors with the specified code.
    """
    if not errors:
        return []
    filtered_errors = [error for error in errors if error.errorCode == code]
    return filtered_errors


def execute_with_error_checking(func: Callable, *args, **kwargs) -> Any:
    """Execute a function with error checking using Tf.Error.Mark.

    Args:
        func (Callable): The function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        Any: The result of the function execution.
    """
    mark = Tf.Error.Mark()
    try:
        mark.SetMark()
        result = func(*args, **kwargs)
        if mark.IsClean():
            return result
        else:
            errors = mark.GetErrors()
            error_messages = [str(error) for error in errors]
            raise RuntimeError(f"Errors occurred during function execution: {', '.join(error_messages)}")
    finally:
        mark.Clear()


def test_function(value: int) -> int:
    """A test function that multiplies the input by 2."""
    return value * 2


def mark_and_execute(func: Callable) -> None:
    """Execute a function and collect any errors that occur.

    Args:
        func (Callable): The function to execute.
    """
    mark = Tf.Error.Mark()
    try:
        func()
        if mark.IsClean():
            print("Function executed successfully with no errors.")
        else:
            errors = mark.GetErrors()
            for error in errors:
                print(f"Error: {error}")
    except Exception as e:
        print(f"An exception occurred: {str(e)}")
    finally:
        mark.Clear()


def test_func1():
    print("Executing test_func1")


def test_func2():
    print("Executing test_func2")
    raise ValueError("An error occurred in test_func2")


def initialize_malloc_tag() -> bool:
    """Initialize the memory tagging system.

    Returns:
        bool: True if initialization succeeded, False otherwise.
    """
    err_msg = ""
    success = Tf.MallocTag.Initialize(err_msg)
    if not success:
        print(f"Failed to initialize memory tagging system: {err_msg}")
    return success


def get_max_memory_footprint() -> int:
    """Get the maximum total number of bytes that have ever been allocated."""
    if not Tf.MallocTag.IsInitialized():
        error_msg = ""
        if not Tf.MallocTag.Initialize(error_msg):
            raise RuntimeError(f"Failed to initialize MallocTag: {error_msg}")
    max_bytes = Tf.MallocTag.GetMaxTotalBytes()
    return max_bytes


def get_current_memory_footprint() -> int:
    """Get the current memory footprint in bytes."""
    if not Tf.MallocTag.IsInitialized():
        err_msg = ""
        if not Tf.MallocTag.Initialize(err_msg):
            raise RuntimeError(f"Failed to initialize memory tagging system: {err_msg}")
    total_bytes = Tf.MallocTag.GetTotalBytes()
    return total_bytes


def is_malloc_tag_initialized() -> bool:
    """Check if the MallocTag system is initialized."""
    is_initialized = Tf.MallocTag.IsInitialized()
    return is_initialized


def fetch_tracked_malloc_stacks() -> List[Tuple[str, List[str]]]:
    """
    Fetch the stack traces for tracked malloc tags.

    Returns:
        A list of tuples, where each tuple contains the malloc tag name and a list of stack frames.
    """
    if not Tf.MallocTag.IsInitialized():
        Tf.MallocTag.Initialize()
    captured_stacks = Tf.MallocTag.GetCallStacks()
    tracked_stacks = []
    for malloc_tag, stack_frames in captured_stacks:
        stack_trace = [str(frame) for frame in stack_frames]
        tracked_stacks.append((malloc_tag, stack_trace))
    return tracked_stacks


def track_memory_allocation(match_list: str) -> None:
    """
    Track memory allocations for tags matching the given list.

    Args:
        match_list (str): Comma, tab or newline separated list of malloc tag names to match.
    """
    if not Tf.MallocTag.IsInitialized():
        print("Memory tagging system is not initialized. Skipping memory tracking.")
        return
    Tf.MallocTag.SetCapturedMallocStacksMatchList(match_list)
    call_stacks = Tf.MallocTag.GetCallStacks()
    for stack in call_stacks:
        print(f"Memory allocation stack trace:")
        for frame in stack:
            print(f"  {frame}")


def enable_debug_on_tags(match_list: str) -> None:
    """Enable debugger traps for malloc tags matching the given list.

    Args:
        match_list (str): Comma, tab or newline separated list of malloc tag names.
                          Names can have internal spaces, but leading/trailing spaces are stripped.
                          If a name ends with '*', the suffix is wildcarded.
                          A name can have a leading '-' or '+' to prevent or allow a match.
                          Names are considered in order, with later matches overriding earlier ones.
                          An empty string disables debugging traps.

    Raises:
        TypeError: If match_list is not a string.

    Example:
        enable_debug_on_tags("Csd*, -CsdScene._Populate*, +CsdScene._PopulatePrimCacheLocal")
        This matches any malloc tag starting with 'Csd', but nothing starting with 'CsdScene._Populate',
        except 'CsdScene._PopulatePrimCacheLocal'.
    """
    if not isinstance(match_list, str):
        raise TypeError(f"Expected string for match_list, got {type(match_list)}")
    Tf.MallocTag.SetDebugMatchList(match_list)


def collect_memory_reports(num_reports: int = 3) -> List[str]:
    """Collect memory reports using Tf.MallocTag.CallTree.

    Args:
        num_reports (int): The number of memory reports to collect. Default is 3.

    Returns:
        List[str]: A list of memory report strings.
    """
    reports = []
    if num_reports <= 0:
        raise ValueError("num_reports must be a positive integer.")
    for _ in range(num_reports):
        root = Tf.MallocTag.CallTree.GetRoot()
        if not root:
            raise RuntimeError("Failed to get the call tree root.")
        report = Tf.MallocTag.CallTree.GetPrettyPrintString(root)
        reports.append(report)
    return reports


def compare_memory_reports(report1: Dict[str, int], report2: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    """Compare two memory reports and return a dictionary of differing call sites.

    Args:
        report1 (Dict[str, int]): The first memory report dictionary.
        report2 (Dict[str, int]): The second memory report dictionary.

    Returns:
        Dict[str, Tuple[int, int]]: A dictionary where the keys are the differing call sites,
            and the values are tuples of the form (bytes_in_report1, bytes_in_report2).
    """
    diff_sites = {}
    for site, bytes1 in report1.items():
        bytes2 = report2.get(site)
        if bytes2 is None or bytes1 != bytes2:
            diff_sites[site] = (bytes1, bytes2 or 0)
    for site, bytes2 in report2.items():
        if site not in report1:
            diff_sites[site] = (0, bytes2)
    return diff_sites


def generate_detailed_memory_analysis() -> str:
    """Generate a detailed memory analysis report using Tf.MallocTag."""
    try:
        call_tree = Tf.MallocTag.GetCallTree()
        if not call_tree:
            return "Failed to retrieve the call tree."
        call_sites = call_tree.GetCallSites()
        if not call_sites:
            return "No memory usage data available."
        report = call_tree.GetPrettyPrintString()
        call_tree.LogReport()
        return report
    except Exception as e:
        return f"An error occurred while generating the memory analysis report: {str(e)}"


class CallSite:

    def __init__(self):
        self._nBytes = None
        self._name = None

    @property
    def nBytes(self):
        return self._nBytes

    @nBytes.setter
    def nBytes(self, value):
        self._nBytes = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value


def aggregate_memory_usage(callsites: List[CallSite]) -> int:
    """Aggregate the memory usage from a list of CallSite objects.

    Args:
        callsites (List[CallSite]): A list of CallSite objects.

    Returns:
        int: The total memory usage in bytes.
    """
    total_bytes = 0
    for callsite in callsites:
        if callsite.nBytes is not None:
            total_bytes += callsite.nBytes
        else:
            continue
    return total_bytes


def filter_call_sites_by_memory_threshold(call_sites: List[Dict], threshold: int) -> List[Dict]:
    """Filter call sites based on a memory threshold.

    Args:
        call_sites (List[Dict]): A list of call sites represented as dictionaries.
        threshold (int): The memory threshold in bytes.

    Returns:
        List[Dict]: A list of call sites that exceed the memory threshold.
    """
    filtered_call_sites: List[Dict] = []
    for call_site in call_sites:
        if call_site["nBytes"] > threshold:
            filtered_call_sites.append(call_site)
    return filtered_call_sites


class PathNode:

    def __init__(self):
        self._children: List[PathNode] = []
        self.nAllocations: int = 0
        self.nBytes: int = 0
        self.nBytesDirect: int = 0
        self.siteName: str = ""

    def GetChildren(self) -> List["PathNode"]:
        return self._children


def analyze_memory_allocation_tree(root: PathNode) -> Dict[str, Dict[str, int]]:
    """Analyze the memory allocation tree and return a dictionary with site names and their allocation info.

    Args:
        root (PathNode): The root node of the memory allocation tree.

    Returns:
        Dict[str, Dict[str, int]]: A dictionary mapping site names to their allocation info.
                                   The allocation info includes 'nAllocations', 'nBytes', and 'nBytesDirect'.
    """
    result: Dict[str, Dict[str, int]] = {}

    def traverse(node: PathNode) -> None:
        site_name: str = node.siteName
        if site_name not in result:
            result[site_name] = {"nAllocations": 0, "nBytes": 0, "nBytesDirect": 0}
        result[site_name]["nAllocations"] += node.nAllocations
        result[site_name]["nBytes"] += node.nBytes
        result[site_name]["nBytesDirect"] += node.nBytesDirect
        for child in node.GetChildren():
            traverse(child)

    traverse(root)
    return result


class PathNode:

    def __init__(self, site_name: str, n_bytes: int):
        self.siteName = site_name
        self.nBytes = n_bytes
        self.children: List[PathNode] = []

    def GetChildren(self) -> List["PathNode"]:
        return self.children


def find_top_memory_consumers(root_node: PathNode, top_n: int = 5) -> List[Tuple[str, int]]:
    """
    Find the top N memory consumers in the call tree.

    Args:
        root_node (PathNode): The root node of the call tree.
        top_n (int, optional): The number of top memory consumers to return. Defaults to 5.

    Returns:
        List[Tuple[str, int]]: A list of tuples containing the site name and total bytes allocated for the top N memory consumers.
    """
    all_nodes: List[PathNode] = []

    def traverse_tree(node: PathNode):
        all_nodes.append(node)
        for child_node in node.GetChildren():
            traverse_tree(child_node)

    traverse_tree(root_node)
    sorted_nodes = sorted(all_nodes, key=lambda node: node.nBytes, reverse=True)
    return [(node.siteName, node.nBytes) for node in sorted_nodes[:top_n]]


class PathNode:

    def __init__(self, site_name: str):
        self.siteName = site_name
        self.children = []

    def add_child(self, child_node: "PathNode"):
        self.children.append(child_node)

    def GetChildren(self):
        return self.children


def get_allocation_path_to_site(node: PathNode, site_name: str) -> List[str]:
    """
    Recursively traverse the allocation call tree to find the path to a specific site.

    Args:
        node (PathNode): The current node in the call tree.
        site_name (str): The name of the site to find the path to.

    Returns:
        List[str]: The path to the specified site as a list of site names.
    """
    if node.siteName == site_name:
        return [node.siteName]
    for child_node in node.GetChildren():
        path = get_allocation_path_to_site(child_node, site_name)
        if path:
            return [node.siteName] + path
    return []


class MockPathNode:

    def __init__(self, site_name: str, n_allocations: int, n_bytes: int, n_bytes_direct: int):
        self._site_name = site_name
        self._n_allocations = n_allocations
        self._n_bytes = n_bytes
        self._n_bytes_direct = n_bytes_direct
        self._children: List[MockPathNode] = []

    def GetChildren(self) -> List["MockPathNode"]:
        return self._children

    @property
    def nAllocations(self) -> int:
        return self._n_allocations

    @property
    def nBytes(self) -> int:
        return self._n_bytes

    @property
    def nBytesDirect(self) -> int:
        return self._n_bytes_direct

    @property
    def siteName(self) -> str:
        return self._site_name


def compare_allocation_patterns(node1: MockPathNode, node2: MockPathNode) -> bool:
    """
    Compare the allocation patterns of two PathNodes.

    Args:
        node1 (MockPathNode): The first PathNode to compare.
        node2 (MockPathNode): The second PathNode to compare.

    Returns:
        bool: True if the allocation patterns are the same, False otherwise.
    """
    if node1.siteName != node2.siteName:
        return False
    if node1.nAllocations != node2.nAllocations:
        return False
    if node1.nBytes != node2.nBytes:
        return False
    if node1.nBytesDirect != node2.nBytesDirect:
        return False
    children1 = node1.GetChildren()
    children2 = node2.GetChildren()
    if len(children1) != len(children2):
        return False
    for child1, child2 in zip(children1, children2):
        if not compare_allocation_patterns(child1, child2):
            return False
    return True


def get_total_allocations_and_bytes(node: Tf.MallocTag.CallTree.PathNode) -> Tuple[int, int]:
    """
    Recursively traverse the CallTree.PathNode and its children to calculate the total allocations and bytes.

    Args:
        node (Tf.MallocTag.CallTree.PathNode): The root node to start the traversal from.

    Returns:
        Tuple[int, int]: A tuple containing the total allocations and total bytes.
    """
    total_allocations = node.nAllocations
    total_bytes = node.nBytes
    for child_node in node.GetChildren():
        (child_allocations, child_bytes) = get_total_allocations_and_bytes(child_node)
        total_allocations += child_allocations
        total_bytes += child_bytes
    return (total_allocations, total_bytes)


def handle_module_loaded_event(notice: Tf.PyModuleWasLoaded):
    """Handle the PyModuleWasLoaded event and print the loaded module name.

    Args:
        notice (Tf.PyModuleWasLoaded): The PyModuleWasLoaded notice.
    """
    module_name = notice.name()
    print(f"Loaded module: {module_name}")


class MockPyModuleWasLoaded:

    def __init__(self, module_name):
        self._module_name = module_name

    def name(self):
        return self._module_name


def send_global_notice(notice: Tf.Notice) -> int:
    """
    Deliver the notice to interested listeners globally.

    Args:
        notice (Tf.Notice): The notice to send.

    Returns:
        int: The number of interested listeners that received the notice.
    """
    if not isinstance(notice, Tf.Notice):
        raise TypeError("Invalid notice type. Expected Tf.Notice.")
    num_listeners = notice.SendGlobally()
    return num_listeners


def global_listener(notice):
    print("Global listener received notice:", type(notice))


class MockListener:
    """A mock class that mimics the behavior of Tf.Notice.Listener."""

    def __init__(self):
        self.revoked = False

    def Revoke(self):
        self.revoked = True


def manage_listener_connections(listeners: List[MockListener], revoke_indices: List[int]) -> List[MockListener]:
    """
    Manages a list of listener connections, revoking specific ones based on the provided indices.

    Args:
        listeners (List[MockListener]): A list of listener connections.
        revoke_indices (List[int]): A list of indices indicating which listeners to revoke.

    Returns:
        List[MockListener]: The updated list of listener connections.
    """
    updated_listeners = []
    for i, listener in enumerate(listeners):
        if i in revoke_indices:
            listener.Revoke()
        else:
            updated_listeners.append(listener)
    return updated_listeners


def filter_and_watch_prims(stage: Usd.Stage, prim_filter: Callable[[Usd.Prim], bool]) -> List[Tf.RefPtrTracker]:
    """
    Filter and watch prims in a USD stage using a custom filter function.

    Args:
        stage (Usd.Stage): The USD stage to filter prims from.
        prim_filter (Callable[[Usd.Prim], bool]): A function that takes a Usd.Prim and returns True if the prim should be watched.

    Returns:
        List[Tf.RefPtrTracker]: A list of RefPtrTrackers for the watched prims.
    """
    trackers: List[Tf.RefPtrTracker] = []
    for prim in stage.Traverse():
        if prim_filter(prim):
            tracker = Tf.RefPtrTracker(prim)
            trackers.append(tracker)
    return trackers


def is_mesh_prim(prim: Usd.Prim) -> bool:
    """Filter function to check if a prim is a mesh."""
    return prim.IsA(UsdGeom.Mesh)


def set_scope_description_for_operations(description: str):
    """Set the description for the current scope of operations.

    Args:
        description (str): The description to set for the current scope.

    Returns:
        None
    """
    scope_description = Tf.ScopeDescription(description)


def load_script_modules_for_prims(stage: Usd.Stage, prim_paths: List[str]) -> None:
    """
    Load script modules for the specified prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The paths of the prims to load script modules for.
    """
    loader = Tf.ScriptModuleLoader()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} is not valid. Skipping.")
            continue
        type_name = prim.GetTypeName()
        if type_name in loader.GetModulesDict():
            loader.LoadModule(type_name)
        else:
            print(f"Warning: No script module found for prim type {type_name}.")


def get_dependency_ordered_modules() -> List[str]:
    """Get a list of all currently known modules in a valid dependency order."""
    try:
        module_names = Tf.ScriptModuleLoader.GetModuleNames()
        if not module_names:
            return []
        return module_names
    except Exception as e:
        print(f"Error getting dependency ordered modules: {str(e)}")
        return []


def write_dependency_graph(file_path: str) -> None:
    """Write a graphviz dot-file for the dependency graph of all currently known modules.

    Args:
        file_path (str): The path to the output file.

    Raises:
        ValueError: If the file path is empty or None.
        IOError: If there is an error writing to the file.
    """
    if not file_path:
        raise ValueError("File path cannot be empty or None.")
    try:
        module_names = Tf.ScriptModuleLoader().GetModuleNames()
        modules_dict = Tf.ScriptModuleLoader().GetModulesDict()
        with open(file_path, "w") as file:
            file.write("digraph DependencyGraph {\n")
            for module_name in module_names:
                file.write(f'  "{module_name}";\n')
            for module_name, module in modules_dict.items():
                for dependency in module.__dict__.get("__depends__", []):
                    file.write(f'  "{module_name}" -> "{dependency}";\n')
            file.write("}\n")
    except IOError as e:
        raise IOError(f"Error writing to file: {e}")


def get_modules_info() -> Dict[str, Any]:
    """Get information about currently known USD modules.

    Returns:
        Dict[str, Any]: A dictionary containing module information.
            - 'module_names' (List[str]): A list of module names in dependency order.
            - 'modules_dict' (Dict[str, Any]): A dictionary of module objects keyed by their canonical names.
    """
    module_names: List[str] = Tf.ScriptModuleLoader().GetModuleNames()
    modules_dict: Dict[str, Any] = Tf.ScriptModuleLoader().GetModulesDict()
    modules_info: Dict[str, Any] = {"module_names": module_names, "modules_dict": modules_dict}
    return modules_info


def register_and_load_libraries(module_names: List[str]) -> None:
    """
    Register and load the given module names using ScriptModuleLoader.

    Args:
        module_names (List[str]): A list of module names to register and load.

    Raises:
        ValueError: If any of the module names are not found in the registered modules.
    """
    modules_dict = Tf.ScriptModuleLoader().GetModulesDict()
    for module_name in module_names:
        if module_name not in modules_dict:
            raise ValueError(f"Module '{module_name}' not found in registered modules.")
        Tf.ScriptModuleLoader()._LoadModulesForLibrary(module_name)


def get_singleton_instance(cls):
    """Get the singleton instance of the class.

    If the singleton instance does not exist, create it and return it.
    If the singleton instance already exists, return the existing instance.
    """
    if not hasattr(cls, "_singleton_instance"):
        cls._singleton_instance = cls()
    return cls._singleton_instance


class MySingleton(Tf.Singleton):

    def __init__(self):
        self.value = 42


def list_all_singletons() -> List[Tf.Type]:
    """List all singleton types in the TfType registry."""
    singleton_base_type = Tf.Type.Find(Tf.Singleton)
    singleton_types = singleton_base_type.GetAllDerivedTypes()
    return list(singleton_types)


def monitor_and_report_transform_changes(stage: Usd.Stage, prim_path: str):
    """Monitor a prim's transform changes and report them using Tf.Status"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    translate_op = add_translate_op(xformable)
    rotate_op = add_rotate_xyz_op(xformable)
    scale_op = add_scale_op(xformable)
    translate = translate_op.Get()
    rotate = rotate_op.Get()
    scale = scale_op.Get()

    def notice_listener(notice, stage, listener):
        if notice.GetStage() != stage or not notice.ResyncedObject(stage.GetPrimAtPath(prim_path)):
            return
        new_translate = translate_op.Get()
        new_rotate = rotate_op.Get()
        new_scale = scale_op.Get()
        if not Gf.IsClose(new_translate, translate, 1e-05):
            Tf.Status("Translation changed").Msg(f"Previous: {translate}, Current: {new_translate}")
        if not Gf.IsClose(new_rotate, rotate, 1e-05):
            Tf.Status("Rotation changed").Msg(f"Previous: {rotate}, Current: {new_rotate}")
        if not Gf.IsClose(new_scale, scale, 1e-05):
            Tf.Status("Scale changed").Msg(f"Previous: {scale}, Current: {new_scale}")

    listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, notice_listener, stage)
    return listener


def measure_prim_creation_time(stage: Usd.Stage, prim_type: str, prim_path: str, num_prims: int) -> float:
    """Measure the time it takes to create a specified number of prims of a given type.

    Args:
        stage (Usd.Stage): The USD stage to create the prims on.
        prim_type (str): The type of prim to create (e.g., "Xform", "Mesh", "Sphere").
        prim_path (str): The base path where the prims will be created.
        num_prims (int): The number of prims to create.

    Returns:
        float: The time taken to create the prims in seconds.
    """
    if num_prims <= 0:
        raise ValueError("num_prims must be a positive integer.")
    stopwatch = Tf.Stopwatch()
    stopwatch.Start()
    for i in range(num_prims):
        path = f"{prim_path}_{i}"
        if prim_type == "Xform":
            UsdGeom.Xform.Define(stage, path)
        elif prim_type == "Mesh":
            UsdGeom.Mesh.Define(stage, path)
        elif prim_type == "Sphere":
            UsdGeom.Sphere.Define(stage, path)
        else:
            raise ValueError(f"Unsupported prim type: {prim_type}")
    stopwatch.Stop()
    elapsed_time = stopwatch.seconds
    return elapsed_time


def measure_hierarchy_traversal_time(stage: Usd.Stage, root_prim_path: str) -> float:
    """Measure the time it takes to traverse the hierarchy of a given root prim."""
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid():
        raise ValueError(f"Invalid root prim path: {root_prim_path}")
    stopwatch = Tf.Stopwatch()
    stopwatch.Start()
    for prim in Usd.PrimRange(root_prim):
        _ = prim.GetName()
        _ = prim.GetTypeName()
    stopwatch.Stop()
    return stopwatch.seconds


def measure_property_change_time(
    stage: Usd.Stage, prim_path: str, property_name: str, num_iterations: int = 100
) -> float:
    """Measure the time it takes to change a property value on a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        property_name (str): The name of the property to modify.
        num_iterations (int, optional): The number of iterations to perform. Defaults to 100.

    Returns:
        float: The average time in seconds to change the property value.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim not found at path: {prim_path}")
    property = prim.GetProperty(property_name)
    if not property:
        raise ValueError(f"Property '{property_name}' not found on prim: {prim_path}")
    stopwatch = Tf.Stopwatch()
    for _ in range(num_iterations):
        stopwatch.Start()
        property.Set(0.0)
        stopwatch.Stop()
    average_time = stopwatch.seconds / num_iterations
    return average_time


def measure_attribute_modification_time(stage: Usd.Stage, prim_path: str, attr_name: str) -> Tuple[float, float]:
    """
    Measure the time it takes to modify an attribute on a prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.
        attr_name (str): The name of the attribute to modify.

    Returns:
        Tuple[float, float]: A tuple containing the start and end times of the attribute modification.

    Raises:
        ValueError: If the prim or attribute does not exist.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    attr = prim.GetAttribute(attr_name)
    if not attr.IsValid():
        raise ValueError(f"Attribute {attr_name} does not exist on prim at path {prim_path}.")
    start_time = time.time()
    attr.Set(42.0)
    end_time = time.time()
    return (start_time, end_time)


def measure_prim_removal_time(stage: Usd.Stage, prim_path: str) -> Tuple[float, bool]:
    """
    Measure the time it takes to remove a prim from a stage.

    Args:
        stage (Usd.Stage): The stage to remove the prim from.
        prim_path (str): The path of the prim to remove.

    Returns:
        Tuple[float, bool]: A tuple containing the elapsed time in seconds and a boolean
            indicating whether the prim was successfully removed.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    start_time = time.time()
    success = prim.SetActive(False)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return (elapsed_time, success)


def measure_material_assignment_time(stage: Usd.Stage, prim_path: str, material_path: str) -> Tuple[float, float]:
    """Measure the time it takes to assign a material to a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to assign the material to.
        material_path (str): The path to the material to assign.

    Returns:
        Tuple[float, float]: A tuple containing the elapsed time in seconds and the number of samples.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    if not material:
        raise ValueError(f"Material at path {material_path} does not exist.")
    stopwatch = Tf.Stopwatch()
    stopwatch.Start()
    UsdShade.MaterialBindingAPI(prim).Bind(material)
    stopwatch.Stop()
    elapsed_time = stopwatch.seconds
    sample_count = stopwatch.sampleCount
    return (elapsed_time, sample_count)


def measure_prim_copy_time(stage: Usd.Stage, source_prim_path: str, dest_prim_path: str) -> float:
    """Measure the time it takes to copy a prim within a stage.

    Args:
        stage (Usd.Stage): The USD stage.
        source_prim_path (str): The path of the source prim to be copied.
        dest_prim_path (str): The path of the destination prim.

    Returns:
        float: The time taken to copy the prim in seconds.

    Raises:
        ValueError: If the source prim does not exist or is not valid.
    """
    source_prim = stage.GetPrimAtPath(source_prim_path)
    if not source_prim.IsValid():
        raise ValueError(f"Source prim at path {source_prim_path} does not exist or is not valid.")
    stopwatch = Tf.Stopwatch()
    stopwatch.Start()
    dest_prim = stage.DefinePrim(dest_prim_path)
    dest_prim.GetReferences().AddReference(assetPath="", primPath=source_prim.GetPath())
    stopwatch.Stop()
    return stopwatch.seconds


def substitute_template_for_prims(template: Tf.TemplateString, prim_paths: List[Sdf.Path]) -> List[str]:
    """
    Substitute a template string with prim paths and return the resulting strings.

    Args:
        template (Tf.TemplateString): The template string to substitute.
        prim_paths (List[Sdf.Path]): A list of prim paths to substitute into the template.

    Returns:
        List[str]: A list of strings with the template substituted for each prim path.

    Raises:
        ValueError: If the template is invalid or if any prim path is invalid.
    """
    if not template.valid:
        raise ValueError("Invalid template string")
    results = []
    for prim_path in prim_paths:
        if not prim_path.IsAbsolutePath():
            raise ValueError(f"Invalid prim path: {prim_path}")
        mapping = {"path": str(prim_path), "name": prim_path.name, "parent": str(prim_path.GetParentPath())}
        substituted_str = template.SafeSubstitute(mapping)
        results.append(substituted_str)
    return results


def substitute_template_with_fallback(
    template_string: Tf.TemplateString, mapping: Dict[str, str], fallback_mapping: Dict[str, str]
) -> str:
    """
    Performs template substitution with a fallback mapping.

    If a placeholder is not found in the primary mapping, it will attempt to use the fallback mapping.
    If the placeholder is not found in either mapping, the original placeholder will be left intact.

    Args:
        template_string (Tf.TemplateString): The template string to perform substitution on.
        mapping (Dict[str, str]): The primary mapping of placeholders to their corresponding values.
        fallback_mapping (Dict[str, str]): The fallback mapping to use if a placeholder is not found in the primary mapping.

    Returns:
        str: The substituted string with placeholders replaced by their corresponding values.
    """
    combined_mapping = {**fallback_mapping, **mapping}
    substituted_string = template_string.SafeSubstitute(combined_mapping)
    return substituted_string


def batch_substitute_templates(templates: List[Tf.TemplateString], mapping: Dict[str, str]) -> List[str]:
    """
    Performs batch substitution on a list of templates using the provided mapping.

    Args:
        templates (List[Tf.TemplateString]): A list of template strings.
        mapping (Dict[str, str]): A mapping of placeholders to their corresponding values.

    Returns:
        List[str]: A list of substituted strings.

    Raises:
        ValueError: If any of the templates are invalid or if a placeholder is missing from the mapping.
    """
    substituted_strings: List[str] = []
    for template in templates:
        if not template.valid:
            raise ValueError(f"Invalid template: {template.template}")
        try:
            substituted_string = template.Substitute(mapping)
            substituted_strings.append(substituted_string)
        except Tf.ErrorException as e:
            raise ValueError(f"Missing placeholder in mapping for template: {template.template}") from e
    return substituted_strings


def verify_and_substitute_template(template_string: Tf.TemplateString, mapping: Mapping) -> str:
    """
    Verifies the validity of a template string and performs substitution with the provided mapping.

    Args:
        template_string (Tf.TemplateString): The template string to verify and substitute.
        mapping (Mapping): The mapping of placeholders to their corresponding values.

    Returns:
        str: The resulting string after template substitution.

    Raises:
        ValueError: If the template string is invalid or if any placeholders are missing from the mapping.
    """
    if not template_string.valid:
        parse_errors = template_string.GetParseErrors()
        raise ValueError(f"Invalid template string. Parse errors: {', '.join(parse_errors)}")
    try:
        result = template_string.Substitute(mapping)
    except Tf.ErrorException as e:
        missing_placeholders = str(e).split(":")[1].strip().split(", ")
        raise ValueError(f"Missing placeholders in the mapping: {', '.join(missing_placeholders)}")
    return result


def create_and_substitute_template(template: str, mapping: Dict[str, str]) -> str:
    """
    Create a Tf.TemplateString from the given template string and substitute values from the mapping.

    Args:
        template (str): The template string to create the Tf.TemplateString from.
        mapping (Dict[str, str]): A dictionary mapping placeholder names to their corresponding values.

    Returns:
        str: The substituted string with placeholders replaced by their mapped values.

    Raises:
        ValueError: If the template string is invalid.
    """
    template_string = Tf.TemplateString(template)
    if not template_string.valid:
        parse_errors = template_string.GetParseErrors()
        error_message = "Invalid template string. Parse errors:\n" + "\n".join(parse_errors)
        raise ValueError(error_message)
    substituted_string = template_string.SafeSubstitute(mapping)
    return substituted_string


def validate_and_parse_template(template: str) -> Tf.TemplateString:
    """
    Validate and parse a template string.

    Args:
        template (str): The template string to validate and parse.

    Returns:
        Tf.TemplateString: The parsed template string if valid.

    Raises:
        ValueError: If the template string is invalid.
    """
    ts = Tf.TemplateString(template)
    if not ts.valid:
        parse_errors = ts.GetParseErrors()
        error_message = "Invalid template string:\n" + "\n".join(parse_errors)
        raise ValueError(error_message)
    return ts


def get_all_base_types(tf_type: Type) -> list[Type]:
    """
    Get all base types for a given TfType.

    Args:
        tf_type (Type): The TfType to get base types for.

    Returns:
        list[Type]: A list of all base types for the given TfType.
    """
    base_types = []
    if not tf_type:
        return base_types
    for base_type in tf_type.baseTypes:
        base_types.append(base_type)
        base_types.extend(get_all_base_types(base_type))
    return base_types


class BaseType1(Tf.Type):
    pass


class BaseType2(Tf.Type):
    pass


class DerivedType(BaseType1, BaseType2):
    pass


def get_direct_derived_types(tf_type: Type) -> list[Type]:
    """
    Get the direct derived types of the given TfType.

    Args:
        tf_type (Type): The TfType to get the direct derived types for.

    Returns:
        list[Type]: A list of the direct derived types.
    """
    direct_derived_types: set[Type] = set()
    for derived_type in tf_type.GetAllDerivedTypes():
        if tf_type in derived_type.baseTypes:
            direct_derived_types.add(derived_type)
    return list(direct_derived_types)


class A(Tf.Type):
    pass


class B(A):
    pass


class C(A):
    pass


class D(B, C):
    pass


def get_type_size(type_obj: Tf.Type) -> int:
    """Get the size of a type in bytes.

    Args:
        type_obj (Tf.Type): The type object to get the size of.

    Returns:
        int: The size of the type in bytes.
    """
    if not type_obj:
        raise ValueError("Invalid type object")
    size = type_obj.sizeof
    return size


def find_type_by_cpp_type_name(cpp_type_name: str) -> Tf.Type:
    """
    Find a TfType by its C++ type name.

    Args:
        cpp_type_name (str): The C++ type name to search for.

    Returns:
        Tf.Type: The found TfType, or Tf.Type.Unknown if not found.
    """
    root_type = Tf.Type.GetRoot()
    found_type = root_type.FindDerivedByName(cpp_type_name)
    if found_type != Tf.Type.Unknown:
        return found_type
    else:
        pxr_type_name = "pxr::" + cpp_type_name
        found_type = root_type.FindDerivedByName(pxr_type_name)
        return found_type


def compare_types_by_name(type_a: Tf.Type, type_b: Tf.Type) -> int:
    """Compare two TfTypes by their type names.

    Args:
        type_a (Tf.Type): The first type to compare.
        type_b (Tf.Type): The second type to compare.

    Returns:
        int: -1 if type_a < type_b, 0 if equal, 1 if type_a > type_b.
    """
    name_a = type_a.typeName
    name_b = type_b.typeName
    if name_a < name_b:
        return -1
    elif name_a > name_b:
        return 1
    else:
        return 0


def get_all_types_in_hierarchy(root_type: type) -> List[type]:
    """
    Get all types in the hierarchy starting from the given root type.

    This function traverses the type hierarchy starting from the given root type
    and returns a list of all types in the hierarchy, including the root type.

    Args:
        root_type (type): The root type to start the traversal from.

    Returns:
        List[type]: A list of all types in the hierarchy.
    """
    result = []
    queue = [root_type]
    while queue:
        current_type = queue.pop(0)
        result.append(current_type)
        subclasses = current_type.__subclasses__()
        queue.extend(subclasses)
    return result


class Root:
    pass


class Child1(Root):
    pass


class Child2(Root):
    pass


class Grandchild1(Child1):
    pass


class Grandchild2(Child2):
    pass


def get_all_aliases_for_type(tf_type: Tf.Type) -> List[str]:
    """
    Get all aliases for a given TfType.

    This function traverses the type hierarchy starting from the root type
    and collects all aliases defined for the given type under each ancestor.

    Parameters:
        tf_type (Tf.Type): The TfType to get aliases for.

    Returns:
        List[str]: A list of all aliases defined for the given type.
    """
    aliases = []
    root_type = Tf.Type.GetRoot()
    to_visit = [root_type]
    while to_visit:
        current_type = to_visit.pop()
        current_aliases = current_type.GetAliases(tf_type)
        aliases.extend(current_aliases)
        to_visit.extend(current_type.baseTypes)
    return aliases


class ClassA(object):
    pass


class ClassB(ClassA):
    pass


class ClassC(ClassB):
    pass


def find_type_by_name_in_hierarchy(base_type: Tf.Type, type_name: str) -> Tf.Type:
    """
    Recursively search for a type with the given name in the type hierarchy starting from the base type.

    Args:
        base_type (Tf.Type): The base type to start the search from.
        type_name (str): The name of the type to find.

    Returns:
        Tf.Type: The found type, or Tf.Type.Unknown if not found.
    """
    if base_type.typeName == type_name:
        return base_type
    alias_type = base_type.FindDerivedByName(type_name)
    if alias_type != Tf.Type.Unknown:
        return alias_type
    for derived_type in base_type.derivedTypes:
        found_type = find_type_by_name_in_hierarchy(derived_type, type_name)
        if found_type != Tf.Type.Unknown:
            return found_type
    return Tf.Type.Unknown


class CustomBaseType:
    pass


class CustomDerivedType1(CustomBaseType):
    pass


class CustomDerivedType2(CustomDerivedType1):
    pass


class CustomDerivedType3(CustomDerivedType2):
    pass


def get_python_class_for_type(tf_type: Tf.Type) -> Optional[Type]:
    """
    Get the Python class object for a given Tf.Type.

    Args:
        tf_type (Tf.Type): The Tf.Type to get the Python class for.

    Returns:
        Optional[Type]: The Python class object if found, None otherwise.
    """
    if tf_type.isUnknown:
        return None
    py_class = tf_type.pythonClass
    if py_class is None:
        return None
    return py_class


def get_type_information(type_name: str) -> Optional[Tf.Type]:
    """
    Retrieves the Tf.Type object for the given type name.

    Args:
        type_name (str): The name of the type to retrieve information for.

    Returns:
        Optional[Tf.Type]: The Tf.Type object if found, None otherwise.
    """
    tf_type = Tf.Type.FindByName(type_name)
    if not tf_type:
        return None
    return tf_type


def get_type_hierarchy_as_string(tf_type: Tf.Type) -> str:
    """Return a string representation of the type hierarchy for the given TfType."""
    if not isinstance(tf_type, Tf.Type):
        raise TypeError(f"Expected a TfType, got {type(tf_type)}")
    ancestors = Tf.Type.GetAllAncestorTypes(tf_type)
    type_names = [tf_type.typeName]
    for ancestor in ancestors[1:]:
        type_names.append(ancestor.typeName)
    hierarchy_string = " -> ".join(type_names)
    return hierarchy_string


def filter_warnings_by_message(warnings: List[Warning], message: str) -> List[Warning]:
    """Filter warnings by message.

    Args:
        warnings (List[Warning]): A list of Warning objects.
        message (str): The message to filter by.

    Returns:
        List[Warning]: A list of Warning objects that contain the specified message.
    """
    if not isinstance(warnings, list):
        raise TypeError("warnings must be a list")
    for warning in warnings:
        if not isinstance(warning, Warning):
            raise TypeError("All elements in warnings must be Warning objects")
    if not isinstance(message, str):
        raise TypeError("message must be a string")
    filtered_warnings = [warning for warning in warnings if message in str(warning)]
    return filtered_warnings
