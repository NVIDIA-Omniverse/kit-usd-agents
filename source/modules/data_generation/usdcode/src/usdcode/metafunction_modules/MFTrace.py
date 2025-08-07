## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from typing import List, Optional, Tuple

from pxr import Gf, Sdf, Trace, Usd, UsdGeom, UsdShade


class AggregateNode:

    def __init__(self):
        self._count = 0
        self._children: List[AggregateNode] = []

    @property
    def count(self) -> int:
        return self._count

    @count.setter
    def count(self, value: int):
        self._count = value

    @property
    def children(self) -> List["AggregateNode"]:
        return self._children


def get_total_call_count(node: AggregateNode, recursive: bool = True) -> int:
    """
    Get the total call count for the given AggregateNode.

    Args:
        node (AggregateNode): The AggregateNode to get the total call count for.
        recursive (bool, optional): Whether to include the call counts of child nodes. Defaults to True.

    Returns:
        int: The total call count.
    """
    total_count = node.count
    if recursive:
        for child_node in node.children:
            total_count += get_total_call_count(child_node, recursive)
    return total_count


def find_node_by_key(root_node: Trace.AggregateNode, key: str) -> Optional[Trace.AggregateNode]:
    """
    Recursively searches for a node with the given key in the tree starting from the root node.

    Args:
        root_node (Trace.AggregateNode): The root node of the tree to search.
        key (str): The key of the node to find.

    Returns:
        Optional[Trace.AggregateNode]: The node with the given key if found, None otherwise.
    """
    if root_node.key == key:
        return root_node
    for child_node in root_node.GetChildren():
        found_node = find_node_by_key(child_node, key)
        if found_node is not None:
            return found_node
    return None


class MockAggregateNode:

    def __init__(self, key):
        self.key = key
        self.children = []

    def GetChildren(self):
        return self.children


def expand_all_nodes(node: Trace.AggregateNode) -> None:
    """Recursively expand all nodes in the tree."""
    if node is None or node.expired:
        raise ValueError("Invalid or expired node.")
    node.expanded = True
    for child_node in node.children:
        expand_all_nodes(child_node)


def summarize_aggregate_tree(node: Trace.AggregateNode, depth: int = 0) -> List[Tuple[int, str, int, float]]:
    """
    Summarize an aggregate tree node and its children recursively.

    Args:
        node (Trace.AggregateNode): The root node of the aggregate tree.
        depth (int): The current depth in the tree. Defaults to 0.

    Returns:
        List[Tuple[int, str, int, float]]: A list of tuples representing the tree summary.
            Each tuple contains (depth, key, inclusive count, inclusive time).
    """
    summary = []
    summary.append((depth, node.key, node.count, node.inclusiveTime))
    for child in node.children:
        child_summary = summarize_aggregate_tree(child, depth + 1)
        summary.extend(child_summary)
    return summary


class MockAggregateNode:

    def __init__(self, key, count, inclusive_time, children=None):
        self._key = key
        self._count = count
        self._inclusiveTime = inclusive_time
        self._children = children if children else []

    @property
    def key(self):
        return self._key

    @property
    def count(self):
        return self._count

    @property
    def inclusiveTime(self):
        return self._inclusiveTime

    @property
    def children(self):
        return self._children


class AggregateNode:
    """Simplified version of Trace.AggregateNode for testing purposes."""

    def __init__(self, key="", count=0, exclusive_count=0, inclusive_time=0.0, exclusive_time=0.0):
        self._key = key
        self._count = count
        self._exclusive_count = exclusive_count
        self._inclusive_time = inclusive_time
        self._exclusive_time = exclusive_time

    @property
    def key(self):
        return self._key

    @property
    def count(self):
        return self._count

    @property
    def exclusiveCount(self):
        return self._exclusive_count

    @property
    def inclusiveTime(self):
        return self._inclusive_time

    @property
    def exclusiveTime(self):
        return self._exclusive_time


def filter_aggregate_nodes_by_time(nodes: List[AggregateNode], min_time: float, max_time: float) -> List[AggregateNode]:
    """
    Filter a list of AggregateNodes based on their inclusive time.

    Args:
        nodes (List[AggregateNode]): The list of nodes to filter.
        min_time (float): The minimum inclusive time for a node to be included.
        max_time (float): The maximum inclusive time for a node to be included.

    Returns:
        List[AggregateNode]: The filtered list of nodes.
    """
    if min_time < 0 or max_time < 0:
        raise ValueError("min_time and max_time must be non-negative.")
    if min_time > max_time:
        raise ValueError("min_time must be less than or equal to max_time.")
    filtered_nodes = []
    for node in nodes:
        inclusive_time = node.inclusiveTime
        if min_time <= inclusive_time <= max_time:
            filtered_nodes.append(node)
    return filtered_nodes


def trace_bulk_property_change(
    collector: Trace.Collector, prim: Usd.Prim, property_names: List[str], key: Optional[str] = None
):
    """
    Record TraceEvents for changes to multiple properties on a prim.

    Args:
        collector (Trace.Collector): The collector instance to record events.
        prim (Usd.Prim): The prim whose properties are being changed.
        property_names (List[str]): The names of the properties being changed.
        key (Optional[str]): Optional key to use for the trace events.
            If not provided, a default key based on the prim path will be used.

    Returns:
        None
    """
    if not collector or not collector.enabled:
        return
    if not prim:
        raise ValueError("Invalid prim.")
    if not property_names:
        raise ValueError("No properties specified.")
    if not key:
        key = prim.GetPath().pathString
    event_key = key + "_BulkPropertyChange"
    ts_start = collector.BeginEvent(event_key)
    try:
        for prop_name in property_names:
            prop_key = key + "_" + prop_name
            prop_ts = collector.BeginEvent(prop_key)
            collector.EndEvent(prop_key)
    finally:
        collector.EndEvent(event_key)


def trace_advanced_material_manipulation(stage: Usd.Stage, prim_path: str, material_path: str):
    """Trace advanced material manipulation operations on a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Invalid prim path: {prim_path}")
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim:
        raise ValueError(f"Invalid material path: {material_path}")
    with Trace.TraceScope("Bind material"):
        binding_api = UsdShade.MaterialBindingAPI(prim)
        material = UsdShade.Material(material_prim)
        binding_api.Bind(material)
    with Trace.TraceScope("Modify material properties"):
        shader_output = material.GetSurfaceOutput()
        if shader_output:
            shader_prim = shader_output.GetConnectedSource()[0]
            if shader_prim:
                shader = UsdShade.Shader(shader_prim)
                with Trace.TraceScope("Modify shader parameters"):
                    diffuse_color_input = shader.GetInput("diffuseColor")
                    if diffuse_color_input:
                        diffuse_color_input.Set(Gf.Vec3f(1.0, 0.0, 0.0))
                    roughness_input = shader.GetInput("roughness")
                    if roughness_input:
                        roughness_input.Set(0.5)
                with Trace.TraceScope("Create and connect texture"):
                    texture_path = f"{material_path}/Texture"
                    texture = UsdShade.Shader.Define(stage, texture_path)
                    texture.CreateIdAttr("UsdUVTexture")
                    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("texture.png")
                    normal_input = shader.GetInput("normal")
                    if normal_input:
                        normal_input.ConnectToSource(texture.ConnectableAPI(), "rgb")


def compare_performance_reports(baseline_report: List[str], current_report: List[str]) -> List[str]:
    """Compare two performance reports and return a list of differences."""
    baseline_timings = {}
    current_timings = {}
    for line in baseline_report:
        if ":" in line:
            (key, value) = line.split(":")
            baseline_timings[key.strip()] = float(value.strip())
    for line in current_report:
        if ":" in line:
            (key, value) = line.split(":")
            current_timings[key.strip()] = float(value.strip())
    differences = []
    for key in baseline_timings:
        if key in current_timings:
            baseline_time = baseline_timings[key]
            current_time = current_timings[key]
            if baseline_time != 0:
                percentage_diff = (current_time - baseline_time) / baseline_time * 100
            else:
                percentage_diff = float("inf") if current_time != 0 else 0
            differences.append(f"{key}: {percentage_diff:.2f}%")
        else:
            differences.append(f"{key}: Missing in current report")
    for key in current_timings:
        if key not in baseline_timings:
            differences.append(f"{key}: New in current report")
    return differences


def set_reporter_properties(
    reporter: Trace.Reporter,
    fold_recursive_calls: bool = True,
    group_by_function: bool = True,
    adjust_for_overhead_and_noise: bool = True,
) -> None:
    """Set properties of a Trace.Reporter.

    Args:
        reporter (Trace.Reporter): The reporter to set properties for.
        fold_recursive_calls (bool, optional): Whether to fold recursive calls in stack trace event reporting. Defaults to True.
        group_by_function (bool, optional): Whether to group events by function in stack trace event reporting. Defaults to True.
        adjust_for_overhead_and_noise (bool, optional): Whether the reporter should adjust scope times for overhead and noise. Defaults to True.
    """
    reporter.foldRecursiveCalls = fold_recursive_calls
    reporter.groupByFunction = group_by_function
    reporter.shouldAdjustForOverheadAndNoise = adjust_for_overhead_and_noise


def create_chrome_trace_report(reporter: Trace.Reporter, output_file: str) -> None:
    """Generate a Chrome Trace report from a Trace.Reporter and write it to a file.

    Args:
        reporter (Trace.Reporter): The Trace.Reporter instance containing the trace data.
        output_file (str): The path to the output file where the report will be written.

    Raises:
        ValueError: If the reporter is expired or the output file path is empty.
    """
    if reporter.expired:
        raise ValueError("Cannot generate report from an expired Trace.Reporter.")
    if not output_file:
        raise ValueError("Output file path cannot be empty.")
    reporter.UpdateTraceTrees()
    with open(output_file, "w") as file:
        reporter.ReportChromeTracingToFile(output_file)


def export_performance_data(reporter: Trace.Reporter, filepath: str, iteration_count: int = 1) -> None:
    """
    Export performance data from a Trace Reporter to a file.

    Args:
        reporter (Trace.Reporter): The Trace Reporter instance containing the performance data.
        filepath (str): The path to the file where the performance data will be written.
        iteration_count (int, optional): The number of iterations to divide the times by. Defaults to 1.

    Raises:
        ValueError: If the provided reporter is not a valid Trace.Reporter instance.
        IOError: If there is an error writing to the specified file path.
    """
    if not isinstance(reporter, Trace.Reporter):
        raise ValueError("Invalid reporter instance provided.")
    reporter.UpdateTraceTrees()
    try:
        with open(filepath, "w") as file:
            reporter.Report(filepath, iteration_count)
    except IOError as e:
        raise IOError(f"Error writing to file '{filepath}': {str(e)}") from e


def visualize_trace_tree(reporter: Trace.Reporter, output_file: str) -> None:
    """
    Visualize the trace tree from a Trace.Reporter and save it to a file.

    Args:
        reporter (Trace.Reporter): The reporter containing the trace data.
        output_file (str): The path to the output file.

    Raises:
        ValueError: If the reporter is expired or the output file path is empty.
    """
    if reporter.expired:
        raise ValueError("The provided reporter has expired.")
    if not output_file:
        raise ValueError("The output file path cannot be empty.")
    reporter.UpdateTraceTrees()
    root_node = reporter.aggregateTreeRoot
    with open(output_file, "w") as file:
        file.write("digraph TraceTree {\n")
        stack = [root_node]
        while stack:
            node = stack.pop()
            file.write(
                f"""  "{id(node)}" [label="{(node.key.GetName() if hasattr(node.key, 'GetName') else str(node.key))}"];\n"""
            )
            for child in node.children:
                stack.append(child)
                file.write(f'  "{id(node)}" -> "{id(child)}";\n')
        file.write("}\n")


def filter_events_by_time(reporter: Trace.Reporter, min_time: float, max_time: float) -> List[Trace.AggregateNode]:
    """Filter trace events by inclusive time range.

    Args:
        reporter (Trace.Reporter): The trace reporter.
        min_time (float): The minimum inclusive time in seconds.
        max_time (float): The maximum inclusive time in seconds.

    Returns:
        List[Trace.AggregateNode]: The filtered list of trace aggregate nodes.
    """
    root_node = reporter.aggregateTreeRoot
    filtered_nodes = []

    def traverse_node(node):
        if min_time <= node.exclusiveTime <= max_time:
            filtered_nodes.append(node)
        for child_node in node.children:
            traverse_node(child_node)

    traverse_node(root_node)
    return filtered_nodes


def event1():
    pass


def event2():
    pass


def event3():
    pass


def event4():
    pass


def generate_detailed_report(reporter: Trace.Reporter, output_file: str, iteration_count: int = 1) -> None:
    """
    Generate a detailed report of the trace events captured by the reporter.

    Args:
        reporter (Trace.Reporter): The reporter object containing the trace events.
        output_file (str): The path to the output file to write the report to.
        iteration_count (int, optional): The number of iterations to divide the times by. Defaults to 1.

    Raises:
        ValueError: If the reporter is expired or the iteration count is less than or equal to zero.
    """
    if reporter.expired:
        raise ValueError("Cannot generate report from an expired reporter.")
    if iteration_count <= 0:
        raise ValueError("Iteration count must be greater than zero.")
    reporter.UpdateTraceTrees()
    reporter.Report(output_file, iterationCount=iteration_count)
