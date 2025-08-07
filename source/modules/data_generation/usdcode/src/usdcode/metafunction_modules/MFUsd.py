## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import json
import os
import random
import shutil
import string
import tempfile
import zipfile
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from pxr import Ar, Gf, Kind, Pcp, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdSkel, Vt

from .add_op import *


def collect_prims_with_api_schema(stage: Usd.Stage, api_schema_type: Tf.Type) -> List[Usd.Prim]:
    """
    Collect all prims on the given stage that have the specified applied API schema.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        api_schema_type (Tf.Type): The TfType of the applied API schema to search for.

    Returns:
        List[Usd.Prim]: A list of prims that have the specified applied API schema.
    """
    if not Usd.SchemaRegistry.IsAppliedAPISchema(api_schema_type):
        raise ValueError(f"{api_schema_type} is not a valid applied API schema.")
    prims_with_api_schema: List[Usd.Prim] = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.HasAPI(api_schema_type):
            prims_with_api_schema.append(prim)
    return prims_with_api_schema


def apply_api_schema_to_prims(stage: Usd.Stage, api_schema_class: Type[Usd.SchemaBase], prim_paths: List[str]) -> None:
    """Apply an API schema to a list of prims.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        api_schema_class (Type[Usd.SchemaBase]): The API schema class to apply.
        prim_paths (List[str]): The paths of the prims to apply the API schema to.
    """
    if not issubclass(api_schema_class, Usd.APISchemaBase):
        raise TypeError(f"{api_schema_class} is not a subclass of Usd.APISchemaBase")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Prim at path {prim_path} does not exist. Skipping.")
            continue
        api_schema = api_schema_class.Apply(prim)
        if not api_schema:
            print(f"Failed to apply {api_schema_class} to prim at path {prim_path}")


def copy_asset_info(source_prim: Usd.Prim, destination_prim: Usd.Prim) -> bool:
    """Copy asset info from one prim to another.

    Args:
        source_prim (Usd.Prim): The source prim to copy asset info from.
        destination_prim (Usd.Prim): The destination prim to copy asset info to.

    Returns:
        bool: True if the asset info was successfully copied, False otherwise.
    """
    if not source_prim.IsValid():
        return False
    if not destination_prim.IsValid():
        return False
    source_asset_info = source_prim.GetAssetInfo()
    for key, value in source_asset_info.items():
        destination_prim.SetAssetInfoByKey(key, value)
    return True


def remove_asset_info_key(obj: Usd.Object, key_path: str) -> bool:
    """Remove an element identified by key path in the object's assetInfo dictionary.

    Args:
        obj (Usd.Object): The object to remove the assetInfo key from.
        key_path (str): The key path identifying the element to remove.

    Returns:
        bool: True if the element was successfully removed, False otherwise.
    """
    if not obj.HasAuthoredAssetInfo():
        return False
    asset_info = obj.GetAssetInfo()
    keys = key_path.split(":")
    current_dict = asset_info
    for key in keys[:-1]:
        if key not in current_dict:
            return False
        current_dict = current_dict[key]
    last_key = keys[-1]
    if last_key in current_dict:
        del current_dict[last_key]
    else:
        return False
    obj.SetAssetInfo(asset_info)
    return True


def find_prims_with_asset_identifier(stage: Usd.Stage, identifier: str) -> List[Usd.Prim]:
    """Find all prims with a specific asset identifier.

    Args:
        stage (Usd.Stage): The stage to search.
        identifier (str): The asset identifier to search for.

    Returns:
        List[Usd.Prim]: A list of prims with the specified asset identifier.
    """
    prims_with_identifier = []
    for prim in stage.TraverseAll():
        if prim.HasAssetInfo():
            if prim.HasAssetInfoKey(Usd.AssetInfoKeys.identifier):
                prim_identifier = prim.GetAssetInfoByKey(Usd.AssetInfoKeys.identifier)
                if prim_identifier == identifier:
                    prims_with_identifier.append(prim)
    return prims_with_identifier


def update_asset_identifier(layer: Sdf.Layer, identifier: str) -> None:
    """Update the asset identifier in the layer metadata.

    Args:
        layer (Sdf.Layer): The layer to update the asset identifier for.
        identifier (str): The new asset identifier value.

    Raises:
        ValueError: If the layer is invalid or not editable.
    """
    if not layer:
        raise ValueError("Invalid layer provided.")
    if not layer.permissionToEdit:
        raise ValueError("Layer is not editable.")
    asset_info = layer.assetInfo if layer.assetInfo else {}
    asset_info["identifier"] = identifier
    with Sdf.ChangeBlock():
        layer.assetInfo = asset_info


def query_attribute_values(
    stage: Usd.Stage, prim_path: str, attribute_name: str
) -> Tuple[bool, List[Tuple[float, float]]]:
    """Query the attribute values for a given prim and attribute name.

    Args:
        stage (Usd.Stage): The USD stage to query.
        prim_path (str): The path to the prim.
        attribute_name (str): The name of the attribute to query.

    Returns:
        Tuple[bool, List[Tuple[float, float]]]: A tuple containing a boolean indicating if the attribute has a value,
        and a list of tuples representing (time, value) pairs for the attribute.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim not found at path: {prim_path}")
    attribute = prim.GetAttribute(attribute_name)
    if not attribute:
        raise ValueError(f"Attribute '{attribute_name}' not found on prim at path: {prim_path}")
    query = Usd.AttributeQuery(attribute)
    has_value = query.HasValue()
    time_samples = attribute.GetTimeSamples()
    values = []
    for time in time_samples:
        value = attribute.Get(time)
        values.append((time, value))
    return (has_value, values)


def analyze_value_variability(attr_query: Usd.AttributeQuery) -> Tuple[bool, bool]:
    """Analyze if an attribute's value varies over time or is uniform.

    Args:
        attr_query (Usd.AttributeQuery): The attribute query to analyze.

    Returns:
        Tuple[bool, bool]: A tuple with two booleans:
            - The first indicates if the value is uniform (single value)
            - The second indicates if the value varies over time
    """
    if not attr_query.IsValid():
        raise ValueError("Invalid attribute query")
    has_authored_value = attr_query.HasAuthoredValue()
    might_be_time_varying = attr_query.ValueMightBeTimeVarying()
    has_fallback = attr_query.HasFallbackValue()
    is_uniform = has_fallback or (has_authored_value and (not might_be_time_varying))
    is_varying = might_be_time_varying or attr_query.GetNumTimeSamples() > 0
    return (is_uniform, is_varying)


def check_attribute_existence(attribute_query: Usd.AttributeQuery) -> bool:
    """Check if the attribute associated with the given AttributeQuery exists.

    Args:
        attribute_query (Usd.AttributeQuery): The AttributeQuery to check.

    Returns:
        bool: True if the attribute exists, False otherwise.
    """
    attribute = attribute_query.GetAttribute()
    if not attribute.IsValid():
        return False
    if not attribute_query.HasValue():
        return False
    return True


def set_attribute_values(
    attr_queries: List[Usd.AttributeQuery], values: List[Any], time_code: Usd.TimeCode = Usd.TimeCode.Default()
):
    """Set attribute values for a list of attribute queries.

    Args:
        attr_queries (List[Usd.AttributeQuery]): List of attribute queries.
        values (List[Any]): List of values to set, matching the attr_queries list.
        time_code (Usd.TimeCode): Time code to set the attribute value at. Defaults to Default time code.

    Raises:
        ValueError: If the lengths of attr_queries and values don't match.
    """
    if len(attr_queries) != len(values):
        raise ValueError(f"Length of attr_queries ({len(attr_queries)}) must match length of values ({len(values)}).")
    for attr_query, value in zip(attr_queries, values):
        if not attr_query.IsValid():
            continue
        attr = attr_query.GetAttribute()
        if not attr.IsValid():
            continue
        attr.Set(value, time_code)


def get_prim_attributes_summary(prim: Usd.Prim) -> Dict[str, Any]:
    """
    Get a summary of attributes for a given prim.

    Args:
        prim (Usd.Prim): The prim to get attribute summary for.

    Returns:
        Dict[str, Any]: A dictionary mapping attribute names to their values.
    """
    summary = {}
    for attr in prim.GetAttributes():
        attr_name = attr.GetName()
        attr_type = attr.GetTypeName()
        if attr_type == Sdf.ValueTypeNames.Bool:
            value = attr.Get(Usd.TimeCode.Default())
        elif attr_type == Sdf.ValueTypeNames.Int:
            value = attr.Get(Usd.TimeCode.Default())
        elif attr_type == Sdf.ValueTypeNames.Float:
            value = attr.Get(Usd.TimeCode.Default())
        elif attr_type == Sdf.ValueTypeNames.Double:
            value = attr.Get(Usd.TimeCode.Default())
        elif attr_type == Sdf.ValueTypeNames.String:
            value = attr.Get(Usd.TimeCode.Default())
        elif attr_type == Sdf.ValueTypeNames.Float2:
            value = attr.Get(Usd.TimeCode.Default())
        elif attr_type == Sdf.ValueTypeNames.Float3:
            value = attr.Get(Usd.TimeCode.Default())
        elif attr_type == Sdf.ValueTypeNames.Float4:
            value = attr.Get(Usd.TimeCode.Default())
        elif attr_type == Sdf.ValueTypeNames.Matrix4d:
            value = attr.Get(Usd.TimeCode.Default())
        else:
            value = None
        summary[attr_name] = value
    return summary


def get_unioned_time_samples(
    attr_queries: List[Usd.AttributeQuery], interval: Optional[Gf.Interval] = None
) -> List[float]:
    """
    Get the union of authored time samples from the given attribute queries.

    Args:
        attr_queries (List[Usd.AttributeQuery]): The list of attribute queries.
        interval (Optional[Gf.Interval]): The optional time interval to restrict the results.
            If None, all time samples are considered. Default is None.

    Returns:
        List[float]: The sorted list of unique time sample values.
    """
    if not attr_queries:
        return []
    time_samples = set()
    for attr_query in attr_queries:
        if not attr_query.IsValid():
            continue
        if interval is None:
            curr_time_samples = attr_query.GetTimeSamples()
        else:
            curr_time_samples = attr_query.GetTimeSamplesInInterval(interval)
        time_samples.update(curr_time_samples)
    return sorted(time_samples)


def is_attribute_value_varying(attribute_query: Usd.AttributeQuery) -> bool:
    """Check if the attribute value may vary over time."""
    if not attribute_query.IsValid():
        raise ValueError("Invalid attribute query.")
    if attribute_query.GetNumTimeSamples() > 0:
        return True
    attr = attribute_query.GetAttribute()
    if attr.HasAuthoredValue():
        value_type = attr.GetTypeName()
        if value_type in [Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.TimeCode]:
            return True
    return False


def copy_attribute_values(source_prim: Usd.Prim, dest_prim: Usd.Prim, skip_attrs: List[str] = None) -> None:
    """Copy attribute values from source_prim to dest_prim.

    Args:
        source_prim (Usd.Prim): The source prim to copy attribute values from.
        dest_prim (Usd.Prim): The destination prim to copy attribute values to.
        skip_attrs (List[str], optional): A list of attribute names to skip copying. Defaults to None.
    """
    if not source_prim.IsValid():
        raise ValueError("Invalid source prim.")
    if not dest_prim.IsValid():
        raise ValueError("Invalid destination prim.")
    attrs_to_copy = []
    for attr in source_prim.GetAttributes():
        if skip_attrs and attr.GetName() in skip_attrs:
            continue
        attrs_to_copy.append(attr)
    for attr in attrs_to_copy:
        attr_name = attr.GetName()
        attr_value = attr.Get()
        if not dest_prim.HasAttribute(attr_name):
            dest_attr = dest_prim.CreateAttribute(attr_name, attr.GetTypeName())
        else:
            dest_attr = dest_prim.GetAttribute(attr_name)
        if attr_value is not None:
            dest_attr.Set(attr_value)


def get_attribute_value_at_time(attribute: Usd.Attribute, time_code: Usd.TimeCode) -> Sdf.ValueBlock:
    """Get the value of an attribute at a specific time code.

    Args:
        attribute (Usd.Attribute): The attribute to get the value from.
        time_code (Usd.TimeCode): The time code to get the value at.

    Returns:
        Sdf.ValueBlock: The value of the attribute at the specified time code.
                        If the attribute has no value, a Sdf.ValueBlock is returned.
    """
    if attribute.HasValue():
        value = attribute.Get(time_code)
        if isinstance(value, Sdf.ValueBlock):
            return value
        else:
            return value
    else:
        return Sdf.ValueBlock()


def get_attribute_values_across_times(attr_query: Usd.AttributeQuery, times: List[Usd.TimeCode]) -> List[object]:
    """
    Get the attribute values at the specified times using an AttributeQuery.

    Args:
        attr_query (Usd.AttributeQuery): The AttributeQuery to use for value resolution.
        times (List[Usd.TimeCode]): The list of times at which to query the attribute values.

    Returns:
        List[object]: The list of resolved attribute values at the specified times.
    """
    values = []
    for time in times:
        value = attr_query.Get(time)
        if value is not None:
            values.append(value)
        else:
            values.append(None)
    return values


def get_all_time_samples(attr_query: Usd.AttributeQuery) -> List[float]:
    """
    Retrieve all authored time samples for the attribute associated with the given AttributeQuery.

    Args:
        attr_query (Usd.AttributeQuery): The AttributeQuery object to retrieve time samples for.

    Returns:
        List[float]: A list of all authored time samples for the attribute.
    """
    if not attr_query.IsValid():
        raise ValueError("Invalid AttributeQuery object provided.")
    num_time_samples = attr_query.GetNumTimeSamples()
    if num_time_samples == 0:
        return []
    result = []
    for i in range(num_time_samples):
        time_sample = attr_query.GetTimeSamples()[i]
        result.append(time_sample)
    return result


def has_fallback_value(self) -> bool:
    """Return true if the attribute associated with this query has a fallback
    value provided by a registered schema.
    """
    attr = self.GetAttribute()
    if not attr.IsValid():
        return False
    fallback_value = attr.Get(Usd.TimeCode.Default())
    if fallback_value is None:
        return False
    attr_def = attr.GetAttributeDefinition()
    if attr_def.HasDefaultValue():
        return True
    return False


def has_authored_value(attr_query: Usd.AttributeQuery) -> bool:
    """Return true if this attribute has either an authored default value or authored time samples."""
    attr = attr_query.GetAttribute()
    if not attr:
        return False
    return attr.HasAuthoredValueOpinion() or attr.GetNumTimeSamples() > 0


def get_bracketing_samples(attr_query: Usd.AttributeQuery, time: float) -> Tuple[float, float, bool]:
    """
    Get the next greater and lesser value relative to the given time.

    Args:
        attr_query (Usd.AttributeQuery): The attribute query object.
        time (float): The time at which to find bracketing samples.

    Returns:
        Tuple[float, float, bool]: A tuple containing the lower and upper bracketing times,
            and a boolean indicating whether there are any time samples.
    """
    has_time_samples = attr_query.GetNumTimeSamples() > 0
    if has_time_samples:
        result = attr_query.GetBracketingTimeSamples(time)
        return (result[0], result[1], has_time_samples)
    else:
        return (time, time, has_time_samples)


def get_attribute_variability(attribute: Usd.Attribute) -> Sdf.Variability:
    """Get the variability of a USD attribute.

    Args:
        attribute (Usd.Attribute): The attribute to query.

    Returns:
        Sdf.Variability: The variability of the attribute.
    """
    if not attribute.IsDefined():
        raise ValueError("Attribute is not defined.")
    variability = attribute.GetVariability()
    return variability


def set_attribute_variability(attribute: Usd.Attribute, variability: Sdf.Variability) -> bool:
    """Set the variability of a USD attribute.

    Args:
        attribute (Usd.Attribute): The attribute to set the variability on.
        variability (Sdf.Variability): The variability to set.

    Returns:
        bool: True if the variability was set successfully, False otherwise.
    """
    if not attribute.IsValid():
        return False
    current_variability = attribute.GetVariability()
    if current_variability == variability:
        return True
    success = attribute.SetVariability(variability)
    return success


def get_authored_attribute_values(attribute: Usd.Attribute) -> Optional[List]:
    """
    Returns a list of all authored values for the given attribute.
    The list includes the default value and all time samples if present.
    Returns None if there are no authored values.
    """
    if not attribute.IsValid():
        raise ValueError("Invalid attribute")
    if not attribute.HasAuthoredValue():
        return None
    default_value = attribute.Get() if attribute.HasAuthoredValue() else None
    time_samples = attribute.GetTimeSamples()
    if not time_samples:
        return [default_value] if default_value is not None else None
    values = [attribute.Get(time) for time in time_samples]
    if default_value is not None:
        values.insert(0, default_value)
    return values


def set_attribute_values_over_time(attribute: Usd.Attribute, values: list[float], times: list[Usd.TimeCode]) -> None:
    """Set attribute values over time.

    Args:
        attribute (Usd.Attribute): The attribute to set values on.
        values (list[float]): The values to set at each time.
        times (list[Usd.TimeCode]): The time codes corresponding to each value.

    Raises:
        ValueError: If the length of values and times lists do not match.
    """
    if not attribute.IsValid():
        raise ValueError("Invalid attribute.")
    if len(values) != len(times):
        raise ValueError("Length of values and times lists do not match.")
    attribute.Clear()
    for value, time in zip(values, times):
        attribute.Set(value, time)


def clear_attribute_color_space(attribute: Usd.Attribute) -> bool:
    """Clear the authored color space value for the given attribute.

    Returns:
        True if the color space was successfully cleared, False otherwise.
    """
    if not attribute:
        return False
    if not attribute.HasColorSpace():
        return True
    return attribute.ClearColorSpace()


def get_prim_attributes_by_type(prim: Usd.Prim, type_name: str) -> list[Usd.Attribute]:
    """Get all attributes of a given type on a prim.

    Args:
        prim (Usd.Prim): The prim to query attributes from.
        type_name (str): The type name of the attributes to find (e.g. "float", "float3", "token").

    Returns:
        list[Usd.Attribute]: A list of attributes with the specified type.
    """
    attr_names_and_types = [(attr.GetName(), attr.GetTypeName()) for attr in prim.GetAttributes()]
    attrs_of_type = [
        prim.GetAttribute(attr_name) for (attr_name, attr_type) in attr_names_and_types if attr_type == type_name
    ]
    return attrs_of_type


def get_attribute_samples_and_values(attribute: Usd.Attribute) -> List[Tuple[float, Sdf.ValueBlock]]:
    """
    Get a list of tuples containing time samples and their corresponding values for a given attribute.

    Args:
        attribute (Usd.Attribute): The attribute to query for time samples and values.

    Returns:
        List[Tuple[float, Sdf.ValueBlock]]: A list of tuples, where each tuple contains a time sample
                                            and its corresponding value. If the attribute has no time samples,
                                            the default value will be returned with a time sample of 0.0.
                                            If the attribute has no default value, an empty list is returned.
    """
    if not attribute.IsValid():
        raise ValueError("Invalid attribute")
    time_samples = attribute.GetTimeSamples()
    if time_samples:
        return [(sample, attribute.Get(sample)) for sample in time_samples]
    if attribute.HasDefault():
        return [(0.0, attribute.Get())]
    return []


def set_attribute_connection_paths(attribute: Usd.Attribute, paths: List[Sdf.Path]) -> bool:
    """Set the connection paths for a UsdAttribute.

    Args:
        attribute (Usd.Attribute): The attribute to set the connection paths on.
        paths (List[Sdf.Path]): The list of connection paths to set.

    Returns:
        bool: True if the connection paths were set successfully, False otherwise.
    """
    if not attribute.IsValid():
        return False
    edit_target = attribute.GetStage().GetEditTarget()
    attribute.ClearConnections()
    for path in paths:
        if not path.IsAbsolutePath():
            continue
        if not attribute.GetStage().GetPrimAtPath(path):
            continue
        attribute.AddConnection(path, Usd.ListPositionBackOfAppendList)
    return True


def remove_attribute_connection(attr: Usd.Attribute, source: Sdf.Path) -> bool:
    """Remove a connection from an attribute.

    Args:
        attr (Usd.Attribute): The attribute to remove the connection from.
        source (Sdf.Path): The source path of the connection to remove.

    Returns:
        bool: True if the connection was successfully removed, False otherwise.
    """
    if not attr.IsValid():
        return False
    connections = attr.GetConnections()
    if source not in connections:
        return False
    success = attr.RemoveConnection(source)
    return success


def get_attribute_color_space(attribute: Usd.Attribute) -> Optional[str]:
    """Get the color space of an attribute if authored.

    Args:
        attribute (Usd.Attribute): The attribute to query.

    Returns:
        Optional[str]: The color space if authored, otherwise None.
    """
    if not attribute.IsValid():
        return None
    if attribute.HasColorSpace():
        color_space = attribute.GetColorSpace()
        return color_space
    else:
        return None


def set_attribute_color_space(attribute: Usd.Attribute, color_space: str) -> None:
    """Set the color space of the given attribute.

    Args:
        attribute (Usd.Attribute): The attribute to set the color space on.
        color_space (str): The color space to set.

    Raises:
        ValueError: If the attribute is invalid or the color space is empty.
    """
    if not attribute or not attribute.IsValid():
        raise ValueError("Invalid attribute.")
    if not color_space:
        raise ValueError("Color space cannot be empty.")
    attribute.SetColorSpace(color_space)
    if attribute.HasColorSpace():
        set_color_space = attribute.GetColorSpace()
        if set_color_space != color_space:
            raise ValueError(f"Failed to set color space. Expected: {color_space}, Got: {set_color_space}")
    else:
        raise ValueError("Color space not set on the attribute.")


def clear_attribute_default_value(attribute: Usd.Attribute) -> bool:
    """Clear the default value of a USD attribute.

    Args:
        attribute (Usd.Attribute): The attribute to clear the default value for.

    Returns:
        bool: True if the default value was successfully cleared, False otherwise.
    """
    if not attribute.IsValid():
        return False
    default_time_code = Usd.TimeCode.Default()
    if not attribute.HasAuthoredValue():
        return True
    success = attribute.ClearAtTime(default_time_code)
    return success


def get_attribute_value_blocking_intervals(attr: Usd.Attribute) -> list[Gf.Interval]:
    """
    Return a list of Gf.Intervals over which the attribute value is blocked.

    An attribute value can be blocked over specific time intervals using
    SdfValueBlock. This function returns a list of those intervals.
    If the attribute value is not blocked at all, an empty list is returned.
    """
    layer = attr.GetStage().GetEditTarget().GetLayer()
    attr_spec = layer.GetAttributeAtPath(attr.GetPath())
    if not attr_spec:
        return []
    time_sample_map = attr_spec.GetInfo("timeSamples")
    if not time_sample_map:
        return []
    blocking_intervals = []
    for time, value in time_sample_map.items():
        if value == Sdf.ValueBlock:
            if not blocking_intervals:
                blocking_intervals.append(Gf.Interval(time, time))
            else:
                blocking_intervals[-1].SetMaxOpen(time)
    return blocking_intervals


def is_attribute_value_time_varying(attr: Usd.Attribute) -> bool:
    """Check if the attribute's value might be varying over time."""
    if not attr.IsValid():
        raise ValueError("Invalid attribute")
    if attr.HasAuthoredValueOpinion() and (not attr.HasAuthoredValue()):
        return False
    variability = attr.GetVariability()
    if variability == Sdf.VariabilityUniform:
        return False
    if attr.GetNumTimeSamples() > 1:
        return True
    if attr.HasAuthoredValue() and attr.GetNumTimeSamples() == 1:
        default_value = attr.Get()
        time_sample_value = attr.Get(attr.GetTimeSamples()[0])
        if default_value != time_sample_value:
            return True
    return False


def assign_attribute_value_block(
    prim: Usd.Prim, attr_name: str, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> bool:
    """Assign a value block to an attribute at a specific time code.

    Args:
        prim (Usd.Prim): The prim to assign the attribute value block to.
        attr_name (str): The name of the attribute to assign the value block to.
        time_code (Usd.TimeCode, optional): The time code to assign the value block at. Defaults to Usd.TimeCode.Default().

    Returns:
        bool: True if the value block was successfully assigned, False otherwise.
    """
    if not prim.IsValid():
        return False
    attribute = prim.GetAttribute(attr_name)
    if not attribute:
        return False
    try:
        attribute.Set(Sdf.ValueBlock(), time_code)
    except:
        return False
    return True


def get_prims_with_attribute(stage: Usd.Stage, attribute_name: str) -> list[Usd.Prim]:
    """
    Get all prims on the stage that have an attribute with the given name.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        attribute_name (str): The name of the attribute to search for.

    Returns:
        list[Usd.Prim]: A list of prims that have the specified attribute.
    """
    prims_with_attr = []
    for prim in stage.TraverseAll():
        if prim.HasAttribute(attribute_name):
            prims_with_attr.append(prim)
    return prims_with_attr


def clear_attribute_value_at_time(attr: Usd.Attribute, time: Usd.TimeCode):
    """Clear the value of an attribute at a specific time.

    If no value is authored at the given time, do nothing.
    If a default value is present, clear the value only at the given time.
    If time is Default, clear the default value.
    """
    if not attr.IsValid():
        raise ValueError("Invalid attribute.")
    if attr.HasAuthoredValueOpinion():
        if time == Usd.TimeCode.Default():
            success = attr.ClearDefault()
        else:
            success = attr.ClearAtTime(time)
        if not success:
            raise RuntimeError("Failed to clear attribute value.")
    else:
        pass


def set_attribute_type_name(attribute: Usd.Attribute, type_name: Sdf.ValueTypeName) -> bool:
    """Set the type name for a USD attribute.

    Args:
        attribute (Usd.Attribute): The attribute to set the type name for.
        type_name (Sdf.ValueTypeName): The type name to set.

    Returns:
        bool: True if the type name was set successfully, False otherwise.
    """
    if not attribute.IsValid():
        return False
    result = attribute.SetTypeName(type_name)
    return result


def clear_all_attribute_time_samples(attribute: Usd.Attribute) -> bool:
    """Clear all time samples for the given attribute.

    Args:
        attribute (Usd.Attribute): The attribute to clear time samples for.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    if not attribute.IsValid():
        return False
    time_samples = attribute.GetTimeSamples()
    for time in time_samples:
        success = attribute.ClearAtTime(time)
        if not success:
            return False
    return True


def set_attribute_values_in_interval(attr: Usd.Attribute, interval: Gf.Interval, value: Any):
    """Sets the attribute values over the specified interval.

    If any timeSamples already exist in the interval, they will be overwritten.
    No timeSamples outside the specified interval will be affected.

    Args:
        attr (Usd.Attribute): The attribute to set values on.
        interval (Gf.Interval): The interval over which to set values.
        value (Any): The value to set across the interval.
    """
    existing_times = attr.GetTimeSamples()
    times_to_replace = [t for t in existing_times if t >= interval.GetMin() and t <= interval.GetMax()]
    for t in times_to_replace:
        attr.Set(value, t)
    if interval.GetMin() not in existing_times:
        attr.Set(value, interval.GetMin())
    if interval.GetMax() not in existing_times:
        attr.Set(value, interval.GetMax())


def set_attribute_value_with_interpolation(
    attribute: Usd.Attribute, value: float, time: Usd.TimeCode, interpolation: str
) -> bool:
    """Set the value of an attribute at a specific time with the given interpolation type.

    Args:
        attribute (Usd.Attribute): The attribute to set the value on.
        value (float): The value to set on the attribute.
        time (Usd.TimeCode): The time at which to set the value.
        interpolation (str): The interpolation type to use. Valid values are "held" and "linear".

    Returns:
        bool: True if the value was set successfully, False otherwise.
    """
    if not attribute.IsValid():
        return False
    if attribute.GetTypeName() != Sdf.ValueTypeNames.Float:
        return False
    if interpolation not in ["held", "linear"]:
        return False
    attribute.SetMetadata("interpolation", interpolation)
    return attribute.Set(value, time)


def add_attribute_connection(
    attribute: Usd.Attribute, source: Sdf.Path, position=Usd.ListPositionBackOfAppendList
) -> bool:
    """Add a connection target to the attribute.

    Args:
        attribute (Usd.Attribute): The attribute to add the connection to.
        source (Sdf.Path): The prim path of the connection source.
        position (Usd.ListPosition, optional): The list position for the new connection. Defaults to Usd.ListPositionBackOfAppendList.

    Returns:
        bool: True if the connection was added successfully, False otherwise.
    """
    if not attribute.IsValid():
        return False
    if not source.IsAbsolutePath():
        return False
    stage = attribute.GetStage()
    if not stage.GetPrimAtPath(source).IsValid():
        return False
    source_prim = stage.GetPrimAtPath(source)
    if source_prim.IsPrototype() or source_prim.IsInPrototype():
        return False
    try:
        attribute.AddConnection(source, position)
    except Tf.ErrorException:
        return False
    return True


def get_attribute_type_name(attribute: Usd.Attribute) -> str:
    """Get the type name of a USD attribute.

    Args:
        attribute (Usd.Attribute): The attribute to get the type name for.

    Returns:
        str: The type name of the attribute.

    Raises:
        ValueError: If the attribute is invalid.
    """
    if not attribute.IsValid():
        raise ValueError("Invalid attribute.")
    type_name = attribute.GetTypeName()
    return type_name.cppTypeName


def has_authored_attribute_connections(attr: UsdShade.Input) -> bool:
    """Return true if the shade input has any authored opinions regarding connections."""
    has_connections = attr.HasConnectedSource()
    return has_connections


def get_time_sampled_values(attr: Usd.Attribute) -> list[tuple[float, object]]:
    """
    Get all time-sampled values from a USD attribute.

    Args:
        attr (Usd.Attribute): The USD attribute to retrieve time samples from.

    Returns:
        list[tuple[float, object]]: A list of tuples containing time sample times and values.
    """
    if not attr.IsValid():
        raise ValueError("Invalid attribute.")
    if not attr.ValueMightBeTimeVarying():
        default_value = attr.Get()
        if default_value is not None:
            return [(0.0, default_value)]
        else:
            return []
    time_samples = attr.GetTimeSamples()
    values = [attr.Get(time) for time in time_samples]
    result = list(zip(time_samples, values))
    return result


def get_attribute_role_name(attr: Usd.Attribute) -> str:
    """Get the role name for an attribute's type name."""
    if not attr.IsValid():
        raise ValueError("Invalid attribute.")
    type_name = attr.GetTypeName()
    role_name = ""
    if type_name:
        role_name = attr.GetRoleName()
    return role_name


def set_attribute_values_and_interpolation(
    attr: Usd.Attribute, values: list, times: list[float], interpolation: str = "linear"
) -> None:
    """Set values and interpolation for a USD attribute.

    Args:
        attr (Usd.Attribute): The USD attribute to set values and interpolation on.
        values (list): List of values to set on the attribute.
        times (list[float]): List of times corresponding to each value.
        interpolation (str): Interpolation type to use. Defaults to "linear".

    Raises:
        ValueError: If the length of values and times lists do not match.
    """
    if not attr.IsValid():
        raise ValueError("Invalid USD Attribute.")
    if len(values) != len(times):
        raise ValueError("Length of values and times lists must match.")
    attr.Clear()
    attr.SetMetadata("interpolation", interpolation)
    for value, time in zip(values, times):
        attr.Set(value, Usd.TimeCode(time))


def get_attribute_fallback_value(attr: Usd.Attribute):
    """Get the fallback value of a USD attribute if it exists.

    Args:
        attr (Usd.Attribute): The USD attribute to get the fallback value from.

    Returns:
        Any: The fallback value of the attribute if it exists, None otherwise.
    """
    if not attr or not attr.IsValid():
        return None
    prim = attr.GetPrim()
    if not prim.IsValid():
        return None
    prim_def = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(prim.GetTypeName())
    if not prim_def:
        prim_def = Usd.SchemaRegistry().FindConcretePrimDefinition(prim.GetTypeName())
    if not prim_def:
        return None
    prop_def = prim_def.GetSchemaAttributeSpec(attr.GetName())
    if prop_def:
        fallback = prop_def.default
        if fallback is not None:
            fallback_type = type(fallback)
            attr_type = attr.GetTypeName()
            if fallback_type != Tf.Type.Find(attr_type).pythonClass:
                fallback = attr.Get(Usd.TimeCode.Default())
        return fallback
    else:
        return None


def get_attribute_connection_paths(attribute: Usd.Attribute) -> List[Sdf.Path]:
    """
    Get the connection paths for a given attribute.

    Args:
        attribute (Usd.Attribute): The attribute to get the connection paths for.

    Returns:
        List[Sdf.Path]: A list of connection paths. If no connections exist, an empty list is returned.
    """
    if not attribute.IsValid():
        raise ValueError("Invalid attribute")
    connection_paths: List[Sdf.Path] = []
    if attribute.HasAuthoredConnections():
        connection_paths = attribute.GetConnections()
    return connection_paths


def get_attribute_samples_in_interval(attr: Usd.Attribute, start_time: float, end_time: float) -> List[float]:
    """
    Get the time samples for an attribute within a given time interval.

    Args:
        attr (Usd.Attribute): The attribute to query for time samples.
        start_time (float): The start time of the interval.
        end_time (float): The end time of the interval.

    Returns:
        List[float]: The time samples within the interval.
    """
    if not attr.IsValid():
        raise ValueError("Invalid attribute.")
    if start_time > end_time:
        raise ValueError("Start time must be less than or equal to end time.")
    interval = Gf.Interval(start_time, end_time)
    time_samples = attr.GetTimeSamplesInInterval(interval)
    return list(time_samples)


def create_and_apply_collection(
    stage: Usd.Stage, prim_path: str, collection_name: str, expansion_rule: str = "expandPrims"
) -> Usd.CollectionAPI:
    """Create a collection on a prim and apply the CollectionAPI schema.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to create the collection on.
        collection_name (str): The name of the collection.
        expansion_rule (str, optional): The expansion rule for the collection.
            Must be one of "explicitOnly", "expandPrims", or "expandPrimsAndProperties".
            Defaults to "expandPrims".

    Returns:
        Usd.CollectionAPI: The applied CollectionAPI schema.

    Raises:
        ValueError: If the prim_path is invalid or the expansion_rule is not a valid value.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Invalid prim path: {prim_path}")
    valid_expansion_rules = ["explicitOnly", "expandPrims", "expandPrimsAndProperties"]
    if expansion_rule not in valid_expansion_rules:
        raise ValueError(f"Invalid expansion rule: {expansion_rule}. Must be one of {valid_expansion_rules}")
    collection_api = Usd.CollectionAPI.Apply(prim, collection_name)
    expansion_rule_attr = collection_api.CreateExpansionRuleAttr()
    expansion_rule_attr.Set(expansion_rule)
    return collection_api


def remove_from_collection(collection_api: Usd.CollectionAPI, path: Sdf.Path) -> bool:
    """Remove the given path from the collection.

    Args:
        collection_api (Usd.CollectionAPI): The collection to remove the path from.
        path (Sdf.Path): The path to remove from the collection.

    Returns:
        bool: True if the path was removed, False if it was not in the collection.
    """
    includes_rel = collection_api.GetIncludesRel()
    if path in includes_rel.GetTargets():
        targets = includes_rel.GetTargets()
        targets.remove(path)
        includes_rel.SetTargets(targets)
        return True
    else:
        return False


def is_path_in_collection(collection_api: Usd.CollectionAPI, path: Sdf.Path) -> bool:
    """Check if a given path is included in a collection.

    Args:
        collection_api (Usd.CollectionAPI): The collection to check.
        path (Sdf.Path): The path to check for inclusion.

    Returns:
        bool: True if the path is included in the collection, False otherwise.
    """
    if not collection_api.GetPrim().IsValid():
        raise ValueError("Invalid collection")
    if not path.IsAbsolutePath():
        raise ValueError("Path must be an absolute path")
    if collection_api.GetIncludeRootAttr().Get() and path == Sdf.Path.absoluteRootPath:
        return True
    included_paths = collection_api.GetIncludesRel().GetTargets()
    excluded_paths = collection_api.GetExcludesRel().GetTargets()
    if path in included_paths:
        return True
    for included_path in included_paths:
        if path.HasPrefix(included_path):
            if path in excluded_paths:
                return False
            for excluded_path in excluded_paths:
                if path.HasPrefix(excluded_path):
                    return False
            return True
    return False


def validate_collection(collection_api: Usd.CollectionAPI) -> bool:
    """Validate a collection by checking its expansion rule, circular dependencies, and top-level rules."""
    valid_expansion_rules = ["explicitOnly", "expandPrims", "expandPrimsAndProperties"]
    expansion_rule = collection_api.GetExpansionRuleAttr().Get()
    if expansion_rule not in valid_expansion_rules:
        return False
    included_collections = collection_api.GetIncludesRel().GetTargets()
    for included_collection in included_collections:
        if collection_api.GetPath().HasPrefix(included_collection):
            return False
    includes = collection_api.GetIncludesRel().GetTargets()
    excludes = collection_api.GetExcludesRel().GetTargets()
    for include in includes:
        for exclude in excludes:
            if Sdf.Path(include).GetPrimPath() == Sdf.Path(exclude).GetPrimPath():
                return False
    return True


def merge_collections(collection1: Usd.CollectionAPI, collection2: Usd.CollectionAPI) -> Usd.CollectionAPI:
    """Merge two collections into a new collection.

    The new collection will contain the union of the included paths and the
    union of the excluded paths from both input collections. If the
    collections have different expansion rules, it will use the broader
    rule (expandPrimsAndProperties > expandPrims > explicitOnly).
    """
    if not collection1.GetPrim().IsValid():
        raise ValueError("Invalid collection1")
    if not collection2.GetPrim().IsValid():
        raise ValueError("Invalid collection2")
    prim = collection1.GetPrim()
    name = f"{collection1.GetName()}_{collection2.GetName()}"
    merged_collection = Usd.CollectionAPI.Apply(prim, name)
    includes1 = collection1.GetIncludesRel().GetTargets()
    includes2 = collection2.GetIncludesRel().GetTargets()
    merged_includes = includes1 + includes2
    merged_collection.CreateIncludesRel().SetTargets(merged_includes)
    excludes1 = collection1.GetExcludesRel().GetTargets()
    excludes2 = collection2.GetExcludesRel().GetTargets()
    merged_excludes = excludes1 + excludes2
    merged_collection.CreateExcludesRel().SetTargets(merged_excludes)
    expansion_rule1 = collection1.GetExpansionRuleAttr().Get()
    expansion_rule2 = collection2.GetExpansionRuleAttr().Get()
    if expansion_rule1 == "expandPrimsAndProperties" or expansion_rule2 == "expandPrimsAndProperties":
        merged_collection.CreateExpansionRuleAttr().Set("expandPrimsAndProperties")
    elif expansion_rule1 == "expandPrims" or expansion_rule2 == "expandPrims":
        merged_collection.CreateExpansionRuleAttr().Set("expandPrims")
    else:
        merged_collection.CreateExpansionRuleAttr().Set("explicitOnly")
    include_root = collection1.GetIncludeRootAttr().Get() or collection2.GetIncludeRootAttr().Get()
    merged_collection.CreateIncludeRootAttr().Set(include_root)
    return merged_collection


def get_collection_expansion_rule(collection_api: Usd.CollectionAPI) -> Usd.Tokens:
    """Get the expansion rule for a collection.

    Args:
        collection_api (Usd.CollectionAPI): The collection API object.

    Returns:
        Usd.Tokens: The expansion rule of the collection.
    """
    if not collection_api.GetPrim().IsValid():
        raise ValueError("Collection API prim is not valid.")
    attr = collection_api.GetExpansionRuleAttr()
    if not attr.IsValid():
        return Usd.Tokens.expandPrims
    rule = attr.Get()
    if rule not in [Usd.Tokens.explicitOnly, Usd.Tokens.expandPrims, Usd.Tokens.expandPrimsAndProperties]:
        raise ValueError(f"Unsupported expansion rule: {rule}")
    return rule


def get_collection_excludes(collection_api: Usd.CollectionAPI) -> List[Sdf.Path]:
    """Get the list of excluded paths in a collection."""
    excludes_rel = collection_api.GetExcludesRel()
    if not excludes_rel:
        return []
    excluded_paths = []
    for target in excludes_rel.GetTargets():
        if target.IsPropertyPath():
            excluded_paths.append(target.GetPrimPath())
        elif target.IsAbsoluteRootOrPrimPath():
            excluded_paths.append(target)
    return excluded_paths


def set_collection_includes(collection_api: Usd.CollectionAPI, includes: List[str]) -> None:
    """Set the includes relationship targets for a collection.

    Args:
        collection_api (Usd.CollectionAPI): The collection API object.
        includes (List[str]): A list of paths to set as includes targets.
    """
    includes_rel = collection_api.GetIncludesRel()
    if not includes_rel:
        includes_rel = collection_api.CreateIncludesRel()
    targets = includes_rel.GetTargets()
    for target in targets:
        includes_rel.RemoveTarget(target)
    for include in includes:
        if not Sdf.Path(include).IsAbsolutePath():
            raise ValueError(f"Include path '{include}' is not an absolute path.")
        includes_rel.AddTarget(include)


def update_collection(collection: Usd.CollectionAPI, include_paths: List[str], exclude_paths: List[str]) -> bool:
    """Update the collection by setting new include and exclude paths."""
    for path in include_paths + exclude_paths:
        if not Sdf.Path(path).IsPrimPath() and (not Sdf.Path(path).IsPropertyPath()):
            raise ValueError(f"Path {path} must be a prim or property path.")
    collection.GetIncludesRel().ClearTargets(True)
    collection.GetExcludesRel().ClearTargets(True)
    for path in include_paths:
        collection.IncludePath(path)
    for path in exclude_paths:
        collection.ExcludePath(path)
    (isValid, reason) = collection.Validate()
    if not isValid:
        raise ValueError(f"Updated collection is not valid: {reason}")
    return True


def add_to_collection(collection_api: Usd.CollectionAPI, prim_path: str) -> bool:
    """Add a prim to a collection."""
    if not collection_api.GetPrim().IsValid():
        raise ValueError("Invalid collection API.")
    prim = collection_api.GetPrim().GetStage().GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    if collection_api.HasNoIncludedPaths() or collection_api.GetIncludesRel().GetTargets():
        includes = collection_api.GetIncludesRel().GetTargets()
        if prim_path in [str(target) for target in includes]:
            return True
    try:
        collection_api.IncludePath(prim_path)
        return True
    except Exception as e:
        print(f"Error adding prim to collection: {str(e)}")
        return False


def get_collection_paths(stage: Usd.Stage) -> List[Sdf.Path]:
    """Get all collection paths in the stage."""
    collection_paths = []
    for prim in stage.Traverse():
        applied_schemas = prim.GetAppliedSchemas()
        for schema in applied_schemas:
            if schema.startswith("CollectionAPI:"):
                collection_name = schema.split(":")[1]
                collection_path = prim.GetPath().AppendProperty("collection:" + collection_name)
                collection_paths.append(collection_path)
    return collection_paths


def get_collection_include_root(collection_api: Usd.CollectionAPI) -> bool:
    """Get the value of the includeRoot attribute for a collection.

    Args:
        collection_api (Usd.CollectionAPI): The collection to query.

    Returns:
        bool: The value of the includeRoot attribute, or False if not set.
    """
    include_root_attr = collection_api.GetIncludeRootAttr()
    if include_root_attr.HasAuthoredValue():
        include_root = include_root_attr.Get()
        return include_root
    else:
        return False


def list_collections(prim: Usd.Prim) -> List[Tuple[str, Usd.CollectionAPI]]:
    """Return a list of tuples containing collection name and CollectionAPI object for each collection on the prim."""
    collections = []
    collection_schemas = Usd.CollectionAPI.GetAll(prim)
    for collection in collection_schemas:
        collection_name = collection.GetName()
        collections.append((collection_name, collection))
    return collections


def get_collection_name(collection_api: Usd.CollectionAPI) -> str:
    """Get the name of the collection from a CollectionAPI object."""
    if not collection_api or not collection_api.GetPrim().IsValid():
        raise ValueError("Invalid CollectionAPI object.")
    name = collection_api.GetName()
    if not name:
        raise ValueError("Collection name is empty.")
    return name


class UsdCollectionAPI:

    def __init__(self, prim, name):
        self.prim = prim
        self.name = name

    @staticmethod
    def Apply(prim, name):
        collection = UsdCollectionAPI(prim, name)
        collection._create_includes_rel()
        collection._create_excludes_rel()
        return collection

    def GetPrim(self):
        return self.prim

    def GetExpansionRuleAttr(self):
        return self.prim.CreateAttribute("collection:expansionRule", Sdf.ValueTypeNames.Token)

    def CreateExpansionRuleAttr(self, rule):
        attr = self.GetExpansionRuleAttr()
        attr.Set(rule)

    def GetIncludeRootAttr(self):
        return self.prim.CreateAttribute("collection:includeRoot", Sdf.ValueTypeNames.Bool)

    def CreateIncludeRootAttr(self, include_root):
        attr = self.GetIncludeRootAttr()
        attr.Set(include_root)

    def GetIncludesRel(self):
        return self.prim.GetRelationship("collection:includes")

    def GetExcludesRel(self):
        return self.prim.GetRelationship("collection:excludes")

    def _create_includes_rel(self):
        self.prim.CreateRelationship("collection:includes")

    def _create_excludes_rel(self):
        self.prim.CreateRelationship("collection:excludes")


def copy_collection(source_collection: UsdCollectionAPI, dest_collection: UsdCollectionAPI) -> bool:
    """Copy the contents of one collection to another."""
    if not source_collection.GetPrim().IsValid():
        raise ValueError("Source collection is invalid.")
    if not dest_collection.GetPrim().IsValid():
        raise ValueError("Destination collection is invalid.")
    src_expansion_rule = source_collection.GetExpansionRuleAttr().Get()
    dest_collection.CreateExpansionRuleAttr(src_expansion_rule)
    src_include_root = source_collection.GetIncludeRootAttr().Get()
    dest_collection.CreateIncludeRootAttr(src_include_root)
    src_includes = source_collection.GetIncludesRel().GetTargets()
    src_excludes = source_collection.GetExcludesRel().GetTargets()
    dest_collection.GetIncludesRel().ClearTargets(True)
    dest_collection.GetExcludesRel().ClearTargets(True)
    dest_includes = dest_collection.GetIncludesRel()
    for target in src_includes:
        dest_includes.AddTarget(target)
    dest_excludes = dest_collection.GetExcludesRel()
    for target in src_excludes:
        dest_excludes.AddTarget(target)
    return True


def include_root_in_collection(collection_api: Usd.CollectionAPI) -> bool:
    """Include the pseudo-root path </> in the given collection.

    Args:
        collection_api (Usd.CollectionAPI): The collection to modify.

    Returns:
        bool: True if the operation succeeded, False otherwise.
    """
    if not collection_api.GetPrim().IsValid():
        return False
    include_root_attr = collection_api.GetIncludeRootAttr()
    if not include_root_attr.IsValid():
        include_root_attr = collection_api.CreateIncludeRootAttr(True)
    else:
        include_root_attr.Set(True)
    collection_api.GetIncludesRel().ClearTargets(True)
    collection_api.GetExcludesRel().ClearTargets(True)
    expansion_rule_attr = collection_api.GetExpansionRuleAttr()
    if not expansion_rule_attr.IsValid():
        expansion_rule_attr = collection_api.CreateExpansionRuleAttr("explicitOnly")
    else:
        expansion_rule_attr.Set("explicitOnly")
    return True


def get_all_collections(prim: Usd.Prim) -> List[Usd.CollectionAPI]:
    """
    Get all collection APIs applied to the given prim.

    Args:
        prim (Usd.Prim): The prim to retrieve collections from.

    Returns:
        List[Usd.CollectionAPI]: A list of CollectionAPI objects.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    applied_schemas = prim.GetAppliedSchemas()
    collections = []
    for schema in applied_schemas:
        if Usd.CollectionAPI.IsSchemaPropertyBaseName(schema):
            collection_name = schema.split(":")[1]
            collection = Usd.CollectionAPI(prim, collection_name)
            collections.append(collection)
    return collections


def set_collection_excludes(collection_api: Usd.CollectionAPI, exclude_paths: List[Sdf.Path]) -> None:
    """Set the excludes list for a collection.

    Args:
        collection_api (Usd.CollectionAPI): The collection API object.
        exclude_paths (List[Sdf.Path]): A list of paths to exclude from the collection.
    """
    excludes_rel = collection_api.GetExcludesRel()
    excludes_rel.ClearTargets(removeSpec=True)
    for path in exclude_paths:
        if not path.IsAbsolutePath():
            raise ValueError(f"Path {path} is not an absolute path.")
        if not collection_api.GetPrim().GetStage().GetPrimAtPath(path):
            raise ValueError(f"Prim at path {path} does not exist.")
        excludes_rel.AddTarget(path)


def find_collections_containing_path(stage: Usd.Stage, path: Sdf.Path) -> List[Usd.CollectionAPI]:
    """
    Find all collections that contain the given path.

    Args:
        stage (Usd.Stage): The USD stage to search for collections.
        path (Sdf.Path): The path to check for inclusion in collections.

    Returns:
        List[Usd.CollectionAPI]: A list of CollectionAPI objects representing the collections that contain the path.
    """
    matching_collections = []
    for prim in stage.TraverseAll():
        if not prim.IsValid():
            continue
        collection_apis = Usd.CollectionAPI.GetAll(prim)
        for collection_api in collection_apis:
            query = collection_api.ComputeMembershipQuery()
            if query.IsPathIncluded(path):
                matching_collections.append(collection_api)
    return matching_collections


def list_collection_names(prim: Usd.Prim) -> List[str]:
    """Return a list of collection names applied to the given prim."""
    collection_names = []
    for prop in prim.GetPropertiesInNamespace("collection:"):
        collection_name = prop.GetName().replace("collection:", "")
        collection_names.append(collection_name)
    return collection_names


def apply_collection_to_prims(stage: Usd.Stage, collection_path: str, prim_paths: List[str]) -> None:
    """Apply a collection to a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        collection_path (str): The path to the collection prim.
        prim_paths (List[str]): The paths to the prims to add to the collection.

    Raises:
        ValueError: If the collection prim does not exist or is not a valid collection.
    """
    collection_prim = stage.GetPrimAtPath(collection_path)
    if not collection_prim.IsValid():
        raise ValueError(f"Collection prim at path {collection_path} does not exist.")
    includes_rel_name = "collection:includes"
    if not collection_prim.HasRelationship(includes_rel_name):
        raise ValueError(f"Prim at path {collection_path} is not a valid collection.")
    includes_rel = collection_prim.GetRelationship(includes_rel_name)
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        includes_rel.AddTarget(Sdf.Path(prim_path))


def can_apply_collection(prim: Usd.Prim, name: str, whyNot: str) -> bool:
    """Returns true if the CollectionAPI schema can be applied to the given prim with the given instance name.

    If the schema cannot be applied, returns false and populates whyNot with the reason.
    """
    if not prim.IsValid():
        whyNot = "Prim is not valid"
        return False
    if not name:
        whyNot = "Instance name is empty"
        return False
    if prim.IsInstanceProxy():
        whyNot = "Prim is an instance proxy"
        return False
    if prim.HasAPI(Usd.CollectionAPI, name):
        whyNot = f"CollectionAPI with instance name '{name}' is already applied"
        return False
    return True


def can_include_in_collection(collection_api: Usd.CollectionAPI, prim_path: Sdf.Path) -> bool:
    """Check if a given prim path can be included in a collection.

    Args:
        collection_api (Usd.CollectionAPI): The collection to check against.
        prim_path (Sdf.Path): The prim path to check for inclusion.

    Returns:
        bool: True if the prim path can be included in the collection, False otherwise.
    """
    if not collection_api.GetPrim().IsValid():
        raise ValueError("The provided collection is not valid.")
    if not prim_path.IsAbsolutePath() or not prim_path.IsPrimPath():
        raise ValueError("The provided prim path is not a valid absolute prim path.")
    if prim_path in collection_api.GetExcludesRel().GetTargets():
        return False
    expansion_rule = collection_api.GetExpansionRuleAttr().Get()
    if expansion_rule == Usd.Tokens.explicitOnly:
        return prim_path in collection_api.GetIncludesRel().GetTargets()
    elif expansion_rule == Usd.Tokens.expandPrims:
        current_path = prim_path
        while current_path != Sdf.Path.absoluteRootPath:
            if current_path in collection_api.GetIncludesRel().GetTargets():
                return True
            current_path = current_path.GetParentPath()
    elif expansion_rule == Usd.Tokens.expandPrimsAndProperties:
        current_path = prim_path
        while current_path != Sdf.Path.absoluteRootPath:
            if current_path in collection_api.GetIncludesRel().GetTargets():
                return True
            current_path = current_path.GetParentPath()
    return False


def get_collection_includes(collection_api: Usd.CollectionAPI) -> Sdf.PathListOp:
    """Get the includes list operation for a collection."""
    includes_rel = collection_api.GetIncludesRel()
    if not includes_rel.GetTargets():
        return Sdf.PathListOp()
    includes_targets = includes_rel.GetTargets()
    includes_listop = Sdf.PathListOp()
    includes_listop.explicitItems = includes_targets
    return includes_listop


def exclude_from_collection(collection_api: Usd.CollectionAPI, exclude_path: Sdf.Path) -> bool:
    """Exclude a path from a collection.

    Args:
        collection_api (Usd.CollectionAPI): The collection to exclude from.
        exclude_path (Sdf.Path): The path to exclude from the collection.

    Returns:
        bool: True if the path was successfully excluded, False otherwise.
    """
    if not collection_api.GetPrim().IsValid():
        return False
    excludes_rel = collection_api.GetExcludesRel()
    if excludes_rel.GetTargets().count(exclude_path) > 0:
        return True
    includes_rel = collection_api.GetIncludesRel()
    if includes_rel.GetTargets().count(exclude_path) == 0:
        excludes_rel.AddTarget(exclude_path)
        return True
    includes_rel.RemoveTarget(exclude_path)
    excludes_rel.AddTarget(exclude_path)
    return True


def remove_collection(prim: Usd.Prim, collection_name: str) -> bool:
    """Remove a collection from the given prim.

    Args:
        prim (Usd.Prim): The prim to remove the collection from.
        collection_name (str): The name of the collection to remove.

    Returns:
        bool: True if the collection was removed, False otherwise.
    """
    if not prim.IsValid():
        return False
    if prim.HasAPI(Usd.CollectionAPI, collection_name):
        collection_api = Usd.CollectionAPI(prim, collection_name)
        collection_api.GetIncludesRel().ClearTargets(True)
        collection_api.GetExcludesRel().ClearTargets(True)
        prim.RemoveAPI(Usd.CollectionAPI, collection_name)
        return True
    return False


def compute_included_paths(
    collection: Usd.CollectionAPI, stage: Usd.Stage, pred: Usd.PrimIsActive = Usd.PrimIsActive
) -> set:
    """Compute the paths included in a collection based on the membership query.

    Args:
        collection (Usd.CollectionAPI): The collection to compute included paths for.
        stage (Usd.Stage): The USD stage.
        pred (Usd.PrimIsActive, optional): Optional predicate for filtering prims.
            Defaults to Usd.PrimIsActive which includes all active prims.

    Returns:
        set[Sdf.Path]: The set of paths included in the collection.
    """
    if not collection:
        raise ValueError("Invalid CollectionAPI object")
    query = collection.ComputeMembershipQuery()
    included_paths = set(query.ComputeIncludedPaths(stage, pred))
    if query.HasIncludeRoot() and query.GetIncludeRoot():
        included_paths.add(Sdf.Path.absoluteRootPath())
    expansion_rule = query.GetExpansionRule()
    if expansion_rule == Usd.Tokens.expandPrims:
        included_paths = set(Usd.CollectionAPI.ComputeIncludedPaths(query, stage, Usd.TraverseInstanceProxies(pred)))
    elif expansion_rule == Usd.Tokens.expandPrimsAndProperties:
        included_paths = set(
            Usd.CollectionAPI.ComputeIncludedPaths(
                query, stage, Usd.TraverseInstanceProxies(pred & Usd.PrimIsLoaded & ~Usd.PrimIsAbstract)
            )
        )
    return included_paths


def compute_included_objects(query, stage: Usd.Stage, pred: Usd._PrimFlagsPredicate) -> set[Usd.Object]:
    """Returns all the usd objects that satisfy the predicate in the collection represented by the query."""
    included_objects: set[Usd.Object] = set()
    included_paths = query.GetAsPathExpansionRuleMap()
    for path, expansionRule in included_paths.items():
        if path.IsPrimPath():
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid() and pred(prim):
                included_objects.add(prim)
            if expansionRule == Usd.Tokens.expandPrims or expansionRule == Usd.Tokens.expandPrimsAndProperties:
                for child in prim.GetAllChildren():
                    if pred(child):
                        included_objects.add(child)
        elif path.IsPropertyPath():
            prop = stage.GetPropertyAtPath(path)
            if prop.IsValid() and pred(prop.GetPrim()):
                included_objects.add(prop)
        else:
            continue
    return included_objects


def always_true_pred(prim):
    return True


def get_arc_target_prim_paths(node: Pcp.NodeRef) -> List[str]:
    """Get the target prim paths from a composition node."""
    target_prim_paths = []
    for child_node in node.children:
        arc = child_node.arcType
        if arc == Pcp.ArcTypeReference or arc == Pcp.ArcTypePayload:
            target_prim_path = child_node.pathAtIntroduction
            if target_prim_path:
                target_prim_paths.append(str(target_prim_path))
            else:
                continue
    return target_prim_paths


def check_implicit_arcs(node: Pcp.NodeRef) -> bool:
    """Check if any of the composition arcs in the node are implicit.

    Args:
        node (Pcp.NodeRef): A node in the PcpPrimIndex to check for implicit arcs.

    Returns:
        bool: True if any of the arcs are implicit, False otherwise.
    """
    for child_node in node.children:
        arc = child_node.arcToParent
        if arc.IsImplicit():
            return True
    return False


def get_arc_introducing_layers(arc: Usd.CompositionArc) -> List[Sdf.Layer]:
    """Return the layers that introduce the given composition arc."""
    if not arc:
        raise ValueError("Invalid composition arc.")
    layer_stack = arc.GetIntroducingNode().layerStack
    start_layer = arc.GetIntroducingLayer()
    end_layer = layer_stack.identifier.rootLayer
    introducing_layers = []
    for layer in layer_stack.layers:
        introducing_layers.append(layer)
        if layer == end_layer:
            break
    return introducing_layers


def get_target_layers_of_arcs(prim: Usd.Prim) -> List[Sdf.Layer]:
    """Get the target layers for a prim's composition arcs.

    Args:
        prim (Usd.Prim): The prim to get the composition arcs from.

    Returns:
        List[Sdf.Layer]: A list of target layers for the prim's composition arcs.
    """
    target_layers: List[Sdf.Layer] = []
    stage = prim.GetStage()
    root_layer = stage.GetRootLayer()
    layer_stack = root_layer.subLayerPaths
    for layer_path in layer_stack:
        layer = Sdf.Layer.FindOrOpen(layer_path)
        if layer and layer.permissionToEdit:
            target_layers.append(layer)
    return target_layers


def modify_arc_targets(arc: Sdf.PrimSpec, new_targets: List[Sdf.Path]) -> bool:
    """Modify the targets of a composition arc.

    Args:
        arc (Sdf.PrimSpec): The composition arc to modify.
        new_targets (List[Sdf.Path]): The new target paths for the arc.

    Returns:
        bool: True if the targets were successfully modified, False otherwise.
    """
    if not arc or not arc.layer:
        return False
    current_targets = (
        arc.referenceList.GetAddedOrExplicitItems()
        if arc.typeName == "reference"
        else arc.inheritPathList.GetAddedOrExplicitItems()
    )
    current_target_paths = [target for target in current_targets]
    if current_target_paths == new_targets:
        return True
    if arc.typeName == "reference":
        arc.referenceList.ClearEditsAndMakeExplicit()
        for target in new_targets:
            arc.referenceList.Add(Sdf.Reference(target))
    else:
        arc.inheritPathList.ClearEditsAndMakeExplicit()
        for target in new_targets:
            arc.inheritPathList.Add(target)
    updated_targets = (
        arc.referenceList.GetAddedOrExplicitItems()
        if arc.typeName == "reference"
        else arc.inheritPathList.GetAddedOrExplicitItems()
    )
    updated_target_paths = [target for target in updated_targets]
    if updated_target_paths != new_targets:
        return False
    return True


def compare_node_resolutions(node1: Pcp.NodeRef, node2: Pcp.NodeRef) -> int:
    """Compare the strength of two composition nodes.

    Returns:
        int: -1 if node1 is weaker than node2, 0 if they are equal, 1 if node1 is stronger than node2.
    """
    if not node1 or not node2:
        raise ValueError("One or both nodes are invalid.")
    arc1_type = node1.GetArcType()
    arc2_type = node2.GetArcType()
    if arc1_type != arc2_type:
        raise ValueError("Cannot compare composition nodes with different arc types.")
    arc1_target_path = node1.GetTargetNode().GetPath()
    arc2_target_path = node2.GetTargetNode().GetPath()
    if arc1_target_path != arc2_target_path:
        raise ValueError("Cannot compare composition nodes with different target prim paths.")
    if node1.HasStrongerThan([node2]):
        return 1
    elif node2.HasStrongerThan([node1]):
        return -1
    else:
        return 0


def compare_crate_files(file_paths: List[str]) -> List[Tuple[str, str, str, str]]:
    """Compare the file version, software version, and summary stats of multiple crate files.

    Args:
        file_paths (List[str]): List of file paths to the crate files to compare.

    Returns:
        List[Tuple[str, str, str, str]]: List of tuples containing the file path, file version,
                                         software version, and summary stats for each file.
    """
    results = []
    for file_path in file_paths:
        if not Sdf.Layer.FindOrOpen(file_path):
            results.append((file_path, "N/A", "N/A", "File not found or cannot be opened"))
            continue
        try:
            crate_info = Usd.CrateInfo.Open(file_path)
            file_version = crate_info.GetFileVersion()
            software_version = crate_info.GetSoftwareVersion()
            summary_stats = crate_info.GetSummaryStats()
            results.append((file_path, file_version, software_version, str(summary_stats)))
        except RuntimeError as e:
            results.append((file_path, "N/A", "N/A", f"Error: {str(e)}"))
    return results


def extract_section_from_crate(crate_file: str, section_name: str) -> bytes:
    """Extract a named section from a USD crate file.

    Args:
        crate_file (str): The path to the USD crate file.
        section_name (str): The name of the section to extract.

    Returns:
        bytes: The raw bytes of the extracted section.

    Raises:
        ValueError: If the crate file or section is not found.
    """
    if not os.path.exists(crate_file):
        raise ValueError(f"Crate file not found: {crate_file}")
    crate_info = Usd.CrateInfo.Open(crate_file)
    if not crate_info:
        raise ValueError(f"Failed to open crate file: {crate_file}")
    sections = crate_info.GetSections()
    target_section = None
    for section in sections:
        if section.name == section_name:
            target_section = section
            break
    if target_section is None:
        raise ValueError(f"Section '{section_name}' not found in crate file.")
    with open(crate_file, "rb") as file:
        file.seek(target_section.start)
        section_data = file.read(target_section.size)
    return section_data


def list_large_sections(crate_info: Usd.CrateInfo, min_size: int) -> List[Usd.CrateInfo.Section]:
    """Return a list of sections in the crate file that are larger than min_size.

    Args:
        crate_info (Usd.CrateInfo): The CrateInfo object to query sections from.
        min_size (int): The minimum size in bytes for a section to be considered large.

    Returns:
        List[Usd.CrateInfo.Section]: A list of sections larger than min_size.
    """
    large_sections = []
    for section in crate_info.GetSections():
        if section.size > min_size:
            large_sections.append(section)
    return large_sections


def merge_scenes(source_layer: Sdf.Layer, target_layer: Sdf.Layer) -> bool:
    """Merge the contents of source_layer into target_layer."""
    try:
        if not source_layer.IsValid() or not target_layer.IsValid():
            raise ValueError("Invalid layer(s) provided.")
        target_stage = Usd.Stage.Open(target_layer)
        edit_target = target_stage.GetRootLayer()
        edit_target.TransferContent(source_layer)
        target_stage.Save()
        return True
    except Exception as e:
        print(f"Error merging scenes: {str(e)}")
        return False


class SummaryStats:

    def __init__(self):
        self.numSpecs = 0
        self.numUniqueFieldSets = 0
        self.numUniqueFields = 0
        self.numUniquePaths = 0
        self.numUniqueStrings = 0
        self.numUniqueTokens = 0


def compare_scene_complexity(
    stats1: SummaryStats, stats2: SummaryStats
) -> Tuple[float, float, float, float, float, float]:
    """Compare the complexity of two USD scenes using their SummaryStats.

    Returns a tuple with the percentage difference for each stat, where a positive value means
    the first scene has a higher value for that stat, and a negative value means the second
    scene has a higher value.
    """
    if not isinstance(stats1, SummaryStats) or not isinstance(stats2, SummaryStats):
        raise TypeError("Both arguments must be of type SummaryStats")
    pct_specs = (stats1.numSpecs - stats2.numSpecs) / max(stats1.numSpecs, stats2.numSpecs, 1) * 100
    pct_fieldsets = (
        (stats1.numUniqueFieldSets - stats2.numUniqueFieldSets)
        / max(stats1.numUniqueFieldSets, stats2.numUniqueFieldSets, 1)
        * 100
    )
    pct_fields = (
        (stats1.numUniqueFields - stats2.numUniqueFields) / max(stats1.numUniqueFields, stats2.numUniqueFields, 1) * 100
    )
    pct_paths = (
        (stats1.numUniquePaths - stats2.numUniquePaths) / max(stats1.numUniquePaths, stats2.numUniquePaths, 1) * 100
    )
    pct_strings = (
        (stats1.numUniqueStrings - stats2.numUniqueStrings)
        / max(stats1.numUniqueStrings, stats2.numUniqueStrings, 1)
        * 100
    )
    pct_tokens = (
        (stats1.numUniqueTokens - stats2.numUniqueTokens) / max(stats1.numUniqueTokens, stats2.numUniqueTokens, 1) * 100
    )
    return (pct_specs, pct_fieldsets, pct_fields, pct_paths, pct_strings, pct_tokens)


def filter_prims_by_complexity(stage: Usd.Stage, max_complexity: int) -> List[Usd.Prim]:
    """Filter prims in a stage by their complexity.

    Args:
        stage (Usd.Stage): The USD stage to filter prims from.
        max_complexity (int): The maximum complexity allowed for a prim to be included.

    Returns:
        List[Usd.Prim]: A list of prims that meet the complexity criteria.
    """
    filtered_prims = []
    for prim in stage.Traverse():
        complexity = (
            len(prim.GetAttributes())
            + len(prim.GetRelationships())
            + len(prim.GetChildren())
            + len(prim.GetAllMetadata())
        )
        if complexity <= max_complexity:
            filtered_prims.append(prim)
    return filtered_prims


def override_materials(stage: Usd.Stage, prim_path: str, material_path: str, edit_context: Usd.EditContext):
    """Override the materials of a prim and its descendants within an EditContext.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to override materials for.
        material_path (str): The path to the material to assign.
        edit_context (Usd.EditContext): The EditContext to perform the overrides within.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise ValueError(f"Material prim at path {material_path} does not exist.")
    material = UsdShade.Material(material_prim)
    if not material:
        raise ValueError(f"Prim at path {material_path} is not a valid UsdShadeMaterial.")
    for descendant in Usd.PrimRange(prim):
        gprim = UsdGeom.Gprim(descendant)
        if gprim:
            with edit_context:
                UsdShade.MaterialBindingAPI(gprim).Bind(material)


def apply_transformations(
    stage: Usd.Stage, prim_path: str, translate: Gf.Vec3d = None, rotate: Gf.Vec3f = None, scale: Gf.Vec3f = None
):
    """Apply transformations to a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to transform.
        translate (Gf.Vec3d, optional): Translation vector. Defaults to None.
        rotate (Gf.Vec3f, optional): Rotation vector in degrees. Defaults to None.
        scale (Gf.Vec3f, optional): Scaling vector. Defaults to None.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim not found at path: {prim_path}")
    xform_api = UsdGeom.XformCommonAPI(prim)
    if translate is not None:
        xform_api.SetTranslate(translate)
    if rotate is not None:
        rotation_order = UsdGeom.XformCommonAPI.RotationOrderXYZ
        xform_api.SetRotate(rotate, rotation_order)
    if scale is not None:
        xform_api.SetScale(scale)


def batch_create_prims(stage: Usd.Stage, prim_paths: List[str], prim_type: Optional[str] = None) -> List[Usd.Prim]:
    """
    Create multiple prims at the given paths on the specified stage.

    Args:
        stage (Usd.Stage): The stage to create the prims on.
        prim_paths (List[str]): A list of prim paths to create.
        prim_type (str, optional): The type of the prims to create. If not provided, uses "Xform" by default.

    Returns:
        List[Usd.Prim]: The list of created prims.
    """
    if prim_type is None:
        prim_type = "Xform"
    created_prims = []
    for prim_path in prim_paths:
        existing_prim = stage.GetPrimAtPath(prim_path)
        if existing_prim.IsValid():
            created_prims.append(existing_prim)
        else:
            new_prim = stage.DefinePrim(prim_path, prim_type)
            created_prims.append(new_prim)
    return created_prims


def change_interpolation_type(attribute: Usd.Attribute, interpolation_type: Sdf.ValueTypeName) -> None:
    """Change the interpolation type of an attribute.

    Args:
        attribute (Usd.Attribute): The attribute to modify.
        interpolation_type (Sdf.ValueTypeName): The new interpolation type.

    Raises:
        ValueError: If the attribute is invalid.
        TypeError: If the interpolation type is not a valid Sdf.ValueTypeName.
    """
    if not attribute.IsValid():
        raise ValueError("Invalid attribute.")
    if not isinstance(interpolation_type, Sdf.ValueTypeName):
        raise TypeError("Invalid interpolation type. Must be a Sdf.ValueTypeName.")
    attribute.SetTypeName(interpolation_type)


def get_prim_and_direct_inherits(prim: Usd.Prim) -> Tuple[Usd.Prim, list[Sdf.Path]]:
    """Get the prim and its direct inherits.

    Args:
        prim (Usd.Prim): The prim to get the direct inherits for.

    Returns:
        Tuple[Usd.Prim, list[Sdf.Path]]: A tuple containing the prim and a list of its direct inherits.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    inherits = prim.GetInherits()
    direct_inherits = inherits.GetAllDirectInherits()
    return (prim, direct_inherits)


def replace_inherit_path(prim: Usd.Prim, old_path: Sdf.Path, new_path: Sdf.Path) -> bool:
    """Replace an inherited path on the given prim.

    Args:
        prim (Usd.Prim): The prim to modify the inherits on.
        old_path (Sdf.Path): The inherit path to replace.
        new_path (Sdf.Path): The new inherit path.

    Returns:
        bool: True if the inherit path was replaced, False otherwise.
    """
    inherits = prim.GetInherits()
    if old_path not in inherits.GetAllDirectInherits():
        return False
    if not inherits.RemoveInherit(old_path):
        return False
    if not inherits.AddInherit(new_path):
        return False
    return True


def batch_add_inherit_paths(
    prim: Usd.Prim, inherit_paths: List[Sdf.Path], position=Usd.ListPositionFrontOfPrependList
) -> bool:
    """Batch add inherit paths to a prim.

    Args:
        prim (Usd.Prim): The prim to add inherit paths to.
        inherit_paths (List[Sdf.Path]): The list of inherit paths to add.
        position (Usd.ListPosition, optional): The position to add the inherit paths. Defaults to Usd.ListPositionFrontOfPrependList.

    Returns:
        bool: True if the inherit paths were added successfully, False otherwise.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    inherits = prim.GetInherits()
    for inherit_path in inherit_paths:
        if not inherit_path.IsAbsolutePath():
            raise ValueError(f"Inherit path must be an absolute path: {inherit_path}")
        success = inherits.AddInherit(inherit_path, position)
        if not success:
            return False
    return True


def clear_and_set_inherits(prim: Usd.Prim, inherit_paths: list[Sdf.Path]) -> bool:
    """Clear any existing inherits and set the specified inherit paths.

    Args:
        prim (Usd.Prim): The prim to set the inherits on.
        inherit_paths (list[Sdf.Path]): The list of paths to set as inherits.

    Returns:
        bool: True if the operation succeeded, False otherwise.
    """
    inherits = prim.GetInherits()
    if not inherits.ClearInherits():
        return False
    return inherits.SetInherits(inherit_paths)


def remove_inherit_and_reset(prim: Usd.Prim, inherit_path: Sdf.Path) -> bool:
    """Remove an inherit path from a prim and reset the inherited prim.

    Args:
        prim (Usd.Prim): The prim to remove the inherit path from.
        inherit_path (Sdf.Path): The inherit path to remove.

    Returns:
        bool: True if the inherit path was successfully removed and the prim was reset, False otherwise.
    """
    inherits = prim.GetInherits()
    if not inherits.RemoveInherit(inherit_path):
        return False
    inherited_prim = prim.GetStage().GetPrimAtPath(inherit_path)
    if not inherited_prim.IsValid():
        return False
    for attr in inherited_prim.GetAttributes():
        attr.Clear()
    for rel in inherited_prim.GetRelationships():
        rel.ClearTargets(True)
    return True


def get_all_inherit_paths(prim: Usd.Prim) -> List[Sdf.Path]:
    """
    Return all the paths in the prim's stage's local layer stack that would compose into
    this prim via direct inherits (excluding prim specs that would be composed into this
    prim due to inherits authored on ancestral prims) in strong-to-weak order.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim '{prim.GetPath()}' is not valid.")
    inherits = prim.GetInherits()
    if not inherits:
        return []
    inherit_paths = inherits.GetAllDirectInherits()
    return inherit_paths


def reorder_prim_attributes(prim: Usd.Prim, attribute_names: List[str]) -> bool:
    """Reorder the attributes on a prim.

    Args:
        prim (Usd.Prim): The prim to reorder the attributes on.
        attribute_names (List[str]): The new order of the attribute names.

    Returns:
        bool: True if the reordering was successful, False otherwise.
    """
    if not prim.IsValid():
        return False
    current_attributes = prim.GetAttributes()
    current_names = [attr.GetName() for attr in current_attributes]
    if set(current_names) != set(attribute_names):
        return False
    reordered_attributes = []
    for attr_name in attribute_names:
        for attr in current_attributes:
            if attr.GetName() == attr_name:
                reordered_attributes.append(attr)
                break
    for attr in current_attributes:
        attr.Clear()
    for attr in reordered_attributes:
        prim.CreateAttribute(attr.GetName(), attr.GetTypeName())
    return True


def revert_to_default_edit_target(stage: Usd.Stage) -> None:
    """Revert the stage's EditTarget to the default, root layer of the stage."""
    root_layer = stage.GetRootLayer()
    if not root_layer:
        raise ValueError("Stage has no root layer.")
    default_edit_target = Usd.EditTarget(root_layer)
    stage.SetEditTarget(default_edit_target)


def get_prim_spec_for_edit_target(edit_target: Usd.EditTarget, scene_path: Sdf.Path) -> Sdf.PrimSpec:
    """
    Get the PrimSpec in the edit target's layer for the given scene path.

    Args:
        edit_target (Usd.EditTarget): The edit target to map the scene path to.
        scene_path (Sdf.Path): The scene path to get the PrimSpec for.

    Returns:
        Sdf.PrimSpec: The PrimSpec in the edit target's layer for the given scene path,
                      or None if the edit target is invalid or there is no valid mapping.
    """
    if edit_target.IsNull() or not edit_target.IsValid():
        return None
    spec_path = edit_target.MapToSpecPath(scene_path)
    if not spec_path.IsAbsoluteRootOrPrimPath():
        return None
    layer = edit_target.GetLayer()
    if not layer:
        return None
    prim_spec = layer.GetPrimAtPath(spec_path)
    return prim_spec if prim_spec else None


def get_property_spec_for_edit_target(edit_target: Usd.EditTarget, scene_path: Sdf.Path) -> Sdf.PropertySpec:
    """Get the property spec for a given scene path in an edit target.

    Args:
        edit_target (Usd.EditTarget): The edit target to query.
        scene_path (Sdf.Path): The scene path to the property.

    Returns:
        Sdf.PropertySpec: The property spec for the given scene path in the edit target's layer.

    Raises:
        ValueError: If the edit target is invalid or the scene path is not a property path.
    """
    if not edit_target.IsValid():
        raise ValueError("Invalid edit target.")
    if not scene_path.IsPropertyPath():
        raise ValueError(f"Path {scene_path} is not a property path.")
    spec_path = edit_target.MapToSpecPath(scene_path)
    layer = edit_target.GetLayer()
    property_spec = layer.GetPropertyAtPath(spec_path)
    return property_spec


def validate_edit_target(stage: Usd.Stage, edit_target: Usd.EditTarget) -> bool:
    """Validate if the given edit target is valid for the stage.

    Args:
        stage (Usd.Stage): The USD stage.
        edit_target (Usd.EditTarget): The edit target to validate.

    Returns:
        bool: True if the edit target is valid for the stage, False otherwise.
    """
    if edit_target.IsNull():
        return False
    if not edit_target.GetLayer():
        return False
    if edit_target.GetLayer() not in stage.GetLayerStack():
        return False
    root_prim = stage.GetPseudoRoot()
    try:
        mapped_path = edit_target.MapToSpecPath(root_prim.GetPath())
        if not Sdf.Path.IsValidPathString(mapped_path.pathString):
            return False
    except Sdf.MapperError:
        return False
    return True


def get_layer_of_current_edit_target(stage: Usd.Stage) -> Sdf.Layer:
    """Get the layer of the current edit target for the given stage."""
    edit_target = stage.GetEditTarget()
    if not edit_target.IsValid():
        raise ValueError("The edit target is not valid.")
    layer = edit_target.GetLayer()
    if not layer:
        raise ValueError("The edit target does not have a valid layer.")
    return layer


def set_edit_target_to_variant(
    stage: Usd.Stage, prim_path: str, variant_set_name: str, variant_selection: str
) -> Usd.EditTarget:
    """Set the edit target to a variant of a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        variant_set_name (str): The name of the variant set.
        variant_selection (str): The name of the variant selection.

    Returns:
        Usd.EditTarget: The edit target for the variant.

    Raises:
        ValueError: If the prim or variant set does not exist, or if the variant selection is not valid.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    variant_set = prim.GetVariantSet(variant_set_name)
    if not variant_set:
        raise ValueError(f"Variant set {variant_set_name} does not exist on prim {prim_path}.")
    if not variant_set.HasAuthoredVariant(variant_selection):
        raise ValueError(
            f"Variant selection {variant_selection} does not exist in variant set {variant_set_name} on prim {prim_path}."
        )
    return Usd.EditTarget.ForLocalDirectVariant(
        stage.GetRootLayer(), prim.GetPath().AppendVariantSelection(variant_set_name, variant_selection)
    )


def map_scene_path_to_spec_path(edit_target: Usd.EditTarget, scene_path: Sdf.Path) -> Optional[Sdf.Path]:
    """Map a scene path to a spec path using the given EditTarget.

    Args:
        edit_target (Usd.EditTarget): The EditTarget to use for mapping.
        scene_path (Sdf.Path): The scene path to map.

    Returns:
        Optional[Sdf.Path]: The mapped spec path, or None if the mapping is not valid.
    """
    if not edit_target.IsValid():
        return None
    if scene_path.isEmpty:
        return None
    spec_path = edit_target.MapToSpecPath(scene_path)
    if spec_path.isEmpty:
        return None
    return spec_path


def recursively_load_prims(prim: Usd.Prim, policy=Usd.Stage.LoadAll) -> None:
    """Recursively load the given prim and its descendants based on the load policy.

    Args:
        prim (Usd.Prim): The prim to start loading from.
        policy (Usd.Stage.LoadAll): The load policy to use. Defaults to Usd.Stage.LoadAll.

    Raises:
        ValueError: If the given prim is not valid.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if policy == Usd.Stage.LoadAll:
        prim.Load()
    elif policy == Usd.Stage.LoadNone:
        stage = prim.GetStage()
        stage.Load(prim.GetPath())
    if policy == Usd.Stage.LoadAll:
        for child in prim.GetChildren():
            recursively_load_prims(child, policy)


def validate_and_convert_kind(kind: str) -> Usd.ModelAPI.KindValidation:
    """Validate and convert a string kind value to a KindValidation enum.

    Args:
        kind (str): The string representation of the KindValidation enum.

    Returns:
        Usd.ModelAPI.KindValidation: The corresponding KindValidation enum value.

    Raises:
        ValueError: If the provided kind string is not a valid KindValidation enum name.
    """
    try:
        kind_validation = Usd.ModelAPI.KindValidation.GetValueFromName(kind)
    except RuntimeError as e:
        raise ValueError(
            f"Invalid KindValidation value: {kind}. Valid values are: {', '.join(e.args[0].split())}"
        ) from None
    return kind_validation


def get_layers_muting_status(stage: Usd.Stage) -> Dict[str, bool]:
    """Get the muting status of all layers in the stage.

    Args:
        stage (Usd.Stage): The USD stage to query.

    Returns:
        Dict[str, bool]: A dictionary mapping layer paths to their muting status.
    """
    all_layers = set(stage.GetUsedLayers())
    muting_status = {}
    for layer in all_layers:
        layer_path = layer.identifier
        muting_status[layer_path] = stage.IsLayerMuted(layer_path)
    return muting_status


class Notice:
    """
    Container class for Usd notices
    """

    def __init__(self):
        self._registry = []

    def register_notice_listener(self, listener: Callable):
        self._registry.append(listener)

    def unregister_notice_listener(self, listener: Callable):
        """Unregister a notice listener.

        Args:
            listener (Callable): The notice listener to unregister.

        Raises:
            ValueError: If the listener is not currently registered.
        """
        if listener not in self._registry:
            raise ValueError("Listener is not registered.")
        self._registry.remove(listener)
        if not self._registry:
            self._registry.clear()


def test_listener(notice, *args):
    print(f"Received notice: {notice}")


def filter_notices_by_prim(notices: List[Tf.Notice], prim_path: str) -> List[Tf.Notice]:
    """Filter a list of notices by a specific prim path.

    Args:
        notices (List[Tf.Notice]): A list of Tf.Notice objects to filter.
        prim_path (str): The prim path to filter the notices by.

    Returns:
        List[Tf.Notice]: A list of notices that match the given prim path.
    """
    if not notices:
        return []
    filtered_notices = []
    for notice in notices:
        if isinstance(notice, Usd.Notice) and hasattr(notice, "GetPrimPath"):
            if notice.GetPrimPath() == prim_path:
                filtered_notices.append(notice)
        else:
            continue
    return filtered_notices


def get_issued_notices() -> List[Usd.Notice]:
    """Return a list of all notices that have been issued."""
    issued_notices = []
    usd_notice_type = Tf.Type.Find(Usd.Notice)
    for notice_type in usd_notice_type.GetAllDerivedTypes():
        notice_class = notice_type.pythonClass
        notices = notice_class.GetNotices()
        issued_notices.extend(notices)
    return issued_notices


def summarize_notices(notices: List[Tf.Notice]) -> Dict[str, int]:
    """Summarize a list of Tf.Notice objects by type.

    Args:
        notices (List[Tf.Notice]): A list of Tf.Notice objects.

    Returns:
        Dict[str, int]: A dictionary mapping notice type names to their counts.
    """
    notice_counts: Dict[str, int] = {}
    for notice in notices:
        notice_type = type(notice).__name__
        notice_counts[notice_type] = notice_counts.get(notice_type, 0) + 1
    return notice_counts


class TestNotice1(Tf.Notice):
    pass


class TestNotice2(Tf.Notice):
    pass


class TestNotice3(Tf.Notice):
    pass


def save_changes_to_layer(stage: Usd.Stage, layer_path: str) -> bool:
    """Save the changes made to the stage to a specified layer.

    Args:
        stage (Usd.Stage): The stage with changes to save.
        layer_path (str): The path to the layer to save changes to.

    Returns:
        bool: True if the changes were saved successfully, False otherwise.
    """
    if not stage or not stage.GetRootLayer():
        return False
    layer = Sdf.Layer.FindOrOpen(layer_path)
    if not layer:
        layer = Sdf.Layer.CreateNew(layer_path)
        if not layer:
            return False
    try:
        stage.Export(layer.identifier)
    except Exception as e:
        print(f"Error exporting changes to layer: {e}")
        return False
    if not layer.Save():
        print(f"Error saving layer to disk: {layer_path}")
        return False
    return True


def batch_update_changed_prims(stage: Usd.Stage, notice: Usd.Notice.ObjectsChanged, func) -> List[Usd.Prim]:
    """
    Apply a function to all prims that have changed in a ObjectsChanged notice.

    Args:
        stage (Usd.Stage): The USD stage.
        notice (Usd.Notice.ObjectsChanged): The ObjectsChanged notice.
        func (function): The function to apply to each changed prim.
                         It should take a Usd.Prim as input and return a Usd.Prim.

    Returns:
        List[Usd.Prim]: A list of the updated prims.
    """
    updated_prims = []
    changed_paths = notice.GetResyncedPaths()
    for path in changed_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        updated_prim = func(prim)
        updated_prims.append(updated_prim)
    return updated_prims


def test_func(prim: Usd.Prim) -> Usd.Prim:
    """A test function that sets a custom attribute on a prim."""
    prim.CreateAttribute("test_attr", Sdf.ValueTypeNames.Int).Set(42)
    return prim


class MockNotice:

    def GetResyncedPaths(self):
        return [prim1.GetPath(), prim2.GetPath()]


def update_affected_prims(notice: Usd.Notice.ObjectsChanged) -> Dict[Usd.Prim, List[str]]:
    """Update affected prims based on the ObjectsChanged notice.

    Args:
        notice (Usd.Notice.ObjectsChanged): The ObjectsChanged notice.

    Returns:
        Dict[Usd.Prim, List[str]]: A dictionary mapping affected prims to a list of changed fields.
    """
    resync_paths: List[Sdf.Path] = notice.GetResyncedPaths()
    affected_prims: Dict[Usd.Prim, List[str]] = {}
    for path in resync_paths:
        prim: Usd.Prim = notice.GetStage().GetPrimAtPath(path)
        if prim.IsValid():
            changed_fields: List[str] = notice.GetChangedFields(path)
            affected_prims[prim] = changed_fields
    return affected_prims


def on_objects_changed(notice, stage):
    affected_prims = update_affected_prims(notice)
    print(f"Affected prims: {affected_prims}")


def get_changed_materials(notice: Usd.Notice.ObjectsChanged) -> List[Sdf.Path]:
    """Get a list of changed material prim paths from an ObjectsChanged notice."""
    changed_materials: List[Sdf.Path] = []
    for resync_path in notice.GetResyncedPaths():
        prim = notice.GetStage().GetPrimAtPath(resync_path)
        if prim.IsValid() and prim.IsA(UsdShade.Material):
            changed_materials.append(prim.GetPath())
    return changed_materials


class TestNotice:

    def __init__(self, stage):
        self.stage = stage
        self.resyncedPaths = [Sdf.Path(material_path)]

    def GetResyncedPaths(self):
        return self.resyncedPaths

    def GetStage(self):
        return self.stage


def get_hierarchy_of_changed_prims(notice: Usd.Notice.ObjectsChanged) -> Dict[str, List[str]]:
    """
    Given an ObjectsChanged notice, return a dictionary mapping prim paths to a list of their
    changed descendant prim paths (if any) that are also included in the notice.
    """
    changed_paths = notice.GetResyncedPaths()
    hierarchy: Dict[str, List[str]] = {}
    for path in changed_paths:
        path_str = str(path)
        prefix_path = path_str
        while prefix_path != "/" and prefix_path not in hierarchy:
            prefix_path = Sdf.Path(prefix_path).GetParentPath()
        if prefix_path in hierarchy:
            hierarchy[str(prefix_path)].append(path_str)
        else:
            hierarchy[path_str] = []
    return hierarchy


class MockNotice:

    def GetResyncedPaths(self):
        return resync_paths


def optimize_scene_after_changes(
    stage: Usd.Stage, resync_paths: List[Sdf.Path], fast_update_paths: List[Sdf.Path]
) -> None:
    """Optimize the scene after changes have been made.

    This function retrieves the paths of the changed prims and performs
    optimizations based on the type of change (resync or fast update).

    Args:
        stage (Usd.Stage): The stage to optimize.
        resync_paths (List[Sdf.Path]): The paths of the resynced prims.
        fast_update_paths (List[Sdf.Path]): The paths of the fast updated prims.

    Raises:
        ValueError: If the input stage is invalid.
    """
    if not stage:
        raise ValueError("Invalid stage provided.")
    for path in resync_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            _optimize_resync(prim)
    for path in fast_update_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            _optimize_fast_update(prim)


def _optimize_resync(prim: Usd.Prim) -> None:
    """Optimize a prim after resync."""
    print(f"Optimizing resync for prim: {prim.GetPath()}")


def _optimize_fast_update(prim: Usd.Prim) -> None:
    """Optimize a prim after fast update."""
    print(f"Optimizing fast update for prim: {prim.GetPath()}")


def get_masked_prims_attributes(stage: Usd.Stage, mask: Usd.StagePopulationMask) -> Dict[Sdf.Path, List[Usd.Attribute]]:
    """
    Returns a dictionary mapping each prim path in the given population mask to a list of its attributes.

    Args:
        stage (Usd.Stage): The USD stage to query.
        mask (Usd.StagePopulationMask): The population mask defining the set of prims to include.

    Returns:
        Dict[Sdf.Path, List[Usd.Attribute]]: A dictionary mapping prim paths to lists of attributes.
    """
    prim_attributes = {}
    for path in mask.GetPaths():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        attributes = prim.GetAttributes()
        prim_attributes[path] = attributes
    return prim_attributes


def create_prim(stage, prim_type, prim_path):
    prim = stage.DefinePrim(prim_path, prim_type)
    return prim


def find_models_with_asset_version(stage: Usd.Stage, asset_version: str) -> List[Usd.Prim]:
    """
    Find all model prims in the given stage that have the specified asset version.

    Args:
        stage (Usd.Stage): The USD stage to search for models.
        asset_version (str): The asset version to match.

    Returns:
        List[Usd.Prim]: A list of model prims with the specified asset version.
    """
    matching_models = []
    for prim in stage.TraverseAll():
        model_api = Usd.ModelAPI(prim)
        if model_api.IsModel():
            version = ""
            if model_api.GetAssetVersion(version):
                if version == asset_version:
                    matching_models.append(prim)
    return matching_models


def validate_model_hierarchy(prim: Usd.Prim) -> bool:
    """Validate that the prim conforms to the rules of model hierarchy.

    A prim is considered a model if it satisfies the following constraints:
        1. Its kind metadata is or inherits from "model".
        2. It is not an instance proxy.
        3. If it is a group (i.e., its kind inherits from "group"), it must
           not have any properties authored directly on it.

    Args:
        prim (Usd.Prim): The prim to validate.

    Returns:
        bool: True if the prim conforms to the model hierarchy rules, False otherwise.
    """
    model_api = Usd.ModelAPI(prim)
    if not model_api.IsKind("model", Usd.ModelAPI.KindValidationNone):
        return False
    if prim.IsInstanceProxy():
        return False
    if model_api.IsGroup():
        if prim.HasAuthoredProperties():
            return False
    return True


def update_model_asset_version(prim: Usd.Prim, version: str) -> bool:
    """Update the asset version for a model prim.

    Args:
        prim (Usd.Prim): The model prim to update the asset version for.
        version (str): The new asset version string.

    Returns:
        bool: True if the asset version was successfully updated, False otherwise.
    """
    if not prim.IsValid():
        print(f"Error: Invalid prim '{prim.GetPath()}'")
        return False
    model_api = Usd.ModelAPI(prim)
    if not model_api.IsModel():
        print(f"Error: Prim '{prim.GetPath()}' is not a model")
        return False
    model_api.SetAssetVersion(version)
    return True


def assign_new_asset_identifier(model_api: Usd.ModelAPI, new_identifier: str) -> bool:
    """Assign a new asset identifier to the model.

    Args:
        model_api (Usd.ModelAPI): The ModelAPI object to assign the identifier to.
        new_identifier (str): The new asset identifier to assign.

    Returns:
        bool: True if the asset identifier was successfully assigned, False otherwise.
    """
    if not model_api.GetPrim().IsValid():
        print(f"Error: Invalid ModelAPI object")
        return False
    if not isinstance(new_identifier, str) or not new_identifier:
        print(f"Error: Invalid asset identifier '{new_identifier}'")
        return False
    try:
        model_api.SetAssetIdentifier(new_identifier)
    except Exception as e:
        print(f"Error: Failed to set asset identifier - {str(e)}")
        return False
    if model_api.GetAssetIdentifier() != new_identifier:
        print(f"Error: Asset identifier was not set correctly")
        return False
    return True


def batch_set_payload_dependencies(
    stage: Usd.Stage, prim_paths: List[str], asset_dependencies: List[Sdf.AssetPath]
) -> None:
    """
    Set the payload asset dependencies for multiple prims in a single operation.

    Parameters:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (List[str]): A list of paths to the prims to set the dependencies for.
        asset_dependencies (List[Sdf.AssetPath]): A list of asset dependencies to set.

    Raises:
        ValueError: If any of the prim paths are invalid or do not have the UsdModelAPI applied.
    """
    vt_asset_deps = Sdf.AssetPathArray(asset_dependencies)
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")
        model_api = Usd.ModelAPI(prim)
        if not model_api:
            raise ValueError(f"Prim at path {prim_path} does not have UsdModelAPI applied")
        model_api.SetPayloadAssetDependencies(vt_asset_deps)


def get_models_by_kind(stage: Usd.Stage, kind: str) -> List[Usd.Prim]:
    """
    Get all model prims on the stage of a given kind.

    Args:
        stage (Usd.Stage): The USD stage to search for models.
        kind (str): The kind of model to search for.

    Returns:
        List[Usd.Prim]: A list of model prims of the specified kind.
    """
    prims = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        model_api = Usd.ModelAPI(prim)
        if model_api.IsModel() and model_api.IsKind(kind):
            prims.append(prim)
    return prims


def is_valid_model(prim: Usd.Prim) -> bool:
    """Check if a given prim is a valid model based on its kind metadata."""
    model_api = Usd.ModelAPI(prim)
    if not model_api.IsModel():
        return False
    kind = model_api.GetKind()
    if not kind:
        return False
    if not Kind.Registry.IsA(kind, Kind.Tokens.model):
        return False
    return True


def set_model_kind_bulk(stage: Usd.Stage, prim_paths: List[str], kind: str) -> List[bool]:
    """Set the model kind for multiple prims in bulk.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (List[str]): A list of prim paths to set the model kind for.
        kind (str): The model kind to set.

    Returns:
        List[bool]: A list of booleans indicating whether each prim's model kind was successfully set.
    """
    results = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            results.append(False)
            continue
        model_api = Usd.ModelAPI(prim)
        if not model_api:
            results.append(False)
            continue
        results.append(model_api.SetKind(kind))
    return results


def remove_asset_info(prim: Usd.Prim) -> bool:
    """Remove the assetInfo dictionary from the prim."""
    model_api = Usd.ModelAPI(prim)
    if not model_api:
        return False
    asset_info = model_api.GetAssetInfo()
    if not asset_info:
        return True
    model_api.SetAssetInfo({})
    return True


def monitor_edit_target_changes(stage: Usd.Stage, callback: callable) -> Tf.Notice:
    """Monitor changes to the edit target of a stage.

    Args:
        stage (Usd.Stage): The stage to monitor for edit target changes.
        callback (callable): A callback function to be invoked when the edit target changes.
            The callback should accept a single argument of type Usd.Notice.StageEditTargetChanged.

    Returns:
        Tf.Notice: A notice object representing the registered callback. This can be used to
            unregister the callback later if needed.
    """
    if not stage:
        raise ValueError("Invalid stage provided.")
    if not callable(callback):
        raise ValueError("Invalid callback function provided.")

    def callback_wrapper(notice, sender):
        callback(notice)

    notice = Tf.Notice.Register(Usd.Notice.StageEditTargetChanged, callback_wrapper, stage)
    return notice


def on_edit_target_changed(notice: Usd.Notice.StageEditTargetChanged):
    print(f"Edit target changed: {notice.GetStage().GetEditTarget().GetLayer().identifier}")


def get_all_payload_prims(stage: Usd.Stage) -> List[Usd.Prim]:
    """
    Get all prims in the stage that have a payload.

    Args:
        stage (Usd.Stage): The USD stage to search for payloads.

    Returns:
        List[Usd.Prim]: A list of prims that have a payload.
    """
    payload_prims: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        if prim.HasPayload():
            payload_prims.append(prim)
    return payload_prims


def assign_material_to_hierarchy(stage: Usd.Stage, root_prim_path: str, material: UsdShade.Material):
    """Assign a material to a hierarchy of prims."""
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid():
        raise ValueError(f"Prim at path {root_prim_path} does not exist.")
    for prim in Usd.PrimRange(root_prim):
        mesh = UsdGeom.Mesh(prim)
        if mesh:
            binding_api = UsdShade.MaterialBindingAPI(mesh)
            binding_api.Bind(material)


def get_world_transform(prim: Usd.Prim) -> Gf.Matrix4d:
    """
    Returns the world transformation matrix for a given prim.

    Args:
        prim (Usd.Prim): The prim to get the world transform for.

    Returns:
        Gf.Matrix4d: The world transformation matrix.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim {prim} is not transformable")
    local_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    parent = prim.GetParent()
    while parent.IsValid():
        parent_xformable = UsdGeom.Xformable(parent)
        if parent_xformable:
            parent_transform = parent_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            local_transform = local_transform * parent_transform
        parent = parent.GetParent()
    return local_transform


def create_hierarchy_from_json(stage: Usd.Stage, json_data: dict, parent_path: str = "/") -> None:
    """Create a USD hierarchy from a JSON dictionary.

    Args:
        stage (Usd.Stage): The USD stage to create the hierarchy on.
        json_data (dict): The JSON dictionary representing the hierarchy.
        parent_path (str, optional): The parent path for the hierarchy. Defaults to "/".
    """
    for prim_name, prim_data in json_data.items():
        prim_path = f"{parent_path}{prim_name}"
        prim_type = prim_data.get("type", "Xform")
        prim = stage.DefinePrim(prim_path, prim_type)
        attributes = prim_data.get("attributes", {})
        for attr_name, attr_value in attributes.items():
            if not prim.HasAttribute(attr_name):
                prim.CreateAttribute(attr_name, attr_value.__class__.__name__)
            attr = prim.GetAttribute(attr_name)
            attr.Set(attr_value)
        children = prim_data.get("children", {})
        create_hierarchy_from_json(stage, children, f"{prim_path}/")


def sync_prim_variants_across_stages(source_stage: Usd.Stage, dest_stage: Usd.Stage, prim_path: str) -> bool:
    """Sync variant selections for a prim from one stage to another.

    Args:
        source_stage (Usd.Stage): The stage to copy variant selections from.
        dest_stage (Usd.Stage): The stage to copy variant selections to.
        prim_path (str): The path of the prim to sync variant selections for.

    Returns:
        bool: True if the variant selections were synced successfully, False otherwise.
    """
    source_prim = source_stage.GetPrimAtPath(prim_path)
    dest_prim = dest_stage.GetPrimAtPath(prim_path)
    if not source_prim.IsValid() or not dest_prim.IsValid():
        print(f"Invalid prim path: {prim_path}")
        return False
    source_vsets = source_prim.GetVariantSets()
    for vset_name in source_vsets.GetNames():
        source_vset = source_vsets.GetVariantSet(vset_name)
        if dest_prim.HasVariantSets():
            dest_vsets = dest_prim.GetVariantSets()
            if dest_vsets.HasVariantSet(vset_name):
                dest_vset = dest_vsets.GetVariantSet(vset_name)
                selected_variant = source_vset.GetVariantSelection()
                if dest_vset.GetVariantNames().count(selected_variant) > 0:
                    dest_vset.SetVariantSelection(selected_variant)
                else:
                    print(f"Variant {selected_variant} not found in destination variant set {vset_name}")
            else:
                print(f"Variant set {vset_name} not found on destination prim")
        else:
            print(f"No variant sets found on destination prim")
    return True


def collect_all_materials_in_stage(stage: Usd.Stage) -> List[UsdShade.Material]:
    """
    Collect all UsdShadeMaterial prims in the given USD stage.

    Args:
        stage (Usd.Stage): The USD stage to search for materials.

    Returns:
        List[UsdShade.Material]: A list of all UsdShadeMaterial prims in the stage.
    """
    materials: List[UsdShade.Material] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            material = UsdShade.Material(prim)
            materials.append(material)
    return materials


def find_and_replace_text_in_metadata(prim: Usd.Prim, metadata_key: str, old_text: str, new_text: str):
    """Find and replace text in a metadata field on a prim.

    Args:
        prim (Usd.Prim): The prim to update metadata on.
        metadata_key (str): The key of the metadata field to update.
        old_text (str): The text to find and replace.
        new_text (str): The replacement text.

    Raises:
        ValueError: If the prim is not valid.
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid")
    metadata_value = prim.GetMetadata(metadata_key)
    if metadata_value is None or not isinstance(metadata_value, str):
        return
    new_metadata_value = metadata_value.replace(old_text, new_text)
    if new_metadata_value != metadata_value:
        prim.SetMetadata(metadata_key, new_metadata_value)


def batch_remove_prims(stage: Usd.Stage, prim_paths: list[str]) -> None:
    """Remove multiple prims from the stage given their paths."""
    prim_paths.sort(reverse=True)
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            continue
        prim.SetActive(False)
        stage.RemovePrim(prim_path)


def create_prim_with_default_material(
    stage: Usd.Stage, prim_type: str, prim_path: str, material_path: str = ""
) -> Usd.Prim:
    """Create a prim with a default material assigned.

    Args:
        stage (Usd.Stage): The USD stage to create the prim on.
        prim_type (str): The type of prim to create (e.g., "Sphere", "Cube").
        prim_path (str): The path where the prim should be created.
        material_path (str, optional): The path to the material to assign. If empty, a default material will be created. Defaults to "".

    Returns:
        Usd.Prim: The created prim with the default material assigned.
    """
    prim = stage.DefinePrim(prim_path, prim_type)
    if not prim.IsValid():
        raise ValueError(f"Failed to create prim at path {prim_path}")
    if not material_path:
        material_path = f"{prim_path}_Material"
        material = UsdShade.Material.Define(stage, material_path)
        shader_path = f"{material_path}_Shader"
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.8, 0.8))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(stage.GetPrimAtPath(material_path)))
    return prim


def create_hierarchy_from_blueprint(stage: Usd.Stage, blueprint: dict) -> None:
    """Create a hierarchy of prims from a blueprint dictionary.

    The blueprint dictionary should have the following structure:
    {
        "<prim_name>": {
            "type": "<prim_type>",
            "xformOp:translate": [<tx>, <ty>, <tz>],
            "children": {
                "<child_prim_name>": {
                    "type": "<child_prim_type>",
                    ...
                },
                ...
            }
        },
        ...
    }

    Args:
        stage (Usd.Stage): The stage to create the prims on.
        blueprint (dict): The blueprint dictionary specifying the hierarchy.

    Raises:
        ValueError: If the blueprint dictionary is not valid.
    """

    def create_prim_from_blueprint(parent_prim: Usd.Prim, prim_blueprint: dict):
        if len(prim_blueprint) != 1:
            raise ValueError("Each prim blueprint must have exactly one key")
        prim_name = next(iter(prim_blueprint))
        prim_type = prim_blueprint[prim_name].get("type", "Xform")
        prim = stage.DefinePrim(parent_prim.GetPath().AppendChild(prim_name), prim_type)
        if "xformOp:translate" in prim_blueprint[prim_name]:
            translation = prim_blueprint[prim_name]["xformOp:translate"]
            if len(translation) != 3:
                raise ValueError("Translation must have 3 components")
            UsdGeom.XformCommonAPI(prim).SetTranslate(translation)
        if "children" in prim_blueprint[prim_name]:
            for child_blueprint in prim_blueprint[prim_name]["children"].values():
                create_prim_from_blueprint(prim, {prim_name: child_blueprint})

    for root_blueprint in blueprint.values():
        create_prim_from_blueprint(stage.GetPseudoRoot(), {"root": root_blueprint})


def create_motion_trail_visualization(stage: Usd.Stage, prim_path: str, num_points: int) -> Usd.Prim:
    """Create a motion trail visualization for a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path of the prim to create the motion trail for.
        num_points (int): The number of points in the motion trail.

    Returns:
        Usd.Prim: The created motion trail prim.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if not translate_ops:
        raise ValueError(f"Prim at path {prim_path} does not have a translate op.")
    motion_trail_prim = stage.DefinePrim(f"{prim_path}_MotionTrail", "PointInstancer")
    points_attr = motion_trail_prim.CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray)
    translate_attr = translate_ops[0].GetAttr()
    time_samples = translate_attr.GetTimeSamples()
    if not time_samples:
        translate = translate_attr.Get()
        points_attr.Set([Gf.Vec3f(translate[0], translate[1], translate[2])] * num_points)
    else:
        start_time = time_samples[0]
        end_time = time_samples[-1]
        delta = (end_time - start_time) / (num_points - 1)
        points = []
        for i in range(num_points):
            t = start_time + i * delta
            translate = translate_attr.Get(t)
            points.append(Gf.Vec3f(translate[0], translate[1], translate[2]))
        points_attr.Set(points)
    return motion_trail_prim


def find_and_replace_text_in_attributes(prim: Usd.Prim, find_text: str, replace_text: str) -> int:
    """Find and replace text in string attributes on the given prim.

    Args:
        prim (Usd.Prim): The prim to search for attributes.
        find_text (str): The text to find in attribute values.
        replace_text (str): The text to replace the found text with.

    Returns:
        int: The number of attributes modified.
    """
    num_modified = 0
    for attr in prim.GetAttributes():
        if attr.GetTypeName() == Sdf.ValueTypeNames.String:
            old_value = attr.Get()
            if old_value is not None:
                new_value = old_value.replace(find_text, replace_text)
                if new_value != old_value:
                    success = attr.Set(new_value)
                    if success:
                        num_modified += 1
    return num_modified


def batch_create_relationships(prim: Usd.Prim, relationship_names: list[str]) -> None:
    """Batch create multiple relationships on a prim.

    Args:
        prim (Usd.Prim): The prim to create the relationships on.
        relationship_names (list[str]): A list of relationship names to create.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    for rel_name in relationship_names:
        if not Sdf.Path.IsValidNamespacedIdentifier(rel_name):
            raise ValueError(f"Invalid relationship name: {rel_name}")
        if prim.HasRelationship(rel_name):
            continue
        prim.CreateRelationship(rel_name)


def check_prim_relationships(prim: Usd.Prim) -> dict:
    """Check the relationships of a given prim and return a dictionary of relationship names and targets."""
    relationships = {}
    for rel in prim.GetRelationships():
        rel_name = rel.GetName()
        targets = rel.GetTargets()
        if targets:
            relationships[rel_name] = [str(target) for target in targets]
        else:
            relationships[rel_name] = []
    return relationships


def mirror_prim_attributes(source_prim: Usd.Prim, target_prim: Usd.Prim):
    """Mirror the attributes from source_prim to target_prim."""
    if not source_prim.IsValid() or not target_prim.IsValid():
        raise ValueError("Both source_prim and target_prim must be valid.")
    for attr in source_prim.GetAttributes():
        attr_name = attr.GetName()
        attr_value = attr.Get()
        if target_prim.HasAttribute(attr_name):
            target_attr = target_prim.GetAttribute(attr_name)
            target_attr.Set(attr_value)
        else:
            target_attr = target_prim.CreateAttribute(attr_name, attr.GetTypeName())
            target_attr.Set(attr_value)


def gather_statistics_for_hierarchy(prim: Usd.Prim) -> Dict[str, int]:
    """Gather statistics for the prim hierarchy.

    Returns a dictionary with the following keys:
    - totalPrims: Total number of prims in the hierarchy
    - totalMeshes: Total number of meshes in the hierarchy
    - totalXforms: Total number of transforms in the hierarchy
    - totalMeshXforms: Total number of meshes that have a transform

    Args:
        prim (Usd.Prim): The root prim to gather statistics for.

    Returns:
        Dict[str, int]: The statistics dictionary.
    """
    stats = {"totalPrims": 0, "totalMeshes": 0, "totalXforms": 0, "totalMeshXforms": 0}
    for descendant in Usd.PrimRange(prim):
        stats["totalPrims"] += 1
        if descendant.IsA(UsdGeom.Mesh):
            stats["totalMeshes"] += 1
        if UsdGeom.Xformable(descendant):
            stats["totalXforms"] += 1
            if descendant.IsA(UsdGeom.Mesh):
                stats["totalMeshXforms"] += 1
    return stats


def find_prims_with_metadata(stage: Usd.Stage, metadata_name: str) -> List[Usd.Prim]:
    """
    Find all prims on the given stage that have the specified metadata field.

    Args:
        stage (Usd.Stage): The stage to search for prims.
        metadata_name (str): The name of the metadata field to search for.

    Returns:
        List[Usd.Prim]: A list of prims that have the specified metadata field.
    """
    prims_with_metadata = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.HasMetadata(metadata_name):
            prims_with_metadata.append(prim)
    return prims_with_metadata


def set_prim_attributes_from_dict(prim: Usd.Prim, attr_dict: Dict[str, Any]) -> None:
    """Set attributes on a prim from a dictionary.

    The dictionary keys are attribute names and the values are the attribute values.
    If an attribute doesn't exist, it will be created.

    Args:
        prim (Usd.Prim): The prim to set attributes on.
        attr_dict (Dict[str, Any]): The dictionary of attribute names and values.

    Raises:
        ValueError: If the prim is not valid.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    for attr_name, attr_value in attr_dict.items():
        attr = prim.GetAttribute(attr_name)
        if not attr:
            if isinstance(attr_value, bool):
                attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Bool)
            elif isinstance(attr_value, int):
                attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Int)
            elif isinstance(attr_value, float):
                attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
            elif isinstance(attr_value, str):
                attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.String)
            else:
                raise ValueError(f"Unsupported attribute value type: {type(attr_value)}")
        attr.Set(attr_value)


def copy_prim_attributes(src_prim: Usd.Prim, dst_prim: Usd.Prim):
    """Copy attributes from one prim to another."""
    if not src_prim.IsValid():
        raise ValueError("Source prim is not valid.")
    if not dst_prim.IsValid():
        raise ValueError("Destination prim is not valid.")
    for attr in src_prim.GetAttributes():
        attr_name = attr.GetName()
        if dst_prim.HasAttribute(attr_name):
            continue
        attr_value = attr.Get()
        dst_attr = dst_prim.CreateAttribute(attr_name, attr.GetTypeName())
        dst_attr.Set(attr_value)


def bake_animation_to_keyframes(stage: Usd.Stage, prim_path: str, start_time: float, end_time: float, time_step: float):
    """Bake the animation of a prim to keyframes.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to bake.
        start_time (float): The start time of the baking range.
        end_time (float): The end time of the baking range.
        time_step (float): The time step between keyframes.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    translate_op = None
    rotate_op = None
    scale_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            rotate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
    current_time = start_time
    while current_time <= end_time:
        if translate_op:
            translate_op.Set(time=current_time, value=translate_op.Get(time=current_time))
        if rotate_op:
            rotate_op.Set(time=current_time, value=rotate_op.Get(time=current_time))
        if scale_op:
            scale_op.Set(time=current_time, value=scale_op.Get(time=current_time))
        current_time += time_step


def mirror_prim(stage: Usd.Stage, prim_path: str, axis: str = "x") -> Usd.Prim:
    """Mirror a prim and its subtree across the specified axis.

    Args:
        stage (Usd.Stage): The stage containing the prim to mirror.
        prim_path (str): The path of the prim to mirror.
        axis (str): The axis to mirror across. Must be "x", "y", or "z". Defaults to "x".

    Returns:
        Usd.Prim: The newly created mirrored prim.
    """
    if axis not in ["x", "y", "z"]:
        raise ValueError(f"Invalid mirror axis '{axis}'. Must be 'x', 'y', or 'z'.")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"No prim found at path '{prim_path}'")
    prim_name = prim.GetName()
    parent_path = prim.GetPath().GetParentPath()
    mirrored_prim_name = f"{prim_name}_mirrored"
    mirrored_prim_path = parent_path.AppendChild(mirrored_prim_name)
    mirrored_prim = stage.DefinePrim(mirrored_prim_path)
    xformable = UsdGeom.Xformable(prim)
    if xformable:
        transform_ops = xformable.GetOrderedXformOps()
        for op in transform_ops:
            op_name = op.GetOpName()
            if op_name == "xformOp:translate":
                translation = op.Get()
                if axis == "x":
                    translation[0] *= -1
                elif axis == "y":
                    translation[1] *= -1
                else:
                    translation[2] *= -1
                add_translate_op(UsdGeom.Xformable(mirrored_prim)).Set(translation)
            elif op_name == "xformOp:rotateXYZ":
                rotation = op.Get()
                if axis == "x":
                    rotation[1] *= -1
                    rotation[2] *= -1
                elif axis == "y":
                    rotation[0] *= -1
                    rotation[2] *= -1
                else:
                    rotation[0] *= -1
                    rotation[1] *= -1
                add_rotate_xyz_op(UsdGeom.Xformable(mirrored_prim)).Set(rotation)
            elif op_name == "xformOp:scale":
                add_scale_op(UsdGeom.Xformable(mirrored_prim)).Set(op.Get())
    if prim.IsA(UsdGeom.Imageable):
        mesh = UsdGeom.Mesh(prim)
        if mesh:
            mirrored_mesh = UsdGeom.Mesh.Define(stage, mirrored_prim_path)
            points = mesh.GetPointsAttr().Get()
            if points:
                mirrored_points = []
                for point in points:
                    if axis == "x":
                        mirrored_points.append(Gf.Vec3f(-point[0], point[1], point[2]))
                    elif axis == "y":
                        mirrored_points.append(Gf.Vec3f(point[0], -point[1], point[2]))
                    else:
                        mirrored_points.append(Gf.Vec3f(point[0], point[1], -point[2]))
                mirrored_mesh.CreatePointsAttr().Set(mirrored_points)
            mirrored_mesh.CreateFaceVertexCountsAttr().Set(mesh.GetFaceVertexCountsAttr().Get())
            mirrored_mesh.CreateFaceVertexIndicesAttr().Set(mesh.GetFaceVertexIndicesAttr().Get())
    for child in prim.GetChildren():
        mirror_prim(stage, child.GetPath(), axis)
    return mirrored_prim


def check_prim_for_cycles(prim: Usd.Prim) -> bool:
    """Check if the given prim or any of its descendants has a cycle in its composition."""
    visited = set()

    def traverse_prim(current_prim: Usd.Prim) -> bool:
        if current_prim in visited:
            return True
        visited.add(current_prim)
        for child in current_prim.GetAllChildren():
            if traverse_prim(child):
                return True
        for rel in current_prim.GetRelationships():
            for target_path in rel.GetTargets():
                target_prim = current_prim.GetStage().GetPrimAtPath(target_path)
                if target_prim.IsValid() and traverse_prim(target_prim):
                    return True
        return False

    return traverse_prim(prim)


def create_geometry_variants_from_shapes(stage: Usd.Stage, prim_path: str, shapes: List[str]) -> Usd.Prim:
    """Create a prim with geometry variants from a list of shape names.

    Args:
        stage (Usd.Stage): The USD stage to create the prim on.
        prim_path (str): The path where the prim will be created.
        shapes (List[str]): A list of shape names to create as geometry variants.

    Returns:
        Usd.Prim: The created prim with geometry variants.
    """
    prim = stage.DefinePrim(prim_path)
    if not prim:
        raise ValueError(f"Failed to create prim at path {prim_path}")
    variant_set = prim.GetVariantSets().AddVariantSet("geometry")
    for shape in shapes:
        variant_set.AddVariant(shape)
        variant_set.SetVariantSelection(shape)
        with variant_set.GetVariantEditContext():
            mesh_prim = UsdGeom.Mesh.Define(stage, f"{prim_path}/{shape}")
            if not mesh_prim:
                raise ValueError(f"Failed to create mesh prim for variant {shape}")
    variant_set.SetVariantSelection(shapes[0])
    return prim


def collect_all_authored_properties(prim: Usd.Prim, recurse_instances: bool = False) -> List[Usd.Property]:
    """
    Collect all authored properties (attributes and relationships) on the given prim and its descendants.

    Args:
        prim (Usd.Prim): The prim to collect properties from.
        recurse_instances (bool, optional): If True, recurse into instances. Defaults to False.

    Returns:
        List[Usd.Property]: A list of all authored properties.
    """
    properties = []
    properties.extend(prim.GetAuthoredProperties())
    for child in prim.GetAllChildren():
        if not recurse_instances and child.IsInstance():
            continue
        properties.extend(collect_all_authored_properties(child, recurse_instances))
    return properties


def find_prims_with_attribute_value(stage: Usd.Stage, attribute_name: str, attribute_value: Any) -> List[Usd.Prim]:
    """
    Find all prims on the given stage that have an attribute with the given name and value.

    Args:
        stage (Usd.Stage): The stage to search for prims.
        attribute_name (str): The name of the attribute to search for.
        attribute_value (Any): The value to match against the attribute.

    Returns:
        List[Usd.Prim]: A list of prims that have an attribute with the given name and value.
    """
    matching_prims = []
    for prim in stage.Traverse():
        if prim.HasAttribute(attribute_name):
            attribute = prim.GetAttribute(attribute_name)
            value = attribute.Get()
            if value == attribute_value:
                matching_prims.append(prim)
    return matching_prims


def synchronize_prim_visibility(prim: Usd.Prim, visibility: str) -> bool:
    """Synchronize the visibility of a prim and its descendants.

    Args:
        prim (Usd.Prim): The prim to synchronize visibility for.
        visibility (str): The visibility value to set. Must be one of 'inherited', 'invisible'.

    Returns:
        bool: True if the visibility was successfully set, False otherwise.
    """
    if not prim.IsValid():
        return False
    if visibility not in ["inherited", "invisible"]:
        return False
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        return False
    imageable.GetVisibilityAttr().Set(visibility)
    for child in prim.GetAllChildren():
        synchronize_prim_visibility(child, visibility)
    return True


def batch_toggle_prims_active_state(stage: Usd.Stage, prim_paths: List[str]) -> List[Tuple[str, bool]]:
    """Toggle the active state of multiple prims in a single operation.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (List[str]): The paths of the prims to toggle.

    Returns:
        List[Tuple[str, bool]]: A list of tuples containing the prim path and the new active state.
    """
    results = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            results.append((prim_path, None))
            continue
        current_state = prim.IsActive()
        new_state = not current_state
        prim.SetActive(new_state)
        results.append((prim_path, new_state))
    return results


def batch_remove_payloads(stage: Usd.Stage, prim_paths: Sequence[str]) -> int:
    """Remove payloads from the specified prims.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (Sequence[str]): The paths of the prims to remove payloads from.

    Returns:
        int: The number of prims that had their payloads removed.
    """
    num_removed = 0
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        if prim.HasPayload():
            prim.ClearPayload()
            num_removed += 1
    return num_removed


def generate_texture_variants(stage: Usd.Stage, prim_path: str, texture_variants: Dict[str, str]) -> None:
    """
    Generate texture variants for a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        texture_variants (Dict[str, str]): A dictionary mapping variant set names to variant names.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    variant_sets = prim.GetVariantSets()
    if not variant_sets:
        raise ValueError(f"Prim at path {prim_path} has no variant sets.")
    for variant_set_name, variant_name in texture_variants.items():
        if not variant_sets.HasVariantSet(variant_set_name):
            variant_set = variant_sets.AddVariantSet(variant_set_name)
        else:
            variant_set = variant_sets.GetVariantSet(variant_set_name)
        if not variant_set.HasAuthoredVariant(variant_name):
            variant_set.AddVariant(variant_name)
        variant_selection = variant_set.GetVariantSelection()
        if variant_selection != variant_name:
            variant_set.SetVariantSelection(variant_name)


def create_collision_shape_from_prims(
    stage: Usd.Stage, prim_paths: List[str], collision_prim_path: str
) -> UsdPhysics.CollisionAPI:
    """Create a collision shape from a list of prim paths."""
    if not stage:
        raise ValueError("Stage is invalid.")
    if not prim_paths:
        raise ValueError("Prim paths list is empty.")
    if not collision_prim_path:
        raise ValueError("Collision prim path is empty.")
    collision_prim = stage.DefinePrim(collision_prim_path, "Xform")
    if not collision_prim.IsValid():
        raise RuntimeError(f"Failed to create collision prim at path {collision_prim_path}")
    collision_api = UsdPhysics.CollisionAPI.Apply(collision_prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(collision_prim)
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            print(f"Warning: Prim at path {prim_path} is not a valid mesh. Skipping.")
            continue
        mesh_collision.CreateApproximationAttr().Set("none")
        mesh_collision.CreateMeshAttr().Set(mesh.GetPrim().GetPath())
    return collision_api


def collect_all_prims_with_attribute(stage: Usd.Stage, attr_name: str) -> list[Usd.Prim]:
    """
    Collect all prims on the stage that have an attribute with the given name.

    Args:
        stage (Usd.Stage): The stage to search for prims.
        attr_name (str): The name of the attribute to search for.

    Returns:
        list[Usd.Prim]: A list of prims that have the specified attribute.
    """
    prims_with_attr = []
    for prim in stage.TraverseAll():
        if prim.HasAttribute(attr_name):
            prims_with_attr.append(prim)
    return prims_with_attr


def apply_variant_set_to_hierarchy(prim: Usd.Prim, variant_set_name: str, variant_selection: str):
    """Recursively apply a variant selection to a prim and its descendents.

    Args:
        prim (Usd.Prim): The starting prim to apply the variant selection.
        variant_set_name (str): The name of the variant set.
        variant_selection (str): The name of the variant selection.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    if prim.HasVariantSets():
        variant_sets = prim.GetVariantSets()
        if variant_sets.HasVariantSet(variant_set_name):
            variant_set = variant_sets.GetVariantSet(variant_set_name)
            if variant_set.HasAuthoredVariant(variant_selection):
                variant_set.SetVariantSelection(variant_selection)
            else:
                raise ValueError(f"Variant set {variant_set_name} does not have variant selection {variant_selection}.")
    for child in prim.GetAllChildren():
        apply_variant_set_to_hierarchy(child, variant_set_name, variant_selection)


def get_prim_attributes_as_dict(prim: Usd.Prim) -> Dict[str, Any]:
    """
    Returns a dictionary containing all attributes of the given prim.

    Args:
        prim (Usd.Prim): The prim to retrieve attributes from.

    Returns:
        Dict[str, Any]: A dictionary where keys are attribute names and values are attribute values.
    """
    attributes_dict: Dict[str, Any] = {}
    for attr in prim.GetAttributes():
        attr_name = attr.GetName()
        attr_value = attr.Get()
        attributes_dict[attr_name] = attr_value
    return attributes_dict


def align_prims_to_grid(stage: Usd.Stage, grid_size: float = 10.0, up_axis="Y"):
    """Align all prims on the stage to the nearest grid point.

    Args:
        stage (Usd.Stage): The stage containing the prims to align.
        grid_size (float): The size of the grid cells. Defaults to 10.0.
        up_axis (str): The up axis, either "Y" or "Z". Defaults to "Y".

    Returns:
        None
    """
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        xformable = UsdGeom.Xformable(prim)
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if not translate_op:
            continue
        translation = translate_op.Get()
        if up_axis == "Y":
            rounded_translation = Gf.Vec3d(
                grid_size * round(translation[0] / grid_size),
                grid_size * round(translation[1] / grid_size),
                grid_size * round(translation[2] / grid_size),
            )
        elif up_axis == "Z":
            rounded_translation = Gf.Vec3d(
                grid_size * round(translation[0] / grid_size),
                grid_size * round(translation[2] / grid_size),
                grid_size * round(translation[1] / grid_size),
            )
        else:
            raise ValueError(f"Invalid up axis: {up_axis}")
        translate_op.Set(rounded_translation)


def create_animated_transform(
    stage: Usd.Stage,
    prim_path: str,
    rotate_x_values: list[tuple[float, float]],
    translate_y_values: list[tuple[float, float]],
) -> Usd.Prim:
    """Create a prim with animated rotation and translation.

    Args:
        stage (Usd.Stage): The stage to create the prim on.
        prim_path (str): The path of the prim to create.
        rotate_x_values (list[tuple[float, float]]): A list of (time, value) tuples for animating rotation around the X-axis.
        translate_y_values (list[tuple[float, float]]): A list of (time, value) tuples for animating translation along the Y-axis.

    Returns:
        Usd.Prim: The created prim.

    Raises:
        ValueError: If the prim path is invalid or the prim already exists.
    """
    if not Sdf.Path.IsValidPathString(prim_path):
        raise ValueError(f"Invalid prim path: {prim_path}")
    existing_prim = stage.GetPrimAtPath(prim_path)
    if existing_prim.IsValid():
        raise ValueError(f"Prim already exists at path: {prim_path}")
    prim = stage.DefinePrim(prim_path, "Xform")
    xform = UsdGeom.Xform(prim)
    rotate_attr = xform.AddRotateXOp().GetAttr()
    for time, value in rotate_x_values:
        rotate_attr.Set(value, Usd.TimeCode(time))
    translate_attr = add_translate_op(xform).GetAttr()
    for time, value in translate_y_values:
        translate_attr.Set(Gf.Vec3d(0, value, 0), Usd.TimeCode(time))
    return prim


def create_hierarchy_from_table(stage: Usd.Stage, table: List[Dict[str, str]]) -> None:
    """Create a prim hierarchy from a list of dictionaries.

    Each dictionary represents a prim and should have these keys:
    - path: The prim path.
    - type: The prim type.
    - parent: The parent prim path, or empty string for the root prim.
    """
    prims: Dict[str, Usd.Prim] = {}
    for row in table:
        path = row["path"]
        prim_type = row["type"]
        parent_path = row["parent"]
        if parent_path and parent_path not in prims:
            raise ValueError(f"Parent path {parent_path} does not exist.")
        parent_prim = prims.get(parent_path)
        if parent_prim:
            prim_path = parent_prim.GetPath().AppendChild(path)
        else:
            prim_path = Sdf.Path.absoluteRootPath.AppendChild(path)
        prim = stage.DefinePrim(prim_path, prim_type)
        prims[str(prim_path)] = prim


def unload_all_descendants_of_prims(stage: Usd.Stage, prim_paths: List[str]):
    """Unload all descendants of the specified prims using UsdStageLoadRules.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths for which to unload descendants.
    """
    load_rules = Usd.StageLoadRules()
    for prim_path in prim_paths:
        if not Sdf.Path(prim_path).IsPrimPath():
            raise ValueError(f"Invalid prim path: {prim_path}")
        if not stage.GetPrimAtPath(prim_path):
            raise ValueError(f"Prim does not exist: {prim_path}")
        load_rules.Unload(prim_path)
    stage.SetLoadRules(load_rules)
    stage.Reload()


def create_test_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World")
    stage.DefinePrim("/World/CharacterA")
    stage.DefinePrim("/World/CharacterA/Mesh")
    stage.DefinePrim("/World/CharacterA/Rig")
    stage.DefinePrim("/World/CharacterB")
    stage.DefinePrim("/World/CharacterB/Mesh")
    stage.DefinePrim("/World/CharacterB/Rig")
    stage.DefinePrim("/World/Prop")
    return stage


def find_and_replace_text_in_shaders(stage: Usd.Stage, search_text: str, replace_text: str) -> int:
    """Find and replace text in all shader source code on the given stage.

    Args:
        stage (Usd.Stage): The USD stage to search for shaders.
        search_text (str): The text to search for in the shader source code.
        replace_text (str): The text to replace the search text with.

    Returns:
        int: The number of substitutions made.
    """
    num_substitutions = 0
    for prim in stage.TraverseAll():
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            source_attr = shader.GetImplementationSourceAttr()
            if source_attr.IsValid():
                source_code = source_attr.Get()
                if search_text in source_code:
                    new_source_code = source_code.replace(search_text, replace_text)
                    source_attr.Set(new_source_code)
                    num_substitutions += source_code.count(search_text)
    return num_substitutions


def batch_set_attributes(prim: Usd.Prim, attr_dict: Dict[str, Any]) -> bool:
    """Batch set attribute values on a prim from a dictionary.

    Args:
        prim (Usd.Prim): The prim to set attributes on.
        attr_dict (Dict[str, Any]): A dictionary mapping attribute names to their values.

    Returns:
        bool: True if all attributes were set successfully, False otherwise.
    """
    for attr_name, attr_value in attr_dict.items():
        attr = prim.GetAttribute(attr_name)
        if not attr:
            if isinstance(attr_value, bool):
                value_type = Sdf.ValueTypeNames.Bool
            elif isinstance(attr_value, int):
                value_type = Sdf.ValueTypeNames.Int
            elif isinstance(attr_value, float):
                value_type = Sdf.ValueTypeNames.Float
            elif isinstance(attr_value, str):
                value_type = Sdf.ValueTypeNames.String
            elif isinstance(attr_value, Gf.Vec3d):
                value_type = Sdf.ValueTypeNames.Double3
            else:
                raise ValueError(f"Unsupported attribute value type: {type(attr_value)}")
            attr = prim.CreateAttribute(attr_name, value_type)
        success = attr.Set(attr_value)
        if not success:
            return False
    return True


def rename_prim(stage: Usd.Stage, old_path: str, new_name: str) -> Usd.Prim:
    """Rename a prim at the given path with the new name."""
    old_prim = stage.GetPrimAtPath(old_path)
    if not old_prim.IsValid():
        raise ValueError(f"Prim at path {old_path} does not exist.")
    if not Sdf.Path.IsValidIdentifier(new_name):
        raise ValueError(f"{new_name} is not a valid prim name.")
    parent_path = old_prim.GetPath().GetParentPath()
    new_path = parent_path.AppendChild(new_name)
    if stage.GetPrimAtPath(new_path).IsValid():
        raise ValueError(f"A prim already exists at path {new_path}.")
    try:
        stage.DefinePrim(new_path)
        old_prim.SetActive(False)
    except Tf.ErrorException as e:
        raise RuntimeError(f"Failed to rename prim: {e}")
    new_prim = stage.GetPrimAtPath(new_path)
    return new_prim


def set_variant_for_prims(stage: Usd.Stage, prim_paths: List[str], variant_set_name: str, variant_name: str) -> None:
    """Set a variant for multiple prims on a stage.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The paths of the prims to set the variant for.
        variant_set_name (str): The name of the variant set.
        variant_name (str): The name of the variant to set.

    Raises:
        ValueError: If any prim path is invalid or does not have the specified variant set.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")
        variant_sets = prim.GetVariantSets()
        if not variant_sets.HasVariantSet(variant_set_name):
            raise ValueError(f"Prim at path {prim_path} does not have variant set {variant_set_name}")
        variant_set = variant_sets.GetVariantSet(variant_set_name)
        variant_set.SetVariantSelection(variant_name)


def remove_all_children(prim: Usd.Prim):
    """Remove all children from the given prim."""
    children = prim.GetAllChildren()
    for child in children:
        child.SetActive(False)
        child.GetStage().RemovePrim(child.GetPath())


def generate_lod_variants(stage: Usd.Stage, prim_path: str, lod_names: List[str]) -> None:
    """Generate LOD variants for a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        lod_names (List[str]): The names of the LOD variants to create.

    Raises:
        ValueError: If the prim does not exist or is not a UsdGeom.Imageable.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    imageable = UsdGeom.Imageable(prim)
    if not imageable:
        raise ValueError(f"Prim at path {prim_path} is not a UsdGeom.Imageable.")
    variant_set = prim.GetVariantSets().AddVariantSet("LOD")
    for lod_name in lod_names:
        variant_set.AddVariant(lod_name)
        variant_set.SetVariantSelection(lod_name)
        with variant_set.GetVariantEditContext():
            imageable.CreateVisibilityAttr().Set("invisible" if lod_name != "LOD0" else "inherited")


def batch_add_references(prim: Usd.Prim, reference_paths: List[str], position=Usd.ListPositionBackOfAppendList) -> bool:
    """
    Batch add a list of references to the specified prim.

    Args:
        prim (Usd.Prim): The prim to add the references to.
        reference_paths (List[str]): A list of reference paths to add.
        position (Usd.ListPosition): The list position to add the references at.
            Defaults to Usd.ListPositionBackOfAppendList.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not prim.IsValid():
        return False
    references = prim.GetReferences()
    for reference_path in reference_paths:
        reference = Sdf.Reference(reference_path)
        if references.AddReference(reference, position):
            position = Usd.ListPositionBackOfAppendList
        else:
            return False
    return True


def set_motion_paths_for_prims(
    stage: Usd.Stage, prim_paths: list[str], start_time: float, end_time: float, translation_only: bool = True
):
    """Set motion paths for a list of prims between a start and end time.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (list[str]): A list of prim paths to set motion paths for.
        start_time (float): The start time of the motion path range.
        end_time (float): The end time of the motion path range.
        translation_only (bool, optional): If True, only set translation motion path, otherwise also set rotation and scale. Defaults to True.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        translate_op = add_translate_op(xformable)
        translate_op.Set(time=start_time, value=Gf.Vec3d(0, 0, 0))
        translate_op.Set(time=end_time, value=Gf.Vec3d(10, 20, 30))
        if not translation_only:
            rotate_op = add_rotate_xyz_op(xformable)
            rotate_op.Set(time=start_time, value=Gf.Vec3f(0, 0, 0))
            rotate_op.Set(time=end_time, value=Gf.Vec3f(45, 90, 180))
            scale_op = add_scale_op(xformable)
            scale_op.Set(time=start_time, value=Gf.Vec3f(1, 1, 1))
            scale_op.Set(time=end_time, value=Gf.Vec3f(2, 3, 4))


def validate_and_fix_prim_paths(prim_paths: List[str]) -> List[str]:
    """
    Validate and fix a list of prim paths.

    Args:
        prim_paths (List[str]): A list of prim paths.

    Returns:
        List[str]: A list of validated and fixed prim paths.
    """
    fixed_paths = []
    for path in prim_paths:
        if not path:
            raise ValueError("Prim path cannot be empty.")
        if not path.startswith("/"):
            path = "/" + path
        segments = path.split("/")
        segments = [seg for seg in segments if seg]
        for segment in segments[1:]:
            if not segment[0].isalpha():
                raise ValueError(f"Prim path segment '{segment}' in '{path}' must start with a letter.")
        fixed_path = "/".join(segments)
        fixed_paths.append(fixed_path)
    return fixed_paths


def analyze_and_optimize_hierarchy(prim: Usd.Prim) -> bool:
    """Analyze and optimize the hierarchy of the given prim.

    This function recursively traverses the prim hierarchy, identifies
    and removes unnecessary intermediate grouping prims, and returns
    True if any changes were made to the hierarchy.

    Args:
        prim (Usd.Prim): The root prim to start the analysis from.

    Returns:
        bool: True if any changes were made to the hierarchy, False otherwise.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    hierarchy_changed = False
    for child in prim.GetChildren():
        if analyze_and_optimize_hierarchy(child):
            hierarchy_changed = True
    if prim.IsA(UsdGeom.Xform) and (not prim.GetChildren()):
        if not prim.GetAuthoredProperties() and (not prim.GetRelationships()):
            prim.SetActive(False)
            hierarchy_changed = True
    return hierarchy_changed


def collect_all_variant_sets(prim: Usd.Prim) -> Dict[str, Usd.VariantSet]:
    """Collect all variant sets on the given prim and its ancestors recursively.

    Args:
        prim (Usd.Prim): The prim to start collecting variant sets from.

    Returns:
        Dict[str, Usd.VariantSet]: A dictionary mapping variant set names to
        variant sets that were found on the prim or its ancestors.
    """
    variant_sets = {}
    current_prim = prim
    while current_prim.IsValid():
        prim_variant_sets = current_prim.GetVariantSets()
        for variant_set_name in prim_variant_sets.GetNames():
            if variant_set_name not in variant_sets:
                variant_set = prim_variant_sets.GetVariantSet(variant_set_name)
                variant_sets[variant_set_name] = variant_set
        current_prim = current_prim.GetParent()
    return variant_sets


def synchronize_prim_inherits(prim: Usd.Prim, inherits: List[Sdf.Path]) -> None:
    """Synchronize the inherits for a prim with the given list of paths."""
    prim_inherits = prim.GetInherits()
    current_inherits = prim_inherits.GetAllDirectInherits()
    inherits_set = set(inherits)
    for inherit in current_inherits:
        if str(inherit) not in inherits_set:
            prim_inherits.RemoveInherit(inherit)
    for inherit in inherits:
        if inherit not in current_inherits:
            prim_inherits.AddInherit(inherit)


def create_hierarchy_from_graph(stage: Usd.Stage, graph: Dict[Sdf.Path, List[str]]) -> None:
    """Create a prim hierarchy from a graph represented as a dictionary.

    Args:
        stage (Usd.Stage): The USD stage to create the prims on.
        graph (Dict[Sdf.Path, List[str]]): A dictionary representing the graph. Keys are prim paths,
            and values are lists of child prim names.
    """
    for prim_path, child_names in graph.items():
        if not prim_path.IsAbsoluteRootOrPrimPath():
            raise ValueError(f"Invalid prim path: {prim_path}")
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            prim = stage.DefinePrim(prim_path)
        for child_name in child_names:
            child_path = prim_path.AppendChild(child_name)
            if not stage.GetPrimAtPath(child_path):
                stage.DefinePrim(child_path)


def create_animated_visibility(prim: Usd.Prim, time_samples: Dict[float, UsdGeom.Tokens.visibility]) -> None:
    """Create animated visibility on a prim.

    Args:
        prim (Usd.Prim): The prim to create animated visibility on.
        time_samples (Dict[float, UsdGeom.Tokens.visibility]): A dictionary of time samples and their visibility values.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    visibility_attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
    for time, value in time_samples.items():
        visibility_attr.Set(value, Usd.TimeCode(time))


def generate_uv_variants(
    stage: Usd.Stage,
    prim_path: str,
    variant_set_name: str,
    variant_names: List[str],
    uv_ranges: List[Tuple[float, float, float, float]],
) -> None:
    """Generate UV variants for a mesh prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the mesh prim.
        variant_set_name (str): The name of the variant set to create.
        variant_names (List[str]): The names of the variants to create.
        uv_ranges (List[Tuple[float, float, float, float]]): The UV ranges for each variant as a list of tuples (min_u, max_u, min_v, max_v).

    Raises:
        ValueError: If the prim is not a valid mesh prim or if the number of variant names and UV ranges do not match.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    if not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"Prim at path {prim_path} is not a mesh.")
    if len(variant_names) != len(uv_ranges):
        raise ValueError("Number of variant names and UV ranges do not match.")
    mesh = UsdGeom.Mesh(prim)
    variant_set = prim.GetVariantSets().AddVariantSet(variant_set_name)
    for variant_name, uv_range in zip(variant_names, uv_ranges):
        variant_set.SetVariantSelection(variant_name)
        with variant_set.GetVariantEditContext():
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
            ).Set(
                Vt.Vec2fArray(
                    [
                        (uv_range[0], uv_range[2]),
                        (uv_range[1], uv_range[2]),
                        (uv_range[1], uv_range[3]),
                        (uv_range[0], uv_range[3]),
                    ]
                )
            )


def create_geometry_variants_from_geometry(
    stage: Usd.Stage, root_prim_path: str, variant_set_name: str, geometry_paths: List[str]
) -> None:
    """Create a variant set on the root prim with variants for each geometry path."""
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim.IsValid():
        raise ValueError(f"Invalid root prim path: {root_prim_path}")
    variant_set = root_prim.GetVariantSets().AddVariantSet(variant_set_name)
    for geometry_path in geometry_paths:
        geometry_prim = stage.GetPrimAtPath(geometry_path)
        if not geometry_prim.IsValid():
            raise ValueError(f"Invalid geometry prim path: {geometry_path}")
        geometry_name = geometry_prim.GetName()
        variant = variant_set.AddVariant(geometry_name)
        variant_set.SetVariantSelection(geometry_name)
        with variant_set.GetVariantEditContext():
            UsdGeom.Imageable(geometry_prim).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            for other_geometry_path in geometry_paths:
                if other_geometry_path != geometry_path:
                    other_geometry_prim = stage.GetPrimAtPath(other_geometry_path)
                    UsdGeom.Imageable(other_geometry_prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)


def batch_remove_relationships(prim: Usd.Prim, relationship_names: List[str]) -> bool:
    """
    Remove multiple relationships from a prim in a single operation.

    Args:
        prim (Usd.Prim): The prim to remove relationships from.
        relationship_names (List[str]): A list of relationship names to remove.

    Returns:
        bool: True if all relationships were successfully removed, False otherwise.
    """
    if not prim.IsValid():
        return False
    success = True
    for rel_name in relationship_names:
        if prim.HasRelationship(rel_name):
            rel = prim.GetRelationship(rel_name)
            if not prim.RemoveProperty(rel_name):
                success = False
        else:
            success = False
    return success


def create_hierarchy_from_network(stage: Usd.Stage, network: Dict[str, List[str]]) -> None:
    """Create a hierarchy of prims in the given stage based on the provided network.

    Args:
        stage (Usd.Stage): The stage to create the prims in.
        network (Dict[str, List[str]]): A dictionary representing the network of prims.
            Each key is a prim path, and the corresponding value is a list of its child prim paths.
    """
    if not stage:
        raise ValueError("Invalid stage provided.")
    for prim_path, children in network.items():
        prim = stage.DefinePrim(prim_path)
        if prim_path != "/":
            prim.SetTypeName("Xform")
        for child_path in children:
            child_prim = stage.DefinePrim(child_path)
            child_prim.SetTypeName("Xform")


def create_animation_from_transforms(stage: Usd.Stage, prim_paths: list[str], time_samples: list[float]) -> bool:
    """Create animation for a list of prims based on their local transforms at different time samples.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (list[str]): The paths of the prims to animate.
        time_samples (list[float]): The time samples for the animation.

    Returns:
        bool: True if the animation was created successfully, False otherwise.
    """
    if not prim_paths or not time_samples:
        return False
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            continue
        for time in time_samples:
            transform = xformable.GetLocalTransformation(Usd.TimeCode(time))
            xform_ops = xformable.GetOrderedXformOps()
            for op in xform_ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
                    op.Set(transform, Usd.TimeCode(time))
                    break
    return True


def set_world_transform(
    stage: Usd.Stage,
    prim_path: str,
    translation: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    scale: Tuple[float, float, float],
) -> None:
    """Set the world transform for a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    xformable.ClearXformOpOrder()
    translate_op = add_translate_op(xformable)
    translate_op.Set(Gf.Vec3d(translation))
    rotate_op = add_rotate_xyz_op(xformable)
    rotate_op.Set(Gf.Vec3d(rotation))
    scale_op = add_scale_op(xformable)
    scale_op.Set(Gf.Vec3d(scale))


def convert_prim_to_mesh(prim: Usd.Prim) -> bool:
    """Convert a prim to a mesh.

    Args:
        prim (Usd.Prim): The prim to convert.

    Returns:
        bool: True if the conversion was successful, False otherwise.
    """
    if not prim.IsValid():
        return False
    if prim.IsA(UsdGeom.Mesh):
        return True
    if not prim.IsA(UsdGeom.Xformable):
        return False
    mesh = UsdGeom.Mesh.Define(prim.GetStage(), prim.GetPath())
    for attr in prim.GetAttributes():
        if attr.IsAuthored():
            attr_name = attr.GetName()
            if mesh.GetPrim().HasAttribute(attr_name):
                mesh_attr = mesh.GetPrim().GetAttribute(attr_name)
                mesh_attr.Set(attr.Get())
    for rel in prim.GetRelationships():
        if rel.IsAuthored():
            rel_name = rel.GetName()
            if mesh.GetPrim().HasRelationship(rel_name):
                mesh_rel = mesh.GetPrim().GetRelationship(rel_name)
                for target in rel.GetTargets():
                    mesh_rel.AddTarget(target)
    prim.GetStage().RemovePrim(prim.GetPath())
    return True


def check_and_fix_prim_attribute_conflicts(prim: Usd.Prim):
    """Check and fix attribute conflicts on a prim.

    This function checks for attributes with the same base name but different
    namespaces on the given prim. If any conflicts are found, it will create
    new attributes with the namespace included in the name and remove the
    old conflicting attributes.

    Args:
        prim (Usd.Prim): The prim to check for attribute conflicts.
    """
    attrs = prim.GetAttributes()
    attr_dict = {}
    for attr in attrs:
        base_name = attr.GetBaseName()
        if base_name not in attr_dict:
            attr_dict[base_name] = []
        attr_dict[base_name].append(attr)
    for base_name, attr_list in attr_dict.items():
        if len(attr_list) > 1:
            for attr in attr_list:
                namespace = attr.GetNamespace()
                if namespace:
                    new_name = f"{namespace}:{base_name}"
                else:
                    new_name = f"_{base_name}"
                value = attr.Get()
                typename = attr.GetTypeName()
                new_attr = prim.CreateAttribute(new_name, typename)
                new_attr.Set(value)
                prim.RemoveProperty(attr.GetName())


def align_prims_to_surface(stage: Usd.Stage, surface_path: str, prims_to_align: List[str]) -> None:
    """Align the given prims to the surface prim's top plane.

    Args:
        stage (Usd.Stage): The USD stage.
        surface_path (str): The path to the surface prim.
        prims_to_align (List[str]): The paths of the prims to align.

    Raises:
        ValueError: If the surface prim or any of the prims to align are not valid.
    """
    surface_prim = stage.GetPrimAtPath(surface_path)
    if not surface_prim.IsValid():
        raise ValueError(f"Surface prim at path {surface_path} does not exist.")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    surface_bbox = bbox_cache.ComputeWorldBound(surface_prim)
    surface_range = surface_bbox.ComputeAlignedRange()
    top_center_pos = Gf.Vec3d(
        (surface_range.min[0] + surface_range.max[0]) / 2,
        surface_range.max[1],
        (surface_range.min[2] + surface_range.max[2]) / 2,
    )
    for prim_path in prims_to_align:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        prim_bbox = bbox_cache.ComputeWorldBound(prim)
        prim_range = prim_bbox.ComputeAlignedRange()
        prim_center_pos = Gf.Vec3d(
            (prim_range.min[0] + prim_range.max[0]) / 2,
            (prim_range.min[1] + prim_range.max[1]) / 2,
            (prim_range.min[2] + prim_range.max[2]) / 2,
        )
        translation = top_center_pos - prim_center_pos
        translation[1] -= prim_range.max[1] - prim_center_pos[1]
        UsdGeom.XformCommonAPI(prim).SetTranslate(translation)


def convert_hierarchy_to_payloads(stage: Usd.Stage, prim_path: str, payload_layer: Sdf.Layer):
    """Converts the hierarchy starting at prim_path into a payload reference.

    Args:
        stage (Usd.Stage): The stage containing the hierarchy to convert.
        prim_path (str): The path to the root prim of the hierarchy to convert.
        payload_layer (Sdf.Layer): The layer to author the payload prims into.

    Raises:
        ValueError: If the prim at prim_path does not exist.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    payload_prim = Sdf.CreatePrimInLayer(payload_layer, prim_path)
    Sdf.CopySpec(prim.GetPrimStack()[0].layer, prim.GetPrimStack()[0].path, payload_layer, payload_prim.path)
    for child in prim.GetAllChildren():
        convert_hierarchy_to_payloads(stage, child.GetPath(), payload_layer)
    payload_asset_path = payload_layer.identifier
    payload_prim_path = payload_prim.path
    prim.GetPayloads().AddPayload(payload_asset_path, payload_prim_path)
    source_prim = stage.GetEditTarget().GetPrimSpecForScenePath(prim_path)
    source_prim.nameChildren.clear()


def create_animation_from_attributes(
    stage: Usd.Stage, prim_path: str, attribute_names: List[str], time_range: Tuple[float, float], time_samples: int
):
    """Create animation for a prim based on a list of attributes.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to animate.
        attribute_names (List[str]): The names of the attributes to use for animation.
        time_range (Tuple[float, float]): The start and end times for the animation.
        time_samples (int): The number of time samples to generate.

    Returns:
        None
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    attributes = [prim.GetAttribute(attr_name) for attr_name in attribute_names]
    if not all(attributes):
        invalid_attrs = [attr_name for (attr_name, attr) in zip(attribute_names, attributes) if not attr]
        raise ValueError(f"Invalid attributes: {invalid_attrs}")
    start_values = [attr.Get(time_range[0]) for attr in attributes]
    end_values = [attr.Get(time_range[1]) for attr in attributes]
    time_codes = [Usd.TimeCode(t) for t in range(time_samples)]
    time_step = (time_range[1] - time_range[0]) / (time_samples - 1)
    for i, time_code in enumerate(time_codes):
        t = i / (time_samples - 1)
        for attr, start_value, end_value in zip(attributes, start_values, end_values):
            interpolated_value = start_value + (end_value - start_value) * t
            attr.Set(interpolated_value, time_code)


def convert_prim_to_subdivision_surface(prim: Usd.Prim) -> bool:
    """Convert a prim to a subdivision surface."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        if mesh.GetSubdivisionSchemeAttr().Get() != "none":
            return True
    valid_types = (
        UsdGeom.Mesh,
        UsdGeom.Cube,
        UsdGeom.Sphere,
        UsdGeom.Cylinder,
        UsdGeom.Cone,
        UsdGeom.Capsule,
        UsdSkel.Skeleton,
        UsdShade.Material,
    )
    if not any((prim.IsA(t) for t in valid_types)):
        raise ValueError(f"Prim {prim.GetPath()} is not a valid type for conversion to a subdivision surface.")
    prim_type = prim.GetTypeName()
    mesh = UsdGeom.Mesh.Define(prim.GetStage(), prim.GetPath())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.catmullClark)
    mesh.CreateInterpolateBoundaryAttr(UsdGeom.Tokens.edgeAndCorner)
    mesh.CreateFaceVaryingLinearInterpolationAttr(UsdGeom.Tokens.boundaries)
    attrs_to_transfer = ["extent", "doubleSided", "purpose", "visibility"]
    for attr_name in attrs_to_transfer:
        old_attr = prim.GetAttribute(attr_name)
        if old_attr.HasValue():
            new_attr = mesh.GetPrim().CreateAttribute(attr_name, old_attr.GetTypeName())
            new_attr.Set(old_attr.Get())
    prim.SetActive(False)
    return True


def create_inherited_variants(
    stage: Usd.Stage, prim_path: str, variant_set_name: str, variants: Dict[str, Dict[str, Any]]
) -> None:
    """Create inherited variants for a prim.

    Args:
        stage (Usd.Stage): The stage to create the variants on.
        prim_path (str): The path to the prim to create the variants on.
        variant_set_name (str): The name of the variant set to create.
        variants (Dict[str, Dict[str, Any]]): A dictionary of variant names and their attributes.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path {prim_path}")
    vset = prim.GetVariantSets().AddVariantSet(variant_set_name)
    for variant_name, attributes in variants.items():
        vset.AddVariant(variant_name)
        vset.SetVariantSelection(variant_name)
        with vset.GetVariantEditContext():
            for attr_name, attr_value in attributes.items():
                attr_type = type(attr_value)
                if attr_type == float:
                    attr_type_name = Sdf.ValueTypeNames.Float
                elif attr_type == int:
                    attr_type_name = Sdf.ValueTypeNames.Int
                elif attr_type == bool:
                    attr_type_name = Sdf.ValueTypeNames.Bool
                elif attr_type == str:
                    attr_type_name = Sdf.ValueTypeNames.String
                else:
                    raise ValueError(f"Unsupported attribute type: {attr_type}")
                attr = prim.CreateAttribute(attr_name, attr_type_name)
                attr.Set(attr_value)


def convert_group_to_model(prim: Usd.Prim) -> bool:
    """Convert a group prim to a model prim.

    Args:
        prim (Usd.Prim): The group prim to convert.

    Returns:
        bool: True if the conversion was successful, False otherwise.
    """
    if not prim.IsValid():
        return False
    if Usd.ModelAPI(prim).IsModel():
        return True
    if not Usd.ModelAPI(prim).IsGroup():
        return False
    prim.SetMetadata("kind", "model")
    return True


def consolidate_duplicate_materials(stage: Usd.Stage) -> int:
    """Consolidate duplicate materials on the given stage.

    Returns the number of duplicate materials removed.
    """
    materials = [p for p in stage.Traverse() if p.IsA(UsdShade.Material)]
    unique_materials = {}
    num_removed = 0
    for material in materials:
        attrs_key = tuple(sorted(((attr.GetName(), attr.Get()) for attr in material.GetAttributes())))
        if attrs_key in unique_materials:
            unique_material = unique_materials[attrs_key]
            for prim in stage.TraverseAll():
                for rel in prim.GetRelationships():
                    targets = rel.GetTargets()
                    if material.GetPath() in targets:
                        targets.remove(material.GetPath())
                        targets.append(unique_material.GetPath())
                        rel.SetTargets(targets)
            material.GetPrim().SetActive(False)
            stage.RemovePrim(material.GetPath())
            num_removed += 1
        else:
            unique_materials[attrs_key] = material
    return num_removed


def filter_prims_by_metadata(stage: Usd.Stage, key: str, value: str) -> List[Usd.Prim]:
    """
    Filter prims on the given stage by metadata key and value.

    Args:
        stage (Usd.Stage): The USD stage to filter prims from.
        key (str): The metadata key to filter by.
        value (str): The expected metadata value to match.

    Returns:
        List[Usd.Prim]: A list of prims that have the specified metadata key and value.
    """
    result = []
    for prim in stage.Traverse():
        if prim.HasMetadata(key):
            metadata_value = prim.GetMetadata(key)
            if metadata_value == value:
                result.append(prim)
    return result


def filter_prims_by_custom_data(stage: Usd.Stage, custom_data: Dict[str, Any]) -> List[Usd.Prim]:
    """Filter prims in a stage by custom data.

    Args:
        stage (Usd.Stage): The stage to filter prims from.
        custom_data (Dict[str, Any]): The custom data to filter prims by.
            The keys are the custom data keys and the values are the corresponding values to match.

    Returns:
        List[Usd.Prim]: A list of prims that match the given custom data.
    """
    prims = [prim for prim in stage.TraverseAll()]
    filtered_prims = []
    for prim in prims:
        if all(
            (
                prim.HasCustomDataKey(key) and prim.GetCustomDataByKey(key) == value
                for (key, value) in custom_data.items()
            )
        ):
            filtered_prims.append(prim)
    return filtered_prims


def create_prim_with_custom_data(stage, prim_path, custom_data):
    prim = stage.DefinePrim(prim_path)
    for key, value in custom_data.items():
        prim.SetCustomDataByKey(key, value)
    return prim


def filter_prims_by_asset_info(stage: Usd.Stage, key: str, value: str) -> List[Usd.Prim]:
    """Filter prims on the given stage by presence of a specific asset info key-value pair.

    Args:
        stage (Usd.Stage): The stage to search for prims.
        key (str): The asset info key to match.
        value (str): The asset info value to match.

    Returns:
        List[Usd.Prim]: A list of prims that contain the specified asset info key-value pair.
    """
    matching_prims: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        if prim.HasAssetInfoKey(key):
            prim_value = prim.GetAssetInfoByKey(key)
            if prim_value == value:
                matching_prims.append(prim)
    return matching_prims


def transfer_custom_data(source_prim: Usd.Prim, dest_prim: Usd.Prim):
    """Transfer custom data from one prim to another."""
    if not source_prim.IsValid():
        raise ValueError("Source prim is not valid.")
    if not dest_prim.IsValid():
        raise ValueError("Destination prim is not valid.")
    source_custom_data = source_prim.GetCustomData()
    if not source_custom_data:
        return
    edit_target = source_prim.GetStage().GetEditTarget()
    for key, value in source_custom_data.items():
        dest_prim.SetCustomDataByKey(key, value)
    if edit_target != dest_prim.GetStage().GetEditTarget():
        with Usd.EditContext(dest_prim.GetStage(), edit_target):
            for key, value in source_custom_data.items():
                dest_prim.SetCustomDataByKey(key, value)


def transfer_asset_info(source_prim: Usd.Prim, dest_prim: Usd.Prim) -> None:
    """Transfer the asset info from one prim to another."""
    if not source_prim.IsValid():
        raise ValueError("Source prim is not valid.")
    if not dest_prim.IsValid():
        raise ValueError("Destination prim is not valid.")
    source_asset_info = source_prim.GetAssetInfo()
    if not source_asset_info:
        return
    edit_target = dest_prim.GetStage().GetEditTarget()
    dest_prim.ClearAssetInfo()
    for key, value in source_asset_info.items():
        dest_prim.SetAssetInfoByKey(key, value)
    dest_prim.GetStage().Save()


def get_all_custom_data_keys(obj: Usd.Object) -> List[str]:
    """
    Get all the keys in the customData dictionary of a USD object.

    Args:
        obj (Usd.Object): The USD object to get the customData keys from.

    Returns:
        List[str]: A list of all keys in the customData dictionary.
    """
    custom_data = obj.GetCustomData()
    if not custom_data:
        return []
    keys = []

    def collect_keys(data, prefix=""):
        for key, value in data.items():
            full_key = prefix + key
            keys.append(full_key)
            if isinstance(value, dict):
                collect_keys(value, full_key + ":")

    collect_keys(custom_data)
    return keys


def get_custom_data_value(obj: Usd.Object, key: str, keyPath: str) -> Any:
    """Get the value of a custom data field identified by key and keyPath.

    Args:
        obj (Usd.Object): The object to retrieve custom data from.
        key (str): The key identifying the custom data dictionary.
        keyPath (str): The ':'-separated path to the value in nested dictionaries.

    Returns:
        Any: The value of the custom data field or None if not found.
    """
    customData = obj.GetCustomData()
    if key not in customData:
        return None
    subDict = customData[key]
    keys = keyPath.split(":")
    for k in keys:
        if k in subDict:
            subDict = subDict[k]
        else:
            return None
    return subDict


def clear_all_custom_data(obj: Usd.Object) -> None:
    """Clear all custom data authored on the object."""
    if not obj.IsValid():
        raise ValueError("Invalid USD object.")
    custom_data = obj.GetCustomData()
    if not custom_data:
        return
    obj.ClearCustomData()


def get_all_documentation(obj: Usd.Object) -> Dict[str, str]:
    """Get all documentation metadata for a Usd.Object and its properties."""
    docs: Dict[str, str] = {}
    if not obj.IsValid():
        return docs
    obj_doc = obj.GetDocumentation()
    if obj_doc:
        docs[""] = obj_doc
    if isinstance(obj, Usd.Prim):
        for prop in obj.GetProperties():
            prop_name = prop.GetName()
            prop_doc = prop.GetDocumentation()
            if prop_doc:
                docs[prop_name] = prop_doc
    return docs


def synchronize_asset_info_across_prims(stage: Usd.Stage, source_prim_path: str, target_prim_paths: List[str]) -> None:
    """
    Synchronize the assetInfo metadata from a source prim to a list of target prims.

    The assetInfo from the source prim will be copied to each target prim, overwriting any existing assetInfo.

    Parameters:
        stage (Usd.Stage): The stage containing the prims.
        source_prim_path (str): The path of the prim to copy the assetInfo from.
        target_prim_paths (List[str]): A list of prim paths to copy the assetInfo to.

    Raises:
        ValueError: If the source prim or any target prim does not exist.
    """
    source_prim = stage.GetPrimAtPath(source_prim_path)
    if not source_prim.IsValid():
        raise ValueError(f"Source prim at path {source_prim_path} does not exist.")
    source_asset_info = source_prim.GetAssetInfo()
    if not source_asset_info:
        return
    for target_prim_path in target_prim_paths:
        target_prim = stage.GetPrimAtPath(target_prim_path)
        if not target_prim.IsValid():
            raise ValueError(f"Target prim at path {target_prim_path} does not exist.")
        target_prim.SetAssetInfo(source_asset_info)


def get_prim_visibility(prim: Usd.Prim) -> Usd.Tokens:
    """Get the visibility of a prim.

    Args:
        prim (Usd.Prim): The prim to get the visibility of.

    Returns:
        Usd.Tokens: The visibility of the prim (inherited, invisible).
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim")
    visibility_attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
    if visibility_attr.IsValid() and visibility_attr.HasAuthoredValue():
        visibility = visibility_attr.Get()
    else:
        visibility = UsdGeom.Tokens.inherited
    return visibility


def update_prim_documentation(prim: Usd.Prim, new_doc: str) -> bool:
    """Update the documentation metadata for a prim.

    Args:
        prim (Usd.Prim): The prim to update the documentation for.
        new_doc (str): The new documentation string to set.

    Returns:
        bool: True if the documentation was updated successfully, False otherwise.
    """
    if not prim.IsValid():
        return False
    current_doc = prim.GetDocumentation()
    if new_doc == current_doc:
        return True
    success = prim.SetDocumentation(new_doc)
    return success


def recursive_clear_custom_data(prim: Usd.Prim):
    """Recursively clear customData on the given prim and all its descendants."""
    prim.ClearCustomData()
    for child in prim.GetAllChildren():
        recursive_clear_custom_data(child)


def recursive_clear_asset_info(obj: Usd.Object):
    """Recursively clear the assetInfo dictionary for the given object and its descendants."""
    obj.ClearAssetInfo()
    if isinstance(obj, Usd.Prim):
        for child in obj.GetAllChildren():
            recursive_clear_asset_info(child)


def get_prim_description(prim: Usd.Prim) -> str:
    """Get the description for a prim."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    if prim.HasAuthoredDocumentation():
        return prim.GetDocumentation()
    else:
        path_str = prim.GetPath().pathString
        return f"No description for prim at {path_str}."


def set_prim_description(stage: Usd.Stage, prim_path: str, description: str) -> bool:
    """Set the description for a prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    try:
        prim.SetMetadata("documentation", description)
    except Tf.ErrorException as e:
        print(f"Error setting description: {e}")
        return False
    return True


def clear_hidden_metadata_for_all_prims(stage: Usd.Stage) -> None:
    """Clear the 'hidden' metadata for all prims on the given stage."""
    for prim in stage.TraverseAll():
        if prim.HasAuthoredHidden():
            success = prim.ClearHidden()
            if not success:
                raise RuntimeError(f"Failed to clear 'hidden' metadata for prim {prim.GetPath()}")


def transfer_custom_data_and_asset_info(source: Usd.Object, dest: Usd.Object):
    """Transfer all custom data and asset info from source to destination prim."""
    if not source.IsValid() or not dest.IsValid():
        raise ValueError("Source or destination object is invalid.")
    if source.HasCustomData():
        custom_data = source.GetCustomData()
        dest.SetCustomData(custom_data)
    if source.HasAssetInfo():
        asset_info = source.GetAssetInfo()
        dest.SetAssetInfo(asset_info)
    for key in source.GetCustomData().keys():
        if isinstance(source.GetCustomData()[key], dict):
            for subkey in source.GetCustomData()[key].keys():
                key_path = f"{key}:{subkey}"
                if source.HasCustomDataKey(key_path):
                    value = source.GetCustomDataByKey(key_path)
                    dest.SetCustomDataByKey(key_path, value)
    for key in source.GetAssetInfo().keys():
        if isinstance(source.GetAssetInfo()[key], dict):
            for subkey in source.GetAssetInfo()[key].keys():
                key_path = f"{key}:{subkey}"
                if source.HasAssetInfoKey(key_path):
                    value = source.GetAssetInfoByKey(key_path)
                    dest.SetAssetInfoByKey(key_path, value)


def copy_documentation_to_prim(source_prim: Usd.Prim, target_prim: Usd.Prim) -> bool:
    """Copy documentation from source_prim to target_prim."""
    if not source_prim.IsValid():
        raise ValueError("Source prim is not valid.")
    if not target_prim.IsValid():
        raise ValueError("Target prim is not valid.")
    source_doc = source_prim.GetDocumentation()
    if not source_doc:
        return False
    return target_prim.SetDocumentation(source_doc)


def merge_custom_data(obj: Usd.Object, custom_data: Dict[str, Any]):
    """Merge the given custom data dictionary into the object's custom data."""
    existing_custom_data = obj.GetCustomData()
    for key, value in custom_data.items():
        if key in existing_custom_data:
            if isinstance(existing_custom_data[key], dict) and isinstance(value, dict):
                existing_custom_data[key] = {**existing_custom_data[key], **value}
            else:
                existing_custom_data[key] = value
        else:
            existing_custom_data[key] = value
    obj.SetCustomData(existing_custom_data)


def set_asset_info_value(usd_object: Usd.Prim, key_path: str, value: Any):
    """Set a value in the asset info dictionary by key path.

    Args:
        usd_object (Usd.Prim): The USD prim to set the asset info on.
        key_path (str): The ':'-separated path to the asset info value.
        value (Any): The value to set for the specified key path.
    """
    keys = key_path.split(":")
    asset_info = usd_object.GetAssetInfo()
    current_dict = asset_info
    for key in keys[:-1]:
        if key not in current_dict:
            current_dict[key] = {}
        current_dict = current_dict[key]
    current_dict[keys[-1]] = value
    usd_object.SetAssetInfo(asset_info)


def get_all_prim_paths_with_metadata(stage: Usd.Stage, metadata_key: str) -> List[Sdf.Path]:
    """
    Return a list of prim paths that have the specified metadata key.

    Args:
        stage (Usd.Stage): The USD stage to search for prims with the metadata key.
        metadata_key (str): The metadata key to search for on prims.

    Returns:
        List[Sdf.Path]: A list of prim paths that have the specified metadata key.
    """
    prim_paths_with_metadata = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.HasMetadata(metadata_key):
            prim_paths_with_metadata.append(prim.GetPath())
    return prim_paths_with_metadata


def remove_custom_data_key(obj: Usd.Object, key_path: str) -> bool:
    """Remove a key from the customData dictionary of a Usd.Object.

    Args:
        obj (Usd.Object): The object to remove the customData key from.
        key_path (str): The ':'-separated path to the key to remove.

    Returns:
        bool: True if the key was successfully removed, False otherwise.
    """
    if not obj.HasCustomData():
        return False
    if not obj.HasCustomDataKey(key_path):
        return False
    obj.ClearCustomDataByKey(key_path)
    if obj.HasCustomDataKey(key_path):
        return False
    else:
        return True


def list_all_hidden_prims(stage: Usd.Stage) -> List[Usd.Prim]:
    """Return a list of all hidden prims in the stage."""
    hidden_prims = []
    for prim in stage.TraverseAll():
        if prim.HasAuthoredHidden():
            if prim.IsHidden():
                hidden_prims.append(prim)
    return hidden_prims


def list_all_prims_with_asset_info(stage: Usd.Stage) -> List[Usd.Prim]:
    """Return a list of all prims on the stage that have authored assetInfo."""
    prims_with_asset_info = []
    for prim in stage.TraverseAll():
        if prim.HasAuthoredAssetInfo():
            prims_with_asset_info.append(prim)
    return prims_with_asset_info


def set_custom_data_value(prim: Usd.Prim, key_path: str, value: Any) -> None:
    """Set a custom data value on a USD prim.

    Args:
        prim (Usd.Prim): The USD prim to set the custom data on.
        key_path (str): The key path to set the value at, using ':' as the separator.
        value (Any): The value to set at the specified key path.

    Raises:
        ValueError: If the USD prim is not valid.
    """
    if not prim.IsValid():
        raise ValueError("USD prim is not valid")
    keys = key_path.split(":")
    custom_data = prim.GetCustomData()
    data = custom_data
    for key in keys[:-1]:
        if key not in data:
            data[key] = {}
        data = data[key]
    data[keys[-1]] = value
    prim.SetCustomData(custom_data)


def merge_asset_info(object: Usd.Object, asset_info: dict) -> bool:
    """Merge the given asset info dictionary with the object's existing asset info.

    If a key already exists in the object's asset info, its value will be
    overwritten if the new value has a different type or is not empty.
    If the new value is of the same type as the existing value and is empty,
    the existing value will be preserved.

    Parameters:
        object (Usd.Object): The object to merge the asset info into.
        asset_info (dict): The asset info dictionary to merge.

    Returns:
        bool: True if the asset info was successfully merged, False otherwise.
    """
    existing_asset_info = object.GetAssetInfo()
    for key, value in asset_info.items():
        if key in existing_asset_info:
            existing_value = existing_asset_info[key]
            if type(value) != type(existing_value) or value:
                existing_asset_info[key] = value
        else:
            existing_asset_info[key] = value
    object.SetAssetInfo(existing_asset_info)
    return True


def transfer_metadata(source_prim: Usd.Prim, dest_prim: Usd.Prim) -> None:
    """Transfer all metadata from the source prim to the destination prim."""
    if not source_prim.IsValid():
        raise ValueError("Source prim is not valid.")
    if not dest_prim.IsValid():
        raise ValueError("Destination prim is not valid.")
    metadata = source_prim.GetAllAuthoredMetadata()
    for key, value in metadata.items():
        if Tf.Type.FindByName(key):
            metadata_type = Tf.Type.FindByName(key).GetTypeName()
            if isinstance(value, eval(metadata_type)):
                success = dest_prim.SetMetadata(key, value)
                if not success:
                    print(f"Failed to set metadata key '{key}' on destination prim.")
            else:
                print(f"Skipping metadata key '{key}' with mismatched value type.")
        else:
            print(f"Skipping unregistered metadata key: '{key}'")


def copy_metadata(src_prim: Usd.Prim, dest_prim: Usd.Prim, skip_keys: Optional[Tuple[str, ...]] = None):
    """Copy metadata from one prim to another.

    Args:
        src_prim (Usd.Prim): The source prim to copy metadata from.
        dest_prim (Usd.Prim): The destination prim to copy metadata to.
        skip_keys (Optional[Tuple[str, ...]]): Optional tuple of metadata keys to skip copying. Defaults to None.

    Returns:
        None
    """
    if not src_prim.IsValid():
        raise ValueError("Source prim is not valid.")
    if not dest_prim.IsValid():
        raise ValueError("Destination prim is not valid.")
    src_metadata = src_prim.GetAllAuthoredMetadata()
    for key, value in src_metadata.items():
        if skip_keys and key in skip_keys:
            continue
        try:
            success = dest_prim.SetMetadata(key, value)
            if not success:
                raise ValueError(f"Failed to set metadata key '{key}' on destination prim.")
        except Tf.ErrorException as e:
            print(f"Error setting metadata key '{key}': {str(e)}")


def transfer_all_metadata(src_object: Usd.Object, dest_object: Usd.Object) -> bool:
    """
    Transfer all authored metadata from one object to another.

    Args:
        src_object (Usd.Object): The source object to transfer metadata from.
        dest_object (Usd.Object): The destination object to transfer metadata to.

    Returns:
        bool: True if the metadata was successfully transferred, False otherwise.
    """
    if not src_object.IsValid() or not dest_object.IsValid():
        return False
    metadata_fields = src_object.GetAllAuthoredMetadata()
    for key, value in metadata_fields.items():
        if Usd.SchemaRegistry.IsDisallowedField(key):
            continue
        if dest_object.HasAuthoredMetadata(key):
            dest_object.ClearMetadata(key)
        success = dest_object.SetMetadata(key, value)
        if not success:
            return False
    return True


def get_prim_metadata_value(prim: Usd.Prim, key: str, default=None) -> Any:
    """
    Get the metadata value for a given key on a prim.

    Args:
        prim (Usd.Prim): The prim to retrieve the metadata from.
        key (str): The key of the metadata to retrieve.
        default (Any, optional): The default value to return if the metadata is not found. Defaults to None.

    Returns:
        Any: The value of the metadata or the default value if not found.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if not prim.HasMetadata(key):
        return default
    try:
        value = prim.GetMetadata(key)
    except Tf.ErrorException:
        return default
    if not value:
        return default
    return value


def list_all_prims_with_custom_data(stage: Usd.Stage) -> list:
    """Return a list of prims on the stage that have custom data."""
    prims_with_custom_data = []
    for prim in stage.TraverseAll():
        if prim.HasAuthoredCustomData():
            prims_with_custom_data.append(prim)
    return prims_with_custom_data


def get_asset_info_value(usd_object: Usd.Object, key_path: str):
    """Get a value from the assetInfo dictionary of a USD object.

    Args:
        usd_object (Usd.Object): The USD object to retrieve assetInfo from.
        key_path (str): The ':'-separated path to the desired value in the assetInfo dictionary.

    Returns:
        The value at the specified key path in the assetInfo dictionary.

    Raises:
        ValueError: If the USD object is invalid or the key path is invalid.
    """
    if not usd_object.IsValid():
        raise ValueError("Invalid USD object")
    asset_info = usd_object.GetAssetInfo()
    if not key_path:
        return asset_info
    keys = key_path.split(":")
    current_dict = asset_info
    for key in keys:
        if key not in current_dict:
            raise ValueError(f"Key '{key}' not found in assetInfo dictionary")
        current_dict = current_dict[key]
    return current_dict


def synchronize_custom_data_across_prims(prim_1: Usd.Prim, prim_2: Usd.Prim) -> bool:
    """
    Synchronize the custom data from prim_1 to prim_2.

    This function copies all custom data fields from prim_1 to prim_2. If a field
    already exists on prim_2, it will be overwritten with the value from prim_1.

    Args:
        prim_1 (Usd.Prim): The source prim to copy custom data from.
        prim_2 (Usd.Prim): The destination prim to copy custom data to.

    Returns:
        bool: True if the synchronization was successful, False otherwise.
    """
    if not prim_1.IsValid() or not prim_2.IsValid():
        return False
    custom_data_1 = prim_1.GetCustomData()
    custom_data_2 = prim_2.GetCustomData()
    for key, value in custom_data_1.items():
        if isinstance(value, str):
            value = Sdf.AssetPath(value)
        elif isinstance(value, list) and all((isinstance(item, str) for item in value)):
            value = Vt.StringArray(value)
        try:
            prim_2.SetCustomDataByKey(key, value)
        except ValueError as e:
            print(f"Error setting custom data for key '{key}': {str(e)}")
            return False
    updated_custom_data_2 = prim_2.GetCustomData()
    if custom_data_1 != updated_custom_data_2:
        return False
    return True


def bulk_set_custom_data(usd_object: Usd.Object, custom_data: Dict[str, Any]):
    """Bulk set custom data on a USD object.

    Args:
        usd_object (Usd.Object): The USD object to set custom data on.
        custom_data (Dict[str, Any]): A dictionary of custom data to set.
            Nested dictionaries are supported.
    """
    for key, value in custom_data.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                key_path = f"{key}:{nested_key}"
                usd_object.SetCustomDataByKey(key_path, nested_value)
        else:
            usd_object.SetCustomDataByKey(key, value)


def bulk_set_asset_info(prim: Usd.Prim, asset_info: Dict[str, Sdf.AssetPath]) -> None:
    """Bulk set multiple entries in a prim's assetInfo dictionary.

    Args:
        prim (Usd.Prim): The prim to set asset info on.
        asset_info (Dict[str, Sdf.AssetPath]): The asset info dictionary to set.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    current_asset_info = prim.GetAssetInfo()
    for key, value in asset_info.items():
        if not isinstance(value, Sdf.AssetPath):
            raise TypeError(f"Value for key {key} is not an Sdf.AssetPath.")
        current_asset_info[key] = value
    prim.SetAssetInfo(current_asset_info)


def get_all_authored_metadata(obj: Usd.Object) -> Dict[str, Sdf.ValueBlock]:
    """Get all authored metadata for a USD object.

    Args:
        obj (Usd.Object): The USD object to get metadata for.

    Returns:
        Dict[str, Sdf.ValueBlock]: A dictionary containing all authored metadata key-value pairs.
    """
    prim = obj.GetPrim()
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    metadata_dict = prim.GetAllMetadata()
    authored_metadata = {}
    for key, value in metadata_dict.items():
        if prim.HasAuthoredMetadata(key):
            authored_metadata[key] = value
    return authored_metadata


def set_prim_metadata_value(prim: Usd.Prim, key: str, value: Any) -> None:
    """Set a metadata value on a prim.

    Args:
        prim (Usd.Prim): The prim to set the metadata on.
        key (str): The metadata key.
        value (Any): The value to set for the metadata key.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid")
    try:
        success = prim.SetMetadata(key, value)
        if not success:
            raise ValueError(f"Failed to set metadata {key} to {value} on prim {prim.GetPath()}")
    except Tf.ErrorException as e:
        if "Unregistered metadata key" in str(e):
            print(f"Warning: {str(e)}")
        else:
            raise ValueError(f"Failed to set metadata {key} to {value} on prim {prim.GetPath()}: {str(e)}")


def set_clip_asset_paths_for_prim(prim: Usd.Prim, clip_set_name: str, asset_paths: List[str]) -> bool:
    """Set the clip asset paths for the specified clip set on the given prim."""
    clips_api = Usd.ClipsAPI(prim)
    if not clips_api:
        raise ValueError(f"Prim at path {prim.GetPath()} does not have a valid ClipsAPI.")
    clips_dict = clips_api.GetClips()
    if clip_set_name in clips_dict:
        clips_dict[clip_set_name]["assetPaths"] = Vt.StringArray(asset_paths)
    else:
        clips_dict[clip_set_name] = {
            "assetPaths": Vt.StringArray(asset_paths),
            "primPath": prim.GetPath().pathString,
            "times": Vt.Vec2dArray([(0.0, 0.0)]),
        }
    success = clips_api.SetClips(clips_dict)
    return success


def generate_and_set_clip_manifest(prim: Usd.Prim, clip_layers: List[Sdf.Layer], clip_prim_path: str) -> bool:
    """
    Generate a clip manifest from the given clip layers and set it on the prim.

    Args:
        prim (Usd.Prim): The prim to set the clip manifest on.
        clip_layers (List[Sdf.Layer]): The list of clip layers to generate the manifest from.
        clip_prim_path (str): The path to the prim in the clip layers to generate the manifest for.

    Returns:
        bool: True if the manifest was successfully generated and set, False otherwise.
    """
    if not prim.IsValid():
        print(f"Error: Invalid prim '{prim.GetPath()}'")
        return False
    if not Sdf.Path.IsValidPathString(clip_prim_path):
        print(f"Error: Invalid clip prim path '{clip_prim_path}'")
        return False
    clip_manifest = Usd.ClipsAPI.GenerateClipManifestFromLayers(clip_layers, clip_prim_path)
    if not clip_manifest:
        print("Error: Failed to generate clip manifest")
        return False
    clips_api = Usd.ClipsAPI(prim)
    success = clips_api.SetClipManifestAssetPath(clip_manifest.identifier)
    return success


def set_clip_prim_path(prim: Usd.Prim, clip_set_name: str, prim_path: Sdf.Path) -> bool:
    """Set the path to a prim within the clips for a clip set.

    The clip prim path is the path to the prim in the clips that provides
    the opinions for the prim. If this is not specified, the prim path is
    assumed to be the same as the path to the prim with the clip metadata.
    """
    clips_api = Usd.ClipsAPI(prim)
    clips_dict = clips_api.GetClips()
    if clip_set_name not in clips_dict:
        return False
    clip_set_dict = clips_dict[clip_set_name]
    clip_set_dict["clipPrimPath"] = prim_path
    success = clips_api.SetClips(clips_dict)
    return success


def get_clip_template_stride(prim: Usd.Prim, clip_set_name: str = "default") -> Optional[float]:
    """Get the template stride value for a clip set on a prim.

    Args:
        prim (Usd.Prim): The prim to get the clip template stride from.
        clip_set_name (str, optional): The name of the clip set. Defaults to "default".

    Returns:
        Optional[float]: The template stride value, or None if not set.
    """
    clip_api = Usd.ClipsAPI(prim)
    clip_dict = clip_api.GetClips().get(clip_set_name)
    if not clip_dict:
        print(f"Clip set '{clip_set_name}' not found on prim '{prim.GetPath()}'.")
        return None
    if "templateStride" not in clip_dict:
        print(f"templateStride not set for clip set '{clip_set_name}' on prim '{prim.GetPath()}'.")
        return None
    template_stride = clip_dict["templateStride"]
    if not isinstance(template_stride, float) and (not isinstance(template_stride, int)):
        print(
            f"Invalid templateStride value '{template_stride}' for clip set '{clip_set_name}' on prim '{prim.GetPath()}'."
        )
        return None
    return float(template_stride)


def get_clip_template_active_offset(prim: Usd.Prim, clip_set_name: str = "default") -> Vt.Vec2dArray:
    """
    Get the clip template active offset for the specified prim and clip set.

    Args:
        prim (Usd.Prim): The prim to get the clip template active offset for.
        clip_set_name (str): The name of the clip set to get the active offset for.
                             Defaults to "default".

    Returns:
        Vt.Vec2dArray: The clip template active offset array.
    """
    clips_api = Usd.ClipsAPI(prim)
    clip_sets = clips_api.GetClips()
    if clip_set_name not in clip_sets:
        raise ValueError(f"Clip set '{clip_set_name}' does not exist on prim '{prim.GetPath()}'")
    active_offset = clips_api.GetClipTemplateActiveOffset(clip_set_name)
    if not active_offset:
        return Vt.Vec2dArray()
    return active_offset


def merge_clip_sets(clips_a: Dict, clips_b: Dict) -> Dict:
    """Merges two clip set dictionaries, with clips_b taking precedence."""
    merged_clips = clips_a.copy()
    for clip_set_name, clip_set_b in clips_b.items():
        if clip_set_name in merged_clips:
            clip_set_a = merged_clips[clip_set_name]
            for key, value in clip_set_b.items():
                clip_set_a[key] = value
        else:
            merged_clips[clip_set_name] = clip_set_b
    return merged_clips


def set_clip_metadata(prim: Usd.Prim, clip_set_name: str, clip_dict: Dict[str, Any]) -> None:
    """Set the clip metadata for a given prim and clip set.

    Args:
        prim (Usd.Prim): The prim to set the clip metadata on.
        clip_set_name (str): The name of the clip set.
        clip_dict (Dict[str, Any]): A dictionary containing the clip metadata.

    Raises:
        ValueError: If the prim is not valid or if the clip dictionary is empty.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    if not clip_dict:
        raise ValueError("Clip dictionary cannot be empty.")
    clips_api = Usd.ClipsAPI(prim)
    clips = clips_api.GetClips()
    clips[clip_set_name] = clip_dict
    clips_api.SetClips(clips)


def set_clip_reference_mode(prim: Usd.Prim, clip_set_name: str, reference_mode: str) -> bool:
    """Set the clip reference mode for a given clip set on a prim.

    Args:
        prim (Usd.Prim): The prim to set the clip reference mode on.
        clip_set_name (str): The name of the clip set to set the reference mode for.
        reference_mode (str): The reference mode to set. Must be either "append" or "replace".

    Returns:
        bool: True if the reference mode was set successfully, False otherwise.
    """
    if not prim.IsValid():
        return False
    clips_api = Usd.ClipsAPI(prim)
    clips_dict = clips_api.GetClips()
    if clip_set_name not in clips_dict:
        return False
    clip_set_dict = clips_dict[clip_set_name]
    if reference_mode not in ["append", "replace"]:
        return False
    clip_set_dict["referenceMode"] = reference_mode
    clips_dict[clip_set_name] = clip_set_dict
    return clips_api.SetClips(clips_dict)


def set_clip_layer_priority(prim: Usd.Prim, clip_set_name: str, priority: int) -> bool:
    """Set the priority of a clip layer for the specified prim and clip set.

    Args:
        prim (Usd.Prim): The prim to set the clip layer priority for.
        clip_set_name (str): The name of the clip set.
        priority (int): The priority value to set for the clip layer.

    Returns:
        bool: True if the priority was set successfully, False otherwise.
    """
    clips_api = Usd.ClipsAPI(prim)
    clip_sets = clips_api.GetClips()
    if clip_set_name not in clip_sets:
        print(f"Clip set '{clip_set_name}' does not exist on prim '{prim.GetPath()}'")
        return False
    clip_dict = clip_sets[clip_set_name]
    if "priority" not in clip_dict:
        clip_dict["priority"] = priority
    else:
        clip_dict["priority"] = priority
    success = clips_api.SetClips(clip_sets)
    return success


def get_clip_layer_priority(prim: Usd.Prim, clip_set_name: str) -> Optional[int]:
    """
    Get the layer priority for a given clip set on a prim.

    Args:
        prim (Usd.Prim): The prim to query the clip layer priority for.
        clip_set_name (str): The name of the clip set to get the layer priority for.

    Returns:
        int or None: The layer priority of the clip set, or None if the clip set doesn't exist or has no priority opinion.
    """
    clips_api = Usd.ClipsAPI(prim)
    clips_dict = clips_api.GetClips()
    if clip_set_name not in clips_dict:
        return None
    clip_set_dict = clips_dict[clip_set_name]
    if "clipPrimPath" in clip_set_dict:
        return clip_set_dict.get("layerPriority")
    else:
        return None


def set_clip_time_offset(prim: Usd.Prim, clip_set_name: str, time_offset: float) -> bool:
    """Set the time offset for a clip set on the given prim.

    Args:
        prim (Usd.Prim): The prim to set the clip metadata on.
        clip_set_name (str): The name of the clip set to set the offset for.
        time_offset (float): The time offset value to set.

    Returns:
        bool: True if the offset was set successfully, False otherwise.
    """
    clips_api = Usd.ClipsAPI(prim)
    clips_dict = clips_api.GetClips()
    if clip_set_name not in clips_dict:
        print(f"Clip set '{clip_set_name}' does not exist on prim '{prim.GetPath()}'")
        return False
    clip_set_dict = clips_dict[clip_set_name]
    try:
        clip_set_dict["templateActiveOffset"] = time_offset
    except (KeyError, TypeError) as e:
        print(f"Failed to set time offset: {str(e)}")
        return False
    clips_dict[clip_set_name] = clip_set_dict
    if not clips_api.SetClips(clips_dict):
        print("Failed to set updated clips dictionary")
        return False
    return True


def get_clip_time_offset(clip_api: Usd.ClipsAPI, clip_set_name: str) -> float:
    """Get the time offset for the specified clip set on the given prim."""
    clips_dictionary = clip_api.GetClips()
    if clip_set_name not in clips_dictionary:
        raise ValueError(f"Clip set '{clip_set_name}' does not exist on prim.")
    clip_set_dict = clips_dictionary[clip_set_name]
    if "templateActiveOffset" in clip_set_dict:
        return clip_set_dict["templateActiveOffset"]
    else:
        return 0.0


def get_clip_loop_count(prim: Usd.Prim, clip_set_name: str) -> int:
    """Get the loop count for a clip set on a prim.

    Args:
        prim (Usd.Prim): The prim to get the clip loop count for.
        clip_set_name (str): The name of the clip set.

    Returns:
        int: The loop count for the specified clip set.
    """
    clips_api = Usd.ClipsAPI(prim)
    clips_dict = clips_api.GetClips()
    if clip_set_name not in clips_dict:
        raise ValueError(f"Clip set '{clip_set_name}' not found on prim '{prim.GetPath()}'")
    loop_count = 1
    if "loop" in clips_dict[clip_set_name]:
        loop_count = clips_dict[clip_set_name]["loop"]
        if not isinstance(loop_count, int) or loop_count < 1:
            raise ValueError(
                f"Invalid loop count '{loop_count}' for clip set '{clip_set_name}' on prim '{prim.GetPath()}'"
            )
    return loop_count


def create_clip_template(
    prim: Usd.Prim,
    clip_set_name: str,
    template_asset_path: str,
    template_start_time: float,
    template_end_time: float,
    template_stride: float,
) -> bool:
    """Create a clip template for the specified prim and clip set.

    Args:
        prim (Usd.Prim): The prim to create the clip template for.
        clip_set_name (str): The name of the clip set to create the template for.
        template_asset_path (str): The template asset path for the clip set.
        template_start_time (float): The template start time for the clip set.
        template_end_time (float): The template end time for the clip set.
        template_stride (float): The stride value for the clip set template.

    Returns:
        bool: True if the clip template was created successfully, False otherwise.
    """
    clips_api = Usd.ClipsAPI(prim)
    if template_start_time >= template_end_time:
        print(
            f"Error: template_start_time ({template_start_time}) must be less than template_end_time ({template_end_time})"
        )
        return False
    if template_stride <= 0:
        print(f"Error: template_stride ({template_stride}) must be greater than 0")
        return False
    if not Sdf.Path.IsValidIdentifier(clip_set_name):
        print(f"Error: clip set name '{clip_set_name}' is not a valid identifier")
        return False
    if not clips_api.SetClipTemplateAssetPath(template_asset_path):
        print(f"Error setting clip template asset path")
        return False
    if not clips_api.SetClipTemplateStartTime(template_start_time):
        print(f"Error setting clip template start time")
        return False
    if not clips_api.SetClipTemplateEndTime(template_end_time):
        print(f"Error setting clip template end time")
        return False
    if not clips_api.SetClipTemplateStride(template_stride):
        print(f"Error setting clip template stride")
        return False
    return True


def set_clip_active_range(prim: Usd.Prim, clip_set_name: str, start_time: float, end_time: float) -> bool:
    """Set the active range for a clip set on a prim.

    Args:
        prim (Usd.Prim): The prim to set the clip active range on.
        clip_set_name (str): The name of the clip set to modify.
        start_time (float): The start time of the active range.
        end_time (float): The end time of the active range.

    Returns:
        bool: True if the operation succeeded, False otherwise.
    """
    clips_api = Usd.ClipsAPI(prim)
    clips_dict = clips_api.GetClips()
    if clip_set_name not in clips_dict:
        print(f"Clip set '{clip_set_name}' does not exist on prim '{prim.GetPath()}'")
        return False
    active_times = clips_dict[clip_set_name].get("active", Vt.Vec2dArray())
    active_times_list = list(active_times)
    active_times_list.append(Gf.Vec2d(start_time, end_time))
    active_times = Vt.Vec2dArray(active_times_list)
    clips_dict[clip_set_name]["active"] = active_times
    success = clips_api.SetClips(clips_dict)
    if not success:
        print(f"Failed to set clip active range on prim '{prim.GetPath()}'")
        return False
    return True


def get_clip_active_range(prim: Usd.Prim) -> Tuple[float, float]:
    """Get the active range for the clip on the given prim."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is invalid.")
    clips_api = Usd.ClipsAPI(prim)
    clip_sets = Vt.Dictionary()
    if not clips_api.GetClips(clip_sets) or not clip_sets:
        raise ValueError(f"Prim {prim.GetPath()} has no clip sets.")
    clip_set_name = next(iter(clip_sets))
    active_times = Vt.Vec2dArray()
    if not clips_api.GetClipActive(active_times) or not active_times:
        raise ValueError(f"Clip set {clip_set_name} has no active times.")
    start_time = active_times[0][0]
    end_time = active_times[-1][1]
    clip_times = Vt.Vec2dArray()
    if clips_api.GetClipTimes(clip_times) and clip_times:
        end_time = clip_times[-1][1]
    return (start_time, end_time)


def set_clip_template_asset_path(prim: Usd.Prim, clip_set_name: str, template_asset_path: str) -> bool:
    """
    Set the clip template asset path for the specified clip set on the given prim.

    Args:
        prim (Usd.Prim): The prim to set the clip template asset path on.
        clip_set_name (str): The name of the clip set to set the template asset path for.
        template_asset_path (str): The template asset path to set.

    Returns:
        bool: True if the clip template asset path was set successfully, False otherwise.
    """
    clips_api = Usd.ClipsAPI(prim)
    clip_sets = clips_api.GetClips()
    if clip_set_name not in clip_sets:
        clip_sets[clip_set_name] = {}
    clip_sets[clip_set_name]["templateAssetPath"] = Sdf.AssetPath(template_asset_path)
    return clips_api.SetClips(clip_sets)


def get_interpolate_missing_clip_values(self) -> bool:
    """
    Get the interpolateMissingClipValues value for the given clip set.

    Returns:
        bool: The interpolateMissingClipValues value.
    """
    clips = self.GetClips()
    if not clips:
        return False
    clip_set_name = next(iter(clips))
    interpolate_missing_clip_values = clips[clip_set_name].get("interpolateMissingClipValues", False)
    return bool(interpolate_missing_clip_values)


def add_clip_to_set(
    prim: Usd.Prim,
    clip_set_name: str,
    clip_asset_path: str,
    clip_prim_path: str,
    start_time: Usd.TimeCode,
    end_time: Usd.TimeCode,
) -> None:
    """Add a clip to the specified clip set on the given prim."""
    clips_api = Usd.ClipsAPI(prim)
    clips_dict = clips_api.GetClips()
    if clip_set_name not in clips_dict:
        clips_dict[clip_set_name] = {}
    clip_set_dict = clips_dict[clip_set_name]
    if "assetPaths" not in clip_set_dict:
        clip_set_dict["assetPaths"] = []
    clip_set_dict["assetPaths"].append(clip_asset_path)
    if "primPath" not in clip_set_dict:
        clip_set_dict["primPath"] = clip_prim_path
    if "times" not in clip_set_dict:
        clip_set_dict["times"] = []
    clip_set_dict["times"].append(Gf.Vec2d(start_time.GetValue(), end_time.GetValue()))
    clips_api.SetClips(clips_dict)


def remove_clip_from_set(prim: Usd.Prim, clip_set_name: str, clip_identifier: str) -> bool:
    """Remove a clip from a clip set on the given prim.

    Args:
        prim (Usd.Prim): The prim to remove the clip from.
        clip_set_name (str): The name of the clip set to remove the clip from.
        clip_identifier (str): The identifier of the clip to remove (e.g., the asset path).

    Returns:
        bool: True if the clip was successfully removed, False otherwise.
    """
    clip_api = Usd.ClipsAPI(prim)
    clips_dict = clip_api.GetClips()
    if clips_dict:
        if clip_set_name in clips_dict:
            clip_set = clips_dict[clip_set_name]
            asset_paths = clip_set.get("assetPaths", [])
            if clip_identifier in asset_paths:
                asset_paths.remove(clip_identifier)
                clip_set["assetPaths"] = asset_paths
                if clip_api.SetClips(clips_dict):
                    return True
                else:
                    print(f"Failed to update clips dictionary on prim {prim.GetPath()}")
            else:
                print(f"Clip identifier '{clip_identifier}' not found in clip set '{clip_set_name}'")
        else:
            print(f"Clip set '{clip_set_name}' not found on prim {prim.GetPath()}")
    else:
        print(f"No clips found on prim {prim.GetPath()}")
    return False


def resolve_clip_asset(stage: Usd.Stage, prim_path: str, clip_set_name: str) -> Sdf.AssetPath:
    """Resolve the clip asset path for a given prim and clip set."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    clips_api = Usd.ClipsAPI(prim)
    if not clips_api:
        raise ValueError(f"Prim at path {prim_path} does not have ClipsAPI applied.")
    clip_sets = clips_api.GetClips()
    if not clip_sets:
        raise ValueError(f"No clip sets found for prim at path {prim_path}.")
    if clip_set_name not in clip_sets:
        raise ValueError(f"Clip set '{clip_set_name}' not found for prim at path {prim_path}.")
    clip_set = clip_sets[clip_set_name]
    asset_paths = clip_set.get("assetPaths")
    if not asset_paths:
        raise ValueError(f"No asset paths found for clip set '{clip_set_name}' on prim at path {prim_path}.")
    return Sdf.AssetPath(asset_paths[0])


def set_clip_layer_offset(prim: Usd.Prim, clip_set_name: str, layer_offset: float) -> bool:
    """Set the layer offset for a clip set on the given prim.

    Args:
        prim (Usd.Prim): The prim to set the clip layer offset on.
        clip_set_name (str): The name of the clip set to set the offset for.
        layer_offset (float): The layer offset value to set.

    Returns:
        bool: True if the offset was set successfully, False otherwise.
    """
    clips_api = Usd.ClipsAPI(prim)
    clips_dict = clips_api.GetClips()
    if clip_set_name not in clips_dict:
        print(f"Warning: Clip set '{clip_set_name}' does not exist on prim '{prim.GetPath()}'.")
        return False
    clip_set_dict = clips_dict[clip_set_name]
    clip_set_dict["layerOffset"] = layer_offset
    success = clips_api.SetClips(clips_dict)
    return success


def delete_clip_template(prim: Usd.Prim) -> bool:
    """Delete the clip template metadata from the specified prim."""
    clips_api = Usd.ClipsAPI(prim)
    if not clips_api.GetClipTemplateAssetPath():
        return False
    clips_api.SetClipTemplateAssetPath("")
    clips_api.SetClipTemplateStartTime(0.0)
    clips_api.SetClipTemplateEndTime(0.0)
    clips_api.SetClipTemplateStride(1.0)
    clips_api.SetClipTemplateActiveOffset(0.0)
    return True


def set_clip_time_scale(prim: Usd.Prim, clip_set_name: str, time_scale: float) -> bool:
    """Set the time scale for a clip set on the given prim.

    Args:
        prim (Usd.Prim): The prim to set the clip time scale on.
        clip_set_name (str): The name of the clip set to set the time scale for.
        time_scale (float): The time scale value to set.

    Returns:
        bool: True if the time scale was set successfully, False otherwise.
    """
    clips_api = Usd.ClipsAPI(prim)
    clips_dict = clips_api.GetClips()
    if clip_set_name not in clips_dict:
        print(f"Clip set '{clip_set_name}' does not exist on prim '{prim.GetPath()}'")
        return False
    clip_set_dict = clips_dict[clip_set_name]
    try:
        clip_set_dict["time_scale"] = time_scale
    except (TypeError, ValueError) as e:
        print(f"Error setting time scale: {str(e)}")
        return False
    clips_api.SetClips(clips_dict)
    return True


def set_clip_loop_count(prim: Usd.Prim, clip_set_name: str, loop_count: int) -> bool:
    """Set the loop count for a given clip set on a prim.

    Args:
        prim (Usd.Prim): The prim to set the clip loop count on.
        clip_set_name (str): The name of the clip set to set the loop count for.
        loop_count (int): The loop count value to set. Must be greater than or equal to 0.

    Returns:
        bool: True if the loop count was set successfully, False otherwise.
    """
    if not prim.IsValid():
        return False
    clips_api = Usd.ClipsAPI(prim)
    clips_dictionary = clips_api.GetClips()
    if clip_set_name not in clips_dictionary:
        return False
    if loop_count < 0:
        return False
    clip_set_dictionary = clips_dictionary[clip_set_name]
    clip_set_dictionary["loop"] = loop_count
    clips_api.SetClips(clips_dictionary)
    return True


def set_clip_template_start_end_time(prim: Usd.Prim, clip_set_name: str, start_time: float, end_time: float) -> bool:
    """Set the template clip start and end times for the specified clip set on the given prim.

    Args:
        prim (Usd.Prim): The prim to set the clip metadata on.
        clip_set_name (str): The name of the clip set to set the template times for.
        start_time (float): The template start time.
        end_time (float): The template end time.

    Returns:
        bool: True if successful, False otherwise.
    """
    clip_api = Usd.ClipsAPI(prim)
    if start_time > end_time:
        print(f"Error: start_time {start_time} is greater than end_time {end_time}")
        return False
    set_start = clip_api.SetClipTemplateStartTime(start_time, clip_set_name)
    set_end = clip_api.SetClipTemplateEndTime(end_time, clip_set_name)
    if not set_start or not set_end:
        print(f"Error setting template clip times for clip set '{clip_set_name}' on prim '{prim.GetPath()}'")
        return False
    return True


def remove_clip_set(prim: Usd.Prim, clip_set_name: str) -> bool:
    """Remove a clip set from the prim if it exists."""
    clips_api = Usd.ClipsAPI(prim)
    clips_dict = clips_api.GetClips()
    if clip_set_name not in clips_dict:
        return False
    del clips_dict[clip_set_name]
    success = clips_api.SetClips(clips_dict)
    return success


def list_clip_sets(prim: Usd.Prim) -> List[str]:
    """Return a list of all clip set names defined on the given prim."""
    clips_dict = prim.GetMetadata("clips")
    if clips_dict:
        return list(clips_dict.keys())
    else:
        return []


def set_clip_template_active_offset(prim: Usd.Prim, clip_set_name: str, active_offset: float) -> bool:
    """Set the active offset for a clip set using a template.

    Args:
        prim (Usd.Prim): The prim to set the clip metadata on.
        clip_set_name (str): The name of the clip set to set the active offset for.
        active_offset (float): The active offset value to set.

    Returns:
        bool: True if the active offset was set successfully, False otherwise.
    """
    clips_dictionary = prim.GetMetadata("clips")
    if not clips_dictionary:
        print(f"No clips dictionary found on prim '{prim.GetPath()}'")
        return False
    if clip_set_name not in clips_dictionary:
        print(f"Clip set '{clip_set_name}' does not exist on prim '{prim.GetPath()}'")
        return False
    clip_set = clips_dictionary[clip_set_name]
    clip_set["templateActiveOffset"] = active_offset
    prim.SetMetadata("clips", clips_dictionary)
    return True


def import_clip_manifest_from_file(prim: Usd.Prim, clip_set_name: str, manifest_file_path: str) -> bool:
    """Import a clip manifest from a file and associate it with the specified clip set on the given prim.

    Args:
        prim (Usd.Prim): The prim to associate the clip manifest with.
        clip_set_name (str): The name of the clip set to associate the manifest with.
        manifest_file_path (str): The file path to the clip manifest to import.

    Returns:
        bool: True if the import was successful, False otherwise.
    """
    if not prim.IsValid():
        print(f"Error: Invalid prim '{prim.GetPath()}'")
        return False
    if not os.path.isfile(manifest_file_path):
        print(f"Error: Manifest file '{manifest_file_path}' does not exist")
        return False
    clips_api = Usd.ClipsAPI(prim)
    clips_dict = clips_api.GetClips()
    if clip_set_name in clips_dict:
        clips_dict[clip_set_name]["manifestAssetPath"] = manifest_file_path
    else:
        clips_dict[clip_set_name] = {"manifestAssetPath": manifest_file_path}
    clips_api.SetClips(clips_dict)
    return True


def get_clip_manifest_asset_path(prim: Usd.Prim, clip_set_name: str = "default") -> str:
    """
    Retrieve the clip manifest asset path for the specified prim and clip set.

    Args:
        prim (Usd.Prim): The prim to retrieve the clip manifest asset path for.
        clip_set_name (str): The name of the clip set. Defaults to "default".

    Returns:
        str: The clip manifest asset path, or an empty string if not found.
    """
    clips_dict = prim.GetMetadata("clips")
    if not clips_dict:
        return ""
    if clip_set_name not in clips_dict:
        return ""
    clip_set_dict = clips_dict[clip_set_name]
    manifest_asset_path = ""
    if "manifestAssetPath" in clip_set_dict:
        manifest_asset_path = clip_set_dict["manifestAssetPath"]
    return manifest_asset_path


def optimize_clip_data(clips_api: Usd.ClipsAPI) -> bool:
    """Optimize the clip data for the given prim by removing duplicate entries."""
    clip_dict = clips_api.GetClips()
    if not clip_dict:
        return True
    new_clip_dict = {}
    for clip_set_name, clip_data in clip_dict.items():
        asset_paths = clip_data.get("assetPaths", Sdf.AssetPathArray())
        active_times = clip_data.get("active", Vt.Vec2dArray())
        times = clip_data.get("times", Vt.Vec2dArray())
        prim_path = clip_data.get("primPath", Sdf.Path())
        manifest_asset_path = clip_data.get("manifestAssetPath", Sdf.AssetPath())
        seen = set()
        new_asset_paths = []
        new_active_times = []
        new_times = []
        for i in range(len(asset_paths)):
            if asset_paths[i] not in seen:
                seen.add(asset_paths[i])
                new_asset_paths.append(asset_paths[i])
                new_active_times.append(active_times[i])
                new_times.append(times[i])
        new_clip_data = {}
        new_clip_data["assetPaths"] = Sdf.AssetPathArray(new_asset_paths)
        new_clip_data["active"] = Vt.Vec2dArray(new_active_times)
        new_clip_data["times"] = Vt.Vec2dArray(new_times)
        new_clip_data["primPath"] = prim_path
        new_clip_data["manifestAssetPath"] = manifest_asset_path
        new_clip_dict[clip_set_name] = new_clip_data
    return clips_api.SetClips(new_clip_dict)


def get_clip_time_mapping(
    clip_api: Usd.ClipsAPI, clip_set_name: str = ""
) -> Dict[Sdf.Path, List[Tuple[Gf.Vec2d, Gf.Vec2d]]]:
    """
    Get the time mapping for all clips in a given clip set.

    Args:
        clip_api (Usd.ClipsAPI): The ClipsAPI object to query.
        clip_set_name (str): The name of the clip set to query. If empty, the default clip set is used.

    Returns:
        Dict[Sdf.Path, List[Tuple[Gf.Vec2d, Gf.Vec2d]]]: A dictionary mapping clip prim paths to a list of
        tuples representing the time mapping for each clip. Each tuple contains the source time range
        and the corresponding stage time range.
    """
    if not clip_api.GetPrim().IsValid():
        raise ValueError("Accessed schema on invalid prim")
    clips_dict = clip_api.GetClips()
    if clip_set_name and clip_set_name not in clips_dict:
        raise ValueError(f"Clip set '{clip_set_name}' does not exist.")
    if not clip_set_name:
        if not clips_dict:
            return {}
        clip_set_name = next(iter(clips_dict))
    clip_set_dict = clips_dict[clip_set_name]
    clip_asset_paths = clip_set_dict.get("assetPaths", Sdf.AssetPathArray())
    clip_times = clip_set_dict.get("times", Vt.Vec2dArray())
    time_mapping = {}
    for asset_path, (start_time, end_time) in zip(clip_asset_paths, clip_times):
        clip_layer = Sdf.Layer.FindOrOpen(asset_path.resolvedPath)
        if not clip_layer:
            raise ValueError(f"Failed to open clip layer: {asset_path.resolvedPath}")
        clip_prim_path = clip_set_dict.get("primPath", Sdf.Path.emptyPath)
        if clip_prim_path not in time_mapping:
            time_mapping[clip_prim_path] = []
        time_mapping[clip_prim_path].append(
            ((start_time, end_time), (clip_layer.startTimeCode, clip_layer.endTimeCode))
        )
    return time_mapping


def get_clip_template_asset_path(prim: Usd.Prim) -> str:
    """Get the clip template asset path for a prim, if it exists."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    clips_dictionary = prim.GetMetadata("clips")
    if not clips_dictionary:
        return ""
    clip_template_asset_path = clips_dictionary.get("templateAssetPath", "")
    return clip_template_asset_path


def export_clip_manifest_to_file(prim: Usd.Prim, file_path: str) -> bool:
    """
    Export the clip manifest for the given prim to a file.

    Args:
        prim (Usd.Prim): The prim to export the clip manifest for.
        file_path (str): The file path to export the clip manifest to.

    Returns:
        bool: True if the export was successful, False otherwise.
    """
    if not prim.IsValid():
        print(f"Invalid prim: {prim.GetPath()}")
        return False
    clip_manifest_asset_path = ""
    if not prim.GetMetadata("clips", clip_manifest_asset_path):
        print(f"Prim {prim.GetPath()} does not have a clip manifest asset path.")
        return False
    clip_prim_path = Sdf.Path()
    if not prim.GetMetadata("clipPrimPath", clip_prim_path):
        print(f"Prim {prim.GetPath()} does not have a clip prim path.")
        return False
    clip_layers = []
    clip_set_names = prim.GetMetadata("clips")
    for clip_set_name in clip_set_names.keys():
        clip_asset_paths = prim.GetMetadata("clips:" + clip_set_name + ":assetPaths")
        for clip_asset_path in clip_asset_paths:
            clip_layer = Sdf.Layer.FindOrOpen(clip_asset_path)
            if clip_layer:
                clip_layers.append(clip_layer)
    clip_manifest_layer = Usd.ClipsAPI.GenerateClipManifestFromLayers(clip_layers, clip_prim_path)
    if not clip_manifest_layer:
        print("Failed to generate clip manifest layer.")
        return False
    try:
        clip_manifest_layer.Export(file_path)
    except Tf.ErrorException as e:
        print(f"Error exporting clip manifest layer: {e}")
        return False
    return True


def set_clip_namespace(prim: Usd.Prim, clip_namespace: str) -> bool:
    """Set the clip namespace for the given prim."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    clips_api = Usd.ClipsAPI(prim)
    clips_dict = clips_api.GetClips()
    for clip_set_name in clips_dict:
        clip_set = clips_dict[clip_set_name]
        old_clip_prim_path = clip_set.get("primPath", Sdf.Path.emptyPath)
        if isinstance(old_clip_prim_path, Sdf.Path):
            new_clip_prim_path = old_clip_prim_path.ReplacePrefix(old_clip_prim_path.GetPrimPath(), clip_namespace)
        else:
            new_clip_prim_path = Sdf.Path(clip_namespace)
        clip_set["primPath"] = new_clip_prim_path
    success = clips_api.SetClips(clips_dict)
    return success


def get_clip_layer_offset(prim: Usd.Prim, clip_set_name: str) -> Vt.Vec2dArray:
    """Get the layer offsets for a clip set on a prim."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    clip_api = Usd.ClipsAPI(prim)
    if not clip_api.GetClips():
        raise ValueError(f"Prim {prim} does not have clip metadata.")
    clip_dict = clip_api.GetClips()
    if clip_set_name not in clip_dict:
        raise ValueError(f"Clip set '{clip_set_name}' not found on prim {prim}")
    (times_attr, offsets_attr) = ("times", "active")
    if times_attr not in clip_dict[clip_set_name] or offsets_attr not in clip_dict[clip_set_name]:
        return Vt.Vec2dArray()
    times = clip_dict[clip_set_name][times_attr]
    offsets = clip_dict[clip_set_name][offsets_attr]
    if len(times) != len(offsets):
        return Vt.Vec2dArray()
    result = Vt.Vec2dArray(len(times))
    for i in range(len(times)):
        result[i] = Gf.Vec2d(times[i], offsets[i])
    return result


def get_clip_time_scale(prim: Usd.Prim, clip_set_name: str = "default") -> float:
    """Get the time scale for a clip set on a prim.

    Args:
        prim (Usd.Prim): The prim to get the clip time scale for.
        clip_set_name (str, optional): The name of the clip set. Defaults to "default".

    Returns:
        float: The clip time scale, or 1.0 if not found.
    """
    clips_api = Usd.ClipsAPI(prim)
    clips_dict = clips_api.GetClips()
    if not clips_dict:
        return 1.0
    if clip_set_name not in clips_dict:
        return 1.0
    clip_set = clips_dict[clip_set_name]
    time_scale = clip_set.get("templateStride", 1.0)
    return time_scale


def set_clip_time_mapping(
    prim: Usd.Prim, clip_set_name: str, source_times: Vt.Vec2dArray, stage_times: Vt.Vec2dArray
) -> bool:
    """
    Set the time mapping for a clip set on a prim.

    The time mapping maps times in the clip to times on the stage.

    Args:
        prim (Usd.Prim): The prim to set the clip time mapping on.
        clip_set_name (str): The name of the clip set to set the time mapping for.
        source_times (Vt.Vec2dArray): The times in the clip to map from.
        stage_times (Vt.Vec2dArray): The times on the stage to map to.

    Returns:
        bool: True if the time mapping was set successfully, False otherwise.
    """
    clips_api = Usd.ClipsAPI(prim)
    if not clips_api.GetClips():
        print(f"No clips defined on prim '{prim.GetPath()}'")
        return False
    clip_sets = clips_api.GetClips().keys()
    if clip_set_name not in clip_sets:
        print(f"Clip set '{clip_set_name}' does not exist on prim '{prim.GetPath()}'")
        print(f"Available clip sets: {clip_sets}")
        return False
    if len(source_times) != len(stage_times):
        print("Source times and stage times must be the same length")
        return False
    time_mapping = []
    for i in range(len(source_times)):
        time_mapping.append(Gf.Vec2d(source_times[i][0], stage_times[i][0]))
        time_mapping.append(Gf.Vec2d(source_times[i][1], stage_times[i][1]))
    time_mapping = Vt.Vec2dArray(time_mapping)
    result = clips_api.SetClipTimes(time_mapping, clip_set_name)
    return result


def get_composition_summary(prim: Usd.Prim) -> List[str]:
    """Get a summary of the composition arcs for a prim.

    Args:
        prim (Usd.Prim): The prim to get the composition summary for.

    Returns:
        List[str]: A list of strings summarizing the composition arcs.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    query = Usd.PrimCompositionQuery(prim)
    arcs = query.GetCompositionArcs()
    summary = []
    for arc in arcs:
        arc_type = arc.GetArcType()
        arc_path = arc.GetTargetPrimPath()
        summary_str = f"{arc_type}: {arc_path}"
        summary.append(summary_str)
    return summary


def find_prims_with_inherit_arcs(stage: Usd.Stage) -> List[Usd.Prim]:
    """Find all prims on the stage that have inherit arcs."""
    prims_with_inherits = []
    for prim in stage.TraverseAll():
        prim_comp_query = Usd.PrimCompositionQuery(prim)
        inherit_arcs = Usd.PrimCompositionQuery.GetDirectInherits(prim).GetCompositionArcs()
        if inherit_arcs:
            prims_with_inherits.append(prim)
    return prims_with_inherits


def compare_composition_arcs(prim: Usd.Prim) -> bool:
    """Compare the composition arcs of a prim.

    Args:
        prim (Usd.Prim): The prim to query the composition arcs of.

    Returns:
        bool: True if the prim has composition arcs, False otherwise.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim")
    query = Usd.PrimCompositionQuery(prim)
    arcs = query.GetCompositionArcs()
    return bool(arcs)


def summarize_composition_arcs(
    prim: Usd.Prim,
    filter: Usd.PrimCompositionQuery.ArcIntroducedFilter = Usd.PrimCompositionQuery.ArcIntroducedFilter.All,
) -> Dict[str, List[str]]:
    """Summarizes the composition arcs for a given prim.

    Args:
        prim (Usd.Prim): The prim to summarize composition arcs for.
        filter (Usd.PrimCompositionQuery.ArcIntroducedFilter, optional): Filter to apply to the arcs. Defaults to Usd.PrimCompositionQuery.ArcIntroducedFilter.All.

    Returns:
        Dict[str, List[str]]: A dictionary mapping arc types to a list of contributing layer paths.
    """
    prim_composition_query = Usd.PrimCompositionQuery(prim)
    composition_arcs = prim_composition_query.GetCompositionArcs()
    summary = {}
    for arc in composition_arcs:
        if (
            filter == Usd.PrimCompositionQuery.ArcIntroducedFilter.All
            or (
                filter == Usd.PrimCompositionQuery.ArcIntroducedFilter.IntroducedInLocalLayerStack
                and arc.IsIntroducedInLocalLayerStack()
            )
            or (
                filter == Usd.PrimCompositionQuery.ArcIntroducedFilter.IntroducedInRootLayerStack
                and arc.IsIntroducedInRootLayerStack()
            )
        ):
            arc_type = str(arc.GetArcType())
            introducing_layer = arc.GetIntroducingLayer()
            layer_path = introducing_layer.realPath if introducing_layer else "None"
            if arc_type not in summary:
                summary[arc_type] = []
            summary[arc_type].append(layer_path)
    return summary


def visualize_composition_arcs(prim: Usd.Prim, arc_type_filter: Usd.PrimCompositionQuery.ArcTypeFilter) -> str:
    """Visualize the composition arcs for a given prim, filtered by arc type.

    Args:
        prim (Usd.Prim): The prim to visualize the composition arcs for.
        arc_type_filter (Usd.PrimCompositionQuery.ArcTypeFilter): The filter to apply to the arcs.

    Returns:
        str: A string representation of the composition arcs.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    prim_comp_query = Usd.PrimCompositionQuery(prim)
    arcs = prim_comp_query.GetCompositionArcs()
    arc_strings = []
    for arc in arcs:
        if arc.GetArcType() & arc_type_filter:
            source_path = arc.GetIntroducingPrimPath()
            target_path = arc.GetTargetPrimPath()
            arc_type = arc.GetArcType()
            arc_string = f"{source_path} -> {target_path} ({arc_type})"
            arc_strings.append(arc_string)
    result = "\n".join(arc_strings)
    return result


def update_dependency_type(
    dependency_type: Usd.PrimCompositionQuery.DependencyTypeFilter,
    new_dependency_type: Usd.PrimCompositionQuery.DependencyTypeFilter,
) -> Usd.PrimCompositionQuery.DependencyTypeFilter:
    """Updates the dependency type filter with a new dependency type.

    Args:
        dependency_type (Usd.PrimCompositionQuery.DependencyTypeFilter): The current dependency type filter.
        new_dependency_type (Usd.PrimCompositionQuery.DependencyTypeFilter): The new dependency type to add to the filter.

    Returns:
        Usd.PrimCompositionQuery.DependencyTypeFilter: The updated dependency type filter.
    """
    if dependency_type & new_dependency_type:
        return dependency_type
    updated_dependency_type = dependency_type | new_dependency_type
    return updated_dependency_type


def filter_prims_by_arc_type(prims: List[Usd.Prim], arc_type: Usd.PrimCompositionQuery.ArcTypeFilter) -> List[Usd.Prim]:
    """Filter a list of prims by arc type.

    Args:
        prims (List[Usd.Prim]): List of prims to filter.
        arc_type (Usd.PrimCompositionQuery.ArcTypeFilter): Arc type to filter by.

    Returns:
        List[Usd.Prim]: List of prims that match the given arc type.
    """
    filter = Usd.PrimCompositionQuery.Filter()
    filter.arcTypeFilter = arc_type
    filtered_prims = []
    for prim in prims:
        query = Usd.PrimCompositionQuery(prim)
        if any((arc.GetArcType() == arc_type for arc in query.GetCompositionArcs())):
            filtered_prims.append(prim)
    return filtered_prims


def filter_prims_by_specs(prims: List[Usd.Prim], filter: Usd.PrimCompositionQuery.HasSpecsFilter) -> List[Usd.Prim]:
    """Filter prims based on whether they have specs contributing to them.

    Args:
        prims (List[Usd.Prim]): List of prims to filter.
        filter (Usd.PrimCompositionQuery.HasSpecsFilter): Filter to apply.

    Returns:
        List[Usd.Prim]: List of prims that pass the filter.
    """
    filtered_prims = []
    for prim in prims:
        if not prim.IsValid():
            continue
        if filter == Usd.PrimCompositionQuery.HasSpecsFilter.HasSpecs and prim.HasAuthoredSpecializes():
            filtered_prims.append(prim)
        elif filter == Usd.PrimCompositionQuery.HasSpecsFilter.HasNoSpecs and (not prim.HasAuthoredSpecializes()):
            filtered_prims.append(prim)
    return filtered_prims


def get_prim_definition_documentation(prim_def: Usd.PrimDefinition) -> str:
    """Get the documentation metadata for a prim definition."""
    if not prim_def:
        raise ValueError("Invalid prim definition.")
    documentation = prim_def.GetDocumentation()
    return documentation


def get_schema_attribute_spec(prim_def: Usd.PrimDefinition, attr_name: str) -> Sdf.AttributeSpec:
    """
    Get the attribute spec that defines the fallback for the attribute named attr_name
    on prims of this prim definition's type.

    Args:
        prim_def (Usd.PrimDefinition): The prim definition to query.
        attr_name (str): The name of the attribute to get the spec for.

    Returns:
        Sdf.AttributeSpec: The attribute spec for the named attribute, or None if no
        such attribute spec exists.
    """
    prop_spec = prim_def.GetSchemaPropertySpec(attr_name)
    if not isinstance(prop_spec, Sdf.AttributeSpec):
        return None
    return prop_spec


def get_relationship_spec(prim_def: Usd.PrimDefinition, rel_name: str) -> Sdf.RelationshipSpec:
    """
    Get the relationship spec for a given relationship name in a prim definition.

    Args:
        prim_def (Usd.PrimDefinition): The prim definition to query.
        rel_name (str): The name of the relationship to retrieve the spec for.

    Returns:
        Sdf.RelationshipSpec: The relationship spec if found, None otherwise.
    """
    prop_spec = prim_def.GetSchemaPropertySpec(rel_name)
    if prop_spec and isinstance(prop_spec, Sdf.RelationshipSpec):
        return prop_spec
    return None


def get_all_applied_schemas(prim_def: Usd.PrimDefinition) -> List[Tf.Type]:
    """
    Get the list of all applied API schemas for a given prim definition.

    Args:
        prim_def (Usd.PrimDefinition): The prim definition to query.

    Returns:
        List[Tf.Type]: The list of applied API schema types.
    """
    if not prim_def:
        raise ValueError("Invalid prim definition.")
    applied_schemas = prim_def.GetAppliedAPISchemas()
    schema_types = [Tf.Type.FindByName(schema_name) for schema_name in applied_schemas]
    return schema_types


def merge_prim_definitions(prim_definitions: List[Usd.PrimDefinition]) -> Usd.PrimDefinition:
    """Merge multiple prim definitions into a single prim definition."""
    if not prim_definitions:
        raise ValueError("At least one prim definition must be provided.")
    valid_prim_definitions = [prim_def for prim_def in prim_definitions if prim_def is not None]
    if not valid_prim_definitions:
        raise ValueError("All provided prim definitions are None.")
    prim_type_name = valid_prim_definitions[0].GetTypeName()
    merged_prim_spec = Sdf.PrimSpec(Sdf.Path.absoluteRootPath, prim_type_name, Sdf.SpecifierDef)
    for prim_def in valid_prim_definitions:
        if prim_def.GetTypeName() != prim_type_name:
            raise ValueError(f"Inconsistent prim type found: {prim_def.GetTypeName()}")
        doc = prim_def.GetDocumentation()
        if doc:
            merged_prim_spec.documentation = doc
        for metadata_field in prim_def.ListMetadataFields():
            value = prim_def.GetMetadata(metadata_field)
            merged_prim_spec.SetInfo(metadata_field, value)
        for prop_name in prim_def.GetPropertyNames():
            prop_spec = prim_def.GetSchemaPropertySpec(prop_name)
            merged_prim_spec.properties.append(prop_spec)
            prop_doc = prim_def.GetPropertyDocumentation(prop_name)
            if prop_doc:
                merged_prim_spec.properties[prop_name].documentation = prop_doc
            for prop_metadata_field in prim_def.ListPropertyMetadataFields(prop_name):
                prop_metadata_value = prim_def.GetPropertyMetadata(prop_name, prop_metadata_field)
                merged_prim_spec.properties[prop_name].SetInfo(prop_metadata_field, prop_metadata_value)
    merged_prim_def = Usd.SchemaRegistry().BuildComposedPrimDefinition(merged_prim_spec)
    return merged_prim_def


def get_all_fallback_values(prim_def: Usd.PrimDefinition) -> Dict[str, Any]:
    """
    Return a dictionary containing all fallback values for properties
    defined in the given prim definition.
    """
    fallback_values = {}
    for prop_name in prim_def.GetPropertyNames():
        prop_spec = prim_def.GetSchemaPropertySpec(prop_name)
        if isinstance(prop_spec, Sdf.AttributeSpec):
            attr_name = prop_name
            attr_type = prop_spec.typeName
            if attr_type == Sdf.ValueTypeNames.Float:
                fallback_value = 0.0
            elif attr_type == Sdf.ValueTypeNames.Double:
                fallback_value = 0.0
            elif attr_type == Sdf.ValueTypeNames.Int:
                fallback_value = 0
            elif attr_type == Sdf.ValueTypeNames.String:
                fallback_value = ""
            elif attr_type == Sdf.ValueTypeNames.Bool:
                fallback_value = False
            elif attr_type == Sdf.ValueTypeNames.Float2:
                fallback_value = Gf.Vec2f()
            elif attr_type == Sdf.ValueTypeNames.Float3:
                fallback_value = Gf.Vec3f()
            elif attr_type == Sdf.ValueTypeNames.Float4:
                fallback_value = Gf.Vec4f()
            elif attr_type == Sdf.ValueTypeNames.Double2:
                fallback_value = Gf.Vec2d()
            elif attr_type == Sdf.ValueTypeNames.Double3:
                fallback_value = Gf.Vec3d()
            elif attr_type == Sdf.ValueTypeNames.Double4:
                fallback_value = Gf.Vec4d()
            elif attr_type == Sdf.ValueTypeNames.Point3h:
                fallback_value = Gf.Vec3h()
            elif attr_type == Sdf.ValueTypeNames.Point3f:
                fallback_value = Gf.Vec3f()
            elif attr_type == Sdf.ValueTypeNames.Point3d:
                fallback_value = Gf.Vec3d()
            elif attr_type == Sdf.ValueTypeNames.Vector3h:
                fallback_value = Gf.Vec3h()
            elif attr_type == Sdf.ValueTypeNames.Vector3f:
                fallback_value = Gf.Vec3f()
            elif attr_type == Sdf.ValueTypeNames.Vector3d:
                fallback_value = Gf.Vec3d()
            elif attr_type == Sdf.ValueTypeNames.Normal3h:
                fallback_value = Gf.Vec3h()
            elif attr_type == Sdf.ValueTypeNames.Normal3f:
                fallback_value = Gf.Vec3f()
            elif attr_type == Sdf.ValueTypeNames.Normal3d:
                fallback_value = Gf.Vec3d()
            elif attr_type == Sdf.ValueTypeNames.Color3h:
                fallback_value = Gf.Vec3h()
            elif attr_type == Sdf.ValueTypeNames.Color3f:
                fallback_value = Gf.Vec3f()
            elif attr_type == Sdf.ValueTypeNames.Color3d:
                fallback_value = Gf.Vec3d()
            elif attr_type == Sdf.ValueTypeNames.Color4h:
                fallback_value = Gf.Vec4h()
            elif attr_type == Sdf.ValueTypeNames.Color4f:
                fallback_value = Gf.Vec4f()
            elif attr_type == Sdf.ValueTypeNames.Color4d:
                fallback_value = Gf.Vec4d()
            elif attr_type == Sdf.ValueTypeNames.Quath:
                fallback_value = Gf.Quath()
            elif attr_type == Sdf.ValueTypeNames.Quatf:
                fallback_value = Gf.Quatf()
            elif attr_type == Sdf.ValueTypeNames.Quatd:
                fallback_value = Gf.Quatd()
            elif attr_type == Sdf.ValueTypeNames.Matrix2d:
                fallback_value = Gf.Matrix2d()
            elif attr_type == Sdf.ValueTypeNames.Matrix3d:
                fallback_value = Gf.Matrix3d()
            elif attr_type == Sdf.ValueTypeNames.Matrix4d:
                fallback_value = Gf.Matrix4d()
            elif attr_type == Sdf.ValueTypeNames.Frame4d:
                fallback_value = Gf.Matrix4d()
            elif attr_type == Sdf.ValueTypeNames.TexCoord2f:
                fallback_value = Gf.Vec2f()
            elif attr_type == Sdf.ValueTypeNames.TexCoord2d:
                fallback_value = Gf.Vec2d()
            elif attr_type == Sdf.ValueTypeNames.TexCoord2h:
                fallback_value = Gf.Vec2h()
            elif attr_type == Sdf.ValueTypeNames.TexCoord3f:
                fallback_value = Gf.Vec3f()
            elif attr_type == Sdf.ValueTypeNames.TexCoord3d:
                fallback_value = Gf.Vec3d()
            elif attr_type == Sdf.ValueTypeNames.TexCoord3h:
                fallback_value = Gf.Vec3h()
            else:
                continue
            attr_spec = prim_def.GetSchemaAttributeSpec(attr_name)
            if attr_spec and attr_spec.default:
                fallback_value = attr_spec.default
                fallback_values[attr_name] = fallback_value
    return fallback_values


def traverse_and_modify_materials(prim: Usd.Prim, opacity: Optional[float] = None, roughness: Optional[float] = None):
    """Traverse the prim hierarchy and modify material properties.

    Args:
        prim (Usd.Prim): The starting prim to traverse from.
        opacity (float, optional): The opacity value to set on materials. Defaults to None.
        roughness (float, optional): The roughness value to set on materials. Defaults to None.
    """
    for child_prim in Usd.PrimRange.PreAndPostVisit(prim):
        if not child_prim.IsInstance():
            if child_prim.IsA(UsdShade.Material):
                material = UsdShade.Material(child_prim)
                shader = material.GetSurfaceOutput().GetConnectedSource()[0]
                if opacity is not None:
                    if shader.GetPrim().HasAttribute("inputs:opacity"):
                        shader.GetPrim().GetAttribute("inputs:opacity").Set(opacity)
                    else:
                        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
                if roughness is not None:
                    if shader.GetPrim().HasAttribute("inputs:roughness"):
                        shader.GetPrim().GetAttribute("inputs:roughness").Set(roughness)
                    else:
                        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)


def traverse_and_apply_transform(prim_range: Usd.PrimRange, transform: Gf.Matrix4d):
    """
    Traverse the prims in the given range and apply the specified transform to each prim.

    Args:
        prim_range (Usd.PrimRange): The range of prims to traverse.
        transform (Gf.Matrix4d): The transform matrix to apply to each prim.

    Returns:
        None
    """
    for prim in prim_range:
        if not prim.IsA(UsdGeom.Xformable):
            continue
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            continue
        local_transform = xformable.GetLocalTransformation()
        new_transform = local_transform * transform
        transform_op = xformable.AddTransformOp()
        transform_op.Set(new_transform)


def find_prims_with_fallback_schema(stage: Usd.Stage) -> List[Usd.Prim]:
    """
    Find all prims on the given stage that have a fallback schema type.

    A prim has a fallback schema type if its prim type is not a recognized
    concrete schema, but the stage provides a fallback type for it.

    Returns:
        List[Usd.Prim]: A list of prims with fallback schema types.
    """
    prims_with_fallback = []
    for prim in stage.TraverseAll():
        prim_type_info = prim.GetPrimTypeInfo()
        if prim_type_info.GetTypeName() != prim_type_info.GetSchemaTypeName():
            prims_with_fallback.append(prim)
    return prims_with_fallback


def get_full_prim_type_info(prim: Usd.Prim) -> Usd.PrimTypeInfo:
    """Get the full PrimTypeInfo for a prim."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    prim_type_info = prim.GetPrimTypeInfo()
    if not prim_type_info:
        raise ValueError(f"Could not get PrimTypeInfo for prim {prim.GetPath()}.")
    return prim_type_info


def apply_api_schemas_to_prims(stage: Usd.Stage, api_schemas: List[Tuple[str, str]]) -> None:
    """Apply the given API schemas to the prims with matching paths on the stage.

    Args:
        stage (Usd.Stage): The USD stage to apply the API schemas to.
        api_schemas (List[Tuple[str, str]]): A list of tuples, where each tuple contains the prim path and the API schema name to apply.
    """
    for prim_path, schema_name in api_schemas:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Prim at path {prim_path} does not exist. Skipping...")
            continue
        prim_type_info = prim.GetPrimTypeInfo()
        applied_schemas = prim_type_info.GetAppliedAPISchemas()
        if schema_name in applied_schemas:
            print(f"API schema {schema_name} is already applied to prim at path {prim_path}. Skipping...")
            continue
        if schema_name == "ModelAPI":
            UsdGeom.ModelAPI.Apply(prim)
        elif schema_name == "MaterialBindingAPI":
            UsdShade.MaterialBindingAPI.Apply(prim)
        else:
            print(f"Unsupported API schema {schema_name}. Skipping...")


def set_property_display_metadata(
    property: Usd.Property,
    display_name: Optional[str] = None,
    display_group: Optional[str] = None,
    nested_display_groups: Optional[List[str]] = None,
) -> bool:
    """Sets display metadata for a property.

    Args:
        property (Usd.Property): The property to set display metadata on.
        display_name (str, optional): The display name to set. If None, the display name is cleared.
        display_group (str, optional): The display group to set. If None, the display group is cleared.
        nested_display_groups (List[str], optional): The nested display groups to set. If None, the nested display groups are cleared.

    Returns:
        bool: True if setting the metadata succeeded, False otherwise.
    """
    if not property.IsDefined():
        return False
    if display_name is not None:
        success = property.SetDisplayName(display_name)
    else:
        success = property.ClearDisplayName()
    if not success:
        return False
    if display_group is not None:
        success = property.SetDisplayGroup(display_group)
    else:
        success = property.ClearDisplayGroup()
    if not success:
        return False
    if nested_display_groups is not None:
        success = property.SetNestedDisplayGroups(nested_display_groups)
    else:
        success = property.SetNestedDisplayGroups([])
    return success


def copy_property_metadata(source_prop: Usd.Property, dest_prop: Usd.Property) -> None:
    """Copy metadata from one property to another."""
    if not source_prop.IsDefined():
        raise ValueError("Source property is not defined.")
    if not dest_prop.IsDefined():
        raise ValueError("Destination property is not defined.")
    metadata_fields = source_prop.GetAllMetadata()
    for key, value in metadata_fields.items():
        if key == "documentation":
            continue
        dest_prop.SetMetadata(key, value)
        if dest_prop.GetMetadata(key) != value:
            raise ValueError(f"Failed to set metadata field '{key}' on destination property.")


def get_property_base_and_namespace(property: Usd.Property) -> Tuple[str, str]:
    """
    Get the base name and namespace of a property.

    Args:
        property (Usd.Property): The property to get the base name and namespace from.

    Returns:
        Tuple[str, str]: A tuple containing the base name and namespace of the property.
    """
    if not property.IsDefined():
        raise ValueError("The provided property is not defined.")
    full_name = property.GetName()
    name_parts = full_name.split(":")
    base_name = name_parts[-1]
    namespace = ":".join(name_parts[:-1])
    return (base_name, namespace)


def set_custom_property(prop: Usd.Property, is_custom: bool) -> bool:
    """Set the value for custom on the given property.

    Args:
        prop (Usd.Property): The property to set the custom flag on.
        is_custom (bool): Whether the property should be marked as custom.

    Returns:
        bool: True if the value was set successfully, False otherwise.
    """
    if not prop.IsDefined():
        return False
    edit_target = prop.GetStage().GetEditTarget()
    if prop.IsAuthoredAt(edit_target):
        return prop.SetCustom(is_custom)
    else:
        spec = edit_target.CreateSpec(prop.GetPath())
        if spec:
            spec.SetCustom(is_custom)
            return True
        else:
            return False


def get_property_stack_with_offsets(
    property: Usd.Property, time: Usd.TimeCode
) -> List[Tuple[Sdf.PropertySpec, Sdf.LayerOffset]]:
    """
    Get the property stack with layer offsets for a given property at a specific time.

    Args:
        property (Usd.Property): The property to get the stack for.
        time (Usd.TimeCode): The time at which to get the property stack.

    Returns:
        List[Tuple[Sdf.PropertySpec, Sdf.LayerOffset]]: A list of tuples containing the property specs and their corresponding layer offsets.
    """
    stack_with_offsets = property.GetPropertyStackWithLayerOffsets(time)
    if not stack_with_offsets:
        return []
    result = []
    for spec, offset in stack_with_offsets:
        if not spec:
            continue
        result.append((spec, offset))
    return result


def clear_property_display_metadata(prop: Usd.Property) -> bool:
    """Clear the display name and display group metadata for a property."""
    display_name_cleared = prop.ClearDisplayName()
    display_group_cleared = prop.ClearDisplayGroup()
    return display_name_cleared and display_group_cleared


def retrieve_nested_display_groups(prop: Usd.Property) -> List[str]:
    """Retrieve the nested display groups for a USD property.

    Args:
        prop (Usd.Property): The USD property to retrieve display groups for.

    Returns:
        List[str]: The list of nested display groups. Returns an empty list if no display groups are authored.
    """
    if not prop.HasAuthoredDisplayGroup():
        return []
    display_group = prop.GetDisplayGroup()
    if not display_group:
        return []
    return display_group.split(":")


def batch_set_display_names(properties: List[Usd.Property], display_names: List[str]) -> bool:
    """
    Sets the display name (metadata) for each property in the given list.

    Args:
        properties (List[Usd.Property]): The list of properties to set display names for.
        display_names (List[str]): The corresponding list of display names to set.

    Returns:
        bool: True if all display names were set successfully, False otherwise.
    """
    if len(properties) != len(display_names):
        print("Error: The number of properties and display names must be equal.")
        return False
    for prop, display_name in zip(properties, display_names):
        if not prop.IsDefined():
            print(f"Error: Property {prop.GetPath()} is not defined.")
            return False
        if not prop.SetDisplayName(display_name):
            print(f"Error: Failed to set display name for property {prop.GetPath()}.")
            return False
    return True


def batch_clear_display_groups(properties: list[Usd.Property]) -> dict[Usd.Property, bool]:
    """Clear the display group for multiple properties in a batch.

    Args:
        properties (list[Usd.Property]): List of properties to clear display group for.

    Returns:
        dict[Usd.Property, bool]: Dictionary mapping each property to the success status
        of clearing its display group.
    """
    results = {}
    for prop in properties:
        if not prop.IsDefined():
            results[prop] = False
            continue
        success = prop.ClearDisplayGroup()
        results[prop] = success
    return results


def get_all_custom_properties(prim: Usd.Prim) -> List[Usd.Property]:
    """Get all custom properties on a prim.

    Args:
        prim (Usd.Prim): The prim to get custom properties from.

    Returns:
        List[Usd.Property]: A list of all custom properties on the prim.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    properties = prim.GetProperties()
    custom_properties = [prop for prop in properties if prop.IsCustom()]
    return custom_properties


def flatten_property_to(property: Usd.Property, dest_property: Usd.Property) -> Usd.Property:
    """
    Flattens the property onto the destination property.

    This will copy all authored metadata and values from the given property
    onto the destination property, recursively if the property is a
    namespace property. This does not flatten properties from the property's
    namespace ancestors.

    If the destination property already exists, this method will replace all
    authored metadata and values on the destination property.
    """
    if not property.IsDefined():
        raise ValueError("The source property is not defined.")
    if not dest_property.IsDefined():
        raise ValueError("The destination property is not defined.")
    spec_type = type(property.GetPropertyStack(Usd.TimeCode.Default())[0])
    edit_target = dest_property.GetStage().GetEditTarget()
    for key, value in property.GetAllMetadata().items():
        dest_property.SetMetadata(key, value)
    if isinstance(property, Usd.Attribute):
        if property.GetNamespace():
            for child_prop in property.GetChildren():
                dest_child_prop = dest_property.GetStage().OverridePrim(
                    dest_property.GetPrim().GetPath().AppendProperty(child_prop.GetName())
                )
                flatten_property_to(child_prop, dest_child_prop)
        elif property.HasValue():
            (value, time_samples) = (property.Get(Usd.TimeCode.Default()), {})
            for sample in property.GetTimeSamples():
                time_samples[sample] = property.Get(sample)
            if time_samples:
                dest_property.Set(value, Usd.TimeCode.Default())
                for time, sample_val in time_samples.items():
                    dest_property.Set(sample_val, time)
            else:
                dest_property.Set(value)
    elif isinstance(property, Usd.Relationship):
        targets = property.GetTargets()
        dest_property.SetTargets(targets)
    return dest_property


def batch_set_custom(properties: List[Usd.Property], is_custom: bool) -> List[bool]:
    """
    Batch set the custom flag on a list of properties.

    Args:
        properties (List[Usd.Property]): List of properties to set the custom flag on.
        is_custom (bool): Value to set for the custom flag.

    Returns:
        List[bool]: List of boolean values indicating success for each property.
    """
    results = []
    for prop in properties:
        if not prop.IsDefined():
            results.append(False)
            continue
        success = prop.SetCustom(is_custom)
        results.append(success)
    return results


def get_property_namespace_hierarchy(prop: Usd.Property) -> List[str]:
    """Returns the namespace hierarchy of the given property as a list of strings."""
    if not prop.IsValid():
        raise ValueError("Invalid property")
    full_name = prop.GetName()
    components = full_name.split(":")
    namespace_hierarchy = components[:-1]
    return namespace_hierarchy


def get_all_property_namespaces(prop: Usd.Property) -> List[str]:
    """
    Returns a list of all namespaces for the given property.

    Args:
        prop (Usd.Property): The property to get the namespaces for.

    Returns:
        List[str]: A list of all namespaces for the property.
    """
    name_parts = prop.SplitName()
    name_parts = name_parts[:-1]
    namespaces = []
    namespace = ""
    for part in name_parts:
        if namespace:
            namespace += ":"
        namespace += part
        namespaces.append(namespace)
    return namespaces


def validate_property_definition(property_: Usd.Property) -> bool:
    """Validate if a property is properly defined.

    A property is considered properly defined if it meets the following criteria:
    1. The property is defined (IsDefined returns True)
    2. If the property is a custom property, it must be marked as custom
    3. The property must have an authored value or metadata

    Args:
        property_ (Usd.Property): The property to validate.

    Returns:
        bool: True if the property is properly defined, False otherwise.
    """
    if not property_.IsDefined():
        return False
    if property_.IsCustom() and (not property_.GetMetadata("custom")):
        return False
    if not property_.IsAuthored():
        return False
    return True


def get_all_defined_properties(prim: Usd.Prim) -> List[Usd.Property]:
    """
    Get all defined properties on a given prim.

    Args:
        prim (Usd.Prim): The prim to get defined properties for.

    Returns:
        List[Usd.Property]: A list of all defined properties on the prim.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    properties = prim.GetProperties()
    defined_properties = [prop for prop in properties if prop.IsDefined()]
    return defined_properties


def retrieve_property_stack(prop: Usd.Property, time: Usd.TimeCode = Usd.TimeCode.Default()) -> List[Sdf.PropertySpec]:
    """Retrieve the property stack for a given property at a specific time.

    Args:
        prop (Usd.Property): The property to retrieve the stack for.
        time (Usd.TimeCode, optional): The time at which to retrieve the stack. Defaults to Usd.TimeCode.Default().

    Returns:
        List[Sdf.PropertySpec]: The property stack, ordered from strongest to weakest opinion.
    """
    if not prop.IsDefined():
        raise ValueError("The provided property is not defined.")
    property_stack = prop.GetPropertyStack(time)
    if not property_stack:
        raise ValueError("The property stack is empty.")
    return property_stack


def get_all_properties_with_metadata(prim: Usd.Prim) -> List[Tuple[Usd.Property, Sdf.PropertySpec]]:
    """
    Returns a list of tuples containing all properties on the given prim along with their property spec.

    Args:
        prim (Usd.Prim): The prim to retrieve properties and metadata from.

    Returns:
        List[Tuple[Usd.Property, Sdf.PropertySpec]]: A list of tuples containing properties and their specs.
        The list will be empty if the prim is invalid or has no properties.
    """
    if not prim.IsValid():
        return []
    properties_with_metadata = []
    for prop in prim.GetProperties():
        prop_spec = prop.GetPropertyStack(Usd.TimeCode.Default())[-1]
        if prop_spec:
            properties_with_metadata.append((prop, prop_spec))
    return properties_with_metadata


def analyze_property_hierarchy(prop: Usd.Property) -> Dict[str, List[str]]:
    """
    Analyze the property hierarchy for the given property.

    Returns a dictionary where keys are the namespace levels and
    values are lists of property names at that level.
    """
    result: Dict[str, List[str]] = {}
    name_parts = prop.SplitName()
    cur_ns = ""
    for part in name_parts[:-1]:
        cur_ns = f"{cur_ns}/{part}" if cur_ns else part
        if cur_ns not in result:
            result[cur_ns] = []
    if cur_ns in result:
        result[cur_ns].append(name_parts[-1])
    else:
        result[""] = [name_parts[-1]]
    return result


def set_references_for_prims(
    stage: Usd.Stage, prim_paths: list[str], reference_path: str, layer_offset: Sdf.LayerOffset = Sdf.LayerOffset()
) -> bool:
    """Set the same reference on multiple prims.

    Args:
        stage (Usd.Stage): The stage to modify.
        prim_paths (list[str]): The paths of the prims to set the reference on.
        reference_path (str): The path to the referenced layer.
        layer_offset (Sdf.LayerOffset, optional): The layer offset to apply to the reference. Defaults to Sdf.LayerOffset().

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    ref = Sdf.Reference(reference_path, layerOffset=layer_offset)
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            print(f"Error: Prim at path {prim_path} does not exist.")
            return False
        references = prim.GetReferences()
        if not references.SetReferences([ref]):
            print(f"Error: Failed to set reference on prim at path {prim_path}.")
            return False
    return True


def add_internal_reference_to_prims(
    stage: Usd.Stage,
    prim_paths: List[str],
    reference_prim_path: str,
    position: Usd.ListPosition = Usd.ListPositionFrontOfPrependList,
) -> bool:
    """Add an internal reference to the specified prims.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (List[str]): The paths of the prims to add the reference to.
        reference_prim_path (str): The path of the prim being referenced.
        position (Usd.ListPosition, optional): The position to insert the reference at. Defaults to ListPositionFrontOfPrependList.

    Returns:
        bool: True if the reference was added successfully to all prims, False otherwise.
    """
    if not stage.GetPrimAtPath(reference_prim_path):
        print(f"Error: Referenced prim path {reference_prim_path} does not exist on the stage.")
        return False
    all_success = True
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            print(f"Error: Prim path {prim_path} does not exist on the stage. Skipping.")
            all_success = False
            continue
        success = prim.GetReferences().AddInternalReference(reference_prim_path, Sdf.LayerOffset(), position)
        if not success:
            print(f"Error: Failed to add internal reference to prim {prim_path}.")
            all_success = False
    return all_success


def create_relationship(prim: Usd.Prim, relationship_name: str, target_paths: list[str]) -> Usd.Relationship:
    """Create a relationship on a prim with the given name and target paths.

    Args:
        prim (Usd.Prim): The prim to create the relationship on.
        relationship_name (str): The name of the relationship to create.
        target_paths (list[str]): The target paths for the relationship.

    Returns:
        Usd.Relationship: The created relationship.

    Raises:
        ValueError: If the prim is not valid or if any of the target paths are invalid.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    relationship = prim.CreateRelationship(relationship_name)
    sdf_paths = [Sdf.Path(path) for path in target_paths]
    success = relationship.SetTargets(sdf_paths)
    if not success:
        raise ValueError(f"Failed to set targets on relationship: {relationship_name}")
    return relationship


def copy_relationship(source_prim: Usd.Prim, dest_prim: Usd.Prim, relationship_name: str) -> bool:
    """Copy a relationship from one prim to another.

    Args:
        source_prim (Usd.Prim): The prim to copy the relationship from.
        dest_prim (Usd.Prim): The prim to copy the relationship to.
        relationship_name (str): The name of the relationship to copy.

    Returns:
        bool: True if the relationship was successfully copied, False otherwise.
    """
    if not source_prim.IsValid():
        print(f"Error: Source prim '{source_prim.GetPath()}' is not valid.")
        return False
    if not dest_prim.IsValid():
        print(f"Error: Destination prim '{dest_prim.GetPath()}' is not valid.")
        return False
    if not source_prim.HasRelationship(relationship_name):
        print(f"Error: Source prim '{source_prim.GetPath()}' does not have relationship '{relationship_name}'.")
        return False
    source_rel = source_prim.GetRelationship(relationship_name)
    targets = source_rel.GetTargets()
    dest_rel = dest_prim.CreateRelationship(relationship_name)
    dest_rel.SetTargets(targets)
    return True


def clear_relationship_targets(relationship: Usd.Relationship, remove_spec: bool = False) -> bool:
    """Clear all target paths from the given relationship.

    Args:
        relationship (Usd.Relationship): The relationship to clear targets from.
        remove_spec (bool, optional): If True, remove the relationship spec entirely.
            Defaults to False.

    Returns:
        bool: True if the operation succeeded, False otherwise.
    """
    if not relationship.IsValid():
        raise ValueError("Invalid relationship.")
    if not relationship.HasAuthoredTargets():
        return True
    try:
        success = relationship.ClearTargets(remove_spec)
        return success
    except Tf.ErrorException as e:
        print(f"Error clearing targets: {e}")
        return False


def check_relationship_consistency(relationship: Usd.Relationship) -> bool:
    """Check if a relationship has consistent targets.

    A relationship is considered consistent if all its targets are either prims,
    properties (attributes or relationships), or a mix of both. It is
    inconsistent if it contains a mix of prims and properties.

    Args:
        relationship (Usd.Relationship): The relationship to check for consistency.

    Returns:
        bool: True if the relationship has consistent targets, False otherwise.
    """
    targets = relationship.GetTargets()
    if not targets:
        return True
    target_types = set((target.GetPrimPath() == target for target in targets))
    is_consistent = len(target_types) == 1
    return is_consistent


def get_all_relationship_targets(prim: Usd.Prim, relationship_name: str) -> List[Usd.Prim]:
    """
    Get all target prims of a relationship on a given prim.

    Args:
        prim (Usd.Prim): The prim to query the relationship on.
        relationship_name (str): The name of the relationship to query.

    Returns:
        List[Usd.Prim]: A list of target prims for the relationship.

    Raises:
        ValueError: If the prim is not valid or the relationship does not exist.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim '{prim.GetPath()}' is not valid.")
    relationship = prim.GetRelationship(relationship_name)
    if not relationship:
        raise ValueError(f"Relationship '{relationship_name}' does not exist on prim '{prim.GetPath()}'.")
    target_paths = relationship.GetTargets()
    target_prims = []
    for target_path in target_paths:
        target_prim = prim.GetStage().GetPrimAtPath(target_path)
        if target_prim.IsValid():
            target_prims.append(target_prim)
    return target_prims


def add_relationship_targets(
    relationship: Usd.Relationship, targets: List[Sdf.Path], position=Usd.ListPositionFrontOfAppendList
) -> bool:
    """Add target paths to a relationship.

    Args:
        relationship (Usd.Relationship): The relationship to add targets to.
        targets (List[Sdf.Path]): The target paths to add.
        position (Usd.ListPosition, optional): The position to add the targets at. Defaults to ListPositionFrontOfAppendList.

    Returns:
        bool: True if the targets were added successfully, False otherwise.
    """
    if not relationship.IsValid():
        print(f"Error: Invalid relationship {relationship.GetPath()}")
        return False
    for target in targets:
        if not target.IsPrimPath() and (not target.IsPropertyPath()):
            print(f"Error: Invalid target path {target}")
            return False
    for target in targets:
        success = relationship.AddTarget(target, position)
        if not success:
            print(f"Error: Failed to add target {target} to relationship {relationship.GetPath()}")
            return False
    return True


def is_attribute_value_blocked(attribute: Usd.Attribute) -> bool:
    """Check if an attribute's value is blocked."""
    resolve_info = attribute.GetResolveInfo()
    is_blocked = resolve_info.ValueIsBlocked()
    return is_blocked


def get_resolved_attribute_value(attribute: Usd.Attribute) -> Tuple[Any, Usd.ResolveInfo]:
    """
    Get the resolved value and resolve info for a given attribute.

    Args:
        attribute (Usd.Attribute): The attribute to resolve.

    Returns:
        Tuple[Any, Usd.ResolveInfo]: A tuple containing the resolved value and the resolve info.
    """
    resolved_value = attribute.Get()
    resolve_info = attribute.GetResolveInfo()
    if resolve_info.ValueIsBlocked():
        raise RuntimeError(f"Attribute {attribute.GetPath()} is blocked.")
    source = resolve_info.GetSource()
    node = resolve_info.GetNode()
    return (resolved_value, resolve_info)


def resolve_attribute_source(attribute: Usd.Attribute) -> Tuple[Usd.ResolveInfoSource, Pcp.NodeRef, bool]:
    """
    Resolves the source of an attribute's value.

    Args:
        attribute (Usd.Attribute): The attribute to resolve the source for.

    Returns:
        Tuple[Usd.ResolveInfoSource, Pcp.NodeRef, bool]: A tuple containing the resolved source,
        the node that provided the value opinion, and a boolean indicating if the value is blocked.
    """
    resolve_info = attribute.GetResolveInfo()
    source = resolve_info.GetSource()
    node = resolve_info.GetNode()
    is_blocked = resolve_info.ValueIsBlocked()
    return (source, node, is_blocked)


def resolve_and_get_property_value(prim: Usd.Prim, property_name: str) -> Usd.ResolveInfo:
    """Resolves a property on a prim and returns the resolved value and source info.

    Args:
        prim (Usd.Prim): The prim to resolve the property on.
        property_name (str): The name of the property to resolve.

    Returns:
        Usd.ResolveInfo: The resolved value and source info of the property.

    Raises:
        ValueError: If the property does not exist on the prim.
    """
    if not prim.HasProperty(property_name):
        raise ValueError(f"Property '{property_name}' does not exist on prim '{prim.GetPath()}'.")
    prop = prim.GetProperty(property_name)
    resolve_info = prop.GetResolveInfo()
    source = resolve_info.GetSource()
    if source == Usd.ResolveInfoSourceValueClips:
        resolved_value = resolve_info.GetNode().GetClips().GetClipAssetPaths()
    elif source == Usd.ResolveInfoSourceFallback:
        resolved_value = resolve_info.GetNode().GetFallbackValue()
    elif source == Usd.ResolveInfoSourceTimeSamples:
        resolved_value = resolve_info.GetNode().GetTimeSamples()
    elif source == Usd.ResolveInfoSourceDefault:
        resolved_value = prop.Get()
    elif source == Usd.ResolveInfoSourceNone:
        resolved_value = None
    else:
        resolved_value = None
    return (resolved_value, resolve_info)


def create_resolve_target_from_edit_target(prim: Usd.Prim, edit_target: Usd.EditTarget) -> Usd.ResolveTarget:
    """
    Create a resolve target for the given prim that will limit value resolution
    to up to the spec that would be edited when setting a value on the prim using
    the given edit target.

    Args:
        prim (Usd.Prim): The prim to create the resolve target for.
        edit_target (Usd.EditTarget): The edit target that defines the limit for value resolution.

    Returns:
        Usd.ResolveTarget: The created resolve target, or a null resolve target if the prim is invalid.
    """
    if not prim.IsValid():
        return Usd.ResolveTarget()
    edit_layer = edit_target.GetLayer()
    if not edit_layer:
        return Usd.ResolveTarget()
    resolve_target = prim.MakeResolveTargetUpToEditTarget(edit_target)
    return resolve_target


def create_attribute_query_with_resolve_target(
    prim: Usd.Prim, attribute_name: str, resolve_target: Usd.ResolveTarget
) -> Usd.AttributeQuery:
    """Create an attribute query with a specific resolve target.

    Args:
        prim (Usd.Prim): The prim to query the attribute on.
        attribute_name (str): The name of the attribute to query.
        resolve_target (Usd.ResolveTarget): The resolve target to use for the query.

    Returns:
        Usd.AttributeQuery: The attribute query with the specified resolve target.

    Raises:
        ValueError: If the prim is invalid or the attribute does not exist.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    attribute = prim.GetAttribute(attribute_name)
    if not attribute.IsDefined():
        raise ValueError(f"Attribute '{attribute_name}' does not exist on prim '{prim.GetPath()}'")
    query = Usd.AttributeQuery(attribute, resolve_target)
    return query


def compare_schema_attributes(schema1: Usd.SchemaBase, schema2: Usd.SchemaBase) -> bool:
    """Compare the attributes of two schemas.

    Returns True if the schemas have the same attributes, False otherwise.
    """
    attrs1 = schema1.GetSchemaAttributeNames(includeInherited=True)
    attrs2 = schema2.GetSchemaAttributeNames(includeInherited=True)
    if len(attrs1) != len(attrs2):
        return False
    for attr_name in attrs1:
        if not schema2.GetPrim().HasAttribute(attr_name):
            return False
        attr1 = schema1.GetPrim().GetAttribute(attr_name)
        attr2 = schema2.GetPrim().GetAttribute(attr_name)
        if attr1.Get() != attr2.Get():
            return False
    return True


def transfer_schema(src_prim: Usd.Prim, dest_prim: Usd.Prim) -> bool:
    """Transfer the schema from one prim to another."""
    if not src_prim.IsValid():
        raise ValueError("Source prim is not valid.")
    if not dest_prim.IsValid():
        raise ValueError("Destination prim is not valid.")
    src_schema = src_prim.GetTypeName()
    if src_schema == "":
        return False
    try:
        if src_schema == "Sphere":
            UsdGeom.Sphere(dest_prim)
        elif src_schema == "Cube":
            UsdGeom.Cube(dest_prim)
        elif src_schema == "Cone":
            UsdGeom.Cone(dest_prim)
        else:
            raise ValueError(f"Unsupported schema type: {src_schema}")
    except Tf.ErrorException as e:
        print(f"Error applying schema: {e}")
        return False
    for src_attr in src_prim.GetAttributes():
        dest_attr = dest_prim.GetAttribute(src_attr.GetName())
        if dest_attr.IsValid():
            if src_attr.HasValue():
                value = src_attr.Get()
                dest_attr.Set(value)
    return True


def find_prims_by_schema(stage: Usd.Stage, schema_class: Usd.SchemaBase) -> List[Usd.Prim]:
    """Find all prims on the stage that have the specified schema applied.

    Args:
        stage (Usd.Stage): The stage to search for prims.
        schema_class (Usd.SchemaBase): The schema class to match prims against.

    Returns:
        List[Usd.Prim]: A list of prims that have the specified schema applied.
    """
    matching_prims: List[Usd.Prim] = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        if prim.IsA(schema_class):
            matching_prims.append(prim)
    return matching_prims


def copy_schema_attributes(
    source_schema: UsdGeom.Xformable, dest_schema: UsdGeom.Xformable, skip_attrs: List[str] = None
) -> None:
    """Copy attributes from one schema to another.

    Args:
        source_schema (UsdGeom.Xformable): The source schema to copy attributes from.
        dest_schema (UsdGeom.Xformable): The destination schema to copy attributes to.
        skip_attrs (List[str], optional): List of attribute names to skip copying. Defaults to None.

    Raises:
        ValueError: If source_schema or dest_schema is invalid.
    """
    if not source_schema or not source_schema.GetPrim().IsValid():
        raise ValueError("Invalid source schema.")
    if not dest_schema or not dest_schema.GetPrim().IsValid():
        raise ValueError("Invalid destination schema.")
    attr_names = [attr.GetName() for attr in source_schema.GetPrim().GetAttributes()]
    for attr_name in attr_names:
        if skip_attrs and attr_name in skip_attrs:
            continue
        source_attr = source_schema.GetPrim().GetAttribute(attr_name)
        if not source_attr.HasValue():
            continue
        attr_value = source_attr.Get()
        dest_attr = dest_schema.GetPrim().CreateAttribute(attr_name, source_attr.GetTypeName())
        dest_attr.Set(attr_value)


def validate_schema_kind(schema_kind: Usd.SchemaKind) -> bool:
    """Validate if a given SchemaKind is valid.

    Args:
        schema_kind (Usd.SchemaKind): The SchemaKind to validate.

    Returns:
        bool: True if the SchemaKind is valid, False otherwise.
    """
    if schema_kind not in (
        Usd.SchemaKind.AbstractBase,
        Usd.SchemaKind.AbstractTyped,
        Usd.SchemaKind.ConcreteTyped,
        Usd.SchemaKind.NonAppliedAPI,
    ):
        return False
    return True


def assign_schema_kind_to_prims(stage: Usd.Stage, schema_kind: str, prim_paths: List[str]) -> None:
    """Assign a schema kind to a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        schema_kind (str): The schema kind to assign as a string.
        prim_paths (List[str]): The list of prim paths to assign the schema kind to.

    Raises:
        ValueError: If any of the prim paths are invalid or the prims don't exist.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")
        prim.SetMetadata("kind", schema_kind)


def copy_prims_with_schema_kind(
    source_prim: Usd.Prim, destination_prim: Usd.Prim, schema_kind: Usd.SchemaKind
) -> List[Usd.Prim]:
    """Copy prims with a specific schema kind from a source prim to a destination prim.

    Args:
        source_prim (Usd.Prim): The source prim to copy from.
        destination_prim (Usd.Prim): The destination prim to copy to.
        schema_kind (Usd.SchemaKind): The schema kind to filter prims by.

    Returns:
        List[Usd.Prim]: A list of the copied prims.
    """
    copied_prims = []
    for descendant in Usd.PrimRange(source_prim):
        applied_schemas = descendant.GetAppliedSchemas()
        if applied_schemas:
            if applied_schemas[0].GetSchemaKind() == schema_kind:
                new_path = destination_prim.GetPath().AppendPath(
                    descendant.GetPath().MakeRelativePath(source_prim.GetPath())
                )
                copied_prim = descendant.CopyTo(descendant.GetStage(), new_path)
                copied_prims.append(copied_prim)
    return copied_prims


def find_prims_with_schema(stage: Usd.Stage, schema_type: Tf.Type) -> List[Usd.Prim]:
    """Find all prims on the stage with the given schema type."""
    if not stage:
        raise ValueError("Invalid stage")
    if not schema_type:
        raise ValueError("Invalid schema type")
    prims_with_schema: List[Usd.Prim] = []
    for prim in stage.Traverse():
        if prim.IsA(schema_type):
            prims_with_schema.append(prim)
    return prims_with_schema


def get_type_from_name(type_name: str) -> Tf.Type:
    """
    Find the TfType of a schema with the given type name.

    Args:
        type_name (str): The name of the schema type.

    Returns:
        Tf.Type: The TfType of the schema, or Tf.Type.Unknown if not found.
    """
    tf_type = Tf.Type.FindByName(type_name)
    if not tf_type:
        geom_type_name = f"UsdGeom{type_name}"
        tf_type = Tf.Type.FindByName(geom_type_name)
    if not tf_type:
        return Tf.Type.Unknown
    return tf_type


def is_abstract_schema(schema_type_name: str) -> bool:
    """Returns True if the given schema type name is an abstract type, False otherwise."""
    schema_type = Usd.SchemaRegistry.GetTypeFromSchemaTypeName(schema_type_name)
    if not schema_type:
        raise ValueError(f"{schema_type_name} is not a registered schema type")
    is_abstract = Usd.SchemaRegistry.IsAbstract(schema_type)
    return is_abstract


def get_schema_instance_name(schema_name: str) -> Tuple[str, str]:
    """
    Get the schema type name and instance name from a multiple apply schema name.

    Args:
        schema_name (str): The name of the multiple apply schema.

    Returns:
        Tuple[str, str]: A tuple containing the schema type name and instance name.
    """
    tokens = schema_name.split(":")
    if len(tokens) < 2:
        return (schema_name, "")
    instance_name = tokens[-1]
    type_name = ":".join(tokens[:-1])
    return (type_name, instance_name)


def get_applicable_prim_types_for_schema(schema_type_name: str) -> List[str]:
    """
    Get the list of prim type names that the given schema type name can be applied to.

    If the schema type is not a valid applied API schema, an empty list is returned.
    If the schema type has no explicit applicability restriction, an empty list is returned,
    indicating it can be applied to any prim type.
    """
    if not Usd.SchemaRegistry.IsAppliedAPISchema(schema_type_name):
        return []
    prim_type_names = Usd.SchemaRegistry.GetAPISchemaCanOnlyApplyToTypeNames(schema_type_name, instanceName="")
    if not prim_type_names:
        return []
    prim_type_names = [str(name) for name in prim_type_names]
    expanded_prim_type_names = set(prim_type_names)
    for prim_type_name in prim_type_names:
        prim_type = Usd.SchemaRegistry.GetTypeFromSchemaTypeName(prim_type_name)
        if prim_type:
            derived_types = prim_type.GetAllDerivedTypes()
            derived_type_names = [
                str(Usd.SchemaRegistry.GetConcreteSchemaTypeName(derived_type)) for derived_type in derived_types
            ]
            expanded_prim_type_names.update(derived_type_names)
    return list(expanded_prim_type_names)


def validate_schema_application(prim: Usd.Prim, schema_type: Tf.Type) -> bool:
    """Validate that a prim has a specific schema applied.

    Args:
        prim (Usd.Prim): The prim to validate.
        schema_type (Tf.Type): The type of the schema to check for.

    Returns:
        bool: True if the prim has the schema applied, False otherwise.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    schema_type_name = Usd.SchemaRegistry.GetAPISchemaTypeName(schema_type)
    if not Usd.SchemaRegistry.IsAppliedAPISchema(schema_type_name):
        raise ValueError(f"{schema_type} is not a valid applied API schema")
    if schema_type_name in prim.GetAppliedSchemas():
        return True
    else:
        return False


def test_validate_schema_application():
    stage = Usd.Stage.CreateInMemory()
    prim_path = "/TestPrim"
    prim = stage.DefinePrim(prim_path)
    schema_type = UsdGeom.MotionAPI
    prim.ApplyAPI(schema_type)
    result = validate_schema_application(prim, schema_type)
    print(result)
    result = validate_schema_application(prim, UsdGeom.ModelAPI)
    print(result)
    invalid_prim = Usd.Prim()
    try:
        validate_schema_application(invalid_prim, schema_type)
    except ValueError as e:
        print(f"Caught expected exception: {str(e)}")
    invalid_schema_type = UsdGeom.Mesh
    try:
        validate_schema_application(prim, invalid_schema_type)
    except ValueError as e:
        print(f"Caught expected exception: {str(e)}")


def get_prim_schema_kind(prim_type_name: str) -> Tf.Type:
    """Get the schema kind for a given prim type name."""
    schema_kind = Usd.SchemaRegistry.GetSchemaKind(prim_type_name)
    if schema_kind == Usd.SchemaKind.Invalid:
        raise ValueError(f"Invalid schema kind for prim type {prim_type_name}")
    return schema_kind


def remove_schema_from_prim(prim: Usd.Prim, schema_type: Type) -> bool:
    """Remove a schema from a prim.

    Args:
        prim (Usd.Prim): The prim to remove the schema from.
        schema_type (Type): The type of the schema to remove.

    Returns:
        bool: True if the schema was successfully removed, False otherwise.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if not prim.HasAPI(schema_type):
        print(f"Prim {prim.GetPath()} does not have schema {schema_type.__name__}")
        return False
    prim.RemoveAPI(schema_type)
    if prim.HasAPI(schema_type):
        raise RuntimeError(f"Failed to remove schema {schema_type.__name__} from prim {prim.GetPath()}")
    return True


def is_prim_of_schema_type(prim: Usd.Prim, schema_type_name: str) -> bool:
    """Check if the given prim is of the specified schema type or derives from it.

    Args:
        prim (Usd.Prim): The prim to check the schema type of.
        schema_type_name (str): The name of the schema type to check for.

    Returns:
        bool: True if the prim is of the specified schema type or derives from it, False otherwise.
    """
    schema_type = Tf.Type.FindByName(schema_type_name)
    if not schema_type:
        raise ValueError(f"Invalid schema type name: {schema_type_name}")
    return prim.IsA(schema_type)


def create_prim_with_multiple_schemas(
    stage: Usd.Stage, prim_path: str, prim_type: str, applied_schemas: List[str]
) -> Usd.Prim:
    """Create a prim with the specified type and applied API schemas.

    Args:
        stage (Usd.Stage): The stage to create the prim on.
        prim_path (str): The path of the prim to create.
        prim_type (str): The type of the prim to create.
        applied_schemas (List[str]): A list of API schemas to apply to the prim.

    Returns:
        Usd.Prim: The created prim.

    Raises:
        ValueError: If the prim type or any of the applied schemas are invalid.
    """
    if not Usd.SchemaRegistry().FindConcretePrimDefinition(prim_type):
        raise ValueError(f"Invalid prim type: {prim_type}")
    prim = stage.DefinePrim(prim_path, prim_type)
    for schema_name in applied_schemas:
        schema_type = Usd.SchemaRegistry().GetTypeFromName(schema_name)
        if not schema_type or not Usd.SchemaRegistry().IsAppliedAPISchema(schema_name):
            raise ValueError(f"Invalid applied API schema: {schema_name}")
        prim.ApplyAPI(schema_type)
    return prim


def get_concrete_prim_definition(schema_registry: Usd.SchemaRegistry, prim_type_name: str) -> Usd.SchemaBase:
    """Get the concrete prim definition for a given prim type name."""
    if not isinstance(prim_type_name, str):
        raise TypeError(f"Expected prim type name to be a string, got {type(prim_type_name)}")
    prim_definition = schema_registry.FindConcretePrimDefinition(prim_type_name)
    if prim_definition is None:
        raise ValueError(f"No concrete prim definition found for prim type '{prim_type_name}'")
    return prim_definition


def create_prim_with_concrete_schema(stage: Usd.Stage, prim_path: str, schema_type_name: str) -> Usd.Prim:
    """Create a prim with a concrete schema type.

    Args:
        stage (Usd.Stage): The USD stage to create the prim on.
        prim_path (str): The path where the prim should be created.
        schema_type_name (str): The name of the concrete schema type to apply.

    Returns:
        Usd.Prim: The created prim.

    Raises:
        ValueError: If the schema type name is not a valid concrete schema type.
    """
    schema_type = Usd.SchemaRegistry.GetConcreteTypeFromSchemaTypeName(schema_type_name)
    if not schema_type:
        raise ValueError(f"{schema_type_name} is not a valid concrete schema type.")
    type_name = Usd.SchemaRegistry.GetSchemaTypeName(schema_type)
    prim = stage.DefinePrim(Sdf.Path(prim_path), type_name)
    return prim


def list_specializes(prim: Usd.Prim) -> List[Sdf.Path]:
    """
    Returns the list of specializes paths for the given prim.

    Args:
        prim (Usd.Prim): The prim to get the specializes paths for.

    Returns:
        List[Sdf.Path]: The list of specializes paths for the prim.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim")
    specializes_attr = prim.GetAttribute("specializes")
    if not specializes_attr.IsValid():
        return []
    specializes_paths = specializes_attr.Get()
    if specializes_paths is None:
        return []
    return list(specializes_paths)


def add_specializes_to_prim(prim: Usd.Prim, specializes_path: str, position=Usd.ListPositionFrontOfAppendList) -> None:
    """Add a specializes to a prim at the given list position.

    Args:
        prim (Usd.Prim): The prim to add the specializes to.
        specializes_path (str): The path of the prim to specialize.
        position (Usd.ListPosition, optional): Where to add the specializes. Defaults to APPEND.

    Raises:
        ValueError: If the prim is not valid or if the specializes path is empty.
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid")
    if not specializes_path:
        raise ValueError("Specializes path cannot be empty")
    path = Sdf.Path(specializes_path)
    success = prim.GetSpecializes().AddSpecialize(path, position)
    if not success:
        raise RuntimeError(f"Failed to add specializes '{specializes_path}' to prim '{prim.GetPath()}'")


def clear_and_set_specializes(prim: Usd.Prim, prim_paths: List[Sdf.Path]) -> bool:
    """Clear any existing specializes and set the given prim paths as specializes.

    Args:
        prim (Usd.Prim): The prim to modify the specializes for.
        prim_paths (List[Sdf.Path]): The list of prim paths to set as specializes.

    Returns:
        bool: True if the operation succeeded, False otherwise.
    """
    specializes = prim.GetSpecializes()
    if not specializes.ClearSpecializes():
        return False
    return specializes.SetSpecializes(prim_paths)


def create_material_with_shaders(stage: Usd.Stage, prim_path: str) -> UsdShade.Material:
    """Create a new material prim with shader prims."""
    if not stage:
        raise ValueError("Invalid stage.")
    if not prim_path:
        raise ValueError("Invalid prim path.")
    existing_prim = stage.GetPrimAtPath(prim_path)
    if existing_prim:
        if UsdShade.Material(existing_prim):
            return UsdShade.Material(existing_prim)
        else:
            raise ValueError(f"A prim already exists at {prim_path} which is not a Material.")
    material_prim = UsdShade.Material.Define(stage, prim_path)
    surface_shader_path = f"{prim_path}/Surface"
    surface_shader = UsdShade.Shader.Define(stage, surface_shader_path)
    surface_shader.CreateIdAttr("UsdPreviewSurface")
    surface_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.8, 0.0, 0.0))
    surface_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    displacement_shader_path = f"{prim_path}/Displacement"
    displacement_shader = UsdShade.Shader.Define(stage, displacement_shader_path)
    displacement_shader.CreateIdAttr("UsdPrimvarReader_float2")
    displacement_shader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    material_prim.CreateSurfaceOutput().ConnectToSource(surface_shader.ConnectableAPI(), "surface")
    material_prim.CreateDisplacementOutput().ConnectToSource(displacement_shader.ConnectableAPI(), "result")
    return material_prim


def generate_collision_meshes(stage: Usd.Stage) -> None:
    """Generate UsdPhysics collision meshes for all meshes on the stage."""
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            mesh_name = prim.GetName()
            collision_mesh_name = f"{mesh_name}_collision"
            mesh_xform = mesh.GetPrim().GetParent()
            if not mesh_xform:
                continue
            collision_mesh_path = mesh_xform.GetPath().AppendChild(collision_mesh_name)
            collision_mesh = UsdPhysics.CollisionMesh.Define(stage, collision_mesh_path)
            collision_mesh.CreateSimulationOwnerRel().SetTargets([mesh.GetPath()])
            mesh_xform_ops = mesh_xform.GetOrderedXformOps()
            for op in mesh_xform_ops:
                op.ApplyTo(collision_mesh)


def apply_depth_of_field(stage: Usd.Stage, camera_path: str, focal_distance: float, f_stop: float) -> None:
    """Apply depth of field settings to a camera.

    Args:
        stage (Usd.Stage): The stage containing the camera prim.
        camera_path (str): The path to the camera prim.
        focal_distance (float): The focal distance in world units.
        f_stop (float): The f-stop value (aperture).

    Raises:
        ValueError: If the camera prim is not valid or is not a UsdGeom.Camera.
    """
    camera_prim = stage.GetPrimAtPath(camera_path)
    if not camera_prim.IsValid():
        raise ValueError(f"Invalid camera prim path: {camera_path}")
    camera = UsdGeom.Camera(camera_prim)
    if not camera:
        raise ValueError(f"Prim at path {camera_path} is not a UsdGeom.Camera")
    camera.GetFocusDistanceAttr().Set(focal_distance)
    camera.GetFStopAttr().Set(f_stop)


def batch_reparent_prims(stage: Usd.Stage, parent_path: str, prim_paths: list[str]) -> None:
    """Batch reparent multiple prims under a new parent prim path.

    If the target parent prim doesn't exist, it will be created.
    If any source prim doesn't exist, it will be skipped.

    Args:
        stage (Usd.Stage): The USD stage.
        parent_path (str): The target parent prim path.
        prim_paths (list[str]): A list of prim paths to reparent.
    """
    parent_prim = stage.GetPrimAtPath(parent_path)
    if not parent_prim:
        parent_prim = stage.DefinePrim(parent_path)
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim:
            prim.GetPath().ReplacePrefix(prim.GetPath().GetParentPath(), parent_prim.GetPath())
        else:
            print(f"Warning: Prim {prim_path} does not exist. Skipping.")


def generate_detail_maps(stage: Usd.Stage, prim_path: str, num_maps: int) -> None:
    """Generate a set of detail maps for a given prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to generate detail maps for.
        num_maps (int): The number of detail maps to generate.

    Raises:
        ValueError: If the prim does not exist or is not a mesh.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    if not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"Prim at path {prim_path} is not a mesh.")
    mesh = UsdGeom.Mesh(prim)
    for i in range(num_maps):
        detail_map_path = f"{prim_path}/DetailMap_{i}"
        detail_map_prim = stage.DefinePrim(detail_map_path, "Mesh")
        detail_map_mesh = UsdGeom.Mesh(detail_map_prim)
        detail_map_mesh.GetPointsAttr().Set(mesh.GetPointsAttr().Get())
        detail_map_mesh.GetFaceVertexCountsAttr().Set(mesh.GetFaceVertexCountsAttr().Get())
        detail_map_mesh.GetFaceVertexIndicesAttr().Set(mesh.GetFaceVertexIndicesAttr().Get())
        detail_map_mesh.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
        detail_map_mesh.GetPurposeAttr().Set("render")
        detail_map_mesh.GetVisibilityAttr().Set("invisible")
        detail_map_prim.GetInherits().AddInherit(prim.GetPath())


def integrate_fluid_simulation(stage: Usd.Stage, prim_path: str, time_code: Usd.TimeCode, dt: float) -> None:
    """Integrate fluid simulation on a prim for one timestep.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.
        time_code (Usd.TimeCode): The current time code.
        dt (float): The timestep for integration.

    Raises:
        ValueError: If the prim is not a valid mesh.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    if not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"Prim at path {prim_path} is not a mesh.")
    mesh = UsdGeom.Mesh(prim)
    points_attr = mesh.GetPointsAttr()
    if not points_attr.HasValue():
        raise ValueError(f"Mesh at path {prim_path} does not have vertex positions.")
    curr_points = points_attr.Get(time_code)
    velocities_attr = mesh.GetVelocitiesAttr()
    if velocities_attr.HasValue():
        velocities = velocities_attr.Get(time_code)
        new_points = [p + v * dt for (p, v) in zip(curr_points, velocities)]
    else:
        new_points = curr_points
    points_attr.Set(new_points, time_code)


def retarget_animation(
    source_prim: Usd.Prim, target_prim: Usd.Prim, attributes: List[str] = None, time_range: Tuple[float, float] = None
) -> None:
    """
    Retarget animation from source_prim to target_prim.

    Args:
        source_prim (Usd.Prim): The source prim with the animation to retarget.
        target_prim (Usd.Prim): The target prim to apply the retargeted animation.
        attributes (List[str], optional): List of attribute names to retarget.
            If None, retarget all animatable attributes. Defaults to None.
        time_range (Tuple[float, float], optional): The time range (start, end)
            to retarget. If None, use the full animation range. Defaults to None.

    Raises:
        ValueError: If source_prim or target_prim is not a valid prim.
    """
    if not source_prim.IsValid():
        raise ValueError("Source prim is not valid")
    if not target_prim.IsValid():
        raise ValueError("Target prim is not valid")
    if attributes is None:
        attributes = [attr.GetName() for attr in source_prim.GetAttributes() if attr.ValueMightBeTimeVarying()]
    if time_range is None:
        time_range = (source_prim.GetStage().GetStartTimeCode(), source_prim.GetStage().GetEndTimeCode())
    for attr_name in attributes:
        source_attr = source_prim.GetAttribute(attr_name)
        target_attr = target_prim.CreateAttribute(attr_name, source_attr.GetTypeName())
        for time_sample in source_attr.GetTimeSamples():
            if time_range[0] <= time_sample <= time_range[1]:
                value = source_attr.Get(time_sample)
                target_attr.Set(value, time_sample)


def batch_edit_prim_metadata(stage: Usd.Stage, prim_paths: List[str], metadata_key: str, metadata_value: Any):
    """Batch edit metadata on a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths.
        metadata_key (str): The metadata key to set.
        metadata_value (Any): The metadata value to set.
    """
    if not stage:
        raise ValueError("Invalid stage")
    if not prim_paths:
        raise ValueError("No prim paths provided")
    if not metadata_key:
        raise ValueError("No metadata key provided")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim '{prim_path}' does not exist. Skipping.")
            continue
        try:
            prim.SetMetadata(metadata_key, metadata_value)
        except Tf.ErrorException as e:
            print(f"Error setting metadata on prim '{prim_path}': {e}")


def duplicate_prim_hierarchy(stage: Usd.Stage, root_prim_path: str, dest_prim_path: str) -> Usd.Prim:
    """
    Duplicate a prim hierarchy from one location to another in the stage.

    Args:
        stage (Usd.Stage): The stage containing the prim hierarchy to duplicate.
        root_prim_path (str): The path of the root prim to duplicate.
        dest_prim_path (str): The destination path for the duplicated prim hierarchy.

    Returns:
        Usd.Prim: The root prim of the duplicated hierarchy.
    """
    source_root_prim = stage.GetPrimAtPath(root_prim_path)
    if not source_root_prim.IsValid():
        raise ValueError(f"Source prim at path {root_prim_path} does not exist.")
    dest_root_prim = stage.DefinePrim(dest_prim_path)
    for source_prim in Usd.PrimRange(source_root_prim):
        dest_prim_path = dest_root_prim.GetPath().AppendPath(
            source_prim.GetPath().MakeRelativePath(source_root_prim.GetPath())
        )
        dest_prim = stage.DefinePrim(dest_prim_path, source_prim.GetTypeName())
        for attr in source_prim.GetAttributes():
            if attr.IsAuthored():
                attr_name = attr.GetName()
                attr_value = attr.Get()
                dest_prim.CreateAttribute(attr_name, attr.GetTypeName()).Set(attr_value)
        for rel in source_prim.GetRelationships():
            if rel.IsAuthored():
                rel_name = rel.GetName()
                rel_targets = rel.GetTargets()
                dest_prim.CreateRelationship(rel_name).SetTargets(rel_targets)
    return dest_root_prim


def perform_advanced_search(stage: Usd.Stage, query: str, exact: bool = False) -> List[Usd.Prim]:
    """Perform an advanced prim search on the stage using a query string.

    Args:
        stage (Usd.Stage): The stage to search.
        query (str): The search query string.
        exact (bool, optional): If True, match the query exactly. Defaults to False.

    Returns:
        List[Usd.Prim]: A list of matching prims.
    """
    terms = query.split()
    matching_prims = []
    for prim in stage.TraverseAll():
        matches_all_terms = True
        for term in terms:
            if exact:
                if term != prim.GetName() and term != prim.GetTypeName():
                    matches_all_terms = False
                    break
            elif term not in prim.GetName() and term not in prim.GetTypeName():
                matches_all_terms = False
                break
        if matches_all_terms:
            matching_prims.append(prim)
    return matching_prims


def transfer_animation_keys(source_prim: Usd.Prim, dest_prim: Usd.Prim, attribute_name: str):
    """
    Transfer animation keys from one prim to another for a specific attribute.

    Args:
        source_prim (Usd.Prim): The source prim to transfer animation from.
        dest_prim (Usd.Prim): The destination prim to transfer animation to.
        attribute_name (str): The name of the attribute to transfer animation for.
    """
    source_attr = source_prim.GetAttribute(attribute_name)
    if not source_attr.IsValid():
        raise ValueError(f"Attribute '{attribute_name}' does not exist on source prim.")
    dest_attr = dest_prim.GetAttribute(attribute_name)
    if not dest_attr.IsValid():
        type_name = source_attr.GetTypeName()
        dest_attr = dest_prim.CreateAttribute(attribute_name, type_name)
    dest_attr.Clear()
    time_samples = source_attr.GetTimeSamples()
    for time_code in time_samples:
        value = source_attr.Get(time_code)
        dest_attr.Set(value, time_code)


def integrate_vfx_elements(stage: Usd.Stage, vfx_root_path: str, vfx_variants: Dict[str, str]) -> None:
    """Integrate VFX elements into the main scene.

    Args:
        stage (Usd.Stage): The main USD stage to integrate VFX elements into.
        vfx_root_path (str): The root path where the VFX elements should be integrated.
        vfx_variants (Dict[str, str]): A dictionary mapping VFX element paths to their selected variants.

    Raises:
        ValueError: If the vfx_root_path does not exist on the stage.
    """
    vfx_root_prim = stage.GetPrimAtPath(vfx_root_path)
    if not vfx_root_prim.IsValid():
        raise ValueError(f"The prim path '{vfx_root_path}' does not exist on the stage.")
    for element_path, selected_variant in vfx_variants.items():
        full_element_path = f"{vfx_root_path}{element_path}"
        element_prim = stage.GetPrimAtPath(full_element_path)
        if not element_prim.IsValid():
            element_prim = stage.DefinePrim(full_element_path, "Xform")
        element_prim.GetVariantSet("type").SetVariantSelection(selected_variant)
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            element_prim.GetVariantSet("type").SetVariantSelection(selected_variant)


def perform_geometry_search(stage: Usd.Stage, prim_path: str, radius: float) -> List[Tuple[str, Gf.Vec3f]]:
    """
    Search for all geometry prims within a given radius from the specified prim.

    Args:
        stage (Usd.Stage): The USD stage to search.
        prim_path (str): The path to the prim to search from.
        radius (float): The search radius.

    Returns:
        List[Tuple[str, Gf.Vec3f]]: A list of tuples containing the prim path and world position
                                    of prims within the search radius.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    prim_world_pos = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).Transform(Gf.Vec3f(0, 0, 0))
    results = []
    for geom_prim in stage.Traverse():
        if not geom_prim.IsA(UsdGeom.Gprim):
            continue
        geom_xformable = UsdGeom.Xformable(geom_prim)
        if not geom_xformable:
            continue
        geom_world_pos = geom_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).Transform(
            Gf.Vec3f(0, 0, 0)
        )
        dist = (prim_world_pos - geom_world_pos).GetLength()
        if dist <= radius:
            results.append((str(geom_prim.GetPath()), geom_world_pos))
    return results


def extract_camera_rig_to_new_layer(stage: Usd.Stage, rig_prim_path: str, output_layer_path: str) -> Usd.Stage:
    """Extract a camera rig subtree to a new USD layer.

    Args:
        stage (Usd.Stage): The input USD stage.
        rig_prim_path (str): The path to the camera rig prim.
        output_layer_path (str): The path to save the new USD layer.

    Returns:
        Usd.Stage: The new USD stage with the extracted camera rig.
    """
    rig_prim = stage.GetPrimAtPath(rig_prim_path)
    if not rig_prim.IsValid():
        raise ValueError(f"Invalid camera rig prim path: {rig_prim_path}")
    new_stage = Usd.Stage.CreateNew(output_layer_path)
    new_rig_prim = new_stage.DefinePrim(rig_prim.GetPath())
    new_rig_prim.GetReferences().AddReference(stage.GetRootLayer().identifier, rig_prim.GetPath())
    new_stage.SetDefaultPrim(new_rig_prim)
    new_stage.GetRootLayer().Save()
    return new_stage


def update_rig_parameters(stage: Usd.Stage, rig_path: str, params: Dict[str, float]):
    """Update the parameter values on a rig."""
    rig_prim = stage.GetPrimAtPath(rig_path)
    if not rig_prim.IsValid():
        raise ValueError(f"Rig prim at path {rig_path} does not exist.")
    for param_name, param_value in params.items():
        attr_path = f"{rig_path}.{param_name}"
        attr = stage.GetAttributeAtPath(attr_path)
        if not attr.IsValid():
            prim = stage.GetPrimAtPath(rig_path)
            attr = prim.CreateAttribute(param_name, Sdf.ValueTypeNames.Float)
        attr.Set(param_value)


def create_camera_rig(stage: Usd.Stage, rig_path: str, camera_name: str = "MainCamera") -> UsdGeom.Xform:
    """Create a simple camera rig with a camera and a parent Xform.

    Args:
        stage (Usd.Stage): The USD stage to create the camera rig in.
        rig_path (str): The path where the camera rig should be created.
        camera_name (str): The name of the camera prim. Defaults to "MainCamera".

    Returns:
        UsdGeom.Xform: The created camera rig Xform prim.
    """
    if not Sdf.Path(rig_path).IsPrimPath():
        raise ValueError(f"Invalid rig path: {rig_path}")
    rig_xform = UsdGeom.Xform.Define(stage, rig_path)
    camera_path = f"{rig_path}/{camera_name}"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
    camera.CreateFocalLengthAttr().Set(35.0)
    UsdGeom.XformCommonAPI(camera).SetTranslate((0.0, 0.0, 5.0))
    return rig_xform


def link_prims_across_stages(src_stage: Usd.Stage, src_prim_path: str, dest_stage: Usd.Stage, dest_prim_path: str):
    """Link a prim from one stage to another."""
    src_prim = src_stage.GetPrimAtPath(src_prim_path)
    if not src_prim.IsValid():
        raise ValueError(f"Source prim at path {src_prim_path} does not exist.")
    dest_prim = dest_stage.GetPrimAtPath(dest_prim_path)
    if not dest_prim.IsValid():
        dest_prim = dest_stage.DefinePrim(dest_prim_path, src_prim.GetTypeName())
    dest_prim.GetReferences().AddReference(src_stage.GetRootLayer().identifier, src_prim_path)


def resolve_material_conflicts(stage: Usd.Stage, ancestor_scope: str, descendant_scope: str) -> None:
    """Resolves material binding conflicts between an ancestor and descendant scope.

    This function unbinds any bindings on prims in the ancestor scope that are also bound
    to prims in the descendant scope, giving preference to the descendant bindings.
    """
    ancestor_prim = stage.GetPrimAtPath(ancestor_scope)
    if not ancestor_prim.IsValid():
        raise ValueError(f"Prim at path {ancestor_scope} does not exist.")
    descendant_prim = stage.GetPrimAtPath(descendant_scope)
    if not descendant_prim.IsValid():
        raise ValueError(f"Prim at path {descendant_scope} does not exist.")
    descendant_bindings = {}
    for prim in Usd.PrimRange(descendant_prim):
        binding_api = UsdShade.MaterialBindingAPI(prim)
        direct_binding = binding_api.GetDirectBinding()
        if direct_binding.GetMaterial():
            descendant_bindings[str(prim.GetPath())] = direct_binding
    for prim in Usd.PrimRange(ancestor_prim):
        binding_api = UsdShade.MaterialBindingAPI(prim)
        direct_binding = binding_api.GetDirectBinding()
        if direct_binding.GetMaterial():
            prim_path = str(prim.GetPath())
            if prim_path in descendant_bindings:
                binding_api.UnbindDirectBinding()


def create_deforming_mesh(
    stage: Usd.Stage, mesh_path: str, points: Gf.Vec3f, time_samples: dict[float, list[Gf.Vec3f]]
) -> UsdGeom.Mesh:
    """Create a mesh prim with time-sampled points for deformation.

    Args:
        stage (Usd.Stage): The USD stage to create the mesh prim on.
        mesh_path (str): The path where the mesh prim should be created.
        points (Gf.Vec3f): The base mesh points.
        time_samples (dict[float, list[Gf.Vec3f]]): A dictionary mapping time codes to lists of points for deformation.

    Returns:
        UsdGeom.Mesh: The created mesh prim.

    Raises:
        ValueError: If the mesh_path is invalid or points are empty.
    """
    if not Sdf.Path.IsValidPathString(mesh_path):
        raise ValueError(f"Invalid mesh path: {mesh_path}")
    if not points:
        raise ValueError("Points cannot be empty")
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set([len(points)])
    mesh.GetFaceVertexIndicesAttr().Set(list(range(len(points))))
    points_attr = mesh.GetPointsAttr()
    for time_code, deformed_points in time_samples.items():
        points_attr.Set(deformed_points, time_code)
    return mesh


def resolve_texture_conflicts(prim: Usd.Prim, texture_file_name: str) -> bool:
    """Resolves conflicting texture file assignments by selecting the strongest opinion.

    Args:
        prim (Usd.Prim): The prim to resolve texture file conflicts for.
        texture_file_name (str): The texture file name to resolve conflicts for.

    Returns:
        bool: True if a conflict was resolved, False otherwise.
    """
    shader_binding_rels = [rel for rel in prim.GetRelationships() if rel.GetName().startswith("material:binding")]
    resolved_conflict = False
    for binding_rel in shader_binding_rels:
        shader_prim = binding_rel.GetTargets()[0]
        shader = UsdShade.Shader(stage.GetPrimAtPath(shader_prim))
        texture_file_inputs = [input for input in shader.GetInputs() if input.GetBaseName() == texture_file_name]
        if not texture_file_inputs:
            continue
        if len(texture_file_inputs) > 1:
            texture_file_inputs = sorted(texture_file_inputs, key=lambda x: x.GetResolveInfo().source)
            strongest_input = texture_file_inputs[-1]
            shader.GetInput(texture_file_name).Set(strongest_input.Get())
            resolved_conflict = True
    return resolved_conflict


def create_procedural_geometry(
    stage: Usd.Stage,
    prim_path: str,
    point_positions: list[Gf.Vec3f],
    point_normals: list[Gf.Vec3f],
    triangle_vertex_indices: list[int],
    triangle_orientation=UsdGeom.Tokens.rightHanded,
) -> UsdGeom.Mesh:
    """Create a procedurally generated mesh prim.

    Args:
        stage (Usd.Stage): The stage to create the mesh prim on.
        prim_path (str): The path where the mesh prim should be created.
        point_positions (list[Gf.Vec3f]): List of 3D point positions for the mesh vertices.
        point_normals (list[Gf.Vec3f]): List of 3D normal vectors for the mesh vertices.
        triangle_vertex_indices (list[int]): Flattened list of indices into the point_positions list
            representing triangles. Every 3 consecutive indices form one triangle.
        triangle_orientation (UsdGeom.Tokens): Winding order of the triangles, either rightHanded or leftHanded.

    Returns:
        UsdGeom.Mesh: The newly created mesh prim.
    """
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    num_points = len(point_positions)
    if num_points != len(point_normals):
        raise ValueError("Number of point positions must match number of point normals.")
    point_attr = mesh.GetPointsAttr()
    point_attr.Set(point_positions)
    normal_attr = mesh.GetNormalsAttr()
    normal_attr.Set(point_normals)
    vertex_count = [3] * (len(triangle_vertex_indices) // 3)
    mesh.GetFaceVertexCountsAttr().Set(vertex_count)
    mesh.GetFaceVertexIndicesAttr().Set(triangle_vertex_indices)
    mesh.CreateOrientationAttr().Set(triangle_orientation)
    mesh.CreateDoubleSidedAttr().Set(True)
    return mesh


def integrate_hair_simulation(stage: Usd.Stage, hair_prim_path: str, time_step: float, num_substeps: int) -> None:
    """Integrate hair simulation on a hair prim over a time step with substeps.

    Args:
        stage (Usd.Stage): The USD stage.
        hair_prim_path (str): The path to the hair prim.
        time_step (float): The time step size in seconds.
        num_substeps (int): The number of substeps to divide the time step into.

    Raises:
        ValueError: If the hair prim is not found or is not a valid hair prim.
    """
    hair_prim = stage.GetPrimAtPath(hair_prim_path)
    if not hair_prim.IsValid():
        raise ValueError(f"Hair prim not found at path: {hair_prim_path}")
    hair_geom = UsdGeom.BasisCurves(hair_prim)
    if not hair_geom:
        raise ValueError(f"Prim at path {hair_prim_path} is not a valid hair geometry")
    current_time = Usd.TimeCode.Default()
    points_attr = hair_geom.GetPointsAttr()
    if not points_attr.IsValid():
        raise ValueError(f"Hair geometry at path {hair_prim_path} does not have a valid points attribute")
    hair_points = points_attr.Get(current_time)
    substep_size = time_step / num_substeps
    for _ in range(num_substeps):
        velocity = Gf.Vec3f(0.1, 0.0, 0.0)
        hair_points = [p + velocity * substep_size for p in hair_points]
    points_attr.Set(hair_points, current_time)


def perform_texture_search(stage: Usd.Stage, material_path: str, texture_attr_name: str) -> str:
    """
    Perform a texture search on the given material path and texture attribute name.

    Args:
        stage (Usd.Stage): The USD stage to search.
        material_path (str): The path to the material prim.
        texture_attr_name (str): The name of the texture attribute to search for.

    Returns:
        str: The path to the texture file if found, empty string otherwise.
    """
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise ValueError(f"Invalid material path: {material_path}")
    material = UsdShade.Material(material_prim)
    if not material:
        raise ValueError(f"Prim at path {material_path} is not a valid Material")
    texture_attr = material.GetPrim().GetAttribute(texture_attr_name)
    if not texture_attr.IsValid():
        return ""
    texture_file_path = ""
    if texture_attr.HasValue():
        texture_file_path = texture_attr.Get().path
    return texture_file_path


def create_custom_prim(
    stage: Usd.Stage, type_name: str, path: str, specifier: Sdf.SpecifierDef = Sdf.SpecifierDef
) -> Usd.Prim:
    """Create a custom prim with the given type name and specifier at the given path on the stage.

    Args:
        stage (Usd.Stage): The stage to create the prim on.
        type_name (str): The type name of the custom prim.
        path (str): The path at which to create the prim.
        specifier (Sdf.SpecifierDef, optional): The specifier for the prim. Defaults to Sdf.SpecifierDef.

    Returns:
        Usd.Prim: The created prim.
    """
    if not Sdf.Path(path).IsAbsolutePath():
        raise Tf.ErrorException(f"Path '{path}' is not an absolute path.")
    existing_prim = stage.GetPrimAtPath(path)
    if existing_prim.IsValid():
        if existing_prim.GetTypeName() == type_name and existing_prim.GetSpecifier() == specifier:
            return existing_prim
        else:
            raise Tf.ErrorException(f"A prim of type '{existing_prim.GetTypeName()}' already exists at path '{path}'.")
    prim = stage.DefinePrim(path, type_name)
    prim.SetSpecifier(specifier)
    return prim


def create_hierarchical_prim(
    stage: Usd.Stage, prim_type: str, prim_path: str, prim_children: Dict[str, Dict]
) -> Usd.Prim:
    """Create a prim with the given type and path, and create its children prims recursively.

    Args:
        stage (Usd.Stage): The stage to create the prim on.
        prim_type (str): The type of the prim to create (e.g., "Xform", "Sphere").
        prim_path (str): The path where the prim should be created.
        prim_children (Dict[str, Dict]): A dictionary mapping child prim names to their data (type and children).

    Returns:
        Usd.Prim: The created prim.
    """
    if not stage:
        raise ValueError("Invalid stage")
    if not prim_type:
        raise ValueError("Invalid prim type")
    if not prim_path:
        raise ValueError("Invalid prim path")
    prim = stage.DefinePrim(prim_path, prim_type)
    if not prim.IsValid():
        return None
    for child_name, child_data in prim_children.items():
        child_type = child_data["type"]
        child_path = f"{prim_path}/{child_name}"
        child_children = child_data.get("children", {})
        create_hierarchical_prim(stage, child_type, child_path, child_children)
    return prim


def extract_subtree_to_new_layer(stage: Usd.Stage, prim_path: str, new_layer_identifier: str) -> Usd.Stage:
    """Extract a prim subtree into a new layer.

    The extracted layer will contain the prim and all its descendants.
    The original stage will be modified to reference the extracted layer.

    Args:
        stage (Usd.Stage): The stage containing the subtree to extract.
        prim_path (str): The path of the prim at the root of the subtree.
        new_layer_identifier (str): The identifier to use for the new layer.

    Returns:
        Usd.Stage: A new stage with the extracted subtree.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"No prim exists at path '{prim_path}'")
    new_layer = Sdf.Layer.CreateNew(new_layer_identifier)
    new_stage = Usd.Stage.Open(new_layer)
    new_prim = new_stage.DefinePrim(prim.GetPath())
    new_prim.GetReferences().AddReference(assetPath=stage.GetRootLayer().identifier, primPath=prim.GetPath())
    original_prim = stage.GetPrimAtPath(prim_path)
    original_prim.GetReferences().AddReference(assetPath=new_layer.identifier, primPath=prim.GetPath())
    with Usd.EditContext(stage, stage.GetRootLayer()):
        stage.RemovePrim(prim_path)
    return new_stage


def import_json_to_stage(json_path: str, usd_path: str) -> Usd.Stage:
    """
    Import a JSON file into a USD stage.

    Args:
        json_path (str): Path to the input JSON file.
        usd_path (str): Path to save the output USD file.

    Returns:
        Usd.Stage: The created USD stage.
    """
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)
    stage = Usd.Stage.CreateNew(usd_path)

    def create_prims_recursive(json_obj, parent_prim):
        for key, value in json_obj.items():
            prim_path = parent_prim.GetPath().AppendChild(key)
            prim = stage.DefinePrim(prim_path)
            if isinstance(value, dict):
                create_prims_recursive(value, prim)
            else:
                attr = prim.CreateAttribute("value", Sdf.ValueTypeNames.String)
                attr.Set(str(value))

    create_prims_recursive(json_data, stage.GetPseudoRoot())
    stage.Save()
    return stage


def create_prim(stage, prim_type, prim_path):
    """Create a new prim at the given path with the specified type."""
    return stage.DefinePrim(prim_path, prim_type)


def create_variant_set(prim, variant_set_name, variants):
    """Create a new variant set on the given prim with the specified variants."""
    variant_set = prim.GetVariantSets().AddVariantSet(variant_set_name)
    for variant in variants:
        variant_set.AddVariant(variant)


def change_variant_set_for_prims(stage, variant_set_name, variant_name, prim_paths):
    """Change the variant selection for multiple prims in a variant set."""
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        variant_sets = prim.GetVariantSets()
        if not variant_sets.HasVariantSet(variant_set_name):
            raise ValueError(f"Prim at path {prim_path} does not have variant set {variant_set_name}.")
        variant_set = variant_sets.GetVariantSet(variant_set_name)
        if not variant_set.HasAuthoredVariant(variant_name):
            raise ValueError(
                f"Variant set {variant_set_name} on prim at path {prim_path} does not have variant {variant_name}."
            )
        variant_set.SetVariantSelection(variant_name)


def perform_scenegraph_query(stage: Usd.Stage, path: str, pred: Callable[[Usd.Prim], bool]) -> List[str]:
    """
    Perform a scenegraph query starting from the specified root path.
    Returns a list of prim paths that match the predicate.
    """
    root_prim = stage.GetPrimAtPath(path)
    if not root_prim:
        raise ValueError(f"Invalid root path: {path}")
    if not callable(pred):
        raise TypeError("pred must be a callable")
    results = []
    stack = [root_prim]
    while stack:
        prim = stack.pop()
        if pred(prim):
            results.append(str(prim.GetPath()))
        stack.extend(prim.GetChildren())
    return results


def is_mesh(prim: Usd.Prim) -> bool:
    return prim.IsA(UsdGeom.Mesh)


def reassign_prim_parents(stage: Usd.Stage, prim_paths: List[str], new_parent_path: str) -> None:
    """Reassign the parents of the given prims to a new parent prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths to reassign.
        new_parent_path (str): The path of the new parent prim.

    Raises:
        ValueError: If any of the prim paths or the new parent path are invalid.
    """
    new_parent_prim = stage.GetPrimAtPath(new_parent_path)
    if not new_parent_prim.IsValid():
        raise ValueError(f"New parent prim at path {new_parent_path} does not exist.")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        current_parent_path = prim.GetPath().GetParentPath()
        if current_parent_path == new_parent_prim.GetPath():
            continue
        stage.DefinePrim(new_parent_prim.GetPath().AppendChild(prim.GetName()))
        stage.RemovePrim(prim.GetPath())


def extract_textures_to_directory(stage: Usd.Stage, output_dir: str) -> Dict[str, str]:
    """
    Extract all textures from a USD stage to a directory.

    Args:
        stage (Usd.Stage): The USD stage to extract textures from.
        output_dir (str): The directory to save the extracted textures to.

    Returns:
        A dictionary mapping texture file path to the saved texture path.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    texture_paths = {}
    for prim in stage.Traverse():
        material_binding = UsdShade.MaterialBindingAPI(prim)
        if material_binding:
            material = material_binding.GetDirectBinding().GetMaterial()
            for input in material.GetInputs():
                if input.GetTypeName() == "SdfAssetPath":
                    texture_file_path = input.Get().path
                    texture_file_name = os.path.basename(texture_file_path)
                    saved_texture_path = os.path.join(output_dir, texture_file_name)
                    shutil.copy(texture_file_path, saved_texture_path)
                    texture_paths[texture_file_path] = saved_texture_path
    return texture_paths


def update_shader_parameters(stage: Usd.Stage, material_path: str, parameters: Dict[str, Any]):
    """Update the parameters of a shader with the given values.

    Args:
        stage (Usd.Stage): The USD stage.
        material_path (str): The path to the material prim.
        parameters (Dict[str, Any]): A dictionary of parameter names and values.

    Raises:
        ValueError: If the material prim is not found or is not a valid shader.
    """
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise ValueError(f"Material prim not found at path: {material_path}")
    material = UsdShade.Material(material_prim)
    surface_output = material.GetOutput(UsdShade.Tokens.surface)
    if not surface_output:
        raise ValueError(f"Material prim at path {material_path} does not have a valid surface output")
    shader_prim = surface_output.GetConnectedSource()[0]
    if not shader_prim:
        raise ValueError(f"Material prim at path {material_path} does not have a connected shader")
    shader = UsdShade.Shader(shader_prim)
    for param_name, param_value in parameters.items():
        shader_input = shader.GetInput(param_name)
        if not shader_input:
            shader_input = shader.CreateInput(param_name, Sdf.ValueTypeNames.Float)
        shader_input.Set(param_value)


def apply_post_processing_effects(stage: Usd.Stage, prim_path: str, effect_names: List[str]):
    """Apply post-processing effects to a prim's material output.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to apply effects to.
        effect_names (List[str]): A list of effect names to apply, in order.

    Raises:
        ValueError: If the prim or its material does not exist.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    binding_api = UsdShade.MaterialBindingAPI(prim)
    material = binding_api.ComputeBoundMaterial()[0]
    if not material:
        raise ValueError(f"Prim at path {prim_path} does not have a bound material.")
    surface_output = material.GetOutput(UsdShade.Tokens.surface)
    if not surface_output:
        raise ValueError(f"Material {material.GetPath()} does not have a surface output.")
    effect_prims = []
    for effect_name in effect_names:
        effect_prim_path = f"{material.GetPath()}/{effect_name}"
        effect_prim = UsdShade.Shader.Define(stage, effect_prim_path)
        effect_prim.CreateIdAttr(effect_name)
        effect_prim.CreateInput("color", Sdf.ValueTypeNames.Color3f).ConnectToSource(surface_output)
        surface_output = effect_prim.CreateOutput(UsdShade.Tokens.surface, Sdf.ValueTypeNames.Color3f)
        effect_prims.append(effect_prim)
    material.CreateOutput(UsdShade.Tokens.surface, Sdf.ValueTypeNames.Color3f).ConnectToSource(surface_output)


def create_complex_physics_simulation(stage: Usd.Stage) -> Usd.Prim:
    """Create a complex physics simulation with multiple interacting objects."""
    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, -1.0, 0.0))
    scene.CreateGravityMagnitudeAttr().Set(981.0)
    ground = stage.DefinePrim("/physicsScene/ground", "Cube")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    Usd.ModelAPI(ground).SetKind(Kind.Tokens.component)
    ground_xform = UsdGeom.Xformable(ground)
    add_scale_op(ground_xform).Set((10.0, 0.1, 10.0))
    add_translate_op(ground_xform).Set((0.0, -5.0, 0.0))
    UsdPhysics.CollisionAPI.Apply(ground)
    UsdPhysics.RigidBodyAPI.Apply(ground)
    cube = stage.DefinePrim("/physicsScene/cube", "Cube")
    Usd.ModelAPI(cube).SetKind(Kind.Tokens.component)
    cube_xform = UsdGeom.Xformable(cube)
    add_translate_op(cube_xform).Set((0.0, 5.0, 0.0))
    UsdPhysics.CollisionAPI.Apply(cube)
    UsdPhysics.RigidBodyAPI.Apply(cube)
    sphere = stage.DefinePrim("/physicsScene/sphere", "Sphere")
    Usd.ModelAPI(sphere).SetKind(Kind.Tokens.component)
    sphere_xform = UsdGeom.Xformable(sphere)
    add_translate_op(sphere_xform).Set((3.0, 8.0, 0.0))
    UsdPhysics.CollisionAPI.Apply(sphere)
    UsdPhysics.RigidBodyAPI.Apply(sphere)
    joint = UsdPhysics.DistanceJoint.Define(stage, "/physicsScene/joint")
    joint.CreateBody0Rel().SetTargets([cube.GetPath()])
    joint.CreateBody1Rel().SetTargets([sphere.GetPath()])
    joint.CreateLocalPos0Attr().Set((0.5, 0.5, 0.5))
    joint.CreateLocalPos1Attr().Set((-0.5, -0.5, -0.5))
    joint.CreateMinDistanceAttr().Set(1.0)
    joint.CreateMaxDistanceAttr().Set(3.0)
    return scene.GetPrim()


def reassign_textures_for_prims(stage: Usd.Stage, texture_dict: Dict[str, str], prim_paths: List[str]):
    """Reassign textures for prims based on a texture dictionary.

    Args:
        stage (Usd.Stage): The USD stage.
        texture_dict (Dict[str, str]): Dictionary mapping old texture paths to new texture paths.
        prim_paths (List[str]): List of prim paths to process.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        binding_api = UsdShade.MaterialBindingAPI(prim)
        material = binding_api.ComputeBoundMaterial()[0]
        if not material:
            print(f"Warning: No material bound to prim {prim_path}. Skipping.")
            continue
        for shader_input in material.GetInputs():
            if shader_input.GetTypeName() != "SdfAssetPath":
                continue
            connection_source = shader_input.GetConnectedSources()[0]
            if not connection_source:
                print(
                    f"Warning: Shader input {shader_input.GetBaseName()} on material {material.GetPath()} has no connection. Skipping."
                )
                continue
            connected_shader_output = connection_source[0]
            texture_shader = UsdShade.Shader(connected_shader_output.GetPrim())
            if not texture_shader:
                print(
                    f"Warning: Shader input {shader_input.GetBaseName()} on material {material.GetPath()} is not connected to a texture shader. Skipping."
                )
                continue
            file_attr = texture_shader.GetInput("file")
            if not file_attr:
                print(f"Warning: Texture shader {texture_shader.GetPath()} has no 'file' attribute. Skipping.")
                continue
            file_path = file_attr.Get().path
            if file_path in texture_dict:
                new_file_path = texture_dict[file_path]
                file_attr.Set(new_file_path)
                print(f"Reassigned texture {file_path} to {new_file_path} on shader {texture_shader.GetPath()}")
            else:
                print(
                    f"Warning: No replacement found for texture {file_path} on shader {texture_shader.GetPath()}. Skipping."
                )


def transfer_deformation_curves(source_prim: Usd.Prim, target_prim: Usd.Prim):
    """Transfer deformation curves from source_prim to target_prim."""
    if not source_prim.IsValid():
        raise ValueError("Source prim is not valid.")
    if not target_prim.IsValid():
        raise ValueError("Target prim is not valid.")
    source_curves = UsdGeom.BasisCurves(source_prim)
    target_curves = UsdGeom.BasisCurves(target_prim)
    if not source_curves or not target_curves:
        raise ValueError("Both source and target prims must be curve prims.")
    points_attr = source_curves.GetPointsAttr()
    if points_attr.HasValue():
        points = points_attr.Get()
        target_curves.GetPointsAttr().Set(points)
    velocities_attr = source_curves.GetVelocitiesAttr()
    if velocities_attr.HasValue():
        velocities = velocities_attr.Get()
        target_curves.CreateVelocitiesAttr().Set(velocities)
    widths_attr = source_curves.GetWidthsAttr()
    if widths_attr.HasValue():
        widths = widths_attr.Get()
        target_curves.CreateWidthsAttr().Set(widths)


def update_texture_parameters(
    stage: Usd.Stage,
    prim_path: str,
    file_path: str,
    st: Tuple[float, float] = (0.0, 0.0),
    scale: Tuple[float, float] = (1.0, 1.0),
) -> None:
    """Update texture parameters for a shader prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the shader prim.
        file_path (str): The file path for the texture.
        st (Tuple[float, float]): The texture coordinate offset. Defaults to (0.0, 0.0).
        scale (Tuple[float, float]): The texture coordinate scale. Defaults to (1.0, 1.0).

    Raises:
        ValueError: If the prim does not exist or is not a shader.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    shader = UsdShade.Shader(prim)
    if not shader:
        raise ValueError(f"Prim at path {prim_path} is not a shader.")
    texture_input = shader.GetInput("file")
    if not texture_input:
        texture_input = shader.CreateInput("file", Sdf.ValueTypeNames.Asset)
    texture_input.Set(file_path)
    st_input = shader.GetInput("st")
    if not st_input:
        st_input = shader.CreateInput("st", Sdf.ValueTypeNames.Float2)
    st_input.Set(Gf.Vec2f(st[0], st[1]))
    scale_input = shader.GetInput("scale")
    if not scale_input:
        scale_input = shader.CreateInput("scale", Sdf.ValueTypeNames.Float2)
    scale_input.Set(Gf.Vec2f(scale[0], scale[1]))


def create_prim(stage, prim_type, prim_path):
    """Helper function to create a prim."""
    return stage.DefinePrim(prim_path, prim_type)


def validate_hierarchical_lods(stage: Usd.Stage, prim_path: str) -> bool:
    """Validate if the prim at prim_path has valid hierarchical LODs.

    LODs must be specified as children prims named 'LOD0', 'LOD1', etc.
    Each successive LOD must have fewer children than the previous.

    Args:
        stage (Usd.Stage): The stage to query.
        prim_path (str): The path to the prim to validate.

    Returns:
        bool: True if the prim has valid hierarchical LODs, False otherwise.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    lod_prims = [child for child in prim.GetChildren() if child.GetName().startswith("LOD")]
    lod_prims.sort(key=lambda p: p.GetName())
    if not lod_prims:
        return False
    for i, lod_prim in enumerate(lod_prims):
        if lod_prim.GetName() != f"LOD{i}":
            return False
    prev_num_children = float("inf")
    for lod_prim in lod_prims:
        num_children = len(lod_prim.GetChildren())
        if num_children >= prev_num_children:
            return False
        prev_num_children = num_children
    return True


def serialize_stage_to_usdz(stage: Usd.Stage, output_path: str) -> bool:
    """Serialize a USD stage to a USDZ file.

    Args:
        stage (Usd.Stage): The input USD stage to serialize.
        output_path (str): The output file path for the USDZ file.

    Returns:
        bool: True if the serialization was successful, False otherwise.
    """
    if not stage:
        return False
    try:
        bool_result = stage.Export(output_path)
        if not bool_result:
            return False
    except Exception as e:
        print(f"Error exporting stage: {e}")
        return False
    return True


def setup_color_variations(stage: Usd.Stage, prim_path: str, variations: Dict[str, Gf.Vec3f]) -> None:
    """Set up color variations on a prim using UsdPrimVariants.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        variations (Dict[str, Gf.Vec3f]): A dictionary mapping variation names to color values.

    Raises:
        ValueError: If the prim does not exist or is not a valid UsdGeomGprim.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    gprim = UsdGeom.Gprim(prim)
    if not gprim:
        raise ValueError(f"Prim at path {prim_path} is not a valid UsdGeomGprim.")
    color_variantSet = prim.GetVariantSets().AddVariantSet("color")
    for var_name, color in variations.items():
        color_variantSet.AddVariant(var_name)
        color_variantSet.SetVariantSelection(var_name)
        with color_variantSet.GetVariantEditContext():
            material_path = f"{prim_path}_Material_{var_name}"
            material = UsdShade.Material.Define(stage, material_path)
            UsdShade.MaterialBindingAPI(prim).Bind(material)
            input_name = "diffuseColor"
            material.CreateInput(input_name, Sdf.ValueTypeNames.Color3f).Set(color)


def batch_duplicate_prims(stage: Usd.Stage, prim_paths: List[str], num_copies: int, variation: str) -> List[str]:
    """
    Duplicate a list of prims on the given stage a specified number of times with a name variation.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths to duplicate.
        num_copies (int): Number of copies to create for each prim.
        variation (str): String to append to each duplicated prim name for uniqueness.

    Returns:
        List[str]: List of paths of the newly created duplicate prims.
    """
    if num_copies < 1:
        raise ValueError("num_copies must be greater than 0")
    new_prim_paths = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist")
        for i in range(num_copies):
            new_prim_name = f"{prim.GetName()}_{variation}_{i + 1}"
            new_prim_path = prim.GetPath().GetParentPath().AppendChild(new_prim_name)
            new_prim = stage.DefinePrim(new_prim_path)
            new_prim.GetReferences().AddReference(str(prim.GetPath()))
            new_prim_paths.append(new_prim.GetPath().pathString)
    return new_prim_paths


def link_geometry_across_stages(
    source_stage: Usd.Stage, dest_stage: Usd.Stage, prim_paths: List[Tuple[str, str]]
) -> None:
    """Link geometry prims from one stage to another.

    Args:
        source_stage (Usd.Stage): The stage containing the source geometry prims.
        dest_stage (Usd.Stage): The stage where the geometry will be linked.
        prim_paths (List[Tuple[str, str]]): A list of tuples containing the source and destination prim paths.
    """
    for source_path, dest_path in prim_paths:
        source_prim = source_stage.GetPrimAtPath(source_path)
        if not source_prim.IsValid():
            raise ValueError(f"Source prim at path {source_path} does not exist.")
        if not source_prim.IsA(UsdGeom.Imageable):
            raise ValueError(f"Source prim at path {source_path} is not a geometry prim.")
        dest_prim = dest_stage.GetPrimAtPath(dest_path)
        if not dest_prim.IsValid():
            dest_prim = dest_stage.DefinePrim(dest_path, source_prim.GetTypeName())
        source_layer_path = source_stage.GetRootLayer().realPath
        dest_prim.GetReferences().AddReference(assetPath=source_layer_path, primPath=source_prim.GetPath())


def reassign_rigs_for_prims(stage: Usd.Stage, prim_paths: Sequence[Sdf.Path], rig_path: Sdf.Path):
    """Reassign the rig binding for the given prims to the specified rig prim path."""
    rig_prim = stage.GetPrimAtPath(rig_path)
    if not rig_prim.IsValid():
        raise ValueError(f"Invalid rig prim path: {rig_path}")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Invalid prim path: {prim_path}. Skipping.")
            continue
        rig_binding_api = UsdSkel.BindingAPI(prim)
        if not rig_binding_api.GetSkeletonRel().GetTargets():
            print(f"Warning: Prim {prim_path} has no valid rig binding. Skipping.")
            continue
        try:
            rig_binding_api.GetSkeletonRel().SetTargets([rig_path])
            print(f"Reassigned rig binding for prim {prim_path} to {rig_path}")
        except Exception as e:
            print(f"Error reassigning rig binding for prim {prim_path}: {str(e)}")


def setup_transparency_variations(stage: Usd.Stage, prim_path: str, variations: List[Tuple[str, float]]) -> None:
    """
    Set up transparency variations on a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        variations (List[Tuple[str, float]]): A list of tuples representing the variations,
            where each tuple contains the variation name and the transparency value.

    Raises:
        ValueError: If the prim path is invalid or the prim is not a UsdGeomGprim.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")
    gprim = UsdGeom.Gprim(prim)
    if not gprim:
        raise ValueError(f"Prim at path {prim_path} is not a UsdGeomGprim")
    display_opacity_attr = gprim.GetDisplayOpacityAttr()
    variant_set = prim.GetVariantSets().AddVariantSet("transparency")
    variant_set.SetVariantSelection(variations[0][0])
    for variant_name, transparency in variations:
        variant_set.SetVariantSelection(variant_name)
        display_opacity_attr.Set([1.0 - transparency])


def transfer_texture_data(source_prim: Usd.Prim, dest_prim: Usd.Prim):
    """Transfer texture data from one prim to another."""
    if not source_prim.IsValid():
        raise ValueError("Source prim is not valid.")
    if not dest_prim.IsValid():
        raise ValueError("Destination prim is not valid.")
    source_binding_api = UsdShade.MaterialBindingAPI(source_prim)
    dest_binding_api = UsdShade.MaterialBindingAPI(dest_prim)
    if not source_binding_api.GetDirectBinding():
        raise ValueError("Source prim does not have a bound material.")
    source_material = UsdShade.MaterialBindingAPI(source_prim).ComputeBoundMaterial()[0]
    dest_binding_api.Bind(source_material)
    shader_prims = source_material.ComputeInterfaceInputConsumersMap().values()
    for shader_prim in shader_prims:
        texture_input = shader_prim.GetInput("file")
        if texture_input:
            texture_asset_path = texture_input.Get().path
            dest_layer = dest_prim.GetStage().GetEditTarget().GetLayer()
            texture_asset_path_copy = texture_asset_path.ReplacePrefix(source_prim.GetPath(), dest_prim.GetPath())
            dest_layer.Copy(texture_asset_path, texture_asset_path_copy)
            dest_shader_prim = dest_prim.GetStage().GetPrimAtPath(
                shader_prim.GetPath().ReplacePrefix(source_prim.GetPath(), dest_prim.GetPath())
            )
            dest_shader_prim.GetInput("file").Set(texture_asset_path_copy)


def batch_delete_prims(stage: Usd.Stage, prim_paths: List[str]) -> None:
    """
    Batch delete prims from the stage.

    Args:
        stage (Usd.Stage): The stage to delete prims from.
        prim_paths (List[str]): List of absolute prim paths to delete.

    Raises:
        ValueError: If any path in prim_paths is invalid.
    """
    for path in prim_paths:
        if not Sdf.Path.IsValidPathString(path):
            raise ValueError(f"Invalid prim path: {path}")
    with Usd.EditContext(stage, stage.GetEditTarget()):
        for path in prim_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                removed = stage.RemovePrim(path)
                if not removed:
                    print(f"Prim at path {path} was already removed or inactive.")
            else:
                print(f"No prim exists at path {path}. Skipping.")


def perform_shader_search(stage: Usd.Stage, shader_path: str) -> Tuple[str, List[Usd.Prim]]:
    """
    Search for a shader prim and its bound prims in the given stage.

    Args:
        stage (Usd.Stage): The stage to search in.
        shader_path (str): The path to the shader prim.

    Returns:
        A tuple containing the shader ID and a list of bound prims.
        If the shader prim is not found or invalid, returns (None, []).
    """
    shader_prim = stage.GetPrimAtPath(shader_path)
    if not shader_prim.IsValid():
        return (None, [])
    shader_id = shader_prim.GetAttribute("info:id").Get()
    if not shader_id:
        return (None, [])
    bound_prims = []
    for prim in stage.TraverseAll():
        if UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel().GetTargets():
            bound_shader_prim = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if bound_shader_prim == shader_prim:
                bound_prims.append(prim)
    return (shader_id, bound_prims)


def update_geometric_transforms(
    stage: Usd.Stage, prim_path: str, translation: Gf.Vec3f, rotation: Gf.Vec3f, scale: Gf.Vec3f
):
    """Update the geometric transforms for a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): Path to the prim to update transforms for.
        translation (Gf.Vec3f): New translation value.
        rotation (Gf.Vec3f): New rotation value in degrees.
        scale (Gf.Vec3f): New scale value.

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
    rotation_op = add_rotate_xyz_op(xformable)
    rotation_op.Set(rotation)
    scale_op = add_scale_op(xformable)
    scale_op.Set(scale)


def create_prim(stage, prim_type, prim_path):
    """Create a prim with a given type at the specified path."""
    return stage.DefinePrim(prim_path, prim_type)


def setup_reflection_probes(stage: Usd.Stage) -> None:
    """Create reflection probes for the environment."""
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        raise ValueError("Stage has no default prim")
    probes_prim = create_prim(stage, "Xform", root_prim.GetPath().AppendChild("Probes"))
    if not probes_prim:
        raise ValueError("Failed to create Probes prim")
    for i in range(2):
        for j in range(3):
            probe_prim = create_prim(stage, "Sphere", probes_prim.GetPath().AppendChild(f"Probe_{i}_{j}"))
            if not probe_prim:
                raise ValueError(f"Failed to create Probe_{i}_{j} prim")
            probe_sphere = UsdGeom.Sphere(probe_prim)
            probe_sphere.CreateRadiusAttr().Set(100.0)
            probe_light = UsdLux.SphereLight.Define(stage, probe_prim.GetPath())
            probe_xform = UsdGeom.Xformable(probe_prim)
            add_translate_op(probe_xform).Set((i * 500.0, 200.0, j * 500.0))


def create_prim(stage, prim_type, prim_path):
    """Create a new prim at the given path with the specified type."""
    return stage.DefinePrim(prim_path, prim_type)


def create_light_rig(stage: Usd.Stage, root_path: str = "/LightRig") -> Usd.Prim:
    """Create a light rig with a dome and key lights.

    Args:
        stage (Usd.Stage): The USD stage to create the light rig on.
        root_path (str, optional): The root path for the light rig. Defaults to "/LightRig".

    Returns:
        Usd.Prim: The root prim of the created light rig.
    """
    root_prim = create_prim(stage, "Xform", root_path)
    dome_light_path = f"{root_path}/DomeLight"
    dome_light = create_prim(stage, "DomeLight", dome_light_path)
    dome_light_api = UsdLux.DomeLight(dome_light)
    dome_light_api.CreateIntensityAttr().Set(5000.0)
    dome_light_api.CreateTextureFileAttr().Set("@domeTexture.hdr@")
    key_light_path = f"{root_path}/KeyLight"
    key_light = create_prim(stage, "DistantLight", key_light_path)
    key_light_api = UsdLux.DistantLight(key_light)
    key_light_api.CreateIntensityAttr().Set(10000.0)
    key_light_api.CreateAngleAttr().Set(1.0)
    key_light_xform = UsdGeom.Xformable(key_light)
    key_light_xform.AddRotateXOp().Set(45.0)
    key_light_xform.AddRotateYOp().Set(-60.0)
    fill_light_path = f"{root_path}/FillLight"
    fill_light = create_prim(stage, "DistantLight", fill_light_path)
    fill_light_api = UsdLux.DistantLight(fill_light)
    fill_light_api.CreateIntensityAttr().Set(2500.0)
    fill_light_api.CreateAngleAttr().Set(5.0)
    fill_light_xform = UsdGeom.Xformable(fill_light)
    fill_light_xform.AddRotateXOp().Set(45.0)
    fill_light_xform.AddRotateYOp().Set(60.0)
    return root_prim


def attach_physics_properties(stage: Usd.Stage, prim_path: str, mass: float, density: float) -> None:
    """Attach physics properties to a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        mass (float): The mass value to set.
        density (float): The density value to set.

    Raises:
        ValueError: If the prim does not exist or is not a valid UsdGeomGprim.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    gprim = UsdGeom.Gprim(prim)
    if not gprim:
        raise ValueError(f"Prim at path {prim_path} is not a valid UsdGeomGprim.")
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(mass)
    mass_api.CreateDensityAttr().Set(density)


def reassign_materials_for_prims(stage: Usd.Stage, prim_paths: List[str], material_prim_path: str) -> bool:
    """Reassign materials for the specified prims."""
    material_prim = stage.GetPrimAtPath(material_prim_path)
    if not material_prim.IsValid():
        raise ValueError(f"Material prim at path {material_prim_path} does not exist.")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            print(f"Warning: Prim at path {prim_path} is not a mesh. Skipping.")
            continue
        binding_api = UsdShade.MaterialBindingAPI(prim)
        material_binding_rel_name = "material:binding"
        if not prim.HasRelationship(material_binding_rel_name):
            binding_api.CreateMaterialBindSubset("allPurpose", [], "face")
            binding_api.GetDirectBindingRel().SetTargets([material_prim.GetPath()])
        else:
            material_binding_rel = prim.GetRelationship(material_binding_rel_name)
            material_binding_rel.SetTargets([material_prim.GetPath()])
    return True


def link_shaders_across_stages(source_stage: Usd.Stage, target_stage: Usd.Stage, source_path: str, target_path: str):
    """Link shaders from one stage to another.

    Args:
        source_stage (Usd.Stage): The stage containing the source shader.
        target_stage (Usd.Stage): The stage where the shader will be linked.
        source_path (str): The path to the source shader in the source stage.
        target_path (str): The path where the shader will be linked in the target stage.
    """
    source_shader = UsdShade.Shader.Get(source_stage, source_path)
    if not source_shader:
        raise ValueError(f"No shader found at path {source_path} in the source stage.")
    target_prim = target_stage.GetPrimAtPath(target_path)
    if not target_prim:
        target_prim = UsdShade.Shader.Define(target_stage, target_path).GetPrim()
    if not target_prim.IsA(UsdShade.Material) and (not target_prim.IsA(UsdShade.Shader)):
        raise ValueError(f"Target prim at path {target_path} is not a valid shader container.")
    target_prim.GetReferences().AddReference(
        assetPath=source_stage.GetRootLayer().identifier, primPath=source_shader.GetPath()
    )


LIGHT_TYPES = {
    "DistantLight": UsdLux.DistantLight,
    "DomeLight": UsdLux.DomeLight,
    "RectLight": UsdLux.RectLight,
    "DiskLight": UsdLux.DiskLight,
    "CylinderLight": UsdLux.CylinderLight,
    "SphereLight": UsdLux.SphereLight,
}


def setup_light_intensity_variations(
    stage: Usd.Stage, prim_path: str, intensity_min: float, intensity_max: float, num_variations: int
) -> List[Usd.Prim]:
    """Create variations of a light prim with different intensity values."""
    if intensity_min < 0 or intensity_max < 0:
        raise ValueError("Intensity values must be non-negative.")
    if intensity_min > intensity_max:
        raise ValueError("intensity_min must be less than or equal to intensity_max.")
    if num_variations < 1:
        raise ValueError("num_variations must be a positive integer.")
    light_prim = stage.GetPrimAtPath(prim_path)
    if not light_prim.IsValid():
        raise ValueError(f"No prim found at path {prim_path}.")
    light_type = light_prim.GetTypeName()
    if light_type not in LIGHT_TYPES:
        raise ValueError(f"Unsupported light type: {light_type}")
    variations = []
    for i in range(num_variations):
        t = i / (num_variations - 1) if num_variations > 1 else 0
        intensity = intensity_min + (intensity_max - intensity_min) * t
        var_prim_name = f"{light_prim.GetName()}_var{i}"
        var_prim = stage.DefinePrim(light_prim.GetPath().AppendChild(var_prim_name), light_prim.GetTypeName())
        LIGHT_TYPES[light_type](var_prim).GetIntensityAttr().Set(intensity)
        variations.append(var_prim)
    return variations


def validate_shading_networks(stage: Usd.Stage) -> bool:
    """Validate the shading networks on the given stage.

    This function checks if all shading networks on the stage are valid, meaning:
    1. All shading nodes have valid types.
    2. All input connections are valid (types match).

    Returns:
        bool: True if all networks are valid, False otherwise.
    """
    for prim in stage.Traverse():
        shader = UsdShade.Shader(prim)
        if shader:
            if not shader.GetShaderId():
                print(f"Invalid shader type for {shader.GetPath()}")
                return False
            for input in shader.GetInputs():
                if input.HasConnectedSource():
                    source = input.GetConnectedSource()
                    if not source[0].GetTypeName() == input.GetTypeName():
                        print(f"Invalid connection type for {input.GetPath()}")
                        return False
    return True


def perform_advanced_attribute_search(
    stage: Usd.Stage, prim_path: str, attribute_name: str
) -> List[Tuple[str, str, object]]:
    """
    Perform an advanced attribute search on a prim and its descendants.

    Args:
        stage (Usd.Stage): The stage to search.
        prim_path (str): The path to the prim to start the search from.
        attribute_name (str): The name of the attribute to search for.

    Returns:
        List[Tuple[str, str, object]]: A list of tuples containing the prim path, attribute name, and attribute value
                                       for each attribute found.
    """
    results = []
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")
    for descendant in iter(Usd.PrimRange(prim)):
        if descendant.HasAttribute(attribute_name):
            attr = descendant.GetAttribute(attribute_name)
            attr_name = attr.GetName()
            attr_value = attr.Get()
            results.append((str(descendant.GetPath()), attr_name, attr_value))
        relationships = descendant.GetRelationships()
        for rel in relationships:
            target_paths = rel.GetTargets()
            for target_path in target_paths:
                target_prim = stage.GetPrimAtPath(target_path)
                if target_prim.HasAttribute(attribute_name):
                    attr = target_prim.GetAttribute(attribute_name)
                    attr_name = attr.GetName()
                    attr_value = attr.Get()
                    results.append((str(target_path), attr_name, attr_value))
    return results


def serialize_stage_to_binary_blob(stage: Usd.Stage) -> bytes:
    """Serialize a USD stage to a binary blob."""
    with tempfile.NamedTemporaryFile(suffix=".usd", delete=False) as temp_file:
        temp_file_path = temp_file.name
    try:
        stage.Export(temp_file_path)
        with open(temp_file_path, "rb") as file:
            binary_data = file.read()
    finally:
        os.remove(temp_file_path)
    return binary_data


def update_animation_parameters(
    stage: Usd.Stage,
    prim_path: str,
    translate: Tuple[float, float, float],
    rotate: float,
    scale: Tuple[float, float, float],
    time_code: Usd.TimeCode,
) -> None:
    """Update the animation parameters (translate, rotate, scale) for a prim at a specific time code."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable.")
    translate_op = xformable.GetOrderedXformOps()[0]
    translate_op.Set(Gf.Vec3d(translate), time_code)
    rotate_op = xformable.GetOrderedXformOps()[1]
    rotate_op.Set(rotate, time_code)
    scale_op = xformable.GetOrderedXformOps()[2]
    scale_op.Set(Gf.Vec3f(scale), time_code)


def apply_bloom_effect(
    stage: Usd.Stage, prim_path: str, width: float, height: float, threshold: float = 0.5, intensity: float = 1.0
) -> None:
    """Apply a bloom post-process effect to a Mesh prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the Mesh prim.
        width (float): The width of the render target.
        height (float): The height of the render target.
        threshold (float, optional): The luminance threshold for the bloom. Defaults to 0.5.
        intensity (float, optional): The intensity multiplier for the bloom. Defaults to 1.0.

    Raises:
        ValueError: If the prim at prim_path is not a valid Mesh prim.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"Prim at path {prim_path} is not a valid Mesh prim.")
    material_path = Sdf.Path(prim_path).AppendChild("BloomMaterial")
    if not stage.GetPrimAtPath(material_path):
        stage.DefinePrim(material_path, "Material")
    shader = UsdShade.Shader.Define(stage, material_path.AppendChild("BloomShader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
    shader.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(threshold)
    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material = UsdShade.Material.Get(stage, material_path)
    material.CreateSurfaceOutput().ConnectToSource(shader_output)
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def copy_prim_with_variants(stage: Usd.Stage, prim_path: Sdf.Path, dest_path: Sdf.Path) -> Usd.Prim:
    """Copy a prim and its variants to a new location in the stage.

    Args:
        stage (Usd.Stage): The stage containing the prim to copy.
        prim_path (Sdf.Path): The path of the prim to copy.
        dest_path (Sdf.Path): The destination path for the copied prim.

    Returns:
        Usd.Prim: The newly created prim copy.
    """
    source_prim = stage.GetPrimAtPath(prim_path)
    if not source_prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    dest_prim = stage.GetPrimAtPath(dest_path)
    if not dest_prim.IsValid():
        dest_prim = stage.DefinePrim(dest_path, source_prim.GetTypeName())
    for key in source_prim.GetAllMetadata():
        if source_prim.HasAuthoredMetadata(key):
            value = source_prim.GetMetadata(key)
            dest_prim.SetMetadata(key, value)
    for prop in source_prim.GetAuthoredProperties():
        dest_prop = dest_prim.GetProperty(prop.GetName())
        if not dest_prop.IsValid():
            dest_prop = dest_prim.CreateAttribute(prop.GetName(), prop.GetTypeName())
            dest_prop.Set(prop.Get())
        else:
            dest_prop.Set(prop.Get())
    variants = source_prim.GetVariantSets()
    for variant_set in variants.GetNames():
        source_vset = source_prim.GetVariantSet(variant_set)
        dest_vset = dest_prim.GetVariantSet(variant_set)
        dest_vset.SetVariantSelection(source_vset.GetVariantSelection())
        for variant in source_vset.GetVariantNames():
            source_vset.SetVariantSelection(variant)
            dest_vset.SetVariantSelection(variant)
            for child in source_prim.GetAllChildren():
                child_dest_path = dest_path.AppendChild(child.GetName())
                copy_prim_with_variants(stage, child.GetPath(), child_dest_path)
    return dest_prim


def create_prim(stage: Usd.Stage, prim_type: str, prim_path: str) -> Usd.Prim:
    """Helper function to create a prim."""
    return stage.DefinePrim(prim_path, prim_type)


def find_and_replace_materials(stage: Usd.Stage, search_material: str, replace_material: str) -> int:
    """
    Find and replace materials on a USD stage.

    Args:
        stage (Usd.Stage): The USD stage to search and replace materials on.
        search_material (str): The material to search for, specified as a prim path.
        replace_material (str): The material to replace with, specified as a prim path.

    Returns:
        int: The number of material bindings that were replaced.
    """
    if not stage:
        raise ValueError("Invalid stage")
    search_prim = stage.GetPrimAtPath(search_material)
    if not search_prim.IsValid():
        raise ValueError(f"Search material prim '{search_material}' not found in stage")
    replace_prim = stage.GetPrimAtPath(replace_material)
    if not replace_prim.IsValid():
        raise ValueError(f"Replace material prim '{replace_material}' not found in stage")
    binding_rels = []
    for prim in stage.TraverseAll():
        binding_api = UsdShade.MaterialBindingAPI(prim)
        direct_binding_rel = binding_api.GetDirectBindingRel()
        if direct_binding_rel:
            binding_rels.append(direct_binding_rel)
    num_replaced = 0
    for rel in binding_rels:
        target_paths = rel.GetTargets()
        if search_prim.GetPath() not in target_paths:
            continue
        rel.RemoveTarget(search_prim.GetPath())
        rel.AddTarget(replace_prim.GetPath())
        num_replaced += 1
    return num_replaced


def apply_vignette_effect(stage: Usd.Stage, radius: float, scale: float) -> None:
    """Apply a vignette effect to all cameras on the given stage.

    The vignette darkens the edges of the rendered image based on the given radius and scale.

    Args:
        stage (Usd.Stage): The USD stage to apply the vignette effect to.
        radius (float): The radius of the vignette effect. Should be between 0 and 1.
        scale (float): The scale of the vignette effect. Should be greater than 0.

    Raises:
        ValueError: If the stage is invalid, or if radius or scale are outside their valid ranges.
    """
    if not stage:
        raise ValueError("Invalid stage")
    if radius < 0 or radius > 1:
        raise ValueError(f"Invalid radius value {radius}. Must be between 0 and 1.")
    if scale <= 0:
        raise ValueError(f"Invalid scale value {scale}. Must be greater than 0.")
    cameras = [p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
    for camera in cameras:
        camera_schema = UsdGeom.Camera(camera)
        splines_attr = camera_schema.GetClippingRangeAttr()
        if not splines_attr:
            splines_attr = camera_schema.CreateClippingRangeAttr()
        splines_attr.Set(Gf.Vec2f(1.0 - radius, 1.0 + radius * scale))


def create_camera(stage: Usd.Stage, path: str) -> UsdGeom.Camera:
    """Helper function to create a camera prim"""
    prim = stage.DefinePrim(path, "Camera")
    return UsdGeom.Camera(prim)


def validate_physics_properties(stage: Usd.Stage, prim_path: str) -> bool:
    """Validate that a prim has the necessary physics properties."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        print(f"No prim found at path: {prim_path}")
        return False
    if not prim.IsA(UsdGeom.Xformable):
        print(f"Prim at path {prim_path} is not a UsdGeomXformable")
        return False
    physics_api = UsdPhysics.RigidBodyAPI(prim)
    if not physics_api:
        return False
    mass_attr = physics_api.GetRigidBodyEnabledAttr()
    if not mass_attr or not mass_attr.HasAuthoredValue():
        return False
    collider_api = UsdPhysics.CollisionAPI(prim)
    if not collider_api:
        return False
    return True


def create_prim(stage, prim_type, prim_path):
    """Create a new prim with the specified type and path."""
    prim = stage.DefinePrim(prim_path, prim_type)
    return prim


def align_objects_to_bounding_box(stage: Usd.Stage, prim_paths: List[str], target_prim_path: str):
    """Align objects to the bounding box of a target prim."""
    if not stage:
        raise ValueError("Invalid stage")
    if not prim_paths:
        raise ValueError("No prim paths provided")
    if not target_prim_path:
        raise ValueError("No target prim path provided")
    target_prim = stage.GetPrimAtPath(target_prim_path)
    if not target_prim.IsValid():
        raise ValueError(f"Target prim '{target_prim_path}' does not exist")
    bounding_box_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    bounding_box = bounding_box_cache.ComputeWorldBound(target_prim)
    target_translation = bounding_box.ComputeCentroid()
    target_rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), 0)
    target_scale = max(bounding_box.GetBox().GetSize())
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim '{prim_path}' does not exist. Skipping.")
            continue
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            print(f"Warning: Prim '{prim_path}' is not transformable. Skipping.")
            continue
        translate_op = add_translate_op(xformable)
        translate_op.Set(target_translation)
        rotate_op = add_rotate_xyz_op(xformable)
        rotate_op.Set(target_rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)))
        scale_op = add_scale_op(xformable)
        scale_op.Set((target_scale, target_scale, target_scale))


def transfer_materials_between_stages(source_stage: Usd.Stage, dest_stage: Usd.Stage):
    """Transfer materials from one stage to another."""
    source_materials = []
    for prim in source_stage.TraverseAll():
        if prim.IsA(UsdShade.Material):
            source_materials.append(prim)
    for source_material_prim in source_materials:
        material_path = source_material_prim.GetPath()
        if not dest_stage.GetPrimAtPath(material_path):
            dest_material = UsdShade.Material.Define(dest_stage, material_path)
            source_shader = UsdShade.Shader(source_material_prim.GetChild("Shader"))
            if source_shader:
                dest_shader = UsdShade.Shader.Define(dest_stage, dest_material.GetPath().AppendChild("Shader"))
                dest_shader.CreateIdAttr(source_shader.GetIdAttr().Get())
                for input in source_shader.GetInputs():
                    if input.HasConnectedSource():
                        source = input.GetConnectedSource()
                        UsdShade.ConnectableAPI.ConnectToSource(
                            dest_shader.CreateInput(input.GetBaseName(), input.GetTypeName()),
                            UsdShade.ConnectableAPI(dest_stage.OverridePrim(source[0].GetPath())),
                            source[1].GetBaseName(),
                        )
                    else:
                        dest_shader.CreateInput(input.GetBaseName(), input.GetTypeName()).Set(input.Get())
                for output in source_shader.GetOutputs():
                    dest_shader.CreateOutput(output.GetBaseName(), output.GetTypeName())


def create_dynamic_skybox(stage: Usd.Stage, prim_path: str = "/World/DynamicSkybox") -> Usd.Prim:
    """Create a dynamic skybox prim with a dome light and skysphere mesh.

    Args:
        stage (Usd.Stage): The USD stage to create the skybox prim on.
        prim_path (str, optional): The path to create the skybox prim at. Defaults to "/World/DynamicSkybox".

    Returns:
        Usd.Prim: The created skybox prim.
    """
    skybox_prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    dome_light_prim = UsdLux.DomeLight.Define(stage, prim_path + "/DomeLight").GetPrim()
    UsdLux.DomeLight(dome_light_prim).CreateTextureFileAttr("sky_texture.exr")
    UsdLux.DomeLight(dome_light_prim).CreateTextureFormatAttr("latlong")
    UsdLux.DomeLight(dome_light_prim).CreateIntensityAttr(1000.0)
    skysphere_prim = UsdGeom.Sphere.Define(stage, prim_path + "/SkySphere").GetPrim()
    UsdGeom.Sphere(skysphere_prim).CreateRadiusAttr(1000000.0)
    material_prim = UsdShade.Material.Define(stage, prim_path + "/SkySphereMaterial")
    dome_light_shader = UsdShade.Shader.Define(stage, material_prim.GetPath().AppendChild("DomeLightShader"))
    dome_light_shader.CreateIdAttr("UsdPreviewSurface")
    dome_light_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("sky_texture.exr")
    material_prim.CreateSurfaceOutput().ConnectToSource(dome_light_shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(skysphere_prim).Bind(material_prim)
    Usd.ModelAPI(skybox_prim).SetKind(Kind.Tokens.group)
    return skybox_prim


def test_create_dynamic_skybox():
    stage = Usd.Stage.CreateInMemory()
    skybox_prim = create_dynamic_skybox(stage)
    assert skybox_prim.GetTypeName() == "Xform"
    assert skybox_prim.GetPath() == Sdf.Path("/World/DynamicSkybox")
    dome_light_prim = skybox_prim.GetChild("DomeLight")
    assert dome_light_prim
    assert dome_light_prim.GetTypeName() == "DomeLight"
    assert UsdLux.DomeLight(dome_light_prim).GetTextureFileAttr().Get() == "sky_texture.exr"
    assert UsdLux.DomeLight(dome_light_prim).GetTextureFormatAttr().Get() == "latlong"
    assert UsdLux.DomeLight(dome_light_prim).GetIntensityAttr().Get() == 1000.0
    skysphere_prim = skybox_prim.GetChild("SkySphere")
    assert skysphere_prim
    assert skysphere_prim.GetTypeName() == "Sphere"
    assert UsdGeom.Sphere(skysphere_prim).GetRadiusAttr().Get() == 1000000.0
    material_prim = skybox_prim.GetChild("SkySphereMaterial")
    assert material_prim
    dome_light_shader = material_prim.GetChild("DomeLightShader")
    assert dome_light_shader
    assert UsdShade.Shader(dome_light_shader).GetIdAttr().Get() == "UsdPreviewSurface"
    assert UsdShade.Shader(dome_light_shader).GetInput("file").Get() == "sky_texture.exr"
    assert Usd.ModelAPI(skybox_prim).GetKind() == Kind.Tokens.group
    print(stage.GetRootLayer().ExportToString())


def update_material_textures(stage: Usd.Stage, material_path: str, textures: Dict[str, str]):
    """Update the textures of a material prim.

    Args:
        stage (Usd.Stage): The stage containing the material prim.
        material_path (str): The path to the material prim.
        textures (Dict[str, str]): A dictionary mapping texture file attribute names to texture file paths.
    """
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim:
        raise ValueError(f"Material prim not found at path: {material_path}")
    material = UsdShade.Material(material_prim)
    if not material:
        raise ValueError(f"Prim at path {material_path} is not a Material")
    for attr_name, file_path in textures.items():
        shader_input = material.CreateInput(attr_name, Sdf.ValueTypeNames.Asset)
        shader_input.Set(file_path)


def generate_light_maps(stage: Usd.Stage) -> None:
    """Generate light maps for all UsdLux light prims on the given stage."""
    for prim in stage.TraverseAll():
        if prim.IsA(UsdLux.LightAPI):
            light = UsdLux.LightAPI(prim)
            if not light.GetPath().HasPrefix("/World/Lights"):
                continue
            transform = light.GetTransform()
            if not transform:
                continue
            intensity = light.GetIntensityAttr().Get()
            if intensity is None:
                intensity = 1.0
            color = light.GetColorAttr().Get()
            if color is None:
                color = Gf.Vec3f(1, 1, 1)
            light_map = f"Light map for {prim.GetPath()} with intensity {intensity} and color {color}"
            prim.CreateAttribute("lightMap", Sdf.ValueTypeNames.String).Set(light_map)


def load_specific_prims(stage: Usd.Stage, prim_paths: List[str]) -> None:
    """Load only the specified prims and their descendants.

    Args:
        stage (Usd.Stage): The stage to load prims on.
        prim_paths (List[str]): A list of prim paths to load.

    Raises:
        ValueError: If any of the specified prim paths are invalid.
    """
    for prim_path in prim_paths:
        if not Sdf.Path(prim_path).IsAbsolutePath():
            raise ValueError(f"Prim path {prim_path} is not an absolute path.")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim:
            prim.Load()


def get_all_cached_stage_paths(cache: Usd.StageCache) -> List[str]:
    """Get the paths of all stages in the cache."""
    stages = cache.GetAllStages()
    stage_paths = []
    for stage in stages:
        root_layer = stage.GetRootLayer()
        if root_layer:
            stage_paths.append(root_layer.realPath)
    return stage_paths


def merge_caches(cache1: Usd.StageCache, cache2: Usd.StageCache) -> Usd.StageCache:
    """Merge two stage caches into a new cache."""
    merged_cache = Usd.StageCache()
    for stage in cache1.GetAllStages():
        merged_cache.Insert(stage)
    for stage in cache2.GetAllStages():
        stage_id = cache2.GetId(stage)
        if not merged_cache.Contains(stage_id):
            merged_cache.Insert(stage)
        else:
            pass
    return merged_cache


def swap_cache_contents(cache1: Usd.StageCache, cache2: Usd.StageCache):
    """Swap the contents of two Usd.StageCache objects."""
    if cache1 is cache2:
        raise ValueError("Cannot swap a cache with itself")
    cache1.swap(cache2)


def clear_cache_except_paths(cache: Usd.StageCache, keep_paths: List[str]):
    """Clear all stages from the cache except those with root layer paths in keep_paths."""
    stages = cache.GetAllStages()
    stages_to_erase = [stage for stage in stages if stage.GetRootLayer().realPath not in keep_paths]
    for stage in stages_to_erase:
        cache.Erase(stage)
        stage = None


def find_or_create_stage(
    cache: Usd.StageCache, layer: Sdf.Layer, pathResolverContext: Optional[Ar.ResolverContext] = None
) -> Usd.Stage:
    """Find or create a new stage in the cache based on the given root layer.

    Args:
        cache (Usd.StageCache): The stage cache to search or insert into.
        layer (Sdf.Layer): The root layer for the stage.
        pathResolverContext (Optional[Ar.ResolverContext], optional): Optional
            path resolver context to use when creating the stage. Defaults to None.

    Returns:
        Usd.Stage: The found or created stage.
    """
    existing_stage = cache.FindOneMatching(layer, sessionLayer=None, pathResolverContext=pathResolverContext)
    if existing_stage:
        return existing_stage
    new_stage = Usd.Stage.Open(layer, sessionLayer=None, pathResolverContext=pathResolverContext)
    cache_id = cache.Insert(new_stage)
    return new_stage


def get_cache_size(cache: Usd.StageCache) -> int:
    """Get the number of stages in the given UsdStageCache.

    Args:
        cache (Usd.StageCache): The stage cache to query.

    Returns:
        int: The number of stages in the cache.
    """
    num_stages = cache.Size()
    return num_stages


def get_debug_info(cache: Usd.StageCache) -> str:
    """Return debug information about the stage cache."""
    debug_name = cache.GetDebugName()
    if not debug_name:
        debug_name = f"Cache at {hex(id(cache))}"
    num_stages = cache.Size()
    stages = cache.GetAllStages()
    stage_info = "\n".join((f"  {cache.GetId(stage)}: {stage.GetRootLayer().identifier}" for stage in stages))
    debug_info = f"{debug_name}:\nNumber of stages: {num_stages}\nStages:\n{stage_info}"
    return debug_info


def set_debug_name_and_return(cache: Usd.StageCache, debug_name: str) -> Usd.StageCache:
    """Set the debug name for the cache and return the cache."""
    if not cache:
        raise ValueError("Invalid StageCache object")
    cache.SetDebugName(debug_name)
    return cache


def get_stage_ids(cache: Usd.StageCache) -> List[Usd.StageCache.Id]:
    """Get a list of all stage IDs in the cache."""
    stages = cache.GetAllStages()
    ids = [cache.GetId(stage) for stage in stages]
    return ids


def retrieve_and_clear_cache(stage_cache: Usd.StageCache) -> List[Usd.Stage]:
    """Retrieve all stages from the cache and clear it.

    Args:
        stage_cache (Usd.StageCache): The stage cache to retrieve stages from and clear.

    Returns:
        List[Usd.Stage]: A list of all stages that were in the cache.
    """
    stages = stage_cache.GetAllStages()
    stage_cache.Clear()
    return stages


def generate_temp_usd_file_path(output_dir: str = None) -> str:
    """Generate a temporary file path with a random name for a USD file."""
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=10))
    if output_dir:
        return output_dir + "/temp_" + random_string + ".usda"
    else:
        return "C:\\Users\\horde\\AppData\\Local\\Temp/usda_" + random_string + ".usda"


def find_stage_by_debug_name(cache: Usd.StageCache, debug_name: str) -> Optional[Usd.Stage]:
    """Find a stage in the cache by its debug name.

    Args:
        cache (Usd.StageCache): The stage cache to search.
        debug_name (str): The debug name of the stage to find.

    Returns:
        Optional[Usd.Stage]: The stage with the given debug name, or None if not found.
    """
    stages = cache.GetAllStages()
    for stage in stages:
        stage_id = cache.GetId(stage)
        stage_debug_name = cache.GetDebugName()
        if stage_debug_name == debug_name:
            return stage
    return None


def erase_and_return_count(
    cache: Usd.StageCache,
    root_layer: Sdf.Layer,
    session_layer: Sdf.Layer = None,
    path_resolver_context: Ar.ResolverContext = None,
) -> int:
    """Erase all stages from the cache that match the given arguments and return the number erased.

    This is equivalent to:
    num_erased = len(cache.FindAllMatching(root_layer, session_layer, path_resolver_context))
    for stage in cache.FindAllMatching(root_layer, session_layer, path_resolver_context):
        cache.Erase(stage)
    return num_erased

    Parameters:
        cache (Usd.StageCache): The stage cache to erase stages from.
        root_layer (Sdf.Layer): The root layer of the stages to erase.
        session_layer (Sdf.Layer): Optional session layer of the stages to erase.
        path_resolver_context (Ar.ResolverContext): Optional asset resolver context.

    Returns:
        int: The number of stages erased from the cache.
    """
    matching_stages = cache.FindAllMatching(root_layer, session_layer, path_resolver_context)
    matching_stages_list = list(matching_stages)
    num_erased = len(matching_stages_list)
    for stage in matching_stages_list:
        cache.Erase(stage)
    return num_erased


def remove_all_stages_by_layer(cache: Usd.StageCache, layer: Sdf.Layer) -> int:
    """Remove all stages from the cache that have the given layer as root layer.

    Args:
        cache (Usd.StageCache): The stage cache to remove stages from.
        layer (Sdf.Layer): The layer to match against root layers.

    Returns:
        int: The number of stages removed from the cache.
    """
    stages = cache.GetAllStages()
    num_removed = 0
    for stage in stages:
        if stage.GetRootLayer() == layer:
            if cache.Erase(stage):
                num_removed += 1
    return num_removed


def cache_and_return_all_stages(cache: Usd.StageCache, stage_paths: List[str]) -> List[Usd.Stage]:
    """
    Cache the stages at the given paths and return all stages in the cache.

    Args:
        cache (Usd.StageCache): The stage cache to use.
        stage_paths (List[str]): The paths of the stages to cache.

    Returns:
        List[Usd.Stage]: All stages in the cache after caching the given stages.
    """
    for stage_path in stage_paths:
        stage = Usd.Stage.Open(stage_path)
        if stage:
            cache.Insert(stage)
        else:
            print(f"Warning: Failed to open stage at path {stage_path}")
    all_stages = list(cache.GetAllStages())
    return all_stages


def get_stage_by_id(cache: Usd.StageCache, stage_id: Usd.StageCache.Id) -> Usd.Stage:
    """
    Retrieve a stage from the cache by its ID.

    Args:
        cache (Usd.StageCache): The stage cache to search.
        stage_id (Usd.StageCache.Id): The ID of the stage to retrieve.

    Returns:
        Usd.Stage: The stage with the given ID, or None if not found.
    """
    if cache.Contains(stage_id):
        stage = cache.Find(stage_id)
        return stage
    else:
        return None


def find_stages_by_session_layer(cache: Usd.StageCache, session_layer: Sdf.Layer) -> List[Usd.Stage]:
    """Find all stages in the cache that have the given session layer."""
    matching_stages = []
    for stage in cache.GetAllStages():
        if stage.GetSessionLayer() == session_layer:
            matching_stages.append(stage)
    return matching_stages


def get_or_open_stage(
    cache: Usd.StageCache, layer_path: str, path_resolver_context: Optional[Usd.StagePopulationMask] = None
) -> Usd.Stage:
    """
    Get a stage from the cache if it exists, otherwise open it and add to the cache.

    Args:
        cache (Usd.StageCache): The stage cache to use.
        layer_path (str): The path to the layer to open.
        path_resolver_context (Optional[Usd.StagePopulationMask], optional): The stage population
            mask to use when opening the stage. Defaults to None.

    Returns:
        Usd.Stage: The stage from the cache or newly opened.
    """
    root_layer = Sdf.Layer.FindOrOpen(layer_path)
    stage = cache.FindOneMatching(root_layer)
    if not stage:
        stage = Usd.Stage.Open(root_layer, path_resolver_context)
        if not stage:
            raise RuntimeError(f"Failed to open stage at path: {layer_path}")
        cache.Insert(stage)
    return stage


def insert_and_get_id(cache: Usd.StageCache, stage: Optional[Usd.Stage]) -> Optional[Usd.StageCache.Id]:
    """Insert a stage into the cache and return its Id.

    If the stage is already in the cache, return its existing Id.
    If the stage is invalid or None, return None.
    """
    if not stage:
        return None
    existing_id = cache.GetId(stage)
    if existing_id:
        return existing_id
    return cache.Insert(stage)


def get_stage_cache_id_from_long_int(long_int: int) -> Usd.StageCache.Id:
    """
    Get a Usd.StageCache.Id from a long integer.

    Args:
        long_int (int): The long integer to convert to a Usd.StageCache.Id.

    Returns:
        Usd.StageCache.Id: The Usd.StageCache.Id created from the long integer.

    Raises:
        ValueError: If the provided long integer is not a valid Usd.StageCache.Id.
    """
    stage_cache_id = Usd.StageCache.Id.FromLongInt(long_int)
    if not Usd.StageCache.Id.IsValid(stage_cache_id):
        raise ValueError(f"The provided long integer {long_int} is not a valid Usd.StageCache.Id.")
    return stage_cache_id


def convert_stage_cache_id_to_long_int(stage_cache_id: Usd.StageCache.Id) -> int:
    """Convert a Usd.StageCache.Id to a long int.

    Args:
        stage_cache_id (Usd.StageCache.Id): The stage cache ID to convert.

    Returns:
        int: The converted long int value.

    Raises:
        ValueError: If the input stage_cache_id is not valid.
    """
    if not Usd.StageCache.Id.IsValid(stage_cache_id):
        raise ValueError("Invalid stage cache ID.")
    long_int_value = Usd.StageCache.Id.ToLongInt(stage_cache_id)
    return long_int_value


def convert_stage_cache_id_to_string(stage_cache_id: Usd.StageCache.Id) -> str:
    """Convert a Usd.StageCache.Id to a string representation."""
    if not stage_cache_id.IsValid():
        raise ValueError("Invalid stage cache ID.")
    stage_cache_id_long = stage_cache_id.ToLongInt()
    stage_cache_id_str = str(stage_cache_id_long)
    return stage_cache_id_str


def validate_stage_cache_id(cache_id: str) -> bool:
    """Validate a Usd.StageCache.Id.

    Args:
        cache_id (str): The cache ID to validate.

    Returns:
        bool: True if the cache ID is valid, False otherwise.
    """
    try:
        cache_id_obj = Usd.StageCache.Id.FromString(cache_id)
        if not cache_id_obj.IsValid():
            return False
        cache_id_str = cache_id_obj.ToString()
        converted_id = Usd.StageCache.Id.FromString(cache_id_str)
        if cache_id_obj != converted_id:
            return False
        cache_id_int = cache_id_obj.ToLongInt()
        converted_id = Usd.StageCache.Id.FromLongInt(cache_id_int)
        if cache_id_obj != converted_id:
            return False
    except Exception:
        return False
    return True


def open_stage_with_cache(stage_url: str, stage_cache: Usd.StageCache) -> Usd.Stage:
    """Open a USD stage using a cache context.

    Args:
        stage_url (str): The URL of the stage to open.
        stage_cache (Usd.StageCache): The cache to use when opening the stage.

    Returns:
        Usd.Stage: The opened stage.
    """
    with Usd.StageCacheContext(stage_cache):
        stage = Usd.Stage.Open(stage_url)
        if not stage:
            raise RuntimeError(f"Failed to open stage: {stage_url}")
        if stage_cache.Contains(stage):
            print(f"Retrieved stage from cache: {stage_url}")
        else:
            print(f"Opened new stage and added to cache: {stage_url}")
    return stage


def merge_stages_from_caches(stage_caches: List[Usd.StageCache]) -> Usd.StageCache:
    """
    Merge the stages from multiple UsdStageCaches into a single cache.

    Args:
        stage_caches (List[Usd.StageCache]): A list of UsdStageCaches to merge.

    Returns:
        Usd.StageCache: A new UsdStageCache containing all the stages from the input caches.
    """
    merged_cache = Usd.StageCache()
    for cache in stage_caches:
        stages = cache.GetAllStages()
        for stage in stages:
            root_layer_id = stage.GetRootLayer().identifier
            stage_id = Usd.StageCache.Id.FromString(root_layer_id)
            existing_stage = merged_cache.Find(stage_id)
            if existing_stage is None:
                merged_cache.Insert(stage)
            else:
                existing_edit_target = existing_stage.GetEditTarget().GetLayer()
                current_edit_target = stage.GetEditTarget().GetLayer()
                if current_edit_target.empty:
                    continue
                elif existing_edit_target.empty or current_edit_target.identifier != existing_edit_target.identifier:
                    merged_cache.EraseAll(stage_id)
                    merged_cache.Insert(stage)
    return merged_cache


def cache_optimized_open(cache: Usd.StageCache, file_path: str, populate_cache: bool = True) -> Optional[Usd.Stage]:
    """
    Open a USD stage using a cache for optimization.

    Args:
        cache (Usd.StageCache): The cache to use for optimizing stage loading.
        file_path (str): The path to the USD file to open.
        populate_cache (bool, optional): Whether to populate the cache with the newly
            opened stage. Defaults to True.

    Returns:
        Optional[Usd.Stage]: The opened stage, or None if the stage could not be opened.
    """
    root_layer = Sdf.Layer.FindOrOpen(file_path)
    if not root_layer:
        return None
    cached_stage = cache.FindOneMatching(root_layer)
    if cached_stage:
        return cached_stage
    if populate_cache:
        with Usd.StageCacheContext(cache):
            stage = Usd.Stage.Open(root_layer)
    else:
        stage = Usd.Stage.Open(root_layer)
    return stage


def open_and_modify_stage_with_cache(cache: Usd.StageCache, file_path: str) -> Tuple[Usd.Stage, Usd.Prim]:
    """
    Open a USD stage using a cache context, modify the stage, and return the stage and modified prim.

    Args:
        cache (Usd.StageCache): The stage cache to use.
        file_path (str): The path to the USD file to open.

    Returns:
        A tuple containing the opened stage and the modified prim.
    """
    with Usd.StageCacheContext(cache):
        stage = Usd.Stage.Open(file_path)
        default_prim = stage.GetDefaultPrim()
        if not default_prim:
            raise ValueError("No default prim found in the stage.")
        new_prim_path = default_prim.GetPath().AppendChild("NewPrim")
        new_prim = stage.DefinePrim(new_prim_path, "Xform")
        UsdGeom.XformCommonAPI(new_prim).SetTranslate(Gf.Vec3d(1.0, 2.0, 3.0))
        stage.GetRootLayer().Save()
    return (stage, new_prim)


def cache_context_operations(value_by_name_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Get the values for each name in the dictionary using StageCacheContextBlockType."""
    result_dict = {}
    for name, value in value_by_name_dict.items():
        result_value = Usd.StageCacheContextBlockType.GetValueFromName(name)
        result_dict[name] = result_value
    return result_dict


def get_loading_state_summary(stage: Usd.Stage, rules: Usd.StageLoadRules) -> Dict[str, int]:
    """Get a summary of the loading state of prims in a stage based on the given rules.

    Args:
        stage (Usd.Stage): The USD stage to analyze.
        rules (Usd.StageLoadRules): The stage load rules to apply.

    Returns:
        Dict[str, int]: A dictionary with keys 'loaded', 'unloaded', and 'loaded_with_no_descendants',
                        and values representing the count of prims in each state.
    """
    loaded_count = 0
    unloaded_count = 0
    loaded_with_no_descendants_count = 0
    for prim in stage.TraverseAll():
        prim_path = prim.GetPath()
        if rules.IsLoaded(prim_path):
            loaded_count += 1
            if rules.IsLoadedWithNoDescendants(prim_path):
                loaded_with_no_descendants_count += 1
        else:
            unloaded_count += 1
    summary = {
        "loaded": loaded_count,
        "unloaded": unloaded_count,
        "loaded_with_no_descendants": loaded_with_no_descendants_count,
    }
    return summary


def unload_all_except_inclusions(rules: Usd.StageLoadRules, inclusions):
    """Unloads all payloads except those specified in inclusions.

    Args:
        rules (Usd.StageLoadRules): The stage load rules to modify.
        inclusions (list[Sdf.Path]): A list of paths to exclude from unloading.
    """
    rules.LoadNone()
    for path in inclusions:
        rules.LoadWithDescendants(path)
    rules.Minimize()


def reset_and_apply_load_rules(stage: Usd.Stage, load_paths: list, unload_paths: list) -> None:
    """Reset the stage load rules and apply new load/unload paths.

    Args:
        stage (Usd.Stage): The stage to modify load rules for.
        load_paths (list): List of prim paths to load.
        unload_paths (list): List of prim paths to unload.
    """
    load_rules = stage.GetLoadRules()
    load_rules.SetRules([])
    load_sdf_paths = [Sdf.Path(path) for path in load_paths]
    unload_sdf_paths = [Sdf.Path(path) for path in unload_paths]
    for unload_path in unload_sdf_paths:
        load_rules.Unload(unload_path)
    for load_path in load_sdf_paths:
        load_rules.LoadWithDescendants(load_path)
    stage.SetLoadRules(load_rules)


def get_effective_load_rules(stage: Usd.Stage, prim_path: str) -> Usd.StageLoadRules.Rule:
    """
    Get the effective load rule for a given prim path in a USD stage.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The prim path to get the effective load rule for.

    Returns:
        Usd.StageLoadRules.Rule: The effective load rule for the prim path.
    """
    load_rules = stage.GetLoadRules()
    path = Sdf.Path(prim_path)
    effective_rule = load_rules.GetEffectiveRuleForPath(path)
    return effective_rule


def get_load_rules_for_subtree(rules: Usd.StageLoadRules, path: Sdf.Path) -> Usd.StageLoadRules:
    """Get a subset of the given load rules that apply to the given prim path and its descendants."""
    subtree_rules = Usd.StageLoadRules()
    for rule_path, rule in rules.GetRules():
        if path.HasPrefix(rule_path) or rule_path.HasPrefix(path):
            subtree_rules.AddRule(rule_path, rule)
    subtree_rules.Minimize()
    return subtree_rules


def remove_redundant_load_rules(rules: Usd.StageLoadRules) -> None:
    """Remove redundant rules from a StageLoadRules object.

    A rule is considered redundant if it does not change the effective load state
    of any path when compared to the other rules in the set.

    Parameters:
        rules (Usd.StageLoadRules): The StageLoadRules object to minimize.
    """
    current_rules = rules.GetRules()
    minimized_rules = Usd.StageLoadRules()
    for path, rule in current_rules:
        effective_rule = minimized_rules.GetEffectiveRuleForPath(path)
        if effective_rule != rule:
            minimized_rules.AddRule(path, rule)
    rules.SetRules(minimized_rules.GetRules())


def unload_prims_by_pattern(stage: Usd.Stage, pattern: str) -> None:
    """Unload prims from the stage that match the given pattern.

    Args:
        stage (Usd.Stage): The stage to unload prims from.
        pattern (str): The pattern to match prim paths against.
    """
    load_rules = Usd.StageLoadRules()
    for prim in stage.Traverse():
        if prim.IsLoaded() and str(prim.GetPath()).startswith(pattern):
            load_rules.Unload(prim.GetPath())
    stage.SetLoadRules(load_rules)


def load_all_except_exclusions(stage: Usd.Stage, exclusion_paths: set) -> Usd.StageLoadRules:
    """Create StageLoadRules that load everything except the specified paths and their descendants."""
    load_rules = Usd.StageLoadRules()
    load_rules.LoadAll()
    for path in exclusion_paths:
        if not path.IsPrimPath():
            raise ValueError(f"Invalid exclusion path: {path}")
        load_rules.Unload(path)
    load_rules.Minimize()
    return load_rules


def toggle_prim_loading(stage: Usd.Stage, prim_path: str) -> bool:
    """Toggle the load state of a prim and its descendants.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.

    Returns:
        bool: True if the prim is now loaded, False if it is now unloaded.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"No prim found at path {prim_path}")
    load_rules = stage.GetLoadRules()
    is_loaded = load_rules.IsLoaded(prim_path)
    if is_loaded:
        load_rules.Unload(prim_path)
    else:
        load_rules.LoadWithDescendants(prim_path)
    stage.SetLoadRules(load_rules)
    return not is_loaded


def sync_load_rules_between_stages(source_stage: Usd.Stage, target_stage: Usd.Stage):
    """Synchronize the load rules from the source stage to the target stage."""
    source_load_rules = source_stage.GetLoadRules()
    rules = source_load_rules.GetRules()
    target_load_rules = Usd.StageLoadRules()
    for path, rule in rules:
        target_load_rules.AddRule(path, rule)
    target_stage.SetLoadRules(target_load_rules)


def load_all_descendants_of_prims(stage: Usd.Stage, prim_paths: Sequence[str]) -> None:
    """Load all descendant prims of the given prim paths.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (Sequence[str]): The paths of the prims to load descendants for.
    """
    load_rules = Usd.StageLoadRules()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        load_rules.LoadWithDescendants(prim.GetPath())
    stage.SetLoadRules(load_rules)
    stage.Reload()


def apply_load_rules_to_prims(
    stage: Usd.Stage, prim_paths: List[str], load_rules: Dict[str, Usd.StageLoadRules.Rule]
) -> None:
    """Apply load rules to the specified prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths to apply load rules to.
        load_rules (Dict[str, Usd.StageLoadRules.Rule]): Dictionary mapping prim paths to load rules.
    """
    if not stage:
        raise ValueError("Invalid USD stage.")
    if not prim_paths:
        raise ValueError("Prim paths list is empty.")
    if not load_rules:
        raise ValueError("Load rules dictionary is empty.")
    stage_load_rules = Usd.StageLoadRules()
    for prim_path in prim_paths:
        if prim_path in load_rules:
            load_rule = load_rules[prim_path]
            stage_load_rules.AddRule(prim_path, load_rule)
        else:
            stage_load_rules.AddRule(prim_path, Usd.StageLoadRules.NoneRule)
    stage.SetLoadRules(stage_load_rules)


def get_masked_subtree_paths(stage: Usd.Stage, mask: Usd.StagePopulationMask) -> list:
    """
    Return a list of prim paths that are included in the given population mask and represent subtrees.

    A subtree path is a path where the mask includes the path itself and all its descendants.

    Parameters:
        stage (Usd.Stage): The USD stage to query.
        mask (Usd.StagePopulationMask): The population mask to filter the prim paths.

    Returns:
        list: A list of prim paths representing subtrees included in the mask.
    """
    subtree_paths = []
    for prim in stage.TraverseAll():
        prim_path = prim.GetPath()
        if mask.IncludesSubtree(prim_path):
            subtree_paths.append(prim_path)
    return subtree_paths


def combine_masks_and_optimize(*masks: Usd.StagePopulationMask) -> Usd.StagePopulationMask:
    """Combine multiple StagePopulationMask instances and optimize the result."""
    if not masks:
        return Usd.StagePopulationMask.All()
    combined_mask = Usd.StagePopulationMask()
    for mask in masks:
        combined_mask = combined_mask.GetUnion(mask)
    optimized_mask = Usd.StagePopulationMask()
    for path in combined_mask.GetPaths():
        if not optimized_mask.IncludesSubtree(path):
            optimized_mask = optimized_mask.GetUnion(path)
    return optimized_mask


def get_mask_including_subtree(path: Sdf.Path) -> Usd.StagePopulationMask:
    """
    Return a StagePopulationMask that includes the given path and all paths descendant to it.

    Parameters:
        path (Sdf.Path): The path to include in the mask along with its subtree.

    Returns:
        Usd.StagePopulationMask: A mask that includes the given path and its subtree.
    """
    mask = Usd.StagePopulationMask()
    if not path.IsAbsolutePath():
        raise ValueError(f"Path '{path}' is not an absolute path.")
    if not path.IsPrimPath():
        raise ValueError(f"Path '{path}' is not a prim path.")
    mask = mask.Add(path)
    return mask


def create_mask_with_paths(paths: List[Union[str, Sdf.Path]]) -> Usd.StagePopulationMask:
    """Create a StagePopulationMask from a list of paths.

    The paths can be either strings or Sdf.Path objects. They must be absolute
    prim paths (or the absolute root path). No path in the set can be an ancestor
    path of any other path in the set other than itself.

    Parameters:
        paths (List[Union[str, Sdf.Path]]): The paths to include in the mask.

    Returns:
        Usd.StagePopulationMask: The created mask.

    Raises:
        ValueError: If any of the paths are not absolute or if any path is an
                    ancestor of another path in the set.
    """
    mask = Usd.StagePopulationMask()
    sdf_paths = [Sdf.Path(p) if isinstance(p, str) else p for p in paths]
    if not all((p.IsAbsolutePath() for p in sdf_paths)):
        raise ValueError("All paths must be absolute.")
    for i, p1 in enumerate(sdf_paths):
        for p2 in sdf_paths[i + 1 :]:
            if p1.HasPrefix(p2) or p2.HasPrefix(p1):
                raise ValueError(f"Path {p1} is an ancestor or descendant of {p2}.")
    for path in sdf_paths:
        mask = mask.GetUnion(path)
    return mask


def add_paths_to_mask(mask: Usd.StagePopulationMask, paths: List[str]) -> Usd.StagePopulationMask:
    """Add a list of paths to a StagePopulationMask.

    Args:
        mask (Usd.StagePopulationMask): The mask to add paths to.
        paths (List[str]): A list of absolute prim paths to add to the mask.

    Returns:
        Usd.StagePopulationMask: The updated mask with the paths added.
    """
    new_mask = Usd.StagePopulationMask()
    for path in mask.GetPaths():
        new_mask = new_mask.Add(path)
    for path in paths:
        sdf_path = Sdf.Path(path)
        if not sdf_path.IsAbsolutePath() or not sdf_path.IsPrimPath():
            raise ValueError(f"Path {path} is not an absolute prim path.")
        new_mask = new_mask.Add(sdf_path)
    return new_mask


def mask_intersection_with_paths(mask: Usd.StagePopulationMask, paths: Sequence[Sdf.Path]) -> Usd.StagePopulationMask:
    """
    Return a new mask that is the intersection of the given mask and the mask containing the given paths.

    The resulting mask will include only paths that are included by both the given mask and the paths.
    """
    path_mask = Usd.StagePopulationMask()
    for path in paths:
        path_mask = path_mask.Add(path)
    result_mask = mask.GetIntersection(path_mask)
    return result_mask


def save_stage_with_mask(stage: Usd.Stage, mask: Usd.StagePopulationMask, file_path: str) -> bool:
    """Saves the given stage with the specified population mask to a file.

    Args:
        stage (Usd.Stage): The stage to be saved.
        mask (Usd.StagePopulationMask): The population mask to apply when saving the stage.
        file_path (str): The file path where the stage should be saved.

    Returns:
        bool: True if the stage was successfully saved, False otherwise.
    """
    if not stage or not stage.GetPseudoRoot():
        print("Error: Invalid stage.")
        return False
    if not file_path:
        print("Error: Invalid file path.")
        return False
    try:
        masked_stage = Usd.Stage.OpenMasked(stage.GetRootLayer(), mask)
        if not masked_stage.Export(file_path):
            print(f"Error: Failed to export stage to {file_path}")
            return False
        print(f"Stage successfully saved to {file_path} with population mask.")
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def create_intersection_mask(mask1: Usd.StagePopulationMask, mask2: Usd.StagePopulationMask) -> Usd.StagePopulationMask:
    """Create a new mask that is the intersection of two given masks.

    Args:
        mask1 (Usd.StagePopulationMask): The first mask.
        mask2 (Usd.StagePopulationMask): The second mask.

    Returns:
        Usd.StagePopulationMask: A new mask that is the intersection of mask1 and mask2.
    """
    if mask1.IsEmpty() or mask2.IsEmpty():
        return Usd.StagePopulationMask()
    if mask1.GetPaths() == [Sdf.Path.absoluteRootPath] or mask2.GetPaths() == [Sdf.Path.absoluteRootPath]:
        return mask1 if mask2.GetPaths() == [Sdf.Path.absoluteRootPath] else mask2
    intersection_mask = Usd.StagePopulationMask()
    for path in mask1.GetPaths():
        if mask2.Includes(path):
            intersection_mask = intersection_mask.Add(path)
    return intersection_mask


def filter_prims_by_mask(stage: Usd.Stage, mask: Usd.StagePopulationMask) -> List[Usd.Prim]:
    """Filter prims in a stage by a population mask.

    Args:
        stage (Usd.Stage): The USD stage to filter prims from.
        mask (Usd.StagePopulationMask): The population mask to filter prims with.

    Returns:
        List[Usd.Prim]: A list of prims that match the population mask.
    """
    matched_prims: List[Usd.Prim] = []
    for prim in stage.Traverse():
        if mask.Includes(prim.GetPath()):
            matched_prims.append(prim)
        elif not mask.IncludesSubtree(prim.GetPath()):
            prim.SetActive(False)
    return matched_prims


def mask_union_with_paths(mask: Usd.StagePopulationMask, paths: Sequence[Sdf.Path]) -> Usd.StagePopulationMask:
    """Return a new mask that is the union of given mask and paths.

    The resulting mask will include all paths from the input mask and the
    additional paths provided. Redundant paths are removed.

    Parameters
    ----------
    mask : Usd.StagePopulationMask
        The input mask to union with.
    paths : Sequence[Sdf.Path]
        Additional paths to include in the union.

    Returns
    -------
    Usd.StagePopulationMask
        A new mask that is the union of the input mask and paths.
    """
    result_mask = Usd.StagePopulationMask()
    for path in mask.GetPaths():
        result_mask = result_mask.Add(path)
    for path in paths:
        if not path.IsAbsolutePath():
            raise ValueError(f"Path {path} is not an absolute path.")
        result_mask = result_mask.Add(path)
    return result_mask


def remove_paths_from_mask(mask: Usd.StagePopulationMask, paths_to_remove: List[Sdf.Path]) -> Usd.StagePopulationMask:
    """
    Remove the specified paths from the given population mask.

    Args:
        mask (Usd.StagePopulationMask): The input population mask.
        paths_to_remove (List[Sdf.Path]): The list of paths to remove from the mask.

    Returns:
        Usd.StagePopulationMask: A new population mask with the specified paths removed.
    """
    result_mask = Usd.StagePopulationMask()
    for path in mask.GetPaths():
        if path not in paths_to_remove:
            result_mask = result_mask.Add(path)
    return result_mask


def create_union_mask(mask1: Usd.StagePopulationMask, mask2: Usd.StagePopulationMask) -> Usd.StagePopulationMask:
    """Create a union mask from two given masks."""
    if mask1.IsEmpty():
        return mask2
    if mask2.IsEmpty():
        return mask1
    if Sdf.Path.absoluteRootPath in mask1.GetPaths():
        return mask1
    if Sdf.Path.absoluteRootPath in mask2.GetPaths():
        return mask2
    union_mask = Usd.StagePopulationMask()
    for path in mask1.GetPaths():
        union_mask = union_mask.GetUnion(path)
    for path in mask2.GetPaths():
        union_mask = union_mask.GetUnion(path)
    return union_mask


def convert_time_samples_to_keyframes(time_samples: Dict[float, float]) -> List[Usd.TimeCode]:
    """Convert a dictionary of time samples to a list of Usd.TimeCode keyframes.

    Args:
        time_samples (Dict[float, float]): A dictionary of time samples, where the keys are time
            values and the values are the corresponding attribute values.

    Returns:
        List[Usd.TimeCode]: A list of Usd.TimeCode objects representing the keyframes.
    """
    keyframes = []
    for time, value in time_samples.items():
        keyframe = Usd.TimeCode(time)
        if keyframe.IsNumeric():
            keyframes.append(keyframe)
        else:
            continue
    keyframes.sort()
    return keyframes


def find_prims_with_animation(stage: Usd.Stage) -> List[Usd.Prim]:
    """Find all prims in the stage that have animated properties.

    Args:
        stage (Usd.Stage): The USD stage to search for animated prims.

    Returns:
        List[Usd.Prim]: A list of prims that have animated properties.
    """
    animated_prims: List[Usd.Prim] = []
    for prim in stage.Traverse():
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            for xform_op in xformable.GetOrderedXformOps():
                if xform_op.GetNumTimeSamples() > 1:
                    animated_prims.append(prim)
                    break
        for attribute in prim.GetAttributes():
            if attribute.GetNumTimeSamples() > 1:
                animated_prims.append(prim)
                break
    return animated_prims


def blend_animations(
    stage: Usd.Stage, prim_path: str, anim_attr_names: List[str], time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> Sdf.LayerOffset:
    """Blends multiple animation attributes on a prim additively.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim with the animation attributes.
        anim_attr_names (List[str]): The names of the animation attributes to blend.
        time_code (Usd.TimeCode): The time code at which to evaluate the animations.
            Defaults to Usd.TimeCode.Default().

    Returns:
        Sdf.LayerOffset: The combined layer offset from blending the animations.

    Raises:
        ValueError: If the specified prim or any of the animation attributes don't exist.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim '{prim_path}' does not exist")
    combined_offset = Sdf.LayerOffset()
    for attr_name in anim_attr_names:
        attr = prim.GetAttribute(attr_name)
        if not attr:
            raise ValueError(f"Attribute '{attr_name}' does not exist on prim '{prim_path}'")
        layer_offset = attr.Get(time_code)
        if layer_offset:
            combined_offset = combined_offset * layer_offset
    return combined_offset


def duplicate_animation(source_prim: Usd.Prim, dest_prim: Usd.Prim, time_codes: List[Usd.TimeCode]) -> None:
    """
    Duplicate the animation from the source prim to the destination prim for the given time codes.

    Args:
        source_prim (Usd.Prim): The source prim to copy the animation from.
        dest_prim (Usd.Prim): The destination prim to copy the animation to.
        time_codes (List[Usd.TimeCode]): The list of time codes to copy the animation for.

    Raises:
        ValueError: If the source or destination prim is not valid or not transformable.
    """
    if not source_prim.IsValid() or not source_prim.IsA(UsdGeom.Xformable):
        raise ValueError("Source prim is not valid or not transformable.")
    if not dest_prim.IsValid() or not dest_prim.IsA(UsdGeom.Xformable):
        raise ValueError("Destination prim is not valid or not transformable.")
    source_xform = UsdGeom.Xformable(source_prim)
    dest_xform = UsdGeom.Xformable(dest_prim)
    for time_code in time_codes:
        source_matrix = source_xform.ComputeLocalToWorldTransform(time_code)
        dest_xform.MakeMatrixXform().Set(source_matrix, time_code)


def merge_time_samples(time_samples: Dict[Usd.TimeCode, float]) -> List[Usd.TimeCode]:
    """Merges close time samples together to reduce the number of time samples.

    Args:
        time_samples (Dict[Usd.TimeCode, float]): A dictionary of time samples, where the keys are Usd.TimeCode
            objects and the values are the corresponding attribute values.

    Returns:
        List[Usd.TimeCode]: A list of merged time codes.
    """
    if not time_samples:
        return []
    sorted_time_samples = sorted(time_samples.items(), key=lambda x: x[0])
    merged_time_samples = [sorted_time_samples[0][0]]
    prev_time_code = sorted_time_samples[0][0]
    prev_value = sorted_time_samples[0][1]
    for time_code, value in sorted_time_samples[1:]:
        if time_code.GetValue() - prev_time_code.GetValue() < Usd.TimeCode.SafeStep() and value == prev_value:
            continue
        else:
            merged_time_samples.append(time_code)
            prev_time_code = time_code
            prev_value = value
    return merged_time_samples


def resample_animation(
    stage: Usd.Stage, prim_path: str, times: List[Usd.TimeCode], attribute_name: str = "xformOp:translate"
) -> None:
    """Resample the animation of a prim's attribute at specified times.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        times (List[Usd.TimeCode]): The list of time codes to resample at.
        attribute_name (str, optional): The name of the attribute to resample. Defaults to "xformOp:translate".

    Raises:
        ValueError: If the prim or attribute does not exist.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim '{prim_path}' does not exist.")
    attribute = prim.GetAttribute(attribute_name)
    if not attribute:
        raise ValueError(f"Attribute '{attribute_name}' does not exist on prim '{prim_path}'.")
    for time in times:
        value = attribute.Get(time)
        if value is not None:
            attribute.Set(value, time)


def analyze_token_usage(stage: Usd.Stage, token_to_check: str) -> Tuple[int, List[Sdf.Path]]:
    """Analyze the usage of a specific token in a USD stage.

    Args:
        stage (Usd.Stage): The USD stage to analyze.
        token_to_check (str): The token to check for usage.

    Returns:
        Tuple[int, List[Sdf.Path]]: A tuple containing the count of prims using the token
                                    and a list of paths to those prims.
    """
    count = 0
    prim_paths = []
    for prim in stage.Traverse():
        if prim.HasAuthoredMetadata(token_to_check):
            count += 1
            prim_paths.append(prim.GetPath())
    return (count, prim_paths)


def filter_prims_by_token(stage: Usd.Stage, token: str) -> List[Usd.Prim]:
    """
    Filter prims in a stage by a specific token.

    Args:
        stage (Usd.Stage): The USD stage to filter prims from.
        token (str): The token to filter prims by.

    Returns:
        List[Usd.Prim]: A list of prims that have the specified token.
    """
    matching_prims: List[Usd.Prim] = []
    for prim in stage.Traverse():
        if prim.HasAuthoredMetadata(token):
            matching_prims.append(prim)
    return matching_prims


def apply_token_to_prims(stage: Usd.Stage, token: str, prim_paths: List[str]) -> None:
    """Apply a token to a list of prims.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        token (str): The token to apply.
        prim_paths (List[str]): The paths of the prims to apply the token to.

    Raises:
        ValueError: If any of the prim paths are invalid.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")
        prim.SetMetadata("kind", token)


def replace_token_in_prim_names(stage: Usd.Stage, old_token: str, new_token: str) -> List[Usd.Prim]:
    """Replace a token in prim names on a USD stage.

    Args:
        stage (Usd.Stage): The USD stage to operate on.
        old_token (str): The token to replace in prim names.
        new_token (str): The token to replace the old token with.

    Returns:
        List[Usd.Prim]: A list of prims whose names were modified.
    """
    modified_prims: List[Usd.Prim] = []
    prims_to_modify: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        prim_name = prim.GetName()
        if old_token in prim_name:
            prims_to_modify.append(prim)
    for prim in prims_to_modify:
        prim_name = prim.GetName()
        new_prim_name = prim_name.replace(old_token, new_token)
        new_prim_path = prim.GetPath().GetParentPath().AppendChild(new_prim_name)
        new_prim = stage.DefinePrim(new_prim_path)
        for prop in prim.GetProperties():
            new_prim.CreateProperty(prop.GetName(), prop.GetTypeName()).Set(prop.Get())
        new_prim.SetMetadata("documentation", prim.GetMetadata("documentation"))
        stage.RemovePrim(prim.GetPath())
        modified_prims.append(new_prim)
    return modified_prims


def analyze_and_modify_typed_prims(stage: Usd.Stage):
    """Analyzes all typed prims on the stage and modifies their attributes.

    For each typed prim found, it creates a new attribute "is_typed" and sets
    its value to True. If the prim has a bounding box, it also computes the
    bounding box and sets the "bbox" attribute with the value.

    Parameters:
        stage (Usd.Stage): The USD stage to analyze and modify.
    """
    for prim in stage.TraverseAll():
        if prim.IsA(Usd.Typed):
            prim.CreateAttribute("is_typed", Sdf.ValueTypeNames.Bool).Set(True)
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
            bbox = bbox_cache.ComputeWorldBound(prim)
            if bbox.GetRange().IsEmpty():
                continue
            bbox_min = bbox.GetRange().GetMin()
            bbox_max = bbox.GetRange().GetMax()
            prim.CreateAttribute("bbox", Sdf.ValueTypeNames.Float3Array).Set([bbox_min, bbox_max])


def get_collection_expansion_rules(collection_query: Usd.UsdCollectionMembershipQuery) -> Tuple[set, set]:
    """
    Get the included and excluded paths along with their expansion rules from a collection membership query.

    Args:
        collection_query (Usd.UsdCollectionMembershipQuery): The collection membership query object.

    Returns:
        Tuple[set, set]: A tuple containing two sets:
            - The first set contains tuples of (path, expansion_rule) for included paths.
            - The second set contains the excluded paths.
    """
    path_expansion_rule_map = collection_query.GetAsPathExpansionRuleMap()
    included_paths = set()
    excluded_paths = set()
    for path, expansion_rule in path_expansion_rule_map.items():
        if expansion_rule == Usd.Tokens.exclude:
            excluded_paths.add(path)
        else:
            included_paths.add((path, expansion_rule))
    return (included_paths, excluded_paths)


def convert_layer_format(src_layer: Sdf.Layer, format: str) -> Sdf.Layer:
    """Converts a given layer to a new format.

    Args:
        src_layer (Sdf.Layer): The source layer to convert.
        format (str): The format to convert to. Must be a supported format id.

    Returns:
        Sdf.Layer: The converted layer in the new format.

    Raises:
        ValueError: If the source layer is invalid or the format is not supported.
    """
    if not src_layer:
        raise ValueError("Invalid source layer")
    if format not in [Sdf.FileFormat.FindById("usda"), Sdf.FileFormat.FindById("usdc")]:
        raise ValueError(f"Unsupported format: {format}")
    src_format = Usd.UsdFileFormat.GetUnderlyingFormatForLayer(src_layer)
    if src_format == format:
        return src_layer
    dst_layer = Sdf.Layer.CreateAnonymous(".usda" if format == Sdf.FileFormat.FindById("usda") else ".usdc")
    dst_layer.TransferContent(src_layer)
    return dst_layer


def list_all_variants(variantSet: Usd.VariantSet) -> List[str]:
    """Return a list of all variant names in the given VariantSet.

    Args:
        variantSet (Usd.VariantSet): The VariantSet to list variants for.

    Returns:
        List[str]: A list of variant names in the VariantSet.

    Raises:
        ValueError: If the provided VariantSet is not valid.
    """
    if not variantSet.IsValid():
        raise ValueError("Invalid VariantSet provided.")
    variant_names = variantSet.GetVariantNames()
    return variant_names


def remove_variant(variant_set: Usd.VariantSet, variant_name: str) -> bool:
    """Remove a variant from a VariantSet.

    Args:
        variant_set (Usd.VariantSet): The VariantSet to remove the variant from.
        variant_name (str): The name of the variant to remove.

    Returns:
        bool: True if the variant was successfully removed, False otherwise.
    """
    if not variant_set.IsValid():
        raise ValueError("Invalid VariantSet.")
    if not variant_set.HasAuthoredVariant(variant_name):
        print(f"Variant '{variant_name}' does not exist in the VariantSet.")
        return False
    prim = variant_set.GetPrim()
    variant_path = prim.GetPath().AppendVariantSelection(variant_set.GetName(), variant_name)
    stage = prim.GetStage()
    stage.RemovePrim(variant_path)
    return True


def check_variant_existence(variant_set: Usd.VariantSet, variant_name: str) -> bool:
    """Check if a variant exists in a VariantSet.

    Args:
        variant_set (Usd.VariantSet): The VariantSet to check.
        variant_name (str): The name of the variant to check for existence.

    Returns:
        bool: True if the variant exists in the VariantSet, False otherwise.
    """
    if not variant_set.IsValid():
        raise ValueError("Invalid VariantSet.")
    variant_names = variant_set.GetVariantNames()
    if variant_name in variant_names:
        return True
    else:
        return False


def set_default_variant_selection(variant_set: Usd.VariantSet, variant_name: str) -> bool:
    """Set the default variant selection for a VariantSet.

    Args:
        variant_set (Usd.VariantSet): The VariantSet to set the default selection for.
        variant_name (str): The name of the variant to select as default.

    Returns:
        bool: True if the default selection was set successfully, False otherwise.
    """
    if not variant_set.IsValid():
        raise ValueError("Invalid VariantSet")
    if not variant_set.HasAuthoredVariant(variant_name):
        raise ValueError(f"Variant '{variant_name}' does not exist in the VariantSet")
    prim = variant_set.GetPrim()
    try:
        variant_set.SetVariantSelection(variant_name)
    except Tf.ErrorException as e:
        print(f"Error setting variant selection: {e}")
        return False
    stage = prim.GetStage()
    try:
        stage.SetDefaultPrim(prim)
    except Tf.ErrorException as e:
        print(f"Error setting default prim: {e}")
        return False
    return True


def clear_all_variant_selections(prim: Usd.Prim) -> bool:
    """Clear all variant selections on the given prim.

    Args:
        prim (Usd.Prim): The prim to clear variant selections on.

    Returns:
        bool: True if all variant selections were cleared successfully, False otherwise.
    """
    variant_set_names = prim.GetVariantSets().GetNames()
    all_cleared = True
    for variant_set_name in variant_set_names:
        variant_set = prim.GetVariantSet(variant_set_name)
        if variant_set.HasAuthoredVariantSelection():
            if not variant_set.ClearVariantSelection():
                all_cleared = False
    return all_cleared


def synchronize_variant_selections(source_prim: Usd.Prim, target_prim: Usd.Prim) -> None:
    """Synchronize variant selections from source_prim to target_prim."""
    source_variant_set_names = source_prim.GetVariantSets().GetNames()
    for variant_set_name in source_variant_set_names:
        source_variant_set = source_prim.GetVariantSet(variant_set_name)
        source_selection = source_variant_set.GetVariantSelection()
        if source_selection:
            target_variant_set = target_prim.GetVariantSets().AddVariantSet(variant_set_name)
            if not target_variant_set.HasAuthoredVariant(source_selection):
                target_variant_set.AddVariant(source_selection)
            target_variant_set.SetVariantSelection(source_selection)


def create_and_select_variant(variant_set: Usd.VariantSet, variant_name: str) -> None:
    """Create a new variant in the given variant set and select it.

    Args:
        variant_set (Usd.VariantSet): The variant set to create the variant in.
        variant_name (str): The name of the variant to create and select.

    Raises:
        ValueError: If the variant set is invalid or the variant name is empty.
    """
    if not variant_set.IsValid():
        raise ValueError("Invalid variant set")
    if not variant_name:
        raise ValueError("Variant name cannot be empty")
    added = variant_set.AddVariant(variant_name)
    if not added:
        raise RuntimeError(f"Failed to add variant '{variant_name}' to the variant set")
    selected = variant_set.SetVariantSelection(variant_name)
    if not selected:
        raise RuntimeError(f"Failed to select variant '{variant_name}' in the variant set")


def get_variant_edit_context(
    variantSet: Usd.VariantSet, layer: Optional[Sdf.Layer] = None
) -> Tuple[Usd.Stage, Usd.EditTarget]:
    """
    Helper function for configuring a UsdStage's EditTarget to author into the currently selected variant.

    Returns configuration for a UsdEditContext.

    To begin editing into VariantSet varSet's currently selected variant:

    with varSet.get_variant_edit_context():
        # Now sending mutations to current variant

    If there is no currently selected variant in this VariantSet, return (None, None).

    If layer is unspecified, then we will use the layer of our prim's stage's current UsdEditTarget.

    Currently, we require layer to be in the stage's local LayerStack (see UsdStage.HasLocalLayer()),
    and will raise a ValueError if layer is not. We may relax this restriction in the future, if need arises,
    but it introduces several complications in specification and behavior.

    Args:
        variantSet (Usd.VariantSet): The VariantSet to get the edit context for.
        layer (Optional[Sdf.Layer]): The layer to use for editing. If None, uses the current edit target's layer.

    Returns:
        Tuple[Usd.Stage, Usd.EditTarget]: The stage and edit target for editing the selected variant.

    Raises:
        ValueError: If the specified layer is not in the stage's local LayerStack.
    """
    selected_variant = variantSet.GetVariantSelection()
    if not selected_variant:
        return (None, None)
    if layer is None:
        layer = variantSet.GetPrim().GetStage().GetEditTarget().GetLayer()
    stage = variantSet.GetPrim().GetStage()
    if not stage.HasLocalLayer(layer):
        raise ValueError("The specified layer is not in the stage's local LayerStack.")
    edit_target = variantSet.GetVariantEditTarget(layer)
    return (stage, edit_target)


def copy_variant_to_another_set(
    src_varset: Usd.VariantSet, src_variant: str, dst_varset: Usd.VariantSet, dst_variant: str
) -> bool:
    """Copy a variant from one VariantSet to another.

    Args:
        src_varset (Usd.VariantSet): The source VariantSet containing the variant to copy.
        src_variant (str): The name of the variant to copy from the source VariantSet.
        dst_varset (Usd.VariantSet): The destination VariantSet to copy the variant to.
        dst_variant (str): The name of the variant to create in the destination VariantSet.

    Returns:
        bool: True if the variant was successfully copied, False otherwise.
    """
    if not src_varset.IsValid() or not src_varset.HasAuthoredVariant(src_variant):
        print(f"Error: Invalid source VariantSet or variant '{src_variant}'")
        return False
    if not dst_varset.IsValid():
        print("Error: Invalid destination VariantSet")
        return False
    if dst_varset.HasAuthoredVariant(dst_variant):
        print(f"Error: Destination variant '{dst_variant}' already exists")
        return False
    if not dst_varset.AddVariant(dst_variant, Usd.ListPositionBackOfAppendList):
        print(f"Error: Failed to add variant '{dst_variant}' to the destination VariantSet")
        return False
    src_edit_target = src_varset.GetVariantEditTarget()
    with Usd.EditContext(src_varset.GetPrim().GetStage(), src_edit_target):
        dst_edit_target = dst_varset.GetVariantEditTarget()
        with Usd.EditContext(dst_varset.GetPrim().GetStage(), dst_edit_target):
            src_prim_spec = src_varset.GetPrim().GetPrimStack()[0]
            dst_prim_spec = dst_varset.GetPrim().GetPrimStack()[0]
            for prop_spec in src_prim_spec.properties:
                dst_prim_spec.properties.append(prop_spec)
    return True


def get_variant_prim(variantSet: Usd.VariantSet) -> Usd.Prim:
    """Get the prim associated with the currently selected variant of a VariantSet.

    Args:
        variantSet (Usd.VariantSet): The VariantSet to get the variant prim for.

    Returns:
        Usd.Prim: The prim associated with the currently selected variant.

    Raises:
        ValueError: If the VariantSet is not valid or has no variant selection.
    """
    if not variantSet.IsValid():
        raise ValueError("Invalid VariantSet")
    selected_variant = variantSet.GetVariantSelection()
    if not selected_variant:
        raise ValueError("No variant selected in the VariantSet")
    variantSet_prim = variantSet.GetPrim()
    variant_prim_path = variantSet_prim.GetPath().AppendVariantSelection(variantSet.GetName(), selected_variant)
    stage = variantSet_prim.GetStage()
    variant_prim = stage.GetPrimAtPath(variant_prim_path)
    if not variant_prim.IsValid():
        return variantSet_prim
    return variant_prim


def merge_variants(varset: Usd.VariantSet, source_variant: str, target_variant: str) -> bool:
    """Merge the contents of source_variant into target_variant for the given varset.

    Args:
        varset (Usd.VariantSet): The variant set to operate on.
        source_variant (str): The name of the source variant to merge from.
        target_variant (str): The name of the target variant to merge into.

    Returns:
        bool: True if the merge was successful, False otherwise.
    """
    if not varset.IsValid():
        raise ValueError("Invalid VariantSet provided.")
    if not varset.HasAuthoredVariant(source_variant):
        raise ValueError(f"Source variant '{source_variant}' does not exist.")
    if not varset.HasAuthoredVariant(target_variant):
        raise ValueError(f"Target variant '{target_variant}' does not exist.")
    stage = varset.GetPrim().GetStage()
    with varset.GetVariantEditContext(stage.GetEditTarget().GetLayer()):
        varset.SetVariantSelection(source_variant)
        source_prim = stage.GetPrimAtPath(varset.GetPrim().GetPath())
        varset.SetVariantSelection(target_variant)
        target_prim = stage.GetPrimAtPath(varset.GetPrim().GetPath())
        for prop in source_prim.GetProperties():
            prop_name = prop.GetName()
            prop_type_name = prop.GetTypeName()
            prop_value = prop.Get()
            if not target_prim.HasProperty(prop_name):
                target_prim.CreateAttribute(prop_name, prop_type_name)
            target_prop = target_prim.GetProperty(prop_name)
            target_prop.Set(prop_value)
    return True


def set_multiple_variant_selections(variant_sets: Usd.VariantSets, selections: Dict[str, str]) -> None:
    """
    Set multiple variant selections on a prim's VariantSets.

    Args:
        variant_sets (Usd.VariantSets): The VariantSets object of the prim.
        selections (Dict[str, str]): A dictionary mapping variant set names to the desired selections.

    Raises:
        ValueError: If any of the specified variant sets or variants do not exist.
    """
    for variant_set_name, variant_name in selections.items():
        if not variant_sets.HasVariantSet(variant_set_name):
            raise ValueError(f"Variant set '{variant_set_name}' does not exist on the prim.")
        variant_set = variant_sets.GetVariantSet(variant_set_name)
        if not variant_set.HasAuthoredVariant(variant_name):
            raise ValueError(f"Variant '{variant_name}' does not exist in variant set '{variant_set_name}'.")
        variant_sets.SetSelection(variant_set_name, variant_name)


def clear_variant_selection(variantSets: Usd.VariantSets, variantSetName: str) -> None:
    """Clear the variant selection for the specified variant set.

    If the variant set does not exist, this function does nothing.

    Args:
        variantSets (Usd.VariantSets): The VariantSets object.
        variantSetName (str): The name of the variant set.
    """
    if variantSets.HasVariantSet(variantSetName):
        variantSet = variantSets.GetVariantSet(variantSetName)
        variantSet.SetVariantSelection("")


def list_all_variant_sets(variant_sets: Usd.VariantSets) -> List[str]:
    """Return a list of all variant set names on the originating prim."""
    variant_set_names = variant_sets.GetNames()
    if not variant_set_names:
        return []
    result = []
    for name in variant_set_names:
        if variant_sets.HasVariantSet(name):
            result.append(name)
    return result


def add_variant_and_select(variantSets: Usd.VariantSets, variantSetName: str, variantName: str) -> bool:
    """Add a variant to a VariantSet and select it."""
    if not variantSets.HasVariantSet(variantSetName):
        variantSet = variantSets.AddVariantSet(variantSetName)
    else:
        variantSet = variantSets.GetVariantSet(variantSetName)
    if variantSet.HasAuthoredVariant(variantName):
        pass
    else:
        variantSet.AddVariant(variantName)
    return variantSets.SetSelection(variantSetName, variantName)


def get_all_variant_selections(prim: Usd.Prim) -> dict:
    """
    Returns the composed map of all variant selections authored on the
    given UsdPrim, regardless of whether a corresponding variant set
    exists.

    Args:
        prim (Usd.Prim): The prim to query for variant selections.

    Returns:
        dict: The composed map of all variant selections.
    """
    if not prim.IsValid():
        return {}
    variant_sets = prim.GetVariantSets()
    selection_map = variant_sets.GetAllVariantSelections()
    selection_dict = dict(selection_map)
    return selection_dict


def validate_usdz_integrity(usdz_file_path: str) -> bool:
    """
    Validate the integrity of a .usdz file.

    Args:
        usdz_file_path (str): The file path to the .usdz file.

    Returns:
        bool: True if the .usdz file is valid, False otherwise.
    """
    try:
        zip_file = Usd.ZipFile.Open(usdz_file_path)
        if not zip_file:
            return False
        file_names = zip_file.GetFileNames()
        if not file_names:
            return False
        if "model.usda" not in file_names:
            return False
        for file_name in file_names:
            file_info = zip_file.GetFileInfo(file_name)
            if not file_info:
                return False
            if file_info.dataSize != file_info.uncompressedSize:
                return False
        return True
    except:
        return False


def copy_usdz_file_to_path(src_file: str, dst_dir: str) -> str:
    """
    Copy a .usdz file to a destination directory and return the new file path.

    Args:
        src_file (str): The source .usdz file path.
        dst_dir (str): The destination directory path.

    Returns:
        str: The new file path in the destination directory.

    Raises:
        ValueError: If the source file is not a .usdz file.
        FileNotFoundError: If the source file does not exist.
        OSError: If the destination directory is not writable.
    """
    if not src_file.lower().endswith(".usdz"):
        raise ValueError(f"Source file {src_file} is not a .usdz file.")
    if not os.path.isfile(src_file):
        raise FileNotFoundError(f"Source file {src_file} does not exist.")
    base_name = os.path.basename(src_file)
    dst_file = os.path.join(dst_dir, base_name)
    shutil.copy2(src_file, dst_file)
    return dst_file


class MockFileInfo:

    def __init__(self, uncompressed_size: int):
        self.uncompressedSize = uncompressed_size


def aggregate_zip_file_sizes(file_infos: List[MockFileInfo]) -> int:
    """Aggregates the uncompressed sizes of the given MockFileInfo objects.

    Args:
        file_infos (List[MockFileInfo]): A list of MockFileInfo objects.

    Returns:
        int: The total uncompressed size of all the files.
    """
    total_size = 0
    for file_info in file_infos:
        if file_info is None:
            continue
        uncompressed_size = file_info.uncompressedSize
        if uncompressed_size is None or uncompressed_size < 0:
            continue
        total_size += uncompressed_size
    return total_size


class MockFileInfo:

    def __init__(self, compressionMethod: int, dataOffset: int, encrypted: bool, size: int, uncompressedSize: int):
        self._compressionMethod = compressionMethod
        self._dataOffset = dataOffset
        self._encrypted = encrypted
        self._size = size
        self._uncompressedSize = uncompressedSize

    @property
    def compressionMethod(self):
        return self._compressionMethod

    @property
    def dataOffset(self):
        return self._dataOffset

    @property
    def encrypted(self):
        return self._encrypted

    @property
    def size(self):
        return self._size

    @property
    def uncompressedSize(self):
        return self._uncompressedSize


def filter_files_by_compression_method(files: List[MockFileInfo], method: int) -> List[MockFileInfo]:
    """Filter a list of FileInfo objects by the specified compression method.

    Args:
        files (List[MockFileInfo]): List of FileInfo objects to filter.
        method (int): The compression method to filter by.

    Returns:
        List[MockFileInfo]: Filtered list of FileInfo objects.
    """
    if not isinstance(files, list):
        raise TypeError("files must be a list of FileInfo objects")
    filtered_files = [file for file in files if file.compressionMethod == method]
    return filtered_files


class MockFileInfo:

    def __init__(self, size, uncompressed_size):
        self.size = size
        self.uncompressedSize = uncompressed_size


def compare_zip_file_sizes(zip_file_info_list: List[MockFileInfo]) -> List[int]:
    """
    Compare the compressed and uncompressed sizes of a list of zip file infos.

    Args:
        zip_file_info_list: A list of MockFileInfo objects.

    Returns:
        A list of integers, where each integer represents the comparison result:
        - 1 if compressed size is smaller than uncompressed size
        - 0 if compressed size is equal to uncompressed size
        - -1 if compressed size is larger than uncompressed size
    """
    comparison_results = []
    for file_info in zip_file_info_list:
        compressed_size = file_info.size
        uncompressed_size = file_info.uncompressedSize
        if compressed_size < uncompressed_size:
            comparison_results.append(1)
        elif compressed_size == uncompressed_size:
            comparison_results.append(0)
        else:
            comparison_results.append(-1)
    return comparison_results


def create_usdz_with_referenced_usd(usdz_file_path: str, usd_file_path: str) -> bool:
    """
    Create a .usdz file that references an external .usd file.

    Args:
        usdz_file_path (str): The file path where the .usdz file will be created.
        usd_file_path (str): The file path to the external .usd file to reference.

    Returns:
        bool: True if the .usdz file was created successfully, False otherwise.
    """
    zip_writer = Usd.ZipFileWriter.CreateNew(usdz_file_path)
    if not zip_writer:
        return False
    try:
        stage = Usd.Stage.CreateInMemory()
        root_layer = stage.GetRootLayer()
        root_layer.GetExternalReferences().AddReference(usd_file_path)
        usd_data = root_layer.ExportToString()
        zip_writer.AddFile(usd_data, "data.usd")
        if not zip_writer.Save():
            return False
    except Exception as e:
        zip_writer.Discard()
        return False
    return True


def extract_usdz_contents(usdz_file_path: str, output_dir: str) -> bool:
    """
    Extracts the contents of a .usdz file to a specified output directory.

    Args:
        usdz_file_path (str): The path to the .usdz file to extract.
        output_dir (str): The directory where the extracted contents will be saved.

    Returns:
        bool: True if the extraction was successful, False otherwise.
    """
    if not os.path.exists(usdz_file_path):
        print(f"Error: File '{usdz_file_path}' does not exist.")
        return False
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        with zipfile.ZipFile(usdz_file_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
    except zipfile.BadZipFile:
        print(f"Error: File '{usdz_file_path}' is not a valid zip archive.")
        return False
    except Exception as e:
        print(f"Error: An unexpected error occurred during extraction: {str(e)}")
        return False
    return True
