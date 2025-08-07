## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from typing import Any, Dict, List, Optional, Tuple, Union

from pxr import Gf, Ndr, Sdf, Sdr, Usd, UsdShade


def optimize_shader_connections(shaderNode: Sdr.ShaderNode, nodeContext: Sdr.NodeContext) -> bool:
    """Optimize the shader connections for a given shader node.

    Args:
        shaderNode (Sdr.ShaderNode): The shader node to optimize connections for.
        nodeContext (Sdr.NodeContext): The node context for the shader node.

    Returns:
        bool: True if optimization was successful, False otherwise.
    """
    if not shaderNode:
        return False
    shaderId = shaderNode.GetIdentifier()
    if not shaderId:
        return False
    inputNames = shaderNode.GetInputNames()
    outputNames = shaderNode.GetOutputNames()
    for inputName in inputNames:
        connectedSource = nodeContext.GetConnectedSource(shaderId, inputName)
        if not connectedSource:
            continue
        outputName = connectedSource.outputName
        if not outputName:
            continue
    for outputName in outputNames:
        connectedTargets = nodeContext.GetConnectedTargets(shaderId, outputName)
        if not connectedTargets:
            continue
    return True


def validate_shader_network(shader_node_prim: Usd.Prim) -> bool:
    """Validate that the specified shader node prim is a valid shader network.

    Args:
        shader_node_prim (Usd.Prim): The shader node prim to validate.

    Returns:
        bool: True if the shader node prim is a valid shader network, False otherwise.
    """
    sdr_registry = Sdr.Registry()
    if not shader_node_prim.IsValid():
        return False
    shader_type = shader_node_prim.GetAttribute("info:id").Get()
    if not sdr_registry.GetNodeByIdentifier(shader_type):
        return False
    if not shader_node_prim.GetAttribute("info:implementationSource").IsValid():
        return False
    if not shader_node_prim.GetAttribute("info:context").IsValid():
        return False
    if not shader_node_prim.GetAttribute("info:type").IsValid():
        return False
    inputs = shader_node_prim.GetAttributes()
    for input_attr in inputs:
        if input_attr.GetNamespace() == "inputs" and (not input_attr.IsValid()):
            return False
    outputs = shader_node_prim.GetAttributes()
    for output_attr in outputs:
        if output_attr.GetNamespace() == "outputs" and (not output_attr.IsValid()):
            return False
    return True


def gather_node_metadata(
    node_paths: List[str], metadata_fields: List[str], options: Dict[str, str] = None
) -> Dict[str, Dict[str, Any]]:
    """Gather metadata for multiple nodes.

    Args:
        node_paths (List[str]): The paths to the nodes to gather metadata for.
        metadata_fields (List[str]): The metadata fields to retrieve for each node.
        options (Dict[str, str], optional): Additional options for the metadata query.
            Defaults to None.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary mapping node paths to dictionaries of
            metadata key-value pairs.
    """
    all_metadata = {}
    for node_path in node_paths:
        node_metadata = {}
        node = Sdr.Registry().GetNodeByIdentifier(node_path)
        if not node:
            print(f"Warning: Node at path {node_path} does not exist. Skipping.")
            continue
        for field in metadata_fields:
            if field == "primvars":
                primvar_metadata = node.GetMetadata().get("primvars", {})
                node_metadata[field] = primvar_metadata
            else:
                value = node.GetMetadata().get(field)
                if value is not None:
                    node_metadata[field] = value
        all_metadata[node_path] = node_metadata
    return all_metadata


def filter_prims_by_metadata(prims: List[Usd.Prim], metadata: Dict[str, Any]) -> List[Usd.Prim]:
    """Filter a list of prims by the given metadata.

    Args:
        prims (List[Usd.Prim]): List of prims to filter.
        metadata (Dict[str, Any]): Metadata to filter by.

    Returns:
        List[Usd.Prim]: List of prims that match the given metadata.
    """
    filtered_prims: List[Usd.Prim] = []
    for prim in prims:
        match = True
        for key, value in metadata.items():
            if not prim.HasCustomDataKey(key) or prim.GetCustomDataByKey(key) != value:
                match = False
                break
        if match:
            filtered_prims.append(prim)
    return filtered_prims


def validate_node_roles_in_stage(stage: Usd.Stage) -> List[Sdf.Path]:
    """Validate that all shaders in the stage have valid node roles.

    Args:
        stage (Usd.Stage): The USD stage to validate.

    Returns:
        List[Sdf.Path]: A list of paths to shaders with invalid node roles.
    """
    invalid_shaders: List[Sdf.Path] = []
    for prim in stage.Traverse():
        shader = UsdShade.Shader(prim)
        if shader:
            shader_attr = shader.GetIdAttr()
            if shader_attr:
                shader_id = shader_attr.Get()
                registry = Sdr.Registry()
                shader_node = registry.GetNodeByIdentifier(shader_id)
                if shader_node:
                    role = shader_node.GetRole()
                    if role == "none":
                        invalid_shaders.append(prim.GetPath())
                else:
                    invalid_shaders.append(prim.GetPath())
    return invalid_shaders


def create_shader_stage() -> Usd.Stage:
    """Create a stage with valid and invalid shader roles."""
    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/Material")
    valid_shader = UsdShade.Shader.Define(stage, "/Material/ValidShader")
    valid_shader.CreateIdAttr("UsdPreviewSurface")
    valid_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((1, 0, 0))
    material.CreateSurfaceOutput().ConnectToSource(valid_shader.ConnectableAPI(), "surface")
    invalid_shader = UsdShade.Shader.Define(stage, "/Material/InvalidShader")
    invalid_shader.CreateIdAttr("NonexistentShader")
    return stage


def assign_node_role_to_material(stage: Usd.Stage, material_path: str, node_role: str) -> None:
    """Assign a node role to a material.

    Args:
        stage (Usd.Stage): The USD stage.
        material_path (str): The path to the material prim.
        node_role (str): The node role to assign.

    Raises:
        ValueError: If the material prim is not valid or not a UsdShade.Material.
    """
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise ValueError(f"Invalid material prim path: {material_path}")
    if not UsdShade.Material(material_prim):
        raise ValueError(f"Prim at path {material_path} is not a UsdShade.Material")
    shade_api = UsdShade.Material(material_prim)
    shade_api.CreateSurfaceOutput().GetAttr().Set(node_role)


def list_all_metadata_for_property(prop_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return a dictionary containing all metadata for a given property.

    Args:
        prop_metadata (Dict[str, Any]): The PropertyMetadata dictionary to query.

    Returns:
        Dict[str, Any]: A dictionary where keys are metadata names and values are metadata values.
    """
    metadata_dict: Dict[str, Any] = {}
    possible_metadata_names = [
        "default",
        "type",
        "label",
        "page",
        "help",
        "widget",
        "options",
        "connectable",
        "implementationName",
        "isDynamicArray",
        "role",
        "vstructMemberOf",
        "vstructMemberName",
    ]
    for metadata_name in possible_metadata_names:
        if metadata_name in prop_metadata:
            metadata_value = prop_metadata[metadata_name]
            metadata_dict[metadata_name] = metadata_value
    return metadata_dict


def search_prims_by_property_role(stage: Usd.Stage, role: Sdr.PropertyRole) -> List[Usd.Prim]:
    """Search for prims on a stage that have a property with a specific role.

    Args:
        stage (Usd.Stage): The stage to search for prims.
        role (Sdr.PropertyRole): The property role to search for.

    Returns:
        List[Usd.Prim]: A list of prims that have a property with the specified role.
    """
    prims_with_role: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        properties = prim.GetProperties()
        for prop in properties:
            if prop.HasCustomDataKey(Sdr.PropertyMetadata.Role):
                prop_role = prop.GetCustomDataByKey(Sdr.PropertyMetadata.Role)
                if prop_role == role:
                    prims_with_role.append(prim)
                    break
    return prims_with_role


def set_prim_property_value(prim: Usd.Prim, property_name: str, property_type: Sdf.ValueTypeName, value: Any):
    """Set a property value on a prim.

    Args:
        prim (Usd.Prim): The prim to set the property on.
        property_name (str): The name of the property to set.
        property_type (Sdf.ValueTypeName): The type of the property.
        value (Any): The value to set the property to.

    Raises:
        ValueError: If the prim is not valid.
        TypeError: If the property type is not supported.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    if prim.HasProperty(property_name):
        prop = prim.GetProperty(property_name)
    else:
        prop = prim.CreateAttribute(property_name, property_type)
    if property_type == Sdf.ValueTypeNames.Bool:
        prop.Set(bool(value))
    elif property_type == Sdf.ValueTypeNames.Int:
        prop.Set(int(value))
    elif property_type == Sdf.ValueTypeNames.Float:
        prop.Set(float(value))
    elif property_type == Sdf.ValueTypeNames.Double:
        prop.Set(float(value))
    elif property_type == Sdf.ValueTypeNames.String:
        prop.Set(str(value))
    elif property_type == Sdf.ValueTypeNames.Float2:
        prop.Set(Gf.Vec2f(value))
    elif property_type == Sdf.ValueTypeNames.Float3:
        prop.Set(Gf.Vec3f(value))
    elif property_type == Sdf.ValueTypeNames.Float4:
        prop.Set(Gf.Vec4f(value))
    elif property_type == Sdf.ValueTypeNames.Color3f:
        prop.Set(Gf.Vec3f(value))
    elif property_type == Sdf.ValueTypeNames.Color4f:
        prop.Set(Gf.Vec4f(value))
    else:
        raise TypeError(f"Unsupported property type: {property_type}")


def convert_prim_property_type(sdf_value_type: Sdf.ValueTypeName) -> Union[str, None]:
    """Convert an Sdf.ValueTypeName to an Sdr.PropertyType string.

    Args:
        sdf_value_type (Sdf.ValueTypeName): The Sdf value type to convert.

    Returns:
        Union[str, None]: The corresponding Sdr.PropertyType string or None if no match.
    """
    if sdf_value_type == Sdf.ValueTypeNames.Asset:
        return "Asset"
    elif sdf_value_type == Sdf.ValueTypeNames.Bool:
        return "Boolean"
    elif sdf_value_type == Sdf.ValueTypeNames.Color3f:
        return "Color"
    elif sdf_value_type == Sdf.ValueTypeNames.Float:
        return "Float"
    elif sdf_value_type == Sdf.ValueTypeNames.Int:
        return "Integer"
    elif sdf_value_type == Sdf.ValueTypeNames.Matrix4d:
        return "Matrix"
    elif sdf_value_type == Sdf.ValueTypeNames.String:
        return "String"
    elif sdf_value_type == Sdf.ValueTypeNames.Vector3f:
        return "Vector"
    elif sdf_value_type == Sdf.ValueTypeNames.FloatArray:
        return "Float[]"
    elif sdf_value_type == Sdf.ValueTypeNames.IntArray:
        return "Integer[]"
    elif sdf_value_type == Sdf.ValueTypeNames.StringArray:
        return "String[]"
    else:
        return None


def validate_prim_properties(prim: Usd.Prim, schema: Sdr.ShaderNode) -> bool:
    """Validate that a prim has all the properties required by a shader schema.

    Args:
        prim (Usd.Prim): The prim to validate.
        schema (Sdr.ShaderNode): The shader schema to validate against.

    Returns:
        bool: True if the prim has all the required properties, False otherwise.
    """
    if schema is None:
        print("Invalid shader schema")
        return False
    shader_props = schema.GetInputNames()
    for prop_name in shader_props:
        prop = schema.GetInput(prop_name)
        prop_type = prop.GetType()
        if not prim.HasAttribute(prop_name):
            print(f"Prim is missing attribute: {prop_name}")
            return False
        attr = prim.GetAttribute(prop_name)
        if attr.GetTypeName().type == prop_type:
            print(f"Attribute {prop_name} has wrong type. Expected: {prop_type}, Got: {attr.GetTypeName().type}")
            return False
    return True


def collect_shaders_by_family(reg: Sdr.Registry) -> Dict[str, List[Sdr.ShaderNode]]:
    """Collect all shader nodes in the registry and group them by family."""
    ids = reg.GetNodeIdentifiers()
    result: Dict[str, List[Sdr.ShaderNode]] = {}
    for identifier in ids:
        node = reg.GetShaderNodeByIdentifier(identifier, [])
        if not node:
            continue
        family = node.GetFamily()
        if family not in result:
            result[family] = []
        result[family].append(node)
    return result


def find_and_replace_shader(stage: Usd.Stage, old_id: str, new_id: str) -> List[Tuple[Sdf.Path, str, str]]:
    """Find and replace shader nodes with a given identifier.

    Args:
        stage (Usd.Stage): The USD stage to search for shader nodes.
        old_id (str): The identifier of the shader nodes to replace.
        new_id (str): The identifier of the replacement shader nodes.

    Returns:
        List[Tuple[Sdf.Path, str, str]]: A list of tuples containing the prim path, old shader ID, and new shader ID for each replaced shader node.
    """
    results = []
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            shader_id = shader.GetShaderId()
            if shader_id == old_id:
                shader.SetShaderId(new_id)
                results.append((prim.GetPath(), old_id, new_id))
    return results


def batch_update_shader_parameters(
    stage: Usd.Stage, shader_prim_paths: List[str], parameter_name: str, parameter_value: Any
) -> None:
    """
    Update a specific parameter for multiple shader prims in a single operation.

    Args:
        stage (Usd.Stage): The USD stage containing the shader prims.
        shader_prim_paths (List[str]): A list of paths to the shader prims to update.
        parameter_name (str): The name of the parameter to update.
        parameter_value (Any): The new value for the parameter.

    Raises:
        ValueError: If any of the shader prims are invalid or don't have the specified parameter.
    """
    shader_prims = []
    for prim_path in shader_prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Shader prim at path {prim_path} is invalid.")
        if not UsdShade.Shader(prim):
            raise ValueError(f"Prim at path {prim_path} is not a shader.")
        shader_prims.append(prim)
    for shader_prim in shader_prims:
        shader = UsdShade.Shader(shader_prim)
        if not shader.GetInput(parameter_name):
            raise ValueError(f"Shader {shader_prim.GetPath()} does not have parameter {parameter_name}.")
    with Sdf.ChangeBlock():
        for shader_prim in shader_prims:
            shader = UsdShade.Shader(shader_prim)
            shader.GetInput(parameter_name).Set(parameter_value)


def merge_shader_parameters(shader_node: Sdr.ShaderNode, additional_params: Dict[str, Any]) -> Dict[str, Any]:
    """Merge additional parameters with the shader node's existing parameters.

    Args:
        shader_node (Sdr.ShaderNode): The shader node to retrieve parameters from.
        additional_params (Dict[str, Any]): Additional parameters to merge with the shader node's parameters.

    Returns:
        Dict[str, Any]: The merged parameter dictionary.
    """
    merged_params = {}
    if shader_node:
        for input_name in shader_node.GetInputNames():
            input_param = shader_node.GetInput(input_name)
            param_default = input_param.GetDefaultValue()
            merged_params[input_name] = param_default
    for param_name, param_value in additional_params.items():
        if param_name in merged_params:
            merged_params[param_name] = param_value
        else:
            merged_params[param_name] = param_value
    return merged_params


def get_shader_node_info(shader_node: Sdr.ShaderNode) -> Dict[str, Any]:
    """Get information about a shader node."""
    info = {}
    if not shader_node:
        raise ValueError("Invalid shader node")
    info["name"] = shader_node.GetName()
    info["label"] = shader_node.GetLabel()
    info["implementation_name"] = shader_node.GetImplementationName()
    info["category"] = shader_node.GetCategory()
    info["departments"] = shader_node.GetDepartments()
    info["help"] = shader_node.GetHelp()
    info["role"] = shader_node.GetRole()
    info["primvars"] = shader_node.GetPrimvars()
    info["additional_primvar_properties"] = shader_node.GetAdditionalPrimvarProperties()
    pages = shader_node.GetPages()
    info["pages"] = pages
    info["properties_by_page"] = {page: shader_node.GetPropertyNamesForPage(page) for page in pages}
    default_input = shader_node.GetDefaultInput()
    info["default_input"] = default_input.GetName() if default_input else None
    info["asset_identifier_input_names"] = shader_node.GetAssetIdentifierInputNames()
    info["vstructs"] = shader_node.GetAllVstructNames()
    return info


def get_shader_node_asset_identifiers(shader_node: Sdr.ShaderNode) -> List[str]:
    """Get the list of asset identifier input names for a shader node."""
    if not shader_node:
        raise ValueError("Invalid shader node.")
    asset_identifier_input_names = shader_node.GetAssetIdentifierInputNames()
    asset_identifier_names = [input_name.GetString() for input_name in asset_identifier_input_names]
    return asset_identifier_names


def get_shader_node_help(shader_node: Sdr.ShaderNode) -> str:
    """Get the help message assigned to the shader node, if any."""
    if not shader_node:
        raise ValueError("Invalid shader node.")
    help_message = shader_node.GetHelp()
    return help_message


def get_shader_node_role(shader_node: Sdr.ShaderNode) -> str:
    """Get the role of a shader node."""
    if not shader_node:
        raise ValueError("Invalid shader node.")
    role = shader_node.GetRole()
    return role


def get_shader_node_category_and_label(shader_node: Sdr.ShaderNode) -> Tuple[str, str]:
    """
    Get the category and label of a shader node.

    Args:
        shader_node (Sdr.ShaderNode): The shader node to get the category and label from.

    Returns:
        Tuple[str, str]: A tuple containing the category and label of the shader node.
    """
    category = shader_node.GetCategory()
    label = shader_node.GetLabel()
    if not label:
        label = shader_node.GetName()
    return (category, label)


def get_shader_node_primvars(shader_node: Sdr.ShaderNode) -> List[str]:
    """Get the list of primvars required by a shader node."""
    primvars = []
    if shader_node:
        primvars = shader_node.GetPrimvars()
        additional_primvar_properties = shader_node.GetAdditionalPrimvarProperties()
        for prop_name in additional_primvar_properties:
            prop = shader_node.GetShaderInput(prop_name)
            if prop:
                primvar_name = prop.GetDefaultValue()
                if primvar_name:
                    primvars.append(primvar_name)
    return primvars


def get_shader_node_additional_primvar_properties(shader_node: Sdr.ShaderNode) -> List[str]:
    """Get the additional primvar properties for a shader node.

    Args:
        shader_node (Sdr.ShaderNode): The shader node to query.

    Returns:
        List[str]: The list of additional primvar property names.
    """
    if shader_node is None:
        raise ValueError("Invalid shader node. The input shader_node is None.")
    additional_primvar_properties = shader_node.GetAdditionalPrimvarProperties()
    additional_primvar_names = [prop.GetString() for prop in additional_primvar_properties]
    return additional_primvar_names


class MockShaderNode:

    def __init__(self):
        self.name = "TestShaderNode"
        self.label = "TestLabel"
        self.category = "TestCategory"
        self.help = "TestHelp"
        self.departments = ["TestDepartment1", "TestDepartment2"]
        self.pages = ["TestPage1", "TestPage2"]
        self.role = "TestRole"
        self.primvars = ["TestPrimvar1", "TestPrimvar2"]
        self.additional_primvar_properties = ["TestAdditionalPrimvarProperty1", "TestAdditionalPrimvarProperty2"]
        self.implementation_name = "TestImplementationName"
        self.asset_identifier_input_names = ["TestAssetIdentifierInputName1", "TestAssetIdentifierInputName2"]
        self.default_input = "TestDefaultInput"
        self.vstructs = ["TestVstruct1", "TestVstruct2"]

    def GetName(self):
        return self.name

    def GetLabel(self):
        return self.label

    def GetCategory(self):
        return self.category

    def GetHelp(self):
        return self.help

    def GetDepartments(self):
        return self.departments

    def GetPages(self):
        return self.pages

    def GetRole(self):
        return self.role

    def GetPrimvars(self):
        return self.primvars

    def GetAdditionalPrimvarProperties(self):
        return self.additional_primvar_properties

    def GetImplementationName(self):
        return self.implementation_name

    def GetAssetIdentifierInputNames(self):
        return self.asset_identifier_input_names

    def GetDefaultInput(self):

        class MockShaderProperty:

            def GetName(self):
                return "TestDefaultInput"

        return MockShaderProperty()

    def GetAllVstructNames(self):
        return self.vstructs


def get_shader_node_info_summary(shader_node: MockShaderNode) -> Dict[str, Any]:
    """Get a summary of the shader node information.

    Args:
        shader_node (MockShaderNode): The shader node to get the summary for.

    Returns:
        Dict[str, Any]: A dictionary containing the shader node information summary.
    """
    info_summary = {}
    info_summary["name"] = shader_node.GetName()
    info_summary["label"] = shader_node.GetLabel()
    info_summary["category"] = shader_node.GetCategory()
    info_summary["help"] = shader_node.GetHelp()
    info_summary["departments"] = shader_node.GetDepartments()
    info_summary["pages"] = shader_node.GetPages()
    info_summary["role"] = shader_node.GetRole()
    info_summary["primvars"] = shader_node.GetPrimvars()
    info_summary["additional_primvar_properties"] = shader_node.GetAdditionalPrimvarProperties()
    info_summary["implementation_name"] = shader_node.GetImplementationName()
    info_summary["asset_identifier_input_names"] = shader_node.GetAssetIdentifierInputNames()
    default_input = shader_node.GetDefaultInput()
    info_summary["default_input"] = default_input.GetName() if default_input else None
    info_summary["vstructs"] = shader_node.GetAllVstructNames()
    return info_summary


def get_shader_node_shader_outputs(shader_node: Sdr.ShaderNode) -> List[Sdr.ShaderProperty]:
    """Get all shader output properties of a shader node.

    Args:
        shader_node (Sdr.ShaderNode): The shader node to get outputs from.

    Returns:
        List[Sdr.ShaderProperty]: A list of shader output properties.
    """
    if not shader_node:
        raise ValueError("Invalid shader node provided.")
    pages = shader_node.GetPages()
    shader_outputs = []
    for page in pages:
        property_names = shader_node.GetPropertyNamesForPage(page)
        for prop_name in property_names:
            prop = shader_node.GetShaderOutput(prop_name)
            if prop:
                shader_outputs.append(prop)
    property_names = shader_node.GetPropertyNamesForPage("")
    for prop_name in property_names:
        prop = shader_node.GetShaderOutput(prop_name)
        if prop:
            shader_outputs.append(prop)
    return shader_outputs


def get_shader_node_complete_primvar_list(shader_node: Sdr.ShaderNode) -> List[str]:
    """Get the complete list of primvar names used by a shader node.

    This includes the primvars directly used by the node, as well as any
    additional primvars specified through the node's input properties.

    Args:
        shader_node (Sdr.ShaderNode): The shader node to get the primvar list for.

    Returns:
        List[str]: The list of primvar names used by the shader node.
    """
    primvars = list(shader_node.GetPrimvars())
    additional_primvar_properties = shader_node.GetAdditionalPrimvarProperties()
    for property_name in additional_primvar_properties:
        input_property = shader_node.GetShaderInput(property_name)
        if input_property:
            primvar_name = input_property.GetDefaultValue()
            if primvar_name:
                primvars.append(primvar_name)
    return primvars


def get_shader_node_properties_by_page(shader_node: Sdr.ShaderNode, page_name: str) -> List[Sdr.ShaderProperty]:
    """Get a list of shader properties for a given page name.

    Args:
        shader_node (Sdr.ShaderNode): The shader node to query properties from.
        page_name (str): The name of the page to filter properties by.

    Returns:
        List[Sdr.ShaderProperty]: A list of shader properties for the given page name.
    """
    if not shader_node:
        raise ValueError("Invalid shader node.")
    property_names = shader_node.GetPropertyNamesForPage(page_name)
    shader_properties = []
    for property_name in property_names:
        input_property = shader_node.GetShaderInput(property_name)
        if input_property:
            shader_properties.append(input_property)
        else:
            output_property = shader_node.GetShaderOutput(property_name)
            if output_property:
                shader_properties.append(output_property)
    return shader_properties


def replace_shader_node_in_prims(stage: Usd.Stage, old_shader_node: str, new_shader_node: str, prims: List[str]) -> int:
    """Replace a shader node in the given prims with a new shader node.

    Args:
        stage (Usd.Stage): The USD stage.
        old_shader_node (str): The path to the old shader node to replace.
        new_shader_node (str): The path to the new shader node to use.
        prims (List[str]): A list of prim paths to process.

    Returns:
        int: The number of prims that were modified.
    """
    num_prims_modified = 0
    for prim_path in prims:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            continue
        binding_api = UsdShade.MaterialBindingAPI(prim)
        material = binding_api.ComputeBoundMaterial()
        if not material:
            continue
        shader_nodes = []
        for output in material.GetOutputs():
            if output.GetBaseName() == "surface":
                shader_nodes = UsdShade.ShaderNodeList.append(shader_nodes, output.GetConnectedSources())
        old_node = stage.GetPrimAtPath(old_shader_node)
        if old_node in shader_nodes:
            new_node = stage.GetPrimAtPath(new_shader_node)
            for output in material.GetOutputs():
                if output.GetBaseName() == "surface":
                    for i, node in enumerate(output.GetConnectedSources()):
                        if node == old_node:
                            output.ConnectToSource(new_node.GetOutputs()[0], Usd.ListPositionFrontOfAppendList)
                            output.DisconnectSource(old_node.GetOutputs()[0])
                            num_prims_modified += 1
                            break
    return num_prims_modified


def filter_shaders_by_help_text(registry: Sdr.Registry, keywords: List[str]) -> List[Sdr.ShaderNode]:
    """
    Filter shaders in a registry by help text keywords.

    Args:
        registry (Sdr.Registry): The shader registry to search.
        keywords (List[str]): A list of keywords to match against shader help text.

    Returns:
        List[Sdr.ShaderNode]: A list of shader nodes whose help text contains all the provided keywords.
    """
    shader_nodes = registry.GetShaderNodesByFamily()
    filtered_nodes = []
    for shader_node in shader_nodes:
        help_text = shader_node.GetHelp()
        if all((keyword.lower() in help_text.lower() for keyword in keywords)):
            filtered_nodes.append(shader_node)
    return filtered_nodes


def get_shader_properties_by_hint(shader: Sdr.ShaderNode, hint_name: str) -> Dict[str, Sdr.ShaderProperty]:
    """
    Get a dictionary of shader properties that have a specific hint.

    Args:
        shader (Sdr.ShaderNode): The shader node to query properties from.
        hint_name (str): The name of the hint to filter properties by.

    Returns:
        Dict[str, Sdr.ShaderProperty]: A dictionary mapping property names to ShaderProperty objects
            that have the specified hint.
    """
    if not shader:
        raise ValueError("Invalid shader node.")
    input_properties = shader.GetInputNames()
    output_properties = shader.GetOutputNames()
    filtered_properties = {}
    for prop_name in input_properties + output_properties:
        prop = shader.GetShaderInput(prop_name) or shader.GetShaderOutput(prop_name)
        if prop:
            hints = prop.GetHints()
            if hint_name in hints:
                filtered_properties[prop.GetName()] = prop
    return filtered_properties


class ShaderProperty:

    def __init__(self, name: str, type: Sdf.ValueTypeName, label: str = "", default=None):
        self._name = name
        self._type = type
        self._label = label
        self._default = default

    def GetName(self) -> str:
        return self._name

    def GetLabel(self) -> str:
        return self._label


def get_shader_property_labels(shader_properties: List[ShaderProperty]) -> List[str]:
    """
    Get the labels of a list of ShaderProperty objects.

    Args:
        shader_properties (List[ShaderProperty]): A list of ShaderProperty objects.

    Returns:
        List[str]: A list of labels corresponding to the input ShaderProperty objects.
    """
    labels = []
    for prop in shader_properties:
        label = prop.GetLabel()
        if not label:
            label = prop.GetName()
        labels.append(label)
    return labels


def filter_shaders_by_widget_type(shaders: List[Sdr.ShaderNode], widget_type: str) -> List[Sdr.ShaderNode]:
    """Filter a list of ShaderNodes by the widget type of their input properties.

    Args:
        shaders (List[Sdr.ShaderNode]): List of ShaderNodes to filter.
        widget_type (str): The widget type to filter by.

    Returns:
        List[Sdr.ShaderNode]: List of ShaderNodes that have at least one input property with the specified widget type.
    """
    filtered_shaders = []
    for shader in shaders:
        properties = shader.GetInputNames()
        for prop_name in properties:
            prop = shader.GetInput(prop_name)
            if prop.GetWidget() == widget_type:
                filtered_shaders.append(shader)
                break
    return filtered_shaders


def get_shader_implementation_names(shader_property: Sdr.ShaderProperty) -> List[str]:
    """Get the implementation names for a shader property."""
    if not isinstance(shader_property, Sdr.ShaderProperty):
        raise TypeError("Input must be a valid Sdr.ShaderProperty")
    impl_name = shader_property.GetImplementationName()
    return [impl_name]


def get_shader_property_valid_options(shader_property: Sdr.ShaderProperty) -> List[Tuple[str, Optional[str]]]:
    """
    Get the valid options for a shader property.

    Args:
        shader_property (Sdr.ShaderProperty): The shader property to get valid options for.

    Returns:
        List[Tuple[str, Optional[str]]]: A list of tuples containing the option name and value (if any).
    """
    options: Sdr.NdrOptionVec = shader_property.GetOptions()
    valid_options: List[Tuple[str, Optional[str]]] = []
    for option in options:
        option_name: str = option.GetName()
        if option.GetDefaultValue():
            valid_options.append((option_name, option.GetDefaultValue()))
        else:
            valid_options.append((option_name, None))
    return valid_options


def get_shader_property_defaults(shader_property: Sdr.ShaderProperty) -> Dict[str, Any]:
    """
    Get the default values for a shader property as a dictionary.

    Args:
        shader_property (Sdr.ShaderProperty): The shader property to get defaults for.

    Returns:
        Dict[str, Any]: A dictionary containing the default values for the shader property.
    """
    defaults = {}
    default_value = shader_property.GetDefaultValueAsSdfType()
    if default_value:
        defaults["default"] = default_value
    help_str = shader_property.GetHelp()
    if help_str:
        defaults["help"] = help_str
    hints = shader_property.GetHints()
    if hints:
        defaults["hints"] = hints
    impl_name = shader_property.GetImplementationName()
    if impl_name:
        defaults["implementationName"] = impl_name
    label = shader_property.GetLabel()
    if label:
        defaults["label"] = label
    options = shader_property.GetOptions()
    if options:
        defaults["options"] = options
    page = shader_property.GetPage()
    if page:
        defaults["page"] = page
    vstruct_cond_expr = shader_property.GetVStructConditionalExpr()
    if vstruct_cond_expr:
        defaults["vstructConditionalExpr"] = vstruct_cond_expr
    vstruct_member_name = shader_property.GetVStructMemberName()
    if vstruct_member_name:
        defaults["vstructMemberName"] = vstruct_member_name
    vstruct_member_of = shader_property.GetVStructMemberOf()
    if vstruct_member_of:
        defaults["vstructMemberOf"] = vstruct_member_of
    valid_conn_types = shader_property.GetValidConnectionTypes()
    if valid_conn_types:
        defaults["validConnectionTypes"] = valid_conn_types
    widget = shader_property.GetWidget()
    if widget:
        defaults["widget"] = widget
    is_asset_identifier = shader_property.IsAssetIdentifier()
    defaults["isAssetIdentifier"] = is_asset_identifier
    is_default_input = shader_property.IsDefaultInput()
    defaults["isDefaultInput"] = is_default_input
    is_vstruct = shader_property.IsVStruct()
    defaults["isVStruct"] = is_vstruct
    is_vstruct_member = shader_property.IsVStructMember()
    defaults["isVStructMember"] = is_vstruct_member
    return defaults
