## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from typing import Dict, List, Optional, Tuple, Type

from pxr import Plug, Sdf, Tf, Usd, UsdGeom, UsdShade


def notify_on_prim_removal(notice: Usd.Notice, stage: Usd.Stage) -> None:
    """Notify when a prim is removed from the stage.

    Args:
        notice (Usd.Notice): The notice object.
        stage (Usd.Stage): The USD stage.
    """
    for p in notice.GetChangedInfoOnlyPaths():
        if Sdf.Path(p).IsPrimPath():
            if not stage.GetPrimAtPath(p).IsValid():
                Tf.Status("Prim removed: " + str(p))


def listener(notice, stage):
    notify_on_prim_removal(notice, stage)


def notify_on_stage_changes(stage: Usd.Stage):
    """Register a callback to be notified whenever the stage changes."""

    def notice_callback(notice: Tf.Notice, stage: Usd.Stage) -> None:
        """Callback function to handle stage change notices."""
        if isinstance(notice, Usd.Notice.ObjectsChanged):
            for path in notice.GetChangedInfoOnlyPaths():
                print(f"Object changed: {path}")
        else:
            print(f"Received notice: {type(notice).__name__}")

    Tf.Notice.Register(Usd.Notice.ObjectsChanged, notice_callback, stage)
    Tf.Notice.Register(Usd.Notice.StageEditTargetChanged, notice_callback, stage)


def batch_update_animations(stage: Usd.Stage, anim_data: Dict[str, List[float]]) -> None:
    """Update animation values for multiple prims in a batch.

    Args:
        stage (Usd.Stage): The USD stage to update animations on.
        anim_data (Dict[str, List[float]]): A dictionary mapping prim paths to a list of animation values.

    Raises:
        ValueError: If any of the prim paths are invalid or don't have an animation attribute.
    """
    for prim_path, anim_values in anim_data.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")
        anim_attr = prim.GetAttribute("anim")
        if not anim_attr.IsValid():
            raise ValueError(f"Prim {prim_path} does not have an 'anim' attribute")
        for i, value in enumerate(anim_values):
            anim_attr.Set(value, Usd.TimeCode(i))


def search_and_filter_prims(stage: Usd.Stage, prim_path: str, prim_type: Optional[str] = None) -> List[Usd.Prim]:
    """
    Search for prims under a given prim path and optionally filter by prim type.

    Args:
        stage (Usd.Stage): The USD stage to search.
        prim_path (str): The path of the prim to start the search from.
        prim_type (str, optional): The type of prims to filter. If None, no filtering is applied. Defaults to None.

    Returns:
        List[Usd.Prim]: A list of prims that match the search criteria.
    """
    start_prim = stage.GetPrimAtPath(prim_path)
    if not start_prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")
    found_prims: List[Usd.Prim] = []
    for prim in Usd.PrimRange(start_prim):
        if prim_type is None or prim.GetTypeName() == prim_type:
            found_prims.append(prim)
    return found_prims


def analyze_scene_hierarchy(stage: Usd.Stage) -> Tuple[int, int, int, List[str]]:
    """
    Analyze the scene hierarchy and return statistics.

    Args:
        stage (Usd.Stage): The USD stage to analyze.

    Returns:
        Tuple[int, int, int, List[str]]: A tuple containing:
            - Total number of prims in the stage
            - Number of Xform prims
            - Number of Mesh prims
            - List of prim paths for Mesh prims
    """
    total_prims = 0
    xform_count = 0
    mesh_count = 0
    mesh_paths = []
    for prim in stage.Traverse():
        total_prims += 1
        if prim.IsA(UsdGeom.Xform):
            xform_count += 1
        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1
            mesh_paths.append(str(prim.GetPath()))
    return (total_prims, xform_count, mesh_count, mesh_paths)


def apply_material_to_hierarchy(stage: Usd.Stage, material_prim: UsdShade.Material, prim_path: str) -> bool:
    """Apply a material to all prims in a hierarchy.

    Args:
        stage (Usd.Stage): The USD stage.
        material_prim (UsdShade.Material): The material prim to apply.
        prim_path (str): The path to the root prim of the hierarchy.

    Returns:
        bool: True if the material was applied successfully, False otherwise.
    """
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        print(f"Error: Invalid prim path '{prim_path}'")
        return False
    if not material_prim.GetPrim().IsA(UsdShade.Material):
        print(f"Error: '{material_prim.GetPath()}' is not a valid UsdShadeMaterial")
        return False
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh) and (not prim.IsA(UsdGeom.Gprim)):
            continue
        binding_api = UsdShade.MaterialBindingAPI(prim)
        binding_api.Bind(material_prim)
    return True


def optimize_scene_visibility(stage: Usd.Stage) -> bool:
    """Optimize the scene visibility by pruning invisible prims."""
    root_layer = stage.GetRootLayer()

    def pruning_predicate(prim):
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            return not imageable.ComputeVisibility()
        return False

    prims_to_prune = []
    for prim in stage.Traverse():
        if pruning_predicate(prim):
            prims_to_prune.append(prim.GetPath())
    if not prims_to_prune:
        return False
    edit_target = Usd.EditTarget(root_layer)
    with Usd.EditContext(stage, edit_target):
        for prim_path in prims_to_prune:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                prim.SetActive(False)
    return True


def register_and_report_plugins() -> List[Plug.Plugin]:
    """Register plugins and return a list of newly registered plugins."""
    previously_registered_plugins = set(Plug.Registry().GetAllPlugins())
    Plug.Registry().RegisterPlugins([""])
    currently_registered_plugins = set(Plug.Registry().GetAllPlugins())
    newly_registered_plugins = currently_registered_plugins - previously_registered_plugins
    return list(newly_registered_plugins)


def get_all_plugin_resource_paths() -> List[str]:
    """
    Returns a list of resource paths for all registered plugins.

    Returns:
        List[str]: A list of resource paths for all registered plugins.
    """
    resource_paths = []
    plugins = Plug.Registry().GetAllPlugins()
    for plugin in plugins:
        if plugin.isResource:
            resource_path = plugin.resourcePath
            resource_paths.append(resource_path)
    return resource_paths


def get_loaded_python_plugins() -> List[Plug.Plugin]:
    """Return a list of loaded Python plugins."""
    plugin_registry = Plug.Registry()
    plugins = plugin_registry.GetAllPlugins()
    loaded_python_plugins = []
    for plugin in plugins:
        if not plugin.isLoaded:
            continue
        if not plugin.isPythonModule:
            continue
        loaded_python_plugins.append(plugin)
    return loaded_python_plugins


def load_and_verify_plugin(plugin: Plug.Plugin) -> bool:
    """Load a plugin and verify it loaded successfully.

    Args:
        plugin (Plug.Plugin): The plugin to load.

    Returns:
        bool: True if the plugin loaded successfully, False otherwise.
    """
    try:
        loaded = plugin.Load()
    except Exception as e:
        print(f"Error loading plugin: {e}")
        return False
    if not loaded:
        return False
    if not plugin.isLoaded:
        return False
    metadata = plugin.metadata
    if not metadata:
        return False
    path = plugin.path
    if not path:
        return False
    if plugin.isResource:
        resource_path = plugin.resourcePath
        if not resource_path:
            return False
    return True


def load_plugin_if_needed(plugin: Plug.Plugin) -> bool:
    """Load a plugin if it's not already loaded.

    Args:
        plugin (Plug.Plugin): The plugin to load.

    Returns:
        bool: True if the plugin was loaded or already loaded, False otherwise.
    """
    if plugin.isLoaded:
        return True
    if plugin.isResource:
        return True
    try:
        loaded = plugin.Load()
    except Exception as e:
        print(f"Error loading plugin {plugin.name}: {e}")
        loaded = False
    return loaded


def get_all_plugin_metadata() -> Dict[str, Dict]:
    """
    Retrieve metadata for all registered plugins.

    Returns:
        Dict[str, Dict]: A dictionary mapping plugin names to their metadata.
    """
    all_metadata = {}
    plugins = Plug.Registry().GetAllPlugins()
    for plugin in plugins:
        name = plugin.name
        metadata = plugin.metadata
        all_metadata[name] = metadata
    return all_metadata


def verify_all_plugins_loaded() -> bool:
    """
    Verify that all registered plugins are loaded.

    Returns:
        bool: True if all plugins are loaded, False otherwise.
    """
    plugin_registry = Plug.Registry()
    plugins = plugin_registry.GetAllPlugins()
    all_loaded = True
    for plugin in plugins:
        if plugin.isResource:
            continue
        if not plugin.isLoaded:
            print(f"Warning: Plugin '{plugin.name}' is not loaded.")
            all_loaded = False
    return all_loaded


def get_expired_plugins() -> List[Plug.Plugin]:
    """Return a list of expired Plug.Plugin objects."""
    all_plugins = Plug.Registry().GetAllPlugins()
    expired_plugins = []
    for plugin in all_plugins:
        if plugin.expired:
            expired_plugins.append(plugin)
    return expired_plugins


def find_and_load_plugin_resource(plugin_name: str, resource_path: str, verify: bool = True) -> Optional[str]:
    """
    Find and load a plugin resource by relative path, optionally verifying that the file exists.

    Args:
        plugin_name (str): The name of the plugin to find the resource in.
        resource_path (str): The relative path to the resource within the plugin's resource path.
        verify (bool, optional): Whether to verify that the resource file exists. Defaults to True.

    Returns:
        Optional[str]: The full path to the resource if found, None otherwise.
    """
    plugin = Plug.Registry().GetPluginWithName(plugin_name)
    if not plugin:
        return None
    full_path = plugin.FindPluginResource(resource_path, verify)
    if not full_path:
        return None
    if not plugin.isLoaded:
        loaded = plugin.Load()
        if not loaded:
            return None
    return full_path


def build_and_verify_resource_path(plugin: Plug.Plugin, path: str, verify: bool = True) -> str:
    """
    Build a plugin resource path and optionally verify it exists.

    Args:
        plugin (Plug.Plugin): The plugin to build the resource path for.
        path (str): The path to the resource, either absolute or relative to the plugin's resource path.
        verify (bool, optional): Whether to verify the resource exists. Defaults to True.

    Returns:
        str: The built resource path, or an empty string if verification fails.
    """
    if not plugin:
        raise ValueError("Invalid plugin provided.")
    if os.path.isabs(path):
        if verify and (not os.path.exists(path)):
            return ""
        return path
    else:
        resource_path = plugin.MakeResourcePath(path)
        if verify and (not os.path.exists(resource_path)):
            return ""
        return resource_path


def check_plugin_declaration(plugin: Plug.Plugin, type_to_check: Type, include_subclasses: bool = False) -> bool:
    """
    Check if a given type is declared by the provided plugin.

    Args:
        plugin (Plug.Plugin): The plugin to check for type declaration.
        type_to_check (Type): The type to check for declaration in the plugin.
        include_subclasses (bool, optional): If True, also check for subclasses of the given type. Defaults to False.

    Returns:
        bool: True if the type is declared by the plugin, False otherwise.
    """
    if plugin is None:
        raise ValueError("Invalid plugin provided.")
    if not plugin.isLoaded:
        plugin.Load()
    if plugin.DeclaresType(type_to_check, includeSubclasses=False):
        return True
    if include_subclasses:
        declared_types = [
            cls for cls in type_to_check.__subclasses__() if plugin.DeclaresType(cls, includeSubclasses=False)
        ]
        if declared_types:
            return True
    return False


def get_plugin_status(plugin_name: str) -> Dict[str, bool]:
    """Get the status of a plugin.

    Args:
        plugin_name (str): The name of the plugin to query.

    Returns:
        Dict[str, bool]: A dictionary with keys 'loaded', 'resource', 'python_module'
                         indicating the status of the plugin.
    """
    registry = Plug.Registry()
    plugin = registry.GetPluginWithName(plugin_name)
    if not plugin:
        raise ValueError(f"Plugin '{plugin_name}' not found.")
    is_loaded = plugin.isLoaded
    is_resource = plugin.isResource
    is_python_module = plugin.isPythonModule
    return {"loaded": is_loaded, "resource": is_resource, "python_module": is_python_module}


def load_and_get_metadata(plugin: Plug.Plugin) -> dict:
    """Load the plugin and return its metadata.

    Args:
        plugin (Plug.Plugin): The plugin to load and get metadata from.

    Returns:
        dict: The metadata dictionary for the plugin.

    Raises:
        ValueError: If the plugin is None.
        RuntimeError: If the plugin fails to load.
    """
    if plugin is None:
        raise ValueError("Plugin cannot be None.")
    if not plugin.isLoaded:
        if not plugin.Load():
            raise RuntimeError(f"Failed to load plugin: {plugin.name}")
    metadata = plugin.metadata
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Invalid metadata format for plugin: {plugin.name}")
    return metadata


def get_directly_derived_plugin_types(base_type: Type) -> List[Type]:
    """
    Get a list of types directly derived from the given base type that are provided by plugins.

    Args:
        base_type (Type): The base type to find derived types for.

    Returns:
        List[Type]: A list of types directly derived from the base type.
    """
    derived_types = []
    try:
        Plug.Registry.GetDirectlyDerivedTypes(base_type, derived_types)
    except Exception as e:
        print(f"Error getting directly derived types for {base_type}: {e}")
    valid_types = [t for t in derived_types if t.IsValid()]
    return valid_types


class MyBaseClass(Tf.Type):
    pass


class DerivedClass1(MyBaseClass):
    pass


class DerivedClass2(MyBaseClass):
    pass


def load_all_plugins() -> List[Plug.Plugin]:
    """Load all registered plugins and return a list of loaded plugins."""
    plugins = Plug.Registry().GetAllPlugins()
    loaded_plugins = []
    for plugin in plugins:
        if not plugin.Load():
            print(f"Failed to load plugin: {plugin.name}")
            continue
        loaded_plugins.append(plugin)
    return loaded_plugins
