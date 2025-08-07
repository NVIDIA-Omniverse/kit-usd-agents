## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from contextlib import ExitStack, contextmanager
from typing import Callable, Dict, List, Optional, T, Tuple, Type, TypeVar

from pxr import Ar, Sdf, Usd


def set_search_path_and_resolve_asset(search_path: list[str], asset_path: str) -> Optional[str]:
    """
    Set the default search path and resolve the given asset path.

    Args:
        search_path (list[str]): List of directories to search for the asset.
        asset_path (str): The path to the asset to resolve.

    Returns:
        Optional[str]: The resolved absolute path to the asset if found, otherwise None.
    """
    Ar.DefaultResolver.SetDefaultSearchPath(search_path)
    resolver = Ar.GetResolver()
    resolved_path = resolver.Resolve(asset_path)
    if resolved_path:
        return resolved_path
    return None


def resolve_asset_path_with_context(asset_path: str, context: Ar.ResolverContext) -> Ar.ResolvedPath:
    """
    Resolve the given asset path using the provided resolver context.

    Args:
        asset_path (str): The asset path to resolve.
        context (Ar.ResolverContext): The resolver context to use for resolution.

    Returns:
        Ar.ResolvedPath: The resolved path for the asset, or an empty path if unresolvable.
    """
    resolver = Ar.GetResolver()
    with Ar.ResolverContextBinder(context):
        resolved_path = resolver.Resolve(asset_path)
        if resolved_path == Ar.ResolvedPath():
            return Ar.ResolvedPath()
        else:
            return resolved_path


class Notice:

    def __init__(self):
        self._callbacks: List[Callable[[Notice], None]] = []

    def register_notice_callback(self, callback: Callable[["Notice"], None]) -> None:
        self._callbacks.append(callback)

    def unregister_notice_callback(self, callback: Callable[["Notice"], None]) -> bool:
        """Unregister a previously registered callback for this notice type.

        Args:
            callback (Callable[[Notice], None]): The callable to be unregistered.

        Returns:
            bool: True if the callback was successfully unregistered, False otherwise.
        """
        if callback not in self._callbacks:
            return False
        self._callbacks.remove(callback)
        return True


def test_callback(notice: Notice):
    print(f"Received notice: {notice}")


class MockNotice:

    def __init__(self, notice_type: str):
        self._notice_type = notice_type

    def GetType(self) -> str:
        return self._notice_type


def get_notices_by_type(notices: List[MockNotice], notice_type: str) -> List[MockNotice]:
    """Get a list of notices filtered by the specified notice type.

    Args:
        notices (List[MockNotice]): A list of notices to filter.
        notice_type (str): The specific notice type to filter by.

    Returns:
        List[MockNotice]: A list of notices that match the specified type.
    """
    filtered_notices: List[MockNotice] = []
    for notice in notices:
        if notice.GetType() == notice_type:
            filtered_notices.append(notice)
    return filtered_notices


def handle_resolver_changes() -> None:
    """Handle resolver changes by updating the asset resolver context."""
    resolver = Ar.GetResolver()
    new_context = Ar.ResolverContext()
    if resolver.GetCurrentContext() != new_context:
        resolver.RefreshContext(new_context)
        print("Asset resolver context updated.")
    else:
        print("Asset resolver context is already up to date.")


class MockResolverContext:

    def __init__(self):
        self._dependencies = []

    def AddResourceDependency(self, path):
        self._dependencies.append(path)

    def GetResourceDependencyChain(self):
        return self._dependencies


class MockResolverNotice:

    def __init__(self):
        self._primary_context = None

    def SetPrimaryContext(self, context):
        self._primary_context = context

    def GetPrimaryContext(self):
        return self._primary_context

    def HasPrimaryContext(self):
        return self._primary_context is not None


def get_resource_dependency_chain(notice: MockResolverNotice) -> List[str]:
    """Get the resource dependency chain for this notice.

    The dependency chain is a list of resolved asset paths that were
    encountered during the resolution of the primary context.

    Returns:
        List[str]: The resource dependency chain.
    """
    if not notice.HasPrimaryContext():
        return []
    primary_context = notice.GetPrimaryContext()
    dependency_chain = primary_context.GetResourceDependencyChain()
    dependency_chain_str = [str(path) for path in dependency_chain]
    return dependency_chain_str


def resolve_and_validate_asset_path(asset_path: str) -> str:
    """Resolve and validate an asset path.

    Args:
        asset_path (str): The asset path to resolve and validate.

    Returns:
        str: The resolved asset path if valid, otherwise an empty string.
    """
    resolver = Ar.GetResolver()
    resolved_path = resolver.Resolve(asset_path)
    if resolved_path:
        resolved_path_str = resolved_path.GetPathString()
        if Ar.GetResolver().Exists(resolved_path_str):
            return resolved_path_str
        else:
            return ""
    else:
        return ""


def refresh_and_resolve_context(context: Ar.ResolverContext, asset_path: str) -> Ar.ResolvedPath:
    """
    Refresh the given context and resolve the asset path using the refreshed context.

    Args:
        context (Ar.ResolverContext): The context to refresh.
        asset_path (str): The asset path to resolve.

    Returns:
        Ar.ResolvedPath: The resolved path for the given asset path using the refreshed context.
    """
    Ar.GetResolver().RefreshContext(context)
    with Ar.ResolverContextBinder(context):
        resolved_path = Ar.GetResolver().Resolve(asset_path)
    return resolved_path


def resolve_multiple_assets(asset_paths: List[str], resolver: Ar.Resolver) -> List[Ar.ResolvedPath]:
    """
    Resolve multiple asset paths using the given resolver.

    Args:
        asset_paths (List[str]): A list of asset paths to resolve.
        resolver (Ar.Resolver): The asset resolver to use.

    Returns:
        List[Ar.ResolvedPath]: A list of resolved paths corresponding to the input asset paths.
    """
    resolved_paths = []
    for asset_path in asset_paths:
        resolved_path = resolver.Resolve(asset_path)
        if resolved_path:
            resolved_paths.append(resolved_path)
        else:
            resolved_paths.append(Ar.ResolvedPath())
    return resolved_paths


def get_asset_modification_timestamps(resolver: Ar.Resolver, asset_paths: Sdf.AssetPath) -> Dict[str, Ar.Timestamp]:
    """
    Get the modification timestamps for a list of asset paths.

    Args:
        resolver (Ar.Resolver): The asset resolver instance.
        asset_paths (list[str]): List of asset paths to query.

    Returns:
        Dict[str, Ar.Timestamp]: Dictionary mapping asset paths to their modification timestamps.
    """
    timestamps = {}
    for asset_path in asset_paths:
        resolved_path = resolver.Resolve(asset_path)
        if not resolved_path:
            continue
        timestamp = resolver.GetModificationTimestamp(asset_path, resolved_path)
        if timestamp:
            timestamps[asset_path] = timestamp
        else:
            timestamps[asset_path] = Ar.Timestamp()
    return timestamps


def resolve_assets_for_new(asset_paths: List[str]) -> List[Ar.ResolvedPath]:
    """
    Resolve the given asset paths for creating new assets.

    Args:
        asset_paths (List[str]): List of asset paths to resolve.

    Returns:
        List[Ar.ResolvedPath]: List of resolved paths for the given asset paths.
    """
    resolved_paths = []
    for asset_path in asset_paths:
        resolved_path = Ar.GetResolver().ResolveForNewAsset(asset_path)
        if resolved_path:
            resolved_paths.append(resolved_path)
        else:
            print(f"Warning: Could not resolve path for new asset: {asset_path}")
    return resolved_paths


def get_context_dependent_paths(resolver: Ar.Resolver, paths: List[str]) -> List[str]:
    """
    Returns a list of context-dependent paths from the given list of paths.

    Args:
        resolver (Ar.Resolver): The asset resolver instance.
        paths (List[str]): The list of paths to check.

    Returns:
        List[str]: The list of context-dependent paths.
    """
    context_dependent_paths = []
    for path in paths:
        if resolver.IsContextDependentPath(path):
            context_dependent_paths.append(path)
    return context_dependent_paths


def get_extension_for_multiple_assets(resolver: Ar.Resolver, asset_paths: List[str]) -> List[str]:
    """
    Get the file extension for multiple asset paths using the given resolver.

    Args:
        resolver (Ar.Resolver): The asset resolver to use.
        asset_paths (List[str]): List of asset paths to get extensions for.

    Returns:
        List[str]: List of file extensions corresponding to the asset paths.
    """
    extensions = []
    for asset_path in asset_paths:
        if not asset_path:
            extensions.append("")
            continue
        ext = resolver.GetExtension(asset_path)
        extensions.append(ext)
    return extensions


def create_identifier_for_new_assets(asset_path: str, anchor_asset_path: Ar.ResolvedPath) -> str:
    """
    Returns an identifier for a new asset specified by asset_path.

    If anchor_asset_path is not empty, it is the resolved asset path
    that asset_path should be anchored to if it is a relative path.
    """
    if not asset_path:
        raise ValueError("Asset path cannot be empty.")
    resolver_context = Ar.ResolverContext()
    asset_resolver = Ar.GetResolver()
    resolved_path = asset_resolver.ResolveForNewAsset(asset_path)
    if not resolved_path:
        raise ValueError(f"Could not resolve path for new asset: {asset_path}")
    identifier = asset_resolver.CreateIdentifierForNewAsset(asset_path, anchor_asset_path)
    return identifier


def create_context_for_asset_and_refresh(resolver: Ar.Resolver, asset_path: str) -> Ar.ResolverContext:
    """
    Create a default context for the given asset path and refresh the context.

    Args:
        resolver (Ar.Resolver): The asset resolver instance.
        asset_path (str): The path to the asset.

    Returns:
        Ar.ResolverContext: The created and refreshed resolver context.
    """
    context = resolver.CreateDefaultContextForAsset(asset_path)
    if not context:
        raise ValueError(f"Failed to create default context for asset path: {asset_path}")
    resolver.RefreshContext(context)
    return context


def get_asset_info_with_timestamp(asset_path: str, resolved_path: Ar.ResolvedPath):
    """Get asset info and modification timestamp for an asset path.

    Args:
        asset_path (str): The asset path to query.
        resolved_path (Ar.ResolvedPath): The resolved path for the asset.

    Returns:
        Tuple[Ar.AssetInfo, Ar.Timestamp]: A tuple containing the asset info and modification timestamp.

    Raises:
        ValueError: If the asset path or resolved path is invalid.
    """
    if not asset_path:
        raise ValueError("Asset path cannot be empty.")
    if not resolved_path:
        raise ValueError("Resolved path cannot be empty.")
    resolver = Ar.GetResolver()
    asset_info = resolver.GetAssetInfo(asset_path, resolved_path)
    timestamp = resolver.GetModificationTimestamp(asset_path, resolved_path)
    return (asset_info, timestamp)


def can_write_asset(resolved_path: Ar.ResolvedPath) -> bool:
    """Check if an asset can be written to the given resolved path."""
    resolver = Ar.GetResolver()
    can_write = resolver.CanWriteAssetToPath(resolved_path)
    return can_write


def get_asset_info_and_resolve(assetPath: str):
    """
    Resolves the given asset path and returns the asset info and resolved path.

    Args:
        assetPath (str): The asset path to resolve.

    Returns:
        Tuple[Ar.AssetInfo, Ar.ResolvedPath]: A tuple containing the asset info and resolved path.
        If the asset cannot be resolved, returns (None, empty ResolvedPath).
    """
    resolver = Ar.GetResolver()
    resolvedPath = resolver.Resolve(assetPath)
    if not resolvedPath:
        return (None, Ar.ResolvedPath())
    assetInfo = resolver.GetAssetInfo(assetPath, resolvedPath)
    return (assetInfo, resolvedPath)


def manage_context_lifecycle(context: Optional[Ar.ResolverContext] = None) -> Ar.ResolverContext:
    """
    Manage the lifecycle of an Ar.ResolverContext object.

    Args:
        context (Optional[Ar.ResolverContext]): An optional existing context to use.
            If not provided, a new context will be created.

    Returns:
        Ar.ResolverContext: The managed resolver context.
    """
    if context is None:
        context = Ar.ResolverContext()
    binder = Ar.ResolverContextBinder(context)
    try:
        bound_context = context
        assert bound_context == context, "Bound context does not match the provided context"
    finally:
        del binder
    return context


@contextmanager
def switch_context_with_logging(resolver: Ar.Resolver, new_context: Optional[Ar.ResolverContext] = None) -> None:
    """
    Context manager to switch the resolver context and log the change.

    Args:
        resolver (Ar.Resolver): The asset resolver instance.
        new_context (Optional[Ar.ResolverContext]): The new context to switch to.
            If None, unbinds the current context. Defaults to None.

    Yields:
        None
    """
    old_context = resolver.GetCurrentContext()
    try:
        if new_context is not None:
            with Ar.ResolverContextBinder(new_context):
                print(f"Switched context from {old_context.GetDebugString()} to {new_context.GetDebugString()}")
                yield
        else:
            with Ar.ResolverContextBinder(Ar.ResolverContext()):
                print(f"Unbound context {old_context.GetDebugString()}")
                yield
    finally:
        if not old_context.IsEmpty():
            with Ar.ResolverContextBinder(old_context):
                print(f"Restored context to {old_context.GetDebugString()}")
        else:
            print("Restored to empty context")


def batch_bind_contexts_and_modify(context_list: List[Ar.ResolverContext], modifier_fn: Callable[[None], None]) -> None:
    """
    Bind a list of ArResolverContext objects in batch, call a modifier
    function, and then unbind them all safely using ExitStack.

    Args:
        context_list (List[Ar.ResolverContext]): List of resolver contexts to bind.
        modifier_fn (Callable[[None], None]): Function to call while contexts are bound.
    """
    if not context_list:
        raise ValueError("Context list cannot be empty.")
    if not callable(modifier_fn):
        raise TypeError("modifier_fn must be a callable.")
    with ExitStack() as stack:
        for ctx in context_list:
            binder = Ar.ResolverContextBinder(ctx)
            stack.enter_context(binder)
        modifier_fn()


def test_modifier():
    print("Modifier function called.")


def bind_context_and_collect_asset_paths(context: Ar.ResolverContext) -> List[str]:
    """
    Bind the given context to the asset resolver and collect all resolved asset paths.

    Args:
        context: The resolver context to bind.

    Returns:
        A list of resolved asset paths.
    """
    asset_paths = []

    @contextmanager
    def bind_context():
        try:
            with Ar.ResolverContextBinder(context):
                yield
        except Exception as e:
            raise RuntimeError(f"Failed to bind resolver context: {str(e)}") from e

    try:
        with bind_context():
            resolver = Ar.GetResolver()
            resolved_paths = resolver.Resolve("")
            if isinstance(resolved_paths, Ar.ResolvedPath):
                asset_paths.append(resolved_paths)
            else:
                asset_paths.extend(resolved_paths)
    except Exception as e:
        raise RuntimeError(f"Failed to resolve assets: {str(e)}") from e
    return [str(path) for path in asset_paths]


T = TypeVar("T")


class ResolverContextBinder:

    def bind_and_execute(self, resolver_context: Ar.ResolverContext, func: Callable[[], T]) -> T:
        """
        Bind the given resolver context and execute the given function.

        Args:
            resolver_context (Ar.ResolverContext): The resolver context to bind.
            func (Callable[[], T]): The function to execute while the resolver context is bound.

        Returns:
            T: The return value of the executed function.

        Raises:
            TypeError: If the given resolver context is not of type Ar.ResolverContext.
            RuntimeError: If the given function raises an exception during execution.
        """
        if not isinstance(resolver_context, Ar.ResolverContext):
            raise TypeError(f"Expected Ar.ResolverContext, got {type(resolver_context)}")
        with Ar.ResolverContextBinder(resolver_context):
            try:
                result = func()
            except Exception as e:
                raise RuntimeError(f"Function execution failed: {str(e)}") from e
        return result


def test_function() -> str:
    return "Test successful"


def cache_prim_references(stage: Usd.Stage, prim: Usd.Prim, resolver_cache: Ar.ResolverScopedCache) -> None:
    """Cache all references on the given prim and its descendants using the provided resolver cache.

    Args:
        stage (Usd.Stage): The USD stage.
        prim (Usd.Prim): The prim to start caching references from.
        resolver_cache (Ar.ResolverScopedCache): The resolver cache to use for caching.
    """
    if not prim.IsValid():
        return
    prim_index = prim.GetPrimIndex()
    node_iter = prim_index.GetNodeIterator()
    for node in node_iter:
        if node.GetArcType() == "reference":
            ref = node.GetPath().pathString
            with resolver_cache:
                resolved_ref = Ar.GetResolver().Resolve(ref)
                if Ar.GetResolver().IsAssetPath(resolved_ref):
                    referenced_layer = Sdf.Layer.Find(resolved_ref)
    for child in prim.GetAllChildren():
        cache_prim_references(stage, child, resolver_cache)


def resolve_asset_path(resolver: Ar.Resolver, asset_path: str) -> Optional[str]:
    """
    Resolve the given asset path using the provided asset resolver.

    Args:
        resolver (Ar.Resolver): The asset resolver to use for resolution.
        asset_path (str): The asset path to resolve.

    Returns:
        Optional[str]: The resolved asset path if successful, None otherwise.
    """
    with Ar.ResolverContextBinder(resolver.CreateDefaultContext()):
        resolved_path = resolver.Resolve(asset_path)
        if resolved_path:
            resolved_path_str = resolved_path.GetPathString()
            return resolved_path_str
    return None


def batch_resolve_asset_paths(resolver: Ar.Resolver, paths: List[str]) -> List[str]:
    """
    Resolve multiple asset paths at once, using the same cache for all resolutions.

    Args:
        resolver (Ar.Resolver): The asset resolver instance.
        paths (List[str]): A list of asset paths to resolve.

    Returns:
        List[str]: A list of resolved paths, in the same order as the input paths.
    """
    with Ar.ResolverContextBinder(resolver.CreateDefaultContext()):
        resolved_paths = []
        for path in paths:
            try:
                resolved_path = resolver.Resolve(path)
                resolved_paths.append(resolved_path)
            except Ar.ResolverError as e:
                resolved_paths.append(None)
                print(f"Failed to resolve path {path}: {str(e)}")
    return resolved_paths


@contextmanager
def manage_asset_cache_lifetime(resolver: Ar.Resolver, cache_scope_path: str) -> None:
    """
    Manage the lifetime of an asset resolver cache scope.

    Args:
        resolver (Ar.Resolver): The asset resolver instance.
        cache_scope_path (str): The path to the cache scope.

    Raises:
        ValueError: If the cache scope path is empty.
    """
    if not cache_scope_path:
        raise ValueError("Cache scope path cannot be empty.")
    with Ar.ResolverContextBinder(resolver.CreateDefaultContextForAsset(cache_scope_path)):
        try:
            yield
        finally:
            pass


def compare_asset_timestamps(timestamp1: Ar.Timestamp, timestamp2: Ar.Timestamp) -> int:
    """
    Compare two asset timestamps and return their relative order.

    Args:
        timestamp1 (Ar.Timestamp): The first timestamp to compare.
        timestamp2 (Ar.Timestamp): The second timestamp to compare.

    Returns:
        int: -1 if timestamp1 is earlier than timestamp2,
             0 if timestamp1 is equal to timestamp2,
             1 if timestamp1 is later than timestamp2.

    Raises:
        ValueError: If either timestamp is invalid.
    """
    if not timestamp1.IsValid() or not timestamp2.IsValid():
        raise ValueError("One or both timestamps are invalid.")
    time1 = timestamp1.GetTime()
    time2 = timestamp2.GetTime()
    if time1 < time2:
        return -1
    elif time1 > time2:
        return 1
    else:
        return 0


def get_latest_asset_modification_time(resolver: Ar.Resolver, asset_paths: List[str]) -> float:
    """
    Get the latest modification time among a list of asset paths.

    Args:
        resolver (Ar.Resolver): The asset resolver to use.
        asset_paths (List[str]): The list of asset paths to check.

    Returns:
        float: The latest modification time as a Unix timestamp, or 0 if no valid timestamp is found.
    """
    latest_time = 0.0
    for asset_path in asset_paths:
        resolved_path = resolver.Resolve(asset_path)
        if resolved_path:
            timestamp = resolver.GetModificationTimestamp(asset_path, resolved_path)
            if timestamp.IsValid():
                time = timestamp.GetTime()
                if time > latest_time:
                    latest_time = time
    return latest_time
