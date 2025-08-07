## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

__all__ = ["get_node_factory"]

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union
import inspect

DEBUG_CREATED_AT = "/exts/omni.ai_node.core/factory/debug/created_at"


def is_debug_created_at() -> bool:
    return False
    # return bool(carb.settings.get_settings().get(DEBUG_CREATED_AT))


class NodeFactory:
    """
    Registry for Node types. It's used to create the node by registered name.

    It's a singleton class. Use get_node_factory() to get the instance.
    """

    def __init__(self):
        """
        Initializes a new instance of the NodeFactory class.
        """
        self._registered_nodes = {}

    def register(self, node_type: Type, *args, **kwargs):
        """
        Registers a new node type with the node registry.

        Args:
            node_type (type): The class of the node.
            *args: Additional arguments to be passed to the node constructor.
            **kwargs: Additional keyword arguments to be passed to the node constructor.
        """
        name = kwargs.get("name", None) or str(node_type.__name__)

        self._registered_nodes[name] = (node_type, args, kwargs)

    def unregister(self, node_type: Union[Type, str]):
        """
        Unregisters an node type from the node registry.

        Args:
            node_type (Union[Type, str]): The class of the node or name of the node type to be unregistered.
        """
        if isinstance(node_type, str):
            name = node_type
        elif hasattr(node_type, "name"):
            name = node_type.name
        else:
            name = str(node_type.__name__)

        self._registered_nodes.pop(name, None)

    def has_registered(self, name: str) -> bool:
        """
        Checks if the node registry has a registered node type with the given name.

        Args:
            name (str): The name of the node type.

        Returns:
            bool: True if the node registry has a registered node type with the given name, False otherwise.
        """
        name = self._find_by_name_or_type(name)
        return name is not None

    def create_node(self, node_name: str, *args, **kwargs) -> "RunnableNode":
        """
        Creates a new instance of a registered node type.

        Args:
            name (str): The name of the node type.
            *args: Additional arguments to be passed to the node constructor.
            **kwargs: Additional keyword arguments to be passed to the node constructor.

        Returns:
            Node: A new instance of the registered node type.

        Raises:
            KeyError: If the specified node type is not registered.
        """
        node_name = self._find_by_name_or_type(node_name)
        if node_name is None:
            # No exception. Just return None and let the caller handle it.
            return None

        node_type, base_args, base_kwargs = self._registered_nodes[node_name]
        merged_args = base_args + args

        # Deep merge kwargs, with kwargs taking precedence
        merged_kwargs = {}
        for key, value in base_kwargs.items():
            if key in kwargs:
                # If both are dicts, merge them with kwargs taking precedence
                if isinstance(value, dict) and isinstance(kwargs[key], dict):
                    merged_kwargs[key] = {**value, **kwargs[key]}
                else:
                    # Non-dict values from kwargs take precedence
                    merged_kwargs[key] = kwargs[key]
            else:
                # Key only in base_kwargs
                merged_kwargs[key] = value

        # Add remaining kwargs keys
        for key, value in kwargs.items():
            if key not in merged_kwargs:
                merged_kwargs[key] = value

        created_node = node_type(*merged_args, **merged_kwargs)

        if is_debug_created_at():
            # Get the stack frame of the previous function call (up one level from the current)
            previous_frame = inspect.currentframe().f_back
            frame_info = inspect.getframeinfo(previous_frame)

            # Extract file name and line number
            file_name = frame_info.filename
            line_number = frame_info.lineno

            # Set the debug information as metadata on the created node
            metadata_value = f"{file_name}:{line_number}"
            created_node.metadata["__debug_created_at"] = metadata_value

        return created_node

    def get_registered_node_names(self, hidden=True) -> List[str]:
        """
        Retrieves a list of names for all registered node types.

        Returns:
            List[str]: A list of names for all registered node types.
        """
        if hidden:
            return list(self._registered_nodes.keys())
        else:
            return [
                name
                for name in self._registered_nodes.keys()
                if self._registered_nodes[name][2].get("hidden", False) is False
            ]

    def get_base_node_names(self, name: str) -> List[str]:
        """
        Retrieves the names of all registered base types for the node of the
        specified name.

        Args:
            name (str): The name of the registered node type.

        Returns:
            List[str]: A list of names of registered base types for the node
                    with the given name.
        """

        def get_all_base_types(node_type):
            # Get the type of the object
            all_types = [node_type]

            # Get all base types recursively
            def get_bases(t):
                if t.__bases__:
                    for base in t.__bases__:
                        all_types.append(base)
                        get_bases(base)

            get_bases(node_type)
            return all_types

        node_type = self.get_registered_node_type(name)

        all_base_types = get_all_base_types(node_type)

        # Find and return registered names
        base_node_names = []
        for base_type in all_base_types:
            for registered_name, (
                registered_type,
                _,
                _,
            ) in self._registered_nodes.items():
                if registered_type is base_type:
                    base_node_names.append(registered_name)

        return base_node_names

    def get_registered_node_type(self, name: str) -> Optional[Type]:
        name = self._find_by_name_or_type(name)
        if name is None:
            # No exception. Just return None and let the caller handle it.
            return None

        node_type, base_args, base_kwargs = self._registered_nodes[name]

        return node_type

    def get_registered_node_kwargs(self, name: str) -> Optional[Dict[str, Any]]:
        name = self._find_by_name_or_type(name)
        if name is None:
            # No exception. Just return None and let the caller handle it.
            return None

        return self._registered_nodes[name][2]

    def _find_by_name_or_type(self, node_name: str) -> Optional[str]:
        if node_name not in self._registered_nodes:
            # Not available by name, search by type
            found = False
            for name, (node_type, _, _) in self._registered_nodes.items():
                if node_type.__name__ == node_name:
                    node_name = name
                    found = True
                    break

            if not found:
                # No exception. Just return None and let the caller handle it.
                return None

        return node_name


# Global
FACTORY = NodeFactory()


def get_node_factory():
    """
    Returns the default node registry instance
    """
    global FACTORY
    return FACTORY
