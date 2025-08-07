## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

__all__ = ["NetworkList"]

from typing import Any
from typing import Dict
from typing import List
from typing import Callable
import abc
import enum


class NetworkList:
    """
    A specialized container for managing `RunnableNetwork` instances.

    Overview:
    --------
    While `NetworkList` is similar to standard Python lists, it extends
    typical list functionalities to suit the specific needs of handling
    `RunnableNetwork` objects. The class can be seamlessly integrated with the Omni
    UI system, thanks to its implementation of select methods from
    `AbstractItemModel`, even without direct inheritance.

    Features:
    --------
    - **Basic List Operations**: The class provides foundational list operations
                                 such as appending, removing, and indexing.
    - **Enhanced Search**: It offers powerful searching capabilities to find
                           networks or nodes based on given criteria or metadata.
    - **Event-driven Programming**: Users can register callback functions to be
                                    invoked for specific events, such as when
                                    networks are added or removed.

    The methods `save` and `load` are left abstract, allowing subclasses to
    customize how the networks should be stored and retrieved.
    """

    class Event(enum.Enum):
        """Enumeration of events to which callbacks can be registered."""

        ALL = 0
        NETWORK_ADDED = 1
        NETWORK_REMOVED = 2
        # ETC...

    def __init__(self):
        """Initializes an empty network list and a dictionary for event callbacks."""
        super().__init__()
        self._networks = []
        self._callbacks = {}

    def append(self, network: "RunnableNetwork"):
        """
        Adds an node network to the internal list and triggers the NETWORK_ADDED event.

        Args:
            network (RunnableNetwork): The node network to add.
        """
        self._networks.append(network)
        self.__event_callback(self.Event.NETWORK_ADDED, {"network": network})

    def remove(self, network: "RunnableNetwork"):
        """
        Removes an node network from the internal list and triggers the NETWORK_REMOVED event.

        Args:
            network (RunnableNetwork): The node network to remove.
        """
        self._networks.remove(network)
        self.__event_callback(self.Event.NETWORK_REMOVED, {"network": network})

    def __getitem__(self, index: int) -> "RunnableNetwork":
        """
        Retrieves an node network by its index.

        Args:
            index (int): Index of the desired node network.

        Returns:
            RunnableNetwork: The node network at the specified index.
        """
        return self._networks[index]

    def __iter__(self):
        yield from self._networks

    def __len__(self):
        return len(self._networks)

    def clear(self):
        """
        list compatible

        Clears the history, removing all node networks.
        """
        self._networks.clear()

    def find_network(
        self, criteria: Callable[["RunnableNetwork"], bool]
    ) -> List["RunnableNetwork"]:
        """
        Searches the list for networks meeting a specific criteria.

        Args:
            criteria (Callable[[RunnableNetwork], bool]): A function to evaluate each RunnableNetwork.

        Returns:
            List[RunnableNetwork]: Networks satisfying the given criteria.
        """
        result = []
        for network in self:
            if criteria(network):
                result.append(network)
        return result

    def find_node(
        self, criteria: Callable[["RunnableNode"], bool]
    ) -> List["RunnableNode"]:
        """
        Searches for nodes in all the networks that meet a given criteria.

        Args:
            criteria (Callable[[RunnableNode], bool]): A function that takes an RunnableNode as input
                and returns True if the network meets the desired criteria, False otherwise.

        Returns:
            List[RunnableNode]: A list of RunnableNode objects that meet the criteria.
        """
        pass

    def find_network_by_metadata(
        self, metadata: Dict[str, Any]
    ) -> List["RunnableNetwork"]:
        """
        Searches for network that has the given metadata.

        Args:
            metadata (Dict[str, Any]): The network meets the criteria if it has
                                       metadata with all the name and values of
                                       this dict.

        Returns:
            List[RunnableNetwork]: A list of RunnableNetwork objects that meet the criteria.
        """

        def criteria(network, metadata=metadata):
            meets = True
            for key, value in metadata.items():
                if network.metadata[key] != value:
                    meets = False
                    break
            return meets

        return self.find_network(criteria)

    def find_node_by_metadata(self, metadata: Dict[str, Any]) -> List["RunnableNode"]:
        """
        Searches for nodes in all the network that meet a given criteria.

        Useful to search the bookmarked nodes.

        Args:
            metadata (Dict[str, Any]): The node meets the criteria if it has
                                       metadata with all the name and values of
                                       this dict.

        Returns:
            List[RunnableNode]: A list of RunnableNode objects that meet the criteria.
        """
        pass

    def filter_conversations(self, criteria: Callable[["RunnableNetwork"], bool]):
        """
        Temporarily hides networks from view if they don't meet the provided criteria.

        Args:
            criteria (Callable[[RunnableNetwork], bool]): Evaluation function for each RunnableNetwork.
        """
        pass

    @abc.abstractmethod
    def save(self, network: "RunnableNetwork" = None):
        """
        Abstract method for saving a network. Subclasses should provide a concrete implementation.

        Args:
            network (RunnableNetwork, optional): The network to save. If None, all networks may be saved.
        """
        pass

    @abc.abstractmethod
    def load(self):
        """
        Abstract method to load networks. Subclasses should provide a concrete implementation.
        """
        pass

    @abc.abstractmethod
    def delete(self, network: "RunnableNetwork"):
        """
        Abstract method for deleting a network async. Subclasses should provide a concrete implementation.

        Args:
            network (RunnableNetwork): The network to delete.
        """
        pass

    @abc.abstractmethod
    async def save_async(self, network: "RunnableNetwork" = None):
        """
        Abstract method for saving a network async. Subclasses should provide a concrete implementation.

        Args:
            network (RunnableNetwork, optional): The network to save. If None, all networks may be saved.
        """
        return self.save(network)

    @abc.abstractmethod
    async def load_async(self):
        """
        Abstract method to load networks. Subclasses should provide a concrete implementation.
        """
        return self.load()

    @abc.abstractmethod
    async def delete_async(self, network: "RunnableNetwork"):
        """
        Abstract method for deleting a network async. Subclasses should provide a concrete implementation.

        Args:
            network (RunnableNetwork): The network to delete.
        """
        return self.delete(network)

    def set_event_fn(
        self,
        callable: Callable[["NetworkList.Event", "Payload"], None],
        event: "NetworkList.Event" = Event.ALL,
        priority: int = 100,
    ) -> int:
        """
        Adds a callback to the events like added/removed, etc...

        Args:
            callable: The callable that will be called on event.
            event: The event to subscribe.
            priority (int): Used to order the process.

        Returns:
            int: id to be able to remove it
        """
        event_id = len(self._callbacks)
        self._callbacks[event_id] = callable
        return event_id

    def remove_event_fn(self, event_id: int):
        """
        Removes the callback.

        Args:
            event_id (int): The id from set_event_fn.
        """
        self._callbacks[event_id] = None

    def __event_callback(self, event: "NetworkList.Event", payload: Dict[str, Any]):
        for i, c in self._callbacks.items():
            if c:
                c(event, payload)
