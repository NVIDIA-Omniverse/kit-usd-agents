## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import threading


class RetrieverRegistry:
    def __init__(self):
        """
        Instantiate the Registry with an empty list for storing class names
        and a dictionary for storing BaseRetriever objects.
        """
        self.registered_names = []
        self.retrievers = {}
        self._ref_counts = {}  # Reference counting for concurrent request safety
        self._lock = threading.RLock()  # Thread-safe access to registrations

    def register(self, name: str, retriever):
        """
        Register a BaseRetriever under the given name.

        This method uses reference counting to support concurrent requests that
        register the same retriever name. Each register() call increments the
        reference count, and the retriever is only removed when all corresponding
        unregister() calls have been made.

        Args:
            name (str): Name under which the BaseRetriever will be registered.
            retriever: The BaseRetriever object to store.
        """
        with self._lock:
            # Increment reference count
            self._ref_counts[name] = self._ref_counts.get(name, 0) + 1

            # Only add to registered_names if not already present (avoid duplicates)
            if name not in self.registered_names:
                self.registered_names.append(name)

            self.retrievers[name] = retriever

    def unregister(self, name: str):
        """
        Unregister a BaseRetriever under a given name.

        This method decrements the reference count for the retriever. The retriever
        is only actually removed when the reference count reaches zero, ensuring that
        concurrent requests that share the same retriever name don't interfere with
        each other.

        Args:
            name (str): Name under which the BaseRetriever was registered.

        Raises:
            ValueError: If the name was never registered.
        """
        with self._lock:
            # Raise ValueError if name was never registered (preserves original behavior)
            if name not in self._ref_counts:
                raise ValueError(f"Retriever '{name}' is not registered")

            # Decrement reference count
            self._ref_counts[name] -= 1
            # Only remove when no more references
            if self._ref_counts[name] <= 0:
                self.registered_names.remove(name)
                self.retrievers.pop(name)
                del self._ref_counts[name]

    def get_retriever(self, name: str):
        """
        Get the BaseRetriever registered under a given name. If name is not provided,
        it defaults to the first registered name.

        Args:
            name (str): Name under which the BaseRetriever was registered.
        """
        if not name and self.registered_names:
            # Default is the first one
            return self.retrievers.get(self.registered_names[0])
        return self.retrievers.get(name)

    def get_registered_names(self):
        """
        Get a list of all names under which BaseRetrievers have been registered.

        Returns:
            List of registered names.
        """
        return self.registered_names[:]


RETRIEVER_REGISTRY = RetrieverRegistry()


def get_retriever_registry():
    """
    Get the global BaseRetriever Registry.

    Returns:
        The global BaseRetriever Registry.
    """
    global RETRIEVER_REGISTRY
    return RETRIEVER_REGISTRY
