## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from dataclasses import dataclass
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union
import threading


class ChatModelRegistry:
    @dataclass
    class _ChatModelEntry:
        chat_model: Optional[BaseChatModel]
        chat_model_factory: Optional[Callable[[], BaseChatModel]]
        tokenizer: Optional[Callable]
        max_tokens: Optional[int]
        hidden: bool

    def __init__(self):
        """
        Instantiate the Registry with empty lists and dictionaries for storing class names,
        Chat Models, and optional Generators.
        """
        self.registered_names = []
        self.chat_models: Dict[str, "ChatModelRegistry._ChatModelEntry"] = {}
        self._ref_counts = {}  # Reference counting for concurrent request safety
        self._lock = threading.RLock()  # Thread-safe access to registrations

    def register(
        self,
        name: str,
        chat_model: Union[BaseChatModel, Callable[[], BaseChatModel]],
        tokenizer: Optional[Callable] = None,
        max_tokens: Optional[int] = None,
        hidden: bool = False,
    ):
        """
        Register a Chat Model and optionally a Generator under the same name.

        This method uses reference counting to support concurrent requests that
        register the same model name. Each register() call increments the reference
        count, and the model is only removed when all corresponding unregister()
        calls have been made.

        Args:
            name (str): Name under which the Chat Model and optional Generator will be registered.
            chat_model: The Chat Model object to store or a callable that returns a Chat Model.
            tokenizer: The optional tokenizer function.
            max_tokens: The optional max tokens value.
            hidden: Whether to hide this model from the list of registered names.
        """
        with self._lock:
            # Increment reference count
            self._ref_counts[name] = self._ref_counts.get(name, 0) + 1

            # Only add to registered_names if not already present (avoid duplicates)
            if name not in self.registered_names:
                self.registered_names.append(name)

            # Check if chat_model is a callable (factory function)
            if callable(chat_model) and not isinstance(chat_model, BaseChatModel):
                # Store the factory function, not the model itself
                self.chat_models[name] = self._ChatModelEntry(None, chat_model, tokenizer, max_tokens, hidden)
            else:
                # Store the model directly
                self.chat_models[name] = self._ChatModelEntry(chat_model, None, tokenizer, max_tokens, hidden)

    def unregister(self, name: str):
        """
        Unregister a Chat Model and optionally a Generator under a given name.

        This method decrements the reference count for the model. The model is only
        actually removed when the reference count reaches zero, ensuring that
        concurrent requests that share the same model name don't interfere with
        each other.

        Args:
            name (str): Name under which the Chat Model and optional Generator were registered.
        """
        with self._lock:
            # Decrement reference count
            if name in self._ref_counts:
                self._ref_counts[name] -= 1
                # Only remove when no more references
                if self._ref_counts[name] <= 0:
                    if name in self.registered_names:
                        self.registered_names.remove(name)
                    if name in self.chat_models:
                        self.chat_models.pop(name)
                    del self._ref_counts[name]

    def get_model(self, name: str) -> Optional[BaseChatModel]:
        """
        Get the Chat Model registered under a given name. If name is not provided,
        it defaults to the first registered name.

        Args:
            name (str): Name under which the Chat Model was registered.
        """
        if not self.chat_models or not self.registered_names:
            return None

        if not name:
            # Default is the first one
            name = self.registered_names[0]  # Default is the first one

        entry = self.chat_models.get(name)
        if entry:
            # If the chat model hasn't been created yet but we have a factory function
            if entry.chat_model is None and entry.chat_model_factory is not None:
                # Create the model using the factory function and cache it
                entry.chat_model = entry.chat_model_factory()

            return entry.chat_model

    def get_tokenizer(self, name: str = None) -> Optional[Callable]:
        """
        Get the tokenizer registered under a given name. If name is not provided,
        it defaults to the first registered name.

        Args:
            name (str): Name under which the tokenizer was registered.

        Returns:
            The tokenizer function or None if not registered.
        """
        if not self.chat_models or not self.registered_names:
            return None

        if not name:
            name = self.registered_names[0]  # Default is the first one

        entry = self.chat_models.get(name)
        if entry:
            return entry.tokenizer

    def get_max_tokens(self, name: str = None) -> Optional[int]:
        """
        Get the max tokens registered under a given name. If name is not provided,
        it defaults to the first registered name.

        Args:
            name (str): Name under which the max tokens was registered.

        Returns:
            The max tokens value or None if not registered.
        """
        if not self.chat_models or not self.registered_names:
            return None

        if not name:
            name = self.registered_names[0]  # Default is the first one

        entry = self.chat_models.get(name)
        if entry:
            return entry.max_tokens

    def get_registered_names(self):
        """
        Get a list of all names under which Chat Models and Generators have been registered.

        Returns:
            List of registered names.
        """
        return [name for name in self.registered_names if not self.chat_models[name].hidden]


REGISTRY = ChatModelRegistry()


def get_chat_model_registry():
    """
    Get the global Chat Model Registry.

    Returns:
        The global Chat Model Registry.
    """
    global REGISTRY
    return REGISTRY
