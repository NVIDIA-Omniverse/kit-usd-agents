## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

__all__ = ["NetworkModifier"]


class NetworkModifier:
    """
    The `NetworkModifier` class provides mechanisms to dynamically alter the state
    and behavior of an node-based network during its invokeing cycle.

    This class primarily supports asynchronous operations, with synchronous operations
    reserved for potential future implementation. The modifications facilitated by this
    class encompass operations like adding, removing, reordering, reconnecting, and
    cloning nodes within the network.

    Note:
        Direct network modifications by nodes are restricted to prevent infinite loops
        or unintended behaviors, especially during network restoration post-deserialization.
    """

    def on_begin_invoke(self, network: "RunnableNetwork"):
        """
        Callback triggered at the start of the network invokeing cycle.

        Note:
            This synchronous version is not the primary focus. The emphasis is on its
            asynchronous counterpart.

        Args:
            network (RunnableNetwork): The node network initiating the invoke.
        """
        pass

    def on_pre_invoke(self, network: "RunnableNetwork", node: "RunnableNode"):
        """
        Callback executed before an individual node's invokeing begins.

        Note:
            Current focus is on the async version. Synchronous implementation may be
            considered in the future.

        Args:
            network (RunnableNetwork): The node network associated with the invoke.
            node (Node): The node about to be invokeed.
            force (bool, optional): Indicates if the network expects the new node to be created. Defaults to False.
        """
        pass

    def on_post_invoke(self, network: "RunnableNetwork", node: "RunnableNode"):
        """
        Callback triggered after an node's invokeing is completed.

        Note:
            Emphasis is on the async version, with the sync version considered for future
            implementation.

        Args:
            network (RunnableNetwork): The node network associated with the invoke.
            node (Node): The node that has been invokeed.
            force (bool, optional): Indicates if the network desired the new node to be created. Defaults to False.
        """
        pass

    def on_end_invoke(self, network: "RunnableNetwork"):
        """
        Callback activated at the conclusion of the network invokeing cycle.

        Note:
            The main focus remains on the async counterpart, with this synchronous version
            reserved for potential future inclusion.

        Args:
            network (RunnableNetwork): The node network concluding the invoke.
        """
        pass

    async def on_begin_invoke_async(self, network: "RunnableNetwork"):
        """
        Asynchronous callback triggered at the start of the network invokeing cycle.

        Args:
            network (RunnableNetwork): The node network initiating the invoke.

        Returns:
            Output of the synchronous counterpart `on_begin_invoke`.
        """
        return self.on_begin_invoke(network)

    async def on_pre_invoke_async(
        self, network: "RunnableNetwork", node: "RunnableNode"
    ):
        """
        Asynchronous callback executed before the invokeing of an individual node begins.

        Args:
            network (RunnableNetwork): The node network associated with the invoke.
            node (Node): The node about to undergo invokeing.
            force (bool, optional): Indicates if the network expects an output. Defaults to False.

        Returns:
            Output of the synchronous counterpart `on_pre_invoke`.
        """
        return self.on_pre_invoke(network, node)

    async def on_post_invoke_async(
        self, network: "RunnableNetwork", node: "RunnableNode"
    ):
        """
        Asynchronous callback activated after the invokeing of an node concludes.

        Args:
            network (RunnableNetwork): The node network associated with the invoke.
            node (Node): The node that was invokeed.
            force (bool, optional): Indicates if the network desired an output. Defaults to False.

        Returns:
            Output of the synchronous counterpart `on_post_invoke`.
        """
        return self.on_post_invoke(network, node)

    async def on_end_invoke_async(self, network: "RunnableNetwork"):
        """
        Asynchronous callback initiated at the end of the network's invokeing cycle.

        Args:
            network (RunnableNetwork): The node network wrapping up the invoke.

        Returns:
            Output of the synchronous counterpart `on_end_invoke`.
        """
        return self.on_end_invoke(network)
