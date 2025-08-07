## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from ..runnable_network import RunnableNetwork
from ..utils.pydantic import is_using_pydantic_v1
from .network_list import NetworkList
from typing import Optional
import aioredis
import asyncio
import getpass
import json
import time

_REDIS_HOST = "omni-chatusd-redis.nvidia.com"
_REDIS_PORT = 6379


class RedisNetworkList(NetworkList):
    """Custom save/load for Network List with async Redis operations"""

    def __init__(self, username, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network_hashes = {}
        self._username: str = username or getpass.getuser() or ""
        self._hash_key = f"lc_agent:user:{self._username}:entries"
        self._sorted_set_key = f"lc_agent:user:{self._username}:timeline"

    async def initialize_redis(self):
        # When you use asyncio.run(), it creates a new event loop and if the
        # Redis client was created in this event loop, it automatically closes.
        # To avoid this, we need to create the Redis client in the current event
        # loop every time.
        return await aioredis.Redis.from_url(f"redis://{_REDIS_HOST}:{_REDIS_PORT}", decode_responses=True)

    def _compute_hash(self, network: "RunnableNetwork") -> str:
        """Compute a fast hash for a given data"""
        network_str = network.json(sort_keys=True)
        # hash() returns a signed integer, make it positive and convert to hex
        return hex(hash(network_str) & 0xFFFFFFFF)[2:]

    async def _store_json(self, network: "RunnableNetwork"):
        redis = await self.initialize_redis()
        try:
            network_id = network.uuid()
            json_string = network.json()
            timestamp = time.time()

            if await redis.hexists(self._hash_key, network_id):
                await redis.hset(self._hash_key, network_id, json_string)
            else:
                await redis.hset(self._hash_key, network_id, json_string)
                await redis.zadd(self._sorted_set_key, {network_id: timestamp})
        finally:
            await redis.close()

    async def delete_async(self, network: "RunnableNetwork"):
        redis = await self.initialize_redis()
        try:
            if network not in self:
                print(f"Network not found: {network}")
                return

            self.remove(network)
            network_id = network.uuid()
            await redis.hdel(self._hash_key, network_id)
            await redis.zrem(self._sorted_set_key, network_id)
        finally:
            await redis.close()

    def delete(self, network: "RunnableNetwork"):
        self._run_async(self.delete_async(network))

    async def save_async(self, network: Optional["RunnableNetwork"] = None):
        await self.initialize_redis()

        if network:
            current_id = network.uuid()
            current_hash = self._compute_hash(network)
            await self._store_json(network)

        else:
            for n in self:
                current_id = n.uuid()
                current_hash = self._compute_hash(n)

                if current_id not in self.network_hashes or self.network_hashes[current_id] != current_hash:
                    self.network_hashes[current_id] = current_hash
                    await self._store_json(n)

    def save(self, network: Optional["RunnableNetwork"] = None):
        self._run_async(self.save_async(network))

    async def _retrieve_json(self):
        redis = await self.initialize_redis()
        try:
            entry_ids = await redis.zrange(self._sorted_set_key, 0, -1)
            json_strings = await asyncio.gather(*[redis.hget(self._hash_key, entry_id) for entry_id in entry_ids])
            return [json.loads(j) if j else None for j in json_strings]
        finally:
            await redis.close()

    async def load_async(self):
        from lc_agent import RunnableNetwork

        data = await self._retrieve_json()
        self.clear()

        for entry in data:
            if entry:
                try:
                    if is_using_pydantic_v1():
                        network = RunnableNetwork.parse_obj(entry)
                    else:
                        # Use model_validate for Pydantic v2
                        network = RunnableNetwork.model_validate(entry)
                    self.append(network)
                except Exception as e:
                    print(f"Error loading network: {e}")

    def load(self):
        self._run_async(self.load_async())

    def _run_async(self, coro):
        async def wrapper_async(coro):
            try:
                await coro
            except Exception as e:
                import traceback

                traceback.print_exc()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # If there's no running event loop, create a new one and run the coroutine
            asyncio.run(wrapper_async(coro))
        else:
            # If there's already a running event loop, use it to create a task
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(wrapper_async(coro), loop)
            else:
                loop.run_until_complete(wrapper_async(coro))
