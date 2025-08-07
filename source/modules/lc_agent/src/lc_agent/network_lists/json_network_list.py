## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from ..utils.pydantic import is_using_pydantic_v1
from .network_list import NetworkList
from typing import Optional
import getpass
import glob
import json
import os
import re


def _sanitize_filename(filename: str) -> str:
    """Sanitize the filename by removing invalid characters."""
    s = re.sub(r"[^\w\s-]", "", filename)  # Remove non-alphanumeric characters except whitespace and hyphen
    s = re.sub(r"\s+", "_", s)  # Replace whitespace with underscore
    return s


class JsonNetworkList(NetworkList):
    """Custom save/load for Network List"""

    SAVE_PATH = "%%tmp%%/networks/%%user%%"

    def __init__(self, username: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A dictionary to keep track of the checksums for each filename
        self.network_hashes = {}
        self._username = username

    def _compute_hash(self, network: "RunnableNetwork") -> str:
        """Compute a fast hash for a given data"""
        # Handle different Pydantic versions
        if is_using_pydantic_v1():
            # Pydantic v1 way
            network_str = network.json(sort_keys=True)
        else:
            # Pydantic v2 way - use model_dump instead of json
            network_dict = network.model_dump()
            # Convert to string for hashing
            network_str = json.dumps(network_dict, sort_keys=True)

        # hash() returns a signed integer, make it positive and convert to hex
        return hex(hash(network_str) & 0xFFFFFFFF)[2:]

    def _get_save_path(self, filename: str = "") -> str:
        resolved = self.SAVE_PATH

        if "%%user%%" in resolved:
            username = self._username or getpass.getuser() or ""
            resolved = resolved.replace("%%user%%", username)

        if "%%tmp%%" in resolved:
            tmpdir = os.path.join(os.path.expanduser("~"), ".lc_agent")
            resolved = resolved.replace("%%tmp%%", tmpdir)

        return os.path.join(resolved, filename)

    def save(self, network: Optional["RunnableNetwork"] = None):
        networks_to_serialize = [network] if network else self
        saved_files = []
        # Networks that have changes and need to be saved
        to_save = []

        dirpath = self._get_save_path()
        os.makedirs(dirpath, exist_ok=True)

        for idx, n in enumerate(networks_to_serialize):
            network_name = _sanitize_filename(n.name or n.metadata.get("name", None) or "RunnableNetwork")
            filename = f"{idx:04}_{network_name}.json"
            current_hash = self._compute_hash(n)

            saved_files.append(filename)

            # Determine which networks have changed
            if filename not in self.network_hashes or self.network_hashes[filename] != current_hash:
                to_save.append(n)
                # Update hash
                self.network_hashes[filename] = current_hash

                filepath = self._get_save_path(filename)
                with open(filepath, "w") as file:
                    # Use json library to format the output with indentation
                    network_data = json.loads(n.json())
                    json.dump(network_data, file, indent=2)

                print(f"Network saved: {filepath}")

        # Remove old files that aren't in current list
        all_files = set(glob.glob(os.path.join(dirpath, "*.json")))
        all_files_shortnames = {os.path.basename(f) for f in all_files}

        for old_file in all_files_shortnames - set(saved_files):
            os.remove(os.path.join(dirpath, old_file))
            self.network_hashes.pop(old_file, None)
            print(f"Deleted old network file: {os.path.join(dirpath, old_file)}")

    def load(self):
        from lc_agent import RunnableNetwork

        dirpath = self._get_save_path()  # To get the directory path

        if not os.path.exists(dirpath):
            print(f"Can't load history. Directory doesn't exist: {dirpath}")
            return

        files = sorted(glob.glob(os.path.join(dirpath, "*.json")))

        self.clear()

        self.network_hashes.clear()
        for filepath in files:
            filename = os.path.basename(filepath)
            try:
                if is_using_pydantic_v1():
                    network = RunnableNetwork.parse_file(filepath)
                else:
                    # Read the JSON file and parse it into a Python dictionary
                    with open(filepath, "r", encoding="utf-8") as f:
                        data_dict = json.load(f)

                    # Use model_validate with the dictionary
                    network = RunnableNetwork.model_validate(data_dict)
            except TypeError as e:
                import traceback

                traceback.print_exc()
                print(f"Can't load {filename} because {e}")
                continue
            except KeyError as e:
                import traceback

                traceback.print_exc()
                print(f"Can't load {filename} because {e}")
                continue
            self.append(network)
            self.network_hashes[filename] = self._compute_hash(network)

    def delete(self, network: "RunnableNetwork"):
        if network not in self:
            print(f"Network not found: {network}")
            return

        self.remove(network)

        filename = _sanitize_filename(network.name or network.metadata.get("name", None) or "RunnableNetwork")
        filename = f"{filename}.json"
        filepath = self._get_save_path(filename)

        if os.path.exists(filepath):
            os.remove(filepath)
            self.network_hashes.pop(filename, None)
            print(f"Deleted network file: {filepath}")
        else:
            print(f"File not found: {filepath}")
