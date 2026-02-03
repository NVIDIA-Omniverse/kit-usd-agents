# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import os
import re
import tempfile

import carb
import carb.settings
import requests
from lc_agent import AIMessage, NetworkModifier, RunnableNetwork, RunnableNode

from .usd_search_node import USDSearchNode


def get_api_key():
    settings = carb.settings.get_settings()

    # Check in order of precedence:
    # 1. USD Search specific API key from settings
    api_key = settings.get("/exts/omni.ai.chat_usd.bundle/usdsearch_api_key")
    if api_key:
        return api_key

    # 2. USD Search specific API key from environment
    api_key = os.environ.get("USDSEARCH_API_KEY")
    if api_key:
        return api_key

    # 3. Generic NVIDIA API key from settings
    api_key = settings.get("/exts/omni.ai.chat_usd.bundle/nvidia_api_key")
    if api_key:
        return api_key

    # 4. Generic NVIDIA API key from environment
    api_key = os.environ.get("NVIDIA_API_KEY")
    return api_key


def get_username():
    settings = carb.settings.get_settings()

    # Check in order of precedence:
    # 1. USD Search username from settings
    username = settings.get("/exts/omni.ai.chat_usd.bundle/usdsearch_username")
    if username:
        return username

    # 2. USD Search username from environment
    username = os.environ.get("USDSEARCH_USERNAME")
    return username


class USDSearchModifier(NetworkModifier):
    """USDSearch API Command:
    @USDSearch(query, metadata: bool, limit: int)@

    Description: Searches the USD API with either text or image queries.

    Parameters:
    - query: Can be either:
        - A text string for text search: "search terms"
        - An image path for image search: <image(path)>
    - metadata: Whether to include metadata in the search results (true/false)
    - limit: The maximum number of results to return

    Examples:
    Text search: @USDSearch("big box", false, 10)@
    Image search: @USDSearch(<image(/path/to/reference.png)>, false, 10)@"""

    def __init__(
        self,
        host_url=None,
        api_key=None,
        username=None,
        url_replacements=None,
        search_path=None,
    ):
        """Initialize USDSearchModifier with optional configuration parameters.

        Note: While api_key can be passed directly as a parameter, this is intentional to support
        flexible configuration scenarios. The implementation provides multiple secure methods for
        handling sensitive data:
        1. Direct parameter (for programmatic/dynamic configuration)
        2. AIQ configuration file
        3. Environment variables (USDSEARCH_API_KEY, NVIDIA_API_KEY)
        """
        self._settings = carb.settings.get_settings()

        # Use provided host_url if available, otherwise fall back to settings
        if host_url is not None:
            self._service_url = host_url
        else:
            self._service_url = self._settings.get("/exts/omni.ai.chat_usd.bundle/usd_search_host_url")

        # Use provided username if available, otherwise fall back to get_username()
        if username is not None:
            self._username = username
        else:
            self._username = get_username()

        # Use provided api_key if available, otherwise fall back to get_api_key()
        if api_key is not None:
            self._api_key = api_key
        else:
            self._api_key = get_api_key()

        # Use provided search_path if available, otherwise fall back to settings
        if search_path is not None:
            self._search_path = search_path
        else:
            self._search_path = self._settings.get("/exts/omni.ai.chat_usd.bundle/usdsearch_search_path")

        # Store custom URL replacements
        # Expected format: dict {old_url: new_url}
        self._custom_url_replacements = url_replacements or {}

    def prepare_basic_auth(self, username: str, password: str) -> str:
        """Prepare the basic auth header."""
        user_pass = f"{username}:{password}"

        # Encode to base64
        encoded_bytes = base64.b64encode(user_pass.encode("utf-8"))
        encoded_str = encoded_bytes.decode("utf-8")

        # Create the Basic Auth header
        return f"Basic {encoded_str}"

    def _process_json_data(self, json_data):
        """Process the JSON data returned by the USD Search API."""
        for item in json_data:
            # Apply default replacements
            item["url"] = (
                item["url"]
                .replace(
                    "s3://deepsearch-demo-content/",
                    "https://omniverse-content-production.s3.us-west-2.amazonaws.com/",
                )
                .replace(
                    "s3://deepsearch-content-staging-bucket/",
                    "https://deepsearch-content-staging-bucket.s3.us-east-2.amazonaws.com/",
                )
            )

            # Apply custom replacements
            for old_url, new_url in self._custom_url_replacements.items():
                item["url"] = item["url"].replace(old_url, new_url)

            if "image" in item:
                # Create a temporary file in the system's temp directory
                with tempfile.NamedTemporaryFile(prefix="temp_", suffix=".png", delete=False) as temp_file:
                    # Decode the base64 image data and write it to the temp file
                    image_data = base64.b64decode(item["image"])
                    temp_file.write(image_data)
                    full_path = temp_file.name

                # Replace the base64 encoded image with the file path
                item["image"] = full_path

                if "bbox_dimension_x" in item:
                    item["bbox_dimension"] = [
                        item["bbox_dimension_x"],
                        item["bbox_dimension_y"],
                        item["bbox_dimension_z"],
                    ]

        clean_json_data = []
        for item in json_data:
            new_item = {}
            # Remove any other keys that we dont care about
            for key in item.keys():
                if key in ["url", "image", "bbox_dimension"]:
                    new_item[key] = item[key]

            clean_json_data.append(new_item)

        return clean_json_data

    def on_post_invoke(self, network: "RunnableNetwork", node: RunnableNode):
        output = node.outputs.content if node.outputs else ""

        # Updated regex to capture both text queries and image queries
        # This will match both:
        # @USDSearch("text query", true, 10)@
        # @USDSearch(<image(path)>, false, 10)@
        matches = re.findall(r'@USDSearch\((?:"([^"]+)"|<image\(([^)]+)\)>), (.*?), (\d+)\)@', output)

        search_results = {}

        for text_query, image_path, metadata, limit in matches:
            # Cast to proper Python types
            metadata = metadata.lower() == "true"
            limit = int(limit)

            if text_query:
                # This is a text search
                api_response = self.usd_search_post(query=text_query, return_metadata=metadata, limit=limit)
                search_results[text_query] = api_response
            elif image_path:
                # This is an image search - use the unified method with image_path parameter
                api_response = self.usd_search_post(image_path=image_path, return_metadata=metadata, limit=limit)
                search_results[f"Image: {image_path}"] = api_response

        if search_results:
            search_results_str = json.dumps(search_results, indent=2) + "\n\n"
            search_result_node = USDSearchNode()
            search_result_node.outputs = AIMessage(search_results_str)
            network.outputs = search_result_node.outputs
            network._event_callback(
                RunnableNetwork.Event.NODE_INVOKED,
                {"node": network, "network": network},
            )
            node >> search_result_node

    def usd_search_post(self, query=None, return_metadata=False, limit=10, image_path=None):
        """Call the USD Search API with either text query or image similarity search.

        Args:
            query: Text search query (used if image_path is None)
            return_metadata: Whether to include metadata in results
            limit: Maximum number of results to return
            image_path: Path to image file for similarity search (if provided, overrides text query)
        """
        # fixed parameters
        # USD File only for now
        filter = "usd*"
        # we get the images
        images = True

        url = self._service_url
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        # Set authorization header based on username availability
        if self._username:
            # Use basic auth with username and API key as password
            headers["Authorization"] = self.prepare_basic_auth(self._username, self._api_key)
        else:
            # Use bearer token auth
            headers["Authorization"] = "Bearer {}".format(self._api_key)

        # Build base payload - common fields for both search types
        payload = {
            "return_metadata": return_metadata,
            "limit": limit,
            "file_extension_include": filter,
            "return_images": images,
            "return_root_prims": False,
        }

        # Determine search type and add specific fields
        if image_path:
            # Image similarity search
            try:
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                    image_base64 = base64.b64encode(image_data).decode("utf-8")
            except Exception as e:
                carb.log_error(f"Failed to read image file {image_path}: {str(e)}")
                return {"error": f"Failed to read image file: {str(e)}"}

            # For image search, use a single base64 string (not a list)
            payload["image_similarity_search"] = image_base64
            # Optionally add description for image search if provided
            if query:
                payload["description"] = query
            search_type = "image"
        else:
            # Text search
            payload["description"] = query
            search_type = "text"

        # Add search_path to payload if it's set
        if self._search_path:
            payload["search_path"] = self._search_path

        # Check if we should print curl commands
        print_curl = self._settings.get("/exts/omni.ai.chat_usd.bundle/print_curl_commands")
        if print_curl is None:
            print_curl = os.environ.get("USDSEARCH_PRINT_CURL", "").lower() in [
                "true",
                "1",
                "yes",
            ]

        if print_curl:
            # For debugging, show appropriate message and potentially truncated data
            debug_payload = payload.copy()
            if search_type == "image":
                debug_payload["image_similarity_search"] = image_base64[:50] + "..."
                debug_message = "Windows curl command for image search (image data truncated):"
            else:
                debug_message = "Windows curl command (copy and paste):"

            payload_json = json.dumps(debug_payload)
            curl_cmd_parts = [
                f'curl -X POST "{url}"',
            ]

            for header_key, header_value in headers.items():
                curl_cmd_parts.append(f'  -H "{header_key}: {header_value}"')

            escaped_payload = payload_json.replace('"', '\\"')
            curl_cmd_parts.append(f'  -d "{escaped_payload}"')

            curl_cmd = " ^\n".join(curl_cmd_parts)

            print("\n" + "=" * 60)
            print(debug_message)
            print("=" * 60)
            print(curl_cmd)
            print("=" * 60 + "\n")

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()  # Raise an exception for non-200 status codes

            result = response.json()

            filtered_result = self._process_json_data(result)
            return filtered_result

        except requests.RequestException as e:
            error_prefix = "Image search" if search_type == "image" else "Text search"
            return {"error": f"{error_prefix} API request failed: {str(e)}"}
