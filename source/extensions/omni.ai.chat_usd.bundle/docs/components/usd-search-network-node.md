# USDSearchNetworkNode

The `USDSearchNetworkNode` is a specialized agent in the Chat USD system that handles USD asset search. It extends the `NetworkNode` class from the LC Agent framework and is responsible for searching for USD assets based on natural language queries.

## Overview

The `USDSearchNetworkNode` serves as the search engine of the Chat USD system. It can interpret natural language search queries, search for relevant USD assets, and present the results to the user. This enables users to find and import USD assets through natural language interaction.

## Implementation

The `USDSearchNetworkNode` is implemented as a Python class that extends `NetworkNode`:

```python
class USDSearchNetworkNode(NetworkNode):
    """
    Use this node to search any asset in Deep Search. It can search, to import call another tool after this one.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Add the USDSearchModifier to the network
        self.add_modifier(USDSearchModifier())

        # Set the default node to USDSearchNode
        self.default_node = "USDSearchNode"

        self.metadata[
            "description"
        ] = """Agent to search and Import Assets.
Connect to the USD Search NIM to find USD assets base on the natural language query.
Drag and drop discovered assets directly into your scene for seamless integration"""

        self.metadata["examples"] = [
            "What can you do?",
            "Find 3 traffic cones and 2 Boxes",
            "I need 3 office chairs",
            "10 warehouse shelves",
        ]
```

The class initializes itself with a `USDSearchModifier` to extend its functionality, sets the default node to `USDSearchNode`, and sets metadata for the node.

## USDSearchNode

The `USDSearchNode` is the default node for the `USDSearchNetworkNode`. It extends `RunnableNode` and is responsible for generating search queries based on natural language descriptions:

```python
class USDSearchNode(RunnableNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.inputs.append(RunnableSystemAppend(system_message=USD_SEARCH_SYSTEM))
```

The node is initialized with a system message (`USD_SEARCH_SYSTEM`) that provides instructions on how to generate search queries.

## System Message

The system message (`USD_SEARCH_SYSTEM`) is a critical component of the `USDSearchNode`. It provides detailed instructions on how to generate search queries:

```python
USD_SEARCH_SYSTEM = """You are an AI assistant specialized in generating queries for the USDSearch API.
Your task is to interpret user requests and generate appropriate queries for searching USD-related information.
The query should be concise and relevant to the user's request.

@USDSearch("<search terms>", True|False, int)@

For example:
to search Box with metadata and limit 10 results:
@USDSearch("Box", True, 10)@

or

to search Small chair without metadata and limit 10 results:
@USDSearch("Small chair", False, 10)@

or

to search blue table with metadata and limit 3 results:
@USDSearch("blue table", True, 3)@

if you don't know how many results to return, you never omit the limit parameter but use 10 as default
also for meta always use False as default, only set to True if you where asked to do so

you never do
@USDSearch("Box", True)@
or
@USDSearch("Crate")@

when you get ask for multiple type of things make sure to break the query fully like
Questions:
I need to build some shelving with security railing around them also might need few cones
Answer:
@USDSearch("shelving", False, 10)@
@USDSearch("security railing", False, 10)@
@USDSearch("cones", False, 10)@

Always use the full command with all parameters
"""
```

This system message instructs the node on how to generate search queries in the correct format, with examples and guidelines for different types of queries.

## USDSearchModifier

The `USDSearchModifier` is a key component of the `USDSearchNetworkNode`. It extends `NetworkModifier` and is responsible for intercepting search queries, calling the USD Search API, and processing the results:

```python
class USDSearchModifier(NetworkModifier):
    """USDSearch API Command:
    @USDSearch(query: str, metadata: bool, limit: int)@

    Description: Searches the USD API with the given query and parameters.
    - query: The search query string
    - metadata: Whether to include metadata in the search results (true/false)
    - limit: The maximum number of results to return

    Example: @USDSearch("big box", false, 10)@"""

    def __init__(self):
        self._settings = carb.settings.get_settings()
        self._service_url = self._settings.get("exts/omni.ai.chat_usd.bundle/usd_search_host_url")
        self._api_key = get_api_key()

    def on_post_invoke(self, network: "RunnableNetwork", node: RunnableNode):
        output = node.outputs.content if node.outputs else ""
        matches = re.findall(r'@USDSearch\("(.*?)", (.*?), (\d+)\)@', output)

        search_results = {}
        for query, metadata, limit in matches:
            # Cast to proper Python types
            metadata = metadata.lower() == "true"
            limit = int(limit)

            # Call the actual USD Search API
            api_response = self.usd_search_post(query, metadata, limit)
            search_results[query] = api_response

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
```

The modifier initializes itself with the service URL and API key, and implements the `on_post_invoke` method to intercept search queries, call the USD Search API, and process the results.

## Search Process

The search process in `USDSearchNetworkNode` follows these steps:

1. **Query Generation**: The `USDSearchNode` generates a search query based on the user's natural language description
2. **Query Interception**: The `USDSearchModifier` intercepts the search query
3. **API Call**: The modifier calls the USD Search API with the query
4. **Result Processing**: The modifier processes the API response
5. **Result Presentation**: The results are presented to the user

This process enables users to search for USD assets using natural language queries.

## API Call

The `USDSearchModifier` calls the USD Search API using the `usd_search_post` method:

```python
def usd_search_post(self, query, return_metadata, limit):
    """Call the USD Search API with the given query and parameters."""
    # fixed parameters
    # USD File only for now
    filter = "usd*"
    # we get the images
    images = True

    url = self._service_url
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(self._api_key),
    }

    payload = {
        "description": query,
        "return_metadata": return_metadata,
        "limit": limit,
        "file_extension_include": filter,
        "return_images": images,
        "return_root_prims": False,
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for non-200 status codes

        result = response.json()

        filtered_result = self._process_json_data(result)
        return filtered_result

    except requests.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
```

This method sends a POST request to the USD Search API with the query, metadata flag, and limit, and returns the processed results.

## Result Processing

The `USDSearchModifier` processes the API response using the `_process_json_data` method:

```python
def _process_json_data(self, json_data):
    """Process the JSON data returned by the USD Search API."""
    for item in json_data:
        item["url"] = item["url"].replace(
            "s3://deepsearch-demo-content/", "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
        )

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
```

This method processes the JSON data returned by the API, converting URLs, decoding images, and cleaning up the data for presentation to the user.

## UI Integration

The `USDSearchNetworkNode` integrates with the Omniverse Kit UI through the `ChatView` class, which provides a chat interface for interacting with Chat USD:

```python
try:
    from omni.ai.langchain.widget.core import ChatView

    from .search.usd_search_delegate import USDSearchImageDelegate

    ChatView.add_delegate("USDSearchNode", USDSearchImageDelegate())
    ChatView.add_delegate("USDSearchNetworkNode", USDSearchImageDelegate())
except ImportError:
    # this extension is not available in the current environment
    # print("ChatView not available")
    pass
```

This integration allows users to interact with the search functionality through a user-friendly chat interface.
