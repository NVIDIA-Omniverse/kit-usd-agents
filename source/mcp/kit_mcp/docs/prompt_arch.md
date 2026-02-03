

Okay, so we are going to talk about the OmniUI MCP.

So it's going to be a developer MCP, so this one is going to be very focused on developer workflow.
There are a few things that are key that we're going to do:

1.  It's a hierarchical data structure. So first, there is an overview set of instructions that may be part of Cursor rules or agentic workflow. Effectively, there is an initial, kind of large system prompt about what is OmniUI and how to use it.

2.  Then it's a hierarchical set of tools in order to retrieve data and information about it into a scaffolding way.

1) there is a tool for search extension. This tool will enable you to find extensions about certain topics,
so it will use some rags and some good prompting techniques in order to search into the almost 400 to 450 extensions that we have.
And then it will sort of return to the caller, to the agent, the list of extensions that are related and relevant for the task that the user is asking.

2) a tool will be that once I have extensions, I might want to go a bit deeper into what actually do they do. So I will be able to pass these extensions to another MCP tool that will collect detailed information. So then I take all of these extension IDs and pass them to the extension detail MCP that will then collect some shortened documentation about key features, key objects, key objectives, how to use it, and this type of things. And I will be able to retrieve that and get a general sense of what that does.

3) Then there will be one tools for extension APIs. You would pass an extension ID to this extension API, or you will pass a list of extension IDs, and it will return you the equivalent of this python_api.md file which is the short version of the API listing all the APIs in that case they don't have any definitions, they are pretty compact, but you get to see what they're doing. Then you are able to ask for API documentation

4) so given an extension and some API (either a method or a class), you will be able to ask for clarifications on those APIs, giving you a more outlook of every method with their docstrings, so now you can get a real sense of the API.

5) Another set of MCPs will be to get code examples. The tool will enable you to get code examples about certain topics, and then you'll be able to ask for something and it will return you some files or some extensions that are implementing the type of methods or workflows that you're trying to go after. So get code examples will be a pretty clear one that would be really useful.

6) Finally, when you're doing this code, you might have to validate that it's working, so you'll be able to connect to a running kit and one of the methods that you would be able to ask the running kit is to ask for the current logs. Where that'd be really useful is because based on this current log, you'll be able to figure out if there was some error, some warning, or if you're coding and you're trying to debug something, you can do some printouts and then you can look into the log and see when it executed what happens.

7) Another really useful method is when you're doing a UI or you're building some widgets that you need to be able to show on screen. You'd be able to validate that by asking kit to give you the image of a certain window. And so there is an API that will be called Get Window Screenshot that will return you the visuals of that current window in its current state at that moment. So I think those are the scaffolded APIs that we will all be able to get access to.

8) Another useful tool will be to be able to grab extension dependency. We have an API that enables you to search extensions, find out who are their owners, who created them, on which Git repository they are in, and give you further details about them. That would also be something that would be really useful. That would be mostly for internal development but yet extremely valuable.

9) There needs to be an MCP that will give some details and instructions on how to write tests and what's the structure, how to run them and use them, so this way the AI can discover best practices about Omniverse KIT tests.
I suspect that these MCP sections or maybe some other type of command will also give some test examples. In case you need to find some inspiration to write some tests or some test APIs and this value things.

Based on this I have defined the following API

# Kit Docs MCP

## Documentation

- get_kit_instructions(systems: [str]) - gather System Instruction about Kit
    - General Kit
    - Extensions
    - Testing

## Extension Discovery

- search_kit_extensions(request: str) - Find relevant extensions by topic/functionality
- get_kit_extension_details(ext_ids: [str]) - Get detailed information about extensions
- get_kit_extension_dependencies(ext_ids: str) - Analyze extension dependency graphs


## Extensions API Reference

- get_kit_extension_apis(ext_ids: [str]) - List APIs for each extensions return structured
- get_extension_full_api(ext_id: str) - Get detailed API documentation with docstrings
- get_kit_api_details(apis=[str]) - Get class or methods docstring format "<ext_id>@symbol"

## Code & Examples

- search_kit_code_examples(request:str) - Search for relevant code examples
- search_kit_test_examples(request:str) - Find test implementations


# Kit Runtime MCP Server Tools

# Debugging & Monitoring

- get_kit_logs(time:timestand) - Retrieve current Kit logs since a certain time

# Visual Validation

- capture_window_screenshot(window_name:str) -> image - Capture window screenshot
- capture_viewport_screenshot() -> image - Capture image of the viewport (will need some index to support multiple start with only this API )
- capture_application_screenshot() -> image - capture an image fo the full applications


# Runtime Inspection

- execute_code(code: str) - Execute some kit code , will always require approval for security reason and MCP will have that feature off by default