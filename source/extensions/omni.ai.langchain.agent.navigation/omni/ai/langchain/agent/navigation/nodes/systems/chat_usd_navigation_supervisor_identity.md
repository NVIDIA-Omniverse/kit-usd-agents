# ChatUSD_Navigation Function

## Use Case: Scene Navigation

The ChatUSD_Navigation function enables natural language navigation within USD scenes. It allows users to navigate to Points of Interest (POIs) that can be locations or specific objects within the scene.

## When to Call ChatUSD_Navigation:

Call this function when the user request involves:
1. Navigating to a specific location or viewpoint in the scene
2. Listing available points of interest in the scene
3. Saving the current camera position as a point of interest
4. Requesting information about scene navigation capabilities
5. Exploring a digital twin facility through different viewpoints
6. Requesting to move the camera to view specific objects or areas

## When NOT to Call ChatUSD_Navigation:

1. When the exact same navigation request was just processed in the previous turn
2. When a navigation operation has already been completed successfully and the user hasn't requested a new navigation action
3. When the user is asking about the current view or location (use ChatUSD_SceneInfo instead)
4. When the user is asking for information about scene contents rather than navigation
5. When the previous call to ChatUSD_Navigation has already addressed the user's request completely

## Avoiding Repetitive Calls:

To prevent unnecessary repetitive calls to ChatUSD_Navigation:
1. Track the last navigation request and its result
2. If the current request is identical to the previous one, do not call ChatUSD_Navigation again
3. Instead, refer to the previous result or provide additional context about the current view
4. Only call ChatUSD_Navigation again if the user explicitly requests a different navigation action
5. If a navigation request fails, provide feedback to the user rather than repeatedly trying the same operation

## Examples of Requests for ChatUSD_Navigation:

- "Show me all the available viewpoints in this facility"
- "Take me to the kitchen area"
- "Navigate to the main entrance"
- "Save this view as 'production line overview'"
- "I want to see the building from the south entrance"
- "Move the camera to focus on the assembly station"
- "What points of interest are available in this scene?"
- "Can you save my current position as 'maintenance access point'?"
- "Give me a tour of the facility"

## Integration with Other Functions:

- For scene information queries, use ChatUSD_SceneInfo first
- For USD code generation, use ChatUSD_USDCodeInteractive
- For asset searches, use ChatUSD_USDSearch
- For navigation operations, use ChatUSD_Navigation

## Limitations and Functional Boundaries:

- ChatUSD_Navigation is NOT able to create cameras or modify the scene structure
- It stores navigation data only in custom metadata in the root layer
- It cannot access or modify other aspects of the scene beyond navigation
- To create a camera or any prim in the scene, use ChatUSD_USDCodeInteractive instead
- ChatUSD_Navigation should be used exclusively for scene navigation operations within existing content
