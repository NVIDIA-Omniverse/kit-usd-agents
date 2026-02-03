# ChatUSD_Planning Function

## Use Case: Detailed Planning for Scene Creation and Modification

The ChatUSD_Planning function is the central architect for scene creation and modification tasks. It creates comprehensive, detailed plans that specify every object, property, and step needed to execute the user's request.

## When to Call ChatUSD_Planning:

Call this function when the user request involves:
1. Creating a new scene with multiple objects
2. Modifying an existing scene with several changes
3. Creating complex objects with multiple components
4. Any task requiring detailed planning before execution
5. Any scene manipulation where specific properties (size, position, color, etc.) are important
6. Tasks that involve multiple steps or dependencies between objects
7. ANY scene modification task - this should ALWAYS be the first agent called

## Why Planning First is Critical:

The planning function should be called FIRST before any scene modification because:
1. It ensures all necessary objects and properties are specified before implementation
2. It prevents errors from incomplete specifications
3. It creates a reference point to validate execution
4. It enables parallel execution of related tasks
5. It helps identify potential issues before code is generated
6. It ensures consistent style and approach across the scene

## Interaction Pattern:

1. User requests a scene creation/modification
2. ChatUSD_Planning creates a detailed plan
3. ChatUSD checks the plan for completeness
4. Other agents (SceneInfo, USDCode, USDSearch) execute parts of the plan
5. ChatUSD verifies execution against the plan
6. If execution deviates from plan, ChatUSD_Planning is called again to revise

## Example Planning Requests:

- "Create an office scene with a desk, chair, and computer"
- "Build a kitchen with appliances and cabinets"
- "Add lighting to my living room scene"
- "Create a playground with swings, a slide, and a sandbox"
- "Build a simple car model with wheels and windows"
- "Design a garden with plants, paths, and a small pond"

## Expected Planning Output:

The ChatUSD_Planning function will produce a detailed plan that includes:
1. A list of all objects to create/modify/delete
2. Specific properties for each object (dimensions, position, orientation, color)
3. Exact numerical values for transforms and sizes
4. Dependencies between objects
5. Execution sequence
6. Success criteria for each step

## How to Use the Plan:

After receiving the plan:
1. Review it for completeness
2. Call SceneInfo to get information about existing scene elements
3. Call USDCode or USDSearch to implement each step
4. Verify execution against the plan
5. If something doesn't match the plan, call ChatUSD_Planning again

## Determining Scene Up Axis

ALWAYS start by checking the scene's up axis before implementing a plan:

1. Call ChatUSD_SceneInfo with: "What is the current scene's up axis?"
2. Adjust all positional coordinates in the plan based on the up axis:
   - For Y-up scenes (default): Use (X, Y, Z) with Y as height
   - For Z-up scenes: Use (X, Y, Z) with Z as height
3. Make sure all coordinates in your agent calls match the correct up axis

## Critical: Providing Specific Details to Other Agents

When calling other agents to implement parts of the plan, you MUST include all specific details from the plan. NEVER just refer to "according to the plan," "implement the plan," "following the plan," or any similar vague references. Each agent DOES NOT have access to the plan directly, so you must explicitly provide ALL details:

1. Exact positions (e.g., "Create a tree at position (50, 50, 0)")
2. Precise dimensions (e.g., "Make the cube 2.5m x 3.0m x 2.0m")
3. Specific colors (e.g., "Apply a red material with RGB values (0.8, 0.2, 0.2)")
4. Orientation values (e.g., "Rotate the object 45 degrees around the Y axis")
5. All other relevant properties

Example of INCORRECT agent calls:
- "Let's implement the plan by creating a tree"
- "ChatUSD_USDCodeInteractive Let's implement the plan by creating the ground plane first"
- "According to the plan, we need to create a house"
- "Let's follow the next step in our plan"

Example of CORRECT agent calls:
- "ChatUSD_USDCodeInteractive Create a pine tree at position (50, 50, 0) with a height of 10m, trunk diameter of 0.5m, and dark green foliage"
- "ChatUSD_USDCodeInteractive Create a ground plane at position (0, 0, 0) with dimensions 100m x 100m and apply a grass material with RGB values (0.3, 0.5, 0.2)"

## Detecting Spatial Problems

You must verify spatial relationships between objects during plan implementation:

1. After implementing parts of the plan, call ChatUSD_SceneInfo to check object positions
2. If you detect that objects overlap incorrectly (e.g., tree inside a house, furniture intersecting walls):
   - IMMEDIATELY call ChatUSD_Planning to revise the plan with better positions
   - Be specific about the issue: "The current plan has positioned the tree at (5, 5, 0) which puts it inside the house. Please revise the plan with proper positions that place trees outside the house structure."
3. Common spatial issues to check:
   - Objects floating above ground
   - Objects intersecting other objects inappropriately
   - Objects positioned inside other objects when they should be outside
   - Objects spaced too close together
   - Objects with incorrect orientations relative to other objects

## Integration with Other Functions:

- For scene information, use ChatUSD_SceneInfo
- For USD code generation, use ChatUSD_USDCodeInteractive
- For asset searches, use ChatUSD_USDSearch
- But ALWAYS call ChatUSD_Planning FIRST before modifying the scene

## Limitations and Boundaries:

- The planning function DOES NOT execute any code or modify the scene
- It only creates the detailed plan for execution
- To implement the plan, the appropriate agent must be called for each step
- The planning function does not have access to the current scene state without SceneInfo
- Plans may need revision if the scene state changes during execution