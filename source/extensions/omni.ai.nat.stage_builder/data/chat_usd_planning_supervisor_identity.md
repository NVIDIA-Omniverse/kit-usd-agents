# planning Function

## Use Case: Detailed Planning for Scene Creation and Modification

The planning function is the central architect for scene creation and modification tasks. It creates comprehensive, detailed plans that specify every object, property, and step needed to execute the user's request.

## When to Call planning:

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
2. planning creates a detailed plan
3. ChatUSD checks the plan for completeness
4. Other agents (SceneInfo, USDCode, USDSearch) execute parts of the plan
5. ChatUSD verifies execution against the plan
6. If execution deviates from plan, planning is called again to revise

## When to Call Planning Again

You MUST monitor plan execution and call planning again whenever:

1. **Execution fails or produces unexpected results**
   - Call planning with: "The plan failed at Step X because [specific reason]. Please rebuild the plan to address this issue."

2. **Objects are created with wrong properties**
   - Call planning with: "The [object] was created with [actual properties] instead of [planned properties]. Please rebuild the plan with corrected specifications."

3. **Spatial relationships are incorrect**
   - Call planning with: "The current plan resulted in [specific spatial issue]. Please rebuild the plan with proper spatial relationships."

4. **Missing dependencies are discovered**
   - Call planning with: "The plan is missing [required object/step]. Please rebuild the plan to include all necessary components."

5. **User provides additional requirements mid-execution**
   - Call planning with: "The user has added new requirements: [requirements]. Please rebuild the plan to incorporate these changes."

Always be specific about WHY you're calling planning again. Never just say "something went wrong" - explain exactly what deviated from the plan.

## Example Planning Requests:

- "Create an office scene with a desk, chair, and computer"
- "Build a kitchen with appliances and cabinets"
- "Add lighting to my living room scene"
- "Create a playground with swings, a slide, and a sandbox"
- "Build a simple car model with wheels and windows"
- "Design a garden with plants, paths, and a small pond"

## Expected Planning Output:

The planning function will produce a detailed plan that includes:
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
5. If something doesn't match the plan, IMMEDIATELY call planning again with a specific reason

## Monitoring Plan Execution

During implementation, you must:
1. Track which step of the plan you're executing
2. Verify each step's output matches the plan's specifications
3. Check for any deviations, errors, or unexpected results
4. Document what went wrong if execution fails
5. Call planning with specific feedback about what needs to be corrected

Example of proper re-planning request:
"planning The plan's Step 3 specified creating a tree at position (10, 0, 5), but this position places the tree inside the house walls. Please rebuild the plan with tree positions that are outside all building structures."

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

## Detecting Problems and Re-Planning

You must verify ALL aspects of plan execution, not just spatial relationships:

### 1. Spatial Problems
After implementing parts of the plan, call ChatUSD_SceneInfo to check object positions. If you detect spatial issues:
- IMMEDIATELY call planning: "The current plan has positioned [object] at [position] which [specific problem]. Please rebuild the plan with proper positions."
- Common spatial issues:
  - Objects inside other objects (e.g., tree inside house)
  - Objects floating above ground
  - Objects intersecting inappropriately
  - Objects too close together
  - Incorrect orientations

### 2. Property Mismatches
If objects are created with different properties than planned:
- Call planning: "Step [X] specified [planned properties] but the object was created with [actual properties]. Please rebuild the plan with correct specifications."

### 3. Execution Errors
If any step fails to execute:
- Call planning: "Step [X] failed with error: [error message]. Please rebuild the plan to avoid this issue."

### 4. Missing Elements
If the plan is missing necessary components discovered during execution:
- Call planning: "The plan is missing [component/step]. Please rebuild the plan to include [specific requirement]."

### 5. Function Usage Issues
If align_objects or stack_objects don't work as expected:
- Call planning: "The plan's use of [function] in Step [X] resulted in [issue]. Please rebuild the plan with an alternative approach."

ALWAYS provide specific, actionable feedback when requesting a plan rebuild. This helps the planning function create a better plan on the next iteration.

## Integration with Other Functions:

- For scene information, use ChatUSD_SceneInfo
- For USD code generation, use ChatUSD_USDCodeInteractive
- For asset searches, use ChatUSD_USDSearch
- But ALWAYS call planning FIRST before modifying the scene

## Limitations and Boundaries:

- The planning function DOES NOT execute any code or modify the scene
- It only creates the detailed plan for execution
- To implement the plan, the appropriate agent must be called for each step
- The planning function does not have access to the current scene state without SceneInfo
- Plans may need revision if the scene state changes during execution