# Templates

We have many templates and template generator tools to help you get started with your projects.

* You can start out by simply modifying one of our examples to suit your needs: [Custom Interactive Examples](Templates.md)
* You can use the Extension Template Generator to create a new extension projects: [Extension Template Generator](Templates.md). These templates are structured to utilize Isaac Sim libraries and built with robotics applications in mind.
* For more generic extension templates, use [Custom Extensions: C++](Templates.md).
* For extension using any combinations of C++, Python, OmniGraph, GUI elements, and more, refer to the [Advanced Extension Template Generator from VS Code](Templates.md).

These are all for Extension-based projects. For standalone projects, simply browse through our Standalone Examples folder (`PATH_TO_ISAAC_SIM/standalone_examples`), and use them as a starting point.

---

# Custom Interactive Examples

You can create custom examples in NVIDIA Isaac Sim Examples Browser, so that your examples are accessible in the same browser as rest of the examples.

## BaseSampleUITemplate & BaseSample Classes

The BaseSampleUITemplate and BaseSample classes provide the basic structure for creating an interactive examples that looks similar to our other examples in the Examples Browser. It produces a Load button and a Reset button, each button abstracts away the complexity of asynchronously interacting with the simulator and making the interactiveness work.

To create your own, follow the steps below:

1. Copy the current files to the `user_examples` folder under `isaacsim/examples/interactive`.

   > ```python
   > cd exts/isaacsim.examples.interactive/isaacsim/examples/interactive
   > cp hello_world/hello_world* user_examples/
   > ```
2. Edit the highlighted lines in `exts/isaacsim.examples.interactive/isaacsim/examples/interactive/user_examples/hello_world_extension.py`:

   > ```python
   > import os
   >
   > import omni.ext
   > from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
   > from isaacsim.examples.interactive.user_examples import HelloWorld
   >
   >
   > class HelloWorldExtension(omni.ext.IExt):
   >     def on_startup(self, ext_id: str):
   >         self.example_name = "Awesome Example"
   >         self.category = "MyExamples"
   >
   >         ui_kwargs = {
   >             "ext_id": ext_id,
   >             "file_path": os.path.abspath(__file__),
   >             "title": "My Awesome Example",
   >             "doc_link": "https://docs.omniverse.nvidia.com/isaacsim/latest/core_api_tutorials/tutorial_core_hello_world.html",
   >             "overview": "This Example introduces the user on how to do cool stuff with Isaac Sim through scripting in asynchronous mode.",
   >             "sample": HelloWorld(),
   >         }
   >
   >         ui_handle = BaseSampleUITemplate(**ui_kwargs)
   >
   >         # register the example with examples browser
   >         get_browser_instance().register_example(
   >             name=self.example_name,
   >             execute_entrypoint=ui_handle.build_window,
   >             ui_hook=ui_handle.build_ui,
   >             category=self.category,
   >         )
   >
   >     return
   > ```
3. Add the following lines to `exts/isaacsim.examples.interactive/isaacsim/examples/interactive/user_examples/__init__.py`.

   > ```python
   > from isaacsim.examples.interactive.user_examples.hello_world import HelloWorld
   > from isaacsim.examples.interactive.user_examples.hello_world_extension import HelloWorldExtension
   > ```

Note

Every time the code is edited or changed, Press **Ctrl+S** to save the code and hot-reload NVIDIA Isaac Sim.

If you want to add more complexity and more buttons, feel free to browse through the other Examples. You can always access the underlying script by clicking on the folder icon in the upper right hand corner of the Example Browser.

---

# Extension Template Generator

The Extension Template Generator populate a UI-based extensions on your local machine. The available extension templates give a useful starting point for many Isaac Sim
applications and are structured to help you learn how to build a custom UI tool that meets your needs.

## Getting Started

To create and enable a new extension using the Extension Template Generator, follow these steps:

1. Open the extension generator by going to **Utilities > Generate Extension Templates** in the menu bar.
2. Select the [types of templates](#isaac-sim-template-generator-options) to expand the corresponding window. Fill in the fields as follow:

   > * Extension Path: `<Extension_Host_Dir>/my.extension.name`
   > * Extension Name: `my.extension.name`
   > * Extension Description: `My Extension Description`
3. Click Generate Extension.
4. Navigate to **Window > Extensions** in the toolbar to open the Extensions Manager. Click the hamburger icon to the right of search bar, and then *Settings* in the sub-menu to open up the path table. If your selected `<Extension_Host_Dir>` is not already on the list, then scroll all the way down to the end in the “Extension Search Path”. Click on the “+” button in the last row in the “edit” column, and type in the full path to the `<Extension_Host_Dir>`.
5. Search for your new extension.

   > * If your chosen `<Extension_Host_Dir>` was one of the default Extension Search Paths, you should find your extension under **NVIDIA** tab.
   > * If you added a new Extension Search Path, you should find your extension under the **Third Party** tab.
6. Enable the extension. Verify that it appears in the menu bar on the top in Isaac Sim.
7. Alternatively, you can enable extensions by command-line arguments when running Isaac Sim from the terminal: `./isaac-sim.sh --ext-folder {path_to_user_ext_folder} --enable {ext_directory_name}`. On Windows use `python.bat` instead of `python.sh`.
8. Get familiar with the template code by reading the `README.md` file in the provided Python module.

## Template Options

* **Load Scenario Template**: The [Loaded Scenario Template](Templates.md) starts the user off with a simple UI that contains three buttons: *Load*, *Reset*, and *Run*. This is meant to provide as clear a pathway as possible for the user to start writing code to directly affect the USD stage without having to understand much about the internal workings of the underlying simulator.
* **Scripting Template**: The [Scripting Template](Templates.md) demonstrates the implementation of a more advanced framework for programming script-like behavior from a UI-based extension in NVIDIA Isaac Sim. This template uses the same mechanics for loading and resetting the robot position as the “Load Scenario Template”, but it implements the *Run* button as a script.
* **Configuration Tooling Template**: The [Configuration Tooling Template](Templates.md) templates provides fundamental tools for asset configuration, such as finding `Articulation` on the stage and dynamically creates a UI frame through which the user may control each joint in the selected `Articulation`.
* **UI Component Library Template**: The [UI Component Library](Templates.md) template demonstrate the usage of each `UIElementWrapper`, such as the type of arguments and return values required for each callback function that can be attached to each `UIElementWrapper`.

## More Resources

For more detailed explanation regarding the template generator and each template can be found in [Extension Template Generator Explained](Templates.md).

---

# Extension Template Generator Explained

## General Concepts

Each template provided by the *Extension Template Generator* has a common underlying structure with a thin layer of implementation on top.
In each template root directory, there is a folder called `./scripts` where all Python code supporting the extension is stored. Inside
`./scripts`, there are three common Python files:

* global\_variables.py
  :   A script that stores the global variables that the user specified when creating their extension in the *Extension Template Generator*
      such as the Title and Description.
* extension.py
  :   A class containing the standard boilerplate necessary to have the user extension show up on the Toolbar. This
      class is meant to fulfill most use-cases without modification.
      In extension.py, useful standard callback functions are created that the user may complete in ui\_builder.py.
* ui\_builder.py
  :   This file is the user’s main entrypoint into the template. Here, the user can see useful callback functions that have been
      set up for them, and they may also create UI elements that are hooked up to user-defined callback functions. This file is
      the most thoroughly documented, and the user should read through it before making serious modification.

A typical user will only need to modify `./scripts/ui_builder.py` to get their extension working the way they want. Inside `./scripts/ui_builder.py`, the user
will find a set of standard callback functions that connect them to the simulator:

* on\_menu\_callback(): Called when extension is opened
* on\_timeline\_event(): Called when timeline is stopped, paused, or played
* on\_physics\_step(): Called on every physics step. Physics steps only happen while the timeline is playing.
* on\_stage\_event(): Called when stage is opened or closed
* cleanup(): Called when resources such as physics subscriptions should be cleaned up because the extension is being closed
* build\_ui(): User function that creates the UI they want.

In the provided extension templates, most of the implementation is in the `build_ui()` function. The extension templates utilize a set of wrapper classes around
`omni.ui` elements that allow the user to easily create and manage a variety of UI elements. These are referred to in this tutorial as `UIElementWrappers`. Each wrapper is meant to provide the
user with the most common-sense way of interacting with a UI element. For example, the user can create a `FloatField` UI element; any time the user modifies the `FloatField` in the UI,
a user callback function will be called with the new `float` value passed in.

Each extension template builds a UI with a set of governing callback functions in `build_ui()`. These callback functions contain all of the logic to make the UI run smoothly and
make it easy to connect user code for a custom application.

## Loaded Scenario Template

The *Loaded Scenario Template* starts the user off with a simple UI that contains three buttons: *Load*, *Reset*, and *Run*. This is meant to provide
as clear a pathway as possible for the user to start writing code to directly affect the USD stage without having to understand much about the
internal workings of the underlying simulator. There user only needs to know the following simple concepts.

### Important Concepts

In Omniverse Kit Applications, there is a simulation timeline that can be directly stopped, paused, and played on the left-hand side toolbar. Physics
is only running while the timeline is active (not stopped). As such, the user cannot control a robot `Articulation` while the timeline is stopped,
and initialization needs to be performed on certain assets such as an `Articulation` when the timeline goes from stopped to playing. The purpose of the
*Loaded Scenario Template* is to make it easier for the user to interact with the simulator without having to handle things like initialization.

In `isaacsim.core.api.world` there is a singleton class `World` that is designed to set up and properly manage the simulation with simple and clear
user-interaction. In this template, the `World` is managed by the *Load* and *Reset* buttons, leaving the user with clear guarantees about the
state of the simulator at the time that their callback functions are called. The user interaction with the `World` is minimized to the point that
they their only interaction with the `World` takes the form `world.scene.add(user_object)` where `user_object` is any object from `isaacsim.core.api`.

To ensure proper functionality, all manipulation of the timeline should be done by the *Load* and *Reset* buttons. I.e. the user is able to cause trouble
by pressing the *Stop* and *Play* buttons on the left-hand toolbar outside of this UI. For this reason, the template directly handles the cases where
the user messes with the timeline outside of the template UI by resetting the UI when necessary to maintain assumptions on user callback functions.

### Implementation Details

The *Load* button has two callback functions:

* def setup\_scene\_fn():
  :   On pressing the *Load* button, a new instance of `World` is created and then this function is called.
      The user should now load their assets onto the stage and add them to the `World` with `world.scene.add()`.
* def setup\_post\_load\_fn():
  :   The user may assume that their assets have been loaded by their setup\_scene\_fn callback, that
      their objects are properly initialized, and that the timeline is paused on timestep 0.

The *Reset* button has two callback functions:

* pre\_reset\_fn():
  :   This function is called before the `World` is reset, so there are no guarantees on the state of the simulator.
* post\_reset\_fn():
  :   The user may assume that their objects are properly initialized, and that the timeline is paused on timestep 0.

      They may also assume that objects that were added to the `World` have been moved to their default positions.
      I.e. a cube prim will move back to the position it was in when it was created in setup\_scene\_fn().

The *Run* button is not connected to the `World`. It is a `StateButton`, which means that it will switch between two states: *Run* and *Stop*.
A `StateButton` can have three callback functions:

* on\_a\_click():
  :   Function called when the `StateButton` is showing its a\_text
* on\_b\_click():
  :   Function called when the `StateButton` is showing its b\_text
* physics\_callback\_fn():
  :   If specified, the `StateButton` will call this function on every physics step while the state button is in its B state, and
      it will cancel the physics subscription whenever the state button is in its A state.

Note

You can see how these functions are called in the `UIBuilder` class (`template_source_files/loaded_scenario_workflow/ui_builder.py` file in the `isaacsim.examples.extension` extension).

To try it, open the Template Generator (*Utilities > Generate Extension Templates* menu) and create a new extension under the *Loaded Scenario Template* section.
Then, enable the extension (*Window > Extensions* menu, search for the given extension name) and click on the toolbar entry with the same name.

## Scripting Template

The *Scripting Template* is a natural extension of the *Loaded Scenario Template* that demonstrates the
implementation of a more advanced framework for programming script-like behavior from a UI-based
extension in NVIDIA Isaac Sim. This template uses the same mechanics for loading and resetting the robot
position, but it implements the *Run* button as a script.

Using the pattern demonstrated in this template, the user can program script-like behavior by implementing
long-running functions that check in on every physics step to send a new command or determine that it is
time to return. The *Scripting Template* contains an implementation of the functions `goto_position()`,
`open_gripper_franka()` and `close_gripper_franka()`. These functions are used in series in order to
script the simple pick-and-place task shown below.

### Implementation Details

The implementation details of the UI match the *Loaded Scenario Template*, and so this section focuses
on the implementation of script-like behavior. Long-running functions that check in on every frame
can be written using Python’s yield/generator framework. A function `my_script()` is implemented in
the file `scenario.py` that contains the sequence of `goto_position()`, `open_gripper_franka()`, and
`close_gripper_franka()` function calls. The `my_script()` function makes use of `yield` and `yield from` statements.
This allows `my_script()` to be wrapped in a generator with `self._script_generator = self.my_script()`.
Then, on every physics step, `next(self._script_generator)` is called to step the generator and
execute code until the next `yield` statement is encountered (in either `my_script()` or a nested function).

Take the function `open_gripper_franka()` as an example:

```python
def open_gripper_franka(self, articulation):
    open_gripper_action = ArticulationAction(np.array([0.04, 0.04]), joint_indices=np.array([7, 8]))
    articulation.apply_action(open_gripper_action)

    # Check in once a frame until the gripper has been successfully opened.
    while not np.allclose(articulation.get_joint_positions()[7:], np.array([0.04, 0.04]), atol=0.001):
        yield ()

    return True
```

`my_script()` calls `yield from open_gripper_franka()`. The function `open_gripper_franka()` sends
a single command to the Franka `Articulation` that the grippers should open, and then on every subsequent
physics step, it checks if the gripper has made it to the target position. Once the gripper has reached
the target position, the function stops calling `yield` and instead calls `return True` to signal a success.
The control flow goes back to `my_script()` and the next function in the sequence gets called.

To try it, open the Template Generator (*Utilities > Generate Extension Templates* menu) and create a new extension under the *Scripting Template* section.
Then, enable the extension (*Window > Extensions* menu, search for the given extension name) and click on the toolbar entry with the same name.

## Configuration Tooling Template

The *Configuration Tooling Template* provides a simple template that serves as a solid foundation for building tools for asset configuration.
The provided implementation creates a drop-down menu that finds any `Articulation` on the stage and dynamically creates a UI frame through
which the user may control each joint in the selected `Articulation`.

Unlike the *Loaded Scenario Template* this extension assumes no control over the timeline or the stage. Instead, it allows the user to select
whatever is there and start reading and writing its state. Building asset configuration tools is a more advanced use-case, and as such,
it requires a better internal model of the Simulation timeline. For example, because an `Articulation` is only accessible while the timeline
is playing, the provided template only allows the user to attempt to modify their selected `Articulation` while the timeline is playing.

### Implementation Details

The `DropDown` is populated by a function that searches the USD stage for all objects of the specified type. This is provided as a convenience function
directly in the `DropDown` UI wrapper, but a version of the function it is using is left at the bottom of the template to allow the user further
customization.

Whenever a new item is selected from the `DropDown`, the *Robot Control Frame* is rebuilt using a builder function. This is a powerful paradigm for creating robust dynamic UI tools.
In this template, the frame can either report to the user that no robot could be selected, or it can list every joint in the selected robot if everything went well.

To try it, open the Template Generator (*Utilities > Generate Extension Templates* menu) and create a new extension under the *Configuration Tooling Template* section.
Then, enable the extension (*Window > Extensions* menu, search for the given extension name) and click on the toolbar entry with the same name.
Finally, in a new stage (*File > New* menu), add the Franka robot (*Create > Robots > Franka Emika Panda Arm* menu) and play with it.

## UI Component Library

The *UI Component Library* template demonstrates the usage of each `UIElementWrapper` that has been created. This should be used as a reference when
setting up a custom UI tool. Most importantly, this template shows the specific type of arguments and return values required for each callback function that can be
attached to each `UIElementWrapper`. This template omits the *Load* and *Reset* buttons, as these are special case buttons that are demonstrated in
the *Loaded Scenario Template*. None of the UI components shown in this template directly impact the simulation; they only call user callback functions.

The components in the *UI Component Library* template wrap a subset of the elements in `omni.ui`, and each wrapper is opinionated about how the UI component should be placed and labeled so that
it will look good next to other wrapped components. An advanced user may start adding `omni.ui` components next to wrapped components without issue.

To see the UI elements demonstrated by the template, open the Template Generator (*Utilities > Generate Extension Templates* menu) and create a new extension under the *UI Component Library* section.
Then, enable the extension (*Window > Extensions* menu, search for the given extension name) and click on the toolbar entry with the same name. The full set of UI elements is demonstrated in the newly opened window.

## Summary

This tutorial covered the templates provided in the NVIDIA Isaac Sim *Extension Template Generator*. Each template has a common underlying structure with a thin layer of implementation to show a different
use-case. The user will be able to reference one or more of these templates to get started building a highly customized UI-based extension in NVIDIA Isaac Sim without having to build a detailed knowledge
of the internal simulator mechanics.

### Further Learning

In conjunction with these templates, the user will want to reference the [API documentation](../py/source/extensions/isaacsim.gui.components/docs/index.html#ui-element-wrappers) for the `UIElementWrapper` objects.

---

# Custom Extensions: C++

To write your own extension containing C++ code, Omniverse Kit provide an extension template as well as examples of its usage. Go to [Kit C++ Extension Template](https://docs.omniverse.nvidia.com/kit/docs/kit-extension-template-cpp/latest/index.html) for detailed instructions.

---

# Advanced Extension Template Generator from VS Code

The [Isaac Sim VS Code Edition](https://marketplace.visualstudio.com/items?itemName=NVIDIA.isaacsim-vscode-edition) is a Visual Studio Code extension that provides development support for NVIDIA Omniverse in general and Isaac Sim in particular.
One of its features is the generation of advanced extension templates.

Isaac Sim VS Code Edition’s Extension template generator wizard

Extension templates can be generated in the following forms:

* Ready-to-use extensions (**Python**)
* Extensions requiring a Kit-based build system (**C++** and/or **Python**)

The following table list the extension components that can be generated by the template as well as their availability according to the programming language to be used.

Extension components that can be generated by the template

| Component | Python | C++ | Description |
| --- | --- | --- | --- |
| Extension | yes | yes | Define an extension class that derives from `omni.ext.IExt` |
| API | yes | yes | Define/expose codebase Application Programming Interface (API) |
| OmniGraph | yes | yes | Create nodes using the OmniGraph framework for visual scripting |
| Pybind | no | yes | Reflect C++ code using `pybind11` so that it can be called from Python |
| UI | yes | no | Create Graphical User Interfaces (GUI) using Omniverse UI framework |
| Tests | yes | yes | Create test cases for the extension |

---

## Ready-To-Use Extensions

A ready-to-use extension (**Python**), as the name suggests, is an extension that can be used as-is once created without the need for any extra configuration or build system.

The subsequent subsections describe how to generate and run this type of extensions.

### Creating the Extension

Note

The folder containing the extension to be created had to be listed in the Isaac Sim extension search path in order to be discoverable.

If this is not the case, you can use the Isaac Sim’s Extensions Manager (*Window > Extensions* menu) to add it.
Click on the hamburger icon in the Extensions Manager, and then *Settings* in the sub-menu, to add the path to the folder containing your extensions.

Hint

For convenience, the `extsUser` folder at the root of the Isaac Sim installation is listed in the extension search path, so it is recommended to create the extension in that folder.

Open, in the VS Code Editor, the *Isaac Sim VS Code Edition*’s extension template generator wizard (*Templates > Extension*) and fill/check, at least, the following fields:

* **Ext. name**: Given extension name. E.g. `my.custom.extension`.
* **Ext. path**: Folder path that will contain the extension. E.g.: `PATH_TO_ISAAC_SIM/extsUser`.
* Enable the **Ready-to-use extension** checkbox.
* Enable the specific component(s) to generate.

Then, press *Create* to generate the extension. Check that the generated extension exists in the specified path. At this point, the extension is ready to be modified with your own code.

### Running the Extension

Launch Isaac Sim, then search and enable the created extension in the Extension Manager (using the given extension name).
Depending on the component(s) created, the following can be expected (without additional modification):

* Extension/API: Simply, the extension is enabled.
* OmniGraph: The node `OgnMyCustomExtensionPy` can be instantiated in an Action Graph (e.g.: through *Create > Visual Scripting > Action Graph*).
* UI: A sample window can be opened when clicking on the *Window > My Custom Extension* menu.
* Tests: The tests can be run from an opened terminal in the root directory of Isaac Sim as follows:

  > Linux
  >
  > ```python
  > ./kit/kit --empty --enable omni.kit.test --/exts/omni.kit.test/runTestsAndQuit=true --/exts/omni.kit.test/testExts/0='my.custom.extension' --ext-folder "extsUser" --no-window --allow-root
  > ```
  >
  >
  > Windows
  >
  > ```python
  > .\kit\kit --empty --enable omni.kit.test --/exts/omni.kit.test/runTestsAndQuit=true --/exts/omni.kit.test/testExts/0='my.custom.extension' --ext-folder "extsUser" --no-window --allow-root
  > ```

---

## Extensions Requiring a Kit-based Build System

Extensions (**C++** and/or **Python**) requiring Kit-based build system, as the name suggests, need to be configured as part of a Kit SDK-based application (such as the [Isaac Sim App Template](https://github.com/isaac-sim/isaacsim-app-template) or the [Omniverse Kit App Template](https://github.com/NVIDIA-Omniverse/kit-app-template)) in order to be compiled.

The subsequent subsections describe how to generate and run this type of extensions.

### Building the App Template

Get the [Isaac Sim App Template](https://github.com/isaac-sim/isaacsim-app-template), and setup and build it according to its documentation.

### Creating the Extension

Hint

For convenience, the `source/extensions` folder at the root of the Isaac Sim App Template is configured, in the build system, as a place to search for the extensions’ source code.
Therefore, it is recommended to create the extension there. **Create it if the folder doesn’t exist**.

Open, in the VS Code Editor, the *Isaac Sim VS Code Edition*’s extension template generator wizard (*Templates > Extension*) and fill/check, at least, the following fields:

* **Ext. name**: Given extension name. E.g. `my.custom.extension`.
* **Ext. path**: Folder path that will contain the extension source code. E.g.: `PATH_TO_ISAAC_SIM_APP_TEMPLATE/source/extensions`.
* Disable the *Ready-to-use extension* checkbox (if it is already enabled).
* Enable the specific component(s) to generate.

Then, press *Create* to generate the extension. Check that the generated extension exists in the specified path.

### Configuring the Build System

Depending on the component(s) created, the following configuration is necessary:

* OmniGraph: Edit the `tools/deps/kit-sdk-deps.packman.xml` file to include the USD dependency:

  > ```python
  > <import path="...all-deps.packman.xml">
  >     <!-- JUST ADD THE NEXT LINE -->
  >     <filter include="usd-${config}"/>
  > </import>
  >
  > <!-- JUST ADD THE NEXT LINE -->
  > <dependency name="usd-${config}" linkPath="../../_build/target-deps/usd/${config}"/>
  > ```
* Tests: Edit the `tools/deps/kit-sdk-deps.packman.xml` file to include the `doctest` dependency:

  > ```python
  > <import path="...all-deps.packman.xml">
  >     <!-- JUST ADD THE NEXT LINE -->
  >     <filter include="doctest"/>
  > </import>
  >
  > <!-- JUST ADD THE NEXT LINE -->
  > <dependency name="doctest" linkPath="../../_build/target-deps/doctest"/>
  > ```

### Building the Extension

To build the extension, simply run the following command from an opened terminal in the root directory of the Isaac Sim App Template:

Linux

```python
./repo.sh build
```

Windows

```python
.\repo.bat build
```

### Running the Extension

Launch Isaac Sim, then search and enable the created extension in the Extension Manager (using the given extension name).
Depending on the component(s) created, the following can be expected (without additional modification):

* Extension/API: Simply, the extension is enabled.
* OmniGraph: The node `OgnMyCustomExtensionPy` (Python) and/or `OgnMyCustomExtensionCpp` (C++) can be instantiated in an Action Graph (e.g.: through *Create > Visual Scripting > Action Graph*).
* Pybind (C++ only): The exposed C++ API via `pybind11` can be called from Python.

  > For example, execute the following code in the *Script Editor* (*Window > Script Editor* menu):
  >
  > ```python
  > import my.custom.extension
  >
  > interface = my.custom.extension.acquire_extension_interface()
  > my.custom.extension.set_default_status("custom status")
  > interface.register_object(10)
  > my.custom.extension.release_extension_interface()
  > ```
* UI (Python only): A sample window can be opened when clicking on the *Window > My Custom Extension* menu.
* Tests: The tests can be run from an opened terminal in the root directory of the Isaac Sim App Template as follows:

  > Linux
  >
  > ```python
  > ./_build/linux-x86_64/release/tests-my.custom.extension.sh
  > ```
  >
  >
  > Windows
  >
  > ```python
  > .\_build\windows-x86_64\release\tests-my.custom.extension.bat
  > ```