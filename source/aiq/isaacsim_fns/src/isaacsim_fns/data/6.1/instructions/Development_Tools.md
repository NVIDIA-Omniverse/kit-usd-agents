# Development Tools

The development tools tutorials series is for intermediate users who wants to create Python and C++ extensions, use Jupyter notebooks, debug Python scripts, and create OmniGraph nodes.

## Tools

- Visual Studio Code (VS Code)
- Jupyter Notebook
- Omniverse Script Editor

## Tutorials

- Modify Carb Settings

---

# Visual Studio Code (VS Code)

## Isaac Sim VS Code Edition

[Isaac Sim VS Code Edition](https://marketplace.visualstudio.com/items?itemName=NVIDIA.isaacsim-vscode-edition) is an extension for Visual Studio Code that provides development support for NVIDIA Omniverse in general and Isaac Sim in particular.

Key Features:

* Execute Python code, in the Python environment of a running application, locally or remotely from VS Code and show the output in the *Isaac Sim VS Code Edition* panel.
* Browse and insert snippets of code related to Isaac Sim, Omniverse Kit and Universal Scene Description (USD).
* Create templates for Omniverse/Isaac Sim extensions and other development approaches.
* Quick access to the most relevant Omniverse/Isaac Sim documentation sources and resources without leaving the editor.

**Install it now to get started**: [Isaac Sim VS Code Edition](https://marketplace.visualstudio.com/items?itemName=NVIDIA.isaacsim-vscode-edition)

---

## Interactive Scripting

The `isaacsim.code_editor.vscode` extension allows you to edit and execute Python code interactively from the VS Code editor.
It can be enabled or disabled using the [Extension Manager](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html "(in Omniverse Extensions)") by searching for `isaacsim.code_editor.vscode`.

> Note
>
> This extension requires its Visual Studio Code pair extension: [Isaac Sim VS Code Edition](https://marketplace.visualstudio.com/items?itemName=NVIDIA.isaacsim-vscode-edition) to be installed and enabled, in the VS Code editor, in order to execute Python scripts on a running Isaac Sim instance.

1. To begin, enable this extension using the [Extension Manager](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html "(in Omniverse Extensions)") by searching for `isaacsim.code_editor.vscode`.
2. Once the extension is enabled, go to the top menu bar and click on Window > VS Code to open the Isaac Sim folder in a VS Code application.
3. Open a stored file or write the code you want to run in a VS Code editor tab.
4. From the VS Code editor, click on the *Isaac Sim VS Code Edition* container in the Activity Bar (the one with the Isaac Sim logo) to open it.
   Then, click on *Run* (or *Run selected text* if you have selected code statements), in the *Commands* tree view, to execute it.
5. Inspect the execution output, if any, in the *Isaac Sim VS Code Edition* output panel.

---

## VS Code Configuration Files

The Isaac Sim installation provides a `.vscode` workspace with a pre-configured environment under the following three files:

```python
.vscode/launch.json
.vscode/settings.json
.vscode/tasks.json
```

### launch.json

This file provides three different configurations that can be executed using the `Run & Debug` section in VSCode.

* **Python: Current File**: Debug the currently open standalone Python file, should not be used with extension examples/code.
* **Python: Attach**: Attach to a running Isaac Sim application for debugging purposes, most useful when running an interactive GUI application. See [Attaching the Debugger to a Running App](Debugging_Profiling.md) for usage information.
* **(Linux) isaac-sim** Run the main Isaac Sim application with an attached debugger.

### settings.json

This file sets the default Python executable that comes with Isaac Sim:

```python
# "python.pythonPath": "${workspaceFolder}/kit/python/bin/python3",
```

As well as a configuration for `"python.analysis.extraPaths"` which by default includes all of the extensions that are provided by default. You can add additional paths here if needed.

### tasks.json

This is a helper file that contains a task used to automatically setup the Python environment when using the `Python: Current File` option in `Run & Debug`.

```python
# "tasks": [
#     {
#         "label": "setup_python_env",
#         "type": "shell",
#         "linux": {
#             "command": "source ${workspaceFolder}/setup_python_env.sh && printenv >${workspaceFolder}/.vscode/.standalone_examples.env"
#         }
#     }
# ]
```

Once executed, the task generates the `.standalone_examples.env` file used by VS Code to launch the Python debug process.
Refer to [Debugging With Visual Studio Code](Debugging_Profiling.md) for more details.

---

# Jupyter Notebook

## Interactive Scripting

The `isaacsim.code_editor.jupyter` extension allows you to to open a [JupyterLab](https://jupyter.org) (or [Jupyter Notebook](https://jupyter.org)) app in the current Isaac Sim application scope and edit and execute Python code interactively.

1. To begin, enable this extension using the [Extension Manager](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html "(in Omniverse Extensions)") by searching for `isaacsim.code_editor.jupyter`.

   > Note
   >
   > This may take several seconds (and Isaac Sim will freeze) if this is the first time the `isaacsim.code_editor.jupyter` is enabled.
   > Several Python dependencies will be installed.
2. Once the extension is enabled, go to the top menu bar and click on Window > Jupyter Notebook to open a Jupyter app in the default web browser.
3. In the Jupyter app, click on the *Omniverse (Python 3)* kernel (the one with the Omniverse logo) to create a new Untitled notebook.
4. Execute code by clicking the Run button at the top of the notebook. Try it yourself with the same code snippet from above!

   > Warning
   >
   > * The *Omniverse (Python 3)* kernel is designed to run Python code, via the `isaacsim.code_editor.jupyter` extension, on a running Isaac Sim instance (where the Kit application has control over the update/simulation loop).
   > * The *Isaac Sim Python 3* kernel is used to run standalone applications (see [Running Standalone Isaac Sim from Jupyter Notebook](#isaac-sim-python-jupyter-notebook-config) for more details).
   >
   > 

Warning

Execution of blocking code freezes Isaac Sim.

Hint

* Use the Tab key for code autocompletion.
* Use the Ctrl + I keys for code introspection (display docstring if available).

Note

The notebooks are saved, by default, in a folder within the extension itself: `exts/isaacsim.code_editor.jupyter/data/notebooks`. See the location for Isaac Sim packages/extensions in [Location for Isaac Sim app](Installation.md).

**Limitations**

* IPython magic commands are not available.
* Matplotlib plotting is not available in the notebooks.
* Printing, inside callbacks, is not displayed in the notebooks but in the Omniverse terminal.

---

## Running Standalone Isaac Sim from Jupyter Notebook

Warning

* This workflow is only supported on Linux.

### Configuration Files

In order for Isaac Sim to work inside of a Jupyter Notebook we provide a custom Jupyter kernel that is installed the first time you run `./jupyter_notebook.sh`.
The kernel.json itself is fairly simple:

```python
{
    "argv": ["AUTOMATICALLY_REPLACED", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Isaac Sim Python 3",
    "language": "python",
    "env": {"ISAAC_JUPYTER_KERNEL": "1"},
    "metadata": {"debugger": true},
}
```

The important part is that `AUTOMATICALLY_REPLACED` gets replaced by `jupyter_notebook.sh` with the absolute path to the Python executable that is located in the kit/python directory at runtime. Once the variable is replaced, the kernel is installed and the notebook is started. There is an extra variable `ISAAC_JUPYTER_KERNEL` that is used inside of Isaac Sim to setup for notebook usage properly.

Because notebooks require asyncio support, and Isaac Sim itself uses asyncio internally, we automatically execute the following two lines when loading the `isaacsim` module (or the `isaacsim.simulation_app` extension) which provides the `SimulationApp` class:

```python
{
    "argv": ["AUTOMATICALLY_REPLACED", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Isaac Sim Python 3",
    "language": "python",
    "env": {"ISAAC_JUPYTER_KERNEL": "1"},
    "metadata": {"debugger": true},
}
```

This ensures that asyncio calls can be nested inside of the Jupyter Notebook properly.

When writing code in notebooks, it is necessary to first instantiate the `SimulationApp` class (from `isaacsim` or `isaacsim.simulation_app`) after perform any Isaac Sim / Omniverse imports:

```python
{
    "argv": ["AUTOMATICALLY_REPLACED", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Isaac Sim Python 3",
    "language": "python",
    "env": {"ISAAC_JUPYTER_KERNEL": "1"},
    "metadata": {"debugger": true},
}
```

Then, to run the notebook just execute the following commands and play the notebook cells:

```python
./jupyter_notebook.sh PATH_TO_NOTEBOOK.ipynb
```

---

# Omniverse Script Editor

Script Editor is a Python editing environment internal to Omniverse Kit. It can be used to run snippets of Python code to interact with the stage.

1. To open the Script Editor window, go to the Menu Bar and click *Window > Script Editor*.
2. Open multiple tabs by going to the *Tab* Menu in the Script Editor window. All the tabs share the same environment, so libraries that are imported or variables defined in one environment can be accessed and used in other environments.

Refer to [Script Editor](https://docs.omniverse.nvidia.com/extensions/latest/ext_script-editor.html "(in Omniverse Extensions)") in the Omniverse docs for more details.

---

# Modify Carb Settings

[Carbonite (carb)](Glossary.md) settings are used to configure default behaviors of Omniverse and Isaac Sim. They can control a wide ranges of features, such as window properties, ROS versions, browser folders, and more. You may wish to change these settings to suit your needs. Here we show the four ways to change the Carb settings in Isaac Sim.

For this tutorial, we will set a parameter inside extension `isaacsim.my.extension` named `data.foo` to the value `True`. Replace these with your actual extension name, setting parameter, and value when you are working with your project.

## Script Editor Snippet

You can temporarily and quickly change the Carb settings in the [Script Editor](https://docs.omniverse.nvidia.com/extensions/latest/ext_script-editor.html "(in Omniverse Extensions)"). This is useful for testing and debugging, and can be done while Isaac Sim is open. The changes made this way will not be saved after you close the application, and relaunching the simulator will reset the settings.

```python
import carb.settings
import omni.kit

## Set Carb Setting
settings = carb.settings.get_settings()
settings.set("/exts/isaacsim.my.extension/data/foo", True)

## Restart Extension to Apply Changes
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.my.extension", False)
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.my.extension", True)
```

## Command-Line Argument

You can launch Isaac Sim with a command-line argument to change the Carb settings. The changes made this way will not be saved after you close the application, and relaunching the simulator without the arguments will reset the settings.

At the root of your Isaac Sim installation, run the following command:

> Linux
>
> ```python
> ./isaac-sim.sh --/exts/isaacsim.my.extension/data/foo=True
> ```
>
>
> Windows
>
> ```python
> .\isaac-sim.bat --/exts/isaacsim.my.extension/data/foo=True
> ```

## Edit .toml File

For more permanent changes, you can edit the extension’s .toml file. The changes made this way will persist after you close the application.

1. Navigate to the extension’s folder. For example, if you are changing the settings for the isaacsim.my.extension, navigate to <isaac-sim-root\_dir>/exts/isaacsim.my.extension/config.
2. Open the .toml file with a text editor, and add the following line to the file:

   > ```python
   > [settings]
   > exts."isaacsim.my.extension".data.foo = true
   > ```
3. Launch Isaac Sim to see the changes.

## Customize .kit File

If you have multiple settings in multiple extensions that you want to change, you can edit the .kit file for your application. The changes made this way will persist after you close the application.

1. From the root of your Isaac Sim installation, navigate to <isaac-sim-root\_dir>/apps/. Locate the Kit experience app file you are using in this folder. By default, it is the isaacsim.exp.full.kit.
2. Open the app file and add the following line to the file:

   > ```python
   > [settings]
   > exts."isaacsim.my.extension".data.foo = true
   > ```
3. Launch Isaac Sim to see the changes.