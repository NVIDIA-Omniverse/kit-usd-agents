# Quick Install

The Quick Install can be used for demos and to get a quick working idea of what the full product can do. After completing the quick install, you can create a room with a robot in it, which provides an even fuller picture of the product capabilities. These instructions are aimed at installation by someone with basic computer knowledge.

For a quick install on Linux or Windows:

1. Download one of the following:

   > * [Linux (x86\_64)](https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip).
   > * [Linux (aarch64)](https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-aarch64.zip).
   > * [Windows](https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-windows-x86_64.zip).
2. From the terminal or command line, execute the following commands:

   > Linux (x86\_64)
   >
   > ```python
   > mkdir ~/isaacsim
   > cd ~/Downloads
   > unzip "isaac-sim-standalone-5.1.0-linux-x86_64.zip" -d ~/isaacsim
   > cd ~/isaacsim
   > ./post_install.sh
   > ./isaac-sim.sh
   > ```
   >
   >
   > Linux (aarch64)
   >
   > ```python
   > mkdir ~/isaacsim
   > cd ~/Downloads
   > unzip "isaac-sim-standalone-5.1.0-linux-aarch64.zip" -d ~/isaacsim
   > cd ~/isaacsim
   > ./post_install.sh
   > ./isaac-sim.sh
   > ```
   >
   >
   > Windows
   >
   > ```python
   > mkdir C:\isaacsim
   > cd %USERPROFILE%/Downloads
   > tar -xvzf "isaac-sim-standalone-5.1.0-windows-x86_64.zip" -C C:\isaacsim
   > cd C:\isaacsim
   > post_install.bat
   > isaac-sim.bat
   > ```
   >
   > Final load message example:
   >
   > 
3. After the Isaac Sim development environment opens fully, verify that you can see:

   
4. Select **Create > Environment > Simple Room**.
5. Select **Create > Robots > Franka Emika Panda Arm**.

   > 
6. On the leftmost side of your screen, look for an arrow button, and press it to play a short simulation.

## Further Reading

Try out the following tutorials:

> * [Isaac Sim Basic Usage Tutorial](Quick_Tutorials.md)
> * [Basic Robot Tutorial](Quick_Tutorials.md)

Then you can try [Robot Setup Tutorials Series](Robot_Setup.md).