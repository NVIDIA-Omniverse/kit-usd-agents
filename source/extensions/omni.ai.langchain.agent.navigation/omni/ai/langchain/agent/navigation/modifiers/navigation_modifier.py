## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import re
from typing import Any, Dict, List, Optional, Tuple

import omni.usd
from langchain_core.messages import AIMessage
from lc_agent import NetworkModifier, RunnableHumanNode
from pxr import Gf, Sdf, Usd, UsdGeom


class NavigationModifier(NetworkModifier):
    """
    Modifier that handles navigation commands from the NavigationGenNode.

    This modifier recognizes and executes the following commands:
    - LIST - List all points of interest in the scene
    - NAVIGATE <name> - Navigate to a specific point of interest
    - SAVE <name> - Save the current camera position as a new point of interest
    """

    # Command patterns
    LIST_COMMAND = "LIST"
    NAVIGATE_PATTERN = r"NAVIGATE\s+(.+)"
    SAVE_PATTERN = r"SAVE\s+(.+)"
    DONE_COMMAND = "DONE"

    async def on_post_invoke_async(self, network, node):
        """
        Post-invoke hook that processes navigation commands.

        Args:
            network: The current network being executed
            node: The node that was just invoked
        """
        if (
            node.invoked
            and isinstance(node.outputs, AIMessage)
            and node.outputs.content
            and not network.get_children(node)
        ):
            # We are here because this node has the AI message and since it has
            # no children, it is the end of the chain, so we need to extract the
            # command from the node output
            command = node.outputs.content.strip()
            result = await self.process_command(command)
            if result:
                with network:
                    RunnableHumanNode(f"Assistant: {result}")

    async def process_command(self, command):
        """
        Process navigation commands and return appropriate responses.
        This method identifies the command type and delegates to specialized handlers.

        Args:
            command: The command string to process

        Returns:
            str: Response message or None if no response needed
        """
        # Process LIST command
        if command == self.LIST_COMMAND:
            return self._handle_list_command()

        # Process NAVIGATE command
        navigate_match = re.match(self.NAVIGATE_PATTERN, command)
        if navigate_match:
            poi_name = navigate_match.group(1).strip()
            return self._handle_navigate_command(poi_name)

        # Process SAVE command
        save_match = re.match(self.SAVE_PATTERN, command)
        if save_match:
            poi_name = save_match.group(1).strip()
            return await self._handle_save_command(poi_name)

        # If command is DONE, do nothing
        if command == self.DONE_COMMAND:
            return None

        # If command is not recognized, provide a helpful message
        return self._handle_unrecognized_command(command)

    def _handle_unrecognized_command(self, command):
        """
        Handle unrecognized commands by providing a helpful message.

        Args:
            command: The unrecognized command

        Returns:
            str: A helpful message listing available commands
        """
        available_commands = [
            f"- {self.LIST_COMMAND}: List all points of interest",
            f"- NAVIGATE <name>: Navigate to a specific point of interest",
            f"- SAVE <name>: Save the current camera position as a new point of interest",
            f"- {self.DONE_COMMAND}: Exit navigation mode",
        ]

        return f"Command '{command}' is not recognized. Available commands are:\n" f"{chr(10).join(available_commands)}"

    def _handle_list_command(self):
        """
        Handle the LIST command to show all points of interest.

        Returns:
            str: Formatted list of POI names or error message
        """
        pois = self._get_pois()
        if not pois:
            return "No points of interest found in the scene."

        # Format the list of POIs
        poi_names = [poi["name"] for poi in pois]
        return ", ".join(poi_names)

    def _handle_navigate_command(self, poi_name):
        """
        Handle the NAVIGATE command to move camera to a specific POI.

        Args:
            poi_name: Name of the point of interest to navigate to

        Returns:
            str: Success or error message
        """
        pois = self._get_pois()

        # Find the POI by name (case-insensitive)
        for poi in pois:
            if poi["name"].lower() == poi_name.lower():
                return self._navigate_to_poi(poi)

        # POI not found
        return f"Point of interest '{poi_name}' not found. Use the LIST command to see available points."

    def _navigate_to_poi(self, poi):
        """
        Navigate to a specific point of interest.

        Args:
            poi: Point of interest dictionary containing position and look_at

        Returns:
            str: Success or error message
        """
        # Get position and look-at
        position = Gf.Vec3d(*poi["position"])
        look_at = Gf.Vec3d(*poi["look_at"])

        # Set camera transform
        success = self._set_camera_transform(position, look_at)
        if success:
            return f"navigated to {poi['name']}"
        else:
            return "Failed to set camera transform. Please check if the camera is available."

    async def _handle_save_command(self, poi_name):
        """
        Handle the SAVE command to store current camera view as a POI.

        Args:
            poi_name: Name to give to the new point of interest

        Returns:
            str: Success or error message
        """
        # Get current camera transform
        camera_info = self._get_current_camera()
        if not camera_info:
            return "Failed to get current camera information. Please check if the camera is available."

        position, look_at = camera_info

        # Save the POI
        success = self._save_poi(poi_name, position, look_at)
        if success:
            return f'saved view as "{poi_name}"'
        else:
            return "Failed to save point of interest. Please check if the stage is available."

    def _get_stage(self) -> Optional[Usd.Stage]:
        """Get the current USD stage."""
        return omni.usd.get_context().get_stage()

    def _get_current_camera_path(self) -> str:
        """
        Get the path of the current active camera.

        Returns:
            str: Path to the active camera prim
        """
        import omni.kit.viewport

        try:
            # Get the active viewport and camera path using the correct method
            viewport = omni.kit.viewport.utility.get_active_viewport()
            if viewport:
                camera_path = viewport.get_active_camera()
                return str(camera_path)
            return ""
        except Exception as e:
            print(f"Error getting camera path: {e}")
            return ""

    def _get_current_camera(self) -> Optional[Tuple[Gf.Vec3d, Gf.Vec3d]]:
        """
        Get the current camera position and look-at point.

        Returns:
            Tuple containing (position, look_at) or None if camera not available
        """
        try:
            # Get the stage
            stage = self._get_stage()
            if not stage:
                return None

            # Get the camera path
            camera_path = self._get_current_camera_path()
            if not camera_path:
                return None

            # Get the camera prim
            camera_prim = stage.GetPrimAtPath(camera_path)
            if not camera_prim:
                return None

            # Get the camera transform
            xform = UsdGeom.Xformable(camera_prim)
            if not xform:
                return None

            # Get the world transform matrix
            time = Usd.TimeCode.Default()
            world_transform = xform.ComputeLocalToWorldTransform(time)

            # Extract position from the transform matrix
            position = Gf.Vec3d(world_transform.ExtractTranslation())

            # For look-at, we'll use a point in front of the camera
            # Get the camera's forward direction (negative Z axis in camera space)
            rotation = world_transform.ExtractRotation()
            forward = rotation.TransformDir(Gf.Vec3d(0, 0, -1))

            # Look at a point 10 units in front of the camera
            look_at = position + (forward * 10)

            return (position, look_at)
        except Exception as e:
            print(f"Error getting camera information: {e}")
            return None

    def _set_camera_transform(self, position: Gf.Vec3d, look_at: Gf.Vec3d) -> bool:
        """
        Set the camera transform to look from position toward look_at.

        Args:
            position: Camera position (x, y, z)
            look_at: Point to look at (x, y, z)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get the stage
            stage = self._get_stage()
            if not stage:
                return False

            # Get the camera path
            camera_path = self._get_current_camera_path()
            if not camera_path:
                return False

            # Get the camera prim
            camera_prim = stage.GetPrimAtPath(camera_path)
            if not camera_prim:
                return False

            # Get the camera transform
            xform = UsdGeom.Xformable(camera_prim)
            if not xform:
                return False

            # Get the stage's up axis
            up_axis = UsdGeom.GetStageUpAxis(stage)

            # Set the world up vector based on the stage's up axis
            if up_axis == UsdGeom.Tokens.y:
                world_up = Gf.Vec3d(0, 1, 0)  # Y-up
            elif up_axis == UsdGeom.Tokens.z:
                world_up = Gf.Vec3d(0, 0, 1)  # Z-up
            else:
                # Default to Y-up if for some reason the up axis is not recognized
                world_up = Gf.Vec3d(0, 1, 0)

            # Calculate the look-at transform
            # This creates a transform that positions the camera at 'position'
            # and orients it to look at 'look_at'

            # Direction from position to look_at
            forward = (look_at - position).GetNormalized()

            # Check if forward and world_up are too close to parallel
            dot_product = Gf.Dot(forward, world_up)
            if abs(dot_product) > 0.999:
                # If they're nearly parallel, use a different up vector
                if abs(Gf.Dot(Gf.Vec3d(1, 0, 0), forward)) < 0.999:
                    temp_up = Gf.Vec3d(1, 0, 0)
                else:
                    temp_up = Gf.Vec3d(0, 1, 0)
                right = Gf.Cross(forward, temp_up).GetNormalized()
                up = Gf.Cross(right, forward).GetNormalized()
            else:
                # Compute right and up vectors
                right = Gf.Cross(forward, world_up).GetNormalized()
                up = Gf.Cross(right, forward).GetNormalized()

            # Create rotation matrix from the orthonormal basis
            rotation_matrix = Gf.Matrix3d(
                right[0], right[1], right[2], up[0], up[1], up[2], -forward[0], -forward[1], -forward[2]
            )

            # Create the transform matrix
            transform = Gf.Matrix4d().SetTranslate(position)
            transform.SetRotateOnly(rotation_matrix)

            # Clear existing transforms
            xform.ClearXformOpOrder()

            # Add the new transform
            xform_op = xform.AddTransformOp()
            xform_op.Set(transform)

            return True
        except Exception as e:
            print(f"Error setting camera transform: {e}")
            return False

    def _get_pois(self) -> List[Dict]:
        """
        Get all points of interest from the stage metadata.

        Returns:
            List of POI dictionaries with name, position, and look_at
        """
        stage = self._get_stage()
        if not stage:
            return []

        # Get the root layer
        root_layer = stage.GetRootLayer()

        # Get custom data from layer
        custom_data = root_layer.customLayerData

        # Get POIs from custom data
        pois = custom_data.get("points_of_interest", [])

        return pois

    def _save_poi(self, name: str, position: Gf.Vec3d, look_at: Gf.Vec3d) -> bool:
        """
        Save a point of interest to the stage metadata.

        Args:
            name: Name of the POI
            position: Camera position (x, y, z)
            look_at: Point to look at (x, y, z)

        Returns:
            bool: True if successful, False otherwise
        """
        stage = self._get_stage()
        if not stage:
            return False

        # Get the root layer
        root_layer = stage.GetRootLayer()

        # Get custom data from layer
        custom_data = root_layer.customLayerData

        # Get existing POIs or create empty list
        pois = custom_data.get("points_of_interest", [])

        # Get the camera path
        camera_path = self._get_current_camera_path()

        # Convert position and look_at to lists for serialization
        pos_list = [position[0], position[1], position[2]]
        look_at_list = [look_at[0], look_at[1], look_at[2]]

        # Create new POI
        new_poi = {"name": name, "position": pos_list, "look_at": look_at_list, "camera_path": camera_path}

        # Check if POI with same name exists
        for i, poi in enumerate(pois):
            if poi.get("name") == name:
                # Replace existing POI
                pois[i] = new_poi
                break
        else:
            # Add new POI
            pois.append(new_poi)

        # Update custom data
        custom_data["points_of_interest"] = pois
        root_layer.customLayerData = custom_data

        return True
