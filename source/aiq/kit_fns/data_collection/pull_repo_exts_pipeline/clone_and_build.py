#!/usr/bin/env python3
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

"""
Kit App Template Clone and Build Module

This module handles:
1. Cloning the repository
2. Running the build process
3. Locating and validating the generated exts
4. Returning the path for use by other pipeline stages
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import defusedxml.ElementTree as ET  # Use defusedxml for XXE protection


class RepoCloneAndBuilder:
    """Handles cloning and building repository."""

    def __init__(
        self,
        work_dir: Path,
        repo_url: str = "https://gitlab-master.nvidia.com/omniverse/kit-github/kit-app-template.git",
        branch: str = "master",
        exts_path: Optional[str] = "exts",
        app_template_path: Optional[str] = "templates/apps",
        has_version: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the KAT clone and builder.

        Args:
            work_dir: Working directory for pipeline operations
            repo_url: Git URL for repository
            branch: Git branch to checkout
            exts_path: Path to the extensions directory
            app_template_path: Path to the app template (if None, app_template_path return is optional)
            has_version: If True, kit_version is required; if False, it's optional
            logger: Logger instance (creates default if None)
        """
        self.work_dir = Path(work_dir)
        self.repo_url = repo_url
        self.branch = branch
        self.exts_path = exts_path
        self.app_template_path = app_template_path
        self.has_version = has_version
        self.logger = logger or self._create_default_logger()

        if os.path.exists(self.repo_url):
            self.repo_dir = Path(self.repo_url)
        else:
            self.repo_dir = self.work_dir / self.repo_url.split("/")[-1].split(".")[0]

    def _create_default_logger(self) -> logging.Logger:
        """Create default logger if none provided."""
        logger = logging.getLogger("RepoCloneAndBuilder")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def clone_and_build(self) -> Optional[dict]:
        """
        Execute the complete clone and build process.

        Returns:
            Optional[dict] containing at minimum ``exts_path`` and optionally
            ``kit_version`` and ``app_template_path`` if they can be detected,
            or None if the clone/build fails.
        """
        try:
            get_current_branch_command = ["git", "rev-parse", "--abbrev-ref", "HEAD"]

            if self.repo_dir.exists():
                result = subprocess.run(
                    get_current_branch_command,
                    cwd=self.repo_dir,
                    timeout=600,
                    capture_output=True,
                )
                current_branch = result.stdout.decode("utf-8").rstrip()

                if current_branch == self.branch:
                    # If the exts, app template path and kit version already exist, return them
                    paths_and_version = self._locate_and_verify_paths_and_version()
                    if paths_and_version:
                        return paths_and_version

            # Create working directories
            if not self._prepare_directories():
                return None

            # Clone repository
            if not self._clone_repository():
                return None

            # Build repository
            if not self._build_repository():
                return None

            # Verify and locate exts and kit version
            return self._locate_and_verify_paths_and_version()

        except Exception as e:
            self.logger.error(f"Unexpected error during clone and build: {e}")
            return None

    def _prepare_directories(self) -> bool:
        """Create necessary working directories."""
        try:
            self.repo_dir.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created working directory: {self.repo_dir.parent}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create working directories: {e}")
            return False

    def _clone_repository(self) -> bool:
        """Clone the repository."""

        try:
            if not os.path.exists(self.repo_dir):
                self.logger.info(f"Cloning repository from: {self.repo_url} to {self.repo_dir}")
                clone_cmd = [
                    "git",
                    "clone",
                    "--branch",
                    self.branch,
                    "--progress",  # Show progress
                    self.repo_url,
                    self.repo_dir.relative_to(self.work_dir).as_posix(),
                ]
                # Stream output in real-time
                return self._run_command_with_streaming(
                    clone_cmd,
                    cwd=self.work_dir,
                    timeout=600,
                    operation="clone",  # 10 minute timeout
                )
            else:
                self.logger.info(f"Updating repository {self.repo_dir}")
                switch_command = ["git", "switch", self.branch]

                result = self._run_command_with_streaming(
                    switch_command,
                    cwd=self.repo_dir,
                    timeout=600,
                    operation="switch",  # 10 minute timeout
                )

                if not result:
                    return False

                pull_command = ["git", "pull", "origin", self.branch]
                return self._run_command_with_streaming(
                    pull_command,
                    cwd=self.repo_dir,
                    timeout=600,
                    operation="pull",  # 10 minute timeout
                )

        except Exception as e:
            self.logger.error(f"Failed to clone repository: {e}")
            return False

    def _build_repository(self) -> bool:
        """Build the cloned repository."""
        self.logger.info("Starting repository build process...")

        try:
            # First try to clean the _build directory
            if os.path.exists(self.repo_dir / "_build"):
                shutil.rmtree(self.repo_dir / "_build", ignore_errors=True)
                self.logger.info(f"Cleaned _build directory: {self.repo_dir}/_build")

            # Determine appropriate repo command for platform
            if os.name == "nt":
                repo_cmd = ["repo.bat"]
            else:
                repo_cmd = ["./repo.sh"]

            # Add build arguments
            repo_cmd.append("build")

            self.logger.info(f"Running build command: {' '.join(repo_cmd)}")

            # Stream output in real-time
            return self._run_command_with_streaming(
                repo_cmd,
                cwd=self.repo_dir,
                timeout=1800,
                operation="build",  # 30 minute timeout for build
            )

        except Exception as e:
            self.logger.error(f"Failed to build repository: {e}")
            return False

    def _run_command_with_streaming(self, cmd: list, cwd: Path, timeout: int, operation: str) -> bool:
        """
        Run a command with real-time output streaming.

        Args:
            cmd: Command and arguments to run
            cwd: Working directory for the command
            timeout: Timeout in seconds
            operation: Name of operation for logging (e.g., "clone", "build")

        Returns:
            True if command succeeded, False otherwise
        """
        try:
            self.logger.info(f"Running {operation} command: {' '.join(cmd)}")

            # Start the process - combine stderr into stdout for simpler handling
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout
                text=True,
                errors="replace",  # Handle encoding errors gracefully
                shell=True if os.name == "nt" else False,
            )

            # Stream output line by line
            if process.stdout:
                for line in process.stdout:
                    try:
                        # Print line with operation prefix
                        clean_line = line.rstrip()
                        if clean_line:  # Only log non-empty lines
                            self.logger.info(f"[{operation}] {clean_line}")
                    except UnicodeEncodeError as e:
                        # Handle encoding errors like in repo_man
                        self.logger.warning(f"[{operation}] Encoding error: {e}")
                        clean_line = line.encode("ascii", "ignore").decode("utf-8").rstrip()
                        if clean_line:
                            self.logger.info(f"[{operation}] {clean_line}")

            # Wait for process to complete with timeout
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.logger.error(f"{operation.capitalize()} timed out after {timeout} seconds")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                return False

            if return_code == 0:
                self.logger.info(f"Successfully completed {operation}")
                return True
            else:
                self.logger.error(f"{operation.capitalize()} failed with return code {return_code}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to run {operation} command: {e}")
            return False

    def _locate_and_verify_paths_and_version(self) -> Optional[dict]:
        """Locate and verify the generated extensions directory, kit version and app template path.

        Returns a dict with:
        - exts_path (always present)
        - kit_version (optional based on has_version parameter)
        - app_template_path (optional based on app_template_path parameter)
        """
        # Try multiple path patterns for different platforms
        exts_patterns = [
            f"_build/*/release/{self.exts_path}",
            f"_build/windows-x86_64/release/{self.exts_path}",
            f"_build/linux-x86_64/release/{self.exts_path}",
            f"_build/*/*/{self.exts_path}",
        ]

        packman_xml_patterns = [
            "**/deps/kit-sdk.packman.xml",
        ]

        def _locate_path(patterns: List[str]) -> Optional[Path]:
            path = None

            for pattern in patterns:
                matches = list(self.repo_dir.glob(pattern))
                if matches:
                    path = matches[0]
                    self.logger.info(f"Found path using pattern '{pattern}': {path}")
                    break

            if not path:
                self.logger.error("Could not locate path in build output")
                self.logger.error("Searched patterns:")
                for pattern in exts_patterns:
                    self.logger.error(f"  {pattern}")

            return path

        exts_path = _locate_path(exts_patterns)
        if not exts_path or not self._verify_exts(exts_path):
            return None

        # Build result dict starting with required exts_path
        result: dict = {
            "exts_path": str(exts_path),
        }

        # Handle kit version based on has_version parameter
        if self.has_version:
            # Version is required - fail if not found
            packman_xml_path = _locate_path(packman_xml_patterns)
            if packman_xml_path:
                kit_version = self._extract_kit_version(packman_xml_path)
                if kit_version:
                    result["kit_version"] = kit_version
                else:
                    self.logger.error("Could not extract kit version from packman XML")
                    return None
            else:
                self.logger.error("Could not locate kit-sdk.packman.xml (required)")
                return None

        # Handle app template path - optional if app_template_path is None
        if self.app_template_path is not None:
            # app_template_path is required - fail if not found
            app_template_path = _locate_path([self.app_template_path])
            if app_template_path:
                result["app_template_path"] = str(app_template_path)
            else:
                self.logger.error(f"Could not locate app template path: {self.app_template_path} (required)")
                return None
        # If self.app_template_path is None, we skip looking for it entirely

        return result

    def _extract_kit_version(self, packman_xml_path: Path) -> Optional[str]:
        """Extract the kit version from the kit-sdk.packman.xml file."""
        tree = ET.parse(packman_xml_path)
        version = tree.find(".//package").get("version")
        if not version:
            self.logger.error(f"Could not extract version from kit-sdk.packman.xml: {packman_xml_path}")
            return None
        return version.split(".gl.")[0]

    def _verify_exts(self, exts_path: Path) -> bool:
        """Verify the exts directory is valid and contains extensions."""
        if not exts_path.exists():
            self.logger.error(f"Exts path does not exist: {exts_path}")
            return False

        if not exts_path.is_dir():
            self.logger.error(f"Exts path is not a directory: {exts_path}")
            return False

        # Count extension directories
        try:
            extensions = [d for d in exts_path.iterdir() if d.is_dir()]
            extension_count = len(extensions)

            unique_extensions = set([ext.name.split("-")[0] for ext in extensions])

            if len(unique_extensions) != extension_count:
                self.logger.error(f"Exts directory contains duplicate extensions: {exts_path}")
                return False

            if extension_count == 0:
                self.logger.error(f"Exts directory is empty: {exts_path}")
                return False

            self.logger.info(f"Verified exts with {extension_count} extensions: {exts_path}")

            # Log some example extensions for verification
            if extension_count > 0:
                example_extensions = [ext.name for ext in extensions[:5]]
                self.logger.info(f"Sample extensions: {', '.join(example_extensions)}")
                if extension_count > 5:
                    self.logger.info(f"... and {extension_count - 5} more")

            return True

        except Exception as e:
            self.logger.error(f"Failed to verify exts contents: {e}")
            return False


def clone_and_build_repo(
    work_dir: str,
    repo_url: str = "https://gitlab-master.nvidia.com/omniverse/kit-github/kit-app-template.git",
    branch: str = "master",
    exts_path: Optional[str] = "exts",
    app_template_path: Optional[str] = "templates/apps",
    has_version: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Main function to clone and build repository.

    Args:
        work_dir: Working directory for pipeline operations
        repo_url: Git URL for repository
        branch: Git branch to checkout
        exts_path: Path to the extensions directory
        app_template_path: Path to the app template (if None, app_template_path return is optional)
        has_version: If True, kit_version is required; if False, it's optional
        logger: Logger instance

    Returns:
        Optional[dict] containing the paths and version if successful, None otherwise.
        Dict always contains 'exts_path'.
        - If has_version=True, 'kit_version' is required (fails if not found)
        - If has_version=False, 'kit_version' is optional (included if found)
        - If app_template_path is not None, 'app_template_path' is required (fails if not found)
        - If app_template_path is None, 'app_template_path' is not included in result
    """
    builder = RepoCloneAndBuilder(
        work_dir=Path(work_dir),
        repo_url=repo_url,
        branch=branch,
        exts_path=exts_path,
        app_template_path=app_template_path,
        has_version=has_version,
        logger=logger,
    )

    paths_and_version = builder.clone_and_build()
    return paths_and_version


if __name__ == "__main__":
    """Command line interface for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Clone and build repo")
    parser.add_argument("--work-dir", default=".", help="Working directory")
    parser.add_argument(
        "--repo-url",
        default="https://gitlab-master.nvidia.com/omniverse/kit-github/kit-app-template.git",
        help="Repository URL",
    )
    parser.add_argument("--branch", default="feature/109.0", help="Git branch")
    parser.add_argument("--exts-path", default="exts", help="Path to the extensions directory")
    parser.add_argument(
        "--app-template-path",
        default="templates/apps",
        help="Path to the app template (keep empty to make it optional)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("clone_and_build")

    # Handle 'none' string for optional app_template_path
    app_template_path = None if not args.app_template_path else args.app_template_path

    paths_and_version = clone_and_build_repo(
        work_dir=args.work_dir,
        repo_url=args.repo_url,
        branch=args.branch,
        exts_path=args.exts_path,
        app_template_path=app_template_path,
        has_version=True,
        logger=logger,
    )

    if paths_and_version:
        message_parts = [f"SUCCESS: Exts available at: {paths_and_version['exts_path']}"]
        kit_version = paths_and_version.get("kit_version")
        if kit_version:
            message_parts.append(f"Kit Version: {kit_version}")
        app_template_path_result = paths_and_version.get("app_template_path")
        if app_template_path_result:
            message_parts.append(f"App Template Path: {app_template_path_result}")
        print(", ".join(message_parts))
        sys.exit(0)
    else:
        print("FAILED: Could not clone and build repo")
        sys.exit(1)
