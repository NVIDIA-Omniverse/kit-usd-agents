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
Master Data Collection Pipeline for Kit Extensions

This script orchestrates the complete data collection process for Kit extensions,
chaining together all individual pipelines to generate the complete dataset used
by the kit-fns MCP service.

Pipeline Stages:
1. Pull Repo & Build (clone and build repository)
2. Preparation (validate inputs)
3. Extension Data Collection (metadata + Code Atlas)
4. Code Examples Extraction (interesting methods)
5. Settings Discovery (TOML + source code)
6. Embedding Generation (for search)
7. FAISS Database Creation (vector search)
8. Final Assembly (copy to target structure)

Features:
- Automatic checkpoint/resume: Pipeline saves progress after each stage
- Resume from failures: Use --resume to continue from last successful stage
- Force restart: Use --force to ignore checkpoints and start fresh
- Stage selection: Use --start/--end to run specific stage ranges

Usage:
    # Run complete pipeline
    python data_collection_pipeline.py

    # Resume after failure
    python data_collection_pipeline.py --resume

    # Force restart from beginning
    python data_collection_pipeline.py --force

    # Run specific stages
    python data_collection_pipeline.py --start extension_data --end settings
"""

import argparse
import asyncio
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import toml


class PipelineConfig:
    """Configuration management for the data collection pipeline."""

    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path(__file__).parent / "pipeline_config.toml"
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from TOML file or create default."""
        if self.config_file.exists():
            with open(self.config_file, "r") as f:
                return toml.load(f)
        else:
            # Create default configuration
            default_config = {
                "input": {
                    "repo": [
                        {
                            "url": "https://gitlab-master.nvidia.com/omniverse/kit-github/kit-app-template.git",
                            "branch": "production/109.0",
                            "exts_path": "extscache",
                            "has_version": True,
                            "app_template_path": "templates/apps",
                        },
                    ],
                },
                "output": {
                    "work_dir": "../../../../_pipeline_output/kit",
                    "target_dir": "../src/kit_fns/data",
                    "keep_intermediates": False,
                },
                "processing": {
                    "include_source_code": False,
                    "code_examples_min_lines": 50,
                    "code_examples_min_complexity": 3,
                    "max_extensions": -1,
                    "parallel_workers": 4,
                    "excluded_modules": [
                        "ogn",
                        "pip_prebundle",
                        "pip_aiq_prebundle",
                        "pip_core_prebundle",
                        "debugpy",
                        "reportlab",
                        "xlsxwriter",
                    ],
                },
                "embeddings": {
                    "model": "nvidia/nv-embedqa-e5-v5",
                    "batch_size": 50,
                    "nvidia_api_key": "${NVIDIA_API_KEY}",
                    "endpoint_url": "",
                    "max_tokens": 500,
                    "encoding_model": "cl100k_base",
                },
                "logging": {"level": "INFO", "log_file": "pipeline.log"},
            }

            # Save default config
            self.config_file.parent.mkdir(exist_ok=True)
            with open(self.config_file, "w") as f:
                toml.dump(default_config, f)

            return default_config

    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation."""
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        # Handle environment variable substitution
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, default)

        return value


def get_build_platform_path() -> str:
    """
    Detect the current platform and architecture to construct the build path.

    Returns:
        str: Platform-architecture string (e.g., 'linux-x86_64', 'windows-x86_64')
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Normalize platform name
    if system == "linux":
        platform_name = "linux"
    elif system == "windows":
        platform_name = "windows"
    elif system == "darwin":
        platform_name = "macos"
    else:
        platform_name = system

    # Normalize architecture
    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("aarch64", "arm64"):
        arch = "aarch64"
    else:
        arch = machine

    return f"{platform_name}-{arch}"


def find_build_directory(base_path: Path, build_type: Optional[str] = None) -> Optional[Path]:
    """
    Find the appropriate build directory, trying different configurations.

    Args:
        base_path: Base path where _build directory exists
        build_type: Preferred build type ('release' or 'debug'). If None, tries both.

    Returns:
        Path to the build directory if found, None otherwise
    """
    build_base = base_path / "_build"
    if not build_base.exists():
        return None

    platform_path = get_build_platform_path()

    # Determine build type priority
    if build_type:
        build_types = [build_type]
    else:
        # Try release first, then debug
        build_types = ["release", "debug"]

    # Try each combination
    for btype in build_types:
        candidate = build_base / platform_path / btype
        if candidate.exists():
            return candidate

    # If no exact match, try to find any existing build directory
    if build_base.exists():
        for platform_dir in build_base.iterdir():
            if platform_dir.is_dir() and not platform_dir.name.startswith("."):
                for build_dir in platform_dir.iterdir():
                    if build_dir.is_dir() and build_dir.name in ["release", "debug"]:
                        return build_dir

    return None


class PipelineStage:
    """Base class for pipeline stages."""

    def __init__(self, name: str, config: PipelineConfig, logger: logging.Logger):
        self.name = name
        self.config = config
        self.logger = logger
        self.start_time = None
        self.end_time = None
        self.resume_mode = False  # Set to True when resuming from a checkpoint

    def run(self) -> bool:
        """Run the stage. Returns True if successful."""
        self.logger.info(f"Starting stage: {self.name}")
        self.start_time = time.time()

        try:
            success = self._execute()
            self.end_time = time.time()
            duration = self.end_time - self.start_time

            if success:
                self.logger.info(f"Stage {self.name} completed successfully in {duration:.1f}s")
            else:
                self.logger.error(f"Stage {self.name} failed after {duration:.1f}s")

            return success

        except Exception as e:
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.logger.error(f"Stage {self.name} failed with exception after {duration:.1f}s: {e}")
            return False

    def _execute(self) -> bool:
        """Execute the stage logic. Override in subclasses."""
        raise NotImplementedError

    def get_duration(self) -> Optional[float]:
        """Get stage duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class PreparationStage(PipelineStage):
    """Validate inputs and prepare output directories."""

    def _execute(self) -> bool:
        exts_paths = self.config.get("input.exts_paths")
        work_dir = self.config.get("output.work_dir")
        excluded = self.config.get("input.excluded_extensions", [])

        # Validate input directories
        for exts_path in exts_paths:
            if not Path(exts_path).exists():
                self.logger.error(f"Exts directory does not exist: {exts_path}")
                return False

            if not Path(exts_path).is_dir():
                self.logger.error(f"Exts path is not a directory: {exts_path}")
                return False

        # Create a symlink for each extension in the exts_paths
        exts_dir = Path(work_dir) / "all_exts"
        if exts_dir.exists():
            shutil.rmtree(exts_dir)
        exts_dir.mkdir(parents=True, exist_ok=True)
        unique_exts = set()
        for exts_path in exts_paths:
            for ext in os.listdir(exts_path):
                ext_name = ext.split("-")[0]
                if ext_name in unique_exts or ext_name in excluded:
                    continue
                ext_path = Path(exts_path).joinpath(ext).resolve()
                if os.path.isdir(ext_path):
                    symlink_path = Path(exts_dir).joinpath(ext)
                    symlink_path.symlink_to(ext_path, target_is_directory=True)
                    unique_exts.add(ext_name)

        self.logger.info(f"Created symlink for {len(unique_exts)} extensions")
        self.config.config["input"]["exts_path"] = str(exts_dir)

        kit_version = self.config.get("input.kit_version")
        if not kit_version:
            self.logger.error("Kit version not found")
            return False

        # Create output directories
        work_dir = Path(self.config.get("output.work_dir"))
        work_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each pipeline stage
        subdirs = ["extension_data", "code_examples", "settings", "embeddings", "faiss_databases"]

        for subdir in subdirs:
            (work_dir / subdir).mkdir(exist_ok=True)

        self.logger.info(f"Created work directories in: {work_dir}")
        return True


class PullRepoExtsStage(PipelineStage):
    """Clone repository and build it to get exts."""

    def _execute(self) -> bool:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from pull_repo_exts_pipeline.clone_and_build import clone_and_build_repo

            work_dir = self.config.get("output.work_dir")
            repos = self.config.get("input.repo")

            self.logger.info(f"Work directory: {work_dir}")

            kit_version = None
            app_template_path = None
            exts_paths = []

            for repo in repos:
                repo_url = repo.get("url")
                branch = repo.get("branch")
                exts_path = repo.get("exts_path", "exts")
                has_version = repo.get("has_version", False)
                repo_app_template_path = repo.get("app_template_path", None)

                self.logger.info(f"Running repo clone and build process for {repo_url}")
                self.logger.info(f"Branch: {branch}")
                self.logger.info(f"Has version: {has_version}")
                self.logger.info(f"App template path: {repo_app_template_path}")

                # Run the clone and build process
                paths_and_version = clone_and_build_repo(
                    work_dir=work_dir,
                    repo_url=repo_url,
                    branch=branch,
                    exts_path=exts_path,
                    app_template_path=repo_app_template_path,
                    has_version=has_version,
                    logger=self.logger,
                )

                if not paths_and_version:
                    self.logger.error("Failed to clone and build repo")
                    return False

                if not paths_and_version.get("exts_path"):
                    self.logger.error("No exts path returned from build process")
                    return False

                exts_paths.append(paths_and_version.get("exts_path"))

                if has_version:
                    kit_version = paths_and_version.get("kit_version")

                if repo_app_template_path:
                    app_template_path = paths_and_version.get("app_template_path")

            # Update the pipeline config to use this exts
            self.config.config["input"]["exts_paths"] = exts_paths
            self.config.config["input"]["kit_version"] = kit_version
            self.config.config["input"]["app_template_path"] = app_template_path
            self.logger.info(
                f"Updated pipeline config to use repo exts paths: {exts_paths} and kit version: {kit_version} and app template path: {app_template_path}"
            )

            return True

        except Exception as e:
            self.logger.error(f"Pull repo exts stage failed: {e}")
            return False


class ExtensionDataStage(PipelineStage):
    """Run the extension database builder pipeline."""

    def _execute(self) -> bool:
        # Import and run extension database builder
        try:
            from extension_data.build_extension_database import process_all_extensions

            exts_path = self.config.get("input.exts_path")
            work_dir = Path(self.config.get("output.work_dir"))
            output_dir = work_dir / "extension_data"
            include_source = self.config.get("processing.include_source_code", False)
            excluded_modules = self.config.get("processing.excluded_modules", [])

            self.logger.info(f"Processing extensions from: {exts_path}")
            self.logger.info(f"Output directory: {output_dir}")
            self.logger.info(f"Include source code: {include_source}")
            self.logger.info(f"Excluded modules: {excluded_modules}")

            # Run the extension processing
            process_all_extensions(
                extscache_path=exts_path,
                output_dir=str(output_dir),
                include_source_code=include_source,
                excluded_modules=excluded_modules,
            )

            # Verify outputs
            expected_files = [output_dir / "extensions_database.json", output_dir / "extensions_summary.json"]

            for expected_file in expected_files:
                if not expected_file.exists():
                    self.logger.error(f"Expected output file not found: {expected_file}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Extension data stage failed: {e}")
            return False


class CodeExamplesStage(PipelineStage):
    """Run the code examples extraction pipeline."""

    def _execute(self) -> bool:
        try:
            from code_example_pipeline.scan_extensions_codeatlas import analyze_all_extensions_async

            exts_path = self.config.get("input.exts_path")
            work_dir = Path(self.config.get("output.work_dir"))
            output_dir = work_dir / "code_examples"

            min_lines = self.config.get("processing.code_examples_min_lines")
            min_complexity = self.config.get("processing.code_examples_min_complexity")
            max_extensions = self.config.get("processing.max_extensions")
            scan_mode = self.config.get("processing.code_examples_scan_mode")
            excluded_modules = self.config.get("processing.excluded_modules", [])

            self.logger.info(f"Analyzing extensions for code examples")
            self.logger.info(f"Min lines: {min_lines}, Min complexity: {min_complexity}")
            self.logger.info(f"Excluded modules: {excluded_modules}")
            if max_extensions != -1:
                self.logger.info(f"Limited to {max_extensions} extensions")

            # Log resume mode
            if self.resume_mode:
                self.logger.info(f"Resume mode enabled - will skip already-processed extensions")

            # Change to output directory so extracted methods are saved there
            # Run code examples analysis
            asyncio.run(
                analyze_all_extensions_async(
                    output_dir=output_dir,
                    extensions_dir=exts_path,
                    max_extensions=max_extensions,
                    min_lines=min_lines,
                    min_complexity=min_complexity,
                    debug_mode=False,
                    scan_mode=scan_mode,
                    resume=self.resume_mode,
                    excluded_modules=excluded_modules,
                )
            )

            # Verify output
            expected_summary = output_dir / "extension_analysis_summary.json"
            if not expected_summary.exists():
                self.logger.error(f"Code examples summary not found: {expected_summary}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Code examples stage failed: {e}")
            return False


class SettingsStage(PipelineStage):
    """Run the settings extraction pipeline."""

    def _execute(self) -> bool:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from settings_pipeline.scan_extension_settings import ExtensionSettingsScanner

            exts_path = self.config.get("input.exts_path")
            work_dir = Path(self.config.get("output.work_dir"))
            output_dir = work_dir / "settings"

            self.logger.info(f"Scanning extensions for settings")

            # Create scanner and run
            scanner = ExtensionSettingsScanner(exts_path)
            scanner.scan_all_extensions()
            scanner.save_results(output_dir)

            # Verify outputs
            expected_files = [
                output_dir / "setting_summary.json",
                output_dir / "setting_summary_simple.json",
                output_dir / "setting_statistics.json",
            ]

            for expected_file in expected_files:
                if not expected_file.exists():
                    self.logger.error(f"Expected settings file not found: {expected_file}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Settings stage failed: {e}")
            return False


class EmbeddingsStage(PipelineStage):
    """Generate embeddings for extensions, code examples, and settings."""

    def _generate_embeddings_for_extensions(self, input_dir: Path, output_dir: Path) -> bool:
        try:
            from extension_data.generate_embeddings_descriptions import generate_extensions_descriptions

            extensions_dir = self.config.get("input.exts_path")
            database_file = input_dir / "extensions_database.json"
            generate_extensions_descriptions(
                Path(extensions_dir), database_file, output_dir, self.max_tokens, self.encoding_model
            )

            descriptions_file = output_dir / "extensions_descriptions.json"
            if not descriptions_file.exists():
                self.logger.error(f"Descriptions file not found: {descriptions_file}")
                return False
            embeddings_file = output_dir / "extensions_embeddings.json"

            from extension_data.generate_extension_embeddings import generate_embeddings_for_extensions

            generate_embeddings_for_extensions(
                descriptions_file,
                embeddings_file,
                self.api_key,
                self.endpoint_url,
                self.embeddings_model,
                self.batch_size,
            )

            if not embeddings_file.exists():
                self.logger.error(f"Embeddings file not found: {embeddings_file}")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings for extensions: {e}")
            return False

    def _generate_embeddings_for_code_examples(self, input_dir: Path, output_dir: Path) -> bool:
        try:
            from code_example_pipeline.generate_code_examples_embeddings import generate_embeddings_for_code_examples

            scan_mode = self.config.get("processing.code_examples_scan_mode")
            extracted_methods_dir = input_dir / f"extracted_methods_{scan_mode}"
            generate_embeddings_for_code_examples(
                extracted_methods_dir,
                output_dir,
                self.encoding_model,
                self.embeddings_model,
                self.api_key,
                self.endpoint_url,
                self.max_tokens,
                self.batch_size,
            )
            embeddings_file = output_dir / "code_examples_embeddings.json"
            if not embeddings_file.exists():
                self.logger.error(f"Embeddings file not found: {embeddings_file}")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings for code examples: {e}")
            return False

    def _generate_embeddings_for_settings(self, input_dir: Path, output_dir: Path) -> bool:
        try:
            from settings_pipeline.generate_settings_embeddings import generate_embeddings_for_settings

            settings_file = input_dir / "setting_summary.json"
            output_file = output_dir / "settings_embeddings.json"
            generate_embeddings_for_settings(
                settings_file, output_file, self.embeddings_model, self.api_key, self.endpoint_url, self.batch_size
            )

            if not output_file.exists():
                self.logger.error(f"Settings embeddings file not found: {output_file}")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings for settings: {e}")
            return False

    def _execute(self) -> bool:
        try:
            work_dir = Path(self.config.get("output.work_dir"))

            # Generate embeddings for each data type
            stages = [("extensions", "extension_data"), ("code_examples", "code_examples"), ("settings", "settings")]

            output_dir = work_dir / "embeddings"
            output_dir.mkdir(exist_ok=True)
            self.encoding_model = self.config.get("embeddings.encoding_model")
            self.embeddings_model = self.config.get("embeddings.model")
            self.api_key = self.config.get("embeddings.nvidia_api_key")
            self.endpoint_url = self.config.get("embeddings.endpoint_url")
            self.max_tokens = self.config.get("embeddings.max_tokens")
            self.batch_size = self.config.get("embeddings.batch_size")

            # Validate API key is set
            if not self.api_key:
                self.logger.error(
                    "NVIDIA_API_KEY is not set. Please set it using:\n"
                    "  export NVIDIA_API_KEY='your-api-key'\n"
                    "Or configure it in pipeline_config.toml"
                )
                return False

            for data_type, subdir in stages:
                self.logger.info(f"Generating embeddings for {data_type}")

                if not getattr(self, f"_generate_embeddings_for_{data_type}")(work_dir / subdir, output_dir):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Embeddings stage failed: {e}")
            return False


class FAISSStage(PipelineStage):
    """Build FAISS vector databases."""

    def _execute(self) -> bool:
        try:
            work_dir = Path(self.config.get("output.work_dir"))
            faiss_dir = work_dir / "faiss_databases"

            # Build FAISS databases for each data type
            data_types = ["extensions", "code_examples", "settings"]

            self.embeddings_model = self.config.get("embeddings.model")
            self.api_key = self.config.get("embeddings.nvidia_api_key")
            self.endpoint_url = self.config.get("embeddings.endpoint_url")

            # Validate API key is set
            if not self.api_key:
                self.logger.error(
                    "NVIDIA_API_KEY is not set. Please set it using:\n"
                    "  export NVIDIA_API_KEY='your-api-key'\n"
                    "Or configure it in pipeline_config.toml"
                )
                return False

            # Clean up old FAISS databases to ensure fresh build
            if faiss_dir.exists():
                self.logger.info("Removing old FAISS databases to ensure fresh build")
                for data_type in data_types:
                    old_faiss_path = faiss_dir / f"{data_type}_faiss"
                    if old_faiss_path.exists():
                        import shutil

                        shutil.rmtree(old_faiss_path)
                        self.logger.debug(f"Removed old FAISS database: {old_faiss_path}")

            for data_type in data_types:
                self.logger.info(f"Building FAISS database for {data_type}")

                faiss_output_path = faiss_dir / f"{data_type}_faiss"
                if not getattr(self, f"_build_faiss_for_{data_type}")(work_dir, faiss_output_path):
                    return False

                expected_files = [
                    faiss_output_path / "index.faiss",
                    faiss_output_path / "index.pkl",
                    faiss_output_path / "metadata.json",
                ]
                for expected_file in expected_files:
                    if not expected_file.exists():
                        self.logger.error(f"Expected FAISS file not found: {expected_file}")
                        return False

            return True
        except Exception as e:
            self.logger.error(f"FAISS stage failed: {e}")
            return False

    def _build_faiss_for_extensions(self, work_dir: Path, faiss_output_path: Path) -> bool:
        try:
            from extension_data.build_extensions_faiss_database import build_faiss_database

            embeddings_file = work_dir / "embeddings" / "extensions_embeddings.json"
            descriptions_file = work_dir / "embeddings" / "extensions_descriptions.json"
            database_file = work_dir / "extension_data" / "extensions_database.json"
            build_faiss_database(
                embeddings_file,
                descriptions_file,
                database_file,
                faiss_output_path,
                self.embeddings_model,
                self.api_key,
                self.endpoint_url,
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to build FAISS database for extensions: {e}")
            return False

    def _build_faiss_for_code_examples(self, work_dir: Path, faiss_output_path: Path) -> bool:
        try:
            from code_example_pipeline.build_code_examples_faiss_database import build_faiss_database

            scan_mode = self.config.get("processing.code_examples_scan_mode")
            extracted_methods_dir = work_dir / "code_examples" / f"extracted_methods_{scan_mode}"
            embeddings_file = work_dir / "embeddings" / "code_examples_embeddings.json"
            build_faiss_database(
                extracted_methods_dir,
                embeddings_file,
                faiss_output_path,
                self.embeddings_model,
                self.api_key,
                self.endpoint_url,
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to build FAISS database for code examples: {e}")
            return False

    def _build_faiss_for_settings(self, work_dir: Path, faiss_output_path: Path) -> bool:
        try:
            from settings_pipeline.build_settings_faiss_database import build_faiss_database

            embeddings_file = work_dir / "embeddings" / "settings_embeddings.json"
            settings_file = work_dir / "settings" / "setting_summary.json"
            build_faiss_database(
                embeddings_file,
                settings_file,
                faiss_output_path,
                self.embeddings_model,
                self.api_key,
                self.endpoint_url,
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to build FAISS database for settings: {e}")
            return False


class FinalAssemblyStage(PipelineStage):
    """Copy generated files to target directory structure."""

    def _execute(self) -> bool:
        try:
            work_dir = Path(self.config.get("output.work_dir"))
            target_dir = Path(self.config.get("output.target_dir"))
            kit_version = self.config.get("input.kit_version")
            app_template_path = self.config.get("input.app_template_path")
            major_minor_version = ".".join(kit_version.split("+")[0].split(".")[:2])

            self.logger.info(f"Assembling final output to: {target_dir}")

            # Ensure target directory exists
            target_dir.mkdir(parents=True, exist_ok=True)

            # Copy files to target structure
            file_mappings = self._get_file_mappings(work_dir, target_dir, major_minor_version, app_template_path)

            for src, dst in file_mappings:
                if src.exists():
                    # Create destination directory
                    dst.parent.mkdir(parents=True, exist_ok=True)

                    if src.is_file():
                        shutil.copy2(src, dst)
                        self.logger.debug(f"Copied: {src.name} -> {dst.relative_to(target_dir)}")
                    else:
                        # Copy directory
                        if dst.exists():
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                        self.logger.debug(f"Copied dir: {src.name} -> {dst.relative_to(target_dir)}")
                else:
                    self.logger.warning(f"Source file not found: {src}")

            # Write VERSION.md
            version_file = target_dir / major_minor_version / "VERSION.md"
            with open(version_file, "w") as f:
                f.write(kit_version)

            # Generate completion report
            self._generate_completion_report(work_dir, target_dir)

            # Clean up intermediate files if configured
            if not self.config.get("output.keep_intermediates", True):
                self.logger.info("Cleaning up intermediate files")
                shutil.rmtree(work_dir)

            return True

        except Exception as e:
            self.logger.error(f"Final assembly stage failed: {e}")
            return False

    def _get_file_mappings(
        self, work_dir: Path, target_dir: Path, kit_version: str, app_template_path: str
    ) -> List[Tuple[Path, Path]]:
        """Get list of (source, destination) file mappings."""
        mappings = []
        target_version_dir = target_dir / kit_version

        # App templates
        app_templates_src = Path(app_template_path)
        app_templates_dst = target_version_dir / "app_templates"
        mappings.append((app_templates_src, app_templates_dst))

        # Instructions
        instructions_src = Path(__file__).parent / "instructions"
        instructions_dst = target_version_dir / "instructions"
        mappings.append((instructions_src, instructions_dst))

        # Extensions data
        ext_src = work_dir / "extension_data"
        ext_dst = target_version_dir / "extensions"
        mappings.extend(
            [
                (ext_src / "extensions_database.json", ext_dst / "extensions_database.json"),
                (ext_src / "extensions_summary.json", ext_dst / "extensions_summary.json"),
                (ext_src / "codeatlas", ext_dst / "codeatlas"),
                (ext_src / "api_docs", ext_dst / "api_docs"),
            ]
        )

        # Code examples
        code_src = work_dir / "code_examples"
        code_dst = target_version_dir / "code_examples"
        mappings.extend(
            [
                (code_src / "extension_analysis_full.json", code_dst / "extension_analysis_summary.json"),
                (code_src / "extracted_methods", code_dst / "extracted_methods"),
            ]
        )

        # Settings
        settings_src = work_dir / "settings"
        settings_dst = target_version_dir / "settings"
        mappings.extend(
            [
                (settings_src / "setting_summary.json", settings_dst / "setting_summary.json"),
                (settings_src / "setting_summary_simple.json", settings_dst / "setting_summary_simple.json"),
                (settings_src / "setting_statistics.json", settings_dst / "setting_statistics.json"),
            ]
        )

        # FAISS databases
        faiss_src = work_dir / "faiss_databases"
        mappings.extend(
            [
                (faiss_src / "extensions_faiss", ext_dst / "extensions_faiss"),
                (faiss_src / "code_examples_faiss", code_dst / "code_examples_faiss"),
                (faiss_src / "settings_faiss", settings_dst / "settings_faiss"),
            ]
        )

        return mappings

    def _generate_completion_report(self, work_dir: Path, target_dir: Path):
        """Generate a completion report with statistics."""
        report = {
            "pipeline_completion": {
                "timestamp": datetime.now().isoformat(),
                "work_directory": str(work_dir),
                "target_directory": str(target_dir),
                "config": self.config.config,
            },
            "outputs_generated": {},
            "statistics": {},
        }

        # Count generated files
        if target_dir.exists():
            for subdir in ["extensions", "code_examples", "settings"]:
                subdir_path = target_dir / subdir
                if subdir_path.exists():
                    files = list(subdir_path.rglob("*"))
                    report["outputs_generated"][subdir] = len([f for f in files if f.is_file()])

        # Save report
        report_file = target_dir / "pipeline_completion_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Pipeline completion report saved: {report_file}")


class DataCollectionPipeline:
    """Main pipeline orchestrator."""

    def __init__(self, config_file: Optional[Path] = None):
        self.config = PipelineConfig(config_file)
        self.logger = self._setup_logging()

        self.pre_stages = [
            PullRepoExtsStage("pull_repo_exts", self.config, self.logger),
            PreparationStage("preparation", self.config, self.logger),
        ]

        # Define pipeline stages
        self.stages = [
            ExtensionDataStage("extension_data", self.config, self.logger),
            CodeExamplesStage("code_examples", self.config, self.logger),
            SettingsStage("settings", self.config, self.logger),
            EmbeddingsStage("embeddings", self.config, self.logger),
            FAISSStage("faiss", self.config, self.logger),
            FinalAssemblyStage("final_assembly", self.config, self.logger),
        ]

        self.checkpoint_file = Path(self.config.get("output.work_dir")) / ".pipeline_checkpoint.json"

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("DataCollectionPipeline")
        logger.setLevel(self.config.get("logging.level", "INFO"))

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file = self.config.get("logging.log_file")
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)

        return logger

    def _save_checkpoint(self, completed_stages: List[str]):
        """Save pipeline checkpoint after successful stage completion."""
        try:
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "completed_stages": completed_stages,
                "config_snapshot": {
                    "exts_path": self.config.get("input.exts_path"),
                    "kit_version": self.config.get("input.kit_version"),
                    "app_template_path": self.config.get("input.app_template_path"),
                    "work_dir": self.config.get("output.work_dir"),
                },
            }

            # Ensure checkpoint directory exists
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

            self.logger.debug(f"Checkpoint saved: {len(completed_stages)} stages completed")

        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self) -> Optional[Dict]:
        """Load pipeline checkpoint if it exists."""
        try:
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file, "r") as f:
                    checkpoint = json.load(f)

                self.logger.info(f"Found checkpoint from {checkpoint['timestamp']}")
                self.logger.info(f"Completed stages: {len(checkpoint['completed_stages'])}")

                return checkpoint

        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")

        return None

    def _restore_config_from_checkpoint(self, checkpoint: Dict):
        """Restore configuration from checkpoint."""
        try:
            config_snapshot = checkpoint.get("config_snapshot", {})

            if config_snapshot.get("exts_path"):
                self.config.config["input"]["exts_path"] = config_snapshot["exts_path"]

            if config_snapshot.get("kit_version"):
                self.config.config["input"]["kit_version"] = config_snapshot["kit_version"]

            if config_snapshot.get("app_template_path"):
                self.config.config["input"]["app_template_path"] = config_snapshot["app_template_path"]

            self.logger.info("Restored configuration from checkpoint")

        except Exception as e:
            self.logger.warning(f"Failed to restore config from checkpoint: {e}")

    def _clear_checkpoint(self):
        """Clear checkpoint file."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                self.logger.debug("Checkpoint cleared")
        except Exception as e:
            self.logger.warning(f"Failed to clear checkpoint: {e}")

    def run(
        self,
        start_stage: Optional[str] = None,
        end_stage: Optional[str] = None,
        resume: bool = False,
        force: bool = False,
    ) -> bool:
        """Run the complete pipeline or a subset of stages.

        Args:
            start_stage: Name of stage to start from (default: first stage)
            end_stage: Name of stage to end at (default: last stage)
            resume: Resume from last checkpoint (overrides start_stage)
            force: Ignore checkpoints and start fresh

        Returns:
            True if all stages completed successfully
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Kit Extensions Data Collection Pipeline")
        self.logger.info("=" * 80)

        # Handle force flag - clear checkpoint
        if force:
            self.logger.info("Force mode enabled - clearing checkpoint")
            self._clear_checkpoint()

        # Check if auto-resume is enabled in config
        auto_resume = self.config.get("advanced.resume_on_failure", False)
        if auto_resume and not force and not resume:
            self.logger.info("Auto-resume enabled in config - checking for checkpoint")
            resume = True

        # Handle resume flag - load checkpoint
        checkpoint = None
        completed_stages = []
        if resume and not force:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                completed_stages = checkpoint.get("completed_stages", [])
                self._restore_config_from_checkpoint(checkpoint)
                self.logger.info(
                    f"Resuming pipeline from checkpoint ({len(completed_stages)} stages already completed)"
                )
            else:
                self.logger.warning("Resume requested but no checkpoint found - starting from beginning")

        # Build list of stages to run
        pipeline_start = time.time()

        # Determine which stages to include
        if resume and checkpoint and not start_stage and not end_stage:
            # When resuming from checkpoint, run all incomplete stages
            # Build the complete list of all stages in order
            all_stages_in_order = self.pre_stages + self.stages

            # Filter to only run stages that haven't been completed yet
            stages_to_run = [s for s in all_stages_in_order if s.name not in completed_stages]

            if not stages_to_run:
                self.logger.info("All stages already completed according to checkpoint!")
                return True

        elif start_stage or end_stage:
            # User specified explicit stage range
            start_idx = 0
            end_idx = len(self.stages) - 1

            if start_stage:
                found = False
                # Check if start_stage is in pre_stages
                for i, stage in enumerate(self.pre_stages):
                    if stage.name == start_stage:
                        # Starting from a pre_stage - include pre_stages starting from this one
                        stages_to_run = self.pre_stages[i:] + self.stages
                        found = True
                        break

                if not found:
                    # Check if start_stage is in main stages
                    for i, stage in enumerate(self.stages):
                        if stage.name == start_stage:
                            start_idx = i
                            found = True
                            break

                    if not found:
                        self.logger.error(f"Unknown start stage: {start_stage}")
                        return False

                    stages_to_run = self.stages[start_idx:]

            if end_stage:
                # Trim stages_to_run to end at end_stage
                end_idx = -1
                for i, stage in enumerate(stages_to_run):
                    if stage.name == end_stage:
                        end_idx = i
                        break

                if end_idx == -1:
                    self.logger.error(f"Unknown end stage: {end_stage}")
                    return False

                stages_to_run = stages_to_run[: end_idx + 1]

            # If resuming with explicit range, still filter by completed stages
            if resume and checkpoint:
                stages_to_run = [s for s in stages_to_run if s.name not in completed_stages]

        else:
            # Normal run from beginning - include all pre_stages and main stages
            stages_to_run = self.pre_stages + self.stages

        if not stages_to_run:
            self.logger.info("No stages to run - all stages already completed!")
            return True

        self.logger.info(f"Running {len(stages_to_run)} stages: {[s.name for s in stages_to_run]}")
        if completed_stages:
            self.logger.info(f"Skipping {len(completed_stages)} already completed stages: {completed_stages}")

        # Run stages
        for i, stage in enumerate(stages_to_run, 1):
            self.logger.info(f"[{i}/{len(stages_to_run)}] Running stage: {stage.name}")

            # Set resume mode for stages when resuming from checkpoint
            # This allows stages to perform incremental resume (e.g., skip already-processed items)
            if resume and checkpoint:
                stage.resume_mode = True

            success = stage.run()
            if not success:
                self.logger.error(f"Pipeline failed at stage: {stage.name}")
                self.logger.info(f"You can resume from this point using: --resume")
                # Save checkpoint with completed stages before this failed one
                self._save_checkpoint(completed_stages)
                return False

            # Add to completed stages and save checkpoint
            completed_stages.append(stage.name)
            self._save_checkpoint(completed_stages)

        # Pipeline completion
        pipeline_duration = time.time() - pipeline_start
        self.logger.info("=" * 80)
        self.logger.info("Pipeline completed successfully!")
        self.logger.info(f"Total time: {pipeline_duration:.1f}s")

        # Stage timing summary
        self.logger.info("\nStage timing summary:")
        for stage in stages_to_run:
            duration = stage.get_duration()
            if duration:
                self.logger.info(f"  {stage.name}: {duration:.1f}s")

        self.logger.info("=" * 80)

        # Clear checkpoint on successful completion
        self._clear_checkpoint()

        return True

    def list_stages(self):
        """List all available pipeline stages."""
        print("Available pipeline stages:")
        for i, stage in enumerate(self.stages, 1):
            print(f"  {i}. {stage.name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Kit Extensions Data Collection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python data_collection_pipeline.py

  # Run specific stages
  python data_collection_pipeline.py --start extension_data --end settings

  # Resume from last checkpoint after failure
  python data_collection_pipeline.py --resume

  # Force restart from beginning (ignore checkpoint)
  python data_collection_pipeline.py --force

  # List available stages
  python data_collection_pipeline.py --list-stages
        """,
    )
    parser.add_argument("--config", type=Path, help="Path to pipeline configuration file")

    parser.add_argument("--start", type=str, help="Start from specific stage")

    parser.add_argument("--end", type=str, help="End at specific stage")

    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint (automatic recovery)")

    parser.add_argument("--force", action="store_true", help="Force restart from beginning, ignoring any checkpoint")

    parser.add_argument("--list-stages", action="store_true", help="List available pipeline stages and exit")

    args = parser.parse_args()

    # Create pipeline
    pipeline = DataCollectionPipeline(args.config)

    # Handle list stages
    if args.list_stages:
        pipeline.list_stages()
        return

    # Run pipeline
    success = pipeline.run(
        start_stage=args.start,
        end_stage=args.end,
        resume=args.resume,
        force=args.force,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
