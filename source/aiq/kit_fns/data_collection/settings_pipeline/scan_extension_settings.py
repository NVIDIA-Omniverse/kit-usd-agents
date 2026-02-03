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
Comprehensive script to scan all Omniverse Kit extensions and extract settings.
Extracts settings from extension.toml files and source code, converting to canonical slash format.
"""

import argparse
import ast
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import toml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ExtensionSettingsScanner:
    """Scanner for extracting settings from Omniverse Kit extensions."""

    def __init__(self, extensions_dir: str):
        """Initialize the scanner with extensions directory path."""
        self.extensions_dir = Path(extensions_dir)
        self.settings_data = defaultdict(
            lambda: {
                "default_value": None,
                "type": None,
                "description": None,
                "documentation": None,
                "found_in": [],  # List of {"file": path, "line": num, "context": "toml|code"}
                "extensions": set(),  # Set of extensions that use this setting
            }
        )

    def convert_dot_to_slash(self, setting_path: str) -> str:
        """Convert dot notation to canonical slash notation.

        Examples:
            exts."omni.kit.window.console".autoShow -> /exts/omni.kit.window.console/autoShow
            exts.omni.kit.viewport.menubar.timeline.buttonOrder -> /exts/omni.kit.viewport.menubar.timeline/buttonOrder
            app.window.title -> /app/window/title
        """
        # Remove leading slash if present
        setting_path = setting_path.lstrip("/")

        # Handle quoted extension names in exts.* pattern
        if setting_path.startswith("exts."):
            # Pattern 1: exts."extension.name".setting
            match = re.match(r'exts\.["\']([\w\.-]+)["\'](\.(.+))?', setting_path)
            if match:
                ext_name = match.group(1)
                setting_name = match.group(3) if match.group(3) else ""
                if setting_name:
                    # Convert remaining dots in setting_name to slashes
                    setting_name = setting_name.replace(".", "/")
                    return f"/exts/{ext_name}/{setting_name}"
                else:
                    # Just the extension, no additional path
                    return f"/exts/{ext_name}"
            else:
                # Pattern 2: exts.omni.kit.extension.name.setting
                # Need to identify the extension name (usually contains multiple dots)
                # Strategy: Extension names typically follow pattern like omni.kit.* or similar
                # We'll look for known extension patterns
                parts = setting_path.split(".")
                if len(parts) > 1:
                    # Try to identify extension boundary
                    # Extensions typically have 3-5 dot-separated components
                    # Look for where the extension name likely ends

                    # Common extension name patterns
                    ext_parts = []
                    idx = 1  # Start after 'exts'

                    # Build extension name by looking for typical patterns
                    while idx < len(parts):
                        part = parts[idx]
                        ext_parts.append(part)

                        # Check if we've likely found the full extension name
                        # Extension names typically end before settings paths like 'enabled', 'autoLoad', etc.
                        if idx + 1 < len(parts):
                            next_part = parts[idx + 1]
                            # Common setting name indicators
                            if (
                                next_part in ["enabled", "autoLoad", "priority", "order"]
                                or next_part[0].isupper()
                                or (
                                    len(ext_parts) >= 3
                                    and not next_part in ["kit", "omni", "ui", "core", "window", "viewport"]
                                )
                            ):
                                # We've likely found the end of the extension name
                                break
                        idx += 1

                    ext_name = ".".join(ext_parts)
                    remaining = ".".join(parts[idx + 1 :]) if idx + 1 < len(parts) else ""

                    if remaining:
                        return f"/exts/{ext_name}/{remaining.replace('.', '/')}"
                    else:
                        return f"/exts/{ext_name}"

        # For non-exts paths, simply replace dots with slashes
        return "/" + setting_path.replace(".", "/")

    def extract_toml_comment(self, lines: List[str], line_num: int) -> Optional[str]:
        """Extract comment documentation above a setting line in TOML."""
        comments = []
        current_line = line_num - 1

        while current_line >= 0:
            line = lines[current_line].strip()
            if line.startswith("#"):
                # Remove # and leading/trailing whitespace
                comment = line.lstrip("#").strip()
                comments.insert(0, comment)
                current_line -= 1
            elif not line:  # Empty line, continue checking
                current_line -= 1
            else:
                # Non-comment, non-empty line - stop
                break

        return " ".join(comments) if comments else None

    def parse_toml_value(self, value: Any) -> Tuple[Any, str]:
        """Parse TOML value and determine its type."""
        if isinstance(value, bool):
            return value, "bool"
        elif isinstance(value, int):
            return value, "int"
        elif isinstance(value, float):
            return value, "float"
        elif isinstance(value, str):
            return value, "string"
        elif isinstance(value, list):
            return value, "array"
        elif isinstance(value, dict):
            return value, "object"
        else:
            return str(value), "unknown"

    def scan_extension_toml(self, toml_path: Path, extension_name: str):
        """Scan extension.toml file for settings."""
        try:
            with open(toml_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()

            # Parse TOML
            data = toml.loads(content)

            # Extract the clean extension name (without version)
            # e.g., "omni.kit.viewport.menubar.timeline-1.0.2" -> "omni.kit.viewport.menubar.timeline"
            clean_ext_name = extension_name.split("-")[0] if "-" in extension_name else extension_name
            clean_ext_name = clean_ext_name.split("+")[0] if "+" in clean_ext_name else clean_ext_name

            # Extract settings from [settings] section
            if "settings" in data:
                self._process_settings_dict(data["settings"], toml_path, lines, extension_name, "", clean_ext_name)

            # Also check for inline settings in dependencies
            if "dependencies" in data:
                for dep_name, dep_config in data.get("dependencies", {}).items():
                    if isinstance(dep_config, dict) and "settings" in dep_config:
                        self._process_settings_dict(
                            dep_config["settings"], toml_path, lines, extension_name, "", clean_ext_name
                        )

            logger.debug(f"Scanned TOML: {toml_path}")

        except Exception as e:
            logger.warning(f"Error scanning TOML {toml_path}: {e}")

    def _process_settings_dict(
        self,
        settings: Dict,
        file_path: Path,
        lines: List[str],
        extension_name: str,
        prefix: str = "",
        clean_ext_name: str = None,
    ):
        """Process a settings dictionary from TOML."""
        for key, value in settings.items():
            if isinstance(value, dict) and not all(k.startswith("_") for k in value.keys()):
                # Build the new prefix
                new_prefix = f"{prefix}.{key}" if prefix else key

                # Special handling for exts.* paths - check if key is an extension name
                if prefix == "exts" and ("." in key or key == clean_ext_name):
                    # This is likely an extension name, process its contents as settings
                    for setting_key, setting_value in value.items():
                        full_setting_key = f"exts.{key}.{setting_key}"
                        canonical_key = f"/exts/{key}/{setting_key.replace('.', '/')}"

                        # Find line number for documentation
                        line_num = None
                        for i, line in enumerate(lines):
                            if setting_key in line and "=" in line:
                                line_num = i + 1
                                break

                        # Extract documentation comment if available
                        doc = None
                        if line_num:
                            doc = self.extract_toml_comment(lines, line_num - 1)

                        # Parse value and type
                        parsed_value, value_type = self.parse_toml_value(setting_value)

                        self._store_setting(
                            canonical_key, parsed_value, value_type, file_path, lines, line_num, extension_name, doc
                        )
                else:
                    # Check if this is actually a nested settings structure or just a dict value
                    has_nested_dicts = any(isinstance(v, dict) for k, v in value.items() if not k.startswith("_"))

                    # Don't create entries for intermediate paths
                    if has_nested_dicts:
                        # Nested settings - recurse
                        self._process_settings_dict(value, file_path, lines, extension_name, new_prefix, clean_ext_name)
                    else:
                        # This dict might be a value or leaf settings
                        for sub_key, sub_value in value.items():
                            full_key = f"{new_prefix}.{sub_key}"
                            canonical_key = self.convert_dot_to_slash_with_context(full_key, clean_ext_name)

                            # Find line number
                            line_num = None
                            for i, line in enumerate(lines):
                                if sub_key in line and "=" in line:
                                    line_num = i + 1
                                    break

                            doc = None
                            if line_num:
                                doc = self.extract_toml_comment(lines, line_num - 1)

                            parsed_value, value_type = self.parse_toml_value(sub_value)
                            self._store_setting(
                                canonical_key, parsed_value, value_type, file_path, lines, line_num, extension_name, doc
                            )
            else:
                # Actual setting (leaf node)
                full_key = f"{prefix}.{key}" if prefix else key
                canonical_key = self.convert_dot_to_slash_with_context(full_key, clean_ext_name)

                # Find line number for documentation
                line_num = None
                for i, line in enumerate(lines):
                    if key in line and "=" in line:
                        line_num = i + 1
                        break

                # Extract documentation comment if available
                doc = None
                if line_num:
                    doc = self.extract_toml_comment(lines, line_num - 1)

                # Parse value and type
                parsed_value, value_type = self.parse_toml_value(value)

                self._store_setting(
                    canonical_key, parsed_value, value_type, file_path, lines, line_num, extension_name, doc
                )

    def convert_dot_to_slash_with_context(self, setting_path: str, clean_ext_name: str = None) -> str:
        """Convert dot notation to canonical slash notation with context awareness.

        Args:
            setting_path: The setting path to convert
            clean_ext_name: The clean extension name (without version) if known
        """
        # Remove leading slash if present
        setting_path = setting_path.lstrip("/")

        # Handle exts.* patterns when we know the extension name
        if setting_path.startswith("exts.") and clean_ext_name:
            # Remove 'exts.' prefix
            path_without_exts = setting_path[5:]

            # Check if path starts with extension name components
            if path_without_exts.startswith(clean_ext_name):
                # Remove the extension name from the path
                remaining = path_without_exts[len(clean_ext_name) :]
                remaining = remaining.lstrip(".")

                if remaining:
                    return f"/exts/{clean_ext_name}/{remaining.replace('.', '/')}"
                else:
                    # This shouldn't be stored as it's just the extension path
                    return f"/exts/{clean_ext_name}"

        # Fall back to the original conversion method
        return self.convert_dot_to_slash(setting_path)

    def _store_setting(
        self,
        canonical_key: str,
        parsed_value: Any,
        value_type: str,
        file_path: Path,
        lines: Any,
        line_num: Optional[int],
        extension_name: str,
        doc: Optional[str],
    ):
        """Store a setting in the settings data."""
        # Skip partial paths and warn about them
        partial_paths = ["/exts", "/app", "/persistent", "/ext"]
        if canonical_key in partial_paths:
            file_ref = str(file_path.relative_to(self.extensions_dir))
            if line_num:
                file_ref = f"{file_ref}@{line_num}"
            logger.warning(f"Skipping partial path '{canonical_key}' found in {file_ref}")
            logger.warning(f"  Context: extension={extension_name}, value={parsed_value}, type={value_type}")
            # Only try to show line content if lines is a list
            if isinstance(lines, list) and line_num and line_num > 0 and line_num <= len(lines):
                logger.warning(f"  Line content: {lines[line_num-1].strip()}")
            return

        # Update settings data
        setting_info = self.settings_data[canonical_key]
        if setting_info["default_value"] is None:
            setting_info["default_value"] = parsed_value
            setting_info["type"] = value_type

        if doc and not setting_info["documentation"]:
            setting_info["documentation"] = doc

        # Simplified found_in format: "file@line"
        file_ref = str(file_path.relative_to(self.extensions_dir))
        if line_num:
            file_ref = f"{file_ref}@{line_num}"

        setting_info["found_in"].append(file_ref)
        setting_info["extensions"].add(extension_name)

    def scan_python_file(self, py_path: Path, extension_name: str):
        """Scan Python file for settings usage."""
        try:
            with open(py_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()

            # Pattern for carb.settings usage
            patterns = [
                # settings.get("/path/to/setting")
                (r'settings\.get(?:_as_\w+)?\s*\(\s*["\']([^"\']+)["\']', "get"),
                # settings.set("/path/to/setting", value)
                (r'settings\.set\s*\(\s*["\']([^"\']+)["\'](?:\s*,\s*([^,\)]+))?', "set"),
                # Direct string literals that look like settings paths
                (r'["\']/(exts?|app|persistent|ext)/([^"\']+)["\']', "literal"),
            ]

            for line_num, line in enumerate(lines, 1):
                for pattern, context_type in patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        setting_path = match.group(1)

                        # Ensure it starts with /
                        if not setting_path.startswith("/"):
                            setting_path = "/" + setting_path

                        # It's already in slash format from code
                        canonical_key = setting_path

                        # Try to extract default value if it's a set operation
                        default_value = None
                        value_type = None
                        if context_type == "set" and len(match.groups()) > 1:
                            value_str = match.group(2)
                            if value_str:
                                default_value, value_type = self._parse_python_value(value_str.strip())

                        # Skip partial paths
                        if canonical_key in ["/exts", "/app", "/persistent", "/ext"]:
                            logger.debug(
                                f"Skipping partial path '{canonical_key}' in Python file {py_path.name}@{line_num}"
                            )
                            continue

                        # Update settings data
                        setting_info = self.settings_data[canonical_key]

                        if default_value is not None and setting_info["default_value"] is None:
                            setting_info["default_value"] = default_value
                            setting_info["type"] = value_type

                        # Simplified found_in format: "file@line"
                        file_ref = f"{str(py_path.relative_to(self.extensions_dir))}@{line_num}"
                        setting_info["found_in"].append(file_ref)

                        setting_info["extensions"].add(extension_name)

            logger.debug(f"Scanned Python: {py_path}")

        except Exception as e:
            logger.warning(f"Error scanning Python file {py_path}: {e}")

    def scan_cpp_file(self, cpp_path: Path, extension_name: str):
        """Scan C++ file for settings usage."""
        try:
            with open(cpp_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()

            # Patterns for C++ Carbonite settings
            patterns = [
                # settings->get<Type>("/path/to/setting")
                (r'settings->get(?:Bool|Int|Float|String)?\s*(?:<[^>]+>)?\s*\(\s*"([^"]+)"', "get"),
                # settings->set("/path/to/setting", value)
                (r'settings->set(?:Bool|Int|Float|String)?\s*\(\s*"([^"]+)"', "set"),
                # String literals that look like settings paths
                (r'"/(exts?|app|persistent|ext)/([^"]+)"', "literal"),
            ]

            for line_num, line in enumerate(lines, 1):
                for pattern, context_type in patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        setting_path = match.group(1)

                        # Ensure it starts with /
                        if not setting_path.startswith("/"):
                            setting_path = "/" + setting_path

                        canonical_key = setting_path

                        # Skip partial paths
                        if canonical_key in ["/exts", "/app", "/persistent", "/ext"]:
                            logger.debug(
                                f"Skipping partial path '{canonical_key}' in C++ file {cpp_path.name}@{line_num}"
                            )
                            continue

                        # Update settings data
                        setting_info = self.settings_data[canonical_key]

                        # Simplified found_in format: "file@line"
                        file_ref = f"{str(cpp_path.relative_to(self.extensions_dir))}@{line_num}"
                        setting_info["found_in"].append(file_ref)

                        setting_info["extensions"].add(extension_name)

            logger.debug(f"Scanned C++: {cpp_path}")

        except Exception as e:
            logger.warning(f"Error scanning C++ file {cpp_path}: {e}")

    def _parse_python_value(self, value_str: str) -> Tuple[Any, str]:
        """Parse a Python value string to extract value and type."""
        try:
            # Try to evaluate as Python literal
            value = ast.literal_eval(value_str)
            if isinstance(value, bool):
                return value, "bool"
            elif isinstance(value, int):
                return value, "int"
            elif isinstance(value, float):
                return value, "float"
            elif isinstance(value, str):
                return value, "string"
            elif isinstance(value, (list, tuple)):
                return list(value), "array"
            elif isinstance(value, dict):
                return value, "object"
        except:
            # Check for known constants
            if value_str.lower() in ("true", "false"):
                return value_str.lower() == "true", "bool"
            elif value_str.lower() == "none":
                return None, "null"

            # Try to determine type from content
            if value_str.isdigit() or (value_str.startswith("-") and value_str[1:].isdigit()):
                return int(value_str), "int"
            elif re.match(r"^-?\d+\.\d+$", value_str):
                return float(value_str), "float"

        return value_str, "unknown"

    def scan_extension(self, extension_path: Path):
        """Scan a single extension directory."""
        extension_name = extension_path.name
        logger.info(f"Scanning extension: {extension_name}")

        # Scan extension.toml
        toml_path = extension_path / "extension.toml"
        if toml_path.exists():
            self.scan_extension_toml(toml_path, extension_name)

        # Scan config directory for additional TOML files
        config_dir = extension_path / "config"
        if config_dir.exists():
            for toml_file in config_dir.glob("*.toml"):
                self.scan_extension_toml(toml_file, extension_name)

        # Scan Python files
        for py_file in extension_path.rglob("*.py"):
            # Skip test files and __pycache__
            if "__pycache__" not in str(py_file) and "test" not in py_file.name.lower():
                self.scan_python_file(py_file, extension_name)

        # Scan C++ files
        for cpp_ext in ["*.cpp", "*.cc", "*.cxx", "*.h", "*.hpp"]:
            for cpp_file in extension_path.rglob(cpp_ext):
                self.scan_cpp_file(cpp_file, extension_name)

    def scan_all_extensions(self):
        """Scan all extensions in the directory."""
        if not self.extensions_dir.exists():
            logger.error(f"Extensions directory does not exist: {self.extensions_dir}")
            return

        # Get all extension directories
        extension_dirs = [d for d in self.extensions_dir.iterdir() if d.is_dir()]
        total = len(extension_dirs)

        logger.info(f"Found {total} extensions to scan")

        for i, ext_dir in enumerate(extension_dirs, 1):
            logger.info(f"Progress: {i}/{total} - Scanning {ext_dir.name}")
            try:
                self.scan_extension(ext_dir)
            except Exception as e:
                logger.error(f"Error scanning extension {ext_dir.name}: {e}")

    def prepare_output(self) -> Dict[str, Any]:
        """Prepare the final output dictionary."""
        output = {}

        for setting_key, setting_info in self.settings_data.items():
            # Convert set to sorted list for JSON serialization
            extensions_list = sorted(list(setting_info["extensions"]))

            output[setting_key] = {
                "default_value": setting_info["default_value"],
                "type": setting_info["type"],
                "description": setting_info["description"],
                "documentation": setting_info["documentation"],
                "extensions": extensions_list,
                "found_in": setting_info["found_in"],
                "usage_count": len(setting_info["found_in"]),
            }

        return output

    def save_results(self, output_dir: Path):
        """Save scan results to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "setting_summary.json"

        # Prepare output
        output_data = self.prepare_output()

        # Add metadata
        final_output = {
            "metadata": {
                "total_settings": len(output_data),
                "total_extensions_scanned": len(
                    set(ext for s in self.settings_data.values() for ext in s["extensions"])
                ),
                "scan_directory": str(self.extensions_dir),
                "version": "1.0.0",
            },
            "settings": output_data,
        }

        # Save to JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_file}")

        # Also save a simplified version for quick lookup
        simple_output = {}
        for key, info in output_data.items():
            simple_output[key] = {
                "default": info["default_value"],
                "type": info["type"],
                "extensions": info["extensions"][:5] if len(info["extensions"]) > 5 else info["extensions"],
            }

        simple_file = output_dir / "setting_summary_simple.json"
        with open(simple_file, "w", encoding="utf-8") as f:
            json.dump(simple_output, f, indent=2, ensure_ascii=False)

        logger.info(f"Simple summary saved to: {simple_file}")

        # Generate statistics
        self.generate_statistics(output_data, output_dir)

    def generate_statistics(self, output_data: Dict, output_dir: Path):
        """Generate statistics about the settings."""
        stats = {
            "total_settings": len(output_data),
            "settings_by_type": defaultdict(int),
            "settings_by_prefix": defaultdict(int),
            "most_used_settings": [],
            "extensions_with_most_settings": defaultdict(int),
        }

        # Analyze settings
        for key, info in output_data.items():
            # By type
            stats["settings_by_type"][info["type"] or "unknown"] += 1

            # By prefix
            prefix = key.split("/")[1] if "/" in key else "root"
            stats["settings_by_prefix"][prefix] += 1

            # Count by extension
            for ext in info["extensions"]:
                stats["extensions_with_most_settings"][ext] += 1

        # Most used settings (top 20)
        sorted_settings = sorted(output_data.items(), key=lambda x: x[1]["usage_count"], reverse=True)
        stats["most_used_settings"] = [
            {"setting": k, "usage_count": v["usage_count"], "extensions": v["extensions"][:3]}
            for k, v in sorted_settings[:20]
        ]

        # Top extensions by settings count
        sorted_extensions = sorted(stats["extensions_with_most_settings"].items(), key=lambda x: x[1], reverse=True)
        stats["top_extensions"] = sorted_extensions[:20]

        # Convert defaultdicts to regular dicts for JSON
        stats["settings_by_type"] = dict(stats["settings_by_type"])
        stats["settings_by_prefix"] = dict(stats["settings_by_prefix"])
        stats["extensions_with_most_settings"] = dict(sorted_extensions[:20])

        # Save statistics
        stats_file = output_dir / "setting_statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"Statistics saved to: {stats_file}")


def main():
    """Main execution function."""
    # Configuration
    parser = argparse.ArgumentParser(
        description="Scan Kit extensions to extract settings from TOML and source files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--extensions-dir",
        type=str,
        default="/home/horde/repos/kit-app-template/_build/linux-x86_64/release/extscache",
        help="Path to the extensions directory (extscache)",
    )

    args = parser.parse_args()

    # Output directory (next to this script)
    script_dir = Path(__file__).parent
    output_dir = script_dir / "setting_data"

    # Create scanner and run
    scanner = ExtensionSettingsScanner(args.extensions_dir)

    logger.info(f"Starting scan of extensions in: {args.extensions_dir}")
    scanner.scan_all_extensions()

    logger.info("Saving results...")
    scanner.save_results(output_dir)

    logger.info("Scan complete!")

    # Print summary
    output_data = scanner.prepare_output()
    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE")
    print(f"{'='*60}")
    print(f"Total settings found: {len(output_data)}")
    print(
        f"Total extensions scanned: {len(set(ext for s in scanner.settings_data.values() for ext in s['extensions']))}"
    )
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
