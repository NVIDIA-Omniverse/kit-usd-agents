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
Scan Kit extensions and extract interesting methods for code examples.

Usage:
    python scan_extensions_codeatlas.py --mode regular  # For production code
    python scan_extensions_codeatlas.py --mode tests    # For test code
    python scan_extensions_codeatlas.py --mode all      # For all code
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import docstring_parser
import tiktoken

# Import from lc_agent.code_atlas
from lc_agent.code_atlas import CodeAtlasCache


class ExtensionAnalyzer:
    """Analyzes extensions to find interesting methods based on various criteria."""

    def __init__(
        self,
        min_lines: int = 50,
        min_complexity: int = 5,
        debug_mode: bool = False,
        scan_mode: str = "regular",
        model: str = "nvidia-llama-3.3-nemotron-super-49b-v1",
        concurrency: int = 20,
        batch_size: int = 10,
        excluded_modules: List[str] = None,
    ):
        """
        Initialize the analyzer with configurable thresholds.

        Args:
            min_lines: Minimum number of lines for a method to be considered interesting
            min_complexity: Minimum cyclomatic complexity (approximated by control flow keywords)
            debug_mode: Whether to include source code in output
            scan_mode: "regular" (exclude tests), "tests" (only tests), or "all" (include everything)
            model: Model to use for method description generation
            concurrency: Number of concurrent requests to the model
            batch_size: Number of methods to process in a single request
            excluded_modules: List of module names to exclude when scanning Python code
        """
        self.min_lines = min_lines
        self.min_complexity = min_complexity
        self.cache = CodeAtlasCache()
        self.interesting_methods = []
        self.debug_mode = debug_mode  # Debug mode flag
        self.scan_mode = scan_mode  # Control test scanning
        self.excluded_modules = excluded_modules

        # Initialize tiktoken encoder for GPT-4
        try:
            self.encoder = tiktoken.encoding_for_model("gpt-4")
        except:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        # Token tracking
        self.total_tokens = 0
        self.file_tokens = {}

        from .generate_method_descriptions import MethodDescriptionGenerator

        self.description_generator = MethodDescriptionGenerator(model, concurrency, batch_size)

    async def analyze_extension(
        self, extension_path: str, save_to_json: bool = False, output_dir: str = None
    ) -> Dict[str, Any]:
        """
        Analyze a single extension directory.

        Args:
            extension_path: Path to the extension directory
            save_to_json: Whether to save the extracted methods to a JSON file
            output_dir: Directory to save output files (if None, uses default)

        Returns:
            Dictionary containing analysis results
        """
        print(f"\n{'='*80}")
        print(f"Analyzing extension: {extension_path}")
        print(f"{'='*80}")

        # Clear previous cache
        self.cache.clear()

        extension_path_posix = extension_path.replace("\\", "/")

        # Find the Python module path in the extension
        omni_path = Path(extension_path) / "omni"
        if not omni_path.exists():
            print(f"No 'omni' directory found in {extension_path}")
            return {"error": "No omni directory found"}

        # Scan the extension
        try:
            self.cache.scan(extension_path_posix, excluded_modules=self.excluded_modules)
        except Exception as e:
            print(f"Error scanning extension: {e}")
            return {"error": str(e)}

        # Analyze methods
        results = {
            "extension_path": extension_path_posix,
            "extension_name": os.path.basename(extension_path),
            "total_modules": len(self.cache._modules),
            "total_classes": len(self.cache._classes),
            "total_methods": len(self.cache._methods),
            "interesting_methods": [],
        }

        print(f"\nFound:")
        print(f"  - {results['total_modules']} modules")
        print(f"  - {results['total_classes']} classes")
        print(f"  - {results['total_methods']} methods/functions")

        missing_description_methods = {}

        # Analyze each method
        for method_name, method_info in self.cache._methods.items():
            # Filter based on scan mode
            if not self._should_include_method(method_name, method_info):
                continue

            analysis = self._analyze_method(method_info)
            if analysis["is_interesting"]:
                # Use CodeAtlas's lookup_method to get the complete reconstructed source
                complete_source = self.cache.lookup_method(
                    method_name, method_bodies=True, docs=True, pass_in_body=True
                )

                # If lookup_method returns None or empty, fall back to source_code
                if not complete_source:
                    complete_source = method_info.source_code or ""
                    if self.debug_mode:
                        print(f"   ⚠️ lookup_method returned empty for {method_name}")

                # Count tokens in the complete source
                token_count = len(self.encoder.encode(complete_source))
                self.total_tokens += token_count

                # Track tokens per file
                file_key = self._get_file_path(method_info)
                if file_key not in self.file_tokens:
                    self.file_tokens[file_key] = 0
                self.file_tokens[file_key] += token_count

                parsed_docstring = docstring_parser.parse(method_info.docstring)

                method_data = {
                    "name": method_name,
                    "file_path": file_key,
                    "start_line": method_info.line_number,
                    "line_count": analysis["line_count"],
                    "token_count": token_count,
                    "module": method_info.module_name,
                    "is_async": method_info.is_async_method,
                    "decorators": method_info.decorators,
                    "arguments": len(method_info.arguments),
                    "description": parsed_docstring.description if parsed_docstring.description else "",
                    "complexity_score": analysis["complexity_score"],
                    "reasons": analysis["reasons"],
                    "class_usages": method_info.class_usages[:5] if method_info.class_usages else [],
                    "source_code": complete_source if not self.debug_mode else "[SOURCE OMITTED IN DEBUG MODE]",
                    "analysis": analysis,
                    "info": {
                        "module": method_info.module_name,
                        "line_number": method_info.line_number,
                        "is_async": method_info.is_async_method,
                        "decorators": method_info.decorators,
                        "arguments": len(method_info.arguments),
                        "class_usages": method_info.class_usages[:5] if method_info.class_usages else [],
                    },
                }
                results["interesting_methods"].append(method_data)

                if len(method_data["description"]) == 0:
                    missing_description_methods[method_name] = len(results["interesting_methods"]) - 1

        if len(missing_description_methods) > 0:
            method_descriptions = await self.description_generator.generate(
                {
                    results["interesting_methods"][method_id]["name"]: results["interesting_methods"][method_id][
                        "source_code"
                    ]
                    for method_id in missing_description_methods.values()
                }
            )
            missing_description_method_names = set(missing_description_methods.keys()) - set(method_descriptions.keys())
            if len(missing_description_method_names) > 0:
                print(
                    f"  ⚠️  Error: method names missing from description generation: {list(missing_description_method_names)}"
                )
                return {"error": "method names missing from description generation"}
            for method_name, description in method_descriptions.items():
                results["interesting_methods"][missing_description_methods[method_name]]["description"] = description

        # Print interesting methods
        if results["interesting_methods"]:
            print(f"\n🔍 Found {len(results['interesting_methods'])} interesting methods:")
            print("-" * 60)
            for idx, method_data in enumerate(results["interesting_methods"], 1):
                self._print_method_summary(idx, method_data)
        else:
            print("\n✅ No methods matching the 'interesting' criteria were found.")

        # Save individual extension methods to JSON if requested
        if save_to_json and results["interesting_methods"]:
            self._save_extension_methods(results, output_dir)

        return results

    def _get_file_path(self, method_info) -> str:
        """Get the file path for a method."""
        # First check if CodeAtlas provided a file_path
        if hasattr(method_info, "file_path") and method_info.file_path:
            return method_info.file_path

        # Fallback: convert module name to path
        # Handle test modules that might have different naming
        module_name = method_info.module_name

        # Remove any class names from the module path
        if "." in module_name:
            parts = module_name.split(".")
            # Remove the last part if it looks like a class name (starts with uppercase)
            if parts[-1] and parts[-1][0].isupper():
                module_name = ".".join(parts[:-1])

        return module_name.replace(".", "/") + ".py"

    def _should_include_method(self, method_name: str, method_info) -> bool:
        """
        Determine if a method should be included based on scan mode.

        Args:
            method_name: Full method name
            method_info: CodeAtlasMethodInfo object

        Returns:
            True if method should be included, False otherwise
        """
        # Check if method is in a test module
        is_test = False

        if not method_name.startswith("omni."):
            return False

        # Check module name for test patterns
        if method_info.module_name:
            module_parts = method_info.module_name.lower().split(".")
            # Check for common test module patterns
            is_test = any(part in ["test", "tests", "testing"] for part in module_parts)
            # Also check for test_ prefixes
            is_test = is_test or any(part.startswith("test_") for part in module_parts)

        # Check method name for test patterns
        if method_name:
            name_lower = method_name.lower()
            is_test = is_test or ".test" in name_lower or "test_" in name_lower

        # Apply filtering based on scan mode
        if self.scan_mode == "regular":
            # Exclude test methods
            return not is_test
        elif self.scan_mode == "tests":
            # Only include test methods
            return is_test
        else:  # "all"
            # Include everything
            return True

    def _analyze_method(self, method_info) -> Dict[str, Any]:
        """
        Analyze a single method to determine if it's interesting.

        Args:
            method_info: CodeAtlasMethodInfo object

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "is_interesting": False,
            "line_count": 0,
            "complexity_score": 0,
            "has_error_handling": False,
            "has_loops": False,
            "has_conditionals": False,
            "has_recursion": False,
            "has_class_usages": False,
            "reasons": [],
        }

        if not method_info.source_code:
            return analysis

        source_lines = method_info.source_code.split("\n")
        analysis["line_count"] = len(source_lines)

        # Check for various patterns in the source code
        source_lower = method_info.source_code.lower()

        # Complexity indicators
        matches = re.findall(r"\s(if|elif|else:|for|while|try:|except|match|case|with|def|class)\s", source_lower)
        analysis["complexity_score"] += len(matches)

        # Specific pattern checks
        analysis["has_error_handling"] = any(match == "try:" for match in matches)
        analysis["has_loops"] = any(match == "for" or match == "while" for match in matches)
        analysis["has_conditionals"] = any(match == "if" for match in matches)
        # try to match self.method_name(...), cls.method_name(...), method_name(...), etc.
        prefix = (
            rf"({method_info.arguments[0].name}\.)?"
            if method_info.arguments and method_info.arguments[0].name in ["self", "cls"]
            else ""
        )
        analysis["has_recursion"] = re.search(rf"\s{prefix}{method_info.name}\s*\(", source_lower) is not None
        analysis["has_class_usages"] = bool(method_info.class_usages)

        # Determine if method is interesting
        if analysis["line_count"] >= self.min_lines:
            analysis["reasons"].append(f"Large method ({analysis['line_count']} lines)")
            analysis["is_interesting"] = True

        if analysis["complexity_score"] >= self.min_complexity:
            analysis["reasons"].append(f"High complexity (score: {analysis['complexity_score']})")
            analysis["is_interesting"] = True

        if analysis["has_recursion"]:
            analysis["reasons"].append("Uses recursion")
            analysis["is_interesting"] = True

        if analysis["has_error_handling"] and analysis["line_count"] >= 10:
            analysis["reasons"].append("Has error handling")
            analysis["is_interesting"] = True

        if len(method_info.decorators) > 1 and analysis["line_count"] >= 10:
            analysis["reasons"].append(f"Multiple decorators ({len(method_info.decorators)})")
            analysis["is_interesting"] = True

        if method_info.is_async_method and analysis["line_count"] >= 10:
            analysis["reasons"].append("Async method")
            analysis["is_interesting"] = True

        return analysis

    def _print_method_summary(self, idx: int, method_data: Dict[str, Any]):
        """Print a formatted summary of an interesting method."""
        info = method_data["info"]
        analysis = method_data["analysis"]

        print(f"\n{idx}. {method_data['name']}")
        print(f"   📍 Location: {info['module']}:{info['line_number']}")
        print(
            f"   📊 Stats: {analysis['line_count']} lines, {method_data.get('token_count', 0)} tokens, complexity: {analysis['complexity_score']}"
        )
        print(f"   🎯 Reasons: {', '.join(analysis['reasons'])}")

        if info["decorators"]:
            print(f"   🎨 Decorators: {', '.join(info['decorators'])}")

        if info["is_async"]:
            print(f"   ⚡ Async method")

        if info["class_usages"]:
            print(f"   🔗 Uses classes: {', '.join(info['class_usages'][:3])}")

        features = []
        if analysis["has_error_handling"]:
            features.append("error handling")
        if analysis["has_loops"]:
            features.append("loops")
        if analysis["has_recursion"]:
            features.append("recursion")
        if features:
            print(f"   ✨ Features: {', '.join(features)}")

    def _save_extension_methods(self, results: Dict[str, Any], output_dir: str = None):
        """Save extracted methods for an extension to a JSON file."""
        # Create extracted_methods directory if it doesn't exist
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "extracted_methods")
        os.makedirs(output_dir, exist_ok=True)

        # Prepare data for export
        export_data = {
            "extension_name": results["extension_name"],
            "total_methods_analyzed": results["total_methods"],
            "interesting_methods_count": len(results["interesting_methods"]),
            "methods": [],
        }

        for method in results["interesting_methods"]:
            method_entry = {
                "name": method["name"],
                "file_path": method["file_path"],
                "start_line": method["start_line"],
                "line_count": method["line_count"],
                "token_count": method.get("token_count", 0),
                "module": method["module"],
                "is_async": method["is_async"],
                "decorators": method["decorators"],
                "arguments": method["arguments"],
                "description": method["description"],
                "complexity_score": method["complexity_score"],
                "reasons": method["reasons"],
                "class_usages": method["class_usages"],
            }
            # Only include source code if not in debug mode
            if not self.debug_mode:
                method_entry["source_code"] = method["source_code"]
            export_data["methods"].append(method_entry)

        # Save to JSON file
        output_file = os.path.join(output_dir, f"{results['extension_name']}.example.json")
        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"   💾 Saved methods to {output_file}")


async def analyze_all_extensions_async(
    output_dir,
    extensions_dir,
    max_extensions=-1,
    min_lines=50,
    min_complexity=3,
    debug_mode=False,
    scan_mode="regular",
    model="nvidia-llama-3.3-nemotron-super-49b-v1",
    concurrency=20,
    batch_size=10,
    resume=False,
    excluded_modules=None,
):
    """
    Analyze multiple extensions and provide summary.
    Args:
        output_dir: Directory to save output files
        extensions_dir: Directory containing Kit extensions
        max_extensions: Maximum number of extensions to process (default to -1 for all)
        min_lines: Minimum number of lines for a method to be interesting (default: 50)
        min_complexity: Minimum complexity score for a method (default: 3)
        debug_mode: Whether to include source code in output (default: False)
        scan_mode: "regular" (exclude test methods), "tests" (only tests), or "all" (include everything) (default: "regular")
        model: Model to use for method description generation (default: "nvidia-llama-3.3-nemotron-super-49b-v1")
        concurrency: Number of concurrent requests to the model (default: 4)
        batch_size: Number of methods to process in a single request (default: 10)
        resume: Whether to skip already-processed extensions (default: False)
        excluded_modules: List of module names to exclude when scanning Python code
    """
    analyzer = ExtensionAnalyzer(
        min_lines=min_lines,
        min_complexity=min_complexity,
        debug_mode=debug_mode,
        scan_mode=scan_mode,
        model=model,
        concurrency=concurrency,
        batch_size=batch_size,
        excluded_modules=excluded_modules,
    )

    # Determine extracted_methods directory based on scan mode
    if scan_mode == "tests":
        extracted_methods_dir = os.path.join(output_dir, "extracted_methods_tests")
    elif scan_mode == "regular":
        extracted_methods_dir = os.path.join(output_dir, "extracted_methods_regular")
    else:  # "all"
        extracted_methods_dir = os.path.join(output_dir, "extracted_methods_all")

    # Create output directory if it doesn't exist
    os.makedirs(extracted_methods_dir, exist_ok=True)

    # Get all extension directories
    extension_paths = []
    for item in os.listdir(extensions_dir):
        item_path = os.path.join(extensions_dir, item)
        if os.path.isdir(item_path) and item.startswith("omni."):
            extension_paths.append(item_path)

    extension_paths.sort()

    if max_extensions != -1:
        extension_paths = extension_paths[:max_extensions]

    print(f"📦 Found {len(extension_paths)} extensions to analyze")

    # Check for existing .example.json files to enable resume (only if resume flag is set)
    extensions_to_process = []
    extensions_already_done = []

    if resume:
        for extension_path in extension_paths:
            extension_name = os.path.basename(extension_path)
            example_file = os.path.join(extracted_methods_dir, f"{extension_name}.example.json")

            if os.path.exists(example_file):
                extensions_already_done.append((extension_name, example_file))
            else:
                extensions_to_process.append(extension_path)

        if extensions_already_done:
            print(f"✅ Found {len(extensions_already_done)} extensions already processed - skipping them")

        if not extensions_to_process:
            print("✅ All extensions already processed!")
        else:
            print(f"🔄 Will process {len(extensions_to_process)} extensions")
    else:
        # Not resuming - process all extensions
        extensions_to_process = extension_paths
        print(f"🔄 Will process {len(extensions_to_process)} extensions")

    print("=" * 80)

    all_results = []
    total_stats = {"extensions": 0, "modules": 0, "classes": 0, "methods": 0, "interesting": 0}

    # Load stats from already processed extensions
    for extension_name, example_file in extensions_already_done:
        try:
            with open(example_file, "r") as f:
                existing_data = json.load(f)

            # Update statistics
            total_stats["extensions"] += 1
            total_stats["methods"] += existing_data.get("total_methods_analyzed", 0)
            total_stats["interesting"] += existing_data.get("interesting_methods_count", 0)

            # Add to results
            all_results.append(
                {
                    "extension": extension_name,
                    "stats": {
                        "modules": 0,  # Not stored in example.json
                        "classes": 0,  # Not stored in example.json
                        "methods": existing_data.get("total_methods_analyzed", 0),
                        "interesting": existing_data.get("interesting_methods_count", 0),
                    },
                    "skipped": True,
                }
            )

            print(f"   ✅ Loaded existing data for {extension_name}")

        except Exception as e:
            print(f"   ⚠️  Warning: Failed to load {example_file}: {e}")

    tasks = []

    async def _analyze_extension(idx: int, extension_path: str):
        extension_name = os.path.basename(extension_path)
        print(f"\n[{idx}/{len(extensions_to_process)}] Processing: {extension_name}")
        results = await analyzer.analyze_extension(extension_path, save_to_json=True, output_dir=extracted_methods_dir)
        return results

    for idx, extension_path in enumerate(extensions_to_process, 1):
        tasks.append(_analyze_extension(idx, extension_path))

    if tasks:
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        results_list = []

    for results in results_list:
        if isinstance(results, Exception):
            import traceback

            traceback.print_exception(results)
            print(f"  ⚠️  Error: {results}")
            continue

        if "error" not in results:
            total_stats["extensions"] += 1
            total_stats["modules"] += results.get("total_modules", 0)
            total_stats["classes"] += results.get("total_classes", 0)
            total_stats["methods"] += results.get("total_methods", 0)
            total_stats["interesting"] += len(results.get("interesting_methods", []))

            all_results.append(
                {
                    "extension": results.get("extension_name", "unknown"),
                    "stats": {
                        "modules": results.get("total_modules", 0),
                        "classes": results.get("total_classes", 0),
                        "methods": results.get("total_methods", 0),
                        "interesting": len(results.get("interesting_methods", [])),
                    },
                }
            )

    # Print summary
    print("\n" + "=" * 80)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"📦 Extensions analyzed: {total_stats['extensions']}")
    if extensions_already_done:
        print(f"   ✅ Resumed (skipped): {len(extensions_already_done)}")
        print(f"   🔄 Newly processed: {len(extensions_to_process)}")
    print(f"📁 Total modules: {total_stats['modules']}")
    print(f"🏛️  Total classes: {total_stats['classes']}")
    print(f"⚙️  Total methods: {total_stats['methods']}")
    print(f"🔍 Total interesting methods: {total_stats['interesting']}")

    # Print scan mode info
    mode_desc = {
        "regular": "(excluding test methods)",
        "tests": "(test methods only)",
        "all": "(including all methods)",
    }
    print(f"\n📋 Scan mode: {analyzer.scan_mode} {mode_desc.get(analyzer.scan_mode, '')}")

    # Print token statistics
    print(f"\n📈 TOKEN STATISTICS:")
    print(f"   Total tokens extracted: {analyzer.total_tokens:,}")
    print(f"   Average tokens per method: {analyzer.total_tokens // max(total_stats['interesting'], 1):,}")

    if analyzer.file_tokens and len(analyzer.file_tokens) <= 10:
        print(f"\n   Top files by token count:")
        sorted_files = sorted(analyzer.file_tokens.items(), key=lambda x: x[1], reverse=True)[:10]
        for file_path, tokens in sorted_files:
            print(f"     {os.path.basename(file_path)}: {tokens:,} tokens")

    # Save results to output directory
    output_file = os.path.join(output_dir, "extension_analysis_summary.json")
    with open(output_file, "w") as f:
        json.dump({"summary": total_stats, "extensions": all_results}, f, indent=2)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"💾 Summary saved to {output_file}")

    return total_stats


def main():
    """Main function to run the extension analyzer."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Scan Kit extensions and extract interesting methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Extract production code methods:
    python scan_extensions_codeatlas.py --mode regular

  Extract test methods:
    python scan_extensions_codeatlas.py --mode tests

  Extract all methods:
    python scan_extensions_codeatlas.py --mode all

  Process specific number of extensions:
    python scan_extensions_codeatlas.py --mode regular --max-extensions 10

  Debug mode (exclude source code from output):
    python scan_extensions_codeatlas.py --mode regular --debug

  Custom thresholds:
    python scan_extensions_codeatlas.py --mode regular --min-lines 100 --min-complexity 10
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for extracted methods (default: current directory)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["regular", "tests", "all"],
        default="regular",
        help="Type of code to extract: 'regular' (production), 'tests', or 'all' (default: regular)",
    )

    parser.add_argument(
        "--max-extensions", type=int, default=None, help="Maximum number of extensions to process (default: all)"
    )

    parser.add_argument(
        "--min-lines", type=int, default=50, help="Minimum lines for a method to be interesting (default: 50)"
    )

    parser.add_argument(
        "--min-complexity", type=int, default=5, help="Minimum complexity score for a method (default: 5)"
    )

    parser.add_argument("--debug", action="store_true", help="Debug mode - exclude source code from JSON output")

    parser.add_argument(
        "--extensions-dir",
        type=str,
        default="/home/horde/repos/kit-app-template/_build/linux-x86_64/release/extscache",
        help="Directory containing Kit extensions (default: kit-app-template extscache)",
    )

    args = parser.parse_args()

    # Log configuration
    print("Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Extensions directory: {args.extensions_dir}")
    print(f"  Max extensions: {args.max_extensions if args.max_extensions else 'all'}")
    print(f"  Min lines: {args.min_lines}")
    print(f"  Min complexity: {args.min_complexity}")
    print(f"  Debug mode: {args.debug}")

    # Analyze extensions with specified parameters
    asyncio.run(
        analyze_all_extensions_async(
            args.output_dir,
            args.extensions_dir,
            max_extensions=args.max_extensions,
            min_lines=args.min_lines,
            min_complexity=args.min_complexity,
            debug_mode=args.debug,
            scan_mode=args.mode,
        )
    )

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
