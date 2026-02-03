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
Simple runner script for the Kit Extensions Data Collection Pipeline.

This script provides an easy interface to run the complete data collection
pipeline with sensible defaults and automatic Kit cache detection.

Usage:
    # Auto-detect Kit cache and run complete pipeline
    python run_pipeline.py

    # Specify Kit cache path
    python run_pipeline.py --kit-cache /path/to/extscache

    # Run with limited extensions for testing
    python run_pipeline.py --max-extensions 50

    # Quick run (skip embeddings and FAISS)
    python run_pipeline.py --quick

    # Resume from checkpoint after failure
    python run_pipeline.py --resume

    # Force restart from beginning
    python run_pipeline.py --force
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from data_collection_pipeline import DataCollectionPipeline, PipelineConfig


def find_kit_cache_paths() -> List[Path]:
    """Attempt to find Kit cache directories automatically."""
    possible_paths = []

    # Look in common Kit build locations
    search_roots = [
        Path.home() / "repos",
        Path("/home/horde/repos"),
        Path("C:/repos"),
        Path("~/Downloads"),
        Path("../../../.."),  # From data_collection directory
    ]

    for root in search_roots:
        if not root.exists():
            continue

        # Look for kit-app-template or similar
        for subdir in root.iterdir():
            if not subdir.is_dir():
                continue

            # Check for Kit build patterns
            if any(pattern in subdir.name.lower() for pattern in ["kit", "omni", "usd"]):
                # Look for _build/*/release/extscache
                build_dir = subdir / "_build"
                if build_dir.exists():
                    for arch_dir in build_dir.iterdir():
                        if arch_dir.is_dir():
                            for config_dir in arch_dir.iterdir():
                                if config_dir.is_dir() and "release" in config_dir.name:
                                    extscache = config_dir / "extscache"
                                    if extscache.exists() and extscache.is_dir():
                                        possible_paths.append(extscache)

    return possible_paths


def get_extension_count(kit_cache_path: Path) -> int:
    """Count extensions in the cache directory."""
    if not kit_cache_path.exists():
        return 0

    return len([d for d in kit_cache_path.iterdir() if d.is_dir() and d.name.startswith("omni.")])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Easy runner for Kit Extensions Data Collection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect Kit cache and run everything
  python run_pipeline.py

  # Specify Kit cache location
  python run_pipeline.py --kit-cache /path/to/extscache

  # Quick test with limited extensions
  python run_pipeline.py --max-extensions 25 --quick

  # Resume from last checkpoint after failure
  python run_pipeline.py --resume

  # Force restart from beginning (ignore checkpoint)
  python run_pipeline.py --force

  # Full processing including source code
  python run_pipeline.py --include-source
        """,
    )

    parser.add_argument(
        "--kit-cache", type=Path, help="Path to Kit extensions cache directory (auto-detected if not specified)"
    )

    parser.add_argument(
        "--max-extensions", type=int, help="Maximum number of extensions to process (useful for testing)"
    )

    parser.add_argument(
        "--include-source",
        action="store_true",
        help="Include function source code in Code Atlas (increases output size)",
    )

    parser.add_argument("--quick", action="store_true", help="Quick run - skip embeddings and FAISS generation")

    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files after completion")

    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint (automatic recovery)")

    parser.add_argument("--force", action="store_true", help="Force restart from beginning, ignoring any checkpoint")

    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually running")

    args = parser.parse_args()

    print("🚀 Kit Extensions Data Collection Pipeline Runner")
    print("=" * 60)

    # Find or validate Kit cache path
    kit_cache_path = args.kit_cache

    if not kit_cache_path:
        print("🔍 Auto-detecting Kit cache directories...")
        possible_paths = find_kit_cache_paths()

        if not possible_paths:
            print("❌ No Kit cache directories found automatically.")
            print("Please specify --kit-cache /path/to/extscache")
            sys.exit(1)

        if len(possible_paths) == 1:
            kit_cache_path = possible_paths[0]
            print(f"✅ Found Kit cache: {kit_cache_path}")
        else:
            print(f"🔍 Found {len(possible_paths)} possible Kit cache directories:")
            for i, path in enumerate(possible_paths, 1):
                ext_count = get_extension_count(path)
                print(f"  {i}. {path} ({ext_count} extensions)")

            choice = input("\nSelect Kit cache directory (1-{}): ".format(len(possible_paths)))
            try:
                idx = int(choice) - 1
                kit_cache_path = possible_paths[idx]
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                sys.exit(1)

    # Validate Kit cache path
    if not kit_cache_path.exists():
        print(f"❌ Kit cache directory does not exist: {kit_cache_path}")
        sys.exit(1)

    extension_count = get_extension_count(kit_cache_path)
    if extension_count == 0:
        print(f"❌ No extensions found in: {kit_cache_path}")
        sys.exit(1)

    print(f"📦 Found {extension_count} extensions in cache")

    # Apply limits if specified
    if args.max_extensions and args.max_extensions < extension_count:
        print(f"⚠️  Limited to {args.max_extensions} extensions for processing")
        extension_count = args.max_extensions

    # Determine pipeline stages
    if args.quick:
        start_stage = "preparation"
        end_stage = "settings"
        print("⚡ Quick mode: Running preparation → settings (skipping embeddings & FAISS)")
    else:
        start_stage = None
        end_stage = None
        print("🔄 Full mode: Running all pipeline stages")

    # Show configuration summary
    print("\n📋 Configuration Summary:")
    print(f"  Kit cache: {kit_cache_path}")
    print(f"  Extensions: {extension_count}")
    print(f"  Include source: {'Yes' if args.include_source else 'No'}")
    print(f"  Keep intermediates: {'Yes' if args.keep_intermediate else 'No'}")
    print(f"  Stages: {start_stage or 'preparation'} → {end_stage or 'final_assembly'}")

    if args.dry_run:
        print("\n🔍 Dry run mode - would run pipeline with above configuration")
        print("Remove --dry-run to execute")
        return

    # Create and configure pipeline
    config_file = Path(__file__).parent / "pipeline_config.toml"
    pipeline = DataCollectionPipeline(config_file)

    # Update configuration with command line arguments
    pipeline.config.config["input"]["kit_cache_path"] = str(kit_cache_path)

    if args.max_extensions:
        pipeline.config.config["processing"]["max_extensions"] = args.max_extensions

    if args.include_source:
        pipeline.config.config["processing"]["include_source_code"] = True

    if args.keep_intermediate:
        pipeline.config.config["output"]["keep_intermediates"] = True

    # Confirm before running
    if not args.quick and extension_count > 100:
        print(f"\n⚠️  Processing {extension_count} extensions will take significant time")
        confirm = input("Continue? [y/N]: ")
        if confirm.lower() not in ["y", "yes"]:
            print("Aborted")
            return

    # Run pipeline
    print(f"\n🚀 Starting pipeline...")
    try:
        success = pipeline.run(start_stage=start_stage, end_stage=end_stage, resume=args.resume, force=args.force)

        if success:
            print("\n🎉 Pipeline completed successfully!")

            # Show output location
            target_dir = Path(pipeline.config.get("output.target_dir"))
            if target_dir.exists():
                print(f"📂 Output generated in: {target_dir.resolve()}")

                # Count output files
                total_files = len(list(target_dir.rglob("*.json")))
                print(f"📄 Generated {total_files} data files")

            if not args.quick:
                print("\n✨ Ready for kit_fns MCP service!")
                print("The generated data will be automatically loaded by the service.")
        else:
            print("\n❌ Pipeline failed!")
            if not args.resume:
                print("💡 You can resume from this point using: --resume")
            print("Check the logs for details.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
