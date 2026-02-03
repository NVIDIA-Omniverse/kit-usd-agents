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
Analyze documentation files and calculate their token sizes using TikToken.
This script will iterate through all files in the documentation extension
and generate a comprehensive report of file sizes in tokens.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import tiktoken


def get_file_extension(file_path: str) -> str:
    """Get the file extension from a file path."""
    return Path(file_path).suffix.lower()


def read_file_content(file_path: str) -> str:
    """Read file content, handling both text and binary files."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Binary file, return a placeholder
        return f"[Binary file: {Path(file_path).name}]"
    except Exception as e:
        return f"[Error reading file: {e}]"


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def analyze_directory(base_path: str) -> Dict:
    """Analyze all files in the directory and subdirectories."""
    results = {
        "analysis_date": datetime.now().isoformat(),
        "base_path": base_path,
        "encoding": "cl100k_base",
        "files": [],
        "categories": {},
        "summary": {},
    }

    total_tokens = 0
    total_files = 0
    total_size_bytes = 0

    # Walk through all files
    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, base_path)

            # Get file stats
            try:
                file_stat = os.stat(file_path)
                file_size = file_stat.st_size
            except:
                file_size = 0

            # Read content and count tokens
            content = read_file_content(file_path)
            token_count = count_tokens(content)

            # Determine file category
            extension = get_file_extension(file_path)
            if extension in [".md", ".txt"]:
                category = "Documentation"
            elif extension in [".py"]:
                category = "Python Code"
            elif extension in [".toml", ".yaml", ".yml", ".json"]:
                category = "Configuration"
            elif extension in [".svg", ".png", ".jpg", ".jpeg", ".gif"]:
                category = "Images"
            else:
                category = "Other"

            # Store file info
            file_info = {
                "path": relative_path,
                "absolute_path": file_path,
                "category": category,
                "extension": extension,
                "size_bytes": file_size,
                "size_formatted": format_file_size(file_size),
                "token_count": token_count,
                "tokens_per_kb": token_count / (file_size / 1024) if file_size > 0 else 0,
            }

            results["files"].append(file_info)

            # Update category stats
            if category not in results["categories"]:
                results["categories"][category] = {
                    "file_count": 0,
                    "total_tokens": 0,
                    "total_size_bytes": 0,
                    "files": [],
                }

            results["categories"][category]["file_count"] += 1
            results["categories"][category]["total_tokens"] += token_count
            results["categories"][category]["total_size_bytes"] += file_size
            results["categories"][category]["files"].append(relative_path)

            # Update totals
            total_tokens += token_count
            total_files += 1
            total_size_bytes += file_size

    # Sort files by token count
    results["files"].sort(key=lambda x: x["token_count"], reverse=True)

    # Add summary
    results["summary"] = {
        "total_files": total_files,
        "total_tokens": total_tokens,
        "total_size_bytes": total_size_bytes,
        "total_size_formatted": format_file_size(total_size_bytes),
        "average_tokens_per_file": total_tokens / total_files if total_files > 0 else 0,
        "average_file_size": format_file_size(total_size_bytes // total_files) if total_files > 0 else "0 B",
    }

    return results


def generate_markdown_report(results: Dict) -> str:
    """Generate a comprehensive markdown report from the analysis results."""
    report = []

    # Header
    report.append("# Documentation Extension Token Analysis Report")
    report.append(f"\n**Analysis Date:** {results['analysis_date']}")
    report.append(f"**Base Path:** `{results['base_path']}`")
    report.append(f"**Encoding:** {results['encoding']}")
    report.append("")

    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    summary = results["summary"]
    report.append(f"- **Total Files Analyzed:** {summary['total_files']}")
    report.append(f"- **Total Tokens:** {summary['total_tokens']:,}")
    report.append(f"- **Total Size:** {summary['total_size_formatted']}")
    report.append(f"- **Average Tokens per File:** {summary['average_tokens_per_file']:.2f}")
    report.append(f"- **Average File Size:** {summary['average_file_size']}")
    report.append("")

    # Category Breakdown
    report.append("## Category Breakdown")
    report.append("")
    report.append("| Category | Files | Total Tokens | Total Size | Avg Tokens/File |")
    report.append("|----------|-------|--------------|------------|-----------------|")

    for category, stats in sorted(results["categories"].items(), key=lambda x: x[1]["total_tokens"], reverse=True):
        avg_tokens = stats["total_tokens"] / stats["file_count"] if stats["file_count"] > 0 else 0
        report.append(
            f"| {category} | {stats['file_count']} | {stats['total_tokens']:,} | {format_file_size(stats['total_size_bytes'])} | {avg_tokens:.2f} |"
        )

    report.append("")

    # Top 10 Largest Files by Token Count
    report.append("## Top 10 Largest Files by Token Count")
    report.append("")
    report.append("| Rank | File Path | Tokens | Size | Category |")
    report.append("|------|-----------|--------|------|----------|")

    for i, file_info in enumerate(results["files"][:10], 1):
        report.append(
            f"| {i} | `{file_info['path']}` | {file_info['token_count']:,} | {file_info['size_formatted']} | {file_info['category']} |"
        )

    report.append("")

    # All Files Detailed List
    report.append("## All Files Detailed Analysis")
    report.append("")
    report.append("| File Path | Category | Extension | Tokens | Size | Tokens/KB |")
    report.append("|-----------|----------|-----------|--------|------|-----------|")

    for file_info in results["files"]:
        report.append(
            f"| `{file_info['path']}` | {file_info['category']} | {file_info['extension']} | {file_info['token_count']:,} | {file_info['size_formatted']} | {file_info['tokens_per_kb']:.2f} |"
        )

    report.append("")

    # Category Details
    report.append("## Category Details")
    report.append("")

    for category, stats in sorted(results["categories"].items()):
        report.append(f"### {category}")
        report.append(f"- **File Count:** {stats['file_count']}")
        report.append(f"- **Total Tokens:** {stats['total_tokens']:,}")
        report.append(f"- **Total Size:** {format_file_size(stats['total_size_bytes'])}")
        report.append("")
        report.append("**Files:**")
        for file_path in sorted(stats["files"]):
            report.append(f"- `{file_path}`")
        report.append("")

    # Token Distribution Analysis
    report.append("## Token Distribution Analysis")
    report.append("")

    # Calculate distribution
    token_ranges = {"0-100": 0, "101-500": 0, "501-1000": 0, "1001-5000": 0, "5001-10000": 0, "10000+": 0}

    for file_info in results["files"]:
        tokens = file_info["token_count"]
        if tokens <= 100:
            token_ranges["0-100"] += 1
        elif tokens <= 500:
            token_ranges["101-500"] += 1
        elif tokens <= 1000:
            token_ranges["501-1000"] += 1
        elif tokens <= 5000:
            token_ranges["1001-5000"] += 1
        elif tokens <= 10000:
            token_ranges["5001-10000"] += 1
        else:
            token_ranges["10000+"] += 1

    report.append("| Token Range | File Count | Percentage |")
    report.append("|-------------|------------|------------|")

    for range_name, count in token_ranges.items():
        percentage = (count / summary["total_files"] * 100) if summary["total_files"] > 0 else 0
        report.append(f"| {range_name} | {count} | {percentage:.1f}% |")

    report.append("")

    # Recommendations
    report.append("## Recommendations for MCP Integration")
    report.append("")
    report.append("Based on the token analysis, here are recommendations for structuring the data:")
    report.append("")
    report.append(
        "1. **High Priority Files** (>1000 tokens): These files contain substantial documentation and should be indexed with full content retrieval."
    )
    report.append(
        "2. **Medium Priority Files** (500-1000 tokens): These files contain moderate documentation and can be indexed with summaries."
    )
    report.append(
        "3. **Low Priority Files** (<500 tokens): These files can be indexed by title and brief description only."
    )
    report.append("")
    report.append("### Suggested Data Structure")
    report.append("")
    report.append("```python")
    report.append("documentation_index = {")
    report.append("    'high_priority': [  # Full content indexing")
    report.append("        # Files with >1000 tokens")
    report.append("    ],")
    report.append("    'medium_priority': [  # Summary indexing")
    report.append("        # Files with 500-1000 tokens")
    report.append("    ],")
    report.append("    'low_priority': [  # Title/description only")
    report.append("        # Files with <500 tokens")
    report.append("    ]")
    report.append("}")
    report.append("```")
    report.append("")

    # Footer
    report.append("---")
    report.append("*This report was generated automatically using TikToken analysis.*")

    return "\n".join(report)


def main():
    """Main function to run the analysis."""
    # Base path for the documentation extension
    base_path = (
        "/home/horde/repos/kit-lc-agent/source/mcp/omni_ui_mcp/examples/omni.kit.documentation.ui.style-1.0.9+b0a86421"
    )

    print(f"Starting token analysis for: {base_path}")
    print("=" * 60)

    # Run analysis
    results = analyze_directory(base_path)

    # Generate markdown report
    markdown_report = generate_markdown_report(results)

    # Save report to file
    report_path = "/home/horde/repos/kit-lc-agent/source/mcp/omni_ui_mcp/document.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    # Also save JSON results for programmatic access
    json_path = "/home/horde/repos/kit-lc-agent/source/mcp/omni_ui_mcp/documentation_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print summary to console
    print(f"\nAnalysis Complete!")
    print(f"Total Files: {results['summary']['total_files']}")
    print(f"Total Tokens: {results['summary']['total_tokens']:,}")
    print(f"Total Size: {results['summary']['total_size_formatted']}")
    print(f"\nReports saved to:")
    print(f"  - Markdown: {report_path}")
    print(f"  - JSON: {json_path}")


if __name__ == "__main__":
    main()
