## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()


def extract_links(content: str, file_path: Path) -> List[Tuple[str, Path]]:
    """
    Extract markdown links from content and convert to absolute paths.
    Returns a list of tuples (link_text, resolved_path).
    """
    # Regular expression to find markdown links: [text](link)
    link_pattern = re.compile(r"\[(.*?)\]\((.*?)\)")
    matches = link_pattern.findall(content)

    # Convert relative links to absolute paths
    result = []
    for link_text, link_path in matches:
        # Skip external links, anchors, or non-markdown files
        if link_path.startswith(("http://", "https://", "#")) or not link_path.endswith(".md"):
            continue

        # Convert relative path to absolute
        if link_path.startswith("./"):
            link_path = link_path[2:]  # Remove './'

        # Get the directory of the current file
        abs_path = file_path.parent / link_path
        try:
            resolved_path = abs_path.resolve()
            result.append((link_text, resolved_path))
        except Exception as e:
            print(f"Warning: Could not resolve path {abs_path}: {e}")

    return result


def process_markdown_file(
    file_path: Path,
    visited: Set[Path],
    content_map: Dict[Path, str],
    link_order_map: Dict[Path, List[Tuple[str, Path]]],
    docs_dir: Path,
    verbose: bool = False,
) -> None:
    """
    Process a markdown file, extract its content and links.

    Args:
        file_path: Path to the markdown file
        visited: Set of already visited files
        content_map: Dictionary to store file content
        link_order_map: Dictionary to store the order of links in each file
        docs_dir: Root documentation directory
        verbose: Whether to print verbose output
    """
    if file_path in visited:
        return

    if not file_path.exists():
        print(f"Warning: File does not exist: {file_path}")
        return

    print(
        f"Processing: {file_path.relative_to(docs_dir.parent) if docs_dir.parent in file_path.parents else file_path}"
    )
    visited.add(file_path)

    try:
        content = read_file(str(file_path))
        content_map[file_path] = content

        # Extract links and store them in order
        links = extract_links(content, file_path)
        link_order_map[file_path] = links

        # Process all linked files
        for _, link_path in links:
            process_markdown_file(link_path, visited, content_map, link_order_map, docs_dir, verbose)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


def fix_internal_links(content: str, file_path: Path, docs_dir: Path) -> str:
    """Fix internal links in the content to work in the combined MDC file."""
    # Regular expression to find markdown links: [text](link)
    link_pattern = re.compile(r"\[(.*?)\]\((.*?)\)")

    def replace_link(match):
        text, link = match.groups()

        # Skip external links, anchors
        if link.startswith(("http://", "https://", "#")):
            return match.group(0)

        # Handle relative links to markdown files
        if link.endswith(".md"):
            if link.startswith("./"):
                link = link[2:]  # Remove './'

            # Convert to absolute path
            try:
                abs_link_path = (file_path.parent / link).resolve()

                # Create an anchor link to the section in the MDC file
                try:
                    rel_path = abs_link_path.relative_to(docs_dir)
                    return f"[{text}](#{rel_path})"
                except ValueError:
                    # If the link is outside the docs directory, keep it as is
                    return match.group(0)
            except Exception:
                # If there's an error resolving the path, keep the link as is
                return match.group(0)

        # Keep other links as they are
        return match.group(0)

    return link_pattern.sub(replace_link, content)


def organize_files_by_readme_order(
    content_map: Dict[Path, str],
    link_order_map: Dict[Path, List[Tuple[str, Path]]],
    start_file: Path,
    docs_dir: Path,
    verbose: bool = False,
) -> List[Tuple[Path, str]]:
    """
    Organize files based on the order they appear in README.md and other files.

    This ensures that files are ordered according to the logical structure of the documentation.
    """
    organized_files = []
    visited = set()
    extending_file = None
    extending_content = None

    # Debug: Print the links in README.md
    if start_file in link_order_map:
        print("\nLinks in README.md:")
        for i, (text, path) in enumerate(link_order_map[start_file]):
            print(f"  {i+1}. [{text}] -> {path.relative_to(docs_dir)}")

    def process_section_and_children(file_path: Path, depth: int = 0):
        """Process a file and all its children recursively, respecting the order in the file."""
        nonlocal extending_file, extending_content

        if file_path in visited or file_path not in content_map:
            return

        # Check if this is the extending.md file
        if file_path.name.lower() == "extending.md" and "advanced" in str(file_path).lower():
            if verbose:
                print(f"Found extending.md file: {file_path.relative_to(docs_dir)} - will place at the end")
            extending_file = file_path
            extending_content = content_map[file_path]
            visited.add(file_path)
            return

        visited.add(file_path)
        organized_files.append((file_path, content_map[file_path]))
        if verbose:
            print(f"Added to organized files: {file_path.relative_to(docs_dir)}")

        # If this is a README.md file, process all its links in order
        if file_path.name.lower() == "readme.md" and file_path in link_order_map:
            if verbose:
                print(f"Processing links in {file_path.relative_to(docs_dir)}:")
            # Process each linked file in the order they appear
            for link_text, linked_file in link_order_map[file_path]:
                if verbose:
                    print(f"  - Link: [{link_text}] -> {linked_file.relative_to(docs_dir)}")

                # Skip the extending.md file for now
                if linked_file.name.lower() == "extending.md" and "advanced" in str(linked_file).lower():
                    if verbose:
                        print(f"    Skipping extending.md file for now: {linked_file.relative_to(docs_dir)}")
                    extending_file = linked_file
                    extending_content = content_map[linked_file]
                    visited.add(linked_file)
                    continue

                if linked_file not in visited:
                    # If the linked file is a README.md, process it and all its children
                    if linked_file.name.lower() == "readme.md":
                        if verbose:
                            print(f"    Processing README file: {linked_file.relative_to(docs_dir)}")
                        process_section_and_children(linked_file, depth + 1)
                    else:
                        # For non-README files, just add them first
                        if verbose:
                            print(f"    Adding non-README file: {linked_file.relative_to(docs_dir)}")
                        visited.add(linked_file)
                        organized_files.append((linked_file, content_map[linked_file]))

                        # Then process their children if they have any
                        if linked_file in link_order_map:
                            # Get the directory of this file
                            file_dir = linked_file.parent

                            # Process all files linked from this file
                            for _, child_file in link_order_map[linked_file]:
                                # Only process files in the same directory
                                if child_file.parent == file_dir and child_file not in visited:
                                    if verbose:
                                        print(f"      Processing child file: {child_file.relative_to(docs_dir)}")
                                    process_section_and_children(child_file, depth + 1)

    # Start with the main README
    process_section_and_children(start_file)

    # Add any remaining files that weren't linked
    remaining = [(f, content_map[f]) for f in content_map if f not in visited]

    if remaining and verbose:
        print("\nAdding remaining files that weren't linked:")
        for file_path, _ in remaining:
            print(f"  - {file_path.relative_to(docs_dir)}")

    # Sort remaining files by directory and name
    remaining.sort(key=lambda x: (str(x[0].parent), x[0].name))
    organized_files.extend(remaining)

    # Add the extending.md file at the end if it was found
    if extending_file and extending_content:
        if verbose:
            print(f"\nAdding extending.md file at the end: {extending_file.relative_to(docs_dir)}")
        organized_files.append((extending_file, extending_content))

    return organized_files


def create_section_title(file_path: Path, docs_dir: Path) -> str:
    """Create a section title from the file path."""
    rel_path = file_path.relative_to(docs_dir)

    # If it's a README file, use the directory name
    if file_path.name.lower() == "readme.md":
        if rel_path.parent == Path("."):
            return "Introduction"
        else:
            return rel_path.parent.name.replace("-", " ").title()

    # Otherwise use the file name without extension
    return file_path.stem.replace("-", " ").title()


def generate_mdc(docs_dir: Path = None, output_path: Path = None, verbose: bool = False):
    """Generate the chat-usd.mdc file."""
    # Get the base directory
    base_dir = Path(__file__).parent.parent

    # Set default docs directory if not provided
    if docs_dir is None:
        docs_dir = Path(__file__).parent

    # Start with README.md
    start_file = docs_dir / "README.md"

    if not start_file.exists():
        print(f"Error: {start_file} does not exist")
        return

    # Set default output path if not provided
    if output_path is None:
        output_dir = base_dir.parent.parent / ".cursor" / "rules"
        output_path = output_dir / "chat-usd.mdc"
        os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(output_path.parent, exist_ok=True)

    print(f"Starting documentation generation from {start_file}")
    print(f"Output will be written to {output_path}")

    # Process all markdown files starting from README.md
    visited = set()
    content_map = {}
    link_order_map = {}
    process_markdown_file(start_file, visited, content_map, link_order_map, docs_dir, verbose)

    # Organize files based on the order in README.md
    organized_files = organize_files_by_readme_order(content_map, link_order_map, start_file, docs_dir, verbose)

    print(f"\nFinal order of files in the MDC:")
    for i, (file_path, _) in enumerate(organized_files):
        print(f"{i+1}. {file_path.relative_to(docs_dir)}")

    # Header content
    header = """---
description: Comprehensive documentation for Chat USD, a specialized AI assistant for Universal Scene Description (USD) development.
globs:
---

# Chat USD Documentation

This document contains the complete documentation for Chat USD, a specialized AI assistant for Universal Scene Description (USD) development.

"""

    # Use newline='\n' to force Unix line endings in the output file
    try:
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            # Write header
            f.write(header)

            # Write table of contents
            f.write("## Table of Contents\n\n")
            for file_path, _ in organized_files:
                rel_path = file_path.relative_to(docs_dir)
                section_name = create_section_title(file_path, docs_dir)
                f.write(f"- [{section_name}](#{rel_path})\n")
            f.write("\n---\n\n")

            # Write all markdown content
            for file_path, content in organized_files:
                # Add a section header with the file path
                rel_path = file_path.relative_to(docs_dir)
                section_title = create_section_title(file_path, docs_dir)

                f.write(f"<a id='{rel_path}'></a>\n\n")
                f.write(f"# {section_title}\n\n")

                # Fix internal links
                fixed_content = fix_internal_links(content, file_path, docs_dir)

                # Write the content
                f.write(fixed_content)
                f.write("\n\n---\n\n")

        print(f"\nGenerated {output_path}")
        print(f"Processed {len(organized_files)} markdown files")

        # Verify the file was created
        if os.path.exists(output_path):
            print(f"File exists at {output_path} with size {os.path.getsize(output_path)} bytes")
        else:
            print(f"ERROR: File was not created at {output_path}")

    except Exception as e:
        print(f"Error writing to file {output_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MDC file from Markdown documentation")
    parser.add_argument("--docs-dir", type=str, help="Path to the documentation directory")
    parser.add_argument("--output", type=str, help="Path to the output MDC file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir) if args.docs_dir else None
    output_path = Path(args.output) if args.output else None

    generate_mdc(docs_dir, output_path, args.verbose)
