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

import asyncio
import json
import os
from typing import Any, Dict, List

# this currently is only working for Claude Code Register Users
from claude_code_sdk import ClaudeCodeOptions, ClaudeSDKClient


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load and parse JSON file containing function entries"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(file_path: str, data: List[Dict[str, Any]]) -> None:
    """Save updated JSON data back to file"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


async def generate_ui_description(
    client: ClaudeSDKClient, function_body: str, class_name: str, function_name: str
) -> str:
    """Generate description for a UI function using Claude"""

    system_prompt = """You are an expert at analyzing OmniUI code. Your task is to provide a concise description of what UI elements and functionality a given method creates.
    Focus on:
    - The type of UI being created (window, dialog, panel, etc.)
    - Key UI components used (buttons, labels, text fields, etc.)
    - The apparent purpose or objective of the UI
    Keep your description brief (2-3 sentences max) and technical."""

    user_prompt = f"""Analyze this OmniUI method and describe what UI it creates:

Class: {class_name}
Method: {function_name}

Function body:
{function_body}

Provide a brief technical description of what UI this creates and its purpose."""

    try:
        # Send the query with system prompt and user content
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        await client.query(full_prompt)

        # Collect the response
        response_text = ""
        async for message in client.receive_response():
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        response_text += block.text

            # Check if this is the final result message
            if type(message).__name__ == "ResultMessage":
                break

        return response_text.strip() if response_text else "No response generated"
    except Exception as e:
        print(f"Error generating description: {e}")
        return "Error generating description"


def setup_enterprise_auth():
    """Configure environment for enterprise/subscription authentication."""

    # Remove API key if present to force subscription authentication
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]
        print("🔧 Removed API key from environment")

    # Force subscription authentication mode
    os.environ["CLAUDE_USE_SUBSCRIPTION"] = "true"
    print("✅ Configured for subscription authentication")

    # Check for credential files
    claude_dir = os.path.expanduser("~/.claude")
    credentials_file = os.path.join(claude_dir, ".credentials.json")
    oauth_file = os.path.join(claude_dir, "oauth_token.json")

    if os.path.exists(credentials_file):
        print(f"✅ Found credentials: {credentials_file}")
    else:
        print(f"⚠️  Credentials not found: {credentials_file}")
        print("   Run 'claude login' to authenticate with your enterprise account")

    if os.path.exists(oauth_file):
        print(f"✅ Found OAuth tokens: {oauth_file}")
    else:
        print(f"⚠️  OAuth tokens not found: {oauth_file}")


async def process_entries(json_file_path: str, output_file_path: str = None):
    """Main function to process all entries in the JSON file"""

    # Use output path if provided, otherwise overwrite input
    if output_file_path is None:
        output_file_path = json_file_path

    print(f"Loading JSON file from: {json_file_path}")
    entries = load_json_file(json_file_path)
    print(f"Found {len(entries)} entries to process")

    # Setup enterprise authentication
    print("\n🔐 Setting up enterprise authentication...")
    setup_enterprise_auth()

    # Configure options to use Claude 3.5 Haiku for cost efficiency
    # With enterprise subscription, no API key needed
    options = ClaudeCodeOptions(
        model="claude-3-5-haiku-20241022",  # Latest Haiku model
        max_turns=1,  # Simple single-turn interaction
        allowed_tools=[],  # No tools needed for text generation
        permission_mode="plan",  # Read-only mode since we're just generating descriptions
    )

    print("✅ Claude Code SDK client configured with Claude 3.5 Haiku model (Enterprise Subscription)")

    # Process each entry using Claude Code SDK
    async with ClaudeSDKClient(options=options) as client:
        for i, entry in enumerate(entries, 1):
            print(f"\nProcessing entry {i}/{len(entries)}: {entry.get('file_path', 'Unknown')}")
            print(f"  Class: {entry.get('class_name', 'N/A')}, Method: {entry.get('function_name', 'N/A')}")

            # Skip if already has description
            if "description" in entry and entry["description"]:
                print(f"  Skipping - already has description")
                continue

            # C:\repos\kit-app-template\_build\windows-x86_64\release\extscache
            # Get function body
            function_body = entry.get("function_body", "")
            if not function_body:
                print(f"  Warning: No function body found")
                entry["description"] = "No function body available"
                continue

            # Generate description
            print(f"  Generating description...")
            description = await generate_ui_description(
                client, function_body, entry.get("class_name", "Unknown"), entry.get("function_name", "Unknown")
            )

            # Add description to entry
            entry["description"] = description
            print(f"  Description: {description[:100]}...")

            # Rate limiting to avoid API throttling
            await asyncio.sleep(0.5)

            # Save progress every 10 entries
            if i % 10 == 0:
                print(f"\nSaving progress at entry {i}...")
                save_json_file(output_file_path, entries)

    # Final save
    print(f"\nSaving final results to: {output_file_path}")
    save_json_file(output_file_path, entries)
    print(f"Processing complete! All {len(entries)} entries have been processed.")

    # Print summary
    entries_with_desc = sum(1 for e in entries if "description" in e and e["description"])
    print(f"\nSummary: {entries_with_desc}/{len(entries)} entries now have descriptions")


async def main():
    """Main async function to run the processing"""
    # Configuration
    INPUT_JSON_FILE = "C:/repos/kit-app-template/ui_window_atlas.json"  # Change this to your actual file name
    OUTPUT_JSON_FILE = "C:/repos/kit-app-template/ui_functions_with_descriptions.json"  # Optional: separate output file

    # Check if input file exists
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"Error: Input file '{INPUT_JSON_FILE}' not found!")
        print("Please update the INPUT_JSON_FILE variable with the correct path to your JSON file.")
    else:
        # Process the entries
        await process_entries(INPUT_JSON_FILE, OUTPUT_JSON_FILE)


if __name__ == "__main__":
    asyncio.run(main())
