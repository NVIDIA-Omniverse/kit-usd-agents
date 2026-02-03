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
Analyze the token size of the combined styling documentation.
"""

import tiktoken


def analyze_combined_file():
    """Analyze the token count of the combined documentation file."""

    file_path = "/home/horde/repos/kit-lc-agent/source/mcp/omni_ui_mcp/all_styling_combined.md"

    # Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Count tokens
    encoding = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoding.encode(content))

    # Calculate statistics
    lines = content.count("\n")
    chars = len(content)
    words = len(content.split())
    size_kb = len(content.encode("utf-8")) / 1024

    print("=" * 60)
    print("TOKEN ANALYSIS: all_styling_combined.md")
    print("=" * 60)
    print(f"Token Count: {token_count:,} tokens")
    print(f"Lines: {lines:,}")
    print(f"Characters: {chars:,}")
    print(f"Words: {words:,}")
    print(f"File Size: {size_kb:.2f} KB")
    print(f"Tokens per Word: {token_count/words:.2f}")
    print(f"Tokens per KB: {token_count/size_kb:.2f}")
    print("=" * 60)

    return token_count


if __name__ == "__main__":
    analyze_combined_file()
