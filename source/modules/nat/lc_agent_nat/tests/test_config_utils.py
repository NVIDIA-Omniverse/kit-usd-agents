## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
import tempfile
import os
from lc_agent_nat.utils.config_utils import replace_md_file_references


def test_replace_md_file_references_none_input():
    result = replace_md_file_references(None)
    assert result is None


def test_replace_md_file_references_no_references():
    text = "This is a simple string without any file references."
    result = replace_md_file_references(text)
    assert result == text


def test_replace_md_file_references_non_md_reference():
    text = "This has a {file.txt} reference but not markdown."
    result = replace_md_file_references(text)
    assert result == text


def test_replace_md_file_references_missing_file():
    text = "This has a {nonexistent_file_12345.md} reference."
    result = replace_md_file_references(text)
    assert result == text


def test_replace_md_file_references_existing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        md_file = os.path.join(tmpdir, "test.md")
        with open(md_file, "w") as f:
            f.write("# Test Content\nThis is test markdown.")

        text = f"Before {{{md_file}}} After"
        result = replace_md_file_references(text)
        assert "# Test Content" in result
        assert "This is test markdown." in result
        assert "Before" in result
        assert "After" in result


def test_replace_md_file_references_multiple_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        md_file1 = os.path.join(tmpdir, "test1.md")
        md_file2 = os.path.join(tmpdir, "test2.md")

        with open(md_file1, "w") as f:
            f.write("Content One")
        with open(md_file2, "w") as f:
            f.write("Content Two")

        text = f"First: {{{md_file1}}} Second: {{{md_file2}}}"
        result = replace_md_file_references(text)
        assert "Content One" in result
        assert "Content Two" in result

