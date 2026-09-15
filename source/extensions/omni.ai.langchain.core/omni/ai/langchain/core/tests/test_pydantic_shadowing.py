# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.

# SPDX-License-Identifier: Apache-2.0

import os

import omni.kit.test


class TestPydanticPrebundleShadowing(omni.kit.test.AsyncTestCase):
    """Regression for internal issue 6585942: the prebundled pydantic must win over pip_archive's."""

    async def test_langchain_toolcall_schema_builds_with_prebundled_pydantic(self):
        import annotated_types
        import pydantic
        import pydantic_core
        import typing_inspection

        # The whole stack must resolve to pip_core_prebundle, not pip_archive.
        for module in (pydantic, pydantic_core, typing_inspection, annotated_types):
            resolved = os.path.abspath(module.__file__)
            self.assertNotIn(
                "pip_archive",
                resolved,
                f"pip_archive is shadowing {module.__name__} (resolved {resolved}); pydantic={pydantic.VERSION}",
            )
            self.assertIn(
                "pip_core_prebundle",
                resolved,
                f"{module.__name__} did not resolve to pip_core_prebundle (resolved {resolved})",
            )

        # Importing AIMessage builds a schema for its list[ToolCall] field (internal issue 6585942).
        try:
            from langchain_core.messages.ai import AIMessage
            from langchain_core.messages.tool import ToolCall
        except Exception as exc:
            self.fail(f"importing langchain_core AIMessage/ToolCall raised {type(exc).__name__}: {exc}")

        message = AIMessage(content="hi", tool_calls=[ToolCall(name="fn", args={}, id="call-1")])
        self.assertEqual(message.tool_calls[0]["name"], "fn")
