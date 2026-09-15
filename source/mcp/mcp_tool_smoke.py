#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test every tool on every running MCP, 3 inputs per tool.

Usage::

    python3 source/mcp/mcp_tool_smoke.py [--mcp NAME=PORT ...] [--timeout SEC] [--verbose]

Defaults match the public local-compose stack
(``source/mcp/docker-compose.local.yaml``) and the QUICKSTART
port table::

    omni_ui=9901 kit=9902 usd_code=9903 isaacsim=9904

How it talks to the MCP servers:

- POST to ``http://localhost:<PORT>/mcp`` with the ``application/json,
  text/event-stream`` Accept header (NAT 1.8 streamable-http endpoint).
- Captures ``mcp-session-id`` from the initialize response and threads it
  through subsequent requests.
- Parses the first ``data: {...}`` SSE line as the JSON-RPC envelope.

Result categories per (mcp, tool, test):

- ``PASS``  : ``isError=false`` AND the response text is non-trivial
  (>200 chars OR contains an opening ``{`` / ``#`` markdown / "Title '").
- ``EMPTY`` : ``isError=false`` but the response looks like an explicit
  empty-result sentinel (e.g. "No code examples found") or is shorter
  than the heuristic threshold without a structured-result marker.
- ``ERROR`` : MCP-level JSON-RPC error, transport failure, or
  ``isError=true`` in the tool result.

Exit code is ``0`` iff no ``ERROR`` rows were produced.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any

# ---- Test inputs -----------------------------------------------------------
# Hand-curated 3 inputs per tool. Inputs exercise the tool's primary use case,
# not edge cases — the goal is "does this tool produce a real result against a
# real index" rather than thorough functional testing.

TESTS: dict[str, dict[str, list[dict[str, Any]]]] = {
    "kit": {
        "get_kit_instructions": [
            {"instruction_sets": "kit_system"},
            {"instruction_sets": ["extensions", "testing"]},
            {"instruction_sets": None},
        ],
        "search_kit_extensions": [
            {"query": "window management"},
            {"query": "viewport rendering"},
            {"query": "ui widgets and controls"},
        ],
        "get_kit_extension_details": [
            {"extension_ids": "omni.ui"},
            {"extension_ids": ["omni.kit.widget.viewport"]},
            {"extension_ids": None},
        ],
        "get_kit_extension_dependencies": [
            {"extension_id": "omni.ui", "depth": 1},
            {"extension_id": "omni.kit.widget.viewport", "depth": 2},
            {"extension_id": "omni.ui.scene", "include_optional": True},
        ],
        "get_kit_extension_apis": [
            {"extension_ids": "omni.ui"},
            {"extension_ids": ["omni.kit.app", "omni.ui"]},
            {"extension_ids": None},
        ],
        "get_kit_api_details": [
            {"api_references": "omni.ui@Window"},
            {"api_references": ["omni.ui@Button", "omni.ui@Label"]},
            {"api_references": None},
        ],
        "search_kit_code_examples": [
            {"query": "subscribe to stage events"},
            {"query": "create a window with a button"},
            {"query": "register an extension lifecycle handler"},
        ],
        "search_kit_test_examples": [
            {"query": "async test pattern"},
            {"query": "test ui widget"},
            {"query": "extension lifecycle test"},
        ],
        "search_kit_settings": [
            {"query": "rtx rendermode"},
            {"query": "viewport camera"},
            {"query": "rendering", "prefix_filter": "rtx"},
        ],
        "search_kit_app_templates": [
            {"query": "usd viewer"},
            {"query": "streaming application"},
            {"query": "content authoring tool"},
        ],
        "get_kit_app_template_details": [
            {"template_ids": "kit_base_editor"},
            {"template_ids": ["usd_composer", "usd_viewer"]},
            {"template_ids": None},
        ],
        "search_kit_knowledge": [
            {"request": "How does extension lifecycle work?"},
            {"request": "What is RTX rendering in Kit?"},
            {"request": "How do I write a Kit unit test?"},
        ],
    },
    "isaacsim": {
        "get_isaac_sim_instructions": [
            {"instruction_sets": "isaacsim_system"},
            {"instruction_sets": None},
            {"instruction_sets": ["isaacsim_system"]},
        ],
        "search_isaac_sim_extensions": [
            {"query": "robotics manipulation"},
            {"query": "sensors and cameras"},
            {"query": "physics simulation"},
        ],
        "get_isaac_sim_extension_details": [
            {"extension_ids": None},
            {"extension_ids": ["isaacsim.core.api"]},
            {"extension_ids": "isaacsim.sensors.camera"},
        ],
        "search_isaac_sim_code_examples": [
            {"query": "create a cube prim"},
            {"query": "set up a camera"},
            {"query": "spawn a robot"},
        ],
        "search_isaac_sim_settings": [
            {"query": "physics"},
            {"query": "isaac startup"},
            {"query": "rtx", "prefix_filter": "rtx"},
        ],
    },
    "omni_ui": {
        "search_ui_code_examples": [
            {"query": "create a button"},
            {"query": "vstack layout"},
            {"query": "search field widget"},
        ],
        "search_ui_window_examples": [
            {"query": "modal dialog with buttons"},
            {"query": "settings window with sliders"},
            {"query": "error message dialog"},
        ],
        "list_ui_classes": [{}, {}, {}],
        "list_ui_modules": [{}, {}, {}],
        "get_ui_class_detail": [
            {"class_names": "Button"},
            {"class_names": ["TreeView", "Window"]},
            {"class_names": None},
        ],
        "get_ui_module_detail": [
            {"module_names": "omni.ui"},
            {"module_names": ["omni.ui", "omni.ui.scene"]},
            {"module_names": None},
        ],
        "get_ui_method_detail": [
            {"method_names": "__init__"},
            {"method_names": ["set_value", "get_value"]},
            {"method_names": None},
        ],
        "get_ui_instructions": [
            {"name": "agent_system"},
            {"name": "omni_ui_system"},
            {"name": None},
        ],
        "get_ui_class_instructions": [
            {"class_names": "Button"},
            {"class_names": ["TreeView", "Window"]},
            {"class_names": "categories"},
        ],
        "get_ui_style_docs": [
            {"sections": "buttons"},
            {"sections": ["shades", "fonts"]},
            {"sections": None},
        ],
    },
    "usd_code": {
        "search_usd_code_examples": [
            {"request": "create a USD stage"},
            {"request": "add a cube prim"},
            {"request": "set material on prim"},
        ],
        "search_usd_knowledge": [
            {"request": "What is USD?"},
            {"request": "How does composition work?"},
            {"request": "What are USD payloads?"},
        ],
        "list_usd_modules": [{}, {}, {}],
        "list_usd_classes": [{}, {}, {}],
        "get_usd_module_detail": [
            # NOTE: the tool's pydantic schema rejects lists despite the
            # docstring claiming list support — pass only strings. Tracked
            # as a pre-existing tool-input-schema bug separate from !658.
            {"module_names": "pxr.Usd"},
            {"module_names": "pxr.UsdGeom"},
            {"module_names": "pxr.Sdf"},
        ],
        "get_usd_class_detail": [
            {"class_names": "Stage"},
            {"class_names": "Prim"},
            {"class_names": "Sdf.Path"},
        ],
        "get_usd_method_detail": [
            {"method_names": "DefinePrim"},
            {"method_names": "GetPrimAtPath"},
            {"method_names": "Save"},
        ],
    },
}


# ---- MCP transport ---------------------------------------------------------


@dataclass
class McpClient:
    name: str
    port: int
    session_id: str | None = None
    timeout: float = 120.0

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/mcp"

    def _post(self, payload: dict, capture_header: str | None = None) -> tuple[Any, str | None]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            captured_header = resp.headers.get(capture_header) if capture_header else None
            body = resp.read().decode()
        data = None
        for line in body.splitlines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                break
        return data, captured_header

    def initialize(self) -> None:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "smoke", "version": "1"},
            },
        }
        _, session = self._post(payload, capture_header="mcp-session-id")
        self.session_id = session
        # Required notifications/initialized after initialize.
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call_tool(self, tool: str, args: dict) -> tuple[str, str, str]:
        """Return ``(category, snippet, error_msg)``.

        category ∈ {PASS, EMPTY, ERROR}.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000) % 1_000_000,
            "method": "tools/call",
            "params": {"name": tool, "arguments": args},
        }
        try:
            data, _ = self._post(payload)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            return ("ERROR", "", f"transport: {e!r}")
        if not data:
            return ("ERROR", "", "no SSE data line in response")
        if "error" in data:
            return ("ERROR", "", f"jsonrpc error: {data['error']}")
        result = data.get("result") or {}
        if result.get("isError"):
            content = result.get("content") or [{}]
            err_text = content[0].get("text", "") if content else ""
            return ("ERROR", err_text[:200], "tool isError")
        content = result.get("content") or []
        text = ""
        for c in content:
            if c.get("type") == "text":
                text += c.get("text", "")
        text_stripped = text.strip()
        snippet = text_stripped[:160].replace("\n", " ")
        if not text_stripped:
            return ("EMPTY", snippet, "")
        # Explicit empty-result sentinels (lowercase match).
        sentinels_lc = (
            "no code examples found",
            "no relevant",
            "no test examples found",
            "no knowledge found",
            "no usd knowledge found",
            "no isaac sim",
            "no settings found",
            "no templates found",
            "no examples found",
        )
        low = text_stripped.lower()
        if any(s in low for s in sentinels_lc):
            return ("EMPTY", snippet, "")
        # Structured response (JSON / markdown / RAG title) — accept as PASS even if short.
        if text_stripped.startswith(("{", "[", "#", "Title '", "Question:")):
            return ("PASS", snippet, "")
        # Fallback: require enough text for the result to look real.
        if len(text_stripped) < 200:
            return ("EMPTY", snippet, "")
        return ("PASS", snippet, "")


# ---- Test runner -----------------------------------------------------------


def run(mcps: dict[str, int], timeout: float, verbose: bool) -> int:
    overall: Counter[str] = Counter()
    by_mcp: dict[str, Counter[str]] = {}
    failures: list[tuple[str, str, dict, str, str]] = []
    print("MCP                    Tool                                  Test  Result   Snippet")
    print("-" * 110)
    for mcp_name, port in mcps.items():
        client = McpClient(mcp_name, port, timeout=timeout)
        try:
            client.initialize()
        except Exception as e:
            print(f"{mcp_name}: initialize failed: {e!r}")
            overall["ERROR"] += 1
            failures.append((mcp_name, "<init>", {}, "ERROR", repr(e)))
            continue
        tests_for_mcp = TESTS.get(mcp_name, {})
        for tool, inputs in tests_for_mcp.items():
            for i, args in enumerate(inputs, 1):
                cat, snippet, err = client.call_tool(tool, args)
                overall[cat] += 1
                by_mcp.setdefault(mcp_name, Counter())[cat] += 1
                marker = {"PASS": "✓", "EMPTY": "·", "ERROR": "✗"}[cat]
                if verbose or cat == "ERROR":
                    s = snippet if cat != "ERROR" else (err or snippet)
                    print(f"{mcp_name:<22} {tool:<40} #{i}    {marker} {cat:<6} {s[:60]}")
                if cat == "ERROR":
                    failures.append((mcp_name, tool, args, cat, err))
    print("-" * 110)
    print("Per-MCP summary:")
    for mcp, c in sorted(by_mcp.items()):
        total = sum(c.values())
        print(f"  {mcp:<10} PASS={c['PASS']:>3}  EMPTY={c['EMPTY']:>3}  ERROR={c['ERROR']:>3}  total={total}")
    print(
        f"Overall:    PASS={overall['PASS']:>3}  EMPTY={overall['EMPTY']:>3}  "
        f"ERROR={overall['ERROR']:>3}  total={sum(overall.values())}"
    )
    if failures:
        print("\nFailures:")
        for mcp, tool, args, cat, msg in failures:
            print(f"  [{mcp}] {tool} {json.dumps(args)} → {cat}: {msg}")
    return 0 if overall["ERROR"] == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mcp",
        action="append",
        default=[],
        help=(
            "Add an MCP as name=port (can be repeated). Defaults to "
            "omni_ui=9901 kit=9902 usd_code=9903 isaacsim=9904."
        ),
    )
    p.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout (seconds).")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print every test, not just failures.",
    )
    args = p.parse_args()

    if args.mcp:
        mcps = {}
        for spec in args.mcp:
            name, port = spec.split("=", 1)
            mcps[name] = int(port)
    else:
        mcps = {"omni_ui": 9901, "kit": 9902, "usd_code": 9903, "isaacsim": 9904}

    return run(mcps, args.timeout, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
