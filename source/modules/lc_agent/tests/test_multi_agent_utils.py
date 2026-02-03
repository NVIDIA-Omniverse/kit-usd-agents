## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from lc_agent.utils.multi_agent_utils import (
    _line_starts_with_action,
    _find_action_at_line_start,
    parse_classification_result,
)
from unittest.mock import MagicMock


class TestLineStartsWithAction:
    """Tests for _line_starts_with_action helper function."""

    def test_exact_match_returns_action(self):
        valid_actions = ["FINAL", "KitInfo", "UsdQuery"]
        assert _line_starts_with_action("FINAL", valid_actions) == "FINAL"

    def test_action_with_content_returns_action(self):
        valid_actions = ["FINAL", "KitInfo", "UsdQuery"]
        assert _line_starts_with_action("FINAL answer here", valid_actions) == "FINAL"

    def test_case_insensitive_match(self):
        valid_actions = ["FINAL", "KitInfo", "UsdQuery"]
        assert _line_starts_with_action("final answer", valid_actions) == "FINAL"
        assert _line_starts_with_action("kitinfo What is this?", valid_actions) == "KitInfo"
        assert _line_starts_with_action("KITINFO What is this?", valid_actions) == "KitInfo"

    def test_leading_whitespace_is_stripped(self):
        valid_actions = ["FINAL", "KitInfo"]
        assert _line_starts_with_action("  FINAL answer", valid_actions) == "FINAL"
        assert _line_starts_with_action("\t\tKitInfo query", valid_actions) == "KitInfo"

    def test_partial_match_returns_none(self):
        valid_actions = ["FINAL", "KitInfo"]
        # "FINALLY" should not match "FINAL"
        assert _line_starts_with_action("FINALLY done", valid_actions) is None
        # "KitInfoExtra" should not match "KitInfo"
        assert _line_starts_with_action("KitInfoExtra", valid_actions) is None

    def test_no_match_returns_none(self):
        valid_actions = ["FINAL", "KitInfo"]
        assert _line_starts_with_action("Something else", valid_actions) is None
        assert _line_starts_with_action("The answer is FINAL", valid_actions) is None

    def test_empty_line_returns_none(self):
        valid_actions = ["FINAL", "KitInfo"]
        assert _line_starts_with_action("", valid_actions) is None
        assert _line_starts_with_action("   ", valid_actions) is None

    def test_action_followed_by_newline_or_space(self):
        valid_actions = ["FINAL", "KitInfo"]
        assert _line_starts_with_action("KitInfo\t", valid_actions) == "KitInfo"
        assert _line_starts_with_action("FINAL ", valid_actions) == "FINAL"


class TestFindActionAtLineStart:
    """Tests for _find_action_at_line_start helper function."""

    def test_action_on_first_line(self):
        route_nodes = ["KitInfo", "UsdQuery"]
        text = "KitInfo What is the user name?"
        action, content = _find_action_at_line_start(text, route_nodes)
        assert action == "KitInfo"
        assert content == "What is the user name?"

    def test_final_action(self):
        route_nodes = ["KitInfo", "UsdQuery"]
        text = "FINAL Victor is the answer"
        action, content = _find_action_at_line_start(text, route_nodes)
        assert action == "FINAL"
        assert content == "Victor is the answer"

    def test_action_with_preamble(self):
        route_nodes = ["KitInfo", "UsdQuery"]
        text = "Now I understand\n\nKitInfo What is the user name?"
        action, content = _find_action_at_line_start(text, route_nodes)
        assert action == "KitInfo"
        assert content == "What is the user name?"

    def test_action_with_multiline_content(self):
        route_nodes = ["KitInfo", "UsdQuery"]
        text = "Now I understand\n\nKitInfo What is the user name?\nCheck the tools"
        action, content = _find_action_at_line_start(text, route_nodes)
        assert action == "KitInfo"
        assert content == "What is the user name?\nCheck the tools"

    def test_action_content_stops_at_next_action(self):
        route_nodes = ["KitInfo", "UsdQuery"]
        text = "Now I understand\n\nKitInfo What is the user name?\nCheck the tools\nKitInfo What is the address"
        action, content = _find_action_at_line_start(text, route_nodes)
        assert action == "KitInfo"
        assert content == "What is the user name?\nCheck the tools"

    def test_final_action_with_preamble(self):
        route_nodes = ["KitInfo", "UsdQuery"]
        text = "The user said Victor\nFINAL Victor"
        action, content = _find_action_at_line_start(text, route_nodes)
        assert action == "FINAL"
        assert content == "Victor"

    def test_no_action_returns_none(self):
        route_nodes = ["KitInfo", "UsdQuery"]
        text = "I'm thinking about this\nStill thinking"
        action, content = _find_action_at_line_start(text, route_nodes)
        assert action is None
        assert content is None

    def test_empty_text_returns_none(self):
        route_nodes = ["KitInfo", "UsdQuery"]
        action, content = _find_action_at_line_start("", route_nodes)
        assert action is None
        assert content is None

    def test_action_only_no_content(self):
        route_nodes = ["KitInfo", "UsdQuery"]
        text = "Let me check\nKitInfo"
        action, content = _find_action_at_line_start(text, route_nodes)
        assert action == "KitInfo"
        assert content is None

    def test_case_insensitive_action_detection(self):
        route_nodes = ["KitInfo", "UsdQuery"]
        text = "Let me help\nkitinfo What is this?"
        action, content = _find_action_at_line_start(text, route_nodes)
        assert action == "KitInfo"
        assert content == "What is this?"

    def test_action_embedded_in_word_not_matched(self):
        route_nodes = ["Info", "Query"]
        text = "KitInfo should not match\nInfo What is this?"
        action, content = _find_action_at_line_start(text, route_nodes)
        assert action == "Info"
        assert content == "What is this?"


class TestParseClassificationResult:
    """Tests for parse_classification_result function."""

    def _create_mock_network(self, route_nodes):
        """Helper to create a mock network with route_nodes."""
        mock_network = MagicMock()
        mock_network.route_nodes = route_nodes
        # Mock get_leaf_node to return a node that will make _get_human_node_and_tools return quickly
        mock_leaf_node = MagicMock()
        mock_leaf_node.parents = []
        mock_leaf_node.metadata = {}
        mock_network.get_leaf_node.return_value = mock_leaf_node
        return mock_network

    def test_simple_action(self):
        network = self._create_mock_network(["KitInfo", "UsdQuery"])
        result = parse_classification_result("KitInfo What is the user name?", network)
        assert result is not None
        assert result["action"] == "KitInfo"
        assert result["content"] == "What is the user name?"
        assert result["full"] == "KitInfo What is the user name?"
        assert result["is_loop"] is False

    def test_final_action(self):
        network = self._create_mock_network(["KitInfo", "UsdQuery"])
        result = parse_classification_result("FINAL The answer is 42", network)
        assert result is not None
        assert result["action"] == "FINAL"
        assert result["content"] == "The answer is 42"
        assert result["is_loop"] is False

    def test_action_with_preamble(self):
        network = self._create_mock_network(["KitInfo", "UsdQuery"])
        text = "Let me think about this...\n\nKitInfo What is the user name?"
        result = parse_classification_result(text, network)
        assert result is not None
        assert result["action"] == "KitInfo"
        assert result["content"] == "What is the user name?"
        assert result["full"] == text

    def test_final_with_preamble(self):
        network = self._create_mock_network(["KitInfo", "UsdQuery"])
        text = "Based on the previous results\nFINAL The user is Victor"
        result = parse_classification_result(text, network)
        assert result is not None
        assert result["action"] == "FINAL"
        assert result["content"] == "The user is Victor"

    def test_empty_result_returns_none(self):
        network = self._create_mock_network(["KitInfo", "UsdQuery"])
        assert parse_classification_result("", network) is None
        assert parse_classification_result("   ", network) is None

    def test_no_valid_action_returns_none(self):
        network = self._create_mock_network(["KitInfo", "UsdQuery"])
        assert parse_classification_result("I don't know what to do", network) is None
        assert parse_classification_result("Something about KitInfo but not at start", network) is None

    def test_case_insensitive_action(self):
        network = self._create_mock_network(["KitInfo", "UsdQuery"])
        result = parse_classification_result("kitinfo What is this?", network)
        assert result is not None
        assert result["action"] == "KitInfo"

        result = parse_classification_result("final Done", network)
        assert result is not None
        assert result["action"] == "FINAL"

    def test_action_with_multiline_content(self):
        network = self._create_mock_network(["KitInfo", "UsdQuery"])
        text = "KitInfo What is the name?\nAlso check the address"
        result = parse_classification_result(text, network)
        assert result is not None
        assert result["action"] == "KitInfo"
        assert result["content"] == "What is the name?\nAlso check the address"

    def test_action_without_content(self):
        network = self._create_mock_network(["KitInfo", "UsdQuery"])
        result = parse_classification_result("KitInfo", network)
        assert result is not None
        assert result["action"] == "KitInfo"
        assert result["content"] is None


if __name__ == "__main__":
    pytest.main(["-v", "test_multi_agent_utils.py"])

