## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import omni.kit.test
from omni.ai.langchain.agent.usd_code.modifiers.double_run_utils import format_numbers_in_string, merge_layer_content
from pxr import Sdf, Usd


class TestInterpreter(omni.kit.test.AsyncTestCase):
    async def test_format_numbers_in_string(self):
        # Basic tests
        self.assertEqual(format_numbers_in_string("1.2345", 2), "1.23")
        self.assertEqual(format_numbers_in_string("1.2345", 3), "1.234")

        # Integer tests
        self.assertEqual(format_numbers_in_string("100", 2), "100")
        self.assertEqual(format_numbers_in_string("1000000", 2), "1000000")

        # Negative numbers
        self.assertEqual(format_numbers_in_string("-1.2345", 2), "-1.23")
        self.assertEqual(format_numbers_in_string("-100.5", 1), "-100.5")

        # Multiple numbers in a string
        self.assertEqual(format_numbers_in_string("1.23 4.56 7.89", 1), "1.2 4.6 7.9")

        # Numbers with exponents
        self.assertEqual(format_numbers_in_string("1.23e-4", 2), "0")
        self.assertEqual(format_numbers_in_string("1.23e4", 2), "12300")

        # Numbers starting with a dot
        self.assertEqual(format_numbers_in_string(".5 .25 .125", 2), "0.5 0.25 0.12")

        # Very small numbers
        self.assertEqual(format_numbers_in_string("0.0000001", 2), "0")
        self.assertEqual(format_numbers_in_string("0.0000001", 8), "0.0000001")

        # Very large numbers
        self.assertEqual(format_numbers_in_string("1234567890.1234567890", 2), "1234567890.12")

        # Mixed content
        self.assertEqual(format_numbers_in_string("Price: $19.99, Quantity: 3", 1), "Price: $20, Quantity: 3")

        # Numbers in brackets
        self.assertEqual(format_numbers_in_string("[1.2345, 5.6789]", 3), "[1.234, 5.679]")

        # Multiline string
        multiline_input = """
        Line 1: 1.2345
        Line 2: 6.7890
        Line 3: 9.8765
        """
        expected_output = """
        Line 1: 1.23
        Line 2: 6.79
        Line 3: 9.88
        """
        self.assertEqual(format_numbers_in_string(multiline_input, 2), expected_output)

        # Numbers with trailing zeros
        self.assertEqual(format_numbers_in_string("1.2300", 2), "1.23")
        self.assertEqual(format_numbers_in_string("1.2000", 2), "1.2")

        # Numbers with different precisions
        mixed_precision = "0.1 0.12 0.123 0.1234 0.12345"
        self.assertEqual(format_numbers_in_string(mixed_precision, 3), "0.1 0.12 0.123 0.123 0.123")

        # Edge cases
        self.assertEqual(format_numbers_in_string("", 2), "")
        self.assertEqual(format_numbers_in_string("No numbers here", 2), "No numbers here")
        self.assertEqual(format_numbers_in_string("3.14.15.9", 2), "3.14.15.9")  # Not a valid number

        # Numbers next to words
        self.assertEqual(format_numbers_in_string("abc1.23def4.56ghi", 1), "abc1.23def4.56ghi")

        # Numbers with thousands separators (not handled by the current implementation)
        self.assertEqual(format_numbers_in_string("1,234,567.89", 2), "1,234,567.89")

        # Scientific notation
        self.assertEqual(format_numbers_in_string("1.23e-4 5.67E+8", 2), "0 567000000")

        # Mixed positive and negative numbers
        self.assertEqual(format_numbers_in_string("-1.23 +4.56 -7.89", 1), "-1.2 4.6 -7.9")

        # Numbers at the start and end of the string
        self.assertEqual(format_numbers_in_string("1.23 middle 4.56", 1), "1.2 middle 4.6")

        # Repeated decimals
        self.assertEqual(format_numbers_in_string("1.3333333", 2), "1.33")
        self.assertEqual(format_numbers_in_string("1.6666666", 2), "1.67")

        # Additional test cases for precision
        self.assertEqual(format_numbers_in_string("1.2345", 4), "1.2345")
        self.assertEqual(format_numbers_in_string("1.23456", 4), "1.2346")
        self.assertEqual(format_numbers_in_string("1.23454", 4), "1.2345")

        # Test rounding behavior
        self.assertEqual(format_numbers_in_string("1.235", 2), "1.24")
        self.assertEqual(format_numbers_in_string("1.225", 2), "1.23")
        self.assertEqual(format_numbers_in_string("-1.235", 2), "-1.24")
        self.assertEqual(format_numbers_in_string("-1.225", 2), "-1.23")

    async def test_merge_layer_content(self):
        # Create source and destination layers
        src_layer = Sdf.Layer.CreateAnonymous()
        dst_layer = Sdf.Layer.CreateAnonymous()

        # Test 1: Merging a simple prim
        src_prim = Sdf.CreatePrimInLayer(src_layer, "/TestPrim")
        src_prim.specifier = Sdf.SpecifierDef
        src_prim.typeName = "Xform"

        merge_layer_content(src_layer, dst_layer)

        dst_prim = dst_layer.GetPrimAtPath("/TestPrim")
        self.assertIsNotNone(dst_prim)
        self.assertEqual(dst_prim.specifier, Sdf.SpecifierDef)
        self.assertEqual(dst_prim.typeName, "Xform")

        # Test 2: Merging properties
        src_attr = Sdf.AttributeSpec(src_prim, "testAttr", Sdf.ValueTypeNames.Double)
        src_attr.default = 1.0

        merge_layer_content(src_layer, dst_layer)

        dst_attr = dst_prim.attributes.get("testAttr")
        self.assertIsNotNone(dst_attr)
        self.assertEqual(dst_attr.default, 1.0)

        # Test 3: Merging nested prims
        src_child = Sdf.CreatePrimInLayer(src_layer, "/TestPrim/Child")
        src_child.specifier = Sdf.SpecifierDef

        merge_layer_content(src_layer, dst_layer)

        dst_child = dst_layer.GetPrimAtPath("/TestPrim/Child")
        self.assertIsNotNone(dst_child)
        self.assertEqual(dst_child.specifier, Sdf.SpecifierDef)

        # Test 4: Merging over existing content
        dst_existing = Sdf.CreatePrimInLayer(dst_layer, "/ExistingPrim")
        dst_existing.specifier = Sdf.SpecifierDef

        src_over = Sdf.CreatePrimInLayer(src_layer, "/ExistingPrim")
        src_over.specifier = Sdf.SpecifierOver
        src_attr = Sdf.AttributeSpec(src_over, "overAttr", Sdf.ValueTypeNames.String)
        src_attr.default = "overValue"

        merge_layer_content(src_layer, dst_layer)

        dst_merged = dst_layer.GetPrimAtPath("/ExistingPrim")
        self.assertIsNotNone(dst_merged)
        self.assertEqual(dst_merged.specifier, Sdf.SpecifierDef)  # Should remain Def
        self.assertIsNotNone(dst_merged.attributes.get("overAttr"))
        self.assertEqual(dst_merged.attributes.get("overAttr").default, "overValue")

        # Test 5: Merging relationships
        src_rel = Sdf.RelationshipSpec(src_prim, "testRel", custom=True, variability=Sdf.VariabilityUniform)
        src_rel.targetPathList.explicitItems = ["/TestPrim/Child"]

        merge_layer_content(src_layer, dst_layer)

        dst_rel = dst_prim.relationships.get("testRel")
        self.assertIsNotNone(dst_rel)
        self.assertEqual(dst_rel.targetPathList.explicitItems, ["/TestPrim/Child"])
