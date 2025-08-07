## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""
Runnable node implementation for Function wrappers.

This module provides a RunnableNode that wraps a Function and invokes it.
"""

from aiq.builder.function import Function
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from lc_agent import RunnableNode
from pydantic import Field
from pydantic import model_serializer
from typing import Any, List, Optional, Union, Dict
import json
import re
import traceback
import datetime


class FunctionRunnableNode(RunnableNode):
    """A RunnableNode that wraps a Function and invokes it."""

    function: Any = Field(None, description="The function to wrap")
    outputs: Optional[Union[List[BaseMessage], BaseMessage]] = None

    def __init__(self, function, *args, **kwargs):
        # Pass function as a keyword argument to ensure it's properly set during validation
        kwargs["function"] = function
        super().__init__(*args, **kwargs)

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Pydantic 2 serialization method using model_serializer"""
        # Create a base dictionary with all fields except parents
        result = {}
        for field_name, field_value in self:
            if field_name not in ["modifiers", "callbacks", "parents", "function"]:
                result[field_name] = field_value

        # Add type information
        result["__node_type__"] = self.__class__.__name__

        return result

    def _get_chat_model(self, chat_model_name, chat_model_input, invoke_input, config) -> Function:
        """Get the function for this node."""
        # Return the function directly
        return self.function

    def _get_input_value(self, chat_model_input, invoke_input):
        """Extract the input value for function invocation from available sources."""
        # Get the content from the last message or from tool_call_content
        tool_call_content = self.find_metadata("tool_call_content")

        if tool_call_content is not None:
            return tool_call_content
        elif chat_model_input and len(chat_model_input) > 0:
            return chat_model_input[-1].content
        else:
            # Fallback to invoke_input if nothing else is available
            return invoke_input

    def _convert_value_to_schema(self, value, input_schema):
        """Convert a string value to the appropriate input schema type.

        Args:
            value: The input value (typically a string)
            input_schema: The Pydantic model class that defines the schema

        Returns:
            An instance of the input_schema class
        """
        # Get the model fields
        model_fields = input_schema.model_fields
        field_names = list(model_fields.keys())

        # Case 1: If schema has only one field, try direct conversion
        if len(field_names) == 1:
            field_name = field_names[0]
            try:
                return input_schema(**{field_name: value})
            except Exception:
                pass

        # Case 2: If schema has no fields, return the value as is
        elif len(field_names) == 0:
            return input_schema()

        # Case 3: Handle multi-field schemas
        else:
            # Try parsing as JSON
            try:
                json_data = json.loads(value)
                return input_schema(**json_data)
            except json.JSONDecodeError:
                pass

            # Use value as first field and defaults for others
            try:
                kwargs = {field_names[0]: value}
                # Add empty strings for other required string fields
                for name in field_names[1:]:
                    if model_fields[name].annotation == str and name in input_schema.model_fields_set:
                        kwargs[name] = ""
                return input_schema(**kwargs)
            except Exception:
                pass

        # Provide detailed error information when conversion fails
        field_descriptions = []
        for name, field in model_fields.items():
            field_type = field.annotation.__name__ if hasattr(field.annotation, "__name__") else str(field.annotation)
            description = field.description or ""
            field_descriptions.append(f"- {name} ({field_type}): {description}")

        # Join descriptions with newlines first, then use in f-string
        joined_descriptions = "\n".join(field_descriptions)
        error_msg = (
            f"Could not convert input to required schema. Input should be JSON formatted with these fields:\n"
            f"{joined_descriptions}"
        )
        return error_msg

    def _format_error_message(self, function, error, input_schema=None, parsed_input=None):
        """Format an error message with useful debug information.

        Args:
            function: The function that raised the exception
            error: The exception object
            input_schema: The input schema used for validation
            parsed_input: The actual parsed input passed to the function

        Returns:
            A formatted error message string
        """
        # Extract the real function name - try several common attributes
        function_name = None
        for attr in ["name", "function_name", "__name__", "func_name"]:
            if hasattr(function, attr):
                name_value = getattr(function, attr)
                if isinstance(name_value, str) and name_value not in ["FunctionImpl", "Function"]:
                    function_name = name_value
                    break

        # Fallback if we couldn't find a good name
        if not function_name:
            if hasattr(function, "config"):
                function_name = function.config.type
            else:
                function_name = str(function.__class__.__name__)

        error_type = error.__class__.__name__
        error_message = str(error)

        # Get a simplified traceback (last 3 frames)
        tb_lines = traceback.format_exc().strip().split("\n")
        if len(tb_lines) > 6:
            tb_excerpt = "\n".join(tb_lines[-6:])
        else:
            tb_excerpt = "\n".join(tb_lines)

        # Format the message in a structured way similar to success messages
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Add detailed schema information if available
        schema_info = ""
        if input_schema:
            schema_name = getattr(input_schema, "__name__", input_schema.__class__.__name__)
            # Get field descriptions in a detailed format
            field_descriptions = []
            for name, field in input_schema.model_fields.items():
                field_type = (
                    field.annotation.__name__ if hasattr(field.annotation, "__name__") else str(field.annotation)
                )
                description = field.description or ""
                field_descriptions.append(f"  - {name} ({field_type}): {description}")

            # Join field descriptions first to avoid backslash in f-string expression
            joined_field_descriptions = "\n".join(field_descriptions)
            schema_info = f"""
SCHEMA INFORMATION:
  Name:        {schema_name}
  Fields:
{joined_field_descriptions}"""

        # Add input value information in a readable format
        input_info = ""
        if parsed_input is not None:
            input_type = type(parsed_input).__name__

            # Format field values, one per line
            if hasattr(parsed_input, "model_fields") and hasattr(parsed_input, "model_dump"):
                try:
                    field_values = []
                    data = parsed_input.model_dump()
                    for field_name, field_value in data.items():
                        field_str = str(field_value)
                        if len(field_str) > 100:  # Truncate long values
                            field_str = field_str[:100] + "... [truncated]"
                        field_values.append(f"  - {field_name} = {field_str}")

                    # Join field values first to avoid backslash in f-string expression
                    joined_input_value = "\n".join(field_values)
                    input_value = joined_input_value
                except Exception:
                    input_value = str(parsed_input)
            else:
                input_value = str(parsed_input)
                if len(input_value) > 500:
                    input_value = input_value[:500] + "... [truncated]"

            input_info = f"""
INPUT VALUE:
  Type:        {input_type}
  Values:
{input_value}"""

        formatted_message = f"""FUNCTION EXECUTION - ERROR
--------------------------------------------------------------------------------
FUNCTION INFORMATION:
  Name:        {function_name}
  Type:        {function.__class__.__name__}{schema_info}{input_info}
  
ERROR DETAILS:
  Type:        {error_type}
  Message:     {error_message}
  Timestamp:   {timestamp}
  
TRACEBACK:
{tb_excerpt}
--------------------------------------------------------------------------------"""

        return formatted_message

    async def _ainvoke_chat_model(self, function: Function, chat_model_input, invoke_input, config, **kwargs):
        """Invoke function for this node."""
        # Save chat model input to payload.txt if applicable
        self._save_chat_model_input_to_payload(chat_model_input)

        # Get the input value
        value = self._get_input_value(chat_model_input, invoke_input)

        # Get the input schema
        input_schema = function.input_schema

        # Convert string value to appropriate input schema
        if isinstance(value, str):
            converted_value = self._convert_value_to_schema(value, input_schema)
            if isinstance(converted_value, str):  # Error message
                return AIMessage(content=converted_value)
            value = converted_value

        # Call function's ainvoke with the converted value, handling any exceptions
        try:
            result = await function.ainvoke(value)
            return AIMessage(content=result)
        except Exception as e:
            error_message = self._format_error_message(function, e, input_schema, value)
            print(f"Function execution error: {error_message}")
            return AIMessage(content=error_message)

    async def _astream_chat_model(self, function, chat_model_input, invoke_input, config, **kwargs):
        """Stream results from the function."""
        # Save chat model input to payload.txt if applicable
        self._save_chat_model_input_to_payload(chat_model_input)

        # Get the input value
        value = self._get_input_value(chat_model_input, invoke_input)

        # Get input schema and convert value if necessary
        input_schema = function.input_schema
        if isinstance(value, str):
            converted_value = self._convert_value_to_schema(value, input_schema)
            if isinstance(converted_value, str):  # Error message
                yield AIMessageChunk(content=converted_value)
                return
            value = converted_value

        # Stream results from the function, handling any exceptions
        try:
            async for item in function.astream(value):
                yield AIMessageChunk(content=item)
        except Exception as e:
            error_message = self._format_error_message(function, e, input_schema, value)
            print(f"Function streaming error: {error_message}")
            yield AIMessageChunk(content=error_message)

    def _invoke_chat_model(self, function, chat_model_input, invoke_input, config, **kwargs):
        """Synchronous invoke for this node (not typically used)."""
        # Function interface is async, so this is not generally used
        raise NotImplementedError("Synchronous invocation not supported for Function nodes")
