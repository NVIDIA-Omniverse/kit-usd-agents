:: SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
:: SPDX-License-Identifier: Apache-2.0
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
:: http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.

@echo off
REM Windows script to run OmniUI MCP Server locally
REM This script starts the server for local development

echo ========================================
echo OmniUI MCP Server - Local Development
echo ========================================
echo.

REM Check if Poetry environment exists
poetry env info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Poetry environment not found
    echo Please run setup-dev.bat first to set up the development environment
    pause
    exit /b 1
)

REM Check if Poetry is available
poetry --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Poetry not found
    echo Please install Poetry or run setup-dev.bat to set up the environment
    pause
    exit /b 1
)

REM Set environment variables for development
set OMNI_UI_DISABLE_USAGE_LOGGING=false
set MCP_PORT=9901

REM Display startup information
echo Starting OmniUI MCP Server...
echo Port: %MCP_PORT%
echo Development mode: ENABLED
echo.
echo Server will be available at: http://localhost:%MCP_PORT%
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the server using Poetry
REM First check if local_config.yaml exists, otherwise use config.yaml
if exist "workflow\local_config.yaml" (
    echo Config: workflow\local_config.yaml
    poetry run omni-ui-aiq workflow\local_config.yaml
) else (
    echo Note: Using config.yaml as local_config.yaml not found
    echo Config: workflow\config.yaml
    poetry run omni-ui-aiq workflow\config.yaml
)

REM If we get here, the server has stopped
echo.
echo Server stopped.
pause
