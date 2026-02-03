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
REM Build script for Kit MCP Docker container
REM
REM This script builds both the AIQ functions and MCP server wheels,
REM then builds the Docker image.
REM
REM Prerequisites:
REM   - Python 3.11+
REM   - Poetry
REM   - Docker

setlocal enabledelayedexpansion

echo Building Kit MCP Docker container...

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%..\..\"
set "AIQ_DIR=%ROOT_DIR%aiq\kit_fns"

REM Clean previous builds
echo Cleaning previous builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

REM Build the AIQ functions package first
echo Building AIQ functions package (kit_fns)...
pushd "%AIQ_DIR%"
if %errorlevel% neq 0 (
    echo ERROR: Cannot find AIQ directory: %AIQ_DIR%
    exit /b 1
)
poetry build
if %errorlevel% neq 0 (
    echo ERROR: Failed to build AIQ functions package
    popd
    exit /b 1
)
popd

REM Copy the AIQ wheel to the current dist/ directory
echo Copying AIQ wheel to dist...
if not exist dist mkdir dist
copy "%AIQ_DIR%\dist\*.whl" dist\
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy AIQ wheel
    exit /b 1
)

REM Build the MCP package using Poetry
echo Building MCP Python package (kit_mcp)...
poetry build
if %errorlevel% neq 0 (
    echo ERROR: Failed to build MCP package
    exit /b 1
)

REM Check if both wheels were created
dir /b dist\kit_fns-*.whl >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: AIQ wheel not found in dist\
    exit /b 1
)

dir /b dist\kit_mcp-*.whl >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: MCP wheel not found in dist\
    exit /b 1
)

REM Build Docker image
echo Building Docker image...
set DOCKER_TAG=kit-mcp:latest
docker build -t %DOCKER_TAG% .

if %errorlevel% neq 0 (
    echo ERROR: Docker build failed
    exit /b 1
)

echo.
echo Docker build complete!
echo To run the container:
echo   docker run --rm -p 9902:9902 -e NVIDIA_API_KEY=%%NVIDIA_API_KEY%% %DOCKER_TAG%
echo.
echo To run with custom port:
echo   docker run --rm -e MCP_PORT=8080 -e NVIDIA_API_KEY=%%NVIDIA_API_KEY%% -p 8080:8080 %DOCKER_TAG%

endlocal
