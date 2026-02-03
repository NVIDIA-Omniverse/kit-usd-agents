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
REM Build wheels for MCP servers (Windows)
REM
REM This script builds the required wheel files for Docker image construction.
REM Run this before 'docker compose -f docker-compose.local.yaml up --build'
REM
REM Prerequisites:
REM   - Python 3.11+
REM   - Poetry (https://python-poetry.org/docs/#installation)
REM
REM Usage:
REM   build-wheels.bat        - Build all wheels
REM   build-wheels.bat kit    - Build only kit-mcp wheels
REM   build-wheels.bat omni   - Build only omni-ui-mcp wheels
REM   build-wheels.bat usd    - Build only usd-code-mcp wheels

setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%\..\.."
set "ROOT_DIR=%CD%"
popd

REM Check prerequisites
echo [INFO] Checking prerequisites...

where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is required but not installed.
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] %PYTHON_VERSION%

where poetry >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Poetry is required but not installed.
    echo [INFO] Install with: pip install poetry
    exit /b 1
)

for /f "tokens=*" %%i in ('poetry --version 2^>^&1') do set POETRY_VERSION=%%i
echo [INFO] %POETRY_VERSION%

REM Parse argument
set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=all"

if /i "%TARGET%"=="kit" goto :build_kit
if /i "%TARGET%"=="omni" goto :build_omni
if /i "%TARGET%"=="usd" goto :build_usd
if /i "%TARGET%"=="all" goto :build_all

echo Usage: %~nx0 [kit^|omni^|usd^|all]
exit /b 1

:build_kit
echo [INFO] === Building Kit MCP wheels ===
call :build_wheel "%ROOT_DIR%\source\aiq\kit_fns" "kit_fns"
if %ERRORLEVEL% neq 0 exit /b 1
call :build_wheel "%SCRIPT_DIR%kit_mcp" "kit_mcp"
if %ERRORLEVEL% neq 0 exit /b 1
REM Copy kit_fns AFTER kit_mcp build to avoid deletion
call :copy_wheel "%ROOT_DIR%\source\aiq\kit_fns" "%SCRIPT_DIR%kit_mcp\dist" "kit_fns"
echo [INFO] Kit MCP wheels ready in: %SCRIPT_DIR%kit_mcp\dist\
if /i "%TARGET%"=="kit" goto :done
goto :eof

:build_omni
echo [INFO] === Building Omni UI MCP wheels ===
call :build_wheel "%ROOT_DIR%\source\aiq\omni_ui_fns" "omni_ui_fns"
if %ERRORLEVEL% neq 0 exit /b 1
call :build_wheel "%SCRIPT_DIR%omni_ui_mcp" "omni_ui_mcp"
if %ERRORLEVEL% neq 0 exit /b 1
REM Copy omni_ui_fns AFTER omni_ui_mcp build to avoid deletion
call :copy_wheel "%ROOT_DIR%\source\aiq\omni_ui_fns" "%SCRIPT_DIR%omni_ui_mcp\dist" "omni_ui_fns"
echo [INFO] Omni UI MCP wheels ready in: %SCRIPT_DIR%omni_ui_mcp\dist\
if /i "%TARGET%"=="omni" goto :done
goto :eof

:build_usd
echo [INFO] === Building USD Code MCP wheels ===
call :build_wheel "%ROOT_DIR%\source\aiq\usd_code_fns" "usd_code_fns"
if %ERRORLEVEL% neq 0 exit /b 1
call :build_wheel "%SCRIPT_DIR%usd_code_mcp" "usd_code_mcp"
if %ERRORLEVEL% neq 0 exit /b 1
REM Copy usd_code_fns AFTER usd_code_mcp build to avoid deletion
call :copy_wheel "%ROOT_DIR%\source\aiq\usd_code_fns" "%SCRIPT_DIR%usd_code_mcp\dist" "usd_code_fns"
echo [INFO] USD Code MCP wheels ready in: %SCRIPT_DIR%usd_code_mcp\dist\
if /i "%TARGET%"=="usd" goto :done
goto :eof

:build_all
call :build_kit
call :build_omni
call :build_usd
goto :done

:build_wheel
REM %1 = package directory, %2 = package name
echo [INFO] Building %~2 wheel...
pushd "%~1"
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
poetry build
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to build %~2
    popd
    exit /b 1
)
echo [INFO] %~2 wheel built successfully
dir /b dist\*.whl
popd
goto :eof

:copy_wheel
REM %1 = source dir, %2 = dest dir, %3 = package name
echo [INFO] Copying %~3 wheel to %~2...
if not exist "%~2" mkdir "%~2"
copy /y "%~1\dist\*.whl" "%~2\" >nul
goto :eof

:done
echo.
echo [INFO] === Build Complete ===
echo [INFO] You can now run: docker compose -f docker-compose.local.yaml up --build
exit /b 0
