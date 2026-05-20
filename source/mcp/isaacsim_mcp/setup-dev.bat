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
REM Windows development setup script for Isaac Sim MCP Server
REM This script sets up the local development environment

echo ========================================
echo Isaac Sim MCP Server - Development Setup
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ and try again
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python version:
python --version

REM Check if Poetry is available
poetry --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo Poetry not found. Installing Poetry...
    echo.
    curl -sSL https://install.python-poetry.org | python -
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install Poetry
        echo Please install Poetry manually from https://python-poetry.org/docs/#installation
        pause
        exit /b 1
    )
    echo.
    echo Poetry installed successfully!
    echo Please restart this script or refresh your PATH to use Poetry
    echo You may need to restart your command prompt
    pause
    exit /b 0
)

echo Poetry version:
poetry --version

REM Configure Poetry to create virtual environment in project directory
echo.
echo Configuring Poetry to use local virtual environment...
poetry config virtualenvs.in-project true

REM Install dependencies. If the shipped poetry.lock is older than the current
REM pyproject.toml (which happens any time a new dep lands without `poetry lock`
REM being re-run on the maintainer side), `poetry install` will refuse with a
REM "pyproject.toml changed significantly..." error. Auto-recover by running
REM `poetry lock` once and retrying — same behavior as setup-dev.sh
REM.
REM
REM Implementation note: cmd.exe expands %errorlevel% at PARSE time of an
REM outer "if (...)" block, so nested ``if %errorlevel% neq 0`` checks
REM inside the same block see the stale value from when the block was
REM entered. Flatten the recovery flow with ``goto`` so each errorlevel
REM check is at top level and re-evaluates correctly.
echo.
echo Installing dependencies...
poetry install
if not errorlevel 1 goto :install_done

echo.
echo poetry install failed ^(likely a stale lock file^). Refreshing lock and retrying...
poetry lock
if errorlevel 1 goto :lock_failed
poetry install
if errorlevel 1 goto :retry_failed
goto :install_done

:lock_failed
echo ERROR: poetry lock failed
pause
exit /b 1

:retry_failed
echo.
echo ERROR: Failed to install dependencies even after refreshing the lock
echo Please check your internet connection and try again
pause
exit /b 1

:install_done

REM Create local directories if they don't exist
if not exist "logs" mkdir logs

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Run 'run.bat' to start the Isaac Sim MCP server
echo 2. The server will be available at http://localhost:9904
echo.
echo For development:
echo - Use 'poetry shell' to activate the virtual environment
echo - Use 'poetry run isaacsim-mcp' to run the server manually
echo - Edit workflows/local_config.yaml to customize configuration
echo.
pause

