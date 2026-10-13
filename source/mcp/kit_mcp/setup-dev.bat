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
setlocal
REM Windows development setup script for Kit MCP Server
REM This script sets up the local development environment

echo ========================================
echo Kit MCP Server - Development Setup
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

REM These MCP servers resolve every dependency from public PyPI, so Poetry never
REM needs stored credentials. Poetry still queries the Windows Credential Manager
REM for each package URL, and in sessions without a logon-session credential
REM store that raises an uncaught "(1312, 'CredRead', ...)" error which aborts
REM both `poetry lock` and `poetry install`. Disable the keyring lookup for this
REM script unless the caller has already chosen a setting. The variable is scoped
REM to this script by the setlocal at the top.
if not defined POETRY_KEYRING_ENABLED set "POETRY_KEYRING_ENABLED=false"

REM If the shipped poetry.lock no longer matches pyproject.toml, `poetry install`
REM refuses with a "pyproject.toml changed significantly..." error. Detect that up
REM front with `poetry check --lock` and refresh the lock only in that case, so a
REM later install failure is reported as what it really is.
REM
REM Implementation note: cmd.exe expands environment variables at PARSE time of
REM an outer "if (...)" block, so nested errorlevel checks inside the same block
REM see the stale value from when the block was entered. Keep the flow flat with
REM ``goto`` so each errorlevel check is at top level. Compare the errorlevel
REM with 0 (equ/neq) rather than using ``if errorlevel 1``: the latter treats negative exit
REM codes (a crash, or Ctrl-C answered with N) as success.
echo.
echo Checking that poetry.lock matches pyproject.toml...
poetry check --lock >nul 2>&1
if %errorlevel% equ 0 goto :lock_ok
echo poetry.lock is out of date. Refreshing it with 'poetry lock'...
poetry lock
if %errorlevel% neq 0 goto :lock_failed
:lock_ok

echo.
echo Installing dependencies...
poetry install
if %errorlevel% equ 0 goto :install_done
REM Poetry treats only "true" or "1" (any case) as enabled. With the keyring
REM lookup disabled the failure cannot be a keyring error.
if /i "%POETRY_KEYRING_ENABLED%"=="true" goto :install_failed_keyring
if "%POETRY_KEYRING_ENABLED%"=="1" goto :install_failed_keyring
goto :install_failed

:install_failed_keyring

echo.
echo ERROR: poetry install failed with Poetry's keyring lookup enabled
echo ^(POETRY_KEYRING_ENABLED=%POETRY_KEYRING_ENABLED%^).
echo If the output above mentions CredRead, 1312 or
echo "A specified logon session does not exist", the Windows credential store
echo is unavailable in this session.
echo These MCP servers only use public PyPI and need no credentials. Disable
echo Poetry's keyring lookup and re-run this script:
echo     set POETRY_KEYRING_ENABLED=false
pause
exit /b 1

:lock_failed
echo.
echo ERROR: 'poetry lock' failed. See the output above for the cause.
pause
exit /b 1

:install_failed
echo.
echo ERROR: poetry install failed. See the output above for the cause.
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
echo 1. Run 'run.bat' to start the Kit MCP server
echo 2. The server will be available at http://localhost:9902
echo.
echo For development:
echo - Use 'poetry shell' to activate the virtual environment
echo - Use 'poetry run kit-mcp' to run the server manually
echo - Edit workflows\local_config.yaml to customize configuration
echo.
pause
