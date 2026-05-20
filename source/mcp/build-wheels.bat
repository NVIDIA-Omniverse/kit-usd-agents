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
REM   build-wheels.bat         - Build all wheels
REM   build-wheels.bat kit     - Build only kit-mcp wheels
REM   build-wheels.bat omni    - Build only omni-ui-mcp wheels
REM   build-wheels.bat usd     - Build only usd-code-mcp wheels
REM   build-wheels.bat isaac   - Build only isaacsim-mcp wheels

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

REM Check Git LFS + probe canonical data files. Building wheels off LFS pointer
REM stubs produces a much smaller wheel that silently breaks retrieval at runtime.
REM Mirrors the bash version's check_git_lfs.
call :check_git_lfs
if %ERRORLEVEL% neq 0 exit /b 1

REM Parse argument
set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=all"

if /i "%TARGET%"=="kit" goto :build_kit
if /i "%TARGET%"=="omni" goto :build_omni
if /i "%TARGET%"=="usd" goto :build_usd
if /i "%TARGET%"=="isaac" goto :build_isaac
if /i "%TARGET%"=="all" goto :build_all

echo Usage: %~nx0 [kit^|omni^|usd^|isaac^|all]
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

:build_isaac
echo [INFO] === Building Isaac Sim MCP wheels ===
call :build_wheel "%ROOT_DIR%\source\aiq\isaacsim_fns" "isaacsim_fns"
if %ERRORLEVEL% neq 0 exit /b 1
call :build_wheel "%SCRIPT_DIR%isaacsim_mcp" "isaacsim_mcp"
if %ERRORLEVEL% neq 0 exit /b 1
REM Copy isaacsim_fns AFTER isaacsim_mcp build to avoid deletion
call :copy_wheel "%ROOT_DIR%\source\aiq\isaacsim_fns" "%SCRIPT_DIR%isaacsim_mcp\dist" "isaacsim_fns"
echo [INFO] Isaac Sim MCP wheels ready in: %SCRIPT_DIR%isaacsim_mcp\dist\
if /i "%TARGET%"=="isaac" goto :done
goto :eof

:build_all
call :build_kit
call :build_omni
call :build_usd
call :build_isaac
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

:check_git_lfs
where git >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [WARN] git not installed; skipping LFS pointer check.
    REM Explicit ``exit /b 0`` — the failed ``where git`` left errorlevel=1,
    REM and a bare ``goto :eof`` would propagate that to the caller, which
    REM does ``if %%ERRORLEVEL%% neq 0 exit /b 1`` and aborts the build.
    exit /b 0
)
where git-lfs >nul 2>&1
if %ERRORLEVEL% neq 0 (
    git lfs version >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Git LFS is required but not installed.
        echo [INFO]  Install with one of:
        echo [INFO]    Windows ^(winget^):   winget install GitHub.GitLFS
        echo [INFO]    Windows ^(choco^):    choco install git-lfs
        echo [INFO]  Then re-run this script -- LFS objects will be auto-pulled.
        exit /b 1
    )
)

set "LFS_POINTERS_FOUND=0"
set "LFS_POINTER_LIST="
call :probe_lfs "%ROOT_DIR%\source\aiq\isaacsim_fns\src\isaacsim_fns\data\6.0\extensions\extensions_database.json"
call :probe_lfs "%ROOT_DIR%\source\aiq\isaacsim_fns\src\isaacsim_fns\data\6.0\extensions\extensions_faiss\index.faiss"
call :probe_lfs "%ROOT_DIR%\source\aiq\kit_fns\src\kit_fns\data\110.0\knowledge\index.json"
call :probe_lfs "%ROOT_DIR%\source\aiq\omni_ui_fns\src\omni_ui_fns\data\faiss_index_omni_ui\index.faiss"
call :probe_lfs "%ROOT_DIR%\source\aiq\usd_code_fns\src\omni_aiq_usd_code\data\v25.11\code_rag\index.faiss"

if "!LFS_POINTERS_FOUND!"=="0" (
    echo [INFO] Git LFS check: data files resolved ^(no pointer stubs^).
    exit /b 0
)

echo [WARN] Detected LFS pointer stubs:
echo !LFS_POINTER_LIST!
echo [INFO] Attempting auto-recovery ^(git lfs install --local ^&^& git lfs pull^)...
pushd "%ROOT_DIR%"
git rev-parse --git-dir >nul 2>&1
if !ERRORLEVEL! neq 0 (
    popd
    echo [ERROR] %ROOT_DIR% is not a git working tree; cannot auto-pull.
    echo [ERROR] Re-clone with git ^(not a tarball download^), then re-run this script.
    exit /b 1
)
git lfs install --local
if !ERRORLEVEL! neq 0 (
    popd
    echo [ERROR] git lfs install --local failed.
    echo [INFO]  Run manually from the repo root: git lfs install ^&^& git lfs pull
    exit /b 1
)
git lfs pull
if !ERRORLEVEL! neq 0 (
    popd
    echo [ERROR] git lfs pull failed ^(network? LFS auth? remote not configured for LFS?^).
    exit /b 1
)
popd

REM Re-probe
set "LFS_POINTERS_FOUND=0"
set "LFS_POINTER_LIST="
call :probe_lfs "%ROOT_DIR%\source\aiq\isaacsim_fns\src\isaacsim_fns\data\6.0\extensions\extensions_database.json"
call :probe_lfs "%ROOT_DIR%\source\aiq\isaacsim_fns\src\isaacsim_fns\data\6.0\extensions\extensions_faiss\index.faiss"
call :probe_lfs "%ROOT_DIR%\source\aiq\kit_fns\src\kit_fns\data\110.0\knowledge\index.json"
call :probe_lfs "%ROOT_DIR%\source\aiq\omni_ui_fns\src\omni_ui_fns\data\faiss_index_omni_ui\index.faiss"
call :probe_lfs "%ROOT_DIR%\source\aiq\usd_code_fns\src\omni_aiq_usd_code\data\v25.11\code_rag\index.faiss"

if "!LFS_POINTERS_FOUND!"=="0" (
    echo [INFO] Git LFS auto-recovery succeeded -- data files now resolved.
    exit /b 0
)
echo [ERROR] git lfs pull ran but pointer stubs still present:
echo !LFS_POINTER_LIST!
echo [ERROR] Investigate with: git lfs ls-files --debug ^| head
exit /b 1

:probe_lfs
REM %1 = file path; if it's a pointer stub, increment LFS_POINTERS_FOUND
if not exist "%~1" goto :eof
for %%A in ("%~1") do set "FILE_SIZE=%%~zA"
if !FILE_SIZE! geq 1024 goto :eof
findstr /B /C:"version https://git-lfs.github.com/spec/v1" "%~1" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set /a LFS_POINTERS_FOUND=!LFS_POINTERS_FOUND!+1
    set "LFS_POINTER_LIST=!LFS_POINTER_LIST!  %~1!LF!"
    echo [ERROR] LFS pointer ^(not real content^) detected: %~1
)
goto :eof

:done
echo.
echo [INFO] === Build Complete ===
echo [INFO] You can now run: docker compose -f docker-compose.local.yaml up --build
exit /b 0
