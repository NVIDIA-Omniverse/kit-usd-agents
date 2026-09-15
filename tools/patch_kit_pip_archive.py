#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Align a few pip packages in Kit's omni.kit.pip_archive with the langchain stack.

Kit's ``omni.kit.pip_archive`` is loaded before any project extension, so the
package versions it ships win on ``sys.path`` for the whole process. The ChatUSD
agent stack pulls ``langchain-core`` 1.4.x / ``langgraph`` 1.2.x (required by
lc_agent's CVE-2026-44843 floor), which need newer versions of two packages than
pip_archive bundles:

* ``typing_extensions`` >= 4.13 -- ``langchain_protocol`` uses PEP-728
  ``extra_items`` TypedDicts; pip_archive ships 4.12.2 ->
  ``_TypedDictMeta.__new__() got an unexpected keyword argument 'extra_items'``.
* ``websockets`` >= 14 -- ``langgraph_sdk`` does ``from websockets.client import
  backoff``; pip_archive ships 12.0 -> ``cannot import name 'backoff'``.

Fix (minimal blast radius): copy just those packages from the langchain core
prebundle (which already resolved the correct, mutually-compatible versions) into
pip_archive, but only when pip_archive's copy is older. No ``pip install
--target`` (that litters the prebundle with ``bin/`` + ``__pycache__`` and
disturbs namespace scanning) and no network. Idempotent; a no-op once Kit's
pip_archive ships versions new enough.

Runs from both repo-build layout (``_build/<platform>/<config>/...``) and the
unpacked test-package layout (``<test_root>/...``), so it works as a repo_build
post_build step AND a repo_test archive_post_unpack step (after pull_kit_sdk
re-fetches a pristine SDK at test time).
"""
import glob
import os
import re
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# package -> minimum (major, minor) required by the langchain/langgraph stack
PACKAGES = {
    "typing_extensions": (4, 13),
    "websockets": (14, 0),
}

# pip_archive (target) can live in extscache or under kit/exts, in either the
# repo _build tree or an unpacked test package rooted at the repo.
TARGET_GLOBS = [
    "_build/*/*/extscache/omni.kit.pip_archive-*/pip_prebundle",
    "_build/*/*/kit/exts/omni.kit.pip_archive*/pip_prebundle",
    "extscache/omni.kit.pip_archive-*/pip_prebundle",
    "kit/exts/omni.kit.pip_archive*/pip_prebundle",
]
# pip_core_prebundle (source of the known-good versions) ships with the
# omni.ai.langchain.core extension.
SOURCE_GLOBS = [
    "_build/*/*/exts/omni.ai.langchain.core/pip_core_prebundle",
    "exts/omni.ai.langchain.core/pip_core_prebundle",
]


def _resolve(patterns):
    found = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(REPO_ROOT, pat)))
    return sorted(set(found))


def _dist_version(prebundle, pkg):
    for info in glob.glob(os.path.join(prebundle, f"{pkg}-*.dist-info")):
        m = re.search(rf"{re.escape(pkg)}-([0-9]+)\.([0-9]+)", os.path.basename(info))
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return None


def _copy_package(pkg, src_prebundle, dst_prebundle):
    """Copy a package's module/dir + dist-info from src to dst (overwriting)."""
    copied = False
    for name in (f"{pkg}.py", pkg):  # module file or package dir
        src = os.path.join(src_prebundle, name)
        if os.path.exists(src):
            dst = os.path.join(dst_prebundle, name)
            if os.path.isdir(src):
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            copied = True
    for old in glob.glob(os.path.join(dst_prebundle, f"{pkg}-*.dist-info")):
        shutil.rmtree(old, ignore_errors=True)
    for src_info in glob.glob(os.path.join(src_prebundle, f"{pkg}-*.dist-info")):
        shutil.copytree(src_info, os.path.join(dst_prebundle, os.path.basename(src_info)), dirs_exist_ok=True)
    return copied


def _pick_source(pkg, min_ver):
    for src in _resolve(SOURCE_GLOBS):
        v = _dist_version(src, pkg)
        if v is not None and v >= min_ver:
            return src
    return None


def main():
    pip_archives = _resolve(TARGET_GLOBS)
    if not pip_archives:
        print("[patch_kit_pip_archive] no pip_archive prebundle found (skipping)")
        return 0

    patched = 0
    for dst in pip_archives:
        for pkg, min_ver in PACKAGES.items():
            cur = _dist_version(dst, pkg)
            if cur is not None and cur >= min_ver:
                continue
            src = _pick_source(pkg, min_ver)
            if src is None:
                print(f"[patch_kit_pip_archive] WARNING: no source with {pkg}>={min_ver}; skipping")
                continue
            if _copy_package(pkg, src, dst):
                print(f"[patch_kit_pip_archive] aligned {pkg} {cur} -> {_dist_version(src, pkg)} in {dst}")
                patched += 1

    print(f"[patch_kit_pip_archive] done ({patched} package(s) aligned across {len(pip_archives)} prebundle(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
