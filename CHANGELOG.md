# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.5.3] - 2026-09-03

### Security

- Constrained documentation and template paths to their repository-owned
  roots, rejecting traversal and symlink escapes before file or URL access.
- Validated GitLab project and branch inputs before invoking Git, and removed
  control characters from values written to logs.
- Bounded both start and end times in recording schedules and moved all
  recording defaults from predictable shared temporary paths to the current
  user's cache directory.
- Added lockfiles for every AIQ library and data-collection project so local
  development and CI resolve the same reviewed dependency graphs.
- Raised the AIQ dependency floors to patched Cryptography and LiteLLM
  releases, and the development toolchain to patched Pytest and Black releases.

## [1.5.2] - 2026-09-02

### Changed

- Migrated every embedding and reranking path from the retired
  `nvidia/nv-embedqa-e5-v5` and
  `nvidia/llama-nemotron-rerank-1b-v2` models to
  `nvidia/nemotron-3-embed-1b` and
  `nvidia/llama-nemotron-rerank-vl-1b-v2`.
- Re-embedded all shipped FAISS bundles at 2048 dimensions. Index manifests
  now record the model, vector dimension, and document count;
  loaders reject stale or mismatched bundles instead of returning invalid
  retrieval results.
- Updated the retrieval client stack to `langchain-nvidia-ai-endpoints>=1.4.3`
  and `langchain-core>=1.4.7`, and aligned all MCP and MaaS runtimes on the
  compatible NVIDIA Agent Toolkit `1.8.0a20260514` package set.
- Updated MaaS images to `maas-sdk==2.4.3`, regenerated all Poetry and uv
  lockfiles, refreshed NIM images, and patch-bumped affected packages and
  extensions.
- Updated headless ChatUSD to use the available `openai/gpt-oss-120b` model by
  default and disabled its unsupported Fabric scene delegate.
- Restored the public ChatUSD app, extension graph, and Kit dependency
  manifests to `sync-to-staging`; local caches and build artifacts are removed
  before a staging commit is created.

### Fixed

- Limited the NAT prebundle's `uvloop` dependency to Linux so the shared
  dependency manifest also builds successfully on Windows.

### Security

- Removed the obsolete `ragas` and Pillow dependency paths from MCP runtime
  environments. Container builds now verify dependency consistency and fail
  if either package is present.
- Upgraded Lupa, PyArrow, setuptools, soupsieve, and ujson to eliminate
  CVE-2026-34444, CVE-2026-25087, CVE-2026-59890, CVE-2026-49476,
  CVE-2026-49477, CVE-2026-32874, CVE-2026-32875, CVE-2026-44660, and
  CVE-2026-54911 from the assembled Kit runtime.
- Raised every standard MCP lock and image to MCP 1.28.1 or newer and to
  patched aiohttp, Click, cryptography, JOSE RFC, and pip releases for
  CVE-2026-59881, CVE-2026-69243, CVE-2026-69244, CVE-2026-7246,
  CVE-2026-69247, CVE-2026-48990, CVE-2026-49852, and CVE-2026-13346.
- Added PyJWT 2.13.0 explicitly to the no-deps NAT prebundle, preserving the
  public ChatUSD authentication runtime while avoiding the PyJWT 2.12.x
  CVE-2026-48522 through CVE-2026-48526 findings.
- FAISS generation and loading use native FAISS plus JSON metadata only, with
  strict row, ID, model, and dimension validation.

## [1.5.1] - 2026-06-05

### Fixed

- **MaaS MCP containers crashed at startup with
  `ImportError: cannot import name 'eval_type_backport' from
  'pydantic._internal._typing_extra'`.** All 4 MaaS variants
  (`kit_mcp_maas`, `omni_ui_mcp_maas`, `usd_code_mcp_maas`,
  `isaacsim_mcp_maas`) failed to load
  `maas_sdk.app.start_nat_mcp_server`. The crash chain:

  1. `maas-sdk==2.4.0.8` hard-pins `mcp[cli]==1.16.0` as a
     dependency.
  2. `mcp 1.16.0`'s `mcp/server/fastmcp/utilities/func_metadata.py`
     imports
     `from pydantic._internal._typing_extra import eval_type_backport`.
  3. `nvidia-nat 1.4.0a20251118` forces `--prerelease=allow` on the
     `uv pip install` step, so the resolver was free to select
     `pydantic 2.14.0a1` — a pre-release that removed
     `eval_type_backport` from `_typing_extra` as part of a typing
     internals cleanup.
  4. At container startup, the very first `from maas_sdk.app import
     start_nat_mcp_server` raised `ImportError`. (No "could not find
     mcp" — the package was installed; the symbol on the pydantic
     side just wasn't there.)

  The NGC images do not hit this because they install `mcp 1.27.2`
  (via the newer nvidia-nat-mcp dependency chain), and mcp 1.27+
  uses `typing_inspection` instead of pydantic-internal symbols.

  Fix: add `"mcp>=1.27,<2.0"` to the security-upgrade `uv pip
  install --system --upgrade` block in all 4 MaaS Dockerfiles, so
  that mcp is force-upgraded past the maas-sdk pin. The pip
  resolver emits a warning ("maas-sdk 2.4.0.8 requires
  mcp[cli]==1.16.0, but you have mcp 1.27.2") — benign:
  `maas_sdk.patches` and the rest of the maas-sdk import chain
  load successfully against mcp 1.27 (verified locally
  2026-06-05).

### Changed

- MaaS package versions patch-bumped to force fresh image builds:
  `kit_mcp_maas` 0.2.0 → 0.2.1, `omni_ui_mcp_maas` 0.2.0 → 0.2.1,
  `usd_code_mcp_maas` 0.2.0 → 0.2.1, `isaacsim_mcp_maas`
  2.1.0 → 2.1.1. The 4 NGC images are unaffected; only the MaaS
  Dockerfiles changed.

## [1.5.0] - 2026-06-03

### Fixed

- **Hybrid retrieval was silently broken on all 4 production MCPs.**
  Helm charts set `KIT_RERANKER_BACKEND=nvidia_api` without setting any
  `OVAI_*` companion env vars, so the legacy detector in
  `ovgenai_retrieval.shim._rerank_enabled()` flipped rerank ON, the
  HybridRetriever was constructed with `rerank_backend="nvidia"` and
  a default model name that didn't start with `nvidia/`, and the
  fallback inside `HybridRetriever.__init__` constructed
  `NvidiaReranker()` with the canonical Nemotron model. At query time
  the hybrid pipeline behaved inconsistently across services — three
  of four MCPs (`kit`, `omni_ui`, `usd_code`) still returned content
  through their keyword fallbacks against atlas data, but
  `isaacsim_fns.services.code_search_service._load_fallback_data()`
  looks for `extracted_methods_regular/` while the wheel ships
  `extracted_methods/` (no `_regular` suffix), so `search_isaac_sim_
  code_examples` had no working fallback and returned the
  `"No code examples found"` sentinel for every query. Fix: add the
  six `OVAI_*` env vars (`OVAI_RETRIEVAL_MODE=hybrid`,
  `OVAI_FUSION=rrf`, `OVAI_RERANK=false`, `OVAI_RERANK_BACKEND=nvidia`,
  `OVAI_RERANK_MODEL=""`, `OVAI_QE_LOG=1`) to all four helm charts'
  `values.yaml` and `templates/deployment.yaml`, mirroring the
  per-MCP `docker-compose.ngc.yaml` env block that works locally and
  in the smoke test.

- **Same OVAI_\* gap was present in every other docker-compose file**
  that a user might run — fixed by extending the same env block to:
  - `source/mcp/docker-compose.local.yaml` (external users with their
    own local NIM embedder + reranker — synced to public mirror)
  - `source/mcp/docker-compose.internal.yaml` (NVIDIA-internal local
    NIM deployments)
  - `source/mcp/kit_mcp_maas/docker-compose.yaml`,
    `omni_ui_mcp_maas/docker-compose.yaml`,
    `usd_code_mcp_maas/docker-compose.yaml`,
    `isaacsim_mcp_maas/docker-compose.yaml` (MaaS OAuth-protected
    deployments — internal only, not synced to public).

  Without this fix an external user running
  `docker compose -f docker-compose.local.yaml up` would hit the
  same broken hybrid pipeline behavior that prod had — exactly the
  symptom we fixed for isaac-sim. The added vars use the same
  `${OVAI_RERANK:-false}` default as the ngc compose so an operator
  with a working local NIM reranker can override `OVAI_RERANK=true`
  via their `.env` without editing the compose file.

### Changed

- **All 8 packages bumped to a minor release** to align with the
  external 1.5.0 release naming (container scanner program-version row is read
  from top-level VERSION.md after a sync-to-staging run; jumping
  from internal 1.4.x to external 1.5.0 keeps the staging-mirror
  version coherent without burning multiple patch rows). The minor
  bumps also force fresh docker-build runs across all 4 NGC + 4 MaaS
  images so the helm chart env-var change actually rolls new pods on
  master deploy.
  - `kit_fns` 0.7.1 → 0.8.0, `kit_mcp` 1.1.1 → 1.2.0, `kit_mcp_maas`
    0.1.1 → 0.2.0
  - `omni_ui_fns` 0.7.1 → 0.8.0, `omni_ui_mcp` 1.1.1 → 1.2.0,
    `omni_ui_mcp_maas` 0.1.1 → 0.2.0
  - `usd_code_fns` 0.4.1 → 0.5.0, `usd_code_mcp` 1.1.1 → 1.2.0,
    `usd_code_mcp_maas` 0.1.1 → 0.2.0
  - `isaacsim_fns` 2.1.2 → 2.2.0, `isaacsim_mcp` 2.1.2 → 2.2.0,
    `isaacsim_mcp_maas` 2.0.2 → 2.1.0

## [1.4.2] - 2026-06-02

### Fixed

- **Force redeploy of `isaac-sim-mcp` production pod.** The 1.4.1 master
  deploy left the existing isaac-sim-mcp pod running its previous image
  (Helm computed an identical Deployment spec because `image.tag` was
  unchanged across the 1.4.0 → 1.4.1 cycle for this service's
  registry), so the pod never picked up the v1.4.1 wheel. Bumping
  `isaacsim_fns` 2.1.1 → 2.1.2, `isaacsim_mcp` 2.1.1 → 2.1.2, and
  `isaacsim_mcp_maas` 2.0.1 → 2.0.2 changes the docker-build inputs,
  produces a fresh image at the new `CI_COMMIT_SHORT_SHA`, and forces
  Helm to issue a real rollout. Symptom that prompted this:
  `search_isaac_sim_code_examples` returned the `"No code examples
  found"` sentinel for every query (including the tool's own
  documented usage examples) — a fresh local build from the same
  source returned 10K–28K chars of rich content per query, so the
  problem was the deployed image's contents, not the code.

### Changed

- isaacsim_fns 2.1.1 → 2.1.2, isaacsim_mcp 2.1.1 → 2.1.2,
  isaacsim_mcp_maas 2.0.1 → 2.0.2. Other packages unchanged: the kit,
  omni-ui, and usd-code MCPs are healthy in production on the 1.4.1
  images and don't need a forced rollout.

## [1.4.1] - 2026-05-29

### Fixed

- **Hybrid retrieval was silently broken in every published MCP image.**
  The CI's `.build-ovgenai-retrieval-wheel` step cloned the gitlab
  pull-mirror of github.com/NVIDIA-dev/ovgenai-retrieval and built the
  wheel from it. The upstream repo does not contain `shim.py` — that
  file (and its `maybe_load_hybrid` export) is a kit-usd-agents-specific
  extension allowlisted by `_vendor_drift_check.VENDORED_ONLY_ALLOWED`.
  So the wheel shipped to all 8 production images (4 NGC + 4 MaaS)
  was missing `maybe_load_hybrid`. omni-ui-mcp and usd-code-mcp
  surfaced `cannot import name 'maybe_load_hybrid' from
  'ovgenai_retrieval'` at request time; kit-dev-mcp and
  isaac-sim-mcp silently degraded to dense-only retrieval because
  their service files wrap the import in `try/except`. The fix
  changes the CI step to build from the in-tree
  `source/aiq/ovgenai_retrieval/` vendored copy, which does contain
  `shim.py` — same path that `source/mcp/build-wheels.sh` already
  uses for local dev.

### Changed

- Docker-build auto-rebuild rules for all four NGC and four MaaS
  images now include `source/aiq/ovgenai_retrieval/**/*` in their
  `changes` patterns, so changes to the shared library trigger fresh
  image builds.
- All 5 deploy jobs (`deploy-omni-ui-mcp`, `deploy-kit-dev-mcp`,
  `deploy-kit-dev-mcp-109`, `deploy-usd-code-mcp`,
  `deploy-isaac-sim-mcp`) now hardcode
  `kitEmbedderBackend="nvidia_api"` and
  `kitRerankerBackend="nvidia_api"` for consistency with the policy
  that all hosted deploys route through build.nvidia.com NIM
  endpoints. The `${KIT_EMBEDDER_BACKEND:-...}` env-var fallback was
  removed so a misconfigured pipeline can no longer silently route
  to a local NIM.
- `deploy/isaac_sim_mcp/values.yaml` +
  `templates/deployment.yaml` now expose the same four embedder/
  reranker value knobs as the other three charts (consistency only;
  `isaacsim_fns` tools currently exercise the embedder only).

- Per-package `__init__.py` version fallback strings updated to
  match the bumped versions (cosmetic — runtime queries
  `importlib.metadata.version()` first). Two distribution-name
  mismatches fixed at the same time: `omni_ui_mcp/__init__.py`
  queried `omni-ui-mcp` but the wheel's distribution name is
  `omni-ui-aiq`; `usd_code_fns/__init__.py` queried `usd-code-fns`
  but the distribution is `usd-code-aiq`. Both would silently fall
  through to the stale hardcoded fallback in installed-but-out-of-
  source environments.

### Removed

- `deploy-kit-dev-mcp-nv-rerank` and `deploy-usd-code-mcp-nv-rerank`
  manual jobs. After the lockdown above, both did exactly the same
  thing as their default counterparts.

## [1.4.0] - 2026-05-19

### Changed (April 2026 — MCP retrieval and tool description refresh)

Pass across all four MCP servers (Kit, OmniUI, USD Code, Isaac Sim) focused
on retrieval quality, agent tool selection, and security hardening.

**Retrieval engine**

- Default search mode flipped from semantic-only to *hybrid*: BM25 lexical
  + dense semantic fused via Reciprocal Rank Fusion (RRF), with optional
  cross-encoder reranking. Configured via `OVAI_RETRIEVAL_MODE=hybrid`
  (default) / `OVAI_FUSION=rrf` / `OVAI_RERANK=true`.
- Query expansion runs on every search: 215+ term curated glossary +
  auto-generated USD v25.02/v25.11 glossary expand abbreviations
  (`SSS`, `PBR`, `LIVRPS`, `Gf`, `Sdf`, ...) before retrieval. Set
  `OVAI_QE_LOG=1` to log expansion firings at INFO level.
- BM25 sidecar migrated from `bm25.pkl` to `bm25.json`. Closes the
  pickle-safety hardening for the BM25 path; no `pickle` code paths remain.
- Shared `ovgenai-retrieval` library introduced under
  `source/aiq/ovgenai_retrieval/` and built into a wheel by
  `source/mcp/build-wheels.sh`. Each MCP Docker image installs it
  alongside its own fns/mcp wheels.

**Tool descriptions**

- All 34 tools across the 4 MCPs got refreshed descriptions with
  PRIMARY / WHEN-TO-USE / ARGUMENTS / RETURNS / USAGE EXAMPLES /
  WHEN-TO-USE-A-DIFFERENT-TOOL-INSTEAD blocks. Resolves the long-running
  routing failure where agents wasted ~10 calls retrying
  `search_kit_settings` before reaching `search_kit_knowledge`.

**Security and result guardrails**

- Every search tool sanitizes input (HTML/control chars stripped, entities
  escaped, whitespace normalized) before it reaches the retriever.
- Empty result sets return a deterministic
  `"No relevant documentation found for this query."` sentinel string
  rather than an empty list, so agents stop retrying with reworded
  queries indefinitely.
- `search_*_settings` caps output at 12,000 characters with a
  `prefix_filter` / `type_filter` narrowing hint when truncated.
- Log injection, error-leak, and settings-cap hardening (R4 / R5 / R6
  audit findings) closed.

**Protocol**

- MCP protocol version bumped from `2024-11-05` to `2025-11-25`,
  validated end-to-end against NAT 1.4+.

**Tooling**

- Dockerfiles normalized to a single shape across all 4 MCPs (R3 audit):
  one wheel-install step, one external-dep step, identical verification.
- Local poetry-only dev gracefully falls back when `ovgenai-retrieval`
  isn't installed (per-package `_retrieval_compat` shim) — degraded
  retrieval but server starts cleanly. Docker path is unchanged.

## [1.3.0] - 2026-05-19

First public release covering the four MCP servers and their supporting packages under a single distribution version. Individual components keep their own SemVer; the table below records which component version is included in this release.

| Component | Version |
| --- | --- |
| `source/mcp/isaacsim_mcp`, `source/mcp/isaacsim_mcp_maas` | 2.0.0 |
| `source/mcp/kit_mcp`, `source/mcp/omni_ui_mcp`, `source/mcp/usd_code_mcp` | 1.0.0 |
| `source/mcp/kit_mcp_maas`, `source/mcp/omni_ui_mcp_maas`, `source/mcp/usd_code_mcp_maas` | 0.1.0 |
| `source/aiq/isaacsim_fns` | 2.0.1 |
| `source/aiq/kit_fns`, `source/aiq/omni_ui_fns` | 0.6.0 |
| `source/aiq/usd_code_fns` | 0.3.0 |

### Added

#### MCP Servers (Model Context Protocol)

Four new MCP servers provide AI-powered assistance for Omniverse development:

- **USD Code MCP** (`source/mcp/usd_code_mcp`)
  - USD API documentation and code examples
  - Semantic search across USD knowledge base
  - Class, module, and method detail lookups
  - Port: 9903

- **Kit MCP** (`source/mcp/kit_mcp`)
  - Omniverse Kit extension documentation
  - Extension dependency analysis
  - Kit API reference and code examples
  - Settings and configuration search
  - Port: 9902

- **OmniUI MCP** (`source/mcp/omni_ui_mcp`)
  - omni.ui widget documentation and styling
  - UI code examples and window patterns
  - Class and method reference
  - Port: 9901

- **Isaac Sim MCP** (`source/mcp/isaacsim_mcp`)
  - Isaac Sim extension documentation and code examples
  - Settings discovery across Isaac Sim configuration
  - Robotics-focused developer instructions
  - Port: 9904

#### Deployment Options

- **NVIDIA API**: Cloud-based embeddings and reranking (no GPU required)
- **Local NIMs**: On-premise deployment with NVIDIA NIM containers
- **Docker Compose**: Ready-to-use configurations for both options

#### Documentation

- `QUICKSTART.md`: Pure Python setup guide for Windows, macOS, and Linux
- `LOCAL_DEPLOYMENT.md`: Comprehensive deployment guide with Docker

---

## Previous Release

The initial public release contained **Chat USD**, a multi-agent AI assistant for USD development within Omniverse Kit:
- USD code generation and execution
- USD asset search
- Scene information retrieval

[Unreleased]: https://github.com/NVIDIA-Omniverse/kit-usd-agents/compare/v1.5.3...HEAD
[1.5.3]: https://github.com/NVIDIA-Omniverse/kit-usd-agents/compare/v1.5.2...v1.5.3
[1.5.2]: https://github.com/NVIDIA-Omniverse/kit-usd-agents/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/NVIDIA-Omniverse/kit-usd-agents/releases/tag/v1.5.1
[1.5.0]: https://github.com/NVIDIA-Omniverse/kit-usd-agents/releases/tag/v1.5.0
[1.4.2]: https://github.com/NVIDIA-Omniverse/kit-usd-agents/releases/tag/v1.4.2
[1.4.1]: https://github.com/NVIDIA-Omniverse/kit-usd-agents/releases/tag/v1.4.1
[1.4.0]: https://github.com/NVIDIA-Omniverse/kit-usd-agents/releases/tag/v1.4.0
[1.3.0]: https://github.com/NVIDIA-Omniverse/kit-usd-agents/releases/tag/v1.3.0
