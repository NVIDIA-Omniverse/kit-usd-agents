# Changelog

## [2.0.15] - 2025-09-04

### Changed
- Updated omni.ai.langchain.agent.usd_code dependency to 2.0.7

## [2.0.14] - 2025-09-02
- Added Nemotron model support (nvidia/llama-3.3-nemotron-super-49b-v1.5)
- Added NoThinkChatNVIDIA class for models requiring no_think mode
- Added support for /no_think directive in system messages

## [2.0.13] - 2025-08-07
- Fixed openai/gpt-oss-120b

## [2.0.12] - 2025-08-06
- Enabled developer mode
- Added Maverick chat model (meta/llama-4-maverick-17b-128e-instruct)
- Added native OpenAI GPT-4o support (openai/gpt-4o)
  - Requires OPENAI_API_KEY environment variable or openai_api_key setting
  - Model won't register without API key

## [2.0.11] - 2025-08-02
- Updated omni.ai.langchain.widget.core to 2.0.6

## [2.0.10] - 2025-08-01
- Updated omni.ai.langchain.widget.core to 2.0.5

## [2.0.9] - 2025-07-30
- Added search_path parameter to USD Search for filtering search results (e.g., SimReady content with scale alignment)

## [2.0.8] - 2025-07-17
- Additional customization of USDSearchNetworkNode

## [2.0.7] - 2025-07-10

### Changed
- Added NVIDIA copyright headers to Python files

## [2.0.6] - 2025-07-04
- Updated omni.ai.langchain.agent.usd_code to 2.0.4
- Updated omni.ai.langchain.widget.core to 2.0.3

## [2.0.5] - 2025-05-21
- Fixed /exts/omni.ai.langchain.agent.usd_code/enable_undo_stack

## [2.0.4] - 2025-05-20
- Don't fail when there is no API key

## [2.0.3] - 2025-05-15
- Updated omni.ai.langchain.agent.usd_code to 2.0.2

## [2.0.2] - 2025-04-10
- Republishing extension

## [2.0.1] - 2025-03-25
- Added package.writeTarget.kit to the config to make sure the ext is only visible to kit 107

## [2.0.0] - 2025-03-18
- Updated to support Pydantic v2
- Fixed chat model payload handling
- Updated LC Agent dependencies to 0.2.1

## [1.2.4] - 2025-02-28
- Updated lc_agent_usd to version 0.1.26

## [1.2.3] - 2025-02-27
- Added documentation about Chat USD

## [1.2.2] - 2025-02-03
- Added USD Tutor agent (enable /exts/omni.ai.chat_usd.bundle/register_usd_tutor_agent)
- Added support for registering all available chat models (/exts/omni.ai.chat_usd.bundle/register_all_chat_models)

## [1.2.1] - 2025-01-06
- Chat USD with omni.ui

## [1.2.0] - 2024-12-06
- Release

## [1.0.0] - 2024-07-29
- Initial release
