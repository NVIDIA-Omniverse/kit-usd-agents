## 0.2.4 - 2025-09-04
- Fixed package URL

## 0.2.3 - 2025-08-22
- Adde async in code interpreter

## 0.2.2 - 2025-07-31
- Added profiling to RAG (AsyncUSDClassAppender)

## 0.2.1 - 2025-07-10
- Added NVIDIA license headers to all Python files

## 0.2.0 - 2025-03-17
- Switched to use Pydantic v2 and langchain 0.3.x

## 0.1.27 - 2025-03-17
- Updated to use lc_agent's Pydantic compatibility layer
- Added model_serializer for AsyncUSDClassAppender and CodeAtlasErrorMessage
- Added name field with Literal type for serialization support
- Enhanced MFRetrieverMessage with proper name field

## 0.1.26 - 2025-02-28
- Create system message runtime, not on init

## 0.1.25 - 2025-02-27
- Added parsers for usdcode metafunctions

## 0.1.24 - 2025-01-06
- More USD Code Interactive examples

## 0.1.23 - 2024-12-03
- Better USD Code Interactive prompts

## 0.1.22 - 2024-11-28
- Ability to use custom system in USDCodeInteractiveNetworkNode

## 0.1.21 - 2024-11-28
- Added RAG parameter support (top_k, max_tokens) to MFRagModifier
- Enhanced USDCodeGenRagModifier with new RAG controls
- Updated USDCodeInteractiveNetworkNode to handle RAG metadata

## 0.1.20 - 2024-11-15
- Renamed mf to usdcode

## 0.1.19 - 2024-11-15
- Improved system of USD Code Interactive

## 0.1.18 - 2024-11-11
- Missing kwargs in AsyncUSDClassAppender

## 0.1.17 - 2024-11-08
- Don't import usd-code when importing lc_agent_usd

## 0.1.16 - 2024-11-07
- Added "USD Code Interactive" node

## 0.1.15 - 2024-11-05
- Ability to set custom message to NetworkLenghtModifier

## 0.1.14 - 2024-08-15
- fixed USDCodeGenInterpreterModifier to use "/" in paths instead of "\"

## 0.1.13 - 2024-07-24
- requirements

## 0.1.12 - 2024-07-23
- ability to disable code interpreter
- ability to secure code interpreter with a list of refused modules and methods

## 0.1.11 - 2024-07-17
- rename ChatUSD to USDCode

## 0.1.10 - 2024-07-15
- NetworkLenghtModifier makes nodes with error message

## 0.1.9 - 2024-07-05
- Ability to set custom error/success messages to `CodeInterpreterModifier`

## 0.1.8 - 2024-06-27
- Updated `CodeExtractorModifier` to wrap extracted code snippets in Python code block delimiters.