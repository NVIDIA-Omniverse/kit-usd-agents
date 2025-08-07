## 0.2.3 - 2025-09-04
- Fixed package URL

## 0.2.2 - 2025-07-31
- Added profiling to BaseRetrieverMessage._execute method

## 0.2.1 - 2025-07-10
- Added NVIDIA license headers to all Python files

## 0.2.0 - 2025-03-17
- Switched to use Pydantic v2 and langchain 0.3.x

## 0.1.18 - 2025-03-17
- Updated to use lc_agent's Pydantic compatibility layer
- Added model_serializer for BaseRetrieverMessage
- Added name field with Literal type for RetrieverMessage and BaseRetrieverMessage
- Improved serialization support for message classes

## 0.1.17 - 2024-12-03
- Added "Request:" to the RAG human message

## 0.1.16 - 2024-12-02
- Set both "k" and "fetch_k" for the retriever

## 0.1.15 - 2024-11-29
- Don't call retriever if top-k is 0

## 0.1.14 - 2024-11-28
- Added top_k and max_tokens parameters to RetrieverMessage
- Implemented token filtering functionality
- Enhanced HumanRagModifier and SystemRagModifier with new parameters
- Improved RAG message formatting

## 0.1.13 - 2024-11-15
- Fixed error when BaseRetrieverMessage is the first node

## 0.1.12 - 2024-11-07
- BaseRagModifier supports sub-networks
