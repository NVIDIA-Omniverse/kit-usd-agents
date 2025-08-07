# LC Agent Retrievers

This package provides utility modules for the LC Agent project, specifically focusing on retrievers.

## Installation

You can install the `lc_agent_retrievers` package using pip:

```
pip install lc_agent_retrievers
```

## Usage

The `lc_agent_retrievers` package includes a `register_retrievers` module that allows you to register retrievers for the LC Agent project.

Example usage:

```python
from lc_agent_retrievers.register_retrievers import register_all

# Register all retrievers
register_all()
```

## Included Retrievers

The package includes the following retrievers:

- FAISS Retriever: A retriever based on the FAISS library for efficient similarity search.

## Data

The package includes pre-built data files for the FAISS retriever:

- `faiss_index_embedqa_3346`: Pre-built FAISS index for the EmbedQA-3346 dataset.
- `faiss_index_ai-embed-qa-4_ousd_sdgqa`: Pre-built FAISS index for the QA dataset.
- `faiss_index_ai-embed-qa-4_code06262024`: Pre-built FAISS index for the USD Code dataset.
