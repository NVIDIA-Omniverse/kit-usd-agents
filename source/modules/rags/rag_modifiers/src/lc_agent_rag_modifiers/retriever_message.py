## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from abc import ABC, abstractmethod
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import _convert_to_message as convert_to_message
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables.base import RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever
from lc_agent import get_retriever_registry
from lc_agent.utils.profiling_utils import Profiler
from lc_agent.utils.pydantic import BaseModel
from pydantic import model_serializer
from typing import Callable, Literal, Optional
import time


# Define constants for RAG prompt and item templates
RAG_PROMPT = """Following examples may or may not help you with this request.

{rag}

If you think any of these examples don't help, ignore it completely.
Only use relevant examples.
"""

RAG_PROMPT_FOOTER_HUMAN = "\nRequest:\n"

CODE_RAG_ITEM = """Example {idx}
{question}
Code:
```
{info}
```

"""

CODE_RAG_ITEM_HUMAN = """{question}
Code:
{info}

"""

QA_RAG_ITEM = """Info {idx}
{info}
"""


class BaseRetrieverMessage(BaseModel, RunnableLambda, ABC):
    """Base class for handling retrieving and formatting messages with examples."""

    name: Literal["base_retriever_message"] = "base_retriever_message"
    question: Optional[str] = None
    # This is "role". In BaseMessage it's called "type"
    type: str = "system"
    func: Optional[Callable] = None
    # TODO: keep the result here and don't format and don't invoke if it's already there
    result: Optional[str] = None

    def __init__(self, **kwargs):
        BaseModel.__init__(self, **kwargs)
        RunnableLambda.__init__(self, lambda x, s=self, **kwargs: s._execute(x))

    def _iter(self, *args, **kwargs):
        """Pydantic serialization method"""
        # No func
        kwargs["exclude"] = (kwargs.get("exclude", None) or set()) | {"func"}

        # Call super
        yield from super()._iter(*args, **kwargs)

    @model_serializer
    def serialize_model(self) -> dict:
        """Pydantic 2 serialization method using model_serializer"""
        # Create a base dictionary with all fields except func
        result = {}
        for field_name, field_value in self:
            if field_name != "func":
                result[field_name] = field_value

        return result

    @abstractmethod
    def _process_question(self, question):
        """Process the question and return the formatted message."""
        pass

    def _execute(self, input):
        """Execute the retriever and format the message."""
        p = Profiler(
            "retriever_execute_" + type(self).__name__,
            "retriever",
            retriever_name=getattr(self, 'retriever_name', None),
            question_type=type(input).__name__ if input else 'None'
        )
        
        result = input
        start_time = time.time()

        if self.question:
            question = self.question
        else:
            if isinstance(input, list) and input:
                extracted = input[-1]
            else:
                extracted = input

            if isinstance(extracted, str):
                question = extracted
            elif isinstance(extracted, ChatPromptValue):
                question = extracted.messages[-1].content
            elif isinstance(extracted, BaseMessage):
                question = extracted.content
            elif isinstance(extracted, dict) and "content" in extracted:
                question = extracted["content"]
            else:
                question = None

        if not question:
            # Question not found. Pass through.
            return input

        if isinstance(input, list):
            input = input[0] if input else None
        if not isinstance(input, dict) or "role" in input or "content" in input:
            input = None

        # Format the question
        if input:
            for k, v in input.items():
                question = question.replace("{" + k + "}", str(v))

        message = convert_to_message((self.type, question))
        text = self._process_question(message.content)
        if text:
            if isinstance(text, BaseMessage):
                message = text
            else:
                message.content = text

            if isinstance(result, ChatPromptValue):
                result.messages.append(message)
            elif isinstance(result, list):
                result.append(message)
            elif isinstance(result, dict):
                result = [result, message]
            else:
                # TODO: We need more types here
                result = ChatPromptValue(messages=[message])

        print(f"RetrieverMessage took {time.time() - start_time} seconds")

        return result


class RetrieverMessage(BaseRetrieverMessage):
    """Class to handle retrieving and formatting messages with examples."""

    name: Literal["retriever_message"] = "retriever_message"
    retriever_name: str
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _process_question(self, question):
        """Process the question and return the formatted message."""
        if self.top_k == 0 or self.max_tokens == 0 or not self.retriever_name:
            # No retriever or constraints. Pass through.
            return None

        retriever_results = self._invoke_retriever(question)
        if retriever_results:
            formatted_message = self._format_message(retriever_results)
            return formatted_message

    def _invoke_retriever(self, question: str):
        """Invoke the retriever with the given question and apply search constraints.

        Args:
            question: The query string to search for relevant documents.

        Returns:
            Retrieved documents if successful, None otherwise.
        """
        if not self.retriever_name:
            return None

        retriever = get_retriever_registry().get_retriever(self.retriever_name)
        if not retriever:
            return None

        search_kwargs = self._build_search_kwargs(retriever)
        if search_kwargs:
            retriever = retriever.copy(update={"search_kwargs": search_kwargs})

        return retriever.invoke(question)

    def _build_search_kwargs(self, retriever):
        """Build search parameters based on configured constraints.

        Args:
            retriever: The base retriever instance.

        Returns:
            dict: Search parameters including top_k and filter if applicable.
        """
        if not (self.top_k or self.max_tokens):
            return None

        k = self.top_k if self.top_k is not None else retriever.search_kwargs.get("k")
        filter = self._create_token_filter() if self.max_tokens else None

        # We need to set fetch_k because of the implementation of FAISS retriever:
        #
        # The file langchain_community\vectorstores\faiss.py
        # In the method similarity_search_with_score_by_vector there is the code:
        # scores, indices = self.index.search(vector, k if filter is None else fetch_k)
        #
        # It means FAISS ignores k if filter is set.
        search_kwargs = {}
        if k is not None:
            search_kwargs["k"] = k
            search_kwargs["fetch_k"] = k
        if filter is not None:
            search_kwargs["filter"] = filter

        return search_kwargs

    def _create_token_filter(self):
        """Create a filter function to limit total tokens in retrieved documents.

        Returns:
            Callable: Filter function that tracks cumulative token count.
        """
        tokens_state = [0, self.max_tokens]  # [current_total, max_allowed]

        def filter_func(metadata):
            index_tokens = metadata.get("index_text_tokens", 0)
            content_tokens = metadata.get("content_tokens", 0)
            total_tokens = index_tokens + content_tokens

            if tokens_state[0] + total_tokens > tokens_state[1]:
                return False

            tokens_state[0] += total_tokens
            return True

        return filter_func

    def _format_message(self, retriever_results):
        """Format the retriever results into a message."""
        rag_bit = ""
        code_item = CODE_RAG_ITEM_HUMAN if self.type == "human" else CODE_RAG_ITEM
        for idx, rag_result in enumerate(retriever_results):
            rag_index = idx + 1
            has_question = "question" in rag_result.metadata.keys()
            if has_question:
                rag_question = f"Question: '{rag_result.metadata['question']}'"
            elif "index_text" in rag_result.metadata:
                rag_question = f"Title: '{rag_result.metadata['index_text']}'"
                has_question = True  # Treat title as a question for formatting

            if has_question and "url" in rag_result.metadata:
                url = rag_result.metadata["url"]
                if url:
                    rag_question += f", URL: {url}"

            rag_content = rag_result.page_content

            if has_question and rag_content:
                rag_bit += (
                    code_item.replace("{idx}", str(rag_index))
                    .replace("{question}", str(rag_question))
                    .replace("{info}", str(rag_content))
                )
            else:
                rag_bit += QA_RAG_ITEM.replace("{idx}", str(rag_index)).replace("{info}", str(rag_content))

        if rag_bit:
            body = RAG_PROMPT.replace("{rag}", rag_bit)
            if self.type == "human":
                body += RAG_PROMPT_FOOTER_HUMAN
            return body
