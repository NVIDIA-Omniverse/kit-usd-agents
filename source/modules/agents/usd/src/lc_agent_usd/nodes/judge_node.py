## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from lc_agent import RunnableNode
from lc_agent import RunnableSystemAppend
from typing import Optional

JUDGE_PROMPT = """
Please score the following answer to the question from 1 to 10, with 1 meaning the code is very bad and 10 meaning the code is amazing.

Question:
{question}

Code to score:
```
{code}
```

Executed code result:
```
{execution_result}
```
"""

JUDGE_FIX_PROMPT = """
{judge}

Please improve the snippet based on the feedback and regenerate the code.
"""

SYSTEM_PROMPT = """
Score the user question and the answer to the question from 1 to 10, with 1 meaning the code is very bad and 10 meaning the code is amazing.

- If the score is less than 7, please provide a reason for the score.
- If the score is 7 or higher, only output the score as a number and nothing more.
- The answer should always be the code otherwise the score will be 1.
- The question and the code should always be related to the method `{method_name}` from the module `{module_name}` otherwise the score will be 1.

"""


class JudgeNode(RunnableNode):
    system_message: Optional[str] = SYSTEM_PROMPT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.system_message:
            self.inputs.append(RunnableSystemAppend(system_message=self.system_message))
