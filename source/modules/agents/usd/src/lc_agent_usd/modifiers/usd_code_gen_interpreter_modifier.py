## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from .code_interpreter_modifier import CodeInterpreterModifier
from pathlib import Path
import random
import re
import string
import tempfile

DATA_PATH = Path(__file__).parent.joinpath("../data")
TEMP_PATH = Path(tempfile.gettempdir())


def replace_stage_file(code, stage_open, file_name):
    stage_open_re = stage_open.replace(".", r"\.")

    # Replace the file name in Usd.Stage.Open() with the given file name
    file_name = file_name.replace("\\", "/")
    fixed_code = re.sub(stage_open_re + r"\(.*?\)", stage_open + f'("{file_name}")', code)

    return fixed_code


def replace_sublayerpaths_file(code, sub_layer_paths):
    stage_open_re = sub_layer_paths.replace(".", r"\.")

    # Replace the file name in Usd.Stage.Open() with the given file name
    fixed_code = re.sub(stage_open_re + r"\(.*?\)", sub_layer_paths + "(_get_random_file_name())", code)

    prefix_code = f"""import random
import string

def _get_random_file_name():
    random_string = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    return "{TEMP_PATH}/usda_" + random_string + ".usda"

"""

    return prefix_code + fixed_code


class USDCodeGenInterpreterModifier(CodeInterpreterModifier):
    def _fix_before_run(self, code):
        # Replace some stage open methods. When we run the code, we can't use random
        # file names becouse it will make the error. So we need to replace the
        # random file names with a fixed file name
        for stage in [".subLayerPaths.append"]:
            if stage in code:
                code = replace_sublayerpaths_file(code, stage)

        # Replace stage file
        for stage in ["Usd.Stage.Open"]:
            if stage in code:
                code = replace_stage_file(code, stage, f"{DATA_PATH}/lc_agent_usd_example.usda")

        # Create new stage with random name
        for stage in ["Usd.Stage.CreateNew"]:
            # Generate random filename
            random_string = "".join(random.choices(string.ascii_letters + string.digits, k=8))
            filename = f"{TEMP_PATH}/usda_" + random_string + ".usda"
            if stage in code:
                code = replace_stage_file(code, stage, filename)

        return code
