## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from setuptools import find_packages
from setuptools import setup
import os


def req_file(filename, folder="."):
    with open(os.path.join(folder, filename), encoding="utf-8") as f:
        content = f.readlines()

    lines = [x.strip() for x in content]

    # remove whitespace characters
    # Example: `\n` at the end of each line
    lines = [x for x in lines if x]

    # remove lines that start with #
    lines = [x for x in lines if not x.startswith("#")]

    return lines


install_requires = req_file("requirements.txt")


setup(
    name="lc_agent_retrievers",
    version="0.2.6",
    author="Omniverse GenAI Team",
    author_email="doyopk-org@exchange.nvidia.com",
    description="Utility modules for LC Agent",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA-Omniverse/kit-usd-agents",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "lc_agent_retrievers": [
            "data/**/*",
        ]
    },
    include_package_data=True,
    install_requires=install_requires,
    classifiers=[
        # https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
