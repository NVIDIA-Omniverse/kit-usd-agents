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
    name="omni_aiq_planning",
    version="0.1.2",
    author="NVIDIA",
    author_email="doyopk-org@exchange.nvidia.com",
    description="Planning Agent plugin for AgentIQ",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA-Omniverse/kit-usd-agents",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data={
        "omni_aiq_planning": [
            "nodes/systems/*.md",
        ],
    },
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    entry_points={
        "aiq.components": [
            "omni_aiq_planning = omni_aiq_planning.register",
        ],
    },
) 