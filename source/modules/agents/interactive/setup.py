from setuptools import find_packages, setup
import os


def req_file(filename, folder="."):
    with open(os.path.join(folder, filename), encoding="utf-8") as f:
        content = f.readlines()
    lines = [x.strip() for x in content]
    lines = [x for x in lines if x]
    lines = [x for x in lines if not x.startswith("#")]
    return lines


install_requires = req_file("requirements.txt")


setup(
    name="lc_agent_interactive",
    version="0.1.1",
    author="Omniverse GenAI Team",
    author_email="doyopk-org@exchange.nvidia.com",
    description="Interactive base for module-guided assistants",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA-Omniverse/kit-usd-agents",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
