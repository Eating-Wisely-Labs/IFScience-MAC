from setuptools import setup, find_packages

setup(
    name="ifsci_agentic",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "aiohttp>=3.8.0",
        "requests>=2.25.0",
        "python-dotenv>=0.19.0",
        "loguru>=0.5.3",
    ],
    author="IFSci Team",
    author_email="team@ifsci.com",
    description="LLM Framework for IFSci Server",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ifsci/ifsci_agentic",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
