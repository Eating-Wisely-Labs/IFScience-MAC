from setuptools import setup, find_packages


# Read the contents of your requirements file
def read_requirements():
    with open("requirements.txt") as req:
        return req.read().splitlines()


setup(
    name="ifsci_agentic",
    version="0.1.0",
    packages=find_packages(),
    install_requires=read_requirements(),
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
