from setuptools import setup, find_packages
import sys

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Base requirements that work for both backends
base_requirements = [
    "torch>=2.0.0",
    "transformers>=4.30.0", 
    "stable_baselines3==2.6.0",
    "gymnasium>=0.29.1",
    "numpy>=1.20.0",
    "requests>=2.25.0",
    "tqdm>=4.65.0",
    "huggingface-hub>=0.16.0"
]

# Check Python version to determine supported features
python_version = sys.version_info
supported_extras = {}

# Azure OpenAI support (Python 3.10+)
if python_version >= (3, 10):
    supported_extras["azure"] = ["openai>=1.0.0"]

# GPU/LLM support (Python 3.11+ for full open-source features)
if python_version >= (3, 11):
    supported_extras["gpu"] = ["llama-cpp-python"]
    supported_extras["full"] = ["openai>=1.0.0", "llama-cpp-python"]
elif python_version >= (3, 10):
    # Limited GPU support for Azure backend
    supported_extras["gpu"] = ["llama-cpp-python"]
    supported_extras["azure-full"] = ["openai>=1.0.0", "llama-cpp-python"]

# Development dependencies
supported_extras["dev"] = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "isort>=5.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0"
]

# Determine Python version support
if python_version >= (3, 11):
    python_requires = ">=3.11"
    classifiers_python = [
        "Programming Language :: Python :: 3.11",
    ]
elif python_version >= (3, 10):
    python_requires = ">=3.10"
    classifiers_python = [
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
    ]
else:
    python_requires = ">=3.10"
    classifiers_python = [
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ]

setup(
    name="deepmost",
    version="0.1.1",
    author="DeepMost Innovations",
    author_email="support@deepmostai.com",
    description="Sales conversion prediction using reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepMostInnovations/deepmost",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ] + classifiers_python,
    python_requires=python_requires,
    install_requires=base_requirements,
    extras_require=supported_extras,
    keywords="sales, conversion, prediction, reinforcement-learning, ai, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/DeepMostInnovations/deepmost/issues",
        "Source": "https://github.com/DeepMostInnovations/deepmost",
        "Documentation": "https://deepmost.readthedocs.io/",
    },
)