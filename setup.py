from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Base requirements needed for the core functionality (excluding optional LLM)
base_requirements = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "stable-baselines3>=2.0.0",
    "gymnasium>=0.28.0",
    "numpy>=1.23.0",
    "requests>=2.28.0",
    "tqdm>=4.60.0",
    "huggingface-hub>=0.17.0",
]

# Optional dependencies
extras_require = {
    "gpu": [
        "llama-cpp-python[server]>=0.2.20", # For local GGUF model support.
    ],
    # No "azure" or "full" if focusing only on open-source
}

# 'dev' dependencies for development (testing, linting, formatting)
# It includes 'gpu' extra to ensure all functionalities can be tested.
extras_require["dev"] = extras_require.get("gpu", []) + [ # Use .get for safety if gpu extra were conditional
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.10.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "ruff>=0.1.0",
]


# Python version requirement
python_requires_str = ">=3.11"

# Classifiers for PyPI
classifiers_python = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

setup(
    name="deepmost",
    version="0.1.4", # Updated version
    author="DeepMost Innovations",
    author_email="support@deepmostai.com",
    description="Sales conversion prediction using reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepMostInnovations/deepmost",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ] + classifiers_python,
    python_requires=python_requires_str,
    install_requires=base_requirements,
    extras_require=extras_require,
    keywords="sales, conversion, prediction, reinforcement-learning, ai, machine-learning, llm, nlp, gguf",
    project_urls={
        "Bug Reports": "https://github.com/DeepMostInnovations/deepmost/issues",
        "Source": "https://github.com/DeepMostInnovations/deepmost",
        "Documentation": "https://deepmost.readthedocs.io/",
        "Model Repository": "https://huggingface.co/DeepMostInnovations",
    },
)