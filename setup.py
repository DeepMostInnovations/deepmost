from setuptools import setup, find_packages
import sys

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

base_requirements = [
    "torch==2.6.0",
    "transformers==4.51.3",
    "stable_baselines3==2.6.0",
    "gymnasium==1.1.1",
    "numpy==2.0.2",
    "requests==2.32.3",
    "tqdm==4.67.1",
    "huggingface-hub==0.31.2",
    "llama-cpp-python==0.3.9",
    "openai==1.78.1"
]

python_version = sys.version_info
supported_extras = {}

if python_version >= (3, 10):
    supported_extras["azure"] = ["openai==1.78.1"]

if python_version >= (3, 11):
    supported_extras["gpu"] = ["llama-cpp-python==0.3.9"]
    supported_extras["full"] = ["openai==1.78.1", "llama-cpp-python==0.3.9"]
elif python_version >= (3, 10):
    supported_extras["gpu"] = ["llama-cpp-python==0.3.9"]
    supported_extras["azure-full"] = ["openai==1.78.1", "llama-cpp-python==0.3.9"]

dev_requirements = list(base_requirements) # Start with base
if "azure" in supported_extras:
    dev_requirements.extend(supported_extras["azure"])
if "gpu" in supported_extras:
    # Ensure llama-cpp-python is added only once if 'full'/'azure-full' also used
    if "llama-cpp-python==0.3.9" not in [dep for extra_deps in supported_extras.values() for dep in extra_deps if dep.startswith("llama-cpp-python")]:
         if "llama-cpp-python==0.3.9" not in dev_requirements: # final check for safety
            dev_requirements.extend(supported_extras["gpu"])
    # Add if it's in full but not standalone gpu (e.g. python < 3.11 but >=3.10 has azure-full)
    elif "llama-cpp-python==0.3.9" in supported_extras.get("full", []) and "llama-cpp-python==0.3.9" not in dev_requirements:
         if "llama-cpp-python==0.3.9" not in dev_requirements:
            dev_requirements.append("llama-cpp-python==0.3.9")
    elif "llama-cpp-python==0.3.9" in supported_extras.get("azure-full", []) and "llama-cpp-python==0.3.9" not in dev_requirements:
         if "llama-cpp-python==0.3.9" not in dev_requirements:
            dev_requirements.append("llama-cpp-python==0.3.9")


supported_extras["dev"] = list(set(dev_requirements + [ # Use set to remove duplicates then list
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "isort>=5.0.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0"
]))


# Simplified python_requires, assuming 3.10 as a practical minimum for full features.
# Adjust if you strongly need to support 3.8/3.9 for base non-LLM/Azure functionality.
# The Colab env was 3.11.
python_requires_str = ">=3.11"
classifiers_python = [
    "Programming Language :: Python :: 3.11"
]

# If strict 3.8 support is needed for non-LLM/Azure base, uncomment and adapt:
# if python_version < (3,10):
#    python_requires_str = ">=3.8"
#    classifiers_python = [
#        "Programming Language :: Python :: 3.8",
#        "Programming Language :: Python :: 3.9",
#    ] + classifiers_python


setup(
    name="deepmost",
    version="0.1.1",
    author="DeepMost Innovations",
    author_email="support@deepmost.ai",
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
        "Programming Language :: Python :: 3",
    ] + classifiers_python,
    python_requires=python_requires_str,
    install_requires=base_requirements,
    extras_require=supported_extras,
    keywords="sales, conversion, prediction, reinforcement-learning, ai, machine-learning, llm, nlp",
    project_urls={
        "Bug Reports": "https://github.com/DeepMostInnovations/deepmost/issues",
        "Source": "https://github.com/DeepMostInnovations/deepmost",
        "Documentation": "https://deepmost.readthedocs.io/",
        "Model Repository": "https://huggingface.co/DeepMostInnovations",
    },
)