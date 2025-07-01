# deepmost/__init__.py

"""
DeepMost - Sales Conversion Prediction and Prospecting Package
A powerful Python package for predicting sales conversion probability using
reinforcement learning and generating AI-powered prospecting plans.
"""

__version__ = "0.5.0" 


from . import sales
from . import prospecting

__all__ = [
    "sales",
    "prospecting",
    "__version__"
]