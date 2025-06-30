"""
DeepMost - Sales Conversion Prediction Package
A powerful Python package for predicting sales conversion probability using
reinforcement learning.
"""

__version__ = "0.5.0" 


from . import sales
from . import prospecting # Added

__all__ = [
    "sales",
    "prospecting", # Added
    "__version__"
]