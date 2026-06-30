"""
Models package for the Flexible-TADA framework.

This package contains the core logic for dynamically unfreezing specific layers
(Flexible TADA), applying PEFT methods like LoRA, and loading baseline models.
"""

# Exposing the main factory function so it can be cleanly imported in main.py
# Example usage in main.py: from models import get_model

from .model_factory import get_model

# Define what gets imported when a user runs `from models import *`
__all__ = ["get_model"]