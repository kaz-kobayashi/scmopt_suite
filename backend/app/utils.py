"""
Utility functions for the SCM Optimization Suite backend.
"""
import numpy as np
import pandas as pd
from typing import Any


def convert_to_json_safe(obj: Any) -> Any:
    """
    Convert various Python objects to JSON-serializable format.
    
    This function handles:
    - NumPy integers and floats
    - NumPy arrays
    - Python tuples (converts to lists)
    - Pandas NaN values
    - Nested lists and dictionaries
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        val = float(obj)
        if np.isinf(val) or np.isnan(val):
            return None
        return val
    elif isinstance(obj, (int, float)):
        if np.isinf(obj) or np.isnan(obj):
            return None
        return obj
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        # Convert tuples to lists to avoid JSON serialization issues
        return list(obj)
    elif isinstance(obj, list):
        return [convert_to_json_safe(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_safe(value) for key, value in obj.items()}
    elif isinstance(obj, (int, float, str, bool)) and pd.isna(obj):
        return None
    else:
        return obj


def safe_json_response(data: Any) -> Any:
    """
    Create a JSON-safe response by converting all non-serializable objects.
    
    Args:
        data: Data to make JSON-safe
        
    Returns:
        JSON-serializable data
    """
    return convert_to_json_safe(data)