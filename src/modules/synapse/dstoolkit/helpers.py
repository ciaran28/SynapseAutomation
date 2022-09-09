import os
import json
from typing import Any


def current_directory() -> str:
    """
    Get current directory.

    Returns
    -------
    str
        The current directory path
    """
    return os.path.dirname(os.path.realpath(__file__))


def add_folder_in_current_directory(folder_name: str) -> bool:
    """
    Add a folder in the current directory.

    Parameters
    ----------
    folder_name : str
        New folder name

    Returns
    -------
    bool
        True if success
    """
    output_folder = os.path.join(current_directory(), folder_name)
    os.makedirs(output_folder)
    return True


def is_json_serializable(x: Any) -> bool:
    """
    Check if the object is serializable.

    Parameters
    ----------
    x : Any
        Object to validate

    Returns
    -------
    bool
        True if success
    """
    try:
        json.dumps(x)
        return True
    except Exception:
        return False
