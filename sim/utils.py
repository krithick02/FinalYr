"""
utils.py — Shared helpers for the IoT offloading simulation.
"""

import os
import yaml


def load_config(path: str = None) -> dict:
    """
    Load config.yaml.
    Looks for the file in the same directory as this module if path is None.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dir(directory: str) -> None:
    """Create directory (and parents) if it does not exist."""
    os.makedirs(directory, exist_ok=True)
