"""
settings_loader.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Bridge module that exposes 01_config/settings.py under a clean
    import path, since folder names starting with digits cannot be
    imported directly. All other modules load configuration via this.

Inputs:
    01_config/settings.py      configuration module loaded dynamically

Outputs:
    (none — library module; returns the cached settings module object)

Usage:
    from settings_loader import load_settings
    settings = load_settings()
"""

import importlib.util
from pathlib import Path

_settings_cache = None


def load_settings():
    """Load and return the settings module from 01_config/settings.py.

    The module is loaded once and cached; subsequent calls return the
    same object (avoids re-executing load_dotenv / mkdir on every import).
    """
    global _settings_cache
    if _settings_cache is not None:
        return _settings_cache
    settings_path = Path(__file__).resolve().parent / "01_config" / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    _settings_cache = settings
    return settings
