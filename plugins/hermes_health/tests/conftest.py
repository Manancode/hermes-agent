"""Conftest — make hermes_health importable for tests."""

import sys
from pathlib import Path

_plugin_dir = Path(__file__).resolve().parent.parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))
