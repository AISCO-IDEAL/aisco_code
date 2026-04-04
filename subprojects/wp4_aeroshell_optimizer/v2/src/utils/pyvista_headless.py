"""Utilities to configure PyVista for headless HPC execution."""

from __future__ import annotations

import importlib
import os
from typing import Any, cast


def is_headless_runtime() -> bool:
    """Return True when running without an X display (typical Slurm batch jobs)."""
    display = os.environ.get("DISPLAY", "")
    return display.strip() == ""


def configure_headless_pyvista() -> None:
    """Set environment for headless rendering before importing PyVista."""
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")


def ensure_pyvista_headless():
    """Configure environment, import PyVista, and enable off-screen mode."""
    configure_headless_pyvista()
    pv = cast(Any, importlib.import_module("pyvista"))

    pv.OFF_SCREEN = True
    return pv
