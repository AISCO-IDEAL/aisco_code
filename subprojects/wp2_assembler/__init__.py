"""
WP2 Assembler - Solar car chassis assembly generation and and regulation implementation.

This package contains different versions of the assembler tool.
Import from specific version directories as needed.
"""

__version__ = "2.2.2"
__author__ = ("vdiekmann", "fsanchez")

from subprojects.wp2_assembler.version2.components.nbz import call_assembly

__all__ = ["call_assembly"]