"""
Modular Roundtrip Generation Package

This package provides a modular implementation of roundtrip generation
that supports multiple models (BLIP3o, MMaDA, etc.).
"""

from .roundtrip_base import RoundtripGenerator, create_parser
from .roundtrip_factory import RoundtripGeneratorFactory, create_roundtrip_generator
from .blip3o_roundtrip import BLIP3oRoundtripGenerator
from .mmada_roundtrip import MMaDARoundtripGenerator
from .emu3_roundtrip import EMU3RoundtripGenerator

__version__ = "1.0.0"
__author__ = "UniVLM Team"

__all__ = [
    "RoundtripGenerator",
    "RoundtripGeneratorFactory", 
    "create_roundtrip_generator",
    "create_parser",
    "BLIP3oRoundtripGenerator",
    "MMaDARoundtripGenerator",
    "EMU3RoundtripGenerator",
] 