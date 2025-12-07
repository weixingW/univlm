#!/usr/bin/env python3
"""
Reverse Roundtrip Generator Factory

This module provides a factory pattern for creating reverse roundtrip generators
for different model types.

NOTE: We use lazy imports to avoid path conflicts between different models.
Each model adds its own directory to sys.path, and importing all at once
causes conflicts (e.g., Show-o's 'models' package vs MMaDA's 'models' package).
"""

from typing import Dict, Type, Optional, List
from reverse_roundtrip_base import ReverseRoundtripGenerator


# Lazy import functions to avoid path conflicts between different models
def _get_blip3o_generator():
    from blip3o_reverse_roundtrip import BLIP3oReverseRoundtripGenerator
    return BLIP3oReverseRoundtripGenerator

def _get_mmada_generator():
    from mmada_reverse_roundtrip import MMaDAReverseRoundtripGenerator
    return MMaDAReverseRoundtripGenerator

def _get_emu3_generator():
    from emu3_reverse_roundtrip import EMU3ReverseRoundtripGenerator
    return EMU3ReverseRoundtripGenerator

def _get_omnigen2_generator():
    from omnigen2_reverse_roundtrip import OmniGen2ReverseRoundtripGenerator
    return OmniGen2ReverseRoundtripGenerator

def _get_januspro_generator():
    from januspro_reverse_roundtrip import JanusProReverseRoundtripGenerator
    return JanusProReverseRoundtripGenerator

def _get_showo2_generator():
    from showo2_reverse_roundtrip import Showo2ReverseRoundtripGenerator
    return Showo2ReverseRoundtripGenerator

def _get_showo_generator():
    from showo_reverse_roundtrip import ShowoReverseRoundtripGenerator
    return ShowoReverseRoundtripGenerator


class ReverseRoundtripGeneratorFactory:
    """Factory for creating reverse roundtrip generators."""
    
    # Map model types to lazy import functions instead of classes directly
    _generator_loaders: Dict[str, callable] = {
        "blip3o": _get_blip3o_generator,
        "mmada": _get_mmada_generator,
        "emu3": _get_emu3_generator,
        "omnigen2": _get_omnigen2_generator,
        "januspro": _get_januspro_generator,
        "showo2": _get_showo2_generator,
        "showo": _get_showo_generator,
    }
    
    @classmethod
    def register_generator(cls, model_type: str, generator_loader: callable):
        """Register a new generator type with a lazy loader function."""
        cls._generator_loaders[model_type] = generator_loader
    
    @classmethod
    def get_generator_class(cls, model_type: str) -> Type[ReverseRoundtripGenerator]:
        """Get the generator class for a given model type (lazy loaded)."""
        if model_type not in cls._generator_loaders:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(cls._generator_loaders.keys())}"
            )
        # Lazy load the generator class
        return cls._generator_loaders[model_type]()
    
    @classmethod
    def create_generator(cls, model_type: str, model_path: str, device: int = 0, seed: int = 42, config_path: Optional[str] = None) -> ReverseRoundtripGenerator:
        """Create a reverse roundtrip generator instance."""
        generator_class = cls.get_generator_class(model_type)
        return generator_class(model_path=model_path, device=device, seed=seed, config_path=config_path)
    
    @classmethod
    def get_supported_models(cls) -> list:
        """Get list of supported model types."""
        return list(cls._generator_loaders.keys())


# Convenience function
def create_reverse_roundtrip_generator(model_type: str, model_path: str, device: int = 0, seed: int = 42, config_path: Optional[str] = None) -> ReverseRoundtripGenerator:
    """Create a reverse roundtrip generator instance."""
    return ReverseRoundtripGeneratorFactory.create_generator(model_type, model_path, device, seed, config_path) 