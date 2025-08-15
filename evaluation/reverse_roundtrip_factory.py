#!/usr/bin/env python3
"""
Reverse Roundtrip Generator Factory

This module provides a factory pattern for creating reverse roundtrip generators
for different model types.
"""

from typing import Dict, Type, Optional, List
from reverse_roundtrip_base import ReverseRoundtripGenerator
from blip3o_reverse_roundtrip import BLIP3oReverseRoundtripGenerator
from mmada_reverse_roundtrip import MMaDAReverseRoundtripGenerator
from emu3_reverse_roundtrip import EMU3ReverseRoundtripGenerator
from omnigen2_reverse_roundtrip import OmniGen2ReverseRoundtripGenerator
from januspro_reverse_roundtrip import JanusProReverseRoundtripGenerator
from showo2_reverse_roundtrip import Showo2ReverseRoundtripGenerator
from showo_reverse_roundtrip import ShowoReverseRoundtripGenerator


class ReverseRoundtripGeneratorFactory:
    """Factory for creating reverse roundtrip generators."""
    
    _generators: Dict[str, Type[ReverseRoundtripGenerator]] = {
        "blip3o": BLIP3oReverseRoundtripGenerator,
        "mmada": MMaDAReverseRoundtripGenerator,
        "emu3": EMU3ReverseRoundtripGenerator,
        "omnigen2": OmniGen2ReverseRoundtripGenerator,
        "januspro": JanusProReverseRoundtripGenerator,
        "showo2": Showo2ReverseRoundtripGenerator,
        "showo": ShowoReverseRoundtripGenerator,
    }
    
    @classmethod
    def register_generator(cls, model_type: str, generator_class: Type[ReverseRoundtripGenerator]):
        """Register a new generator type."""
        cls._generators[model_type] = generator_class
    
    @classmethod
    def get_generator_class(cls, model_type: str) -> Type[ReverseRoundtripGenerator]:
        """Get the generator class for a given model type."""
        if model_type not in cls._generators:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(cls._generators.keys())}"
            )
        return cls._generators[model_type]
    
    @classmethod
    def create_generator(cls, model_type: str, model_path: str, device: int = 0, seed: int = 42, config_path: Optional[str] = None) -> ReverseRoundtripGenerator:
        """Create a reverse roundtrip generator instance."""
        generator_class = cls.get_generator_class(model_type)
        return generator_class(model_path=model_path, device=device, seed=seed, config_path=config_path)
    
    @classmethod
    def get_supported_models(cls) -> list:
        """Get list of supported model types."""
        return list(cls._generators.keys())


# Convenience function
def create_reverse_roundtrip_generator(model_type: str, model_path: str, device: int = 0, seed: int = 42, config_path: Optional[str] = None) -> ReverseRoundtripGenerator:
    """Create a reverse roundtrip generator instance."""
    return ReverseRoundtripGeneratorFactory.create_generator(model_type, model_path, device, seed, config_path) 