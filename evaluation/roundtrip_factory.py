#!/usr/bin/env python3
"""
Roundtrip Generator Factory

This module provides a factory pattern for creating roundtrip generators
for different model types.
"""

from typing import Dict, Type, Optional, List
from roundtrip_base import RoundtripGenerator
from blip3o_roundtrip import BLIP3oRoundtripGenerator
from mmada_roundtrip import MMaDARoundtripGenerator
from emu3_roundtrip import EMU3RoundtripGenerator
from omnigen2_roundtrip import OmniGen2RoundtripGenerator
from januspro_roundtrip import JanusProRoundtripGenerator
from showo2_roundtrip import Showo2RoundtripGenerator
from showo_roundtrip import ShowoRoundtripGenerator


class RoundtripGeneratorFactory:
    """Factory for creating roundtrip generators."""
    
    _generators: Dict[str, Type[RoundtripGenerator]] = {
        "blip3o": BLIP3oRoundtripGenerator,
        "mmada": MMaDARoundtripGenerator,
        "emu3": EMU3RoundtripGenerator,
        "omnigen2": OmniGen2RoundtripGenerator,
        "januspro": JanusProRoundtripGenerator,
        "showo2": Showo2RoundtripGenerator,
        "showo": ShowoRoundtripGenerator,
    }
    
    @classmethod
    def register_generator(cls, model_type: str, generator_class: Type[RoundtripGenerator]):
        """Register a new generator type."""
        cls._generators[model_type] = generator_class
    
    @classmethod
    def get_generator_class(cls, model_type: str) -> Type[RoundtripGenerator]:
        """Get the generator class for a given model type."""
        if model_type not in cls._generators:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(cls._generators.keys())}"
            )
        return cls._generators[model_type]
    
    @classmethod
    def create_generator(cls, model_type: str, model_path: str, device: int = 0, seed: int = 42, config_path: Optional[str] = None) -> RoundtripGenerator:
        """Create a roundtrip generator instance."""
        generator_class = cls.get_generator_class(model_type)
        return generator_class(model_path=model_path, device=device, seed=seed, config_path=config_path)
    
    @classmethod
    def get_supported_models(cls) -> list:
        """Get list of supported model types."""
        return list(cls._generators.keys())


# Convenience function
def create_roundtrip_generator(model_type: str, model_path: str, device: int = 0, seed: int = 42, config_path: Optional[str] = None) -> RoundtripGenerator:
    """Create a roundtrip generator instance."""
    return RoundtripGeneratorFactory.create_generator(model_type, model_path, device, seed, config_path) 