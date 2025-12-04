#!/usr/bin/env python3
"""
Example Reverse Roundtrip Generation

This script demonstrates how to use the reverse roundtrip generation system
with different models.
"""

import os
import sys
from pathlib import Path
from reverse_roundtrip_factory import create_reverse_roundtrip_generator


def example_blip3o_reverse_roundtrip():
    """Example using BLIP3o for reverse roundtrip generation."""
    print("=== BLIP3o Reverse Roundtrip Example ===")
    
    # Example paths - modify these for your setup
    model_path = "/path/to/blip3o/model"
    image_dir = "/path/to/images"
    output_dir = "blip3o_reverse_example"
    
    try:
        # Create generator
        generator = create_reverse_roundtrip_generator(
            model_type="blip3o",
            model_path=model_path,
            device=0,
            seed=42
        )
        
        # Run reverse roundtrip generation on first 5 images
        results = generator.run_reverse_roundtrip_generation(
            image_dir=image_dir,
            output_dir=output_dir,
            start_idx=0,
            end_idx=5,
            seed_offset=20000
        )
        
        print(f"✓ BLIP3o reverse roundtrip completed! Processed {len(results)} images")
        
    except Exception as e:
        print(f"Error in BLIP3o example: {str(e)}")


def example_mmada_reverse_roundtrip():
    """Example using MMaDA for reverse roundtrip generation."""
    print("=== MMaDA Reverse Roundtrip Example ===")
    
    # Example paths - modify these for your setup
    model_path = "/path/to/mmada/model"
    config_path = "/path/to/mmada_config.yaml"
    image_dir = "/path/to/images"
    output_dir = "mmada_reverse_example"
    
    try:
        # Create generator
        generator = create_reverse_roundtrip_generator(
            model_type="mmada",
            model_path=model_path,
            device=0,
            seed=42,
            config_path=config_path
        )
        
        # Run reverse roundtrip generation on first 5 images
        results = generator.run_reverse_roundtrip_generation(
            image_dir=image_dir,
            output_dir=output_dir,
            start_idx=0,
            end_idx=5,
            seed_offset=20000
        )
        
        print(f"✓ MMaDA reverse roundtrip completed! Processed {len(results)} images")
        
    except Exception as e:
        print(f"Error in MMaDA example: {str(e)}")


def example_emu3_reverse_roundtrip():
    """Example using EMU3 for reverse roundtrip generation."""
    print("=== EMU3 Reverse Roundtrip Example ===")
    
    # Example paths - modify these for your setup
    model_path = "BAAI/Emu3-Gen"  # Can use HuggingFace model names
    image_dir = "/path/to/images"
    output_dir = "emu3_reverse_example"
    
    try:
        # Create generator
        generator = create_reverse_roundtrip_generator(
            model_type="emu3",
            model_path=model_path,
            device=0,
            seed=42
        )
        
        # Run reverse roundtrip generation on first 5 images
        results = generator.run_reverse_roundtrip_generation(
            image_dir=image_dir,
            output_dir=output_dir,
            start_idx=0,
            end_idx=5,
            seed_offset=20000
        )
        
        print(f"✓ EMU3 reverse roundtrip completed! Processed {len(results)} images")
        
    except Exception as e:
        print(f"Error in EMU3 example: {str(e)}")


def example_showo2_reverse_roundtrip():
    """Example using Showo2 for reverse roundtrip generation."""
    print("=== Showo2 Reverse Roundtrip Example ===")
    
    # Example paths - modify these for your setup
    model_path = "/path/to/showo2/model"
    config_path = "/path/to/showo2_config.yaml"
    image_dir = "/path/to/images"
    output_dir = "showo2_reverse_example"
    
    try:
        # Create generator
        generator = create_reverse_roundtrip_generator(
            model_type="showo2",
            model_path=model_path,
            device=0,
            seed=42,
            config_path=config_path
        )
        
        # Run reverse roundtrip generation on first 5 images
        results = generator.run_reverse_roundtrip_generation(
            image_dir=image_dir,
            output_dir=output_dir,
            start_idx=0,
            end_idx=5,
            seed_offset=20000
        )
        
        print(f"✓ Showo2 reverse roundtrip completed! Processed {len(results)} images")
        
    except Exception as e:
        print(f"Error in Showo2 example: {str(e)}")


def example_custom_reverse_roundtrip():
    """Example showing custom usage of reverse roundtrip generation."""
    print("=== Custom Reverse Roundtrip Example ===")
    
    # Example paths - modify these for your setup
    model_path = "/path/to/model"
    image_dir = "/path/to/images"
    output_dir = "custom_reverse_example"
    
    # List of models to try
    models_to_try = ["blip3o", "mmada", "emu3", "omnigen2", "januspro", "showo2", "showo"]
    
    for model_type in models_to_try:
        print(f"\nTrying {model_type.upper()}...")
        
        try:
            # Create generator
            generator = create_reverse_roundtrip_generator(
                model_type=model_type,
                model_path=model_path,
                device=0,
                seed=42
            )
            
            # Run reverse roundtrip generation on first 2 images
            results = generator.run_reverse_roundtrip_generation(
                image_dir=image_dir,
                output_dir=f"{output_dir}_{model_type}",
                start_idx=0,
                end_idx=2,
                seed_offset=20000
            )
            
            print(f"✓ {model_type.upper()} completed! Processed {len(results)} images")
            
        except Exception as e:
            print(f"✗ {model_type.upper()} failed: {str(e)}")


def main():
    """Main function to run examples."""
    print("Reverse Roundtrip Generation Examples")
    print("=" * 50)
    
    # Check if we have the required directories
    if not os.path.exists("/path/to/images"):
        print("Warning: Example image directory '/path/to/images' does not exist.")
        print("Please modify the paths in this script to point to your actual directories.")
        print("\nExample usage:")
        print("1. Modify the paths in this script")
        print("2. Run individual examples or the main function")
        print("3. Check the output directories for results")
        return
    
    # Run individual examples
    example_blip3o_reverse_roundtrip()
    example_mmada_reverse_roundtrip()
    example_emu3_reverse_roundtrip()
    example_showo2_reverse_roundtrip()
    
    # Or run custom example
    # example_custom_reverse_roundtrip()
    
    print("\n" + "=" * 50)
    print("Examples completed! Check the output directories for results.")


if __name__ == "__main__":
    main() 