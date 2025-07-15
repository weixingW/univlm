#!/usr/bin/env python3
"""
Example Usage of Modular Roundtrip Generation

This script demonstrates how to use the modular roundtrip generation system
programmatically.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from roundtrip_factory import create_roundtrip_generator, RoundtripGeneratorFactory


def example_basic_usage():
    """Example of basic roundtrip generation usage."""
    print("=== Basic Roundtrip Generation Example ===")
    
    # Check supported models
    supported_models = RoundtripGeneratorFactory.get_supported_models()
    print(f"Supported models: {supported_models}")
    
    # Create a generator (this would require an actual model path)
    try:
        # Note: Replace with actual model path
        model_path = "/path/to/your/blip3o/model"
        
        generator = create_roundtrip_generator(
            model_type="blip3o",
            model_path=model_path,
            device=0,
            seed=42
        )
        
        print("Generator created successfully!")
        
        # Example of running roundtrip generation
        # results = generator.run_roundtrip_generation(
        #     prompts_file="prompts.txt",
        #     output_dir="example_results",
        #     start_idx=0,
        #     end_idx=5
        # )
        # print(f"Generated {len(results)} results")
        
    except Exception as e:
        print(f"Error creating generator: {e}")
        print("This is expected if the model path doesn't exist.")


def example_caption_to_image():
    """Example of generating images from existing captions."""
    print("\n=== Caption to Image Generation Example ===")
    
    try:
        # Note: Replace with actual model path
        model_path = "/path/to/your/blip3o/model"
        
        generator = create_roundtrip_generator(
            model_type="blip3o",
            model_path=model_path,
            device=0,
            seed=42
        )
        
        print("Generator created successfully!")
        
        # Example of generating images from captions
        # results = generator.generate_images_from_captions(
        #     captions_dir="captions",
        #     output_dir="caption_images",
        #     start_idx=0,
        #     end_idx=3,
        #     seed_offset=10000
        # )
        # print(f"Generated {len(results)} images from captions")
        
    except Exception as e:
        print(f"Error creating generator: {e}")
        print("This is expected if the model path doesn't exist.")


def example_custom_implementation():
    """Example of how to create a custom implementation."""
    print("\n=== Custom Implementation Example ===")
    
    from roundtrip_base import RoundtripGenerator
    from typing import Optional
    from PIL import Image
    
    class CustomRoundtripGenerator(RoundtripGenerator):
        """Example custom implementation."""
        
        def _initialize_models(self):
            """Initialize custom models."""
            print("Initializing custom models...")
            # Add your model initialization here
            pass
        
        def generate_image_from_text(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
            """Generate image from text prompt."""
            print(f"Custom image generation for: {prompt[:50]}...")
            # Add your image generation logic here
            # For now, return a placeholder
            return Image.new('RGB', (512, 512), color='white')
        
        def generate_caption_from_image(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
            """Generate caption from image."""
            print(f"Custom caption generation for image of size {image.size}")
            # Add your caption generation logic here
            return "A custom generated caption for the image."
    
    # Register the custom generator
    RoundtripGeneratorFactory.register_generator("custom", CustomRoundtripGenerator)
    
    # Create and use the custom generator
    try:
        generator = create_roundtrip_generator(
            model_type="custom",
            model_path="/dummy/path",
            device=0,
            seed=42
        )
        
        print("Custom generator created successfully!")
        
        # Test the custom implementation
        image = generator.generate_image_from_text("A beautiful sunset")
        caption = generator.generate_caption_from_image(image)
        
        print(f"Generated caption: {caption}")
        
    except Exception as e:
        print(f"Error with custom generator: {e}")


def main():
    """Run all examples."""
    print("Modular Roundtrip Generation Examples")
    print("=" * 50)
    
    example_basic_usage()
    example_caption_to_image()
    example_custom_implementation()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo run actual roundtrip generation:")
    print("python roundtrip_generation.py /path/to/model --model_type blip3o --prompts_file prompts.txt")


if __name__ == "__main__":
    main() 