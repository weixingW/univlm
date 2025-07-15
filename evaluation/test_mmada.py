#!/usr/bin/env python3
"""
Test script for MMaDA roundtrip generation

This script tests the MMaDA implementation to ensure it works correctly.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from mmada_roundtrip import MMaDARoundtripGenerator


def test_mmada_initialization():
    """Test MMaDA model initialization."""
    print("Testing MMaDA initialization...")
    
    try:
        # Note: Replace with actual model path
        model_path = "Gen-Verse/MMaDA-8B-Base"  # or local path
        
        generator = MMaDARoundtripGenerator(
            model_path=model_path,
            device=0,
            seed=42
        )
        
        print("✓ MMaDA initialization successful!")
        return generator
        
    except Exception as e:
        print(f"✗ MMaDA initialization failed: {e}")
        print("This is expected if the model path doesn't exist or dependencies are missing.")
        return None


def test_text_to_image(generator):
    """Test text-to-image generation."""
    print("\nTesting text-to-image generation...")
    
    try:
        prompt = "A beautiful sunset over the ocean with palm trees"
        
        print(f"Generating image for: {prompt}")
        image = generator.generate_image_from_text(prompt, seed=42)
        
        print(f"✓ Generated image with size: {image.size}")
        
        # Save test image
        test_image_path = "test_mmada_generated_image.jpg"
        image.save(test_image_path)
        print(f"✓ Saved test image to: {test_image_path}")
        
        return image
        
    except Exception as e:
        print(f"✗ Text-to-image generation failed: {e}")
        return None


def test_image_to_text(generator, image):
    """Test image-to-text generation."""
    print("\nTesting image-to-text generation...")
    
    try:
        caption = generator.generate_caption_from_image(
            image, 
            "Describe this image in detail."
        )
        
        print(f"✓ Generated caption: {caption}")
        return caption
        
    except Exception as e:
        print(f"✗ Image-to-text generation failed: {e}")
        return None


def test_roundtrip(generator):
    """Test full roundtrip generation."""
    print("\nTesting full roundtrip generation...")
    
    try:
        original_prompt = "A cat sitting on a windowsill looking outside"
        
        print(f"Original prompt: {original_prompt}")
        
        # Step 1: Text -> Image
        print("Step 1: Generating image from text...")
        generated_image = generator.generate_image_from_text(original_prompt, seed=42)
        print(f"✓ Generated image with size: {generated_image.size}")
        
        # Step 2: Image -> Text
        print("Step 2: Generating caption from image...")
        generated_caption = generator.generate_caption_from_image(
            generated_image, 
            "Describe this image in detail."
        )
        print(f"✓ Generated caption: {generated_caption}")
        
        # Save roundtrip results
        roundtrip_image_path = "test_mmada_roundtrip_image.jpg"
        generated_image.save(roundtrip_image_path)
        
        print(f"\n=== Roundtrip Results ===")
        print(f"Original prompt: {original_prompt}")
        print(f"Generated caption: {generated_caption}")
        print(f"Image saved to: {roundtrip_image_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Roundtrip generation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("MMaDA Roundtrip Generation Test")
    print("=" * 50)
    
    # Test initialization
    generator = test_mmada_initialization()
    
    if generator is None:
        print("\nSkipping further tests due to initialization failure.")
        print("Please ensure:")
        print("1. MMaDA model is available at the specified path")
        print("2. All dependencies are installed")
        print("3. CUDA is available (if using GPU)")
        return
    
    # Test individual components
    image = test_text_to_image(generator)
    if image is not None:
        test_image_to_text(generator, image)
    
    # Test full roundtrip
    test_roundtrip(generator)
    
    print("\n" + "=" * 50)
    print("Tests completed!")


if __name__ == "__main__":
    main() 