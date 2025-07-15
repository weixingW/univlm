#!/usr/bin/env python3
"""
Test EMU3 Roundtrip Generation

This script tests the EMU3 roundtrip generation implementation.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from roundtrip_factory import create_roundtrip_generator


def test_emu3_initialization():
    """Test EMU3 model initialization."""
    print("Testing EMU3 initialization...")
    
    try:
        # Test with HuggingFace model names
        generator = create_roundtrip_generator(
            model_type="emu3",
            model_path="BAAI/Emu3-Stage1",  # Use Stage1 for both generation and understanding
            device=0,
            seed=42
        )
        print("✓ EMU3 initialization successful!")
        return generator
    except Exception as e:
        print(f"✗ EMU3 initialization failed: {e}")
        return None


def test_text_to_image(generator):
    """Test text-to-image generation."""
    print("\nTesting text-to-image generation...")
    
    test_prompt = "A beautiful sunset over the ocean"
    
    try:
        image = generator.generate_image_from_text(test_prompt, seed=42)
        print(f"✓ Generated image: {image.size}")
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
        print(f"✓ Generated caption: {caption[:100]}...")
        return caption
    except Exception as e:
        print(f"✗ Image-to-text generation failed: {e}")
        return None


def test_roundtrip(generator):
    """Test full roundtrip generation."""
    print("\nTesting full roundtrip generation...")
    
    test_prompt = "A cat sitting on a windowsill"
    
    try:
        # Generate image from text
        print(f"  Generating image from: '{test_prompt}'")
        image = generator.generate_image_from_text(test_prompt, seed=42)
        print(f"  ✓ Generated image: {image.size}")
        
        # Generate caption from image
        print(f"  Generating caption from image...")
        caption = generator.generate_caption_from_image(
            image, 
            "Describe this image in detail."
        )
        print(f"  ✓ Generated caption: {caption[:100]}...")
        
        # Save test results
        image.save("test_emu3_roundtrip_image.jpg")
        
        print("✓ Full roundtrip generation successful!")
        print(f"  Original prompt: {test_prompt}")
        print(f"  Generated caption: {caption}")
        print(f"  Image saved as: test_emu3_roundtrip_image.jpg")
        
        return True
    except Exception as e:
        print(f"✗ Full roundtrip generation failed: {e}")
        return False


def main():
    """Main test function."""
    print("EMU3 Roundtrip Generation Test")
    print("=" * 50)
    
    # Test initialization
    generator = test_emu3_initialization()
    if generator is None:
        print("\nInitialization failed. Exiting.")
        return
    
    # Test individual components
    image = test_text_to_image(generator)
    if image is not None:
        test_image_to_text(generator, image)
    
    # Test full roundtrip
    test_roundtrip(generator)
    
    print("\n" + "=" * 50)
    print("EMU3 test completed!")
    print("\nTo run actual roundtrip generation:")
    print("python roundtrip_generation.py BAAI/Emu3-Stage1 --model_type emu3 --prompts_file prompts.txt")


if __name__ == "__main__":
    main() 