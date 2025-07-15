#!/usr/bin/env python3
"""
Test Show-o2 Roundtrip Generation

This script tests the Show-o2 roundtrip generation implementation.
"""

import os
import sys
from pathlib import Path
import torch
from PIL import Image

# Add the evaluation directory to the path
sys.path.append(str(Path(__file__).parent))

# Add Show-o2 path for imports
from pathlib import Path
showo_path = Path(__file__).parent.parent / "Show-o"
showo2_path = showo_path / "show-o2"
sys.path.insert(0, str(showo2_path))

from showo2_roundtrip import Showo2RoundtripGenerator


def test_showo2_roundtrip():
    """Test Show-o2 roundtrip generation."""
    print("Testing Show-o2 roundtrip generation...")
    
    # Test parameters
    model_path = "showlab/show-o2-7b"  # Example model path
    config_path = "../configs/showo2_config.yaml"  # Path to config file
    device = 0
    seed = 42
    
    try:
        # Initialize the generator
        print("Initializing Show-o2 generator...")
        generator = Showo2RoundtripGenerator(
            model_path=model_path,
            device=device,
            seed=seed,
            config_path=config_path
        )
        print("✓ Show-o2 generator initialized successfully!")
        
        # Test text-to-image generation
        test_prompt = "A beautiful sunset over the ocean with palm trees"
        print(f"\nTesting text-to-image generation with prompt: '{test_prompt}'")
        
        generated_image = generator.generate_image_from_text(test_prompt, seed=seed)
        print("✓ Image generation successful!")
        print(f"Generated image size: {generated_image.size}")
        
        # Test image-to-text generation
        print("\nTesting image-to-text generation...")
        caption = generator.generate_caption_from_image(generated_image)
        print("✓ Caption generation successful!")
        print(f"Generated caption: {caption}")
        
        # Test roundtrip generation
        print("\nTesting full roundtrip generation...")
        roundtrip_caption = generator.generate_caption_from_image(generated_image)
        print("✓ Roundtrip generation successful!")
        print(f"Roundtrip caption: {roundtrip_caption}")
        
        print("\n🎉 All Show-o2 roundtrip tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_showo2_with_config():
    """Test Show-o2 with custom configuration."""
    print("\nTesting Show-o2 with custom configuration...")
    
    # Test parameters
    model_path = "showlab/show-o2-7b"
    config_path = "../configs/showo2_config.yaml"
    device = 0
    seed = 42
    
    try:
        # Initialize the generator with custom config
        generator = Showo2RoundtripGenerator(
            model_path=model_path,
            device=device,
            seed=seed,
            config_path=config_path
        )
        
        # Override some config parameters
        generator.config.transport.num_inference_steps = 25  # Faster generation
        generator.config.guidance_scale = 3.0  # Lower guidance scale
        
        print("✓ Show-o2 generator with custom config initialized!")
        
        # Test generation with custom config
        test_prompt = "A futuristic city with flying cars"
        generated_image = generator.generate_image_from_text(test_prompt, seed=seed)
        caption = generator.generate_caption_from_image(generated_image)
        
        print("✓ Custom config test successful!")
        print(f"Generated caption: {caption}")
        
    except Exception as e:
        print(f"❌ Custom config test failed: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Show-o2 Roundtrip Generation Test")
    print("=" * 60)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available, using CPU")
    
    # Run tests
    success = True
    success &= test_showo2_roundtrip()
    success &= test_showo2_with_config()
    
    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 