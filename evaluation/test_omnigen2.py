#!/usr/bin/env python3
"""
Test script for OmniGen2 roundtrip generation.
"""

import os
import sys
from pathlib import Path

# Add the evaluation directory to the path
sys.path.append(str(Path(__file__).parent))

from roundtrip_factory import create_roundtrip_generator


def test_omnigen2_roundtrip():
    """Test OmniGen2 roundtrip generation."""
    
    # Model path - you'll need to update this to your actual OmniGen2 model path
    model_path = "/path/to/your/omnigen2/model"  # Update this path
    
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist!")
        print("Please update the model_path variable in this script to point to your OmniGen2 model.")
        return
    
    try:
        # Create OmniGen2 roundtrip generator
        print("Creating OmniGen2 roundtrip generator...")
        generator = create_roundtrip_generator(
            model_type="omnigen2",
            model_path=model_path,
            device=0,
            seed=42
        )
        
        # Test text-to-image generation
        print("\nTesting text-to-image generation...")
        test_prompt = "A beautiful sunset over the ocean with palm trees"
        generated_image = generator.generate_image_from_text(test_prompt)
        print(f"✓ Generated image from prompt: {test_prompt}")
        
        # Save the generated image
        output_path = "test_omnigen2_generated_image.jpg"
        generated_image.save(output_path)
        print(f"✓ Saved generated image to: {output_path}")
        
        # Test image-to-text generation
        print("\nTesting image-to-text generation...")
        generated_caption = generator.generate_caption_from_image(generated_image)
        print(f"✓ Generated caption: {generated_caption}")
        
        print("\n✓ OmniGen2 roundtrip test completed successfully!")
        
    except Exception as e:
        print(f"Error during OmniGen2 roundtrip test: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_omnigen2_roundtrip() 