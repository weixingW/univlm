#!/usr/bin/env python3
"""
Model Comparison Script

This script demonstrates how to compare roundtrip generation between
different models (BLIP3o and MMaDA).
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from roundtrip_factory import create_roundtrip_generator


def compare_models():
    """Compare roundtrip generation between BLIP3o and MMaDA."""
    print("Model Comparison: BLIP3o vs MMaDA")
    print("=" * 50)
    
    # Test prompt
    test_prompt = "A beautiful sunset over the ocean with palm trees"
    
    # Model configurations
    models = [
        {
            "name": "BLIP3o",
            "type": "blip3o",
            "path": "/path/to/blip3o/model"  # Replace with actual path
        },
        {
            "name": "MMaDA",
            "type": "mmada", 
            "path": "Gen-Verse/MMaDA-8B-Base"
        }
    ]
    
    results = {}
    
    for model_config in models:
        print(f"\nTesting {model_config['name']}...")
        
        try:
            # Create generator
            generator = create_roundtrip_generator(
                model_type=model_config["type"],
                model_path=model_config["path"],
                device=0,
                seed=42
            )
            
            print(f"✓ {model_config['name']} initialized successfully")
            
            # Test text-to-image
            print(f"  Generating image from: '{test_prompt}'")
            image = generator.generate_image_from_text(test_prompt, seed=42)
            print(f"  ✓ Generated image: {image.size}")
            
            # Test image-to-text
            print(f"  Generating caption from image...")
            caption = generator.generate_caption_from_image(
                image, 
                "Describe this image in detail."
            )
            print(f"  ✓ Generated caption: {caption[:100]}...")
            
            # Save results
            image_path = f"comparison_{model_config['type']}_image.jpg"
            image.save(image_path)
            
            results[model_config["name"]] = {
                "success": True,
                "image_path": image_path,
                "caption": caption,
                "image_size": image.size
            }
            
        except Exception as e:
            print(f"✗ {model_config['name']} failed: {e}")
            results[model_config["name"]] = {
                "success": False,
                "error": str(e)
            }
    
    # Print comparison results
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        if result["success"]:
            print(f"  ✓ Success")
            print(f"  Image size: {result['image_size']}")
            print(f"  Image saved: {result['image_path']}")
            print(f"  Caption: {result['caption'][:150]}...")
        else:
            print(f"  ✗ Failed: {result['error']}")
    
    print(f"\nTest prompt: {test_prompt}")


def main():
    """Main function."""
    compare_models()


if __name__ == "__main__":
    main() 