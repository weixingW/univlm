#!/usr/bin/env python3
"""
Test script for Show-o Roundtrip Generator

This script tests the Show-o roundtrip generation functionality.
"""

import os
import sys
from pathlib import Path
import argparse

# Add the evaluation directory to the path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from showo_roundtrip import ShowoRoundtripGenerator
from roundtrip_factory import create_roundtrip_generator


def test_showo_roundtrip(config):
    """Test Show-o roundtrip generation with a simple prompt."""
    
    print("Testing Show-o Roundtrip Generator...")
    print(f"Model path: {config['model_path']}")
    print(f"Config path: {config['config_path']}")
    print(f"Test prompt: {config['test_prompt']}")
    
    try:
        # Create generator using factory
        print("\n1. Creating Show-o roundtrip generator...")
        generator = create_roundtrip_generator(
            model_type="showo",
            model_path=config["model_path"],
            device=config["device"],
            seed=config["seed"],
            config_path=config["config_path"]
        )
        print("✓ Generator created successfully!")
        
        # Test image generation
        print("\n2. Testing image generation...")
        generated_image = generator.generate_image_from_text(
            config["test_prompt"], 
            seed=config["seed"]
        )
        print("✓ Image generated successfully!")
        print(f"Image size: {generated_image.size}")
        
        # Test caption generation
        print("\n3. Testing caption generation...")
        generated_caption = generator.generate_caption_from_image(
            generated_image,
            "Describe this image in detail."
        )
        print("✓ Caption generated successfully!")
        print(f"Generated caption: {generated_caption}")
        
        # Save test results
        output_dir = Path("test_showo_output")
        output_dir.mkdir(exist_ok=True)
        
        image_path = output_dir / "test_generated_image.jpg"
        generated_image.save(image_path)
        print(f"✓ Test image saved to: {image_path}")
        
        caption_path = output_dir / "test_generated_caption.txt"
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(f"Original Prompt: {config['test_prompt']}\n\n")
            f.write(f"Generated Caption: {generated_caption}\n")
        print(f"✓ Test caption saved to: {caption_path}")
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_showo_direct(config):
    """Test Show-o roundtrip generator directly without factory."""
    
    print("Testing Show-o Roundtrip Generator (Direct)...")
    
    try:
        # Create generator directly
        print("\n1. Creating Show-o roundtrip generator directly...")
        generator = ShowoRoundtripGenerator(
            model_path=config["model_path"],
            device=config["device"],
            seed=config["seed"],
            config_path=config["config_path"]
        )
        print("✓ Generator created successfully!")
        
        # Test image generation
        print("\n2. Testing image generation...")
        generated_image = generator.generate_image_from_text(
            config["test_prompt"], 
            seed=config["seed"]
        )
        print("✓ Image generated successfully!")
        
        # Test caption generation
        print("\n3. Testing caption generation...")
        generated_caption = generator.generate_caption_from_image(
            generated_image,
            "Describe this image in detail."
        )
        print("✓ Caption generated successfully!")
        print(f"Generated caption: {generated_caption}")
        
        print("\n🎉 Direct test passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Direct test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description="Test Show-o roundtrip generator")
    parser.add_argument("--model_path", required=True, help="Path to Show-o model")
    parser.add_argument("--config_path", required=True, help="Path to Show-o config file")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--test_prompt", default="A beautiful sunset over the ocean", help="Test prompt for generation")
    parser.add_argument("--direct", action="store_true", help="Test direct instantiation instead of factory")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = {
        "model_path": args.model_path,
        "config_path": args.config_path,
        "device": args.device,
        "seed": 42,
        "test_prompt": args.test_prompt
    }
    
    print("=" * 60)
    print("Show-o Roundtrip Generator Test")
    print("=" * 60)
    print(f"Model path: {config['model_path']}")
    print(f"Config path: {config['config_path']}")
    print(f"Device: {config['device']}")
    print(f"Test prompt: {config['test_prompt']}")
    print("=" * 60)
    
    if args.direct:
        success = test_showo_direct(config)
    else:
        success = test_showo_roundtrip(config)
    
    if success:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 