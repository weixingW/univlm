#!/usr/bin/env python3
"""
Example EMU3 Roundtrip Generation Usage

This script demonstrates how to use the EMU3 roundtrip generation system.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from roundtrip_factory import create_roundtrip_generator


def example_emu3_basic_usage():
    """Example of basic EMU3 roundtrip generation usage."""
    print("=== EMU3 Basic Roundtrip Generation Example ===")
    
    # Create an EMU3 generator
    try:
        # You can use different EMU3 model variants:
        # - "BAAI/Emu3-Stage1": Base model for both generation and understanding
        # - "BAAI/Emu3-Gen": Specialized for image generation
        # - "BAAI/Emu3-Chat": Specialized for vision-language understanding
        
        generator = create_roundtrip_generator(
            model_type="emu3",
            model_path="BAAI/Emu3-Stage1",  # Recommended for roundtrip
            device=0,
            seed=42
        )
        
        print("✓ EMU3 generator created successfully!")
        
        # Example of running roundtrip generation
        # results = generator.run_roundtrip_generation(
        #     prompts_file="prompts.txt",
        #     output_dir="emu3_results",
        #     start_idx=0,
        #     end_idx=5
        # )
        # print(f"Generated {len(results)} results")
        
    except Exception as e:
        print(f"Error creating EMU3 generator: {e}")
        print("This is expected if the model path doesn't exist or dependencies are missing.")


def example_emu3_model_variants():
    """Example of different EMU3 model variants."""
    print("\n=== EMU3 Model Variants Example ===")
    
    model_variants = [
        {
            "name": "Emu3-Stage1",
            "path": "BAAI/Emu3-Stage1",
            "description": "Base model for both generation and understanding"
        },
        {
            "name": "Emu3-Gen", 
            "path": "BAAI/Emu3-Gen",
            "description": "Specialized for image generation"
        },
        {
            "name": "Emu3-Chat",
            "path": "BAAI/Emu3-Chat", 
            "description": "Specialized for vision-language understanding"
        }
    ]
    
    for variant in model_variants:
        print(f"\n{variant['name']}:")
        print(f"  Path: {variant['path']}")
        print(f"  Description: {variant['description']}")
        
        try:
            generator = create_roundtrip_generator(
                model_type="emu3",
                model_path=variant["path"],
                device=0,
                seed=42
            )
            print(f"  ✓ Initialization: Successful")
        except Exception as e:
            print(f"  ✗ Initialization: Failed - {e}")


def example_emu3_caption_to_image():
    """Example of generating images from existing captions with EMU3."""
    print("\n=== EMU3 Caption to Image Generation Example ===")
    
    try:
        generator = create_roundtrip_generator(
            model_type="emu3",
            model_path="BAAI/Emu3-Stage1",
            device=0,
            seed=42
        )
        
        print("✓ EMU3 generator created successfully!")
        
        # Example of generating images from captions
        # results = generator.generate_images_from_captions(
        #     captions_dir="captions",
        #     output_dir="emu3_caption_images",
        #     start_idx=0,
        #     end_idx=3,
        #     seed_offset=10000
        # )
        # print(f"Generated {len(results)} images from captions")
        
    except Exception as e:
        print(f"Error creating EMU3 generator: {e}")
        print("This is expected if the model path doesn't exist.")


def example_emu3_custom_prompts():
    """Example of using custom prompts with EMU3."""
    print("\n=== EMU3 Custom Prompts Example ===")
    
    try:
        generator = create_roundtrip_generator(
            model_type="emu3",
            model_path="BAAI/Emu3-Stage1",
            device=0,
            seed=42
        )
        
        print("✓ EMU3 generator created successfully!")
        
        # Example custom prompts
        custom_prompts = [
            "A majestic dragon soaring over a medieval castle",
            "A serene Japanese garden with cherry blossoms",
            "A futuristic cityscape with flying cars",
            "A cozy coffee shop on a rainy day",
            "A magical forest with glowing mushrooms"
        ]
        
        print("Example custom prompts for EMU3:")
        for i, prompt in enumerate(custom_prompts):
            print(f"  {i+1}. {prompt}")
        
        # Example of processing custom prompts
        # for i, prompt in enumerate(custom_prompts):
        #     print(f"\nProcessing prompt {i+1}: {prompt}")
        #     image = generator.generate_image_from_text(prompt, seed=42+i)
        #     caption = generator.generate_caption_from_image(image)
        #     print(f"Generated caption: {caption[:100]}...")
        
    except Exception as e:
        print(f"Error creating EMU3 generator: {e}")
        print("This is expected if the model path doesn't exist.")


def main():
    """Run all EMU3 examples."""
    print("EMU3 Roundtrip Generation Examples")
    print("=" * 50)
    
    example_emu3_basic_usage()
    example_emu3_model_variants()
    example_emu3_caption_to_image()
    example_emu3_custom_prompts()
    
    print("\n" + "=" * 50)
    print("EMU3 Examples completed!")
    print("\nTo run actual EMU3 roundtrip generation:")
    print("python roundtrip_generation.py BAAI/Emu3-Stage1 --model_type emu3 --prompts_file prompts.txt")
    print("\nTo test EMU3 functionality:")
    print("python test_emu3.py")


if __name__ == "__main__":
    main() 