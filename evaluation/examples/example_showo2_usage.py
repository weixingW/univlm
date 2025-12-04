#!/usr/bin/env python3
"""
Show-o2 Roundtrip Generation Example Usage

This script demonstrates how to use the Show-o2 roundtrip generation implementation.
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


def example_basic_usage():
    """Example of basic Show-o2 roundtrip usage."""
    print("=" * 60)
    print("Show-o2 Basic Roundtrip Generation Example")
    print("=" * 60)
    
    # Initialize the generator
    model_path = "showlab/show-o2-7b"  # Replace with your model path
    config_path = "../configs/showo2_config.yaml"  # Path to config file
    device = 0
    seed = 42
    
    generator = Showo2RoundtripGenerator(
        model_path=model_path,
        device=device,
        seed=seed,
        config_path=config_path
    )
    
    # Example prompts
    prompts = [
        "A serene mountain landscape with snow-capped peaks and a crystal clear lake",
        "A cozy coffee shop interior with warm lighting and people reading books",
        "A futuristic robot in a neon-lit cyberpunk city street"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- Example {i+1} ---")
        print(f"Original prompt: {prompt}")
        
        # Generate image from text
        print("Generating image...")
        generated_image = generator.generate_image_from_text(prompt, seed=seed + i)
        
        # Generate caption from image
        print("Generating caption...")
        caption = generator.generate_caption_from_image(generated_image)
        
        print(f"Generated caption: {caption}")
        
        # Save the image
        output_path = f"showo2_example_{i+1}.jpg"
        generated_image.save(output_path)
        print(f"Saved image to: {output_path}")


def example_custom_config():
    """Example with custom configuration."""
    print("\n" + "=" * 60)
    print("Show-o2 Custom Configuration Example")
    print("=" * 60)
    
    # Initialize the generator
    generator = Showo2RoundtripGenerator(
        model_path="showlab/show-o2-7b",
        device=0,
        seed=42,
        config_path="../configs/showo2_config.yaml"
    )
    
    # Customize configuration for faster generation
    generator.config.transport.num_inference_steps = 20  # Fewer steps = faster
    generator.config.guidance_scale = 2.0  # Lower guidance = more creative
    generator.config.dataset.preprocessing.resolution = 256  # Lower resolution
    
    print("Using custom configuration:")
    print(f"- Inference steps: {generator.config.transport.num_inference_steps}")
    print(f"- Guidance scale: {generator.config.guidance_scale}")
    print(f"- Resolution: {generator.config.dataset.preprocessing.resolution}")
    
    # Generate with custom config
    prompt = "A magical forest with glowing mushrooms and fairy lights"
    print(f"\nGenerating with prompt: {prompt}")
    
    generated_image = generator.generate_image_from_text(prompt)
    caption = generator.generate_caption_from_image(generated_image)
    
    print(f"Generated caption: {caption}")
    
    # Save the image
    generated_image.save("showo2_custom_config.jpg")
    print("Saved image to: showo2_custom_config.jpg")


def example_batch_processing():
    """Example of batch processing multiple prompts."""
    print("\n" + "=" * 60)
    print("Show-o2 Batch Processing Example")
    print("=" * 60)
    
    # Initialize the generator
    generator = Showo2RoundtripGenerator(
        model_path="showlab/show-o2-7b",
        device=0,
        seed=42,
        config_path="../configs/showo2_config.yaml"
    )
    
    # Batch of prompts
    prompts = [
        "A vintage car driving through a desert landscape",
        "A modern kitchen with stainless steel appliances",
        "A fantasy castle on a floating island in the sky",
        "A peaceful garden with blooming flowers and butterflies"
    ]
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}/{len(prompts)}: {prompt}")
        
        try:
            # Generate image
            image = generator.generate_image_from_text(prompt, seed=42 + i)
            
            # Generate caption
            caption = generator.generate_caption_from_image(image)
            
            # Save image
            image_path = f"showo2_batch_{i+1}.jpg"
            image.save(image_path)
            
            results.append({
                'prompt': prompt,
                'caption': caption,
                'image_path': image_path
            })
            
            print(f"✓ Completed: {image_path}")
            
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            results.append({
                'prompt': prompt,
                'caption': f"Error: {str(e)}",
                'image_path': None
            })
    
    # Print summary
    print(f"\n--- Batch Processing Summary ---")
    print(f"Total prompts: {len(prompts)}")
    print(f"Successful: {len([r for r in results if r['image_path']])}")
    print(f"Failed: {len([r for r in results if not r['image_path']])}")
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['prompt']}")
        print(f"   Caption: {result['caption']}")
        print(f"   Image: {result['image_path']}")


def example_roundtrip_analysis():
    """Example of analyzing roundtrip consistency."""
    print("\n" + "=" * 60)
    print("Show-o2 Roundtrip Analysis Example")
    print("=" * 60)
    
    # Initialize the generator
    generator = Showo2RoundtripGenerator(
        model_path="showlab/show-o2-7b",
        device=0,
        seed=42,
        config_path="../configs/showo2_config.yaml"
    )
    
    # Test prompts with different complexity levels
    test_cases = [
        {
            'name': 'Simple Object',
            'prompt': 'A red apple on a white table'
        },
        {
            'name': 'Complex Scene',
            'prompt': 'A bustling city street during rush hour with people walking, cars driving, and neon signs lighting up the night sky'
        },
        {
            'name': 'Abstract Concept',
            'prompt': 'The feeling of nostalgia represented through warm colors and vintage objects'
        }
    ]
    
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        print(f"Original: {case['prompt']}")
        
        # Generate image
        image = generator.generate_image_from_text(case['prompt'])
        
        # Generate caption
        caption = generator.generate_caption_from_image(image)
        print(f"Roundtrip: {caption}")
        
        # Simple similarity analysis (word overlap)
        original_words = set(case['prompt'].lower().split())
        caption_words = set(caption.lower().split())
        overlap = len(original_words.intersection(caption_words))
        total_unique = len(original_words.union(caption_words))
        similarity = overlap / total_unique if total_unique > 0 else 0
        
        print(f"Word overlap similarity: {similarity:.2f}")
        
        # Save image
        image.save(f"showo2_analysis_{case['name'].replace(' ', '_').lower()}.jpg")


if __name__ == "__main__":
    print("Show-o2 Roundtrip Generation Examples")
    print("Make sure you have the Show-o2 model available!")
    
    try:
        # Run examples
        example_basic_usage()
        example_custom_config()
        example_batch_processing()
        example_roundtrip_analysis()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        import traceback
        traceback.print_exc() 