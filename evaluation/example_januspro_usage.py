#!/usr/bin/env python3
"""
Example usage of Janus Pro roundtrip generation.

This script demonstrates how to use the Janus Pro roundtrip generator
with the existing roundtrip infrastructure.
"""

import os
import sys
from pathlib import Path

# Add the evaluation directory to the path
sys.path.append(str(Path(__file__).parent))

from roundtrip_factory import create_roundtrip_generator


def example_januspro_roundtrip():
    """Example of using Janus Pro for roundtrip generation."""
    
    # Configuration
    model_path = "/path/to/your/januspro/model"  # Update this path
    prompts_file = "prompts.txt"  # Path to your prompts file
    output_dir = "januspro_roundtrip_results"
    device = 0
    seed = 42
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist!")
        print("Please update the model_path variable in this script.")
        print("You can download Janus Pro models from:")
        print("- Janus-Pro-1B: https://huggingface.co/deepseek-ai/Janus-Pro-1B")
        print("- Janus-Pro-7B: https://huggingface.co/deepseek-ai/Janus-Pro-7B")
        return
    
    # Check if prompts file exists
    if not os.path.exists(prompts_file):
        print(f"Prompts file {prompts_file} not found!")
        print("Please create a prompts.txt file or update the prompts_file variable.")
        return
    
    try:
        # Create Janus Pro roundtrip generator
        print("Creating Janus Pro roundtrip generator...")
        generator = create_roundtrip_generator(
            model_type="januspro",
            model_path=model_path,
            device=device,
            seed=seed
        )
        
        # Run full roundtrip generation
        print(f"\nRunning Janus Pro roundtrip generation...")
        print(f"Prompts file: {prompts_file}")
        print(f"Output directory: {output_dir}")
        
        results = generator.run_roundtrip_generation(
            prompts_file=prompts_file,
            output_dir=output_dir,
            start_idx=0,
            end_idx=None  # Process all prompts
        )
        
        print(f"\n✓ Janus Pro roundtrip generation completed!")
        print(f"Processed {len(results)} prompts")
        print(f"Results saved to: {output_dir}")
        
        # Example of generating images from existing captions
        print(f"\nExample: Generating images from existing captions...")
        captions_dir = os.path.join(output_dir, "captions")
        caption_output_dir = os.path.join(output_dir, "caption_to_image")
        
        if os.path.exists(captions_dir):
            caption_results = generator.generate_images_from_captions(
                captions_dir=captions_dir,
                output_dir=caption_output_dir,
                start_idx=0,
                end_idx=None,
                seed_offset=10000
            )
            print(f"✓ Generated {len(caption_results)} images from captions")
            print(f"Caption-to-image results saved to: {caption_output_dir}")
        else:
            print(f"Captions directory {captions_dir} not found. Skipping caption-to-image generation.")
        
    except Exception as e:
        print(f"Error during Janus Pro roundtrip generation: {str(e)}")
        import traceback
        traceback.print_exc()


def example_single_prompt():
    """Example of processing a single prompt with Janus Pro."""
    
    model_path = "/path/to/your/januspro/model"  # Update this path
    device = 0
    seed = 42
    
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist!")
        print("You can download Janus Pro models from:")
        print("- Janus-Pro-1B: https://huggingface.co/deepseek-ai/Janus-Pro-1B")
        print("- Janus-Pro-7B: https://huggingface.co/deepseek-ai/Janus-Pro-7B")
        return
    
    try:
        # Create generator
        generator = create_roundtrip_generator(
            model_type="januspro",
            model_path=model_path,
            device=device,
            seed=seed
        )
        
        # Test with a single prompt
        test_prompt = "A majestic dragon flying over a medieval castle at sunset"
        
        print(f"Original prompt: {test_prompt}")
        
        # Generate image
        print("Generating image...")
        generated_image = generator.generate_image_from_text(test_prompt)
        
        # Save image
        image_path = "januspro_single_test_image.jpg"
        generated_image.save(image_path)
        print(f"✓ Image saved to: {image_path}")
        
        # Generate caption
        print("Generating caption...")
        generated_caption = generator.generate_caption_from_image(generated_image)
        print(f"✓ Generated caption: {generated_caption}")
        
        # Save caption
        caption_path = "januspro_single_test_caption.txt"
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(f"Original Prompt: {test_prompt}\n\n")
            f.write(f"Generated Caption: {generated_caption}\n")
        print(f"✓ Caption saved to: {caption_path}")
        
    except Exception as e:
        print(f"Error during single prompt test: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Janus Pro Roundtrip Generation Examples")
    print("=" * 50)
    
    # Uncomment the example you want to run:
    
    # Example 1: Full roundtrip generation with prompts file
    # example_januspro_roundtrip()
    
    # Example 2: Single prompt test
    example_single_prompt() 