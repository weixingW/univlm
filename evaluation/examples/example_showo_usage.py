#!/usr/bin/env python3
"""
Example Usage of Show-o Roundtrip Generator

This script demonstrates how to use the Show-o roundtrip generator
for text-to-image and image-to-text generation.
"""

import os
import sys
from pathlib import Path
import argparse
import json

# Add the evaluation directory to the path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from roundtrip_factory import create_roundtrip_generator


def example_single_roundtrip():
    """Example of a single roundtrip generation."""
    
    # Configuration - update these paths
    config = {
        "model_path": "/path/to/showo/model",  # Update this path
        "config_path": "/path/to/showo/config.yaml",  # Update this path
        "device": 0,
        "seed": 42,
        "prompt": "A majestic dragon flying over a medieval castle at sunset"
    }
    
    print("=" * 60)
    print("Show-o Single Roundtrip Generation Example")
    print("=" * 60)
    print(f"Prompt: {config['prompt']}")
    print("=" * 60)
    
    try:
        # Create generator
        print("1. Creating Show-o roundtrip generator...")
        generator = create_roundtrip_generator(
            model_type="showo",
            model_path=config["model_path"],
            device=config["device"],
            seed=config["seed"],
            config_path=config["config_path"]
        )
        print("✓ Generator created successfully!")
        
        # Step 1: Generate image from text
        print("\n2. Generating image from text...")
        generated_image = generator.generate_image_from_text(
            config["prompt"], 
            seed=config["seed"]
        )
        print("✓ Image generated successfully!")
        
        # Step 2: Generate caption from image
        print("\n3. Generating caption from image...")
        generated_caption = generator.generate_caption_from_image(
            generated_image,
            "Describe this image in detail."
        )
        print("✓ Caption generated successfully!")
        
        # Save results
        output_dir = Path("example_showo_output")
        output_dir.mkdir(exist_ok=True)
        
        # Save image
        image_path = output_dir / "example_generated_image.jpg"
        generated_image.save(image_path)
        print(f"✓ Image saved to: {image_path}")
        
        # Save caption
        caption_path = output_dir / "example_generated_caption.txt"
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(f"Original Prompt: {config['prompt']}\n\n")
            f.write(f"Generated Caption: {generated_caption}\n")
        print(f"✓ Caption saved to: {caption_path}")
        
        # Print results
        print("\n" + "=" * 60)
        print("ROUNDTRIP RESULTS")
        print("=" * 60)
        print(f"Original Prompt: {config['prompt']}")
        print(f"Generated Caption: {generated_caption}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def example_batch_roundtrip():
    """Example of batch roundtrip generation."""
    
    # Configuration - update these paths
    config = {
        "model_path": "/path/to/showo/model",  # Update this path
        "config_path": "/path/to/showo/config.yaml",  # Update this path
        "device": 0,
        "seed": 42,
        "prompts": [
            "A serene lake surrounded by mountains at dawn",
            "A futuristic city with flying cars and neon lights",
            "A cozy coffee shop with warm lighting and people reading books",
            "A magical forest with glowing mushrooms and fairy lights"
        ]
    }
    
    print("=" * 60)
    print("Show-o Batch Roundtrip Generation Example")
    print("=" * 60)
    print(f"Number of prompts: {len(config['prompts'])}")
    print("=" * 60)
    
    try:
        # Create generator
        print("1. Creating Show-o roundtrip generator...")
        generator = create_roundtrip_generator(
            model_type="showo",
            model_path=config["model_path"],
            device=config["device"],
            seed=config["seed"],
            config_path=config["config_path"]
        )
        print("✓ Generator created successfully!")
        
        # Create output directory
        output_dir = Path("example_showo_batch_output")
        output_dir.mkdir(exist_ok=True)
        
        results = []
        
        # Process each prompt
        for i, prompt in enumerate(config["prompts"]):
            print(f"\n--- Processing prompt {i+1}/{len(config['prompts'])} ---")
            print(f"Prompt: {prompt}")
            
            try:
                # Generate image
                generated_image = generator.generate_image_from_text(
                    prompt, 
                    seed=config["seed"] + i
                )
                
                # Generate caption
                generated_caption = generator.generate_caption_from_image(
                    generated_image,
                    "Describe this image in detail."
                )
                
                # Save image
                image_filename = f"prompt_{i+1:02d}_image.jpg"
                image_path = output_dir / image_filename
                generated_image.save(image_path)
                
                # Save caption
                caption_filename = f"prompt_{i+1:02d}_caption.txt"
                caption_path = output_dir / caption_filename
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(f"Original Prompt: {prompt}\n\n")
                    f.write(f"Generated Caption: {generated_caption}\n")
                
                # Store result
                result = {
                    "prompt_id": i + 1,
                    "original_prompt": prompt,
                    "generated_caption": generated_caption,
                    "image_path": str(image_path),
                    "caption_path": str(caption_path)
                }
                results.append(result)
                
                print(f"✓ Completed prompt {i+1}")
                
            except Exception as e:
                print(f"❌ Error processing prompt {i+1}: {str(e)}")
                continue
        
        # Save batch results
        results_path = output_dir / "batch_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Batch processing completed!")
        print(f"✓ Results saved to: {results_path}")
        print(f"✓ Successfully processed: {len(results)}/{len(config['prompts'])} prompts")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def example_roundtrip_from_file():
    """Example of roundtrip generation from a prompts file."""
    
    # Configuration - update these paths
    config = {
        "model_path": "/path/to/showo/model",  # Update this path
        "config_path": "/path/to/showo/config.yaml",  # Update this path
        "device": 0,
        "seed": 42,
        "prompts_file": "test_prompts.txt",  # Update this path
        "output_dir": "example_showo_file_output"
    }
    
    print("=" * 60)
    print("Show-o File-based Roundtrip Generation Example")
    print("=" * 60)
    print(f"Prompts file: {config['prompts_file']}")
    print(f"Output directory: {config['output_dir']}")
    print("=" * 60)
    
    try:
        # Create generator
        print("1. Creating Show-o roundtrip generator...")
        generator = create_roundtrip_generator(
            model_type="showo",
            model_path=config["model_path"],
            device=config["device"],
            seed=config["seed"],
            config_path=config["config_path"]
        )
        print("✓ Generator created successfully!")
        
        # Run roundtrip generation
        print("\n2. Running roundtrip generation from file...")
        results = generator.run_roundtrip_generation(
            prompts_file=config["prompts_file"],
            output_dir=config["output_dir"],
            start_idx=0,
            end_idx=None  # Process all prompts
        )
        
        print(f"\n✓ File-based roundtrip generation completed!")
        print(f"✓ Processed {len(results)} prompts")
        print(f"✓ Results saved to: {config['output_dir']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(description="Show-o roundtrip generator examples")
    parser.add_argument("--model_path", required=True, help="Path to Show-o model")
    parser.add_argument("--config_path", required=True, help="Path to Show-o config file")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--example", choices=["single", "batch", "file"], default="single",
                       help="Example to run: single, batch, or file")
    parser.add_argument("--prompts_file", default="test_prompts.txt", 
                       help="Path to prompts file (for file example)")
    
    args = parser.parse_args()
    
    print("Show-o Roundtrip Generator Examples")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Config path: {args.config_path}")
    print(f"Device: {args.device}")
    print(f"Example: {args.example}")
    print("=" * 60)
    
    # Update global config
    global_config = {
        "model_path": args.model_path,
        "config_path": args.config_path,
        "device": args.device,
        "seed": 42
    }
    
    success = False
    
    if args.example == "single":
        success = example_single_roundtrip()
    elif args.example == "batch":
        success = example_batch_roundtrip()
    elif args.example == "file":
        # Update config for file example
        global_config["prompts_file"] = args.prompts_file
        success = example_roundtrip_from_file()
    
    if success:
        print("\n✅ Example completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Example failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 