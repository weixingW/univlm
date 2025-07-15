#!/usr/bin/env python3
"""
Metadata Handling Example

This script demonstrates the new iterative metadata handling in the roundtrip generation system.
"""

import json
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from roundtrip_factory import create_roundtrip_generator


def demonstrate_metadata_handling():
    """Demonstrate how metadata is handled iteratively."""
    print("Metadata Handling Demonstration")
    print("=" * 50)
    
    # Example metadata structure
    example_metadata = {
        "metadata": {
            "total_prompts": 10,
            "successful_generations": 3,
            "start_idx": 0,
            "end_idx": 10,
            "model_path": "example/model/path",
            "created_at": "2024-01-01T10:00:00",
            "last_updated": "2024-01-01T10:15:00"
        },
        "results": [
            {
                "prompt_id": 0,
                "original_prompt": "A beautiful sunset over the ocean",
                "generated_caption": "A stunning sunset scene with vibrant colors",
                "image_path": "images/prompt_0000_sunset.jpg",
                "caption_path": "captions/prompt_0000_caption.txt",
                "processed_at": "2024-01-01T10:05:00"
            },
            {
                "prompt_id": 1,
                "original_prompt": "A cat sitting on a windowsill",
                "generated_caption": "A domestic cat perched on a windowsill looking outside",
                "image_path": "images/prompt_0001_cat.jpg",
                "caption_path": "captions/prompt_0001_caption.txt",
                "processed_at": "2024-01-01T10:10:00"
            },
            {
                "prompt_id": 2,
                "original_prompt": "A mountain landscape at dawn",
                "generated_caption": "Majestic mountain peaks bathed in the golden light of dawn",
                "image_path": "images/prompt_0002_mountain.jpg",
                "caption_path": "captions/prompt_0002_caption.txt",
                "processed_at": "2024-01-01T10:15:00"
            }
        ]
    }
    
    print("Example metadata structure:")
    print(json.dumps(example_metadata, indent=2))
    
    print("\n" + "=" * 50)
    print("Key Features of Iterative Metadata Handling:")
    print("=" * 50)
    
    print("1. **Progress Tracking**:")
    print(f"   - Total prompts: {example_metadata['metadata']['total_prompts']}")
    print(f"   - Successful generations: {example_metadata['metadata']['successful_generations']}")
    print(f"   - Progress: {example_metadata['metadata']['successful_generations']}/{example_metadata['metadata']['total_prompts']} ({example_metadata['metadata']['successful_generations']/example_metadata['metadata']['total_prompts']*100:.1f}%)")
    
    print("\n2. **Resume Capability**:")
    print("   - If process is interrupted, it can resume from the last successful generation")
    print("   - No need to restart from the beginning")
    
    print("\n3. **Real-time Updates**:")
    print("   - Metadata is updated after each successful generation")
    print("   - Can monitor progress in real-time")
    
    print("\n4. **Comprehensive Information**:")
    print("   - Each result includes processing timestamp")
    print("   - File paths for images and captions")
    print("   - Original prompts and generated captions")
    
    print("\n5. **Error Recovery**:")
    print("   - Failed generations don't affect successful ones")
    print("   - Can identify which prompts failed and retry them")


def show_metadata_file_structure():
    """Show the expected file structure with metadata."""
    print("\n" + "=" * 50)
    print("Expected File Structure with Metadata")
    print("=" * 50)
    
    structure = """
output_dir/
├── images/                           # Generated images
│   ├── prompt_0000_sunset.jpg
│   ├── prompt_0001_cat.jpg
│   └── prompt_0002_mountain.jpg
├── captions/                         # Generated captions
│   ├── prompt_0000_caption.txt
│   ├── prompt_0001_caption.txt
│   └── prompt_0002_caption.txt
└── roundtrip_metadata.json          # Iteratively updated metadata
    """
    
    print(structure)
    
    print("For caption-to-image generation:")
    caption_structure = """
caption_output_dir/
├── caption_0000_generated_*.jpg
├── caption_0001_generated_*.jpg
├── caption_0002_generated_*.jpg
└── caption_to_image_metadata.json   # Iteratively updated metadata
    """
    
    print(caption_structure)


def demonstrate_resume_capability():
    """Demonstrate how resume capability works."""
    print("\n" + "=" * 50)
    print("Resume Capability Demonstration")
    print("=" * 50)
    
    print("Scenario: Process interrupted after 3 generations")
    print("1. Original plan: Generate 10 images")
    print("2. Process interrupted after 3 successful generations")
    print("3. Metadata file contains 3 results")
    print("4. Restart process with same parameters")
    print("5. System loads existing metadata and continues from prompt 4")
    print("6. No duplicate work, seamless continuation")


def main():
    """Main function."""
    demonstrate_metadata_handling()
    show_metadata_file_structure()
    demonstrate_resume_capability()
    
    print("\n" + "=" * 50)
    print("Benefits of Iterative Metadata:")
    print("=" * 50)
    print("✓ Progress tracking and monitoring")
    print("✓ Resume capability after interruptions")
    print("✓ No duplicate work on restart")
    print("✓ Real-time progress updates")
    print("✓ Comprehensive result tracking")
    print("✓ Easy debugging and error identification")


if __name__ == "__main__":
    main() 