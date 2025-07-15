#!/usr/bin/env python3
"""
Modular Roundtrip Generation Script

This script performs roundtrip generation using a modular architecture that supports
multiple models (BLIP3o, MMaDA, etc.).

Usage: python roundtrip_generation.py <model_path> --model_type <model_type>
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from roundtrip_base import create_parser
from roundtrip_factory import RoundtripGeneratorFactory


def main():
    """Main function for roundtrip generation."""
    parser = create_parser()
    args = parser.parse_args()
    config_path = args.config_path
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Create the generator using the factory
        print(f"Initializing {args.model_type.upper()} roundtrip generator...")
        generator = RoundtripGeneratorFactory.create_generator(
            model_type=args.model_type,
            model_path=args.model_path,
            device=args.device,
            seed=args.seed,
            config_path=args.config_path
        )
        
        if args.generate_from_captions:
            # Generate images from existing captions
            captions_dir = args.captions_dir if args.captions_dir else output_dir / "captions"
            caption_images_dir = output_dir / "caption_generated_images"
            
            generator.generate_images_from_captions(
                captions_dir=captions_dir,
                output_dir=caption_images_dir,
                start_idx=args.start_idx,
                end_idx=args.end_idx,
                seed_offset=args.seed + 10000
            )
        else:
            # Run full roundtrip generation
            generator.run_roundtrip_generation(
                prompts_file=args.prompts_file,
                output_dir=output_dir,
                start_idx=args.start_idx,
                end_idx=args.end_idx
            )
            
    except Exception as e:
        print(f"Error during roundtrip generation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 