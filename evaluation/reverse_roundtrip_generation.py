#!/usr/bin/env python3
"""
Reverse Roundtrip Generation Main Script

This script provides a unified interface for running reverse roundtrip generation
with any supported model type.
"""

import argparse
import sys
from pathlib import Path

from reverse_roundtrip_factory import create_reverse_roundtrip_generator
from reverse_roundtrip_base import create_reverse_parser


def main():
    """Main function for reverse roundtrip generation."""
    
    # Create parser
    parser = create_reverse_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image_dir:
        print("Error: --image_dir is required")
        sys.exit(1)
    
    if not Path(args.image_dir).exists():
        print(f"Error: Image directory {args.image_dir} does not exist")
        sys.exit(1)
    
    # Create generator
    try:
        print(f"Creating {args.model_type} reverse roundtrip generator...")
        generator = create_reverse_roundtrip_generator(
            model_type=args.model_type,
            model_path=args.model_path,
            device=args.device,
            seed=args.seed,
            config_path=args.config_path
        )
        print(f"✓ {args.model_type} reverse roundtrip generator created successfully!")
    except Exception as e:
        print(f"Error creating generator: {str(e)}")
        sys.exit(1)
    
    # Run reverse roundtrip generation
    try:
        print(f"\nStarting reverse roundtrip generation...")
        print(f"Model: {args.model_type}")
        print(f"Model path: {args.model_path}")
        print(f"Image directory: {args.image_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Device: {args.device}")
        print(f"Seed: {args.seed}")
        print(f"Seed offset: {args.seed_offset}")
        print(f"Range: {args.start_idx} to {args.end_idx if args.end_idx else 'end'}")
        
        results = generator.run_reverse_roundtrip_generation(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            seed_offset=args.seed_offset
        )
        
        print(f"\n✓ Reverse roundtrip generation completed successfully!")
        print(f"✓ Processed {len(results)} images")
        print(f"✓ Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during reverse roundtrip generation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 