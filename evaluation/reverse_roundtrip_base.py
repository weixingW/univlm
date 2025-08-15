#!/usr/bin/env python3
"""
Base Reverse Roundtrip Generation Class

This module provides a base class for reverse roundtrip generation that can be extended
by different model implementations (BLIP3o, MMaDA, etc.).
"""

import os
import json
import argparse
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import torch
import random
import numpy as np
from PIL import Image
import re
from tqdm import tqdm

from roundtrip_base import RoundtripGenerator


class ReverseRoundtripGenerator(RoundtripGenerator):
    """Base class for reverse roundtrip generation models."""
    
    def __init__(self, model_path: str, device: int = 0, seed: int = 42, config_path: Optional[str] = None):
        """
        Initialize the reverse roundtrip generator.
        
        Args:
            model_path: Path to the model
            device: CUDA device ID
            seed: Random seed for reproducibility
            config_path: Optional path to configuration file
        """
        super().__init__(model_path, device, seed, config_path)
    
    def get_image_files(self, image_dir: str, extensions: List[str] = None) -> List[Path]:
        """Get all image files from a directory."""
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory {image_dir} not found!")
        
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    def run_reverse_roundtrip_generation(self, 
                                        image_dir: str,
                                        output_dir: str,
                                        start_idx: int = 0,
                                        end_idx: Optional[int] = None,
                                        seed_offset: int = 20000) -> List[Dict[str, Any]]:
        """
        Run reverse roundtrip generation: Image -> Caption -> Image.
        
        Args:
            image_dir: Directory containing input images
            output_dir: Output directory
            start_idx: Starting image index
            end_idx: Ending image index
            seed_offset: Seed offset for generation
            
        Returns:
            List of results dictionaries
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        captions_dir = output_dir / "captions"
        reconstructed_images_dir = output_dir / "reconstructed_images"
        captions_dir.mkdir(exist_ok=True)
        reconstructed_images_dir.mkdir(exist_ok=True)
        
        # Initialize metadata JSON file
        metadata_path = output_dir / "reverse_roundtrip_metadata.json"
        metadata = {
            "metadata": {
                "total_images": 0,
                "successful_generations": 0,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "seed_offset": seed_offset,
                "model_path": self.model_path,
                "image_dir": str(image_dir),
                "created_at": None,
                "last_updated": None
            },
            "results": []
        }
        
        # Load existing metadata if file exists
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"Loaded existing metadata from {metadata_path}")
            except Exception as e:
                print(f"Warning: Could not load existing metadata: {e}")
        
        # Get all image files
        image_files = self.get_image_files(image_dir)
        print(f"Found {len(image_files)} images in {image_dir}")
        
        # Set range for processing
        end_idx = end_idx if end_idx is not None else len(image_files)
        images_to_process = image_files[start_idx:end_idx]
        
        # Update metadata
        metadata["metadata"]["total_images"] = len(images_to_process)
        metadata["metadata"]["end_idx"] = end_idx
        metadata["metadata"]["created_at"] = metadata["metadata"].get("created_at") or str(Path().cwd())
        metadata["metadata"]["last_updated"] = str(Path().cwd())
        
        print(f"Processing images {start_idx} to {end_idx-1}...")
        
        for idx, image_file in enumerate(tqdm(images_to_process, desc="Processing images")):
            try:
                actual_idx = start_idx + idx
                
                # Load image
                print(f"\nProcessing image {actual_idx}: {image_file.name}...")
                original_image = Image.open(image_file).convert('RGB')
                
                # Generate caption from image
                generated_caption = self.generate_caption_from_image(
                    original_image,
                    "Describe this image in detail."
                )
                
                # Generate reconstructed image from caption
                reconstructed_image = self.generate_image_from_text(
                    generated_caption, 
                    seed=seed_offset + actual_idx
                )
                
                # Save caption
                caption_filename = f"image_{actual_idx:04d}_{image_file.stem}_caption.txt"
                caption_path = captions_dir / caption_filename
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(f"Original Image: {image_file.name}\n\n")
                    f.write(f"Generated Caption: {generated_caption}\n")
                
                # Save reconstructed image
                sanitized_caption = self.sanitize_filename(generated_caption)
                reconstructed_image_filename = f"image_{actual_idx:04d}_{image_file.stem}_reconstructed_{sanitized_caption[:50]}.jpg"
                reconstructed_image_path = reconstructed_images_dir / reconstructed_image_filename
                reconstructed_image.save(reconstructed_image_path)
                
                # Create result entry
                result = {
                    "image_id": actual_idx,
                    "original_image_path": str(image_file),
                    "original_image_name": image_file.name,
                    "generated_caption": generated_caption,
                    "reconstructed_image_path": str(reconstructed_image_path),
                    "caption_path": str(caption_path),
                    "seed": seed_offset + actual_idx,
                    "processed_at": str(Path().cwd())
                }
                
                # Add to results and update metadata
                metadata["results"].append(result)
                metadata["metadata"]["successful_generations"] = len(metadata["results"])
                metadata["metadata"]["last_updated"] = str(Path().cwd())
                
                # Save updated metadata after each successful generation
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                print(f"✓ Saved caption: {caption_filename}")
                print(f"✓ Saved reconstructed image: {reconstructed_image_filename}")
                print(f"✓ Updated metadata: {metadata_path}")
                print(f"Generated caption: {generated_caption[:100]}...")
                
            except Exception as e:
                print(f"Error processing image {actual_idx}: {str(e)}")
                continue
        
        print(f"\nReverse roundtrip generation completed!")
        print(f"Processed {metadata['metadata']['successful_generations']} out of {metadata['metadata']['total_images']} images")
        print(f"Results saved to: {output_dir}")
        print(f"Metadata saved to: {metadata_path}")
        
        return metadata["results"]


def create_reverse_parser() -> argparse.ArgumentParser:
    """Create argument parser for reverse roundtrip generation."""
    parser = argparse.ArgumentParser(description="Reverse roundtrip generation with modular model support")
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("--model_type", required=True, choices=["mmada","blip3o", "mmada", "emu3", "omnigen2", "januspro", "showo2", "showo"], 
                       help="Type of model to use")
    parser.add_argument("--config_path", default=None, help="Path to configuration file")
    parser.add_argument("--image_dir", required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", default="reverse_roundtrip_results", help="Output directory")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting image index")
    parser.add_argument("--end_idx", type=int, default=None, help="Ending image index")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seed_offset", type=int, default=20000, help="Seed offset for generation")
    
    return parser 