#!/usr/bin/env python3
"""
Base Roundtrip Generation Class

This module provides a base class for roundtrip generation that can be extended
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


class RoundtripGenerator(ABC):
    """Base class for roundtrip generation models."""
    
    def __init__(self, model_path: str, device: int = 0, seed: int = 42, config_path: Optional[str] = None):
        """
        Initialize the roundtrip generator.
        
        Args:
            model_path: Path to the model
            device: CUDA device ID
            seed: Random seed for reproducibility
            config_path: Optional path to configuration file
        """
        self.model_path = model_path
        self.device = device
        self.seed = seed
        self.config_path = config_path
        self.set_global_seed(seed)
        
        # Initialize models
        self._initialize_models()
    
    @abstractmethod
    def _initialize_models(self):
        """Initialize the specific models for this implementation."""
        pass
    
    @abstractmethod
    def generate_image_from_text(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """Generate image from text prompt."""
        pass
    
    @abstractmethod
    def generate_caption_from_image(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
        """Generate caption from image."""
        pass
    
    def set_global_seed(self, seed: int = 42):
        """Set global seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by removing/replacing invalid characters."""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limit length
        if len(filename) > 100:
            filename = filename[:100]
        return filename
    
    def parse_prompts_file(self, prompts_file_path: str) -> List[str]:
        """Parse prompts from the prompts.txt file."""
        prompts = []
        current_prompt = ""
        
        with open(prompts_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("=== Prompt") and line.endswith("==="):
                    if current_prompt:
                        prompts.append(current_prompt.strip())
                    current_prompt = ""
                elif line and not line.startswith("===") and not line.startswith("="):
                    current_prompt += line + " "
        
        # Add the last prompt
        if current_prompt:
            prompts.append(current_prompt.strip())
        
        return prompts
    
    def parse_caption_file(self, caption_file_path: str) -> Tuple[str, str]:
        """Parse caption from a caption file."""
        with open(caption_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract both original prompt and generated caption
        lines = content.split('\n')
        original_prompt = ""
        generated_caption = ""
        found_caption = False
        
        for line in lines:
            # Ensure line is a string before calling startswith
            if not isinstance(line, str):
                print(f"Warning: Non-string line in caption file: {type(line)}")
                continue
                
            if line.startswith("Original Prompt:"):
                original_prompt = line.replace("Original Prompt:", "").strip()
            elif line.startswith("Generated Caption:"):
                # Start collecting caption content
                generated_caption = line.replace("Generated Caption:", "").strip()
                found_caption = True
            elif found_caption and line.strip():
                # Continue collecting subsequent lines if they're not empty
                generated_caption += " " + line.strip()
        
        return original_prompt, generated_caption
    
    def run_roundtrip_generation(self, 
                                prompts_file: str,
                                output_dir: str,
                                start_idx: int = 0,
                                end_idx: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run full roundtrip generation: Text -> Image -> Caption.
        
        Args:
            prompts_file: Path to prompts file
            output_dir: Output directory
            start_idx: Starting prompt index
            end_idx: Ending prompt index
            
        Returns:
            List of results dictionaries
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        images_dir = output_dir / "images"
        captions_dir = output_dir / "captions"
        images_dir.mkdir(exist_ok=True)
        captions_dir.mkdir(exist_ok=True)
        
        # Initialize metadata JSON file
        metadata_path = output_dir / "roundtrip_metadata.json"
        metadata = {
            "metadata": {
                "total_prompts": 0,
                "successful_generations": 0,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "model_path": self.model_path,
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
        
        # Parse prompts
        prompts_file_path = Path(prompts_file)
        if not prompts_file_path.exists():
            raise FileNotFoundError(f"Prompts file {prompts_file_path} not found!")
        
        prompts = self.parse_prompts_file(prompts_file_path)
        print(f"Loaded {len(prompts)} prompts from {prompts_file_path}")
        
        # Set range for processing
        end_idx = end_idx if end_idx is not None else len(prompts)
        prompts_to_process = prompts[start_idx:end_idx]
        
        # Update metadata
        metadata["metadata"]["total_prompts"] = len(prompts_to_process)
        metadata["metadata"]["end_idx"] = end_idx
        metadata["metadata"]["created_at"] = metadata["metadata"].get("created_at") or str(Path().cwd())
        metadata["metadata"]["last_updated"] = str(Path().cwd())
        
        print(f"Processing prompts {start_idx} to {end_idx-1}...")
        
        for idx, prompt in enumerate(tqdm(prompts_to_process, desc="Processing prompts")):
            try:
                actual_idx = start_idx + idx
                
                # Generate image from text
                print(f"\nProcessing prompt {actual_idx}: {prompt[:100]}...")
                
                # Generate image
                generated_image = self.generate_image_from_text(
                    prompt, 
                    seed=self.seed + actual_idx
                )
                
                # Generate caption from image
                generated_caption = self.generate_caption_from_image(
                    generated_image,
                    "Describe this image in detail."
                )
                
                # Save image
                sanitized_prompt = self.sanitize_filename(prompt)
                image_filename = f"prompt_{actual_idx:04d}_{sanitized_prompt[:50]}.jpg"
                image_path = images_dir / image_filename
                generated_image.save(image_path)
                
                # Save caption
                caption_filename = f"prompt_{actual_idx:04d}_caption.txt"
                caption_path = captions_dir / caption_filename
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(f"Original Prompt: {prompt}\n\n")
                    f.write(f"Generated Caption: {generated_caption}\n")
                
                # Create result entry
                result = {
                    "prompt_id": actual_idx,
                    "original_prompt": prompt,
                    "generated_caption": generated_caption,
                    "image_path": str(image_path),
                    "caption_path": str(caption_path),
                    "processed_at": str(Path().cwd())
                }
                
                # Add to results and update metadata
                metadata["results"].append(result)
                metadata["metadata"]["successful_generations"] = len(metadata["results"])
                metadata["metadata"]["last_updated"] = str(Path().cwd())
                
                # Save updated metadata after each successful generation
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                print(f"✓ Saved image: {image_filename}")
                print(f"✓ Saved caption: {caption_filename}")
                print(f"✓ Updated metadata: {metadata_path}")
                print(f"Generated caption: {generated_caption[:100]}...")
                
            except Exception as e:
                print(f"Error processing prompt {actual_idx}: {str(e)}")
                continue
        
        print(f"\nRoundtrip generation completed!")
        print(f"Processed {metadata['metadata']['successful_generations']} out of {metadata['metadata']['total_prompts']} prompts")
        print(f"Results saved to: {output_dir}")
        print(f"Metadata saved to: {metadata_path}")
        
        return metadata["results"]
    
    def generate_images_from_captions(self, 
                                    captions_dir: str,
                                    output_dir: str,
                                    start_idx: int = 0,
                                    end_idx: Optional[int] = None,
                                    seed_offset: int = 10000) -> List[Dict[str, Any]]:
        """
        Generate images from existing captions.
        
        Args:
            captions_dir: Directory containing caption files
            output_dir: Output directory for generated images
            start_idx: Starting caption index
            end_idx: Ending caption index
            seed_offset: Seed offset for generation
            
        Returns:
            List of results dictionaries
        """
        captions_dir = Path(captions_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Initialize metadata JSON file
        metadata_path = output_dir / "caption_to_image_metadata.json"
        metadata = {
            "metadata": {
                "total_captions": 0,
                "successful_generations": 0,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "seed_offset": seed_offset,
                "model_path": self.model_path,
                "captions_dir": str(captions_dir),
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
        
        # Get all caption files
        caption_files = sorted([f for f in captions_dir.glob("prompt_*_caption.txt")])
        
        if end_idx is None:
            end_idx = len(caption_files)
        
        caption_files = caption_files[start_idx:end_idx]
        
        # Update metadata
        metadata["metadata"]["total_captions"] = len(caption_files)
        metadata["metadata"]["end_idx"] = end_idx
        metadata["metadata"]["created_at"] = metadata["metadata"].get("created_at") or str(Path().cwd())
        metadata["metadata"]["last_updated"] = str(Path().cwd())
        
        print(f"Generating images from {len(caption_files)} captions...")
        
        for idx, caption_file in enumerate(tqdm(caption_files, desc="Generating images from captions")):
            try:
                actual_idx = start_idx + idx
                
                # Extract prompt ID from filename
                prompt_id = caption_file.stem.split('_')[1]
                
                # Parse caption
                original_prompt, generated_caption = self.parse_caption_file(caption_file)
                
                if not original_prompt or not generated_caption:
                    print(f"Warning: No prompt or caption found in {caption_file}")
                    continue
                
                print(f"\nProcessing caption {actual_idx} (prompt {prompt_id}): {generated_caption[:100]}...")
                
                # Generate image from caption
                generated_image = self.generate_image_from_text(
                    generated_caption, 
                    seed=seed_offset + actual_idx
                )
                
                # Save image
                sanitized_caption = self.sanitize_filename(generated_caption)
                image_filename = f"caption_{prompt_id}_generated_{sanitized_caption[:50]}.jpg"
                image_path = output_dir / image_filename
                generated_image.save(image_path)
                
                # Create result entry
                result = {
                    "prompt_id": prompt_id,
                    "caption_file": str(caption_file),
                    "original_prompt": original_prompt,
                    "generated_caption": generated_caption,
                    "generated_image_path": str(image_path),
                    "seed": seed_offset + int(prompt_id),
                    "processed_at": str(Path().cwd())
                }
                
                # Add to results and update metadata
                metadata["results"].append(result)
                metadata["metadata"]["successful_generations"] = len(metadata["results"])
                metadata["metadata"]["last_updated"] = str(Path().cwd())
                
                # Save updated metadata after each successful generation
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                print(f"✓ Saved image: {image_filename}")
                print(f"✓ Updated metadata: {metadata_path}")
                
            except Exception as e:
                print(f"Error processing caption {actual_idx}: {str(e)}")
                continue
        
        print(f"\nCaption-to-image generation completed!")
        print(f"Processed {metadata['metadata']['successful_generations']} out of {metadata['metadata']['total_captions']} captions")
        print(f"Results saved to: {output_dir}")
        print(f"Metadata saved to: {metadata_path}")
        
        return metadata["results"]


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for roundtrip generation."""
    parser = argparse.ArgumentParser(description="Roundtrip generation with modular model support")
    parser.add_argument("model_path", help="Path to the model")
    parser.add_argument("--model_type", required=True, choices=["mmada","blip3o", "mmada", "emu3", "omnigen2", "januspro", "showo2", "showo"], 
                       help="Type of model to use")
    parser.add_argument("--config_path", default=None, help="Path to configuration file")
    parser.add_argument("--prompts_file", default="prompts.txt", help="Path to prompts file")
    parser.add_argument("--output_dir", default="roundtrip_results", help="Output directory")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting prompt index")
    parser.add_argument("--end_idx", type=int, default=None, help="Ending prompt index")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--generate_from_captions", action="store_true", 
                       help="Generate images from existing captions instead of running full roundtrip")
    parser.add_argument("--captions_dir", default=None, 
                       help="Directory containing caption files (default: output_dir/captions)")
    
    return parser 