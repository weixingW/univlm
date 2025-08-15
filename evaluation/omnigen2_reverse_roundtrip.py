#!/usr/bin/env python3
"""
OmniGen2 Reverse Roundtrip Generation Implementation

This module provides the OmniGen2-specific implementation of reverse roundtrip generation.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
import torch
from PIL import Image
from accelerate import Accelerator

# Add OmniGen2 directory to path to import modules
omnigen2_path = Path(__file__).parent.parent / "OmniGen2"
sys.path.append(str(omnigen2_path))

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.pipelines.omnigen2.pipeline_omnigen2_chat import OmniGen2ChatPipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

from reverse_roundtrip_base import ReverseRoundtripGenerator


class OmniGen2ReverseRoundtripGenerator(ReverseRoundtripGenerator):
    """OmniGen2-specific implementation of reverse roundtrip generation."""
    
    def _initialize_models(self):
        """Initialize OmniGen2 models."""
        print("Initializing OmniGen2 models for reverse roundtrip...")
        
        # Initialize accelerator
        self.accelerator = Accelerator(mixed_precision='bf16')
        
        # Set weight dtype
        self.weight_dtype = torch.bfloat16
        
        # Load image generation pipeline
        self.image_pipeline = OmniGen2Pipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.weight_dtype,
            trust_remote_code=True,
        )
        
        # Load transformer for image generation
        self.image_pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            self.model_path,
            subfolder="transformer",
            torch_dtype=self.weight_dtype,
        )
        
        # Load chat pipeline for text generation
        self.chat_pipeline = OmniGen2ChatPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.weight_dtype,
            trust_remote_code=True,
        )
        
        # Load transformer for chat pipeline
        self.chat_pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            self.model_path,
            subfolder="transformer",
            torch_dtype=self.weight_dtype,
        )
        
        # Move pipelines to device
        self.image_pipeline = self.image_pipeline.to(f'cuda:{self.device}')
        self.chat_pipeline = self.chat_pipeline.to(f'cuda:{self.device}')
        
        # Set default parameters
        self.default_negative_prompt = "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"
        
        print("OmniGen2 models initialized successfully for reverse roundtrip!")
    
    def generate_image_from_text(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """Generate image from text prompt using OmniGen2."""
        if seed is not None:
            self.set_global_seed(seed)
        
        generator = torch.Generator(device=f'cuda:{self.device}').manual_seed(seed if seed is not None else self.seed)
        
        results = self.image_pipeline(
            prompt=prompt,
            input_images=None,  # No input images for text-to-image generation
            width=1024,
            height=1024,
            num_inference_steps=50,
            max_sequence_length=1024,
            text_guidance_scale=5.0,
            image_guidance_scale=2.0,
            cfg_range=(0.0, 1.0),
            negative_prompt=self.default_negative_prompt,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil",
        )
        
        # Return the first (and only) generated image
        return results.images[0]
    
    def generate_caption_from_image(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
        """Generate caption from image using OmniGen2."""
        # For OmniGen2, we need to use the chat pipeline for text generation
        # The chat pipeline can handle both image and text inputs
        
        results = self.chat_pipeline(
            prompt=prompt,
            input_images=[image.convert("RGB")],  # Pass the image as input
            width=1024,
            height=1024,
            num_inference_steps=50,
            max_sequence_length=1024,
            text_guidance_scale=5.0,
            image_guidance_scale=2.0,
            cfg_range=(0.0, 1.0),
            negative_prompt=self.default_negative_prompt,
            num_images_per_prompt=1,
            generator=torch.Generator(device=f'cuda:{self.device}').manual_seed(self.seed),
            output_type="pil",
        )
        
        # Return the generated text
        return results.text.strip()


if __name__ == "__main__":
    from reverse_roundtrip_base import create_reverse_parser
    
    parser = create_reverse_parser()
    args = parser.parse_args()
    
    # Create generator
    generator = OmniGen2ReverseRoundtripGenerator(
        model_path=args.model_path,
        device=args.device,
        seed=args.seed,
        config_path=args.config_path
    )
    
    # Run reverse roundtrip generation
    results = generator.run_reverse_roundtrip_generation(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        seed_offset=args.seed_offset
    )
    
    print(f"Reverse roundtrip generation completed with {len(results)} results!") 