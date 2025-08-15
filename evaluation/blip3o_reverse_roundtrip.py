#!/usr/bin/env python3
"""
BLIP3o Reverse Roundtrip Generation Implementation

This module provides the BLIP3o-specific implementation of reverse roundtrip generation.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import torch
from PIL import Image

# Add BLIP3o directory to path to import modules
blip3o_path = Path(__file__).parent.parent / "BLIP3o"
sys.path.append(str(blip3o_path))

from diffusers import DiffusionPipeline
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from reverse_roundtrip_base import ReverseRoundtripGenerator
from blip3o.constants import *
from blip3o.conversation import conv_templates, SeparatorStyle
from blip3o.model.builder import load_pretrained_model
from blip3o.utils import disable_torch_init
from blip3o.mm_utils import get_model_name_from_path
from qwen_vl_utils import process_vision_info


class BLIP3oReverseRoundtripGenerator(ReverseRoundtripGenerator):
    """BLIP3o-specific implementation of reverse roundtrip generation."""
    
    def _initialize_models(self):
        """Initialize BLIP3o models."""
        print("Initializing BLIP3o models for reverse roundtrip...")
        disable_torch_init()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        
        # Load BLIP3o model
        model_path = os.path.expanduser(self.model_path)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.multi_model, self.context_len = load_pretrained_model(
            model_path, None, model_name
        )
         
        # Load diffusion pipeline
        diffusion_path = model_path + "/diffusion-decoder"
        self.pipe = DiffusionPipeline.from_pretrained(
            diffusion_path,
            custom_pipeline="pipeline_llava_gen",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="bf16",
            multimodal_encoder=self.multi_model,
            tokenizer=self.tokenizer,
            safety_checker=None
        )
        
        self.pipe.vae.to(f'cuda:{self.device}')
        self.pipe.unet.to(f'cuda:{self.device}')
        
        print("BLIP3o models initialized successfully for reverse roundtrip!")
    
    def _add_template(self, prompt: str) -> list:
        """Add conversation template to prompt."""
        conv = conv_templates['qwen'].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return [prompt]
    
    def generate_image_from_text(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """Generate image from text prompt using BLIP3o."""
        if seed is not None:
            self.set_global_seed(seed)
        
        template_prompt = f"Please generate image based on the following caption: {prompt}"
        gen_img = self.pipe(
            self._add_template(template_prompt)[0], 
            guidance_scale=3.0
        )
        return gen_img.image
    
    def generate_caption_from_image(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
        """Generate caption from image using BLIP3o."""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.multi_model.device)

        # Generate caption
        generated_ids = self.multi_model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0].strip()


if __name__ == "__main__":
    from reverse_roundtrip_base import create_reverse_parser
    
    parser = create_reverse_parser()
    args = parser.parse_args()
    
    # Create generator
    generator = BLIP3oReverseRoundtripGenerator(
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