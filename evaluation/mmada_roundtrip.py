#!/usr/bin/env python3
"""
MMaDA Roundtrip Generation Implementation

This module provides the MMaDA-specific implementation of roundtrip generation.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from PIL import Image

# Add MMaDA directory to path to import modules
mmada_path = Path(__file__).parent.parent / "MMaDA"
sys.path.append(str(mmada_path))

# Set environment variable for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from roundtrip_base import RoundtripGenerator
from models import MAGVITv2, get_mask_schedule, MMadaModelLM
from training.prompting_utils import UniversalPrompting
from training.utils import image_transform, get_config
from transformers import AutoTokenizer
from omegaconf import OmegaConf


class MMaDARoundtripGenerator(RoundtripGenerator):
    """MMaDA-specific implementation of roundtrip generation."""
    
    def _initialize_models(self):
        """Initialize MMaDA models."""
        print("Initializing MMaDA models...")
        
        # Set device
        self.device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        
        # Create a minimal config structure similar to the demo config
        self.config = OmegaConf.create({
            'model': {
                'vq_model': {
                    'type': 'magvitv2',
                    'vq_model_name': 'showlab/magvitv2'
                },
                'mmada': {
                    'pretrained_model_path': self.model_path,
                    'w_clip_vit': False,
                    'new_vocab_size': 134656,
                    'llm_vocab_size': 126464,
                    'codebook_size': 8192,
                    'num_vq_tokens': 1024,
                    'num_new_special_tokens': 0,
                    'tie_word_embeddings': False
                }
            },
            'dataset': {
                'preprocessing': {
                    'max_seq_length': 512
                }
            },
            'training': {
                'cond_dropout_prob': 0.1,
                'guidance_scale': 3.0,
                'generation_timesteps': 15,
                'generation_temperature': 1.0,
                'noise_type': 'mask',
                'mask_schedule': 'cosine'
            }
        })
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            padding_side="left"
        )
        
        # Initialize universal prompting
        self.uni_prompting = UniversalPrompting(
            self.tokenizer, 
            max_text_len=self.config.dataset.preprocessing.max_seq_length,
            special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
            ignore_id=-100, 
            cond_dropout_prob=self.config.training.cond_dropout_prob,
            use_reserved_token=True
        )
        
        # Load VQ model (MAGVITv2)
        self.vq_model = MAGVITv2.from_pretrained(self.config.model.vq_model.vq_model_name).to(self.device)
        self.vq_model.requires_grad_(False)
        self.vq_model.eval()
        
        # Load MMaDA model
        self.model = MMadaModelLM.from_pretrained(
            self.model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        # Get mask token ID
        self.mask_token_id = self.model.config.mask_token_id
        
        # Set generation parameters from config
        self.guidance_scale = self.config.training.guidance_scale
        self.generation_timesteps = self.config.training.generation_timesteps
        self.temperature = self.config.training.generation_temperature
        self.noise_type = self.config.training.noise_type
        self.mask_schedule = get_mask_schedule(self.config.training.mask_schedule)
        
        print("MMaDA models initialized successfully!")
    
    def generate_image_from_text(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """Generate image from text prompt using MMaDA."""
        if seed is not None:
            self.set_global_seed(seed)
        
        # Set batch size to 1 for single prompt
        batch_size = 1
        prompts = [prompt]
        
        # Create image tokens with mask tokens
        image_tokens = torch.ones(
            (batch_size, self.config.model.mmada.num_vq_tokens),
            dtype=torch.long, 
            device=self.device
        ) * self.mask_token_id
        
        # Prepare input with universal prompting
        input_ids, attention_mask = self.uni_prompting((prompts, image_tokens), 't2i_gen')
        
        # Prepare unconditional input for guidance
        if self.guidance_scale > 0:
            uncond_input_ids, uncond_attention_mask = self.uni_prompting(
                ([''] * batch_size, image_tokens), 't2i_gen'
            )
        else:
            uncond_input_ids = None
            uncond_attention_mask = None
        
        # Generate image tokens
        with torch.no_grad():
            gen_token_ids = self.model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                uncond_attention_mask=uncond_attention_mask,
                guidance_scale=self.guidance_scale,
                temperature=self.temperature,
                timesteps=self.generation_timesteps,
                noise_schedule=self.mask_schedule,
                noise_type=self.noise_type,
                seq_len=self.config.model.mmada.num_vq_tokens,
                uni_prompting=self.uni_prompting,
                config=self.config,  # Pass the config object
            )
        
        # Decode tokens to image
        gen_token_ids = torch.clamp(
            gen_token_ids, 
            max=self.config.model.mmada.codebook_size - 1, 
            min=0
        )
        images = self.vq_model.decode_code(gen_token_ids)
        
        # Convert to PIL image
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        
        # Return the first (and only) image
        return Image.fromarray(images[0])
    
    def generate_caption_from_image(self, image: Image.Image, prompt: str = "Describe this image in detail.", max_new_tokens: int = 128, steps: int = 128, block_length: int = 64, guidance_scale: float = 0.0) -> str:
        """Generate caption from image using MMaDA."""
        # Convert PIL image to tensor
        image_tensor = image_transform(image, resolution=512).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Get image tokens from VQ model
        image_tokens = self.vq_model.get_code(image_tensor) + len(self.uni_prompting.text_tokenizer)
        
        # Prepare input for multimodal understanding
        input_ids = self.uni_prompting.text_tokenizer([
            '<|start_header_id|>user<|end_header_id|>\n' + prompt + '<eot_id><|start_header_id|>assistant<|end_header_id|>\n'
        ])['input_ids']
        input_ids = torch.tensor(input_ids).to(self.device)
        
        # Construct the full input sequence
        input_ids = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(self.device),
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.device),
            image_tokens,
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.device),
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|sot|>']).to(self.device),
            input_ids
        ], dim=1).long()
        
        # Generate caption
        with torch.no_grad():
            output_ids = self.model.mmu_generate(
                input_ids, 
                max_new_tokens=max_new_tokens,  # Reduced from 1024 for faster generation
                steps=steps,  # Reduced from 512 for faster generation
                block_length=block_length,  # Reduced from 1024 for faster generation
                cfg_scale=guidance_scale
            )
        
        # Decode the generated text
        text = self.uni_prompting.text_tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return text[0].strip() 