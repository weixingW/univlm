#!/usr/bin/env python3
"""
Show-o Roundtrip Generation Implementation

This module provides the Show-o-specific implementation of roundtrip generation.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
import torch
import numpy as np
from PIL import Image

# Add Show-o directory to path to import modules
if __name__ == "__main__":
    # When run as script, use current directory
    current_dir = Path.cwd()
    showo_path = current_dir.parent / "Show-o"
else:
    # When imported as module, use __file__
    showo_path = Path(__file__).parent.parent / "Show-o"

# Add the Show-o path to sys.path
sys.path.insert(0, str(showo_path))


modules_to_clear = ['models', 'utils', 'transport', 'datasets', 'training']
for module in modules_to_clear:
    if module in sys.modules:
        del sys.modules[module]
    # Also clear any submodules
    for key in list(sys.modules.keys()):
        if key.startswith(f"{module}."):
            del sys.modules[key]

# Import from Show-o models and utilities
from models import Showo, MAGVITv2, CLIPVisionTower, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, create_attention_mask_for_mmu, create_attention_mask_for_mmu_vit
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer, CLIPImageProcessor

from roundtrip_base import RoundtripGenerator


def get_vq_model_class(model_type):
    """Get VQ model class based on type."""
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


class ShowoRoundtripGenerator(RoundtripGenerator):
    """Show-o-specific implementation of roundtrip generation."""
    
    def _initialize_models(self):
        """Initialize Show-o models."""
        print("Initializing Show-o models...")
        
        # Load configuration
        if self.config_path and os.path.exists(self.config_path):
            print(f"Loading configuration from: {self.config_path}")
            self.config = self._load_config_from_file(self.config_path)
        else:
            print("No config file provided or file not found, using minimal default config")
            raise ValueError("No config file provided or file not found")
        
        # Set device
        self.device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.showo.llm_model_path, 
            padding_side="left"
        )
        
        # Initialize universal prompting
        self.uni_prompting = UniversalPrompting(
            self.tokenizer, 
            max_text_len=self.config.dataset.preprocessing.max_seq_length,
            special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
            ignore_id=-100, 
            cond_dropout_prob=self.config.training.cond_dropout_prob
        )
        # Initialize VQ model
        vq_model_class = get_vq_model_class(self.config.model.vq_model.type)
        self.vq_model = vq_model_class.from_pretrained(self.config.model.vq_model.vq_model_name).to(self.device)
        self.vq_model.requires_grad_(False)
        self.vq_model.eval()
        
        
        # Initialize vision tower for MMU (if using CLIP ViT)
        if hasattr(self.config.model.showo, 'w_clip_vit') and self.config.model.showo.w_clip_vit:
            vision_tower_name = "openai/clip-vit-large-patch14-336"
            self.vision_tower = CLIPVisionTower(vision_tower_name).to(self.device)
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
        else:
            self.vision_tower = None
            self.clip_image_processor = None
        
        # Initialize Show-o model
        self.model = Showo.from_pretrained(self.config.model.showo.pretrained_model_path, low_cpu_mem_usage=False).to(self.device)
        self.model.eval()
        
        # Get mask token ID
        self.mask_token_id = self.model.config.mask_token_id
        
        # Set default generation parameters
        self.guidance_scale = getattr(self.config.training, 'guidance_scale', 0.0)
        self.generation_timesteps = getattr(self.config.training, 'generation_timesteps', 50)
        self.generation_temperature = getattr(self.config.training, 'generation_temperature', 1.0)
        self.max_new_tokens = getattr(self.config, 'max_new_tokens', 300)
        
        print("Show-o models initialized successfully!")
    
    def _load_config_from_file(self, config_path: str):
        """Load configuration from a YAML file."""
        from omegaconf import OmegaConf
        
        # Load the YAML config file
        config = OmegaConf.load(config_path)
        
        return config
    
    def generate_image_from_text(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """Generate image from text prompt using Show-o."""
        if seed is not None:
            self.set_global_seed(seed)
        
        # Prepare image tokens (all mask tokens for text-to-image generation)
        image_tokens = torch.ones(
            (1, self.config.model.showo.num_vq_tokens),
            dtype=torch.long, 
            device=self.device
        ) * self.mask_token_id
        
        # Prepare input using universal prompting
        input_ids, _ = self.uni_prompting(([prompt], image_tokens), 't2i_gen')
        
        # Apply classifier-free guidance if enabled
        if self.guidance_scale > 0:
            uncond_input_ids, _ = self.uni_prompting(([''], image_tokens), 't2i_gen')
            attention_mask = create_attention_mask_predict_next(
                torch.cat([input_ids, uncond_input_ids], dim=0),
                pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                rm_pad_in_image=True
            )
        else:
            attention_mask = create_attention_mask_predict_next(
                input_ids,
                pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                rm_pad_in_image=True
            )
            uncond_input_ids = None
        
        # Get mask schedule
        mask_schedule = get_mask_chedule(self.config.training.get("mask_schedule", "cosine"))
        
        # Generate image tokens
        with torch.no_grad():
            gen_token_ids = self.model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                guidance_scale=self.guidance_scale,
                temperature=self.generation_temperature,
                timesteps=self.generation_timesteps,
                noise_schedule=mask_schedule,
                noise_type=self.config.training.get("noise_type", "mask"),
                seq_len=self.config.model.showo.num_vq_tokens,
                uni_prompting=self.uni_prompting,
                config=self.config,
            )
        
        # Decode tokens to image
        gen_token_ids = torch.clamp(gen_token_ids, max=self.config.model.showo.codebook_size - 1, min=0)
        images = self.vq_model.decode_code(gen_token_ids)
        
        # Convert to PIL image
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(images[0])
        
        print(f"Image generated successfully!")
        return pil_image
    
    def generate_caption_from_image(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
        """Generate caption from image using Show-o."""
        
        # Ensure image is in the right format
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")
        
        # Preprocess image
        image = image.convert("RGB")
        image_tensor = image_transform(image, resolution=self.config.dataset.params.resolution).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)
        
        # Get image tokens
        image_tokens = self.vq_model.get_code(image_tensor) + len(self.uni_prompting.text_tokenizer)
        
        # Generate caption based on whether using CLIP ViT or not
        if self.vision_tower is not None and self.config.model.showo.w_clip_vit:
            return self._generate_caption_with_clip_vit(image, prompt)
        else:
            return self._generate_caption_without_clip_vit(image_tokens, prompt)
    
    def _generate_caption_with_clip_vit(self, image: Image.Image, prompt: str) -> str:
        """Generate caption using CLIP ViT vision tower."""
        from llava.llava import conversation as conversation_lib
        
        conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
        SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        SYSTEM_PROMPT_LEN = 28
        
        # Process image with CLIP
        pixel_values = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        
        # Prepare conversation
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        # Tokenize input
        input_ids_system = self.uni_prompting.text_tokenizer(
            SYSTEM_PROMPT, return_tensors="pt", padding="longest"
        ).input_ids.to(self.device)
        
        input_ids = self.uni_prompting.text_tokenizer(
            prompt_question.strip(), return_tensors="pt", padding="longest"
        ).input_ids.to(self.device)
        
        # Create input sequence
        input_ids_llava = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(self.device),
            input_ids_system,
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.device),
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.device),
            input_ids,
        ], dim=1).long()
        
        # Get image embeddings
        images_embeddings = self.vision_tower(pixel_values[None])
        images_embeddings = self.model.mm_projector(images_embeddings)
        
        # Get text embeddings
        text_embeddings = self.model.showo.model.embed_tokens(input_ids_llava)
        
        # Combine embeddings
        part1 = text_embeddings[:, :2 + SYSTEM_PROMPT_LEN, :]
        part2 = text_embeddings[:, 2 + SYSTEM_PROMPT_LEN:, :]
        input_embeddings = torch.cat((part1, images_embeddings, part2), dim=1)
        
        # Create attention mask
        attention_mask_llava = create_attention_mask_for_mmu_vit(
            input_embeddings, system_prompt_len=SYSTEM_PROMPT_LEN
        )
        
        # Generate response
        cont_toks_list = self.model.mmu_generate(
            input_embeddings=input_embeddings,
            attention_mask=attention_mask_llava[0].unsqueeze(0),
            max_new_tokens=self.max_new_tokens,
            top_k=1,
            eot_token=self.tokenizer.eos_token_id
        )
        
        cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]
        text = self.uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)
        
        return text[0].strip()
    
    def _generate_caption_without_clip_vit(self, image_tokens: torch.Tensor, prompt: str) -> str:
        """Generate caption without CLIP ViT vision tower."""
        
        # Prepare input
        input_ids = self.uni_prompting.text_tokenizer(['USER: \n' + prompt + ' ASSISTANT:'])['input_ids']
        input_ids = torch.tensor(input_ids).to(self.device)
        
        # Create input sequence
        input_ids = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(self.device),
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.device),
            image_tokens,
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.device),
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|sot|>']).to(self.device),
            input_ids
        ], dim=1).long()
        
        # Create attention mask
        attention_mask = create_attention_mask_for_mmu(
            input_ids.to(self.device),
            eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>'])
        )
        
        # Generate response
        cont_toks_list = self.model.mmu_generate(
            input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens, 
            top_k=1,
            eot_token=self.uni_prompting.sptids_dict['<|eot|>']
        )
        
        cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]
        text = self.uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)
        
        return text[0].strip() 