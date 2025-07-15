#!/usr/bin/env python3
"""
Show-o2 Roundtrip Generation Implementation

This module provides the Show-o2-specific implementation of roundtrip generation.
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

showo2_path = showo_path / "show-o2"

# Add only the show-o2 path to avoid conflicts with parent Show-o models
# Insert at the beginning to ensure it takes precedence
sys.path.insert(0, str(showo2_path))

# Clear any cached imports to force re-import from the correct path
modules_to_clear = ['models', 'utils', 'transport', 'datasets']
for module in modules_to_clear:
    if module in sys.modules:
        del sys.modules[module]
    # Also clear any submodules
    for key in list(sys.modules.keys()):
        if key.startswith(f"{module}."):
            del sys.modules[key]

# Import from show-o2 models (these will be found in the show-o2 directory)
from models import Showo2Qwen2_5, omni_attn_mask_naive, WanVAE
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import get_config, denorm, get_hyper_params, path_to_llm_name, load_state_dict
from transport import Sampler, create_transport
from datasets.utils import image_transform

from roundtrip_base import RoundtripGenerator


class Showo2RoundtripGenerator(RoundtripGenerator):
    """Show-o2-specific implementation of roundtrip generation."""
    
    def _initialize_models(self):
        """Initialize Show-o2 models."""
        print("Initializing Show-o2 models...")
        
        
        # Load configuration
        if self.config_path and os.path.exists(self.config_path):
            print(f"Loading configuration from: {self.config_path}")
            self.config = self._load_config_from_file(self.config_path)
        else:
            print("No config file provided or file not found, using minimal default config")
            raise ValueError("No config file provided or file not found")
        
        # Set device and weight type
        self.device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        if self.config.model.weight_type == "bfloat16":
            self.weight_type = torch.bfloat16
        elif self.config.model.weight_type == "float32":
            self.weight_type = torch.float32
        else:
            raise NotImplementedError(f"Weight type {self.config.model.weight_type} not supported")
        
        # Initialize VAE model
        if self.config.model.vae_model.type == 'wan21':
            self.vae_model = WanVAE(
                vae_pth=self.config.model.vae_model.pretrained_model_path, 
                dtype=self.weight_type, 
                device=self.device
            )
        else:
            raise NotImplementedError(f"VAE model type {self.config.model.vae_model.type} not supported")
        
        # Initialize text tokenizer
        self.text_tokenizer, self.showo_token_ids = get_text_tokenizer(
            self.config.model.showo.llm_model_path,
            add_showo_tokens=True,
            return_showo_token_ids=True,
            llm_name=path_to_llm_name[self.config.model.showo.llm_model_path]
        )
        self.config.model.showo.llm_vocab_size = len(self.text_tokenizer)
        
        # Initialize Show-o2 model
        if self.config.model.showo.load_from_showo:
            self.model = Showo2Qwen2_5.from_pretrained(
                self.config.model.showo.pretrained_model_path, 
                use_safetensors=False
            ).to(self.device)
        else:
            self.model = Showo2Qwen2_5(**self.config.model.showo).to(self.device)
            if hasattr(self.config, 'model_path'):
                state_dict = load_state_dict(self.config.model_path)
                self.model.load_state_dict(state_dict)
        
        self.model.to(self.weight_type)
        self.model.eval()
        
        # for time embedding - adjust token counts if needed
        if self.config.model.showo.add_time_embeds:
            # we prepend the time embedding to vision tokens
            self.config.dataset.preprocessing.num_t2i_image_tokens += 1
            self.config.dataset.preprocessing.num_mmu_image_tokens += 1
            self.config.dataset.preprocessing.num_video_tokens += 1
        
        # Get hyperparameters
        self.num_t2i_image_tokens, self.num_mmu_image_tokens, self.num_video_tokens, \
        self.max_seq_len, self.max_text_len, self.image_latent_dim, self.patch_size, \
        self.latent_width, self.latent_height, self.pad_id, self.bos_id, self.eos_id, \
        self.boi_id, self.eoi_id, self.bov_id, self.eov_id, self.img_pad_id, \
        self.vid_pad_id, self.guidance_scale = get_hyper_params(
            self.config, self.text_tokenizer, self.showo_token_ids
        )
        
        # Initialize transport and sampler for image generation
        self.transport = create_transport(
            path_type=self.config.transport.path_type,
            prediction=self.config.transport.prediction,
            loss_weight=self.config.transport.loss_weight,
            train_eps=self.config.transport.train_eps,
            sample_eps=self.config.transport.sample_eps,
            snr_type=self.config.transport.snr_type,
            do_shift=self.config.transport.do_shift,
            seq_len=self.num_t2i_image_tokens,
        )
        self.sampler = Sampler(self.transport)
        
        print("Show-o2 models initialized successfully!")
    
    def _load_config_from_file(self, config_path: str):
        """Load configuration from a YAML file."""
        from omegaconf import OmegaConf
        
        # Load the YAML config file
        config = OmegaConf.load(config_path)
        
        return config
    
    
    def generate_image_from_text(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """Generate image from text prompt using Show-o2."""
        if seed is not None:
            self.set_global_seed(seed)
        
        # Prepare input for generation
        batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = prepare_gen_input(
                [prompt], self.text_tokenizer, self.num_t2i_image_tokens, 
                self.bos_id, self.eos_id, self.boi_id, self.eoi_id, 
                self.pad_id, self.img_pad_id, self.max_text_len, self.device
            )
        
        # Initialize noise
        z = torch.randn((1, self.image_latent_dim, 
                        self.latent_height * self.patch_size,
                        self.latent_width * self.patch_size)).to(self.weight_type).to(self.device)
        
        # Apply classifier-free guidance
        if self.guidance_scale > 0:
            z = torch.cat([z, z], dim=0)
            text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
            modality_positions = torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0)
            block_mask = omni_attn_mask_naive(
                text_tokens.size(0), self.max_seq_len, modality_positions, self.device
            ).to(self.weight_type)
        else:
            text_tokens = batch_text_tokens
            modality_positions = batch_modality_positions
            block_mask = omni_attn_mask_naive(
                text_tokens.size(0), self.max_seq_len, modality_positions, self.device
            ).to(self.weight_type)
        
        # Model kwargs for generation
        model_kwargs = dict(
            text_tokens=text_tokens,
            attention_mask=block_mask,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=self.max_seq_len,
            guidance_scale=self.guidance_scale
        )
        
        # Sample using transport
        sample_fn = self.sampler.sample_ode(
            sampling_method=self.config.transport.sampling_method,
            num_steps=self.config.transport.num_inference_steps,
            atol=self.config.transport.atol,
            rtol=self.config.transport.rtol,
            reverse=self.config.transport.reverse,
            time_shifting_factor=self.config.transport.time_shifting_factor
        )
        
        samples = sample_fn(z, self.model.t2i_generate, **model_kwargs)[-1]
        if self.guidance_scale > 0:
            samples = torch.chunk(samples, 2)[0]  # Take first half for guided generation
        
        # Decode samples to images
        samples = samples.unsqueeze(2).to(self.weight_type)
        images = self.vae_model.batch_decode(samples)
        images = images.squeeze(2)
        
        # Convert to PIL image
        images = denorm(images)
        pil_image = Image.fromarray(images[0])
        print(f"Image generated successfully!")
        
        return pil_image
    
    def generate_caption_from_image(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
        """Generate caption from image using Show-o2."""

        self.weight_type = torch.float32
        
        # Ensure image is in the right format
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")
        
        # Preprocess image
        image = image.convert("RGB")
        image_tensor = image_transform(image, resolution=self.config.dataset.preprocessing.resolution).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)
        
        # Get image latents
        image_latents = self.vae_model.sample(image_tensor.unsqueeze(2)).squeeze(2).to(self.weight_type)
        
        # Get image embeddings
        image_embeds_und = self.model.image_embedder_und(image_latents)
        image_embeds_gen = self.model.image_embedder_gen(image_latents)
        image_embeds_und = image_embeds_und + self.model.position_embedding(self.model.image_position_ids)
        image_embeds_und = self.model.und_trans(image_embeds_und)['last_hidden_state']
        image_embeds = self.model.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1)).to(self.weight_type)
        
        # Prepare text input
        input_ids = self.text_tokenizer(prompt, add_special_tokens=False).input_ids
        
        # System prompt and role tokens
        sys_prompt_ids = self.text_tokenizer(
            "system\nYou are a helpful assistant.<|im_end|>", 
            add_special_tokens=False
        )['input_ids']
        role_a = self.text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)['input_ids']
        role_b = self.text_tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)['input_ids']
        
        # Create text embeddings
        text_tokens_a = torch.tensor([self.showo_token_ids['bos_id']] + sys_prompt_ids + role_a, dtype=torch.long).to(self.device)[None, :]
        text_tokens_b = torch.tensor([self.showo_token_ids['boi_id'], self.showo_token_ids['eoi_id']] + input_ids + role_b, dtype=torch.long).to(self.device)[None, :]
        text_embeds_a = self.model.showo.model.embed_tokens(text_tokens_a).to(self.weight_type)
        text_embeds_b = self.model.showo.model.embed_tokens(text_tokens_b).to(self.weight_type)
        
        # Prepare input embeddings
        if self.config.model.showo.add_time_embeds:
            time_embeds = self.model.time_embed(torch.Tensor([[1.0]]).to(self.device).to(self.weight_type), text_embeds_a.dtype)
            if hasattr(self.model, 'time_embed_proj'):
                time_embeds = self.model.time_embed_proj(time_embeds)
            input_embeds = torch.cat([
                text_embeds_a,
                text_embeds_b[:, :1],
                time_embeds,
                image_embeds,
                text_embeds_b[:, 1:]
            ], dim=1).to(self.weight_type)
            modality_positions = torch.tensor([text_tokens_a.shape[1] + 2, self.num_mmu_image_tokens], dtype=torch.long)[None, None, :].to(self.device)
        else:
            input_embeds = torch.cat([
                text_embeds_a,
                text_embeds_b[:, :1],
                image_embeds,
                text_embeds_b[:, 1:]
            ], dim=1).to(self.weight_type)
            modality_positions = torch.tensor([text_tokens_a.shape[1] + 1, self.num_mmu_image_tokens], dtype=torch.long)[None, None, :].to(self.device)
        
        # Create attention mask
        attention_mask = omni_attn_mask_naive(
            B=input_embeds.size(0),
            LEN=input_embeds.size(1),
            modalities=modality_positions,
            device=self.device, 
            inverted=True
        ).to(self.weight_type)
        
        # Generate response
        output_tokens = self.model.mmu_generate(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            top_k=1,
            max_new_tokens=300,
            eos_token=self.text_tokenizer.eos_token_id
        )
        
        output_tokens = torch.stack(output_tokens).squeeze()[None]
        text = self.text_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        
        return text[0].strip() 