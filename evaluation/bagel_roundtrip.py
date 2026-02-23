import sys
import os

# Hack to fix missing CUDA_HOME preventing flash_attn from loading properly in some environments
try:
    import flash_attn
except ImportError:
    # If flash_attn is not installed or broken, force Qwen2 to use eager attention implementation
    sys.modules["flash_attn"] = None

from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download

from roundtrip_base import RoundtripGenerator


class BagelRoundtripGenerator(RoundtripGenerator):
    def _initialize_models(self):
        # 1. Add Bagel repo to sys.path
        repo_root = Path(__file__).resolve().parents[1] / "Bagel"
        sys.path.insert(0, str(repo_root))

        try:
            # 2. Imports from Bagel
            from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
            from data.data_utils import add_special_tokens, pil_img2rgb
            from data.transforms import ImageTransform
            from inferencer import InterleaveInferencer
            from modeling.autoencoder import load_ae
            from modeling.bagel import (
                BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
                SiglipVisionConfig, SiglipVisionModel
            )
            from modeling.qwen2 import Qwen2Tokenizer
            
            # 3. Model Initialization (adapted from Bagel/app.py)
            model_path = self.model_path
            
            if not os.path.exists(model_path):
                 print(f"Model path {model_path} not found locally. Attempting download from HF...")
                 try:
                     model_path = snapshot_download(repo_id=model_path)
                     print(f"Model downloaded to {model_path}")
                 except Exception as e:
                     raise ValueError(f"Could not find model at {self.model_path} and failed to download: {e}")
            
            # Use os.path.join for safety
            llm_config_path = os.path.join(model_path, "llm_config.json")
            if not os.path.exists(llm_config_path):
                print(f"Warning: {llm_config_path} not found. Ensure Bagel model is downloaded to {model_path}")
            
            llm_config = Qwen2Config.from_json_file(llm_config_path)
            llm_config.qk_norm = True
            llm_config.tie_word_embeddings = False
            llm_config.layer_module = "Qwen2MoTDecoderLayer"

            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
            vit_config.rope = False
            vit_config.num_hidden_layers -= 1

            vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

            config = BagelConfig(
                visual_gen=True,
                visual_und=True,
                llm_config=llm_config, 
                vit_config=vit_config,
                vae_config=vae_config,
                vit_max_num_patch_per_side=70,
                connector_act='gelu_pytorch_tanh',
                latent_patch_size=2,
                max_latent_size=64,
            )

            with init_empty_weights():
                language_model = Qwen2ForCausalLM(llm_config)
                vit_model      = SiglipVisionModel(vit_config)
                model          = Bagel(language_model, vit_model, config)
                model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

            tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
            tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

            vae_transform = ImageTransform(1024, 512, 16)
            vit_transform = ImageTransform(980, 224, 14)

            # Model Loading and Multi GPU Inference Preparing
            device_map = infer_auto_device_map(
                model,
                max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
                no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
            )

            same_device_modules = [
                'language_model.model.embed_tokens',
                'time_embedder',
                'latent_pos_embed',
                'vae2llm',
                'llm2vae',
                'connector',
                'vit_pos_embed'
            ]
            
            # Handling device map logic from app.py
            if torch.cuda.device_count() == 1:
                first_device = device_map.get(same_device_modules[0], "cuda:0")
                for k in same_device_modules:
                    if k in device_map:
                        device_map[k] = first_device
                    else:
                        device_map[k] = "cuda:0"
            else:
                first_device = device_map.get(same_device_modules[0], "cuda:0") # fallback
                for k in same_device_modules:
                    if k in device_map:
                        device_map[k] = first_device

            # Load checkpoint
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=os.path.join(model_path, "ema.safetensors"),
                device_map=device_map,
                offload_buffers=True,
                offload_folder="offload",
                dtype=torch.bfloat16,
                force_hooks=True,
            ).eval()

            # Initialize Inferencer
            self.inferencer = InterleaveInferencer(
                model=model,
                vae_model=vae_model,
                tokenizer=tokenizer,
                vae_transform=vae_transform,
                vit_transform=vit_transform,
                new_token_ids=new_token_ids,
            )
            
            # Helper function dependencies
            self.pil_img2rgb = pil_img2rgb
            
            self._loaded = True
        
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Bagel model: {e}")

    def generate_image_from_text(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        if seed is None:
            seed = self.seed
            
        # Set seed
        if seed > 0:
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        
        # Hyperparameters (defaults from app.py)
        # Using 1:1 ratio (1024x1024)
        image_shapes = (1024, 1024)
        
        inference_hyper = dict(
            max_think_token_n=1024,
            do_sample=False, 
            text_temperature=0.3, # not used if do_sample is False
            cfg_text_scale=4.0,
            cfg_interval=[0.4, 1.0],
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
            image_shapes=image_shapes,
        )
        
        # Call inferencer
        result = self.inferencer(text=prompt, think=False, **inference_hyper)
        return result["image"]

    def generate_caption_from_image(
        self,
        image: Image.Image,
        prompt: str = "Describe this image in detail.",
    ) -> str:
        
        if image is None:
            return ""

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = self.pil_img2rgb(image)
        
        inference_hyper = dict(
            do_sample=False,
            text_temperature=0.3,
            max_think_token_n=512, 
        )
        
        result = self.inferencer(image=image, text=prompt, think=False, 
                                understanding_output=True, **inference_hyper)
        return result["text"]
