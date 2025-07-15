#!/usr/bin/env python3
"""
EMU3 Roundtrip Generation Implementation

This module provides the EMU3-specific implementation of roundtrip generation.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from PIL import Image

# Add EMU3 directory to path to import modules
emu3_path = Path(__file__).parent.parent / "Emu3"
sys.path.append(str(emu3_path))

from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor

from roundtrip_base import RoundtripGenerator
from emu3.mllm.processing_emu3 import Emu3Processor


class EMU3RoundtripGenerator(RoundtripGenerator):
    """EMU3-specific implementation of roundtrip generation."""
    
    def _initialize_models(self):
        """Initialize EMU3 models."""
        print("Initializing EMU3 models...")
        
        # Set device
        self.device = torch.device(f"cuda:{self.device}" if torch.cuda.is_available() else "cpu")
        
        # Model paths - use Emu3-Gen and Emu3-Chat
        if self.model_path in ["BAAI/Emu3-Gen", "BAAI/Emu3-Chat"]:
            # Use HuggingFace model names directly
            self.emu_gen_hub = "BAAI/Emu3-Gen"
            self.emu_chat_hub = "BAAI/Emu3-Chat"
            self.vq_hub = "BAAI/Emu3-VisionTokenizer"
        else:
            # Use local model paths
            self.emu_gen_hub = os.path.join(self.model_path, "gen")
            self.emu_chat_hub = os.path.join(self.model_path, "chat")
            self.vq_hub = os.path.join(self.model_path, "vision_tokenizer")
        
        # Load EMU3-Gen model for image generation
        self.gen_model = AutoModelForCausalLM.from_pretrained(
            self.emu_gen_hub,
            device_map="auto",  # Use auto device mapping for better memory management
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        self.gen_model.eval()
        
        # Load EMU3-Chat model for image understanding
        self.chat_model = AutoModelForCausalLM.from_pretrained(
            self.emu_chat_hub,
            device_map="auto",  # Use auto device mapping for better memory management
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        self.chat_model.eval()
        
        # Load tokenizers
        self.gen_tokenizer = AutoTokenizer.from_pretrained(
            self.emu_gen_hub, 
            trust_remote_code=True, 
            padding_side="left"
        )
        self.chat_tokenizer = AutoTokenizer.from_pretrained(
            self.emu_chat_hub, 
            trust_remote_code=True, 
            padding_side="left"
        )
        
        # Load vision tokenizer and processor
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.vq_hub, 
            trust_remote_code=True
        )
        self.image_tokenizer = AutoModel.from_pretrained(
            self.vq_hub, 
            device_map="auto",  # Use auto device mapping
            trust_remote_code=True
        ).eval()
        
        # Create EMU3 processors for generation and chat
        self.gen_processor = Emu3Processor(
            self.image_processor, 
            self.image_tokenizer, 
            self.gen_tokenizer
        )
        self.chat_processor = Emu3Processor(
            self.image_processor, 
            self.image_tokenizer, 
            self.chat_tokenizer
        )
        
        # Set generation parameters
        self.positive_prompt = " masterpiece, film grained, best quality."
        
        # Track which model is currently loaded
        self.current_model = None
        
        print("EMU3 models initialized successfully!")
    
    def _load_gen_model(self):
        """Load Gen model to GPU and offload Chat model if needed."""
        if self.current_model != "gen":
            print("Loading Gen model to GPU...")
            
            # Offload chat model to CPU if it's currently on GPU
            if self.current_model == "chat":
                self.chat_model.to("cpu")
                torch.cuda.empty_cache()
            
            # Load gen model to GPU
            self.gen_model.to(self.device)
            self.current_model = "gen"
            print("Gen model loaded to GPU")
    
    def _load_chat_model(self):
        """Load Chat model to GPU and offload Gen model if needed."""
        if self.current_model != "chat":
            print("Loading Chat model to GPU...")
            
            # Offload gen model to CPU if it's currently on GPU
            if self.current_model == "gen":
                self.gen_model.to("cpu")
                torch.cuda.empty_cache()
            
            # Load chat model to GPU
            self.chat_model.to(self.device)
            self.current_model = "chat"
            print("Chat model loaded to GPU")
    
    def generate_image_from_text(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """Generate image from text prompt using EMU3-Gen."""
        if seed is not None:
            self.set_global_seed(seed)
        
        # Load Gen model to GPU
        self._load_gen_model()
        
        # Prepare input
        full_prompt = prompt + self.positive_prompt
        
        kwargs = dict(
            mode='G',
            ratio="1:1",  # Default to square image
            image_area=self.gen_model.config.image_area,
            return_tensors="pt",
            padding="longest",
        )
        
        pos_inputs = self.gen_processor(text=full_prompt, **kwargs)
        
        # Prepare generation config
        generation_config = GenerationConfig(
            use_cache=False,  # Disable cache to avoid DynamicCache issues
            eos_token_id=self.gen_model.config.eos_token_id,
            pad_token_id=self.gen_model.config.pad_token_id,
            max_new_tokens=40960,
            do_sample=True,
            top_k=2048,
        )
        
        # Prepare logits processor for prefix constraint only
        h = pos_inputs.image_size[:, 0]
        w = pos_inputs.image_size[:, 1]
        constrained_fn = self.gen_processor.build_prefix_constrained_fn(h, w)
        
        logits_processor = LogitsProcessorList([
            PrefixConstrainedLogitsProcessor(
                constrained_fn,
                num_beams=1,
            ),
        ])
        
        # Generate without classifier-free guidance to avoid DynamicCache issues
        with torch.no_grad():
            outputs = self.gen_model.generate(
                pos_inputs.input_ids.to(self.device),
                generation_config,
                logits_processor=logits_processor,
                attention_mask=pos_inputs.attention_mask.to(self.device),
            )
        
        # Decode the generated image
        mm_list = self.gen_processor.decode(outputs[0])
        for item in mm_list:
            if isinstance(item, Image.Image):
                return item
        
        # If no image found, raise error
        raise RuntimeError("No image was generated from the text prompt")
    
    def generate_caption_from_image(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
        """Generate caption from image using EMU3-Chat."""
        # Load Chat model to GPU
        self._load_chat_model()
        
        # Prepare input
        inputs = self.chat_processor(
            text=prompt,
            image=image,
            mode='U',
            return_tensors="pt",
            padding="longest",
        )
        
        # Prepare generation config
        generation_config = GenerationConfig(
            pad_token_id=self.chat_tokenizer.pad_token_id,
            bos_token_id=self.chat_tokenizer.bos_token_id,
            eos_token_id=self.chat_tokenizer.eos_token_id,
            max_new_tokens=1024,
        )
        
        # Generate caption
        with torch.no_grad():
            outputs = self.chat_model.generate(
                inputs.input_ids.to(self.device),
                generation_config,
                attention_mask=inputs.attention_mask.to(self.device),
            )
        
        # Decode the generated text
        outputs = outputs[:, inputs.input_ids.shape[-1]:]
        caption = self.chat_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return caption.strip() 