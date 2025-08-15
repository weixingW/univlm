#!/usr/bin/env python3
"""
Janus Pro Reverse Roundtrip Generation Implementation

This module provides the Janus Pro-specific implementation of reverse roundtrip generation.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
import torch
import numpy as np
from PIL import Image

# Add Janus directory to path to import modules
janus_path = Path(__file__).parent.parent / "Janus"
sys.path.append(str(janus_path))

from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

from reverse_roundtrip_base import ReverseRoundtripGenerator


class JanusProReverseRoundtripGenerator(ReverseRoundtripGenerator):
    """Janus Pro-specific implementation of reverse roundtrip generation."""
    
    def _initialize_models(self):
        """Initialize Janus Pro models."""
        print("Initializing Janus Pro models for reverse roundtrip...")
        
        # Load processor and tokenizer
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        # Load Janus Pro model
        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda(self.device).eval()
        
        print("Janus Pro models initialized successfully for reverse roundtrip!")
    
    def generate_image_from_text(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        """Generate image from text prompt using Janus Pro."""
        if seed is not None:
            self.set_global_seed(seed)
        
        # Create conversation format for Janus Pro
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # Apply SFT template
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt_text = sft_format + self.vl_chat_processor.image_start_tag
        
        # Generate image using Janus Pro's generation function
        generated_image = self._generate_image(
            prompt_text,
            temperature=1.0,
            parallel_size=1,  # Generate single image
            cfg_weight=5.0,
            image_token_num_per_image=576,
            img_size=384,
            patch_size=16,
        )
        
        # Ensure we return a PIL Image
        if not isinstance(generated_image, Image.Image):
            print(f"Warning: generate_image_from_text returned non-Image: {type(generated_image)}")
            if generated_image is None:
                raise ValueError("Image generation failed - returned None")
            generated_image = Image.fromarray(generated_image) if hasattr(generated_image, 'shape') else Image.new('RGB', (384, 384))
        
        return generated_image
    
    def generate_caption_from_image(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
        """Generate caption from image using Janus Pro."""
        # Create conversation format for Janus Pro
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # Load images and prepare for inputs
        pil_images = [image.convert("RGB")]
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)
        
        # Run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        
        # Run the model to get the response
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
        
        # Decode the response
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        print(f"Debug: Raw decoded answer: {answer}")
        
        # Extract only the generated part (remove the input prompt)
        # For Janus Pro, we need to find where the actual response starts
        # Look for the assistant role marker
        assistant_marker = "<|Assistant|>"
        if assistant_marker in answer:
            # Split by assistant marker and take the last part
            parts = answer.split(assistant_marker)
            if len(parts) > 1:
                answer = parts[-1].strip()
        
        print(f"Debug: Final answer: {answer}")
        print(f"Debug: Answer type: {type(answer)}")
        
        # Ensure we return a string, not an image object
        if not isinstance(answer, str):
            print(f"Warning: generate_caption_from_image returned non-string: {type(answer)}")
            answer = str(answer) if answer is not None else "No caption generated"
        
        return answer
    
    def _generate_image(
        self,
        prompt: str,
        temperature: float = 1.0,
        parallel_size: int = 1,
        cfg_weight: float = 5.0,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
    ) -> Image.Image:
        """Internal method to generate image using Janus Pro's generation logic."""
        
        @torch.inference_mode()
        def generate_internal():
            input_ids = self.vl_chat_processor.tokenizer.encode(prompt)
            input_ids = torch.LongTensor(input_ids)
            
            tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda(self.device)
            for i in range(parallel_size*2):
                tokens[i, :] = input_ids
                if i % 2 != 0:
                    tokens[i, 1:-1] = self.vl_chat_processor.pad_id
            
            inputs_embeds = self.vl_gpt.language_model.get_input_embeddings()(tokens)
            
            generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda(self.device)
            
            outputs = None
            for i in range(image_token_num_per_image):
                outputs = self.vl_gpt.language_model.model(
                    inputs_embeds=inputs_embeds, 
                    use_cache=True, 
                    past_key_values=outputs.past_key_values if i != 0 else None
                )
                hidden_states = outputs.last_hidden_state
                
                logits = self.vl_gpt.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                
                logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
                probs = torch.softmax(logits / temperature, dim=-1)
                
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)
                
                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = self.vl_gpt.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)
            
            # Decode the generated tokens to image
            dec = self.vl_gpt.gen_vision_model.decode_code(
                generated_tokens.to(dtype=torch.int), 
                shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
            )
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            
            dec = np.clip((dec + 1) / 2 * 255, 0, 255)
            
            # Return the first (and only) generated image
            visual_img = dec[0].astype(np.uint8)
            return Image.fromarray(visual_img)
        
        return generate_internal()


if __name__ == "__main__":
    from reverse_roundtrip_base import create_reverse_parser
    
    parser = create_reverse_parser()
    args = parser.parse_args()
    
    # Create generator
    generator = JanusProReverseRoundtripGenerator(
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