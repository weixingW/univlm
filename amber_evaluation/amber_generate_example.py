import os
import glob
import re
import torch
import argparse
import sys
sys.path.append(".")
sys.path.append("..")
from transformers import (
    ChameleonForConditionalGeneration,
    ChameleonProcessor,
    set_seed,
    GenerationConfig,
)
from mmsg.utils import load_image
import logging
import json
from typing import Dict, List, Optional, Tuple
import tqdm
from latent_gen import run_latent_generation_with_gnn, run_latent_generation, load_models
from vq_vis import load_model_and_processor
from vcd_utils.vcd_sample import evolve_vcd_sampling
from vcd_utils.vcd_add_noise import add_diffusion_noise
from torchvision import transforms
from PIL import Image


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

def extract(line):
    NEG_WORDS = ["No", "not", "no", "NO"]
    line = line.replace('.', '')
    line = line.replace(',', '')
    words = line.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"

def load_query_data(query_file: str) -> List[Dict]:
    """Load query data from JSON file."""
    if not os.path.exists(query_file):
        raise FileNotFoundError(f"Query file not found: {query_file}")
        
    with open(query_file, 'r') as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} queries from {query_file}")
    return queries

def get_amber_images_with_queries(image_dir: str, queries: List[Dict]) -> List[Tuple[str, int, str]]:
    """Get AMBER image paths with their IDs and queries."""
    # Create a list of tuples with image paths, IDs and queries
    image_tuples = []
    for query in queries:
        image_name = query["image"]
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            image_tuples.append((image_path, query["id"], query["query"]))
    
    # Sort by ID
    sorted_images = sorted(image_tuples, key=lambda x: x[1])
    logger.info(f"Found {len(sorted_images)} matching images with queries")
    return sorted_images

def load_existing_captions(output_file: str) -> Dict[int, str]:
    """Load existing captions from amber_out.json if it exists."""
    if not os.path.exists(output_file):
        return {}
        
    with open(output_file, "r") as f:
        data = json.load(f)
        return {item["id"]: item["response"] for item in data}

def save_amber_response(id, response, output_file: str):
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
    else:
        data = []
            
    # Update existing entry or add new one
    entry = {"id": id, "response": response}
    for i, item in enumerate(data):
        if item["id"] == id:
            data[i] = entry
            break
    else:
        data.append(entry)
        
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

def tensor_to_pil(tensor):
    """Convert a tensor to PIL Image in RGB mode."""
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Make sure tensor is on CPU
    tensor = tensor.cpu()
    
    # If values are in [0, 1], scale to [0, 255]
    if tensor.max() <= 1:
        tensor = tensor * 255
    
    # Convert to uint8
    tensor = tensor.clamp(0, 255).byte()
    
    # Convert to PIL
    if tensor.shape[0] == 1:  # Grayscale
        pil_image = Image.fromarray(tensor.squeeze().numpy(), mode='L').convert('RGB')
    elif tensor.shape[0] == 3:  # RGB
        pil_image = Image.fromarray(tensor.permute(1, 2, 0).numpy(), mode='RGB')
    elif tensor.shape[0] == 4:  # RGBA
        pil_image = Image.fromarray(tensor.permute(1, 2, 0).numpy(), mode='RGBA').convert('RGB')
    else:
        raise ValueError(f"Unsupported number of channels: {tensor.shape[0]}")
    
    return pil_image

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with latent image/text embedding manipulation."
    )
    parser.add_argument(
        "--query_file",
        type=str,
        required=False,
        default="outputs/amber_output/query_all.json",
        help="Path to query_all.json containing image IDs and queries",
    )
    parser.add_argument("--model_id", type=str, default="leloy/Anole-7b-v0.1-hf", help="Model ID")
    parser.add_argument("--operation", type=str, default="subtract", help="Operation to perform")
    parser.add_argument("--weight", type=float, default=1.0, help="Weight for projection subtraction")
    parser.add_argument("--use_mean", action="store_true", help="Use mean pooling for text embeddings")
    parser.add_argument("--fast", action="store_true", help="Use fast settings")
    parser.add_argument("--model_cache_dir", type=str, default="/scratch/weixing.wang", help="Model cache directory")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--prompt", type=str, default="Please describe this image in detail.", help="Prompt for caption generation")
    parser.add_argument("--layer", type=int, default=12, help="Layer to perform the operation")
    parser.add_argument("--gen_type", type=str, default="sample", 
                       choices=["sample", "opera", "vcd", "gnn", "sid"],
                       help="Generation type: 'sample', 'opera', 'vcd', 'gnn', or 'sid'")
    parser.add_argument("--noise_step", type=int, default=500,
                       help="Number of noise steps for VCD")
    parser.add_argument("--cd_alpha", type=float, default=0.5,
                       help="Alpha parameter for contrastive decoding")
    parser.add_argument("--cd_beta", type=float, default=0.1,
                       help="Beta parameter for contrastive decoding")
    parser.add_argument("--cluster_results_path", type=str, 
                       help="Path to cluster results (required for GNN generation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--find_non_input_indices", action="store_true", 
                       help="Whether to find indices that don't appear in the input image")
    parser.add_argument("--model_type", type=str, default="chameleon", help="Model type")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens")

    # fast token merging
    parser.add_argument("--fast-v-inplace", default=False)
    parser.add_argument("--fast-v-attention-rank", type=int, default=100)
    parser.add_argument("--fast-v-attention-rank-add", type=int, default=100)
    parser.add_argument("--fast-v-agg-layer", type=int, default=2)
    # auto-generation
    parser.add_argument("--fast-v-sys-length", default=None, type=int, help='the length of system prompt')
    parser.add_argument("--fast-v-image-token-length", default=None, type=int, help='the length of image token')

    return parser.parse_args()

def get_generation_config(model_type, gen_type, args, processor, inputs, model, key_position=None):
    """Get appropriate generation config based on model type and generation mode."""
    
    if gen_type == "vcd" or gen_type == "sid":
        # Prepare noisy image for VCD
        if isinstance(inputs["pixel_values"], str):
            # If it's a path, load the image first
            image = Image.open(inputs["pixel_values"])
            image = image.convert("RGB")
            if model_type == "chameleon":
                # Process with dummy prompt since processor requires text
                dummy_inputs = processor(text="<image>", images=image, return_tensors="pt")
                image_tensor = dummy_inputs["pixel_values"]
            elif model_type == "Emu3":
                image_tensor = processor.image_processor([image], return_tensors="pt")["pixel_values"]
            elif model_type == "janus":
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": "<image_placeholder>",
                        "images": [image],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                prepare_inputs = processor(
                    conversations=conversation,
                    images=[image],
                    force_batchify=True
                )
                image_tensor = prepare_inputs["pixel_values"]
        else:
            image_tensor = inputs["pixel_values"]

        if gen_type == "vcd":
            image_cd = add_diffusion_noise(image_tensor, args.noise_step)
        elif gen_type == "sid":
            image_cd = image_tensor
        image_cd = image_cd.to(model.device, dtype=torch.bfloat16)
        
        config = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            #"temperature": 0.7,
            #"top_p": 0.9,
            "vcd_sample": True,
            "cd_alpha": args.cd_alpha,
            "cd_beta": args.cd_beta,
            "vcd_inputs": image_cd,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict_in_generate": True,
            #"pad_token_id": model.config.pad_token_id if model.config.pad_token_id is not None else processor.tokenizer.pad_token_id,
            #"eos_token_id": model.config.eos_token_id if model.config.eos_token_id is not None else processor.tokenizer.eos_token_id
            "key_position": key_position
        }
        if gen_type == "sid":
            config["use_sid"] = True
        
        
        if model_type == "Emu3":
            config.update({
                "pad_token_id": processor.tokenizer.pad_token_id,
                "bos_token_id": processor.tokenizer.bos_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            })
            
        return GenerationConfig(**config)
        
    elif gen_type == "opera":
        config = {
            "max_length": 600,
            "output_attentions": True,
            "num_beams": 5,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "opera_decoding": True,
            "key_position": key_position,
            "scale_factor": 50,
            "threshold": 15,
            "num_attn_candidates": 5,
            "penalty_weights": 1,
            "return_dict_in_generate": True,
            "pad_token_id": model.config.pad_token_id if model.config.pad_token_id is not None else processor.tokenizer.pad_token_id,
            "eos_token_id": model.config.eos_token_id if model.config.eos_token_id is not None else processor.tokenizer.eos_token_id
        }
        
        if model_type == "Emu3":
            config.update({
                "pad_token_id": processor.tokenizer.pad_token_id,
                "bos_token_id": processor.tokenizer.bos_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            })
            
        return GenerationConfig(**config)
        
    else:  # Default sampling or gnn
        config = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": model.config.pad_token_id if model.config.pad_token_id is not None else processor.tokenizer.pad_token_id,
            "eos_token_id": model.config.eos_token_id if model.config.eos_token_id is not None else processor.tokenizer.eos_token_id
        }
        
        if model_type == "Emu3":
            config.update({
                "pad_token_id": processor.tokenizer.pad_token_id,
                "bos_token_id": processor.tokenizer.bos_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            })
            
        return GenerationConfig(**config)

def main():
    args = parse_arguments()
    logger.info(f"Loading model and processor from {args.model_id}")
    
    # Load model and processor based on model type
    model, processor = load_model_and_processor(
        model_path=args.model_id,
        model_type=args.model_type
    )
    
    
    # Load query data and get images
    queries = load_query_data(args.query_file)
    image_dir = "/hpi/fs00/share/fg-meinel/weixing.wang/datasets/AMBER/image/"
    image_data = get_amber_images_with_queries(image_dir, queries)
    logger.info(f"Found {len(image_data)} AMBER images with queries")
    
    # Load existing captions
    existing_captions = load_existing_captions(args.output_file)
    logger.info(f"Loaded {len(existing_captions)} existing captions")
    
    
    for image_path, amber_id, query in tqdm.tqdm(image_data, desc="Generating captions"):
        if amber_id in existing_captions:
            logger.info(f"Skipping AMBER_{amber_id}, caption already exists")
            continue
        if args.gen_type == "gnn":
            if not args.cluster_results_path:
                raise ValueError("cluster_results_path must be provided for GNN generation")
                
            response = run_latent_generation_with_gnn(
                model_type=args.model_type,
                model=model,
                processor=processor,
                model_id=args.model_id,
                image_1_path=image_path,
                prompt=query,
                layer=args.layer,
                operation=args.operation,
                weight=args.weight,
                use_mean=args.use_mean,
                max_new_tokens=args.max_new_tokens,
                fast=args.fast,
                model_cache_dir=args.model_cache_dir,
                seed=args.seed,
                cluster_results_path=args.cluster_results_path,
                find_non_input_indices=args.find_non_input_indices,
            )
        else:
            image = load_image(image_path)
            image = image.convert("RGB")
            
            # Process inputs based on model type
            if args.model_type == "chameleon":
                inputs = processor(
                    text=query + "<image>",
                    images=image,
                    return_tensors="pt"
                )
                # Convert inputs to bfloat16 to match model dtype
                model_dtype = next(model.parameters()).dtype
                device = next(model.parameters()).device
                inputs = {
                    "input_ids": inputs["input_ids"].to(device, dtype=torch.long),
                    "attention_mask": inputs["attention_mask"].to(device, dtype=torch.long),
                    "pixel_values": inputs["pixel_values"].to(device, dtype=model_dtype)
                }
                
                # Get key positions
                #import pdb; pdb.set_trace()
                image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
                image_token_pos = (inputs["input_ids"] == image_token_id).nonzero()[0, 1].item()
                key_position = {
                    "image_start": torch.tensor(image_token_pos).to(model.device),
                    "image_end": torch.tensor(image_token_pos + 1023).to(model.device),
                    "response_start": torch.tensor(inputs["input_ids"].shape[-1]).to(model.device),
                }
                if args.gen_type == "sid":
                    model.model.config.use_fast_v = True
                    model.model.config.fast_v_inplace = args.fast_v_inplace
                    model.model.config.fast_v_sys_length = args.fast_v_sys_length
                    model.model.config.fast_v_image_token_length = args.fast_v_image_token_length
                    model.model.config.fast_v_attention_rank = args.fast_v_attention_rank
                    model.model.config.fast_v_attention_rank_add = args.fast_v_attention_rank_add
                    model.model.config.fast_v_agg_layer = args.fast_v_agg_layer
                else:
                    model.model.config.use_fast_v = False
                model.model.reset_fastv()
                
            elif args.model_type == "Emu3":
                
                #torch.cuda.empty_cache()
                inputs, image_start_list, image_end_list = processor(
                    text=query,
                    image=image,
                    mode="U",
                    return_tensors="pt",
                    padding="longest",
                )
                
                image_tensor = processor.image_processor([image], return_tensors="pt")["pixel_values"]
                image_cd = add_diffusion_noise(image_tensor, args.noise_step)
                image_cd =  image_cd.to(model.device, dtype=torch.bfloat16)
                image_cd = tensor_to_pil(image_cd)
                inputs_cd, _, _ = processor(
                    text=query,
                    image=image_cd,
                    mode="U",
                    return_tensors="pt",
                    padding="longest",
                )
                
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                key_position = {
                    "image_start": torch.tensor(image_start_list[0]).to(model.device),
                    "image_end": torch.tensor(image_end_list[0]).to(model.device),
                    "response_start": torch.tensor(inputs["input_ids"].shape[-1]).to(model.device),
                }
                if args.gen_type == "sid":
                    model.model.config.use_fast_v = True
                    model.model.config.fast_v_inplace = args.fast_v_inplace
                    model.model.config.fast_v_sys_length = args.fast_v_sys_length
                    model.model.config.fast_v_image_token_length = args.fast_v_image_token_length
                    model.model.config.fast_v_attention_rank = args.fast_v_attention_rank
                    model.model.config.fast_v_attention_rank_add = args.fast_v_attention_rank_add
                    model.model.config.fast_v_agg_layer = args.fast_v_agg_layer
                else:
                    model.model.config.use_fast_v = False
                model.model.reset_fastv()
                
            elif args.model_type == "janus":
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{query}",
                        "images": [image],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                
                prepare_inputs = processor(
                    conversations=conversation,
                    images=[image],
                    force_batchify=True
                ).to(model.device)
                inputs = model.prepare_inputs_embeds(**prepare_inputs)
                
                img_start_id = processor.image_start_id
                img_end_id = processor.image_end_id
                img_start_pos = (prepare_inputs["input_ids"] == img_start_id).nonzero()[0, 1].item()
                img_end_pos = (prepare_inputs["input_ids"] == img_end_id).nonzero()[0, 1].item()
                
                
                key_position = {
                    "image_start": torch.tensor(img_start_pos).to(model.device),
                    "image_end": torch.tensor(img_end_pos).to(model.device),
                    "response_start": torch.tensor(inputs.shape[-2]).to(model.device),
                }
                if args.gen_type == "sid":
                    model.language_model.model.config.use_fast_v = True
                    model.language_model.model.config.fast_v_inplace = args.fast_v_inplace
                    model.language_model.model.config.fast_v_sys_length = args.fast_v_sys_length
                    model.language_model.model.config.fast_v_image_token_length = args.fast_v_image_token_length
                    model.language_model.model.config.fast_v_attention_rank = args.fast_v_attention_rank
                    model.language_model.model.config.fast_v_attention_rank_add = args.fast_v_attention_rank_add
                    model.language_model.model.config.fast_v_agg_layer = args.fast_v_agg_layer
                else:
                    model.language_model.model.config.use_fast_v = False
                model.language_model.model.reset_fastv()
            
            # Get generation config
            generation_config = get_generation_config(
                model_type=args.model_type,
                gen_type=args.gen_type,
                args=args,
                processor=processor,
                inputs={"pixel_values": image_path, "text": query},
                model=model,
                key_position=key_position
            )
            
            # Generate response
            with torch.inference_mode():
                if args.model_type == "chameleon":
                    output = model.generate(
                        **inputs,
                        generation_config=generation_config,
                        #eos_token_id=processor.tokenizer.eos_token_id,
                        #pad_token_id=processor.tokenizer.pad_token_id
                    )
                    
                    if isinstance(output, dict):
                        output = output["sequences"]
                    
                    response = processor.decode(
                        output[0][len(inputs["input_ids"][0]):], 
                        skip_special_tokens=True
                    )
                    
                elif args.model_type == "Emu3":
                    generation_config.vcd_inputs = inputs_cd["input_ids"]
                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        generation_config=generation_config
                    )
                    if isinstance(outputs, dict):
                        generated_sequence = outputs.sequences
                    else:
                        generated_sequence = outputs
                    outputs = generated_sequence[:, inputs["input_ids"].shape[-1]:]
                    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                    
                elif args.model_type == "janus":
                    if args.gen_type == "vcd" or args.gen_type == "sid":
                        # Detach and clone the tensors before passing to generate
                        detached_inputs = inputs.detach().clone()
                        detached_attention_mask = prepare_inputs.attention_mask.detach().clone()
                        image_cd = tensor_to_pil(generation_config.vcd_inputs.squeeze(0))
                        conversation = [
                            {
                                "role": "<|User|>",
                                "content": f"<image_placeholder>\n{query}",
                                "images": [image_cd],
                            },
                            {"role": "<|Assistant|>", "content": ""},
                        ]
                        prepare_inputs_cd = processor(
                            conversations=conversation,
                            images=[image_cd],
                            return_tensors="pt"
                        ).to(model.device)
                        inputs_cd = model.prepare_inputs_embeds(**prepare_inputs_cd)
                        generation_config.vcd_inputs=inputs_cd
                                
                        outputs = model.language_model.generate(
                            inputs_embeds=detached_inputs,
                            attention_mask=detached_attention_mask,
                            generation_config=generation_config
                        )
                    else:
                        outputs = model.language_model.generate(
                            inputs_embeds=inputs,
                            attention_mask=prepare_inputs.attention_mask,
                            generation_config=generation_config
                        )
                        
                    
                    # Handle outputs whether they're a dictionary or tensor
                    if isinstance(outputs, dict):
                        generated_sequence = outputs.sequences
                    else:
                        generated_sequence = outputs
                        
                    response = processor.tokenizer.decode(
                        generated_sequence[0].cpu().tolist(), 
                        skip_special_tokens=True
                    )
        
        
        # Extract yes/no answer if needed
        if amber_id >= 1005:
            response = extract(response)
            
        save_amber_response(amber_id, response, args.output_file)
        logger.info(f"Generated caption for AMBER_{amber_id}")
            
        
            

    logger.info("Finished generating captions")

if __name__ == "__main__":
    main() 