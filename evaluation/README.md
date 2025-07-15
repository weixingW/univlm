# Modular Roundtrip Generation

This directory contains a modular implementation of roundtrip generation that supports multiple models (BLIP3o, MMaDA, etc.).

## Overview

Roundtrip generation performs the following sequence:
1. **Text → Image**: Generate image from text prompt
2. **Image → Text**: Generate caption from the generated image
3. **Store Results**: Save both the generated image and caption
4. **(Optional) Caption → Image**: Generate new images from generated captions

## Architecture

The implementation uses a modular architecture with the following components:

### Core Components

- **`roundtrip_base.py`**: Abstract base class `RoundtripGenerator` that defines the interface for all roundtrip generators
- **`roundtrip_factory.py`**: Factory pattern for creating generators for different model types
- **`roundtrip_generation.py`**: Main script that orchestrates the roundtrip generation process

### Model-Specific Implementations

- **`blip3o_roundtrip.py`**: BLIP3o-specific implementation
- **`mmada_roundtrip.py`**: MMaDA-specific implementation
- **`emu3_roundtrip.py`**: EMU3-specific implementation
- **`omnigen2_roundtrip.py`**: OmniGen2-specific implementation
- **`januspro_roundtrip.py`**: Janus Pro-specific implementation
- **`showo2_roundtrip.py`**: Show-o2-specific implementation
- **`showo_roundtrip.py`**: Show-o-specific implementation

## Usage

### Basic Usage

```bash
# Run full roundtrip generation with BLIP3o
python roundtrip_generation.py /path/to/blip3o/model --model_type blip3o --prompts_file prompts.txt

# Run full roundtrip generation with MMaDA
python roundtrip_generation.py Gen-Verse/MMaDA-8B-Base --model_type mmada --prompts_file prompts.txt

# Run full roundtrip generation with EMU3
python roundtrip_generation.py BAAI/Emu3-Stage1 --model_type emu3 --prompts_file prompts.txt

# Run full roundtrip generation with OmniGen2
python roundtrip_generation.py /path/to/omnigen2/model --model_type omnigen2 --prompts_file prompts.txt

# Run full roundtrip generation with Janus Pro
python roundtrip_generation.py /path/to/januspro/model --model_type januspro --prompts_file prompts.txt

# Run full roundtrip generation with Show-o2
python roundtrip_generation.py showlab/show-o2-7b --model_type showo2 --prompts_file prompts.txt

# Run full roundtrip generation with Show-o
python roundtrip_generation.py /path/to/showo/model --model_type showo --config_path /path/to/config.yaml --prompts_file prompts.txt

# Generate images from existing captions
python roundtrip_generation.py /path/to/blip3o/model --model_type blip3o --generate_from_captions --captions_dir /path/to/captions
```

### Command Line Arguments

- `model_path`: Path to the model directory
- `--model_type`: Type of model (`blip3o`, `mmada`, `emu3`, `omnigen2`, `januspro`, `showo2`, `showo`)
- `--config_path`: Path to configuration file (optional, model-specific)
- `--prompts_file`: Path to prompts file (default: `prompts.txt`)
- `--output_dir`: Output directory (default: `roundtrip_results`)
- `--start_idx`: Starting prompt index (default: 0)
- `--end_idx`: Ending prompt index (default: all prompts)
- `--device`: CUDA device ID (default: 0)
- `--seed`: Random seed (default: 42)
- `--generate_from_captions`: Generate images from existing captions instead of full roundtrip
- `--captions_dir`: Directory containing caption files (default: `output_dir/captions`)

### Programmatic Usage

```python
from roundtrip_factory import create_roundtrip_generator

# Create a BLIP3o generator
generator = create_roundtrip_generator(
    model_type="blip3o",
    model_path="/path/to/blip3o/model",
    device=0,
    seed=42
)

# Create a MMaDA generator
generator = create_roundtrip_generator(
    model_type="mmada", 
    model_path="Gen-Verse/MMaDA-8B-Base",
    device=0,
    seed=42
)

# Create an EMU3 generator
generator = create_roundtrip_generator(
    model_type="emu3",
    model_path="BAAI/Emu3-Stage1",
    device=0,
    seed=42
)

# Create an OmniGen2 generator
generator = create_roundtrip_generator(
    model_type="omnigen2",
    model_path="/path/to/omnigen2/model",
    device=0,
    seed=42
)

# Create a Janus Pro generator
generator = create_roundtrip_generator(
    model_type="januspro",
    model_path="/path/to/januspro/model",
    device=0,
    seed=42
)

# Create a Show-o2 generator
generator = create_roundtrip_generator(
    model_type="showo2",
    model_path="showlab/show-o2-7b",
    device=0,
    seed=42,
    config_path="../configs/showo2_config.yaml"
)

# Create a Show-o generator
generator = create_roundtrip_generator(
    model_type="showo",
    model_path="/path/to/showo/model",
    device=0,
    seed=42,
    config_path="/path/to/showo/config.yaml"
)
```

# Run full roundtrip generation
results = generator.run_roundtrip_generation(
    prompts_file="prompts.txt",
    output_dir="results",
    start_idx=0,
    end_idx=10
)

# Generate images from captions
results = generator.generate_images_from_captions(
    captions_dir="captions",
    output_dir="caption_images",
    start_idx=0,
    end_idx=5
)
```

## Adding New Models

To add support for a new model:

1. **Create a new implementation file** (e.g., `new_model_roundtrip.py`):
   ```python
   from roundtrip_base import RoundtripGenerator
   
   class NewModelRoundtripGenerator(RoundtripGenerator):
       def _initialize_models(self):
           # Initialize your model-specific components
           pass
       
       def generate_image_from_text(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
           # Implement image generation
           pass
       
       def generate_caption_from_image(self, image: Image.Image, prompt: str = "Describe this image in detail.") -> str:
           # Implement caption generation
           pass
   ```

2. **Register the new generator** in `roundtrip_factory.py`:
   ```python
   from new_model_roundtrip import NewModelRoundtripGenerator
   
   class RoundtripGeneratorFactory:
       _generators = {
           "blip3o": BLIP3oRoundtripGenerator,
           "mmada": MMaDARoundtripGenerator,
           "emu3": EMU3RoundtripGenerator,
           "omnigen2": OmniGen2RoundtripGenerator,
           "januspro": JanusProRoundtripGenerator,
           "showo2": Showo2RoundtripGenerator,  # Add this line
           "showo": ShowoRoundtripGenerator,  # Add this line
           "new_model": NewModelRoundtripGenerator,  # Add this line
       }
   ```

3. **Update the argument parser** in `roundtrip_base.py`:
   ```python
   parser.add_argument("--model_type", required=True, 
                      choices=["blip3o", "mmada", "emu3", "omnigen2", "januspro", "showo2", "showo", "new_model"],  # Add new_model
                      help="Type of model to use")
   ```

## Output Structure

The roundtrip generation creates the following directory structure:

```
output_dir/
├── images/                    # Generated images from prompts
│   ├── prompt_0000_*.jpg
│   ├── prompt_0001_*.jpg
│   └── ...
├── captions/                  # Generated captions
│   ├── prompt_0000_caption.txt
│   ├── prompt_0001_caption.txt
│   └── ...
├── caption_generated_images/  # Images generated from captions (if using --generate_from_captions)
│   ├── caption_0000_*.jpg
│   ├── caption_0001_*.jpg
│   └── ...
├── roundtrip_metadata.json    # Metadata for roundtrip generation (updated iteratively)
└── caption_to_image_metadata.json  # Metadata for caption-to-image generation (updated iteratively)
```

## File Formats

### Prompts File
Prompts should be formatted with separators:
```
=== Prompt 1 ===
A beautiful sunset over the ocean

=== Prompt 2 ===
A cat sitting on a windowsill
```

### Caption Files
Caption files contain both original prompt and generated caption:
```
Original Prompt: A beautiful sunset over the ocean

Generated Caption: A stunning sunset scene with vibrant orange and pink hues reflecting off the calm ocean waters, creating a peaceful and serene atmosphere.
```

### Metadata Files
The system maintains JSON metadata files that are updated iteratively after each successful generation:

**roundtrip_metadata.json** (for roundtrip generation):
```json
{
  "metadata": {
    "total_prompts": 100,
    "successful_generations": 95,
    "start_idx": 0,
    "end_idx": 100,
    "model_path": "/path/to/model",
    "created_at": "2024-01-01T10:00:00",
    "last_updated": "2024-01-01T12:30:00"
  },
  "results": [
    {
      "prompt_id": 0,
      "original_prompt": "A beautiful sunset...",
      "generated_caption": "A stunning sunset scene...",
      "image_path": "images/prompt_0000_*.jpg",
      "caption_path": "captions/prompt_0000_caption.txt",
      "processed_at": "2024-01-01T10:05:00"
    }
  ]
}
```

**caption_to_image_metadata.json** (for caption-to-image generation):
```json
{
  "metadata": {
    "total_captions": 50,
    "successful_generations": 48,
    "start_idx": 0,
    "end_idx": 50,
    "seed_offset": 10000,
    "model_path": "/path/to/model",
    "captions_dir": "/path/to/captions",
    "created_at": "2024-01-01T10:00:00",
    "last_updated": "2024-01-01T11:30:00"
  },
  "results": [
    {
      "prompt_id": "0000",
      "caption_file": "/path/to/captions/prompt_0000_caption.txt",
      "original_prompt": "A beautiful sunset...",
      "generated_caption": "A stunning sunset scene...",
      "generated_image_path": "caption_generated_images/caption_0000_*.jpg",
      "seed": 10000,
      "processed_at": "2024-01-01T10:05:00"
    }
  ]
}
```

## OmniGen2-Specific Features

OmniGen2 is a powerful multimodal generation model with the following capabilities:

### Key Features
- **Visual Understanding**: Inherits robust image interpretation from Qwen-VL-2.5 foundation
- **Text-to-Image Generation**: Creates high-fidelity images from textual prompts
- **Instruction-guided Image Editing**: Executes complex image modifications with high precision
- **In-context Generation**: Processes diverse inputs to produce novel visual outputs

### Usage Tips for OmniGen2
- **Model Path**: Use the path to your OmniGen2 model directory
- **Memory Management**: OmniGen2 requires significant VRAM (~17GB). Consider using CPU offload options if needed
- **Image Quality**: For best results, use high-resolution images (>512×512 pixels) and detailed English prompts
- **Subject Consistency**: For better subject consistency, increase image guidance scale (2.5-3.0) and use larger input images
- **Performance**: Consider reducing `cfg_range_end` for faster inference with minimal quality loss

### Example OmniGen2 Usage
```bash
# Basic roundtrip generation
python roundtrip_generation.py /path/to/omnigen2/model --model_type omnigen2 --prompts_file prompts.txt

# Using the provided shell script
./run_omnigen2_roundtrip.sh

# Programmatic usage
python example_omnigen2_usage.py
```

## Janus Pro-Specific Features

Janus Pro is an advanced version of Janus with significant improvements in both multimodal understanding and visual generation capabilities.

### Key Features
- **Enhanced Multimodal Understanding**: Improved visual interpretation and analysis capabilities
- **Advanced Text-to-Image Generation**: Higher quality and more stable image generation
- **Optimized Training Strategy**: Better training methodology for improved performance
- **Expanded Training Data**: Larger and more diverse training dataset
- **Scaled Model Size**: Available in both 1B and 7B parameter versions

### Usage Tips for Janus Pro
- **Model Path**: Use the path to your Janus Pro model directory
- **Model Variants**: Choose between Janus-Pro-1B (faster) or Janus-Pro-7B (higher quality)
- **Memory Requirements**: 7B model requires more VRAM than the 1B version
- **Conversation Format**: Uses `<|User|>` and `<|Assistant|>` role format
- **Image Quality**: Generates 384x384 images by default with high fidelity

### Example Janus Pro Usage
```bash
# Basic roundtrip generation
python roundtrip_generation.py /path/to/januspro/model --model_type januspro --prompts_file prompts.txt

# Using the provided shell script
./run_januspro_roundtrip.sh

# Programmatic usage
python example_januspro_usage.py
```

## Show-o2-Specific Features

Show-o2 is an improved native unified multimodal model that performs unified learning of multimodal understanding and generation on text tokens and 3D Causal VAE space. It's designed to handle both text-to-image generation and image-to-text understanding in a single unified framework.

### Key Features
- **Unified Architecture**: Single transformer for both understanding and generation tasks
- **3D Causal VAE Space**: Scalable representation for text, image, and video modalities
- **Dual-Path Fusion**: Spatial-temporal fusion accommodating distinct feature dependencies
- **Transport-based Generation**: Advanced sampling methods for high-quality image generation
- **Multiple Model Sizes**: Available in 1.5B and 7B parameter versions

### Architecture Details

#### Model Components
- **VAE Model**: Uses WanVAE for processing images into discrete tokens
- **Text Tokenizer**: Based on Qwen2.5-7B-Instruct for text processing
- **Transport Sampler**: Implements ODE-based sampling for image generation
- **Omni Attention**: Naive attention masking for multimodal fusion

#### Generation Process
1. **Text-to-Image**: Uses transport sampling with classifier-free guidance
2. **Image-to-Text**: Uses multimodal understanding with attention masking
3. **Roundtrip**: Combines both processes for full roundtrip evaluation

### Installation and Setup

#### Prerequisites
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Show-o2 dependencies
pip install omegaconf transformers accelerate diffusers

# Install additional dependencies
pip install wandb tqdm pillow torchdiffeq
```

#### Directory Structure
The Show-o2 implementation expects the following directory structure:
```
univlm/
├── evaluation/           # This directory
│   ├── showo2_roundtrip.py
│   ├── test_showo2_roundtrip.py
│   └── ...
└── Show-o/              # Show-o repository
    └── show-o2/         # Show-o2 specific code
        ├── models/
        ├── utils.py
        ├── transport/
        └── ...
```

#### Model Download
```bash
# Download Show-o2 model (choose one)
# 7B version (higher quality, more memory)
huggingface-cli download showlab/show-o2-7b

# 1.5B version (faster, less memory)
huggingface-cli download showlab/show-o2-1.5b
```

### Usage Tips for Show-o2
- **Model Path**: Use the path to your Show-o2 model (e.g., `showlab/show-o2-7b`)
- **VAE Model**: Uses WanVAE for image processing into discrete tokens
- **Transport-based Generation**: Uses transport sampling for image generation
- **Memory Requirements**: 7B model requires significant VRAM, 1.5B is more memory-efficient
- **Configuration**: Supports custom transport parameters and generation settings
- **Resolution**: Default 256x256 resolution, configurable

### Configuration Options

#### Model Configuration
- `model.weight_type`: Data type for model weights (`bfloat16`, `float32`)
- `model.vae_model.pretrained_model_path`: Path to VAE model
- `model.showo.llm_model_path`: Path to language model
- `model.showo.add_time_embeds`: Whether to add time embeddings

#### Transport Configuration
- `transport.num_inference_steps`: Number of sampling steps (20-100)
- `transport.sampling_method`: Sampling method (`euler`, `dopri5`)
- `transport.guidance_scale`: Classifier-free guidance scale (1.0-10.0)
- `transport.path_type`: Transport path type (`linear`, `cosine`)

#### Generation Parameters
- `roundtrip.max_new_tokens`: Maximum tokens for caption generation
- `roundtrip.top_k`: Top-k sampling for text generation
- `roundtrip.temperature`: Temperature for text generation

### Performance Optimization

#### Memory Management
```python
# For lower memory usage
generator.config.model.weight_type = "float16"
generator.config.transport.num_inference_steps = 20
generator.config.dataset.preprocessing.resolution = 256

# For higher quality
generator.config.transport.num_inference_steps = 100
generator.config.guidance_scale = 7.0
generator.config.dataset.preprocessing.resolution = 512
```

#### Speed Optimization
```python
# Faster generation settings
generator.config.transport.num_inference_steps = 15
generator.config.transport.sampling_method = "euler"
generator.config.guidance_scale = 2.0
```

### Example Show-o2 Usage

#### Basic Roundtrip Generation
```bash
# Using the main script
python roundtrip_generation.py showlab/show-o2-7b \
    --model_type showo2 \
    --config_path ../configs/showo2_config.yaml \
    --prompts_file prompts.txt \
    --output_dir showo2_results \
    --device 0 \
    --seed 42

# Using the provided shell script
./run_showo2_roundtrip.sh \
    -m showlab/show-o2-7b \
    -c ../configs/showo2_config.yaml \
    -p prompts.txt \
    -o showo2_results \
    -d 0 \
    -s 42
```

#### Programmatic Usage
```python
from showo2_roundtrip import Showo2RoundtripGenerator

# Initialize generator
generator = Showo2RoundtripGenerator(
    model_path="showlab/show-o2-7b",
    device=0,
    seed=42,
    config_path="../configs/showo2_config.yaml"
)

# Generate image from text
image = generator.generate_image_from_text(
    "A beautiful sunset over the ocean",
    seed=42
)

# Generate caption from image
caption = generator.generate_caption_from_image(
    image,
    "Describe this image in detail."
)

# Run full roundtrip
results = generator.run_roundtrip_generation(
    prompts_file="prompts.txt",
    output_dir="results",
    start_idx=0,
    end_idx=10
)
```

#### Custom Configuration
```python
# Customize generation parameters
generator.config.transport.num_inference_steps = 25  # Faster generation
generator.config.guidance_scale = 3.0  # Lower guidance = more creative
generator.config.dataset.preprocessing.resolution = 256  # Lower resolution

# Generate with custom settings
image = generator.generate_image_from_text("A magical forest")
```

#### Advanced Examples
```python
# Basic example
generator = Showo2RoundtripGenerator("showlab/show-o2-7b")
prompt = "A cat sitting on a windowsill"
image = generator.generate_image_from_text(prompt)
caption = generator.generate_caption_from_image(image)
print(f"Original: {prompt}")
print(f"Generated: {caption}")

# Batch processing
prompts = ["A mountain landscape", "A city street", "A forest scene"]
for i, prompt in enumerate(prompts):
    image = generator.generate_image_from_text(prompt, seed=42+i)
    caption = generator.generate_caption_from_image(image)
    image.save(f"result_{i}.jpg")
    print(f"{i}: {caption}")
```

### Testing Show-o2
```bash
# Test basic functionality
python test_showo2_roundtrip.py

# Test with custom configuration
python test_showo2_roundtrip.py --custom-config

# Run examples
python example_showo2_usage.py

# Test with provided prompts
python roundtrip_generation.py showlab/show-o2-7b \
    --model_type showo2 \
    --config_path ../configs/showo2_config.yaml \
    --prompts_file showo2_test_prompts.txt \
    --output_dir test_results

### Troubleshooting Show-o2

#### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce memory usage
   generator.config.transport.num_inference_steps = 20
   generator.config.dataset.preprocessing.resolution = 256
   ```

2. **Model Loading Errors**
   ```bash
   # Ensure model path is correct
   ls -la showlab/show-o2-7b/
   
   # Check model files
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('showlab/show-o2-7b')"
   ```

3. **Import Errors**
   ```bash
   # Add Show-o directory to Python path
   export PYTHONPATH="${PYTHONPATH}:/path/to/Show-o:/path/to/Show-o/show-o2"
   ```

#### Debug Mode
```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Check model initialization
print(f"Device: {generator.device}")
print(f"Model type: {type(generator.model)}")
print(f"VAE type: {type(generator.vae_model)}")
```

### References
- [Show-o2 Paper](https://arxiv.org/abs/2506.15564)
- [Show-o2 Repository](https://github.com/showlab/Show-o/tree/main/show-o2)
- [Hugging Face Models](https://huggingface.co/showlab)

## Show-o-Specific Features

Show-o is a unified multimodal model that performs text-to-image generation and image-to-text understanding using a transformer-based architecture with VQ tokenization. It's designed to handle both generation and understanding tasks in a single framework.

### Key Features
- **Unified Architecture**: Single transformer for both understanding and generation tasks
- **VQ Tokenization**: Uses MAGVITv2 for image tokenization and reconstruction
- **Universal Prompting**: Flexible prompting system for different modalities
- **CLIP ViT Support**: Optional CLIP vision tower for enhanced image understanding
- **Mask-based Generation**: Uses mask scheduling for controlled image generation

### Architecture Details

#### Model Components
- **VQ Model**: Uses MAGVITv2 for processing images into discrete tokens
- **Text Tokenizer**: Based on the specified language model (e.g., Phi-1.5)
- **Universal Prompting**: Handles special tokens for different modalities
- **Show-o Model**: Main transformer for unified multimodal processing

#### Generation Process
1. **Text-to-Image**: Uses mask-based generation with noise scheduling
2. **Image-to-Text**: Uses multimodal understanding with attention masking
3. **Roundtrip**: Combines both processes for full roundtrip evaluation

### Installation and Setup

#### Prerequisites
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Show-o dependencies
pip install transformers accelerate omegaconf

# Install additional dependencies
pip install wandb tqdm pillow
```

#### Directory Structure
The Show-o implementation expects the following directory structure:
```
univlm/
├── evaluation/           # This directory
│   ├── showo_roundtrip.py
│   ├── test_showo.py
│   └── ...
└── Show-o/              # Show-o repository
    ├── models/
    ├── training/
    ├── configs/
    └── ...
```

### Usage Tips for Show-o
- **Model Path**: Use the path to your Show-o model checkpoint
- **Config Path**: Required YAML configuration file for model parameters
- **VQ Model**: Uses MAGVITv2 for image processing into discrete tokens
- **Memory Requirements**: Depends on model size and resolution
- **Configuration**: Supports custom generation parameters and model settings
- **CLIP ViT**: Optional feature for enhanced image understanding

### Configuration Options

#### Model Configuration
- `model.showo.pretrained_model_path`: Path to Show-o model
- `model.showo.llm_model_path`: Path to language model
- `model.vq_model.vq_model_name`: Path to VQ model
- `model.showo.w_clip_vit`: Whether to use CLIP ViT vision tower

#### Generation Configuration
- `training.guidance_scale`: Classifier-free guidance scale (0.0-10.0)
- `training.generation_timesteps`: Number of generation timesteps (20-100)
- `training.generation_temperature`: Temperature for generation (0.1-2.0)
- `training.mask_schedule`: Mask scheduling method (`cosine`, `linear`)

#### Dataset Configuration
- `dataset.params.resolution`: Image resolution (256, 512, etc.)
- `dataset.preprocessing.max_seq_length`: Maximum sequence length

### Performance Optimization

#### Memory Management
```python
# For lower memory usage
generator.config.dataset.params.resolution = 256
generator.config.training.generation_timesteps = 20
generator.config.training.guidance_scale = 1.0

# For higher quality
generator.config.training.generation_timesteps = 50
generator.config.training.guidance_scale = 5.0
generator.config.dataset.params.resolution = 512
```

#### Speed Optimization
```python
# Faster generation settings
generator.config.training.generation_timesteps = 15
generator.config.training.guidance_scale = 2.0
generator.config.training.generation_temperature = 0.8
```

### Example Show-o Usage

#### Basic Roundtrip Generation
```bash
# Using the main script
python roundtrip_generation.py /path/to/showo/model \
    --model_type showo \
    --config_path /path/to/config.yaml \
    --prompts_file prompts.txt \
    --output_dir showo_results \
    --device 0 \
    --seed 42

# Using the provided shell script
./run_showo_roundtrip.sh \
    -m /path/to/showo/model \
    -c /path/to/config.yaml \
    -p prompts.txt \
    -o showo_results \
    -d 0 \
    -s 42
```

#### Programmatic Usage
```python
from showo_roundtrip import ShowoRoundtripGenerator

# Initialize generator
generator = ShowoRoundtripGenerator(
    model_path="/path/to/showo/model",
    device=0,
    seed=42,
    config_path="/path/to/config.yaml"
)

# Generate image from text
image = generator.generate_image_from_text(
    "A beautiful sunset over the ocean",
    seed=42
)

# Generate caption from image
caption = generator.generate_caption_from_image(
    image,
    "Describe this image in detail."
)

# Run full roundtrip
results = generator.run_roundtrip_generation(
    prompts_file="prompts.txt",
    output_dir="results",
    start_idx=0,
    end_idx=10
)
```

#### Custom Configuration
```python
# Customize generation parameters
generator.config.training.generation_timesteps = 25  # Faster generation
generator.config.training.guidance_scale = 3.0  # Lower guidance = more creative
generator.config.dataset.params.resolution = 256  # Lower resolution

# Generate with custom settings
image = generator.generate_image_from_text("A magical forest")
```

#### Testing and Examples
```bash
# Test the Show-o implementation
python test_showo.py \
    --model_path /path/to/showo/model \
    --config_path /path/to/config.yaml \
    --device 0

# Run example usage
python example_showo_usage.py \
    --model_path /path/to/showo/model \
    --config_path /path/to/config.yaml \
    --device 0 \
    --example single

# Run with test prompts
python roundtrip_generation.py /path/to/showo/model \
    --model_type showo \
    --config_path /path/to/config.yaml \
    --prompts_file showo_test_prompts.txt \
    --output_dir test_showo_output
```

### Troubleshooting Show-o

#### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce memory usage
   generator.config.training.generation_timesteps = 20
   generator.config.dataset.params.resolution = 256
   ```

2. **Model Loading Errors**
   ```bash
   # Ensure model path is correct
   ls -la /path/to/showo/model/
   
   # Check config file
   python -c "from omegaconf import OmegaConf; OmegaConf.load('/path/to/config.yaml')"
   ```

3. **Import Errors**
   ```bash
   # Add Show-o directory to Python path
   export PYTHONPATH="${PYTHONPATH}:/path/to/Show-o"
   ```

#### Debug Mode
```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Check model initialization
print(f"Device: {generator.device}")
print(f"Model type: {type(generator.model)}")
print(f"VQ model type: {type(generator.vq_model)}")
```

### References
- [Show-o Repository](https://github.com/showlab/Show-o)

## Testing

You can test the implementations using the provided test scripts:

```bash
# Test MMaDA roundtrip generation
python test_mmada.py

# Test EMU3 roundtrip generation
python test_emu3.py

# Test OmniGen2 roundtrip generation
python test_omnigen2.py

# Test Janus Pro roundtrip generation
python test_januspro.py

# Test Show-o2 roundtrip generation
python test_showo2_roundtrip.py

# Test Show-o roundtrip generation
python test_showo.py
```

These will test:
- Model initialization
- Text-to-image generation
- Image-to-text generation
- Full roundtrip generation

## Dependencies

The implementation requires the following dependencies:
- PyTorch
- PIL (Pillow)
- tqdm
- transformers (for BLIP3o and MMaDA)
- diffusers (for BLIP3o)
- Model-specific dependencies (see individual model implementations)

## Notes

- BLIP3o, MMaDA, EMU3, OmniGen2, and Janus Pro implementations are fully functional
- All generators inherit from the base class, ensuring consistent interface and behavior
- The factory pattern makes it easy to add new models without modifying existing code
- Error handling is built into the base class to ensure robust operation
- MMaDA uses MAGVITv2 as the VQ model and supports both text-to-image and image-to-text generation
- EMU3 uses next-token prediction for both image generation and understanding, supporting multiple model variants (Stage1, Chat, Gen)
- OmniGen2 is a powerful multimodal generation model with two distinct decoding pathways for text and image modalities, supporting visual understanding, text-to-image generation, instruction-guided image editing, and in-context generation
- Janus Pro is an advanced unified multimodal model with decoupled visual encoding, supporting both understanding and generation tasks with improved performance over the original Janus
- Metadata files are updated iteratively after each successful generation, allowing for progress tracking and recovery from interruptions
- If a process is interrupted, it can resume from where it left off by loading the existing metadata file
- Show-o is a unified multimodal model that supports both text-to-image generation and image-to-text understanding using VQ tokenization and transformer architecture 