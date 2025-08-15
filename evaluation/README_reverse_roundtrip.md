# Reverse Roundtrip Generation

This directory contains implementations for reverse roundtrip generation across different multimodal models. The reverse roundtrip process is: **Image → Caption → Reconstructed Image**.

## Overview

Reverse roundtrip generation takes a folder of images, generates captions for each image, and then attempts to reconstruct the original image from the generated caption. This is useful for:

- Evaluating how well models can understand and reconstruct visual content
- Analyzing the fidelity of image-to-text and text-to-image capabilities
- Comparing different models' performance on reverse roundtrip tasks

## Supported Models

The following models are supported for reverse roundtrip generation:

- **BLIP3o**: BLIP3o model for image understanding and generation
- **MMaDA**: MMaDA model for multimodal understanding and generation
- **EMU3**: EMU3 model for image understanding and generation
- **OmniGen2**: OmniGen2 model for image understanding and generation
- **JanusPro**: Janus Pro model for image understanding and generation
- **Showo2**: Show-o2 model for image understanding and generation
- **Showo**: Show-o model for image understanding and generation

## File Structure

```
evaluation/
├── reverse_roundtrip_base.py              # Base class for reverse roundtrip generation
├── reverse_roundtrip_factory.py           # Factory for creating generators
├── reverse_roundtrip_generation.py        # Main script for running reverse roundtrip
├── blip3o_reverse_roundtrip.py           # BLIP3o-specific implementation
├── mmada_reverse_roundtrip.py            # MMaDA-specific implementation
├── emu3_reverse_roundtrip.py             # EMU3-specific implementation
├── omnigen2_reverse_roundtrip.py         # OmniGen2-specific implementation
├── januspro_reverse_roundtrip.py         # JanusPro-specific implementation
├── showo2_reverse_roundtrip.py           # Showo2-specific implementation
├── showo_reverse_roundtrip.py            # Showo-specific implementation
└── README_reverse_roundtrip.md           # This file
```

## Usage

### Command Line Interface

Use the main script to run reverse roundtrip generation:

```bash
python reverse_roundtrip_generation.py <model_path> --model_type <type> --image_dir <path> [options]
```

### Arguments

- `model_path`: Path to the model checkpoint
- `--model_type`: Type of model (blip3o, mmada, emu3, omnigen2, januspro, showo2, showo)
- `--image_dir`: Directory containing input images (required)
- `--output_dir`: Output directory (default: "reverse_roundtrip_results")
- `--start_idx`: Starting image index (default: 0)
- `--end_idx`: Ending image index (default: None, processes all images)
- `--device`: CUDA device ID (default: 0)
- `--seed`: Random seed (default: 42)
- `--seed_offset`: Seed offset for generation (default: 20000)
- `--config_path`: Path to configuration file (optional, required for some models)

### Examples

#### BLIP3o Reverse Roundtrip
```bash
python reverse_roundtrip_generation.py /path/to/blip3o/model \
    --model_type blip3o \
    --image_dir /path/to/images \
    --output_dir blip3o_reverse_results \
    --device 0
```

#### MMaDA Reverse Roundtrip
```bash
python reverse_roundtrip_generation.py /path/to/mmada/model \
    --model_type mmada \
    --image_dir /path/to/images \
    --output_dir mmada_reverse_results \
    --config_path /path/to/mmada_config.yaml \
    --device 0
```

#### EMU3 Reverse Roundtrip
```bash
python reverse_roundtrip_generation.py /path/to/emu3/model \
    --model_type emu3 \
    --image_dir /path/to/images \
    --output_dir emu3_reverse_results \
    --device 0
```

#### Showo2 Reverse Roundtrip
```bash
python reverse_roundtrip_generation.py /path/to/showo2/model \
    --model_type showo2 \
    --image_dir /path/to/images \
    --output_dir showo2_reverse_results \
    --config_path /path/to/showo2_config.yaml \
    --device 0
```

### Programmatic Usage

You can also use the reverse roundtrip generators programmatically:

```python
from reverse_roundtrip_factory import create_reverse_roundtrip_generator

# Create generator
generator = create_reverse_roundtrip_generator(
    model_type="blip3o",
    model_path="/path/to/model",
    device=0,
    seed=42
)

# Run reverse roundtrip generation
results = generator.run_reverse_roundtrip_generation(
    image_dir="/path/to/images",
    output_dir="results",
    start_idx=0,
    end_idx=10,
    seed_offset=20000
)

print(f"Processed {len(results)} images")
```

## Output Structure

The reverse roundtrip generation creates the following output structure:

```
output_dir/
├── reverse_roundtrip_metadata.json       # Metadata about the generation run
├── captions/                             # Generated captions
│   ├── image_0000_original_name_caption.txt
│   ├── image_0001_original_name_caption.txt
│   └── ...
└── reconstructed_images/                 # Reconstructed images
    ├── image_0000_original_name_reconstructed_caption.jpg
    ├── image_0001_original_name_reconstructed_caption.jpg
    └── ...
```

### Caption Files

Each caption file contains:
```
Original Image: original_image_name.jpg

Generated Caption: A detailed description of the image...
```

### Metadata File

The metadata file contains:
```json
{
  "metadata": {
    "total_images": 100,
    "successful_generations": 95,
    "start_idx": 0,
    "end_idx": 100,
    "seed_offset": 20000,
    "model_path": "/path/to/model",
    "image_dir": "/path/to/images",
    "created_at": "2024-01-01T00:00:00",
    "last_updated": "2024-01-01T01:00:00"
  },
  "results": [
    {
      "image_id": 0,
      "original_image_path": "/path/to/images/image1.jpg",
      "original_image_name": "image1.jpg",
      "generated_caption": "A detailed description...",
      "reconstructed_image_path": "/path/to/output/reconstructed_images/...",
      "caption_path": "/path/to/output/captions/...",
      "seed": 20000,
      "processed_at": "2024-01-01T00:00:00"
    }
  ]
}
```

## Model-Specific Requirements

### BLIP3o
- Requires BLIP3o model checkpoint
- No additional configuration needed

### MMaDA
- Requires MMaDA model checkpoint
- Requires configuration file (`--config_path`)
- Uses MAGVITv2 VQ model

### EMU3
- Requires EMU3-Gen and EMU3-Chat models
- Can use HuggingFace model names or local paths
- Automatically handles model switching between generation and understanding

### OmniGen2
- Requires OmniGen2 model checkpoint
- Uses OmniGen2 pipeline for generation and chat

### JanusPro
- Requires Janus Pro model checkpoint
- Uses VLChatProcessor for processing

### Showo2
- Requires Showo2 model checkpoint
- Requires configuration file (`--config_path`)
- Uses WanVAE for image encoding/decoding

### Showo
- Requires Showo model checkpoint
- Requires configuration file (`--config_path`)
- Supports both CLIP ViT and VQ-based vision towers

## Supported Image Formats

The system supports the following image formats:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## Error Handling

The system includes robust error handling:
- Continues processing if individual images fail
- Logs errors for failed generations
- Saves progress after each successful generation
- Can resume from where it left off if interrupted

## Performance Considerations

- Use appropriate batch sizes for your hardware
- Consider using multiple GPUs for large datasets
- Monitor memory usage, especially for large models
- Some models may require specific CUDA versions or dependencies

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all model dependencies are installed
2. **CUDA Out of Memory**: Reduce batch size or use smaller models
3. **Configuration Errors**: Verify config file paths and format
4. **Model Loading Errors**: Check model paths and compatibility

### Getting Help

For issues specific to individual models, refer to their respective documentation:
- BLIP3o: Check BLIP3o directory documentation
- MMaDA: Check MMaDA directory documentation
- EMU3: Check Emu3 directory documentation
- OmniGen2: Check OmniGen2 directory documentation
- JanusPro: Check Janus directory documentation
- Showo2/Showo: Check Show-o directory documentation 