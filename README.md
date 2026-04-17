# UniVLM - Unified Vision-Language Models

A comprehensive collection of vision-language models and implementations for multimodal AI research and applications.

## Overview

This repository contains implementations and evaluations of various state-of-the-art unified vision-language models including:

- **BLIP3o**
- **Show-o**
- **Janus**
- **OmniGen2**
- **Emu3**
- **MMaDA**
- **Transfusion**



## Getting Started

### Prerequisites

- Conda (Miniconda or Anaconda)
- CUDA 12.6 compatible GPU

### Installation

1. Clone this repository with all submodules:
```bash
git clone --recursive https://github.com/weixingW/univlm.git
cd univlm
```

**Note:** If you already cloned without `--recursive`, you can initialize and update submodules with:
```bash
git submodule update --init --recursive
```

2. Create the conda environment from `environment.yml`:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate univlm
```


### Updating the Environment

If `environment.yml` is updated, you can sync your environment with:
```bash
conda env update -f environment.yml --prune
```

## Usage

Each model implementation has its own directory with specific usage instructions. Please refer to the README files in each subdirectory for detailed usage examples.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite the original papers of the respective models.

## Acknowledgments

- Original model authors and research teams
- Open-source community contributors 


## Integration of Tar, Bagel and Unitok for existing users of the main repo

# Add fork as a new remote
git remote add conscht https://github.com/Conscht/univlm.git
git fetch conscht

# Switch to the model branch
git checkout -b add-models conscht/add-models

# CRITICAL: Initialize the new submodules you added
git submodule update --init --recursive

## Additional requirements for TAR

You need to get access to gemma (and flux granted instandly) and set your HF token:

https://huggingface.co/black-forest-labs/FLUX.1-dev

https://huggingface.co/google/gemma-2-2b-it

export HUGGINGFACE_HUB_TOKEN="hf_...yourtoken..."

Runs with univlm env

## For Bagle & UniTok

Should be downloaded and run automatically