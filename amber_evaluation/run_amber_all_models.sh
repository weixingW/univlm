#!/bin/bash
#SBATCH --job-name=amber_all_models
#SBATCH --output=amber_output/amber_all_models.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH -w fb10dl03
#SBATCH --ntasks=1
#SBATCH --time=10-00:00:00




cd ~/univlm/amber_evaluation/

source ~/.bashrc
conda activate univlm



set -euo pipefail

# This script runs AMBER caption generation for all supported models using
# univlm/amber_evaluation/amber_generation.py
#
# Configure the paths below (env vars can override):
#   AMBER_IMAGE_DIR     - Path to AMBER images directory
#   AMBER_QUERY_FILE    - Path to AMBER query JSON (query_all.json)
#   OUTPUT_DIR          - Directory to place per-model amber_out_*.json files
#   DEVICE              - CUDA device index passed to --device
#   SEED                - Random seed
#
# Model-specific variables (set to your local paths or HF hub ids):
#   BLIP3O_MODEL_PATH   - Path to BLIP3o model (must include diffusion-decoder subdir)
#   MMADA_MODEL_PATH    - e.g., "Gen-Verse/MMaDA-8B-Base"
#   EMU3_MODEL_PATH     - Use "BAAI/Emu3-Gen" to auto-pick Chat as well, or a local dir with gen/chat subdirs
#   OMNIGEN2_MODEL_PATH - Path to OmniGen2 model directory
#   JANUSPRO_MODEL_PATH - Path to Janus Pro model
#   SHOWO2_MODEL_PATH   - e.g., "showlab/show-o2-7b"
#   SHOWO2_CONFIG_PATH  - Path to Show-o2 YAML config
#   SHOWO_MODEL_PATH    - Path to Show-o model
#   SHOWO_CONFIG_PATH   - Path to Show-o YAML config

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults (override via env)
: "${AMBER_IMAGE_DIR:="/mnt/ssd/AMBER/image"}"
: "${AMBER_QUERY_FILE:="${SCRIPT_DIR}/data/query/query_all.json"}"
: "${OUTPUT_DIR:="${SCRIPT_DIR}/outputs"}"
: "${DEVICE:=0}"
: "${SEED:=42}"
: "${BLIP3O_MODEL_PATH:="~/.huggingface/BLIP3o-Model-8B"}"
: "${MMADA_MODEL_PATH:="Gen-Verse/MMaDA-8B-Base"}"
: "${EMU3_MODEL_PATH:="BAAI/Emu3-Chat"}"
: "${OMNIGEN2_MODEL_PATH:="OmniGen2/OmniGen2"}"
: "${JANUSPRO_MODEL_PATH:="deepseek-ai/Janus-Pro-7B"}"
: "${SHOWO2_MODEL_PATH:="showlab/show-o2-7B"}"
: "${SHOWO_MODEL_PATH:="showlab/show-o"}"

: "${SHOWO2_CONFIG_PATH:="${REPO_ROOT}/configs/showo2.yaml"}"
: "${SHOWO_CONFIG_PATH:="${REPO_ROOT}/configs/showo.yaml"}"

mkdir -p "${OUTPUT_DIR}"

echo "AMBER image dir : ${AMBER_IMAGE_DIR}"
echo "AMBER query file: ${AMBER_QUERY_FILE}"
echo "Output dir      : ${OUTPUT_DIR}"
echo "Device          : ${DEVICE}"
echo "Seed            : ${SEED}"

run_model() {
  local model_type="$1"      # blip3o|mmada|emu3|omnigen2|januspro|showo2|showo
  local model_path="$2"
  local output_file="$3"
  local extra_args=("${@:4}")

  if [[ -z "${model_path}" ]]; then
    echo "[SKIP] ${model_type}: MODEL_PATH not set"
    return 0
  fi

  echo "[RUN] ${model_type} -> ${output_file}"
  python -u "${SCRIPT_DIR}/amber_generation.py" "${model_path}" \
    --model_type "${model_type}" \
    --output_file "${output_file}" \
    --query_file "${AMBER_QUERY_FILE}" \
    --image_dir "${AMBER_IMAGE_DIR}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --skip_existing \
    "${extra_args[@]}"
}

# BLIP3o
: "${BLIP3O_MODEL_PATH:=}"
run_model "blip3o" "${BLIP3O_MODEL_PATH}" "${OUTPUT_DIR}/amber_out_blip3o.json"

# MMaDA
: "${MMADA_MODEL_PATH:=}"
run_model "mmada" "${MMADA_MODEL_PATH}" "${OUTPUT_DIR}/amber_out_mmada.json"

# EMU3 (prefer hub ids: BAAI/Emu3-Gen & BAAI/Emu3-Chat)
: "${EMU3_MODEL_PATH:=BAAI/Emu3-Gen}"
run_model "emu3" "${EMU3_MODEL_PATH}" "${OUTPUT_DIR}/amber_out_emu3.json"

# OmniGen2
: "${OMNIGEN2_MODEL_PATH:=}"
run_model "omnigen2" "${OMNIGEN2_MODEL_PATH}" "${OUTPUT_DIR}/amber_out_omnigen2.json"

# Janus Pro
: "${JANUSPRO_MODEL_PATH:=}"
run_model "januspro" "${JANUSPRO_MODEL_PATH}" "${OUTPUT_DIR}/amber_out_januspro.json"

# Show-o2 (requires config)
: "${SHOWO2_MODEL_PATH:=showlab/show-o2-7b}"
: "${SHOWO2_CONFIG_PATH:=}"
if [[ -n "${SHOWO2_MODEL_PATH}" && -n "${SHOWO2_CONFIG_PATH}" && -f "${SHOWO2_CONFIG_PATH}" ]]; then
  run_model "showo2" "${SHOWO2_MODEL_PATH}" "${OUTPUT_DIR}/amber_out_showo2.json" --config_path "${SHOWO2_CONFIG_PATH}"
else
  echo "[SKIP] showo2: SHOWO2_MODEL_PATH/SHOWO2_CONFIG_PATH not set or config missing"
fi

# Show-o (requires config)
: "${SHOWO_MODEL_PATH:=}"
: "${SHOWO_CONFIG_PATH:=}"
if [[ -n "${SHOWO_MODEL_PATH}" && -n "${SHOWO_CONFIG_PATH}" && -f "${SHOWO_CONFIG_PATH}" ]]; then
  run_model "showo" "${SHOWO_MODEL_PATH}" "${OUTPUT_DIR}/amber_out_showo.json" --config_path "${SHOWO_CONFIG_PATH}"
else
  echo "[SKIP] showo: SHOWO_MODEL_PATH/SHOWO_CONFIG_PATH not set or config missing"
fi

echo "All AMBER runs attempted."


