import os
import sys
import json
import argparse
from typing import List, Dict, Tuple
from PIL import Image
import tqdm

# Ensure evaluation modules can be imported (they use local-relative imports)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "evaluation"))
if EVAL_DIR not in sys.path:
    sys.path.append(EVAL_DIR)

from roundtrip_factory import RoundtripGeneratorFactory  # type: ignore  # noqa: E402


def extract_yes_no(line: str) -> str:
    """Heuristic yes/no extractor like amber_generate_example.py."""
    neg_words = ["No", "not", "no", "NO"]
    line = line.replace('.', '').replace(',', '')
    words = line.split(' ')
    if any(word in neg_words for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    return "Yes"


def load_query_data(query_file: str) -> List[Dict]:
    if not os.path.exists(query_file):
        raise FileNotFoundError(f"Query file not found: {query_file}")
    with open(query_file, 'r') as f:
        queries = json.load(f)
    return queries


def get_amber_images_with_queries(image_dir: str, queries: List[Dict]) -> List[Tuple[str, int, str]]:
    image_tuples: List[Tuple[str, int, str]] = []
    for query in queries:
        image_name = query["image"]
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            image_tuples.append((image_path, query["id"], query["query"]))
    sorted_images = sorted(image_tuples, key=lambda x: x[1])
    return sorted_images


def load_existing_captions(output_file: str) -> Dict[int, str]:
    if not os.path.exists(output_file):
        return {}
    with open(output_file, 'r') as f:
        data = json.load(f)
    return {item["id"]: item["response"] for item in data}


def save_amber_response(example_id: int, response: str, output_file: str) -> None:
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            data = json.load(f)
    else:
        data = []
    entry = {"id": example_id, "response": response}
    for i, item in enumerate(data):
        if item["id"] == example_id:
            data[i] = entry
            break
    else:
        data.append(entry)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AMBER evaluation captioning using modular evaluation models")
    parser.add_argument("model_path", type=str, help="Path or hub id for the model")
    parser.add_argument("--model_type", required=True, choices=[
        "blip3o", "mmada", "emu3", "omnigen2", "januspro", "showo2", "showo"
    ], help="Model family to use (see evaluation/README.md)")
    parser.add_argument("--config_path", type=str, default=None, help="Optional config path for models that require it")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--query_file", type=str, default="outputs/amber_output/query_all.json", help="Path to AMBER query file")
    parser.add_argument("--image_dir", type=str, default="/hpi/fs00/share/fg-meinel/weixing.wang/datasets/AMBER/image/", help="Directory containing AMBER images")
    parser.add_argument("--output_file", type=str, required=True, help="Path to write amber_out.json style results")
    parser.add_argument("--skip_existing", action="store_true", help="Skip IDs already present in output file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Create evaluation generator
    generator = RoundtripGeneratorFactory.create_generator(
        model_type=args.model_type,
        model_path=args.model_path,
        device=args.device,
        seed=args.seed,
        config_path=args.config_path,
    )

    # Load AMBER queries and images
    queries = load_query_data(args.query_file)
    image_data = get_amber_images_with_queries(args.image_dir, queries)

    # Load existing results if any
    existing = load_existing_captions(args.output_file) if args.skip_existing else {}

    for image_path, amber_id, query in tqdm.tqdm(image_data, desc="AMBER captioning"):
        if args.skip_existing and amber_id in existing:
            continue

        image = Image.open(image_path).convert("RGB")
        response = generator.generate_caption_from_image(image, prompt=query)

        # Yes/No extraction for VQA-style subset (match example behavior)
        if amber_id >= 1005:
            response = extract_yes_no(response)

        save_amber_response(amber_id, response, args.output_file)


if __name__ == "__main__":
    main()


