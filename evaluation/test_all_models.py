#!/usr/bin/env python3
"""
Comprehensive Test Suite for Roundtrip and Reverse Roundtrip Generation

This script tests both roundtrip (Text → Image → Text) and reverse roundtrip
(Image → Text → Image) generation for all supported models:
- BLIP3o
- MMaDA
- EMU3
- OmniGen2
- JanusPro
- Showo2
- Showo

Usage:
    # Test all models with both roundtrip and reverse roundtrip
    python test_all_models.py --all

    # Test specific model
    python test_all_models.py --model_type blip3o --model_path /path/to/model

    # Test only roundtrip generation
    python test_all_models.py --model_type mmada --model_path /path/to/model --test_type roundtrip

    # Test only reverse roundtrip generation
    python test_all_models.py --model_type emu3 --model_path /path/to/model --test_type reverse

    # List all supported models
    python test_all_models.py --list-models
"""

import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from PIL import Image
import traceback

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    model_type: str
    default_model_path: str
    requires_config: bool = False
    default_config_path: Optional[str] = None
    description: str = ""


# Base paths (adjust these to your environment)
HOME_DIR = os.path.expanduser("~")
REPO_ROOT = Path(__file__).parent.parent.absolute()

# Model configurations with pre-filled paths from run_amber_all_models.sh
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "blip3o": ModelConfig(
        name="BLIP3o",
        model_type="blip3o",
        default_model_path=f"{HOME_DIR}/.huggingface",
        requires_config=False,
        description="BLIP3o model for image understanding and generation"
    ),
    "mmada": ModelConfig(
        name="MMaDA",
        model_type="mmada",
        default_model_path="Gen-Verse/MMaDA-8B-Base",
        requires_config=False,
        description="MMaDA model for multimodal understanding and generation"
    ),
    "emu3": ModelConfig(
        name="EMU3",
        model_type="emu3",
        default_model_path="BAAI/Emu3-Chat",
        requires_config=False,
        description="EMU3 model for image understanding and generation"
    ),
    "omnigen2": ModelConfig(
        name="OmniGen2",
        model_type="omnigen2",
        default_model_path="OmniGen2/OmniGen2",
        requires_config=False,
        description="OmniGen2 model for multimodal generation"
    ),
    "januspro": ModelConfig(
        name="JanusPro",
        model_type="januspro",
        default_model_path="deepseek-ai/Janus-Pro-7B",
        requires_config=False,
        description="Janus Pro model for unified understanding and generation"
    ),
    "showo2": ModelConfig(
        name="Show-o2",
        model_type="showo2",
        default_model_path="showlab/show-o2-7B",
        requires_config=True,
        default_config_path=str(REPO_ROOT / "configs" / "showo2_config.yaml"),
        description="Show-o2 unified multimodal model"
    ),
    "showo": ModelConfig(
        name="Show-o",
        model_type="showo",
        default_model_path="showlab/show-o",
        requires_config=True,
        default_config_path=str(REPO_ROOT / "configs" / "showo_config.yaml"),
        description="Show-o unified multimodal model with VQ tokenization"
    ),
}


class TestResult:
    """Container for test results."""
    
    def __init__(self, model_type: str, test_type: str):
        self.model_type = model_type
        self.test_type = test_type
        self.passed = False
        self.error_message: Optional[str] = None
        self.details: Dict[str, Any] = {}
    
    def __str__(self):
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        msg = f"{status}: {self.model_type} - {self.test_type}"
        if self.error_message:
            msg += f"\n    Error: {self.error_message}"
        return msg


class RoundtripTester:
    """Test suite for roundtrip and reverse roundtrip generation."""
    
    def __init__(self, device: int = 0, seed: int = 42, verbose: bool = True):
        self.device = device
        self.seed = seed
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.temp_dir: Optional[Path] = None
    
    def setup_test_environment(self) -> Path:
        """Create a temporary directory with test resources."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="roundtrip_test_"))
        
        # Create test prompts file
        prompts_file = self.temp_dir / "test_prompts.txt"
        prompts_content = """=== Prompt 1 ===
            A beautiful sunset over the ocean with vibrant orange and pink colors

            === Prompt 2 ===
            A fluffy white cat sitting on a cozy windowsill looking outside

            === Prompt 3 ===
            A futuristic city skyline at night with neon lights
            """
        prompts_file.write_text(prompts_content)
        
        # Create test image directory with sample images
        images_dir = self.temp_dir / "test_images"
        images_dir.mkdir(exist_ok=True)
        
        # Create sample test images (simple colored images)
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
        for i, color in enumerate(colors):
            img = Image.new('RGB', (256, 256), color)
            img.save(images_dir / f"test_image_{i:04d}.jpg")
        
        if self.verbose:
            print(f"✓ Created test environment at: {self.temp_dir}")
            print(f"  - Test prompts file: {prompts_file}")
            print(f"  - Test images directory: {images_dir}")
        
        return self.temp_dir
    
    def cleanup_test_environment(self):
        """Clean up the temporary test directory."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            if self.verbose:
                print(f"✓ Cleaned up test environment: {self.temp_dir}")
    
    def log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def test_roundtrip_initialization(self, model_type: str, model_path: str, 
                                       config_path: Optional[str] = None) -> TestResult:
        """Test roundtrip generator initialization."""
        result = TestResult(model_type, "roundtrip_initialization")
        
        try:
            from roundtrip_factory import create_roundtrip_generator
            
            self.log(f"\n--- Testing {model_type} Roundtrip Initialization ---")
            self.log(f"Model path: {model_path}")
            if config_path:
                self.log(f"Config path: {config_path}")
            
            generator = create_roundtrip_generator(
                model_type=model_type,
                model_path=model_path,
                device=self.device,
                seed=self.seed,
                config_path=config_path
            )
            
            result.passed = True
            result.details["generator_type"] = type(generator).__name__
            self.log(f"✓ Successfully initialized {type(generator).__name__}")
            
        except Exception as e:
            result.passed = False
            result.error_message = str(e)
            self.log(f"✗ Failed to initialize: {e}")
            if self.verbose:
                traceback.print_exc()
        
        self.results.append(result)
        return result
    
    def test_reverse_roundtrip_initialization(self, model_type: str, model_path: str,
                                               config_path: Optional[str] = None) -> TestResult:
        """Test reverse roundtrip generator initialization."""
        result = TestResult(model_type, "reverse_roundtrip_initialization")
        
        try:
            from reverse_roundtrip_factory import create_reverse_roundtrip_generator
            
            self.log(f"\n--- Testing {model_type} Reverse Roundtrip Initialization ---")
            self.log(f"Model path: {model_path}")
            if config_path:
                self.log(f"Config path: {config_path}")
            
            generator = create_reverse_roundtrip_generator(
                model_type=model_type,
                model_path=model_path,
                device=self.device,
                seed=self.seed,
                config_path=config_path
            )
            
            result.passed = True
            result.details["generator_type"] = type(generator).__name__
            self.log(f"✓ Successfully initialized {type(generator).__name__}")
            
        except Exception as e:
            result.passed = False
            result.error_message = str(e)
            self.log(f"✗ Failed to initialize: {e}")
            if self.verbose:
                traceback.print_exc()
        
        self.results.append(result)
        return result
    
    def test_text_to_image_generation(self, model_type: str, model_path: str,
                                       config_path: Optional[str] = None) -> TestResult:
        """Test text-to-image generation."""
        result = TestResult(model_type, "text_to_image")
        
        try:
            from roundtrip_factory import create_roundtrip_generator
            
            self.log(f"\n--- Testing {model_type} Text-to-Image Generation ---")
            
            generator = create_roundtrip_generator(
                model_type=model_type,
                model_path=model_path,
                device=self.device,
                seed=self.seed,
                config_path=config_path
            )
            
            test_prompt = "A beautiful landscape with mountains and a lake"
            self.log(f"Generating image from prompt: '{test_prompt}'")
            
            image = generator.generate_image_from_text(test_prompt, seed=self.seed)
            
            if image is not None and isinstance(image, Image.Image):
                result.passed = True
                result.details["image_size"] = image.size
                result.details["image_mode"] = image.mode
                
                # Save test image
                if self.temp_dir:
                    output_path = self.temp_dir / f"{model_type}_text_to_image.jpg"
                    image.save(output_path)
                    result.details["output_path"] = str(output_path)
                
                self.log(f"✓ Generated image: {image.size} ({image.mode})")
            else:
                result.passed = False
                result.error_message = "Generated image is None or invalid type"
                
        except Exception as e:
            result.passed = False
            result.error_message = str(e)
            self.log(f"✗ Failed: {e}")
            if self.verbose:
                traceback.print_exc()
        
        self.results.append(result)
        return result
    
    def test_image_to_text_generation(self, model_type: str, model_path: str,
                                       config_path: Optional[str] = None) -> TestResult:
        """Test image-to-text (caption) generation."""
        result = TestResult(model_type, "image_to_text")
        
        try:
            from roundtrip_factory import create_roundtrip_generator
            
            self.log(f"\n--- Testing {model_type} Image-to-Text Generation ---")
            
            generator = create_roundtrip_generator(
                model_type=model_type,
                model_path=model_path,
                device=self.device,
                seed=self.seed,
                config_path=config_path
            )
            
            # Create a test image
            test_image = Image.new('RGB', (256, 256), (100, 150, 200))
            self.log("Generating caption from test image...")
            
            caption = generator.generate_caption_from_image(
                test_image,
                "Describe this image in detail."
            )
            
            if caption and isinstance(caption, str) and len(caption) > 0:
                result.passed = True
                result.details["caption_length"] = len(caption)
                result.details["caption_preview"] = caption[:100] + "..." if len(caption) > 100 else caption
                self.log(f"✓ Generated caption ({len(caption)} chars): {caption[:100]}...")
            else:
                result.passed = False
                result.error_message = "Generated caption is empty or invalid"
                
        except Exception as e:
            result.passed = False
            result.error_message = str(e)
            self.log(f"✗ Failed: {e}")
            if self.verbose:
                traceback.print_exc()
        
        self.results.append(result)
        return result
    
    def test_full_roundtrip(self, model_type: str, model_path: str,
                            config_path: Optional[str] = None) -> TestResult:
        """Test full roundtrip generation (Text → Image → Text)."""
        result = TestResult(model_type, "full_roundtrip")
        
        try:
            from roundtrip_factory import create_roundtrip_generator
            
            self.log(f"\n--- Testing {model_type} Full Roundtrip Generation ---")
            
            generator = create_roundtrip_generator(
                model_type=model_type,
                model_path=model_path,
                device=self.device,
                seed=self.seed,
                config_path=config_path
            )
            
            if not self.temp_dir:
                self.setup_test_environment()
            
            prompts_file = self.temp_dir / "test_prompts.txt"
            output_dir = self.temp_dir / f"{model_type}_roundtrip_output"
            
            self.log(f"Running roundtrip with prompts from: {prompts_file}")
            self.log(f"Output directory: {output_dir}")
            
            # Run roundtrip generation with just 1 prompt for testing
            results = generator.run_roundtrip_generation(
                prompts_file=str(prompts_file),
                output_dir=str(output_dir),
                start_idx=0,
                end_idx=1
            )
            
            if results and len(results) > 0:
                result.passed = True
                result.details["num_results"] = len(results)
                result.details["output_dir"] = str(output_dir)
                self.log(f"✓ Completed roundtrip generation: {len(results)} results")
            else:
                result.passed = False
                result.error_message = "No results generated"
                
        except Exception as e:
            result.passed = False
            result.error_message = str(e)
            self.log(f"✗ Failed: {e}")
            if self.verbose:
                traceback.print_exc()
        
        self.results.append(result)
        return result
    
    def test_full_reverse_roundtrip(self, model_type: str, model_path: str,
                                     config_path: Optional[str] = None) -> TestResult:
        """Test full reverse roundtrip generation (Image → Text → Image)."""
        result = TestResult(model_type, "full_reverse_roundtrip")
        
        try:
            from reverse_roundtrip_factory import create_reverse_roundtrip_generator
            
            self.log(f"\n--- Testing {model_type} Full Reverse Roundtrip Generation ---")
            
            generator = create_reverse_roundtrip_generator(
                model_type=model_type,
                model_path=model_path,
                device=self.device,
                seed=self.seed,
                config_path=config_path
            )
            
            if not self.temp_dir:
                self.setup_test_environment()
            
            images_dir = self.temp_dir / "test_images"
            output_dir = self.temp_dir / f"{model_type}_reverse_roundtrip_output"
            
            self.log(f"Running reverse roundtrip with images from: {images_dir}")
            self.log(f"Output directory: {output_dir}")
            
            # Run reverse roundtrip generation with just 1 image for testing
            results = generator.run_reverse_roundtrip_generation(
                image_dir=str(images_dir),
                output_dir=str(output_dir),
                start_idx=0,
                end_idx=1,
                seed_offset=20000
            )
            
            if results and len(results) > 0:
                result.passed = True
                result.details["num_results"] = len(results)
                result.details["output_dir"] = str(output_dir)
                self.log(f"✓ Completed reverse roundtrip generation: {len(results)} results")
            else:
                result.passed = False
                result.error_message = "No results generated"
                
        except Exception as e:
            result.passed = False
            result.error_message = str(e)
            self.log(f"✗ Failed: {e}")
            if self.verbose:
                traceback.print_exc()
        
        self.results.append(result)
        return result
    
    def run_all_tests_for_model(self, model_type: str, model_path: str,
                                 config_path: Optional[str] = None,
                                 test_types: Optional[List[str]] = None) -> List[TestResult]:
        """Run all tests for a specific model."""
        if test_types is None:
            test_types = ["roundtrip", "reverse"]
        
        model_results = []
        
        self.log(f"\n{'='*60}")
        self.log(f"Testing Model: {model_type.upper()}")
        self.log(f"{'='*60}")
        
        if "roundtrip" in test_types:
            # Test roundtrip
            model_results.append(
                self.test_roundtrip_initialization(model_type, model_path, config_path)
            )
            model_results.append(
                self.test_text_to_image_generation(model_type, model_path, config_path)
            )
            model_results.append(
                self.test_image_to_text_generation(model_type, model_path, config_path)
            )
            model_results.append(
                self.test_full_roundtrip(model_type, model_path, config_path)
            )
        
        if "reverse" in test_types:
            # Test reverse roundtrip
            model_results.append(
                self.test_reverse_roundtrip_initialization(model_type, model_path, config_path)
            )
            model_results.append(
                self.test_full_reverse_roundtrip(model_type, model_path, config_path)
            )
        
        return model_results
    
    def print_summary(self):
        """Print a summary of all test results."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        # Group results by model type
        results_by_model: Dict[str, List[TestResult]] = {}
        for result in self.results:
            if result.model_type not in results_by_model:
                results_by_model[result.model_type] = []
            results_by_model[result.model_type].append(result)
        
        for model_type, model_results in results_by_model.items():
            model_passed = sum(1 for r in model_results if r.passed)
            model_total = len(model_results)
            status = "✓" if model_passed == model_total else "✗"
            print(f"\n{status} {model_type.upper()}: {model_passed}/{model_total} tests passed")
            
            for result in model_results:
                status_icon = "  ✓" if result.passed else "  ✗"
                print(f"  {status_icon} {result.test_type}")
                if not result.passed and result.error_message:
                    print(f"      Error: {result.error_message[:80]}...")
        
        print(f"\n{'='*60}")
        print(f"TOTAL: {passed}/{total} tests passed ({failed} failed)")
        print("="*60)
        
        return passed == total


def list_supported_models():
    """List all supported models and their configurations."""
    print("\n" + "="*60)
    print("SUPPORTED MODELS")
    print("="*60)
    
    for model_type, config in MODEL_CONFIGS.items():
        print(f"\n{config.name} ({model_type})")
        print(f"  Description: {config.description}")
        print(f"  Default path: {config.default_model_path}")
        print(f"  Requires config: {config.requires_config}")
        if config.requires_config and config.default_config_path:
            print(f"  Default config: {config.default_config_path}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Test roundtrip and reverse roundtrip generation for all models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all supported models
    python test_all_models.py --list-models

    # Test a specific model
    python test_all_models.py --model_type blip3o --model_path /path/to/model

    # Test only roundtrip generation
    python test_all_models.py --model_type mmada --model_path /path/to/model --test_type roundtrip

    # Test only reverse roundtrip generation
    python test_all_models.py --model_type emu3 --model_path /path/to/model --test_type reverse

    # Test all models (uses default paths - may not work without proper setup)
    python test_all_models.py --all

    # Test with config file (required for showo/showo2)
    python test_all_models.py --model_type showo2 --model_path showlab/show-o2-7b --config_path ../configs/showo2_config.yaml
        """
    )
    
    parser.add_argument("--list-models", action="store_true",
                        help="List all supported models")
    parser.add_argument("--all", action="store_true",
                        help="Test all models with default paths")
    parser.add_argument("--model_type", type=str, choices=list(MODEL_CONFIGS.keys()),
                        help="Type of model to test")
    parser.add_argument("--model_path", type=str,
                        help="Path to the model")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to configuration file (required for showo/showo2)")
    parser.add_argument("--test_type", type=str, choices=["roundtrip", "reverse", "both"],
                        default="both", help="Type of test to run")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device ID")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Enable verbose output")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable verbose output")
    parser.add_argument("--cleanup", action="store_true", default=True,
                        help="Clean up temporary test files after testing")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Keep temporary test files after testing")
    
    args = parser.parse_args()
    
    if args.list_models:
        list_supported_models()
        return 0
    
    verbose = args.verbose and not args.quiet
    cleanup = args.cleanup and not args.no_cleanup
    
    # Determine test types
    if args.test_type == "both":
        test_types = ["roundtrip", "reverse"]
    else:
        test_types = [args.test_type]
    
    tester = RoundtripTester(device=args.device, seed=args.seed, verbose=verbose)
    
    try:
        # Setup test environment
        tester.setup_test_environment()
        
        if args.all:
            # Test all models with default paths
            print("\n" + "="*60)
            print("TESTING ALL MODELS")
            print("="*60)
            
            for model_type, config in MODEL_CONFIGS.items():
                config_path = config.default_config_path if config.requires_config else None
                tester.run_all_tests_for_model(
                    model_type=model_type,
                    model_path=config.default_model_path,
                    config_path=config_path,
                    test_types=test_types
                )
        
        elif args.model_type:
            # Test specific model
            model_path = args.model_path
            config_path = args.config_path
            
            # Use default paths if not provided
            if not model_path:
                config = MODEL_CONFIGS.get(args.model_type)
                if config:
                    model_path = config.default_model_path
                    if config.requires_config and not config_path:
                        config_path = config.default_config_path
                else:
                    print(f"Error: Unknown model type: {args.model_type}")
                    return 1
            
            tester.run_all_tests_for_model(
                model_type=args.model_type,
                model_path=model_path,
                config_path=config_path,
                test_types=test_types
            )
        
        else:
            print("Error: Please specify --model_type, --all, or --list-models")
            parser.print_help()
            return 1
        
        # Print summary
        all_passed = tester.print_summary()
        
        return 0 if all_passed else 1
        
    finally:
        if cleanup:
            tester.cleanup_test_environment()


if __name__ == "__main__":
    sys.exit(main())

