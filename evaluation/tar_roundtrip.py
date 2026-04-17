import sys
import os

# Ultra-aggressive hack to disable flash_attn if it causes issues
# We must do this BEFORE any other import that might touch transformers/flash_attn
try:
    import flash_attn
except ImportError:
    # If it fails to import (e.g. missing CUDA_HOME), we mock it or set to None
    sys.modules["flash_attn"] = None
except OSError:
     # Also catch OSError which happens when shared query failed
    sys.modules["flash_attn"] = None
except Exception:
    sys.modules["flash_attn"] = None

# Also ensure tri_attn or other similar fused kernels don't break things
os.environ["FLASH_ATTENTION_FORCE_BUILD"] = "FALSE"
os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"

from pathlib import Path
from typing import Optional

import torch
import tempfile
from PIL import Image

from roundtrip_base import RoundtripGenerator
from easydict import EasyDict
import torch.serialization
import argparse
torch.serialization.add_safe_globals([EasyDict, argparse.Namespace])

class TarRoundtripGenerator(RoundtripGenerator):
    def _initialize_models(self):
        repo_root = Path(__file__).resolve().parents[1] / "Tar"
        sys.path.insert(0, str(repo_root))

        try:
            # --- T2I: Lumina2 Dif-DTok (recommended in README) ---
            from t2i_inference_lumina2 import T2IConfig as T2IConfigLumina2, TextToImageInference as T2IInferLumina2
            from i2t_inference import I2TConfig, ImageToTextInference
            from huggingface_hub import snapshot_download, hf_hub_download
        except ImportError as e:
            raise RuntimeError(f"Failed to import Tar modules: {e}. Ensure Tar dependencies are installed.")

        # --- T2I Setup ---
        self.t2i_config = T2IConfigLumina2()
        self.t2i_config.device = f"cuda:{self.device}" if torch.cuda.is_available() else "cpu"
        
        # Override text_encoder for Lumina2 to use a non-gated model if possible, 
        # or rely on env var.
        import os
        env_text_encoder = os.environ.get("TAR_TEXT_ENCODER", None)
        if env_text_encoder:
            self.t2i_config.text_encoder = env_text_encoder

        # Model Path Logic:
        # Expand user path (~) if present
        import os
        if self.model_path:
            self.model_path = os.path.expanduser(self.model_path)

        # If user explicitly provided a path, try to use it if it makes sense.
        if self.model_path and Path(self.model_path).exists():
            if "Lumina" in str(self.model_path):
                self.t2i_config.lumina2_path = str(self.model_path)
            else:
                self.t2i_config.model_path = str(self.model_path)
        
        # Auto-download valid paths if needed
        # 1. Lumina2
        if self.t2i_config.lumina2_path == "csuhan/Tar-Lumina2":
             try:
                 print(f"Downloading Lumina2 from {self.t2i_config.lumina2_path}...")
                 self.t2i_config.lumina2_path = snapshot_download(self.t2i_config.lumina2_path)
             except Exception as e:
                 print(f"Warning: Could not auto-download Lumina2 ({e}). Ensure it is available.")

        # 2. TA-Tok
        if not Path(self.t2i_config.ta_tok_path).exists():
            try:
                print("Downloading TA-Tok path...")
                self.t2i_config.ta_tok_path = hf_hub_download("csuhan/TA-Tok", "ta_tok.pth")
            except Exception:
                pass

        self.t2i = T2IInferLumina2(self.t2i_config)

        # --- I2T Setup ---
        self.i2t_config = I2TConfig()
        self.i2t_config.device = self.t2i_config.device
        
        # Allow overriding I2T LLM via env var (e.g. to avoid gated models if defaults were gated)
        env_i2t_llm = os.environ.get("TAR_I2T_LLM", None)
        if env_i2t_llm:
            self.i2t_config.model_path = env_i2t_llm

        # If user provided a generic model path, assume it might be the base LLM for I2T as well
        if self.model_path and Path(self.model_path).exists() and "Lumina" not in str(self.model_path):
             self.i2t_config.model_path = str(self.model_path)
             
        self.i2t_config.ta_tok_path = self.t2i_config.ta_tok_path
        
        self.i2t = ImageToTextInference(self.i2t_config)
        self._loaded = True

    def generate_image_from_text(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        if seed is None:
            seed = self.seed
            
        # Set seed for reproducibility
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Tar’s inference API
        img = self.t2i.generate_image(prompt)
        
        if not isinstance(img, Image.Image):
            # Check if it's a list (some batch inference return lists)
            if isinstance(img, list) and len(img) > 0:
                img = img[0]
            if not isinstance(img, Image.Image):
                 try:
                     img = Image.fromarray(img)
                 except:
                     pass 
                     
        return img.convert("RGB")

    def generate_caption_from_image(
        self,
        image: Image.Image,
        prompt: str = "Describe the image shortly.",
    ) -> str:
        # i2t_inference.generate expects an image path
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "input.png"
            image.save(p)
            text = self.i2t.generate(str(p), prompt)
            return text.strip() if isinstance(text, str) else str(text)
