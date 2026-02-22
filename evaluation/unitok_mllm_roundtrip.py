import sys
import os
from pathlib import Path
from typing import Optional

import torch
import random
import numpy as np
from PIL import Image

from roundtrip_base import RoundtripGenerator
from easydict import EasyDict
import torch.serialization
torch.serialization.add_safe_globals([EasyDict])

class UniTokRoundtripGenerator(RoundtripGenerator):
    def _initialize_models(self):
        import os
        import sys
        import shutil
        import traceback
        from pathlib import Path
        
        # 1. Resolve workspace paths dynamically
        current_script_path = Path(__file__).resolve()
        workspace_dir = current_script_path.parents[2]
        
        repo_root = workspace_dir / "univlm" / "UniTok"
        if not repo_root.exists():
            repo_root = workspace_dir / "UniTok"

        if not repo_root.exists():
            raise RuntimeError(f"UniTok not found at {repo_root}.")

        mllm_dir = repo_root / "UniTokMLLM"
        tokenizer_dir = repo_root / "UniTok"
        lumina_mgpt_subdir = workspace_dir / "univlm" / "Lumina-mGPT" / "lumina_mgpt"
        lumina_root_dir = workspace_dir / "univlm" / "Lumina-mGPT"

        # Push the repo folders to the top of sys.path
        # Order matters: later items in list are inserted first (prepended to sys.path)
        # We want: UniTok > lumina_mgpt > xllmx (lumina_root)
        paths_to_add = [str(lumina_root_dir), str(lumina_mgpt_subdir), str(repo_root), str(tokenizer_dir), str(mllm_dir)]
        for p in paths_to_add:
            if Path(p).exists() and p not in sys.path:
                sys.path.insert(0, p)

        # 1.5. Monkey-Patch transformers to prevent LogitsWarper crash
        try:
            import transformers.generation.logits_process as logits_process
            if not hasattr(logits_process, "LogitsWarper"):
                logits_process.LogitsWarper = logits_process.LogitsProcessor
        except ImportError:
            pass 

        # 2. Identify the core inference directory
        target_import_dir = repo_root
        if (mllm_dir / "inference_solver.py").exists():
            target_import_dir = mllm_dir
        elif (tokenizer_dir / "inference_solver.py").exists():
            target_import_dir = tokenizer_dir
        elif (lumina_mgpt_subdir / "inference_solver.py").exists():
            target_import_dir = lumina_mgpt_subdir

        # 3. Auto-download the missing Meta Chameleon tokenizer weights expected by the codebase
        tokenizer_path = target_import_dir / "ckpts" / "chameleon" / "tokenizer"
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        
        required_files = ["text_tokenizer.json", "vqgan.yaml", "vqgan.ckpt"]
        from huggingface_hub import hf_hub_download
        
        for f in required_files:
            local_f = tokenizer_path / f
            if not local_f.exists():
                print(f"Auto-downloading missing Chameleon tokenizer file: {f}...")
                try:
                    # Fetching the clean Meta Chameleon tokenizer weights hosted on HF
                    cached_f = hf_hub_download(repo_id="Alibaba-DAMO-Academy/WorldVLA", filename=f"chameleon/tokenizer/{f}")
                    shutil.copy(cached_f, local_f)
                except Exception as e:
                    print(f"Warning: Failed to download {f}: {e}")

        # 4. Resolve the model path
        if self.model_path:
            self.model_path = os.path.expanduser(self.model_path)
            model_path_to_load = self.model_path
        else:
            model_path_to_load = "Alpha-VLLM/Lumina-mGPT-7B-768-Omni"

        original_cwd = os.getcwd()
        
        # Trick Python into thinking we are running the script locally so it finds ./ckpts
        os.chdir(target_import_dir)

        try:
            from inference_solver import FlexARInferenceSolver
            
            print(f"Loading unified UniTok MLLM from: {model_path_to_load}...")
            # INITIALIZATION MOVED HERE: Must happen while os.getcwd() matches the ./ckpts folder
            self.solver = FlexARInferenceSolver(
                model_path=model_path_to_load,
                precision="bf16",
                target_size=768, 
            )
            self._loaded = True
            
        except Exception as e:
            print("\n" + "="*50)
            print("RAW ERROR TRACEBACK")
            print("="*50)
            traceback.print_exc()
            print("="*50 + "\n")
            raise RuntimeError(f"Failed to load UniTok: {e}")
        finally:
            # Always revert the terminal directory
            os.chdir(original_cwd)

    def generate_image_from_text(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        if seed is None:
            seed = self.seed
            
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # UniTok relies on a specific prompt format for generation
        query = f"Generate an image of 768x768 according to the following prompt: {prompt}"
        
        # Generate the image
        generated = self.solver.generate(
            images=[],
            qas=[[query, None]],
            max_gen_len=8192,
            temperature=1.0,
            # Note: UniTok favors CFG-free generation (cfg=1.0) for better gFID scores
            logits_processor=self.solver.create_logits_processor(cfg=1.0, image_top_k=2000), 
        )
        
        # The solver returns a tuple: (text_response, [generated_images])
        new_image = generated[1][0]
        
        # Ensure it is a PIL Image
        if not isinstance(new_image, Image.Image):
            try:
                new_image = Image.fromarray(new_image)
            except Exception:
                pass 
                
        return new_image.convert("RGB")

    def generate_caption_from_image(
        self,
        image: Image.Image,
        prompt: str = "Describe the image in detail.",
    ) -> str:
        
        # Following the official UniTok documentation exactly:
        q1 = f"{prompt} <|image|>" if "<|image|>" not in prompt else prompt

        # Their exact generation call
        generated = self.solver.generate(
            images=[image],
            qas=[[q1, None]],
            max_gen_len=8192,
            temperature=1.0,
            logits_processor=self.solver.create_logits_processor(cfg=4.0, image_top_k=2000),
        )
        
        # generated[0] is the text response
        # generated[1] is the list of generated images (should be empty here)
        a1 = generated[0]
        
        return a1.strip() if isinstance(a1, str) else str(a1)