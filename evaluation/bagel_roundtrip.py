import sys
from pathlib import Path
from typing import Optional

from PIL import Image
from roundtrip_base import RoundtripGenerator


class BagelRoundtripGenerator(RoundtripGenerator):
    def _initialize_models(self):
        repo_root = Path(__file__).resolve().parents[1] / "Bagel"
        sys.path.insert(0, str(repo_root))

        # TODO: load Bagel here
        self._loaded = True

    def generate_image_from_text(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        raise NotImplementedError("Implement Bagel text->image")

    def generate_caption_from_image(
        self,
        image: Image.Image,
        prompt: str = "Describe this image in detail.",
    ) -> str:
        raise NotImplementedError("Implement Bagel image->text")
