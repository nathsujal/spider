import logging

import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger(__name__)

def _best_device() -> torch.device:
    """Pick the best available accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BLIPEngine:
    """BLIP engine for image captioning."""

    def __init__(self, batch_size: int = 8):
        """
        Initialize BLIP Engine.

        Args:
            batch_size: Batch size for processing multiple objects.
        """
        self.blip_model = "Salesforce/blip-image-captioning-large"
        self.device = _best_device()
        self.batch_size = batch_size

        try:
            self._initialize_captioning_model()
        except Exception as e:
            logger.error(f"Failed to initialize {self.blip_model}: {e}")
            raise RuntimeError(f"BLIP Engine initialization failed: {e}")

        logger.info(
            f"BLIP Engine initialized | Device: {self.device}"
        )

    def _initialize_captioning_model(self):
        """Initialize captioning model."""
        logger.info(f"Loading captioning model: {self.blip_model}")
        self.caption_processor = BlipProcessor.from_pretrained(self.blip_model)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(self.blip_model).to(self.device)
        self.caption_model.eval()

    def describe(
        self,
        image: str | np.ndarray | Image.Image,
        conditional_prompt: str | None = None,
        max_length: int = 30,
        num_beams: int = 3,
    ) -> str:
        """
        Generate caption for image.

        Args:
            image: Input image
            conditional_prompt: Optional text prompt to guide generation
            max_length: Maximum caption length in tokens
            num_beams: Beam search width (higher = better quality, slower)
        
        Returns:
            Generated caption string
        """

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        inputs = self.caption_processor(
            image,
            text=conditional_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.caption_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                return_dict_in_generate=True,
            )

        caption = self.caption_processor.decode(
            outputs.sequences[0],
            skip_special_tokens=True,
        )

        return caption


if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python -m python.engines.vision.blip <image_path>")
        sys.exit(1)

    engine = BLIPEngine()

    # unconditional caption
    t0 = time.perf_counter()
    caption = engine.describe(path)
    elapsed = time.perf_counter() - t0
    print(f"\n  Caption : {caption}")
    print(f"  Time    : {elapsed:.2f}s")

    # conditional caption
    t0 = time.perf_counter()
    guided = engine.describe(path, conditional_prompt="a photograph of")
    elapsed = time.perf_counter() - t0
    print(f"  Guided  : {guided}")
    print(f"  Time    : {elapsed:.2f}s")