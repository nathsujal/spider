import logging
from PIL import Image
from typing import List
import pytesseract
from pytesseract import Output

logger = logging.getLogger(__name__)

class TesseractOCREngine():

    def __init__(self, min_confidence: int = 30):
        self.min_confidence = min_confidence

    def run(self, image: Image.Image) -> list[str]:
        """Returns a list of paragraph text strings, in reading order."""
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
        return self._group_paragraphs(data)

    def _group_paragraphs(self, data: dict) -> List[str]:
        """
        Groups word-level OCR output into paragraphs.
        Each unique (block_num, par_num) pair becomes one paragraph.
        """
        paragraphs: dict[tuple[int, int], list[str]] = {}

        for i, word in enumerate(data["text"]):
            word = word.strip()
            if not word:
                continue

            conf = int(data["conf"][i])
            if conf < self.min_confidence:
                continue

            key = (data["block_num"][i], data["par_num"][i])
            paragraphs.setdefault(key, []).append(word)

        # join words within each paragraph, preserve reading order
        return [
            " ".join(words)
            for words in paragraphs.values()
            if words
        ]


if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python -m python.engines.tesseract <image_path>")
        sys.exit(1)

    engine = TesseractOCREngine()
    img = Image.open(path)

    t0 = time.perf_counter()
    paragraphs = engine.run(img)
    elapsed = time.perf_counter() - t0

    print(f"\n  Found {len(paragraphs)} paragraph(s) in {elapsed:.2f}s\n")
    for i, p in enumerate(paragraphs, 1):
        print(f"  [{i}] {p}")