import hashlib
from python.models import Chunk, Proposition
from python.intelligence.slm import OllamaClient


SYSTEM_PROMPT = """You are a knowledge extraction engine. Extract atomic facts from the given text.

Each fact must be:
- Self-contained (understandable without the original text)
- Specific (include names, numbers, dates when present)
- One claim per fact (no compound sentences)
- Faithful to the source (no inference or hallucination)

Return a JSON object: {"propositions": ["fact1", "fact2", ...]}"""


class PropositionExtractor:
    """Extracts atomic propositions from chunks using an SLM via Ollama."""

    def __init__(self):
        self.client = OllamaClient()

    def extract(self, chunks: list[Chunk]) -> list[Proposition]:
        """Extract propositions from all chunks.

        Returns a flat list of Proposition objects, each linked
        back to its source chunk via chunk_index.
        """
        if not self.client.is_available():
            print(f"[WARNING] Ollama model '{self.client.model}' not available. Skipping extraction.")
            return []

        all_props: list[Proposition] = []

        for i, chunk in enumerate(chunks):
            try:
                props = self._extract_one(chunk)
                all_props.extend(props)
                print(f"  [{i+1}/{len(chunks)}] chunk {chunk.index} → {len(props)} propositions")
            except Exception as e:
                print(f"  [{i+1}/{len(chunks)}] chunk {chunk.index} FAILED: {e}")
                continue

        return all_props

    def _extract_one(self, chunk: Chunk) -> list[Proposition]:
        """Extract propositions from a single chunk."""
        prompt = f"Extract atomic facts from the following text:\n\n{chunk.text}"

        result = self.client.generate_json(prompt, system_prompt=SYSTEM_PROMPT)

        facts = result.get("propositions", [])

        propositions = []
        for fact in facts:
            if not isinstance(fact, str) or len(fact.strip()) < 10:
                continue  # skip garbage

            propositions.append(Proposition(
                text=fact.strip(),
                chunk_index=chunk.index,
                section_title=chunk.section_title,
                page_num=chunk.page_num,
                content_hash=hashlib.sha256(fact.strip().encode()).hexdigest(),
            ))

        return propositions