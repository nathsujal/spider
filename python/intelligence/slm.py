import json
import logging
import re

from ollama import chat, list as ollama_list, ChatResponse

from .utils import retry

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Thin wrapper around Ollama's chat API with structured JSON output,
    retry logic, and configurable generation parameters.
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        temperature: float = 0.2,
        timeout: float = 120.0,
    ):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        """Simple text generation."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        opts = {"temperature": self.temperature, **kwargs.pop("options", {})}

        response: ChatResponse = chat(
            model=self.model,
            messages=messages,
            options=opts,
            **kwargs,
        )
        return response.message.content

    @retry(retries=3, delay=1, backoff=2)
    def generate_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> dict:
        """Generate structured JSON output.

        Uses Ollama's native format="json" to constrain the model output,
        with a regex fallback for models that still wrap in code fences.
        """
        raw = self.generate(
            prompt,
            system_prompt,
            format="json",
            **kwargs,
        )
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract and parse JSON from SLM output."""
        # Strip markdown code fences if present (shouldn't happen with format="json"
        # but some models still do it)
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # last resort: find first { ... } or [ ... ] block
            match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Could not parse JSON from SLM output:\n{text[:200]}")

    def is_available(self) -> bool:
        """Check if the configured model is pulled and Ollama is running."""
        try:
            models = ollama_list()
            available = [m.model for m in models.models]
            return any(self.model in m for m in available)
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            return False


if __name__ == "__main__":
    client = OllamaClient()
    system_prompt = """You are a knowledge extraction engine. Extract atomic facts 
from the given text. Each fact must be:
- Self-contained (understandable without the original text)
- Specific (include names, numbers, dates)
- One claim per fact (no compound sentences)
Return a JSON object: {"propositions": ["fact1", "fact2", ...]}"""

    chunk = """\
DeepSeekMath-Base is initialized with DeepSeek-Coder-Base-v1.5 7B (Guo et al., 2024), \
as we notice that starting from a code training model is a better choice compared to a general LLM.

Furthermore, we observe the math training also improves model capability on MMLU \
(Hendrycks et al., 2020) and BBH benchmarks (Suzgun et al., 2022), indicating it does \
not only enhance the model's mathematical abilities but also amplifies general reasoning capabilities.

The resulting model DeepSeekMath-Instruct 7B beats all 7B counterparts and is comparable \
with 70B open-source instruction-tuned models.

Furthermore, we introduce the Group Relative Policy Optimization (GRPO), a variant \
reinforcement learning (RL) algorithm of Proximal Policy Optimization (PPO) (Schulman et al., 2017).

GRPO foregoes the critic model, instead estimating the baseline from group scores, \
significantly reducing training resources."""

    prompt = f"Extract propositions from:\n\n{chunk}"

    result = client.generate_json(prompt, system_prompt)
    for i, prop in enumerate(result.get("propositions", []), 1):
        print(f"  {i}. {prop}")