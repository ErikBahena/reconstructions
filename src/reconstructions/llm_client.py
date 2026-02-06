"""
LLM client for memory-enhanced reconstruction.

Provides structured access to a local LLM (Ollama) for reranking,
compression, and synthesis operations. All features are optional —
the system degrades gracefully when Ollama is unavailable.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests


@dataclass
class LLMConfig:
    """Configuration for LLM-enhanced reconstruction features."""

    base_url: str = "http://localhost:11434"
    model: str = "gemma3:4b"

    # Feature toggles
    enable_reranking: bool = True
    enable_compression: bool = True
    enable_synthesis: bool = True

    # Per-operation timeouts (seconds)
    rerank_timeout: int = 10
    compress_timeout: int = 8
    synthesis_timeout: int = 15

    # Temperature settings
    rerank_temperature: float = 0.1
    compress_temperature: float = 0.3
    synthesis_temperature: float = 0.5

    # Reranking threshold (0-10 scale)
    rerank_min_score: int = 3

    # Availability cache TTL (seconds)
    availability_cache_ttl: float = 60.0


@dataclass
class LLMResult:
    """Structured result from an LLM operation."""

    success: bool
    content: str = ""
    parsed: Any = None  # For JSON responses
    error: Optional[str] = None
    latency_ms: float = 0.0


class MemoryLLMClient:
    """
    Client for LLM operations in the memory system.

    Uses Ollama's HTTP API. Caches availability checks to avoid
    repeated connection attempts when the server is down.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._available: Optional[bool] = None
        self._available_checked_at: float = 0.0

    def is_available(self) -> bool:
        """
        Check if the LLM server is reachable. Cached with TTL.

        Returns:
            True if server responded within the TTL window
        """
        now = time.time()
        if (self._available is not None and
                now - self._available_checked_at < self.config.availability_cache_ttl):
            return self._available

        try:
            resp = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=3
            )
            self._available = resp.status_code == 200
        except Exception:
            self._available = False

        self._available_checked_at = now
        return self._available

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        timeout: int = 10
    ) -> LLMResult:
        """
        Generate text from the LLM.

        Args:
            prompt: User prompt
            system: System prompt
            temperature: Sampling temperature
            timeout: Request timeout in seconds

        Returns:
            LLMResult with generated text or error
        """
        if not self.is_available():
            return LLMResult(success=False, error="LLM server not available")

        start = time.time()
        try:
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            }
            if system:
                payload["system"] = system

            resp = requests.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                timeout=timeout,
            )
            latency_ms = (time.time() - start) * 1000

            if resp.status_code != 200:
                return LLMResult(
                    success=False,
                    error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                    latency_ms=latency_ms,
                )

            data = resp.json()
            content = data.get("response", "").strip()
            return LLMResult(
                success=True,
                content=content,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return LLMResult(
                success=False,
                error=str(e),
                latency_ms=latency_ms,
            )

    def generate_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        timeout: int = 10
    ) -> LLMResult:
        """
        Generate and parse JSON from the LLM.

        Attempts to extract valid JSON from the response, handling
        markdown code fences and other common wrapping.

        Args:
            prompt: User prompt (should request JSON output)
            system: System prompt
            temperature: Sampling temperature
            timeout: Request timeout in seconds

        Returns:
            LLMResult with parsed JSON in .parsed field
        """
        result = self.generate(prompt, system, temperature, timeout)
        if not result.success:
            return result

        # Try to parse JSON from the response
        text = result.content

        # Strip markdown code fences if present
        if "```json" in text:
            text = text.split("```json", 1)[1]
            if "```" in text:
                text = text.split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1]
            if "```" in text:
                text = text.split("```", 1)[0]

        text = text.strip()

        try:
            parsed = json.loads(text)
            return LLMResult(
                success=True,
                content=result.content,
                parsed=parsed,
                latency_ms=result.latency_ms,
            )
        except json.JSONDecodeError as e:
            return LLMResult(
                success=False,
                content=result.content,
                error=f"JSON parse error: {e}",
                latency_ms=result.latency_ms,
            )


# Module-level singleton
_client: Optional[MemoryLLMClient] = None


def get_llm_client(config: Optional[LLMConfig] = None) -> MemoryLLMClient:
    """
    Get or create the module-level LLM client singleton.

    Args:
        config: Optional config (only used on first call)

    Returns:
        MemoryLLMClient instance
    """
    global _client
    if _client is None:
        _client = MemoryLLMClient(config)
    return _client
