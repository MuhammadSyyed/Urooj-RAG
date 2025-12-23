"""
Lightweight LLM client abstractions used across the project.

Supported providers:
- OpenAI-compatible APIs (incl. Azure, local OpenAI-compatible gateways)
- Ollama (local)

Environment variables:
- PROVIDER: "openai" or "ollama" (defaults to "openai")
- OPENAI_API_KEY: required for openai provider
- OPENAI_BASE_URL: optional override (e.g., http://localhost:8000/v1)
- OLLAMA_BASE_URL: optional override (default http://localhost:11434)

Usage:
    from llm_clients import client_from_env
    llm = client_from_env(model="gpt-4o-mini")
    response = llm.generate("Hello?")
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


class LLMClient:
    """Abstract base for LLM providers."""

    def generate(self, prompt: str, **kwargs: Any) -> str:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class OpenAIClient(LLMClient):
    model: str
    api_key: str
    base_url: Optional[str] = None

    def __post_init__(self) -> None:
        from openai import OpenAI  # local import to keep dependency optional

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", 0.2)
        max_tokens = kwargs.get("max_tokens", 512)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()


@dataclass
class OllamaClient(LLMClient):
    model: str
    base_url: str = "http://localhost:11434"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", 0.2)
        resp = requests.post(
            f"{self.base_url.rstrip('/')}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": temperature},
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        # Ollama may return a 'message' dict or a 'choices' list depending on version.
        if "message" in data:
            return data["message"]["content"].strip()
        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        raise ValueError(f"Unexpected Ollama response shape: {data}")


def client_from_env(model: str, provider: Optional[str] = None) -> LLMClient:
    """Return an LLM client based on environment variables."""
    selected = (provider or os.getenv("PROVIDER") or "openai").lower()
    if selected == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is required for OpenAI provider.")
        base_url = os.getenv("OPENAI_BASE_URL")
        return OpenAIClient(model=model, api_key=api_key, base_url=base_url)
    if selected == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaClient(model=model, base_url=base_url)
    raise ValueError(f"Unsupported provider: {selected}")


def safe_generate(client: LLMClient, prompt: str, **kwargs: Any) -> str:
    """Best-effort generation that surfaces concise errors."""
    try:
        return client.generate(prompt, **kwargs)
    except Exception as exc:  # pragma: no cover - pass-through
        raise RuntimeError(f"LLM generation failed: {exc}") from exc


