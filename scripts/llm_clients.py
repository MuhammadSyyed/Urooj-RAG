from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
import requests

class LLMClient:
    def generate(self, prompt: str, **kwargs: Any) -> str:
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
    base_url: str = os.getenv("OLLAMA_BASE_URL")

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
        
        if "message" in data:
            return data["message"]["content"].strip()
        if "choices" in data:
            return data["choices"][0]["message"]["content"].strip()
        raise ValueError(f"Unexpected Ollama response shape: {data}")

def client_from_env(model: str, provider: Optional[str] = None) -> LLMClient:
    selected = (provider or os.getenv("PROVIDER") or "ollama").lower()
    if selected == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is required for OpenAI provider.")
        base_url = os.getenv("OPENAI_BASE_URL")
        return OpenAIClient(model=model, api_key=api_key, base_url=base_url)
    if selected == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL")
        return OllamaClient(model=model, base_url=base_url)
    raise ValueError(f"Unsupported provider: {selected}")

def safe_generate(client: LLMClient, prompt: str, **kwargs: Any) -> str:
    try:
        return client.generate(prompt, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"LLM generation failed: {exc}") from exc


