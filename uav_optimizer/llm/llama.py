"""Llama 3 interface with Ollama and Groq-compatible API support."""

import json
import os
from typing import Optional
import requests


class LlamaInterface:
    """Interface to Llama 3 via Ollama or a Groq-compatible API."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: Optional[str] = None,
        timeout: int = 60,
    ):
        """Initialize the Llama interface."""
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.timeout = timeout
        self.provider = os.getenv("LLM_PROVIDER", "groq").lower()
        self.api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
        if self.provider == "groq":
            self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        else:
            self.model = model or os.getenv("OLLAMA_MODEL", "llama3")

    def is_available(self) -> bool:
        """Return True when the configured backend is reachable."""
        if self.provider == "groq":
            return bool(self.api_key)

        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def query(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """Send a prompt to the configured LLM backend and return the text response."""
        if self.provider == "groq":
            return self._query_groq(prompt, temperature, max_tokens)

        return self._query_ollama(prompt, temperature, max_tokens)

    def _query_ollama(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Query Ollama's generate endpoint."""
        if not self.is_available():
            raise RuntimeError(f"Ollama server not available at {self.ollama_url}")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
            "num_predict": max_tokens,
        }

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama request failed: {str(e)}") from e

    def _query_groq(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Query a Groq-compatible chat completions endpoint."""
        if not self.api_key:
            raise RuntimeError("No LLM API key configured for Groq-compatible provider")

        payload = {
            "model": self.model or "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a UAV trajectory planning assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Groq request failed: {str(e)}") from e
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Groq response format was unexpected: {str(e)}") from e

    def query_json(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 500,
    ) -> dict:
        """Query the configured backend and parse JSON from the response."""
        response_text = self.query(prompt, temperature, max_tokens)

        try:
            start = response_text.find("{")
            if start >= 0:
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
                return json.loads(json_str)
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from response: {str(e)}") from e

    def __repr__(self) -> str:
        return f"LlamaInterface(provider={self.provider}, model={self.model}, url={self.ollama_url})"
