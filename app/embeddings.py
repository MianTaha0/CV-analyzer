from __future__ import annotations
from typing import List, Any
import os
import math
import hashlib


class BaseEmbedder:
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class LocalEmbedder(BaseEmbedder):
    def __init__(self, dims: int = 2048) -> None:
        self.dims = dims

    def _tokenize(self, text: str) -> List[str]:
        tokens = []
        word = []
        for ch in text.lower():
            if ch.isalnum():
                word.append(ch)
            else:
                if word:
                    tokens.append("".join(word))
                    word = []
        if word:
            tokens.append("".join(word))
        return tokens

    def embed(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return [0.0] * self.dims
        tokens = self._tokenize(text)
        vec = [0.0] * self.dims
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dims
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, api_key: str) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = "text-embedding-3-small"

    def embed(self, text: str) -> List[float]:
        text = (text or "").strip()
        if not text:
            return []
        if len(text) > 8000:
            text = text[:8000]
        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding


_embedder: BaseEmbedder | None = None


def init_embeddings(app: Any) -> None:
    global _embedder
    provider = app.config.get("EMBEDDING_PROVIDER", "local").lower()
    if provider == "openai" and app.config.get("OPENAI_API_KEY"):
        _embedder = OpenAIEmbedder(app.config["OPENAI_API_KEY"])
    else:
        _embedder = LocalEmbedder()
    app.embedder = _embedder


def get_embedder() -> BaseEmbedder:
    if _embedder is None:
        raise RuntimeError("Embedder not initialized")
    return _embedder