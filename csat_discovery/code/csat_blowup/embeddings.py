"""Pluggable text encoders. Every encoder maps list[str] -> (n, d) float32 array with unit-norm rows.

- "minilm": sentence-transformers all-MiniLM-L6-v2 (the encoder used in the paper; downloads once).
- "hashing": hashed word uni/bi-grams, dependency-free fallback for tests and offline demos. It is a
  bag-of-words stand-in, not a substitute for the paper's encoder.
- "auto": minilm if it can be loaded, else hashing.
Any callable with the same signature can be passed wherever an encoder is expected.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

Encoder = Callable[[list[str]], np.ndarray]
MODEL_NAME = "all-MiniLM-L6-v2"


def _unit(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.where(n == 0, 1, n)).astype(np.float32)


def hashing_encoder(n_features: int = 512) -> Encoder:
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(n_features=n_features, ngram_range=(1, 2), alternate_sign=False, norm="l2")

    def encode(texts: list[str]) -> np.ndarray:
        return _unit(hv.transform([t or "" for t in texts]).toarray())
    return encode


def minilm_encoder(model_name: str = MODEL_NAME, batch_size: int = 64) -> Encoder:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    def encode(texts: list[str]) -> np.ndarray:
        return _unit(model.encode([t or "" for t in texts], batch_size=batch_size,
                                  normalize_embeddings=True, show_progress_bar=False))
    return encode


def get_encoder(name: str = "auto") -> tuple[str, Encoder]:
    """Returns (resolved_name, encoder)."""
    if name in ("minilm", "auto"):
        try:
            return "minilm", minilm_encoder()
        except Exception as e:  # not installed or no network / no cache
            if name == "minilm":
                raise
            print(f"[embeddings] MiniLM unavailable ({type(e).__name__}); using hashing fallback")
    return "hashing", hashing_encoder()
