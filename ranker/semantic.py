"""Semantic similarity layer.

Default engine is TF-IDF (scikit-learn): fully offline, deterministic, and fast
enough to vectorise 100K candidates in a few seconds on CPU - which keeps us
comfortably inside the 5-minute / 16 GB / no-network budget.

An optional sentence-transformers engine can be enabled with ``--semantic st``
for denser semantic matching; it requires the model to be cached locally
(no network at ranking time). TF-IDF is the reproducible default.
"""
from __future__ import annotations

import numpy as np

from . import jd


class TfidfSemantic:
    """Character- and word-level TF-IDF cosine similarity against the JD query."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Word n-grams capture phrases like "learning to rank"; sublinear tf and
        # English stop-words reduce the influence of boilerplate.
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.6,
            sublinear_tf=True,
            max_features=200_000,
        )
        self._matrix = None
        self._jd_vec = None

    def fit_transform(self, docs: list[str]) -> None:
        self._matrix = self.vectorizer.fit_transform(docs)
        self._jd_vec = self.vectorizer.transform([jd.JD_SEMANTIC_QUERY])

    def similarities(self) -> np.ndarray:
        """Cosine similarity of every doc to the JD query, scaled to [0, 1]."""
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(self._matrix, self._jd_vec).ravel()
        # TF-IDF cosines are typically small; rescale by a robust max so the
        # component spreads across [0, 1] instead of bunching near zero.
        hi = np.quantile(sims, 0.999) or 1.0
        return np.clip(sims / hi, 0.0, 1.0)


class SentenceTransformerSemantic:
    """Optional dense engine. Requires a locally cached model (no network)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self._emb = None
        self._jd_emb = None

    def fit_transform(self, docs: list[str]) -> None:
        self._emb = self.model.encode(
            docs, batch_size=256, normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._jd_emb = self.model.encode(
            [jd.JD_SEMANTIC_QUERY], normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

    def similarities(self) -> np.ndarray:
        sims = self._emb @ self._jd_emb
        return np.clip((sims + 1.0) / 2.0, 0.0, 1.0)


def build_semantic(engine: str):
    if engine == "st":
        return SentenceTransformerSemantic()
    return TfidfSemantic()
