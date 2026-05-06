"""Semantic search over audio segments.

We embed each ASR segment with a multilingual sentence-transformer, build a FAISS
inner-product index (cosine via L2-normalised vectors), and persist a JSON
sidecar with the segment metadata so a query can return:
    (audio_path, start, end, segment_text, score).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence
import numpy as np
import faiss

from config import EMBED_MODEL, EMBED_DIM, INDEX_DIR, DEVICE


@dataclass
class IndexedSegment:
    audio_path: str
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchHit:
    audio_path: str
    start: float
    end: float
    text: str
    score: float


@lru_cache(maxsize=1)
def _embedder(name: str = EMBED_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=DEVICE)


def embed(texts: Sequence[str], model_name: str = EMBED_MODEL) -> np.ndarray:
    model = _embedder(model_name)
    vecs = model.encode(
        list(texts),
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    return vecs


class AudioSearchIndex:
    """Cosine-similarity FAISS index over ASR segments."""

    INDEX_FILE = "audio.faiss"
    META_FILE = "audio.meta.json"

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.meta: List[IndexedSegment] = []

    # ---- build ---------------------------------------------------------
    def add(self, segments: Sequence[IndexedSegment]) -> None:
        if not segments:
            return
        texts = [s.text for s in segments]
        vecs = embed(texts)
        if vecs.shape[1] != self.dim:
            # adjust on first add (handles model dim mismatch)
            self.dim = vecs.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)
        self.meta.extend(segments)

    # ---- query ---------------------------------------------------------
    def search(self, query: str, k: int = 5) -> List[SearchHit]:
        if self.index.ntotal == 0:
            return []
        qv = embed([query])
        scores, idx = self.index.search(qv, min(k, self.index.ntotal))
        hits: list[SearchHit] = []
        for score, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            m = self.meta[i]
            hits.append(SearchHit(m.audio_path, m.start, m.end, m.text, float(score)))
        return hits

    # ---- persistence ---------------------------------------------------
    def save(self, folder: str | Path = INDEX_DIR) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / self.INDEX_FILE))
        with (folder / self.META_FILE).open("w", encoding="utf-8") as f:
            json.dump(
                {"dim": self.dim, "items": [m.to_dict() for m in self.meta]},
                f, ensure_ascii=False,
            )

    @classmethod
    def load(cls, folder: str | Path = INDEX_DIR) -> "AudioSearchIndex":
        folder = Path(folder)
        with (folder / cls.META_FILE).open(encoding="utf-8") as f:
            blob = json.load(f)
        obj = cls(dim=blob["dim"])
        obj.index = faiss.read_index(str(folder / cls.INDEX_FILE))
        obj.meta = [IndexedSegment(**m) for m in blob["items"]]
        return obj
