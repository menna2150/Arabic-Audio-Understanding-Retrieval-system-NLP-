"""Speaker identification / diarisation (optional).

Two strategies:
  1. `embed(wav)` -> a fixed-dim speaker embedding using SpeechBrain's
     ECAPA / x-vector model.
  2. `diarize(wavs)` -> agglomerative clustering of those embeddings to
     assign speaker IDs across many short clips.

SpeechBrain is imported lazily so the rest of the pipeline works without it.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
import numpy as np

from src.utils import load_audio


def _lazy_encoder():
    try:
        from speechbrain.pretrained import EncoderClassifier
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "speechbrain not installed — `pip install speechbrain` to enable "
            "speaker_id."
        ) from e
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="data/spk_xvect",
        run_opts={"device": "cpu"},
    )


_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = _lazy_encoder()
    return _encoder


def embed(audio: str | Path | np.ndarray) -> np.ndarray:
    import torch
    enc = _get_encoder()
    wav = load_audio(audio) if not isinstance(audio, np.ndarray) else audio
    t = torch.from_numpy(wav).unsqueeze(0)
    return enc.encode_batch(t).squeeze().cpu().numpy()


def diarize(wavs: Iterable[str | Path], n_speakers: int | None = None) -> List[int]:
    """Cluster a list of audio files by speaker.  Returns one label per input."""
    from sklearn.cluster import AgglomerativeClustering

    embs = np.stack([embed(w) for w in wavs])
    if n_speakers is None:
        n_speakers = max(1, min(8, len(embs) // 4))
    clu = AgglomerativeClustering(n_clusters=n_speakers, metric="cosine", linkage="average")
    return clu.fit_predict(embs).tolist()
