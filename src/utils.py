"""Audio + text helpers."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable
import numpy as np
import librosa

from config import SAMPLE_RATE


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float32 waveform at `sr` Hz."""
    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    return wav.astype(np.float32)


def chunk_audio(wav: np.ndarray, chunk_s: float = 30.0, sr: int = SAMPLE_RATE) -> list[np.ndarray]:
    """Split into fixed-length chunks (last chunk may be shorter)."""
    n = int(chunk_s * sr)
    return [wav[i : i + n] for i in range(0, len(wav), n)]


def iter_wavs(folder: str | Path) -> Iterable[Path]:
    folder = Path(folder)
    for p in sorted(folder.glob("*.wav")):
        yield p


def clean_text(s: str) -> str:
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()
