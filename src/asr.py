"""Arabic Speech Recognition with Whisper.

Returns timestamped segments so downstream search can point back to audio
positions.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List
import numpy as np
import torch

from config import ASR_MODEL, ASR_LANG, DEVICE, SAMPLE_RATE
from src.utils import load_audio


@dataclass
class Segment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}


@dataclass
class ASRResult:
    text: str
    segments: List[Segment]
    audio_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "audio_path": self.audio_path,
        }


@lru_cache(maxsize=1)
def _load_model(name: str = ASR_MODEL):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name)
    model.to(DEVICE).eval()
    return processor, model


def _chunked(wav: np.ndarray, chunk_s: float = 30.0):
    n = int(chunk_s * SAMPLE_RATE)
    for i in range(0, len(wav), n):
        yield i / SAMPLE_RATE, wav[i : i + n]


@torch.inference_mode()
def transcribe(audio: str | Path | np.ndarray, model_name: str = ASR_MODEL) -> ASRResult:
    """Transcribe an Arabic audio file or waveform."""
    processor, model = _load_model(model_name)

    if isinstance(audio, (str, Path)):
        wav = load_audio(audio)
        path = str(audio)
    else:
        wav = audio
        path = None

    segments: list[Segment] = []
    for offset_s, chunk in _chunked(wav, 30.0):
        if len(chunk) < SAMPLE_RATE * 0.2:
            continue  # skip <200 ms tail
        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        feats = inputs.input_features.to(DEVICE)
        forced = processor.get_decoder_prompt_ids(language=ASR_LANG, task="transcribe")
        out = model.generate(
            feats,
            forced_decoder_ids=forced,
            max_new_tokens=440,
            num_beams=1,
        )
        text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        if not text:
            continue
        seg_end = offset_s + min(30.0, len(chunk) / SAMPLE_RATE)
        segments.append(Segment(start=offset_s, end=seg_end, text=text))

    full = " ".join(s.text for s in segments).strip()
    return ASRResult(text=full, segments=segments, audio_path=path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m src.asr <wav-file>")
        raise SystemExit(2)
    res = transcribe(sys.argv[1])
    for s in res.segments:
        print(f"[{s.start:6.2f} -> {s.end:6.2f}] {s.text}")
    print("\nFULL:\n", res.text)
