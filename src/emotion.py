"""Speech-emotion recognition (optional).

Defaults to `superb/wav2vec2-base-superb-er` which classifies audio into
{neu, hap, sad, ang}.  This is an English-trained model — for Arabic-specific
emotion, swap `EMOTION_MODEL` in config.py to an Arabic emotion checkpoint.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Dict
import numpy as np
import torch

from config import EMOTION_MODEL, DEVICE, SAMPLE_RATE
from src.utils import load_audio


@lru_cache(maxsize=1)
def _load(name: str = EMOTION_MODEL):
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    fe = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModelForAudioClassification.from_pretrained(name)
    model.to(DEVICE).eval()
    return fe, model


@torch.inference_mode()
def classify(audio: str | Path | np.ndarray, model_name: str = EMOTION_MODEL) -> Dict[str, float]:
    fe, model = _load(model_name)
    wav = load_audio(audio) if not isinstance(audio, np.ndarray) else audio
    if len(wav) > SAMPLE_RATE * 10:
        wav = wav[: SAMPLE_RATE * 10]  # cap to 10 s
    inputs = fe(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    labels = model.config.id2label
    return {labels[i]: float(probs[i]) for i in range(len(probs))}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m src.emotion <wav>")
        raise SystemExit(2)
    print(classify(sys.argv[1]))
