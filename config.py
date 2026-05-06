"""Central config: model names, paths, knobs."""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = (ROOT.parent / "arabic-speech-corpus").resolve()
INDEX_DIR = ROOT / "index"
DATA_DIR = ROOT / "data"
INDEX_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

ASR_MODEL = os.environ.get("ASR_MODEL", "openai/whisper-small")
ASR_LANG = "ar"

SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "csebuetnlp/mT5_multilingual_XLSum")
SUMMARY_MAX_INPUT = 512
SUMMARY_MAX_OUTPUT = 84
SUMMARY_MIN_OUTPUT = 10

EMBED_MODEL = os.environ.get(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBED_DIM = 384

EMOTION_MODEL = os.environ.get("EMOTION_MODEL", "superb/wav2vec2-base-superb-er")

DEVICE = os.environ.get("DEVICE", "cpu")

SAMPLE_RATE = 16_000
SEGMENT_SECONDS = 30
