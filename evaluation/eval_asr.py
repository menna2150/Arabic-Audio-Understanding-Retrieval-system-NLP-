"""Word Error Rate on the Arabic Speech Corpus.

The corpus transcripts are in Buckwalter; we convert to Arabic and lightly
normalise (strip diacritics, unify alef forms) before scoring against the ASR
output.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
from tqdm import tqdm
from jiwer import wer

from config import CORPUS_DIR
from src.asr import transcribe
from src.buckwalter import bw_to_ar, normalize_arabic


def _read_orthographic(corpus_dir: Path) -> dict[str, str]:
    """Return {wav_basename: buckwalter_transcript}."""
    transcripts: dict[str, str] = {}
    f = corpus_dir / "orthographic-transcript.txt"
    pattern = re.compile(r'"([^"]+\.wav)"\s+"([^"]*)"')
    with f.open(encoding="utf-8") as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                transcripts[m.group(1)] = m.group(2)
    return transcripts


def evaluate(corpus_dir: Path, limit: int | None = None) -> float:
    transcripts = _read_orthographic(corpus_dir)
    wav_dir = corpus_dir / "wav"
    wavs = sorted(wav_dir.glob("*.wav"))
    if limit:
        wavs = wavs[:limit]

    refs, hyps = [], []
    for wav in tqdm(wavs, desc="ASR-eval"):
        ref_bw = transcripts.get(wav.name)
        if not ref_bw:
            continue
        ref = normalize_arabic(bw_to_ar(ref_bw))
        try:
            hyp_text = transcribe(wav).text
        except Exception as e:
            print(f"! skip {wav.name}: {e}")
            continue
        hyp = normalize_arabic(hyp_text)
        if ref and hyp:
            refs.append(ref)
            hyps.append(hyp)
            print(f"\nREF: {ref[:80]}")
            print(f"HYP: {hyp[:80]}")

    if not refs:
        print("No pairs evaluated.")
        return float("nan")
    score = wer(refs, hyps)
    print(f"\n=== WER on {len(refs)} files: {score:.3f} ===")
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default=CORPUS_DIR, type=Path)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()
    evaluate(args.corpus_dir, args.limit)


if __name__ == "__main__":
    main()
