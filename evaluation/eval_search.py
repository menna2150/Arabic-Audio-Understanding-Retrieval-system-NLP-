"""Precision@K and Recall@K for semantic search over corpus transcripts.

Uses the corpus's orthographic transcript file as the document collection,
and a small set of Arabic queries with hand-labelled relevant doc IDs.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Dict
import numpy as np

from config import CORPUS_DIR
from src.search import AudioSearchIndex, IndexedSegment
from src.buckwalter import bw_to_ar, normalize_arabic


_QUERIES: list[tuple[str, list[str]]] = [
    # Climate / glaciers / desertification block at the start of the corpus
    ("الانهار الجليديه والتصحر بسبب ارتفاع الحراره",
     ["ARA NORM  0002.wav", "ARA NORM  0003.wav", "ARA NORM  0004.wav"]),
    # Football match in Paris
    ("مباراه كره القدم في باريس",
     ["ARA NORM  0006.wav"]),
]


def _read_orthographic(corpus_dir: Path) -> Dict[str, str]:
    """Return {wav_basename: normalised_arabic_text}."""
    transcripts: dict[str, str] = {}
    f = corpus_dir / "orthographic-transcript.txt"
    pattern = re.compile(r'"([^"]+\.wav)"\s+"([^"]*)"')
    with f.open(encoding="utf-8") as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                transcripts[m.group(1)] = normalize_arabic(bw_to_ar(m.group(2)))
    return transcripts


def evaluate(corpus_dir: Path, k: int = 5) -> Dict[str, float]:
    transcripts = _read_orthographic(corpus_dir)

    idx = AudioSearchIndex()
    items = [
        IndexedSegment(audio_path=name, start=0.0, end=0.0, text=text)
        for name, text in transcripts.items()
        if text.strip()
    ]
    print(f"Indexing {len(items)} corpus utterances ...")
    idx.add(items)

    p_at_k, r_at_k = [], []
    for q, relevant in _QUERIES:
        hits = idx.search(q, k=k)
        retrieved = [Path(h.audio_path).name for h in hits]
        retrieved_set = set(retrieved)
        relevant_set = set(relevant)
        tp = len(retrieved_set & relevant_set)
        p = tp / max(1, len(retrieved))
        r = tp / max(1, len(relevant_set))
        p_at_k.append(p); r_at_k.append(r)
        print(f"\nQUERY: {q}")
        for h in hits:
            mark = "*" if Path(h.audio_path).name in relevant_set else " "
            print(f"  {mark} {h.score:.3f}  {Path(h.audio_path).name}: {h.text[:70]}")
        print(f"  P@{k}={p:.2f}  R@{k}={r:.2f}")

    avg = {"P@K": float(np.mean(p_at_k)), "R@K": float(np.mean(r_at_k))}
    print(f"\n=== Avg: P@{k}={avg['P@K']:.3f}  R@{k}={avg['R@K']:.3f} ===")
    return avg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", default=CORPUS_DIR, type=Path)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    evaluate(args.corpus_dir, args.k)


if __name__ == "__main__":
    main()
