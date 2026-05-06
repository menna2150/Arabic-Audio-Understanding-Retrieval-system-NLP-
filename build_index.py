"""Index a folder of .wav files for semantic search.

Usage:
    python build_index.py --audio-dir <folder> [--limit N] [--out index/]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from config import INDEX_DIR
from src.asr import transcribe
from src.search import AudioSearchIndex, IndexedSegment
from src.summarize import summarize


def build(audio_dir: Path, out_dir: Path, limit: int | None = None,
          skip_summary: bool = False) -> None:
    wavs = sorted(audio_dir.glob("*.wav"))
    if limit:
        wavs = wavs[:limit]
    if not wavs:
        raise SystemExit(f"No .wav files found in {audio_dir}")

    print(f"Indexing {len(wavs)} files from {audio_dir}")
    idx = AudioSearchIndex()
    summaries: dict[str, str] = {}

    for wav in tqdm(wavs, desc="ASR"):
        try:
            asr = transcribe(wav)
        except Exception as e:
            print(f"  ! skip {wav.name}: {e}")
            continue
        segs = [
            IndexedSegment(str(wav), s.start, s.end, s.text)
            for s in asr.segments
            if s.text.strip()
        ]
        idx.add(segs)
        if not skip_summary and asr.text:
            try:
                summaries[str(wav)] = summarize(asr.text)
            except Exception as e:
                print(f"  ! summary failed on {wav.name}: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    idx.save(out_dir)
    if summaries:
        with (out_dir / "summaries.json").open("w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"Saved index ({idx.index.ntotal} segments) to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True, type=Path)
    ap.add_argument("--out", default=INDEX_DIR, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-summary", action="store_true")
    args = ap.parse_args()
    build(args.audio_dir, args.out, args.limit, args.skip_summary)


if __name__ == "__main__":
    main()
