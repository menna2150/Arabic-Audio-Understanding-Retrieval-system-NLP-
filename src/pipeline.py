"""End-to-end orchestrator: audio -> transcript+segments -> summary -> indexable."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from src.asr import transcribe, ASRResult
from src.summarize import summarize
from src.search import IndexedSegment


@dataclass
class PipelineResult:
    audio_path: str
    transcript: str
    summary: str
    segments: List[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def process(audio_path: str | Path, do_summary: bool = True) -> PipelineResult:
    asr: ASRResult = transcribe(audio_path)
    summary = summarize(asr.text) if (do_summary and asr.text) else ""
    return PipelineResult(
        audio_path=str(audio_path),
        transcript=asr.text,
        summary=summary,
        segments=[s.to_dict() for s in asr.segments],
    )


def to_indexed_segments(asr_path: str, asr: ASRResult) -> list[IndexedSegment]:
    return [
        IndexedSegment(audio_path=asr_path, start=s.start, end=s.end, text=s.text)
        for s in asr.segments
        if s.text.strip()
    ]


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("usage: python -m src.pipeline <wav-file>")
        raise SystemExit(2)
    res = process(sys.argv[1])
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
