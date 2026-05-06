"""Lexical keyword spotting on ASR segments.

Given an ASR result with timestamped segments, return the time intervals where
each keyword appears.  Matching is whitespace-tokenised and substring-aware so
that morphological variants (e.g. "اختبار" vs "الاختبار") still match.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable, List

from src.asr import ASRResult, Segment
from src.buckwalter import normalize_arabic


@dataclass
class KeywordHit:
    keyword: str
    start: float
    end: float
    text: str


def _norm(s: str) -> str:
    s = normalize_arabic(s)
    return re.sub(r"\s+", " ", s).strip()


def spot(asr: ASRResult, keywords: Iterable[str]) -> List[KeywordHit]:
    norm_kws = [(_norm(k), k) for k in keywords if k.strip()]
    hits: list[KeywordHit] = []
    for seg in asr.segments:
        seg_norm = _norm(seg.text)
        for nk, original in norm_kws:
            if nk and nk in seg_norm:
                hits.append(KeywordHit(original, seg.start, seg.end, seg.text))
    return hits


if __name__ == "__main__":
    fake = ASRResult(
        text="هذا اختبار للكلمات المفتاحية",
        segments=[
            Segment(0.0, 2.5, "هذا اختبار"),
            Segment(2.5, 5.0, "للكلمات المفتاحية"),
        ],
    )
    for h in spot(fake, ["اختبار", "deadline"]):
        print(h)
