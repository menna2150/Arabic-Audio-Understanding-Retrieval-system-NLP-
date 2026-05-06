"""Arabic-capable neural summarisation.

Default: csebuetnlp/mT5_multilingual_XLSum (44 languages incl. Arabic, news-style).
"""
from __future__ import annotations
import re
from functools import lru_cache
import torch

from config import (
    SUMMARY_MODEL,
    SUMMARY_MAX_INPUT,
    SUMMARY_MAX_OUTPUT,
    SUMMARY_MIN_OUTPUT,
    DEVICE,
)


_WS = re.compile(r"\s+")


@lru_cache(maxsize=1)
def _load_model(name: str = SUMMARY_MODEL):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.to(DEVICE).eval()
    return tokenizer, model


@torch.inference_mode()
def summarize(text: str, model_name: str = SUMMARY_MODEL,
              max_new_tokens: int = SUMMARY_MAX_OUTPUT,
              min_new_tokens: int = SUMMARY_MIN_OUTPUT) -> str:
    text = _WS.sub(" ", text).strip()
    if not text:
        return ""
    tokenizer, model = _load_model(model_name)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=SUMMARY_MAX_INPUT,
    ).to(DEVICE)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        num_beams=4,
        no_repeat_ngram_size=3,
        length_penalty=0.8,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


if __name__ == "__main__":
    sample = (
        "قال معهد أبحاث هضبة التبت في الأكاديمية الصينية للعلوم إن درجات الحرارة "
        "ومستويات الرطوبة في الارتفاعات العالية ستستمر في الارتفاع طوال هذا القرن، "
        "مما قد يؤدي إلى تراجع مساحات الأنهار الجليدية وانتشار التصحر."
    )
    print(summarize(sample))
