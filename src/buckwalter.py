"""Buckwalter <-> Arabic Unicode conversion.

The Arabic Speech Corpus (Halabi 2016) ships transcripts in Buckwalter
transliteration. We convert to standard Arabic so we can compare against
modern Arabic ASR/LM outputs.
"""
from __future__ import annotations

# Standard Buckwalter table.  Source: Tim Buckwalter, "Issues in Arabic Orthography
# and Morphology Analysis" (2004).  We use the strict form used by the corpus.
_BW2AR = {
    "'": "ء",  # hamza
    "|": "آ",  # alef with madda
    ">": "أ",  # alef with hamza above
    "&": "ؤ",  # waw with hamza
    "<": "إ",  # alef with hamza below
    "}": "ئ",  # ya with hamza
    "A": "ا",  # alef
    "b": "ب",
    "p": "ة",  # ta marbuta
    "t": "ت",
    "v": "ث",  # tha
    "j": "ج",
    "H": "ح",
    "x": "خ",
    "d": "د",
    "*": "ذ",  # thal
    "r": "ر",
    "z": "ز",
    "s": "س",
    "$": "ش",  # shin
    "S": "ص",
    "D": "ض",
    "T": "ط",
    "Z": "ظ",
    "E": "ع",  # ain
    "g": "غ",  # ghain
    "_": "ـ",  # tatweel
    "f": "ف",
    "q": "ق",
    "k": "ك",
    "l": "ل",
    "m": "م",
    "n": "ن",
    "h": "ه",
    "w": "و",
    "Y": "ى",  # alef maksura
    "y": "ي",
    "F": "ً",  # fathatan
    "N": "ٌ",  # dammatan
    "K": "ٍ",  # kasratan
    "a": "َ",  # fatha
    "u": "ُ",  # damma
    "i": "ِ",  # kasra
    "~": "ّ",  # shadda
    "o": "ْ",  # sukun
    "^": "ٓ",  # madda above (non-standard, used in this corpus)
    "#": "ٔ",
    "`": "ٰ",  # superscript alef
    "{": "ٱ",  # alef wasla
}
_AR2BW = {v: k for k, v in _BW2AR.items()}


def bw_to_ar(text: str) -> str:
    """Convert a Buckwalter string to Arabic."""
    return "".join(_BW2AR.get(c, c) for c in text)


def ar_to_bw(text: str) -> str:
    """Convert Arabic Unicode to Buckwalter."""
    return "".join(_AR2BW.get(c, c) for c in text)


def normalize_arabic(text: str) -> str:
    """Light normalisation used before WER scoring: strip diacritics, unify alef forms."""
    diacritics = {
        "ً", "ٌ", "ٍ", "َ", "ُ",
        "ِ", "ّ", "ْ", "ٓ", "ٰ",
    }
    out = []
    for c in text:
        if c in diacritics:
            continue
        if c in ("آ", "أ", "إ", "ٱ"):
            out.append("ا")  # unify alef forms
        elif c == "ى":
            out.append("ي")  # alef maksura -> ya
        elif c == "ة":
            out.append("ه")  # ta marbuta -> ha (common WER convention)
        else:
            out.append(c)
    return "".join(out).strip()


if __name__ == "__main__":
    sample = "waraj~aHa Alt~aqoriyru"
    ar = bw_to_ar(sample)
    print(sample, "->", ar)
    print("normalised:", normalize_arabic(ar))
