"""ROUGE evaluation for the summariser.

We ship a tiny built-in evaluation set so the script runs without any external
download.  Plug in a real Arabic news-summary dataset (e.g. XL-Sum Arabic split,
or the Arabic News Summary Dataset) by replacing `_BUILTIN_PAIRS`.
"""
from __future__ import annotations
import argparse
from rouge_score import rouge_scorer

from src.summarize import summarize


_BUILTIN_PAIRS = [
    (
        "أعلنت وزارة الصحة عن انخفاض كبير في عدد الإصابات اليومية بفيروس كورونا "
        "للأسبوع الثالث على التوالي، مشيرةً إلى أن حملات التطعيم وصلت إلى ما يقارب "
        "ثمانين بالمئة من السكان البالغين، وأن المستشفيات استعادت طاقتها الاستيعابية "
        "الطبيعية بعد أشهر من الضغط الشديد. كما حذرت الوزارة من التراخي في الإجراءات "
        "الاحترازية مع اقتراب فصل الشتاء.",
        "انخفاض إصابات كورونا للأسبوع الثالث وتطعيم 80% من البالغين، مع تحذير من التراخي شتاءً.",
    ),
    (
        "في الشوط الثاني من المباراة استعاد فريق باريس سان جيرمان توازنه وحقق "
        "هدف التعادل عن طريق إبراهيموفيتش من تسديدة قوية استعصت على الحارس، قبل "
        "أن يسجل لاعب الوسط هدف الفوز في الدقائق الأخيرة ليحسم النتيجة لصالح فريقه.",
        "باريس سان جيرمان يقلب الطاولة في الشوط الثاني بهدفي إبراهيموفيتش ولاعب الوسط ويحسم المباراة.",
    ),
    (
        "أكد تقرير معهد أبحاث هضبة التبت أن درجات الحرارة ومستويات الرطوبة في "
        "الارتفاعات العالية ستستمر في الصعود طوال هذا القرن، وهو ما يهدد بتراجع "
        "مساحات الأنهار الجليدية وانتشار التصحر، وقد يؤثر على إمدادات المياه لعدد "
        "من الأنهار الآسيوية الرئيسية النابعة من الهضبة، من بينها نهرا يلو ويانغتسي.",
        "ارتفاع متواصل لدرجات الحرارة في هضبة التبت يهدد الأنهار الجليدية وإمدادات الأنهار الآسيوية.",
    ),
]


def evaluate(pairs=_BUILTIN_PAIRS) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for src, ref in pairs:
        hyp = summarize(src)
        scores = scorer.score(ref, hyp)
        for k in agg:
            agg[k] += scores[k].fmeasure
        print(f"\nREF: {ref}\nHYP: {hyp}")
        print({k: round(v.fmeasure, 3) for k, v in scores.items()})
    n = len(pairs)
    avg = {k: v / n for k, v in agg.items()}
    print(f"\n=== ROUGE on {n} samples: " +
          ", ".join(f"{k}={v:.3f}" for k, v in avg.items()))
    return avg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    evaluate()


if __name__ == "__main__":
    main()
