# Deep-Learning-Based Arabic Audio Understanding & Retrieval

End-to-end pipeline that turns Arabic audio into searchable, summarised knowledge.

```
audio  -->  ASR (Whisper)  -->  transcript  -->  summarise (mT5/AraBART)
                                    |
                                    +-->  segment embeddings (multilingual ST) --> FAISS index
                                                                                       |
                                                          query (text)  ----->  semantic search
```

## Layout

```
nlp_project/
  src/
    asr.py            Whisper-based Arabic ASR with timestamped segments
    summarize.py      Arabic-capable seq2seq summariser (mT5 XL-Sum default)
    search.py         Sentence-transformer + FAISS index over audio segments
    pipeline.py       Glue: audio -> transcript+segments -> summary -> indexable
    buckwalter.py     Buckwalter <-> Arabic converter (corpus uses Buckwalter)
    speaker_id.py     Optional speaker diarisation/clustering (x-vectors)
    emotion.py        Optional speech-emotion classifier (wav2vec2-er)
    keyword.py        Keyword spotting on transcripts with timestamps
    utils.py          Audio loading, chunking helpers
  evaluation/
    eval_asr.py       WER on the Arabic Speech Corpus
    eval_summary.py   ROUGE
    eval_search.py    Precision@K / Recall@K
  build_index.py      Index a folder of .wav files
  app.py              Gradio demo
  config.py           Model names, paths, knobs
  requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Build a search index from the local corpus (or any folder of .wav)
python build_index.py --audio-dir "../arabic-speech-corpus/wav" --limit 50

# 2. Launch the demo
python app.py

# 3. Run evaluations
python -m evaluation.eval_asr      --limit 20
python -m evaluation.eval_summary
python -m evaluation.eval_search
```

The first run downloads pretrained weights from Hugging Face (Whisper, mT5/MiniLM,
sentence-transformers). Subsequent runs use the local cache.

## Models used (default; all pretrained, swap-able in `config.py`)

| Stage           | Model                                                       |
| --------------- | ----------------------------------------------------------- |
| ASR             | `openai/whisper-small` (Arabic supported)                   |
| Summarisation   | `csebuetnlp/mT5_multilingual_XLSum`                         |
| Embeddings      | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Emotion (opt.)  | `superb/wav2vec2-base-superb-er`                            |
| Speaker (opt.)  | `speechbrain/spkrec-xvect-voxceleb` (lazy import)           |

## Dataset

Primary: **Arabic Speech Corpus** (Halabi 2016) — 1813 utterances, MSA, single speaker.
Transcripts ship in **Buckwalter** transliteration; `src/buckwalter.py` converts to UTF-8
Arabic so we can train/evaluate against modern Arabic LMs.

Other supported datasets (drop-in via `--audio-dir`):
- Mozilla Common Voice (Arabic)
- MASC
- Arabic Broadcast News (LDC2006S46)

## Evaluation metrics

- **ASR**: Word Error Rate (`jiwer`)
- **Summarisation**: ROUGE-1/2/L (`rouge-score`)
- **Search**: Precision@K, Recall@K on a handcrafted query set

## Deliverables map (per project brief)

| Brief item              | Where                                      |
| ----------------------- | ------------------------------------------ |
| Source code             | `src/`, `app.py`, `build_index.py`         |
| Dataset description     | This README + `evaluation/eval_*.py`       |
| Architecture diagram    | Top of this README                         |
| Experiments             | `evaluation/`                              |
| Evaluation results      | Printed by each `eval_*.py` script         |
| Demo interface          | `app.py` (Gradio)                          |
