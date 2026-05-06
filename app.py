"""Gradio demo: upload Arabic audio, get transcript + summary; query the index.

Run:
    python app.py
"""
from __future__ import annotations
import json
from pathlib import Path
import gradio as gr

from config import INDEX_DIR
from src.pipeline import process
from src.search import AudioSearchIndex


def _load_index() -> AudioSearchIndex | None:
    if (INDEX_DIR / AudioSearchIndex.INDEX_FILE).exists():
        try:
            return AudioSearchIndex.load(INDEX_DIR)
        except Exception as e:
            print(f"index load failed: {e}")
    return None


_INDEX = _load_index()
_index_status = (
    f"Loaded ({_INDEX.index.ntotal} segments)" if _INDEX else "No index. Run build_index.py first."
)


def transcribe_and_summarize(audio_path: str | None):
    if not audio_path:
        return "", "", []
    res = process(audio_path)
    seg_table = [[round(s["start"], 2), round(s["end"], 2), s["text"]] for s in res.segments]
    return res.transcript, res.summary, seg_table


def search(query: str, k: int):
    if not _INDEX or not query.strip():
        return []
    hits = _INDEX.search(query, k=int(k))
    return [
        [round(h.score, 3), Path(h.audio_path).name, round(h.start, 2), round(h.end, 2), h.text]
        for h in hits
    ]


with gr.Blocks(title="Arabic Audio Understanding & Retrieval") as demo:
    gr.Markdown("# 🎙️ Arabic Audio Understanding & Retrieval")
    gr.Markdown(
        "Whisper ASR  →  mT5 Arabic summarisation  →  multilingual semantic search."
    )

    with gr.Tab("Transcribe + Summarise"):
        audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Arabic audio")
        run_btn = gr.Button("Run pipeline")
        transcript_out = gr.Textbox(label="Transcript", lines=4)
        summary_out = gr.Textbox(label="Summary", lines=2)
        seg_out = gr.Dataframe(
            headers=["start", "end", "text"],
            label="Timestamped segments",
            datatype=["number", "number", "str"],
        )
        run_btn.click(
            transcribe_and_summarize,
            inputs=audio_in,
            outputs=[transcript_out, summary_out, seg_out],
        )

    with gr.Tab("Semantic search"):
        gr.Markdown(f"**Index status:** {_index_status}")
        q = gr.Textbox(label="Query (Arabic)")
        k = gr.Slider(1, 20, value=5, step=1, label="Top-K")
        search_btn = gr.Button("Search")
        hits_out = gr.Dataframe(
            headers=["score", "audio", "start", "end", "text"],
            label="Hits",
            datatype=["number", "str", "number", "number", "str"],
        )
        search_btn.click(search, inputs=[q, k], outputs=hits_out)


if __name__ == "__main__":
    demo.launch(share=True)
