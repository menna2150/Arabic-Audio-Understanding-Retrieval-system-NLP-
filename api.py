"""FastAPI server for the Arabic audio pipeline."""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path

from src.pipeline import process
from src.search import AudioSearchIndex

app = FastAPI(title="Arabic Audio Understanding API")
INDEX = None

@app.on_event("startup")
async def load_index():
    global INDEX
    try:
        INDEX = AudioSearchIndex.load("index")
    except FileNotFoundError:
        print("No index found. Build one with: python build_index.py")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Upload audio, get transcript + summary."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = process(tmp_path)
        return result.to_dict()
    finally:
        Path(tmp_path).unlink()

@app.get("/search")
async def search(query: str, k: int = 5):
    """Semantic search over indexed audio."""
    if INDEX is None:
        return JSONResponse({"error": "No index loaded"}, status_code=503)
    hits = INDEX.search(query, k=k)
    return {
        "query": query,
        "hits": [
            {
                "score": h.score,
                "audio_path": h.audio_path,
                "start": h.start,
                "end": h.end,
                "text": h.text,
            }
            for h in hits
        ]
    }

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "index_loaded": INDEX is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
