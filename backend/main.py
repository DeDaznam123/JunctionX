from fastapi import FastAPI, HTTPException
from fastapi import Query
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
import shutil, os, tempfile
from pathlib import Path
import tempfile

from backend.utils.video_processing import extract_audio_from_link
from backend.utils.audio_io import url_to_wav
from backend.processors.noisereduce_nr import denoise_noisereduce
from backend.processors.vocal_extract import extract_vocals_hpss
from backend.processors.transcribe import transcribe_to_segments

app = FastAPI(title="Extremism Screener API")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/analyze")
def preprocess_link(
    link: str = Query(..., description="Page link to a video/audio (e.g. reddit/youtube)."),
    strength: str = Query("light", regex="^(light|strong)$"),
    model_size: str = Query("small.en"),
):
    try:
        media_url = extract_audio_from_link(link)
    except Exception as e:
        raise HTTPException(400, detail=f"Failed to resolve media URL: {e}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # URL to WAV
        wav = url_to_wav(media_url, td, sr=16000)
        # Extract vocal
        wav = extract_vocals_hpss(wav)
        # Denoise
        cleaned = denoise_noisereduce(wav, strength=strength)
        # Transcribe
        transcript = None
        info = None
        try:
            segs, info = transcribe_to_segments(
                str(cleaned),
                model_size=model_size,
            )
            transcript = segs
        except Exception as e:
            raise HTTPException(500, f"transcription failed: {e}")

        return JSONResponse({
            "status": "OK",
            "resolved_media_url": media_url,
            "outputs": {"wav_16k": cleaned.name},
            "transcription": transcript,
            "info": info,
            "next_steps": [
                "Tune thresholds & add classifier later."
            ]
        })
