from fastapi import FastAPI, HTTPException
from fastapi import Query
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
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

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")



@app.post("/analyze")
def preprocess_link(
    link: str = Query(..., description="Page link to a video/audio (e.g. reddit/youtube)."),
    strength: str = Query("light", regex="^(light|strong)$"),
    model_size: str = Query("small.en"),
    chunk_length: int = Query(30),
):
    try:
        # This now returns (audio_path, media_url, title)
        processed_audio_path, resolved_media_url, video_title = extract_audio_from_link(link)
        
        # Denoise the audio
        denoised_audio_path = denoise_noisereduce(processed_audio_path, strength)
        
        # Transcribe the denoised audio
        transcription_data = transcribe_to_segments(denoised_audio_path, model_size)

        # Clean up the temporary audio files
        os.remove(processed_audio_path)
        os.remove(denoised_audio_path)

        return {
            "message": "Analysis successful",
            "video_title": video_title,
            "resolved_media_url": resolved_media_url,
            "transcription": transcription_data
        }
    except Exception as e:
        # Log the exception for debugging
        print(f"An error occurred during analysis: {e}")
        # Optionally, re-raise as an HTTPException to send a specific error response
        raise HTTPException(status_code=500, detail=str(e))
