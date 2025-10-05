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
from backend.classifiers.ai_classifier import HateSpeechClassifier

clf = HateSpeechClassifier(
    model_name="facebook/bart-large-mnli",
    global_threshold=0.52,
    escape_high_conf=0.82,
    smoothing_window=0,
)

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
    processed_audio_path = None
    denoised_audio_path = None

    processed_audio_path = None
    denoised_audio_path = None

    try:
        print(f"[INFO] Extracting audio from: {link}")
        processed_audio_path, resolved_media_url, video_title = extract_audio_from_link(link)

        print(f"[INFO] Denoising audio with strength: {strength}")
        denoised_audio_path = denoise_noisereduce(processed_audio_path, strength)

        print(f"[INFO] Audio denoised: {denoised_audio_path}")

        print(f"[INFO] Transcribing audio with model: {model_size}")
        segments_list, transcription_info = transcribe_to_segments(denoised_audio_path, model_size, chunk_length)
        print(f"[INFO] Transcription complete")

        if processed_audio_path and os.path.exists(processed_audio_path):
            os.remove(processed_audio_path)
        if denoised_audio_path and os.path.exists(denoised_audio_path):
            os.remove(denoised_audio_path)
        
        try:
            hateful_segments = clf.extract_hateful(segments_list or [])
        except Exception as e:
            print(f"ERROR: classification failed: {e}")
            hateful_segments = []

        print(f"[INFO] Analysis complete. Hateful segments count = {len(hateful_segments)}.")

        return {
            "message": "Analysis successful",
            "video_title": video_title,
            "resolved_media_url": resolved_media_url,
            "transcription": hateful_segments,
            "transcription_info": transcription_info
        }
    except Exception as e:
        try:
            if processed_audio_path and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
            if denoised_audio_path and os.path.exists(denoised_audio_path):
                os.remove(denoised_audio_path)
        except:
            pass

        print(f"[ERROR] An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()

        error_msg = str(e)
        if "allocate" in error_msg.lower() or "memory" in error_msg.lower():
            error_msg = "Memory error during processing. The audio file may be too long or corrupted."
            raise HTTPException(status_code=500, detail=error_msg)
        try:
            if processed_audio_path and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
            if denoised_audio_path and os.path.exists(denoised_audio_path):
                os.remove(denoised_audio_path)
        except:
            pass

        print(f"[ERROR] An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()

        error_msg = str(e)
        if "allocate" in error_msg.lower() or "memory" in error_msg.lower():
            error_msg = "Memory error during processing. The audio file may be too long or corrupted."
    raise HTTPException(status_code=500, detail=error_msg)
