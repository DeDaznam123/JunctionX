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
from backend.classifiers.ai_classifier import HateSpeechClassifier
clf = HateSpeechClassifier(
    global_threshold=0.57,
    label_thresholds={"Racist": 0.62, "Personal Attacks": 0.52},
    smoothing_window=1,            # look at previous and next segment
    smoothing_mode="weighted",     # "mean" is simpler; "weighted" often best
    smoothing_weight_center=0.6,   # center more important
    smoothing_weight_neighbor=0.2, # each neighbor
    short_segment_penalty=0.05     # combats 1–2 word spikes
)

app = FastAPI(title="Extremism Screener API")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/analyze")
def preprocess_link(
    link: str = Query(..., description="Page link to a video/audio (e.g. reddit/youtube)."),
    strength: str = Query("light", regex="^(light|strong)$"),
    model_size: str = Query("small.en"),
    chunk_length: int = Query(30),
):
    try:
        media_url = extract_audio_from_link(link)
        print("INFO: Medial URL extracted")
    except Exception as e:
        raise HTTPException(400, detail=f"Failed to resolve media URL: {e}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # URL to WAV
        wav = url_to_wav(media_url, td, sr=16000)
        print("INFO: Video downloaded from URL and transformed to .WAV")
        # Extract vocal
        wav = extract_vocals_hpss(wav)
        print("INFO: Vocal extracted from audio")
        # Denoise
        cleaned = denoise_noisereduce(wav, strength=strength)
        print("INFO: Audio denoised")
        # Transcribe
        transcript = None
        info = None
        try:
            segs, info = transcribe_to_segments(
                str(cleaned),
                model_size=model_size,
                chunk_length=chunk_length,
            )
            transcript = segs
        except Exception as e:
            raise HTTPException(500, f"transcription failed: {e}")
        info("INFO: Audio transcribed")
        # Classify
        try:
            hateful_segments = clf.extract_hateful(transcript or [])
        except Exception as e:
            # Don't break your existing flow; just log and return empty list
            print(f"ERROR: classification failed: {e}")
            hateful_segments = []

        return JSONResponse({
            "status": "OK",
            "resolved_media_url": media_url,
            "outputs": {"wav_16k": cleaned.name},
            "transcription": transcript,
            "hateful_segments": hateful_segments,   # <— added field (non-breaking)
            "info": info,
            "next_steps": [
                "Tune thresholds & add classifier later."
            ]
        })
