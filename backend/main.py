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

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")



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
        # This now returns (audio_path, media_url, title)
        print(f"[INFO] Extracting audio from: {link}")
        processed_audio_path, resolved_media_url, video_title = extract_audio_from_link(link)

        # Denoise the audio
        print(f"[INFO] Denoising audio with strength: {strength}")
        denoised_audio_path = denoise_noisereduce(processed_audio_path, strength)

        print(f"[INFO] Audio denoised: {denoised_audio_path}")

        # Transcribe the denoised audio
        print(f"[INFO] Transcribing audio with model: {model_size}")
        segments_list, transcription_info = transcribe_to_segments(denoised_audio_path, model_size, chunk_length)
        print(f"[INFO] Transcription complete")

        # Clean up the temporary audio files
        if processed_audio_path and os.path.exists(processed_audio_path):
            os.remove(processed_audio_path)
        if denoised_audio_path and os.path.exists(denoised_audio_path):
            os.remove(denoised_audio_path)

        return {
            "message": "Analysis successful",
            "video_title": video_title,
            "resolved_media_url": resolved_media_url,
            "transcription": segments_list,
            "transcription_info": transcription_info
        }
    except Exception as e:
        # Clean up temporary files on error
        try:
            if processed_audio_path and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
            if denoised_audio_path and os.path.exists(denoised_audio_path):
                os.remove(denoised_audio_path)
        except:
            pass

        # Log the exception for debugging
        print(f"[ERROR] An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()

        # Provide more specific error messages
        error_msg = str(e)
        if "allocate" in error_msg.lower() or "memory" in error_msg.lower():
            error_msg = "Memory error during processing. The audio file may be too long or corrupted."

        # Re-raise as an HTTPException to send a specific error response
        raise HTTPException(status_code=500, detail=error_msg)
        # Clean up temporary files on error
        try:
            if processed_audio_path and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)
            if denoised_audio_path and os.path.exists(denoised_audio_path):
                os.remove(denoised_audio_path)
        except:
            pass

        # Log the exception for debugging
        print(f"[ERROR] An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()

        # Provide more specific error messages
        error_msg = str(e)
        if "allocate" in error_msg.lower() or "memory" in error_msg.lower():
            error_msg = "Memory error during processing. The audio file may be too long or corrupted."

        # Re-raise as an HTTPException to send a specific error response
        raise HTTPException(status_code=500, detail=error_msg)
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
