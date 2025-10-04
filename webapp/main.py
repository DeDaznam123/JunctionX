from fastapi import FastAPI, HTTPException
from fastapi import Query
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask
import shutil, os, tempfile
from pathlib import Path
import tempfile

from webapp.utils.video_processing import extract_audio_from_link
from webapp.utils.audio_io import url_to_wav
from webapp.processors.noisereduce_nr import denoise_noisereduce
from webapp.processors.vocal_extract import extract_vocals_hpss

app = FastAPI(title="Extremism-Screener API (minimal preprocess)")

@app.get("/debug_transcode")
def debug_transcode(direct_url: str):
    with tempfile.TemporaryDirectory() as td:
        wav = url_to_wav(direct_url, Path(td), sr=16000)
        return {"ok": True, "wav": str(wav.name)}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/analyze")
def preprocess_link(
    link: str = Query(..., description="Page link to a video/audio (e.g. reddit/youtube)."),
    strength: str = Query("light", regex="^(light|strong)$"),
    download: bool = Query(False, description="If true, return cleaned WAV file; else JSON.")
):
    """
    1) Resolve direct media URL via teammate's yt-dlp helper.
    2) Transcode to 16k mono WAV (ffmpeg).
    3) Denoise with noisereduce (optional).
    """
    try:
        media_url = extract_audio_from_link(link)
    except Exception as e:
        raise HTTPException(400, detail=f"Failed to resolve media URL: {e}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wav = url_to_wav(media_url, td, sr=16000)
        wav = extract_vocals_hpss(wav)
        cleaned = denoise_noisereduce(wav, strength=strength)

        if download:
            outdir = Path("outputs")
            outdir.mkdir(parents=True, exist_ok=True)
            final_path = outdir / cleaned.name
            shutil.copy2(cleaned, final_path)  # move file out of the temp dir

            # optional: auto-delete after response is sent
            task = BackgroundTask(lambda: os.remove(final_path))
            return FileResponse(
                path=str(final_path),
                filename=cleaned.name,
                media_type="audio/wav",
                background=task,   # remove if you want to keep the file
            )

        # else: JSON metadata only
        return {
            "status": "OK",
            "resolved_media_url": media_url,
            "outputs": {"wav_16k": cleaned.name, "note": "Saved in a temp dir only."}
        }
