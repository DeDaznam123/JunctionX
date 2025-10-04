import subprocess, uuid
from pathlib import Path

def url_to_wav(url: str, out_dir: Path, sr: int = 16000) -> Path:
    """Transcodes a remote audio/video URL to mono WAV (sr Hz) using ffmpeg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{uuid.uuid4().hex}.wav"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", url,
        "-ac", "1", "-ar", str(sr),
        "-vn",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out
