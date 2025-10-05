import os
import subprocess
import uuid
import sys
from pathlib import Path

def _resolve_ffmpeg() -> str:
    ff = os.getenv("FFMPEG_PATH")
    if ff and Path(ff).exists():
        return ff
    repo = Path(__file__).resolve().parents[2]  
    bundled = repo / "bin" / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    if bundled.exists():
        return str(bundled)
    return "ffmpeg"

def url_to_wav(url: str, out_dir: Path, sr: int = 16000) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{uuid.uuid4().hex}.wav"
    ff = _resolve_ffmpeg()
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", url, "-ac", "1", "-ar", str(sr), "-vn", str(out)]
    print("FFMPEG_CMD:", cmd, flush=True)
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"ffmpeg not found (tried: {ff})") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed (exit {e.returncode})") from e
    return out