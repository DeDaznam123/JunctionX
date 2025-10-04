import soundfile as sf
import numpy as np
import noisereduce as nr
from pathlib import Path

def denoise_noisereduce(wav_path: Path, strength: str = "light") -> Path:
    """
    Spectral gating denoise. Conservative by default to avoid artifacts.
    strength: 'light' or 'strong'
    """
    y, sr = sf.read(wav_path)
    # ensure float32 for stability
    if y.dtype != np.float32:
        y = y.astype(np.float32, copy=False)

    prop = 0.6 if strength == "light" else 0.8
    # non-stationary handles changing background better
    y_dn = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=prop)
    out = wav_path.with_name(wav_path.stem + "_dn.wav")
    sf.write(out, y_dn, sr)
    return out
