import soundfile as sf
import numpy as np
import noisereduce as nr
from pathlib import Path

def denoise_noisereduce(wav_path, strength: str = "light"):
    """
    Spectral gating denoise. Conservative by default to avoid artifacts.
    strength: 'light' or 'strong'
    wav_path: can be either a string or Path object
    Returns: Path to the denoised audio file
    """
    if isinstance(wav_path, str):
        wav_path = Path(wav_path)

    y, sr = sf.read(wav_path)

    if y.size == 0:
        raise ValueError("Audio file is empty")

    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.mean(y, axis=1)

    y = np.squeeze(y)

    if y.dtype != np.float32:
        y = y.astype(np.float32, copy=False)

    max_samples = 10 * 60 * sr  
    if y.size > max_samples:
        print(f"[WARNING] Audio is very long ({y.size / sr / 60:.1f} minutes). Processing in chunks...")
        chunk_size = 5 * 60 * sr  
        chunks = []
        for i in range(0, y.size, chunk_size):
            chunk = y[i:i + chunk_size]
            prop = 0.6 if strength == "light" else 0.8
            chunk_dn = nr.reduce_noise(
                y=chunk,
                sr=sr,
                stationary=True,  
                prop_decrease=prop
            )
            chunks.append(chunk_dn)
        y_dn = np.concatenate(chunks)
    else:
        prop = 0.6 if strength == "light" else 0.8
        y_dn = nr.reduce_noise(
            y=y,
            sr=sr,
            stationary=True,  
            prop_decrease=prop
        )

    out = wav_path.with_name(wav_path.stem + "_dn.wav")
    sf.write(out, y_dn, sr)
    return out
# Changed to True to reduce memory usage