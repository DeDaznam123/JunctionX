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
    # Convert to Path object if it's a string
    if isinstance(wav_path, str):
        wav_path = Path(wav_path)

    y, sr = sf.read(wav_path)

    # Validate audio data
    if y.size == 0:
        raise ValueError("Audio file is empty")

    # Convert stereo to mono if needed
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = np.mean(y, axis=1)

    # Ensure 1D array
    y = np.squeeze(y)

    # ensure float32 for stability
    if y.dtype != np.float32:
        y = y.astype(np.float32, copy=False)

    # Validate array size to prevent memory issues
    max_samples = 10 * 60 * sr  # 10 minutes at given sample rate
    if y.size > max_samples:
        print(f"[WARNING] Audio is very long ({y.size / sr / 60:.1f} minutes). Processing in chunks...")
        # Process in chunks to avoid memory issues
        chunk_size = 5 * 60 * sr  # 5 minute chunks
        chunks = []
        for i in range(0, y.size, chunk_size):
            chunk = y[i:i + chunk_size]
            prop = 0.6 if strength == "light" else 0.8
            chunk_dn = nr.reduce_noise(
                y=chunk,
                sr=sr,
                stationary=True,  # Use stationary for long audio to reduce memory
                prop_decrease=prop
            )
            chunks.append(chunk_dn)
        y_dn = np.concatenate(chunks)
    else:
        prop = 0.6 if strength == "light" else 0.8
        # non-stationary handles changing background better, but uses more memory
        y_dn = nr.reduce_noise(
            y=y,
            sr=sr,
            stationary=True,  # Changed to True to reduce memory usage
            prop_decrease=prop
        )

    out = wav_path.with_name(wav_path.stem + "_dn.wav")
    sf.write(out, y_dn, sr)
    return out
