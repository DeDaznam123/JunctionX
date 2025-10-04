from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

def extract_vocals_hpss(wav_path: Path, reduction_db: float = 12.0) -> Path:
    """
    Boosts vocal/transient components by suppressing harmonic background (often music).
    Not a perfect vocal isolate, but improves clarity for ASR.
    """
    # Load mono
    y, sr = librosa.load(wav_path, sr=None, mono=True)

    # Harmonic = sustained music bed, Percussive = transients (speech consonants often show up here too)
    y_harm, y_perc = librosa.effects.hpss(y)

    # Down-weight harmonic (music) by N dB
    gain = 10 ** (-reduction_db / 20.0)
    y_vocal = y_perc + gain * y_harm

    # Normalize
    peak = np.max(np.abs(y_vocal)) + 1e-8
    if peak > 1.0:
        y_vocal = y_vocal / peak

    out = wav_path.with_name(wav_path.stem + "_vocals.wav")
    sf.write(out, y_vocal, sr)
    return out
