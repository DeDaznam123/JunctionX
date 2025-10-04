from typing import List, Dict, Tuple
from faster_whisper import WhisperModel

_MODEL = None

def _get_model(model_size: str = "small.en") -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        print(f"[faster-whisper] init device=cpu, compute_type=int8, model={model_size}")
        _MODEL = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _MODEL

def transcribe_to_segments(
    audio_path: str,
    model_size: str = "small.en",
) -> Tuple[List[Dict], Dict]:
    """
    Returns (segments_list, info_dict)
    segments_list: [{start, end, text}]
    """
    model = _get_model(model_size)
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        language="en",   # force English for MVP; remove to auto-detect
    )

    # --- minimal post-processing: merge segments into sentences ---
    PUNCT = ('.', '!', '?')
    SILENCE_GAP = 0.6  # seconds; start new sentence if gap between segments > this

    out_segments: List[Dict] = []
    cur_start = None
    cur_end = None
    cur_text_parts: List[str] = []
    prev_end = None

    for s in segments:
        # compute gap between this and previous chunk
        gap = (s.start - prev_end) if prev_end is not None else 0.0

        # start new sentence if we had a long pause before this segment
        if cur_text_parts and gap is not None and gap > SILENCE_GAP:
            out_segments.append({
                "start": float(cur_start),
                "end": float(cur_end),
                "text": " ".join(cur_text_parts).strip(),
            })
            cur_start, cur_end, cur_text_parts = None, None, []

        # initialize / extend current sentence
        if cur_start is None:
            cur_start = float(s.start)
        cur_end = float(s.end)
        cur_text_parts.append(s.text.strip())
        prev_end = s.end

        # if this segment ends with terminal punctuation, flush as a sentence
        if s.text.strip().endswith(PUNCT):
            out_segments.append({
                "start": float(cur_start),
                "end": float(cur_end),
                "text": " ".join(cur_text_parts).strip(),
            })
            cur_start, cur_end, cur_text_parts = None, None, []

    # flush any trailing text
    if cur_text_parts:
        out_segments.append({
            "start": float(cur_start),
            "end": float(cur_end),
            "text": " ".join(cur_text_parts).strip(),
        })

    info_dict = {
        "duration": float(info.duration) if info.duration else None,
        "language": info.language,
        "language_probability": float(info.language_probability) if info.language_probability else None,
        "model_size": model_size,
    }
    return out_segments, info_dict