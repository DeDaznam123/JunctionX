from typing import List, Dict, Tuple
from faster_whisper import WhisperModel
from tqdm import tqdm

_MODEL = None

def _get_model(model_size: str = "medium.en") -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        # Try GPU first, fallback to CPU if CUDA isn't available
        try:
            print(f"[faster-whisper] init device=cuda, compute_type=float16, model={model_size}")
            _MODEL = WhisperModel(model_size, device="cuda", compute_type="float16")
        except Exception as e:
            print(f"[faster-whisper] CUDA not available ({e}), falling back to CPU")
            print(f"[faster-whisper] init device=cpu, compute_type=int8, model={model_size}")
            _MODEL = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _MODEL

def transcribe_to_segments(
    audio_path: str,
    model_size: str,
    chunk_length: int,
    show_progress: bool = True,
    ) -> Tuple[List[Dict], Dict]:

    """
    Returns (segments_list, info_dict)
    segments_list: [{start, end, text}]
    show_progress: whether to display a tqdm progress bar
    """
    model = _get_model(model_size)

    # Create progress bar if requested
    pbar = None
    if show_progress:
        pbar = tqdm(desc="Transcribing", unit="seg", dynamic_ncols=True)

    segments, info = model.transcribe(
        audio_path,
        language="en",
        # keep sentence timestamps from segment boundaries
        beam_size=1,
        best_of=1,
        word_timestamps=False,          # keeps it fast; segment timestamps still present
        vad_filter=True,
        vad_parameters=dict(            # tune to avoid mid-sentence chops
            min_silence_duration_ms=600,   # raise to merge short pauses
            min_speech_duration_ms=250,    # lower if speech is choppy
            speech_pad_ms=100              # small padding helps sentence continuity
        ),
        chunk_length=chunk_length,                # seconds; good CPU/GIL balance
        condition_on_previous_text=False,
        temperature=0.0,
        no_speech_threshold=0.45,       # slightly stricter: skip very quiet parts
        log_prob_threshold=-1.0,        # avoid fallback decoding
        compression_ratio_threshold=2.4,
        hallucination_silence_threshold=0.5,
    )

    # --- Merge decoder segments into sentences (timestamps preserved) ---
    PUNCT = ('.', '!', '?')
    SILENCE_GAP = 0.7  # seconds; start new sentence if a large pause occurs

    out_segments: List[Dict] = []
    cur_start = None
    cur_end = None
    cur_text_parts: List[str] = []
    prev_end = None

    for s in segments:
        # Update progress bar
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({"time": f"{s.end:.1f}s"})

        # each s has s.start/s.end (segment timestamps from the model)
        gap = (s.start - prev_end) if prev_end is not None else 0.0

        # long pause → flush current sentence
        if cur_text_parts and gap is not None and gap > SILENCE_GAP:
            out_segments.append({
                "start": float(cur_start),
                "end": float(cur_end),
                "text": " ".join(cur_text_parts).strip(),
            })
            cur_start, cur_end, cur_text_parts = None, None, []

        # initialize/extend sentence
        if cur_start is None:
            cur_start = float(s.start)
        cur_end = float(s.end)
        cur_text_parts.append(s.text.strip())
        prev_end = s.end

        # if segment ends with sentence punctuation → flush as sentence
        if s.text.strip().endswith(PUNCT):
            out_segments.append({
                "start": float(cur_start),
                "end": float(cur_end),
                "text": " ".join(cur_text_parts).strip(),
            })
            cur_start, cur_end, cur_text_parts = None, None, []

    # flush trailing
    if cur_text_parts:
        out_segments.append({
            "start": float(cur_start),
            "end": float(cur_end),
            "text": " ".join(cur_text_parts).strip(),
        })

    # Close progress bar
    if pbar is not None:
        pbar.close()

    info_dict = {
        "duration": float(info.duration) if info.duration else None,
        "language": info.language,
        "language_probability": float(info.language_probability) if info.language_probability else None,
        "model_size": model_size,
    }
    return out_segments, info_dict
