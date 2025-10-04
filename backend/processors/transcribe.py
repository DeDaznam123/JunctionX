from typing import List, Dict, Tuple
from faster_whisper import WhisperModel, BatchedInferencePipeline
from tqdm import tqdm
import os

# --- Prevent BLAS/OpenMP oversubscription (good for CPU; harmless on GPU) ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_MODEL: WhisperModel | None = None
_PIPELINE: BatchedInferencePipeline | None = None
GPU: bool | None = None

# Workers: default = all logical CPUs; allow override via env
MAX_WORKERS = max(1, os.cpu_count() or 4)
try:
    if "ASR_MAX_WORKERS" in os.environ:
        MAX_WORKERS = max(1, int(os.environ["ASR_MAX_WORKERS"]))
except ValueError:
    pass

def _ensure_pipeline(model_size: str = "small.en") -> BatchedInferencePipeline:
    """Lazy-init a single pipeline (GPU first, CPU fallback)."""
    global _PIPELINE, GPU
    if _PIPELINE is not None:
        return _PIPELINE

    # Try GPU first
    try:
        print(f"[faster-whisper] init device=cuda, compute_type=float16, model={model_size}")
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        GPU = True
    except Exception as e:
        print(f"[faster-whisper] CUDA not available ({e}); falling back to CPU")
        GPU = False
        print(f"[faster-whisper] init device=cpu, compute_type=int8, cpu_threads={MAX_WORKERS}, model={model_size}")
        model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=MAX_WORKERS)

    _PIPELINE = BatchedInferencePipeline(model=model)
    print(f"[faster-whisper] Batched pipeline ready (runtime={'GPU' if GPU else 'CPU'})")
    return _PIPELINE

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
    pipe = _ensure_pipeline(model_size)

    # Cap batch size a bit on CPU to avoid diminishing returns
    if GPU:
        batch_size = min(4, MAX_WORKERS)
    else:
        batch_size = 1 

    # Create progress bar if requested
    pbar = None
    if show_progress:
        pbar = tqdm(desc="Transcribing", unit="seg", dynamic_ncols=True)

    common = dict(
        language="en",
        beam_size=1,
        best_of=1,
        word_timestamps=False,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=600,
            min_speech_duration_ms=250,
            speech_pad_ms=100
        ),
        chunk_length=chunk_length,
        condition_on_previous_text=False,
        temperature=0.0,
        no_speech_threshold=0.45,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        hallucination_silence_threshold=0.5,
    )

    # Parallelized for both CPU/GPU
    segments, info = pipe.transcribe(audio_path, batch_size=batch_size, **common)

    # --- Merge decoder segments into sentence-level spans ---
    PUNCT = ('.', '!', '?')
    SILENCE_GAP = 0.7

    out_segments: List[Dict] = []
    cur_start = None
    cur_end = None
    cur_text_parts: List[str] = []
    prev_end = None

    for s in segments:
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({"time": f"{s.end:.1f}s"})

        gap = (s.start - prev_end) if prev_end is not None else 0.0

        if cur_text_parts and gap is not None and gap > SILENCE_GAP:
            out_segments.append({
                "start": float(cur_start),
                "end": float(cur_end),
                "text": " ".join(cur_text_parts).strip(),
            })
            cur_start, cur_end, cur_text_parts = None, None, []

        if cur_start is None:
            cur_start = float(s.start)
        cur_end = float(s.end)
        cur_text_parts.append(s.text.strip())
        prev_end = s.end

        if s.text.strip().endswith(PUNCT):
            out_segments.append({
                "start": float(cur_start),
                "end": float(cur_end),
                "text": " ".join(cur_text_parts).strip(),
            })
            cur_start, cur_end, cur_text_parts = None, None, []

    if cur_text_parts:
        out_segments.append({
            "start": float(cur_start),
            "end": float(cur_end),
            "text": " ".join(cur_text_parts).strip(),
        })

    if pbar is not None:
        pbar.close()

    info_dict = {
        "duration": float(info.duration) if info.duration else None,
        "language": info.language,
        "language_probability": float(info.language_probability) if info.language_probability else None,
        "model_size": model_size,
        "gpu": bool(GPU),
        "max_workers": MAX_WORKERS,
        "batch_size": batch_size,
        "chunk_length": chunk_length,
    }
    return out_segments, info_dict
