from typing import List, Dict, Any, Tuple
import torch
from transformers import pipeline

LABEL_DEFS = {
    "Violence": "Content that incites, praises, or describes physical violence, harm, or violent acts.",
    "Extremism": "Content that praises extremist acts or promotes dangerous ideologies or behaviors.",
    "Personal Attacks": "Content that contains personal attacks, insults, or psychological harassment.",
    "Gendered Discrimination": "Content that incites unequal or unfair treatment of a person based on gender identity.",
    "Racist": "Content that targets, discriminates against, or promotes hatred toward ethnic or racial groups."
}

HYPOTHESIS_TEMPLATE = "This text contains: {definition}"
LABELS = list(LABEL_DEFS.keys())
HYPOTHESES = [HYPOTHESIS_TEMPLATE.format(definition=v) for v in LABEL_DEFS.values()]

class HateSpeechClassifier:
    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        device: int = 0 if torch.cuda.is_available() else -1,
        batch_size: int = 16,
        global_threshold: float = 0.55,
        label_thresholds: Dict[str, float] = None,
        max_chars: int = 256,
        # --- smoothing knobs ---
        smoothing_window: int = 1,     # neighbors to each side (±1 by default)
        smoothing_mode: str = "mean",  # "mean" | "max" | "weighted"
        smoothing_weight_center: float = 0.6,   # used when mode="weighted"
        smoothing_weight_neighbor: float = 0.2, # used when mode="weighted"
        min_chars_for_loose: int = 10,          # short segments can be noisy
        short_segment_penalty: float = 0.05     # subtract before threshold if short
    ):
        self.pipe = pipeline("zero-shot-classification", model=model_name, device=device, truncation=True)
        self.batch_size = batch_size
        self.global_threshold = global_threshold
        self.label_thresholds = label_thresholds or {}
        self.max_chars = max_chars

        # smoothing
        self.sw = max(0, int(smoothing_window))
        self.smoothing_mode = smoothing_mode
        self.wc = float(smoothing_weight_center)
        self.wn = float(smoothing_weight_neighbor)
        self.min_chars_for_loose = int(min_chars_for_loose)
        self.short_segment_penalty = float(short_segment_penalty)

    def _thr(self, label: str) -> float:
        return self.label_thresholds.get(label, self.global_threshold)

    def _clip(self, t: str) -> str:
        return t if len(t) <= self.max_chars else t[: self.max_chars] + "..."

    def _nli_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Returns list of dicts: [{label->score,...}, ...] for each text (NO thresholding).
        """
        results = []
        for i in range(0, len(texts), self.batch_size):
            out = self.pipe(texts[i:i+self.batch_size], candidate_labels=HYPOTHESES, multi_label=True)
            if isinstance(out, dict):
                out = [out]
            for r in out:
                label_scores = {}
                for hyp, score in zip(r["labels"], r["scores"]):
                    for label, definition in LABEL_DEFS.items():
                        if hyp.endswith(definition):
                            label_scores[label] = float(score)
                            break
                results.append(label_scores)
        return results

    def _smooth_scores(self, raw: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        raw: per-segment raw scores per label (no thresholds)
        returns: same structure with smoothed scores by label
        """
        if self.sw == 0 or len(raw) <= 1:
            return raw

        n = len(raw)
        smoothed: List[Dict[str, float]] = [dict() for _ in range(n)]

        for li in LABELS:
            # collect 1D series for label
            series = [seg.get(li, 0.0) for seg in raw]

            for idx in range(n):
                left = max(0, idx - self.sw)
                right = min(n - 1, idx + self.sw)
                window_vals = series[left:right+1]

                if self.smoothing_mode == "max":
                    val = max(window_vals)
                elif self.smoothing_mode == "weighted":
                    # center weight + equal neighbor weights
                    k = right - left + 1
                    if k == 1:
                        val = window_vals[0]
                    else:
                        weights = []
                        for j in range(left, right+1):
                            if j == idx:
                                weights.append(self.wc)
                            else:
                                weights.append(self.wn)
                        # normalize just in case
                        s = sum(weights)
                        weights = [w/s for w in weights]
                        val = sum(w * v for w, v in zip(weights, window_vals))
                else:  # "mean"
                    val = sum(window_vals) / len(window_vals)

                smoothed[idx][li] = float(val)

        return smoothed

    def extract_hateful(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not segments:
            return []

        texts = [self._clip(s.get("text", "")) for s in segments]

        # 1) raw per-label scores per segment
        raw_scores = self._nli_batch(texts)

        # 2) smoothing across neighbors
        smooth_scores = self._smooth_scores(raw_scores)

        # 3) optional short-segment penalty BEFORE thresholding
        final_scores: List[Dict[str, float]] = []
        for seg, scores in zip(segments, smooth_scores):
            penalized = scores.copy()
            if len(seg.get("text", "")) < self.min_chars_for_loose:
                for k in penalized:
                    penalized[k] = max(0.0, penalized[k] - self.short_segment_penalty)
            final_scores.append(penalized)

        # 4) threshold & package output
        hateful: List[Dict[str, Any]] = []
        for seg, scores in zip(segments, final_scores):
            passing = [
                {"label": lab, "score": float(sc)}
                for lab, sc in scores.items()
                if sc >= self._thr(lab)
            ]
            if passing:
                hateful.append({
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text", ""),
                    "labels": sorted(passing, key=lambda x: x["score"], reverse=True)
                })

        return hateful