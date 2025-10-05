# ai_classifier.py
from typing import List, Dict, Any
import re
import torch
from transformers import pipeline

# ====== Labels & hypotheses ======
LABEL_DEFS = {
    "Violence": "Content that incites, praises, or describes physical violence, harm, or violent acts.",
    "Extremism": "Content that praises extremist acts or promotes dangerous ideologies or behaviors.",
    "Personal Attacks": "Content that contains personal attacks, insults, or psychological harassment.",
    "Gendered Discrimination": "Content that incites unequal or unfair treatment of a person based on gender identity.",
    "Racist": "Content that targets, discriminates against, or promotes hatred toward ethnic or racial groups.",
    # NEW
    "Religious Discrimination": "Content that targets, discriminates against, or promotes hatred toward people based on religion or belief (e.g., Muslims, Christians, Jews, Hindus, atheists)."
}
LABELS = list(LABEL_DEFS.keys())
HYPOTHESIS_TEMPLATE = "This text contains: {definition}"
HYPOTHESES = [HYPOTHESIS_TEMPLATE.format(definition=LABEL_DEFS[lbl]) for lbl in LABELS]
HYP2LABEL = dict(zip(HYPOTHESES, LABELS))

# Target-term hints (kept gentle)
TARGET_HINTS = {
    "Racist": [
        "asian","black","white","latino","hispanic","african","european","arab",
        "immigrant","refugee","jew","jewish","romani"
    ],
    "Religious Discrimination": [
        "muslim","muslims","christian","christians","jew","jews","jewish","hindu","hindus",
        "buddhist","buddhists","sikh","sikhs","atheist","atheists"
    ],
    "Gendered Discrimination": [
        "woman","women","female","girl","girls","she","her",
        "man","men","male","boy","boys","he","him","nonbinary","trans","transgender"
    ],
}

class HateSpeechClassifier:
    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",  # good recall; swap if needed MoritzLaurer/deberta-v3-large-zeroshot-v2.0
        device: int = 0 if torch.cuda.is_available() else -1,
        batch_size: int = 16,

        # High-recall defaults
        global_threshold: float = 0.50,
        label_thresholds: Dict[str, float] | None = None,

        # Escape hatch: accept very confident spikes
        escape_high_conf: float = 0.82,

        # Light knobs only
        max_chars: int = 256,
        smoothing_window: int = 0,     # OFF (don't wash out spikes)
        min_chars_for_loose: int = 3,
        short_segment_penalty: float = 0.02
    ):
        self.pipe = pipeline("zero-shot-classification", model=model_name, device=device, truncation=True)
        self.batch_size = batch_size
        self.global_threshold = float(global_threshold)
        self.label_thresholds = label_thresholds or {
            "Racist": 0.50,
            "Religious Discrimination": 0.50,
            "Gendered Discrimination": 0.50,
            "Personal Attacks": 0.50,
            "Extremism": 0.52,   # slightly higher
            "Violence": 0.55
        }
        self.escape_high_conf = float(escape_high_conf)
        self.max_chars = int(max_chars)
        self.sw = max(0, int(smoothing_window))
        self.min_chars_for_loose = int(min_chars_for_loose)
        self.short_segment_penalty = float(short_segment_penalty)

    # ----- utilities -----
    def _thr(self, label: str) -> float:
        return float(self.label_thresholds.get(label, self.global_threshold))

    def _clip(self, t: str) -> str:
        return t if len(t) <= self.max_chars else t[: self.max_chars] + "..."

    # Simple rule-based overrides for unmistakable hate patterns
    def _rule_overrides(self, text: str) -> List[str]:
        t = (text or "").lower().strip()
        labels: List[str] = []

        # exact classic pattern: "all X are Y"
        if re.search(r"\ball\s+muslims?\s+are\s+terrorists?\b", t):
            labels += ["Religious Discrimination", "Extremism"]

        # broaden pattern with a few variants
        if re.search(r"\bmuslims?\s+are\s+(all\s+)?terrorists?\b", t):
            if "Religious Discrimination" not in labels:
                labels.append("Religious Discrimination")
            if "Extremism" not in labels:
                labels.append("Extremism")

        # add more rules as needed…

        return labels

    # ----- model call -----
    def _nli_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        results: List[Dict[str, float]] = []
        for i in range(0, len(texts), self.batch_size):
            out = self.pipe(
                texts[i:i+self.batch_size],
                candidate_labels=HYPOTHESES,
                multi_label=True
            )
            if isinstance(out, dict):
                out = [out]
            for r in out:
                scores = { HYP2LABEL[hyp]: float(score) for hyp, score in zip(r["labels"], r["scores"]) }
                results.append(scores)
        return results

    # ----- main API -----
    def extract_hateful(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not segments:
            return []

        texts = [self._clip(s.get("text", "") or "") for s in segments]

        # 0) rule-based overrides (high precision)
        rule_hits: List[List[str]] = [self._rule_overrides(txt) for txt in texts]

        # 1) model scores
        raw_scores = self._nli_batch(texts)

        # 2) tiny penalty for ultra-short segments
        for seg, scs in zip(segments, raw_scores):
            if len((seg.get("text", "") or "")) < self.min_chars_for_loose:
                for k in scs:
                    scs[k] = max(0.0, scs[k] - self.short_segment_penalty)

        # 3) thresholding (+ escape hatch + rule overrides)
        hateful: List[Dict[str, Any]] = []
        for (seg, scores, rules) in zip(segments, raw_scores, rule_hits):
            text_lc = (seg.get("text", "") or "").lower()
            top = max(scores.values() or [0.0])

            # Escape hatch
            if top >= self.escape_high_conf:
                passing = [{"label": lab, "score": float(sc)} for lab, sc in scores.items() if sc >= self.escape_high_conf]
            else:
                passing = [{"label": lab, "score": float(sc)} for lab, sc in scores.items() if sc >= self._thr(lab)]

            # Apply rule-based overrides (inject with high score)
            for lab in rules:
                if lab not in [p["label"] for p in passing]:
                    passing.append({"label": lab, "score": 0.99})

            # Gentle target-term sanity (only if not already very confident)
            if passing:
                filtered = []
                for item in passing:
                    lab, sc = item["label"], item["score"]
                    hints = TARGET_HINTS.get(lab)
                    if hints and sc < (self._thr(lab) + 0.10):
                        has_hint = any(h in text_lc for h in hints)
                        if not has_hint:
                            continue
                    filtered.append(item)
                passing = filtered

            if passing:
                hateful.append({
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text", ""),
                    "labels": sorted(passing, key=lambda x: x["score"], reverse=True)
                })

        return hateful
