from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
except ImportError:  # pragma: no cover - optional dependency
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    torch = None

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FinBERTSentimentAnalyzer:
    model_name: str = "ProsusAI/finbert"
    device: Optional[str] = None
    _model: Optional[AutoModelForSequenceClassification] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _warned_unavailable: bool = False

    def is_available(self) -> bool:
        return AutoModelForSequenceClassification is not None and torch is not None

    def _load(self) -> None:
        if not self.is_available():



    def _load(self) -> None:
        if AutoModelForSequenceClassification is None:
            raise ImportError(
                "transformers and torch are required for FinBERT sentiment analysis."
            )
        if self._model is None or self._tokenizer is None:
            logger.info("Loading FinBERT model %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            if self.device:
                self._model.to(self.device)

    def analyze(self, text: str) -> Dict[str, float]:
        if not self.is_available():  # pragma: no cover - optional dependency path
            if not self._warned_unavailable:
                logger.warning(
                    "FinBERT dependencies missing; returning neutral sentiment. Install optional "
                    "ML extras via `pip install -r requirements-ml.txt` to enable FinBERT."
                )
                self._warned_unavailable = True
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}



 
        self._load()
        assert self._model is not None and self._tokenizer is not None
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        if self.device:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        probs = outputs.logits.softmax(dim=-1).detach().cpu().numpy()[0]
        return {"positive": float(probs[0]), "neutral": float(probs[1]), "negative": float(probs[2])}
