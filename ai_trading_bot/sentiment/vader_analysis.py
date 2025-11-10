from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover - optional dependency
    SentimentIntensityAnalyzer = None

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VaderSentimentAnalyzer:
    def analyze(self, text: str) -> Dict[str, float]:
        if SentimentIntensityAnalyzer is None:
            raise ImportError(
                "vaderSentiment is required for VADER analysis. Install with `pip install vaderSentiment`."
            )
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        return {"positive": scores["pos"], "negative": scores["neg"], "neutral": scores["neu"], "compound": scores["compound"]}
