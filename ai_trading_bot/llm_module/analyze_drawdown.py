from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

try:
    from langchain.prompts import PromptTemplate
    from langchain.llms.base import LLM
except ImportError:  # pragma: no cover - optional dependency
    PromptTemplate = None
    LLM = None


@dataclass
class DrawdownAnalyzer:
    llm: LLM

    def analyze(self, drawdown_stats: Dict[str, str]) -> str:
        if PromptTemplate is None:
            raise ImportError("langchain is required for LLM explanations. Install with `pip install langchain`."
            )
        template = PromptTemplate(
            input_variables=["max_drawdown", "duration", "recovery", "notes"],
            template=(
                "Analysiere den letzten Drawdown. Max Drawdown: {max_drawdown}. Dauer: {duration}. "
                "Erholung: {recovery}. Zusätzliche Hinweise: {notes}."
            ),
        )
        prompt = template.format(**drawdown_stats)
        return self.llm(prompt)
