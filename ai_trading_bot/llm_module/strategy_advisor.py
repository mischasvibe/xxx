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
class StrategyAdvisor:
    llm: LLM

    def suggest(self, performance_snapshot: Dict[str, str]) -> str:
        if PromptTemplate is None:
            raise ImportError("langchain is required for LLM suggestions. Install with `pip install langchain`."
            )
        template = PromptTemplate(
            input_variables=["strategies", "metrics", "market_context"],
            template=(
                "Basierend auf den Strategien {strategies} und den Kennzahlen {metrics} sowie dem Marktumfeld"
                " {market_context}, schlage Optimierungen oder Anpassungen vor."
            ),
        )
        prompt = template.format(**performance_snapshot)
        return self.llm(prompt)
