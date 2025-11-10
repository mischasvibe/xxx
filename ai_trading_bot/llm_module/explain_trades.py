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
class TradeExplainer:
    llm: LLM

    def explain(self, trade_context: Dict[str, str]) -> str:
        if PromptTemplate is None:
            raise ImportError("langchain is required for LLM explanations. Install with `pip install langchain`."
            )
        template = PromptTemplate(
            input_variables=["strategy", "signal", "reason", "metrics"],
            template=(
                "Du bist ein Trading-Assistent. Strategie: {strategy}. Signal: {signal}. "
                "Begründung: {reason}. Leistungskennzahlen: {metrics}."
                " Erkläre den Trade verständlich."
            ),
        )
        prompt = template.format(**trade_context)
        return self.llm(prompt)
