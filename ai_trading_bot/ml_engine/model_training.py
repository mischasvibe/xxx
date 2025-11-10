from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MLSignalModel:
    params: Dict[str, object]
    model: Optional[RandomForestClassifier] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=False)
        logger.info("RandomForest performance:\n%s", report)

    def predict(self, features: pd.DataFrame) -> Tuple[str, float]:
        if self.model is None:
            raise RuntimeError("Model must be trained before calling predict().")
        proba = self.model.predict_proba(features)[0]
        classes = self.model.classes_
        idx = int(np.argmax(proba))
        return classes[idx], float(proba[idx])
