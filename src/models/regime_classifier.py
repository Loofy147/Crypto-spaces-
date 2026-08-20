"""XGBoost Regime Classification & Risk Aversion Engine."""

from enum import Enum
from typing import Any, Dict, List, Tuple
import numpy as np
import xgboost as xgb
from pydantic import BaseModel, ConfigDict


class MarketRegime(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    CONSOLIDATION = "CONSOLIDATION"


class RegimeClassificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    regime: MarketRegime
    probabilities: Dict[str, float]
    risk_aversion_gamma: float  # Adjusted risk aversion parameter


class RegimeClassifier:
    """XGBoost Classifier for market regime detection (Bull/Bear/Consolidation)."""

    def __init__(self) -> None:
        self.model: xgb.XGBClassifier | None = None
        self._is_trained = False
        self.label_map = {0: MarketRegime.BULL, 1: MarketRegime.BEAR, 2: MarketRegime.CONSOLIDATION}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train XGBoost multi-class classifier."""
        if len(X) == 0 or len(y) == 0:
            return

        self.model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
        )
        self.model.fit(X, y)
        self._is_trained = True

    def classify(self, feature_vector: List[float]) -> RegimeClassificationResult:
        """Classify market regime based on feature vector:

        [cvd_delta, depth_ratio, nupl, roc_30d]
        """
        if not self._is_trained or self.model is None:
            # Rule-based fallback if not trained
            cvd_delta, depth_ratio, nupl, roc_30d = (
                feature_vector[0] if len(feature_vector) > 0 else 0.0,
                feature_vector[1] if len(feature_vector) > 1 else 1.0,
                feature_vector[2] if len(feature_vector) > 2 else 0.5,
                feature_vector[3] if len(feature_vector) > 3 else 0.0,
            )

            if nupl > 0.6 or (roc_30d > 0.15 and cvd_delta > 0):
                regime = MarketRegime.BULL
                probs = {"BULL": 0.80, "BEAR": 0.05, "CONSOLIDATION": 0.15}
                gamma = 1.0
            elif nupl < 0.2 or roc_30d < -0.15:
                regime = MarketRegime.BEAR
                probs = {"BULL": 0.05, "BEAR": 0.80, "CONSOLIDATION": 0.15}
                gamma = 3.0
            else:
                regime = MarketRegime.CONSOLIDATION
                probs = {"BULL": 0.20, "BEAR": 0.20, "CONSOLIDATION": 0.60}
                gamma = 2.0

            return RegimeClassificationResult(
                regime=regime,
                probabilities=probs,
                risk_aversion_gamma=gamma,
            )

        X_inp = np.array([feature_vector], dtype=np.float32)
        probs_raw = self.model.predict_proba(X_inp)[0]
        pred_class = int(np.argmax(probs_raw))
        regime = self.label_map[pred_class]

        probs = {
            self.label_map[i].value: float(probs_raw[i]) for i in range(len(probs_raw))
        }

        # Map regime to risk aversion gamma coefficient
        if regime == MarketRegime.BULL:
            gamma = 1.0
        elif regime == MarketRegime.BEAR:
            gamma = 3.0
        else:
            gamma = 2.0

        return RegimeClassificationResult(
            regime=regime,
            probabilities=probs,
            risk_aversion_gamma=gamma,
        )
