"""Derivatives Microstructure, Margin Ratios, and CVD Delta Tracking."""

from enum import Enum
from pydantic import BaseModel, ConfigDict


class MicrostructureRegime(str, Enum):
    ORGANIC_BULLISH = "ORGANIC_BULLISH"
    SPOT_ABSORPTION = "SPOT_ABSORPTION"
    LONG_SQUEEZE_RISK = "LONG_SQUEEZE_RISK"
    BROAD_LIQUIDATION = "BROAD_LIQUIDATION"


class CryptoMarginStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    crypto_margined_oi: float
    cash_margined_oi: float
    total_oi: float
    crypto_margin_ratio: float  # Percentage (0-100)
    leverage_alert_flag: bool  # True if ratio > 50%


class CVDDivergenceStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    spot_cvd: float
    perp_cvd: float
    cvd_delta: float  # spot_cvd - perp_cvd
    regime: MicrostructureRegime
    implied_action: str


class DerivativesFeedCalculator:
    """Calculates derivative microstructure metrics."""

    @staticmethod
    def calculate_crypto_margin_ratio(
        crypto_margined_oi: float, cash_margined_oi: float
    ) -> CryptoMarginStatus:
        """Compute relative ratio of crypto-margined vs cash-margined perpetual open interest."""
        total_oi = crypto_margined_oi + cash_margined_oi
        if total_oi <= 0:
            ratio = 0.0
        else:
            ratio = (crypto_margined_oi / total_oi) * 100.0

        alert_flag = ratio > 50.0

        return CryptoMarginStatus(
            crypto_margined_oi=crypto_margined_oi,
            cash_margined_oi=cash_margined_oi,
            total_oi=total_oi,
            crypto_margin_ratio=ratio,
            leverage_alert_flag=alert_flag,
        )

    @staticmethod
    def detect_cvd_divergence(spot_cvd: float, perp_cvd: float) -> CVDDivergenceStatus:
        """Detect microstructure regime based on Spot CVD vs Perpetual CVD divergence."""
        cvd_delta = spot_cvd - perp_cvd

        if spot_cvd >= 0 and perp_cvd >= 0:
            regime = MicrostructureRegime.ORGANIC_BULLISH
            action = "Expand core long exposure; increase altcoin weights."
        elif spot_cvd >= 0 and perp_cvd < 0:
            regime = MicrostructureRegime.SPOT_ABSORPTION
            action = "Accumulate spot positions; hedge via cash-margined perps."
        elif spot_cvd < 0 and perp_cvd >= 0:
            regime = MicrostructureRegime.LONG_SQUEEZE_RISK
            action = "Reduce leverage; tighten stop-losses; raise cash buffer."
        else:
            regime = MicrostructureRegime.BROAD_LIQUIDATION
            action = "Transition to RWA cash equivalents; execute delta hedges."

        return CVDDivergenceStatus(
            spot_cvd=spot_cvd,
            perp_cvd=perp_cvd,
            cvd_delta=cvd_delta,
            regime=regime,
            implied_action=action,
        )
