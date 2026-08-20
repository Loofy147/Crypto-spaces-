"""Post-Trade Performance & P&L Attribution Engine."""

from typing import Dict, List
import numpy as np
from pydantic import BaseModel, ConfigDict


class PnLAttributionBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_return: float
    factor_delta_return: float  # Directional market return (w * beta * R_m)
    funding_carry_return: float  # Funding rate carry on perpetuals (w * C)
    rwa_yield_return: float  # Tokenized Treasury yield accruals (w * Y)
    residual_drift: float  # Unexplained residual (slippage, fees, friction)
    attribution_error_pct: float


class PnLDecomposer:
    """Decomposes total portfolio returns into factor delta, funding carry, RWA yield, and residual drift."""

    @staticmethod
    def decompose_pnl(
        total_return: float,
        weights: Dict[str, float],
        betas: Dict[str, float],
        benchmark_return: float,
        funding_rates: Dict[str, float],
        rwa_yields: Dict[str, float],
    ) -> PnLAttributionBreakdown:
        """Computes exact return component breakdown."""
        # 1. Factor Delta Return: sum(w_i * beta_i * R_m)
        factor_delta = sum(
            weights.get(asset, 0.0) * betas.get(asset, 1.0) * benchmark_return
            for asset in weights
            if asset not in rwa_yields
        )

        # 2. Funding Carry Return: sum(w_j * C_j)
        funding_carry = sum(
            weights.get(asset, 0.0) * funding_rates.get(asset, 0.0)
            for asset in weights
        )

        # 3. RWA Yield Return: sum(w_k * Y_k)
        rwa_yield = sum(
            weights.get(asset, 0.0) * rwa_yields.get(asset, 0.0)
            for asset in rwa_yields
        )

        # 4. Residual Drift: epsilon = R_p - (Factor_Delta + Funding_Carry + RWA_Yield)
        explained = factor_delta + funding_carry + rwa_yield
        residual = total_return - explained

        error_pct = (abs(residual) / abs(total_return) * 100.0) if abs(total_return) > 1e-6 else 0.0

        return PnLAttributionBreakdown(
            total_return=total_return,
            factor_delta_return=factor_delta,
            funding_carry_return=funding_carry,
            rwa_yield_return=rwa_yield,
            residual_drift=residual,
            attribution_error_pct=error_pct,
        )
