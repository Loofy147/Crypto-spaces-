"""Volatility Targeting & Risk Capital De-leveraging Engine."""

from typing import Dict, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict


class VolatilityTargetingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    scaled_weights: Dict[str, float]
    rwa_sleeve_weight: float
    scaler_factor: float
    realized_volatility: float
    target_volatility: float


class VolatilityTargeter:
    """Scales portfolio weights inversely to realized volatility."""

    def __init__(
        self, target_volatility: float = 0.25, rwa_symbol: str = "BUIDL"
    ) -> None:
        self.target_volatility = target_volatility
        self.rwa_symbol = rwa_symbol

    def scale_positions(
        self,
        base_weights: Dict[str, float],
        realized_vols: Dict[str, float],
        portfolio_cov_matrix: np.ndarray | None = None,
        asset_names: list[str] | None = None,
    ) -> VolatilityTargetingResult:
        """Scales base weights inversely based on realized volatility.

        If realized vol > target vol, reduces risky position weights and diverts excess capital
        into the RWA yield sleeve.
        """
        if portfolio_cov_matrix is not None and asset_names is not None and len(asset_names) > 0:
            w_vec = np.array([base_weights.get(name, 0.0) for name in asset_names])
            portfolio_vol = float(np.sqrt(np.dot(w_vec.T, np.dot(portfolio_cov_matrix, w_vec)))) * np.sqrt(365.0)
        else:
            # Weighted average realized vol
            portfolio_vol = sum(
                base_weights.get(asset, 0.0) * vol
                for asset, vol in realized_vols.items()
            )

        if portfolio_vol <= 0.0:
            scaler = 1.0
        else:
            scaler = min(1.0, self.target_volatility / portfolio_vol)

        scaled_weights: Dict[str, float] = {}
        allocated_risky_weight = 0.0

        for asset, base_w in base_weights.items():
            if asset == self.rwa_symbol:
                continue
            scaled_w = base_w * scaler
            scaled_weights[asset] = scaled_w
            allocated_risky_weight += scaled_w

        # Excess risk capital diverted to RWA yield sleeve
        rwa_weight = 1.0 - allocated_risky_weight
        scaled_weights[self.rwa_symbol] = max(0.0, rwa_weight)

        return VolatilityTargetingResult(
            scaled_weights=scaled_weights,
            rwa_sleeve_weight=scaled_weights[self.rwa_symbol],
            scaler_factor=scaler,
            realized_volatility=portfolio_vol,
            target_volatility=self.target_volatility,
        )
