"""Smart Order Router, Slippage, and Depth-Aware Cost Estimator."""

from typing import Dict
from pydantic import BaseModel, ConfigDict


class TradeCostEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    trade_notional: float
    spread_cost: float
    impact_cost: float
    fee_cost: float
    total_execution_cost: float
    execution_cost_pct: float
    is_cost_effective: bool


class SmartOrderRouter:
    """Estimates execution friction, bid-ask spread drag, and market impact using 2% depth."""

    def __init__(
        self,
        base_fee_bps: float = 5.0,  # 5 bps taker fee
        half_spread_bps: float = 2.5,  # 2.5 bps half-spread
        impact_gamma: float = 0.5,
    ) -> None:
        self.base_fee_rate = base_fee_bps / 10000.0
        self.half_spread_rate = half_spread_bps / 10000.0
        self.impact_gamma = impact_gamma

    def estimate_trade_cost(
        self,
        symbol: str,
        trade_notional: float,
        depth_2pct_notional: float,
        tracking_error_reduction: float = 0.0,
    ) -> TradeCostEstimate:
        """Estimates total execution cost for a trade and compares against tracking-error reduction."""
        abs_notional = abs(trade_notional)
        if abs_notional == 0.0:
            return TradeCostEstimate(
                symbol=symbol,
                trade_notional=0.0,
                spread_cost=0.0,
                impact_cost=0.0,
                fee_cost=0.0,
                total_execution_cost=0.0,
                execution_cost_pct=0.0,
                is_cost_effective=True,
            )

        spread_cost = abs_notional * self.half_spread_rate
        fee_cost = abs_notional * self.base_fee_rate

        # Depth-aware square-root market impact: Gamma * (Notional / Depth2Pct)^0.5 * Notional * HalfSpread
        if depth_2pct_notional > 0:
            participation = abs_notional / depth_2pct_notional
            impact_cost = self.impact_gamma * (participation ** 0.5) * abs_notional * self.half_spread_rate
        else:
            # High penalty for unknown/zero depth
            impact_cost = abs_notional * 0.01  # 100 bps

        total_cost = spread_cost + fee_cost + impact_cost
        cost_pct = total_cost / abs_notional

        # Trade is cost effective if tracking error reduction > total execution cost
        is_effective = tracking_error_reduction > total_cost if tracking_error_reduction > 0 else True

        return TradeCostEstimate(
            symbol=symbol,
            trade_notional=trade_notional,
            spread_cost=spread_cost,
            impact_cost=impact_cost,
            fee_cost=fee_cost,
            total_execution_cost=total_cost,
            execution_cost_pct=cost_pct,
            is_cost_effective=is_effective,
        )
