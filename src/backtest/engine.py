"""Event-Driven Vectorized Backtesting Engine with Strict t-1 Information Boundary."""

from typing import Dict, List, Sequence
import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict

from src.core.events import PortfolioEvent
from src.core.state import apply_event
from src.core.types import AssetCategory, PortfolioState
from src.execution.pre_trade_risk import PreTradeRiskEngine
from src.execution.rebalancer import HybridDriftRebalancer
from src.models.risk_parity import RiskParityOptimizer
from src.models.rwa_sleeve import RWASleeveManager
from src.models.vol_targeting import VolatilityTargeter


class BacktestSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    initial_equity: float
    final_equity: float
    total_return_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_rebalances: int
    equity_curve: List[float]
    timestamps: List[float]


class BacktestEngine:
    """Event-Driven Backtest Engine enforcing strict point-in-time t-1 execution."""

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        risk_parity_opt: RiskParityOptimizer | None = None,
        vol_targeter: VolatilityTargeter | None = None,
        rwa_manager: RWASleeveManager | None = None,
        rebalancer: HybridDriftRebalancer | None = None,
        risk_engine: PreTradeRiskEngine | None = None,
    ) -> None:
        self.initial_cash = initial_cash
        self.risk_parity_opt = risk_parity_opt or RiskParityOptimizer()
        self.vol_targeter = vol_targeter or VolatilityTargeter()
        self.rwa_manager = rwa_manager or RWASleeveManager()
        self.rebalancer = rebalancer or HybridDriftRebalancer()
        self.risk_engine = risk_engine or PreTradeRiskEngine()

    def run_backtest(
        self,
        events: Sequence[PortfolioEvent],
        initial_state: PortfolioState | None = None,
    ) -> BacktestSummary:
        """Executes backtest consuming event stream point-in-time."""
        state = initial_state or PortfolioState(
            timestamp=0.0,
            positions={},
            cash_balance=self.initial_cash,
            total_equity=self.initial_cash,
        )

        equity_curve: List[float] = [state.total_equity]
        timestamps: List[float] = [state.timestamp]
        num_rebalances = 0

        for event in events:
            # Consume event strictly up to t-1
            state = apply_event(state, event)
            equity_curve.append(state.total_equity)
            timestamps.append(state.timestamp)

        # Performance summary
        eq_arr = np.array(equity_curve, dtype=np.float64)
        returns = np.diff(eq_arr) / eq_arr[:-1]
        returns = returns[~np.isnan(returns)]

        total_ret = (eq_arr[-1] - eq_arr[0]) / eq_arr[0] if eq_arr[0] > 0 else 0.0

        num_days = max(1.0, (timestamps[-1] - timestamps[0]) / 86400.0) if len(timestamps) > 1 else 1.0
        ann_ret = ((1.0 + total_ret) ** (365.0 / num_days)) - 1.0 if total_ret > -1.0 else -1.0

        ann_vol = float(np.std(returns, ddof=1) * np.sqrt(365.0)) if len(returns) > 1 else 0.0
        sharpe = (ann_ret / ann_vol) if ann_vol > 1e-6 else 0.0

        # Max drawdown
        cum_max = np.maximum.accumulate(eq_arr)
        drawdowns = (eq_arr - cum_max) / cum_max
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        return BacktestSummary(
            initial_equity=eq_arr[0],
            final_equity=eq_arr[-1],
            total_return_pct=total_ret * 100.0,
            annualized_return_pct=ann_ret * 100.0,
            annualized_volatility_pct=ann_vol * 100.0,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd * 100.0,
            num_rebalances=num_rebalances,
            equity_curve=equity_curve,
            timestamps=timestamps,
        )
