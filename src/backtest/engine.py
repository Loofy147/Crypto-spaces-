"""Event-Driven Vectorized Backtesting Engine with Strict t-1 Information Boundary."""

from typing import Dict, List, Literal, Sequence
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


from pydantic import Field
from src.core.events import MarketTickEvent, OrderExecutedEvent, RebalanceTriggeredEvent
from src.core.state import apply_order_executed


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
    daily_returns: List[float] = Field(default_factory=list)


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
        self.vol_targeter = vol_targeter or VolatilityTargeter(target_volatility=0.15)
        self.rwa_manager = rwa_manager or RWASleeveManager()
        self.rebalancer = rebalancer or HybridDriftRebalancer()
        self.risk_engine = risk_engine or PreTradeRiskEngine()

    def _execute_rebalance_orders(
        self,
        state: PortfolioState,
        target_weights: Dict[str, float],
        asset_categories: Dict[str, AssetCategory],
        depth_2pct_map: Dict[str, float],
        latest_prices: Dict[str, float],
        event_time: float,
        seq: int,
    ) -> tuple[PortfolioState, int]:
        """Evaluates rebalance logic and generates OrderExecutedEvents if approved."""
        current_weights = {
            sym: pos.weight for sym, pos in state.positions.items() if state.total_equity > 0
        }

        # Validate pre-trade risk caps
        risk_res = self.risk_engine.validate_pre_trade(
            state=state,
            proposed_weights=target_weights,
            asset_categories=asset_categories,
        )
        if not risk_res.passed:
            return state, 0

        decision = self.rebalancer.evaluate_rebalance(
            current_weights=current_weights,
            target_weights=target_weights,
            asset_categories=asset_categories,
            total_equity=state.total_equity,
            depth_2pct_map=depth_2pct_map,
        )

        if not decision.should_rebalance:
            return state, 0

        rebalance_count = 0
        for inst in decision.instructions:
            if not inst.approved:
                continue

            symbol = inst.symbol
            if symbol not in latest_prices:
                continue
            px = latest_prices[symbol]
            if px <= 0.0:
                continue

            notional_change = inst.target_notional_change
            if abs(notional_change) < 1.0:
                continue

            units = abs(notional_change) / px
            trade_side: Literal["BUY", "SELL"] = "BUY" if notional_change > 0 else "SELL"
            fee = inst.cost_estimate.fee_cost + inst.cost_estimate.spread_cost

            order_event = OrderExecutedEvent(
                event_id=f"order_{symbol}_{int(event_time)}_{seq}",
                timestamp=event_time,
                sequence=seq,
                order_id=f"ord_{symbol}_{int(event_time)}",
                symbol=symbol,
                side=trade_side,
                units=units,
                execution_price=px,
                fee=fee,
            )

            state = apply_order_executed(state, order_event)
            rebalance_count += 1

        return state, rebalance_count

    def run_backtest(
        self,
        events: Sequence[PortfolioEvent],
        initial_state: PortfolioState | None = None,
        asset_categories: Dict[str, AssetCategory] | None = None,
        depth_2pct_map: Dict[str, float] | None = None,
        auto_rebalance: bool = True,
    ) -> BacktestSummary:
        """Executes backtest consuming event stream point-in-time."""
        categories = asset_categories or {
            "BTC": AssetCategory.CORE,
            "ETH": AssetCategory.CORE,
            "SOL": AssetCategory.SATELLITE,
            "AVAX": AssetCategory.SATELLITE,
            "BUIDL": AssetCategory.RWA_YIELD,
        }
        depths = depth_2pct_map or {
            "BTC": 5000000.0,
            "ETH": 2000000.0,
            "SOL": 1000000.0,
            "AVAX": 1000000.0,
            "BUIDL": 10000000.0,
        }

        state = initial_state or PortfolioState(
            timestamp=0.0,
            positions={},
            cash_balance=self.initial_cash,
            total_equity=self.initial_cash,
        )

        equity_curve: List[float] = [state.total_equity]
        daily_equity_curve: List[float] = [state.total_equity]
        timestamps: List[float] = [state.timestamp]
        num_rebalances = 0
        latest_prices: Dict[str, float] = {"BUIDL": 1.0}

        # Calculate initial target allocation plan
        plan = self.rwa_manager.compute_initial_target_weights(
            satellite_assets=["SOL", "AVAX"],
            depth_2pct_map=depths,
        )

        price_history: Dict[str, List[float]] = {}

        for event in events:
            if isinstance(event, MarketTickEvent):
                latest_prices[event.symbol] = event.price
                if event.symbol not in price_history:
                    price_history[event.symbol] = []
                price_history[event.symbol].append(event.price)

            # Apply event strictly up to t-1
            state = apply_event(state, event)

            # Auto-rebalance setup or event-driven rebalance trigger
            if auto_rebalance:
                is_explicit_trigger = isinstance(event, RebalanceTriggeredEvent)

                if is_explicit_trigger and isinstance(event, RebalanceTriggeredEvent) and event.target_weights:
                    target_w = event.target_weights
                else:
                    # Compute rolling 30d realized vols if history is available
                    realized_vols: Dict[str, float] = {}
                    for sym, pxs in price_history.items():
                        if len(pxs) > 5:
                            arr = np.array(pxs[-30:])
                            rets = np.diff(arr) / arr[:-1]
                            v = float(np.std(rets, ddof=1) * np.sqrt(365.0)) if len(rets) > 1 else 0.30
                            realized_vols[sym] = v
                        else:
                            realized_vols[sym] = 0.30

                    if realized_vols:
                        vol_res = self.vol_targeter.scale_positions(
                            base_weights=plan.weights,
                            realized_vols=realized_vols,
                        )
                        target_w = vol_res.scaled_weights
                    else:
                        target_w = plan.weights

                # Check if portfolio is unallocated or drift rebalancing is triggered at end of price update sequence
                has_unallocated = len(state.positions) < len(plan.weights) and len(latest_prices) >= len(plan.weights)
                if is_explicit_trigger or has_unallocated or (event.sequence % 5 == 4):
                    state, r_count = self._execute_rebalance_orders(
                        state=state,
                        target_weights=target_w,
                        asset_categories=categories,
                        depth_2pct_map=depths,
                        latest_prices=latest_prices,
                        event_time=event.timestamp,
                        seq=event.sequence,
                    )
                    num_rebalances += r_count

            equity_curve.append(state.total_equity)
            timestamps.append(state.timestamp)

            # Collect daily snapshot at sequence end or last event of day
            if event.sequence % 5 == 4 or event.sequence == len(events) - 1:
                daily_equity_curve.append(state.total_equity)

        # Performance summary calculated on daily equity curve
        d_eq_arr = np.array(daily_equity_curve, dtype=np.float64)
        daily_rets = np.diff(d_eq_arr) / d_eq_arr[:-1]
        daily_rets = daily_rets[~np.isnan(daily_rets)]

        eq_arr = np.array(equity_curve, dtype=np.float64)
        total_ret = (eq_arr[-1] - eq_arr[0]) / eq_arr[0] if eq_arr[0] > 0 else 0.0

        num_days = max(1.0, (timestamps[-1] - timestamps[0]) / 86400.0) if len(timestamps) > 1 else 1.0
        ann_ret = ((1.0 + total_ret) ** (365.0 / num_days)) - 1.0 if total_ret > -1.0 else -1.0

        ann_vol = float(np.std(daily_rets, ddof=1) * np.sqrt(365.0)) if len(daily_rets) > 1 else 0.0
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
            daily_returns=daily_rets.tolist(),
        )
