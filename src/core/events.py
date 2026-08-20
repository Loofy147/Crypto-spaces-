"""Immutable event definitions for event-sourced portfolio engine."""

from typing import Dict, Literal, Union
from pydantic import BaseModel, ConfigDict, Field


class BaseEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    timestamp: float
    sequence: int = 0


class MarketTickEvent(BaseEvent):
    event_type: Literal["MARKET_TICK"] = "MARKET_TICK"
    symbol: str
    price: float
    volume_24h: float = 0.0


class OrderBookDepthEvent(BaseEvent):
    event_type: Literal["ORDER_BOOK_DEPTH"] = "ORDER_BOOK_DEPTH"
    symbol: str
    bids: Dict[float, float]  # price -> size
    asks: Dict[float, float]  # price -> size
    depth_2pct_bid: float  # cumulative depth within 2% of mid
    depth_2pct_ask: float


class DerivativesMetricsEvent(BaseEvent):
    event_type: Literal["DERIVATIVES_METRICS"] = "DERIVATIVES_METRICS"
    symbol: str
    crypto_margined_oi: float
    cash_margined_oi: float
    spot_cvd: float
    perp_cvd: float
    funding_rate: float


class RebalanceTriggeredEvent(BaseEvent):
    event_type: Literal["REBALANCE_TRIGGERED"] = "REBALANCE_TRIGGERED"
    reason: str
    target_weights: Dict[str, float]


class OrderExecutedEvent(BaseEvent):
    event_type: Literal["ORDER_EXECUTED"] = "ORDER_EXECUTED"
    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    units: float
    execution_price: float
    fee: float


class YieldAccruedEvent(BaseEvent):
    event_type: Literal["YIELD_ACCRUED"] = "YIELD_ACCRUED"
    asset_symbol: str
    amount: float
    apy_rate: float


PortfolioEvent = Union[
    MarketTickEvent,
    OrderBookDepthEvent,
    DerivativesMetricsEvent,
    RebalanceTriggeredEvent,
    OrderExecutedEvent,
    YieldAccruedEvent,
]
