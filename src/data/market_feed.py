"""Market Feed Normalizer using Polars and PyArrow."""

from typing import Any, Dict, List
import polars as pl
from pydantic import BaseModel, ConfigDict
from src.core.events import DerivativesMetricsEvent, MarketTickEvent, OrderBookDepthEvent


class NormalizedTickData(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: float
    symbol: str
    price: float
    volume: float


class MarketFeedNormalizer:
    """Normalizes raw tick, orderbook depth, and derivatives metrics."""

    @staticmethod
    def normalize_ticks_to_polars(raw_ticks: List[Dict[str, Any]]) -> pl.DataFrame:
        """Converts raw tick dictionary records into a typed Polars DataFrame."""
        if not raw_ticks:
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Float64,
                    "symbol": pl.Utf8,
                    "price": pl.Float64,
                    "volume": pl.Float64,
                }
            )

        df = pl.DataFrame(raw_ticks)
        return df.select(
            [
                pl.col("timestamp").cast(pl.Float64),
                pl.col("symbol").cast(pl.Utf8),
                pl.col("price").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            ]
        )

    @staticmethod
    def create_tick_event(
        event_id: str,
        timestamp: float,
        symbol: str,
        price: float,
        volume_24h: float = 0.0,
        sequence: int = 0,
    ) -> MarketTickEvent:
        """Creates a strongly typed MarketTickEvent."""
        return MarketTickEvent(
            event_id=event_id,
            timestamp=timestamp,
            sequence=sequence,
            symbol=symbol,
            price=price,
            volume_24h=volume_24h,
        )

    @staticmethod
    def create_depth_event(
        event_id: str,
        timestamp: float,
        symbol: str,
        bids: Dict[float, float],
        asks: Dict[float, float],
        sequence: int = 0,
    ) -> OrderBookDepthEvent:
        """Computes top 2% bid/ask cumulative depth and creates OrderBookDepthEvent."""
        if not bids or not asks:
            return OrderBookDepthEvent(
                event_id=event_id,
                timestamp=timestamp,
                sequence=sequence,
                symbol=symbol,
                bids=bids,
                asks=asks,
                depth_2pct_bid=0.0,
                depth_2pct_ask=0.0,
            )

        best_bid = max(bids.keys())
        best_ask = min(asks.keys())
        mid_price = (best_bid + best_ask) / 2.0

        bid_threshold = mid_price * 0.98
        ask_threshold = mid_price * 1.02

        depth_2pct_bid = sum(size for price, size in bids.items() if price >= bid_threshold)
        depth_2pct_ask = sum(size for price, size in asks.items() if price <= ask_threshold)

        return OrderBookDepthEvent(
            event_id=event_id,
            timestamp=timestamp,
            sequence=sequence,
            symbol=symbol,
            bids=bids,
            asks=asks,
            depth_2pct_bid=depth_2pct_bid,
            depth_2pct_ask=depth_2pct_ask,
        )
