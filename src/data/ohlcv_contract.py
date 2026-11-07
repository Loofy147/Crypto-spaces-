"""Canonical OHLCV candle contracts and deterministic validation utilities."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

INTERVAL_SECONDS: Mapping[str, int] = {
    "1m": 60, "5m": 300, "15m": 900, "1h": 3600,
    "4h": 14400, "6h": 21600, "1d": 86400,
}


def payload_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MarketCandle:
    venue: str
    instrument: str
    base_currency: str
    quote_currency: str
    interval: str
    source_bar_timestamp: int
    event_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    ingested_at: int
    is_final: bool
    source_payload_sha256: str
    trade_count: int | None = None
    source_timestamp_semantics: str = "BAR_START"
    canonical_timestamp_semantics: str = "BAR_END_EXCLUSIVE"

    def __post_init__(self) -> None:
        if not all((self.venue, self.instrument, self.base_currency, self.quote_currency)):
            raise ValueError("venue, instrument and currency identifiers must be non-empty")
        if self.interval not in INTERVAL_SECONDS:
            raise ValueError(f"unsupported interval: {self.interval}")
        duration = INTERVAL_SECONDS[self.interval]
        if self.source_timestamp_semantics != "BAR_START":
            raise ValueError("source timestamp semantics must be BAR_START")
        if self.canonical_timestamp_semantics != "BAR_END_EXCLUSIVE":
            raise ValueError("canonical timestamp semantics must be BAR_END_EXCLUSIVE")
        if self.source_bar_timestamp < 0 or self.ingested_at <= 0:
            raise ValueError("timestamps must be non-negative and ingestion time positive")
        if self.source_bar_timestamp % duration != 0:
            raise ValueError("source bar timestamp is not aligned to UTC interval boundary")
        if self.event_time != self.source_bar_timestamp + duration:
            raise ValueError("event_time must equal source_bar_timestamp + interval")
        prices = (self.open, self.high, self.low, self.close)
        if not all(math.isfinite(value) for value in prices):
            raise ValueError("OHLC prices must be finite")
        if self.volume is not None and not math.isfinite(self.volume):
            raise ValueError("volume must be finite when supplied")
        if min(self.open, self.high, self.low, self.close) <= 0.0:
            raise ValueError("OHLC prices must be positive")
        if self.volume is not None and self.volume < 0.0:
            raise ValueError("volume must be non-negative")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("OHLC values violate candle high/low bounds")
        if self.low > self.high:
            raise ValueError("low cannot exceed high")
        if self.trade_count is not None and self.trade_count < 0:
            raise ValueError("trade_count must be non-negative")
        digest = self.source_payload_sha256
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError("source_payload_sha256 must be a 64-character hex digest")

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.venue, self.instrument, self.interval, self.event_time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandleSeriesAudit:
    rows: int
    duplicate_event_times: tuple[int, ...]
    out_of_order_indices: tuple[int, ...]
    unobserved_intervals: tuple[tuple[int, int, int], ...]
    mixed_series: bool

    @property
    def structurally_valid(self) -> bool:
        return self.rows > 0 and not self.duplicate_event_times and not self.out_of_order_indices and not self.mixed_series


@dataclass(frozen=True)
class VenuePriceDifference:
    event_time: int
    left_close: float
    right_close: float
    difference_bps: float


@dataclass(frozen=True)
class VenueComparison:
    left_venue: str
    right_venue: str
    common_observations: int
    mean_difference_bps: float
    median_difference_bps: float
    max_absolute_difference_bps: float
    observations: tuple[VenuePriceDifference, ...]


def audit_candle_series(candles: Sequence[MarketCandle]) -> CandleSeriesAudit:
    """Report temporal integrity and cadence gaps; never fill missing bars."""
    if not candles:
        return CandleSeriesAudit(0, (), (), (), False)
    identity = {(c.venue, c.instrument, c.base_currency, c.quote_currency, c.interval) for c in candles}
    times = [c.event_time for c in candles]
    seen: set[int] = set()
    duplicates: set[int] = set()
    for value in times:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    out_of_order = tuple(i for i in range(1, len(times)) if times[i] <= times[i - 1])
    duration = INTERVAL_SECONDS[candles[0].interval]
    ordered = sorted(set(times))
    gaps = tuple(
        (prior, current, (current - prior) // duration - 1)
        for prior, current in zip(ordered, ordered[1:])
        if current - prior > duration
    )
    return CandleSeriesAudit(
        rows=len(candles),
        duplicate_event_times=tuple(sorted(duplicates)),
        out_of_order_indices=out_of_order,
        unobserved_intervals=gaps,
        mixed_series=len(identity) != 1,
    )


def finalized_contiguous_log_returns(candles: Sequence[MarketCandle]) -> tuple[float, ...]:
    """Create log returns only from finalized contiguous bars of one series."""
    import math as math_module

    finalized = sorted((c for c in candles if c.is_final), key=lambda c: c.event_time)
    if len(finalized) < 2:
        raise ValueError("at least two finalized candles are required")
    identity = {(c.venue, c.instrument, c.base_currency, c.quote_currency, c.interval) for c in finalized}
    if len(identity) != 1:
        raise ValueError("returns cannot mix venues, instruments, quote currencies or intervals")
    duration = INTERVAL_SECONDS[finalized[0].interval]
    result: list[float] = []
    for previous, current in zip(finalized, finalized[1:]):
        if current.event_time - previous.event_time != duration:
            raise ValueError("cannot calculate returns across an unobserved candle interval")
        result.append(math_module.log(current.close / previous.close))
    return tuple(result)


def rollup_candles(candles: Sequence[MarketCandle], target_interval: str) -> tuple[MarketCandle, ...]:
    """Aggregate full, UTC-aligned source bars into a larger interval."""
    if target_interval not in INTERVAL_SECONDS:
        raise ValueError(f"unsupported target interval: {target_interval}")
    if not candles:
        return ()
    source = candles[0]
    source_seconds = INTERVAL_SECONDS[source.interval]
    target_seconds = INTERVAL_SECONDS[target_interval]
    if target_seconds <= source_seconds or target_seconds % source_seconds != 0:
        raise ValueError("target interval must be an integer multiple of source interval")
    if any(
        (c.venue, c.instrument, c.base_currency, c.quote_currency, c.interval)
        != (source.venue, source.instrument, source.base_currency, source.quote_currency, source.interval)
        for c in candles
    ):
        raise ValueError("rollup cannot mix sources, instruments, quote currencies or intervals")
    buckets: dict[int, list[MarketCandle]] = {}
    for candle in candles:
        if candle.is_final:
            start = candle.source_bar_timestamp
            bucket = start - (start % target_seconds)
            buckets.setdefault(bucket, []).append(candle)
    required = target_seconds // source_seconds
    output: list[MarketCandle] = []
    for bucket_start, rows in sorted(buckets.items()):
        rows.sort(key=lambda c: c.source_bar_timestamp)
        expected = [bucket_start + index * source_seconds for index in range(required)]
        if [c.source_bar_timestamp for c in rows] != expected:
            continue
        counts = [c.trade_count for c in rows]
        volumes = [c.volume for c in rows]
        total_volume = (
            sum(float(value) for value in volumes if value is not None)
            if all(value is not None for value in volumes) else None
        )
        total_trade_count = (
            sum(int(value) for value in counts if value is not None)
            if all(value is not None for value in counts) else None
        )
        output.append(MarketCandle(
            venue=source.venue, instrument=source.instrument,
            base_currency=source.base_currency, quote_currency=source.quote_currency,
            interval=target_interval, source_bar_timestamp=bucket_start,
            event_time=bucket_start + target_seconds,
            open=rows[0].open, high=max(c.high for c in rows),
            low=min(c.low for c in rows), close=rows[-1].close,
            volume=total_volume,
            trade_count=total_trade_count,
            ingested_at=max(c.ingested_at for c in rows), is_final=True,
            source_payload_sha256=payload_sha256([c.source_payload_sha256 for c in rows]),
        ))
    return tuple(output)


def compare_venue_closes(left: Sequence[MarketCandle], right: Sequence[MarketCandle]) -> VenueComparison:
    """Compare common finalized closes across venues; never blend or average them."""
    if not left or not right:
        raise ValueError("both venue series must be non-empty")
    a, b = left[0], right[0]
    if (a.base_currency, a.quote_currency, a.interval) != (b.base_currency, b.quote_currency, b.interval):
        raise ValueError("venue series must share base, quote and interval")
    if a.venue == b.venue:
        raise ValueError("venue comparison requires two distinct providers")
    if any(c.venue != a.venue for c in left) or any(c.venue != b.venue for c in right):
        raise ValueError("each comparison input must contain exactly one venue")
    left_map = {c.event_time: c for c in left if c.is_final}
    right_map = {c.event_time: c for c in right if c.is_final}
    common = sorted(set(left_map).intersection(right_map))
    if not common:
        raise ValueError("venue series have no common finalized event_time boundaries")
    observations = tuple(
        VenuePriceDifference(
            event_time=ts, left_close=left_map[ts].close, right_close=right_map[ts].close,
            difference_bps=(left_map[ts].close - right_map[ts].close)
            / ((left_map[ts].close + right_map[ts].close) / 2.0) * 10_000.0,
        )
        for ts in common
    )
    differences = sorted(item.difference_bps for item in observations)
    middle = len(differences) // 2
    median = differences[middle] if len(differences) % 2 else (differences[middle - 1] + differences[middle]) / 2.0
    return VenueComparison(
        left_venue=a.venue, right_venue=b.venue, common_observations=len(observations),
        mean_difference_bps=float(sum(differences) / len(differences)),
        median_difference_bps=float(median),
        max_absolute_difference_bps=float(max(abs(value) for value in differences)),
        observations=observations,
    )


def longest_contiguous_finalized_segment(
    candles: Sequence[MarketCandle],
) -> tuple[MarketCandle, ...]:
    """Return the longest finalized gap-free segment from exactly one series.

    A gap starts a new segment. Partial candles are excluded. Ties prefer the
    most recent segment so the returned sample is not unnecessarily stale.
    """
    finalized = sorted(
        (candle for candle in candles if candle.is_final),
        key=lambda candle: candle.event_time,
    )
    if not finalized:
        return ()
    identity = {
        (c.venue, c.instrument, c.base_currency, c.quote_currency, c.interval)
        for c in finalized
    }
    if len(identity) != 1:
        raise ValueError("segment selection cannot mix venues, instruments, quotes or intervals")
    duration = INTERVAL_SECONDS[finalized[0].interval]
    segments: list[list[MarketCandle]] = []
    current: list[MarketCandle] = []
    for candle in finalized:
        if current and candle.event_time - current[-1].event_time != duration:
            segments.append(current)
            current = []
        current.append(candle)
    if current:
        segments.append(current)
    best = max(segments, key=lambda segment: (len(segment), segment[-1].event_time))
    return tuple(best)
