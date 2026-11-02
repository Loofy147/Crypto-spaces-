"""Deterministic tests for the canonical OHLCV contract and public adapters."""

from __future__ import annotations

import pytest

from src.data.market_data_providers import (
    CoinbaseExchangeClient,
    KrakenSpotClient,
    MarketDataProviderError,
)
from src.data.ohlcv_contract import (
    MarketCandle,
    audit_candle_series,
    compare_venue_closes,
    finalized_contiguous_log_returns,
    payload_sha256,
    rollup_candles,
)


def make_candle(
    *,
    venue: str = "coinbase",
    instrument: str = "ETH-USD",
    interval: str = "1h",
    start: int = 0,
    close: float = 100.0,
    final: bool = True,
    volume: float = 2.0,
) -> MarketCandle:
    return MarketCandle(
        venue=venue,
        instrument=instrument,
        base_currency="ETH",
        quote_currency="USD",
        interval=interval,
        source_bar_timestamp=start,
        event_time=start + {"1h": 3600, "4h": 14400}[interval],
        open=close - 1.0,
        high=close + 1.0,
        low=close - 2.0,
        close=close,
        volume=volume,
        ingested_at=1_800_000_000,
        is_final=final,
        source_payload_sha256=payload_sha256([venue, instrument, start, close]),
    )


def test_contract_preserves_start_and_exclusive_end_boundary() -> None:
    item = make_candle(start=3_600)
    assert item.source_bar_timestamp == 3_600
    assert item.event_time == 7_200
    assert item.source_timestamp_semantics == "BAR_START"
    assert item.canonical_timestamp_semantics == "BAR_END_EXCLUSIVE"


def test_contract_rejects_invalid_ohlc_and_temporal_alignment() -> None:
    with pytest.raises(ValueError, match="high/low"):
        MarketCandle(
            venue="coinbase", instrument="ETH-USD", base_currency="ETH",
            quote_currency="USD", interval="1h", source_bar_timestamp=0,
            event_time=3600, open=100, high=99, low=98, close=100,
            volume=1, ingested_at=1_800_000_000, is_final=True,
            source_payload_sha256=payload_sha256("bad"),
        )
    with pytest.raises(ValueError, match="aligned"):
        make_candle(start=1)


def test_series_audit_reports_gaps_and_never_fills_them() -> None:
    rows = [make_candle(start=0), make_candle(start=7_200)]
    audit = audit_candle_series(rows)
    assert audit.structurally_valid
    assert audit.unobserved_intervals == ((3_600, 10_800, 1),)
    assert len(rows) == 2


def test_series_audit_detects_duplicates_and_out_of_order_rows() -> None:
    first = make_candle(start=0)
    second = make_candle(start=3_600)
    audit = audit_candle_series([second, first, second])
    assert not audit.structurally_valid
    assert audit.duplicate_event_times == (7_200,)
    assert audit.out_of_order_indices == (1,)


def test_log_returns_exclude_partial_bars_and_refuse_gaps() -> None:
    rows = [
        make_candle(start=0, close=100.0),
        make_candle(start=3_600, close=110.0),
        make_candle(start=7_200, close=120.0, final=False),
    ]
    returns = finalized_contiguous_log_returns(rows)
    assert len(returns) == 1
    assert returns[0] == pytest.approx(__import__("math").log(1.1))
    with pytest.raises(ValueError, match="unobserved"):
        finalized_contiguous_log_returns([rows[0], make_candle(start=7_200)])


def test_four_hour_rollup_requires_four_complete_utc_hour_bars() -> None:
    rows = [
        make_candle(start=0, close=101.0, volume=1.0),
        make_candle(start=3_600, close=102.0, volume=2.0),
        make_candle(start=7_200, close=103.0, volume=3.0),
        make_candle(start=10_800, close=104.0, volume=4.0),
        make_candle(start=14_400, close=105.0, volume=5.0),
    ]
    result = rollup_candles(rows, "4h")
    assert len(result) == 1
    assert result[0].source_bar_timestamp == 0
    assert result[0].event_time == 14_400
    assert result[0].open == rows[0].open
    assert result[0].close == rows[3].close
    assert result[0].volume == pytest.approx(10.0)
    assert len(rollup_candles(rows[:3], "4h")) == 0


def test_cross_venue_comparison_uses_common_boundaries_without_blending() -> None:
    coinbase = [make_candle(start=0, close=100.0), make_candle(start=3_600, close=101.0)]
    kraken = [
        make_candle(venue="kraken", instrument="XETHZUSD", start=0, close=99.9),
        make_candle(venue="kraken", instrument="XETHZUSD", start=3_600, close=101.1),
    ]
    comparison = compare_venue_closes(coinbase, kraken)
    assert comparison.common_observations == 2
    assert comparison.left_venue == "coinbase"
    assert comparison.right_venue == "kraken"
    assert comparison.mean_difference_bps != 0.0
    assert len(comparison.observations) == 2


def test_coinbase_adapter_sorts_rows_and_marks_unfinished_candle() -> None:
    start = 36_000
    rows = [
        [start + 7_200, 102, 104, 103, 103.5, 12],
        [start + 3_600, 101, 103, 102, 102.5, 10],
        [start, 100, 102, 101, 101.5, 8],
        [start - 3_600, 99, 101, 100, 100.5, 7],
    ]

    def getter(url: str):
        assert "api.exchange.coinbase.com" in url
        return rows

    batch = CoinbaseExchangeClient(json_getter=getter).fetch_range(
        "ETH-USD", "1h", start, start + 10_800,
        as_of_timestamp=start + 7_200,
    )
    assert batch.provider == "coinbase"
    assert batch.returned_instrument == "ETH-USD"
    assert len(batch.candles) == 3
    assert [c.source_bar_timestamp for c in batch.candles] == [
        start, start + 3_600, start + 7_200
    ]
    assert [c.is_final for c in batch.candles] == [True, True, False]
    assert batch.candles[0].volume == 8.0
    assert batch.raw_response_sha256 == payload_sha256([rows])


def test_coinbase_adapter_rejects_unsupported_interval_and_bad_schema() -> None:
    client = CoinbaseExchangeClient(json_getter=lambda _: [])
    with pytest.raises(ValueError, match="does not document support"):
        client.fetch_range("ETH-USD", "4h", 0, 14_400)
    bad = CoinbaseExchangeClient(json_getter=lambda _: [[0, 1, 2, 1.5]])
    with pytest.raises(MarketDataProviderError, match="timestamp, low, high"):
        bad.fetch_range("ETH-USD", "1h", 0, 3_600, as_of_timestamp=10_000)


def test_kraken_adapter_retains_trade_count_and_current_candle_status() -> None:
    start = 36_000
    payload = {
        "error": [],
        "result": {
            "XETHZUSD": [
                [start, "100", "103", "99", "102", "101", "12.5", 42],
                [start + 3_600, "102", "105", "101", "104", "103", "9.5", 31],
            ],
            "last": start + 7_200,
        },
    }
    batch = KrakenSpotClient(json_getter=lambda _: payload).fetch_latest(
        "ETHUSD", "1h", as_of_timestamp=start + 3_600,
    )
    assert batch.returned_instrument == "XETHZUSD"
    assert len(batch.candles) == 2
    assert batch.candles[0].is_final is True
    assert batch.candles[1].is_final is False
    assert batch.candles[0].trade_count == 42
    assert batch.candles[0].volume == pytest.approx(12.5)


def test_kraken_adapter_surfaces_api_errors() -> None:
    payload = {"error": ["EGeneral:Temporary lockout"], "result": {}}
    client = KrakenSpotClient(json_getter=lambda _: payload)
    with pytest.raises(MarketDataProviderError, match="Temporary lockout"):
        client.fetch_latest()


def test_adapters_keep_exchange_currencies_and_series_separate() -> None:
    rows = [
        make_candle(venue="coinbase", instrument="ETH-USD"),
        make_candle(venue="kraken", instrument="XETHZUSD"),
    ]
    audit = audit_candle_series(rows)
    assert audit.mixed_series is True
    assert not audit.structurally_valid


def test_coinbase_adapter_preserves_missing_volume_as_unknown() -> None:
    rows = [[0, 99, 101, 100, 100.5]]
    batch = CoinbaseExchangeClient(json_getter=lambda _: rows).fetch_range(
        "ETH-USD", "1h", 0, 3_600, as_of_timestamp=10_000,
    )
    assert len(batch.candles) == 1
    assert batch.candles[0].volume is None


def test_coinbase_adapter_splits_large_ranges_below_documented_page_limit() -> None:
    calls: list[str] = []
    client = CoinbaseExchangeClient(
        json_getter=lambda url: calls.append(url) or [],
        page_delay_seconds=0.0,
    )
    client.fetch_range("ETH-USD", "1h", 0, 3_600 * 300)
    assert len(calls) == 2
    from urllib.parse import parse_qs, urlparse
    first = parse_qs(urlparse(calls[0]).query)
    second = parse_qs(urlparse(calls[1]).query)
    assert first["start"][0].startswith("1970-01-01T00:00:00")
    assert first["end"][0] == second["start"][0]
