"""Public spot OHLCV adapters for Coinbase Exchange and Kraken Spot REST."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

from src.data.ohlcv_contract import INTERVAL_SECONDS, MarketCandle, payload_sha256


class MarketDataProviderError(RuntimeError):
    """Raised for provider transport, response, or schema failures."""


JsonGetter = Callable[[str], Any]


def _default_json_getter(url: str) -> Any:
    request = Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "institutional-quant-engine/0.1",
    })
    retryable_statuses = {429, 500, 502, 503, 504}
    for attempt in range(3):
        try:
            with urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code in retryable_statuses and attempt < 2:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = float(retry_after) if retry_after else 0.5 * (2 ** attempt)
                except ValueError:
                    delay = 0.5 * (2 ** attempt)
                time.sleep(min(max(delay, 0.0), 10.0))
                continue
            raise MarketDataProviderError(f"market-data HTTP request failed ({exc.code}): {exc}") from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
                continue
            raise MarketDataProviderError(f"market-data request failed: {exc}") from exc
    raise MarketDataProviderError("market-data request exhausted retry attempts")

def _utc_iso(epoch_seconds: int) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _epoch_now() -> int:
    return int(time.time())


@dataclass(frozen=True)
class CandleFetchBatch:
    provider: str
    requested_instrument: str
    returned_instrument: str
    base_currency: str
    quote_currency: str
    interval: str
    retrieved_at: int
    requested_start: int | None
    requested_end: int | None
    endpoint: str
    source_documentation: str
    raw_response_sha256: str
    candles: tuple[MarketCandle, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "requested_instrument": self.requested_instrument,
            "returned_instrument": self.returned_instrument,
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "interval": self.interval,
            "retrieved_at": self.retrieved_at,
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "endpoint": self.endpoint,
            "source_documentation": self.source_documentation,
            "raw_response_sha256": self.raw_response_sha256,
            "row_count": len(self.candles),
            "finalized_count": sum(1 for candle in self.candles if candle.is_final),
        }


class CoinbaseExchangeClient:
    """Coinbase Exchange public candles endpoint, with bounded time pagination."""

    ENDPOINT = "https://api.exchange.coinbase.com/products/{product_id}/candles"
    DOCUMENTATION = "https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles"
    GRANULARITY_SECONDS: Mapping[str, int] = {
        "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400,
    }

    def __init__(
        self, json_getter: JsonGetter = _default_json_getter, page_delay_seconds: float = 0.15
    ) -> None:
        if page_delay_seconds < 0.0:
            raise ValueError("page_delay_seconds must be non-negative")
        self._get_json = json_getter
        self._page_delay_seconds = page_delay_seconds

    def fetch_range(
        self,
        product_id: str,
        interval: str,
        start_timestamp: int,
        end_timestamp: int,
        *,
        as_of_timestamp: int | None = None,
    ) -> CandleFetchBatch:
        if interval not in self.GRANULARITY_SECONDS:
            raise ValueError(f"Coinbase does not document support for interval {interval}")
        duration = self.GRANULARITY_SECONDS[interval]
        if start_timestamp < 0 or end_timestamp <= start_timestamp:
            raise ValueError("requested time range must be positive and end after start")
        if start_timestamp % duration or end_timestamp % duration:
            raise ValueError("requested start and end must align to the UTC interval boundary")
        retrieved_at = _epoch_now()
        as_of = retrieved_at if as_of_timestamp is None else as_of_timestamp
        pieces: list[Any] = []
        records_by_source_start: dict[int, MarketCandle] = {}
        cursor = start_timestamp
        page_span = duration * 299
        base, separator, quote_currency = product_id.upper().partition("-")
        if not separator or not base or not quote_currency:
            raise ValueError("Coinbase product_id must use BASE-QUOTE notation")
        endpoint = self.ENDPOINT.format(product_id=quote(product_id.upper(), safe="-"))

        while cursor < end_timestamp:
            page_end = min(end_timestamp, cursor + page_span)
            params = urlencode({
                "start": _utc_iso(cursor),
                "end": _utc_iso(page_end),
                "granularity": str(duration),
            })
            url = f"{endpoint}?{params}"
            payload = self._get_json(url)
            if not isinstance(payload, list):
                raise MarketDataProviderError("Coinbase candles response must be a JSON array")
            pieces.append(payload)
            for row in payload:
                if not isinstance(row, (list, tuple)) or len(row) not in (5, 6):
                    raise MarketDataProviderError("Coinbase candle row must contain timestamp, low, high, open, close and optional volume")
                try:
                    source_start = int(row[0])
                    low, high, open_price, close = (float(value) for value in row[1:5])
                    volume = float(row[5]) if len(row) == 6 else None
                except (TypeError, ValueError, OverflowError) as exc:
                    raise MarketDataProviderError(f"Coinbase candle contains an invalid field: {exc}") from exc
                if source_start < start_timestamp or source_start >= end_timestamp:
                    continue
                candle = MarketCandle(
                    venue="coinbase",
                    instrument=product_id.upper(),
                    base_currency=base,
                    quote_currency=quote_currency,
                    interval=interval,
                    source_bar_timestamp=source_start,
                    event_time=source_start + duration,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    ingested_at=retrieved_at,
                    is_final=(source_start + duration) <= as_of,
                    source_payload_sha256=payload_sha256(row),
                )
                previous = records_by_source_start.get(source_start)
                if previous is not None:
                    if previous.source_payload_sha256 != candle.source_payload_sha256:
                        raise MarketDataProviderError(
                            "Coinbase returned conflicting duplicate candle boundaries across pages"
                        )
                    continue
                records_by_source_start[source_start] = candle
            cursor = page_end
            if cursor < end_timestamp and self._page_delay_seconds:
                time.sleep(self._page_delay_seconds)

        records = sorted(
            records_by_source_start.values(),
            key=lambda candle: candle.source_bar_timestamp,
        )
        digest = payload_sha256(pieces)
        return CandleFetchBatch(
            provider="coinbase",
            requested_instrument=product_id.upper(),
            returned_instrument=product_id.upper(),
            base_currency=base,
            quote_currency=quote_currency,
            interval=interval,
            retrieved_at=retrieved_at,
            requested_start=start_timestamp,
            requested_end=end_timestamp,
            endpoint=endpoint,
            source_documentation=self.DOCUMENTATION,
            raw_response_sha256=digest,
            candles=tuple(records),
        )


class KrakenSpotClient:
    """Kraken Spot OHLC endpoint. Kraken returns the current uncommitted candle."""

    ENDPOINT = "https://api.kraken.com/0/public/OHLC"
    DOCUMENTATION = "https://docs.kraken.com/api-reference/market-data/get-ohlc-data"
    INTERVAL_MINUTES: Mapping[str, int] = {
        "1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440,
    }

    def __init__(self, json_getter: JsonGetter = _default_json_getter) -> None:
        self._get_json = json_getter

    def fetch_latest(
        self,
        pair: str = "ETHUSD",
        interval: str = "1h",
        *,
        as_of_timestamp: int | None = None,
    ) -> CandleFetchBatch:
        if interval not in self.INTERVAL_MINUTES:
            raise ValueError(f"Kraken does not document support for interval {interval}")
        retrieved_at = _epoch_now()
        as_of = retrieved_at if as_of_timestamp is None else as_of_timestamp
        minutes = self.INTERVAL_MINUTES[interval]
        params = urlencode({"pair": pair, "interval": str(minutes)})
        url = f"{self.ENDPOINT}?{params}"
        payload = self._get_json(url)
        if not isinstance(payload, dict):
            raise MarketDataProviderError("Kraken OHLC response must be a JSON object")
        errors = payload.get("error")
        if errors:
            raise MarketDataProviderError(f"Kraken OHLC returned errors: {errors}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise MarketDataProviderError("Kraken OHLC response is missing result")
        pair_keys = [key for key in result if key != "last"]
        if len(pair_keys) != 1:
            raise MarketDataProviderError("Kraken response did not resolve exactly one pair series")
        returned_pair = pair_keys[0]
        rows = result[returned_pair]
        if not isinstance(rows, list):
            raise MarketDataProviderError("Kraken pair candles must be a JSON array")
        duration = INTERVAL_SECONDS[interval]
        records: list[MarketCandle] = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 8:
                raise MarketDataProviderError("Kraken candle row must contain 8 fields")
            try:
                source_start = int(row[0])
                open_price, high, low, close = (float(value) for value in row[1:5])
                volume = float(row[6])
                trade_count = int(row[7])
            except (TypeError, ValueError, OverflowError) as exc:
                raise MarketDataProviderError(f"Kraken candle contains an invalid field: {exc}") from exc
            records.append(MarketCandle(
                venue="kraken",
                instrument=returned_pair,
                base_currency="ETH",
                quote_currency="USD",
                interval=interval,
                source_bar_timestamp=source_start,
                event_time=source_start + duration,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                trade_count=trade_count,
                ingested_at=retrieved_at,
                is_final=(source_start + duration) <= as_of,
                source_payload_sha256=payload_sha256(row),
            ))
        records.sort(key=lambda candle: candle.source_bar_timestamp)
        if len({c.event_time for c in records}) != len(records):
            raise MarketDataProviderError("Kraken returned duplicate candle boundaries")
        return CandleFetchBatch(
            provider="kraken",
            requested_instrument=pair.upper(),
            returned_instrument=returned_pair,
            base_currency="ETH",
            quote_currency="USD",
            interval=interval,
            retrieved_at=retrieved_at,
            requested_start=None,
            requested_end=None,
            endpoint=self.ENDPOINT,
            source_documentation=self.DOCUMENTATION,
            raw_response_sha256=payload_sha256(payload),
            candles=tuple(records),
        )
