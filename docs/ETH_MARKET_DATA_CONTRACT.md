# ETH/USD Spot Market Data Contract

## Purpose and scope

This contract defines the first canonical spot OHLCV ingestion path for the ETH
volatility research system. It covers Coinbase Exchange ETH-USD and Kraken Spot
ETH/USD observations. It does not combine spot venues into a synthetic price
series and does not treat ETH-USDT as interchangeable with ETH-USD.

Provider documentation is authoritative for provider-specific response
semantics. Adapter behavior is versioned in source code and deterministic
fixtures; undocumented semantics remain unknown rather than inferred.

## Canonical observation identity

A candle is identified by:

- Venue/provider.
- Exact provider instrument or product identifier.
- Base currency.
- Quote currency.
- Interval.
- Canonical event_time, defined as the exclusive bar-end boundary.

The source bar timestamp is preserved in source_bar_timestamp and is always
interpreted as BAR_START for the two adapters in this increment.

For an interval of duration D seconds:

    event_time = source_bar_timestamp + D

This follows the repository temporal normalization convention. UTC epoch
alignment is required. The original source timestamp is never overwritten.

## Candle fields

Each normalized JSONL row contains:

| Field | Meaning |
|---|---|
| venue | Exact venue adapter identity |
| instrument | Coinbase product ID or Kraken returned pair key |
| base_currency | ETH |
| quote_currency | USD; USDT remains a separate quote |
| interval | Fixed sampling interval |
| source_bar_timestamp | Provider-supplied bar-start epoch seconds |
| event_time | Exclusive bar-end epoch seconds |
| source_timestamp_semantics | BAR_START |
| canonical_timestamp_semantics | BAR_END_EXCLUSIVE |
| open/high/low/close | Provider OHLC values |
| volume | Base-asset volume, or null when not supplied |
| trade_count | Provider count where supplied, otherwise null |
| ingested_at | UTC retrieval time in epoch seconds |
| is_final | True only when event_time is not later than the as-of time |
| source_payload_sha256 | Deterministic hash of the provider row |

Unknown volume is represented as null, never zero. Raw source values are retained
as supplied; normalization does not scale prices, volumes or quote currencies.

## Provider behavior and limitations

### Coinbase Exchange

Endpoint:

    GET https://api.exchange.coinbase.com/products/{product_id}/candles

Supported intervals implemented:

- 1m, 5m, 15m, 1h, 6h, 1d.

The adapter requests bounded UTC windows, splitting them into pages of at most
299 intervals so the request stays inside the documented 300-candle maximum.
It accepts the documented OHLC schema with optional volume; when volume is
absent, the normalized volume is null. Returned candles are filtered to the
requested half-open start/end range and sorted by source start time.

Coinbase documents that historical candle data may be incomplete and that no
data is published for intervals where there are no ticks. It also advises
against frequent polling of historical rates; realtime information should use
trade/book endpoints with the WebSocket feed. The current adapter is a
historical ingestion tool, not a realtime polling loop.

Official documentation:
https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles

### Kraken Spot

Endpoint:

    GET https://api.kraken.com/0/public/OHLC

Intervals implemented:

- 1m, 5m, 15m, 1h, 4h, 1d.

Kraken documents a maximum of 720 recent entries and states that the final
array entry is the current, not-yet-committed timeframe. The adapter preserves
that row and marks it is_final=false until its exclusive bar-end boundary is
at or before the as-of time. It does not claim arbitrary historical pagination
beyond the endpoint's recent-window limit. Kraken volume and trade count are
retained separately.

Official documentation:
https://docs.kraken.com/api-reference/market-data/get-ohlc-data

## Gaps, finality and rollups

- Missing intervals are reported as unobserved gaps. No candle is forward-filled,
  interpolated or fabricated.
- A gap on Coinbase may represent an interval without trades; the adapter does
  not infer why it is absent.
- Incomplete candles are retained with is_final=false, but are excluded from
  contiguous return construction and model training.
- Return construction refuses to bridge any missing interval.
- Four-hour rollups may be formed from complete, contiguous, UTC-aligned 1h bars
  from one venue/instrument only. Incomplete buckets are omitted, not padded.
- Provider-specific data is never averaged or concatenated across venues.

## Cross-venue comparison

Coinbase and Kraken observations may be compared only at common finalized
event_time boundaries with the same base currency, quote currency and interval.
The comparison reports signed close-price differences in basis points, mean,
median and maximum absolute difference. It does not average the two closes or
declare a reference venue without a separate governance decision.

## Running the ingestion tool

Coinbase historical range:

    python scripts/fetch_market_candles.py \
      --provider coinbase --instrument ETH-USD --interval 1h \
      --start 2026-10-01T00:00:00Z --end 2026-11-01T00:00:00Z

Kraken recent window:

    python scripts/fetch_market_candles.py \
      --provider kraken --instrument ETHUSD --interval 1h

The tool writes two local artifacts outside the source tree by default:

- JSONL normalized observations.
- JSON manifest with endpoint, provider, instrument, interval, retrieval time,
  source documentation URL, aggregate response hash, per-row source hashes,
  finality counts and gap audit.

Live market artifacts are not automatically committed. Preserve them as
versioned research artifacts only after the source, retrieval, interval,
row-count and hashes have been reviewed.

## Acceptance and promotion gates

| Gate | Requirement | Current status |
|---|---|---|
| D01 | Canonical temporal and OHLCV schema | Implemented; deterministic tests authored |
| D02 | Coinbase adapter and pagination | Implemented; live endpoint run pending |
| D03 | Kraken adapter and current-candle finality | Implemented; live endpoint run pending |
| D04 | Fixture tests for malformed rows, gaps, duplicates and rollups | Authored; execution pending |
| D05 | Actual independent venue comparison | Pending live retrieval |
| D06 | Persist audited 1h/4h/1d ETH/USD dataset | Pending |
| D07 | Rolling out-of-sample model evaluation | Pending |
| D08 | Forecast API, scheduler and dashboard | Pending |

No live ETH forecast is promoted from this adapter increment. A source adapter
being implemented does not mean its endpoint was reached successfully or that
its observations passed a live audit. Those are separate evidence states.
