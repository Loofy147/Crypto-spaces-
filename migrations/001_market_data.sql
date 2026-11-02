CREATE TABLE IF NOT EXISTS market_candles (
    venue TEXT NOT NULL,
    instrument TEXT NOT NULL,
    base_currency TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    interval TEXT NOT NULL,
    source_bar_timestamp BIGINT NOT NULL,
    event_time BIGINT NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NULL,
    trade_count BIGINT NULL,
    ingested_at BIGINT NOT NULL,
    is_final BOOLEAN NOT NULL,
    source_payload_sha256 CHAR(64) NOT NULL,
    source_timestamp_semantics TEXT NOT NULL DEFAULT 'BAR_START',
    canonical_timestamp_semantics TEXT NOT NULL DEFAULT 'BAR_END_EXCLUSIVE',
    PRIMARY KEY (venue, instrument, interval, event_time),
    CHECK (event_time > source_bar_timestamp),
    CHECK (open > 0 AND high > 0 AND low > 0 AND close > 0),
    CHECK (volume IS NULL OR volume >= 0),
    CHECK (trade_count IS NULL OR trade_count >= 0),
    CHECK (low <= LEAST(open, close) AND high >= GREATEST(open, close))
);

CREATE TABLE IF NOT EXISTS market_candle_observations (
    venue TEXT NOT NULL,
    instrument TEXT NOT NULL,
    base_currency TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    interval TEXT NOT NULL,
    source_bar_timestamp BIGINT NOT NULL,
    event_time BIGINT NOT NULL,
    observed_at BIGINT NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NULL,
    trade_count BIGINT NULL,
    is_final BOOLEAN NOT NULL,
    source_payload_sha256 CHAR(64) NOT NULL,
    PRIMARY KEY (venue, instrument, interval, event_time, source_payload_sha256)
);

CREATE INDEX IF NOT EXISTS idx_candle_observations_history
    ON market_candle_observations (venue, instrument, interval, event_time DESC, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_market_candles_series_time
    ON market_candles (venue, instrument, interval, event_time DESC);

CREATE TABLE IF NOT EXISTS market_data_runs (
    id BIGSERIAL PRIMARY KEY,
    provider TEXT NOT NULL,
    requested_instrument TEXT NOT NULL,
    returned_instrument TEXT NOT NULL,
    base_currency TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    interval TEXT NOT NULL,
    requested_start BIGINT NULL,
    requested_end BIGINT NULL,
    retrieved_at BIGINT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    row_count INTEGER NOT NULL DEFAULT 0,
    finalized_count INTEGER NOT NULL DEFAULT 0,
    raw_response_sha256 CHAR(64) NULL,
    error_message TEXT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_market_data_runs_latest
    ON market_data_runs (provider, interval, started_at DESC);

CREATE TABLE IF NOT EXISTS forecast_cycles (
    id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'partial', 'failed')),
    forecasts_created INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    errors JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS volatility_forecasts (
    id BIGSERIAL PRIMARY KEY,
    venue TEXT NOT NULL,
    instrument TEXT NOT NULL,
    base_currency TEXT NOT NULL,
    quote_currency TEXT NOT NULL,
    interval TEXT NOT NULL,
    forecast_horizon_steps INTEGER NOT NULL DEFAULT 1,
    forecast_origin BIGINT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    model TEXT NOT NULL,
    innovation_distribution TEXT NOT NULL,
    observations INTEGER NOT NULL,
    input_sha256 CHAR(64) NOT NULL,
    model_parameters JSONB NOT NULL,
    log_likelihood DOUBLE PRECISION NULL,
    aic DOUBLE PRECISION NULL,
    bic DOUBLE PRECISION NULL,
    conditional_variance DOUBLE PRECISION NOT NULL,
    conditional_volatility DOUBLE PRECISION NOT NULL,
    annualized_volatility DOUBLE PRECISION NOT NULL,
    reference_price DOUBLE PRECISION NOT NULL,
    mean_log_return DOUBLE PRECISION NOT NULL DEFAULT 0,
    degrees_of_freedom DOUBLE PRECISION NULL,
    price_quantiles JSONB NOT NULL,
    UNIQUE (venue, instrument, interval, forecast_horizon_steps,
            forecast_origin, model, innovation_distribution)
);

CREATE INDEX IF NOT EXISTS idx_volatility_forecasts_latest
    ON volatility_forecasts (instrument, interval, forecast_origin DESC, generated_at DESC);
