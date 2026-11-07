# ETH/USD Volatility Research

## Purpose

This work extends the existing Python quantitative engine with a reproducible
conditional-volatility research layer. It is initially scoped to ETH/USD, but
the model functions are asset-agnostic.

The module estimates conditional variance and converts a one-period forecast
into a probability-based price interval. It does not forecast directional
returns, recommend trades, or grant execution permission.

## Implemented in the first research increment

Entry point: \`src/models/volatility_forecasting.py\`

| Method | Current implementation | Role |
|---|---|---|
| GARCH(1,1) | Gaussian or standardized Student-t innovations | Baseline volatility clustering |
| EGARCH(1,1) | Gaussian or standardized Student-t innovations | Asymmetric response to shocks |
| GJR-GARCH(1,1) | Gaussian or standardized Student-t innovations | Threshold/downside variance response |
| EWMA | Configurable decay, default 0.94 | Responsive benchmark |
| HAR-RV | Daily, weekly and monthly log-RV regressors | Multi-horizon realized-volatility benchmark |
| Realized variance | Sum of squared intraday log returns | Ex-post variation proxy |
| Bipower variation | Intraday bipower estimator | Continuous-path variation proxy |
| Jump variation | Non-negative RV minus BPV proxy | Jump contribution diagnostic |

The GARCH-family estimators use constrained maximum likelihood via SciPy.
The Student-t distribution is standardized to unit variance; its degrees of
freedom must therefore exceed two. Returns are accepted as decimal log returns
and internally scaled to percentage points for estimation.

The price interval function maps the next-period variance estimate to a
conditional return distribution. The resulting price interval is a model-based
quantile range, not a support/resistance estimate and not a guaranteed bound.

## Minimal usage

\`\`\`python
from src.models.volatility_forecasting import (
    fit_volatility_model,
    forecast_price_interval,
)

# Completed, UTC-aligned daily log returns, expressed as decimals.
fit = fit_volatility_model(
    daily_log_returns,
    model="GARCH",
    distribution="student_t",
)

forecast = forecast_price_interval(
    fit,
    spot_price=2300.0,
    confidence=0.95,
    periods_per_year=365,
)

print(forecast.conditional_volatility)
print(forecast.annualized_volatility)
print(forecast.lower_price, forecast.median_price, forecast.upper_price)
\`\`\`

For other sampling frequencies, use one fixed-frequency return series and
annualize consistently:

- Daily: 365 periods/year.
- Four-hour: 365 × 6 periods/year.
- Hourly: 365 × 24 periods/year.

Do not mix bars from different exchanges, quote currencies, time zones or
candle boundaries in a single return series.

## Interpretation and limitations

1. GARCH estimates conditional variance, not the sign of the next return.
2. A Student-t interval is heavier-tailed than a Gaussian interval when its
   degrees of freedom are finite. Estimated degrees of freedom are
   sample- and window-dependent.
3. GJR-GARCH and EGARCH test different asymmetry mechanisms; asymmetry should
   be assessed empirically rather than assumed.
4. HAR-RV requires positive daily realized-variance observations and enough
   history to form daily, weekly and monthly regressors.
5. Daily squared returns are a noisy realized-variance proxy. Intraday realized
   variance is preferred for out-of-sample variance scoring.
6. Variance combination requires explicit non-negative weights that sum to
   one. It intentionally does not learn weights from in-sample fit.
7. The first increment does not yet implement APARCH, FIGARCH, regime-switching
   GARCH, rolling-origin backtests, or calibrated interval coverage. Exchange
   ingestion, snapshot export and the dashboard are now implemented separately.

## Required data contract

Each observation must identify:

- \`venue\`: exchange/provider identifier.
- \`instrument\`: exact symbol (e.g. \`ETH-USD\` or \`ETH-USDT\`).
- \`quote_currency\`: USD and USDT must remain distinct.
- \`interval\`: fixed bar interval (\`1h\`, \`4h\`, \`1d\`, etc.).
- \`event_time\`: canonical UTC bar-end timestamp.
- \`open\`, \`high\`, \`low\`, \`close\`, \`volume\`: source values.
- \`source_timestamp\`: original exchange timestamp and documented meaning.
- \`ingested_at\`: UTC ingestion timestamp.
- \`is_final\`: whether the candle is complete.
- \`source_payload_hash\`: provenance/integrity reference for raw observation.

Validation rejects duplicate event keys, non-monotonic timestamps, invalid
OHLC relationships, negative or non-finite supplied volume, and non-finite
prices. Missing intervals are surfaced as explicit gaps and are never
forward-filled; contiguous-return construction and model fitting must refuse to
bridge them. Partial candles are excluded from completed-bar training sets.
Cross-source divergence is reported for common boundaries and is not silently
averaged away. Keep raw provider values and source hashes; store normalized
observations as derived artifacts with retrieval time and source provenance.

## Validation protocol before publishing a live forecast

1. Freeze a canonical data window and record its source, instrument, quote
   currency, interval, timezone, retrieval time and content hash.
2. Fit each candidate only on observations available at the forecast origin.
3. Use rolling or expanding time-ordered evaluation; never randomly shuffle
   time-series data.
4. Compare against EWMA and a simple historical-variance benchmark.
5. Score variance forecasts with QLIKE, RMSE and MAE. Prefer intraday RV as the
   realized-variance proxy where available.
6. Assess predictive intervals using empirical coverage, Kupiec unconditional
   coverage and Christoffersen independence tests.
7. Compare candidate models with paired out-of-sample loss differentials;
   account for overlapping multi-period targets.
8. Report forecast errors by market regime and time horizon, not just one
   aggregate metric.
9. Version model specification, estimation window, fitted parameters, forecast
   origin, input-data hash and scoring result.

A model is not promoted merely because its in-sample likelihood or AIC is best.
The live candidate must demonstrate stable out-of-sample calibration and
incremental value over the benchmark.

## Next gates

| Gate | Work | Exit evidence |
|---|---|---|
| V01 | Canonical ETH spot candles from two independent venues | Provenance, temporal and OHLC integrity reports |
| V02 | Rolling-origin baseline evaluation | GARCH / EGARCH / GJR / EWMA scorecard |
| V03 | Intraday realized-volatility pipeline | 1h, 4h and daily RV/BPV/jump measures |
| V04 | HAR-RV and distributional comparisons | OOS loss and interval-calibration report |
| V05 | APARCH, FIGARCH and regime-switching candidates | Incremental OOS value demonstrated |
| V06 | Forecast ensemble | Weights learned only from past OOS loss; stability report |
| V07 | Persistence, API and scheduling | Replayable forecast records and monitoring |
| V08 | Dashboard | UI reads persisted, versioned forecast artifacts |

## Market data ingestion

Coinbase Exchange ETH-USD and Kraken Spot ETH/USD public OHLCV adapters are
implemented in `src/data/market_data_providers.py`. Their canonical candle
contract, timestamp semantics, source limitations and invocation examples are
documented in `docs/ETH_MARKET_DATA_CONTRACT.md`. The adapters keep providers
and quote currencies separate, preserve unfinished bars with `is_final=false`,
record source hashes, expose missing intervals, and never fill gaps. A Coinbase
4h series can be derived only from complete contiguous UTC-aligned 1h bars.

Run the ingestion command from the repository root:

```bash
python scripts/fetch_market_candles.py --provider coinbase --instrument ETH-USD --interval 1h --start 2026-10-01T00:00:00Z --end 2026-11-01T00:00:00Z
python scripts/fetch_market_candles.py --provider kraken --instrument ETHUSD --interval 1h
```

The command writes local JSONL observations and a compact manifest under
`artifacts/market-data/`. It does not commit fetched data automatically.

## Current status

The rolling 24-hour backtest now uses a common realized-variance target and expected-variance forecasts. GARCH and GJR-GARCH use analytical conditional-variance recursions; Gaussian EGARCH uses a closed-form normal moment calculation; Student-t EGARCH is interval-only because its exponential recursion has no finite expected variance. HAR and downside-HAR convert log-volatility forecasts back to expected variance with a Duan smearing factor estimated on training residuals only.

The model module, public provider adapters, local ingestion CLI, canonical data
contract and deterministic fixture tests are implemented in the repository.
The provider endpoints have not yet been executed from this environment. This
does not establish live-market connectivity, independently validated ETH/USD
forecast accuracy, scheduled execution, forecast API or deployment. Those
remain separate acceptance gates.


## 24-hour ETH/USD scenario update

The hourly forecasting pipeline now adds a 24-step conditional path simulation for each fitted GARCH, EGARCH and GJR-GARCH model. Each scenario uses 20,000 simulated paths and a fixed seed for reproducibility. The original one-hour conditional variance and interval are retained for continuity.

The published snapshot includes a separate `price_quantiles_24h` object per venue and model. The dashboard currently uses Coinbase as the primary display feed when available and shows the outer envelope across its three model ranges. It does not average exchange prices.

### Validation status

- The source snapshot generated on 2026-11-02 at 17:19:23 UTC contains six 24-hour model scenarios from Coinbase and Kraken.
- All six lower / median / upper triples are finite and strictly ordered.
- The source snapshot status is `succeeded` with no recorded source or model errors.
- Quant research workflow run 41489973252 passed the unit and import checks after the horizon implementation.
- Snapshot workflow run 41489973161 completed successfully and published the six scenarios.

### Interpretation boundary

Each model's 95% interval is conditional on the fitted model and sampled historical data. The dashboard's combined low-to-high envelope takes the lowest lower bound and highest upper bound across models. That combined envelope is not itself a calibrated 95% probability interval. Rolling out-of-sample coverage, interval calibration, and comparative model selection remain open research gates. The scenarios are not directional trading signals.


### Typical-movement summary

The 24-hour scenario output also reports a typical path-volatility summary. It
uses the median of the simulated cumulative path standard deviations rather
than the arithmetic mean. Under heavy-tailed innovations, a small number of
extreme paths can dominate the mean and make a "typical move" figure unusable.
The model's lower and upper price quantiles still retain tail outcomes; this
median-path summary is only a descriptive volatility metric, not a replacement
for the scenario interval.

### Expected 24-hour variance and finite-moment handling

The horizon simulator now distinguishes path simulation from expected-variance estimation:

- GARCH and GJR-GARCH use their exact conditional expected-variance recursions. This avoids relying on a noisy Monte Carlo sample mean when Student-t shocks have finite variance but may have no finite fourth moment.
- EGARCH with Gaussian innovations uses a closed-form normal moment-generating calculation for expected cumulative variance.
- EGARCH with Student-t innovations and non-zero exponential asymmetry terms has no finite expected variance because the Student-t distribution has no moment-generating function. Its expected-variance field is explicitly null. Its simulated price quantiles and median-path movement summary remain available and are evaluated separately.
- The dashboard price quantiles are still generated from the path simulations; these expected-variance calculations do not change the range display.

This distinction prevents a finite Monte Carlo sample average from being presented as a mathematical expectation when the expectation does not exist.

### Updated operational status

The feature-branch GitHub workflows have successfully fetched public Coinbase
and Kraken data, fit six model/venue combinations, and validated six ordered
24-hour scenario ranges. A later median-path-volatility refinement has passed
the focused model and import CI checks. The source workflow publisher is being
hardened against concurrent branch updates. The workflow schedule is not active
on the default branch until the pull request is merged. Out-of-sample interval
calibration and forecast-accuracy validation remain open.
