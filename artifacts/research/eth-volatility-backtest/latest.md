# ETH/USD rolling 24-hour volatility backtest

**Generated:** 2026-11-07T01:04:40.661948Z
**Data:** coinbase ETH-USD 1h candles (3000 finalized candles)
**OOS origins:** 30 daily origins, 90-day rolling training window

## Common-origin scorecard

| Rank | Candidate | QLIKE ↓ | RMSE ↓ | MAE ↓ | Forecast / realized ratio |
|---:|---|---:|---:|---:|---:|
| 1 | Rolling-RV-22d | 0.481460 | 0.00064791 | 0.00044872 | 2.041 |
| 2 | HAR-RV | 0.500931 | 0.00064692 | 0.00041561 | 1.440 |
| 3 | HAR-RV+downside | 0.582922 | 0.00066514 | 0.00042223 | 1.209 |
| 4 | GARCH | 0.624798 | 0.00118590 | 0.00095063 | 4.720 |
| 5 | GJR-GARCH | 0.624799 | 0.00118590 | 0.00095063 | 4.720 |
| 6 | EWMA-hourly | 0.701552 | 0.00085897 | 0.00060191 | 2.906 |
| 7 | EGARCH | 9.228674 | 232086.50638612 | 95208.63587445 | 544817541.352 |

## Candidate comparisons against GARCH

| Candidate | Mean QLIKE difference vs GARCH | DM statistic | p-value | Holm adjusted p |
|---|---:|---:|---:|---:|
| EGARCH | 8.603876 | 7.215 | 0.0000 | 0.0000 |
| GJR-GARCH | 0.000000 | 0.907 | 0.3718 | 1.0000 |
| HAR-RV | -0.123867 | -1.442 | 0.1599 | 0.6395 |
| HAR-RV+downside | -0.041876 | -0.424 | 0.6750 | 1.0000 |
| EWMA-hourly | 0.076754 | 0.710 | 0.4836 | 1.0000 |
| Rolling-RV-22d | -0.143338 | -1.776 | 0.0863 | 0.4313 |

Negative mean QLIKE difference means the candidate had lower average loss than GARCH. The p-values are exploratory because the holdout is short and only one venue is tested.

## GARCH-family 95% interval coverage

| Model | Nominal | Empirical coverage | Average interval width (% of spot) |
|---|---:|---:|---:|
| GARCH | 95% | 90.0% | 10.68% |
| EGARCH | 95% | 90.0% | 11.14% |
| GJR-GARCH | 95% | 90.0% | 10.68% |

## Interpretation and limits

- This is a preliminary single-venue rolling out-of-sample screen, not a final model selection.
- QLIKE, RMSE and MAE score daily realized variance; lower is better.
- The downside-HAR challenger uses only prior daily realized variance and downside semivariance.
- GARCH-family intervals are model-conditional and their empirical coverage is not calibrated by this experiment.
- The daily origins have non-overlapping 24-hour targets; the DM tests still use a five-day Newey-West HAC estimator.
- A model should not be promoted from this short sample without a longer holdout, regime analysis and repeatable results.
- Skipped origins: 0.
- Source response SHA-256: 8140bcf6b6731051a88458d5c96a1d7c3306f59e27e8a6744c3528a7b2df6d39.

The comparison does not establish a winning model. Any promotion requires a longer holdout, more than one venue, regime-specific analysis and interval calibration.
