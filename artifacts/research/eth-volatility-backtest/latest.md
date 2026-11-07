# ETH/USD rolling 24-hour volatility backtest

**Generated:** 2026-11-07T01:26:34.168059Z
**Data:** coinbase ETH-USD 1h candles (3000 finalized candles)
**OOS origins:** 30 daily origins, 90-day rolling training window

## Common-origin scorecard

| Rank | Candidate | QLIKE ↓ | RMSE ↓ | MAE ↓ | Forecast / realized ratio |
|---:|---|---:|---:|---:|---:|
| 1 | HAR-RV+downside | 0.367117 | 0.00059082 | 0.00040619 | 1.834 |
| 2 | HAR-RV | 0.380140 | 0.00059849 | 0.00044754 | 2.260 |
| 3 | EGARCH-normal | 0.465613 | 0.00068903 | 0.00061052 | 3.143 |
| 4 | Rolling-RV-22d | 0.481460 | 0.00064791 | 0.00044872 | 2.041 |
| 5 | EWMA-hourly | 0.701552 | 0.00085897 | 0.00060191 | 2.906 |
| 6 | GARCH | 0.720856 | 0.00142398 | 0.00115079 | 5.613 |
| 7 | GJR-GARCH | 0.720856 | 0.00142398 | 0.00115079 | 5.613 |

## Candidate comparisons against GARCH

| Candidate | Mean QLIKE difference vs GARCH | DM statistic | p-value | Holm adjusted p |
|---|---:|---:|---:|---:|
| EGARCH-normal | -0.255243 | -4.550 | 0.0001 | 0.0004 |
| GJR-GARCH | 0.000000 | 1.073 | 0.2921 | 0.5842 |
| HAR-RV | -0.340716 | -5.274 | 0.0000 | 0.0001 |
| HAR-RV+downside | -0.353738 | -4.856 | 0.0000 | 0.0002 |
| EWMA-hourly | -0.019304 | -0.149 | 0.8827 | 0.8827 |
| Rolling-RV-22d | -0.239395 | -2.137 | 0.0412 | 0.1236 |

Negative mean QLIKE difference means the candidate had lower average loss than GARCH. The p-values are exploratory because the holdout is short and only one venue is tested.

## GARCH-family 95% interval coverage

| Model | Nominal | Empirical coverage | Average interval width (% of spot) |
|---|---:|---:|---:|
| GARCH | 95% | 90.0% | 10.68% |
| EGARCH-student_t | 95% | 90.0% | 11.14% |
| GJR-GARCH | 95% | 90.0% | 10.68% |

## Interpretation and limits

- This is a preliminary single-venue rolling out-of-sample screen, not a final model selection.
- QLIKE, RMSE and MAE score daily realized variance; lower is better.
- The downside-HAR challenger uses only prior daily realized variance and downside semivariance.
- EGARCH is scored with Gaussian innovations because expected variance is undefined under Student-t innovations in the exponential EGARCH recursion.
- Student-t EGARCH is retained as an interval-only candidate; its Monte Carlo sample mean is not treated as an expected variance.
- GARCH-family intervals are model-conditional and their empirical coverage is not calibrated by this experiment.
- The daily origins have non-overlapping 24-hour targets; the DM tests still use a five-day Newey-West HAC estimator.
- A model should not be promoted from this short sample without a longer holdout, regime analysis and repeatable results.
- Skipped origins: 0.
- Source response SHA-256: cba2a2a8936472b18b8172967ac4443ad77f75c2da0bd5856ea562fee329d355.

The comparison does not establish a winning model. Any promotion requires a longer holdout, more than one venue, regime-specific analysis and interval calibration.
