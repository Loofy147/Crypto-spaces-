"""Unit tests for src/data module."""

import polars as pl
from src.data.derivatives_feed import DerivativesFeedCalculator, MicrostructureRegime
from src.data.market_feed import MarketFeedNormalizer
from src.data.onchain_indicators import ASIRegime, OnChainIndicatorsCalculator


def test_market_feed_normalize_ticks() -> None:
    ticks = [
        {"timestamp": 1.0, "symbol": "BTC", "price": 50000.0, "volume": 1.5},
        {"timestamp": 2.0, "symbol": "ETH", "price": 3000.0, "volume": 10.0},
    ]
    df = MarketFeedNormalizer.normalize_ticks_to_polars(ticks)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 2
    assert "price" in df.columns


def test_market_feed_depth_event_creation() -> None:
    bids = {50000.0: 1.0, 49500.0: 2.0, 48000.0: 5.0}  # Mid ~50000. 2% bid is >= 49000
    asks = {50100.0: 1.5, 50500.0: 2.5, 52000.0: 10.0}  # 2% ask is <= 51050

    depth_event = MarketFeedNormalizer.create_depth_event(
        event_id="d1", timestamp=10.0, symbol="BTC", bids=bids, asks=asks
    )

    assert depth_event.depth_2pct_bid == 3.0  # 1.0 + 2.0
    assert depth_event.depth_2pct_ask == 4.0  # 1.5 + 2.5


def test_crypto_margin_ratio_calculator() -> None:
    # Test alert flag > 50%
    status = DerivativesFeedCalculator.calculate_crypto_margin_ratio(
        crypto_margined_oi=600.0, cash_margined_oi=400.0
    )
    assert status.total_oi == 1000.0
    assert status.crypto_margin_ratio == 60.0
    assert status.leverage_alert_flag is True

    # Test safe ratio <= 50%
    status_safe = DerivativesFeedCalculator.calculate_crypto_margin_ratio(
        crypto_margined_oi=400.0, cash_margined_oi=600.0
    )
    assert status_safe.crypto_margin_ratio == 40.0
    assert status_safe.leverage_alert_flag is False


def test_cvd_divergence_regimes() -> None:
    res_bull = DerivativesFeedCalculator.detect_cvd_divergence(spot_cvd=100.0, perp_cvd=50.0)
    assert res_bull.regime == MicrostructureRegime.ORGANIC_BULLISH

    res_abs = DerivativesFeedCalculator.detect_cvd_divergence(spot_cvd=100.0, perp_cvd=-50.0)
    assert res_abs.regime == MicrostructureRegime.SPOT_ABSORPTION

    res_sqz = DerivativesFeedCalculator.detect_cvd_divergence(spot_cvd=-100.0, perp_cvd=50.0)
    assert res_sqz.regime == MicrostructureRegime.LONG_SQUEEZE_RISK

    res_liq = DerivativesFeedCalculator.detect_cvd_divergence(spot_cvd=-100.0, perp_cvd=-50.0)
    assert res_liq.regime == MicrostructureRegime.BROAD_LIQUIDATION


def test_altcoin_season_index() -> None:
    btc_ret = 0.20
    alt_rets = {
        "ETH": 0.30,  # > BTC
        "SOL": 0.50,  # > BTC
        "AVAX": 0.10,  # < BTC
        "LINK": 0.05,  # < BTC
    }

    asi = OnChainIndicatorsCalculator.calculate_altcoin_season_index(btc_ret, alt_rets)
    assert asi.outperforming_count == 2
    assert asi.total_universe_count == 4
    assert asi.asi_score == 50.0
    assert asi.regime == ASIRegime.TRANSITIONAL_ROTATION

    # Test Altseason > 75
    alt_rets_bull = {"A": 0.3, "B": 0.4, "C": 0.5, "D": 0.6}
    asi_bull = OnChainIndicatorsCalculator.calculate_altcoin_season_index(0.1, alt_rets_bull)
    assert asi_bull.asi_score == 100.0
    assert asi_bull.regime == ASIRegime.ALTSEASON

    # Test BTC Dominance < 25
    alt_rets_bear = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    asi_bear = OnChainIndicatorsCalculator.calculate_altcoin_season_index(0.1, alt_rets_bear)
    assert asi_bear.asi_score == 0.0
    assert asi_bear.regime == ASIRegime.BITCOIN_DOMINANCE


def test_onchain_metrics_edge_cases() -> None:
    # Zero market cap / zero realized cap
    m = OnChainIndicatorsCalculator.calculate_onchain_metrics(
        market_cap=0.0,
        realized_cap=100.0,
        prev_realized_cap_30d=0.0,
        hot_capital_val=10.0,
    )
    assert m.nupl == 0.0
    assert m.realized_cap_change_pct == 0.0
    assert m.hot_capital_share == 0.0

    # Normal positive inputs
    m_valid = OnChainIndicatorsCalculator.calculate_onchain_metrics(
        market_cap=1000.0,
        realized_cap=600.0,
        prev_realized_cap_30d=500.0,
        hot_capital_val=200.0,
    )
    assert m_valid.nupl == 0.40
    assert m_valid.realized_cap_change_pct == 20.0
    assert m_valid.hot_capital_share == 0.20

    # Empty altcoin dict for ASI
    empty_asi = OnChainIndicatorsCalculator.calculate_altcoin_season_index(0.1, {})
    assert empty_asi.total_universe_count == 0


def test_market_feed_empty_and_event_creators() -> None:
    # Empty ticks
    df_empty = MarketFeedNormalizer.normalize_ticks_to_polars([])
    assert df_empty.height == 0

    # Tick event creator
    tick_evt = MarketFeedNormalizer.create_tick_event("t_1", 100.0, "BTC", 50000.0, 1000.0)
    assert tick_evt.symbol == "BTC"
    assert tick_evt.price == 50000.0

    # Empty depth event creator
    depth_empty = MarketFeedNormalizer.create_depth_event("d_1", 100.0, "BTC", {}, {})
    assert depth_empty.depth_2pct_bid == 0.0
    assert depth_empty.depth_2pct_ask == 0.0


def test_derivatives_zero_oi() -> None:
    status_zero = DerivativesFeedCalculator.calculate_crypto_margin_ratio(0.0, 0.0)
    assert status_zero.crypto_margin_ratio == 0.0
    assert status_zero.leverage_alert_flag is False
