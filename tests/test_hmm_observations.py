"""Tests for HMM observation pipeline (Task 2)."""

import tempfile
from datetime import datetime, timedelta, timezone

from src.db import TradingDB


def _make_db_with_prices(asset: str, prices: list[float]) -> TradingDB:
    """Create a temp DB and insert prices 1 minute apart."""
    db = TradingDB(db_path=tempfile.mktemp(suffix=".db"))
    base = datetime(2026, 4, 3, 12, 0, 0, tzinfo=timezone.utc)
    for i, p in enumerate(prices):
        ts = (base + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        db.record_crypto_price(asset, p, change_24h_pct=0.0, timestamp=ts)
    return db


def test_compute_features_basic():
    """20 rising prices -> log_return_1m > 0 and realized_vol_15m exists."""
    from src.hmm_observations import compute_observation_features

    prices = [100.0 + i * 0.5 for i in range(20)]
    db = _make_db_with_prices("BTC", prices)
    features = compute_observation_features(db, "BTC")
    assert features is not None
    assert features["log_return_1m"] > 0
    assert "realized_vol_15m" in features
    assert features["realized_vol_15m"] is not None


def test_compute_features_insufficient_data():
    """2 prices -> returns None."""
    from src.hmm_observations import compute_observation_features

    db = _make_db_with_prices("BTC", [100.0, 101.0])
    result = compute_observation_features(db, "BTC")
    assert result is None


def test_compute_features_with_market_data():
    """With a market snapshot, has_active_contract=1 and bid_ask_spread > 0."""
    from src.hmm_observations import compute_observation_features

    prices = [100.0 + i * 0.5 for i in range(20)]
    db = _make_db_with_prices("BTC", prices)
    # Insert a 15M contract snapshot with a recent timestamp
    base = datetime(2026, 4, 3, 12, 0, 0, tzinfo=timezone.utc)
    snap_ts = (base + timedelta(minutes=19)).strftime("%Y-%m-%dT%H:%M:%SZ")
    exp_ts = (base + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    db.record_market_snapshot(
        ticker="KXBTC15M-26APR03-100500",
        yes_bid=0.40,
        yes_ask=0.60,
        no_bid=0.35,
        no_ask=0.55,
        volume=150,
        strike_price=100500,
        spot_price=100000,
        expiration_time=exp_ts,
        asset="BTC",
        timestamp=snap_ts,
    )
    features = compute_observation_features(db, "BTC")
    assert features is not None
    assert features["has_active_contract"] == 1
    assert features["bid_ask_spread"] > 0


def test_observation_pipeline_records_to_db():
    """Pipeline.record_observation stores to DB."""
    from src.hmm_observations import ObservationPipeline

    prices = [100.0 + i * 0.5 for i in range(20)]
    db = _make_db_with_prices("BTC", prices)
    pipeline = ObservationPipeline(db)
    pipeline.record_observation("BTC")
    rows = db.get_hmm_observations(asset="BTC")
    assert len(rows) == 1
    assert rows[0]["log_return_1m"] is not None


def test_log_return_values():
    """Verify log returns are float values."""
    from src.hmm_observations import compute_observation_features

    prices = [100.0 + i * 0.5 for i in range(20)]
    db = _make_db_with_prices("BTC", prices)
    features = compute_observation_features(db, "BTC")
    assert features is not None
    assert isinstance(features["log_return_1m"], float)
    assert isinstance(features["log_return_5m"], float)
    assert isinstance(features["log_return_15m"], float)
