"""Tests for HMM regime detection engine."""

import tempfile
import numpy as np
import pytest

from src.hmm_regime import (
    FEATURE_COLS,
    fit_hmm_select_k,
    decode_posterior,
    RegimeEngine,
    _observations_to_matrix,
    _normalize,
)
from src.db import TradingDB


def _make_synthetic_observations(n=500, n_features=12):
    np.random.seed(42)
    r1 = np.random.randn(n // 2, n_features) * 0.01 + 0.001  # Regime 1
    r2 = np.random.randn(n // 2, n_features) * 0.05 - 0.003  # Regime 2
    return np.vstack([r1, r2])


def _make_db():
    return TradingDB(db_path=tempfile.mktemp(suffix=".db"))


# ------------------------------------------------------------------ #
# Test 1: fit_hmm_select_k returns valid model
# ------------------------------------------------------------------ #
def test_fit_hmm_select_k():
    data = _make_synthetic_observations(500, 12)
    normed, _, _ = _normalize(data)
    model, k, bic = fit_hmm_select_k(normed, k_range=range(3, 7), n_restarts=3)

    assert model is not None
    assert 3 <= k <= 6
    assert isinstance(bic, float)
    assert isinstance(bic, float) and np.isfinite(bic)


# ------------------------------------------------------------------ #
# Test 2: decode_posterior shape and row sums
# ------------------------------------------------------------------ #
def test_decode_posterior():
    data = _make_synthetic_observations(500, 12)
    normed, _, _ = _normalize(data)
    model, k, _ = fit_hmm_select_k(normed, k_range=range(3, 6), n_restarts=3)

    posteriors = decode_posterior(model, normed)
    assert posteriors.shape == (500, k)
    # Each row should sum to ~1
    row_sums = posteriors.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
    # All values between 0 and 1
    assert np.all(posteriors >= 0.0)
    assert np.all(posteriors <= 1.0)


# ------------------------------------------------------------------ #
# Test 3: last-step posterior length and sum
# ------------------------------------------------------------------ #
def test_decode_posterior_last_step():
    data = _make_synthetic_observations(500, 12)
    normed, _, _ = _normalize(data)
    model, k, _ = fit_hmm_select_k(normed, k_range=range(3, 6), n_restarts=3)

    posteriors = decode_posterior(model, normed)
    last = posteriors[-1]
    assert len(last) == k
    assert abs(sum(last) - 1.0) < 1e-6


# ------------------------------------------------------------------ #
# Test 4: RegimeEngine with DB round-trip
# ------------------------------------------------------------------ #
def test_regime_engine_with_db():
    db = _make_db()
    data = _make_synthetic_observations(300, 12)
    feature_names = [
        "log_return_1m", "log_return_5m", "log_return_15m",
        "realized_vol_15m", "realized_vol_1h", "vol_of_vol",
        "momentum_r_sq", "mean_reversion", "bid_ask_spread",
        "spread_vol", "volume_1m", "volume_accel",
    ]

    for i in range(300):
        kwargs = {name: float(data[i, j]) for j, name in enumerate(feature_names)}
        kwargs["asset"] = "btc"
        kwargs["has_active_contract"] = 1
        kwargs["timestamp"] = f"2026-04-03T{i // 60:02d}:{i % 60:02d}:00Z"
        db.record_hmm_observation(**kwargs)

    engine = RegimeEngine(db)
    summary = engine.fit_asset("BTC")
    assert summary is not None
    assert "n_states" in summary
    assert "bic" in summary

    posterior = engine.get_current_posterior("BTC")
    assert posterior is not None
    assert len(posterior) == summary["n_states"]
    assert abs(sum(posterior) - 1.0) < 1e-6


# ------------------------------------------------------------------ #
# Test 5: insufficient data returns None
# ------------------------------------------------------------------ #
def test_regime_engine_insufficient_data():
    db = _make_db()
    engine = RegimeEngine(db)
    result = engine.fit_asset("BTC")
    assert result is None
