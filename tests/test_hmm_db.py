"""Tests for HMM database tables and query methods."""

import tempfile
from src.db import TradingDB


def _make_db():
    return TradingDB(db_path=tempfile.mktemp(suffix=".db"))


def test_record_hmm_observation():
    db = _make_db()
    row_id = db.record_hmm_observation(
        asset="btc",
        log_return_1m=0.001,
        log_return_5m=0.005,
        realized_vol_15m=0.02,
        bid_ask_spread=1.5,
        has_active_contract=1,
        timestamp="2026-04-03T12:00:00Z",
    )
    assert row_id == 1
    rows = db.get_hmm_observations(asset="BTC")
    assert len(rows) == 1
    r = rows[0]
    assert r["asset"] == "BTC"
    assert r["log_return_1m"] == 0.001
    assert r["log_return_5m"] == 0.005
    assert r["realized_vol_15m"] == 0.02
    assert r["bid_ask_spread"] == 1.5
    assert r["has_active_contract"] == 1
    assert r["timestamp"] == "2026-04-03T12:00:00Z"


def test_get_hmm_observations_since():
    db = _make_db()
    db.record_hmm_observation(asset="eth", log_return_1m=0.01, timestamp="2026-04-01T00:00:00Z")
    db.record_hmm_observation(asset="eth", log_return_1m=0.02, timestamp="2026-04-03T00:00:00Z")
    rows = db.get_hmm_observations(asset="ETH", since="2026-04-02T00:00:00Z")
    assert len(rows) == 1
    assert rows[0]["log_return_1m"] == 0.02


def test_record_shadow_prediction():
    db = _make_db()
    row_id = db.record_shadow_prediction(
        asset="btc",
        ticker="KXBTC-26APR03-100000",
        regime_entropy=0.45,
        top_state=2,
        top_state_prob=0.78,
        fair_prob=0.55,
        market_price=0.50,
        edge_cents=5.0,
        ev_cents=3.2,
        confidence=0.8,
        recommendation="BUY_YES",
        position_size=10,
        timestamp="2026-04-03T12:00:00Z",
    )
    assert row_id >= 1
    rows = db.get_shadow_predictions(asset="BTC")
    assert len(rows) == 1
    assert rows[0]["ticker"] == "KXBTC-26APR03-100000"
    assert rows[0]["recommendation"] == "BUY_YES"
    assert rows[0]["outcome"] is None


def test_resolve_shadow_prediction():
    db = _make_db()
    pid = db.record_shadow_prediction(
        asset="sol",
        fair_prob=0.6,
        market_price=0.5,
        edge_cents=10.0,
        recommendation="BUY_YES",
        timestamp="2026-04-03T12:00:00Z",
    )
    db.resolve_shadow_prediction(pid, outcome="WIN", pnl_cents=8.5)

    # resolved_only should return this one
    rows = db.get_shadow_predictions(asset="SOL", resolved_only=True)
    assert len(rows) == 1
    assert rows[0]["outcome"] == "WIN"
    assert rows[0]["pnl_cents"] == 8.5
    assert rows[0]["resolved_at"] is not None

    # unresolved prediction should not appear in resolved_only
    db.record_shadow_prediction(
        asset="sol",
        fair_prob=0.4,
        market_price=0.5,
        recommendation="SKIP",
        timestamp="2026-04-03T13:00:00Z",
    )
    rows = db.get_shadow_predictions(asset="SOL", resolved_only=True)
    assert len(rows) == 1


def test_save_and_load_hmm_model_state():
    db = _make_db()
    row_id = db.save_hmm_model_state(
        asset="btc",
        version=1,
        n_states=3,
        bic=1234.5,
        log_likelihood=-567.8,
        stability_flags=0,
        state_means='[0.01, -0.005, 0.0]',
        transition_matrix='[[0.9,0.05,0.05],[0.1,0.8,0.1],[0.1,0.1,0.8]]',
        observation_count=5000,
        trained_at="2026-04-03T12:00:00Z",
    )
    assert row_id >= 1

    state = db.get_latest_hmm_model_state("BTC")
    assert state is not None
    assert state["asset"] == "BTC"
    assert state["version"] == 1
    assert state["n_states"] == 3
    assert state["bic"] == 1234.5
    assert state["log_likelihood"] == -567.8
    assert state["state_means"] == '[0.01, -0.005, 0.0]'
    assert state["observation_count"] == 5000

    # Save a newer version and verify latest returns it
    db.save_hmm_model_state(
        asset="btc",
        version=2,
        n_states=4,
        bic=1100.0,
        log_likelihood=-500.0,
        trained_at="2026-04-03T13:00:00Z",
    )
    state = db.get_latest_hmm_model_state("BTC")
    assert state["version"] == 2
    assert state["n_states"] == 4

    # Non-existent asset returns None
    assert db.get_latest_hmm_model_state("DOGE") is None
