"""Tests for HMM shadow tracker."""

import tempfile
import numpy as np
import pytest

from src.db import TradingDB
from src.hmm_shadow import ShadowTracker


def _make_db():
    return TradingDB(db_path=tempfile.mktemp(suffix=".db"))


def _make_tracker():
    db = _make_db()
    return ShadowTracker(db), db


# ------------------------------------------------------------------ #
# Test 1: record and resolve a prediction, verify metrics
# ------------------------------------------------------------------ #
def test_record_and_resolve():
    tracker, db = _make_tracker()
    pred_id = tracker.record_prediction(
        asset="BTC",
        ticker="KXBTC-65000-UP",
        regime_posterior=[0.7, 0.2, 0.1],
        regime_entropy=0.8,
        fair_prob=0.6,
        market_price=45,
        edge_cents=9.0,
        ev_cents=5.0,
        confidence=0.75,
        recommendation="buy_yes",
        position_size=2,
    )
    assert isinstance(pred_id, int)
    assert pred_id > 0

    tracker.resolve_prediction(pred_id, outcome="win", pnl_cents=15.0)

    metrics = tracker.get_rolling_metrics(asset="BTC", days=7)
    assert metrics["trade_count"] == 1
    assert metrics["win_rate"] == 1.0
    assert metrics["total_pnl_cents"] == 15.0
    assert metrics["wins"] == 1
    assert metrics["losses"] == 0


# ------------------------------------------------------------------ #
# Test 2: empty metrics when no data
# ------------------------------------------------------------------ #
def test_rolling_metrics_empty():
    tracker, db = _make_tracker()
    metrics = tracker.get_rolling_metrics(days=7)
    assert metrics["trade_count"] == 0
    assert metrics["win_rate"] == 0.0
    assert metrics["total_pnl_cents"] == 0.0


# ------------------------------------------------------------------ #
# Test 3: comparison metrics — 5 predictions (3 wins, 2 losses)
# ------------------------------------------------------------------ #
def test_comparison_metrics():
    tracker, db = _make_tracker()
    results = [
        ("win", 10.0),
        ("win", 8.0),
        ("loss", -12.0),
        ("win", 15.0),
        ("loss", -7.0),
    ]
    for outcome, pnl in results:
        pid = tracker.record_prediction(
            asset="ETH",
            ticker="KXETH-4000-UP",
            regime_posterior=[0.5, 0.5],
            regime_entropy=0.69,
            fair_prob=0.55,
            market_price=48,
            edge_cents=5.0,
            ev_cents=3.0,
            confidence=0.6,
            recommendation="buy_yes",
            position_size=1,
        )
        tracker.resolve_prediction(pid, outcome=outcome, pnl_cents=pnl)

    metrics = tracker.get_rolling_metrics(asset="ETH", days=7)
    assert metrics["trade_count"] == 5
    assert metrics["wins"] == 3
    assert metrics["losses"] == 2
    assert metrics["win_rate"] == pytest.approx(0.6)
    assert metrics["total_pnl_cents"] == pytest.approx(14.0)  # 10+8-12+15-7
    assert metrics["avg_pnl_per_trade"] == pytest.approx(14.0 / 5)


# ------------------------------------------------------------------ #
# Test 4: skip recommendations don't count in metrics
# ------------------------------------------------------------------ #
def test_skip_predictions_not_counted():
    tracker, db = _make_tracker()

    # Record a skip prediction
    pid_skip = tracker.record_prediction(
        asset="SOL",
        ticker="KXSOL-200-UP",
        regime_posterior=[0.3, 0.7],
        regime_entropy=0.6,
        fair_prob=0.50,
        market_price=50,
        edge_cents=0.0,
        ev_cents=0.0,
        confidence=0.2,
        recommendation="skip",
        position_size=0,
    )
    tracker.resolve_prediction(pid_skip, outcome="win", pnl_cents=0.0)

    # Record a real trade
    pid_real = tracker.record_prediction(
        asset="SOL",
        ticker="KXSOL-200-UP",
        regime_posterior=[0.8, 0.2],
        regime_entropy=0.5,
        fair_prob=0.60,
        market_price=42,
        edge_cents=12.0,
        ev_cents=8.0,
        confidence=0.8,
        recommendation="buy_yes",
        position_size=2,
    )
    tracker.resolve_prediction(pid_real, outcome="win", pnl_cents=20.0)

    metrics = tracker.get_rolling_metrics(asset="SOL", days=7)
    assert metrics["trade_count"] == 1  # skip not counted
    assert metrics["wins"] == 1
    assert metrics["total_pnl_cents"] == 20.0
