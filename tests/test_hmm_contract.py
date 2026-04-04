"""Tests for HMM contract opportunity model."""

import numpy as np
import pytest

from src.hmm_contract import (
    FEES_CENTS,
    ContractEvaluation,
    compute_regime_ev,
    position_size,
    evaluate_contract_with_regime,
)


def _make_state_profiles(win_rate, avg_win, avg_loss, count):
    """Helper: single-state profile list."""
    return [{"win_rate": win_rate, "avg_win_cents": avg_win, "avg_loss_cents": avg_loss, "count": count}]


# ------------------------------------------------------------------ #
# Test 1: positive EV from profitable state
# ------------------------------------------------------------------ #
def test_regime_ev_positive():
    posterior = np.array([1.0])
    profiles = _make_state_profiles(0.65, 50, 30, 100)
    ev, confidence = compute_regime_ev(posterior, profiles, edge_cents=10.0)
    assert ev > 0, f"Expected positive EV, got {ev}"
    assert 0.0 <= confidence <= 1.0


# ------------------------------------------------------------------ #
# Test 2: negative EV from losing state
# ------------------------------------------------------------------ #
def test_regime_ev_negative():
    posterior = np.array([1.0])
    profiles = _make_state_profiles(0.3, 50, 30, 100)
    ev, confidence = compute_regime_ev(posterior, profiles, edge_cents=10.0)
    assert ev < 0, f"Expected negative EV, got {ev}"


# ------------------------------------------------------------------ #
# Test 3: insufficient data → break-even prior (0 EV contribution)
# ------------------------------------------------------------------ #
def test_regime_ev_insufficient_data():
    posterior = np.array([1.0])
    profiles = _make_state_profiles(0.65, 50, 30, 5)  # <10 trades
    ev, confidence = compute_regime_ev(posterior, profiles, edge_cents=10.0)
    # With <10 trades, should use break-even prior → 0 EV contribution
    assert ev == 0.0 or abs(ev) < 1e-6
    assert confidence < 0.3  # Low confidence due to insufficient data


# ------------------------------------------------------------------ #
# Test 4: positive EV + confidence → size >= 1
# ------------------------------------------------------------------ #
def test_position_size_positive_ev():
    size = position_size(ev_cents=5.0, confidence=0.8, bankroll=1000)
    assert size >= 1
    assert size <= 3  # max_contracts default


# ------------------------------------------------------------------ #
# Test 5: negative EV → 0
# ------------------------------------------------------------------ #
def test_position_size_negative_ev():
    size = position_size(ev_cents=-2.0, confidence=0.8, bankroll=1000)
    assert size == 0


# ------------------------------------------------------------------ #
# Test 6: low confidence → 0
# ------------------------------------------------------------------ #
def test_position_size_low_confidence():
    size = position_size(ev_cents=5.0, confidence=0.1, bankroll=1000)
    assert size == 0


# ------------------------------------------------------------------ #
# Test 7: full pipeline returns valid ContractEvaluation
# ------------------------------------------------------------------ #
def test_evaluate_contract_with_regime():
    posterior = np.array([0.6, 0.3, 0.1])
    profiles = [
        {"win_rate": 0.7, "avg_win_cents": 40, "avg_loss_cents": 20, "count": 80},
        {"win_rate": 0.5, "avg_win_cents": 30, "avg_loss_cents": 30, "count": 60},
        {"win_rate": 0.3, "avg_win_cents": 20, "avg_loss_cents": 40, "count": 40},
    ]
    result = evaluate_contract_with_regime(
        regime_posterior=posterior,
        state_profiles=profiles,
        spot_price=65000,
        strike_price=65500,
        yes_price_cents=40,
        no_price_cents=60,
        time_to_expiry_secs=3600,
        contract_volume=500,
        bid_ask_spread_cents=3,
        log_normal_prob=0.55,
        bankroll=1000,
    )
    assert isinstance(result, ContractEvaluation)
    assert result.recommendation in ("buy_yes", "buy_no", "skip")
    assert 0.0 <= result.fair_prob <= 1.0
    assert 0.0 <= result.confidence <= 1.0
    assert result.position_size >= 0
    assert isinstance(result.reasons, list)
