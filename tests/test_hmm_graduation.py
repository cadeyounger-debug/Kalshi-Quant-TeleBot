"""Tests for HMM graduation controller."""

import pytest

from src.hmm_graduation import (
    Stage,
    GraduationController,
    MIN_SHADOW_DAYS,
    MIN_TRADE_COUNT,
    MAX_DRAWDOWN_PCT,
    MIN_EV_ADVANTAGE_CENTS,
    KILL_DAILY_LOSS_PCT,
    KILL_CONSECUTIVE_LOSSES,
)


# ------------------------------------------------------------------ #
# Test 1: initial stage is COLLECTION
# ------------------------------------------------------------------ #
def test_initial_stage():
    gc = GraduationController()
    assert gc.stage == Stage.COLLECTION
    assert gc.days_in_stage == 0
    assert gc.has_enough_observations is False


# ------------------------------------------------------------------ #
# Test 2: COLLECTION → SHADOW after 3+ days with observations
# ------------------------------------------------------------------ #
def test_collection_to_shadow():
    gc = GraduationController()
    gc.days_in_stage = 3
    gc.has_enough_observations = True
    assert gc.can_advance() is True
    gc.advance()
    assert gc.stage == Stage.SHADOW


# ------------------------------------------------------------------ #
# Test 3: shadow promotion gate — all 7 criteria met
# ------------------------------------------------------------------ #
def test_shadow_promotion_gate_passes():
    gc = GraduationController()
    gc.stage = Stage.SHADOW
    gc.days_in_stage = 10

    shadow_metrics = {
        "trade_count": 50,
        "win_rate": 0.62,
        "avg_pnl_per_trade": 3.5,
        "max_drawdown_cents": 100,
        "total_pnl_cents": 175,
    }
    live_metrics = {
        "win_rate": 0.55,
        "avg_pnl_per_trade": 2.0,
    }
    result = gc.check_promotion_gate(
        shadow_metrics=shadow_metrics,
        live_metrics=live_metrics,
        bankroll=1000,
        model_stable=True,
    )
    assert result["passes"] is True
    assert result["criteria_met"] == 7
    assert result["criteria_total"] == 7
    assert len(result["failures"]) == 0


# ------------------------------------------------------------------ #
# Test 4: promotion gate fails — low trade count
# ------------------------------------------------------------------ #
def test_shadow_promotion_gate_fails_low_count():
    gc = GraduationController()
    gc.stage = Stage.SHADOW
    gc.days_in_stage = 10

    shadow_metrics = {
        "trade_count": 15,  # < 30
        "win_rate": 0.62,
        "avg_pnl_per_trade": 3.5,
        "max_drawdown_cents": 100,
        "total_pnl_cents": 52.5,
    }
    live_metrics = {
        "win_rate": 0.55,
        "avg_pnl_per_trade": 2.0,
    }
    result = gc.check_promotion_gate(
        shadow_metrics=shadow_metrics,
        live_metrics=live_metrics,
        bankroll=1000,
        model_stable=True,
    )
    assert result["passes"] is False
    assert result["criteria_met"] < 7
    assert any("trade" in f.lower() for f in result["failures"])


# ------------------------------------------------------------------ #
# Test 5: kill switch — 25% loss triggers
# ------------------------------------------------------------------ #
def test_kill_switch_drawdown():
    gc = GraduationController()
    gc.stage = Stage.SMALL_CAP_LIVE

    triggered = gc.check_kill_switch(
        daily_pnl_cents=-25000,  # $250 = 25% of $1000 bankroll
        bankroll=1000,
        consecutive_losses=2,
        model_stable=True,
    )
    assert triggered is True


# ------------------------------------------------------------------ #
# Test 6: kill switch — 5 consecutive losses triggers
# ------------------------------------------------------------------ #
def test_kill_switch_consecutive_losses():
    gc = GraduationController()
    gc.stage = Stage.SMALL_CAP_LIVE

    triggered = gc.check_kill_switch(
        daily_pnl_cents=-50,
        bankroll=1000,
        consecutive_losses=5,
        model_stable=True,
    )
    assert triggered is True


# ------------------------------------------------------------------ #
# Test 7: kill switch — moderate loss + 2 losses does NOT trigger
# ------------------------------------------------------------------ #
def test_kill_switch_ok():
    gc = GraduationController()
    gc.stage = Stage.SMALL_CAP_LIVE

    triggered = gc.check_kill_switch(
        daily_pnl_cents=-50,  # 5% of 1000, well under 20%
        bankroll=1000,
        consecutive_losses=2,
        model_stable=True,
    )
    assert triggered is False
