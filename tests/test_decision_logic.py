"""Tests for decision logic fixes in the trading pipeline."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import MagicMock, patch
from price_predictor import evaluate_contract


def _make_mock_db(prices=None):
    db = MagicMock()
    db.get_crypto_prices.return_value = prices or []
    return db


def _eval(prob_override, yes_price=40, no_price=60, hours_left=0.2):
    """Helper: evaluate a contract with a forced probability."""
    db = _make_mock_db()
    crypto = MagicMock()
    from datetime import datetime, timedelta, timezone
    exp = (datetime.now(timezone.utc) + timedelta(hours=hours_left)).isoformat()

    with patch('price_predictor.estimate_strike_probability', return_value=prob_override), \
         patch('price_predictor.compute_realized_volatility', return_value=0.50), \
         patch('price_predictor.compute_momentum', return_value={
             "has_data": False, "direction": 0, "speed_pct_per_min": 0,
             "r_squared": 0, "price_vs_strike": 0, "adjustment": 0, "data_points": 0,
         }):
        return evaluate_contract(
            db=db, crypto_prices=crypto, ticker="KXBTC15M-TEST",
            strike_price=67000, spot_price=67000, yes_price_cents=yes_price,
            no_price_cents=no_price, expiration_time=exp, asset="BTC",
        )


def test_prob_53_can_buy_yes():
    """P=53% should buy YES — not blocked by no-conviction band."""
    result = _eval(0.53, yes_price=40, no_price=60)
    assert result["recommendation"] == "buy_yes", (
        f"P=53% with edge should buy YES, got {result['recommendation']}. "
        f"Reasons: {result['reasons']}"
    )


def test_prob_47_can_buy_no():
    """P=47% means P(NO)=53% — should buy NO."""
    result = _eval(0.47, yes_price=60, no_price=40)
    assert result["recommendation"] == "buy_no", (
        f"P=47% (P(NO)=53%) with edge should buy NO, got {result['recommendation']}. "
        f"Reasons: {result['reasons']}"
    )


def test_prob_50_skips():
    """P=50% is a true coin flip — should skip."""
    result = _eval(0.50, yes_price=40, no_price=40)
    assert result["recommendation"] == "skip"


def test_prob_49_skips():
    """P=49% is in the 48-52 no-conviction zone — should skip."""
    result = _eval(0.49, yes_price=40, no_price=40)
    assert result["recommendation"] == "skip"


def test_prob_51_skips():
    """P=51% is in the 48-52 no-conviction zone — should skip."""
    result = _eval(0.51, yes_price=40, no_price=60)
    assert result["recommendation"] == "skip"


def test_momentum_cap_fixed_at_15_pct():
    """Momentum adjustment must never exceed ±15% regardless of momentum_weight."""
    db = _make_mock_db()
    crypto = MagicMock()
    from datetime import datetime, timedelta, timezone
    exp = (datetime.now(timezone.utc) + timedelta(hours=0.2)).isoformat()

    with patch('price_predictor.estimate_strike_probability', return_value=0.50), \
         patch('price_predictor.compute_realized_volatility', return_value=0.50), \
         patch('price_predictor.compute_momentum', return_value={
             "has_data": True, "direction": 1.0, "speed_pct_per_min": 5.0,
             "r_squared": 0.99, "price_vs_strike": 1.0, "adjustment": 0.15,
             "data_points": 10,
         }):
        result = evaluate_contract(
            db=db, crypto_prices=crypto, ticker="KXBTC15M-TEST",
            strike_price=67000, spot_price=67100, yes_price_cents=50,
            no_price_cents=50, expiration_time=exp, asset="BTC",
            momentum_weight=2.0,
        )

    # Base = 0.50, max adj should be +0.15, so prob <= 0.65
    assert result["probability"] <= 0.66, (
        f"With momentum_weight=2.0, prob should still be capped near 0.65, "
        f"got {result['probability']}"
    )
    assert result["momentum_adj"] <= 0.15, (
        f"Momentum adj should be capped at 0.15, got {result['momentum_adj']}"
    )
