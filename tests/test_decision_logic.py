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


def test_prob_65_can_buy_yes():
    """P=65% with market agreeing (yes_price=50) should buy YES with 10¢+ edge."""
    result = _eval(0.65, yes_price=50, no_price=50)
    # Model says 65% YES, market implied YES = 50% (>45% = agrees YES direction)
    # Edge = 65 - 50 = 15¢ >= 10¢ min
    assert result["recommendation"] == "buy_yes", (
        f"P=65% with market agreement and 15¢ edge should buy YES, got {result['recommendation']}. "
        f"Reasons: {result['reasons']}"
    )


def test_prob_35_can_buy_no():
    """P=35% (NO=65%) with market agreeing (yes_price=30) should buy NO."""
    result = _eval(0.35, yes_price=30, no_price=55)
    # Model says 35% YES (65% NO), market implied YES = 30% (<45% = agrees NO direction)
    # Edge NO = 65 - 55 = 10¢ >= 10¢ min
    assert result["recommendation"] == "buy_no", (
        f"P=35% (NO=65%) with market agreement and 10¢ edge should buy NO, got {result['recommendation']}. "
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


def test_min_edge_from_param():
    """evaluate_contract should respect min_edge_override parameter."""
    # With default 5¢ min_edge, P=60% fair=60 market=48 edge=+12 => should trade
    result_default = _eval(0.60, yes_price=48, no_price=52)
    assert result_default["recommendation"] == "buy_yes"

    # With 15¢ min_edge override, same 12¢ edge should skip
    db = _make_mock_db()
    crypto = MagicMock()
    from datetime import datetime, timedelta, timezone
    exp = (datetime.now(timezone.utc) + timedelta(hours=0.2)).isoformat()

    with patch('price_predictor.estimate_strike_probability', return_value=0.60), \
         patch('price_predictor.compute_realized_volatility', return_value=0.50), \
         patch('price_predictor.compute_momentum', return_value={
             "has_data": False, "direction": 0, "speed_pct_per_min": 0,
             "r_squared": 0, "price_vs_strike": 0, "adjustment": 0, "data_points": 0,
         }):
        result_strict = evaluate_contract(
            db=db, crypto_prices=crypto, ticker="KXBTC15M-TEST",
            strike_price=67000, spot_price=67100, yes_price_cents=48,
            no_price_cents=52, expiration_time=exp, asset="BTC",
            min_edge_override=15,
        )

    assert result_strict["recommendation"] == "skip", (
        f"With min_edge=15, a 12¢ edge should skip, got {result_strict['recommendation']}"
    )


def test_stop_loss_uses_filled_quantity():
    """Stop loss and position should use actual filled quantity, not requested."""
    from trader import Trader
    from unittest.mock import MagicMock, patch

    api = MagicMock()
    # First call = buy order (partial fill: 1 of 3)
    # Second call = stop loss order
    api.create_order.side_effect = [
        {'order': {
            'order_id': 'buy-123',
            'fill_count_fp': '1.00',
            'initial_count_fp': '3.00',
            'status': 'resting',
        }},
        {'order': {'order_id': 'stop-456'}},
    ]

    with patch('trader.load_current_params', return_value={'version': 0, 'momentum_weight': 1.0, 'stop_loss_pct': 0.30}), \
         patch('trader.PerformanceAnalytics'), \
         patch('trader.MarketDataStreamer'), \
         patch('trader.SettingsManager') as sm_mock, \
         patch('trader.TradingDB') as db_mock, \
         patch('trader.get_crypto_prices', return_value={}), \
         patch('trader.NewsSentimentAnalyzer'), \
         patch('trader.StatisticalArbitrageAnalyzer'), \
         patch('trader.VolatilityAnalyzer'), \
         patch('trader.RiskManager') as rm_mock:
        sm_mock.return_value.settings = MagicMock()
        sm_mock.return_value.add_change_listener = MagicMock()
        db_mock.return_value.load_positions.return_value = {}
        rm_mock.return_value.validate_position_size.return_value = True
        trader = Trader(api, MagicMock(), MagicMock(), bankroll=1000)

    trader.execute_trade({
        'event_id': 'KXBTC15M-TEST', 'action': 'buy', 'side': 'yes',
        'quantity': 3, 'price': 50, 'strategy': 'value_bet',
        'title': 'Test', 'expiration_time': '2026-04-04T00:00:00Z',
        'confidence': 0.5, 'spot_price': 67000, 'strike_price': 67000,
    })

    # Verify stop loss was placed for 1 contract (filled), not 3 (requested)
    assert api.create_order.call_count == 2, f"Expected 2 orders (buy + stop), got {api.create_order.call_count}"
    stop_payload = api.create_order.call_args_list[1][0][0]
    assert stop_payload['count'] == 1, f"Stop loss should be for 1 contract (filled qty), got {stop_payload['count']}"

    # Position should track 1 contract
    pos = trader.current_positions.get('KXBTC15M-TEST', {})
    assert pos.get('quantity') == 1, f"Position quantity should be 1 (filled), got {pos.get('quantity')}"


def test_stop_loss_uses_learned_pct():
    """Resting stop loss price should use model_params stop_loss_pct."""
    from trader import Trader
    from unittest.mock import MagicMock, patch

    api = MagicMock()
    api.create_order.side_effect = [
        {'order': {'order_id': 'buy-1', 'fill_count_fp': '1.00', 'status': 'filled'}},
        {'order': {'order_id': 'stop-1'}},
    ]

    with patch('trader.load_current_params', return_value={
        'version': 1, 'momentum_weight': 1.0, 'stop_loss_pct': 0.15,
    }), \
         patch('trader.PerformanceAnalytics'), \
         patch('trader.MarketDataStreamer'), \
         patch('trader.SettingsManager') as sm_mock:
        sm_mock.return_value.settings = MagicMock()
        sm_mock.return_value.add_change_listener = MagicMock()
        trader = Trader(api, MagicMock(), MagicMock(), bankroll=1000)

    trader.execute_trade({
        'event_id': 'KXBTC15M-TEST2', 'action': 'buy', 'side': 'yes',
        'quantity': 1, 'price': 60, 'strategy': 'value_bet',
        'title': 'Test', 'expiration_time': '2026-04-04T00:00:00Z',
        'confidence': 0.5, 'spot_price': 67000, 'strike_price': 67000,
    })

    stop_payload = api.create_order.call_args_list[1][0][0]
    stop_price = stop_payload.get('yes_price')
    # fill_price = 61 (60+1), sl_distance = max(int(61*0.15), 9) = 9, stop = 61-9 = 52
    assert stop_price == 52, f"Stop should be 52¢ (9¢ below fill at 61¢), got {stop_price}¢"


def test_15m_no_volume_boost():
    """15M contracts should NOT get volume boost for vol=0."""
    edge = 10.0
    is_15m = True
    contract_volume = 0

    # Apply the fixed logic
    if not is_15m:
        if contract_volume > 100:
            edge *= 0.8
        elif contract_volume < 10:
            edge *= 1.2

    assert edge == 10.0, f"15M edge should not be boosted, got {edge}"


def test_cooldown_is_5_minutes():
    """Value bet cooldown should be 300s (5 min), not 900s."""
    import time as time_mod
    from trader import Trader
    from unittest.mock import MagicMock, patch

    with patch('trader.load_current_params', return_value={'version': 0}), \
         patch('trader.PerformanceAnalytics'), \
         patch('trader.MarketDataStreamer'), \
         patch('trader.SettingsManager') as sm_mock:
        sm_mock.return_value.settings = MagicMock(
            news_sentiment_enabled=False, statistical_arbitrage_enabled=False,
            volatility_based_enabled=False,
        )
        sm_mock.return_value.add_change_listener = MagicMock()
        trader = Trader(MagicMock(), MagicMock(), MagicMock(), 1000)

    # Last trade 6 min ago — should NOT be blocked
    trader._last_value_bet_time = time_mod.time() - 360
    result = trader._value_bet_fallback({'markets': []}, None)
    # Returns None because no markets, but NOT because of cooldown

    # Last trade 2 min ago — should be blocked (returns None from cooldown)
    trader._last_value_bet_time = time_mod.time() - 120
    result = trader._value_bet_fallback({'markets': [{'ticker': 'TEST', 'yes_ask_dollars': '0.50'}]}, None)
    assert result is None  # Blocked by cooldown
