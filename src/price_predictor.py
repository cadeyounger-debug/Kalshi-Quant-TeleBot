"""
Short-term price direction predictor for BTC, ETH, SOL.

Uses rolling spot price data from the database to predict whether
the price will be UP or DOWN in the next 15 minutes.

Signals used:
1. Short-term momentum (last 5 min vs last 15 min)
2. Medium-term momentum (last 15 min vs last 60 min)
3. Trend strength (how consistent is the direction?)
4. Volatility regime (high vol = harder to predict)
5. Spot price change (24h trend)
6. Historical win rate at this hour (time-of-day pattern)

Outputs a prediction with confidence:
  {"direction": "up", "confidence": 0.72, "reasons": [...]}
"""

import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def predict_direction(
    db,
    asset: str,
    crypto_prices,
) -> Dict[str, Any]:
    """Predict if asset will be UP or DOWN in next 15 minutes.

    Args:
        db: TradingDB instance
        asset: "BTC", "ETH", or "SOL"
        crypto_prices: CryptoPrices instance for current spot data

    Returns:
        {
            "direction": "up" or "down",
            "confidence": 0.0-1.0,
            "should_trade": bool,
            "reasons": [str, ...],
        }
    """
    result = {
        "direction": None,
        "confidence": 0.0,
        "should_trade": False,
        "reasons": [],
    }

    signals = []  # List of (direction_score, weight, reason)
    # direction_score: positive = up, negative = down

    # --- Signal 1: Short-term momentum (last 5 min) ---
    recent_prices = db.get_crypto_prices(asset=asset, limit=10)
    if len(recent_prices) >= 3:
        prices = [p["price_usd"] for p in recent_prices[-5:] if p.get("price_usd")]
        if len(prices) >= 2:
            short_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
            # Momentum: if price is going up, predict up
            direction_score = np.sign(short_change) * min(abs(short_change) * 100, 1.0)
            signals.append((direction_score, 2.0, f"5-min momentum: {short_change:+.3%}"))

    # --- Signal 2: Medium-term momentum (last 60 min) ---
    since_1h = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    hourly_prices = db.get_crypto_prices(asset=asset, since=since_1h, limit=100)
    if len(hourly_prices) >= 5:
        prices = [p["price_usd"] for p in hourly_prices if p.get("price_usd")]
        if len(prices) >= 5:
            med_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
            direction_score = np.sign(med_change) * min(abs(med_change) * 50, 1.0)
            signals.append((direction_score, 1.5, f"60-min momentum: {med_change:+.3%}"))

    # --- Signal 3: Trend consistency ---
    if len(hourly_prices) >= 10:
        prices = [p["price_usd"] for p in hourly_prices if p.get("price_usd")]
        if len(prices) >= 10:
            changes = [prices[i+1] - prices[i] for i in range(len(prices)-1)]
            up_count = sum(1 for c in changes if c > 0)
            down_count = sum(1 for c in changes if c < 0)
            total = up_count + down_count
            if total > 0:
                consistency = abs(up_count - down_count) / total
                dominant_dir = 1.0 if up_count > down_count else -1.0
                signals.append((dominant_dir * consistency, 1.0,
                               f"Trend consistency: {consistency:.0%} {'up' if dominant_dir > 0 else 'down'}"))

    # --- Signal 4: 24h change from spot API ---
    change_24h = crypto_prices.get_change_24h(asset)
    if change_24h is not None:
        direction_score = np.sign(change_24h) * min(abs(change_24h) / 10, 1.0)
        signals.append((direction_score, 0.5, f"24h change: {change_24h:+.1f}%"))

    # --- Signal 5: Volatility regime ---
    if len(hourly_prices) >= 10:
        prices = [p["price_usd"] for p in hourly_prices if p.get("price_usd")]
        if len(prices) >= 10:
            returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1) if prices[i] > 0]
            if returns:
                volatility = np.std(returns)
                # High volatility = lower confidence in any direction
                if volatility > 0.005:  # > 0.5% per minute = very volatile
                    signals.append((0, 1.0, f"High volatility ({volatility:.4f}) — reducing confidence"))

    # --- Signal 6: Historical outcomes for 15-min contracts at this hour ---
    # Check how past 15-min contracts resolved for this asset
    since_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    past_trades = db.get_trades(asset=asset, since=since_7d, limit=100)
    if past_trades:
        wins = sum(1 for t in past_trades if t.get("pnl") and t["pnl"] > 0)
        losses = sum(1 for t in past_trades if t.get("pnl") and t["pnl"] < 0)
        if wins + losses > 3:
            win_rate = wins / (wins + losses)
            # If we've been winning, slightly boost confidence
            signals.append(((win_rate - 0.5) * 2, 0.5,
                           f"Historical win rate: {win_rate:.0%} ({wins}W/{losses}L)"))

    # --- Combine signals ---
    if not signals:
        result["reasons"].append("Insufficient data for prediction")
        return result

    total_weight = sum(abs(w) for _, w, _ in signals)
    if total_weight == 0:
        return result

    weighted_score = sum(score * weight for score, weight, _ in signals) / total_weight
    confidence = min(abs(weighted_score), 1.0)

    result["direction"] = "up" if weighted_score > 0 else "down"
    result["confidence"] = round(confidence, 3)
    result["should_trade"] = confidence >= 0.15  # Low bar — we're learning
    result["reasons"] = [reason for _, _, reason in signals]

    spot = crypto_prices.get_price(asset)
    spot_str = f"${spot:,.0f}" if spot else "?"
    logger.info(f"{asset} prediction: {result['direction'].upper()} "
                f"({confidence:.0%} confidence, spot: {spot_str}) — "
                f"{', '.join(result['reasons'])}")

    return result
