"""
Contract value predictor for Kalshi binary crypto contracts.

The key question is NOT "will BTC go up or down?" but rather:
"What is the probability that BTC crosses $85,000 in the next 15 minutes,
 and is the market's 5¢ YES price too high or too low for that probability?"

For each contract, we estimate:
1. P(spot crosses strike before expiration) — using recent volatility
2. Fair value of the contract — P(cross) in cents
3. Edge — fair value minus market price
4. Whether the contract is undervalued (buy YES) or overvalued (buy NO)
"""

import logging
import math
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def estimate_strike_probability(
    spot: float,
    strike: float,
    hours_to_expiry: float,
    recent_volatility: float,
    direction: str = "above",
) -> float:
    """Estimate probability that spot crosses strike before expiry.

    Uses a simplified model based on:
    - Distance from spot to strike (as % of spot)
    - Time remaining (more time = more chance of large moves)
    - Recent realized volatility (how much the asset actually moves)

    This is similar to a Black-Scholes-style calculation for binary options,
    but uses realized vol instead of implied vol.

    Args:
        spot: Current spot price (e.g. 67000)
        strike: Contract target price (e.g. 85000)
        hours_to_expiry: Hours until contract settles
        recent_volatility: Annualized volatility (e.g. 0.60 for 60%)
        direction: "above" (YES wins if spot > strike) or "below"

    Returns:
        Probability 0.0 to 1.0
    """
    if spot <= 0 or strike <= 0 or hours_to_expiry <= 0 or recent_volatility <= 0:
        return 0.5  # No data, no edge

    # Convert annualized vol to vol over the contract period
    # vol_period = annual_vol * sqrt(hours / 8760)
    period_vol = recent_volatility * math.sqrt(hours_to_expiry / 8760)

    if period_vol < 0.0001:
        # Essentially no time/vol left — price won't move
        if direction == "above":
            return 1.0 if spot > strike else 0.0
        else:
            return 1.0 if spot < strike else 0.0

    # Log-normal model: z = ln(strike/spot) / period_vol
    log_ratio = math.log(strike / spot)
    z = log_ratio / period_vol

    # P(spot > strike) = 1 - Phi(z) where Phi is standard normal CDF
    # Using the error function approximation
    from math import erf
    phi_z = 0.5 * (1 + erf(z / math.sqrt(2)))

    if direction == "above":
        return 1.0 - phi_z
    else:
        return phi_z


def compute_realized_volatility(db, asset: str, lookback_hours: int = 24) -> float:
    """Compute annualized realized volatility from recent spot prices.

    Args:
        db: TradingDB instance
        asset: "BTC", "ETH", or "SOL"
        lookback_hours: Hours of history to use

    Returns:
        Annualized volatility (e.g. 0.60 for 60%)
    """
    since = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
    prices = db.get_crypto_prices(asset=asset, since=since, limit=1000)

    if len(prices) < 5:
        # Fallback: typical crypto volatility
        defaults = {"BTC": 0.55, "ETH": 0.70, "SOL": 0.85}
        return defaults.get(asset, 0.65)

    price_vals = [p["price_usd"] for p in prices if p.get("price_usd") and p["price_usd"] > 0]
    if len(price_vals) < 5:
        return 0.65

    # Compute log returns
    returns = [math.log(price_vals[i] / price_vals[i - 1])
               for i in range(1, len(price_vals))
               if price_vals[i - 1] > 0]

    if not returns:
        return 0.65

    # Estimate time between observations (in hours)
    # We record spot prices every ~60s, so each return ≈ 1 minute
    obs_per_hour = len(returns) / lookback_hours
    if obs_per_hour < 1:
        obs_per_hour = 1

    # Annualize: vol * sqrt(observations_per_year)
    obs_per_year = obs_per_hour * 8760
    std_dev = float(np.std(returns))
    annualized = std_dev * math.sqrt(obs_per_year)

    # Clamp to reasonable range
    return max(min(annualized, 3.0), 0.10)


def evaluate_contract(
    db,
    crypto_prices,
    ticker: str,
    strike_price: float,
    spot_price: float,
    yes_price_cents: float,
    no_price_cents: float,
    expiration_time: str,
    asset: str,
) -> Dict[str, Any]:
    """Evaluate whether a contract is undervalued.

    Computes:
    - Estimated probability of YES outcome
    - Fair value in cents
    - Edge = fair_value - market_price
    - Recommendation: buy YES, buy NO, or skip

    Args:
        db: TradingDB instance
        crypto_prices: CryptoPrices instance
        ticker: Market ticker
        strike_price: Target price from ticker (e.g. 85000)
        spot_price: Current spot price
        yes_price_cents: Current YES ask in cents
        no_price_cents: Current NO ask in cents
        expiration_time: ISO timestamp of contract expiry
        asset: "BTC", "ETH", or "SOL"

    Returns:
        Evaluation dict with probability, fair_value, edge, recommendation
    """
    result = {
        "ticker": ticker,
        "asset": asset,
        "strike": strike_price,
        "spot": spot_price,
        "yes_price": yes_price_cents,
        "no_price": no_price_cents,
        "probability": 0.5,
        "fair_value_yes": 50,
        "fair_value_no": 50,
        "edge_yes": 0,
        "edge_no": 0,
        "recommendation": "skip",
        "confidence": 0.0,
        "reasons": [],
    }

    if not strike_price or not spot_price or not expiration_time:
        result["reasons"].append("Missing strike, spot, or expiration data")
        return result

    # Compute hours to expiry
    try:
        exp_dt = datetime.fromisoformat(expiration_time.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        hours_left = (exp_dt - now).total_seconds() / 3600
        if hours_left <= 0:
            result["reasons"].append("Contract expired")
            return result
    except (ValueError, TypeError):
        result["reasons"].append("Cannot parse expiration time")
        return result

    # Get recent realized volatility
    vol = compute_realized_volatility(db, asset)

    # Determine direction from ticker
    # MINMON = "how low will X get" → YES wins if price goes BELOW strike
    # MAXMON = "how high will X get" → YES wins if price goes ABOVE strike
    # 15M "up or down" → YES wins if price goes ABOVE target
    if "MINMON" in ticker.upper() or "MIN" in ticker.upper():
        direction = "below"  # YES = price drops below strike
    else:
        direction = "above"  # YES = price rises above strike

    # Estimate probability that spot crosses strike
    prob = estimate_strike_probability(spot_price, strike_price, hours_left, vol, direction)

    # Fair values: prob = chance YES wins
    fair_yes = round(prob * 100, 1)
    fair_no = round((1 - prob) * 100, 1)

    # Edge = fair value - market price (positive = undervalued = buy opportunity)
    edge_yes = fair_yes - yes_price_cents if yes_price_cents else 0
    edge_no = fair_no - no_price_cents if no_price_cents else 0

    result.update({
        "probability": round(prob, 4),
        "fair_value_yes": fair_yes,
        "fair_value_no": fair_no,
        "edge_yes": round(edge_yes, 1),
        "edge_no": round(edge_no, 1),
        "hours_left": round(hours_left, 2),
        "volatility": round(vol, 3),
    })

    # Minimum edge to trade (in cents)
    min_edge = 3  # Need at least 3¢ edge

    distance_pct = abs(spot_price - strike_price) / spot_price * 100
    result["distance_pct"] = round(distance_pct, 2)
    result["reasons"].append(f"Spot ${spot_price:,.0f} vs strike ${strike_price:,.0f} ({distance_pct:.1f}% away)")
    result["reasons"].append(f"Time: {hours_left:.1f}h, Vol: {vol:.0%}")
    result["reasons"].append(f"P({direction} strike): {prob:.1%}, Fair YES: {fair_yes:.0f}¢, Market: {yes_price_cents:.0f}¢")

    if edge_yes >= min_edge and yes_price_cents > 0:
        result["recommendation"] = "buy_yes"
        result["confidence"] = min(edge_yes / 10, 1.0)  # 10¢ edge = 100% confidence
        result["trade_price"] = yes_price_cents
        result["reasons"].append(f"YES undervalued by {edge_yes:.0f}¢ — BUY YES")
    elif edge_no >= min_edge and no_price_cents > 0:
        result["recommendation"] = "buy_no"
        result["confidence"] = min(edge_no / 10, 1.0)
        result["trade_price"] = no_price_cents
        result["reasons"].append(f"NO undervalued by {edge_no:.0f}¢ — BUY NO")
    else:
        result["recommendation"] = "skip"
        result["reasons"].append(f"No edge (YES edge: {edge_yes:+.0f}¢, NO edge: {edge_no:+.0f}¢)")

    return result


# Keep old interface for backward compatibility during transition
def predict_direction(db, asset: str, crypto_prices) -> Dict[str, Any]:
    """Legacy interface — now wraps evaluate_contract logic.

    Returns the old-style prediction dict for callers that still use it.
    """
    result = {
        "direction": None,
        "confidence": 0.0,
        "should_trade": False,
        "reasons": [],
    }

    spot = crypto_prices.get_price(asset)
    if not spot:
        result["reasons"].append(f"No spot price for {asset}")
        return result

    vol = compute_realized_volatility(db, asset)

    # Use 24h change as directional signal
    change_24h = crypto_prices.get_change_24h(asset)
    if change_24h is not None and abs(change_24h) > 1.0:
        result["direction"] = "up" if change_24h > 0 else "down"
        result["confidence"] = min(abs(change_24h) / 10, 0.5)
        result["should_trade"] = result["confidence"] >= 0.15
        result["reasons"].append(f"24h change: {change_24h:+.1f}%, vol: {vol:.0%}")
    else:
        result["direction"] = "up"
        result["confidence"] = 0.05
        result["reasons"].append("No strong directional signal")

    logger.info(f"{asset} prediction: {(result['direction'] or 'NONE').upper()} "
                f"({result['confidence']:.0%} confidence, spot: ${spot:,.0f}) — "
                f"{', '.join(result['reasons'])}")
    return result
