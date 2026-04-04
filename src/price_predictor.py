"""
Contract value predictor for Kalshi binary crypto contracts.

The key question is NOT "will BTC go up or down?" but rather:
"What is the probability that BTC crosses $85,000 in the next 15 minutes,
 and is the market's 5¢ YES price too high or too low for that probability?"

For each contract, we estimate:
1. P(spot crosses strike before expiration) — using recent volatility
2. Momentum — is price trending toward or away from the strike?
3. Fair value of the contract — adjusted P(cross) in cents
4. Edge — fair value minus market price
5. Whether the contract is undervalued (buy YES) or overvalued (buy NO)
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

    # Clamp to reasonable range — floor at 40% for crypto (never assume low vol)
    # Our price samples are ~60s apart from a cached API, which smooths out
    # real intra-minute volatility. 10% floor was way too low and made the
    # model overconfident on near-strike contracts.
    return max(min(annualized, 3.0), 0.40)


def _exponential_weights(n: int, half_life: float) -> np.ndarray:
    """Generate exponential decay weights where the most recent point has weight 1.

    half_life: number of observations at which weight drops to 0.5.
    Example: n=10, half_life=3 → oldest point ~0.1, newest point 1.0
    """
    decay = math.log(2) / max(half_life, 1)
    # Index 0 = oldest, n-1 = newest
    w = np.exp(decay * (np.arange(n, dtype=float) - (n - 1)))
    return w / np.sum(w)  # Normalize so they sum to 1


def compute_momentum(db, asset: str, strike: float, lookback_minutes: int = 10) -> Dict[str, Any]:
    """Compute short-term momentum relative to the strike price.

    Uses exponentially weighted regression — recent minutes count much more
    than older ones. Half-life is ~3 observations (3 minutes), so the last
    3 minutes carry ~50% of the total weight.

    Returns a dict with momentum metrics used to adjust strike probability.
    """
    since = (datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)).isoformat()
    prices = db.get_crypto_prices(asset=asset, since=since, limit=500)

    result = {
        "has_data": False,
        "direction": 0.0,       # +1 = toward strike (above), -1 = away
        "speed_pct_per_min": 0.0,  # % move per minute
        "r_squared": 0.0,       # trend confidence (0-1)
        "price_vs_strike": 0.0, # current distance as % of spot
        "adjustment": 0.0,      # probability adjustment (-0.15 to +0.15)
        "data_points": 0,
    }

    price_vals = [p["price_usd"] for p in prices if p.get("price_usd") and p["price_usd"] > 0]
    if len(price_vals) < 3:
        return result

    result["has_data"] = True
    result["data_points"] = len(price_vals)

    n = len(price_vals)
    current = price_vals[-1]

    x = np.arange(n, dtype=float)
    y = np.array(price_vals, dtype=float)

    # Exponential weights: half-life of 3 observations (~3 min)
    # Last 3 minutes carry ~50% of total weight
    w = _exponential_weights(n, half_life=3.0)

    # Weighted linear regression
    w_sum = np.sum(w)  # = 1.0 after normalization, but keep explicit
    wx_sum = np.sum(w * x)
    wy_sum = np.sum(w * y)
    wxy_sum = np.sum(w * x * y)
    wxx_sum = np.sum(w * x * x)

    denom = w_sum * wxx_sum - wx_sum * wx_sum
    if denom == 0:
        return result

    slope = (w_sum * wxy_sum - wx_sum * wy_sum) / denom
    intercept = (wy_sum - slope * wx_sum) / w_sum

    # Weighted R²
    y_pred = slope * x + intercept
    ss_res = np.sum(w * (y - y_pred) ** 2)
    y_wmean = wy_sum / w_sum
    ss_tot = np.sum(w * (y - y_wmean) ** 2)
    r_squared = max(1 - (ss_res / ss_tot) if ss_tot > 0 else 0, 0.0)

    # Weighted speed: use exponentially weighted returns
    returns = np.diff(y) / y[:-1]  # per-observation returns
    ret_weights = _exponential_weights(len(returns), half_life=3.0)
    weighted_speed = float(np.sum(ret_weights * returns))  # weighted avg return per obs

    # Direction relative to strike
    price_rising = slope > 0
    direction = 1.0 if price_rising else -1.0

    result["direction"] = direction
    result["speed_pct_per_min"] = round(weighted_speed * 100, 4)
    result["r_squared"] = round(r_squared, 3)
    result["price_vs_strike"] = round((current - strike) / current * 100, 3) if current > 0 else 0

    # Probability adjustment:
    # direction * |weighted_speed| * trend_confidence, capped at ±15%
    raw_adj = direction * abs(weighted_speed) * r_squared * 30
    adjustment = max(-0.15, min(0.15, raw_adj))
    result["adjustment"] = round(adjustment, 4)

    return result


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
    momentum_weight: float = 1.0,
    min_edge_override: float = None,
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

    # Base probability from log-normal model (static snapshot)
    base_prob = estimate_strike_probability(spot_price, strike_price, hours_left, vol, direction)

    # Momentum adjustment: is price trending toward or away from strike?
    # momentum_weight is learned daily — scales the ±15% cap up or down
    momentum = compute_momentum(db, asset, strike_price)
    adj = momentum["adjustment"] * min(momentum_weight, 2.0)  # Scale within bounds
    adj = max(-0.15, min(0.15, adj))  # Fixed ±15% cap, never scales
    prob = max(0.01, min(0.99, base_prob + adj))

    # Fair values: prob = chance YES wins
    fair_yes = round(prob * 100, 1)
    fair_no = round((1 - prob) * 100, 1)

    # Edge = fair value - market price (positive = undervalued = buy opportunity)
    edge_yes = fair_yes - yes_price_cents if yes_price_cents else 0
    edge_no = fair_no - no_price_cents if no_price_cents else 0

    result.update({
        "probability": round(prob, 4),
        "base_probability": round(base_prob, 4),
        "momentum_adj": round(adj, 4),
        "momentum": momentum,
        "fair_value_yes": fair_yes,
        "fair_value_no": fair_no,
        "edge_yes": round(edge_yes, 1),
        "edge_no": round(edge_no, 1),
        "hours_left": round(hours_left, 2),
        "volatility": round(vol, 3),
    })

    # Minimum edge: 5¢ for 15-min (4¢ fees + 1¢ profit), 9¢ for longer
    is_short_term = hours_left < 1
    if min_edge_override is not None:
        min_edge = min_edge_override
    else:
        min_edge = 5 if is_short_term else 9

    distance_pct = abs(spot_price - strike_price) / spot_price * 100
    result["distance_pct"] = round(distance_pct, 2)
    result["reasons"].append(f"Spot ${spot_price:,.0f} vs strike ${strike_price:,.0f} ({distance_pct:.1f}% away)")
    result["reasons"].append(f"Time: {hours_left:.1f}h, Vol: {vol:.0%}")
    mom_str = f"Mom: {adj:+.1%}" if momentum["has_data"] else "Mom: no data"
    result["reasons"].append(f"P({direction}): {base_prob:.1%}→{prob:.1%} ({mom_str}, R²={momentum['r_squared']:.2f})")
    result["reasons"].append(f"Fair YES: {fair_yes:.0f}¢, Market: {yes_price_cents:.0f}¢")

    # Don't trade 15-min when probability is between 48-52% (true coin flip)
    if is_short_term and 0.48 < prob < 0.52:
        result["recommendation"] = "skip"
        result["reasons"].append(f"Probability {prob:.0%} too close to 50/50 — no conviction")
        return result

    # Market disagreement check: if our model and the market disagree by >20pp,
    # the market is almost certainly more right (it has order flow info we don't).
    # Skip when the disagreement is too large — we're probably miscalibrated.
    market_implied_yes = yes_price_cents / 100.0 if yes_price_cents else 0.5
    model_market_gap = abs(prob - market_implied_yes)
    if is_short_term and model_market_gap > 0.20:
        result["recommendation"] = "skip"
        result["reasons"].append(
            f"Model-market disagreement too large: model={prob:.0%} vs market={market_implied_yes:.0%} "
            f"(gap={model_market_gap:.0%}). Market likely knows something we don't.")
        return result

    # Only buy the side we think actually wins (prob > 50%)
    # A "good price" on the losing side is still a losing bet
    if prob > 0.50 and edge_yes >= min_edge and yes_price_cents > 0:
        result["recommendation"] = "buy_yes"
        result["confidence"] = min(edge_yes / 20, 1.0)
        result["trade_price"] = yes_price_cents
        result["reasons"].append(f"YES undervalued by {edge_yes:.0f}¢ — BUY YES (P={prob:.0%})")
    elif prob < 0.50 and edge_no >= min_edge and no_price_cents > 0:
        result["recommendation"] = "buy_no"
        result["confidence"] = min(edge_no / 20, 1.0)
        result["trade_price"] = no_price_cents
        result["reasons"].append(f"NO undervalued by {edge_no:.0f}¢ — BUY NO (P(no)={1-prob:.0%})")
    else:
        result["recommendation"] = "skip"
        result["reasons"].append(f"No winning edge (P={prob:.0%}, YES edge: {edge_yes:+.0f}¢, NO edge: {edge_no:+.0f}¢)")

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
