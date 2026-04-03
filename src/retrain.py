#!/usr/bin/env python3
"""
Daily retraining script for the Kalshi trading bot.

Runs every morning at 7am. Analyzes ALL data — not just trades we made,
but every market we observed. This includes:

1. PRICE MOVEMENTS — For every market snapshot, what happened next?
   Did the price go up, down, or stay flat? At what price ranges do
   markets tend to move the most?

2. MISSED OPPORTUNITIES — Markets we saw but didn't trade. If we had
   bought YES at 30¢ and it went to 60¢, that's a missed win. If we
   had bought at 30¢ and it went to 10¢, we correctly avoided it.

3. ACTUAL TRADES — What we traded, what happened, P&L.

4. SENTIMENT vs OUTCOMES — When sentiment was positive, did prices
   actually go up? Calibrates whether news signal is useful.

Outputs model_params.json consumed by the trader each cycle.
"""

import json
import logging
import os
import sqlite3
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "version": 0,
    "trained_at": None,
    "data_points": 0,
    "min_entry_price_cents": 20,
    "max_entry_price_cents": 80,
    "min_edge": 0.05,
    "buy_yes_below": 45,
    "buy_no_above": 55,
    "stop_loss_pct": 0.10,
    "breakeven_trigger": 0.15,
    "trail_trigger": 0.25,
    "trail_pct": 0.20,
    "time_exit_seconds": 120,
    "strategy_weights": {
        "news_sentiment": 1.0,
        "statistical_arbitrage": 1.0,
        "volatility_based": 1.0,
        "value_bet": 1.0,
    },
    "asset_weights": {
        "BTC": 1.0,
        "ETH": 1.0,
        "SOL": 1.0,
    },
    "sentiment_threshold": 0.6,
    "min_sentiment_confidence": 0.2,
    "max_positions": 3,
}

PARAMS_PATH = os.environ.get(
    "MODEL_PARAMS_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_params.json"),
)

DB_PATH = os.environ.get(
    "TRADING_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trading_bot.db"),
)


def load_current_params() -> Dict[str, Any]:
    try:
        with open(PARAMS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_PARAMS.copy()


def save_params(params: Dict[str, Any]):
    with open(PARAMS_PATH, 'w') as f:
        json.dump(params, f, indent=2, default=str)
    logger.info(f"Saved model params to {PARAMS_PATH}")


def query_db(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"DB query error: {e}")
        return []


def get_cutoff_date() -> str:
    cutoff = datetime.now(timezone.utc) - timedelta(days=90)
    return cutoff.isoformat()


def get_yesterday_range() -> Tuple[str, str]:
    """Return (start, end) ISO strings for yesterday UTC."""
    now = datetime.now(timezone.utc)
    end = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=1)
    return start.isoformat(), end.isoformat()


# ---------------------------------------------------------------------------
# Analysis 1: Price movements across ALL observed markets
# ---------------------------------------------------------------------------
def _mid_price(snap, side="yes") -> float:
    """Compute midpoint price for a snapshot to avoid bid-ask spread bias.

    Using (bid + ask) / 2 gives a fair value estimate that doesn't
    artificially show negative movement just because bid < ask.
    """
    if side == "yes":
        bid = snap.get("yes_bid") or 0
        ask = snap.get("yes_ask") or 0
    else:
        bid = snap.get("no_bid") or 0
        ask = snap.get("no_ask") or 0

    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return ask or bid or 0


def analyze_price_movements() -> Dict[str, Any]:
    """For each market ticker, track how the mid-price changed over time.

    Uses mid-price (avg of bid+ask) to avoid the systematic negative bias
    that occurs when comparing entry ask to exit bid (the spread always
    makes this look like a loss even when the market hasn't moved).

    Compares consecutive snapshots rather than first-to-last, which avoids
    conflating expired-at-zero contracts with actual price declines.
    """
    cutoff = get_cutoff_date()

    rows = query_db(
        """SELECT ticker, asset, yes_bid, yes_ask, no_bid, no_ask,
                  strike_price, spot_price, expiration_time, timestamp
           FROM market_snapshots
           WHERE timestamp >= ?
           ORDER BY ticker, timestamp""",
        (cutoff,)
    )

    if not rows:
        return {"total_markets_observed": 0}

    # Group by ticker
    by_ticker = defaultdict(list)
    for r in rows:
        by_ticker[r["ticker"]].append(r)

    results = {
        "total_markets_observed": len(by_ticker),
        "profitable_if_bought_yes": [],
        "profitable_if_bought_no": [],
        "price_range_outcomes": defaultdict(lambda: {"up": 0, "down": 0, "flat": 0, "avg_move": []}),
        "asset_outcomes": defaultdict(lambda: {"up": 0, "down": 0, "flat": 0, "avg_move": []}),
        # New: outcomes bucketed by distance-to-strike and time-to-expiry
        "distance_outcomes": defaultdict(lambda: {"up": 0, "down": 0, "flat": 0, "avg_move": []}),
        "time_to_expiry_outcomes": defaultdict(lambda: {"up": 0, "down": 0, "flat": 0, "avg_move": []}),
    }

    for ticker, snaps in by_ticker.items():
        if len(snaps) < 2:
            continue

        asset = snaps[0]["asset"]

        # --- YES side: compare consecutive mid-prices ---
        for i in range(len(snaps) - 1):
            entry_mid = _mid_price(snaps[i], "yes")
            exit_mid = _mid_price(snaps[i + 1], "yes")
            if entry_mid <= 0:
                continue

            change = (exit_mid - entry_mid) / entry_mid

            entry_cents = int(round(entry_mid * 100))
            bucket = (entry_cents // 10) * 10
            bucket_key = f"{bucket}-{bucket+10}"

            if change > 0.02:
                results["price_range_outcomes"][bucket_key]["up"] += 1
                results["asset_outcomes"][asset]["up"] += 1
            elif change < -0.02:
                results["price_range_outcomes"][bucket_key]["down"] += 1
                results["asset_outcomes"][asset]["down"] += 1
            else:
                results["price_range_outcomes"][bucket_key]["flat"] += 1
                results["asset_outcomes"][asset]["flat"] += 1

            results["price_range_outcomes"][bucket_key]["avg_move"].append(change)
            results["asset_outcomes"][asset]["avg_move"].append(change)

            # --- Bucket by distance-to-strike (spot vs target) ---
            snap = snaps[i]
            strike = snap.get("strike_price") or 0
            spot = snap.get("spot_price") or 0
            if strike > 0 and spot > 0:
                pct_distance = (spot - strike) / strike * 100  # positive = above strike
                if pct_distance > 5:
                    dist_key = "above_5pct"
                elif pct_distance > 0:
                    dist_key = "above_0-5pct"
                elif pct_distance > -5:
                    dist_key = "below_0-5pct"
                else:
                    dist_key = "below_5pct"

                bucket_data = results["distance_outcomes"][dist_key]
                if change > 0.02:
                    bucket_data["up"] += 1
                elif change < -0.02:
                    bucket_data["down"] += 1
                else:
                    bucket_data["flat"] += 1
                bucket_data["avg_move"].append(change)

            # --- Bucket by time-to-expiry ---
            exp_time = snap.get("expiration_time")
            snap_time = snap.get("timestamp")
            if exp_time and snap_time:
                try:
                    from datetime import datetime, timezone
                    exp_dt = datetime.fromisoformat(exp_time.replace("Z", "+00:00"))
                    snap_dt = datetime.fromisoformat(snap_time.replace("Z", "+00:00"))
                    hours_left = (exp_dt - snap_dt).total_seconds() / 3600
                    if hours_left < 0.5:
                        tte_key = "<30min"
                    elif hours_left < 2:
                        tte_key = "30min-2h"
                    elif hours_left < 12:
                        tte_key = "2h-12h"
                    elif hours_left < 48:
                        tte_key = "12h-2d"
                    else:
                        tte_key = ">2d"

                    tte_data = results["time_to_expiry_outcomes"][tte_key]
                    if change > 0.02:
                        tte_data["up"] += 1
                    elif change < -0.02:
                        tte_data["down"] += 1
                    else:
                        tte_data["flat"] += 1
                    tte_data["avg_move"].append(change)
                except (ValueError, TypeError):
                    pass

        # Check if buying YES at first mid would have been profitable at peak
        first_mid = _mid_price(snaps[0], "yes")
        max_mid = max(_mid_price(s, "yes") for s in snaps)
        if first_mid > 0:
            max_gain = (max_mid - first_mid) / first_mid
            if max_gain > 0.10:
                results["profitable_if_bought_yes"].append({
                    "ticker": ticker, "asset": asset,
                    "entry": first_mid, "max": max_mid,
                    "gain": max_gain,
                })

        # --- NO side: same consecutive mid-price approach ---
        for i in range(len(snaps) - 1):
            entry_mid = _mid_price(snaps[i], "no")
            exit_mid = _mid_price(snaps[i + 1], "no")
            if entry_mid <= 0:
                continue

            change = (exit_mid - entry_mid) / entry_mid

            entry_cents = int(round(entry_mid * 100))
            bucket = (entry_cents // 10) * 10
            bucket_key = f"{bucket}-{bucket+10}"

            if change > 0.02:
                results["price_range_outcomes"][bucket_key]["up"] += 1
            elif change < -0.02:
                results["price_range_outcomes"][bucket_key]["down"] += 1
            else:
                results["price_range_outcomes"][bucket_key]["flat"] += 1

            results["price_range_outcomes"][bucket_key]["avg_move"].append(change)

        first_no_mid = _mid_price(snaps[0], "no")
        max_no_mid = max(_mid_price(s, "no") for s in snaps)
        if first_no_mid > 0:
            max_gain = (max_no_mid - first_no_mid) / first_no_mid
            if max_gain > 0.10:
                results["profitable_if_bought_no"].append({
                    "ticker": ticker, "asset": asset,
                    "entry": first_no_mid, "max": max_no_mid,
                    "gain": max_gain,
                })

    logger.info(f"Analyzed {len(by_ticker)} unique markets — "
                f"{len(results['profitable_if_bought_yes'])} YES opportunities, "
                f"{len(results['profitable_if_bought_no'])} NO opportunities")
    return results


# ---------------------------------------------------------------------------
# Analysis 2: Missed opportunities
# ---------------------------------------------------------------------------
def analyze_missed_opportunities() -> Dict[str, Any]:
    """Compare trade decisions (what we considered) with actual outcomes.

    For markets we saw but didn't trade — would trading have been profitable?
    """
    cutoff = get_cutoff_date()

    # Get all trade decisions (including should_trade=False)
    decisions = query_db(
        """SELECT * FROM trade_decisions WHERE timestamp >= ?""",
        (cutoff,)
    )

    # Get actual trades
    trades = query_db(
        """SELECT market_ticker FROM trades WHERE timestamp >= ?""",
        (cutoff,)
    )
    traded_tickers = set(t["market_ticker"] for t in trades)

    # For each market we saw, get price trajectory
    all_snapshots = query_db(
        """SELECT ticker, asset, yes_bid, yes_ask, no_bid, no_ask, timestamp
           FROM market_snapshots WHERE timestamp >= ?
           ORDER BY ticker, timestamp""",
        (cutoff,)
    )

    price_trajectories = defaultdict(list)
    for s in all_snapshots:
        price_trajectories[s["ticker"]].append(s)

    missed_wins = []
    correct_passes = []

    for ticker, snaps in price_trajectories.items():
        if len(snaps) < 3 or ticker in traded_tickers:
            continue

        asset = snaps[0]["asset"]

        # Check YES side using mid-prices to avoid spread bias
        entry_mid = _mid_price(snaps[0], "yes")
        if entry_mid > 0:
            max_mid = max(_mid_price(s, "yes") for s in snaps)
            last_mid = _mid_price(snaps[-1], "yes")
            entry_cents = int(round(entry_mid * 100))
            max_gain = (max_mid - entry_mid) / entry_mid
            final_change = (last_mid - entry_mid) / entry_mid

            if 20 <= entry_cents <= 80:
                if max_gain > 0.20:
                    missed_wins.append({
                        "ticker": ticker, "asset": asset, "side": "yes",
                        "entry": entry_mid, "max": max_mid,
                        "gain_pct": max_gain, "entry_cents": entry_cents,
                    })
                elif final_change < -0.10:
                    correct_passes.append({
                        "ticker": ticker, "asset": asset, "side": "yes",
                        "entry": entry_mid, "loss_pct": final_change,
                    })

        # Check NO side using mid-prices
        entry_no_mid = _mid_price(snaps[0], "no")
        if entry_no_mid > 0:
            max_no_mid = max(_mid_price(s, "no") for s in snaps)
            last_no_mid = _mid_price(snaps[-1], "no")
            no_entry_cents = int(round(entry_no_mid * 100))
            no_max_gain = (max_no_mid - entry_no_mid) / entry_no_mid
            no_final_change = (last_no_mid - entry_no_mid) / entry_no_mid

            if 20 <= no_entry_cents <= 80:
                if no_max_gain > 0.20:
                    missed_wins.append({
                        "ticker": ticker, "asset": asset, "side": "no",
                        "entry": entry_no_mid, "max": max_no_mid,
                        "gain_pct": no_max_gain, "entry_cents": no_entry_cents,
                    })
                elif no_final_change < -0.10:
                    correct_passes.append({
                        "ticker": ticker, "asset": asset, "side": "no",
                        "entry": entry_no_mid, "loss_pct": no_final_change,
                    })

    missed_yes = sum(1 for m in missed_wins if m["side"] == "yes")
    missed_no = sum(1 for m in missed_wins if m["side"] == "no")
    logger.info(f"Missed opportunities: {len(missed_wins)} wins ({missed_yes} YES, {missed_no} NO), "
                f"{len(correct_passes)} correct passes")

    return {
        "missed_wins": missed_wins,
        "correct_passes": correct_passes,
    }


# ---------------------------------------------------------------------------
# Analysis 3: Actual trades
# ---------------------------------------------------------------------------
def analyze_trades() -> Dict[str, Any]:
    cutoff = get_cutoff_date()
    trades = query_db(
        "SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp",
        (cutoff,)
    )

    if not trades:
        return {"total_trades": 0}

    results = {
        "total_trades": len(trades),
        "by_strategy": defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0, "count": 0}),
        "by_asset": defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0, "count": 0}),
        "profitable_entry_prices": [],
        "losing_entry_prices": [],
    }

    # Edge accuracy tracking
    edge_predictions = []  # (predicted_edge, actual_pnl, predicted_prob)

    for t in trades:
        strategy = t.get("strategy", "unknown")
        asset = t.get("asset", "UNKNOWN")
        pnl = t.get("pnl")
        price = t.get("price", 0)
        edge = t.get("edge_cents")
        prob = t.get("predicted_prob")

        results["by_strategy"][strategy]["count"] += 1
        results["by_asset"][asset]["count"] += 1

        if pnl is not None and pnl != 0:
            results["by_strategy"][strategy]["total_pnl"] += pnl
            results["by_asset"][asset]["total_pnl"] += pnl
            if pnl > 0:
                results["by_strategy"][strategy]["wins"] += 1
                results["by_asset"][asset]["wins"] += 1
                results["profitable_entry_prices"].append(price)
            else:
                results["by_strategy"][strategy]["losses"] += 1
                results["by_asset"][asset]["losses"] += 1
                results["losing_entry_prices"].append(price)

        # Track edge prediction accuracy
        if edge is not None and pnl is not None:
            won = pnl > 0
            edge_predictions.append({
                "edge": edge, "pnl": pnl, "prob": prob,
                "won": won, "strategy": strategy, "asset": asset,
            })

    # Analyze edge accuracy
    if edge_predictions:
        total_predicted = len(edge_predictions)
        correct = sum(1 for e in edge_predictions if e["won"])
        accuracy = correct / total_predicted if total_predicted > 0 else 0

        # Does higher edge = higher win rate?
        high_edge = [e for e in edge_predictions if e["edge"] and e["edge"] > 20]
        low_edge = [e for e in edge_predictions if e["edge"] and e["edge"] <= 20]
        high_wr = sum(1 for e in high_edge if e["won"]) / len(high_edge) if high_edge else 0
        low_wr = sum(1 for e in low_edge if e["won"]) / len(low_edge) if low_edge else 0

        results["edge_accuracy"] = {
            "total_with_edge": total_predicted,
            "overall_win_rate": round(accuracy, 3),
            "high_edge_win_rate": round(high_wr, 3),
            "low_edge_win_rate": round(low_wr, 3),
            "avg_edge_winners": round(np.mean([e["edge"] for e in edge_predictions if e["won"]]), 1) if any(e["won"] for e in edge_predictions) else 0,
            "avg_edge_losers": round(np.mean([e["edge"] for e in edge_predictions if not e["won"]]), 1) if any(not e["won"] for e in edge_predictions) else 0,
        }
        logger.info(f"Edge accuracy: {accuracy:.0%} win rate ({correct}/{total_predicted})")
        logger.info(f"  High edge (>20¢): {high_wr:.0%} | Low edge (≤20¢): {low_wr:.0%}")
        logger.info(f"  Avg edge on winners: {results['edge_accuracy']['avg_edge_winners']}¢ | losers: {results['edge_accuracy']['avg_edge_losers']}¢")

    return results


# ---------------------------------------------------------------------------
# Analysis 4: Sentiment vs actual outcomes
# ---------------------------------------------------------------------------
def analyze_sentiment_effectiveness() -> Dict[str, Any]:
    """When sentiment was positive, did prices actually go up within the next hour?

    For each sentiment reading, finds the average contract mid-price in the
    30 minutes BEFORE vs 30-60 minutes AFTER. This measures whether the
    sentiment signal had any predictive value for short-term price direction.
    """
    cutoff = get_cutoff_date()

    sentiments = query_db(
        """SELECT asset, overall_sentiment, confidence, timestamp
           FROM news_sentiment WHERE timestamp >= ?
           ORDER BY timestamp""",
        (cutoff,)
    )

    if not sentiments:
        return {}

    snapshots = query_db(
        """SELECT asset, yes_bid, yes_ask, timestamp
           FROM market_snapshots WHERE timestamp >= ?
           ORDER BY asset, timestamp""",
        (cutoff,)
    )

    # Index by asset for efficient time-window lookups
    asset_prices = defaultdict(list)
    for s in snapshots:
        mid = ((s["yes_bid"] or 0) + (s["yes_ask"] or 0)) / 2 if (s["yes_bid"] or s["yes_ask"]) else 0
        if mid > 0:
            asset_prices[s["asset"]].append({"mid": mid, "ts": s["timestamp"]})

    correct_predictions = 0
    wrong_predictions = 0
    total_checked = 0

    # Deduplicate sentiment readings — only check once per asset per hour
    seen = set()

    for sent in sentiments:
        asset = sent["asset"]
        sentiment = sent["overall_sentiment"] or 0
        if abs(sentiment) < 0.05:  # Too neutral to judge
            continue

        sent_time = sent["timestamp"]
        # Dedupe key: asset + hour
        hour_key = f"{asset}_{sent_time[:13]}"
        if hour_key in seen:
            continue
        seen.add(hour_key)

        prices = asset_prices.get(asset, [])
        if len(prices) < 5:
            continue

        # Find prices in 30-min window BEFORE and 30-60 min AFTER this reading
        before_mids = [p["mid"] for p in prices if p["ts"] < sent_time and p["ts"] >= sent_time[:11]]
        after_mids = [p["mid"] for p in prices if p["ts"] > sent_time]

        # Use a limited after-window (take next 10 snapshots ≈ next hour)
        after_mids = after_mids[:10]

        if len(before_mids) < 2 or len(after_mids) < 2:
            continue

        avg_before = np.mean(before_mids)
        avg_after = np.mean(after_mids)
        price_moved_up = avg_after > avg_before

        total_checked += 1
        if (sentiment > 0 and price_moved_up) or (sentiment < 0 and not price_moved_up):
            correct_predictions += 1
        else:
            wrong_predictions += 1

    accuracy = correct_predictions / total_checked if total_checked > 0 else 0.5
    logger.info(f"Sentiment accuracy: {accuracy:.0%} ({correct_predictions}/{total_checked})")

    return {
        "accuracy": accuracy,
        "total_checked": total_checked,
        "correct": correct_predictions,
        "wrong": wrong_predictions,
    }


# ---------------------------------------------------------------------------
# Analysis 5: Spot price vs contract price correlation
# ---------------------------------------------------------------------------
def analyze_spot_contract_correlation() -> Dict[str, Any]:
    """When spot price moves, do contract prices follow?

    Compares crypto_prices movements with market_snapshots movements
    to learn how responsive Kalshi contracts are to spot moves.
    """
    cutoff = get_cutoff_date()

    spot_prices = query_db(
        """SELECT asset, price_usd, change_24h_pct, timestamp
           FROM crypto_prices WHERE timestamp >= ?
           ORDER BY asset, timestamp""",
        (cutoff,)
    )

    if len(spot_prices) < 10:
        logger.info(f"Only {len(spot_prices)} spot price records — skipping correlation analysis")
        return {}

    contract_snapshots = query_db(
        """SELECT asset, AVG(yes_bid) as avg_yes_bid, timestamp
           FROM market_snapshots WHERE timestamp >= ?
           GROUP BY asset, substr(timestamp, 1, 16)
           ORDER BY asset, timestamp""",
        (cutoff,)
    )

    if len(contract_snapshots) < 10:
        return {}

    # Group by asset
    from collections import defaultdict
    spot_by_asset = defaultdict(list)
    for s in spot_prices:
        spot_by_asset[s["asset"]].append(s)

    contract_by_asset = defaultdict(list)
    for c in contract_snapshots:
        contract_by_asset[c["asset"]].append(c)

    correlations = {}
    for asset in ["BTC", "ETH", "SOL"]:
        spots = spot_by_asset.get(asset, [])
        contracts = contract_by_asset.get(asset, [])
        if len(spots) < 5 or len(contracts) < 5:
            continue

        # Build time-indexed maps (keyed by minute: "YYYY-MM-DDTHH:MM")
        spot_by_minute = {}
        for s in spots:
            minute_key = s["timestamp"][:16]
            spot_by_minute[minute_key] = s["price_usd"]

        contract_by_minute = {}
        for c in contracts:
            minute_key = c["timestamp"][:16]
            contract_by_minute[minute_key] = c["avg_yes_bid"]

        # Find overlapping minutes and compute changes
        common_minutes = sorted(set(spot_by_minute.keys()) & set(contract_by_minute.keys()))
        if len(common_minutes) < 5:
            continue

        spot_changes = []
        contract_changes = []
        for i in range(1, len(common_minutes)):
            prev_m = common_minutes[i - 1]
            curr_m = common_minutes[i]

            sp = spot_by_minute[prev_m]
            sc_val = spot_by_minute[curr_m]
            cp = contract_by_minute[prev_m]
            cc_val = contract_by_minute[curr_m]

            if sp and sc_val and sp > 0 and cp and cc_val and cp > 0:
                spot_changes.append((sc_val - sp) / sp)
                contract_changes.append((cc_val - cp) / cp)

        if len(spot_changes) >= 5:
            sc = np.array(spot_changes)
            cc = np.array(contract_changes)
            corr = float(np.corrcoef(sc, cc)[0, 1]) if np.std(sc) > 0 and np.std(cc) > 0 else 0
            correlations[asset] = {
                "correlation": round(corr, 3),
                "spot_avg_move": round(float(np.mean(np.abs(sc))), 4),
                "contract_avg_move": round(float(np.mean(np.abs(cc))), 4),
                "data_points": len(spot_changes),
            }
            logger.info(f"{asset} spot-contract correlation: {corr:.3f} ({len(spot_changes)} data points)")

    return {"correlations": correlations}


# ---------------------------------------------------------------------------
# Analysis 6: Yesterday's prediction accuracy
# ---------------------------------------------------------------------------
def analyze_yesterday_accuracy() -> Dict[str, Any]:
    """Score how well our model predicted yesterday's 15M contract outcomes.

    For every trade decision we recorded yesterday, compare the predicted
    probability against what actually happened (did the contract settle YES
    or NO?). Returns per-asset and per-signal accuracy so we can tune
    momentum weight, vol weight, and asset weights based on recent performance.
    """
    start, end = get_yesterday_range()

    # Get all trade decisions from yesterday
    decisions = query_db(
        """SELECT market_ticker, asset, strategy, direction, confidence, timestamp
           FROM trade_decisions WHERE timestamp >= ? AND timestamp < ?""",
        (start, end)
    )

    if not decisions:
        logger.info("No trade decisions yesterday — skipping accuracy analysis")
        return {}

    # Get market snapshots to determine outcomes
    # For 15M contracts: look at last snapshot (near expiry) to see final price
    snapshots = query_db(
        """SELECT ticker, yes_bid, yes_ask, no_bid, no_ask, spot_price,
                  strike_price, timestamp
           FROM market_snapshots WHERE timestamp >= ? AND timestamp < ?
           ORDER BY ticker, timestamp""",
        (start, end)
    )

    # Group snapshots by ticker — first and last give us entry state vs outcome
    by_ticker = defaultdict(list)
    for s in snapshots:
        by_ticker[s["ticker"]].append(s)

    # Get actual trades and their P&L
    trades = query_db(
        """SELECT market_ticker, asset, strategy, side, pnl, price, timestamp
           FROM trades WHERE timestamp >= ? AND timestamp < ?""",
        (start, end)
    )

    # Score predictions
    results = {
        "total_decisions": len(decisions),
        "total_trades": len(trades),
        "by_asset": defaultdict(lambda: {
            "correct": 0, "wrong": 0, "total_pnl": 0.0, "trades": 0
        }),
        "by_strategy": defaultdict(lambda: {
            "correct": 0, "wrong": 0, "total_pnl": 0.0, "trades": 0
        }),
        "momentum_accuracy": {"correct": 0, "wrong": 0},
        "overall": {"correct": 0, "wrong": 0},
    }

    # Score each trade by its P&L (direct measure of prediction accuracy)
    for t in trades:
        asset = t.get("asset", "UNKNOWN")
        strategy = t.get("strategy", "unknown")
        pnl = t.get("pnl") or 0

        results["by_asset"][asset]["trades"] += 1
        results["by_asset"][asset]["total_pnl"] += pnl
        results["by_strategy"][strategy]["trades"] += 1
        results["by_strategy"][strategy]["total_pnl"] += pnl

        if pnl > 0:
            results["by_asset"][asset]["correct"] += 1
            results["by_strategy"][strategy]["correct"] += 1
            results["overall"]["correct"] += 1
        elif pnl < 0:
            results["by_asset"][asset]["wrong"] += 1
            results["by_strategy"][strategy]["wrong"] += 1
            results["overall"]["wrong"] += 1

    # Score prediction accuracy by checking if 15M contracts resolved as predicted
    for ticker, snaps in by_ticker.items():
        if "15M" not in ticker.upper() or len(snaps) < 2:
            continue

        first = snaps[0]
        last = snaps[-1]

        strike = first.get("strike_price") or 0
        first_spot = first.get("spot_price") or 0
        last_spot = last.get("spot_price") or 0

        if not (strike > 0 and first_spot > 0 and last_spot > 0):
            continue

        # Did spot end above or below strike?
        resolved_above = last_spot > strike
        # Was spot moving up during the contract?
        spot_went_up = last_spot > first_spot

        # Check if momentum direction matched outcome
        results["momentum_accuracy"]["correct" if spot_went_up == resolved_above else "wrong"] += 1

    # Compute summary stats
    overall = results["overall"]
    total = overall["correct"] + overall["wrong"]
    results["win_rate"] = overall["correct"] / total if total > 0 else 0.5

    mom = results["momentum_accuracy"]
    mom_total = mom["correct"] + mom["wrong"]
    results["momentum_win_rate"] = mom["correct"] / mom_total if mom_total > 0 else 0.5

    logger.info(f"Yesterday: {total} trades, win_rate={results['win_rate']:.0%}, "
                f"momentum_accuracy={results['momentum_win_rate']:.0%}")
    for asset, stats in results["by_asset"].items():
        t = stats["correct"] + stats["wrong"]
        wr = stats["correct"] / t if t > 0 else 0
        logger.info(f"  {asset}: {t} trades, wr={wr:.0%}, pnl=${stats['total_pnl']:.2f}")

    return results


# ---------------------------------------------------------------------------
# Compute optimal params from ALL analyses
# ---------------------------------------------------------------------------
def compute_optimal_params(
    price_analysis: Dict,
    missed_analysis: Dict,
    trade_analysis: Dict,
    sentiment_analysis: Dict,
    spot_correlation: Dict = None,
    yesterday_accuracy: Dict = None,
) -> Dict[str, Any]:
    params = DEFAULT_PARAMS.copy()
    params["trained_at"] = datetime.now(timezone.utc).isoformat()
    params["version"] = load_current_params().get("version", 0) + 1

    total_markets = price_analysis.get("total_markets_observed", 0)
    total_trades = trade_analysis.get("total_trades", 0)
    params["data_points"] = total_markets

    if total_markets < 10:
        logger.info(f"Only {total_markets} market observations — using defaults")
        return params

    # --- Learn entry price range from ALL market movements ---
    price_range_outcomes = price_analysis.get("price_range_outcomes", {})
    best_ranges = []
    for range_key, stats in price_range_outcomes.items():
        total = stats["up"] + stats["down"] + stats["flat"]
        if total < 3:
            continue
        win_rate = stats["up"] / total
        avg_move = np.mean(stats["avg_move"]) if stats["avg_move"] else 0
        best_ranges.append((range_key, win_rate, avg_move, total))

    if best_ranges:
        # Sort by win rate, find the ranges with best outcomes
        best_ranges.sort(key=lambda x: x[1], reverse=True)
        logger.info("Price range outcomes:")
        for r, wr, am, n in best_ranges:
            logger.info(f"  {r}¢: win_rate={wr:.0%}, avg_move={am:.1%}, n={n}")

        # Set entry range to cover the top-performing price ranges
        good_ranges = [r for r, wr, am, n in best_ranges if wr > 0.40 and n >= 3]
        if good_ranges:
            # Parse range strings like "20-30" to get min/max
            all_mins = []
            all_maxs = []
            for r in good_ranges:
                parts = r.split("-")
                all_mins.append(int(parts[0]))
                all_maxs.append(int(parts[1]))
            params["min_entry_price_cents"] = max(min(all_mins), 5)
            params["max_entry_price_cents"] = min(max(all_maxs), 95)
            logger.info(f"Learned entry range: {params['min_entry_price_cents']}-{params['max_entry_price_cents']}¢")

    # --- Log distance-to-strike outcomes (new) ---
    distance_outcomes = price_analysis.get("distance_outcomes", {})
    if distance_outcomes:
        logger.info("Distance-to-strike outcomes:")
        for dist_key in ["above_5pct", "above_0-5pct", "below_0-5pct", "below_5pct"]:
            stats = distance_outcomes.get(dist_key)
            if not stats:
                continue
            total = stats["up"] + stats["down"] + stats["flat"]
            if total < 5:
                continue
            wr = stats["up"] / total
            avg = np.mean(stats["avg_move"]) if stats["avg_move"] else 0
            logger.info(f"  {dist_key}: win_rate={wr:.0%}, avg_move={avg:.2%}, n={total}")
        params["distance_to_strike_outcomes"] = {
            k: {"win_rate": round(v["up"] / max(v["up"] + v["down"] + v["flat"], 1), 3),
                "avg_move": round(float(np.mean(v["avg_move"])) if v["avg_move"] else 0, 4),
                "n": v["up"] + v["down"] + v["flat"]}
            for k, v in distance_outcomes.items()
            if v["up"] + v["down"] + v["flat"] >= 5
        }

    # --- Log time-to-expiry outcomes (new) ---
    tte_outcomes = price_analysis.get("time_to_expiry_outcomes", {})
    if tte_outcomes:
        logger.info("Time-to-expiry outcomes:")
        for tte_key in ["<30min", "30min-2h", "2h-12h", "12h-2d", ">2d"]:
            stats = tte_outcomes.get(tte_key)
            if not stats:
                continue
            total = stats["up"] + stats["down"] + stats["flat"]
            if total < 5:
                continue
            wr = stats["up"] / total
            avg = np.mean(stats["avg_move"]) if stats["avg_move"] else 0
            logger.info(f"  {tte_key}: win_rate={wr:.0%}, avg_move={avg:.2%}, n={total}")
        params["time_to_expiry_outcomes"] = {
            k: {"win_rate": round(v["up"] / max(v["up"] + v["down"] + v["flat"], 1), 3),
                "avg_move": round(float(np.mean(v["avg_move"])) if v["avg_move"] else 0, 4),
                "n": v["up"] + v["down"] + v["flat"]}
            for k, v in tte_outcomes.items()
            if v["up"] + v["down"] + v["flat"] >= 5
        }

    # --- Learn from missed opportunities ---
    missed_wins = missed_analysis.get("missed_wins", [])
    if missed_wins:
        # What entry prices did we miss that would have been profitable?
        missed_entry_prices = [m["entry_cents"] for m in missed_wins]
        missed_assets = defaultdict(int)
        for m in missed_wins:
            missed_assets[m["asset"]] += 1

        logger.info(f"Missed {len(missed_wins)} profitable opportunities:")
        for asset, count in missed_assets.items():
            logger.info(f"  {asset}: {count} missed wins")

        # If we're missing wins in a price range, expand our entry range
        if missed_entry_prices:
            p25 = int(np.percentile(missed_entry_prices, 25))
            p75 = int(np.percentile(missed_entry_prices, 75))
            # Nudge our range toward where we're missing wins
            current_min = params["min_entry_price_cents"]
            current_max = params["max_entry_price_cents"]
            params["min_entry_price_cents"] = max(min(current_min, p25), 5)
            params["max_entry_price_cents"] = min(max(current_max, p75), 95)

        # Boost asset weights for assets with more missed wins
        total_missed = sum(missed_assets.values())
        if total_missed > 0:
            for asset, count in missed_assets.items():
                # More missed wins = we should trade this asset more
                boost = 1.0 + (count / total_missed)
                params["asset_weights"][asset] = round(min(boost, 2.0), 2)
                logger.info(f"  Boosting {asset} weight to {params['asset_weights'][asset]}")

    # --- Learn from actual trades ---
    if total_trades >= 3:
        by_strategy = trade_analysis.get("by_strategy", {})
        for strategy, stats in by_strategy.items():
            count = stats["count"]
            if count < 2:
                continue
            win_rate = stats["wins"] / count if count > 0 else 0.5
            avg_pnl = stats["total_pnl"] / count if count > 0 else 0

            weight = min(max(win_rate * 2, 0.2), 3.0)
            if avg_pnl > 0:
                weight *= 1.2
            elif avg_pnl < 0:
                weight *= 0.8

            params["strategy_weights"][strategy] = round(float(weight), 2)
            logger.info(f"Strategy {strategy}: wr={win_rate:.0%}, pnl=${avg_pnl:.2f}, weight={weight:.2f}")

        # Learn take-profit from actual profitable trades
        profitable_prices = trade_analysis.get("profitable_entry_prices", [])
        losing_prices = trade_analysis.get("losing_entry_prices", [])

        by_asset = trade_analysis.get("by_asset", {})
        for asset, stats in by_asset.items():
            count = stats["count"]
            if count < 2:
                continue
            win_rate = stats["wins"] / count
            weight = min(max(win_rate * 2, 0.3), 3.0)
            # Blend with missed-opportunity weight
            existing = params["asset_weights"].get(asset, 1.0)
            params["asset_weights"][asset] = round((existing + weight) / 2, 2)

    # --- Learn from sentiment accuracy ---
    if sentiment_analysis.get("total_checked", 0) >= 5:
        accuracy = sentiment_analysis["accuracy"]
        if accuracy > 0.60:
            # Sentiment is predictive — lower threshold to use it more
            params["sentiment_threshold"] = 0.3
            params["min_sentiment_confidence"] = 0.15
            params["strategy_weights"]["news_sentiment"] = round(min(accuracy * 2, 2.5), 2)
            logger.info(f"Sentiment is predictive ({accuracy:.0%}) — lowering threshold")
        elif accuracy < 0.40:
            # Sentiment is counter-predictive — either flip it or reduce weight
            params["strategy_weights"]["news_sentiment"] = 0.3
            logger.info(f"Sentiment is counter-predictive ({accuracy:.0%}) — reducing weight")
        else:
            params["strategy_weights"]["news_sentiment"] = 0.8
            logger.info(f"Sentiment is neutral ({accuracy:.0%}) — slightly reducing weight")

    # --- Learn optimal exit thresholds from price volatility ---
    asset_outcomes = price_analysis.get("asset_outcomes", {})
    all_moves = []
    for asset, stats in asset_outcomes.items():
        all_moves.extend(stats.get("avg_move", []))

    if all_moves and len(all_moves) > 20:
        pos_moves = [m for m in all_moves if m > 0]
        neg_moves = [abs(m) for m in all_moves if m < 0]

        if pos_moves:
            # Learn trailing params from positive moves
            # Breakeven trigger: 25th percentile (move stop to entry early)
            be = min(max(float(np.percentile(pos_moves, 25)), 0.08), 0.25)
            params["breakeven_trigger"] = round(be, 2)
            # Trail trigger: 50th percentile (start trailing at typical upside)
            tt = min(max(float(np.percentile(pos_moves, 50)), 0.15), 0.50)
            params["trail_trigger"] = round(tt, 2)
            # Trail distance: tighter when moves are consistent
            spread = float(np.std(pos_moves))
            tp = min(max(spread * 1.5, 0.10), 0.35)
            params["trail_pct"] = round(tp, 2)
            logger.info(f"Learned trailing: breakeven@{be:.0%}, trail@{tt:.0%}, trail_pct={tp:.0%} "
                        f"(from {len(pos_moves)} positive moves)")

        if neg_moves:
            # Set stop-loss tight — kill losers fast
            sl = min(max(float(np.percentile(neg_moves, 40)), 0.05), 0.20)
            params["stop_loss_pct"] = round(sl, 2)
            logger.info(f"Learned stop-loss: {sl:.0%} (from {len(neg_moves)} negative moves)")

    # --- Buy/sell thresholds from where price movements are strongest ---
    # If prices below 40¢ tend to go up, lower the buy_yes_below threshold
    for range_key, stats in price_range_outcomes.items():
        parts = range_key.split("-")
        low = int(parts[0])
        total = stats["up"] + stats["down"] + stats["flat"]
        if total < 5:
            continue
        win_rate = stats["up"] / total

        if low < 50 and win_rate > 0.55:
            # This low range has good upward movement — expand YES buying
            params["buy_yes_below"] = max(params["buy_yes_below"], low + 10)
        if low >= 50 and win_rate < 0.45:
            # This high range has downward pressure — expand NO buying
            params["buy_no_above"] = min(params["buy_no_above"], low)

    # --- Learn from spot-contract correlations ---
    if spot_correlation and spot_correlation.get("correlations"):
        corrs = spot_correlation["correlations"]
        # If spot price strongly correlates with contracts, boost assets where
        # contracts lag behind spot moves (arbitrage opportunity)
        for asset, data in corrs.items():
            corr = data.get("correlation", 0)
            spot_move = data.get("spot_avg_move", 0)
            contract_move = data.get("contract_avg_move", 0)

            if corr > 0.5 and spot_move > contract_move * 1.5:
                # Spot moves faster than contracts — opportunity to front-run
                current_weight = params["asset_weights"].get(asset, 1.0)
                params["asset_weights"][asset] = round(min(current_weight * 1.3, 3.0), 2)
                logger.info(f"{asset}: high spot-contract correlation ({corr:.2f}) with "
                           f"spot leading — boosting weight to {params['asset_weights'][asset]}")

        params["spot_correlations"] = {a: d["correlation"] for a, d in corrs.items()}

    # --- Daily recalibration from yesterday's prediction accuracy ---
    if yesterday_accuracy and yesterday_accuracy.get("total_trades", 0) >= 2:
        yesterday_wr = yesterday_accuracy.get("win_rate", 0.5)
        mom_wr = yesterday_accuracy.get("momentum_win_rate", 0.5)

        # If yesterday's overall win rate was bad, tighten stop loss and raise min edge
        if yesterday_wr < 0.40:
            params["stop_loss_pct"] = max(round(params["stop_loss_pct"] * 0.8, 2), 0.05)
            params["min_edge"] = min(round(params.get("min_edge", 0.05) * 1.3, 2), 0.15)
            logger.info(f"Yesterday win_rate={yesterday_wr:.0%} — tightening SL to "
                        f"{params['stop_loss_pct']:.0%}, min_edge to {params['min_edge']}")
        elif yesterday_wr > 0.65:
            # Winning streak — slightly loosen to capture more trades
            params["min_edge"] = max(round(params.get("min_edge", 0.05) * 0.85, 2), 0.03)
            logger.info(f"Yesterday win_rate={yesterday_wr:.0%} — loosening min_edge to {params['min_edge']}")

        # Tune momentum weight based on yesterday's momentum accuracy
        # This scales the ±15% cap in price_predictor.compute_momentum
        current_mom_weight = params.get("momentum_weight", 1.0)
        if mom_wr > 0.60:
            # Momentum was predictive — trust it more
            params["momentum_weight"] = round(min(current_mom_weight * 1.2, 2.0), 2)
            logger.info(f"Momentum accuracy={mom_wr:.0%} — boosting weight to {params['momentum_weight']}")
        elif mom_wr < 0.40:
            # Momentum was counter-predictive — dampen it
            params["momentum_weight"] = round(max(current_mom_weight * 0.6, 0.2), 2)
            logger.info(f"Momentum accuracy={mom_wr:.0%} — reducing weight to {params['momentum_weight']}")
        else:
            params["momentum_weight"] = round(current_mom_weight, 2)

        # Per-asset tuning: scale asset weights by yesterday's per-asset performance
        by_asset = yesterday_accuracy.get("by_asset", {})
        for asset, stats in by_asset.items():
            t = stats["correct"] + stats["wrong"]
            if t < 2:
                continue
            asset_wr = stats["correct"] / t
            current_w = params["asset_weights"].get(asset, 1.0)
            if asset_wr > 0.60:
                params["asset_weights"][asset] = round(min(current_w * 1.15, 3.0), 2)
            elif asset_wr < 0.35:
                params["asset_weights"][asset] = round(max(current_w * 0.7, 0.2), 2)
            logger.info(f"  {asset}: yesterday wr={asset_wr:.0%} → weight={params['asset_weights'][asset]}")

        # Per-strategy tuning
        by_strategy = yesterday_accuracy.get("by_strategy", {})
        for strategy, stats in by_strategy.items():
            t = stats["correct"] + stats["wrong"]
            if t < 2:
                continue
            strat_wr = stats["correct"] / t
            current_w = params["strategy_weights"].get(strategy, 1.0)
            if strat_wr > 0.60:
                params["strategy_weights"][strategy] = round(min(current_w * 1.2, 3.0), 2)
            elif strat_wr < 0.35:
                params["strategy_weights"][strategy] = round(max(current_w * 0.6, 0.2), 2)
            logger.info(f"  {strategy}: yesterday wr={strat_wr:.0%} → weight={params['strategy_weights'][strategy]}")

        params["yesterday_accuracy"] = {
            "win_rate": round(yesterday_wr, 3),
            "momentum_accuracy": round(mom_wr, 3),
            "total_trades": yesterday_accuracy["total_trades"],
        }

    return params


def retrain():
    """Main retraining pipeline — analyzes ALL data, not just trades."""
    logger.info("=" * 60)
    logger.info("DAILY MODEL RETRAIN — Analyzing all market data")
    logger.info(f"Database: {DB_PATH}")
    logger.info("=" * 60)

    if not os.path.exists(DB_PATH):
        logger.warning(f"Database not found at {DB_PATH} — saving defaults")
        save_params(DEFAULT_PARAMS)
        return DEFAULT_PARAMS

    # Run all analyses
    logger.info("--- Analyzing price movements across ALL markets ---")
    price_analysis = analyze_price_movements()

    logger.info("--- Analyzing missed opportunities ---")
    missed_analysis = analyze_missed_opportunities()

    logger.info("--- Analyzing actual trades ---")
    trade_analysis = analyze_trades()

    logger.info("--- Analyzing sentiment effectiveness ---")
    sentiment_analysis = analyze_sentiment_effectiveness()

    logger.info("--- Analyzing spot price vs contract correlation ---")
    spot_correlation = analyze_spot_contract_correlation()

    logger.info("--- Analyzing yesterday's prediction accuracy ---")
    yesterday_accuracy = analyze_yesterday_accuracy()

    # Summary
    logger.info(f"\nSummary:")
    logger.info(f"  Markets observed: {price_analysis.get('total_markets_observed', 0)}")
    logger.info(f"  Missed wins: {len(missed_analysis.get('missed_wins', []))}")
    logger.info(f"  Correct passes: {len(missed_analysis.get('correct_passes', []))}")
    logger.info(f"  Trades made: {trade_analysis.get('total_trades', 0)}")
    logger.info(f"  Sentiment accuracy: {sentiment_analysis.get('accuracy', 'N/A')}")
    logger.info(f"  Spot correlations: {spot_correlation.get('correlations', {})}")
    logger.info(f"  Yesterday win rate: {yesterday_accuracy.get('win_rate', 'N/A')}")
    logger.info(f"  Yesterday momentum accuracy: {yesterday_accuracy.get('momentum_win_rate', 'N/A')}")

    # Compute and save
    new_params = compute_optimal_params(
        price_analysis, missed_analysis, trade_analysis, sentiment_analysis,
        spot_correlation, yesterday_accuracy,
    )
    save_params(new_params)

    logger.info(f"\nRetrain complete — v{new_params['version']}")
    logger.info(f"Entry: {new_params['min_entry_price_cents']}-{new_params['max_entry_price_cents']}¢")
    logger.info(f"SL: {new_params['stop_loss_pct']:.0%}, "
                f"BE@{new_params.get('breakeven_trigger', 0.15):.0%}, "
                f"Trail@{new_params.get('trail_trigger', 0.25):.0%} ({new_params.get('trail_pct', 0.20):.0%})")
    logger.info(f"Momentum weight: {new_params.get('momentum_weight', 1.0)}")
    logger.info(f"Min edge: {new_params.get('min_edge', 0.05)}")
    logger.info(f"Strategies: {new_params['strategy_weights']}")
    logger.info(f"Assets: {new_params['asset_weights']}")

    return new_params


if __name__ == "__main__":
    retrain()
