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
    "take_profit_pct": 0.50,
    "stop_loss_pct": 0.30,
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
        """SELECT ticker, asset, yes_bid, yes_ask, no_bid, no_ask, timestamp
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

    for t in trades:
        strategy = t.get("strategy", "unknown")
        asset = t.get("asset", "UNKNOWN")
        pnl = t.get("pnl")
        price = t.get("price", 0)

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
# Compute optimal params from ALL analyses
# ---------------------------------------------------------------------------
def compute_optimal_params(
    price_analysis: Dict,
    missed_analysis: Dict,
    trade_analysis: Dict,
    sentiment_analysis: Dict,
    spot_correlation: Dict = None,
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
            # Set take-profit at the 75th percentile of positive moves
            tp = min(max(float(np.percentile(pos_moves, 75)), 0.15), 0.80)
            params["take_profit_pct"] = round(tp, 2)
            logger.info(f"Learned take-profit: {tp:.0%} (from {len(pos_moves)} positive moves)")

        if neg_moves:
            # Set stop-loss at the 60th percentile of negative moves
            sl = min(max(float(np.percentile(neg_moves, 60)), 0.10), 0.50)
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

    # Summary
    logger.info(f"\nSummary:")
    logger.info(f"  Markets observed: {price_analysis.get('total_markets_observed', 0)}")
    logger.info(f"  Missed wins: {len(missed_analysis.get('missed_wins', []))}")
    logger.info(f"  Correct passes: {len(missed_analysis.get('correct_passes', []))}")
    logger.info(f"  Trades made: {trade_analysis.get('total_trades', 0)}")
    logger.info(f"  Sentiment accuracy: {sentiment_analysis.get('accuracy', 'N/A')}")
    logger.info(f"  Spot correlations: {spot_correlation.get('correlations', {})}")

    # Compute and save
    new_params = compute_optimal_params(
        price_analysis, missed_analysis, trade_analysis, sentiment_analysis, spot_correlation
    )
    save_params(new_params)

    logger.info(f"\nRetrain complete — v{new_params['version']}")
    logger.info(f"Entry: {new_params['min_entry_price_cents']}-{new_params['max_entry_price_cents']}¢")
    logger.info(f"TP: {new_params['take_profit_pct']:.0%}, SL: {new_params['stop_loss_pct']:.0%}")
    logger.info(f"Strategies: {new_params['strategy_weights']}")
    logger.info(f"Assets: {new_params['asset_weights']}")

    return new_params


if __name__ == "__main__":
    retrain()
