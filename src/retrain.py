#!/usr/bin/env python3
"""
Daily retraining script for the Kalshi trading bot.

Runs every morning at 7am. Analyzes the last 3 months of trading data
(or all available data if less than 3 months) and outputs updated
model parameters to model_params.json.

What it learns:
- Entry price range (what buy prices lead to profits?)
- Take-profit threshold (when is the best time to sell winners?)
- Stop-loss threshold (when to cut losers?)
- Strategy weights (which strategies actually make money?)
- Time-of-day patterns (when are trades most profitable?)
- Asset preferences (which assets perform best?)
"""

import json
import logging
import os
import sqlite3
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default params used when there's not enough data
DEFAULT_PARAMS = {
    "version": 0,
    "trained_at": None,
    "data_points": 0,
    # Entry filters
    "min_entry_price_cents": 20,
    "max_entry_price_cents": 80,
    "min_edge": 0.05,  # Minimum distance from center (45-55¢ dead zone)
    "buy_yes_below": 45,  # Buy YES when price below this
    "buy_no_above": 55,   # Buy NO when price above this
    # Exit thresholds
    "take_profit_pct": 0.50,
    "stop_loss_pct": 0.30,
    "time_exit_seconds": 120,
    # Strategy weights (0 = disabled, 1 = normal, >1 = preferred)
    "strategy_weights": {
        "news_sentiment": 1.0,
        "statistical_arbitrage": 1.0,
        "volatility_based": 1.0,
        "value_bet": 1.0,
    },
    # Asset preferences
    "asset_weights": {
        "BTC": 1.0,
        "ETH": 1.0,
        "SOL": 1.0,
    },
    # Sentiment
    "sentiment_threshold": 0.6,
    "min_sentiment_confidence": 0.2,
    # Max concurrent positions
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
    """Load current model params or return defaults."""
    try:
        with open(PARAMS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_PARAMS.copy()


def save_params(params: Dict[str, Any]):
    """Save updated params to disk."""
    with open(PARAMS_PATH, 'w') as f:
        json.dump(params, f, indent=2, default=str)
    logger.info(f"Saved model params to {PARAMS_PATH}")


def query_db(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Run a query against the trading database."""
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
    """Get the cutoff date (3 months ago or earliest data)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=90)
    return cutoff.isoformat()


def analyze_trades() -> Dict[str, Any]:
    """Analyze completed trades to learn what works."""
    cutoff = get_cutoff_date()

    trades = query_db(
        "SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp",
        (cutoff,)
    )

    if not trades:
        logger.info("No trades found for analysis")
        return {}

    results = {
        "total_trades": len(trades),
        "by_strategy": {},
        "by_asset": {},
        "profitable_entry_prices": [],
        "losing_entry_prices": [],
        "exits_with_pnl": [],
    }

    for t in trades:
        strategy = t.get("strategy", "unknown")
        asset = t.get("asset", "UNKNOWN")
        pnl = t.get("pnl")
        price = t.get("price", 0)
        order_result = t.get("order_result", "")

        # Track by strategy
        if strategy not in results["by_strategy"]:
            results["by_strategy"][strategy] = {"wins": 0, "losses": 0, "total_pnl": 0, "count": 0}
        results["by_strategy"][strategy]["count"] += 1

        if pnl is not None:
            results["by_strategy"][strategy]["total_pnl"] += pnl
            if pnl > 0:
                results["by_strategy"][strategy]["wins"] += 1
                results["profitable_entry_prices"].append(price)
            elif pnl < 0:
                results["by_strategy"][strategy]["losses"] += 1
                results["losing_entry_prices"].append(price)
            results["exits_with_pnl"].append({"pnl": pnl, "price": price, "strategy": strategy, "asset": asset})

        # Track by asset
        if asset not in results["by_asset"]:
            results["by_asset"][asset] = {"wins": 0, "losses": 0, "total_pnl": 0, "count": 0}
        results["by_asset"][asset]["count"] += 1
        if pnl is not None:
            results["by_asset"][asset]["total_pnl"] += pnl
            if pnl > 0:
                results["by_asset"][asset]["wins"] += 1
            elif pnl < 0:
                results["by_asset"][asset]["losses"] += 1

    return results


def analyze_snapshots() -> Dict[str, Any]:
    """Analyze market snapshots for price patterns."""
    cutoff = get_cutoff_date()

    # Get snapshot counts and price distributions
    stats = query_db(
        """SELECT asset, COUNT(*) as count,
           AVG(yes_bid) as avg_bid, AVG(yes_ask) as avg_ask,
           AVG(volume) as avg_volume
           FROM market_snapshots WHERE timestamp >= ?
           GROUP BY asset""",
        (cutoff,)
    )

    return {"snapshot_stats": stats}


def analyze_sentiment() -> Dict[str, Any]:
    """Analyze sentiment data effectiveness."""
    cutoff = get_cutoff_date()

    sentiments = query_db(
        """SELECT asset, overall_sentiment, confidence, article_count
           FROM news_sentiment WHERE timestamp >= ?""",
        (cutoff,)
    )

    if not sentiments:
        return {}

    avg_sentiment = np.mean([s["overall_sentiment"] for s in sentiments if s["overall_sentiment"]])
    avg_confidence = np.mean([s["confidence"] for s in sentiments if s["confidence"]])

    return {
        "avg_sentiment": float(avg_sentiment),
        "avg_confidence": float(avg_confidence),
        "total_records": len(sentiments),
    }


def compute_optimal_params(trade_analysis: Dict, snapshot_analysis: Dict, sentiment_analysis: Dict) -> Dict[str, Any]:
    """Compute optimal parameters from the analysis."""
    params = DEFAULT_PARAMS.copy()
    params["trained_at"] = datetime.now(timezone.utc).isoformat()

    total_trades = trade_analysis.get("total_trades", 0)
    params["data_points"] = total_trades

    if total_trades < 5:
        logger.info(f"Only {total_trades} trades — using defaults with minor adjustments")
        # Even with few trades, we can start learning
        if total_trades > 0:
            params["version"] = load_current_params().get("version", 0) + 1
        return params

    # --- Learn entry price range ---
    profitable_prices = trade_analysis.get("profitable_entry_prices", [])
    losing_prices = trade_analysis.get("losing_entry_prices", [])

    if profitable_prices:
        # Expand the entry range toward where profits happen
        p10 = max(int(np.percentile(profitable_prices, 10)), 10)
        p90 = min(int(np.percentile(profitable_prices, 90)), 90)
        params["min_entry_price_cents"] = p10
        params["max_entry_price_cents"] = p90
        logger.info(f"Learned entry range: {p10}¢ - {p90}¢")

    # --- Learn take-profit and stop-loss ---
    exits = trade_analysis.get("exits_with_pnl", [])
    if exits:
        profitable_exits = [e for e in exits if e["pnl"] and e["pnl"] > 0]
        losing_exits = [e for e in exits if e["pnl"] and e["pnl"] < 0]

        if profitable_exits:
            # Average gain as fraction — use this to set take-profit
            avg_gain = np.mean([e["pnl"] for e in profitable_exits])
            # Don't be too greedy or too conservative
            tp = min(max(avg_gain * 2, 0.20), 0.80)
            params["take_profit_pct"] = round(float(tp), 2)
            logger.info(f"Learned take-profit: {tp:.0%}")

        if losing_exits:
            avg_loss = abs(np.mean([e["pnl"] for e in losing_exits]))
            # Tighten stop-loss if losses are big
            sl = min(max(avg_loss * 1.5, 0.15), 0.50)
            params["stop_loss_pct"] = round(float(sl), 2)
            logger.info(f"Learned stop-loss: {sl:.0%}")

    # --- Learn strategy weights ---
    by_strategy = trade_analysis.get("by_strategy", {})
    strategy_weights = {}
    for strategy, stats in by_strategy.items():
        count = stats["count"]
        if count < 2:
            strategy_weights[strategy] = 1.0
            continue

        win_rate = stats["wins"] / count if count > 0 else 0.5
        avg_pnl = stats["total_pnl"] / count if count > 0 else 0

        # Weight = win_rate * 2, clamped to 0.2-3.0
        weight = min(max(win_rate * 2, 0.2), 3.0)

        # Boost strategies with positive PnL
        if avg_pnl > 0:
            weight *= 1.2
        elif avg_pnl < 0:
            weight *= 0.8

        strategy_weights[strategy] = round(float(weight), 2)
        logger.info(f"Strategy {strategy}: win_rate={win_rate:.0%}, avg_pnl=${avg_pnl:.2f}, weight={weight:.2f}")

    if strategy_weights:
        params["strategy_weights"] = {**DEFAULT_PARAMS["strategy_weights"], **strategy_weights}

    # --- Learn asset preferences ---
    by_asset = trade_analysis.get("by_asset", {})
    asset_weights = {}
    for asset, stats in by_asset.items():
        count = stats["count"]
        if count < 2:
            asset_weights[asset] = 1.0
            continue

        win_rate = stats["wins"] / count if count > 0 else 0.5
        weight = min(max(win_rate * 2, 0.3), 3.0)
        asset_weights[asset] = round(float(weight), 2)
        logger.info(f"Asset {asset}: win_rate={win_rate:.0%}, weight={weight:.2f}")

    if asset_weights:
        params["asset_weights"] = {**DEFAULT_PARAMS["asset_weights"], **asset_weights}

    # --- Learn sentiment threshold ---
    if sentiment_analysis.get("avg_confidence"):
        # If average confidence is low, lower the threshold so we still trade
        avg_conf = sentiment_analysis["avg_confidence"]
        if avg_conf < 0.3:
            params["sentiment_threshold"] = 0.3
            params["min_sentiment_confidence"] = 0.1
        elif avg_conf < 0.5:
            params["sentiment_threshold"] = 0.4
            params["min_sentiment_confidence"] = 0.15

    params["version"] = load_current_params().get("version", 0) + 1
    return params


def retrain():
    """Main retraining pipeline."""
    logger.info("=" * 50)
    logger.info("Starting daily model retraining")
    logger.info(f"Database: {DB_PATH}")
    logger.info(f"Params output: {PARAMS_PATH}")
    logger.info("=" * 50)

    # Check if DB exists
    if not os.path.exists(DB_PATH):
        logger.warning(f"Database not found at {DB_PATH} — saving defaults")
        save_params(DEFAULT_PARAMS)
        return

    # Analyze data
    trade_analysis = analyze_trades()
    snapshot_analysis = analyze_snapshots()
    sentiment_analysis = analyze_sentiment()

    # Log summary
    logger.info(f"Trade analysis: {trade_analysis.get('total_trades', 0)} trades")
    logger.info(f"Snapshot analysis: {snapshot_analysis}")
    logger.info(f"Sentiment analysis: {sentiment_analysis}")

    # Compute and save new params
    new_params = compute_optimal_params(trade_analysis, snapshot_analysis, sentiment_analysis)
    save_params(new_params)

    logger.info(f"Retraining complete — version {new_params['version']}")
    logger.info(f"Key params: entry={new_params['min_entry_price_cents']}-{new_params['max_entry_price_cents']}¢, "
                f"TP={new_params['take_profit_pct']:.0%}, SL={new_params['stop_loss_pct']:.0%}")

    return new_params


if __name__ == "__main__":
    retrain()
