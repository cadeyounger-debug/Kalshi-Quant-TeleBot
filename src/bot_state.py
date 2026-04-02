#!/usr/bin/env python3
"""Helper CLI for surfacing live Kalshi data to the Node interface."""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List
import logging

from kalshi_api import KalshiAPI

# Add settings manager import
try:
    from settings_manager import SettingsManager
    settings_available = True
except ImportError as e:
    settings_available = False
    logging.warning(f"SettingsManager not available: {e}")

# Initialize settings manager if available
settings_manager = None
if settings_available:
    try:
        settings_manager = SettingsManager()
    except Exception as e:
        logging.error(f"Failed to initialize settings manager: {e}")
        settings_available = False


def _cents_to_dollars(value: Any) -> float | None:
    try:
        return round(float(value) / 100, 2)
    except (TypeError, ValueError):
        return None


def fetch_balance(api: KalshiAPI) -> Dict[str, Any]:
    raw = api.get_account_balance()

    if raw is None:
        return {
            "summary": {
                "available": None,
                "total_equity": None,
                "unrealized_pnl": None,
                "realized_pnl": None,
                "fees_paid": None,
                "timestamp": None,
            },
            "raw": {},
            "error": "Kalshi API returned no data — check KALSHI_API_KEY and KALSHI_PRIVATE_KEY env vars",
        }

    # Get positions to calculate real P&L
    positions_data = api.get_positions() or {}
    event_positions = positions_data.get("event_positions") or []

    total_cost = 0.0
    total_exposure = 0.0
    total_fees = 0.0
    total_realized = 0.0
    for ep in event_positions:
        total_cost += float(ep.get("total_cost_dollars", 0))
        total_exposure += float(ep.get("event_exposure_dollars", 0))
        total_fees += float(ep.get("fees_paid_dollars", 0))
        total_realized += float(ep.get("realized_pnl_dollars", 0))

    # Unrealized P&L = current exposure value - total cost paid
    # (exposure is what positions are worth now)
    unrealized_pnl = total_exposure - total_cost if total_cost > 0 else 0

    available = _cents_to_dollars(raw.get("balance")) or 0
    portfolio_value = _cents_to_dollars(
        raw["portfolio_value"] if "portfolio_value" in raw else 0
    ) or 0

    summary = {
        "available": available,
        "total_equity": round(available + portfolio_value / 100, 2) if portfolio_value else available,
        "unrealized_pnl": round(unrealized_pnl, 2),
        "realized_pnl": round(total_realized, 2),
        "fees_paid": round(total_fees, 2),
        "total_cost": round(total_cost, 2),
        "timestamp": raw.get("updated_ts") or raw.get("timestamp") or raw.get("time"),
    }

    return {
        "summary": summary,
        "raw": raw,
    }


def fetch_positions(api: KalshiAPI) -> Dict[str, Any]:
    response = api.get_positions() or {}
    # Kalshi v2 returns market_positions (individual contracts)
    market_positions = response.get("market_positions") or []
    # Filter to only positions with actual holdings (position != 0)
    active = []
    for p in market_positions:
        pos = float(p.get("position_fp", 0))
        if pos != 0:
            active.append({
                "ticker": p.get("ticker", ""),
                "position": pos,
                "side": "YES" if pos > 0 else "NO",
                "quantity": abs(int(pos)),
                "exposure": p.get("market_exposure_dollars", "0"),
                "cost": p.get("total_traded_dollars", "0"),
                "fees": p.get("fees_paid_dollars", "0"),
                "realized_pnl": p.get("realized_pnl_dollars", "0"),
            })
    return {
        "positions": active,
        "count": len(active),
        "raw": response,
    }


def fetch_status(api: KalshiAPI) -> Dict[str, Any]:
    exchange_status = api.get_exchange_status() or {}
    balance = fetch_balance(api)
    positions = fetch_positions(api)

    # Note: Arbitrage analysis would require market data with price history
    # This is included in the main trader loop, not here for performance

    return {
        "exchange_status": exchange_status,
        "balance_summary": balance.get("summary", {}),
        "positions_count": positions.get("count", 0),
        "active_strategies": ["news_sentiment", "statistical_arbitrage", "volatility_based"],
        "risk_management": {
            "kelly_criterion_enabled": True,
            "dynamic_position_sizing": True,
            "stop_loss_protection": True,
            "take_profit_scaling": False,  # Simplified for Phase 2
            "max_position_size_pct": 10.0,
            "stop_loss_pct": 5.0
        },
        "phase3_features": {
            "real_time_market_data": True,
            "market_data_streaming": True,
            "performance_analytics": True,
            "advanced_reporting": True,
            "market_movement_tracking": True
        },
        "timestamp": time.time(),
    }


def fetch_performance(api: KalshiAPI) -> Dict[str, Any]:
    orders = api.get_orders(params={"limit": 100}) or {}
    orders_list: List[Dict[str, Any]] = orders.get("orders") or orders.get("data") or []

    filled_counts = [order.get("count") for order in orders_list if order.get("count")]
    avg_prices = [order.get("avg_price") or order.get("yes_price") for order in orders_list if order.get("avg_price") or order.get("yes_price")]

    total_trades = len(orders_list)
    total_contracts = sum(int(c) for c in filled_counts if isinstance(c, (int, float)))
    average_price = (
        round(
            sum(float(price) for price in avg_prices if isinstance(price, (int, float)))
            / len(avg_prices),
            4,
        )
        if avg_prices
        else None
    )

    return {
        "totalTrades": total_trades,
        "totalContracts": total_contracts,
        "averagePrice": average_price,
        "rawOrders": orders_list,
    }


def fetch_settings() -> Dict[str, Any]:
    """Fetch current bot settings."""
    if not settings_available or settings_manager is None:
        return {"error": "Settings manager not available", "available": settings_available}

    return settings_manager.get_settings()


def update_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update bot settings."""
    if not settings_available or settings_manager is None:
        return {"success": False, "error": "Settings manager not available", "available": settings_available}

    return settings_manager.update_settings(updates)


def reset_settings() -> Dict[str, Any]:
    """Reset settings to defaults."""
    if not settings_available or settings_manager is None:
        return {"success": False, "error": "Settings manager not available", "available": settings_available}

    return settings_manager.reset_to_defaults()


def fetch_settings_info() -> Dict[str, Any]:
    """Get information about available settings."""
    if not settings_available or settings_manager is None:
        return {"error": "Settings manager not available", "available": settings_available}

    return settings_manager.get_setting_info()


def fetch_dbstats() -> Dict[str, Any]:
    """Get database statistics."""
    try:
        from db import TradingDB
        import sqlite3
        db = TradingDB()

        def count(table, where=""):
            with db._connect() as conn:
                sql = f"SELECT COUNT(*) FROM {table}"
                if where:
                    sql += f" WHERE {where}"
                return conn.execute(sql).fetchone()[0]

        # Count trades excluding FAILED
        successful_trades = count("trades", "order_result NOT LIKE '%FAILED%'")

        return {
            "total_snapshots": count("market_snapshots"),
            "btc_snapshots": count("market_snapshots", "asset='BTC'"),
            "eth_snapshots": count("market_snapshots", "asset='ETH'"),
            "sol_snapshots": count("market_snapshots", "asset='SOL'"),
            "total_sentiment": count("news_sentiment"),
            "total_decisions": count("trade_decisions"),
            "total_trades": successful_trades,
            "crypto_prices": count("crypto_prices"),
            "db_path": db._db_path,
        }
    except Exception as e:
        return {"error": str(e)}


def run(command: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
    api = KalshiAPI()

    if command == "status":
        return fetch_status(api)
    if command == "positions":
        return fetch_positions(api)
    if command == "balance":
        return fetch_balance(api)
    if command == "performance":
        return fetch_performance(api)

    # Phase 4: Settings management commands
    if command == "settings":
        return fetch_settings()
    if command == "update_settings":
        return update_settings(data or {})
    if command == "reset_settings":
        return reset_settings()
    if command == "settings_info":
        return fetch_settings_info()

    if command == "dbstats":
        return fetch_dbstats()

    raise ValueError(f"Unsupported command: {command}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Expose bot state via CLI")
    parser.add_argument(
        "command",
        choices=["status", "positions", "balance", "performance", "settings", "update_settings", "reset_settings", "settings_info", "dbstats"],
        help="State command to execute",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="JSON data for commands that require input (e.g., update_settings)",
    )
    args = parser.parse_args()

    # Parse data if provided
    data = None
    if args.data:
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON data: {e}"}))
            sys.exit(1)

    try:
        payload = run(args.command, data)
        print(json.dumps(payload, default=str))
    except Exception as exc:  # pylint: disable=broad-except
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
