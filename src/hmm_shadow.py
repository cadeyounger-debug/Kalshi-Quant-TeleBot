"""HMM Shadow Tracker — records and evaluates paper predictions.

Tracks prediction accuracy and PnL without real money to validate
the HMM regime model before graduation to live trading.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ShadowTracker:
    """Records shadow (paper) predictions and computes rolling metrics."""

    def __init__(self, db):
        self.db = db

    def record_prediction(
        self,
        asset: str,
        ticker: str,
        regime_posterior: list,
        regime_entropy: float,
        fair_prob: float,
        market_price: float,
        edge_cents: float,
        ev_cents: float,
        confidence: float,
        recommendation: str,
        position_size: int,
    ) -> int:
        """Record a shadow prediction. Compute top_state from posterior. Return pred_id."""
        posterior = np.array(regime_posterior, dtype=float)
        top_state = int(np.argmax(posterior))
        top_state_prob = float(posterior[top_state])

        pred_id = self.db.record_shadow_prediction(
            asset=asset,
            ticker=ticker,
            regime_posterior=json.dumps(regime_posterior),
            regime_entropy=regime_entropy,
            top_state=top_state,
            top_state_prob=top_state_prob,
            fair_prob=fair_prob,
            market_price=market_price,
            edge_cents=edge_cents,
            ev_cents=ev_cents,
            confidence=confidence,
            recommendation=recommendation,
            position_size=position_size,
        )
        return pred_id

    def resolve_prediction(self, prediction_id: int, outcome: str, pnl_cents: float = None):
        """Resolve a shadow prediction with outcome and optional PnL."""
        self.db.resolve_shadow_prediction(prediction_id, outcome, pnl_cents)

    def get_rolling_metrics(self, asset: str = None, days: int = 7) -> Dict:
        """Compute rolling metrics over the last N days.

        Only count predictions where recommendation != 'skip' and outcome is resolved.
        Returns: trade_count, win_rate, total_pnl_cents, avg_pnl_per_trade,
                 max_drawdown_cents, wins, losses.
        """
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        predictions = self.db.get_shadow_predictions(
            asset=asset, since=since, resolved_only=True
        )

        # Filter out skips
        trades = [p for p in predictions if p.get("recommendation") != "skip"]

        if not trades:
            return {
                "trade_count": 0,
                "win_rate": 0.0,
                "total_pnl_cents": 0.0,
                "avg_pnl_per_trade": 0.0,
                "max_drawdown_cents": 0.0,
                "wins": 0,
                "losses": 0,
            }

        wins = sum(1 for t in trades if t.get("outcome") == "win")
        losses = sum(1 for t in trades if t.get("outcome") == "loss")
        total_pnl = sum(t.get("pnl_cents", 0) or 0 for t in trades)

        # Max drawdown: worst cumulative dip from peak
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in sorted(trades, key=lambda x: x.get("timestamp", "")):
            cumulative += t.get("pnl_cents", 0) or 0
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        trade_count = len(trades)
        return {
            "trade_count": trade_count,
            "win_rate": wins / trade_count if trade_count > 0 else 0.0,
            "total_pnl_cents": total_pnl,
            "avg_pnl_per_trade": total_pnl / trade_count if trade_count > 0 else 0.0,
            "max_drawdown_cents": max_dd,
            "wins": wins,
            "losses": losses,
        }

    def format_report(self, days: int = 7) -> str:
        """Format a summary report for Telegram."""
        m = self.get_rolling_metrics(days=days)
        lines = [
            f"Shadow Trading Report ({days}d)",
            f"─────────────────────",
            f"Trades: {m['trade_count']}",
            f"Win Rate: {m['win_rate']:.1%}",
            f"Total PnL: {m['total_pnl_cents']:.1f}¢",
            f"Avg PnL/Trade: {m['avg_pnl_per_trade']:.1f}¢",
            f"Max Drawdown: {m['max_drawdown_cents']:.1f}¢",
            f"W/L: {m['wins']}/{m['losses']}",
        ]
        return "\n".join(lines)
