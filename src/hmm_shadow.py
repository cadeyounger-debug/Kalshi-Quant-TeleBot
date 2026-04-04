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

    def resolve_expired_predictions(self, markets: list):
        """Resolve shadow predictions whose contracts have expired.

        For each unresolved prediction, check if the contract has settled.
        If spot ended above strike → YES wins. Below → NO wins.
        Compute shadow P&L based on what we would have bought.
        """
        unresolved = self.db.get_shadow_predictions(resolved_only=False, limit=200)
        unresolved = [p for p in unresolved
                      if p.get("outcome") is None and p.get("recommendation") not in ("skip", None)]

        if not unresolved:
            return []

        # Build market lookup
        market_lookup = {}
        for m in markets:
            ticker = m.get("ticker", "")
            if ticker:
                market_lookup[ticker] = m

        resolved = []
        now = datetime.now(timezone.utc)

        for pred in unresolved:
            ticker = pred.get("ticker", "")
            m = market_lookup.get(ticker)

            # Check if contract expired by looking at market status or time
            expired = False
            if m and m.get("status") in ("closed", "settled", "finalized"):
                expired = True
            elif m:
                exp_str = m.get("expected_expiration_time") or m.get("expiration_time") or ""
                if exp_str:
                    try:
                        exp_dt = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
                        if now > exp_dt + timedelta(minutes=2):
                            expired = True
                    except (ValueError, TypeError):
                        pass

            if not expired:
                continue

            # Determine outcome: check final price
            rec = pred.get("recommendation", "")
            side = "yes" if "yes" in rec else "no"
            entry_price = pred.get("market_price", 50)

            # If YES price settled near 100 → YES won. Near 0 → NO won.
            final_yes = 0
            if m:
                yes_bid = float(m.get("yes_bid_dollars", 0) or 0) * 100
                yes_ask = float(m.get("yes_ask_dollars", 0) or 0) * 100
                final_yes = (yes_bid + yes_ask) / 2 if (yes_bid or yes_ask) else 50

            yes_won = final_yes > 70  # YES settled if price near 100
            no_won = final_yes < 30   # NO settled if YES price near 0

            if side == "yes":
                if yes_won:
                    outcome = "win"
                    pnl = 100 - entry_price - 6  # Settles at $1, minus fees
                elif no_won:
                    outcome = "loss"
                    pnl = -entry_price - 6  # Worthless, minus fees
                else:
                    continue  # Can't determine yet
            else:  # NO side
                if no_won:
                    outcome = "win"
                    pnl = 100 - entry_price - 6
                elif yes_won:
                    outcome = "loss"
                    pnl = -entry_price - 6
                else:
                    continue

            self.resolve_prediction(pred["id"], outcome=outcome, pnl_cents=pnl)
            resolved.append({
                "ticker": ticker, "side": side, "outcome": outcome,
                "pnl_cents": pnl, "entry_price": entry_price,
            })
            logger.info(f"HMM shadow resolved: {ticker} {side} → {outcome} ({pnl:+.0f}¢)")

        return resolved

    def format_report(self, days: int = 7) -> str:
        """Format a summary report for Telegram."""
        m = self.get_rolling_metrics(days=days)
        lines = [
            f"🤖 HMM Shadow Report ({days}d)",
            f"─────────────────────",
            f"Trades: {m['trade_count']}",
            f"Win Rate: {m['win_rate']:.0%}",
            f"Total PnL: {m['total_pnl_cents']:.0f}¢",
            f"Avg PnL/Trade: {m['avg_pnl_per_trade']:.1f}¢",
            f"Max Drawdown: {m['max_drawdown_cents']:.0f}¢",
            f"W/L: {m['wins']}/{m['losses']}",
        ]

        # Per-asset breakdown
        for asset in ["BTC", "ETH", "SOL"]:
            am = self.get_rolling_metrics(asset=asset, days=days)
            if am["trade_count"] > 0:
                lines.append(f"  {asset}: {am['wins']}W/{am['losses']}L "
                            f"({am['win_rate']:.0%}), PnL={am['total_pnl_cents']:+.0f}¢")

        # Graduation status
        if m["trade_count"] >= 10:
            if m["win_rate"] >= 0.75:
                lines.append(f"\n✅ READY TO GRADUATE (WR={m['win_rate']:.0%} >= 75%)")
            else:
                lines.append(f"\n⏳ Not ready (WR={m['win_rate']:.0%}, need 75%)")

        return "\n".join(lines)

    def check_graduation(self, min_trades: int = 20, min_win_rate: float = 0.75) -> bool:
        """Check if HMM is ready to go live.

        Returns True if win rate >= 75% with at least min_trades resolved.
        """
        m = self.get_rolling_metrics(days=7)
        if m["trade_count"] < min_trades:
            return False
        return m["win_rate"] >= min_win_rate
