"""HMM Observation Pipeline — computes 13 Layer-1 features from raw data.

Public API
----------
compute_observation_features(db, asset) -> Optional[dict]
    Compute one observation vector from recent price / market data.

ObservationPipeline(db)
    .record_observation(asset)   — compute + store
    .record_all_assets()         — BTC, ETH, SOL
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ASSETS = ["BTC", "ETH", "SOL"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_ts(ts_str: str) -> datetime:
    """Parse an ISO-8601 timestamp string to a timezone-aware datetime."""
    # Handle both 'Z' suffix and '+00:00'
    s = ts_str.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def _log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns: ln(p_t / p_{t-1})."""
    return np.diff(np.log(prices))


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """Simple EMA via recursive formula. Returns array same length as input."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation. Returns NaN for positions with < 2 values."""
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        segment = arr[i - window + 1: i + 1]
        out[i] = np.std(segment, ddof=1) if len(segment) >= 2 else np.nan
    return out


def _momentum_r_squared(prices: np.ndarray, window: int = 10) -> float:
    """R² of linear regression of prices over the last *window* bars."""
    if len(prices) < window:
        return 0.0
    y = prices[-window:]
    x = np.arange(window, dtype=float)
    # Fit least-squares line
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx == 0:
        return 0.0
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot == 0:
        return 1.0  # constant price → perfect fit
    return float(1.0 - ss_res / ss_tot)


# ── microstructure features ────────────────────────────────────────────────

def _query_market_snapshots(db, asset: str, since_ts: str, limit: int = 10) -> List[dict]:
    """Fetch recent 15-minute contract snapshots for *asset*."""
    pattern = f"KX{asset.upper()}15M%"
    with db._lock:
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT yes_bid, yes_ask, volume, timestamp FROM market_snapshots "
                "WHERE ticker LIKE ? AND timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
                (pattern, since_ts, limit),
            ).fetchall()
            return [dict(r) for r in rows]


def _compute_microstructure(snapshots: List[dict]) -> Dict[str, Any]:
    """Derive microstructure features from recent market snapshots."""
    if not snapshots:
        return {
            "bid_ask_spread": None,
            "spread_vol": None,
            "volume_1m": None,
            "volume_accel": None,
            "has_active_contract": 0,
        }

    # Latest snapshot
    latest = snapshots[0]
    yes_bid = latest.get("yes_bid") or 0.0
    yes_ask = latest.get("yes_ask") or 0.0
    mid = (yes_bid + yes_ask) / 2.0 if (yes_bid + yes_ask) > 0 else 1.0
    spread = (yes_ask - yes_bid) / mid

    # Spread volatility
    spreads = []
    for s in snapshots:
        b = s.get("yes_bid") or 0.0
        a = s.get("yes_ask") or 0.0
        m = (b + a) / 2.0 if (b + a) > 0 else 1.0
        spreads.append((a - b) / m)
    spread_vol = float(np.std(spreads, ddof=1)) if len(spreads) >= 2 else 0.0

    # Volume
    volume_1m = float(latest.get("volume") or 0)
    volumes = np.array([float(s.get("volume") or 0) for s in snapshots])
    ema_vol = _ema(volumes, span=min(10, len(volumes)))
    volume_accel = float(volumes[0] / ema_vol[-1]) if ema_vol[-1] > 0 else 1.0

    return {
        "bid_ask_spread": float(spread),
        "spread_vol": spread_vol,
        "volume_1m": volume_1m,
        "volume_accel": volume_accel,
        "has_active_contract": 1,
    }


# ── main feature computation ───────────────────────────────────────────────

def compute_observation_features(db, asset: str) -> Optional[Dict[str, Any]]:
    """Compute one observation vector from recent price data.

    Returns ``None`` if fewer than 3 price points are available.
    Uses up to 90 minutes of history for longer-horizon features.
    """
    asset = asset.upper()
    since = (datetime.now(timezone.utc) - timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = db.get_crypto_prices(asset=asset, since=since, limit=1000)

    # Fall back to fetching all available if 'since' yields nothing
    if len(rows) < 3:
        rows = db.get_crypto_prices(asset=asset, limit=1000)

    if len(rows) < 3:
        return None

    # Sort ascending by timestamp
    rows.sort(key=lambda r: r["timestamp"])
    prices = np.array([float(r["price_usd"]) for r in rows])
    n = len(prices)

    # Log returns
    lr = _log_returns(prices)  # length n-1

    log_return_1m = float(lr[-1]) if len(lr) >= 1 else 0.0
    log_return_5m = float(np.sum(lr[-5:])) if len(lr) >= 5 else float(np.sum(lr))
    log_return_15m = float(np.sum(lr[-15:])) if len(lr) >= 15 else float(np.sum(lr))

    # Realized volatility
    realized_vol_15m = float(np.std(lr[-15:], ddof=1)) if len(lr) >= 2 else 0.0
    realized_vol_1h = float(np.std(lr[-60:], ddof=1)) if len(lr) >= 2 else 0.0

    # Vol of vol: std of rolling 15-bar vol over last 30 bars
    if len(lr) >= 16:
        rvol = _rolling_std(lr, 15)
        # Take last 30 valid values
        valid = rvol[~np.isnan(rvol)]
        vol_of_vol = float(np.std(valid[-30:], ddof=1)) if len(valid) >= 2 else 0.0
    else:
        vol_of_vol = 0.0

    # Momentum R²
    momentum_r_sq = _momentum_r_squared(prices, window=min(10, n))

    # Mean reversion z-score: (price - EMA_15) / (price * realized_vol_15m)
    ema15 = _ema(prices, span=15)
    current_price = prices[-1]
    denom = current_price * realized_vol_15m if realized_vol_15m > 0 else 1.0
    mean_reversion = float((current_price - ema15[-1]) / denom)

    # Microstructure features
    latest_ts = rows[-1]["timestamp"]
    lookback_ts = (_parse_ts(latest_ts) - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
    snapshots = _query_market_snapshots(db, asset, lookback_ts)
    micro = _compute_microstructure(snapshots)

    return {
        "log_return_1m": log_return_1m,
        "log_return_5m": log_return_5m,
        "log_return_15m": log_return_15m,
        "realized_vol_15m": realized_vol_15m,
        "realized_vol_1h": realized_vol_1h,
        "vol_of_vol": vol_of_vol,
        "momentum_r_sq": momentum_r_sq,
        "mean_reversion": mean_reversion,
        **micro,
    }


# ── pipeline class ──────────────────────────────────────────────────────────

class ObservationPipeline:
    """Compute and record HMM observation features for tracked assets."""

    def __init__(self, db):
        self.db = db

    def record_observation(self, asset: str) -> Optional[int]:
        """Compute features for *asset* and store to hmm_observations.

        Returns the row id on success, or ``None`` if insufficient data.
        """
        features = compute_observation_features(self.db, asset)
        if features is None:
            logger.info("Skipping %s — insufficient price data", asset)
            return None
        row_id = self.db.record_hmm_observation(asset=asset, **features)
        logger.debug("Recorded HMM observation for %s (id=%d)", asset, row_id)
        return row_id

    def record_all_assets(self) -> Dict[str, Optional[int]]:
        """Record observations for BTC, ETH, SOL. Returns {asset: row_id}."""
        results = {}
        for asset in _ASSETS:
            try:
                results[asset] = self.record_observation(asset)
            except Exception:
                logger.exception("Failed to record observation for %s", asset)
                results[asset] = None
        return results
