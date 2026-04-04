# HMM Regime Detection System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-layer probabilistic trading system — market regime HMM + contract opportunity model — that runs in shadow mode, collects fresh data, and graduates to live trading when it outperforms the current strategy.

**Architecture:** Layer 1 (hmm_regime.py) fits a per-asset Gaussian HMM to 1-minute bars, outputs posterior state distributions. Layer 2 (hmm_contract.py) evaluates specific 15M contracts using regime posteriors + contract features, computes fair value and edge. Shadow tracker logs predictions for comparison. All new code in `src/hmm_*.py`, new DB tables via migration in `db.py`.

**Tech Stack:** Python 3, hmmlearn, numpy, scipy, sqlite3, pytest

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/hmm_observations.py` | Aggregate raw data into 1-min bars, compute Layer 1 features, store to DB |
| `src/hmm_regime.py` | Per-asset HMM: fit (BIC state selection), inference (forward-backward posteriors), model persistence |
| `src/hmm_contract.py` | Layer 2: contract features, regime-conditioned fair value, edge computation, position sizing |
| `src/hmm_shadow.py` | Shadow prediction logging, outcome resolution, rolling performance metrics |
| `src/hmm_graduation.py` | Staged rollout logic, promotion gate checks, kill switch, demotion |
| `src/db.py` | Add HMM tables (hmm_observations, hmm_shadow_predictions, hmm_model_state) + query methods |
| `src/retrain.py` | Add HMM retrain step |
| `src/main.py` | Start observation pipeline, initialize HMM components |
| `src/trader.py` | Call shadow tracker each cycle |
| `tests/test_hmm_observations.py` | Observation pipeline tests |
| `tests/test_hmm_regime.py` | HMM engine tests |
| `tests/test_hmm_contract.py` | Contract evaluator tests |
| `tests/test_hmm_shadow.py` | Shadow tracker tests |
| `tests/test_hmm_graduation.py` | Graduation logic tests |

---

### Task 1: Add HMM Database Tables

Add three new tables to the database schema and query methods for reading/writing observations, shadow predictions, and model state.

**Files:**
- Modify: `src/db.py`
- Test: `tests/test_hmm_db.py`

- [ ] **Step 1: Write failing tests for HMM DB operations**

Create `tests/test_hmm_db.py`:

```python
"""Tests for HMM database tables and query methods."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import TradingDB


def _make_db():
    tmp = tempfile.mktemp(suffix=".db")
    return TradingDB(db_path=tmp)


def test_record_hmm_observation():
    db = _make_db()
    row_id = db.record_hmm_observation(
        asset="BTC",
        log_return_1m=0.001,
        log_return_5m=0.003,
        log_return_15m=0.005,
        realized_vol_15m=0.02,
        realized_vol_1h=0.015,
        vol_of_vol=0.005,
        momentum_r_sq=0.7,
        mean_reversion=-0.5,
        bid_ask_spread=0.03,
        spread_vol=0.01,
        volume_1m=15.0,
        volume_accel=1.2,
        has_active_contract=1,
    )
    assert row_id > 0

    rows = db.get_hmm_observations(asset="BTC", limit=10)
    assert len(rows) == 1
    assert rows[0]["asset"] == "BTC"
    assert abs(rows[0]["log_return_1m"] - 0.001) < 1e-6


def test_get_hmm_observations_since():
    db = _make_db()
    db.record_hmm_observation(asset="BTC", log_return_1m=0.001,
        timestamp="2026-04-01T00:00:00+00:00")
    db.record_hmm_observation(asset="BTC", log_return_1m=0.002,
        timestamp="2026-04-03T00:00:00+00:00")

    rows = db.get_hmm_observations(asset="BTC", since="2026-04-02T00:00:00+00:00")
    assert len(rows) == 1
    assert abs(rows[0]["log_return_1m"] - 0.002) < 1e-6


def test_record_shadow_prediction():
    db = _make_db()
    row_id = db.record_shadow_prediction(
        asset="ETH",
        ticker="KXETH15M-TEST",
        regime_posterior=json.dumps([0.1, 0.2, 0.3, 0.2, 0.2]),
        regime_entropy=1.5,
        top_state=2,
        top_state_prob=0.3,
        fair_prob=0.65,
        market_price=55,
        edge_cents=10.0,
        ev_cents=3.5,
        confidence=0.7,
        recommendation="buy_yes",
        position_size=1,
    )
    assert row_id > 0


def test_resolve_shadow_prediction():
    db = _make_db()
    row_id = db.record_shadow_prediction(
        asset="BTC", ticker="KXBTC15M-TEST",
        regime_posterior="[0.5,0.5]", regime_entropy=0.69,
        top_state=0, top_state_prob=0.5, fair_prob=0.6,
        market_price=50, edge_cents=10, ev_cents=5,
        confidence=0.8, recommendation="buy_yes", position_size=1,
    )
    db.resolve_shadow_prediction(row_id, outcome="win", pnl_cents=44)

    rows = db.get_shadow_predictions(asset="BTC", resolved_only=True)
    assert len(rows) == 1
    assert rows[0]["outcome"] == "win"
    assert rows[0]["pnl_cents"] == 44


def test_save_and_load_hmm_model_state():
    db = _make_db()
    db.save_hmm_model_state(
        asset="SOL", version=1, n_states=5, bic=-1500.0,
        log_likelihood=-1400.0, stability_flags=0,
        state_means=json.dumps([[0.1, 0.2], [0.3, 0.4]]),
        transition_matrix=json.dumps([[0.9, 0.1], [0.1, 0.9]]),
        observation_count=5000,
    )
    state = db.get_latest_hmm_model_state(asset="SOL")
    assert state is not None
    assert state["n_states"] == 5
    assert state["version"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot && python -m pytest tests/test_hmm_db.py -v`

Expected: FAIL — methods don't exist yet.

- [ ] **Step 3: Add HMM tables to DB schema**

In `src/db.py`, add to `_SCHEMA_SQL` (before the closing `"""`):

```sql
CREATE TABLE IF NOT EXISTS hmm_observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset           TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    log_return_1m   REAL,
    log_return_5m   REAL,
    log_return_15m  REAL,
    realized_vol_15m REAL,
    realized_vol_1h REAL,
    vol_of_vol      REAL,
    momentum_r_sq   REAL,
    mean_reversion  REAL,
    bid_ask_spread  REAL,
    spread_vol      REAL,
    volume_1m       REAL,
    volume_accel    REAL,
    has_active_contract INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_hmm_obs_asset_ts ON hmm_observations(asset, timestamp);

CREATE TABLE IF NOT EXISTS hmm_shadow_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset           TEXT NOT NULL,
    ticker          TEXT,
    timestamp       TEXT NOT NULL,
    regime_posterior TEXT,
    regime_entropy  REAL,
    top_state       INTEGER,
    top_state_prob  REAL,
    fair_prob       REAL,
    market_price    REAL,
    edge_cents      REAL,
    ev_cents        REAL,
    confidence      REAL,
    recommendation  TEXT,
    position_size   INTEGER,
    outcome         TEXT,
    pnl_cents       REAL,
    resolved_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_hmm_shadow_asset_ts ON hmm_shadow_predictions(asset, timestamp);
CREATE INDEX IF NOT EXISTS idx_hmm_shadow_outcome ON hmm_shadow_predictions(outcome);

CREATE TABLE IF NOT EXISTS hmm_model_state (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset           TEXT NOT NULL,
    version         INTEGER NOT NULL,
    n_states        INTEGER NOT NULL,
    bic             REAL,
    log_likelihood  REAL,
    stability_flags INTEGER DEFAULT 0,
    state_means     TEXT,
    transition_matrix TEXT,
    trained_at      TEXT NOT NULL,
    observation_count INTEGER
);
```

- [ ] **Step 4: Add query methods to TradingDB class**

Add these methods to the `TradingDB` class in `src/db.py`:

```python
    # ------------------------------------------------------------------
    # HMM Observation methods
    # ------------------------------------------------------------------

    def record_hmm_observation(
        self, asset: str, log_return_1m: float = None, log_return_5m: float = None,
        log_return_15m: float = None, realized_vol_15m: float = None,
        realized_vol_1h: float = None, vol_of_vol: float = None,
        momentum_r_sq: float = None, mean_reversion: float = None,
        bid_ask_spread: float = None, spread_vol: float = None,
        volume_1m: float = None, volume_accel: float = None,
        has_active_contract: int = 0, timestamp: str = None,
    ) -> int:
        timestamp = timestamp or _now_iso()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO hmm_observations "
                    "(asset, timestamp, log_return_1m, log_return_5m, log_return_15m, "
                    "realized_vol_15m, realized_vol_1h, vol_of_vol, momentum_r_sq, "
                    "mean_reversion, bid_ask_spread, spread_vol, volume_1m, "
                    "volume_accel, has_active_contract) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (asset.upper(), timestamp, log_return_1m, log_return_5m,
                     log_return_15m, realized_vol_15m, realized_vol_1h, vol_of_vol,
                     momentum_r_sq, mean_reversion, bid_ask_spread, spread_vol,
                     volume_1m, volume_accel, has_active_contract),
                )
                return cur.lastrowid

    def get_hmm_observations(
        self, asset: str = None, since: str = None, limit: int = 50000,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM hmm_observations"
        conditions = []
        params = []
        if asset:
            conditions.append("asset = ?")
            params.append(asset.upper())
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # HMM Shadow Prediction methods
    # ------------------------------------------------------------------

    def record_shadow_prediction(
        self, asset: str, ticker: str = None, regime_posterior: str = None,
        regime_entropy: float = None, top_state: int = None,
        top_state_prob: float = None, fair_prob: float = None,
        market_price: float = None, edge_cents: float = None,
        ev_cents: float = None, confidence: float = None,
        recommendation: str = None, position_size: int = None,
        timestamp: str = None,
    ) -> int:
        timestamp = timestamp or _now_iso()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO hmm_shadow_predictions "
                    "(asset, ticker, timestamp, regime_posterior, regime_entropy, "
                    "top_state, top_state_prob, fair_prob, market_price, edge_cents, "
                    "ev_cents, confidence, recommendation, position_size) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (asset.upper(), ticker, timestamp, regime_posterior,
                     regime_entropy, top_state, top_state_prob, fair_prob,
                     market_price, edge_cents, ev_cents, confidence,
                     recommendation, position_size),
                )
                return cur.lastrowid

    def resolve_shadow_prediction(self, prediction_id: int, outcome: str, pnl_cents: float = None):
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE hmm_shadow_predictions SET outcome = ?, pnl_cents = ?, "
                    "resolved_at = ? WHERE id = ?",
                    (outcome, pnl_cents, _now_iso(), prediction_id),
                )

    def get_shadow_predictions(
        self, asset: str = None, since: str = None,
        resolved_only: bool = False, limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM hmm_shadow_predictions"
        conditions = []
        params = []
        if asset:
            conditions.append("asset = ?")
            params.append(asset.upper())
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        if resolved_only:
            conditions.append("outcome IS NOT NULL")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # HMM Model State methods
    # ------------------------------------------------------------------

    def save_hmm_model_state(
        self, asset: str, version: int, n_states: int, bic: float = None,
        log_likelihood: float = None, stability_flags: int = 0,
        state_means: str = None, transition_matrix: str = None,
        observation_count: int = 0, trained_at: str = None,
    ) -> int:
        trained_at = trained_at or _now_iso()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO hmm_model_state "
                    "(asset, version, n_states, bic, log_likelihood, "
                    "stability_flags, state_means, transition_matrix, "
                    "trained_at, observation_count) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (asset.upper(), version, n_states, bic, log_likelihood,
                     stability_flags, state_means, transition_matrix,
                     trained_at, observation_count),
                )
                return cur.lastrowid

    def get_latest_hmm_model_state(self, asset: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM hmm_model_state WHERE asset = ? "
                    "ORDER BY version DESC LIMIT 1",
                    (asset.upper(),),
                ).fetchone()
                return dict(row) if row else None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot && python -m pytest tests/test_hmm_db.py -v`

Expected: All 5 PASS.

- [ ] **Step 6: Commit**

```bash
cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot
git add src/db.py tests/test_hmm_db.py
git commit -m "feat: add HMM database tables and query methods

Three new tables: hmm_observations, hmm_shadow_predictions,
hmm_model_state. Full CRUD methods on TradingDB.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Observation Pipeline

Compute Layer 1 features from raw price and market data, aggregate into 1-minute bars, and store to the hmm_observations table.

**Files:**
- Create: `src/hmm_observations.py`
- Test: `tests/test_hmm_observations.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_hmm_observations.py`:

```python
"""Tests for HMM observation pipeline."""
import sys, os, math, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import MagicMock
from db import TradingDB
from hmm_observations import compute_observation_features, ObservationPipeline


def _make_db_with_prices(asset, prices, start_ts="2026-04-03T20:00:00+00:00"):
    """Create a DB with N spot price records 1 minute apart."""
    tmp = tempfile.mktemp(suffix=".db")
    db = TradingDB(db_path=tmp)
    from datetime import datetime, timedelta, timezone
    base = datetime.fromisoformat(start_ts)
    for i, p in enumerate(prices):
        ts = (base + timedelta(minutes=i)).isoformat()
        db.record_crypto_price(asset, p, change_24h_pct=0.0, timestamp=ts)
    return db


def test_compute_features_basic():
    """Should compute log returns and vol from price history."""
    # 20 prices rising steadily from 100 to ~102
    prices = [100 + i * 0.1 for i in range(20)]
    db = _make_db_with_prices("BTC", prices)

    features = compute_observation_features(db, "BTC")
    assert features is not None
    assert "log_return_1m" in features
    assert "realized_vol_15m" in features
    assert features["log_return_1m"] > 0  # Price was rising


def test_compute_features_insufficient_data():
    """Should return None if fewer than 3 price points."""
    db = _make_db_with_prices("BTC", [100, 101])
    features = compute_observation_features(db, "BTC")
    assert features is None


def test_compute_features_with_market_data():
    """Should include microstructure features when contract data available."""
    prices = [100 + i * 0.1 for i in range(20)]
    db = _make_db_with_prices("ETH", prices)
    
    # Add a market snapshot for an active 15M contract
    db.record_market_snapshot(
        "KXETH15M-TEST", title="Test", yes_bid=0.45, yes_ask=0.55,
        no_bid=0.45, no_ask=0.55, volume=50, strike_price=100.5,
        spot_price=101.9, expiration_time="2026-04-03T21:00:00+00:00",
    )

    features = compute_observation_features(db, "ETH")
    assert features is not None
    assert features["has_active_contract"] == 1
    assert features["bid_ask_spread"] > 0


def test_observation_pipeline_records_to_db():
    """Pipeline should compute and store observation."""
    prices = [100 + i * 0.1 for i in range(20)]
    db = _make_db_with_prices("SOL", prices)
    pipeline = ObservationPipeline(db)

    pipeline.record_observation("SOL")

    rows = db.get_hmm_observations(asset="SOL")
    assert len(rows) == 1
    assert rows[0]["asset"] == "SOL"
    assert rows[0]["log_return_1m"] is not None


def test_log_return_values():
    """Log returns should match manual calculation."""
    prices = [100.0, 101.0, 100.5]  # +1%, -0.5%
    db = _make_db_with_prices("BTC", prices + [100.5] * 17)  # pad to 20

    features = compute_observation_features(db, "BTC")
    # Most recent 1-min return: ln(100.5/100.5) = 0
    # The exact value depends on which prices are "most recent"
    assert features is not None
    assert isinstance(features["log_return_1m"], float)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot && python -m pytest tests/test_hmm_observations.py -v`

Expected: FAIL — `hmm_observations` module doesn't exist.

- [ ] **Step 3: Implement observation pipeline**

Create `src/hmm_observations.py`:

```python
"""HMM observation pipeline — aggregate raw data into 1-minute feature bars.

Computes Layer 1 features (spot regime + microstructure) from the
crypto_prices and market_snapshots tables. Called every 60 seconds.
"""

import logging
import math
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def compute_observation_features(db, asset: str) -> Optional[Dict[str, Any]]:
    """Compute a single observation vector from recent price data.

    Requires at least 3 price points. Uses up to 90 minutes of history
    for longer-horizon features (15m return, 1h vol).

    Returns None if insufficient data.
    """
    since = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
    prices_raw = db.get_crypto_prices(asset=asset, since=since, limit=500)

    price_vals = [p["price_usd"] for p in prices_raw
                  if p.get("price_usd") and p["price_usd"] > 0]

    if len(price_vals) < 3:
        return None

    current = price_vals[-1]

    # --- Log returns at multiple horizons ---
    log_return_1m = math.log(price_vals[-1] / price_vals[-2]) if len(price_vals) >= 2 else 0.0
    log_return_5m = math.log(price_vals[-1] / price_vals[-6]) if len(price_vals) >= 6 else log_return_1m
    log_return_15m = math.log(price_vals[-1] / price_vals[-16]) if len(price_vals) >= 16 else log_return_5m

    # --- Realized volatility ---
    def _realized_vol(vals):
        if len(vals) < 3:
            return 0.0
        returns = [math.log(vals[i] / vals[i - 1]) for i in range(1, len(vals)) if vals[i - 1] > 0]
        return float(np.std(returns)) if returns else 0.0

    realized_vol_15m = _realized_vol(price_vals[-16:]) if len(price_vals) >= 16 else _realized_vol(price_vals)
    realized_vol_1h = _realized_vol(price_vals[-61:]) if len(price_vals) >= 61 else realized_vol_15m

    # --- Vol of vol (std of rolling 15-bar vol over 30 bars) ---
    vol_of_vol = 0.0
    if len(price_vals) >= 30:
        rolling_vols = []
        for i in range(15, min(len(price_vals), 45)):
            window = price_vals[max(0, i - 15):i]
            rolling_vols.append(_realized_vol(window))
        vol_of_vol = float(np.std(rolling_vols)) if len(rolling_vols) >= 2 else 0.0

    # --- Momentum R² (linear regression fit over last 10 bars) ---
    momentum_window = price_vals[-10:] if len(price_vals) >= 10 else price_vals
    x = np.arange(len(momentum_window), dtype=float)
    y = np.array(momentum_window, dtype=float)
    if len(x) >= 3:
        n = len(x)
        sx, sy = np.sum(x), np.sum(y)
        sxy, sxx = np.sum(x * y), np.sum(x * x)
        denom = n * sxx - sx * sx
        if denom != 0:
            slope = (n * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / n
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            momentum_r_sq = max(0.0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            momentum_r_sq = 0.0
    else:
        momentum_r_sq = 0.0

    # --- Mean reversion signal: z-score vs 15-bar EMA ---
    if len(price_vals) >= 15 and realized_vol_15m > 0:
        ema_alpha = 2 / (15 + 1)
        ema = price_vals[-15]
        for p in price_vals[-14:]:
            ema = ema_alpha * p + (1 - ema_alpha) * ema
        mean_reversion = (current - ema) / (current * realized_vol_15m) if current > 0 else 0.0
    else:
        mean_reversion = 0.0

    # --- Microstructure features from nearest active 15M contract ---
    bid_ask_spread = 0.0
    spread_vol = 0.0
    volume_1m = 0.0
    volume_accel = 0.0
    has_active_contract = 0

    recent_ts = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    # Query recent 15M snapshots for this asset
    try:
        from db import extract_asset
        snapshots = db.get_hmm_observations.__self__  # won't work — use db directly
    except Exception:
        pass

    # Use the DB's existing query to find recent market snapshots
    try:
        with db._lock:
            with db._connect() as conn:
                prefix = {"BTC": "KXBTC15M", "ETH": "KXETH15M", "SOL": "KXSOL15M"}.get(asset.upper(), "")
                if prefix:
                    rows = conn.execute(
                        "SELECT yes_bid, yes_ask, no_bid, no_ask, volume, timestamp "
                        "FROM market_snapshots "
                        "WHERE ticker LIKE ? AND timestamp >= ? "
                        "ORDER BY timestamp DESC LIMIT 10",
                        (f"{prefix}%", recent_ts),
                    ).fetchall()

                    if rows:
                        has_active_contract = 1
                        latest = rows[0]
                        yes_bid = latest["yes_bid"] or 0
                        yes_ask = latest["yes_ask"] or 0
                        mid = (yes_bid + yes_ask) / 2 if (yes_bid and yes_ask) else 0
                        bid_ask_spread = (yes_ask - yes_bid) / mid if mid > 0 else 0

                        # Spread volatility over recent snapshots
                        if len(rows) >= 3:
                            spreads = []
                            for r in rows:
                                b, a = (r["yes_bid"] or 0), (r["yes_ask"] or 0)
                                m = (b + a) / 2 if (b and a) else 0
                                if m > 0:
                                    spreads.append((a - b) / m)
                            spread_vol = float(np.std(spreads)) if len(spreads) >= 2 else 0.0

                        # Volume: use latest snapshot's volume field
                        volume_1m = float(latest["volume"] or 0)

                        # Volume acceleration: current vs EMA
                        if len(rows) >= 5:
                            vols = [float(r["volume"] or 0) for r in rows[:5]]
                            ema_vol = np.mean(vols[1:]) if len(vols) > 1 else 1.0
                            volume_accel = vols[0] / ema_vol if ema_vol > 0 else 1.0
    except Exception as e:
        logger.debug(f"Microstructure feature error for {asset}: {e}")

    return {
        "log_return_1m": round(log_return_1m, 8),
        "log_return_5m": round(log_return_5m, 8),
        "log_return_15m": round(log_return_15m, 8),
        "realized_vol_15m": round(realized_vol_15m, 8),
        "realized_vol_1h": round(realized_vol_1h, 8),
        "vol_of_vol": round(vol_of_vol, 8),
        "momentum_r_sq": round(momentum_r_sq, 4),
        "mean_reversion": round(mean_reversion, 4),
        "bid_ask_spread": round(bid_ask_spread, 6),
        "spread_vol": round(spread_vol, 6),
        "volume_1m": round(volume_1m, 2),
        "volume_accel": round(volume_accel, 4),
        "has_active_contract": has_active_contract,
    }


class ObservationPipeline:
    """Records 1-minute observations for each asset."""

    def __init__(self, db):
        self.db = db
        self._last_record_time = {}

    def record_observation(self, asset: str):
        """Compute and store one observation for the given asset."""
        features = compute_observation_features(self.db, asset)
        if features is None:
            return

        self.db.record_hmm_observation(asset=asset, **features)

    def record_all_assets(self):
        """Record observations for BTC, ETH, SOL."""
        for asset in ["BTC", "ETH", "SOL"]:
            try:
                self.record_observation(asset)
            except Exception as e:
                logger.error(f"HMM observation error for {asset}: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot && python -m pytest tests/test_hmm_observations.py -v`

Expected: All 5 PASS.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot
git add src/hmm_observations.py tests/test_hmm_observations.py
git commit -m "feat: add HMM observation pipeline — 1-min feature bars from raw data

Computes 13 features per asset per minute: multi-horizon returns,
realized vol, vol-of-vol, momentum R², mean reversion, and
microstructure features from nearest 15M contract.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: HMM Regime Engine

Per-asset Gaussian HMM with BIC state selection and forward-backward posterior inference.

**Files:**
- Create: `src/hmm_regime.py`
- Test: `tests/test_hmm_regime.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_hmm_regime.py`:

```python
"""Tests for HMM regime engine."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from db import TradingDB
from hmm_regime import RegimeEngine, fit_hmm_select_k, decode_posterior


def _make_synthetic_observations(n=500, n_features=8):
    """Generate synthetic observation data with 2 clear regimes."""
    np.random.seed(42)
    # Regime 1: low vol, small positive returns
    r1 = np.random.randn(n // 2, n_features) * 0.01 + 0.001
    # Regime 2: high vol, negative returns
    r2 = np.random.randn(n // 2, n_features) * 0.05 - 0.003
    data = np.vstack([r1, r2])
    return data


def test_fit_hmm_select_k():
    """Should select K by BIC from candidate range."""
    data = _make_synthetic_observations(n=500)
    model, k, bic = fit_hmm_select_k(data, k_range=range(2, 6))
    assert model is not None
    assert 2 <= k <= 5
    assert bic < 0  # BIC is negative log-likelihood scale


def test_decode_posterior():
    """Posterior should be a valid probability distribution."""
    data = _make_synthetic_observations(n=200)
    model, k, _ = fit_hmm_select_k(data, k_range=range(2, 4))

    posterior = decode_posterior(model, data)
    assert posterior.shape == (200, k)
    # Each row should sum to ~1
    row_sums = posterior.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
    # All values should be between 0 and 1
    assert np.all(posterior >= 0)
    assert np.all(posterior <= 1)


def test_decode_posterior_last_step():
    """Should be able to get just the last timestep's posterior."""
    data = _make_synthetic_observations(n=200)
    model, k, _ = fit_hmm_select_k(data, k_range=range(2, 4))

    posterior = decode_posterior(model, data)
    last = posterior[-1]
    assert len(last) == k
    assert abs(sum(last) - 1.0) < 1e-6


def test_regime_engine_with_db():
    """RegimeEngine should fit, persist, and decode from DB data."""
    tmp = tempfile.mktemp(suffix=".db")
    db = TradingDB(db_path=tmp)

    # Insert synthetic observations
    data = _make_synthetic_observations(n=300, n_features=13)
    feature_names = [
        "log_return_1m", "log_return_5m", "log_return_15m",
        "realized_vol_15m", "realized_vol_1h", "vol_of_vol",
        "momentum_r_sq", "mean_reversion", "bid_ask_spread",
        "spread_vol", "volume_1m", "volume_accel", "has_active_contract",
    ]
    from datetime import datetime, timedelta, timezone
    base = datetime(2026, 4, 1, tzinfo=timezone.utc)
    for i in range(300):
        kwargs = {name: float(data[i, j]) for j, name in enumerate(feature_names)}
        kwargs["timestamp"] = (base + timedelta(minutes=i)).isoformat()
        db.record_hmm_observation(asset="BTC", **kwargs)

    engine = RegimeEngine(db)
    result = engine.fit_asset("BTC")

    assert result is not None
    assert "n_states" in result
    assert "bic" in result

    # Decode current regime
    posterior = engine.get_current_posterior("BTC")
    assert posterior is not None
    assert len(posterior) == result["n_states"]
    assert abs(sum(posterior) - 1.0) < 1e-6


def test_regime_engine_insufficient_data():
    """Should return None when not enough observations."""
    tmp = tempfile.mktemp(suffix=".db")
    db = TradingDB(db_path=tmp)

    engine = RegimeEngine(db)
    result = engine.fit_asset("BTC")
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot && python -m pytest tests/test_hmm_regime.py -v`

Expected: FAIL — `hmm_regime` module doesn't exist.

- [ ] **Step 3: Implement regime engine**

Create `src/hmm_regime.py`:

```python
"""HMM Regime Engine — per-asset Gaussian HMM with BIC state selection.

Layer 1 of the two-layer trading system. Learns latent market states
from 1-minute observation bars. Outputs full posterior distributions,
not hard state assignments.
"""

import json
import logging
import math
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

# Feature columns from hmm_observations table (order matters for training)
FEATURE_COLS = [
    "log_return_1m", "log_return_5m", "log_return_15m",
    "realized_vol_15m", "realized_vol_1h", "vol_of_vol",
    "momentum_r_sq", "mean_reversion", "bid_ask_spread",
    "spread_vol", "volume_1m", "volume_accel",
]
# has_active_contract is a mask, not a continuous feature — excluded from HMM input

MIN_OBSERVATIONS = 300  # ~5 hours of 1-min data
DEFAULT_K = 5


def _observations_to_matrix(rows: List[Dict]) -> np.ndarray:
    """Convert DB rows to a (N, D) numpy array of features."""
    data = []
    for r in rows:
        vec = [float(r.get(col) or 0.0) for col in FEATURE_COLS]
        data.append(vec)
    return np.array(data, dtype=np.float64)


def _normalize(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize each column. Returns (normalized, means, stds)."""
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    stds[stds < 1e-10] = 1.0  # Prevent division by zero
    return (data - means) / stds, means, stds


def fit_hmm_select_k(
    data: np.ndarray,
    k_range=range(3, 9),
    n_restarts: int = 5,
) -> Tuple[Any, int, float]:
    """Fit Gaussian HMMs for each K, select by BIC.

    Returns (best_model, best_k, best_bic).
    """
    from hmmlearn.hmm import GaussianHMM

    best_model = None
    best_k = DEFAULT_K
    best_bic = float('inf')

    for k in k_range:
        try:
            best_ll_for_k = -float('inf')
            best_model_for_k = None

            for _ in range(n_restarts):
                model = GaussianHMM(
                    n_components=k,
                    covariance_type="full",
                    n_iter=100,
                    tol=1e-4,
                    random_state=None,  # Different seed each restart
                )
                model.fit(data)
                ll = model.score(data)
                if ll > best_ll_for_k:
                    best_ll_for_k = ll
                    best_model_for_k = model

            # BIC = -2 * log_likelihood + n_params * log(n_samples)
            n_samples = data.shape[0]
            n_features = data.shape[1]
            # Parameters: k*n_features means + k*n_features*(n_features+1)/2 covariances
            # + k*(k-1) transition params + (k-1) initial state params
            n_params = (k * n_features
                       + k * n_features * (n_features + 1) // 2
                       + k * (k - 1)
                       + (k - 1))
            bic = -2 * best_ll_for_k + n_params * math.log(n_samples)

            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_model = best_model_for_k

            logger.info(f"  K={k}: BIC={bic:.1f}, LL={best_ll_for_k:.1f}")

        except Exception as e:
            logger.warning(f"  K={k}: fit failed ({e})")
            continue

    return best_model, best_k, best_bic


def decode_posterior(model, data: np.ndarray) -> np.ndarray:
    """Run forward-backward to get P(state | observations) at each timestep.

    Returns (N, K) array of posterior probabilities.
    """
    posteriors = model.predict_proba(data)
    return posteriors


class RegimeEngine:
    """Per-asset HMM regime detection.

    Fits a Gaussian HMM to stored observations, selects K by BIC,
    and provides posterior state distributions for the current timestep.
    """

    def __init__(self, db):
        self.db = db
        self._models = {}       # asset -> fitted model
        self._norm_params = {}  # asset -> (means, stds)
        self._n_states = {}     # asset -> K

    def fit_asset(self, asset: str) -> Optional[Dict[str, Any]]:
        """Fit HMM for one asset using all stored observations.

        Returns fit summary dict or None if insufficient data.
        """
        rows = self.db.get_hmm_observations(asset=asset)
        if len(rows) < MIN_OBSERVATIONS:
            logger.info(f"HMM {asset}: insufficient data ({len(rows)}/{MIN_OBSERVATIONS})")
            return None

        data = _observations_to_matrix(rows)
        normalized, means, stds = _normalize(data)

        logger.info(f"HMM {asset}: fitting with {len(rows)} observations")
        model, k, bic = fit_hmm_select_k(normalized)

        if model is None:
            logger.error(f"HMM {asset}: all fits failed")
            return None

        self._models[asset] = model
        self._norm_params[asset] = (means, stds)
        self._n_states[asset] = k

        # Persist model state to DB
        ll = model.score(normalized)
        state_means = model.means_.tolist()
        trans_mat = model.transmat_.tolist()

        prior_state = self.db.get_latest_hmm_model_state(asset)
        version = (prior_state["version"] + 1) if prior_state else 1

        # Check stability against prior model
        stability_flags = 0
        if prior_state and prior_state.get("state_means"):
            try:
                old_means = json.loads(prior_state["state_means"])
                old_means_arr = np.array(old_means)
                new_means_arr = np.array(state_means)
                if old_means_arr.shape == new_means_arr.shape:
                    diffs = np.abs(new_means_arr - old_means_arr)
                    old_stds = np.std(old_means_arr, axis=0)
                    old_stds[old_stds < 1e-10] = 1.0
                    shifts = diffs / old_stds
                    unstable = np.any(shifts > 2.0, axis=1).sum()
                    stability_flags = int(unstable)
            except Exception:
                pass

        self.db.save_hmm_model_state(
            asset=asset, version=version, n_states=k,
            bic=bic, log_likelihood=ll,
            stability_flags=stability_flags,
            state_means=json.dumps(state_means),
            transition_matrix=json.dumps(trans_mat),
            observation_count=len(rows),
        )

        logger.info(f"HMM {asset}: K={k}, BIC={bic:.1f}, LL={ll:.1f}, "
                     f"stability_flags={stability_flags}, v{version}")

        return {
            "n_states": k, "bic": bic, "log_likelihood": ll,
            "stability_flags": stability_flags, "version": version,
            "observation_count": len(rows),
        }

    def get_current_posterior(self, asset: str) -> Optional[List[float]]:
        """Get posterior state distribution for the most recent timestep.

        Returns list of K probabilities, or None if model not fitted.
        """
        if asset not in self._models:
            return None

        rows = self.db.get_hmm_observations(asset=asset, limit=200)
        if len(rows) < 10:
            return None

        data = _observations_to_matrix(rows)
        means, stds = self._norm_params[asset]
        normalized = (data - means) / stds

        posteriors = decode_posterior(self._models[asset], normalized)
        last = posteriors[-1].tolist()
        return last

    def get_regime_entropy(self, posterior: List[float]) -> float:
        """Compute entropy of the posterior distribution."""
        entropy = 0.0
        for p in posterior:
            if p > 1e-10:
                entropy -= p * math.log(p)
        return entropy

    def get_transition_matrix(self, asset: str) -> Optional[np.ndarray]:
        """Get the fitted transition matrix for an asset."""
        if asset not in self._models:
            return None
        return self._models[asset].transmat_

    def fit_all_assets(self) -> Dict[str, Any]:
        """Fit HMMs for all three assets. Returns summary dict."""
        results = {}
        for asset in ["BTC", "ETH", "SOL"]:
            results[asset] = self.fit_asset(asset)
        return results
```

- [ ] **Step 4: Add hmmlearn to requirements.txt**

Add to `requirements.txt`:
```
hmmlearn>=0.3.0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot && pip install hmmlearn && python -m pytest tests/test_hmm_regime.py -v`

Expected: All 5 PASS.

- [ ] **Step 6: Commit**

```bash
cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot
git add src/hmm_regime.py tests/test_hmm_regime.py requirements.txt
git commit -m "feat: add HMM regime engine — per-asset Gaussian HMM with BIC selection

Fits K in {3..8}, selects by BIC, forward-backward posteriors,
stability monitoring across retrains. Persists model state to DB.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Contract Opportunity Model

Layer 2: regime-conditioned fair value estimation, edge computation, and position sizing.

**Files:**
- Create: `src/hmm_contract.py`
- Test: `tests/test_hmm_contract.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_hmm_contract.py`:

```python
"""Tests for HMM contract opportunity model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from hmm_contract import (
    compute_regime_ev, position_size, ContractEvaluation,
    evaluate_contract_with_regime,
)


def _make_state_profiles(n_states, win_rate=0.6, avg_win=40, avg_loss=30, count=50):
    """Create uniform state profiles for testing."""
    return [
        {"win_rate": win_rate, "avg_win_cents": avg_win,
         "avg_loss_cents": avg_loss, "trade_count": count}
        for _ in range(n_states)
    ]


def test_regime_ev_positive():
    """Positive EV when win_rate * avg_win > (1-win_rate) * avg_loss + fees."""
    posterior = [0.5, 0.5]
    profiles = _make_state_profiles(2, win_rate=0.65, avg_win=50, avg_loss=30)
    ev, confidence = compute_regime_ev(posterior, profiles, edge_cents=10)
    # 0.65*50 - 0.35*30 - 6 = 32.5 - 10.5 - 6 = 16.0
    assert ev > 0, f"Expected positive EV, got {ev}"
    assert confidence > 0


def test_regime_ev_negative():
    """Negative EV when losses dominate."""
    posterior = [1.0, 0.0]
    profiles = _make_state_profiles(2, win_rate=0.3, avg_win=20, avg_loss=40)
    ev, confidence = compute_regime_ev(posterior, profiles, edge_cents=5)
    # 0.3*20 - 0.7*40 - 6 = 6 - 28 - 6 = -28
    assert ev < 0, f"Expected negative EV, got {ev}"


def test_regime_ev_insufficient_data():
    """States with <10 trades should contribute 0 EV (break even prior)."""
    posterior = [0.5, 0.5]
    profiles = [
        {"win_rate": 0.9, "avg_win_cents": 50, "avg_loss_cents": 10, "trade_count": 5},  # Too few
        {"win_rate": 0.6, "avg_win_cents": 40, "avg_loss_cents": 30, "trade_count": 50},
    ]
    ev, confidence = compute_regime_ev(posterior, profiles, edge_cents=10)
    # State 0 contributes 0 (insufficient data), state 1 contributes positive
    # Confidence should be lower than all-data case
    assert confidence < 1.0


def test_position_size_positive_ev():
    """Should return >0 contracts for positive EV with confidence."""
    size = position_size(ev_cents=10, confidence=0.7, bankroll=1000)
    assert size >= 1


def test_position_size_negative_ev():
    """Should return 0 for negative EV."""
    size = position_size(ev_cents=-5, confidence=0.7, bankroll=1000)
    assert size == 0


def test_position_size_low_confidence():
    """Should return 0 when confidence is too low."""
    size = position_size(ev_cents=10, confidence=0.1, bankroll=1000)
    assert size == 0


def test_evaluate_contract_with_regime():
    """Full evaluation pipeline should produce a ContractEvaluation."""
    posterior = [0.2, 0.3, 0.3, 0.1, 0.1]
    profiles = _make_state_profiles(5, win_rate=0.6, avg_win=40, avg_loss=30, count=50)
    
    result = evaluate_contract_with_regime(
        regime_posterior=posterior,
        state_profiles=profiles,
        spot_price=67000,
        strike_price=66900,
        yes_price_cents=55,
        no_price_cents=45,
        time_to_expiry_secs=600,
        contract_volume=100,
        bid_ask_spread_cents=3,
        log_normal_prob=0.60,
        bankroll=1000,
    )

    assert isinstance(result, ContractEvaluation)
    assert 0 <= result.fair_prob <= 1
    assert result.recommendation in ("buy_yes", "buy_no", "skip")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot && python -m pytest tests/test_hmm_contract.py -v`

Expected: FAIL — `hmm_contract` module doesn't exist.

- [ ] **Step 3: Implement contract opportunity model**

Create `src/hmm_contract.py`:

```python
"""HMM Contract Opportunity Model — Layer 2.

Evaluates specific 15M contracts using regime posteriors + contract
features. Computes fair value probability, edge, and position sizing
based on posterior-weighted expected value.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

FEES_CENTS = 6  # 2c entry + 2c exit + 1c slippage each side


@dataclass
class ContractEvaluation:
    """Result of evaluating a single contract."""
    fair_prob: float
    edge_yes_cents: float
    edge_no_cents: float
    ev_cents: float
    confidence: float
    regime_entropy: float
    recommendation: str  # "buy_yes", "buy_no", "skip"
    position_size: int
    reasons: List[str]


def compute_regime_ev(
    regime_posterior: List[float],
    state_profiles: List[Dict],
    edge_cents: float,
    fees_cents: float = FEES_CENTS,
) -> Tuple[float, float]:
    """Compute posterior-weighted expected value across all states.

    Args:
        regime_posterior: P(state=k) for each state
        state_profiles: Per-state historical performance
        edge_cents: Raw edge of this trade
        fees_cents: Round-trip execution cost

    Returns:
        (ev_cents, confidence) where confidence scales with data availability
    """
    ev = 0.0
    confidence = 0.0

    for k, p_k in enumerate(regime_posterior):
        if k >= len(state_profiles):
            continue
        profile = state_profiles[k]
        count = profile.get("trade_count", 0)

        if count < 10:
            # Insufficient data — use break-even prior
            state_ev = 0.0
        else:
            state_ev = (profile["win_rate"] * profile["avg_win_cents"]
                       - (1 - profile["win_rate"]) * abs(profile["avg_loss_cents"])
                       - fees_cents)

        ev += p_k * state_ev
        confidence += p_k * min(count / 50.0, 1.0)

    return ev, confidence


def position_size(
    ev_cents: float,
    confidence: float,
    bankroll: float,
    max_contracts: int = 3,
) -> int:
    """Kelly-criterion position sizing scaled by regime confidence.

    Returns number of contracts (0 = don't trade).
    """
    if ev_cents <= 0 or confidence < 0.3:
        return 0

    # Half-Kelly with confidence scaling
    kelly_frac = (ev_cents / 100) * confidence * 0.5
    position_value = bankroll * kelly_frac
    contracts = min(int(position_value / 100), max_contracts)
    return max(contracts, 0)


def evaluate_contract_with_regime(
    regime_posterior: List[float],
    state_profiles: List[Dict],
    spot_price: float,
    strike_price: float,
    yes_price_cents: float,
    no_price_cents: float,
    time_to_expiry_secs: float,
    contract_volume: int,
    bid_ask_spread_cents: float,
    log_normal_prob: float,
    bankroll: float = 1000,
) -> ContractEvaluation:
    """Full contract evaluation using regime posteriors.

    Combines the log-normal probability estimate with regime context
    to produce a fair value and trade recommendation.
    """
    reasons = []

    # Don't evaluate with <2 minutes remaining
    if time_to_expiry_secs < 120:
        return ContractEvaluation(
            fair_prob=log_normal_prob, edge_yes_cents=0, edge_no_cents=0,
            ev_cents=0, confidence=0, regime_entropy=0,
            recommendation="skip", position_size=0,
            reasons=["<2 min to expiry"],
        )

    # Regime entropy
    entropy = 0.0
    for p in regime_posterior:
        if p > 1e-10:
            entropy -= p * math.log(p)
    max_entropy = math.log(max(len(regime_posterior), 2))

    # Don't trade during regime transitions (high entropy)
    if entropy > 0.8 * max_entropy:
        return ContractEvaluation(
            fair_prob=log_normal_prob, edge_yes_cents=0, edge_no_cents=0,
            ev_cents=0, confidence=0, regime_entropy=entropy,
            recommendation="skip", position_size=0,
            reasons=[f"Regime uncertain (entropy={entropy:.2f}/{max_entropy:.2f})"],
        )

    # V1 fair value: use log-normal estimate directly
    # (V1.5 will replace with regime-conditioned logistic regression)
    fair_prob = log_normal_prob
    fair_yes = round(fair_prob * 100, 1)
    fair_no = round((1 - fair_prob) * 100, 1)

    edge_yes = fair_yes - yes_price_cents
    edge_no = fair_no - no_price_cents

    # Determine trade direction — only buy the winning side
    if fair_prob > 0.52 and edge_yes >= 5:
        side = "yes"
        edge = edge_yes
    elif fair_prob < 0.48 and edge_no >= 5:
        side = "no"
        edge = edge_no
    else:
        return ContractEvaluation(
            fair_prob=fair_prob, edge_yes_cents=edge_yes, edge_no_cents=edge_no,
            ev_cents=0, confidence=0, regime_entropy=entropy,
            recommendation="skip", position_size=0,
            reasons=[f"No edge (P={fair_prob:.0%}, YES edge={edge_yes:+.0f}c, NO edge={edge_no:+.0f}c)"],
        )

    # Compute posterior-weighted EV
    ev, confidence = compute_regime_ev(regime_posterior, state_profiles, edge)

    reasons.append(f"P={fair_prob:.0%}, edge={edge:+.0f}c, EV={ev:.1f}c, conf={confidence:.2f}")
    reasons.append(f"Regime entropy={entropy:.2f}")

    if ev <= 0:
        return ContractEvaluation(
            fair_prob=fair_prob, edge_yes_cents=edge_yes, edge_no_cents=edge_no,
            ev_cents=ev, confidence=confidence, regime_entropy=entropy,
            recommendation="skip", position_size=0,
            reasons=reasons + [f"Negative EV ({ev:.1f}c)"],
        )

    if confidence < 0.3:
        return ContractEvaluation(
            fair_prob=fair_prob, edge_yes_cents=edge_yes, edge_no_cents=edge_no,
            ev_cents=ev, confidence=confidence, regime_entropy=entropy,
            recommendation="skip", position_size=0,
            reasons=reasons + [f"Low confidence ({confidence:.2f})"],
        )

    size = position_size(ev, confidence, bankroll)
    rec = f"buy_{side}" if size > 0 else "skip"

    return ContractEvaluation(
        fair_prob=fair_prob,
        edge_yes_cents=edge_yes,
        edge_no_cents=edge_no,
        ev_cents=ev,
        confidence=confidence,
        regime_entropy=entropy,
        recommendation=rec,
        position_size=size,
        reasons=reasons,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot && python -m pytest tests/test_hmm_contract.py -v`

Expected: All 7 PASS.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot
git add src/hmm_contract.py tests/test_hmm_contract.py
git commit -m "feat: add contract opportunity model — regime-conditioned fair value + EV sizing

Posterior-weighted expected value across all states, Kelly position
sizing with confidence scaling, entropy-based regime transition
detection.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Shadow Tracker

Log shadow predictions, resolve outcomes, compute rolling comparison metrics.

**Files:**
- Create: `src/hmm_shadow.py`
- Test: `tests/test_hmm_shadow.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_hmm_shadow.py`:

```python
"""Tests for HMM shadow tracker."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from db import TradingDB
from hmm_shadow import ShadowTracker


def _make_db():
    return TradingDB(db_path=tempfile.mktemp(suffix=".db"))


def test_record_and_resolve():
    db = _make_db()
    tracker = ShadowTracker(db)

    pred_id = tracker.record_prediction(
        asset="BTC", ticker="KXBTC15M-TEST",
        regime_posterior=[0.3, 0.7], regime_entropy=0.6,
        fair_prob=0.65, market_price=55, edge_cents=10,
        ev_cents=5, confidence=0.8, recommendation="buy_yes",
        position_size=1,
    )
    assert pred_id > 0

    tracker.resolve_prediction(pred_id, outcome="win", pnl_cents=44)

    metrics = tracker.get_rolling_metrics(asset="BTC", days=7)
    assert metrics["trade_count"] == 1
    assert metrics["win_rate"] == 1.0
    assert metrics["total_pnl_cents"] == 44


def test_rolling_metrics_empty():
    db = _make_db()
    tracker = ShadowTracker(db)
    metrics = tracker.get_rolling_metrics(asset="BTC", days=7)
    assert metrics["trade_count"] == 0
    assert metrics["win_rate"] == 0.0


def test_comparison_metrics():
    db = _make_db()
    tracker = ShadowTracker(db)

    # Add some shadow predictions
    for i in range(5):
        pid = tracker.record_prediction(
            asset="BTC", ticker=f"KXBTC15M-T{i}",
            regime_posterior=[0.5, 0.5], regime_entropy=0.69,
            fair_prob=0.6, market_price=50, edge_cents=10,
            ev_cents=5, confidence=0.7, recommendation="buy_yes",
            position_size=1,
        )
        outcome = "win" if i < 3 else "loss"
        pnl = 40 if outcome == "win" else -50
        tracker.resolve_prediction(pid, outcome=outcome, pnl_cents=pnl)

    metrics = tracker.get_rolling_metrics(asset="BTC", days=7)
    assert metrics["trade_count"] == 5
    assert metrics["win_rate"] == 0.6
    assert metrics["total_pnl_cents"] == 3 * 40 + 2 * (-50)  # 20


def test_skip_predictions_not_counted():
    db = _make_db()
    tracker = ShadowTracker(db)

    # Record a skip — should not count in metrics
    tracker.record_prediction(
        asset="BTC", ticker="KXBTC15M-SKIP",
        regime_posterior=[0.5, 0.5], regime_entropy=0.69,
        fair_prob=0.5, market_price=50, edge_cents=0,
        ev_cents=0, confidence=0.3, recommendation="skip",
        position_size=0,
    )

    metrics = tracker.get_rolling_metrics(asset="BTC", days=7)
    assert metrics["trade_count"] == 0  # Skips don't count
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot && python -m pytest tests/test_hmm_shadow.py -v`

Expected: FAIL — `hmm_shadow` module doesn't exist.

- [ ] **Step 3: Implement shadow tracker**

Create `src/hmm_shadow.py`:

```python
"""HMM Shadow Tracker — log predictions, track outcomes, compare vs live.

Records what the HMM system would do on each trading cycle without
executing. Resolves outcomes when contracts settle. Computes rolling
metrics for promotion gate comparison.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ShadowTracker:
    """Tracks HMM shadow predictions and their outcomes."""

    def __init__(self, db):
        self.db = db

    def record_prediction(
        self, asset: str, ticker: str,
        regime_posterior: List[float], regime_entropy: float,
        fair_prob: float, market_price: float, edge_cents: float,
        ev_cents: float, confidence: float, recommendation: str,
        position_size: int,
    ) -> int:
        """Record a shadow prediction. Returns prediction ID."""
        top_state = int(max(range(len(regime_posterior)),
                           key=lambda i: regime_posterior[i]))
        top_prob = regime_posterior[top_state]

        return self.db.record_shadow_prediction(
            asset=asset, ticker=ticker,
            regime_posterior=json.dumps(regime_posterior),
            regime_entropy=regime_entropy,
            top_state=top_state, top_state_prob=top_prob,
            fair_prob=fair_prob, market_price=market_price,
            edge_cents=edge_cents, ev_cents=ev_cents,
            confidence=confidence, recommendation=recommendation,
            position_size=position_size,
        )

    def resolve_prediction(self, prediction_id: int, outcome: str, pnl_cents: float = None):
        """Resolve a shadow prediction with its outcome."""
        self.db.resolve_shadow_prediction(prediction_id, outcome=outcome, pnl_cents=pnl_cents)

    def get_rolling_metrics(self, asset: str = None, days: int = 7) -> Dict[str, Any]:
        """Compute rolling performance metrics for shadow predictions.

        Only counts predictions where recommendation != 'skip' and
        outcome is resolved.
        """
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        preds = self.db.get_shadow_predictions(
            asset=asset, since=since, resolved_only=True,
        )

        # Filter to actual trade recommendations (not skips)
        trades = [p for p in preds if p.get("recommendation") not in ("skip", None)]

        if not trades:
            return {
                "trade_count": 0, "win_rate": 0.0, "total_pnl_cents": 0,
                "avg_pnl_per_trade": 0.0, "max_drawdown_cents": 0,
                "wins": 0, "losses": 0,
            }

        wins = sum(1 for t in trades if t.get("outcome") == "win")
        losses = sum(1 for t in trades if t.get("outcome") == "loss")
        total_pnl = sum(t.get("pnl_cents", 0) or 0 for t in trades)

        # Max drawdown: worst peak-to-trough in cumulative P&L
        cumulative = 0
        peak = 0
        max_dd = 0
        for t in sorted(trades, key=lambda x: x.get("timestamp", "")):
            cumulative += t.get("pnl_cents", 0) or 0
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        return {
            "trade_count": len(trades),
            "win_rate": wins / len(trades) if trades else 0.0,
            "total_pnl_cents": total_pnl,
            "avg_pnl_per_trade": total_pnl / len(trades) if trades else 0.0,
            "max_drawdown_cents": max_dd,
            "wins": wins,
            "losses": losses,
        }

    def format_report(self, days: int = 7) -> str:
        """Format a summary report for Telegram."""
        lines = [f"HMM Shadow ({days}d rolling):"]
        for asset in ["BTC", "ETH", "SOL"]:
            m = self.get_rolling_metrics(asset=asset, days=days)
            if m["trade_count"] == 0:
                lines.append(f"  {asset}: no trades")
            else:
                lines.append(
                    f"  {asset}: {m['trade_count']} trades, "
                    f"WR={m['win_rate']:.0%}, "
                    f"PnL={m['total_pnl_cents']:+.0f}c, "
                    f"DD={m['max_drawdown_cents']:.0f}c"
                )
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot && python -m pytest tests/test_hmm_shadow.py -v`

Expected: All 4 PASS.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot
git add src/hmm_shadow.py tests/test_hmm_shadow.py
git commit -m "feat: add shadow tracker — prediction logging and rolling metrics

Records shadow predictions, resolves outcomes, computes rolling
win rate, PnL, and max drawdown for promotion gate comparison.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Graduation Controller

Staged rollout logic, promotion gate checks, kill switch.

**Files:**
- Create: `src/hmm_graduation.py`
- Test: `tests/test_hmm_graduation.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_hmm_graduation.py`:

```python
"""Tests for HMM graduation controller."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hmm_graduation import GraduationController, Stage


def test_initial_stage():
    controller = GraduationController()
    assert controller.stage == Stage.COLLECTION


def test_collection_to_shadow():
    controller = GraduationController()
    controller.stage = Stage.COLLECTION
    controller.days_in_stage = 4  # >3 days
    controller.has_enough_observations = True
    
    assert controller.can_advance() == True
    controller.advance()
    assert controller.stage == Stage.SHADOW


def test_shadow_promotion_gate_passes():
    controller = GraduationController()
    controller.stage = Stage.SHADOW
    controller.days_in_stage = 8

    shadow_metrics = {
        "trade_count": 35, "win_rate": 0.65, "total_pnl_cents": 200,
        "avg_pnl_per_trade": 5.7, "max_drawdown_cents": 50,
    }
    live_metrics = {
        "trade_count": 30, "win_rate": 0.50, "avg_pnl_per_trade": 1.0,
    }

    result = controller.check_promotion_gate(
        shadow_metrics, live_metrics, bankroll=1000, model_stable=True,
    )
    assert result["passes"] == True


def test_shadow_promotion_gate_fails_low_count():
    controller = GraduationController()
    controller.stage = Stage.SHADOW
    controller.days_in_stage = 8

    shadow_metrics = {
        "trade_count": 10, "win_rate": 0.80, "total_pnl_cents": 100,
        "avg_pnl_per_trade": 10, "max_drawdown_cents": 20,
    }
    live_metrics = {"trade_count": 30, "win_rate": 0.50, "avg_pnl_per_trade": 1.0}

    result = controller.check_promotion_gate(
        shadow_metrics, live_metrics, bankroll=1000, model_stable=True,
    )
    assert result["passes"] == False
    assert "trade_count" in str(result["failures"])


def test_kill_switch_drawdown():
    controller = GraduationController()
    controller.stage = Stage.SMALL_CAP_LIVE

    should_kill = controller.check_kill_switch(
        daily_pnl_cents=-250, bankroll=1000, consecutive_losses=2,
        model_stable=True,
    )
    assert should_kill == True  # 25% > 20% limit


def test_kill_switch_consecutive_losses():
    controller = GraduationController()
    controller.stage = Stage.SMALL_CAP_LIVE

    should_kill = controller.check_kill_switch(
        daily_pnl_cents=-50, bankroll=1000, consecutive_losses=5,
        model_stable=True,
    )
    assert should_kill == True


def test_kill_switch_ok():
    controller = GraduationController()
    controller.stage = Stage.SMALL_CAP_LIVE

    should_kill = controller.check_kill_switch(
        daily_pnl_cents=-50, bankroll=1000, consecutive_losses=2,
        model_stable=True,
    )
    assert should_kill == False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot && python -m pytest tests/test_hmm_graduation.py -v`

Expected: FAIL — `hmm_graduation` module doesn't exist.

- [ ] **Step 3: Implement graduation controller**

Create `src/hmm_graduation.py`:

```python
"""HMM Graduation Controller — staged rollout, promotion gates, kill switch.

Manages the 5-stage rollout:
  0. COLLECTION — recording observations only
  1. SHADOW — predictions logged, not executed
  2. PAPER — orders placed and immediately cancelled (pipeline test)
  3. SMALL_CAP_LIVE — max 1 contract, max 3 positions
  4. FULL_LIVE — replace current strategy entirely
"""

import json
import logging
from enum import IntEnum
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class Stage(IntEnum):
    COLLECTION = 0
    SHADOW = 1
    PAPER = 2
    SMALL_CAP_LIVE = 3
    FULL_LIVE = 4


# Promotion gate thresholds
MIN_SHADOW_DAYS = 7
MIN_TRADE_COUNT = 30
MAX_DRAWDOWN_PCT = 0.15  # 15% of bankroll
MIN_EV_ADVANTAGE_CENTS = 1.0  # Must beat live by at least 1c/trade

# Kill switch thresholds
KILL_DAILY_LOSS_PCT = 0.20  # 20% of bankroll in one day
KILL_CONSECUTIVE_LOSSES = 5


class GraduationController:
    """Manages HMM system stage progression and safety controls."""

    def __init__(self):
        self.stage = Stage.COLLECTION
        self.days_in_stage = 0
        self.has_enough_observations = False

    def can_advance(self) -> bool:
        """Check if basic time requirements are met to advance."""
        if self.stage == Stage.COLLECTION:
            return self.days_in_stage >= 3 and self.has_enough_observations
        if self.stage == Stage.SHADOW:
            return self.days_in_stage >= MIN_SHADOW_DAYS
        if self.stage == Stage.PAPER:
            return self.days_in_stage >= 3
        if self.stage == Stage.SMALL_CAP_LIVE:
            return self.days_in_stage >= 7
        return False

    def advance(self):
        """Move to the next stage."""
        if self.stage < Stage.FULL_LIVE:
            old = self.stage
            self.stage = Stage(self.stage + 1)
            self.days_in_stage = 0
            logger.info(f"HMM stage advanced: {old.name} → {self.stage.name}")

    def demote(self):
        """Revert to shadow mode."""
        old = self.stage
        self.stage = Stage.SHADOW
        self.days_in_stage = 0
        logger.warning(f"HMM DEMOTED: {old.name} → SHADOW")

    def check_promotion_gate(
        self,
        shadow_metrics: Dict[str, Any],
        live_metrics: Dict[str, Any],
        bankroll: float,
        model_stable: bool,
    ) -> Dict[str, Any]:
        """Check all promotion criteria. Returns pass/fail with details."""
        failures = []

        # 1. Minimum evaluation window
        if self.days_in_stage < MIN_SHADOW_DAYS:
            failures.append(f"days_in_stage={self.days_in_stage} < {MIN_SHADOW_DAYS}")

        # 2. Minimum trade count
        if shadow_metrics.get("trade_count", 0) < MIN_TRADE_COUNT:
            failures.append(f"trade_count={shadow_metrics.get('trade_count', 0)} < {MIN_TRADE_COUNT}")

        # 3. Positive EV after fees
        if shadow_metrics.get("avg_pnl_per_trade", 0) <= 0:
            failures.append(f"avg_pnl={shadow_metrics.get('avg_pnl_per_trade', 0):.1f}c <= 0")

        # 4. Must beat live by at least 1c/trade
        shadow_ev = shadow_metrics.get("avg_pnl_per_trade", 0)
        live_ev = live_metrics.get("avg_pnl_per_trade", 0)
        if shadow_ev - live_ev < MIN_EV_ADVANTAGE_CENTS:
            failures.append(f"EV advantage={shadow_ev - live_ev:.1f}c < {MIN_EV_ADVANTAGE_CENTS}c")

        # 5. Shadow win rate > live win rate
        if shadow_metrics.get("win_rate", 0) <= live_metrics.get("win_rate", 0):
            failures.append(f"shadow_wr={shadow_metrics.get('win_rate', 0):.0%} <= live_wr={live_metrics.get('win_rate', 0):.0%}")

        # 6. Max drawdown check
        max_dd = shadow_metrics.get("max_drawdown_cents", 0)
        if max_dd > bankroll * MAX_DRAWDOWN_PCT:
            failures.append(f"drawdown={max_dd:.0f}c > {bankroll * MAX_DRAWDOWN_PCT:.0f}c")

        # 7. Model stability
        if not model_stable:
            failures.append("model_unstable")

        passes = len(failures) == 0
        return {
            "passes": passes,
            "failures": failures,
            "criteria_met": 7 - len(failures),
            "criteria_total": 7,
        }

    def check_kill_switch(
        self,
        daily_pnl_cents: float,
        bankroll: float,
        consecutive_losses: int,
        model_stable: bool,
    ) -> bool:
        """Check if kill switch should trigger. Returns True to kill."""
        if self.stage < Stage.SMALL_CAP_LIVE:
            return False  # Kill switch only applies to live stages

        if abs(daily_pnl_cents) > bankroll * KILL_DAILY_LOSS_PCT * 100:
            logger.warning(f"KILL SWITCH: daily loss {daily_pnl_cents}c > "
                          f"{bankroll * KILL_DAILY_LOSS_PCT * 100:.0f}c")
            return True

        if consecutive_losses >= KILL_CONSECUTIVE_LOSSES:
            logger.warning(f"KILL SWITCH: {consecutive_losses} consecutive losses")
            return True

        if not model_stable:
            logger.warning("KILL SWITCH: model unstable")
            return True

        return False

    def get_max_contracts(self) -> int:
        """Max contracts per trade at current stage."""
        if self.stage <= Stage.SHADOW:
            return 0
        if self.stage == Stage.PAPER:
            return 0  # Orders cancelled immediately
        if self.stage == Stage.SMALL_CAP_LIVE:
            return 1
        return 3  # FULL_LIVE

    def get_max_positions(self) -> int:
        """Max simultaneous positions at current stage."""
        if self.stage <= Stage.PAPER:
            return 0
        if self.stage == Stage.SMALL_CAP_LIVE:
            return 3
        return 5  # FULL_LIVE

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "stage": self.stage.value,
            "stage_name": self.stage.name,
            "days_in_stage": self.days_in_stage,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'GraduationController':
        """Deserialize from dict."""
        c = cls()
        c.stage = Stage(d.get("stage", 0))
        c.days_in_stage = d.get("days_in_stage", 0)
        return c
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot && python -m pytest tests/test_hmm_graduation.py -v`

Expected: All 7 PASS.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot
git add src/hmm_graduation.py tests/test_hmm_graduation.py
git commit -m "feat: add graduation controller — 5-stage rollout with kill switch

Stages: collection → shadow → paper → small-cap → full-live.
Promotion gate: 7d window, 30 trades, positive EV, beat live,
drawdown < 15%. Kill switch: 20% daily loss, 5 consecutive losses,
or model instability.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Integration — Wire Into Trading Loop

Connect the HMM system to the existing bot: start observation pipeline, call shadow tracker each cycle, add HMM retrain to daily schedule.

**Files:**
- Modify: `src/main.py`
- Modify: `src/trader.py`
- Modify: `src/retrain.py`

- [ ] **Step 1: Add observation pipeline to main.py trading loop**

In `src/main.py`, after the retrain scheduler starts (around line 76), add the observation pipeline:

```python
        # Start HMM observation pipeline
        from hmm_observations import ObservationPipeline
        hmm_pipeline = ObservationPipeline(trader.db)
        hmm_observation_counter = 0
```

Then in the main `while True` loop (around line 85-88), add observation recording every 3rd cycle (~60s at 20s intervals):

```python
        while True:
            logger.info("Running trading strategy with real-time market data")
            trader.run_trading_strategy()

            # Record HMM observations every ~60s (every 3rd cycle at 20s interval)
            hmm_observation_counter += 1
            if hmm_observation_counter >= 3:
                hmm_observation_counter = 0
                try:
                    hmm_pipeline.record_all_assets()
                except Exception as e:
                    logger.debug(f"HMM observation error: {e}")

            time.sleep(TRADE_INTERVAL_SECONDS)
```

- [ ] **Step 2: Add shadow tracking to trader.py**

In `src/trader.py`, in the `run_trading_strategy` method, after the existing trade decision logic, add shadow prediction recording. Import at the top of the file:

```python
from hmm_shadow import ShadowTracker
from hmm_regime import RegimeEngine
from hmm_contract import evaluate_contract_with_regime
```

In `Trader.__init__`, add:

```python
        self.hmm_shadow = ShadowTracker(self.db)
        self.hmm_regime = RegimeEngine(self.db)
```

In `run_trading_strategy`, after `self.check_exits(markets)` and before the trade decision, add:

```python
            # HMM shadow prediction (runs in parallel with existing strategy)
            try:
                for asset in ["BTC", "ETH", "SOL"]:
                    posterior = self.hmm_regime.get_current_posterior(asset)
                    if posterior is None:
                        continue
                    # Find active 15M contract for this asset
                    prefix = f"KX{asset}15M"
                    active_15m = [m for m in markets if m.get('ticker', '').upper().startswith(prefix)]
                    if not active_15m:
                        continue
                    contract = active_15m[0]
                    ticker = contract.get('ticker', '')
                    
                    from price_predictor import estimate_strike_probability, compute_realized_volatility
                    strike = contract.get('strike_price') or _parse_strike_from_ticker(ticker)
                    spot = None
                    if spot_prices and asset in spot_prices:
                        spot = spot_prices[asset].get("price")
                    if not spot or not strike or strike <= 0:
                        continue

                    yes_price = _get_market_price_cents(contract)
                    no_price = _get_no_price_cents(contract)
                    exp_time = contract.get('expected_expiration_time') or contract.get('expiration_time')
                    
                    # Compute time to expiry
                    from datetime import datetime, timezone
                    tte_secs = 900  # default
                    if exp_time:
                        try:
                            exp_dt = datetime.fromisoformat(exp_time.replace("Z", "+00:00"))
                            tte_secs = max(0, (exp_dt - datetime.now(timezone.utc)).total_seconds())
                        except (ValueError, TypeError):
                            pass

                    vol = compute_realized_volatility(self.db, asset)
                    base_prob = estimate_strike_probability(spot, strike, tte_secs / 3600, vol)
                    
                    # Get state profiles (empty until enough data)
                    n_states = len(posterior)
                    state_profiles = [
                        {"win_rate": 0.5, "avg_win_cents": 0, "avg_loss_cents": 0, "trade_count": 0}
                        for _ in range(n_states)
                    ]
                    
                    bid_ask = ((contract.get('yes_ask') or 0) - (contract.get('yes_bid') or 0)) * 100

                    hmm_eval = evaluate_contract_with_regime(
                        regime_posterior=posterior,
                        state_profiles=state_profiles,
                        spot_price=spot, strike_price=strike,
                        yes_price_cents=yes_price, no_price_cents=no_price,
                        time_to_expiry_secs=tte_secs,
                        contract_volume=int(float(contract.get('volume', 0) or 0)),
                        bid_ask_spread_cents=bid_ask,
                        log_normal_prob=base_prob,
                        bankroll=self.risk_manager.current_bankroll,
                    )

                    entropy = self.hmm_regime.get_regime_entropy(posterior)
                    self.hmm_shadow.record_prediction(
                        asset=asset, ticker=ticker,
                        regime_posterior=posterior, regime_entropy=entropy,
                        fair_prob=hmm_eval.fair_prob, market_price=yes_price,
                        edge_cents=hmm_eval.edge_yes_cents, ev_cents=hmm_eval.ev_cents,
                        confidence=hmm_eval.confidence,
                        recommendation=hmm_eval.recommendation,
                        position_size=hmm_eval.position_size,
                    )
            except Exception as e:
                self.logger.debug(f"HMM shadow error: {e}")
```

- [ ] **Step 3: Add HMM retrain to daily retrain schedule**

In `src/retrain.py`, at the end of the `retrain()` function (before `return new_params`), add:

```python
    # --- HMM Regime Model Retrain ---
    try:
        from hmm_regime import RegimeEngine
        from hmm_shadow import ShadowTracker
        
        hmm_db_path = DB_PATH
        import sqlite3
        # Use a temporary DB connection for HMM retrain
        from db import TradingDB
        hmm_db = TradingDB(db_path=hmm_db_path)
        
        engine = RegimeEngine(hmm_db)
        results = engine.fit_all_assets()
        
        shadow = ShadowTracker(hmm_db)
        shadow_report = shadow.format_report(days=7)
        
        logger.info("--- HMM Regime Retrain ---")
        for asset, result in results.items():
            if result:
                logger.info(f"  {asset}: K={result['n_states']}, BIC={result['bic']:.1f}, "
                           f"stability={result['stability_flags']}")
            else:
                logger.info(f"  {asset}: insufficient data")
        logger.info(shadow_report)
    except ImportError:
        logger.info("HMM modules not available — skipping regime retrain")
    except Exception as e:
        logger.error(f"HMM retrain error: {e}")
```

- [ ] **Step 4: Compile check all modified files**

Run: `cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot && python -c "import py_compile; [py_compile.compile(f'src/{f}', doraise=True) for f in ['main.py', 'trader.py', 'retrain.py', 'hmm_observations.py', 'hmm_regime.py', 'hmm_contract.py', 'hmm_shadow.py', 'hmm_graduation.py']]; print('All OK')"`

Expected: "All OK"

- [ ] **Step 5: Run all HMM tests**

Run: `cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot && python -m pytest tests/test_hmm_*.py tests/test_decision_logic.py -v`

Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot
git add src/main.py src/trader.py src/retrain.py
git commit -m "feat: integrate HMM system into trading loop

Observation pipeline records 1-min bars every 60s.
Shadow tracker logs HMM predictions each cycle.
Daily retrain fits per-asset HMMs and reports shadow performance.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Final Integration Test and Push

- [ ] **Step 1: Run full test suite**

Run: `cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot && python -m pytest tests/ -v --ignore=tests/test_system.py --ignore=tests/test_phase1_integration.py --ignore=tests/test_phase1_strategies.py --ignore=tests/test_phase2_phase3.py --ignore=tests/test_phase4_features.py`

Expected: All tests PASS.

- [ ] **Step 2: Compile check all files**

Run: `cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot && python -c "import py_compile; import glob; files = glob.glob('src/hmm_*.py') + glob.glob('src/main.py') + glob.glob('src/trader.py') + glob.glob('src/retrain.py') + glob.glob('src/db.py') + glob.glob('src/price_predictor.py'); [py_compile.compile(f, doraise=True) for f in files]; print(f'All {len(files)} files OK')"`

Expected: All files compile.

- [ ] **Step 3: Push feature branch**

```bash
cd C:/Users/CadeYounger/Kalshi-Quant-TeleBot
git push -u origin feature/hmm-regime-detection
```

Expected: Branch pushed. Does NOT auto-deploy (not main branch). Ready for PR review.
