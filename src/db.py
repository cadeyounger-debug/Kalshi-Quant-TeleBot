"""
SQLite persistence layer for the Kalshi crypto trading bot.

Thread-safe database module that records market snapshots, trade decisions,
executed trades, and news sentiment data. All tables include an `asset` column
for filtering by BTC, ETH, or SOL.
"""

import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# Maps ticker prefixes to asset names.
_TICKER_ASSET_MAP = {
    "KXBTC": "BTC",
    "KXETH": "ETH",
    "KXSOL": "SOL",
}

_TICKER_PATTERN = re.compile(r"^(KXBTC|KXETH|KXSOL)")


def extract_asset(ticker: str) -> str:
    """Extract asset name (BTC/ETH/SOL) from a Kalshi market ticker."""
    if not ticker:
        return "UNKNOWN"
    m = _TICKER_PATTERN.match(ticker.upper())
    if m:
        return _TICKER_ASSET_MAP[m.group(1)]
    return "UNKNOWN"


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS market_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,
    asset           TEXT NOT NULL,
    title           TEXT,
    yes_bid         REAL,
    yes_ask         REAL,
    no_bid          REAL,
    no_ask          REAL,
    volume          INTEGER,
    strike_price    REAL,
    spot_price      REAL,
    expiration_time TEXT,
    timestamp       TEXT NOT NULL
);

-- Add no_bid/no_ask columns if upgrading from older schema


CREATE TABLE IF NOT EXISTS trade_decisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset           TEXT NOT NULL,
    market_ticker   TEXT NOT NULL,
    strategy        TEXT,
    direction       TEXT,
    confidence      REAL,
    sentiment_score REAL,
    should_trade    INTEGER,
    timestamp       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset           TEXT NOT NULL,
    market_ticker   TEXT NOT NULL,
    side            TEXT,
    quantity        INTEGER,
    price           REAL,
    strategy        TEXT,
    order_result    TEXT,
    pnl             REAL,
    edge_cents      REAL,
    predicted_prob  REAL,
    fair_value      REAL,
    spot_price      REAL,
    strike_price    REAL,
    timestamp       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS news_sentiment (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    asset               TEXT NOT NULL,
    asset_keywords_used TEXT,
    overall_sentiment   REAL,
    confidence          REAL,
    article_count       INTEGER,
    positive_count      INTEGER,
    negative_count      INTEGER,
    neutral_count       INTEGER,
    timestamp           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS crypto_prices (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset           TEXT NOT NULL,
    price_usd       REAL NOT NULL,
    change_24h_pct  REAL,
    timestamp       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS open_positions (
    market_id       TEXT PRIMARY KEY,
    asset           TEXT NOT NULL,
    side            TEXT NOT NULL,
    quantity        INTEGER NOT NULL,
    entry_price     REAL NOT NULL,
    strategy        TEXT,
    trade_id        TEXT,
    title           TEXT,
    expiration_time TEXT,
    opened_at       REAL,
    timestamp       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_asset_ts   ON market_snapshots(asset, timestamp);
CREATE INDEX IF NOT EXISTS idx_decisions_asset_ts   ON trade_decisions(asset, timestamp);
CREATE INDEX IF NOT EXISTS idx_decisions_strategy    ON trade_decisions(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_asset_ts       ON trades(asset, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_strategy       ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_sentiment_asset_ts    ON news_sentiment(asset, timestamp);
CREATE INDEX IF NOT EXISTS idx_crypto_prices_asset_ts ON crypto_prices(asset, timestamp);

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
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TradingDB:
    """Thread-safe SQLite database for trading bot persistence."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.environ.get(
                "TRADING_DB_PATH",
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trading_bot.db"),
            )
        self._db_path = db_path
        self._lock = threading.Lock()

        # Ensure the parent directory exists.
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        # Create tables on init.
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            # Migrate: add no_bid/no_ask if missing
            try:
                conn.execute("SELECT no_bid FROM market_snapshots LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE market_snapshots ADD COLUMN no_bid REAL")
                conn.execute("ALTER TABLE market_snapshots ADD COLUMN no_ask REAL")

            # Migrate: add strike_price, spot_price, expiration_time if missing
            for col in ("strike_price REAL", "spot_price REAL", "expiration_time TEXT"):
                col_name = col.split()[0]
                try:
                    conn.execute(f"SELECT {col_name} FROM market_snapshots LIMIT 1")
                except sqlite3.OperationalError:
                    conn.execute(f"ALTER TABLE market_snapshots ADD COLUMN {col}")

            # Migrate: create crypto_prices table if missing (for existing DBs)
            conn.execute(
                "CREATE TABLE IF NOT EXISTS crypto_prices ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "asset TEXT NOT NULL, "
                "price_usd REAL NOT NULL, "
                "change_24h_pct REAL, "
                "timestamp TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_crypto_prices_asset_ts "
                "ON crypto_prices(asset, timestamp)"
            )

            # Migrate: add edge/prediction columns to trades
            for col in ("edge_cents REAL", "predicted_prob REAL", "fair_value REAL",
                         "spot_price REAL", "strike_price REAL"):
                col_name = col.split()[0]
                try:
                    conn.execute(f"SELECT {col_name} FROM trades LIMIT 1")
                except sqlite3.OperationalError:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {col}")

            # Migrate: create open_positions table if missing
            conn.execute(
                "CREATE TABLE IF NOT EXISTS open_positions ("
                "market_id TEXT PRIMARY KEY, "
                "asset TEXT NOT NULL, "
                "side TEXT NOT NULL, "
                "quantity INTEGER NOT NULL, "
                "entry_price REAL NOT NULL, "
                "strategy TEXT, "
                "trade_id TEXT, "
                "title TEXT, "
                "expiration_time TEXT, "
                "opened_at REAL, "
                "timestamp TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ------------------------------------------------------------------
    # Record methods
    # ------------------------------------------------------------------

    def record_market_snapshot(
        self,
        ticker: str,
        title: str = None,
        yes_bid: float = None,
        yes_ask: float = None,
        no_bid: float = None,
        no_ask: float = None,
        volume: int = None,
        strike_price: float = None,
        spot_price: float = None,
        expiration_time: str = None,
        asset: str = None,
        timestamp: str = None,
    ) -> int:
        asset = asset or extract_asset(ticker)
        timestamp = timestamp or _now_iso()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO market_snapshots "
                    "(ticker, asset, title, yes_bid, yes_ask, no_bid, no_ask, volume, "
                    "strike_price, spot_price, expiration_time, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ticker, asset, title, yes_bid, yes_ask, no_bid, no_ask, volume,
                     strike_price, spot_price, expiration_time, timestamp),
                )
                return cur.lastrowid

    def record_trade_decision(
        self,
        market_ticker: str,
        strategy: str = None,
        direction: str = None,
        confidence: float = None,
        sentiment_score: float = None,
        should_trade: bool = None,
        asset: str = None,
        timestamp: str = None,
    ) -> int:
        asset = asset or extract_asset(market_ticker)
        timestamp = timestamp or _now_iso()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO trade_decisions "
                    "(asset, market_ticker, strategy, direction, confidence, sentiment_score, should_trade, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (asset, market_ticker, strategy, direction, confidence,
                     sentiment_score, int(should_trade) if should_trade is not None else None,
                     timestamp),
                )
                return cur.lastrowid

    def record_trade(
        self,
        market_ticker: str,
        side: str = None,
        quantity: int = None,
        price: float = None,
        strategy: str = None,
        order_result: str = None,
        pnl: float = None,
        edge_cents: float = None,
        predicted_prob: float = None,
        fair_value: float = None,
        spot_price: float = None,
        strike_price: float = None,
        asset: str = None,
        timestamp: str = None,
    ) -> int:
        asset = asset or extract_asset(market_ticker)
        timestamp = timestamp or _now_iso()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO trades "
                    "(asset, market_ticker, side, quantity, price, strategy, order_result, pnl, "
                    "edge_cents, predicted_prob, fair_value, spot_price, strike_price, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (asset, market_ticker, side, quantity, price, strategy,
                     order_result, pnl, edge_cents, predicted_prob, fair_value,
                     spot_price, strike_price, timestamp),
                )
                return cur.lastrowid

    def record_news_sentiment(
        self,
        asset: str,
        asset_keywords_used: str = None,
        overall_sentiment: float = None,
        confidence: float = None,
        article_count: int = None,
        positive_count: int = None,
        negative_count: int = None,
        neutral_count: int = None,
        timestamp: str = None,
    ) -> int:
        timestamp = timestamp or _now_iso()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO news_sentiment "
                    "(asset, asset_keywords_used, overall_sentiment, confidence, "
                    "article_count, positive_count, negative_count, neutral_count, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (asset, asset_keywords_used, overall_sentiment, confidence,
                     article_count, positive_count, negative_count, neutral_count,
                     timestamp),
                )
                return cur.lastrowid

    def record_crypto_price(
        self,
        asset: str,
        price_usd: float,
        change_24h_pct: float = None,
        timestamp: str = None,
    ) -> int:
        """Record a crypto spot price snapshot."""
        timestamp = timestamp or _now_iso()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO crypto_prices (asset, price_usd, change_24h_pct, timestamp) "
                    "VALUES (?, ?, ?, ?)",
                    (asset.upper(), price_usd, change_24h_pct, timestamp),
                )
                return cur.lastrowid

    def get_crypto_prices(
        self,
        asset: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Query recorded crypto price snapshots."""
        sql, params = self._build_query(
            "SELECT * FROM crypto_prices", asset=asset, since=since
        )
        sql += " LIMIT ?"
        params.append(limit)
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def _build_query(
        self,
        base_sql: str,
        asset: Optional[str] = None,
        strategy: Optional[str] = None,
        since: Optional[str] = None,
    ) -> tuple:
        """Build a WHERE clause from optional filters. Returns (sql, params)."""
        clauses: List[str] = []
        params: List[Any] = []
        if asset:
            clauses.append("asset = ?")
            params.append(asset.upper())
        if strategy:
            clauses.append("strategy = ?")
            params.append(strategy)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        sql = base_sql
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp DESC"
        return sql, params

    def get_snapshots(
        self,
        asset: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        sql, params = self._build_query("SELECT * FROM market_snapshots", asset=asset, since=since)
        sql += " LIMIT ?"
        params.append(limit)
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]

    def get_trades(
        self,
        asset: Optional[str] = None,
        since: Optional[str] = None,
        strategy: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        sql, params = self._build_query(
            "SELECT * FROM trades", asset=asset, strategy=strategy, since=since
        )
        sql += " LIMIT ?"
        params.append(limit)
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]

    def get_decisions(
        self,
        asset: Optional[str] = None,
        strategy: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        sql, params = self._build_query(
            "SELECT * FROM trade_decisions", asset=asset, strategy=strategy, since=since
        )
        sql += " LIMIT ?"
        params.append(limit)
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]

    def get_sentiment(
        self,
        asset: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        sql, params = self._build_query("SELECT * FROM news_sentiment", asset=asset, since=since)
        sql += " LIMIT ?"
        params.append(limit)
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Open positions persistence
    # ------------------------------------------------------------------

    def save_position(self, market_id: str, position: Dict[str, Any]) -> None:
        """Save or update an open position."""
        asset = extract_asset(market_id)
        timestamp = _now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO open_positions "
                    "(market_id, asset, side, quantity, entry_price, strategy, "
                    "trade_id, title, expiration_time, opened_at, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (market_id, asset,
                     position.get("side", "yes"),
                     position.get("quantity", 1),
                     position.get("entry_price", 0),
                     position.get("strategy", ""),
                     position.get("trade_id", ""),
                     position.get("title", market_id),
                     position.get("expiration_time"),
                     position.get("opened_at"),
                     timestamp),
                )

    def delete_position(self, market_id: str) -> None:
        """Remove a closed position."""
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM open_positions WHERE market_id = ?", (market_id,))

    def load_positions(self) -> Dict[str, Dict[str, Any]]:
        """Load all open positions. Returns dict keyed by market_id."""
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute("SELECT * FROM open_positions").fetchall()
                positions = {}
                for r in rows:
                    row = dict(r)
                    mid = row.pop("market_id")
                    row.pop("asset", None)
                    row.pop("timestamp", None)
                    positions[mid] = row
                return positions

    # ------------------------------------------------------------------
    # HMM observation methods
    # ------------------------------------------------------------------

    def record_hmm_observation(
        self,
        asset: str,
        log_return_1m: float = None,
        log_return_5m: float = None,
        log_return_15m: float = None,
        realized_vol_15m: float = None,
        realized_vol_1h: float = None,
        vol_of_vol: float = None,
        momentum_r_sq: float = None,
        mean_reversion: float = None,
        bid_ask_spread: float = None,
        spread_vol: float = None,
        volume_1m: float = None,
        volume_accel: float = None,
        has_active_contract: int = 0,
        timestamp: str = None,
    ) -> int:
        """Insert an HMM feature observation row. Returns lastrowid."""
        asset = asset.upper()
        timestamp = timestamp or _now_iso()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO hmm_observations "
                    "(asset, timestamp, log_return_1m, log_return_5m, log_return_15m, "
                    "realized_vol_15m, realized_vol_1h, vol_of_vol, momentum_r_sq, "
                    "mean_reversion, bid_ask_spread, spread_vol, volume_1m, volume_accel, "
                    "has_active_contract) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (asset, timestamp, log_return_1m, log_return_5m, log_return_15m,
                     realized_vol_15m, realized_vol_1h, vol_of_vol, momentum_r_sq,
                     mean_reversion, bid_ask_spread, spread_vol, volume_1m, volume_accel,
                     has_active_contract),
                )
                return cur.lastrowid

    def get_hmm_observations(
        self,
        asset: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 50000,
    ) -> List[Dict[str, Any]]:
        """Query HMM observations with optional asset/since filters, ordered by timestamp ASC."""
        clauses: List[str] = []
        params: List[Any] = []
        if asset:
            clauses.append("asset = ?")
            params.append(asset.upper())
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        sql = "SELECT * FROM hmm_observations"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # HMM shadow prediction methods
    # ------------------------------------------------------------------

    def record_shadow_prediction(
        self,
        asset: str,
        ticker: str = None,
        regime_posterior: str = None,
        regime_entropy: float = None,
        top_state: int = None,
        top_state_prob: float = None,
        fair_prob: float = None,
        market_price: float = None,
        edge_cents: float = None,
        ev_cents: float = None,
        confidence: float = None,
        recommendation: str = None,
        position_size: int = None,
        timestamp: str = None,
    ) -> int:
        """Insert a shadow (paper) prediction. Returns lastrowid."""
        asset = asset.upper()
        timestamp = timestamp or _now_iso()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO hmm_shadow_predictions "
                    "(asset, ticker, timestamp, regime_posterior, regime_entropy, "
                    "top_state, top_state_prob, fair_prob, market_price, edge_cents, "
                    "ev_cents, confidence, recommendation, position_size) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (asset, ticker, timestamp, regime_posterior, regime_entropy,
                     top_state, top_state_prob, fair_prob, market_price, edge_cents,
                     ev_cents, confidence, recommendation, position_size),
                )
                return cur.lastrowid

    def resolve_shadow_prediction(
        self,
        prediction_id: int,
        outcome: str,
        pnl_cents: float = None,
    ) -> None:
        """Resolve a shadow prediction with outcome and optional PnL."""
        resolved_at = _now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE hmm_shadow_predictions "
                    "SET outcome = ?, pnl_cents = ?, resolved_at = ? "
                    "WHERE id = ?",
                    (outcome, pnl_cents, resolved_at, prediction_id),
                )

    def get_shadow_predictions(
        self,
        asset: Optional[str] = None,
        since: Optional[str] = None,
        resolved_only: bool = False,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """Query shadow predictions with optional filters."""
        clauses: List[str] = []
        params: List[Any] = []
        if asset:
            clauses.append("asset = ?")
            params.append(asset.upper())
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if resolved_only:
            clauses.append("outcome IS NOT NULL")
        sql = "SELECT * FROM hmm_shadow_predictions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # HMM model state methods
    # ------------------------------------------------------------------

    def save_hmm_model_state(
        self,
        asset: str,
        version: int,
        n_states: int,
        bic: float = None,
        log_likelihood: float = None,
        stability_flags: int = 0,
        state_means: str = None,
        transition_matrix: str = None,
        observation_count: int = 0,
        trained_at: str = None,
    ) -> int:
        """Save a trained HMM model state snapshot. Returns lastrowid."""
        asset = asset.upper()
        trained_at = trained_at or _now_iso()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO hmm_model_state "
                    "(asset, version, n_states, bic, log_likelihood, stability_flags, "
                    "state_means, transition_matrix, trained_at, observation_count) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (asset, version, n_states, bic, log_likelihood, stability_flags,
                     state_means, transition_matrix, trained_at, observation_count),
                )
                return cur.lastrowid

    def get_latest_hmm_model_state(self, asset: str) -> Optional[Dict[str, Any]]:
        """Load the latest model state for an asset (by version DESC)."""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM hmm_model_state WHERE asset = ? "
                    "ORDER BY version DESC LIMIT 1",
                    (asset.upper(),),
                ).fetchone()
                return dict(row) if row else None
