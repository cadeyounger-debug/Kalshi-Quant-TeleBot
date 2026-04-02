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
        asset: str = None,
        timestamp: str = None,
    ) -> int:
        asset = asset or extract_asset(market_ticker)
        timestamp = timestamp or _now_iso()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO trades "
                    "(asset, market_ticker, side, quantity, price, strategy, order_result, pnl, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (asset, market_ticker, side, quantity, price, strategy,
                     order_result, pnl, timestamp),
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
