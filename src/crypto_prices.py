"""
Real-time crypto price fetcher using CoinGecko's free API.

Thread-safe with 60-second caching to respect rate limits.
Never crashes -- returns cached data or None on failure.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_COINGECKO_BASE = "https://api.coingecko.com/api/v3"

_SIMPLE_PRICE_URL = (
    f"{_COINGECKO_BASE}/simple/price"
    "?ids=bitcoin,ethereum,solana"
    "&vs_currencies=usd"
    "&include_24hr_change=true"
)

# CoinGecko id -> our ticker
_ID_MAP = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
}

# Our ticker -> CoinGecko id
_TICKER_TO_ID = {v: k for k, v in _ID_MAP.items()}

_CACHE_TTL = 60  # seconds


class CryptoPrices:
    """Thread-safe crypto price provider with caching."""

    def __init__(self, cache_ttl: int = _CACHE_TTL):
        self._cache_ttl = cache_ttl
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ts: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prices(self) -> Dict[str, Dict[str, Any]]:
        """Return latest prices for BTC, ETH, SOL.

        Returns dict like:
            {"BTC": {"price": 84000.0, "change_24h": 2.5}, ...}
        Falls back to cached data (possibly empty dict) on failure.
        """
        self._refresh_if_stale()
        with self._lock:
            return dict(self._cache)

    def get_price(self, asset: str) -> Optional[float]:
        """Return USD price for a single asset, or None."""
        prices = self.get_prices()
        entry = prices.get(asset.upper())
        return entry["price"] if entry else None

    def get_change_24h(self, asset: str) -> Optional[float]:
        """Return 24h % change for a single asset, or None."""
        prices = self.get_prices()
        entry = prices.get(asset.upper())
        return entry["change_24h"] if entry else None

    def get_price_history(
        self, asset: str, hours: int = 24
    ) -> Optional[List[List[float]]]:
        """Fetch hourly price history from CoinGecko market_chart endpoint.

        Returns list of [timestamp_ms, price_usd] pairs, or None on failure.
        Uses 'hourly' granularity when hours <= 90 days (CoinGecko auto).
        """
        coin_id = _TICKER_TO_ID.get(asset.upper())
        if not coin_id:
            logger.warning("Unknown asset for price history: %s", asset)
            return None

        # CoinGecko market_chart: days param controls granularity
        # <= 1 day  -> ~5-min intervals
        # 1-90 days -> hourly
        # We convert hours to days (minimum 1 to get hourly data)
        days = max(hours / 24, 1)
        # Use integer if whole number, else float
        days_param = int(days) if days == int(days) else round(days, 2)

        url = (
            f"{_COINGECKO_BASE}/coins/{coin_id}/market_chart"
            f"?vs_currency=usd&days={days_param}"
        )

        try:
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("prices", [])
        except Exception:
            logger.exception("Failed to fetch price history for %s", asset)
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh_if_stale(self) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._cache_ts < self._cache_ttl:
                return  # cache is fresh

        # Fetch outside the lock to avoid blocking other readers
        new_data = self._fetch_prices()
        if new_data is not None:
            with self._lock:
                self._cache = new_data
                self._cache_ts = time.monotonic()

    def _fetch_prices(self) -> Optional[Dict[str, Dict[str, Any]]]:
        try:
            resp = self._session.get(_SIMPLE_PRICE_URL, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
        except Exception:
            logger.exception("Failed to fetch crypto prices from CoinGecko")
            return None

        result: Dict[str, Dict[str, Any]] = {}
        for cg_id, ticker in _ID_MAP.items():
            coin = raw.get(cg_id, {})
            price = coin.get("usd")
            change = coin.get("usd_24h_change")
            if price is not None:
                result[ticker] = {
                    "price": float(price),
                    "change_24h": round(float(change), 4) if change is not None else None,
                }
        return result


# Module-level singleton for convenience
_default_instance: Optional[CryptoPrices] = None
_instance_lock = threading.Lock()


def get_default() -> CryptoPrices:
    """Return (or create) the module-level singleton."""
    global _default_instance
    if _default_instance is None:
        with _instance_lock:
            if _default_instance is None:
                _default_instance = CryptoPrices()
    return _default_instance
