"""
Real-time crypto price fetcher.

Primary: FreeCryptoAPI (set FREE_CRYPTO_API_KEY env var)
Fallback: CoinGecko free tier

Thread-safe with 60-second caching. Never crashes — returns cached/None on failure.
"""

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# --- FreeCryptoAPI ---
_FREE_CRYPTO_API_KEY = os.environ.get("FREE_CRYPTO_API_KEY", "")
_FREE_CRYPTO_BASE = "https://api.freecryptoapi.com/v1"

# --- CoinGecko (fallback) ---
_COINGECKO_BASE = "https://api.coingecko.com/api/v3"
_COINGECKO_SIMPLE_URL = (
    f"{_COINGECKO_BASE}/simple/price"
    "?ids=bitcoin,ethereum,solana"
    "&vs_currencies=usd"
    "&include_24hr_change=true"
)

# Asset mapping
_COINGECKO_ID_MAP = {"bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL"}
_TICKER_TO_CG_ID = {v: k for k, v in _COINGECKO_ID_MAP.items()}

# FreeCryptoAPI uses standard symbols
_FREE_CRYPTO_SYMBOLS = {"BTC": "BTC", "ETH": "ETH", "SOL": "SOL"}

_CACHE_TTL = 60  # 1 minute for BTC/ETH, fine for FreeCryptoAPI


class CryptoPrices:
    """Thread-safe crypto price provider with caching."""

    def __init__(self, cache_ttl: int = _CACHE_TTL):
        self._cache_ttl = cache_ttl
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ts: float = 0.0
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._use_free_crypto = bool(_FREE_CRYPTO_API_KEY)
        if self._use_free_crypto:
            logger.info("Using FreeCryptoAPI for price data")
        else:
            logger.info("FREE_CRYPTO_API_KEY not set — falling back to CoinGecko")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prices(self) -> Dict[str, Dict[str, Any]]:
        """Return latest prices for BTC, ETH, SOL.

        Returns dict like:
            {"BTC": {"price": 84000.0, "change_24h": 2.5}, ...}
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
        return entry.get("change_24h") if entry else None

    def get_price_history(self, asset: str, hours: int = 24) -> Optional[List[List[float]]]:
        """Fetch hourly price history. Uses CoinGecko market_chart (FreeCryptoAPI
        doesn't have a history endpoint in free tier)."""
        coin_id = _TICKER_TO_CG_ID.get(asset.upper())
        if not coin_id:
            return None

        days = max(hours / 24, 1)
        days_param = int(days) if days == int(days) else round(days, 2)
        url = f"{_COINGECKO_BASE}/coins/{coin_id}/market_chart?vs_currency=usd&days={days_param}"

        try:
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json().get("prices", [])
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
                return

        new_data = self._fetch_prices()
        if new_data is not None:
            with self._lock:
                self._cache = new_data
                self._cache_ts = time.monotonic()

    def _fetch_prices(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Try FreeCryptoAPI first, fall back to CoinGecko."""
        if self._use_free_crypto:
            result = self._fetch_from_free_crypto()
            if result:
                return result
            logger.warning("FreeCryptoAPI failed, falling back to CoinGecko")

        return self._fetch_from_coingecko()

    def _fetch_from_free_crypto(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Fetch from FreeCryptoAPI: GET /v1/getData?symbol=BTC+ETH+SOL"""
        try:
            url = f"{_FREE_CRYPTO_BASE}/getData?symbol=BTC+ETH+SOL"
            headers = {"Authorization": f"Bearer {_FREE_CRYPTO_API_KEY}"}
            resp = self._session.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            raw = resp.json()

            result: Dict[str, Dict[str, Any]] = {}
            data = raw.get("data", raw)

            for symbol in ["BTC", "ETH", "SOL"]:
                coin_data = None
                if isinstance(data, dict):
                    # Try various key formats
                    coin_data = (data.get(symbol) or data.get(f"{symbol}/USD")
                                 or data.get(symbol.lower()))
                elif isinstance(data, list):
                    # Might be a list of coin objects
                    for item in data:
                        if isinstance(item, dict) and item.get("symbol", "").upper() == symbol:
                            coin_data = item
                            break

                if coin_data and isinstance(coin_data, dict):
                    price = (coin_data.get("price") or coin_data.get("value")
                             or coin_data.get("usd") or coin_data.get("rate"))
                    change = (coin_data.get("change_24h") or coin_data.get("percent_change_24h")
                              or coin_data.get("change24h"))
                    if price is not None:
                        result[symbol] = {
                            "price": float(price),
                            "change_24h": round(float(change), 4) if change is not None else None,
                        }
                elif coin_data is not None:
                    try:
                        result[symbol] = {"price": float(coin_data), "change_24h": None}
                    except (ValueError, TypeError):
                        pass

            if result:
                logger.info(f"FreeCryptoAPI: got prices for {list(result.keys())}")
                return result

            # Log the raw response so we can debug the format
            logger.warning("FreeCryptoAPI: couldn't parse prices from: %s", str(raw)[:300])
            return None

        except Exception:
            logger.exception("Failed to fetch from FreeCryptoAPI")
            return None

    def _fetch_from_coingecko(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Fetch from CoinGecko free API."""
        try:
            resp = self._session.get(_COINGECKO_SIMPLE_URL, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
        except Exception:
            logger.exception("Failed to fetch crypto prices from CoinGecko")
            return None

        result: Dict[str, Dict[str, Any]] = {}
        for cg_id, ticker in _COINGECKO_ID_MAP.items():
            coin = raw.get(cg_id, {})
            price = coin.get("usd")
            change = coin.get("usd_24h_change")
            if price is not None:
                result[ticker] = {
                    "price": float(price),
                    "change_24h": round(float(change), 4) if change is not None else None,
                }
        return result


# Module-level singleton
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
