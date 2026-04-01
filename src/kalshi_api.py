import base64
import hashlib
import logging
import os
import time

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, utils

from config import (
    KALSHI_API_KEY,
    KALSHI_API_BASE_URL,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
)

KALSHI_PRIVATE_KEY_PATH = os.environ.get(
    "KALSHI_PRIVATE_KEY_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kalshi_private_key.pem"),
)


class KalshiAPI:
    def __init__(
        self,
        api_key=None,
        base_url=None,
        max_retries=MAX_RETRIES,
        retry_delay=RETRY_DELAY_SECONDS,
        private_key_path=None,
    ):
        self.api_key = api_key or KALSHI_API_KEY
        self.base_url = base_url or KALSHI_API_BASE_URL
        self.logger = logging.getLogger(__name__)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._private_key = self._load_private_key(private_key_path or KALSHI_PRIVATE_KEY_PATH)

    def _load_private_key(self, path):
        # First try KALSHI_PRIVATE_KEY env var (for Railway/cloud deploys)
        pem_str = os.environ.get("KALSHI_PRIVATE_KEY")
        if pem_str:
            # Railway may escape newlines as literal \n — restore them
            pem_str = pem_str.replace("\\n", "\n")
            try:
                return serialization.load_pem_private_key(pem_str.encode(), password=None)
            except Exception as e:
                self.logger.error(f"Failed to load private key from env var: {e}")
        # Fall back to file path
        try:
            with open(path, "rb") as f:
                return serialization.load_pem_private_key(f.read(), password=None)
        except Exception as e:
            self.logger.error(f"Failed to load private key from {path}: {e}")
            return None

    def _sign_request(self, method, path, timestamp_str):
        """Sign the request using RSA-PSS as required by Kalshi API v2."""
        # Strip query parameters from path before signing
        path_without_query = path.split("?")[0]
        # Kalshi expects direct concatenation with NO separators
        message = f"{timestamp_str}{method}{path_without_query}"
        signature = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _handle_request(self, method, endpoint, **kwargs):
        url = f"{self.base_url}{endpoint}"
        # Extract the full path (e.g. /trade-api/v2/portfolio/balance) for signing
        from urllib.parse import urlparse
        full_path = urlparse(url).path
        timestamp_str = str(int(time.time() * 1000))
        signature = self._sign_request(method.upper(), full_path, timestamp_str) if self._private_key else None
        headers = {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature or "",
            "KALSHI-ACCESS-TIMESTAMP": timestamp_str,
        }
        attempt = 0
        backoff = self.retry_delay

        while attempt < self.max_retries:
            try:
                response = requests.request(method, url, headers=headers, **kwargs)
                response.raise_for_status()
                if response.content:
                    return response.json()
                return {}
            except requests.exceptions.HTTPError as http_err:
                status_code = getattr(http_err.response, "status_code", None)
                if status_code and 400 <= status_code < 500:
                    self.logger.error(
                        f"Non-retriable HTTP error ({status_code}) for {endpoint}: {http_err}"
                    )
                    break
                self.logger.warning(
                    f"HTTP error ({status_code}) on attempt {attempt + 1}/{self.max_retries} "
                    f"for {endpoint}: {http_err}. Retrying in {backoff}s."
                )
            except requests.exceptions.RequestException as req_err:
                self.logger.warning(
                    f"Request exception on attempt {attempt + 1}/{self.max_retries} "
                    f"for {endpoint}: {req_err}. Retrying in {backoff}s."
                )

            attempt += 1
            if attempt < self.max_retries:
                time.sleep(backoff)
                backoff *= 2

        self.logger.error(
            f"Failed to complete request to {endpoint} after {self.max_retries} attempts."
        )
        return None

    # ---- Exchange endpoints ----
    def get_exchange_status(self):
        return self._handle_request("GET", "/exchange/status")

    def get_exchange_announcements(self):
        return self._handle_request("GET", "/exchange/announcements")

    # ---- Market & event data ----
    def get_markets(self, params=None):
        return self._handle_request("GET", "/markets", params=params or {})

    def get_market(self, market_ticker, params=None):
        return self._handle_request(
            "GET", f"/markets/{market_ticker}", params=params or {}
        )

    def get_events(self, params=None):
        return self._handle_request("GET", "/events", params=params or {})

    # ---- Portfolio endpoints ----
    def get_account_balance(self):
        return self._handle_request("GET", "/portfolio/balance")

    def get_positions(self, params=None):
        return self._handle_request("GET", "/portfolio/positions", params=params or {})

    def get_orders(self, params=None):
        return self._handle_request("GET", "/portfolio/orders", params=params or {})

    def create_order(self, order_payload):
        return self._handle_request("POST", "/portfolio/orders", json=order_payload)

    def cancel_order(self, order_id):
        return self._handle_request("DELETE", f"/portfolio/orders/{order_id}")

    # ---- Backwards-compatible helpers ----
    def fetch_market_data(self, params=None):
        """Legacy alias for get_markets used elsewhere in the bot."""
        return self.get_markets(params=params)

    def get_market_data(self, market_id):
        """Legacy alias for get_market to avoid breaking references."""
        return self.get_market(market_id)