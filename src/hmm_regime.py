"""HMM Regime Detection Engine.

Per-asset Gaussian HMM with BIC-based state selection and
forward-backward posterior inference for regime classification.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "log_return_1m", "log_return_5m", "log_return_15m",
    "realized_vol_15m", "realized_vol_1h", "vol_of_vol",
    "momentum_r_sq", "mean_reversion", "bid_ask_spread",
    "spread_vol", "volume_1m", "volume_accel",
]

MIN_OBSERVATIONS = 300  # ~5 hours of 1-min data
DEFAULT_K = 5


# ------------------------------------------------------------------ #
# Helper functions
# ------------------------------------------------------------------ #

def _observations_to_matrix(rows: List[Dict]) -> np.ndarray:
    """Convert DB observation rows to an (N, 12) numpy array."""
    out = np.zeros((len(rows), len(FEATURE_COLS)))
    for i, row in enumerate(rows):
        for j, col in enumerate(FEATURE_COLS):
            out[i, j] = row.get(col) or 0.0
    return out


def _normalize(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize columns. Returns (normalized, means, stds)."""
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    # Prevent division by zero
    stds[stds < 1e-12] = 1.0
    normalized = (data - means) / stds
    return normalized, means, stds


def fit_hmm_select_k(
    data: np.ndarray,
    k_range=range(3, 9),
    n_restarts: int = 5,
) -> Tuple[GaussianHMM, int, float]:
    """Fit GaussianHMM for each K in k_range, return model with lowest BIC.

    Returns (best_model, best_k, best_bic).
    """
    best_model = None
    best_k = DEFAULT_K
    best_bic = np.inf
    n_samples = data.shape[0]

    for k in k_range:
        for restart in range(n_restarts):
            try:
                model = GaussianHMM(
                    n_components=k,
                    covariance_type="full",
                    n_iter=100,
                    random_state=restart * 42 + k,
                )
                model.fit(data)
                ll = model.score(data)
                # BIC = -2 * LL + n_params * ln(N)
                n_features = data.shape[1]
                # params: start probs (k-1), transition (k*(k-1)), means (k*d), covariances (k*d*(d+1)/2)
                n_params = (
                    (k - 1)
                    + k * (k - 1)
                    + k * n_features
                    + k * n_features * (n_features + 1) // 2
                )
                bic = -2 * ll + n_params * np.log(n_samples)
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
                    best_k = k
            except Exception as exc:
                logger.debug("HMM fit failed for K=%d restart=%d: %s", k, restart, exc)
                continue

    return best_model, best_k, float(best_bic)


def decode_posterior(model: GaussianHMM, data: np.ndarray) -> np.ndarray:
    """Run forward-backward and return posterior state probabilities (N, K)."""
    return model.predict_proba(data)


# ------------------------------------------------------------------ #
# RegimeEngine class
# ------------------------------------------------------------------ #

class RegimeEngine:
    """Per-asset HMM regime engine backed by a TradingDB."""

    def __init__(self, db):
        self.db = db
        self._models: Dict[str, dict] = {}  # asset -> {model, means, stds, n_states}

    def fit_asset(self, asset: str) -> Optional[Dict]:
        """Fit HMM for *asset*, persist to DB, return summary dict or None."""
        asset = asset.upper()

        rows = self.db.get_hmm_observations(asset=asset)
        if len(rows) < MIN_OBSERVATIONS:
            logger.info("Insufficient data for %s: %d rows (need %d)", asset, len(rows), MIN_OBSERVATIONS)
            return None

        data = _observations_to_matrix(rows)
        normed, means, stds = _normalize(data)

        model, k, bic = fit_hmm_select_k(normed)
        if model is None:
            logger.warning("All HMM fits failed for %s", asset)
            return None

        # Check stability against prior model
        stability_flags = 0
        prior = self.db.get_latest_hmm_model_state(asset)
        if prior and prior.get("state_means"):
            try:
                old_means = np.array(json.loads(prior["state_means"]))
                new_means = model.means_  # (K, D)
                # Compare per-state mean vectors; flag if any shifted > 2 std
                if old_means.shape == new_means.shape:
                    shift = np.abs(new_means - old_means).max()
                    if shift > 2.0:
                        stability_flags = 1
                        logger.warning("Regime shift detected for %s: max shift %.3f", asset, shift)
                else:
                    stability_flags = 2  # K changed
            except Exception:
                pass

        # Determine version
        version = (prior["version"] + 1) if prior else 1

        # Persist to DB
        self.db.save_hmm_model_state(
            asset=asset,
            version=version,
            n_states=k,
            bic=bic,
            log_likelihood=float(model.score(normed)),
            stability_flags=stability_flags,
            state_means=json.dumps(model.means_.tolist()),
            transition_matrix=json.dumps(model.transmat_.tolist()),
            observation_count=len(rows),
            trained_at=datetime.now(timezone.utc).isoformat(),
        )

        # Cache in memory
        self._models[asset] = {
            "model": model,
            "means": means,
            "stds": stds,
            "n_states": k,
        }

        return {
            "asset": asset,
            "n_states": k,
            "bic": bic,
            "observation_count": len(rows),
            "stability_flags": stability_flags,
            "version": version,
        }

    def get_current_posterior(self, asset: str) -> Optional[List[float]]:
        """Return posterior probabilities for the latest timestep."""
        asset = asset.upper()
        cached = self._models.get(asset)
        if cached is None:
            return None

        rows = self.db.get_hmm_observations(asset=asset, limit=200)
        if not rows:
            return None

        data = _observations_to_matrix(rows)
        normed = (data - cached["means"]) / cached["stds"]

        posteriors = decode_posterior(cached["model"], normed)
        return posteriors[-1].tolist()

    def get_regime_entropy(self, posterior: List[float]) -> float:
        """Compute Shannon entropy: -sum(p * log(p))."""
        p = np.array(posterior)
        p = p[p > 0]  # avoid log(0)
        return float(-np.sum(p * np.log(p)))

    def get_transition_matrix(self, asset: str) -> Optional[np.ndarray]:
        """Return the fitted transition matrix for *asset*."""
        asset = asset.upper()
        cached = self._models.get(asset)
        if cached is None:
            return None
        return cached["model"].transmat_

    def fit_all_assets(self) -> Dict:
        """Fit HMM for BTC, ETH, SOL. Returns dict of summaries."""
        results = {}
        for asset in ("BTC", "ETH", "SOL"):
            results[asset] = self.fit_asset(asset)
        return results
