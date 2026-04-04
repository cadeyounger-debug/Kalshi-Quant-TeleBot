"""HMM Contract Opportunity Model.

Evaluates Kalshi contracts using HMM regime posteriors to compute
expected value, position sizing, and trade recommendations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

FEES_CENTS = 6  # 2c entry + 2c exit + 1c slippage each side


@dataclass
class ContractEvaluation:
    """Result of evaluating a single contract opportunity."""
    fair_prob: float
    edge_yes_cents: float
    edge_no_cents: float
    ev_cents: float
    confidence: float
    regime_entropy: float
    recommendation: str  # "buy_yes", "buy_no", "skip"
    position_size: int
    reasons: List[str] = field(default_factory=list)


def compute_regime_ev(
    regime_posterior: np.ndarray,
    state_profiles: List[dict],
    edge_cents: float,
    fees_cents: float = FEES_CENTS,
) -> Tuple[float, float]:
    """Posterior-weighted EV across regime states.

    States with <10 trades use break-even prior (0 EV contribution).
    Returns (ev, confidence) where confidence scales with data (count/50, capped at 1).
    """
    ev = 0.0
    weighted_confidence = 0.0

    for i, profile in enumerate(state_profiles):
        weight = regime_posterior[i] if i < len(regime_posterior) else 0.0
        count = profile.get("count", 0)

        if count < 10:
            # Insufficient data — break-even prior, contributes 0 EV
            state_confidence = count / 50.0
            weighted_confidence += weight * state_confidence
            continue

        win_rate = profile["win_rate"]
        avg_win = profile["avg_win_cents"]
        avg_loss = profile["avg_loss_cents"]

        # EV = win_rate * avg_win - (1 - win_rate) * avg_loss - fees
        state_ev = win_rate * avg_win - (1.0 - win_rate) * avg_loss - fees_cents
        state_confidence = min(count / 50.0, 1.0)

        ev += weight * state_ev
        weighted_confidence += weight * state_confidence

    return ev, weighted_confidence


def position_size(
    ev_cents: float,
    confidence: float,
    bankroll: float,
    max_contracts: int = 3,
) -> int:
    """Half-Kelly with confidence scaling. Return 0 if ev<=0 or confidence<0.3."""
    if ev_cents <= 0 or confidence < 0.3:
        return 0

    # Half-Kelly fraction: f = 0.5 * edge / odds
    # Simplified: fraction of bankroll per contract (~$1 contract)
    kelly_fraction = 0.5 * (ev_cents / 100.0) * confidence
    raw_size = kelly_fraction * bankroll / 100.0  # bankroll in dollars, contract ~$1

    size = max(1, int(raw_size))
    return min(size, max_contracts)


def evaluate_contract_with_regime(
    regime_posterior: np.ndarray,
    state_profiles: List[dict],
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
    """Full contract evaluation combining regime info with market data.

    Skip if <2min to expiry or entropy > 80% of max.
    Only buy YES if fair_prob > 0.52, NO if fair_prob < 0.48.
    V1 fair_prob = log_normal_prob.
    """
    reasons: List[str] = []
    n_states = len(regime_posterior)

    # Regime entropy
    posterior = np.array(regime_posterior, dtype=float)
    posterior = posterior / posterior.sum()  # normalize
    entropy = -np.sum(posterior * np.log(posterior + 1e-12))
    max_entropy = np.log(n_states) if n_states > 1 else 1.0

    # V1: fair_prob from log-normal model
    fair_prob = log_normal_prob

    # Edge calculations
    edge_yes_cents = (fair_prob * 100.0) - yes_price_cents - FEES_CENTS
    edge_no_cents = ((1.0 - fair_prob) * 100.0) - no_price_cents - FEES_CENTS

    # Compute regime EV
    best_edge = max(edge_yes_cents, edge_no_cents)
    ev, confidence = compute_regime_ev(posterior, state_profiles, best_edge)

    # Default recommendation
    recommendation = "skip"
    size = 0

    # Skip filters
    if time_to_expiry_secs < 120:
        reasons.append("Too close to expiry (<2min)")
    elif entropy > 0.8 * max_entropy:
        reasons.append(f"Regime entropy too high ({entropy:.3f} > {0.8 * max_entropy:.3f})")
    else:
        # Decide direction
        if fair_prob > 0.52 and edge_yes_cents > 0:
            recommendation = "buy_yes"
            ev = edge_yes_cents  # Use edge as EV proxy
            reasons.append(f"YES edge {edge_yes_cents:.1f}c, fair_prob={fair_prob:.3f}")
        elif fair_prob < 0.48 and edge_no_cents > 0:
            recommendation = "buy_no"
            ev = edge_no_cents
            reasons.append(f"NO edge {edge_no_cents:.1f}c, fair_prob={fair_prob:.3f}")
        else:
            reasons.append(f"No sufficient edge (fair_prob={fair_prob:.3f})")

        if recommendation != "skip":
            size = position_size(ev, confidence, bankroll)
            if size == 0:
                reasons.append("Position size 0 — skipping")
                recommendation = "skip"

    return ContractEvaluation(
        fair_prob=fair_prob,
        edge_yes_cents=edge_yes_cents,
        edge_no_cents=edge_no_cents,
        ev_cents=ev,
        confidence=confidence,
        regime_entropy=entropy,
        recommendation=recommendation,
        position_size=size,
        reasons=reasons,
    )
