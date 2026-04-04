"""HMM Graduation Controller — manages stage progression from shadow to live.

Stages: COLLECTION → SHADOW → PAPER → SMALL_CAP_LIVE → FULL_LIVE
Each stage has promotion gates and kill switches to protect capital.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Dict, List

logger = logging.getLogger(__name__)


class Stage(IntEnum):
    COLLECTION = 0
    SHADOW = 1
    PAPER = 2
    SMALL_CAP_LIVE = 3
    FULL_LIVE = 4


# Promotion constants
MIN_SHADOW_DAYS = 7
MIN_TRADE_COUNT = 30
MAX_DRAWDOWN_PCT = 0.15
MIN_EV_ADVANTAGE_CENTS = 1.0

# Kill switch constants
KILL_DAILY_LOSS_PCT = 0.20
KILL_CONSECUTIVE_LOSSES = 5

# Stage limits
_MAX_CONTRACTS = {
    Stage.COLLECTION: 0,
    Stage.SHADOW: 0,
    Stage.PAPER: 0,
    Stage.SMALL_CAP_LIVE: 1,
    Stage.FULL_LIVE: 3,
}

_MAX_POSITIONS = {
    Stage.COLLECTION: 0,
    Stage.SHADOW: 0,
    Stage.PAPER: 0,
    Stage.SMALL_CAP_LIVE: 3,
    Stage.FULL_LIVE: 5,
}

# Minimum days in COLLECTION before advancing
_MIN_COLLECTION_DAYS = 3


class GraduationController:
    """Manages stage progression for the HMM trading system."""

    def __init__(self):
        self.stage: Stage = Stage.COLLECTION
        self.days_in_stage: int = 0
        self.has_enough_observations: bool = False

    def can_advance(self) -> bool:
        """Check if the controller can advance to the next stage."""
        if self.stage == Stage.COLLECTION:
            return self.days_in_stage >= _MIN_COLLECTION_DAYS and self.has_enough_observations
        if self.stage == Stage.FULL_LIVE:
            return False  # Already at max
        # For other stages, promotion gate must be checked externally
        return False

    def advance(self):
        """Move to the next stage."""
        if self.stage < Stage.FULL_LIVE:
            self.stage = Stage(self.stage + 1)
            self.days_in_stage = 0
            logger.info("Advanced to stage %s", self.stage.name)

    def demote(self):
        """Revert to SHADOW stage."""
        self.stage = Stage.SHADOW
        self.days_in_stage = 0
        logger.info("Demoted to SHADOW")

    def check_promotion_gate(
        self,
        shadow_metrics: Dict,
        live_metrics: Dict,
        bankroll: float,
        model_stable: bool,
    ) -> Dict:
        """Check all 7 promotion criteria.

        Returns {passes: bool, failures: list, criteria_met: int, criteria_total: 7}.

        7 criteria:
        1. days >= MIN_SHADOW_DAYS
        2. trades >= MIN_TRADE_COUNT
        3. EV > 0 (positive avg PnL)
        4. EV > live + MIN_EV_ADVANTAGE_CENTS
        5. Win rate > live win rate
        6. Drawdown < MAX_DRAWDOWN_PCT of bankroll
        7. Model is stable
        """
        failures: List[str] = []
        criteria_met = 0

        # 1. Minimum days
        if self.days_in_stage >= MIN_SHADOW_DAYS:
            criteria_met += 1
        else:
            failures.append(f"Days in stage ({self.days_in_stage}) < {MIN_SHADOW_DAYS}")

        # 2. Minimum trade count
        trade_count = shadow_metrics.get("trade_count", 0)
        if trade_count >= MIN_TRADE_COUNT:
            criteria_met += 1
        else:
            failures.append(f"Trade count ({trade_count}) < {MIN_TRADE_COUNT}")

        # 3. Positive EV
        shadow_ev = shadow_metrics.get("avg_pnl_per_trade", 0)
        if shadow_ev > 0:
            criteria_met += 1
        else:
            failures.append(f"Shadow EV ({shadow_ev:.2f}) <= 0")

        # 4. EV advantage over live
        live_ev = live_metrics.get("avg_pnl_per_trade", 0)
        if shadow_ev > live_ev + MIN_EV_ADVANTAGE_CENTS:
            criteria_met += 1
        else:
            failures.append(
                f"Shadow EV ({shadow_ev:.2f}) not > live EV ({live_ev:.2f}) + {MIN_EV_ADVANTAGE_CENTS}c"
            )

        # 5. Win rate better than live
        shadow_wr = shadow_metrics.get("win_rate", 0)
        live_wr = live_metrics.get("win_rate", 0)
        if shadow_wr > live_wr:
            criteria_met += 1
        else:
            failures.append(f"Shadow WR ({shadow_wr:.2%}) <= live WR ({live_wr:.2%})")

        # 6. Drawdown < threshold
        max_dd = shadow_metrics.get("max_drawdown_cents", 0)
        dd_limit = bankroll * MAX_DRAWDOWN_PCT * 100  # bankroll in dollars, dd in cents
        if max_dd < dd_limit:
            criteria_met += 1
        else:
            failures.append(f"Max drawdown ({max_dd:.0f}c) >= {dd_limit:.0f}c ({MAX_DRAWDOWN_PCT:.0%} of bankroll)")

        # 7. Model stability
        if model_stable:
            criteria_met += 1
        else:
            failures.append("Model not stable")

        return {
            "passes": criteria_met == 7,
            "failures": failures,
            "criteria_met": criteria_met,
            "criteria_total": 7,
        }

    def check_kill_switch(
        self,
        daily_pnl_cents: float,
        bankroll: float,
        consecutive_losses: int,
        model_stable: bool,
    ) -> bool:
        """Check kill switch conditions. Only active for SMALL_CAP_LIVE+.

        Returns True if any kill condition triggered:
        - Daily loss > KILL_DAILY_LOSS_PCT of bankroll
        - Consecutive losses >= KILL_CONSECUTIVE_LOSSES
        - Model unstable
        """
        if self.stage < Stage.SMALL_CAP_LIVE:
            return False

        bankroll_cents = bankroll * 100
        loss_threshold = bankroll_cents * KILL_DAILY_LOSS_PCT

        if daily_pnl_cents < 0 and abs(daily_pnl_cents) >= loss_threshold:
            logger.warning("Kill switch: daily loss %.0fc >= %.0fc threshold", abs(daily_pnl_cents), loss_threshold)
            return True

        if consecutive_losses >= KILL_CONSECUTIVE_LOSSES:
            logger.warning("Kill switch: %d consecutive losses >= %d", consecutive_losses, KILL_CONSECUTIVE_LOSSES)
            return True

        if not model_stable:
            logger.warning("Kill switch: model unstable")
            return True

        return False

    def get_max_contracts(self) -> int:
        """Max contracts allowed for current stage."""
        return _MAX_CONTRACTS.get(self.stage, 0)

    def get_max_positions(self) -> int:
        """Max simultaneous positions for current stage."""
        return _MAX_POSITIONS.get(self.stage, 0)

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "stage": int(self.stage),
            "days_in_stage": self.days_in_stage,
            "has_enough_observations": self.has_enough_observations,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> GraduationController:
        """Deserialize from dict."""
        gc = cls()
        gc.stage = Stage(d.get("stage", 0))
        gc.days_in_stage = d.get("days_in_stage", 0)
        gc.has_enough_observations = d.get("has_enough_observations", False)
        return gc
