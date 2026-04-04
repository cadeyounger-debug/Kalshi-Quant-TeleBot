# HMM Regime Detection System — Technical Specification

## Objective

Build a two-layer probabilistic trading system that:
1. Infers latent market regimes from spot price and microstructure data (per-asset)
2. Evaluates specific 15-minute binary contracts for mispricing given the inferred regime
3. Makes position decisions based on posterior-weighted expected value, not hard state labels
4. Runs in shadow mode against the current strategy, graduates through staged rollout

Training begins from a clean slate — no historical data from the prior (buggy) system contaminates the model.

---

## System Overview

```
Layer 1: Market Regime Model (per asset)
  Input:  1-minute bars of spot + microstructure features
  Output: Posterior distribution over K hidden states + transition matrix
  Scope:  "What kind of market are we in right now?"

Layer 2: Contract Opportunity Model
  Input:  Regime posteriors + contract-specific features (moneyness, time-to-expiry, liquidity)
  Output: Fair value probability, edge estimate, position recommendation
  Scope:  "Is this specific contract mispriced, and should we trade it?"
```

Layer 1 and Layer 2 are decoupled. Layer 1 knows nothing about contracts. Layer 2 consumes regime posteriors as conditioning features alongside contract mechanics. This separation ensures the regime model remains stable across contract lifecycles and can generalize to new contract types.

---

## Architecture

### Components

| Component | File | Responsibility |
|-----------|------|----------------|
| Observation Pipeline | `src/hmm_observations.py` | Aggregate 20s ticks into 1-min bars, compute features, store to DB |
| Regime Engine | `src/hmm_regime.py` | Per-asset HMM: fit, decode posteriors, expose current regime distribution |
| Contract Evaluator | `src/hmm_contract.py` | Fair-value model: regime-conditioned probability, edge, position sizing |
| Shadow Tracker | `src/hmm_shadow.py` | Log shadow predictions, track outcomes, compute comparison metrics |
| Graduation Controller | `src/hmm_graduation.py` | Rolling evaluation, staged promotion logic, kill switch |

### Data Flow

```
[Every 20s] Raw spot prices + market snapshots → existing DB tables
                         |
[Every 60s] hmm_observations.py aggregates → hmm_observations table
                         |
[Every 20s trading cycle]
  hmm_regime.py: decode posterior over states from recent observations
                         |
  hmm_contract.py: for each active 15M contract,
    - compute contract-specific features
    - condition fair-value estimate on regime posterior
    - compute edge = fair_value - market_price
    - apply position sizing rules
                         |
  hmm_shadow.py: log prediction + regime state (shadow mode)
    OR
  trader.py: execute trade (live mode, post-graduation)
                         |
[Daily 7am UTC] Retrain HMM on all collected data
  - Fit models for K = {3,4,5,6,7,8}, select by BIC
  - Learn per-state strategy profiles from outcome data
  - Walk-forward backtest on held-out recent data
  - Compare shadow vs live performance → Telegram report
```

---

## Model Design

### Layer 1: Market Regime HMM

**Model class:** Gaussian HMM with full covariance matrices. Per-asset (BTC, ETH, SOL independently).

**State count selection:** Do not hardcode K=7. At each retrain:
- Fit models for K in {3, 4, 5, 6, 7, 8}
- Select K by Bayesian Information Criterion (BIC) on the training set
- Require stability: if optimal K changes between retrains, log a warning and prefer the prior K unless BIC improvement exceeds 2% relative to the prior model
- V1 initial default: K=5 until enough data accumulates for meaningful BIC comparison (~7 days)

**Emission model:** Multivariate Gaussian (V1). Adequate for normalized features. Known limitation: fat tails in crypto returns. Monitor excess kurtosis of residuals; if consistently >6, flag for V1.5 upgrade to Student-t mixture emissions.

**Inference:** Forward-backward algorithm (not Viterbi). Output is the full posterior distribution P(state | observations_1:t) at each timestep. Trade decisions use the posterior vector, not a hard state assignment. This prevents overconfident regime calls during transitions.

**Transition dynamics:** The transition matrix A encodes regime persistence and switching probabilities. High self-transition probability (A[i,i] > 0.9) indicates a sticky regime. Use transition probabilities to estimate regime duration and probability of imminent switch — feed these into Layer 2 as conditioning features.

**Training:**
- Baum-Welch (EM) with 10 random restarts, select best log-likelihood
- Minimum 3 days of 1-minute data before first real fit (~4,300 observations)
- Walk-forward: train on days [1, T-1], validate on day T
- Purge: exclude the last 15 minutes of training data before validation window to prevent label leakage from overlapping contract outcomes

**Stability monitoring:**
- Track state-mean vectors across retrains. If a state's mean shifts by >2 standard deviations, flag as unstable
- Track log-likelihood on a held-out validation window. Declining log-likelihood across retrains indicates model degradation
- If >50% of states are flagged unstable, freeze the model and alert via Telegram

### Layer 2: Contract Opportunity Model

**Model class:** Regime-conditioned logistic regression (V1). Predicts P(contract settles YES) given regime posteriors + contract features. Simple, interpretable, fast to retrain.

**Upgrade path (V1.5):** Gradient-boosted trees (LightGBM) with regime posteriors as features, once sufficient outcome data exists (~500+ resolved contracts per asset).

**Fair value framework:**
```
P_fair(YES) = f(regime_posterior, moneyness, time_to_expiry, liquidity_features)
edge_yes = P_fair(YES) * 100 - market_yes_price_cents
edge_no  = (1 - P_fair(YES)) * 100 - market_no_price_cents
```

This replaces the current log-normal + momentum model. The log-normal estimate can be included as a feature input to the logistic regression rather than discarded.

**Training data:** Only contracts that have resolved (settled YES or NO). Each resolved contract becomes one training example with features computed at the time of evaluation and label = 1 (settled YES) or 0 (settled NO).

**Anti-leakage:** Features must be computed using only data available at evaluation time. No future prices, no post-resolution data. Enforce with timestamp validation in the training pipeline.

---

## Feature Design

### Group 1: Spot Market Regime Features (Layer 1 input)

Computed per asset at 1-minute resolution.

| Feature | Computation | Rationale |
|---------|-------------|-----------|
| log_return_1m | ln(price_t / price_{t-1}) | Instantaneous price change |
| log_return_5m | ln(price_t / price_{t-5}) | Medium-horizon momentum |
| log_return_15m | ln(price_t / price_{t-15}) | Aligns with contract horizon |
| realized_vol_15m | std(log_return_1m) over trailing 15 bars | Short-horizon volatility |
| realized_vol_1h | std(log_return_1m) over trailing 60 bars | Medium-horizon volatility |
| vol_of_vol | std(realized_vol_15m) over trailing 30 bars | Volatility regime instability |
| momentum_r_squared | R² of exp-weighted regression over 10 bars | Trend confidence |
| mean_reversion_signal | (price_t - EMA_15) / realized_vol_15m | Z-score vs moving average |

All features are z-score normalized using a 24-hour rolling window (mean and std computed over trailing 1440 bars). This prevents non-stationarity from drifting feature scales.

### Group 2: Microstructure Features (Layer 1 input)

Computed from the nearest active 15M contract for that asset. When no active contract exists, these features are set to 0 and a binary `has_active_contract` flag is set to 0.

| Feature | Computation | Rationale |
|---------|-------------|-----------|
| bid_ask_spread_pct | (yes_ask - yes_bid) / yes_mid | Liquidity / uncertainty |
| spread_volatility | std(bid_ask_spread_pct) over 5 bars | Spread stability |
| volume_1m | contracts traded in last 1 minute | Trade intensity |
| volume_acceleration | volume_1m / EMA(volume_1m, 10) | Volume surge detection |
| has_active_contract | 1 if active 15M contract exists, else 0 | Mask for microstructure features |

### Group 3: Contract-Specific Features (Layer 2 input)

Computed per contract at evaluation time. These are NOT input to the HMM — they feed the contract opportunity model.

| Feature | Computation | Rationale |
|---------|-------------|-----------|
| moneyness | (spot - strike) / strike | How far in/out of the money |
| abs_moneyness | abs(moneyness) | Distance regardless of direction |
| time_to_expiry_frac | seconds_remaining / 900 | Normalized [0,1] within 15-min window |
| time_to_expiry_sqrt | sqrt(time_to_expiry_frac) | Captures non-linear time decay |
| contract_volume | contracts traded on this specific contract | Specific contract liquidity |
| yes_price_cents | current YES ask | Market's implied probability |
| no_price_cents | current NO ask | Complement check |
| log_normal_prob | existing estimate_strike_probability() output | Base statistical probability |
| bid_ask_spread_cents | yes_ask - yes_bid (in cents) | Execution cost |

### Group 4: Regime Conditioning Features (Layer 2 input, from Layer 1)

| Feature | Source | Rationale |
|---------|--------|-----------|
| regime_posterior[0..K-1] | P(state=k \| obs_1:t) for each state | Full regime belief |
| regime_entropy | -sum(p * log(p)) of posterior | Regime uncertainty (high = transitioning) |
| regime_persistence | A[k,k] for argmax(posterior) | Expected duration of current regime |
| transition_prob_to_volatile | sum of posterior-weighted transitions to historically high-vol states | Risk of regime switch |

---

## Decision Logic

### Posterior-Weighted Expected Value

Do NOT map states to fixed strategy profiles. Instead, compute expected value as a posterior-weighted sum across all states:

```python
def compute_regime_ev(regime_posterior, state_profiles, edge_cents, fees_cents=4):
    """
    Expected value of a trade, weighted by probability of being in each regime.
    
    state_profiles[k] contains historically observed:
      - win_rate: fraction of trades that were profitable in state k
      - avg_win_cents: average profit on winning trades in state k
      - avg_loss_cents: average loss on losing trades in state k
      - trade_count: number of resolved trades observed in state k
    """
    ev = 0.0
    confidence = 0.0
    
    for k, p_k in enumerate(regime_posterior):
        profile = state_profiles[k]
        if profile.trade_count < 10:
            # Insufficient data for this state — use prior (break even)
            state_ev = 0.0
        else:
            state_ev = (profile.win_rate * profile.avg_win_cents 
                       - (1 - profile.win_rate) * abs(profile.avg_loss_cents)
                       - fees_cents)
        ev += p_k * state_ev
        confidence += p_k * min(profile.trade_count / 50, 1.0)  # Confidence scales with data
    
    return ev, confidence
```

### Position Sizing

Kelly-criterion based, scaled by regime confidence:

```python
def position_size(ev_cents, confidence, bankroll, max_contracts=3):
    if ev_cents <= 0 or confidence < 0.3:
        return 0  # No trade
    
    # Half-Kelly with confidence scaling
    kelly_frac = (ev_cents / 100) * confidence * 0.5
    position_value = bankroll * kelly_frac
    contracts = min(int(position_value / 100), max_contracts)
    return max(contracts, 0)
```

### Trade Decision Flow

```python
def should_trade(regime_posterior, state_profiles, contract_eval):
    edge = contract_eval.edge_cents
    fair_prob = contract_eval.fair_probability
    
    # 1. Minimum edge after fees
    if abs(edge) < 5:
        return None
    
    # 2. Must believe in the direction (fair_prob not near 50%)
    if 0.48 < fair_prob < 0.52:
        return None
    
    # 3. Compute posterior-weighted EV
    ev, confidence = compute_regime_ev(regime_posterior, state_profiles, edge)
    
    # 4. Positive EV required
    if ev <= 0:
        return None
    
    # 5. Minimum confidence threshold
    if confidence < 0.3:
        return None
    
    # 6. Regime entropy check — don't trade during transitions
    entropy = -sum(p * log(p + 1e-10) for p in regime_posterior)
    max_entropy = log(len(regime_posterior))
    if entropy > 0.8 * max_entropy:
        return None  # Too uncertain about regime
    
    # 7. Size and return
    side = "yes" if fair_prob > 0.50 else "no"
    size = position_size(ev, confidence, bankroll)
    if size == 0:
        return None
    
    return TradeSignal(side=side, contracts=size, edge=edge, ev=ev, 
                       confidence=confidence, regime_entropy=entropy)
```

---

## Evaluation Framework

### Walk-Forward Backtesting

Every retrain performs walk-forward evaluation:
- Training window: all data from collection start to T-1 day
- Validation window: day T (yesterday)
- No overlap between train and validation within 15 minutes (purge buffer)
- Record: predicted regime, shadow trade decision, actual contract outcome

### Shadow Mode Tracking

Every trading cycle, the shadow system:
1. Decodes regime posterior
2. Evaluates all active 15M contracts
3. Records what it would trade (or not trade), with: regime posterior, fair_prob, edge, EV, confidence, position size
4. After contract resolves, records outcome (win/loss, P&L)

Shadow predictions are stored in `hmm_shadow_predictions` table with full feature snapshots for later analysis.

### Comparison Metrics

Computed on a rolling window against the live strategy:

| Metric | Definition |
|--------|-----------|
| shadow_win_rate | Wins / total shadow trades |
| live_win_rate | Wins / total live trades |
| shadow_ev_per_trade | Mean P&L per shadow trade (after fees) |
| live_ev_per_trade | Mean P&L per live trade (after fees) |
| shadow_sharpe | Mean return / std return (annualized) |
| shadow_max_drawdown | Worst peak-to-trough P&L sequence |
| shadow_trade_count | Total shadow trades in window |

---

## Promotion Framework

### Staged Rollout

| Stage | Criteria to Enter | Behavior | Duration |
|-------|-------------------|----------|----------|
| **0. Collection** | Day 1 | Record observations only. No model, no predictions. | 3 days minimum |
| **1. Shadow** | >= 3 days of 1-min data | HMM trained, shadow predictions logged, no execution | Until promotion |
| **2. Paper** | Passes promotion gate | Shadow trades executed as limit orders that are immediately cancelled (tests order pipeline without risk) | 3 days |
| **3. Small-cap live** | Paper stage clean for 3 days | Live trades with max 1 contract, max 3 positions | 7 days |
| **4. Full live** | Small-cap validates | Replace current strategy entirely, full position sizing | Ongoing |

### Promotion Gate (Shadow → Paper)

ALL must be true over the evaluation window:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Evaluation window | >= 7 days of shadow data | Minimum statistical power |
| Shadow trade count | >= 30 trades in window | Enough samples to estimate win rate |
| Shadow net EV after fees | > 0 (positive) | Must be profitable after 4c/trade fees |
| Shadow net EV > live net EV | By at least 1c/trade | Must materially outperform |
| Shadow win rate | > live win rate | Must predict better |
| Shadow max drawdown | < 15% of bankroll | Risk control |
| Confidence interval | P(shadow_ev > 0) > 80% via bootstrap | Statistical significance — not just lucky |
| Model stability | No regime instability alerts in window | Model must be trustworthy |

### Demotion / Kill Switch

Automatic revert to prior strategy if ANY:
- Max drawdown exceeds 20% of bankroll in any 24-hour period
- 5 consecutive losing trades
- Model stability alert (>50% of states flagged unstable)
- HMM retrain fails or produces degenerate states (any state with <1% posterior mass across all observations)
- Live EV drops below shadow EV for 3 consecutive days after promotion

---

## Risk Controls

### Position Limits

| Limit | Value | Scope |
|-------|-------|-------|
| Max contracts per trade | 3 | Per trade |
| Max open positions | 5 | Across all assets |
| Max daily loss | 10% of bankroll | Per calendar day |
| Max per-asset exposure | 40% of bankroll | Per asset at any time |

### Execution Realism

- All shadow P&L calculations assume:
  - Entry at the ask price (YES) or ask price (NO) — worst case
  - Exit at the bid price — worst case
  - 2c per side fee (4c round trip)
  - 1c additional slippage assumption per side (6c total cost)
- Shadow trades that would have hit a contract with <5 volume are excluded from metrics (unrealistic fill)

### Regime Instability Detection

At each retrain:
- Compare state means, covariances, and transition matrix to prior retrain
- Flag if any state mean shifts >2 std
- Flag if any transition probability changes >0.15
- Flag if optimal K changes
- If >2 flags in a single retrain: freeze model, alert, continue with prior model

### Contract Lifecycle Handling

15-minute contracts are non-stationary — they converge to 0 or 100 at expiry. The contract evaluator must:
- Never evaluate a contract with <2 minutes remaining (time_to_expiry_frac < 0.13)
- Weight contract-level features by time-to-expiry (early observations are more informative about fair value)
- Not train the contract model on observations from the final 2 minutes (extreme convergence behavior is not predictive)

---

## Monitoring and Reporting

### Daily Retrain Report (Telegram)

```
HMM Daily Report (v{version})

Regime Model:
  BTC: K={n_states}, log-lik={ll:.1f}, stability=OK/WARN
  ETH: K={n_states}, log-lik={ll:.1f}, stability=OK/WARN  
  SOL: K={n_states}, log-lik={ll:.1f}, stability=OK/WARN
  
Current Regimes:
  BTC: State {top_state} ({posterior:.0%}), entropy={e:.2f}
  ETH: State {top_state} ({posterior:.0%}), entropy={e:.2f}
  SOL: State {top_state} ({posterior:.0%}), entropy={e:.2f}

Shadow Performance (7d rolling):
  Trades: {n} | Win rate: {wr:.0%} | EV/trade: {ev:+.1f}c
  vs Live: {n_live} trades | {wr_live:.0%} | {ev_live:+.1f}c
  
Promotion Status: {stage} ({days_in_stage}d)
  Gate: {pass_count}/{total_criteria} criteria met
```

### Real-Time Logging

Every trading cycle logs:
- Current regime posterior (top 3 states with probabilities)
- Shadow recommendation (trade/skip, side, edge, EV, confidence)
- Regime entropy
- Contract features for evaluated contracts

---

## Database Schema

### New Tables

```sql
CREATE TABLE hmm_observations (
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

CREATE INDEX idx_hmm_obs_asset_ts ON hmm_observations(asset, timestamp);

CREATE TABLE hmm_shadow_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset           TEXT NOT NULL,
    ticker          TEXT,
    timestamp       TEXT NOT NULL,
    regime_posterior TEXT,  -- JSON array of K floats
    regime_entropy  REAL,
    top_state       INTEGER,
    top_state_prob  REAL,
    fair_prob       REAL,
    market_price    REAL,
    edge_cents      REAL,
    ev_cents        REAL,
    confidence      REAL,
    recommendation  TEXT,  -- "buy_yes", "buy_no", "skip"
    position_size   INTEGER,
    outcome         TEXT,  -- NULL until resolved, then "win" or "loss"
    pnl_cents       REAL,  -- NULL until resolved
    resolved_at     TEXT
);

CREATE INDEX idx_hmm_shadow_asset_ts ON hmm_shadow_predictions(asset, timestamp);
CREATE INDEX idx_hmm_shadow_outcome ON hmm_shadow_predictions(outcome);

CREATE TABLE hmm_model_state (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    asset           TEXT NOT NULL,
    version         INTEGER NOT NULL,
    n_states        INTEGER NOT NULL,
    bic             REAL,
    log_likelihood  REAL,
    stability_flags INTEGER DEFAULT 0,
    state_means     TEXT,  -- JSON
    transition_matrix TEXT,  -- JSON
    trained_at      TEXT NOT NULL,
    observation_count INTEGER
);
```

---

## Implementation Notes

### Dependencies

- `hmmlearn >= 0.3.0` — Gaussian HMM implementation (scikit-learn compatible)
- `scipy` — already in requirements (for bootstrap confidence intervals)
- `numpy` — already in requirements

### File Structure

```
src/
  hmm_observations.py   # Feature pipeline: raw data → 1-min bars → DB
  hmm_regime.py          # Layer 1: HMM training, inference, state selection
  hmm_contract.py        # Layer 2: Contract fair value, edge, sizing
  hmm_shadow.py          # Shadow prediction logging and outcome tracking
  hmm_graduation.py      # Promotion logic, kill switch, stage management
```

### Integration Points

| Existing File | Change |
|---------------|--------|
| `trader.py` | Call `hmm_shadow.record_prediction()` each cycle; after graduation, call `hmm_contract.evaluate()` instead of current `evaluate_contract()` |
| `retrain.py` | Add HMM retrain step after existing retrain |
| `main.py` | Start observation pipeline thread; initialize HMM components |
| `requirements.txt` | Add `hmmlearn>=0.3.0` |

### Cold Start

Days 1-3: Observation pipeline runs, no model. All HMM features return defaults. Current strategy operates normally.

Day 4+: First HMM trained with ~6,000 observations per asset. Shadow mode begins. Initial K=5 (BIC selection begins once >7 days of data).

Day 11+: Earliest possible promotion gate evaluation (7 days of shadow data + 30 trades minimum).

---

## Open Questions

1. **Cross-asset regime correlation:** BTC regime likely influences ETH/SOL. Should Layer 1 include a cross-asset signal (e.g., BTC regime posterior as a feature for ETH/SOL models)? Deferred to V1.5 to keep V1 simple.

2. **Intraday seasonality:** Crypto 15M contract activity varies by time of day. Should observations be deseasonalized? Monitor first, add if feature distributions show strong time-of-day effects.

3. **Online learning:** Current design retrains daily. For faster adaptation, consider online Bayesian updates to the HMM parameters between daily retrains. Deferred to V1.5.

4. **Contract model complexity:** V1 uses logistic regression for contract evaluation. If feature interactions are important (e.g., moneyness * regime), LightGBM in V1.5 would capture this automatically.

5. **SOL inclusion:** SOL has lower volume and wider spreads than BTC/ETH. May need separate liquidity thresholds. Monitor in shadow mode before deciding.

---

## Major Changes from Prior Spec

| Prior Spec | Revised Spec | Rationale |
|------------|-------------|-----------|
| Single-layer HMM (regime + trade decision combined) | Two-layer architecture (regime model + contract evaluator) | Regime detection should be asset-level, not contract-level. Contracts are non-stationary with lifecycle effects that contaminate regime inference. |
| Fixed K=7 states | BIC-selected K from {3..8}, default K=5 | No justification for 7. Let data determine the right number of states. |
| Hard state assignments (Viterbi) | Full posterior distributions (forward-backward) | Hard assignments are overconfident during regime transitions. Posterior-weighted EV is more robust. |
| Fixed state-to-strategy lookup table | Posterior-weighted expected value computation | Binary "state 4 = trade" ignores uncertainty. Weighting across all states with data-driven profiles is more principled. |
| 3-day promotion window | 7-day window + 30 trade minimum + bootstrap CI + drawdown check | 3 days is statistically meaningless. 30 trades at 80% CI is the minimum for credible evaluation. |
| Shadow → Live (2 stages) | Collection → Shadow → Paper → Small-cap → Full (5 stages) | Staged rollout reduces risk. Paper trading tests the order pipeline without capital risk. |
| Log-normal model replaced entirely | Log-normal estimate preserved as a feature input to logistic regression | The log-normal estimate is useful information. Don't discard it — let the contract model learn how much to trust it per regime. |
| Hardcoded 7 features | 13 Layer 1 features + 9 Layer 2 features, clearly separated | Prior spec mixed contract and market features. Separation is critical for model stability. |
| Gaussian emissions assumed sufficient | Gaussian V1 with explicit upgrade path to Student-t mixtures | Crypto returns have fat tails. Monitor kurtosis, upgrade if needed. |
| No anti-leakage procedures | Walk-forward validation, 15-min purge buffer, timestamp enforcement | Backtesting without anti-leakage produces meaningless results. |
| No execution realism | Ask/bid entry/exit + 4c fees + 1c slippage per side | Shadow P&L must reflect real execution costs or promotion metrics are fiction. |
| No kill switch | Automatic demotion on drawdown, consecutive losses, or model instability | Live trading without circuit breakers is reckless. |

---

## Recommended V1 vs V1.5 Scope Split

### V1 (Build Now)

- Observation pipeline + DB tables
- Per-asset Gaussian HMM with BIC state selection
- Full posterior inference (forward-backward)
- Regime-conditioned logistic regression for contract fair value
- Posterior-weighted EV decision logic
- Shadow mode with outcome tracking
- Walk-forward daily retrain
- 5-stage promotion framework
- Kill switch and risk controls
- Daily Telegram report

### V1.5 (Build After 30 Days of V1 Data)

- Student-t mixture emissions (if kurtosis warrants)
- Cross-asset regime features (BTC posterior → ETH/SOL)
- LightGBM contract evaluator (replaces logistic regression)
- Online Bayesian parameter updates between retrains
- Intraday seasonality detrending
- Order book imbalance features (requires Kalshi WebSocket, not currently available)
- Regime-conditioned Kelly sizing with shrinkage estimator
- Multi-contract portfolio optimization (correlation-aware position limits)
