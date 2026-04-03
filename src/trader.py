import pandas as pd
import numpy as np
import logging
import time
from typing import List, Dict, Any
from config import BANKROLL, NEWS_SENTIMENT_THRESHOLD, STAT_ARBITRAGE_THRESHOLD, VOLATILITY_THRESHOLD, MAX_POSITION_SIZE_PERCENTAGE, STOP_LOSS_PERCENTAGE
from news_analyzer import NewsSentimentAnalyzer
from arbitrage_analyzer import StatisticalArbitrageAnalyzer
from volatility_analyzer import VolatilityAnalyzer
from risk_manager import RiskManager
from market_data_streamer import MarketDataStreamer
from performance_analytics import PerformanceAnalytics, Trade
from settings_manager import SettingsManager
from db import TradingDB
from retrain import load_current_params, PARAMS_PATH
from crypto_prices import get_default as get_crypto_prices
from price_predictor import predict_direction, evaluate_contract
import re


def _parse_strike_from_ticker(ticker: str) -> float:
    """Extract strike/target price from Kalshi market ticker.

    Examples:
      KXBTC-26APR020100-T85000      → $85,000
      KXETH-26APR020100-T2100       → $2,100
      KXBTCMAXMON-BTC-26APR30-7250000 → $72,500 (cents → dollars)
      KXSOLMINMON-SOL-26APR30-7500    → $75.00 (cents → dollars)
    """
    if not ticker:
        return 0.0

    # Match -T or -B followed by digits at end
    m = re.search(r'-[TB](\d+\.?\d*)$', ticker)
    if m:
        return float(m.group(1))

    # Monthly min/max tickers: last segment is pure digits (strike in cents)
    # e.g. KXBTCMAXMON-BTC-26APR30-7250000
    m = re.search(r'-(\d{4,})$', ticker)
    if m:
        raw = float(m.group(1))
        # Heuristic: BTC strikes > 10000 are in cents (divide by 100)
        # SOL/ETH strikes < 100000 could be cents or dollars
        # Use the asset prefix to decide
        if 'BTC' in ticker.upper() and raw > 100000:
            return raw / 100  # 7250000 → 72500
        elif 'ETH' in ticker.upper() and raw > 10000:
            return raw / 100  # 250000 → 2500
        elif 'SOL' in ticker.upper() and raw > 1000:
            return raw / 100  # 7500 → 75
        return raw

    return 0.0

def _dollar_to_cents(val) -> int:
    """Convert a dollar string like '0.5000' to cents (50)."""
    try:
        d = float(val)
        return round(d * 100) if d > 0 else 0
    except (ValueError, TypeError):
        return 0


def _get_market_price_cents(market: Dict[str, Any]) -> int:
    """Extract YES price in cents from Kalshi v2 market data."""
    for field in ('yes_ask_dollars', 'yes_bid_dollars'):
        val = market.get(field)
        c = _dollar_to_cents(val)
        if c > 0:
            return c

    # Legacy / streamer fields
    for field in ('yes_ask', 'last_price', 'current_price'):
        val = market.get(field)
        if val is not None:
            try:
                v = float(val)
                if v > 0:
                    return int(v) if v > 1 else round(v * 100)
            except (ValueError, TypeError):
                continue

    return 0


def _get_no_price_cents(market: Dict[str, Any]) -> int:
    """Extract actual NO price in cents from Kalshi v2 market data.

    Uses the real no_ask/no_bid fields — does NOT assume 100 - yes_price.
    """
    for field in ('no_ask_dollars', 'no_bid_dollars'):
        val = market.get(field)
        c = _dollar_to_cents(val)
        if c > 0:
            return c

    # Fallback: if no NO price fields, estimate from YES
    yes = _get_market_price_cents(market)
    return max(100 - yes, 1) if yes > 0 else 0

    return 0


class Trader:
    def __init__(self, api, notifier, logger, bankroll):
        self.api = api
        self.notifier = notifier
        self.logger = logger
        self.bankroll = bankroll
        self.current_positions = {}
        self.news_analyzer = NewsSentimentAnalyzer()
        self.arbitrage_analyzer = StatisticalArbitrageAnalyzer(min_history_points=5)
        self.volatility_analyzer = VolatilityAnalyzer(min_history_points=5)
        self.risk_manager = RiskManager(bankroll)
        self.db = TradingDB()
        self.crypto_prices = get_crypto_prices()
        self.model_params = load_current_params()
        self.logger.info(f"Loaded model params v{self.model_params.get('version', 0)}")

        # Restore open positions from database (survives restarts/deploys)
        restored = self.db.load_positions()
        if restored:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            kept = {}
            for mid, pos in restored.items():
                # Log each position for visibility
                self.logger.info(f"  Restored: {mid} | {pos.get('side')} x{pos.get('quantity')} "
                                 f"@ {pos.get('entry_price')}¢ | {pos.get('strategy')}")

                # Drop expired positions — they've already settled
                exp_str = pos.get('expiration_time')
                if exp_str:
                    try:
                        if exp_str.endswith('Z'):
                            exp_str = exp_str[:-1] + '+00:00'
                        exp_dt = datetime.fromisoformat(exp_str)
                        if now > exp_dt:
                            self.logger.info(f"  → Dropping expired position: {mid}")
                            self.db.delete_position(mid)
                            continue
                    except (ValueError, TypeError):
                        pass

                kept[mid] = pos

            self.current_positions = kept
            self.logger.info(f"Restored {len(kept)} open positions ({len(restored) - len(kept)} expired/dropped)")

        # Phase 3: Enhanced market data — use longer interval to avoid rate limits
        crypto_events = self._build_crypto_event_tickers()
        self.market_data_streamer = MarketDataStreamer(
            api, update_interval=60, event_tickers=crypto_events  # Every 1 min for 15-min markets
        )
        self.performance_analytics = PerformanceAnalytics()

        # Phase 4: Dynamic settings management
        self.settings_manager = SettingsManager()
        self.settings_manager.add_change_listener(self._on_settings_changed)

        # Subscribe to market data updates for real-time monitoring
        self.market_data_streamer.add_subscriber(self._on_market_data_update)

    def _on_settings_changed(self, changed_settings: Dict[str, Any]):
        """Handle dynamic settings changes."""
        self.logger.info(f"Settings updated: {list(changed_settings.keys())}")

        # Update market data streamer interval if changed
        if 'market_data_update_interval' in changed_settings:
            new_interval = self.settings_manager.settings.market_data_update_interval
            self.market_data_streamer.update_interval = new_interval
            self.logger.info(f"Market data update interval changed to {new_interval}s")

        # Update risk manager settings if changed
        if any(key in changed_settings for key in ['kelly_fraction', 'max_position_size_pct', 'stop_loss_pct']):
            # Risk manager will use updated settings automatically
            self.logger.info("Risk management settings updated")

        # Notify via telegram if enabled
        if self.settings_manager.settings.telegram_notifications:
            changes_summary = ", ".join([f"{k}: {v['old_value']} → {v['new_value']}"
                                       for k, v in changed_settings.items()])
            self.notifier.send_trade_notification(f"⚙️ Settings Updated: {changes_summary}")

    def _on_market_data_update(self, updated_markets: List[str], all_market_data: Dict[str, Any]):
        """Handle real-time market data updates."""
        # Check for stop-loss triggers on open positions
        current_prices = {market_id: data.current_price
                         for market_id, data in all_market_data.items()}

        self.check_positions_for_risk_management(current_prices)

        # Log significant market movements
        for market_id in updated_markets:
            if market_id in all_market_data:
                market_data = all_market_data[market_id]
                if market_data.price_change_pct and abs(market_data.price_change_pct) > 2.0:
                    self.logger.info(f"Market movement: {market_data.title} "
                                   f"changed {market_data.price_change_pct:.2f}% "
                                   f"to ${market_data.current_price:.2f}")

    def analyze_market(self, market_data):
        # Enhanced analysis with news sentiment
        return self._make_trade_decision(market_data)

    def _make_trade_decision(self, market_data):
        """
        Enhanced trade decision making with multiple strategies using dynamic settings
        Priority: News Sentiment → Statistical Arbitrage → Volatility Analysis
        """
        trade_decision = None
        settings = self.settings_manager.settings

        # Strategy 1: News Sentiment Analysis (if enabled)
        if settings.news_sentiment_enabled:
            try:
                sentiment_analysis = self.news_analyzer.get_market_relevant_news()
                # Record sentiment to db for each crypto asset
                for asset in ["BTC", "ETH", "SOL"]:
                    self.db.record_news_sentiment(
                        asset,
                        overall_sentiment=sentiment_analysis.get("overall_sentiment", 0),
                        confidence=sentiment_analysis.get("confidence", 0),
                        article_count=sentiment_analysis.get("article_count", 0),
                        positive_count=sentiment_analysis.get("positive_articles", 0),
                        negative_count=sentiment_analysis.get("negative_articles", 0),
                        neutral_count=sentiment_analysis.get("neutral_articles", 0),
                    )
                sentiment_decision = self.news_analyzer.should_trade_based_on_sentiment(
                    sentiment_analysis, settings.news_sentiment_threshold
                )

                if sentiment_decision['should_trade']:
                    self.logger.info(f"News sentiment signal: {sentiment_decision['reason']}")

                    # Find the best market by picking the one with the most
                    # extreme price (closest to 0 or 1) as it has the most
                    # upside potential on a directional sentiment bet.
                    if market_data and 'markets' in market_data and market_data['markets']:
                        market = self._pick_best_market(market_data['markets'], sentiment_decision['direction'])
                        event_id = market.get('ticker') or market.get('id')
                        current_price = _get_market_price_cents(market)

                        if event_id and current_price:
                            # On Kalshi: long = buy YES, short = buy NO
                            side = 'yes' if sentiment_decision['direction'] == 'long' else 'no'
                            trade_price = current_price if side == 'yes' else _get_no_price_cents(market)

                            # Apply dynamic risk management
                            position_size_fraction = self.risk_manager.calculate_position_size_kelly(sentiment_decision['confidence'])
                            position_value = self.risk_manager.current_bankroll * position_size_fraction
                            quantity = max(1, int(position_value / trade_price)) if trade_price > 0 else 1

                            trade_decision = {
                                'event_id': event_id,
                                'action': 'buy',
                                'side': side,
                                'quantity': quantity,
                                'price': trade_price,
                                'strategy': 'news_sentiment',
                                'sentiment_score': sentiment_decision['sentiment_score'],
                                'confidence': sentiment_decision['confidence'],
                                'title': market.get('title', event_id),
                                'reason': sentiment_decision['reason'],
                                'expiration_time': market.get('expected_expiration_time') or market.get('expiration_time'),
                            }

                            self.logger.info(f"News sentiment trade decision: {action} {event_id} "
                                           f"at {current_price} (sentiment: {sentiment_decision['sentiment_score']:.3f})")

            except Exception as e:
                self.logger.error(f"Error in news sentiment analysis: {e}")

        # Strategy 2: Statistical Arbitrage (if enabled and no sentiment signal)
        if not trade_decision and settings.statistical_arbitrage_enabled:
            try:
                arbitrage_opportunities = self._statistical_arbitrage(market_data)
                if arbitrage_opportunities:
                    # Take the highest confidence opportunity
                    best_opportunity = arbitrage_opportunities[0]
                    execution_decision = self.arbitrage_analyzer.should_execute_arbitrage(
                        best_opportunity, risk_tolerance=settings.stat_arbitrage_threshold
                    )

                    if execution_decision['should_execute']:
                        self.logger.info(f"Arbitrage signal: {execution_decision['reason']}")

                        # For simplicity, focus on one side of the arbitrage pair
                        market1 = execution_decision['market1']
                        market2 = execution_decision['market2']

                        if best_opportunity['signal'] == 'LONG_SPREAD':
                            event_id = market1['id']
                            action = 'buy'
                        else:  # SHORT_SPREAD
                            event_id = market1['id']
                            action = 'sell'

                        # Apply dynamic risk management
                        position_size_fraction = self.risk_manager.calculate_position_size_kelly(execution_decision['confidence'])
                        position_value = self.risk_manager.current_bankroll * position_size_fraction
                        quantity = max(1, int(position_value / market1['current_price']))

                        trade_decision = {
                            'event_id': event_id,
                            'action': action,
                            'quantity': quantity,
                            'price': market1['current_price'],
                            'strategy': 'statistical_arbitrage',
                            'z_score': best_opportunity['z_score'],
                            'confidence': execution_decision['confidence'],
                            'arbitrage_pair': [market1['id'], market2['id']]
                        }

                        self.logger.info(f"Arbitrage trade decision: {action} {event_id} "
                                       f"(z-score: {best_opportunity['z_score']:.3f})")

            except Exception as e:
                self.logger.error(f"Error in statistical arbitrage: {e}")

        # Strategy 3: Volatility Analysis (if enabled and no other signals)
        if not trade_decision and settings.volatility_based_enabled:
            try:
                volatility_decision = self._volatility_analysis(market_data)
                if volatility_decision and volatility_decision.get('should_trade'):
                    self.logger.info(f"Volatility signal: {volatility_decision['reason']}")

                    market = volatility_decision.get('market')
                    if market:
                        event_id = market.get('ticker') or market.get('id')
                        current_price = _get_market_price_cents(market)

                        if event_id and current_price and volatility_decision.get('direction'):
                            side = 'yes' if volatility_decision['direction'] == 'long' else 'no'
                            trade_price = current_price if side == 'yes' else _get_no_price_cents(market)

                            # Apply dynamic risk management
                            position_size_fraction = self.risk_manager.calculate_position_size_kelly(volatility_decision['confidence'])
                            position_value = self.risk_manager.current_bankroll * position_size_fraction
                            quantity = max(1, int(position_value / trade_price)) if trade_price > 0 else 1

                            trade_decision = {
                                'event_id': event_id,
                                'action': 'buy',
                                'side': side,
                                'quantity': quantity,
                                'price': trade_price,
                                'strategy': 'volatility_based',
                                'volatility_regime': volatility_decision.get('volatility_regime'),
                                'confidence': volatility_decision['confidence'],
                                'signal_type': volatility_decision.get('signal_type'),
                                'title': market.get('title', event_id),
                                'reason': volatility_decision.get('reason', 'Volatility signal'),
                                'expiration_time': market.get('expected_expiration_time') or market.get('expiration_time'),
                            }

                            self.logger.info(f"Volatility trade decision: {action} {event_id} "
                                           f"(regime: {volatility_decision.get('volatility_regime')})")

            except Exception as e:
                self.logger.error(f"Error in volatility analysis: {e}")

        # Strategy 4: Value betting fallback — make small bets to generate data
        # Fires when all other strategies produce no signal
        if not trade_decision:
            try:
                trade_decision = self._value_bet_fallback(market_data, getattr(self, '_current_spot_prices', None))
            except Exception as e:
                self.logger.error(f"Error in value bet fallback: {e}")

        # Record decision to db (all strategies, for data collection)
        if trade_decision:
            self.db.record_trade_decision(
                trade_decision.get("event_id", ""),
                strategy=trade_decision.get("strategy", ""),
                direction=trade_decision.get("action", ""),
                confidence=trade_decision.get("confidence", 0),
                sentiment_score=trade_decision.get("sentiment_score"),
                should_trade=True,
            )

        # Only execute trades on 15-minute contracts — collect data on everything else
        if trade_decision:
            ticker = trade_decision.get('event_id', '')
            if '15M' not in ticker.upper():
                self.logger.info(f"Skipping non-15M trade (data only): {ticker}")
                return None

        return trade_decision

    def _build_crypto_event_tickers(self):
        """Build list of BTC/ETH/SOL event tickers to watch.

        IMPORTANT: Kalshi 15-min tickers use EASTERN TIME, not UTC.
        e.g. KXBTC15M-26APR022315 = Apr 2 at 11:15 PM ET
        """
        from datetime import datetime, timedelta, timezone

        tickers = []
        now_utc = datetime.now(timezone.utc)

        # Eastern Time = UTC - 4 (EDT) or UTC - 5 (EST)
        # April is EDT (daylight saving)
        ET_OFFSET = timedelta(hours=-4)
        now_et = now_utc + ET_OFFSET

        # Monthly min/max for current month
        next_m = now_et.month + 1 if now_et.month < 12 else 1
        next_y = now_et.year if now_et.month < 12 else now_et.year + 1
        end = datetime(next_y, next_m, 1) - timedelta(days=1)
        dt = end.strftime("%y%b%d").upper()
        for coin, code in [("BTC", "BTC"), ("ETH", "ETH"), ("SOL", "SOL")]:
            tickers.append(f"KX{coin}MAXMON-{code}-{dt}")
            tickers.append(f"KX{coin}MINMON-{code}-{dt}")

        # Daily price events for next 7 days (ET dates)
        for days_ahead in range(0, 7):
            d = now_et + timedelta(days=days_ahead)
            ds = d.strftime("%y%b%d").upper()
            for coin in ["KXBTC", "KXETH", "KXSOL"]:
                tickers.append(f"{coin}-{ds}0100")

        # 15-minute events — use EASTERN TIME for ticker generation
        # Cover past 30 min (catch closing windows) + next 2 hours
        for mins_offset in range(-30, 120, 15):
            t = now_et + timedelta(minutes=mins_offset)
            ds = t.strftime("%y%b%d").upper()
            m15 = (t.minute // 15) * 15
            hm = f"{t.hour:02d}{m15:02d}"
            for coin in ["KXBTC15M", "KXETH15M", "KXSOL15M"]:
                tickers.append(f"{coin}-{ds}{hm}")

        # Deduplicate
        return list(dict.fromkeys(tickers))

    def _fetch_all_markets(self):
        """Fetch active BTC/ETH/SOL crypto markets with liquidity.

        Caches results and refreshes every 5 cycles.
        """
        if not hasattr(self, '_market_cache'):
            self._market_cache = []
            self._cache_cycle = 0

        self._cache_cycle += 1

        # Refresh every 2 cycles (short-term 15-min tickers change frequently)
        if self._market_cache and self._cache_cycle % 2 != 1:
            self.logger.info(f"Using cached crypto markets ({len(self._market_cache)} markets)")
            return self._market_cache

        event_tickers = self._build_crypto_event_tickers()
        all_markets = []
        for event_ticker in event_tickers:
            time.sleep(0.5)  # Rate limit: max 2 requests/sec
            resp = self.api.get_markets(params={"event_ticker": event_ticker, "limit": 200})
            if resp and resp.get("markets"):
                for m in resp["markets"]:
                    if m.get("status") != "active":
                        continue
                    ticker = m.get("ticker", "")
                    has_liquidity = m.get("yes_bid_dollars", "0.0000") != "0.0000"
                    is_15m = "15M" in ticker.upper()
                    # Include 15-min contracts even without bids (they open with no liquidity)
                    if has_liquidity or is_15m:
                        all_markets.append(m)

        self._market_cache = all_markets
        self.logger.info(f"Fetched {len(all_markets)} active BTC/ETH/SOL markets with liquidity")
        return all_markets

    def _reload_model_params(self):
        """Reload model params if they've been updated by retraining."""
        try:
            new_params = load_current_params()
            if new_params.get("version", 0) != self.model_params.get("version", 0):
                self.model_params = new_params
                self.logger.info(f"Reloaded model params v{new_params['version']} "
                               f"(trained at {new_params.get('trained_at', 'unknown')})")
        except Exception:
            pass

    def run_trading_strategy(self):
        """Main trading loop iteration: check exits, fetch markets, analyse, execute."""
        try:
            self._reload_model_params()

            # Fetch and record live crypto spot prices
            spot_prices = self.crypto_prices.get_prices()
            self._current_spot_prices = spot_prices  # Store for use in strategies
            for asset, data in spot_prices.items():
                self.db.record_crypto_price(asset, data["price"], data.get("change_24h"))
            if spot_prices:
                spot_str = ", ".join(f"{a}: ${d['price']:,.0f} ({d.get('change_24h', 0):+.1f}%)" for a, d in spot_prices.items())
                self.logger.info(f"Spot prices: {spot_str}")

            markets = self._fetch_all_markets()
            if not markets:
                self.logger.info("No markets returned from API")
                return

            # Check exits FIRST — take profit, stop loss, time exit
            self.check_exits(markets)

            # Record market snapshots to db — including strike, spot, and expiration
            for m in markets:
                ticker = m.get("ticker", "")
                asset = None
                for prefix, name in [("KXBTC", "BTC"), ("KXETH", "ETH"), ("KXSOL", "SOL")]:
                    if ticker.upper().startswith(prefix):
                        asset = name
                        break

                # Get current spot price for this asset
                current_spot = None
                if asset and spot_prices and asset in spot_prices:
                    current_spot = spot_prices[asset].get("price")

                self.db.record_market_snapshot(
                    ticker,
                    title=m.get("title", ""),
                    yes_bid=float(m.get("yes_bid_dollars", 0) or 0),
                    yes_ask=float(m.get("yes_ask_dollars", 0) or 0),
                    no_bid=float(m.get("no_bid_dollars", 0) or 0),
                    no_ask=float(m.get("no_ask_dollars", 0) or 0),
                    volume=float(m.get("volume_24h_fp", 0) or 0),
                    strike_price=_parse_strike_from_ticker(ticker),
                    spot_price=current_spot,
                    expiration_time=m.get("expected_expiration_time") or m.get("expiration_time") or m.get("close_time"),
                )

            market_data = {"markets": markets}
            trade_decision = self._make_trade_decision(market_data)
            if trade_decision:
                self.execute_trade(trade_decision)
            else:
                self.logger.info(f"No trade signal this cycle ({len(markets)} markets scanned)")
        except Exception as e:
            self.logger.error(f"Error in trading strategy: {e}")

    def _statistical_arbitrage(self, market_data):
        """Find statistical arbitrage opportunities across markets."""
        markets = market_data.get("markets", [])
        # Build enriched list with price history from the streamer
        enriched = []
        for m in markets:
            mid = m.get("ticker") or m.get("id")
            streamer_data = self.market_data_streamer.get_market_data(mid)
            history = streamer_data.price_history if streamer_data else []
            enriched.append({
                "id": mid,
                "title": m.get("title", ""),
                "current_price": _get_market_price_cents(m),
                "price_history": history,
            })
        return self.arbitrage_analyzer.find_arbitrage_opportunities(enriched)

    def _volatility_analysis(self, market_data):
        """Run volatility analysis across markets and return best signal."""
        markets = market_data.get("markets", [])
        best = None
        for m in markets:
            mid = m.get("ticker") or m.get("id")
            streamer_data = self.market_data_streamer.get_market_data(mid)
            history = streamer_data.price_history if streamer_data else []
            if len(history) < 5:
                continue
            vol_analysis = self.volatility_analyzer.analyze_market_volatility({
                "id": mid,
                "title": m.get("title", ""),
                "current_price": _get_market_price_cents(m),
                "price_history": history,
            })
            decision = self.volatility_analyzer.should_trade_based_on_volatility(vol_analysis)
            if decision.get("should_trade"):
                if best is None or decision["confidence"] > best["confidence"]:
                    best = decision
                    best["market"] = m
        return best

    def _value_bet_fallback(self, market_data, spot_prices=None):
        """
        Probability-based value bet on 15-minute BTC/ETH/SOL contracts.

        Two trading windows per 15-min cycle:
        1. OPENING (0-3 min in): max uncertainty, use momentum prediction
           to bet early when YES/NO are near 50/50
        2. CLOSING (last 3 min): trend is clear, bet with conviction
           when price has moved decisively toward YES or NO

        Falls back to monthly contracts when no 15-min are available.
        """
        from db import extract_asset

        markets = market_data.get("markets", [])
        if not markets:
            return None

        # Rate limit: max 1 value bet per 5 minutes
        if not hasattr(self, '_last_value_bet_time'):
            self._last_value_bet_time = 0
        if time.time() - self._last_value_bet_time < 900:  # 15 min cooldown
            return None

        # Count positions separately — monthly and 15-min have different limits
        monthly_positions = sum(
            1 for ticker, p in self.current_positions.items()
            if p.get('strategy') == 'value_bet' and '15M' not in ticker.upper()
        )
        short_positions = sum(
            1 for ticker, p in self.current_positions.items()
            if p.get('strategy') == 'value_bet' and '15M' in ticker.upper()
        )
        # Max 3 monthly + 3 active 15-min positions
        if monthly_positions >= 3 and short_positions >= 3:
            return None

        # Find tradeable contracts — prefer 15-min, allow monthly
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        contracts_15m = []
        contracts_daily = []
        mp = self.model_params
        min_price = mp.get("min_entry_price_cents", 20)
        max_price = mp.get("max_entry_price_cents", 80)

        for m in markets:
            ticker = m.get('ticker') or m.get('id')
            if not ticker or ticker in self.current_positions:
                continue

            # Skip already expired contracts
            exp_str = m.get('expected_expiration_time') or m.get('expiration_time') or ''
            if exp_str:
                try:
                    exp_dt = datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
                    if exp_dt < now:
                        continue
                except (ValueError, TypeError):
                    pass

            yes_price = _get_market_price_cents(m)
            no_price = _get_no_price_cents(m)
            if not yes_price or yes_price <= 0:
                continue
            # Skip markets outside our price range
            if yes_price < min_price or yes_price > max_price:
                continue
            asset = extract_asset(ticker)
            entry = {
                'market': m,
                'ticker': ticker,
                'yes_price': yes_price,
                'no_price': no_price,
                'asset': asset,
            }
            if '15M' in ticker.upper():
                contracts_15m.append(entry)
            else:
                contracts_daily.append(entry)

        # Only trade 15-minute contracts (still collecting data on others above)
        contracts = contracts_15m
        if not contracts:
            return None

        is_15m = True

        # Evaluate each contract — aware of opening vs closing window
        best_trade = None
        best_edge = 0

        self.logger.info(f"Value bet evaluating: {len(contracts_15m)} 15-min, {len(contracts_daily)} other contracts")

        for c in contracts:
            ticker = c['ticker']
            asset = c['asset']
            strike = _parse_strike_from_ticker(ticker)

            # For 15-min contracts, get target from floor_strike or yes_sub_title
            if strike <= 0:
                # floor_strike has the target as a number (e.g. 66521.68)
                floor_strike = c['market'].get('floor_strike')
                if floor_strike:
                    try:
                        strike = float(floor_strike)
                    except (ValueError, TypeError):
                        pass

                # Fallback: parse from yes_sub_title (populated after finalization)
                if strike <= 0:
                    sub_title = c['market'].get('yes_sub_title', '')
                    target_match = re.search(r'[\$]([\d,]+\.?\d*)', sub_title)
                    if target_match:
                        strike = float(target_match.group(1).replace(',', ''))

                if '15M' in ticker.upper():
                    self.logger.info(f"  15M {ticker}: strike={strike}, yes={c['yes_price']}¢, no={c['no_price']}¢")

            if strike <= 0:
                continue

            # Get spot price
            spot = None
            if spot_prices and asset in spot_prices:
                spot = spot_prices[asset].get("price")
            if not spot:
                spot = self.crypto_prices.get_price(asset)
            if not spot:
                continue

            exp_time = c['market'].get('expected_expiration_time') or c['market'].get('expiration_time')

            # Determine timing within the 15-min window
            window_phase = "middle"
            minutes_left = None
            if exp_time:
                try:
                    from datetime import datetime, timezone
                    now = datetime.now(timezone.utc)
                    exp_dt = datetime.fromisoformat(exp_time.replace("Z", "+00:00"))
                    minutes_left = (exp_dt - now).total_seconds() / 60
                    if minutes_left > 12:
                        window_phase = "opening"  # First 3 min of window
                    elif minutes_left < 3:
                        window_phase = "closing"  # Last 3 min
                except (ValueError, TypeError):
                    pass

            evaluation = evaluate_contract(
                db=self.db,
                crypto_prices=self.crypto_prices,
                ticker=ticker,
                strike_price=strike,
                spot_price=spot,
                yes_price_cents=c['yes_price'],
                no_price_cents=c['no_price'],
                expiration_time=exp_time,
                asset=asset,
                momentum_weight=self.model_params.get("momentum_weight", 1.0),
            )

            mom = evaluation.get("momentum", {})
            mom_str = f"mom={evaluation.get('momentum_adj', 0):+.1%}" if mom.get("has_data") else "mom=N/A"
            self.logger.info(
                f"    → P={evaluation['probability']:.0%} (base={evaluation.get('base_probability', 0):.0%}, {mom_str}), "
                f"fair_yes={evaluation['fair_value_yes']:.0f}¢, "
                f"edge_yes={evaluation['edge_yes']:+.0f}¢, edge_no={evaluation['edge_no']:+.0f}¢, "
                f"rec={evaluation['recommendation']}"
            )

            edge_yes = evaluation.get("edge_yes", 0)
            edge_no = evaluation.get("edge_no", 0)
            best_edge_here = max(edge_yes, edge_no)

            if '15M' in ticker.upper():
                self.logger.info(f"    → P={evaluation.get('probability', 0):.0%}, "
                               f"fair_yes={evaluation.get('fair_value_yes', 0):.0f}¢, "
                               f"edge_yes={edge_yes:+.0f}¢, edge_no={edge_no:+.0f}¢, "
                               f"rec={evaluation['recommendation']}")

            if evaluation["recommendation"] == "skip":
                continue

            edge = best_edge_here

            # Boost edge based on window timing
            if is_15m:
                if window_phase == "opening":
                    # Opening: most uncertainty, biggest mispricings
                    edge *= 1.3
                    evaluation["reasons"].append(f"Opening window ({minutes_left:.0f}min left)")
                elif window_phase == "closing":
                    # Closing: trend is clear, high conviction
                    # Only trade if price has moved decisively (far from 50/50)
                    yes_p = c['yes_price']
                    if yes_p < 25 or yes_p > 75:
                        edge *= 1.5  # Strong trend, boost
                        evaluation["reasons"].append(f"Closing window ({minutes_left:.0f}min left, price decisive)")
                    else:
                        edge *= 0.5  # Still uncertain near close, reduce
                        evaluation["reasons"].append(f"Closing but uncertain ({minutes_left:.0f}min left)")

            if edge > best_edge:
                best_edge = edge
                side = "yes" if evaluation["recommendation"] == "buy_yes" else "no"
                trade_price = c['yes_price'] if side == 'yes' else c['no_price']
                best_trade = {
                    'contract': c,
                    'prediction': evaluation,
                    'side': side,
                    'trade_price': trade_price,
                    'asset': asset,
                }

        if not best_trade:
            self.logger.info("Value bet: no mispriced contracts found")
            return None

        c = best_trade['contract']
        p = best_trade['prediction']

        self._last_value_bet_time = time.time()

        reasons_str = "; ".join(p.get("reasons", [])[:3])
        self.logger.info(f"Value bet: {best_trade['asset']} {c['ticker']} — "
                        f"P(strike)={p.get('probability', 0):.1%}, "
                        f"fair={p.get('fair_value_yes', 50):.0f}¢, "
                        f"market={c['yes_price']}¢, "
                        f"edge={best_edge:+.0f}¢ — "
                        f"BUY {best_trade['side'].upper()} at {best_trade['trade_price']}¢")

        # Size up on large mispricings: 3x when edge > 30¢, 2x when > 15¢
        quantity = 1
        if best_edge > 30:
            quantity = 3
        elif best_edge > 15:
            quantity = 2

        return {
            'event_id': c['ticker'],
            'action': 'buy',
            'side': best_trade['side'],
            'quantity': quantity,
            'price': best_trade['trade_price'],
            'strategy': 'value_bet',
            'confidence': p['confidence'],
            'title': c['market'].get('title', c['ticker']),
            'reason': f"{'15-min' if is_15m else 'Monthly'} {best_trade['asset']} — {p.get('recommendation', 'buy').upper()} (P={p.get('probability', 0):.0%}, edge={best_edge:+.0f}¢): {reasons_str}",
            'expiration_time': c['market'].get('expected_expiration_time') or c['market'].get('expiration_time'),
            'edge_cents': round(best_edge, 1),
            'predicted_probability': p.get('probability', 0),
            'fair_value': p.get('fair_value_yes' if best_trade['side'] == 'yes' else 'fair_value_no', 50),
            'spot_price': spot,
            'strike_price': strike,
        }

    def _pick_best_market(self, markets, direction):
        """Pick the market with the most upside for the given direction."""
        scored = []
        for m in markets:
            price = _get_market_price_cents(m)
            if not price or price <= 0:
                continue
            # For a long bet we want cheap yes contracts; for short, expensive ones
            price_frac = price / 100
            if direction == 'long':
                score = 1.0 - price_frac
            else:
                score = price_frac
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else markets[0]

    def execute_trade(self, trade_decision):
        """
        Execute trade via the Kalshi API with risk management.
        """
        if not trade_decision:
            self.logger.info("No trade decision to execute.")
            return

        event_id = trade_decision['event_id']
        action = trade_decision['action']
        quantity = trade_decision['quantity']
        price = trade_decision['price']
        strategy = trade_decision.get('strategy', 'unknown')

        # Price from strategies is already in cents (1-99) via _get_market_price_cents
        price_cents = int(price)

        try:
            # Validate position size
            position_value = quantity * price_cents
            if not self.risk_manager.validate_position_size(position_value):
                self.logger.warning(f"Position size {position_value}¢ exceeds risk limits")
                return

            self.logger.info(f"Executing {strategy} trade: {action} {quantity} units of {event_id} "
                           f"at {price_cents}¢")

            # Generate unique trade ID
            trade_id = f"{strategy}_{event_id}_{int(time.time())}"

            # Build Kalshi v2 order payload
            # On Kalshi we always BUY — either YES or NO side
            # side comes from trade_decision if set, otherwise derive from action
            side = trade_decision.get('side', 'yes' if action.lower() == 'buy' else 'no')
            order_type = 'market' if price_cents < 3 else 'limit'
            order_payload = {
                'ticker': event_id,
                'action': 'buy',
                'side': side,
                'count': quantity,
                'type': order_type,
            }
            if side == 'yes':
                order_payload['yes_price'] = price_cents
            else:
                order_payload['no_price'] = price_cents
            # Remove None values
            order_payload = {k: v for k, v in order_payload.items() if v is not None}
            self.logger.info(f"Order payload: {order_payload}")

            result = self.api.create_order(order_payload)
            if result:
                self.logger.info(f"ORDER PLACED: {result}")
                self.db.record_trade(
                    event_id,
                    side=side,
                    quantity=quantity,
                    price=price_cents,
                    strategy=strategy,
                    order_result=str(result),
                    edge_cents=trade_decision.get('edge_cents'),
                    predicted_prob=trade_decision.get('predicted_probability'),
                    fair_value=trade_decision.get('fair_value'),
                    spot_price=trade_decision.get('spot_price'),
                    strike_price=trade_decision.get('strike_price'),
                )
            else:
                self.logger.error(f"Order failed for {event_id}")
                return

            # Record trade in performance analytics
            trade = Trade(
                trade_id=trade_id,
                market_id=event_id,
                strategy=strategy,
                side=action.lower(),
                quantity=quantity,
                entry_price=price,
                confidence=trade_decision.get('confidence', 0.5)
            )
            self.performance_analytics.record_trade(trade)

            # Store position locally and persist to database
            position = {
                'quantity': quantity,
                'entry_price': price_cents,
                'side': side,
                'type': 'long' if action.lower() == 'buy' else 'short',
                'strategy': strategy,
                'trade_id': trade_id,
                'title': trade_decision.get('title', event_id),
                'expiration_time': trade_decision.get('expiration_time'),
                'opened_at': time.time(),
            }
            self.current_positions[event_id] = position
            self.db.save_position(event_id, position)

            # Send clean trade notification
            title = trade_decision.get('title', event_id)
            reason = trade_decision.get('reason', strategy)
            price_dollars = f"${price_cents / 100:.2f}"
            self.notifier.send_message(
                f"🔔 Trade Opened\n\n"
                f"{title}\n"
                f"{side.upper()} {quantity} contract{'s' if quantity > 1 else ''} @ {price_dollars}\n"
                f"Strategy: {strategy}\n"
                f"Reason: {reason}"
            )

        except Exception as e:
            self.logger.error(f"Error executing {strategy} trade for {event_id}: {e}")
            self.notifier.send_error_notification(f"Trade execution error for {event_id}: {e}")

    def check_exits(self, markets: List[Dict[str, Any]]):
        """Check all open positions for exits.

        Exit priority:
        1. Hard stop loss — kill losers fast (10% drop from entry)
        2. Breakeven stop — once up 15%, move stop to entry price
        3. Trailing stop — once up 25%, trail 20% behind high-water mark
        4. Time exit — close 2 min before expiry
        (No fixed take-profit ceiling — let winners ride via trailing stop)
        """
        from datetime import datetime, timezone

        if not self.current_positions:
            return

        # Build price lookup from current market data
        market_lookup = {}
        for m in markets:
            ticker = m.get('ticker', '')
            if ticker:
                market_lookup[ticker] = m

        positions_to_close = []

        for market_id, position in self.current_positions.items():
            entry = position['entry_price']  # Price we paid (YES or NO cents)
            m = market_lookup.get(market_id)
            if not m:
                continue

            # Get actual current price for the side we hold
            if position['side'] == 'yes':
                current_side_price = _get_market_price_cents(m)
            else:
                current_side_price = _get_no_price_cents(m)

            if not current_side_price:
                continue

            # Track high-water mark for trailing stop
            if 'high_water' not in position or current_side_price > position['high_water']:
                position['high_water'] = current_side_price

            high_water = position['high_water']

            reason = None
            sl_pct = self.model_params.get("stop_loss_pct", 0.10)          # Kill losers at 10%
            breakeven_trigger = self.model_params.get("breakeven_trigger", 0.15)  # Move stop to entry at +15%
            trail_trigger = self.model_params.get("trail_trigger", 0.25)    # Start trailing at +25%
            trail_pct = self.model_params.get("trail_pct", 0.20)            # Trail 20% behind peak
            time_exit_secs = self.model_params.get("time_exit_seconds", 120)

            gain_pct = (current_side_price - entry) / entry if entry > 0 else 0
            peak_gain_pct = (high_water - entry) / entry if entry > 0 else 0
            drop_from_peak_pct = (high_water - current_side_price) / high_water if high_water > 0 else 0

            # --- Exit logic (priority order) ---

            # 1. Hard stop loss: kill losers fast
            if gain_pct <= -sl_pct:
                reason = f"Stop loss ({gain_pct:+.0%} from entry)"

            # 2. Trailing stop: once we've been up 25%+, trail 20% behind the peak
            if not reason and peak_gain_pct >= trail_trigger:
                if drop_from_peak_pct >= trail_pct:
                    reason = f"Trailing stop (peak {high_water}¢, dropped {drop_from_peak_pct:.0%} from peak)"

            # 3. Breakeven stop: once we've been up 15%, don't let it become a loss
            if not reason and peak_gain_pct >= breakeven_trigger:
                if current_side_price <= entry:
                    reason = f"Breakeven stop (was +{peak_gain_pct:.0%}, now back to entry)"

            # 4. Time exit: close before expiration
            if not reason and position.get('expiration_time'):
                try:
                    exp_str = position['expiration_time']
                    if exp_str.endswith('Z'):
                        exp_str = exp_str[:-1] + '+00:00'
                    exp_time = datetime.fromisoformat(exp_str)
                    now = datetime.now(timezone.utc)
                    seconds_left = (exp_time - now).total_seconds()
                    if 0 < seconds_left < time_exit_secs:
                        reason = f"Time exit ({int(seconds_left)}s before expiry)"
                except Exception:
                    pass

            if reason:
                positions_to_close.append({
                    'market_id': market_id,
                    'market': m,
                    'reason': reason,
                })

        for close_info in positions_to_close:
            self.close_position(close_info['market_id'],
                               close_info['market'],
                               close_info['reason'])

    def close_position(self, market_id: str, market: Dict[str, Any], reason: str):
        """Close a position by selling via the Kalshi API.

        Uses actual market bid prices for the side we hold.
        """
        if market_id not in self.current_positions:
            return

        position = self.current_positions[market_id]
        quantity = position['quantity']
        entry = position['entry_price']
        side = position['side']
        title = position.get('title', market_id)

        # Get actual sell price for our side (use bid — that's what we can sell at)
        if side == 'yes':
            sell_price = _dollar_to_cents(market.get('yes_bid_dollars')) or _get_market_price_cents(market)
        else:
            sell_price = _dollar_to_cents(market.get('no_bid_dollars')) or _get_no_price_cents(market)

        # Clamp to Kalshi's valid price range (1-99 cents)
        sell_price = max(1, min(sell_price, 99))

        try:
            # To close: sell as limit order at the current bid price
            order_payload = {
                'ticker': market_id,
                'action': 'sell',
                'side': side,
                'count': quantity,
                'type': 'limit',
            }
            if side == 'yes':
                order_payload['yes_price'] = sell_price
            else:
                order_payload['no_price'] = sell_price

            self.logger.info(f"Closing position: {order_payload}")
            result = self.api.create_order(order_payload)

            if result:
                # P&L = (sell price - entry price) * quantity, always same logic
                pnl_cents = (sell_price - entry) * quantity

                pnl_dollars = pnl_cents / 100

                # Record to db
                self.db.record_trade(
                    market_id, side=side, quantity=quantity, price=sell_price,
                    strategy=position.get('strategy', ''), order_result=f"CLOSE: {reason}",
                    pnl=pnl_dollars,
                )

                # Remove position from memory and database
                del self.current_positions[market_id]
                self.db.delete_position(market_id)

                # Notify
                pnl_emoji = "🟢" if pnl_dollars >= 0 else "🔴"
                self.notifier.send_message(
                    f"🔔 Position Closed\n\n"
                    f"{title}\n"
                    f"{side.upper()} — Entry: {entry}¢ → Exit: {sell_price}¢\n"
                    f"P&L: {pnl_emoji} ${pnl_dollars:.2f}\n"
                    f"Reason: {reason}"
                )
                self.logger.info(f"Closed {market_id} ({side}): {entry}¢→{sell_price}¢, P&L ${pnl_dollars:.2f}, {reason}")
            else:
                self.logger.error(f"Failed to close position {market_id}")

        except Exception as e:
            self.logger.error(f"Error closing position {market_id}: {e}")

    def get_portfolio_status(self):
        """
        Get portfolio status with basic risk metrics.
        """
        return self.risk_manager.get_portfolio_status()


