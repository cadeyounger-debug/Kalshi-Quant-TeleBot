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

class Trader:
    def __init__(self, api, notifier, logger, bankroll):
        self.api = api
        self.notifier = notifier
        self.logger = logger
        self.bankroll = bankroll
        self.current_positions = {}
        self.news_analyzer = NewsSentimentAnalyzer()
        self.arbitrage_analyzer = StatisticalArbitrageAnalyzer(min_history_points=20)
        self.volatility_analyzer = VolatilityAnalyzer(min_history_points=20)
        self.risk_manager = RiskManager(bankroll)

        # Phase 3: Enhanced market data — stream crypto markets specifically
        crypto_events = self._build_crypto_event_tickers()
        self.market_data_streamer = MarketDataStreamer(
            api, update_interval=60, event_tickers=crypto_events
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
                        current_price = market.get('yes_ask') or market.get('last_price') or market.get('current_price')

                        if event_id and current_price:
                            action = 'buy' if sentiment_decision['direction'] == 'long' else 'sell'

                            # Apply dynamic risk management
                            position_size_fraction = self.risk_manager.calculate_position_size_kelly(sentiment_decision['confidence'])
                            position_value = self.risk_manager.current_bankroll * position_size_fraction
                            quantity = max(1, int(position_value / current_price))

                            trade_decision = {
                                'event_id': event_id,
                                'action': action,
                                'quantity': quantity,
                                'price': current_price,
                                'strategy': 'news_sentiment',
                                'sentiment_score': sentiment_decision['sentiment_score'],
                                'confidence': sentiment_decision['confidence']
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
                        current_price = market.get('yes_ask') or market.get('last_price') or market.get('current_price')

                        if event_id and current_price and volatility_decision.get('direction'):
                            action = 'buy' if volatility_decision['direction'] == 'long' else 'sell'

                            # Apply dynamic risk management
                            position_size_fraction = self.risk_manager.calculate_position_size_kelly(volatility_decision['confidence'])
                            position_value = self.risk_manager.current_bankroll * position_size_fraction
                            quantity = max(1, int(position_value / current_price))

                            trade_decision = {
                                'event_id': event_id,
                                'action': action,
                                'quantity': quantity,
                                'price': current_price,
                                'strategy': 'volatility_based',
                                'volatility_regime': volatility_decision.get('volatility_regime'),
                                'confidence': volatility_decision['confidence'],
                                'signal_type': volatility_decision.get('signal_type')
                            }

                            self.logger.info(f"Volatility trade decision: {action} {event_id} "
                                           f"(regime: {volatility_decision.get('volatility_regime')})")

            except Exception as e:
                self.logger.error(f"Error in volatility analysis: {e}")

        return trade_decision

    def _build_crypto_event_tickers(self):
        """Build list of BTC/ETH/SOL event tickers to watch.

        Covers year-end price, monthly min/max, daily, and hourly events.
        """
        from datetime import datetime, timedelta

        tickers = [
            # Year-end price events (highest liquidity)
            "KXBTCY-27JAN0100",
            "KXETHY-27JAN0100",
            "KXSOLD26-27JAN0100",
            "KXSOL26500-27JAN0100",
            # Cross-crypto / thematic
            "KXBTCVSGOLD-26",
            "KXBTCRESERVE-27",
        ]

        now = datetime.now()

        # Monthly min/max for current and next month
        for month_offset in range(0, 2):
            if month_offset == 0:
                next_m = now.month + 1 if now.month < 12 else 1
                next_y = now.year if now.month < 12 else now.year + 1
                end = datetime(next_y, next_m, 1) - timedelta(days=1)
            else:
                next_m = now.month + 2 if now.month < 11 else (now.month + 2 - 12)
                next_y = now.year if now.month < 11 else now.year + 1
                end = datetime(next_y, next_m, 1) - timedelta(days=1)

            dt = end.strftime("%y%b%d").upper()
            for coin, code in [("BTC", "BTC"), ("ETH", "ETH"), ("SOL", "SOL")]:
                tickers.append(f"KX{coin}MAXMON-{code}-{dt}")
                tickers.append(f"KX{coin}MINMON-{code}-{dt}")

        # Daily price events for next 7 days
        for days_ahead in range(0, 7):
            d = now + timedelta(days=days_ahead)
            ds = d.strftime("%y%b%d").upper()
            for coin in ["KXBTC", "KXETH", "KXSOL"]:
                tickers.append(f"{coin}-{ds}0100")

        # Hourly (1H) events for next 12 hours
        for hours_ahead in range(0, 12):
            t = now + timedelta(hours=hours_ahead)
            ds = t.strftime("%y%b%d").upper()
            h = t.hour
            for coin in ["KXBTC1H", "KXETH1H", "KXSOL1H"]:
                tickers.append(f"{coin}-{ds}{h:02d}00")

        # 15-minute events for next 2 hours
        for mins_ahead in range(0, 120, 15):
            t = now + timedelta(minutes=mins_ahead)
            ds = t.strftime("%y%b%d").upper()
            hm = t.strftime("%H%M")
            # Round to nearest 15
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

        if self._market_cache and self._cache_cycle % 5 != 1:
            self.logger.info(f"Using cached crypto markets ({len(self._market_cache)} markets)")
            return self._market_cache

        event_tickers = self._build_crypto_event_tickers()
        all_markets = []
        for event_ticker in event_tickers:
            resp = self.api.get_markets(params={"event_ticker": event_ticker, "limit": 200})
            if resp and resp.get("markets"):
                for m in resp["markets"]:
                    if m.get("status") == "active" and m.get("yes_bid_dollars", "0.0000") != "0.0000":
                        all_markets.append(m)

        self._market_cache = all_markets
        self.logger.info(f"Fetched {len(all_markets)} active BTC/ETH/SOL markets with liquidity")
        return all_markets

    def run_trading_strategy(self):
        """Main trading loop iteration: fetch all markets, analyse, execute."""
        try:
            markets = self._fetch_all_markets()
            if not markets:
                self.logger.info("No markets returned from API")
                return

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
                "current_price": m.get("yes_ask") or m.get("last_price") or 0,
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
            if len(history) < 20:
                continue
            vol_analysis = self.volatility_analyzer.analyze_market_volatility({
                "id": mid,
                "title": m.get("title", ""),
                "current_price": m.get("yes_ask") or m.get("last_price") or 0,
                "price_history": history,
            })
            decision = self.volatility_analyzer.should_trade_based_on_volatility(vol_analysis)
            if decision.get("should_trade"):
                if best is None or decision["confidence"] > best["confidence"]:
                    best = decision
                    best["market"] = m
        return best

    def _pick_best_market(self, markets, direction):
        """Pick the market with the most upside for the given direction."""
        scored = []
        for m in markets:
            price = m.get('yes_ask') or m.get('last_price') or m.get('current_price')
            if not price or price <= 0:
                continue
            # For a long bet we want cheap yes contracts; for short, expensive ones
            if direction == 'long':
                score = 1.0 - (price / 100 if price > 1 else price)
            else:
                score = price / 100 if price > 1 else price
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

        # Kalshi prices are in cents (1-99); normalise if needed
        price_cents = int(price) if price > 1 else int(price * 100)

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

            # Build Kalshi order payload
            side = 'yes' if action.lower() == 'buy' else 'no'
            order_payload = {
                'ticker': event_id,
                'action': 'buy',
                'side': side,
                'count': quantity,
                'type': 'limit',
                'yes_price': price_cents if side == 'yes' else None,
                'no_price': price_cents if side == 'no' else None,
            }
            # Remove None values
            order_payload = {k: v for k, v in order_payload.items() if v is not None}

            result = self.api.create_order(order_payload)
            if result:
                self.logger.info(f"ORDER PLACED: {result}")
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

            # Store position locally for basic tracking
            self.current_positions[event_id] = {
                'quantity': quantity,
                'entry_price': price,
                'type': 'long' if action.lower() == 'buy' else 'short',
                'strategy': strategy,
                'stop_loss_price': self.risk_manager.calculate_stop_loss_price(
                    price, action.lower() == 'buy'
                ),
                'trade_id': trade_id
            }

            # Send notification
            self.notifier.send_trade_notification(
                f"{strategy.upper()}: {action.upper()} {quantity} units of {event_id} at ${price:.2f}"
            )

        except Exception as e:
            self.logger.error(f"Error executing {strategy} trade for {event_id}: {e}")
            self.notifier.send_error_notification(f"Trade execution error for {event_id}: {e}")

    def check_positions_for_risk_management(self, current_prices: Dict[str, float]):
        """
        Check all open positions for stop-loss triggers.
        """
        positions_to_close = []

        for market_id, position in self.current_positions.items():
            current_price = current_prices.get(market_id, position['entry_price'])

            # Check stop-loss
            if self.risk_manager.check_stop_loss_trigger(
                position['entry_price'], current_price, position['type'] == 'long'
            ):
                positions_to_close.append({
                    'market_id': market_id,
                    'exit_price': current_price,
                    'reason': 'stop_loss_triggered'
                })

        # Close positions that hit stop-loss
        for close_info in positions_to_close:
            self.close_position_simple(close_info['market_id'],
                                     close_info['exit_price'],
                                     close_info['reason'])

    def close_position_simple(self, market_id: str, exit_price: float, reason: str):
        """
        Close a position with simple P&L calculation.
        """
        if market_id not in self.current_positions:
            return

        position = self.current_positions[market_id]

        # Calculate P&L
        entry_price = position['entry_price']
        quantity = position['quantity']

        if position['type'] == 'long':
            pnl = (exit_price - entry_price) * quantity
        else:  # short
            pnl = (entry_price - exit_price) * quantity

        # Update bankroll
        self.risk_manager.current_bankroll += pnl

        # Record trade closure in performance analytics
        trade_id = position.get('trade_id')
        if trade_id:
            self.performance_analytics.close_trade(trade_id, exit_price, reason)

        # Remove from positions
        del self.current_positions[market_id]

        # Send notification
        self.notifier.send_trade_notification(
            f"RISK MANAGEMENT: Closed {market_id} at ${exit_price:.2f}, P&L: ${pnl:.2f} ({reason})"
        )

        self.logger.info(f"Closed position {market_id}: P&L ${pnl:.2f}, reason: {reason}")

    def get_portfolio_status(self):
        """
        Get portfolio status with basic risk metrics.
        """
        return self.risk_manager.get_portfolio_status()


