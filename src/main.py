import time
import logging
import threading
from datetime import datetime, timedelta
from config import KALSHI_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, BANKROLL, TRADE_INTERVAL_SECONDS
from kalshi_api import KalshiAPI
from trader import Trader
from notifier import Notifier
from logger import Logger

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = Logger()
    return logger


def start_retrain_scheduler(notifier, logger):
    """Schedule daily retraining at 7:00 AM UTC."""
    def retrain_loop():
        while True:
            now = datetime.utcnow()
            # Next 7am UTC
            next_run = now.replace(hour=7, minute=0, second=0, microsecond=0)
            if now >= next_run:
                next_run += timedelta(days=1)
            wait_seconds = (next_run - now).total_seconds()
            logger.info(f"Next model retrain at {next_run.isoformat()}Z ({wait_seconds/3600:.1f}h from now)")
            time.sleep(wait_seconds)

            # Run retraining
            try:
                logger.info("Starting scheduled model retraining...")
                from retrain import retrain
                params = retrain()
                if params:
                    yday = params.get('yesterday_accuracy', {})
                    yday_str = (f"\nYesterday: {yday.get('win_rate', 0):.0%} win rate "
                                f"({yday.get('total_trades', 0)} trades), "
                                f"momentum {yday.get('momentum_accuracy', 0):.0%}"
                                if yday else "\nYesterday: no data")
                    notifier.send_message(
                        f"🧠 Model Retrained (v{params.get('version', '?')})\n\n"
                        f"Data points: {params.get('data_points', 0)}\n"
                        f"Entry range: {params.get('min_entry_price_cents')}¢-{params.get('max_entry_price_cents')}¢\n"
                        f"Stop loss: {params.get('stop_loss_pct', 0):.0%}\n"
                        f"Breakeven@{params.get('breakeven_trigger', 0.15):.0%} → "
                        f"Trail@{params.get('trail_trigger', 0.25):.0%} ({params.get('trail_pct', 0.20):.0%})\n"
                        f"Momentum weight: {params.get('momentum_weight', 1.0)}"
                        f"{yday_str}"
                    )
                    logger.info("Retraining complete — notified via Telegram")
            except Exception as e:
                logger.error(f"Retraining failed: {e}")
                notifier.send_error_notification(f"Retraining failed: {e}")

    thread = threading.Thread(target=retrain_loop, daemon=True)
    thread.start()
    return thread


def main():
    logger = setup_logging()
    logger.info("Starting Kalshi Advanced Trading Bot with Phase 3 features")

    try:
        api = KalshiAPI(KALSHI_API_KEY)
        notifier = Notifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        trader = Trader(api, notifier, logger, BANKROLL)

        # Start market data streaming
        trader.market_data_streamer.start_streaming()
        logger.info("Market data streaming started")

        # Start daily retraining scheduler (7am UTC)
        start_retrain_scheduler(notifier, logger)
        logger.info("Retrain scheduler started (daily at 7:00 AM UTC)")

        # Run initial retrain to ensure params file exists
        try:
            from retrain import retrain
            retrain()
        except Exception as e:
            logger.warning(f"Initial retrain skipped: {e}")

        # Start HMM observation pipeline
        from hmm_observations import ObservationPipeline
        hmm_pipeline = ObservationPipeline(trader.db)
        hmm_observation_counter = 0

        while True:
            logger.info("Running trading strategy with real-time market data")
            trader.run_trading_strategy()

            # Record HMM observations every ~60s (every 3rd cycle at 20s interval)
            hmm_observation_counter += 1
            if hmm_observation_counter >= 3:
                hmm_observation_counter = 0
                try:
                    hmm_pipeline.record_all_assets()
                except Exception as e:
                    logger.debug(f"HMM observation error: {e}")

            time.sleep(TRADE_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("Bot shutdown requested by user")
        notifier.send_message("⚠️ Trading bot shutting down (user requested)")
        trader.market_data_streamer.stop_streaming()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        notifier.send_error_notification(f"🚨 Bot crashed: {str(e)}")
        trader.market_data_streamer.stop_streaming()
        raise
    finally:
        trader.market_data_streamer.stop_streaming()
        logger.info("Market data streaming stopped")

if __name__ == "__main__":
    main()
