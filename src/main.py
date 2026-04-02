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
                    notifier.send_message(
                        f"🧠 Model Retrained (v{params.get('version', '?')})\n\n"
                        f"Data points: {params.get('data_points', 0)}\n"
                        f"Entry range: {params.get('min_entry_price_cents')}¢-{params.get('max_entry_price_cents')}¢\n"
                        f"Take profit: {params.get('take_profit_pct', 0):.0%}\n"
                        f"Stop loss: {params.get('stop_loss_pct', 0):.0%}"
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

        while True:
            logger.info("Running trading strategy with real-time market data")
            trader.run_trading_strategy()
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
