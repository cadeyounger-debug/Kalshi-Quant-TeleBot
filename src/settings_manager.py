#!/usr/bin/env python3
"""Dynamic settings management for Phase 4 - Kalshi trading bot."""

import json
import logging
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class BotSettings:
    """Dynamic bot configuration settings."""
    # Strategy enable/disable flags
    news_sentiment_enabled: bool = True
    statistical_arbitrage_enabled: bool = True
    volatility_based_enabled: bool = True

    # Risk parameters
    kelly_fraction: float = 0.5  # Half-Kelly for conservatism
    max_position_size_pct: float = 0.10  # Max 10% of bankroll
    stop_loss_pct: float = 0.05  # 5% stop loss

    # Strategy-specific thresholds (lowered to generate trade data)
    news_sentiment_threshold: float = 0.3
    stat_arbitrage_threshold: float = 0.03
    volatility_threshold: float = 0.05

    # Trading parameters
    trade_interval_seconds: int = 60
    max_concurrent_positions: int = 5

    # Market data parameters
    market_data_update_interval: int = 60
    volatility_calculation_window: int = 20

    # Notification preferences
    telegram_notifications: bool = True
    trade_notifications: bool = True
    error_notifications: bool = True
    performance_alerts: bool = True

    # Advanced settings
    risk_free_rate: float = 0.02  # For Sharpe ratio calculations
    max_daily_trades: int = 50
    max_daily_loss_pct: float = 0.05  # Stop trading if daily loss exceeds this

    # System settings
    debug_mode: bool = False
    log_level: str = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return asdict(self)

    def from_dict(self, data: Dict[str, Any]):
        """Update settings from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def validate(self) -> bool:
        """Validate settings are within acceptable ranges."""
        validations = [
            (0 <= self.kelly_fraction <= 1, "kelly_fraction must be between 0 and 1"),
            (0 < self.max_position_size_pct <= 1, "max_position_size_pct must be between 0 and 1"),
            (0 < self.stop_loss_pct <= 0.5, "stop_loss_pct must be between 0 and 0.5"),
            (-1 <= self.news_sentiment_threshold <= 1, "news_sentiment_threshold must be between -1 and 1"),
            (0 <= self.stat_arbitrage_threshold <= 1, "stat_arbitrage_threshold must be between 0 and 1"),
            (0 <= self.volatility_threshold <= 1, "volatility_threshold must be between 0 and 1"),
            (10 <= self.trade_interval_seconds <= 3600, "trade_interval_seconds must be between 10 and 3600"),
            (1 <= self.max_concurrent_positions <= 20, "max_concurrent_positions must be between 1 and 20"),
            (10 <= self.market_data_update_interval <= 3600, "market_data_update_interval must be between 10 and 3600"),
            (5 <= self.volatility_calculation_window <= 100, "volatility_calculation_window must be between 5 and 100"),
            (0 <= self.risk_free_rate <= 0.1, "risk_free_rate must be between 0 and 0.1"),
            (1 <= self.max_daily_trades <= 500, "max_daily_trades must be between 1 and 500"),
            (0 <= self.max_daily_loss_pct <= 1, "max_daily_loss_pct must be between 0 and 1"),
            (self.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], "log_level must be a valid logging level"),
        ]

        for is_valid, error_msg in validations:
            if not is_valid:
                logger.error(f"Settings validation failed: {error_msg}")
                return False

        return True

class SettingsManager:
    """Dynamic settings management system."""

    def __init__(self, settings_file: str = "bot_settings.json"):
        self.settings_file = settings_file
        self.settings = BotSettings()
        self.last_modified = datetime.now()
        self.change_listeners: list = []

        # Load existing settings if available
        self.load_settings()

    def add_change_listener(self, callback):
        """Add a callback to be notified when settings change."""
        self.change_listeners.append(callback)

    def remove_change_listener(self, callback):
        """Remove a change listener."""
        if callback in self.change_listeners:
            self.change_listeners.remove(callback)

    def _notify_listeners(self, changed_settings: Dict[str, Any]):
        """Notify all listeners of settings changes."""
        for listener in self.change_listeners:
            try:
                listener(changed_settings)
            except Exception as e:
                logger.error(f"Error notifying settings listener: {e}")

    def load_settings(self) -> bool:
        """Load settings from file."""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    data = json.load(f)
                    self.settings.from_dict(data)
                    logger.info(f"Loaded settings from {self.settings_file}")
                    return True
            else:
                logger.info(f"Settings file {self.settings_file} not found, using defaults")
                return False
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return False

    def save_settings(self) -> bool:
        """Save current settings to file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings.to_dict(), f, indent=2, default=str)
            logger.info(f"Saved settings to {self.settings_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False

    def update_settings(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update settings with validation.

        Args:
            updates: Dictionary of settings to update

        Returns:
            Dictionary with update results
        """
        old_settings = self.settings.to_dict()

        # Apply updates
        self.settings.from_dict(updates)

        # Validate new settings
        if not self.settings.validate():
            # Revert changes if validation fails
            self.settings.from_dict(old_settings)
            return {
                'success': False,
                'error': 'Settings validation failed',
                'current_settings': self.settings.to_dict()
            }

        # Save to file
        if not self.save_settings():
            return {
                'success': False,
                'error': 'Failed to save settings',
                'current_settings': self.settings.to_dict()
            }

        # Calculate what changed
        changed_settings = {}
        for key, new_value in updates.items():
            if key in old_settings and old_settings[key] != new_value:
                changed_settings[key] = {
                    'old_value': old_settings[key],
                    'new_value': new_value
                }

        # Notify listeners
        if changed_settings:
            self._notify_listeners(changed_settings)
            self.last_modified = datetime.now()

        return {
            'success': True,
            'changed_settings': changed_settings,
            'current_settings': self.settings.to_dict(),
            'timestamp': self.last_modified.isoformat()
        }

    def get_settings(self, keys: Optional[list] = None) -> Dict[str, Any]:
        """
        Get current settings.

        Args:
            keys: Optional list of specific setting keys to return

        Returns:
            Current settings (all or specified keys)
        """
        all_settings = self.settings.to_dict()

        if keys is None:
            return all_settings

        # Return only requested keys
        result = {}
        for key in keys:
            if key in all_settings:
                result[key] = all_settings[key]

        return result

    def reset_to_defaults(self) -> Dict[str, Any]:
        """Reset all settings to defaults."""
        old_settings = self.settings.to_dict()
        self.settings = BotSettings()

        if self.save_settings():
            changed_settings = {}
            for key, new_value in self.settings.to_dict().items():
                if key in old_settings and old_settings[key] != new_value:
                    changed_settings[key] = {
                        'old_value': old_settings[key],
                        'new_value': new_value
                    }

            if changed_settings:
                self._notify_listeners(changed_settings)
                self.last_modified = datetime.now()

            return {
                'success': True,
                'message': 'Settings reset to defaults',
                'changed_settings': changed_settings,
                'current_settings': self.settings.to_dict()
            }
        else:
            return {
                'success': False,
                'error': 'Failed to save default settings'
            }

    def get_setting_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available settings."""
        return {
            'news_sentiment_enabled': {
                'type': 'boolean',
                'description': 'Enable/disable news sentiment trading strategy',
                'default': True
            },
            'statistical_arbitrage_enabled': {
                'type': 'boolean',
                'description': 'Enable/disable statistical arbitrage strategy',
                'default': True
            },
            'volatility_based_enabled': {
                'type': 'boolean',
                'description': 'Enable/disable volatility-based trading strategy',
                'default': True
            },
            'kelly_fraction': {
                'type': 'float',
                'range': [0, 1],
                'description': 'Kelly criterion fraction (0.5 = half-Kelly conservative)',
                'default': 0.5
            },
            'max_position_size_pct': {
                'type': 'float',
                'range': [0, 1],
                'description': 'Maximum position size as percentage of bankroll',
                'default': 0.10
            },
            'stop_loss_pct': {
                'type': 'float',
                'range': [0, 0.5],
                'description': 'Stop-loss percentage for position closure',
                'default': 0.05
            },
            'news_sentiment_threshold': {
                'type': 'float',
                'range': [-1, 1],
                'description': 'Sentiment polarity threshold for trade signals',
                'default': 0.6
            },
            'stat_arbitrage_threshold': {
                'type': 'float',
                'range': [0, 1],
                'description': 'Z-score threshold for arbitrage opportunities',
                'default': 0.05
            },
            'volatility_threshold': {
                'type': 'float',
                'range': [0, 1],
                'description': 'Volatility threshold for trading signals',
                'default': 0.1
            },
            'trade_interval_seconds': {
                'type': 'integer',
                'range': [10, 3600],
                'description': 'Seconds between trading strategy evaluations',
                'default': 60
            },
            'max_concurrent_positions': {
                'type': 'integer',
                'range': [1, 20],
                'description': 'Maximum number of concurrent open positions',
                'default': 5
            },
            'market_data_update_interval': {
                'type': 'integer',
                'range': [10, 3600],
                'description': 'Seconds between market data updates',
                'default': 60
            },
            'volatility_calculation_window': {
                'type': 'integer',
                'range': [5, 100],
                'description': 'Number of periods for volatility calculations',
                'default': 20
            },
            'telegram_notifications': {
                'type': 'boolean',
                'description': 'Enable Telegram notifications',
                'default': True
            },
            'trade_notifications': {
                'type': 'boolean',
                'description': 'Send notifications for executed trades',
                'default': True
            },
            'error_notifications': {
                'type': 'boolean',
                'description': 'Send notifications for errors',
                'default': True
            },
            'performance_alerts': {
                'type': 'boolean',
                'description': 'Send performance milestone alerts',
                'default': True
            },
            'risk_free_rate': {
                'type': 'float',
                'range': [0, 0.1],
                'description': 'Risk-free rate for Sharpe ratio calculations',
                'default': 0.02
            },
            'max_daily_trades': {
                'type': 'integer',
                'range': [1, 500],
                'description': 'Maximum trades allowed per day',
                'default': 50
            },
            'max_daily_loss_pct': {
                'type': 'float',
                'range': [0, 1],
                'description': 'Stop trading if daily loss exceeds this percentage',
                'default': 0.05
            },
            'debug_mode': {
                'type': 'boolean',
                'description': 'Enable debug logging and additional output',
                'default': False
            },
            'log_level': {
                'type': 'string',
                'options': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                'description': 'Logging level for the application',
                'default': 'INFO'
            }
        }
