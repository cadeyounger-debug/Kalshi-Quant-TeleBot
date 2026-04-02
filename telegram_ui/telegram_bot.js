require('dotenv').config();
const TelegramBot = require('node-telegram-bot-api');
const axios = require('axios');
const fs = require('fs');
const path = require('path');

class KalshiTelegramBot {
    constructor(token, pythonBotPath) {
        this.bot = new TelegramBot(token, { polling: true });
        this.pythonBotPath = pythonBotPath;
        this.authorizedUsers = new Set(); // Store authorized user IDs
        this.awaitingApiKeyChats = new Set();
        this.alertChatIds = new Set(); // Chat IDs that receive trade/crash alerts
        this.interfaceBaseUrl = process.env.BOT_INTERFACE_URL || 'http://localhost:3001';
        this.setupCommands();
        this.setupCallbacks();
        this.setupApiKeyCapture();
    }

    // Send alert to all registered chat IDs
    sendAlert(message) {
        for (const chatId of this.alertChatIds) {
            this.bot.sendMessage(chatId, message, { parse_mode: 'Markdown' }).catch(() => {});
        }
    }

    setupCommands() {
        // Start command
        this.bot.onText(/\/start/, (msg) => {
            const chatId = msg.chat.id;
            this.alertChatIds.add(chatId);
            // Save chat ID to file so Python notifier can use it
            const fs = require('fs');
            try {
                fs.mkdirSync('/app/data', { recursive: true });
            } catch(e) {}
            fs.writeFileSync('/app/data/chat_id.txt', String(chatId));
            console.log(`Registered chat ${chatId} for alerts`);
            const welcomeMessage = `
🤖 *Kalshi Trading Bot Control Panel*

Welcome to the advanced Kalshi Trading Bot! This bot provides sophisticated quantitative trading strategies for event-based markets.

*Available Commands:*
/status - Get current bot status
/positions - View current positions
/balance - Check account balance
/start_trading - Start automated trading
/stop_trading - Stop automated trading
/settings - Configure bot settings
/performance - View performance metrics
/help - Show this help message

*Quick Actions:*
Use the inline keyboard below for quick access to common functions.
            `;
            
            const options = {
                parse_mode: 'Markdown',
                reply_markup: {
                    inline_keyboard: [
                        [
                            { text: '📊 Status', callback_data: 'status' },
                            { text: '💰 Balance', callback_data: 'balance' }
                        ],
                        [
                            { text: '📈 Positions', callback_data: 'positions' },
                            { text: '📊 Performance', callback_data: 'performance' }
                        ],
                        [
                            { text: '▶️ Start Trading', callback_data: 'start_trading' },
                            { text: '⏹️ Stop Trading', callback_data: 'stop_trading' }
                        ],
                        [
                            { text: '⚙️ Settings', callback_data: 'settings' },
                            { text: '❓ Help', callback_data: 'help' }
                        ]
                    ]
                }
            };
            
            this.bot.sendMessage(chatId, welcomeMessage, options);
        });

        // Status command
        this.bot.onText(/\/status/, (msg) => {
            this.alertChatIds.add(msg.chat.id);
            this.handleStatusCommand(msg.chat.id);
        });

        // Positions command
        this.bot.onText(/\/positions/, (msg) => {
            this.handlePositionsCommand(msg.chat.id);
        });

        // Balance command
        this.bot.onText(/\/balance/, (msg) => {
            this.handleBalanceCommand(msg.chat.id);
        });

        // Database stats command
        this.bot.onText(/\/dbstats/, (msg) => {
            this.handleDbStatsCommand(msg.chat.id);
        });

        // Start trading command
        this.bot.onText(/\/start_trading/, (msg) => {
            this.handleStartTradingCommand(msg.chat.id);
        });

        // Stop trading command
        this.bot.onText(/\/stop_trading/, (msg) => {
            this.handleStopTradingCommand(msg.chat.id);
        });

        // Settings command
        this.bot.onText(/\/settings/, (msg) => {
            this.handleSettingsCommand(msg.chat.id);
        });

        // Set command for individual settings
        this.bot.onText(/\/set\s+(.+)\s+(.+)/, (msg, match) => {
            const setting = match[1].trim();
            const value = match[2].trim();
            this.handleSetCommand(msg.chat.id, setting, value);
        });

        // Settings info command
        this.bot.onText(/\/settings_info/, (msg) => {
            this.handleSettingsInfoCommand(msg.chat.id);
        });

        // Confirm settings reset command
        this.bot.onText(/\/confirm_reset/, (msg) => {
            this.handleConfirmResetCommand(msg.chat.id);
        });

        // Set Kalshi API key command
        this.bot.onText(/\/set_api_key/, (msg) => {
            const chatId = msg.chat.id;
            this.awaitingApiKeyChats.add(chatId);
            this.bot.sendMessage(
                chatId,
                '🔐 Please reply with your Kalshi API key. It will be stored only in memory on this device until you restart the interface. Send /cancel if you changed your mind.'
            );
        });

        // Cancel pending input
        this.bot.onText(/\/cancel/, (msg) => {
            const chatId = msg.chat.id;
            if (this.awaitingApiKeyChats.has(chatId)) {
                this.awaitingApiKeyChats.delete(chatId);
                this.bot.sendMessage(chatId, '❎ API key entry cancelled.');
            } else {
                this.bot.sendMessage(chatId, 'No pending actions to cancel.');
            }
        });

        // Performance command
        this.bot.onText(/\/performance/, (msg) => {
            this.handlePerformanceCommand(msg.chat.id);
        });

        // Help command
        this.bot.onText(/\/help/, (msg) => {
            this.handleHelpCommand(msg.chat.id);
        });
    }

    setupCallbacks() {
        this.bot.on('callback_query', (callbackQuery) => {
            const action = callbackQuery.data;
            const msg = callbackQuery.message;
            const chatId = msg.chat.id;

            switch (action) {
                case 'status':
                    this.handleStatusCommand(chatId);
                    break;
                case 'balance':
                    this.handleBalanceCommand(chatId);
                    break;
                case 'positions':
                    this.handlePositionsCommand(chatId);
                    break;
                case 'performance':
                    this.handlePerformanceCommand(chatId);
                    break;
                case 'start_trading':
                    this.handleStartTradingCommand(chatId);
                    break;
                case 'stop_trading':
                    this.handleStopTradingCommand(chatId);
                    break;
                case 'settings':
                    this.handleSettingsCommand(chatId);
                    break;
                case 'settings_info':
                    this.handleSettingsInfoCommand(chatId);
                    break;
                case 'set_kelly':
                    this.handleSetKellyPrompt(chatId);
                    break;
                case 'set_stop_loss':
                    this.handleSetStopLossPrompt(chatId);
                    break;
                case 'strategy_control':
                    this.handleStrategyControl(chatId);
                    break;
                case 'notification_settings':
                    this.handleNotificationSettings(chatId);
                    break;
                case 'reset_settings':
                    this.handleResetSettings(chatId);
                    break;
                case 'help':
                    this.handleHelpCommand(chatId);
                    break;
                case 'set_api_key':
                    this.awaitingApiKeyChats.add(chatId);
                    this.bot.sendMessage(
                        chatId,
                        '🔐 Send your Kalshi API key now. Use /cancel to abort.'
                    );
                    break;
                default:
                    this.bot.sendMessage(chatId, 'Unknown action.');
            }

            // Answer the callback query
            this.bot.answerCallbackQuery(callbackQuery.id);
        });
    }

    async handleStatusCommand(chatId) {
        try {
            const status = await this.getBotStatus();
            const strategies = status.active_strategies || status.activeStrategies || [];
            const statusMessage = `
🤖 *Bot Status*

*Trading Status:* ${status.trading ? '🟢 Active' : '🔴 Inactive'}
*Last Update:* ${status.lastUpdate || new Date().toLocaleString()}
*Uptime:* ${status.uptime || 'N/A'}
*API Connection:* ${status.apiConnected ? '🟢 Connected' : '🔴 Disconnected'}
*Active Strategies:* ${strategies.join(', ') || 'None'}
*Positions:* ${status.positionsCount || 0}
            `;

            this.bot.sendMessage(chatId, statusMessage, { parse_mode: 'Markdown' });
        } catch (error) {
            this.bot.sendMessage(chatId, `❌ Error fetching status: ${error.message}`);
        }
    }

    async handlePositionsCommand(chatId) {
        try {
            const positions = await this.getCurrentPositions();
            
            if (positions.length === 0) {
                this.bot.sendMessage(chatId, '📈 *Current Positions*\n\nNo open positions.', { parse_mode: 'Markdown' });
                return;
            }

            let positionsMessage = '📈 *Current Positions*\n\n';
            positions.forEach((position, index) => {
                positionsMessage += `${index + 1}. *${position.eventName}*\n`;
                positionsMessage += `   Position: ${position.quantity} units\n`;
                positionsMessage += `   Entry Price: $${position.entryPrice}\n`;
                positionsMessage += `   Current Price: $${position.currentPrice}\n`;
                positionsMessage += `   P&L: ${position.pnl >= 0 ? '🟢' : '🔴'} $${position.pnl.toFixed(2)}\n\n`;
            });

            this.bot.sendMessage(chatId, positionsMessage, { parse_mode: 'Markdown' });
        } catch (error) {
            this.bot.sendMessage(chatId, `❌ Error fetching positions: ${error.message}`);
        }
    }

    async handleBalanceCommand(chatId) {
        try {
            const response = await axios.get(`${this.interfaceBaseUrl}/api/balance`);
            const data = response.data;

            if (data.error) {
                this.bot.sendMessage(chatId, `❌ *Balance Error*\n\n${data.error}`, { parse_mode: 'Markdown' });
                return;
            }

            const balance = data.summary || {};
            const available = balance.available ?? 0;
            const equity = balance.total_equity ?? balance.totalEquity ?? 0;
            const unrealized = balance.unrealized_pnl ?? balance.unrealizedPnL ?? 0;
            const realized = balance.realized_pnl ?? balance.realizedPnL ?? 0;
            const balanceMessage = `
💰 *Account Balance*

*Available Balance:* $${Number(available).toFixed(2)}
*Total Equity:* $${Number(equity).toFixed(2)}
*Unrealized P&L:* ${unrealized >= 0 ? '🟢' : '🔴'} $${Number(unrealized).toFixed(2)}
*Realized P&L:* ${realized >= 0 ? '🟢' : '🔴'} $${Number(realized).toFixed(2)}
            `;

            this.bot.sendMessage(chatId, balanceMessage, { parse_mode: 'Markdown' });
        } catch (error) {
            this.bot.sendMessage(chatId, `❌ Error fetching balance: ${error.message}`);
        }
    }

    async handleDbStatsCommand(chatId) {
        try {
            const response = await axios.get(`${this.interfaceBaseUrl}/api/dbstats`);
            const s = response.data;
            const msg = `
🗄️ *Database Stats*

*Market Snapshots:* ${s.total_snapshots || 0}
  BTC: ${s.btc_snapshots || 0}
  ETH: ${s.eth_snapshots || 0}
  SOL: ${s.sol_snapshots || 0}

*Sentiment Records:* ${s.total_sentiment || 0}
*Trade Decisions:* ${s.total_decisions || 0}
*Trades Executed:* ${s.total_trades || 0}
            `;
            this.bot.sendMessage(chatId, msg, { parse_mode: 'Markdown' });
        } catch (error) {
            this.bot.sendMessage(chatId, `❌ Error fetching db stats: ${error.message}`);
        }
    }

    async handleStartTradingCommand(chatId) {
        try {
            this.alertChatIds.add(chatId);
            await this.startTrading();
            this.bot.sendMessage(chatId, '▶️ Trading started successfully!');
            this.startMonitoring();
        } catch (error) {
            this.bot.sendMessage(chatId, `❌ Error starting trading: ${error.message}`);
        }
    }

    startMonitoring() {
        // Connect to bot interface WebSocket for crash/trade events
        if (this._ws) return; // Already monitoring
        try {
            const wsUrl = this.interfaceBaseUrl.replace('http', 'ws');
            const WebSocket = require('ws');
            this._ws = new WebSocket(wsUrl);

            this._ws.on('message', (raw) => {
                try {
                    const msg = JSON.parse(raw);
                    if (msg.type === 'bot_stopped') {
                        this.sendAlert(`🚨 *ALERT: Trading bot crashed!*\nExit code: ${msg.code}\nTime: ${new Date().toLocaleString()}`);
                        this._ws = null;
                    }
                } catch (e) {}
            });

            this._ws.on('close', () => {
                this._ws = null;
                // Reconnect after 5 seconds
                setTimeout(() => this.startMonitoring(), 5000);
            });

            this._ws.on('error', () => {
                this._ws = null;
            });
        } catch (e) {
            console.error('Failed to start monitoring:', e.message);
        }
    }

    async handleStopTradingCommand(chatId) {
        try {
            await this.stopTrading();
            this.bot.sendMessage(chatId, '⏹️ Trading stopped successfully!');
        } catch (error) {
            this.bot.sendMessage(chatId, `❌ Error stopping trading: ${error.message}`);
        }
    }

    async handleSettingsCommand(chatId) {
        try {
            // Fetch current settings from the bot interface
            const settingsResponse = await axios.get(`${this.interfaceBaseUrl}/api/settings`);
            const settings = settingsResponse.data;

            // Fetch settings info for descriptions
            const infoResponse = await axios.get(`${this.interfaceBaseUrl}/api/settings/info`);
            const settingsInfo = infoResponse.data;

            let settingsMessage = '⚙️ *Bot Settings*\n\n';
            settingsMessage += '*Current Configuration:*\n\n';

            // Format key settings for display
            const keySettings = [
                'kelly_fraction', 'max_position_size_pct', 'stop_loss_pct',
                'news_sentiment_enabled', 'statistical_arbitrage_enabled', 'volatility_based_enabled',
                'telegram_notifications', 'trade_notifications'
            ];

            keySettings.forEach(key => {
                if (settings[key] !== undefined) {
                    const info = settingsInfo[key] || {};
                    const value = typeof settings[key] === 'boolean' ?
                        (settings[key] ? '✅ Enabled' : '❌ Disabled') :
                        settings[key];
                    const description = info.description || key;
                    settingsMessage += `• *${description}:* ${value}\n`;
                }
            });

            settingsMessage += '\n';
            settingsMessage += '*Available Settings Commands:*\n';
            settingsMessage += 'Use the format: `/set [setting] [value]`\n\n';
            settingsMessage += '*Examples:*\n';
            settingsMessage += '• `/set kelly_fraction 0.4` - Set Kelly fraction to 40%\n';
            settingsMessage += '• `/set stop_loss_pct 3` - Set stop loss to 3%\n';
            settingsMessage += '• `/set news_sentiment_enabled false` - Disable news sentiment strategy\n';
            settingsMessage += '• `/set telegram_notifications false` - Disable Telegram notifications\n\n';
            settingsMessage += '*Strategy Control:*\n';
            settingsMessage += '• `/set news_sentiment_enabled [true/false]`\n';
            settingsMessage += '• `/set statistical_arbitrage_enabled [true/false]`\n';
            settingsMessage += '• `/set volatility_based_enabled [true/false]`\n\n';
            settingsMessage += '*Risk Management:*\n';
            settingsMessage += '• `/set kelly_fraction [0.1-1.0]` - Conservative to aggressive\n';
            settingsMessage += '• `/set max_position_size_pct [1-50]` - Max position size %\n';
            settingsMessage += '• `/set stop_loss_pct [1-10]` - Stop loss percentage\n\n';
            settingsMessage += '*For all available settings, see:* `/settings_info`';

            const options = {
                parse_mode: 'Markdown',
                reply_markup: {
                    inline_keyboard: [
                        [
                            { text: '🔄 Refresh Settings', callback_data: 'settings' },
                            { text: '📋 Settings Info', callback_data: 'settings_info' }
                        ],
                        [
                            { text: '🔧 Set Kelly Fraction', callback_data: 'set_kelly' },
                            { text: '🛡️ Set Stop Loss', callback_data: 'set_stop_loss' }
                        ],
                        [
                            { text: '🎯 Strategy Control', callback_data: 'strategy_control' },
                            { text: '📢 Notification Settings', callback_data: 'notification_settings' }
                        ],
                        [
                            { text: '🔄 Reset to Defaults', callback_data: 'reset_settings' },
                            { text: '❓ Help', callback_data: 'help' }
                        ]
                    ]
                }
            };

            this.bot.sendMessage(chatId, settingsMessage, options);

        } catch (error) {
            console.error('Settings fetch error:', error);
            // Fallback to basic settings display
            const fallbackMessage = `
⚙️ *Bot Settings*

Unable to fetch current settings. The bot interface may not be running.

*Available Settings Commands:*
• /set [setting] [value] - Modify individual settings
• /settings_info - View all available settings

*Examples:*
• /set kelly_fraction 0.5
• /set stop_loss_pct 5
• /set news_sentiment_enabled true

Please ensure the bot interface server is running on ${this.interfaceBaseUrl}
            `;

            this.bot.sendMessage(chatId, fallbackMessage, { parse_mode: 'Markdown' });
        }
    }

    async handleSetCommand(chatId, setting, value) {
        try {
            // Parse the value appropriately
            let parsedValue = value;

            // Convert boolean strings
            if (value.toLowerCase() === 'true') {
                parsedValue = true;
            } else if (value.toLowerCase() === 'false') {
                parsedValue = false;
            }
            // Convert numbers
            else if (!isNaN(value) && !isNaN(parseFloat(value))) {
                parsedValue = parseFloat(value);
                if (parsedValue % 1 === 0) {
                    parsedValue = parseInt(value); // Convert to int if whole number
                }
            }

            // Update the setting
            const response = await axios.post(`${this.interfaceBaseUrl}/api/settings`, {
                [setting]: parsedValue
            });

            if (response.data.success) {
                this.bot.sendMessage(
                    chatId,
                    `✅ Setting updated successfully!\n\n*${setting}:* ${parsedValue}\n\nUse /settings to view all current settings.`,
                    { parse_mode: 'Markdown' }
                );
            } else {
                this.bot.sendMessage(chatId, `❌ Failed to update setting: ${response.data.error || 'Unknown error'}`);
            }

        } catch (error) {
            console.error('Set command error:', error);
            const message = error?.response?.data?.error || error.message;
            this.bot.sendMessage(chatId, `❌ Error updating setting: ${message}`);
        }
    }

    async handleSettingsInfoCommand(chatId) {
        try {
            const response = await axios.get(`${this.interfaceBaseUrl}/api/settings/info`);
            const settingsInfo = response.data;

            let infoMessage = '📋 *Available Settings*\n\n';
            infoMessage += 'All configurable bot parameters:\n\n';

            Object.entries(settingsInfo).forEach(([key, info]) => {
                infoMessage += `*${key}:*\n`;
                infoMessage += `  Type: ${info.type}\n`;
                infoMessage += `  Description: ${info.description}\n`;
                infoMessage += `  Default: ${info.default}\n\n`;
            });

            infoMessage += '*Usage:*\n';
            infoMessage += '• `/set [setting] [value]` to modify\n';
            infoMessage += '• `/settings` to view current values\n';
            infoMessage += '• Example: `/set kelly_fraction 0.5`';

            this.bot.sendMessage(chatId, infoMessage, { parse_mode: 'Markdown' });

        } catch (error) {
            console.error('Settings info error:', error);
            this.bot.sendMessage(chatId, '❌ Unable to fetch settings information. Please ensure the bot interface is running.');
        }
    }

    handleSetKellyPrompt(chatId) {
        const message = `
🔧 *Set Kelly Fraction*

The Kelly fraction determines position sizing aggressiveness:
• *0.1-0.3:* Conservative (recommended)
• *0.4-0.6:* Moderate
• *0.7-1.0:* Aggressive (high risk)

*Current:* Use /settings to view
*Default:* 0.5 (50% Kelly)

*Usage:* /set kelly_fraction [value]
*Example:* /set kelly_fraction 0.4
        `;

        this.bot.sendMessage(chatId, message, { parse_mode: 'Markdown' });
    }

    handleSetStopLossPrompt(chatId) {
        const message = `
🛡️ *Set Stop Loss Percentage*

Stop loss protects against excessive losses:
• *1-3%:* Tight (frequent exits)
• *3-7%:* Moderate (balanced)
• *7-10%:* Wide (fewer exits)

*Current:* Use /settings to view
*Default:* 5%

*Usage:* /set stop_loss_pct [value]
*Example:* /set stop_loss_pct 3
        `;

        this.bot.sendMessage(chatId, message, { parse_mode: 'Markdown' });
    }

    async handleStrategyControl(chatId) {
        try {
            const settingsResponse = await axios.get(`${this.interfaceBaseUrl}/api/settings`);
            const settings = settingsResponse.data;

            const message = `
🎯 *Strategy Control*

Enable or disable individual trading strategies:

*News Sentiment Strategy:*
${settings.news_sentiment_enabled ? '✅' : '❌'} *Enabled*
• Command: /set news_sentiment_enabled ${!settings.news_sentiment_enabled}

*Statistical Arbitrage Strategy:*
${settings.statistical_arbitrage_enabled ? '✅' : '❌'} *Enabled*
• Command: /set statistical_arbitrage_enabled ${!settings.statistical_arbitrage_enabled}

*Volatility-Based Strategy:*
${settings.volatility_based_enabled ? '✅' : '❌'} *Enabled*
• Command: /set volatility_based_enabled ${!settings.volatility_based_enabled}

*Quick Commands:*
• /set news_sentiment_enabled false  (Disable news strategy)
• /set statistical_arbitrage_enabled false  (Disable arbitrage)
• /set volatility_based_enabled false  (Disable volatility)
            `;

            this.bot.sendMessage(chatId, message, { parse_mode: 'Markdown' });

        } catch (error) {
            this.bot.sendMessage(chatId, '❌ Unable to fetch strategy settings.');
        }
    }

    async handleNotificationSettings(chatId) {
        try {
            const settingsResponse = await axios.get(`${this.interfaceBaseUrl}/api/settings`);
            const settings = settingsResponse.data;

            const message = `
📢 *Notification Settings*

Control when the bot sends notifications:

*Telegram Notifications:*
${settings.telegram_notifications ? '✅' : '❌'} *Enabled*
• Command: /set telegram_notifications ${!settings.telegram_notifications}

*Trade Notifications:*
${settings.trade_notifications ? '✅' : '❌'} *Enabled*
• Command: /set trade_notifications ${!settings.trade_notifications}

*Error Notifications:*
${settings.error_notifications ? '✅' : '❌'} *Enabled*
• Command: /set error_notifications ${!settings.error_notifications}

*Performance Alerts:*
${settings.performance_alerts ? '✅' : '❌'} *Enabled*
• Command: /set performance_alerts ${!settings.performance_alerts}

*Quick Commands:*
• /set telegram_notifications false  (Disable all notifications)
• /set trade_notifications false  (Disable trade alerts)
            `;

            this.bot.sendMessage(chatId, message, { parse_mode: 'Markdown' });

        } catch (error) {
            this.bot.sendMessage(chatId, '❌ Unable to fetch notification settings.');
        }
    }

    async handleResetSettings(chatId) {
        try {
            const confirmMessage = `
🔄 *Reset Settings to Defaults*

This will restore all settings to their default values:

• Kelly Fraction: 0.5
• Stop Loss: 5%
• All strategies: Enabled
• Notifications: Enabled
• Risk parameters: Default values

*Are you sure?* This cannot be undone.

Reply with: \`/confirm_reset\` to proceed
or \`/cancel\` to abort.
            `;

            this.bot.sendMessage(chatId, confirmMessage, { parse_mode: 'Markdown' });

        } catch (error) {
            this.bot.sendMessage(chatId, '❌ Error preparing settings reset.');
        }
    }

    async handleConfirmResetCommand(chatId) {
        try {
            const response = await axios.post(`${this.interfaceBaseUrl}/api/settings/reset`);

            if (response.data.success) {
                this.bot.sendMessage(chatId, '✅ Settings have been reset to defaults!\n\nUse /settings to view the updated configuration.');
            } else {
                this.bot.sendMessage(chatId, `❌ Failed to reset settings: ${response.data.error || 'Unknown error'}`);
            }

        } catch (error) {
            console.error('Settings reset error:', error);
            const message = error?.response?.data?.error || error.message;
            this.bot.sendMessage(chatId, `❌ Error resetting settings: ${message}`);
        }
    }

    async handlePerformanceCommand(chatId) {
        try {
            const performance = await this.getPerformanceMetrics();
            const performanceMessage = `
📊 *Performance Metrics*

*Total Return:* ${performance.totalReturn >= 0 ? '🟢' : '🔴'} ${performance.totalReturn.toFixed(2)}%
*Sharpe Ratio:* ${performance.sharpeRatio.toFixed(2)}
*Max Drawdown:* ${performance.maxDrawdown.toFixed(2)}%
*Win Rate:* ${performance.winRate.toFixed(1)}%
*Total Trades:* ${performance.totalTrades}
*Average Trade:* $${performance.avgTrade.toFixed(2)}
*Best Trade:* $${performance.bestTrade.toFixed(2)}
*Worst Trade:* $${performance.worstTrade.toFixed(2)}
            `;
            
            this.bot.sendMessage(chatId, performanceMessage, { parse_mode: 'Markdown' });
        } catch (error) {
            this.bot.sendMessage(chatId, `❌ Error fetching performance: ${error.message}`);
        }
    }

    handleHelpCommand(chatId) {
        const helpMessage = `
❓ *Help - Kalshi Trading Bot*

*Commands:*
/start - Initialize bot and show main menu
/status - Current bot status and health
/positions - View all open positions
/balance - Account balance and P&L
/start_trading - Begin automated trading
/stop_trading - Halt all trading activities
/settings - View and modify bot configuration
/set [setting] [value] - Modify individual settings
/settings_info - View all available settings
/confirm_reset - Confirm settings reset (use with caution)
/performance - Detailed performance metrics
/help - Show this help message

*Settings Examples:*
• /set kelly_fraction 0.4 - Set conservative position sizing
• /set stop_loss_pct 3 - Set tighter stop loss
• /set news_sentiment_enabled false - Disable news strategy
• /set telegram_notifications false - Disable notifications

*Features:*
• Advanced quantitative strategies
• Real-time market monitoring
• Dynamic settings management
• Risk management and position sizing
• News sentiment analysis
• Statistical arbitrage detection
• Volatility-based trading
• Comprehensive logging and alerts

*Support:*
For technical support or questions, contact the development team.
        `;
        
        this.bot.sendMessage(chatId, helpMessage, { parse_mode: 'Markdown' });
    }

    setupApiKeyCapture() {
        this.bot.on('message', async (msg) => {
            const chatId = msg.chat.id;
            const text = msg.text;

            if (!text || text.startsWith('/')) {
                return;
            }

            if (this.awaitingApiKeyChats.has(chatId)) {
                this.awaitingApiKeyChats.delete(chatId);
                await this.handleApiKeySubmission(chatId, text.trim());
            }
        });
    }

    async handleApiKeySubmission(chatId, apiKey) {
        if (!apiKey || apiKey.length < 10) {
            this.bot.sendMessage(chatId, '❌ That API key looks invalid. Please run /set_api_key again.');
            return;
        }

        try {
            await axios.post(`${this.interfaceBaseUrl}/api/credentials`, {
                kalshiApiKey: apiKey
            });
            this.bot.sendMessage(chatId, '✅ Kalshi API key saved in memory for this session.');
        } catch (error) {
            const message = error?.response?.data?.error || error.message;
            this.bot.sendMessage(chatId, `❌ Failed to update API key: ${message}`);
        }
    }

    // Helper methods to interface with the Python bot
    async getBotStatus() {
        const response = await axios.get(`${this.interfaceBaseUrl}/api/status`);
        return response.data;
    }

    async getCurrentPositions() {
        const response = await axios.get(`${this.interfaceBaseUrl}/api/positions`);
        return response.data.positions || [];
    }

    async getAccountBalance() {
        const response = await axios.get(`${this.interfaceBaseUrl}/api/balance`);
        return response.data.summary || {};
    }

    async getPerformanceMetrics() {
        const response = await axios.get(`${this.interfaceBaseUrl}/api/performance`);
        return response.data;
    }

    async startTrading() {
        await axios.post(`${this.interfaceBaseUrl}/api/start-trading`);
    }

    async stopTrading() {
        await axios.post(`${this.interfaceBaseUrl}/api/stop-trading`);
    }

    // Method to send notifications from the Python bot
    sendNotification(chatId, message, options = {}) {
        this.bot.sendMessage(chatId, message, options);
    }

    // Method to send trade notifications
    sendTradeNotification(chatId, tradeData) {
        const message = `
🔔 *Trade Executed*

*Event:* ${tradeData.eventName}
*Action:* ${tradeData.action.toUpperCase()}
*Quantity:* ${tradeData.quantity} units
*Price:* $${tradeData.price}
*Total Value:* $${(tradeData.quantity * tradeData.price).toFixed(2)}
*Strategy:* ${tradeData.strategy}
*Time:* ${new Date().toLocaleString()}
        `;
        
        this.bot.sendMessage(chatId, message, { parse_mode: 'Markdown' });
    }

    // Method to send error notifications
    sendErrorNotification(chatId, error) {
        const message = `
❌ *Error Alert*

*Time:* ${new Date().toLocaleString()}
*Error:* ${error}

The bot will attempt to recover automatically. If the issue persists, please check the logs or contact support.
        `;
        
        this.bot.sendMessage(chatId, message, { parse_mode: 'Markdown' });
    }
}

module.exports = KalshiTelegramBot;

// Example usage
if (require.main === module) {
    const token = process.env.TELEGRAM_BOT_TOKEN;
    const pythonBotPath = process.env.PYTHON_BOT_PATH || '../src/main.py';
    
    if (!token) {
        console.error('TELEGRAM_BOT_TOKEN environment variable is required');
        process.exit(1);
    }
    
    const bot = new KalshiTelegramBot(token, pythonBotPath);
    console.log('Kalshi Telegram Bot started successfully!');
}

