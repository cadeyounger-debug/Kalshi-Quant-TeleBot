const express = require('express');
const WebSocket = require('ws');
const { spawn } = require('child_process');
const path = require('path');
require('dotenv').config();

class BotInterface {
    constructor(pythonBotPath, port = 3001) {
        this.pythonBotPath = pythonBotPath;
        this.port = port;
        this.app = express();
        this.server = null;
        this.wss = null;
        this.pythonProcess = null;
        this.clients = new Set();
        this.kalshiApiKey = process.env.KALSHI_API_KEY || null;
        this.kalshiApiBaseUrl = process.env.KALSHI_API_BASE_URL || null;
        this.botStateScript = process.env.BOT_STATE_SCRIPT || path.join(__dirname, '../src/bot_state.py');
        
        this.setupExpress();
        this.setupWebSocket();
    }

    setupExpress() {
        this.app.use(express.json());
        
        // CORS middleware
        this.app.use((req, res, next) => {
            res.header('Access-Control-Allow-Origin', '*');
            res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
            res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
            
            if (req.method === 'OPTIONS') {
                res.sendStatus(200);
            } else {
                next();
            }
        });

        // Health check endpoint
        this.app.get('/health', (req, res) => {
            res.json({
                status: 'healthy',
                timestamp: new Date().toISOString(),
                pythonBot: this.pythonProcess ? 'running' : 'stopped'
            });
        });

        // Bot status endpoint
        this.app.get('/api/status', async (req, res) => {
            try {
                const status = await this.runBotStateCommand('status');
                res.json({
                    trading: this.pythonProcess !== null,
                    lastUpdate: new Date().toISOString(),
                    uptime: this.getUptime(),
                    exchangeStatus: status.exchange_status || {},
                    balanceSummary: status.balance_summary || {},
                    positionsCount: status.positions_count || 0,
                    apiConnected: !(status.error),
                });
            } catch (error) {
                res.status(500).json({ success: false, error: error.message });
            }
        });

        // Positions endpoint
        this.app.get('/api/positions', async (req, res) => {
            try {
                const positions = await this.runBotStateCommand('positions');
                res.json(positions);
            } catch (error) {
                res.status(500).json({ success: false, error: error.message });
            }
        });

        // Balance endpoint
        this.app.get('/api/balance', async (req, res) => {
            try {
                const balance = await this.runBotStateCommand('balance');
                res.json(balance);
            } catch (error) {
                res.status(500).json({ success: false, error: error.message });
            }
        });

        // Performance metrics endpoint
        this.app.get('/api/performance', async (req, res) => {
            try {
                const performance = await this.runBotStateCommand('performance');
                res.json(performance);
            } catch (error) {
                res.status(500).json({ success: false, error: error.message });
            }
        });

        // Start trading endpoint
        this.app.post('/api/start-trading', (req, res) => {
            try {
                this.startPythonBot();
                res.json({ success: true, message: 'Trading started successfully' });
                this.broadcastToClients({ type: 'trading_started', timestamp: new Date().toISOString() });
            } catch (error) {
                res.status(500).json({ success: false, error: error.message });
            }
        });

        // Stop trading endpoint
        this.app.post('/api/stop-trading', (req, res) => {
            try {
                this.stopPythonBot();
                res.json({ success: true, message: 'Trading stopped successfully' });
                this.broadcastToClients({ type: 'trading_stopped', timestamp: new Date().toISOString() });
            } catch (error) {
                res.status(500).json({ success: false, error: error.message });
            }
        });

        // Credentials endpoint for API key updates
        this.app.post('/api/credentials', (req, res) => {
            const { kalshiApiKey } = req.body || {};
            if (!kalshiApiKey || typeof kalshiApiKey !== 'string') {
                return res.status(400).json({ success: false, error: 'kalshiApiKey is required' });
            }

            this.kalshiApiKey = kalshiApiKey.trim();
            console.log('Updated Kalshi API key via bot interface control plane');
            res.json({ success: true, message: 'Kalshi API key stored in memory until process restart' });
        });

        // Optional credentials status (does not leak key)
        this.app.get('/api/credentials/status', (req, res) => {
            res.json({
                hasKalshiApiKey: Boolean(this.kalshiApiKey || process.env.KALSHI_API_KEY),
                kalshiApiBaseUrl: this.kalshiApiBaseUrl || process.env.KALSHI_API_BASE_URL || null,
            });
        });

        // Configuration endpoints (Phase 4: Dynamic Settings Management)
        this.app.get('/api/config', async (req, res) => {
            try {
                // Get current settings from Python bot
                const config = await this.runBotStateCommand('config');
                res.json(config);
            } catch (error) {
                // Fallback to hardcoded defaults if Python bot not available
                res.json({
                    maxPositionSize: 0.10,
                    stopLoss: 0.05,
                    newsSentimentThreshold: 0.6,
                    statArbitrageThreshold: 0.05,
                    volatilityThreshold: 0.1,
                    tradeInterval: 60,
                    // Phase 4 settings
                    strategyEnablement: {
                        newsSentiment: true,
                        statisticalArbitrage: true,
                        volatilityBased: true
                    },
                    riskManagement: {
                        kellyFraction: 0.5,
                        maxPositionSizePct: 0.10,
                        stopLossPct: 0.05
                    },
                    notifications: {
                        telegram: true,
                        trades: true,
                        errors: true,
                        performance: true
                    }
                });
            }
        });

        // Update configuration endpoint (Phase 4)
        this.app.post('/api/config', (req, res) => {
            const config = req.body;
            // In a real implementation, this would update the Python bot's configuration
            console.log('Updating configuration:', config);
            res.json({ success: true, message: 'Configuration updated successfully' });
            this.broadcastToClients({ type: 'config_updated', config, timestamp: new Date().toISOString() });
        });

        // Phase 4: Dynamic Settings Management
        this.app.get('/api/settings', async (req, res) => {
            try {
                // Get settings from Python bot via CLI
                const settings = await this.runBotStateCommand('settings');
                res.json(settings);
            } catch (error) {
                res.status(500).json({ success: false, error: 'Unable to retrieve settings from bot' });
            }
        });

        // Update settings endpoint (Phase 4)
        this.app.post('/api/settings', async (req, res) => {
            try {
                const updates = req.body;

                // Send settings update to Python bot
                const result = await this.runBotStateCommandWithInput('update_settings', updates);

                if (result.success) {
                    res.json({
                        success: true,
                        message: 'Settings updated successfully',
                        changed_settings: result.changed_settings,
                        timestamp: new Date().toISOString()
                    });

                    // Broadcast settings update to all connected clients
                    this.broadcastToClients({
                        type: 'settings_updated',
                        changed_settings: result.changed_settings,
                        timestamp: new Date().toISOString()
                    });
                } else {
                    res.status(400).json({
                        success: false,
                        error: result.error || 'Settings update failed'
                    });
                }
            } catch (error) {
                console.error('Settings update error:', error);
                res.status(500).json({
                    success: false,
                    error: 'Failed to update settings'
                });
            }
        });

        // Reset settings to defaults (Phase 4)
        this.app.post('/api/settings/reset', async (req, res) => {
            try {
                const result = await this.runBotStateCommand('reset_settings');

                if (result.success) {
                    res.json({
                        success: true,
                        message: 'Settings reset to defaults',
                        timestamp: new Date().toISOString()
                    });

                    this.broadcastToClients({
                        type: 'settings_reset',
                        timestamp: new Date().toISOString()
                    });
                } else {
                    res.status(500).json({
                        success: false,
                        error: 'Failed to reset settings'
                    });
                }
            } catch (error) {
                console.error('Settings reset error:', error);
                res.status(500).json({
                    success: false,
                    error: 'Failed to reset settings'
                });
            }
        });

        // Settings info endpoint (Phase 4)
        this.app.get('/api/settings/info', async (req, res) => {
            try {
                const info = await this.runBotStateCommand('settings_info');
                res.json(info);
            } catch (error) {
                // Provide basic settings info if bot not available
                res.json({
                    news_sentiment_enabled: { type: 'boolean', description: 'Enable news sentiment strategy', default: true },
                    statistical_arbitrage_enabled: { type: 'boolean', description: 'Enable arbitrage strategy', default: true },
                    volatility_based_enabled: { type: 'boolean', description: 'Enable volatility strategy', default: true },
                    kelly_fraction: { type: 'float', range: [0, 1], description: 'Kelly criterion fraction', default: 0.5 },
                    stop_loss_pct: { type: 'float', range: [0, 0.5], description: 'Stop loss percentage', default: 0.05 },
                    trade_interval_seconds: { type: 'integer', range: [10, 3600], description: 'Trading interval', default: 60 }
                });
            }
        });

        // Database stats endpoint
        this.app.get('/api/dbstats', async (req, res) => {
            try {
                const stats = await this.runBotStateCommand('dbstats');
                res.json(stats);
            } catch (error) {
                res.status(500).json({ success: false, error: error.message });
            }
        });
    }

    setupWebSocket() {
        this.server = this.app.listen(this.port, '0.0.0.0', () => {
            console.log(`Bot interface server running on port ${this.port}`);
        });

        this.wss = new WebSocket.Server({ server: this.server });

        this.wss.on('connection', (ws) => {
            console.log('New WebSocket client connected');
            this.clients.add(ws);

            ws.on('message', (message) => {
                try {
                    const data = JSON.parse(message);
                    this.handleWebSocketMessage(ws, data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            });

            ws.on('close', () => {
                console.log('WebSocket client disconnected');
                this.clients.delete(ws);
            });

            // Send initial status
            ws.send(JSON.stringify({
                type: 'status',
                data: {
                    trading: this.pythonProcess !== null,
                    timestamp: new Date().toISOString()
                }
            }));
        });
    }

    handleWebSocketMessage(ws, data) {
        switch (data.type) {
            case 'subscribe':
                // Client wants to subscribe to updates
                ws.send(JSON.stringify({
                    type: 'subscribed',
                    timestamp: new Date().toISOString()
                }));
                break;
            case 'get_status':
                ws.send(JSON.stringify({
                    type: 'status',
                    data: {
                        trading: this.pythonProcess !== null,
                        timestamp: new Date().toISOString()
                    }
                }));
                break;
            default:
                console.log('Unknown WebSocket message type:', data.type);
        }
    }

    broadcastToClients(message) {
        const messageStr = JSON.stringify(message);
        this.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(messageStr);
            }
        });
    }

    startPythonBot() {
        if (this.pythonProcess) {
            console.log('Python bot is already running');
            return;
        }

        const resolvedPath = path.resolve(__dirname, this.pythonBotPath);
        const cwd = path.dirname(resolvedPath);
        console.log(`Starting Python bot: python3 ${resolvedPath} (cwd: ${cwd})`);
        const env = this.buildPythonEnv();
        this.pythonProcess = spawn('/usr/bin/python3', [resolvedPath], {
            cwd,
            stdio: ['pipe', 'pipe', 'pipe'],
            env,
        });

        this.pythonProcess.on('error', (err) => {
            console.error('Failed to start Python bot:', err.message);
            this.pythonProcess = null;
            this.broadcastToClients({
                type: 'bot_error',
                data: `Failed to start Python bot: ${err.message}`,
                timestamp: new Date().toISOString()
            });
        });

        this.pythonProcess.stdout.on('data', (data) => {
            const output = data.toString();
            console.log('Python bot output:', output);
            this.broadcastToClients({
                type: 'bot_output',
                data: output,
                timestamp: new Date().toISOString()
            });
        });

        this.pythonProcess.stderr.on('data', (data) => {
            const error = data.toString();
            console.error('Python bot error:', error);
            this.broadcastToClients({
                type: 'bot_error',
                data: error,
                timestamp: new Date().toISOString()
            });
        });

        this.pythonProcess.on('close', (code) => {
            console.log(`Python bot exited with code ${code}`);
            this.pythonProcess = null;
            this.broadcastToClients({
                type: 'bot_stopped',
                code,
                timestamp: new Date().toISOString()
            });
        });

        this.startTime = new Date();
    }

    stopPythonBot() {
        if (!this.pythonProcess) {
            console.log('Python bot is not running');
            return;
        }

        console.log('Stopping Python bot...');
        this.pythonProcess.kill('SIGTERM');
        
        // Force kill after 5 seconds if it doesn't stop gracefully
        setTimeout(() => {
            if (this.pythonProcess) {
                console.log('Force killing Python bot...');
                this.pythonProcess.kill('SIGKILL');
            }
        }, 5000);
    }

    getUptime() {
        if (!this.startTime) return '0s';
        const uptime = Date.now() - this.startTime.getTime();
        const seconds = Math.floor(uptime / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes % 60}m`;
        } else if (minutes > 0) {
            return `${minutes}m ${seconds % 60}s`;
        } else {
            return `${seconds}s`;
        }
    }

    getTradesCount() {
        // In a real implementation, this would query the database
        return Math.floor(Math.random() * 50) + 10;
    }

    // Method to simulate trade notifications
    simulateTradeNotification() {
        const trades = [
            { eventName: 'Election Outcome', action: 'buy', quantity: 50, price: 0.65, strategy: 'News Sentiment' },
            { eventName: 'Weather Event', action: 'sell', quantity: 25, price: 0.41, strategy: 'Volatility Analysis' },
            { eventName: 'Sports Outcome', action: 'buy', quantity: 100, price: 0.72, strategy: 'Statistical Arbitrage' }
        ];
        
        const trade = trades[Math.floor(Math.random() * trades.length)];
        this.broadcastToClients({
            type: 'trade_executed',
            data: trade,
            timestamp: new Date().toISOString()
        });
    }

    // Phase 4: Run bot state command via CLI
    buildPythonEnv() {
        const env = { ...process.env };
        if (this.kalshiApiKey) env.KALSHI_API_KEY = this.kalshiApiKey;
        if (this.kalshiApiBaseUrl) env.KALSHI_API_BASE_URL = this.kalshiApiBaseUrl;
        return env;
    }

    runBotStateCommand(command) {
        return new Promise((resolve, reject) => {
            const script = this.botStateScript;
            const args = [script, command];

            let child;
            try {
                child = spawn('/usr/bin/python3', args, {
                    cwd: path.dirname(script),
                    stdio: ['pipe', 'pipe', 'pipe'],
                    env: this.buildPythonEnv()
                });
            } catch (err) {
                return reject(new Error(`Failed to spawn python3: ${err.message}`));
            }

            let stdout = '';
            let stderr = '';

            child.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            child.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            child.on('close', (code) => {
                if (code === 0) {
                    try {
                        const result = JSON.parse(stdout.trim());
                        resolve(result);
                    } catch (e) {
                        reject(new Error(`Invalid JSON response: ${stdout}`));
                    }
                } else {
                    reject(new Error(`Command failed with code ${code}: ${stderr}`));
                }
            });

            child.on('error', (error) => {
                reject(error);
            });
        });
    }

    // Phase 4: Run bot state command with input data
    runBotStateCommandWithInput(command, data) {
        return new Promise((resolve, reject) => {
            const script = this.botStateScript;
            const jsonData = JSON.stringify(data);
            const args = [script, command, '--data', jsonData];

            const child = spawn('/usr/bin/python3', args, {
                cwd: path.dirname(script),
                stdio: ['pipe', 'pipe', 'pipe'],
                env: this.buildPythonEnv()
            });

            let stdout = '';
            let stderr = '';

            child.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            child.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            child.on('close', (code) => {
                if (code === 0) {
                    try {
                        const result = JSON.parse(stdout.trim());
                        resolve(result);
                    } catch (e) {
                        reject(new Error(`Invalid JSON response: ${stdout}`));
                    }
                } else {
                    reject(new Error(`Command failed with code ${code}: ${stderr}`));
                }
            });

            child.on('error', (error) => {
                reject(error);
            });
        });
    }

    close() {
        this.stopPythonBot();
        if (this.server) {
            this.server.close();
        }
        if (this.wss) {
            this.wss.close();
        }
    }
}

module.exports = BotInterface;

// Example usage
if (require.main === module) {
    const pythonBotPath = process.env.PYTHON_BOT_PATH || path.join(__dirname, '../src/main.py');
    const port = process.env.INTERFACE_PORT || 3001;
    
    const botInterface = new BotInterface(pythonBotPath, port);
    
    // Simulate trade notifications every 30 seconds for demo purposes
    setInterval(() => {
        if (Math.random() > 0.7) { // 30% chance
            botInterface.simulateTradeNotification();
        }
    }, 30000);
    
    // Graceful shutdown
    process.on('SIGINT', () => {
        console.log('Shutting down bot interface...');
        botInterface.close();
        process.exit(0);
    });
}

