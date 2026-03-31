#!/bin/bash
# Start both the bot interface (Express API) and the Telegram bot
cd telegram_ui

# Start bot interface in background
node bot_interface.js &
BOT_INTERFACE_PID=$!

# Wait for interface to be ready
sleep 2

# Start Telegram bot in foreground (keeps container alive)
node telegram_bot.js &
TELEGRAM_PID=$!

# Wait for either to exit
wait -n $BOT_INTERFACE_PID $TELEGRAM_PID

# If either exits, kill the other and exit
kill $BOT_INTERFACE_PID $TELEGRAM_PID 2>/dev/null
exit 1
