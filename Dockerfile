FROM node:22-slim

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip3 install --break-system-packages -r requirements.txt

# Install Node deps
COPY telegram_ui/package.json telegram_ui/package-lock.json ./telegram_ui/
RUN cd telegram_ui && npm install

# Copy app code
COPY . .

# Start both services
CMD ["bash", "start.sh"]
