#!/bin/bash

# Fourier Forecast Emergency Fix Deployment Script
# This script applies all fixes for the catastrophic failures

echo "========================================="
echo "Fourier Forecast Fix Deployment"
echo "========================================="

# Navigate to project directory
cd /home/ubuntu/Ignacio-s-Fourier-Forecast || {
    echo "ERROR: Project directory not found!"
    exit 1
}

echo "Step 1: Pulling latest code..."
git pull origin main || {
    echo "WARNING: Git pull failed. Continuing with current code..."
}

echo "Step 2: Stopping service..."
sudo systemctl stop fourier-forecast.service

echo "Step 3: Clearing cache database..."
if [ -f cache.db ]; then
    rm -f cache.db
    echo "✓ Cache database cleared"
else
    echo "✓ No cache database found (already clear)"
fi

echo "Step 4: Installing daily restart timer..."
if [ -f scripts/fourier-forecast-restart.timer ]; then
    sudo cp scripts/fourier-forecast-restart.timer /etc/systemd/system/
    sudo cp scripts/fourier-forecast-restart.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable fourier-forecast-restart.timer
    sudo systemctl start fourier-forecast-restart.timer
    echo "✓ Daily restart timer installed and started"
else
    echo "WARNING: Timer files not found in scripts/ directory"
fi

echo "Step 5: Restarting main service..."
sudo systemctl start fourier-forecast.service

echo "Step 6: Verifying services..."
echo ""
echo "Main Service Status:"
sudo systemctl status fourier-forecast.service --no-pager | head -10

echo ""
echo "Restart Timer Status:"
sudo systemctl status fourier-forecast-restart.timer --no-pager 2>/dev/null || echo "Timer not installed"

echo ""
echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""
echo "Recent logs (checking for fix confirmation):"
sudo journalctl -u fourier-forecast.service -n 20 --no-pager | grep -E "(Cleared|cache|Cornell|Miami|Scripture)" || echo "No relevant logs yet"

echo ""
echo "To monitor the next run:"
echo "  sudo journalctl -u fourier-forecast.service -f"