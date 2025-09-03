#!/bin/bash

# Semantic Scholar Citation Velocity Update Deployment
# Run this script on the AWS server to apply the citation velocity changes

echo "========================================="
echo "Semantic Scholar Citation Velocity Update"
echo "========================================="

# Navigate to project directory
cd /home/ubuntu/Ignacio-s-Fourier-Forecast || {
    echo "ERROR: Project directory not found!"
    exit 1
}

echo "Step 1: Backing up existing files..."
cp src/services/semantic_scholar_service.py src/services/semantic_scholar_service.py.backup
cp src/pipeline/content_aggregator.py src/pipeline/content_aggregator.py.backup
echo "✓ Backups created"

echo "Step 2: Stopping service..."
sudo systemctl stop fourier-forecast.service

echo "Step 3: Applying updates..."
echo "Please copy the updated files to the server, then press Enter to continue..."
read -p "Press Enter when files are copied..."

echo "Step 4: Restarting service..."
sudo systemctl start fourier-forecast.service

echo "Step 5: Verifying service status..."
sudo systemctl status fourier-forecast.service --no-pager | head -10

echo ""
echo "========================================="
echo "Update Complete!"
echo "========================================="
echo ""
echo "To monitor the service:"
echo "  sudo journalctl -u fourier-forecast.service -f"
echo ""
echo "To check for citation velocity logs:"
echo "  sudo journalctl -u fourier-forecast.service | grep 'citation velocity'"