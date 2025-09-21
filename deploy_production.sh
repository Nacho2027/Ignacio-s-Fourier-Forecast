#!/bin/bash

# Production deployment script for Fourier Forecast
# Deploys changes to production server
# 
# Usage: ./deploy_production.sh
# 
# Prerequisites:
# - SSH key configured for server access
# - Server details configured in environment variables or modify script below

set -e

# Configuration - modify these for your deployment
SERVER_USER="${DEPLOY_USER:-ubuntu}"
SERVER_HOST="${DEPLOY_HOST:-your-server.com}"
SSH_KEY="${SSH_KEY:-deploy-key.pem}"
PROJECT_DIR="${REMOTE_PROJECT_DIR:-/opt/fourier-forecast}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Deploying Fourier Forecast to Production${NC}"
echo "========================================"

# Check if SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}Error: SSH key $SSH_KEY not found${NC}"
    echo "Please ensure the SSH key is in the current directory or set SSH_KEY environment variable"
    exit 1
fi

echo -e "${YELLOW}Copying files to production server...${NC}"

# Sync files to production (excluding sensitive and cache files)
rsync -avz --exclude='.git' \
           --exclude='__pycache__' \
           --exclude='*.pyc' \
           --exclude='.env*' \
           --exclude='logs/' \
           --exclude='data/' \
           --exclude='cache.db' \
           --exclude='venv/' \
           --exclude='node_modules/' \
           --exclude='*.pem' \
           --exclude='*.key' \
           -e "ssh -i $SSH_KEY" \
           ./ "$SERVER_USER@$SERVER_HOST:$PROJECT_DIR/"

echo -e "${YELLOW}Restarting service on production server...${NC}"

# Restart the service on the remote server
ssh -i "$SSH_KEY" "$SERVER_USER@$SERVER_HOST" << EOF
    cd $PROJECT_DIR
    sudo systemctl restart fourier-forecast
    sleep 3
    
    # Check if service is running
    if sudo systemctl is-active --quiet fourier-forecast; then
        echo "✅ Service restarted successfully"
        sudo systemctl status fourier-forecast --no-pager -l | head -20
    else
        echo "❌ Service failed to start"
        sudo journalctl -u fourier-forecast -n 20 --no-pager
        exit 1
    fi
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Production deployment completed successfully!${NC}"
    echo -e "${YELLOW}Monitor logs with:${NC}"
    echo "ssh -i $SSH_KEY $SERVER_USER@$SERVER_HOST \"sudo journalctl -u fourier-forecast -f\""
else
    echo -e "${RED}❌ Production deployment failed!${NC}"
    exit 1
fi