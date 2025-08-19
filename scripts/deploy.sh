#!/bin/bash

# Deployment script for Ignacio's Fourier Forecast
# Automates deployment, validation, and service configuration.

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ENVIRONMENT=${1:-staging}
PROJECT_DIR="/opt/fourier-forecast"
SERVICE_USER="fourier"
SERVICE_NAME="fourier-forecast"
PYTHON_VERSION_MAJOR=3
PYTHON_VERSION_MINOR=9

echo -e "${GREEN}Ignacio's Fourier Forecast Deployment${NC}"
echo -e "Environment: ${YELLOW}${ENVIRONMENT}${NC}"
echo "================================================"

command_exists() { command -v "$1" >/dev/null 2>&1; }

echo -e "\n${YELLOW}Pre-deployment checks...${NC}"
if ! command_exists python3; then echo -e "${RED}Python3 not found${NC}"; exit 1; fi
if ! python3 - <<EOF >/dev/null 2>&1
import sys
sys.exit(0 if (sys.version_info >= (${PYTHON_VERSION_MAJOR}, ${PYTHON_VERSION_MINOR})) else 1)
EOF
then echo -e "${RED}Python ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}+ required${NC}"; exit 1; fi
echo -e "${GREEN}✓ Python version OK${NC}"

ENV_FILE=".env.${ENVIRONMENT}"
if [ ! -f "$ENV_FILE" ]; then echo -e "${RED}Missing $ENV_FILE${NC}"; exit 1; fi
echo -e "${GREEN}✓ Environment file found${NC}"

echo -e "\n${YELLOW}Validating required env vars...${NC}"
required_vars=(OPENAI_API_KEY PERPLEXITY_API_KEY SMTP_HOST SMTP_PORT SMTP_USER SMTP_PASSWORD RECIPIENT_EMAIL)
set -a; source "$ENV_FILE"; set +a
for v in "${required_vars[@]}"; do
  if [ -z "${!v}" ]; then echo -e "${RED}Missing $v in $ENV_FILE${NC}"; exit 1; fi
  echo -e "${GREEN}✓ ${v}${NC}"
done

echo -e "\n${YELLOW}Setting up directories and user...${NC}"
sudo mkdir -p "$PROJECT_DIR" "$PROJECT_DIR/logs" "$PROJECT_DIR/data" "$PROJECT_DIR/deployment_reports"
if ! id "$SERVICE_USER" &>/dev/null; then sudo useradd -r -s /bin/bash -d "$PROJECT_DIR" "$SERVICE_USER"; fi

echo -e "\n${YELLOW}Deploying application files...${NC}"
sudo rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='venv' --exclude='.env*' --exclude='logs/*' --exclude='data/*' ./ "$PROJECT_DIR/"
sudo cp "$ENV_FILE" "$PROJECT_DIR/.env"
sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR"

echo -e "\n${YELLOW}Setting up Python environment...${NC}"
sudo -u "$SERVICE_USER" python3 -m venv "$PROJECT_DIR/venv"
sudo -u "$SERVICE_USER" "$PROJECT_DIR/venv/bin/pip" install --upgrade pip
sudo -u "$SERVICE_USER" "$PROJECT_DIR/venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

echo -e "\n${YELLOW}Running deployment validation...${NC}"
cd "$PROJECT_DIR"
set +e
sudo -u "$SERVICE_USER" "$PROJECT_DIR/venv/bin/python" src/deployment/validator.py --env "$ENVIRONMENT"
code=$?
set -e
if [ $code -ne 0 ]; then echo -e "${RED}Deployment validation failed${NC}"; exit 1; fi

echo -e "\n${YELLOW}Creating systemd service...${NC}"
sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" > /dev/null <<EOF
[Unit]
Description=Ignacio's Fourier Forecast Daily Newsletter Service
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
Group=$SERVICE_USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$PROJECT_DIR/venv/bin/python $PROJECT_DIR/src/main.py
Restart=on-failure
RestartSec=10s
StandardOutput=append:$PROJECT_DIR/logs/service.log
StandardError=append:$PROJECT_DIR/logs/service.error.log
LimitNOFILE=4096

[Install]
WantedBy=multi-user.target
EOF

echo -e "\n${YELLOW}Configuring logrotate...${NC}"
sudo tee "/etc/logrotate.d/${SERVICE_NAME}" > /dev/null <<EOF
$PROJECT_DIR/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $SERVICE_USER $SERVICE_USER
}
EOF

echo -e "\n${YELLOW}Enabling service...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}.service"
if systemctl is-active --quiet "$SERVICE_NAME"; then
  sudo systemctl restart "$SERVICE_NAME"
else
  sudo systemctl start "$SERVICE_NAME"
fi

sleep 3
if systemctl is-active --quiet "$SERVICE_NAME"; then
  echo -e "${GREEN}✓ Service is running${NC}"
  sudo systemctl status "$SERVICE_NAME" --no-pager | head -n 10
else
  echo -e "${RED}✗ Service failed to start${NC}"
  sudo journalctl -u "$SERVICE_NAME" -n 50 --no-pager
  exit 1
fi

echo -e "\n${GREEN}Deployment completed successfully!${NC}"


