#!/bin/bash

# Ditto TalkingHead Service Installation Script
# Usage: ./install_service.sh [port] [gpu_id]

set -e

# Default values
DEFAULT_PORT=8010
DEFAULT_GPU=0

# Get parameters
PORT=${1:-$DEFAULT_PORT}
GPU_ID=${2:-$DEFAULT_GPU}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Ditto TalkingHead Service Installation ===${NC}"
echo -e "Port: ${GREEN}$PORT${NC}"
echo -e "GPU ID: ${GREEN}$GPU_ID${NC}"
echo ""

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}This script should not be run as root${NC}"
   exit 1
fi

# Get current user and working directory
CURRENT_USER=$(whoami)
CURRENT_DIR=$(pwd)

echo -e "${YELLOW}Current user: $CURRENT_USER${NC}"
echo -e "${YELLOW}Working directory: $CURRENT_DIR${NC}"

# Check if conda environment exists
if ! conda info --envs | grep -q "ditto"; then
    echo -e "${RED}Error: 'ditto' conda environment not found${NC}"
    echo "Please create the conda environment first"
    exit 1
fi

# Get conda path
CONDA_BASE=$(conda info --base)
CONDA_ENV_PATH="$CONDA_BASE/envs/ditto"

if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo -e "${RED}Error: Conda environment path not found: $CONDA_ENV_PATH${NC}"
    exit 1
fi

echo -e "${YELLOW}Conda environment path: $CONDA_ENV_PATH${NC}"

# Create service file with current settings
SERVICE_FILE="ditto-talkinghead.service"
echo -e "${BLUE}Creating service file...${NC}"

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Ditto TalkingHead API Service
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
Group=$CURRENT_USER
WorkingDirectory=$CURRENT_DIR
Environment=PATH=$CONDA_ENV_PATH/bin:/usr/local/bin:/usr/bin:/bin
Environment=CONDA_DEFAULT_ENV=ditto
Environment=CONDA_PREFIX=$CONDA_ENV_PATH
Environment=DITTO_PORT=$PORT
Environment=CUDA_VISIBLE_DEVICES=$GPU_ID
ExecStart=$CONDA_ENV_PATH/bin/python ditto_proxy.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ditto-talkinghead

# Resource limits
LimitNOFILE=65536
MemoryMax=8G

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}Service file created: $SERVICE_FILE${NC}"

# Install service
echo -e "${BLUE}Installing service...${NC}"
sudo cp "$SERVICE_FILE" /etc/systemd/system/
sudo systemctl daemon-reload

echo -e "${GREEN}Service installed successfully!${NC}"

# Enable and start service
echo -e "${BLUE}Enabling and starting service...${NC}"
sudo systemctl enable ditto-talkinghead.service
sudo systemctl start ditto-talkinghead.service

# Check status
echo -e "${BLUE}Service status:${NC}"
sudo systemctl status ditto-talkinghead.service --no-pager

echo ""
echo -e "${GREEN}=== Installation Complete ===${NC}"
echo -e "Service name: ${YELLOW}ditto-talkinghead${NC}"
echo -e "Port: ${YELLOW}$PORT${NC}"
echo -e "GPU ID: ${YELLOW}$GPU_ID${NC}"
echo ""
echo -e "${BLUE}Useful commands:${NC}"
echo -e "  Start:   ${YELLOW}sudo systemctl start ditto-talkinghead${NC}"
echo -e "  Stop:    ${YELLOW}sudo systemctl stop ditto-talkinghead${NC}"
echo -e "  Restart: ${YELLOW}sudo systemctl restart ditto-talkinghead${NC}"
echo -e "  Status:  ${YELLOW}sudo systemctl status ditto-talkinghead${NC}"
echo -e "  Logs:    ${YELLOW}sudo journalctl -u ditto-talkinghead -f${NC}"
echo -e "  Disable: ${YELLOW}sudo systemctl disable ditto-talkinghead${NC}"
echo ""
echo -e "API will be available at: ${GREEN}http://localhost:$PORT${NC}"
