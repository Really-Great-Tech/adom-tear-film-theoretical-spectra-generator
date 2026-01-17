#!/bin/bash
set -e

echo "--- Setting up CloudWatch Logging ---"

# 1. Install CloudWatch Agent
echo "Installing CloudWatch Agent..."
wget https://s3.eu-central-1.amazonaws.com/amazoncloudwatch-agent-eu-central-1/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb -O /tmp/amazon-cloudwatch-agent.deb
sudo dpkg -i /tmp/amazon-cloudwatch-agent.deb

# 2. Configure Agent
echo "Configuring Agent..."
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config \
    -m ec2 \
    -c file:logging/amazon-cloudwatch-agent.json \
    -s

# 3. Update Systemd Service for file logging
echo "Updating pyelli.service logging..."
SERVICE_FILE="/etc/systemd/system/pyelli.service"

# Use sed to ensure StandardOutput/Error are set correctly
# We use 'append' so logs accumulate
if grep -q "StandardOutput" "$SERVICE_FILE"; then
    sudo sed -i 's|^StandardOutput=.*|StandardOutput=append:/var/log/pyelli.log|' "$SERVICE_FILE"
else
    # Insert after [Service]
    sudo sed -i '/^\[Service\]/a StandardOutput=append:/var/log/pyelli.log' "$SERVICE_FILE"
fi

if grep -q "StandardError" "$SERVICE_FILE"; then
    sudo sed -i 's|^StandardError=.*|StandardError=append:/var/log/pyelli.log|' "$SERVICE_FILE"
else
    # Insert after StandardOutput
    sudo sed -i '/^StandardOutput=/a StandardError=append:/var/log/pyelli.log' "$SERVICE_FILE"
fi

# Ensure log file exists and is writable
sudo touch /var/log/pyelli.log
sudo chown ubuntu:ubuntu /var/log/pyelli.log
sudo chmod 644 /var/log/pyelli.log

echo "Reloading systemd..."
sudo systemctl daemon-reload
sudo systemctl restart pyelli

echo "--- Logging Setup Complete ---"
echo "Logs should appear in CloudWatch Log Groups: pyelli-app, pyelli-nginx-access"
