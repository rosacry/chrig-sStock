#!/bin/bash

# Update and install Python 3.8+, pip, and Ray
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.8 python3-pip

# Install Ray
pip3 install ray[default]

# Connect to Ray head node (replace IP and port with your GCE instance details)
ray start --address='gce-server-ip:6379'

# Download the latest AI model (replace with appropriate download command)
python3 download_model.py --bucket ai-models-repository --source latest_model.zip --destination ~/ai_models/latest_model.zip

# Extract and start the peer's bot (adjust path and import as needed)
unzip ~/ai_models/latest_model.zip -d ~/ai_models
python3 ~/ai_models/start_peer_bot.py
