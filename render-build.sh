#!/bin/bash
# Update package list
apt-get update

# Install PortAudio (required for PyAudio)
apt-get install -y portaudio19-dev

# Install Python dependencies
pip install -r requirements.txt
