#!/bin/bash

ENV_NAME=Enric_Task

# Create a new conda environment
conda create --name $ENV_NAME python=3.9 -y

# Activate the new environment
source activate $ENV_NAME

# Install packages from requirements.txt
pip install -r requirements.txt

# Install PyAudio from source
pip install --no-use-pep517 pyaudio

echo "Environment setup complete. To activate, run: conda activate $ENV_NAME"