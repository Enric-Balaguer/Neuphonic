#!/bin/bash

ENV_NAME=Enric_Task

# Create a new virtual environment
python3 -m venv $ENV_NAME

# Activate the new environment
source $ENV_NAME/bin/activate

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Install packages from requirements.txt
pip install -r requirements.txt

echo "Environment setup complete."