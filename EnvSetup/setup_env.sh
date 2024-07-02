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

# Download and install PortAudio
cd $VIRTUAL_ENV
wget http://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz
tar -zxvf pa_stable_v190700_20210406.tgz
cd portaudio
./configure --prefix=$VIRTUAL_ENV
make
make install

# Set environment variables for PortAudio
export CFLAGS="-I$VIRTUAL_ENV/include"
export LDFLAGS="-L$VIRTUAL_ENV/lib"

# Install PyAudio
pip install --global-option='build_ext' --global-option="-I$VIRTUAL_ENV/include" --global-option="-L$VIRTUAL_ENV/lib" pyaudio

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cs/student/msc/ml/2023/ebalague/local/lib:$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH

echo "Environment setup complete. To activate, run: source $ENV_NAME/bin/activate"
echo "Remember to set these environment variables in your session:"
echo "export CFLAGS=\"-I$VIRTUAL_ENV/include\""
echo "export LDFLAGS=\"-L$VIRTUAL_ENV/lib\""
echo "export LD_LIBRARY_PATH=/cs/student/msc/ml/2023/ebalague/local/lib:$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"