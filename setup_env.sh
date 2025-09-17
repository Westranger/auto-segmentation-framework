#!/usr/bin/env bash
set -e

VENV_DIR=".venv"

if ! command -v python3.9 &> /dev/null
then
    echo "❌ python3.9 not found. please install."
    exit 1
fi

python3.12 -m venv $VENV_DIR
echo "virtual environment $VENV_DIR created"

source $VENV_DIR/bin/activate

pip install --upgrade pip
pip install torch==2.0.0+cu121 torchvision==0.15.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo "Setup completed. activate environment with:"
echo "source $VENV_DIR/bin/activate"
