#!/bin/bash
# Level 5 Turn-Key Setup
echo "🚀 Initializing Symbiotic Forecaster Environment..."
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas torch scikit-learn matplotlib
python3 scripts/download_data.py
mkdir -p models figures data/raw
echo "✅ Environment Ready. Run 'python3 train.py' to start."
