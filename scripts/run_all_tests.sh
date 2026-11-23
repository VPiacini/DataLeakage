#!/bin/bash

#source venv/bin/activate

if ! command -v python3 >/dev/null 2>&1;
then
    echo "Python3 is not installed. Please install Python3 to proceed."
    exit 1
fi

echo "Running normalization test..."
python main.py normalization

echo "Running feature selection test..."
python main.py feature_selection

echo "Running tuning test..."
python main.py tuning

echo "Running imputation test..."
python main.py imputation

echo "All tests completed."