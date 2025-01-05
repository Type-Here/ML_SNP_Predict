#!/bin/bash

# Check for conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "No conda environment detected"
    echo "Please activate the conda environment before running the application"
    echo "Default conda env name: fia_env"
    return 1
else
    echo "Conda environment detected: $CONDA_DEFAULT_ENV"
fi

# Run the application
python --version
if [ $? -eq 0 ]; then
    python UI/main.py
else
    python3 UI/main.py
fi