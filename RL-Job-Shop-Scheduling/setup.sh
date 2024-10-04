#!/bin/bash

# Create conda environment named JSS with Python 3.7
conda create --name JSS

# Activate the environment
conda activate JSS

# Install necessary packages using conda and pip
conda install --file conda-specs.txt
pip install -r requirements.txt

echo "Conda environment 'JSS' is set up and ready."
