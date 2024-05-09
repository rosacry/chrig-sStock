# Imports
import os
import sys
from datetime import datetime
import requests
import pandas as pd
import numpy as np
from sklearn.externals import joblib

# Custom modules
import data_processing
import model_training
import trading_algorithm
from load_data import scrape_data, preprocess_data
from model_training import train_model, update_model
from trading_algorithm import make_trades

# Configuration
config_path = "path/to/config.json"
config = load_config(config_path)

# Main orchestration function
def main():
    # Step 1: Data collection and preprocessing
    data = scrape_data()
    processed_data = preprocess_data(data)
    
    # Step 2: Model training
    model = train_model(processed_data)
    
    # Step 3: Making trades
    make_trades(model, processed_data)
    
    # Step 4: Update the model incrementally
    update_model()

# Function to load configurations
def load_config(path):
    import json
    with open(path, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    main()
