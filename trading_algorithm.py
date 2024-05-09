# Imports
import requests
import json
import pandas as pd
import logging

# Custom modules
from models.model_training import load_or_initialize_model, predict
from data.load_data import load_real_time_data, get_features_and_targets
from features.feature_engineering import FeatureEngineeringPipeline

# Setup the API key and endpoint
API_KEY = 'your_alpaca_api_key'
API_SECRET = 'your_alpaca_api_secret'
BASE_URL = 'https://paper-api.alpaca.markets'  # Use https://api.alpaca.markets for live trading

headers = {
    'APCA-API-KEY-ID': API_KEY,
    'APCA-API-SECRET-KEY': API_SECRET,
    'Content-Type': 'application/json'
}

# Configuration
config_path = "path/to/trading_config.json"
config = json.load(open(config_path))

# Setup logging
logging.basicConfig(filename='trading_log.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def fetch_real_time_data():
    """Fetch and process real-time stock data from defined APIs for immediate trading decisions."""
    try:
        real_time_data = load_real_time_data()
        if not real_time_data.empty:
            features, _ = get_features_and_targets(real_time_data)
            return features
        else:
            logging.error("No real-time data available or data is empty")
            return None
    except Exception as e:
        logging.error(f"Failed to fetch or process real-time data: {e}")
        return None

def evaluate_stocks():
    """Evaluate all stocks for buying and selling decisions."""
    market_data = fetch_real_time_data()
    if not market_data.empty:
        market_data['buy_decision'] = market_data.apply(lambda row: predict_and_decide(row, 'buy'), axis=1)
        market_data['sell_decision'] = market_data.apply(lambda row: predict_and_decide(row, 'sell'), axis=1)

        buys = market_data[market_data['buy_decision'] == 'buy']
        sells = market_data[market_data['sell_decision'] == 'sell']

        for index, row in buys.iterrows():
            execute_trade(row['symbol'], 'buy')
        for index, row in sells.iterrows():
            execute_trade(row['symbol'], 'sell')

def predict_and_decide(row, decision_type):
    """Determine the best assets to buy or sell, evaluating all available market assets for buying."""
    model_path = 'path/to/model.pth'
    model = load_or_initialize_model(model_path, decision_type)
    processed_row = FeatureEngineeringPipeline().transform(pd.DataFrame([row]))
    return predict(model, processed_row.iloc[0])

def execute_trade(symbol, decision, quantity=1):
    """Execute trade based on the decision using Alpaca API."""
    endpoint = f"{BASE_URL}/v2/orders"
    data = {
        "symbol": symbol,
        "qty": str(quantity),
        "side": decision,
        "type": "market",
        "time_in_force": "gtc"
    }
    try:
        response = requests.post(endpoint, headers=headers, json=data)
        response.raise_for_status()
        logging.info(f"Executed {decision} for {quantity} shares of {symbol} at market price.")
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err} - {response.text}")
    except Exception as err:
        logging.error(f"An error occurred: {err}")

if __name__ == "__main__":
    evaluate_stocks()
