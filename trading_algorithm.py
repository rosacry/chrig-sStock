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
        # Fetch and preprocess real-time data using the updated load_data.py structure
        real_time_data = load_real_time_data()
        if not real_time_data.empty:
            # Apply feature engineering and get the final features to be used by the model
            features, _ = get_features_and_targets(real_time_data)  # Discard targets since they are not needed for real-time predictions
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

def predict_and_decide(portfolio, market_data):
    """Determine the best assets to buy or sell, evaluating all available market assets for buying."""
    model_path = 'path/to/model.pth'
    buy_model = load_or_initialize_model(model_path, 'buy')
    sell_model = load_or_initialize_model(model_path, 'sell')
    
    # Initialize the feature engineering pipeline
    pipeline = FeatureEngineeringPipeline()

    # Evaluate potential buys among all market assets
    best_buy = None
    highest_buy_score = -float('inf')
    for index, row in market_data.iterrows():
        processed_row = pipeline.transform(pd.DataFrame([row]))  # Transform the row using the pipeline
        buy_score = predict(buy_model, processed_row.iloc[0])
        if buy_score > config['buy_threshold'] and buy_score > highest_buy_score:
            highest_buy_score = buy_score
            best_buy = row['symbol']

    # Evaluate potential sells only within owned assets
    best_sell = None
    highest_sell_score = -float('inf')
    for symbol in portfolio.get_owned_assets():
        if symbol in market_data['symbol'].values:
            row = market_data[market_data['symbol'] == symbol].iloc[0]
            processed_row = pipeline.transform(pd.DataFrame([row]))
            sell_score = predict(sell_model, processed_row.iloc[0])
            if sell_score > config['sell_threshold'] and sell_score > highest_sell_score:
                highest_sell_score = sell_score
                best_sell = symbol

    decisions = {'buy': best_buy, 'sell': best_sell}
    return decisions


def execute_trade(symbol, decision, quantity=1):
    """Execute trade based on the decision using Alpaca API."""
    endpoint = f"{BASE_URL}/v2/orders"
    data = {
        "symbol": symbol,
        "qty": str(quantity),
        "side": decision,  # 'buy' or 'sell'
        "type": "market",  # Market order
        "time_in_force": "gtc"  # Good till canceled
    }

    try:
        response = requests.post(endpoint, headers=headers, json=data)
        response.raise_for_status()  # Check for HTTP request errors
        logging.info(f"Executed {decision} for {quantity} shares of {symbol} at market price.")
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err} - {response.text}")
    except Exception as err:
        logging.error(f"An error occurred: {err}")

# Main function to start the trading process
if __name__ == "__main__":
    evaluate_stocks()
