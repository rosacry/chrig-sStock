# Imports
import requests
import json
import pandas as pd
import logging
import asyncio

# Custom modules
from models.model_training import load_or_initialize_model, predict
from data.load_data import load_real_time_data, get_features_and_targets
from features.feature_engineering import FeatureEngineeringPipeline
from utils.util import load_config, setup_logging, get_logger
from flask import Flask, request, jsonify
from flask import Flask

app = Flask(__name__)

# Load configuration and setup logging
config = load_config('utils/config/trading_config.json')
logger = get_logger(__name__)
setup_logging(config['logging']['filename'])

# Setup logging
logging.basicConfig(filename='utils/config/trading_log.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def fetch_real_time_data(api_key, api_secret):
    """Fetch and process real-time stock data from defined APIs for immediate trading decisions."""
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': api_secret,
        'Content-Type': 'application/json'
    }
    try:
        real_time_data = asyncio.run(load_real_time_data(headers))
        if not real_time_data.empty:
            features, _ = get_features_and_targets(real_time_data)
            return features
        else:
            logging.error("No real-time data available or data is empty")
            return None
    except Exception as e:
        logging.error(f"Failed to fetch or process real-time data: {e}")
        return None


@app.route('/evaluate_stocks', methods=['POST'])
def evaluate_stocks():
    """Evaluate all stocks for buying and selling decisions."""
    data = request.json
    api_key = data['api_key']
    api_secret = data['api_secret']
    market_data = fetch_real_time_data(api_key, api_secret)

    if market_data is not None and not market_data.empty:
        market_data['buy_decision'] = market_data.apply(lambda row: predict_and_decide(row, 'buy'), axis=1)
        market_data['sell_decision'] = market_data.apply(lambda row: predict_and_decide(row, 'sell'), axis=1)

        buys = market_data[market_data['buy_decision'] == 'buy']
        sells = market_data[market_data['sell_decision'] == 'sell']

        for index, row in buys.iterrows():
            execute_trade(row['symbol'], 'buy', api_key, api_secret)
        for index, row in sells.iterrows():
            execute_trade(row['symbol'], 'sell', api_key, api_secret)
    return jsonify({"message": "Evaluation completed"})

def predict_and_decide(row, decision_type):
    """Determine the best assets to buy or sell, evaluating all available market assets for buying."""
    model_path = 'path/to/model.pth'
    model = load_or_initialize_model(model_path, decision_type)
    processed_row = FeatureEngineeringPipeline().transform(pd.DataFrame([row]))
    return predict(model, processed_row.iloc[0])

def execute_trade(symbol, decision, api_key, api_secret, quantity=1):
    """Execute trade based on the decision using Alpaca API."""
    headers = {
        'APCA-API-KEY-ID': api_key,
        'APCA-API-SECRET-KEY': api_secret,
        'Content-Type': 'application/json'
    }
    endpoint = f"https://paper-api.alpaca.markets/v2/orders"
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

#setup api key and secret
#setup cron job to run the script every 5 minutes?