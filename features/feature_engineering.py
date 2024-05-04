import numpy as np
import pandas as pd

class FeatureEngineer:
    def add_technical_indicators(stock_data: pd.DataFrame):
        """Add useful technical indicators to the stock data for machine learning.

        Args:
            stock_data (pd.DataFrame): Cleaned and normalized stock data.

        Returns:
            pd.DataFrame: Stock data enhanced with technical indicators.
        """
        # Simple Moving Averages (SMA) over different periods
        stock_data["sma_20"] = stock_data["close"].rolling(window=20).mean()
        stock_data["sma_50"] = stock_data["close"].rolling(window=50).mean()
        stock_data["sma_200"] = stock_data["close"].rolling(window=200).mean()

        # Exponential Moving Averages (EMA)
        stock_data["ema_20"] = stock_data["close"].ewm(span=20, adjust=False).mean()
        stock_data["ema_50"] = stock_data["close"].ewm(span=50, adjust=False).mean()

        # Percentage change in closing prices
        stock_data["price_pct_change"] = stock_data["close"].pct_change()

        # Volume indicators (e.g., On-Balance Volume)
        stock_data["on_balance_volume"] = (np.sign(stock_data["price_pct_change"]) * stock_data["volume"]).cumsum()

        # Fill any remaining NaN values created by rolling calculations
        stock_data.fillna(0, inplace=True)

        return stock_data