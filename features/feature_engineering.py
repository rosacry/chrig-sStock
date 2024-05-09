import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Custom Transformer for Financial Indicators
class FinancialIndicators(BaseEstimator, TransformerMixin):
    def __init__(self, window=14, atr_window=14):
        self.window = window
        self.atr_window = atr_window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_cp = X.copy()
        # Simple Moving Average
        X_cp['SMA'] = X_cp['close'].rolling(window=self.window).mean()
        # Exponential Moving Average
        X_cp['EMA'] = X_cp['close'].ewm(span=self.window, adjust=False).mean()
        # RSI
        delta = X_cp['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        rs = gain / loss
        X_cp['RSI'] = 100 - (100 / (1 + rs))
        # MACD
        exp1 = X_cp['close'].ewm(span=12, adjust=False).mean()
        exp2 = X_cp['close'].ewm(span=26, adjust=False).mean()
        X_cp['MACD'] = exp1 - exp2
        # Bollinger Bands
        X_cp['middle_band'] = X_cp['close'].rolling(window=self.window).mean()
        X_cp['std_dev'] = X_cp['close'].rolling(window=self.window).std()
        X_cp['upper_band'] = X_cp['middle_band'] + (X_cp['std_dev'] * 2)
        X_cp['lower_band'] = X_cp['middle_band'] - (X_cp['std_dev'] * 2)
        # VWAP
        if 'volume' in X_cp.columns:
            vwap = (X_cp['close'] * X_cp['volume']).cumsum() / X_cp['volume'].cumsum()
            X_cp['VWAP'] = vwap
        # Average True Range (ATR)
        high_low = X_cp['high'] - X_cp['low']
        high_close = np.abs(X_cp['high'] - X_cp['close'].shift())
        low_close = np.abs(X_cp['low'] - X_cp['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        X_cp['ATR'] = true_range.rolling(window=self.atr_window).mean()
        return X_cp

class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.numeric_features = []
        self.categorical_features = []
        self.financial_transformer = FinancialIndicators()
        self.numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        self.categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        self.pca = PCA(n_components=n_components) if n_components else None
        self.preprocessor = None

    def _identify_feature_types(self, X: pd.DataFrame):
        self.numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.preprocessor = ColumnTransformer(transformers=[
            ('fin', self.financial_transformer, ['close', 'high', 'low', 'volume']),
            ('num', self.numeric_transformer, self.numeric_features),
            ('cat', self.categorical_transformer, self.categorical_features)
        ])

    def fit(self, X: pd.DataFrame, y=None):
        self._identify_feature_types(X)
        self.preprocessor.fit(X)
        if self.pca:
            transformed_data = self.preprocessor.transform(X)
            self.pca.fit(transformed_data)
        return self

    def transform(self, X: pd.DataFrame):
        transformed_data = self.preprocessor.transform(X)
        if self.pca:
            transformed_data = self.pca.transform(transformed_data)
        return transformed_data

    def fit_transform(self, X: pd.DataFrame, y=None):
        return self.fit(X).transform(X), y
