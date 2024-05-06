# /mnt/data/feature_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        """Initialize the pipeline with optional PCA components count."""
        # Initial placeholders for feature types
        self.numeric_features = []
        self.categorical_features = []

        # Transformers for numerical scaling and categorical encoding
        self.numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        self.categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # PCA setup (optional)
        self.pca = PCA(n_components=n_components) if n_components else None

        # Placeholder for ColumnTransformer
        self.preprocessor = None

    def _identify_feature_types(self, X: pd.DataFrame):
        """Identify numerical and categorical feature columns."""
        self.numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Define the full preprocessor pipeline
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', self.numeric_transformer, self.numeric_features),
            ('cat', self.categorical_transformer, self.categorical_features)
        ])

    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformations and optional PCA to training data."""
        self._identify_feature_types(X)

        # Fit the preprocessor to the data
        self.preprocessor.fit(X)

        # Optionally fit PCA
        if self.pca:
            transformed_data = self.preprocessor.transform(X)
            self.pca.fit(transformed_data)

        return self

    def transform(self, X: pd.DataFrame):
        """Apply the fitted transformations and optional PCA to new data."""
        # Apply the preprocessor
        transformed_data = self.preprocessor.transform(X)

        # Optionally apply PCA
        if self.pca:
            transformed_data = self.pca.transform(transformed_data)

        return transformed_data

    def fit_transform(self, X: pd.DataFrame, y=None):
        """Fit and then transform the features."""
        return self.fit(X).transform(X), y
