# Import necessary modules
from api.api_clients import PolygonClient, AlphaVantageClient
from data.data_processing import DataProcessor
from features.feature_engineering import FeatureEngineer
from models.model_training import ModelTrainer
from models.model_tuning import ModelTuner
from optimization.unified_tuning import UnifiedTuner
from ui.cli import UserInterface

# Initialize API clients
polygon_client = PolygonClient()
alpha_vantage_client = AlphaVantageClient()

# Initialize data processor
data_processor = DataProcessor()

# Initialize feature engineer
feature_engineer = FeatureEngineer()

# Initialize model trainer and tuner
model_trainer = ModelTrainer()
model_tuner = ModelTuner()

# Initialize optimization tuner
unified_tuner = UnifiedTuner()

# Initialize user interface
ui = UserInterface()

# Start the AI bot
def start_bot():
    # Fetch data
    stock_data = polygon_client.get_stock_data()
    crypto_data = alpha_vantage_client.get_crypto_data()

    # Process data
    processed_stock_data = data_processor.process(stock_data)
    processed_crypto_data = data_processor.process(crypto_data)

    # Generate features
    stock_features = feature_engineer.generate(processed_stock_data)
    crypto_features = feature_engineer.generate(processed_crypto_data)

    # Train and tune models
    stock_model = model_trainer.train(stock_features)
    crypto_model = model_trainer.train(crypto_features)

    tuned_stock_model = model_tuner.tune(stock_model)
    tuned_crypto_model = model_tuner.tune(crypto_model)

    # Optimize models
    optimized_stock_model = unified_tuner.optimize(tuned_stock_model)
    optimized_crypto_model = unified_tuner.optimize(tuned_crypto_model)

    # Interact with user
    ui.interact(optimized_stock_model, optimized_crypto_model)

# Run the bot
if __name__ == "__main__":
    start_bot()