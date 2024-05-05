import argparse
from rich.console import Console
from tqdm import tqdm
from features.feature_engineering import FeatureEngineer
from data.data_processing import DataProcessor
from models.model_training import ModelTrainer
from models.model_tuning import ModelTuner
from optimization.unified_tuning import UnifiedTuner
from distributed.distributed_training import DistributedTraining
from sklearn.model_selection import train_test_split
import pandas as pd

# Initialize components
fe = FeatureEngineer()
mt = ModelTrainer()
mtu = ModelTuner()
ut = UnifiedTuner()
dt = DistributedTraining()
console = Console()

class UserInterface:
    def __init__(self):
        self.console = console

    def display_welcome_message(self):
        self.console.print("[bold cyan]Welcome to Stock AI Bot CLI[/bold cyan]")

    def configure_arguments(self):
        parser = argparse.ArgumentParser(description="Stock AI Bot Command-Line Interface")
        parser.add_argument("--symbol", type=str, required=True, help="Stock symbol to analyze (e.g., AAPL)")
        parser.add_argument("--risk", type=str, choices=["low", "medium", "high"], default="medium", help="Investment risk tolerance")
        parser.add_argument("--capital", type=float, required=True, help="Total capital available for investment")
        parser.add_argument("--asset", type=str, choices=["stocks", "crypto", "options", "auto"], default="auto", help="Asset type to invest in")
        return parser.parse_args()

    def main(self):
        args = self.configure_arguments()
        self.display_welcome_message()

        # Fetch the stock symbol and aggregate its data
        symbol = args.symbol
        self.console.print(f"\n[bold]Analyzing {symbol} with {args.capital} capital and {args.risk} risk tolerance...[/bold]")
        with self.console.status("[bold green]Fetching and processing data...[/bold]") as status:
            from api.api_clients import aggregate_stock_data
            aggregated_data = aggregate_stock_data(symbol)
            cleaned_data = DataProcessor.clean_and_normalize_data(aggregated_data)
            enhanced_data = fe.add_technical_indicators(cleaned_data)

        with tqdm(total=100, desc="Training Progress", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as progress_bar:
            progress_bar.set_description("Training the Model")
            trained_model = mt.train(enhanced_data)
            progress_bar.update(40)

            progress_bar.set_description("Tuning the Model")
            tuned_model = mtu.tune(trained_model)
            progress_bar.update(30)

            progress_bar.set_description("Optimizing the Model")
            optimized_model = ut.unified_tuning(tuned_model)
            progress_bar.update(30)

            self.console.print(f"[bold]Model Training, Tuning, and Optimization completed successfully![/bold]")

        # Prepare data for collaborative training
        feature_columns = ["sma_20", "sma_50", "sma_200", "ema_20", "ema_50", "price_pct_change", "on_balance_volume"]
        target_column = "close"
        X = enhanced_data[feature_columns].fillna(0)
        y = enhanced_data[target_column].fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Execute distributed training
        try:
            dt.run_distributed_training(pd.concat([X_train, y_train], axis=1), optimized_model)
            self.console.print("[bold]Successfully completed the distributed training session.[/bold]")
        except Exception as e:
            self.console.print(f"[bold red]Error during distributed training: {e}[/bold red]")

if __name__ == "__main__":
    ui = UserInterface()
    ui.main()
