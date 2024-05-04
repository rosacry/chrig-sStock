import argparse
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from features.feature_engineering import add_technical_indicators
from data.data_processing import clean_and_normalize_data
from models.model_training import advanced_grid_search_tune_model
from models.optuna_optimization import run_optuna_optimization

class UserInterface:
    def __init__(self):
        self.console = Console()

    def display_welcome_message(self):
        self.console.print("[bold cyan]Welcome to Stock AI Bot CLI[/bold cyan]")

    def select_training_mode(self):
        self.console.print("\n[bold yellow]Select Training Mode:[/bold yellow]")
        table = Table(title="Training Modes")
        table.add_column("Option", justify="right")
        table.add_column("Description")
        table.add_row("1", "Advanced Grid Search Tuning")
        table.add_row("2", "Optuna Optimization")

        self.console.print(table)
        mode = self.console.input("[bold]Enter your choice (1/2):[/bold] ")
        return mode

    def configure_arguments(self):
        parser = argparse.ArgumentParser(description="Stock AI Bot Command-Line Interface")
        parser.add_argument("--symbol", type=str, required=True, help="Stock symbol to analyze (e.g., AAPL)")
        parser.add_argument("--risk", type=str, choices=["low", "medium", "high"], default="medium", help="Investment risk tolerance")
        parser.add_argument("--capital", type=float, required=True, help="Total capital available for investment")
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
            cleaned_data = clean_and_normalize_data(aggregated_data)
            enhanced_data = add_technical_indicators(cleaned_data)

        # Select training mode
        mode = self.select_training_mode()
        with tqdm(total=100, desc="Training Progress", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as progress_bar:
            if mode == "1":
                progress_bar.set_description("Advanced Grid Search Tuning")
                results = advanced_grid_search_tune_model(enhanced_data)
                progress_bar.update(100)
                best_models = results["best_models"]
                evaluation_metrics = results["evaluation_metrics"]
                self.console.print(f"[bold]Model Evaluation Metrics (Grid Search):[/bold] {evaluation_metrics}")
            elif mode == "2":
                progress_bar.set_description("Optuna Optimization")
                run_optuna_optimization(enhanced_data)
                progress_bar.update(100)
            else:
                self.console.print("[bold red]Invalid option. Exiting...[/bold red]")

if __name__ == "__main__":
    ui = UserInterface()
    ui.main()