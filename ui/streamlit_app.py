import streamlit as st
import pandas as pd
from data.data_processing import clean_and_normalize_data
from models.model_training import ModelTrainer
from models.model_tuning import ModelTuner
from optimization.unified_tuning import UnifiedTuner
from distributed.distributed_training import DistributedTraining
from api.api_clients import aggregate_stock_data, get_top_investor_holdings
from features.feature_engineering import add_technical_indicators
import torch
from sklearn.model_selection import train_test_split

# Initialize model components and distributed training
model_trainer = ModelTrainer()
model_tuner = ModelTuner()
unified_tuner = UnifiedTuner()
distributed_training = DistributedTraining()

def main():
    # Streamlit page configuration
    st.set_page_config(page_title="Stock AI Bot", page_icon="📈", layout="wide")
    st.title("📈 Stock AI Bot Dashboard")

    # User inputs
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
    capital = st.number_input("Enter Total Capital Available ($)", min_value=0.0, value=1000.0)
    risk_tolerance = st.selectbox("Investment Risk Tolerance", ["Low", "Medium", "High"])
    investment_type = st.selectbox("Choose Investment Type", ["Stocks", "Crypto", "Options", "AI Decision"])

    # Execute when the "Train AI" button is pressed
    if st.button("Train AI"):
        st.write(f"Training AI for {stock_symbol} with ${capital} capital and {risk_tolerance} risk tolerance.")
        with st.spinner("Fetching and processing data..."):
            # Aggregate and process stock data
            aggregated_data = aggregate_stock_data(stock_symbol, investment_type)
            cleaned_data = clean_and_normalize_data(aggregated_data)
            enhanced_data = add_technical_indicators(cleaned_data)

            # Fetch holdings of top investors if relevant
            top_investor_data = get_top_investor_holdings(stock_symbol)

            # Integrate additional data sources if available
            if top_investor_data:
                st.write("Incorporating top investor holdings into the model.")
                enhanced_data = enhanced_data.join(top_investor_data)

        st.write("Training model...")
        trained_model = model_trainer.train(enhanced_data)

        st.write("Tuning model...")
        tuned_model = model_tuner.tune(trained_model)

        st.write("Running Unified Tuning...")
        if investment_type == "AI Decision":
            optimized_model = unified_tuner.ai_decision(tuned_model)
        else:
            optimized_model = unified_tuner.unified_tuning(tuned_model)

        st.success("Model training and optimization complete!")

    # Allow users to join collaborative training using DistributedDataParallel (DDP)
    if st.button("Join Collaborative Training"):
        st.write("Joining the ongoing distributed training...")

        # Split the enhanced data into training and testing sets
        feature_columns = ["sma_20", "sma_50", "sma_200", "ema_20", "ema_50", "price_pct_change", "on_balance_volume"]
        target_column = "close"
        X = enhanced_data[feature_columns].fillna(0)
        y = enhanced_data[target_column].fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        try:
            # Run distributed training with DDP
            distributed_training.run_distributed_training(pd.concat([X_train, y_train], axis=1), tuned_model)
            st.success("Successfully joined the distributed training session.")
        except Exception as e:
            st.error(f"Failed to join distributed training session: {e}")

if __name__ == "__main__":
    main()
