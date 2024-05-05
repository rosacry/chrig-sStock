# file: stock_ai_bot/ui/streamlit_app.py

import streamlit as st
import pandas as pd
from features.feature_engineering import add_technical_indicators
from data.data_processing import clean_and_normalize_data
from models.model_training import advanced_grid_search_tune_model
from models.optuna_optimization import run_optuna_optimization
from api.api_clients import aggregate_stock_data
from optimization.unified_tuning import UnifiedTuner

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
            # Aggregate, clean, and enhance stock data
            aggregated_data = aggregate_stock_data(stock_symbol, investment_type)
            cleaned_data = clean_and_normalize_data(aggregated_data)
            enhanced_data = add_technical_indicators(cleaned_data)

        st.write("Running Unified Tuning...")
        tuner = UnifiedTuner(stock_symbol, investment_type)
        if investment_type == "AI Decision":
            tuner.ai_decision(enhanced_data)
        else:
            tuner.unified_tuning(enhanced_data)

    # Execute when the "Join Training" button is pressed
    if st.button("Join Training"):
        st.write("Joining the ongoing training...")
        # Implement the logic for joining the ongoing training

if __name__ == "__main__":
    main()
