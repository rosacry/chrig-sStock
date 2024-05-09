import streamlit as st
import threading
import json
from models.model_training import continuous_update
from distributed.distributed_training import manage_training_sessions
from trading_algorithm import evaluate_stocks, fetch_real_time_data

# Set page configuration
st.set_page_config(page_title='Trading Bot Dashboard', layout='wide')

# Load configuration settings
def load_config():
    with open('path/to/config.json', 'r') as file:
        return json.load(file)

config = load_config()

# Continuous updates for the model using real-time data
def continuous_model_updates():
    """Continuous model updates using real-time data."""
    continuous_update('models/model/aiModel.pth', update_interval=86400)

# Display real-time market data
def display_real_time_data():
    st.subheader('Real-Time Market Data')
    data = fetch_real_time_data()
    if data is not None:
        st.dataframe(data)
    else:
        st.error("No real-time data available.")

# Display stock evaluation results
def display_stock_evaluation():
    st.subheader('Evaluate Stocks for Trading Decisions')
    if st.button('Evaluate Stocks'):
        stocks = evaluate_stocks()
        if stocks is not None:
            st.dataframe(stocks)
        else:
            st.error("No stocks to evaluate.")

# Start distributed training
def start_distributed_training():
    if 'training_thread' not in st.session_state or not st.session_state.training_thread.is_alive():
        st.session_state.training_thread = threading.Thread(target=manage_training_sessions, daemon=True)
        st.session_state.training_thread.start()
        st.success('Distributed training started.')
    else:
        st.error("Training already in progress.")

# Stop distributed training
def stop_distributed_training():
    if 'training_thread' in st.session_state and st.session_state.training_thread.is_alive():
        st.session_state.training_thread = None
        st.success("Distributed training has been stopped.")
    else:
        st.error("No active distributed training to stop.")

def start_trading_bot():
    """Start the trading operations."""
    if 'trading_thread' in st.session_state:
        st.warning("Trading bot is already running.")
    else:
        st.session_state.trading_thread = threading.Thread(target=evaluate_stocks, daemon=True)
        st.session_state.trading_thread.start()
        st.success('Trading bot started successfully.')

def stop_trading_bot():
    """Stop the trading bot operation."""
    if 'trading_thread' in st.session_state and st.session_state.trading_thread.is_alive():
        st.session_state.trading_thread = None
        st.success("Trading bot has been stopped.")
    else:
        st.error("No active trading bot to stop.")

# Main app functionality
def main():
    st.title('Trading Bot Dashboard')

    # Continuous model updates setup
    continuous_model_updates()

    # Configuration settings sidebar
    st.sidebar.title("Configuration Settings")
    st.sidebar.json(config)

    # Layout different sections in tabs
    tab1, tab2, tab3 = st.tabs(["Real-Time Data", "Stock Evaluation", "Model Management"])
    
    with tab1:
        display_real_time_data()
    
    with tab2:
        display_stock_evaluation()
    
    with tab3:
        st.subheader('Model Training and Updates')
        if st.button('Start Trading Bot'):
            start_trading_bot()
        if st.button('Stop Trading Bot'):
            stop_trading_bot()
        if st.button('Join Distributed Training'):
            start_distributed_training()
        if st.button('Stop Distributed Training'):
            stop_distributed_training()

if __name__ == '__main__':
    main()
