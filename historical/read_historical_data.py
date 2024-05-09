import yfinance as yf
import pandas as pd

def fetch_tickers_from_file(file_path):
    # This function reads ticker symbols from a CSV file
    return pd.read_csv(file_path)['Symbol'].tolist()

def download_stock_data(tickers):
    data = pd.DataFrame()
    for ticker in tickers:
        print(f"Downloading data for {ticker}")
        stock_data = yf.download(ticker, start="2000-01-01", end="2024-5-09")
        stock_data['Symbol'] = ticker  # Add a symbol column
        if not stock_data.empty:
            data = pd.concat([data, stock_data])
    return data

def main():
    # Load ticker symbols from the CSV files
    sp500_tickers = fetch_tickers_from_file('/mnt/data/sp500_tickers.csv')
    nasdaq_tickers = fetch_tickers_from_file('/mnt/data/nasdaq_tickers.csv')
    
    # Combine all unique tickers from S&P 500 and NASDAQ (ignoring Dow for now)
    all_tickers = list(set(sp500_tickers + nasdaq_tickers))

    # Download stock data
    all_stock_data = download_stock_data(all_tickers)

    # Ensure all required columns are present
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    all_stock_data = all_stock_data.reset_index()
    all_stock_data = all_stock_data[columns]

    # Save the data to CSV
    all_stock_data.to_csv("all_historical_stock_data_2000_to_2024.csv", index=False)

if __name__ == "__main__":
    main()
